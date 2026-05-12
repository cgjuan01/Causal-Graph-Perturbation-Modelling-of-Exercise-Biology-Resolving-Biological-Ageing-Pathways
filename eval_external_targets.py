#!/usr/bin/env python3
"""
Step 4 — External evaluation of gene embeddings on held-out ageing targets.

Validates whether learned gene embeddings encode ageing-relevant biological structure
without any ageing supervision during training.  Frozen embeddings are used as input
features to a regularised linear model (Ridge or Elastic Net) trained to predict
independent ageing targets on predefined degree-stratified splits.

No ageing labels are used during contrastive pretraining (Step 1) or MR-informed
refinement (Step 2); any predictive signal observed here therefore emerges from network
topology and causally anchored exercise biology rather than supervised optimisation.

Targets evaluated
-----------------
* Epigenetic clock DMR summaries: mean absolute methylation change or per-DMR counts
  aggregated per gene, for Horvath, Hannum, PhenoAge, and DunedinPACE clocks.
* Proteomic ageing PC1: first principal component loadings of a multi-cohort proteomic
  ageing axis.

Split design
------------
Degree-stratified train/validation/test splits (80/10/10 default) are used to prevent
high-degree hub genes—which accumulate more MR signal—from dominating evaluation.
The split column in --splits encodes membership as 'train', 'val', or 'test'.

Metrics
-------
* R² on the held-out test set.
* Spearman rank correlation (ρ) on the test set.
* Permutation-based p-value for |ρ| (n_perm resamples of predicted values).

Inputs
------
--embeddings    embeddings_mrrefined.tsv  (or embeddings_contrastive.tsv)
                TSV with columns: gene_symbol, index, emb_*
--targets       paper2_targets_external_epiclocks_proteomicPC1.tsv
                TSV with columns: gene_symbol, <target_cols...>
--splits        paper2_splits_degreeStratified.tsv
                TSV with columns: gene_symbol, <split_col>

Outputs (under --outdir)
------------------------
results_external_metrics.tsv        summary table (one row per target)
predictions_<target_name>.tsv       per-gene y_true / y_pred for test set
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def permutation_pvalue(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute Spearman ρ and a permutation-based p-value for |ρ|.

    The null distribution is constructed by permuting y_pred (predictions) n_perm times
    and counting how often |ρ_null| ≥ |ρ_obs|.

    Returns
    -------
    rho   : observed Spearman ρ
    p_val : empirical two-sided p-value ((count + 1) / (n_perm + 1))
    """
    rho_obs = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho_obs):
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rho_null = spearmanr(y_true, rng.permutation(y_pred)).correlation
        if not np.isnan(rho_null) and abs(rho_null) >= abs(rho_obs):
            count += 1

    return float(rho_obs), float((count + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="External evaluation of gene embeddings on held-out ageing targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embeddings", required=True, help="Gene embedding TSV (emb_* columns).")
    ap.add_argument("--targets",    required=True, help="Ageing target TSV (gene_symbol + target cols).")
    ap.add_argument("--splits",     required=True, help="Degree-stratified split TSV (gene_symbol, split_col).")
    ap.add_argument("--outdir",     required=True, help="Output directory.")

    ap.add_argument(
        "--split_col",
        default="set_801010",
        help="Column in --splits encoding train/val/test membership.",
    )
    ap.add_argument(
        "--model",
        default="ridge",
        choices=["ridge", "elasticnet"],
        help="Regularised linear model for evaluation.",
    )
    ap.add_argument("--alpha",    type=float, default=1.0,  help="Regularisation strength.")
    ap.add_argument("--l1_ratio", type=float, default=0.1,  help="Elastic Net L1 ratio (ignored for Ridge).")
    ap.add_argument("--n_perm",   type=int,   default=2000, help="Number of permutations for p-value.")
    ap.add_argument(
        "--min_train",
        type=int,
        default=50,
        help="Minimum training set size required to evaluate a target.",
    )
    ap.add_argument(
        "--min_test",
        type=int,
        default=30,
        help="Minimum test set size required to evaluate a target.",
    )
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & merge
    # ------------------------------------------------------------------
    emb_df  = pd.read_csv(args.embeddings, sep="\t")
    targ_df = pd.read_csv(args.targets,    sep="\t")
    spl_df  = pd.read_csv(args.splits,     sep="\t")

    df = (
        emb_df
        .merge(targ_df, on="gene_symbol", how="left")
        .merge(spl_df[["gene_symbol", args.split_col]], on="gene_symbol", how="left")
    )

    emb_cols    = [c for c in df.columns if c.startswith("emb_")]
    target_cols = [c for c in targ_df.columns if c != "gene_symbol"]

    if not emb_cols:
        raise ValueError("No 'emb_*' columns found in --embeddings.")
    if not target_cols:
        raise ValueError("No target columns found in --targets.")

    # ------------------------------------------------------------------
    # Build model pipeline
    # ------------------------------------------------------------------
    if args.model == "ridge":
        regressor = Ridge(alpha=args.alpha, random_state=42)
    else:
        regressor = ElasticNet(
            alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42, max_iter=10_000
        )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg",    regressor),
    ])

    # ------------------------------------------------------------------
    # Evaluate each target
    # ------------------------------------------------------------------
    results = []

    for ycol in target_cols:
        sub = (
            df[["gene_symbol", args.split_col] + emb_cols + [ycol]]
            .dropna(subset=[ycol, args.split_col])
            .copy()
        )
        if sub.empty:
            continue

        train = sub[sub[args.split_col] == "train"]
        test  = sub[sub[args.split_col] == "test"]

        if train.shape[0] < args.min_train or test.shape[0] < args.min_test:
            continue

        X_train = train[emb_cols].values
        y_train = train[ycol].values.astype(float)
        X_test  = test[emb_cols].values
        y_test  = test[ycol].values.astype(float)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2       = r2_score(y_test, y_pred)
        rho, pv  = permutation_pvalue(y_test, y_pred, n_perm=args.n_perm)

        results.append({
            "target":              ycol,
            "n_train":             train.shape[0],
            "n_test":              test.shape[0],
            "r2_test":             r2,
            "spearman_rho_test":   rho,
            "perm_p_absrho":       pv,
            "model":               args.model,
            "alpha":               args.alpha,
            "l1_ratio":            args.l1_ratio if args.model == "elasticnet" else float("nan"),
        })

        # Save per-target predictions
        pred_df = test[["gene_symbol"]].copy()
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        safe_name = ycol.replace("/", "_")
        pred_df.to_csv(
            os.path.join(args.outdir, f"predictions_{safe_name}.tsv"),
            sep="\t", index=False,
        )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    res_df = (
        pd.DataFrame(results)
        .sort_values(["perm_p_absrho", "spearman_rho_test"], ascending=[True, False])
    )
    out_path = os.path.join(args.outdir, "results_external_metrics.tsv")
    res_df.to_csv(out_path, sep="\t", index=False)

    print("\nOutputs written:")
    print(f"  {out_path}")
    print(f"  Per-target predictions_*.tsv files in {args.outdir}")


if __name__ == "__main__":
    main()
