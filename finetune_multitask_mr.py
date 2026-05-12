#!/usr/bin/env python3
"""
Step 2 — MR-informed masked multi-task embedding refinement.

Refines pre-trained gene embeddings by training a lightweight projection network to predict
standardised Mendelian randomisation (MR) effect sizes (beta_std) across four omic layers
simultaneously: proteomics, CpG methylation, glycomics, and single-cell transcriptomics.

Key design choices
------------------
* Masking:   MR targets are sparse and unevenly distributed across layers.  Loss is computed
             only for genes that have a valid (non-NaN) MR estimate in each layer; genes
             with missing targets are masked out rather than imputed.
* Huber loss: Robust to outlier effect sizes that arise from MR heterogeneity.
* Stability regularisation: An L2 penalty on ||Z' - Z||² prevents the projection from
             drifting far from the self-supervised pretraining geometry, preserving global
             topological structure encoded during contrastive pretraining.
* No ageing labels: No ageing-related information is used during refinement; associations
             with ageing clocks observed downstream therefore emerge from network topology
             and causal exercise signals rather than supervised optimisation.

Inputs
------
--gene_index    paper2_gene_to_index.tsv         (gene_symbol, index)
--mr_targets    paper2_targets_MR_multitask_beta_std.tsv
                (gene_symbol, y_protein_beta_std, y_cpg_beta_std,
                              y_glycan_beta_std,  y_sc_beta_std)
--embeddings    embeddings_contrastive.tsv        (gene_symbol, index, emb_*)

Outputs (under --outdir)
------------------------
embeddings_mrrefined.tsv        gene_symbol, index, emb_1 … emb_D
training_log_mrrefined.tsv      epoch, loss_total, loss_shift, loss_<task>
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class Projection(nn.Module):
    """
    Lightweight MLP projection Z → Z'.

    Keeps embedding dimensionality fixed (in_dim == out_dim) so refined embeddings
    can be used as drop-in replacements for the base contrastive embeddings.
    """

    def __init__(self, dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RegressionHead(nn.Module):
    """Per-task MLP regression head predicting a scalar MR effect size."""

    def __init__(self, in_dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Huber loss computed only over unmasked (valid) genes.

    Parameters
    ----------
    pred   : [N, 1] predictions
    target : [N, 1] targets (NaN entries have been zero-filled; mask excludes them)
    mask   : [N]    boolean tensor, True where MR target is available
    delta  : Huber transition point
    """
    if mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    return F.huber_loss(pred[mask], target[mask], delta=delta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Canonical MR task names and expected column suffixes
TASKS = ["protein", "cpg", "glycan", "sc"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MR-informed masked multi-task embedding refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gene_index",  required=True, help="Gene-to-index mapping TSV.")
    ap.add_argument("--mr_targets",  required=True, help="MR beta_std target TSV.")
    ap.add_argument("--embeddings",  required=True, help="Base embedding TSV (emb_* columns).")
    ap.add_argument("--outdir",      required=True, help="Output directory.")

    ap.add_argument("--proj_hid",    type=int,   default=256,  help="Projection hidden width.")
    ap.add_argument("--head_hid",    type=int,   default=128,  help="Regression head hidden width.")
    ap.add_argument("--dropout",     type=float, default=0.2,  help="Dropout probability.")
    ap.add_argument("--epochs",      type=int,   default=500,  help="Training epochs.")
    ap.add_argument("--lr",          type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--weight_decay",type=float, default=1e-5, help="Adam weight decay.")
    ap.add_argument("--delta",       type=float, default=1.0,  help="Huber loss delta.")
    ap.add_argument(
        "--l2_embed",
        type=float,
        default=1e-4,
        help="L2 stability penalty on ||Z' - Z||²; set 0 to disable.",
    )
    ap.add_argument("--seed",    type=int,   default=42,  help="Random seed.")
    ap.add_argument("--log_every", type=int, default=10,  help="Log every N epochs.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Load & merge: gene_index ∩ embeddings ∩ MR targets
    # ------------------------------------------------------------------
    gene_df = pd.read_csv(args.gene_index, sep="\t")
    emb_df  = pd.read_csv(args.embeddings,  sep="\t")
    y_df    = pd.read_csv(args.mr_targets,  sep="\t")

    df = (
        gene_df
        .merge(emb_df, on=["gene_symbol", "index"], how="inner")
        .merge(y_df,   on="gene_symbol",             how="left")
        .sort_values("index")
        .reset_index(drop=True)
    )

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No 'emb_*' columns found in --embeddings file.")

    Z0 = torch.tensor(df[emb_cols].values.astype(np.float32), device=args.device)
    emb_dim = Z0.shape[1]

    # ------------------------------------------------------------------
    # Build per-task target tensors + availability masks
    # ------------------------------------------------------------------
    targets, masks = {}, {}
    for task in TASKS:
        col = f"y_{task}_beta_std"
        if col not in df.columns:
            raise ValueError(f"Missing MR target column '{col}' in --mr_targets.")
        vals = df[col].astype(float).values
        valid = ~np.isnan(vals)
        targets[task] = torch.tensor(
            np.where(valid, vals, 0.0), dtype=torch.float32, device=args.device
        ).view(-1, 1)
        masks[task] = torch.tensor(valid, dtype=torch.bool, device=args.device)

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    projection = Projection(dim=emb_dim, hid_dim=args.proj_hid, dropout=args.dropout).to(args.device)
    heads = nn.ModuleDict(
        {t: RegressionHead(in_dim=emb_dim, hid_dim=args.head_hid, dropout=args.dropout)
         for t in TASKS}
    ).to(args.device)

    params = list(projection.parameters()) + list(heads.parameters())
    optimiser = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        projection.train()
        heads.train()
        optimiser.zero_grad()

        Zp = projection(Z0)

        # Multi-task masked Huber loss
        task_losses = {t: masked_huber_loss(heads[t](Zp), targets[t], masks[t], args.delta)
                       for t in TASKS}
        loss_total = sum(task_losses.values())

        # Stability regularisation
        loss_shift = ((Zp - Z0) ** 2).mean() if args.l2_embed > 0 else Zp.new_tensor(0.0)
        if args.l2_embed > 0:
            loss_total = loss_total + args.l2_embed * loss_shift

        loss_total.backward()
        optimiser.step()

        if epoch == 1 or epoch % args.log_every == 0:
            row = {
                "epoch":      epoch,
                "loss_total": float(loss_total.item()),
                "loss_shift": float(loss_shift.item()),
            }
            row.update({f"loss_{t}": float(task_losses[t].item()) for t in TASKS})
            log_rows.append(row)

            task_str = "  ".join(f"{t}={task_losses[t].item():.4f}" for t in TASKS)
            print(
                f"[epoch {epoch:5d}]  total={loss_total.item():.6f}  "
                f"shift={loss_shift.item():.6f}  {task_str}"
            )

    # ------------------------------------------------------------------
    # Save refined embeddings + training log
    # ------------------------------------------------------------------
    projection.eval()
    with torch.no_grad():
        Zp = projection(Z0).detach().cpu().numpy()

    out_df = pd.DataFrame(Zp, columns=emb_cols)
    out_df.insert(0, "index",       df["index"].values)
    out_df.insert(0, "gene_symbol", df["gene_symbol"].values)

    out_emb = os.path.join(args.outdir, "embeddings_mrrefined.tsv")
    out_log = os.path.join(args.outdir, "training_log_mrrefined.tsv")

    out_df.to_csv(out_emb, sep="\t", index=False)
    pd.DataFrame(log_rows).to_csv(out_log, sep="\t", index=False)

    print("\nOutputs written:")
    print(f"  {out_emb}")
    print(f"  {out_log}")


if __name__ == "__main__":
    main()
