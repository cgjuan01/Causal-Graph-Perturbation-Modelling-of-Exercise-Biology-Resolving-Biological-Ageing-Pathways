#!/usr/bin/env python3
"""
Step 3a — In silico gene perturbation via network diffusion (topology metrics only).

Simulates acute single-event gene-level perturbations by modifying a target gene's embedding
vector and propagating the perturbation signal across the gene interaction network using
Personalised PageRank (PPR)-style diffusion.  Outputs ranked lists of affected genes and
quantifies whether perturbation effects are localised (concentrated) or global (diffuse),
using Gini coefficient and Shannon entropy as complementary breadth metrics.

Perturbation model
------------------
For a target gene g with embedding z_g, perturbation is defined as:
    z_g' = (1 ± s) * z_g    [OE: +s,  KD: −s]
    Δ_g  = ||z_g' − z_g||₂  (scaling scalar for impact scores)

The magnitude Δ_g scales the downstream impact scores; ranking of affected genes is
determined solely by the PPR diffusion profile p, which depends on network topology.

Note: under pure PPR diffusion, the *ranking* of affected genes is identical for OE and KD
at the same |s|.  The mode label affects only impact score magnitudes via Δ_g.

Diffusion model
---------------
PPR power iteration:
    p^(t+1) = (1 − α) e_g + α D⁻¹ A p^(t)
where e_g is a one-hot seed at gene g, α is the restart probability, A is the adjacency
matrix, and D is the diagonal degree matrix.

Degree-matched null models
--------------------------
For each perturbed gene, null seeds with similar network degree are sampled to assess
whether observed diffusion patterns (concentration, entropy) exceed topology-driven
expectations.  Empirical two-sided p-values are computed against the null distribution.

Inputs
------
--embeddings    embeddings_mrrefined.tsv  (or embeddings_contrastive.tsv)
                TSV with columns: gene_symbol, index, emb_*
--gene_index    paper2_gene_to_index.tsv  (gene_symbol, index)
--edge_index    paper2_edge_index_int.tsv  (src, dst — integer node indices)
--gene / --genes  gene(s) to perturb (comma-separated list for --genes)

Outputs (under --outdir)
------------------------
perturb_summary_all.tsv
<GENE>_<KD|OE>/
    affected_genes.tsv
    null_degree_matched.tsv
    perturb_summary.tsv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gene_set(path: str | None) -> set | None:
    """Load a one-column TSV of gene symbols; returns None if path is None."""
    if path is None:
        return None
    gs = (
        pd.read_csv(path, sep="\t", header=None)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
    )
    return set(gs[gs != ""].values)


def load_graph(edge_path: str, n_nodes: int):
    """
    Build an undirected adjacency list and degree array from an integer edge index TSV.

    Supports headers 'src'/'dst', 'from'/'to', or falls back to the first two columns.

    Returns
    -------
    adj : list[list[int]]   — undirected adjacency list indexed by node_id
    deg : np.ndarray[int32] — node degree array
    """
    edges = pd.read_csv(edge_path, sep="\t")
    cols = set(edges.columns)

    if {"src", "dst"}.issubset(cols):
        src_arr = edges["src"].astype(int).values
        dst_arr = edges["dst"].astype(int).values
    elif {"from", "to"}.issubset(cols):
        src_arr = edges["from"].astype(int).values
        dst_arr = edges["to"].astype(int).values
    else:
        if edges.shape[1] < 2:
            raise ValueError("--edge_index must have at least 2 columns.")
        src_arr = edges.iloc[:, 0].astype(int).values
        dst_arr = edges.iloc[:, 1].astype(int).values

    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(src_arr, dst_arr):
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u].append(v)
            adj[v].append(u)  # undirected

    deg = np.array([len(nbrs) for nbrs in adj], dtype=np.int32)
    return adj, deg


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def ppr_diffusion(
    adj: list[list[int]],
    deg: np.ndarray,
    seed_idx: int,
    alpha: float = 0.85,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Personalised PageRank diffusion centred on seed_idx.

    Iteratively updates:
        p^(t+1) = (1 − α) e_seed + α Σ_{j: j→i} p^(t)[j] / deg[j]

    Parameters
    ----------
    adj      : undirected adjacency list
    deg      : node degree array
    seed_idx : index of the seeded (perturbed) gene
    alpha    : restart probability (default 0.85; higher = more global propagation)
    n_iter   : number of power iterations

    Returns
    -------
    p : [N] steady-state influence vector (sums to ~1)
    """
    n = len(adj)
    e = np.zeros(n, dtype=np.float64)
    e[seed_idx] = 1.0
    p = e.copy()

    for _ in range(n_iter):
        p_new = (1.0 - alpha) * e
        for j in range(n):
            if deg[j] == 0:
                continue
            contrib = alpha * p[j] / deg[j]
            for i in adj[j]:
                p_new[i] += contrib
        p = p_new

    return p


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------

def degree_matched_null(
    deg: np.ndarray,
    target_deg: int,
    n_null: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    Sample null seed nodes with degree close to target_deg.

    Tolerance is expanded iteratively until at least max(10, n_null) candidates are found.

    Returns
    -------
    indices  : [n_null] sampled node indices
    tol_used : tolerance value at which the sample was drawn
    """
    tol = 0
    n = len(deg)
    while True:
        cand = np.where((deg >= target_deg - tol) & (deg <= target_deg + tol))[0]
        if len(cand) >= max(10, n_null):
            break
        tol += 1
        if tol > 50:
            cand = np.arange(n)
            break
    return rng.choice(cand, size=n_null, replace=len(cand) < n_null), tol


# ---------------------------------------------------------------------------
# Breadth / concentration metrics
# ---------------------------------------------------------------------------

def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a non-negative vector (natural log)."""
    p = p.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return float("nan")
    q = p / total
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def gini_coefficient(p: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, 1 = maximally concentrated)."""
    x = np.sort(p.astype(np.float64))
    if x.sum() == 0:
        return float("nan")
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * cumx.sum() / cumx[-1]) / n)


def emp_p_two_sided(obs: float, null: np.ndarray) -> float:
    """
    Two-sided empirical p-value based on absolute deviation from the null mean.

    p_emp = (#{ |x_null − μ| ≥ |x_obs − μ| } + 1) / (n_null + 1)
    """
    null = np.asarray(null, dtype=np.float64)
    mu = float(np.mean(null))
    count = int(np.sum(np.abs(null - mu) >= np.abs(obs - mu)))
    return float((count + 1) / (len(null) + 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="In silico gene perturbation + PPR network diffusion (topology metrics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--embeddings",  required=True, help="Gene embedding TSV (gene_symbol, index, emb_*).")
    ap.add_argument("--gene_index",  required=True, help="Gene-to-index TSV (gene_symbol, index).")
    ap.add_argument("--edge_index",  required=True, help="Edge index TSV (integer src/dst).")

    gene_group = ap.add_mutually_exclusive_group(required=True)
    gene_group.add_argument("--gene",  default=None, help="Single gene symbol.")
    gene_group.add_argument("--genes", default=None, help="Comma-separated gene symbols.")

    ap.add_argument("--mode",     choices=["knockdown", "overexpression"], default="overexpression")
    ap.add_argument("--strength", type=float, default=1.0, help="Perturbation strength s.")
    ap.add_argument("--alpha",    type=float, default=0.85, help="PPR restart probability.")
    ap.add_argument("--n_iter",   type=int,   default=50,  help="PPR power iterations.")
    ap.add_argument("--null_n",   type=int,   default=2000, help="Null seed sample size.")
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--gene_set", default=None, help="Optional TSV of gene symbols for module overlap.")
    ap.add_argument("--topk",     type=int,   default=200, help="Top-K neighbourhood size for overlap.")
    ap.add_argument("--outdir",   required=True)
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve gene list
    if args.genes is not None:
        pert_genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        pert_genes = [args.gene.strip()]

    gene_set = load_gene_set(args.gene_set)

    # ------------------------------------------------------------------
    # Load embeddings + gene index
    # ------------------------------------------------------------------
    gene_df = pd.read_csv(args.gene_index, sep="\t")
    emb_df  = pd.read_csv(args.embeddings,  sep="\t")

    for df_name, df_obj in [("--gene_index", gene_df), ("--embeddings", emb_df)]:
        missing = {"gene_symbol", "index"} - set(df_obj.columns)
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    df = (
        gene_df
        .merge(emb_df, on=["gene_symbol", "index"], how="inner")
        .sort_values("index")
        .reset_index(drop=True)
    )

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No 'emb_*' columns found in embeddings file.")

    n_nodes = df.shape[0]
    Z = df[emb_cols].values.astype(np.float64)

    gene_to_index = dict(zip(df["gene_symbol"].values, df["index"].values))
    index_to_row  = {int(idx): i for i, idx in enumerate(df["index"].values)}

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    adj, deg = load_graph(args.edge_index, n_nodes)

    # ------------------------------------------------------------------
    # Perturbation loop
    # ------------------------------------------------------------------
    all_summaries = []

    for pgene in pert_genes:
        if pgene not in gene_to_index:
            print(f"[WARN] Gene '{pgene}' not found — skipping.", file=sys.stderr)
            continue

        g_idx = int(gene_to_index[pgene])
        if g_idx not in index_to_row:
            print(f"[WARN] Index for '{pgene}' not in row map — skipping.", file=sys.stderr)
            continue

        g_row = index_to_row[g_idx]
        z0 = Z[g_row].copy()

        # embedding perturbation
        z1 = (1.0 + args.strength) * z0 if args.mode == "overexpression" else (1.0 - args.strength) * z0
        delta_norm = float(np.linalg.norm(z1 - z0))
        if delta_norm == 0.0:
            print(f"[WARN] Δ_g = 0 for '{pgene}'; increase --strength.", file=sys.stderr)

        # PPR diffusion seeded at the perturbed gene
        p = ppr_diffusion(adj, deg, seed_idx=g_idx, alpha=args.alpha, n_iter=args.n_iter)

        # Ranked output table
        tag = "KD" if args.mode == "knockdown" else "OE"
        out = pd.DataFrame({
            "gene_symbol":    df["gene_symbol"].values,
            "index":          df["index"].values,
            "degree":         deg,
            "ppr_mass":       p,
            "diffusion_score": p * delta_norm,
        }).sort_values("diffusion_score", ascending=False)

        subdir = os.path.join(args.outdir, f"{pgene}_{tag}")
        os.makedirs(subdir, exist_ok=True)
        out.to_csv(os.path.join(subdir, "affected_genes.tsv"), sep="\t", index=False)

        # Concentration / breadth
        top10_mass  = float(out["ppr_mass"].head(10).sum())
        top50_mass  = float(out["ppr_mass"].head(50).sum())
        top200_mass = float(out["ppr_mass"].head(200).sum())
        ent  = shannon_entropy(p)
        gini = gini_coefficient(p)

        # Module overlap (optional)
        overlap = float("nan")
        frac    = float("nan")
        if gene_set is not None:
            top_genes = out["gene_symbol"].head(args.topk).astype(str).values
            overlap   = int(sum(g in gene_set for g in top_genes))
            frac      = overlap / args.topk

        # Degree-matched null
        target_deg = int(deg[g_idx])
        null_idx, tol_used = degree_matched_null(deg, target_deg, args.null_n, rng)

        null_top10, null_top50, null_ent, null_gini = [], [], [], []
        for sidx in null_idx:
            pp = ppr_diffusion(adj, deg, seed_idx=int(sidx), alpha=args.alpha, n_iter=args.n_iter)
            ms = np.sort(pp)[::-1]
            null_top10.append(float(ms[:10].sum()))
            null_top50.append(float(ms[:50].sum()))
            null_ent.append(shannon_entropy(pp))
            null_gini.append(gini_coefficient(pp))

        null_df = pd.DataFrame({
            "target_gene":    pgene,
            "target_degree":  target_deg,
            "tol_used":       tol_used,
            "null_top10_mass": null_top10,
            "null_top50_mass": null_top50,
            "null_entropy":   null_ent,
            "null_gini":      null_gini,
        })
        null_df.to_csv(os.path.join(subdir, "null_degree_matched.tsv"), sep="\t", index=False)

        # Empirical p-values
        summ = {
            "gene":              pgene,
            "mode":              args.mode,
            "strength":          args.strength,
            "alpha":             args.alpha,
            "n_iter":            args.n_iter,
            "degree":            target_deg,
            "delta_norm":        delta_norm,
            "top10_ppr_mass":    top10_mass,
            "top50_ppr_mass":    top50_mass,
            "top200_ppr_mass":   top200_mass,
            "entropy_ppr":       ent,
            "gini_ppr":          gini,
            "null_n":            args.null_n,
            "null_tol_used":     tol_used,
            "emp_p_top10_mass":  emp_p_two_sided(top10_mass,  np.array(null_top10)),
            "emp_p_top50_mass":  emp_p_two_sided(top50_mass,  np.array(null_top50)),
            "emp_p_entropy":     emp_p_two_sided(ent,         np.array(null_ent)),
            "emp_p_gini":        emp_p_two_sided(gini,        np.array(null_gini)),
            "topk":              args.topk,
            "module_overlap_topk": overlap,
            "module_frac_topk":  frac,
            "outdir_gene":       subdir,
        }
        pd.DataFrame([summ]).to_csv(os.path.join(subdir, "perturb_summary.tsv"), sep="\t", index=False)
        all_summaries.append(summ)

        print(f"[OK] {pgene}: written to {subdir}", file=sys.stderr)

    if not all_summaries:
        raise RuntimeError("No genes were processed (none found in the embeddings table).")

    summ_path = os.path.join(args.outdir, "perturb_summary_all.tsv")
    pd.DataFrame(all_summaries).to_csv(summ_path, sep="\t", index=False)

    print("\nOutputs written:")
    print(f"  {summ_path}")
    print(f"  Per-gene folders under {args.outdir}")


if __name__ == "__main__":
    main()
