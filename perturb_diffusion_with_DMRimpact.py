#!/usr/bin/env python3
"""
Step 3b — In silico perturbation with quantitative ageing-clock annotation.

Extends the base diffusion perturbation script (perturb_diffusion.py) with three families
of ageing-clock impact metrics, each evaluated against degree-matched null models:

(A) Epigenetic clock DMR gene sets
    Horvath, Hannum, PhenoAge, DunedinPACE.
    Metrics: diffusion mass captured in the top-K neighbourhood (total and top-K).
    Optional per-gene weighting by n_DMRs or mean_abs_delta methylation change.

(B) Proteomic ageing PC1 loadings
    Metric: diffusion-mass-weighted sum of |PC1 loading| over all genes in the network.

(C) Glyco7 validated glycosylation enzyme panel
    Default panel: B4GALT1, ST6GAL1, ST3GAL1, ST3GAL4, MGAT3, MGAT5, MGAT5B.
    Metric: diffusion mass captured by the panel (total and top-K).

All annotation-level metrics are evaluated against degree-matched null distributions and
reported with two-sided empirical p-values, controlling for topology-driven confounding.

Note on OE vs KD
----------------
Under pure PPR diffusion, gene *ranking* is identical for OE and KD at the same |s|.
The perturbation mode affects only the impact score magnitude (via Δ_g); all clock-
alignment metrics depend on ranking and are therefore symmetric across modes.

Inputs
------
--nodes         TSV with column: gene_symbol  (1 row per gene, no duplicates)
--edges         TSV edge list: from/to or src/dst gene symbols (or first 2 columns)
--genes         Comma-separated primary gene list to perturb
--genes_extra   Comma-separated supplementary gene list (optional)

Ageing annotation inputs (all optional)
--dmr_horvath, --dmr_hannum, --dmr_phenoage, --dmr_dunedinpace
                TSV with columns: gene_symbol, n_DMRs, mean_abs_delta
--pc1_proteomic TSV with columns: Gene, PC1_abs_mean_loading
--glyco7        Comma-separated gene symbols (default: 7-enzyme panel above)

Outputs (under --outdir)
------------------------
perturb_summary_all.tsv
skipped_genes.tsv  (if any genes were not found in the node table)
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
# Gene list helpers
# ---------------------------------------------------------------------------

def parse_gene_list(s: str | None) -> list[str]:
    """Parse a comma-separated gene string into a deduplicated list."""
    if not s or not s.strip():
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def deduplicate_ordered(lists: list[list[str]]) -> list[str]:
    """Concatenate multiple gene lists, preserving order and removing duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for lst in lists:
        for g in lst:
            if g not in seen:
                seen.add(g)
                result.append(g)
    return result


# ---------------------------------------------------------------------------
# Node / edge loaders
# ---------------------------------------------------------------------------

def load_nodes(path: str) -> pd.DataFrame:
    """
    Load the gene node table.

    Expects a TSV with at least a 'gene_symbol' column, one row per gene (no duplicates).
    Assigns canonical 0..N-1 node IDs.

    Returns
    -------
    DataFrame with columns: gene_symbol, node_id
    """
    df = pd.read_csv(path, sep="\t")
    if "gene_symbol" not in df.columns:
        raise ValueError(f"--nodes missing required column 'gene_symbol': {path}")

    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df = df[df["gene_symbol"] != ""].copy()

    dups = df.loc[df["gene_symbol"].duplicated(), "gene_symbol"].head(10).tolist()
    if dups:
        raise ValueError(
            f"--nodes contains duplicated gene_symbol rows (examples: {dups}). "
            "Use a CLEAN nodes table with exactly 1 row per gene."
        )

    df = df.reset_index(drop=True)
    df["node_id"] = np.arange(df.shape[0], dtype=np.int32)
    return df[["gene_symbol", "node_id"]].copy()


def load_edges_symbol(path: str) -> pd.DataFrame:
    """
    Load a gene-symbol edge list.

    Accepts headers: 'from'/'to', 'src'/'dst', or falls back to the first two columns.

    Returns
    -------
    DataFrame with columns: from, to  (string gene symbols, stripped)
    """
    e = pd.read_csv(path, sep="\t")
    cols = set(e.columns)

    if {"from", "to"}.issubset(cols):
        a, b = e["from"], e["to"]
    elif {"src", "dst"}.issubset(cols):
        a, b = e["src"], e["dst"]
    else:
        if e.shape[1] < 2:
            raise ValueError("--edges must have at least 2 columns.")
        a, b = e.iloc[:, 0], e.iloc[:, 1]

    out = pd.DataFrame({
        "from": a.astype(str).str.strip(),
        "to":   b.astype(str).str.strip(),
    })
    return out[(out["from"] != "") & (out["to"] != "")].copy()


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    undirected: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Convert symbol-level edges to node-ID arrays; build degree array and fast diffusion
    arrays (directed edge src/dst arrays + inverse degree).

    Returns
    -------
    deg     : [N] int32 degree array
    src_e   : [E] int32 directed edge source indices
    dst_e   : [E] int32 directed edge destination indices
    invdeg  : [N] float64 reciprocal degree (0 for isolated nodes)
    dropped : dict with counts of dropped edges (missing_gene, self_loop)
    """
    gene_to_id = dict(zip(nodes_df["gene_symbol"].values, nodes_df["node_id"].values))
    n = nodes_df.shape[0]

    src_list, dst_list = [], []
    n_missing, n_self = 0, 0

    for u_sym, v_sym in zip(edges_df["from"].values, edges_df["to"].values):
        if u_sym not in gene_to_id or v_sym not in gene_to_id:
            n_missing += 1
            continue
        u, v = int(gene_to_id[u_sym]), int(gene_to_id[v_sym])
        if u == v:
            n_self += 1
            continue
        src_list.append(u); dst_list.append(v)
        if undirected:
            src_list.append(v); dst_list.append(u)

    src_arr = np.asarray(src_list, dtype=np.int32)
    dst_arr = np.asarray(dst_list, dtype=np.int32)

    # adjacency list → degree
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src_arr, dst_arr):
        adj[u].append(v)
    deg = np.array([len(nbrs) for nbrs in adj], dtype=np.int32)

    # fast edge arrays for np.add.at diffusion
    src_e = np.concatenate([[u] * len(nbrs) for u, nbrs in enumerate(adj)]).astype(np.int32)
    dst_e = np.concatenate([nbrs for nbrs in adj if nbrs]).astype(np.int32) if any(adj) else np.array([], dtype=np.int32)

    invdeg = np.zeros(n, dtype=np.float64)
    invdeg[deg > 0] = 1.0 / deg[deg > 0]

    return deg, src_e, dst_e, invdeg, {"missing_gene": n_missing, "self_loop": n_self}


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def ppr_diffusion_fast(
    src_e: np.ndarray,
    dst_e: np.ndarray,
    invdeg: np.ndarray,
    seed_idx: int,
    n_nodes: int,
    alpha: float = 0.85,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Vectorised PPR diffusion using directed edge arrays and np.add.at.

    Significantly faster than the adjacency-list loop in perturb_diffusion.py for large
    graphs; recommended when iterating over many null seeds.
    """
    e = np.zeros(n_nodes, dtype=np.float64)
    e[seed_idx] = 1.0
    p = e.copy()

    for _ in range(n_iter):
        p_new = (1.0 - alpha) * e
        np.add.at(p_new, dst_e, alpha * p[src_e] * invdeg[src_e])
        p = p_new

    return p


# ---------------------------------------------------------------------------
# Null model + statistics
# ---------------------------------------------------------------------------

def degree_matched_null(
    deg: np.ndarray,
    target_deg: int,
    n_null: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample null seeds with similar degree; expand tolerance until enough candidates."""
    tol = 0
    while True:
        cand = np.where((deg >= target_deg - tol) & (deg <= target_deg + tol))[0]
        if len(cand) >= max(10, n_null):
            break
        tol += 1
        if tol > 50:
            cand = np.arange(len(deg))
            break
    return rng.choice(cand, size=n_null, replace=len(cand) < n_null), tol


def shannon_entropy(p: np.ndarray) -> float:
    p = p.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return float("nan")
    q = p / total
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def gini_coefficient(p: np.ndarray) -> float:
    x = np.sort(p.astype(np.float64))
    if x.sum() == 0:
        return float("nan")
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * cumx.sum() / cumx[-1]) / n)


def emp_p_two_sided(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=np.float64)
    if null.size == 0 or np.isnan(obs):
        return float("nan")
    mu = float(np.mean(null))
    count = int(np.sum(np.abs(null - mu) >= np.abs(obs - mu)))
    return float((count + 1) / (len(null) + 1))


# ---------------------------------------------------------------------------
# DMR helpers
# ---------------------------------------------------------------------------

def load_dmr_table(path: str) -> pd.DataFrame:
    """Load a clock DMR table; validates required columns."""
    df = pd.read_csv(path, sep="\t")
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df = df[df["gene_symbol"] != ""].copy()
    for col in ["gene_symbol", "n_DMRs", "mean_abs_delta"]:
        if col not in df.columns:
            raise ValueError(f"DMR file missing required column '{col}': {path}")
    df["n_DMRs"]          = pd.to_numeric(df["n_DMRs"],          errors="coerce")
    df["mean_abs_delta"]  = pd.to_numeric(df["mean_abs_delta"],  errors="coerce")
    return df


def make_weight_map(dmr_df: pd.DataFrame, weight_col: str) -> dict[str, float]:
    w = dmr_df[["gene_symbol", weight_col]].dropna().copy()
    return dict(zip(w["gene_symbol"].values, w[weight_col].astype(float).values))


def set_mass(
    p_series: pd.Series,
    top_genes: np.ndarray,
    gene_set: set,
) -> tuple[float, float]:
    """Unweighted diffusion mass captured by a gene set (total and within top-K)."""
    in_set  = [g for g in gene_set if g in p_series.index]
    topk_in = [g for g in top_genes if g in gene_set]
    total = float(p_series.loc[in_set].sum()) if in_set else 0.0
    topk  = float(p_series.loc[topk_in].sum()) if topk_in else 0.0
    return total, topk


def set_mass_weighted(
    p_series: pd.Series,
    top_genes: np.ndarray,
    gene_set: set,
    weight_map: dict[str, float],
) -> tuple[float, float]:
    """Weighted diffusion mass captured by a gene set (total and within top-K)."""
    in_set  = [g for g in gene_set if g in p_series.index]
    topk_in = [g for g in top_genes if g in gene_set]
    wtotal = float(sum(p_series.get(g, 0.0) * weight_map.get(g, 0.0) for g in in_set))
    wtopk  = float(sum(p_series.get(g, 0.0) * weight_map.get(g, 0.0) for g in topk_in))
    return wtotal, wtopk


# ---------------------------------------------------------------------------
# Proteomic PC1 helpers
# ---------------------------------------------------------------------------

def load_pc1_table(path: str) -> pd.DataFrame:
    """Load proteomic PC1 loading table; validates required columns."""
    df = pd.read_csv(path, sep="\t")
    for col in ["Gene", "PC1_abs_mean_loading"]:
        if col not in df.columns:
            raise ValueError(f"PC1 file missing required column '{col}': {path}")
    df["Gene"] = df["Gene"].astype(str).str.strip()
    df = df[df["Gene"] != ""].copy()
    df["PC1_abs_mean_loading"] = pd.to_numeric(df["PC1_abs_mean_loading"], errors="coerce")
    return df.dropna(subset=["PC1_abs_mean_loading"])


def pc1_impact(
    p_series: pd.Series,
    top_genes: np.ndarray,
    pc1_weights: dict[str, float],
) -> tuple[float, float]:
    """PC1 impact = Σ p_i × |PC1_loading_i| (total and within top-K)."""
    total = float(sum(p_series.get(g, 0.0) * w for g, w in pc1_weights.items()))
    topk  = float(sum(p_series.get(g, 0.0) * pc1_weights.get(g, 0.0)
                       for g in top_genes if g in pc1_weights))
    return total, topk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Default perturbation genes (glycosylation enzymes + DNA repair panel)
_DEFAULT_GENES       = "B4GALT1,ST6GAL1,ST3GAL1,ST3GAL4,MGAT3,MGAT5,MGAT5B,SIRT1,PARP1,RAD51"
_DEFAULT_GENES_EXTRA = "SIRT3,SIRT6,OGG1"
_DEFAULT_GLYCO7      = "B4GALT1,ST6GAL1,ST3GAL1,ST3GAL4,MGAT3,MGAT5,MGAT5B"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="In silico perturbation + PPR diffusion with ageing-clock annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Graph
    ap.add_argument("--nodes", required=True, help="Node TSV: unique gene_symbol per row.")
    ap.add_argument("--edges", required=True, help="Edge TSV: from/to or src/dst gene symbols.")

    # Genes
    ap.add_argument("--genes",       default=None, help=f"Primary gene list (default: {_DEFAULT_GENES}).")
    ap.add_argument("--genes_extra", default=None, help=f"Supplementary genes (default: {_DEFAULT_GENES_EXTRA}).")
    ap.add_argument("--mode",     choices=["knockdown", "overexpression"], default="overexpression")
    ap.add_argument("--strength", type=float, default=1.0)

    # Diffusion
    ap.add_argument("--alpha",  type=float, default=0.85, help="PPR restart probability.")
    ap.add_argument("--n_iter", type=int,   default=50,   help="PPR power iterations.")

    # Null model
    ap.add_argument("--null_n",         type=int, default=2000)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--topk",           type=int, default=200, help="Top-K neighbourhood size.")
    ap.add_argument("--progress_every", type=int, default=200, help="Log null progress every N samples.")

    # Ageing annotations (all optional)
    ap.add_argument("--dmr_horvath",    default=None, help="Horvath DMR TSV.")
    ap.add_argument("--dmr_hannum",     default=None, help="Hannum DMR TSV.")
    ap.add_argument("--dmr_phenoage",   default=None, help="PhenoAge DMR TSV.")
    ap.add_argument("--dmr_dunedinpace",default=None, help="DunedinPACE DMR TSV.")
    ap.add_argument(
        "--dmr_weight",
        choices=["none", "n_DMRs", "mean_abs_delta"],
        default="none",
        help="Per-gene DMR weighting scheme.",
    )
    ap.add_argument("--pc1_proteomic", default=None,
                    help="Proteomic PC1 TSV (columns: Gene, PC1_abs_mean_loading).")
    ap.add_argument("--glyco7", default=_DEFAULT_GLYCO7,
                    help="Comma-separated Glyco7 enzyme panel.")

    ap.add_argument("--outdir", required=True)
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Resolve perturbation gene list
    # ------------------------------------------------------------------
    primary = parse_gene_list(args.genes)       or parse_gene_list(_DEFAULT_GENES)
    extra   = parse_gene_list(args.genes_extra) or parse_gene_list(_DEFAULT_GENES_EXTRA)
    pert_genes = deduplicate_ordered([primary, extra])

    glyco7_set = set(parse_gene_list(args.glyco7))

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    nodes_df = load_nodes(args.nodes)
    edges_df = load_edges_symbol(args.edges)
    node_genes = set(nodes_df["gene_symbol"].values)

    deg, src_e, dst_e, invdeg, dropped = build_graph(nodes_df, edges_df, undirected=True)
    n_nodes = nodes_df.shape[0]

    print(f"[INFO] nodes={n_nodes}  raw_edges={len(edges_df)}", flush=True)
    if dropped["missing_gene"] or dropped["self_loop"]:
        print(f"[WARN] dropped edges: missing_gene={dropped['missing_gene']}  self_loop={dropped['self_loop']}", flush=True)
    print(f"[INFO] directed_edge_pairs={len(src_e)}  alpha={args.alpha}  n_iter={args.n_iter}  null_n={args.null_n}", flush=True)

    gene_to_id   = dict(zip(nodes_df["gene_symbol"].values, nodes_df["node_id"].values))
    id_to_gene   = dict(zip(nodes_df["node_id"].values,    nodes_df["gene_symbol"].values))
    all_genes    = np.array([id_to_gene[i] for i in range(n_nodes)], dtype=object)

    # ------------------------------------------------------------------
    # Load ageing annotations
    # ------------------------------------------------------------------
    dmr_paths = {
        "Horvath":     args.dmr_horvath,
        "Hannum":      args.dmr_hannum,
        "PhenoAge":    args.dmr_phenoage,
        "DunedinPACE": args.dmr_dunedinpace,
    }
    dmr_sets, dmr_weight_maps = {}, {}
    for clock, path in dmr_paths.items():
        if path is None:
            continue
        dtab = load_dmr_table(path)
        gset = {g for g in dtab["gene_symbol"].values if g in node_genes}
        dmr_sets[clock] = gset
        if args.dmr_weight != "none":
            dmr_weight_maps[clock] = make_weight_map(dtab, args.dmr_weight)

    if dmr_sets:
        print(f"[INFO] Loaded DMR clocks: {', '.join(dmr_sets)} (weight={args.dmr_weight})", flush=True)
    else:
        print("[INFO] No DMR clock files provided.", flush=True)

    pc1_weights = None
    if args.pc1_proteomic is not None:
        pc1_df = load_pc1_table(args.pc1_proteomic)
        pc1_df = pc1_df[pc1_df["Gene"].isin(node_genes)].copy()
        pc1_weights = dict(zip(pc1_df["Gene"].values, pc1_df["PC1_abs_mean_loading"].astype(float).values))
        print(f"[INFO] Loaded proteomic PC1 genes (in-graph): {len(pc1_weights)}", flush=True)
    else:
        print("[INFO] No PC1 file provided.", flush=True)

    glyco7_in_graph = {g for g in glyco7_set if g in node_genes}
    missing_g7 = sorted(glyco7_set - glyco7_in_graph)
    print(f"[INFO] Glyco7 requested={len(glyco7_set)}  in_graph={len(glyco7_in_graph)}", flush=True)
    if missing_g7:
        print(f"[WARN] Glyco7 genes absent from graph: {', '.join(missing_g7)}", flush=True)

    # ------------------------------------------------------------------
    # Perturbation loop
    # ------------------------------------------------------------------
    all_summ: list[dict] = []
    skipped:  list[dict] = []

    for gi, pgene in enumerate(pert_genes, 1):
        print(f"\n[INFO] ({gi}/{len(pert_genes)}) {pgene}", flush=True)

        if pgene not in gene_to_id:
            skipped.append({"gene": pgene, "reason": "not_in_nodes"})
            print(f"[WARN] '{pgene}' not in nodes table — skipping.", file=sys.stderr, flush=True)
            continue

        seed_id = int(gene_to_id[pgene])
        tag = "KD" if args.mode == "knockdown" else "OE"
        delta_norm = float(abs(args.strength))   # bookkeeping scaling under pure PPR

        # PPR diffusion
        p = ppr_diffusion_fast(src_e, dst_e, invdeg, seed_id, n_nodes, args.alpha, args.n_iter)

        # Ranked output
        out = pd.DataFrame({
            "gene_symbol":    all_genes,
            "node_id":        np.arange(n_nodes, dtype=np.int32),
            "degree":         deg,
            "ppr_mass":       p,
            "diffusion_score": p * delta_norm,
        }).sort_values("ppr_mass", ascending=False).reset_index(drop=True)

        subdir = os.path.join(args.outdir, f"{pgene}_{tag}")
        os.makedirs(subdir, exist_ok=True)
        out.to_csv(os.path.join(subdir, "affected_genes.tsv"), sep="\t", index=False)

        # Breadth / concentration
        top10  = float(out["ppr_mass"].head(10).sum())
        top50  = float(out["ppr_mass"].head(50).sum())
        top200 = float(out["ppr_mass"].head(200).sum())
        ent  = shannon_entropy(p)
        gini = gini_coefficient(p)

        p_series    = pd.Series(out["ppr_mass"].values, index=out["gene_symbol"].astype(str).values)
        topk_genes  = out["gene_symbol"].head(args.topk).astype(str).values

        summ: dict = {
            "gene": pgene, "mode": args.mode, "strength": args.strength,
            "alpha": args.alpha, "n_iter": args.n_iter,
            "degree": int(deg[seed_id]), "delta_norm": delta_norm,
            "top10_ppr_mass": top10, "top50_ppr_mass": top50, "top200_ppr_mass": top200,
            "entropy_ppr": ent, "gini_ppr": gini,
            "null_n": args.null_n, "topk": args.topk, "outdir_gene": subdir,
        }

        # Observed ageing-clock metrics
        g7_total, g7_topk = set_mass(p_series, topk_genes, glyco7_in_graph)
        summ.update({"Glyco7_mass_total": g7_total, "Glyco7_mass_topk": g7_topk})

        if pc1_weights is not None:
            pc1_t, pc1_k = pc1_impact(p_series, topk_genes, pc1_weights)
            summ.update({"PC1_mass_total": pc1_t, "PC1_mass_topk": pc1_k})

        for clock, gset in dmr_sets.items():
            mt, mk = set_mass(p_series, topk_genes, gset)
            summ[f"{clock}_dmr_mass_total"] = mt
            summ[f"{clock}_dmr_mass_topk"]  = mk
            if args.dmr_weight != "none":
                wmap = dmr_weight_maps.get(clock, {})
                wmt, wmk = set_mass_weighted(p_series, topk_genes, gset, wmap)
                summ[f"{clock}_dmr_wmass_total"] = wmt
                summ[f"{clock}_dmr_wmass_topk"]  = wmk

        # ------------------------------------------------------------------
        # Degree-matched null model
        # ------------------------------------------------------------------
        target_deg = int(deg[seed_id])
        null_idx, tol_used = degree_matched_null(deg, target_deg, args.null_n, rng)
        summ["null_tol_used"] = tol_used

        null_top10, null_top50, null_ent, null_gini       = [], [], [], []
        null_g7_total, null_g7_topk                       = [], []
        null_pc1_total, null_pc1_topk                     = [], []
        null_dmr: dict[str, dict[str, list]] = {
            c: {"mt": [], "mk": [], "wmt": [], "wmk": []} for c in dmr_sets
        }

        for ni, sidx in enumerate(null_idx, 1):
            if args.progress_every and ni % args.progress_every == 0:
                print(f"  [INFO] null {ni}/{len(null_idx)}", flush=True)

            pp = ppr_diffusion_fast(src_e, dst_e, invdeg, int(sidx), n_nodes, args.alpha, args.n_iter)
            ms = np.sort(pp)[::-1]

            null_top10.append(float(ms[:10].sum()))
            null_top50.append(float(ms[:50].sum()))
            null_ent.append(shannon_entropy(pp))
            null_gini.append(gini_coefficient(pp))

            pp_ser      = pd.Series(pp, index=all_genes)
            topk_null   = all_genes[np.argsort(pp)[::-1][:args.topk]]

            gt, gk = set_mass(pp_ser, topk_null, glyco7_in_graph)
            null_g7_total.append(gt); null_g7_topk.append(gk)

            if pc1_weights is not None:
                pt, pk = pc1_impact(pp_ser, topk_null, pc1_weights)
                null_pc1_total.append(pt); null_pc1_topk.append(pk)

            for clock, gset in dmr_sets.items():
                mt, mk = set_mass(pp_ser, topk_null, gset)
                null_dmr[clock]["mt"].append(mt); null_dmr[clock]["mk"].append(mk)
                if args.dmr_weight != "none":
                    wmap = dmr_weight_maps.get(clock, {})
                    wmt, wmk = set_mass_weighted(pp_ser, topk_null, gset, wmap)
                    null_dmr[clock]["wmt"].append(wmt); null_dmr[clock]["wmk"].append(wmk)

        # Save null table
        null_df = pd.DataFrame({
            "target_gene": pgene, "target_degree": target_deg, "tol_used": tol_used,
            "null_top10_mass": null_top10, "null_top50_mass": null_top50,
            "null_entropy": null_ent, "null_gini": null_gini,
            "null_Glyco7_mass_total": null_g7_total, "null_Glyco7_mass_topk": null_g7_topk,
        })
        if pc1_weights is not None:
            null_df["null_PC1_mass_total"] = null_pc1_total
            null_df["null_PC1_mass_topk"]  = null_pc1_topk
        for clock in dmr_sets:
            null_df[f"null_{clock}_dmr_mass_total"] = null_dmr[clock]["mt"]
            null_df[f"null_{clock}_dmr_mass_topk"]  = null_dmr[clock]["mk"]
            if args.dmr_weight != "none":
                null_df[f"null_{clock}_dmr_wmass_total"] = null_dmr[clock]["wmt"]
                null_df[f"null_{clock}_dmr_wmass_topk"]  = null_dmr[clock]["wmk"]
        null_df.to_csv(os.path.join(subdir, "null_degree_matched.tsv"), sep="\t", index=False)

        # Empirical p-values
        def ep(obs_val: float, null_list: list) -> float:
            return emp_p_two_sided(obs_val, np.asarray(null_list))

        summ.update({
            "emp_p_top10_mass": ep(top10, null_top10),
            "emp_p_top50_mass": ep(top50, null_top50),
            "emp_p_entropy":    ep(ent,   null_ent),
            "emp_p_gini":       ep(gini,  null_gini),
            "Glyco7_emp_p_mass_total": ep(g7_total, null_g7_total),
            "Glyco7_emp_p_mass_topk":  ep(g7_topk,  null_g7_topk),
        })
        if pc1_weights is not None:
            summ["PC1_emp_p_mass_total"] = ep(summ["PC1_mass_total"], null_pc1_total)
            summ["PC1_emp_p_mass_topk"]  = ep(summ["PC1_mass_topk"],  null_pc1_topk)
        for clock in dmr_sets:
            summ[f"{clock}_emp_p_dmr_mass_total"] = ep(summ[f"{clock}_dmr_mass_total"], null_dmr[clock]["mt"])
            summ[f"{clock}_emp_p_dmr_mass_topk"]  = ep(summ[f"{clock}_dmr_mass_topk"],  null_dmr[clock]["mk"])
            if args.dmr_weight != "none":
                summ[f"{clock}_emp_p_dmr_wmass_total"] = ep(summ[f"{clock}_dmr_wmass_total"], null_dmr[clock]["wmt"])
                summ[f"{clock}_emp_p_dmr_wmass_topk"]  = ep(summ[f"{clock}_dmr_wmass_topk"],  null_dmr[clock]["wmk"])

        pd.DataFrame([summ]).to_csv(os.path.join(subdir, "perturb_summary.tsv"), sep="\t", index=False)
        all_summ.append(summ)
        print(f"[OK] {pgene}: written to {subdir}", flush=True)

    # ------------------------------------------------------------------
    # Write combined summary + skipped log
    # ------------------------------------------------------------------
    if not all_summ:
        raise RuntimeError("No genes were processed (none found in nodes table).")

    summ_path = os.path.join(args.outdir, "perturb_summary_all.tsv")
    pd.DataFrame(all_summ).to_csv(summ_path, sep="\t", index=False)
    print(f"\nOutputs written:\n  {summ_path}")

    if skipped:
        skip_path = os.path.join(args.outdir, "skipped_genes.tsv")
        pd.DataFrame(skipped).to_csv(skip_path, sep="\t", index=False)
        print(f"  {skip_path}")

    print(f"  Per-gene folders under {args.outdir}")


if __name__ == "__main__":
    main()
