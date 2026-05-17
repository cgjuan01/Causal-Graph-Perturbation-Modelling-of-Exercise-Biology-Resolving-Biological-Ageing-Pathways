#!/usr/bin/env python3
"""
perturb_biology_vs_topology.py
===============================
Correlation-based analysis to test whether the glycosylation vs DNA repair
diffusion hierarchy is driven by biology or network topology.

IMPORTANT: PPR diffusion is a closed-form mathematical formula — there are
no learned weights, so gradient-based attribution is not applicable here.
This script uses correlation analysis instead, which is the appropriate
method for a non-parametric diffusion model.

Three analyses are performed:

  1. Gini vs degree residual analysis
     Tests whether concentrated diffusion in glycosylation genes can be
     explained by their network degree alone. Genes above the regression
     line have more concentrated diffusion than their degree predicts.

  2. Embedding perturbation magnitude vs diffusion concentration
     Tests whether genes whose embeddings were most affected by the
     perturbation (dz_norm) also show the most concentrated diffusion.
     If topology drove the result, dz_norm and gini_ppr should be
     uncorrelated. If biology drove it, they should correlate within
     gene classes.

  3. Cross-gene class comparison on degree-normalised metrics
     Compares glycosylation vs DNA repair genes on Gini residuals
     (concentration unexplained by degree) and entropy, with
     Mann-Whitney U tests and effect sizes.

Inputs
------
--perturb_summary   perturb_summary_all.tsv from perturb_diffusion.py (Step 3a)
                    Required columns: gene, degree, gini_ppr, entropy_ppr,
                    dz_norm, emp_p_gini, emp_p_entropy, top10_ppr_mass

--perturb_dmr       perturb_summary from perturb_diffusion_with_DMRimpact.py
                    (Step 3b) — optional. If provided, adds clock-DMR mass
                    correlation analysis.

--embeddings_contrastive   embeddings_contrastive.tsv from train_contrastive.py
                           (Step 1) — optional. If provided with embeddings_mrrefined,
                           computes per-gene embedding shift from MR refinement.

--embeddings_mrrefined     embeddings_mrrefined.tsv from finetune_multitask_mr.py
                           (Step 2) — optional.

--out_dir           Output directory (default: biology_vs_topology_outputs)

Usage
-----
# Minimal (topology analysis only, from Step 3a output)
python perturb_biology_vs_topology.py \\
    --perturb_summary results/step3a_perturb/perturb_summary_all.tsv \\
    --out_dir         biology_vs_topology_outputs

# Full analysis (with embeddings and Step 3b clock-DMR output)
python perturb_biology_vs_topology.py \\
    --perturb_summary        results/step3a_perturb/perturb_summary_all.tsv \\
    --perturb_dmr            results/step3b_perturb_annotated/perturb_summary_all.tsv \\
    --embeddings_contrastive results/step1_contrastive/embeddings_contrastive.tsv \\
    --embeddings_mrrefined   results/step2_mrrefined/embeddings_mrrefined.tsv \\
    --out_dir                biology_vs_topology_outputs

Output files
------------
  gini_vs_degree.png/tsv         — Analysis 1: scatter + regression line
  diffusion_class_comparison.png/tsv  — Analysis 3: class-level comparison
  embedding_shift_vs_diffusion.png/tsv — Analysis 2: if embeddings provided
  dmr_correlation.png/tsv        — if Step 3b output provided
  summary_report.txt             — plain-language interpretation of all results
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Gene class definitions — verified against your perturb_summary_all.tsv
# ---------------------------------------------------------------------------

GLYCOSYLATION_GENES = {
    "B4GALT1", "ST6GAL1", "ST3GAL1", "ST3GAL4",
    "MGAT3", "MGAT5", "MGAT5B", "FUT8",
}

DNA_REPAIR_GENES = {
    "SIRT1", "PARP1", "RAD51",
    "SIRT2", "SIRT3", "SIRT6", "OGG1", "PRDX6",
}

# Colours consistent with paper figures
COLOUR_GLYCO   = "#4A9BB5"   # teal
COLOUR_REPAIR  = "#E05A4E"   # red
COLOUR_OTHER   = "#AAAAAA"   # grey
COLOUR_FIT     = "#333333"   # dark grey regression line


def gene_class(gene):
    if gene in GLYCOSYLATION_GENES:
        return "Glycosylation"
    if gene in DNA_REPAIR_GENES:
        return "DNA repair"
    return "Other"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def spearman_with_p(x, y):
    """Return (rho, p) from Spearman correlation, handling NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    return stats.spearmanr(x[mask], y[mask])


def mannwhitney(a, b):
    """Mann-Whitney U with effect size r = Z / sqrt(N)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n = len(a) + len(b)
    z = stats.norm.ppf(1 - p / 2) * np.sign(np.median(a) - np.median(b))
    r = abs(z) / np.sqrt(n)
    return stat, p, r


def add_class_column(df):
    df = df.copy()
    df["gene_class"] = df["gene"].apply(gene_class)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Gini vs degree
# ---------------------------------------------------------------------------

def analysis_gini_vs_degree(df, out_dir):
    """
    Scatter plot of gini_ppr vs degree with OLS regression line.
    Genes above the line have MORE concentrated diffusion than their
    degree alone predicts — evidence the result is biological not topological.
    Computes per-gene Gini residual and saves table.
    """
    print("\n--- Analysis 1: Gini vs degree ---")

    x = df["degree"].values.astype(float)
    y = df["gini_ppr"].values.astype(float)

    # OLS regression
    slope, intercept, r, p_r, _ = stats.linregress(x, y)
    y_fit = intercept + slope * x
    residuals = y - y_fit

    df = df.copy()
    df["gini_residual"] = residuals
    df["gini_predicted"] = y_fit

    print(f"  OLS: gini ~ degree   slope={slope:.4f}  r={r:.3f}  p={p_r:.3f}")
    print()
    for _, row in df.iterrows():
        direction = "ABOVE" if row["gini_residual"] > 0 else "below"
        print(f"  {row['gene']:10s}  degree={int(row['degree']):3d}  "
              f"gini={row['gini_ppr']:.3f}  residual={row['gini_residual']:+.3f}  {direction}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    colours = [COLOUR_GLYCO if g in GLYCOSYLATION_GENES
               else COLOUR_REPAIR if g in DNA_REPAIR_GENES
               else COLOUR_OTHER
               for g in df["gene"]]

    ax.scatter(x, y, c=colours, s=90, zorder=3, edgecolors="white", linewidths=0.5)

    # Regression line
    x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
    ax.plot(x_line, intercept + slope * x_line,
            color=COLOUR_FIT, lw=1.5, ls="--", zorder=2, label=f"OLS fit (r={r:.2f}, p={p_r:.3f})")

    # Gene labels
    for _, row in df.iterrows():
        ax.annotate(row["gene"],
                    (row["degree"], row["gini_ppr"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color="black")

    # Legend patches
    patches = [
        mpatches.Patch(color=COLOUR_GLYCO,  label="Glycosylation"),
        mpatches.Patch(color=COLOUR_REPAIR, label="DNA repair"),
    ]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color=COLOUR_FIT, ls="--", lw=1.5,
                                             label=f"OLS fit (r={r:.2f})")],
              fontsize=9, frameon=False)

    ax.set_xlabel("Network degree", fontsize=11)
    ax.set_ylabel("Gini coefficient (PPR concentration)", fontsize=11)
    ax.set_title("Diffusion concentration vs network degree\n"
                 "Genes above the line: more concentrated than degree predicts",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "gini_vs_degree.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_png}")

    out_tsv = os.path.join(out_dir, "gini_vs_degree.tsv")
    df[["gene", "gene_class", "degree", "gini_ppr", "gini_predicted",
        "gini_residual", "emp_p_gini"]].to_csv(out_tsv, sep="\t", index=False)
    print(f"  Saved: {out_tsv}")

    return df, slope, intercept, r, p_r


# ---------------------------------------------------------------------------
# Analysis 2: Embedding perturbation magnitude vs diffusion concentration
# ---------------------------------------------------------------------------

def analysis_dz_vs_diffusion(df, out_dir):
    """
    Tests whether genes with larger embedding perturbation (dz_norm —
    the L2 norm of the embedding change under perturbation) also show
    more concentrated diffusion.

    dz_norm is already in perturb_summary_all.tsv — it's the impact
    score scaling that drives the PPR seed. If topology drove the result,
    dz_norm and gini should be uncorrelated. If the embedding geometry
    matters, they should correlate.
    """
    print("\n--- Analysis 2: Embedding perturbation magnitude vs diffusion ---")

    if "dz_norm" not in df.columns:
        print("  dz_norm column not found — skipping.")
        return

    x = df["dz_norm"].values.astype(float)
    y_gini = df["gini_ppr"].values.astype(float)
    y_ent  = df["entropy_ppr"].values.astype(float)

    rho_gini, p_gini = spearman_with_p(x, y_gini)
    rho_ent,  p_ent  = spearman_with_p(x, y_ent)

    print(f"  Spearman: dz_norm vs gini_ppr      rho={rho_gini:.3f}  p={p_gini:.3f}")
    print(f"  Spearman: dz_norm vs entropy_ppr   rho={rho_ent:.3f}  p={p_ent:.3f}")

    # Separate correlations per class
    for cls, colour in [("Glycosylation", COLOUR_GLYCO), ("DNA repair", COLOUR_REPAIR)]:
        sub = df[df["gene_class"] == cls]
        if len(sub) >= 3:
            r, p = spearman_with_p(
                sub["dz_norm"].values.astype(float),
                sub["gini_ppr"].values.astype(float)
            )
            print(f"  {cls:15s}: dz_norm vs gini  rho={r:.3f}  p={p:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (y, y_col, ylabel, rho, p_val) in zip(axes, [
        (y_gini, "gini_ppr",    "Gini coefficient (concentration)", rho_gini, p_gini),
        (y_ent,  "entropy_ppr", "Shannon entropy (breadth)",         rho_ent,  p_ent),
    ]):
        colours = [COLOUR_GLYCO if g in GLYCOSYLATION_GENES
                   else COLOUR_REPAIR if g in DNA_REPAIR_GENES
                   else COLOUR_OTHER
                   for g in df["gene"]]
        ax.scatter(x, y, c=colours, s=90, zorder=3,
                   edgecolors="white", linewidths=0.5)
        for _, row in df.iterrows():
            ax.annotate(row["gene"], (row["dz_norm"], row[y_col]),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)
        ax.set_xlabel("Embedding perturbation magnitude (dz_norm)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Spearman ρ = {rho:.3f}, p = {p_val:.3f}", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=COLOUR_GLYCO, label="Glycosylation"),
               mpatches.Patch(color=COLOUR_REPAIR, label="DNA repair")]
    axes[0].legend(handles=patches, fontsize=8, frameon=False)
    fig.suptitle("Embedding perturbation magnitude vs diffusion profile",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "embedding_perturbation_vs_diffusion.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_png}")

    out_tsv = os.path.join(out_dir, "embedding_perturbation_vs_diffusion.tsv")
    df[["gene", "gene_class", "dz_norm", "gini_ppr", "entropy_ppr"]].to_csv(
        out_tsv, sep="\t", index=False)
    print(f"  Saved: {out_tsv}")

    return rho_gini, p_gini, rho_ent, p_ent


# ---------------------------------------------------------------------------
# Analysis 3: Class-level comparison on degree-normalised metrics
# ---------------------------------------------------------------------------

def analysis_class_comparison(df, out_dir):
    """
    Compares glycosylation vs DNA repair genes on:
      - gini_residual (Gini unexplained by degree — from Analysis 1)
      - entropy_ppr
      - top10_ppr_mass
      - emp_p_gini (empirical significance vs degree-matched null)

    Uses Mann-Whitney U with effect size r.
    """
    print("\n--- Analysis 3: Glycosylation vs DNA repair class comparison ---")

    glyco  = df[df["gene_class"] == "Glycosylation"]
    repair = df[df["gene_class"] == "DNA repair"]

    metrics = [
        ("gini_ppr",         "Gini coefficient (raw)"),
        ("gini_residual",    "Gini residual (degree-corrected)"),
        ("entropy_ppr",      "Shannon entropy"),
        ("top10_ppr_mass",   "Top-10 PPR mass"),
        ("dz_norm",          "Embedding perturbation (dz_norm)"),
    ]

    results = []
    for col, label in metrics:
        if col not in df.columns:
            continue
        a = glyco[col].dropna().values.astype(float)
        b = repair[col].dropna().values.astype(float)
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p, r = mannwhitney(a, b)
        direction = "Glyco > Repair" if np.median(a) > np.median(b) else "Repair > Glyco"
        print(f"  {label:35s}  median_glyco={np.median(a):.3f}  "
              f"median_repair={np.median(b):.3f}  U={stat:.0f}  p={p:.3f}  r={r:.3f}  {direction}")
        results.append({
            "metric": col,
            "label": label,
            "median_glycosylation": np.median(a),
            "median_dna_repair": np.median(b),
            "mannwhitney_U": stat,
            "p_value": p,
            "effect_size_r": r,
            "direction": direction,
        })

    results_df = pd.DataFrame(results)
    out_tsv = os.path.join(out_dir, "class_comparison.tsv")
    results_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n  Saved: {out_tsv}")

    # Plot — grouped bar chart of medians
    plot_metrics = [r for r in results
                    if r["metric"] in ("gini_ppr", "gini_residual",
                                       "entropy_ppr", "top10_ppr_mass")]
    if not plot_metrics:
        return results_df

    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4 * len(plot_metrics), 4.5))
    if len(plot_metrics) == 1:
        axes = [axes]

    for ax, r in zip(axes, plot_metrics):
        bars = ax.bar(
            ["Glycosylation", "DNA repair"],
            [r["median_glycosylation"], r["median_dna_repair"]],
            color=[COLOUR_GLYCO, COLOUR_REPAIR],
            width=0.5,
            edgecolor="white",
        )
        ax.set_title(f"{r['label']}\np = {r['p_value']:.3f}, r = {r['effect_size_r']:.2f}",
                     fontsize=9)
        ax.set_ylabel("Median", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Glycosylation vs DNA repair: diffusion profile comparison",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "class_comparison.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")

    return results_df


# ---------------------------------------------------------------------------
# Analysis 4 (optional): Embedding shift from MR refinement
# ---------------------------------------------------------------------------

def analysis_embedding_shift(df, path_contrastive, path_mrrefined, out_dir):
    """
    For each gene, compute how much MR refinement moved its embedding
    from the contrastive baseline: shift = ||z_mrrefined - z_contrastive||

    Then correlate shift with gini_ppr and clock-DMR mass (if available).
    Genes whose embeddings shifted most due to MR anchoring should also
    show the most clock-aligned diffusion — if biology drove the result.
    """
    print("\n--- Analysis 4: MR embedding shift vs diffusion concentration ---")

    try:
        emb_c = pd.read_csv(path_contrastive, sep="\t")
        emb_r = pd.read_csv(path_mrrefined,   sep="\t")
    except Exception as e:
        print(f"  Could not load embeddings: {e}")
        return

    # Find embedding columns
    emb_cols_c = [c for c in emb_c.columns if c.startswith("emb_")]
    emb_cols_r = [c for c in emb_r.columns if c.startswith("emb_")]

    if not emb_cols_c or not emb_cols_r:
        print("  No emb_* columns found in embedding files — skipping.")
        return

    # Align on gene_symbol
    sym_col = "gene_symbol" if "gene_symbol" in emb_c.columns else "gene"
    emb_c = emb_c.rename(columns={sym_col: "gene"})
    emb_r = emb_r.rename(columns={sym_col: "gene"})

    merged = emb_c[["gene"] + emb_cols_c].merge(
        emb_r[["gene"] + emb_cols_r],
        on="gene", suffixes=("_c", "_r")
    )

    # Compute L2 shift
    c_mat = merged[[c + "_c" for c in emb_cols_c]].values.astype(float)
    r_mat = merged[[c + "_r" for c in emb_cols_r]].values.astype(float)
    merged["embedding_shift"] = np.linalg.norm(r_mat - c_mat, axis=1)

    # Merge with perturbation summary
    combined = df.merge(merged[["gene", "embedding_shift"]], on="gene", how="left")

    rho, p = spearman_with_p(
        combined["embedding_shift"].values.astype(float),
        combined["gini_ppr"].values.astype(float)
    )
    print(f"  Spearman: embedding_shift vs gini_ppr   rho={rho:.3f}  p={p:.3f}")
    print()
    print(combined[["gene", "gene_class", "embedding_shift", "gini_ppr"]].to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colours = [COLOUR_GLYCO if g in GLYCOSYLATION_GENES
               else COLOUR_REPAIR if g in DNA_REPAIR_GENES
               else COLOUR_OTHER
               for g in combined["gene"]]
    ax.scatter(combined["embedding_shift"], combined["gini_ppr"],
               c=colours, s=90, zorder=3, edgecolors="white", linewidths=0.5)
    for _, row in combined.iterrows():
        ax.annotate(row["gene"],
                    (row["embedding_shift"], row["gini_ppr"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7.5)
    ax.set_xlabel("Embedding shift from MR refinement (L2 norm)", fontsize=10)
    ax.set_ylabel("Gini coefficient (PPR concentration)", fontsize=10)
    ax.set_title(f"MR embedding shift vs diffusion concentration\n"
                 f"Spearman ρ = {rho:.3f}, p = {p:.3f}", fontsize=9)
    patches = [mpatches.Patch(color=COLOUR_GLYCO,  label="Glycosylation"),
               mpatches.Patch(color=COLOUR_REPAIR, label="DNA repair")]
    ax.legend(handles=patches, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "embedding_shift_vs_gini.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_png}")

    out_tsv = os.path.join(out_dir, "embedding_shift_vs_gini.tsv")
    combined[["gene", "gene_class", "embedding_shift", "gini_ppr",
              "entropy_ppr", "emp_p_gini"]].to_csv(out_tsv, sep="\t", index=False)
    print(f"  Saved: {out_tsv}")

    return combined, rho, p


# ---------------------------------------------------------------------------
# Analysis 5 (optional): Clock-DMR mass correlation (Step 3b output)
# ---------------------------------------------------------------------------

def analysis_dmr_correlation(df, path_dmr, out_dir):
    """
    If Step 3b output is provided, correlates clock-DMR mass with
    degree and Gini residual to test whether clock alignment is
    independent of network position.
    """
    print("\n--- Analysis 5: Clock-DMR mass correlation (Step 3b) ---")

    try:
        dmr = pd.read_csv(path_dmr, sep="\t")
    except Exception as e:
        print(f"  Could not load Step 3b summary: {e}")
        return

    # Find clock DMR mass columns
    dmr_mass_cols = [c for c in dmr.columns
                     if "dmr_mass" in c.lower() or "pc1_mass" in c.lower()]

    if not dmr_mass_cols:
        print("  No DMR mass columns found in Step 3b output — skipping.")
        print(f"  Available columns: {dmr.columns.tolist()}")
        return

    # Align gene column name
    gene_col = "gene" if "gene" in dmr.columns else dmr.columns[0]
    dmr = dmr.rename(columns={gene_col: "gene"})

    combined = df.merge(dmr[["gene"] + dmr_mass_cols], on="gene", how="left")

    print(f"  DMR mass columns found: {dmr_mass_cols}")
    for col in dmr_mass_cols[:4]:
        rho_deg,  p_deg  = spearman_with_p(
            combined["degree"].values.astype(float),
            combined[col].values.astype(float)
        )
        rho_gini, p_gini = spearman_with_p(
            combined["gini_residual"].values.astype(float)
            if "gini_residual" in combined.columns
            else combined["gini_ppr"].values.astype(float),
            combined[col].values.astype(float)
        )
        print(f"  {col:40s}  vs degree: rho={rho_deg:.3f} p={p_deg:.3f}  "
              f"vs gini_residual: rho={rho_gini:.3f} p={p_gini:.3f}")

    out_tsv = os.path.join(out_dir, "dmr_mass_correlation.tsv")
    combined[["gene", "gene_class", "degree", "gini_ppr"] + dmr_mass_cols].to_csv(
        out_tsv, sep="\t", index=False)
    print(f"\n  Saved: {out_tsv}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(df, out_dir, slope, r_degree, p_degree,
                  class_results=None):
    """Write plain-language interpretation of results."""

    glyco  = df[df["gene_class"] == "Glycosylation"]
    repair = df[df["gene_class"] == "DNA repair"]

    lines = [
        "=" * 65,
        "PERTURB BIOLOGY VS TOPOLOGY — SUMMARY REPORT",
        "=" * 65,
        "",
        "QUESTION: Is the glycosylation vs DNA repair diffusion hierarchy",
        "driven by biology (MR-anchored embeddings, causal signal) or by",
        "network topology (degree, clustering, graph position)?",
        "",
        "--- Analysis 1: Gini vs degree ---",
        f"  OLS slope = {slope:.4f}  r = {r_degree:.3f}  p = {p_degree:.3f}",
    ]

    if "gini_residual" in df.columns:
        glyco_res  = glyco["gini_residual"].mean()
        repair_res = repair["gini_residual"].mean()
        lines += [
            f"  Mean Gini residual (glycosylation): {glyco_res:+.3f}",
            f"  Mean Gini residual (DNA repair):    {repair_res:+.3f}",
            "",
        ]
        if glyco_res > 0 and repair_res <= 0:
            lines.append(
                "  INTERPRETATION: Glycosylation genes sit ABOVE the degree-Gini"
                " regression line (positive residuals), meaning their concentrated"
                " diffusion exceeds what their degree alone predicts. DNA repair"
                " genes sit on or below the line. This is evidence the concentration"
                " hierarchy reflects biological signal, not network position."
            )
        else:
            lines.append(
                "  INTERPRETATION: Mixed residuals — topology cannot be fully"
                " ruled out as a driver of the Gini hierarchy."
            )

    lines += ["", "--- Analysis 3: Class comparison ---"]
    if class_results is not None and len(class_results) > 0:
        for _, row in class_results.iterrows():
            sig = "significant" if row["p_value"] < 0.05 else "not significant"
            lines.append(
                f"  {row['label']:35s}  p={row['p_value']:.3f} ({sig})  "
                f"r={row['effect_size_r']:.2f}  {row['direction']}"
            )

    lines += [
        "",
        "--- Honest limitations ---",
        "  1. Only 10 genes are in this analysis — statistical power is low.",
        "     Effect sizes (r) are more informative than p-values here.",
        "  2. Degree-matched null models (in perturb_diffusion.py) are the",
        "     primary defence against the topology artefact. This script",
        "     provides complementary evidence.",
        "  3. The Gini residual analysis controls for degree but not for",
        "     clustering coefficient or community structure. A rewired-network",
        "     null (preserving both degree and clustering) would be the",
        "     strongest test.",
        "",
        "=" * 65,
    ]

    report_path = os.path.join(out_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved summary report: {report_path}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlation-based biology vs topology analysis for PPR diffusion."
    )
    parser.add_argument("--perturb_summary", required=True,
                        help="perturb_summary_all.tsv from perturb_diffusion.py (Step 3a)")
    parser.add_argument("--perturb_dmr", default=None,
                        help="Step 3b summary with clock-DMR mass columns (optional)")
    parser.add_argument("--embeddings_contrastive", default=None,
                        help="embeddings_contrastive.tsv from train_contrastive.py (optional)")
    parser.add_argument("--embeddings_mrrefined", default=None,
                        help="embeddings_mrrefined.tsv from finetune_multitask_mr.py (optional)")
    parser.add_argument("--out_dir", default="biology_vs_topology_outputs",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # Load and label
    df = pd.read_csv(args.perturb_summary, sep="\t")
    df = add_class_column(df)

    print(f"\nLoaded {len(df)} genes from {args.perturb_summary}")
    print(f"Gene classes: {df['gene_class'].value_counts().to_dict()}")

    # Run analyses
    df, slope, intercept, r_deg, p_deg = analysis_gini_vs_degree(df, args.out_dir)

    analysis_dz_vs_diffusion(df, args.out_dir)

    class_results = analysis_class_comparison(df, args.out_dir)

    if args.embeddings_contrastive and args.embeddings_mrrefined:
        analysis_embedding_shift(
            df,
            args.embeddings_contrastive,
            args.embeddings_mrrefined,
            args.out_dir,
        )
    else:
        print("\n  Skipping embedding shift analysis — provide both "
              "--embeddings_contrastive and --embeddings_mrrefined to enable.")

    if args.perturb_dmr:
        analysis_dmr_correlation(df, args.perturb_dmr, args.out_dir)
    else:
        print("\n  Skipping clock-DMR correlation — provide --perturb_dmr "
              "(Step 3b output) to enable.")

    write_summary(df, args.out_dir, slope, r_deg, p_deg, class_results)

    print("\n=== Analysis complete ===")
    print(f"All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
