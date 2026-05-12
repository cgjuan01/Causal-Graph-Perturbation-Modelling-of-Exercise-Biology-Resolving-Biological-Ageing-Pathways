#!/usr/bin/env python3
"""
Step 1 — Self-supervised contrastive pretraining of gene node embeddings.

Learns task-agnostic gene representations via SimCLR-style NT-Xent loss on a gene feature
matrix.  Two stochastic views of each node are created by feature-level dropout; the encoder
is trained to maximise agreement between views of the same gene while separating all other
genes in the batch.  No ageing or MR labels are used, ensuring that downstream perturbation
analyses reflect network topology rather than supervised optimisation toward ageing outcomes.

Inputs
------
--features      paper2_nodes_features_SSL_noPAN_noMR_zscored.tsv
                TSV with columns: gene_symbol, <feature_cols...>
--edge_index    paper2_edge_index_int.tsv
                TSV with columns: src, dst  (integer node indices; kept for compatibility)
--gene_index    paper2_gene_to_index.tsv
                TSV with columns: gene_symbol, index

Outputs (under --outdir)
------------------------
embeddings_contrastive.tsv      gene_symbol, index, emb_1 … emb_D
training_log_contrastive.tsv    epoch, loss
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
# Encoder
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """
    Two-layer MLP encoder.

    A message-passing-free (MLP) baseline is used to keep the pipeline independent of
    torch_geometric.  The encoder operates on raw node feature vectors; structural context
    is implicitly captured via the KNN-derived feature matrix (multi-omic + network
    similarity features).  A GNN encoder (e.g. GAT) can be substituted here once
    graph-level operations are available.

    Parameters
    ----------
    in_dim  : input feature dimensionality
    hid_dim : hidden layer width
    out_dim : embedding dimensionality (D)
    dropout : dropout probability applied after the first linear layer
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (normalised temperature-scaled cross-entropy) loss — SimCLR style.

    For a batch of N genes, the 2N embeddings (two augmented views) form the similarity
    matrix.  Positive pairs are (z1_i, z2_i); all other pairs within the batch are treated
    as negatives.  Self-similarity is masked out before softmax.

    Parameters
    ----------
    z1, z2      : [N, D] embedding tensors (L2-normalised internally)
    temperature : softmax temperature; lower values sharpen the distribution

    Returns
    -------
    Scalar loss (mean over all 2N terms).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)               # [2N, D]
    sim = torch.matmul(z, z.t()) / temperature   # [2N, 2N]

    # mask diagonal (self-similarity)
    mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positive similarities sit on the off-diagonal of size N
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)  # [2N]
    denom = torch.logsumexp(sim, dim=1)                                  # [2N]
    return -(pos - denom).mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Self-supervised contrastive pretraining of gene embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features",    required=True,  help="Node feature TSV (gene_symbol + feature cols).")
    ap.add_argument("--edge_index",  required=True,  help="Edge index TSV (src, dst); not used by MLP encoder.")
    ap.add_argument("--gene_index",  required=True,  help="Gene-to-index mapping TSV (gene_symbol, index).")
    ap.add_argument("--outdir",      required=True,  help="Output directory.")

    ap.add_argument("--emb_dim",           type=int,   default=128,  help="Embedding dimensionality D.")
    ap.add_argument("--hid_dim",           type=int,   default=256,  help="Hidden layer width.")
    ap.add_argument("--dropout",           type=float, default=0.2,  help="Encoder dropout.")
    ap.add_argument("--feature_dropout",   type=float, default=0.2,  help="Feature-level augmentation dropout.")
    ap.add_argument("--epochs",            type=int,   default=500,  help="Training epochs.")
    ap.add_argument("--lr",                type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--weight_decay",      type=float, default=1e-5, help="Adam weight decay.")
    ap.add_argument("--temperature",       type=float, default=0.2,  help="NT-Xent temperature.")
    ap.add_argument("--seed",              type=int,   default=42,   help="Random seed.")
    ap.add_argument("--log_every",         type=int,   default=10,   help="Log loss every N epochs.")
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
    # Load & align features to canonical gene index ordering
    # ------------------------------------------------------------------
    feat_df = pd.read_csv(args.features, sep="\t")
    gene_df = pd.read_csv(args.gene_index, sep="\t")

    feat_df = (
        feat_df
        .merge(gene_df[["gene_symbol", "index"]], on="gene_symbol", how="inner")
        .sort_values("index")
        .reset_index(drop=True)
    )

    feature_cols = [c for c in feat_df.columns if c not in ("gene_symbol", "index")]
    X = torch.tensor(feat_df[feature_cols].values, dtype=torch.float32, device=args.device)

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    encoder = MLPEncoder(
        in_dim=X.shape[1],
        hid_dim=args.hid_dim,
        out_dim=args.emb_dim,
        dropout=args.dropout,
    ).to(args.device)

    optimiser = torch.optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        optimiser.zero_grad()

        # two stochastic views via independent feature dropout masks
        mask1 = (torch.rand_like(X) > args.feature_dropout).float()
        mask2 = (torch.rand_like(X) > args.feature_dropout).float()

        loss = nt_xent_loss(encoder(X * mask1), encoder(X * mask2), args.temperature)
        loss.backward()
        optimiser.step()

        if epoch == 1 or epoch % args.log_every == 0:
            log_rows.append({"epoch": epoch, "loss": float(loss.item())})
            print(f"[epoch {epoch:5d}]  loss = {loss.item():.6f}")

    # ------------------------------------------------------------------
    # Save embeddings + training log
    # ------------------------------------------------------------------
    encoder.eval()
    with torch.no_grad():
        Z = encoder(X).detach().cpu().numpy()

    emb_df = pd.DataFrame(Z, columns=[f"emb_{i + 1}" for i in range(Z.shape[1])])
    emb_df.insert(0, "index",       feat_df["index"].values)
    emb_df.insert(0, "gene_symbol", feat_df["gene_symbol"].values)

    out_emb  = os.path.join(args.outdir, "embeddings_contrastive.tsv")
    out_log  = os.path.join(args.outdir, "training_log_contrastive.tsv")

    emb_df.to_csv(out_emb,  sep="\t", index=False)
    pd.DataFrame(log_rows).to_csv(out_log, sep="\t", index=False)

    print("\nOutputs written:")
    print(f"  {out_emb}")
    print(f"  {out_log}")


if __name__ == "__main__":
    main()
