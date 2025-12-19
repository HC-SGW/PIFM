import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from main_sub import (
    LogitAveragingStitcher,
    build_dataloaders,
    process_subgraph_batch_for_inference,
    split_edges,
    train_subgraph_lp,
)

class DummyDenoiser(torch.nn.Module):
    def forward(self, x, adj, flags, t):
        return torch.zeros_like(adj[:, 0])


def make_args(tmp_path, adj_path):
    return SimpleNamespace(
        single_graph_path=adj_path,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        hidden_dim=8,
        num_layers=2,
        num_linears=1,
        c_init=2,
        c_hid=4,
        c_final=2,
        num_heads=2,
        conv="GCN",
        seed=0,
        drop_prob=0.2,
        train_edge_drop_p=0.3,
        k_hop=2,
        max_nodes=8,
        target_coverage=1,
        lap_pe_dim=4,
        ckpt_dir=None,
        split_seed=123,
        val_ratio=0.1,
        test_ratio=0.1,
        feature_adapter=True,
        subgraph_prior="zero",
    )


def synthetic_adj(n=16, p=0.2):
    rng = np.random.default_rng(0)
    mat = rng.random((n, n))
    mat = ((mat + mat.T) / 2) < p
    np.fill_diagonal(mat, 0)
    return mat.astype(np.float32)


def test_subgraph_training_loss_decreases(tmp_path):
    adj = synthetic_adj()
    adj_path = tmp_path / "adj.npy"
    np.save(adj_path, adj)
    args = make_args(tmp_path, str(adj_path))
    args.epochs = 2

    result = train_subgraph_lp(args)
    train_hist = result["train_loss_history"]
    assert len(train_hist) == args.epochs
    assert train_hist[-1] <= train_hist[0] + 1e-6


def test_clamp_inference_preserves_known_edges(tmp_path):
    adj = synthetic_adj(n=8, p=0.4)
    adj_path = tmp_path / "adj.npy"
    np.save(adj_path, adj)
    args = make_args(tmp_path, str(adj_path))
    args.epochs = 1
    args.n_steps = 2
    args.batch_size = 1

    # Build loaders manually to reuse processing helper
    split = split_edges(adj, args.val_ratio, args.test_ratio, args.split_seed)
    args_single = SimpleNamespace(
        batch_size=args.batch_size,
        lap_pe_dim=args.lap_pe_dim,
        k_hop=args.k_hop,
        max_nodes=args.max_nodes,
        target_coverage=args.target_coverage,
        train_edge_drop_p=args.train_edge_drop_p,
        seed=args.seed,
        subgraph_prior="zero",
    )
    train_loader, _, _, _ = build_dataloaders(split, args_single, torch.device("cpu"))
    batch = next(iter(train_loader))

    denoiser = DummyDenoiser()
    stitcher = LogitAveragingStitcher(adj.shape[0])
    args_infer = SimpleNamespace(
        n_steps=args.n_steps,
        prior_init="baseline",
        subgraph_prior="zero",
    )
    process_subgraph_batch_for_inference(
        batch,
        denoiser,
        feature_adapter=None,
        device=torch.device("cpu"),
        args=args_infer,
        adj_train=split.adj_train,
        stitcher=stitcher,
    )
    probs = stitcher.finalize()
    train_edges = split.train_edges
    if train_edges.size > 0:
        assert np.allclose(probs[train_edges[:, 0], train_edges[:, 1]], 1.0, atol=1e-6)


def test_stitch_idempotence():
    N = 5
    stitcher = LogitAveragingStitcher(N)
    mat = np.triu(np.ones((N, N)) - np.eye(N))
    probs = mat + mat.T - np.diag(np.diag(mat))
    stitcher.add_subgraph_probs(np.arange(N), probs)
    aggregated = stitcher.finalize()
    assert np.allclose(aggregated, probs, atol=1e-6)


def test_edge_split_leakage_guard(tmp_path):
    adj = synthetic_adj(n=10, p=0.3)
    split = split_edges(adj, 0.2, 0.2, seed=42)
    for edges in [split.val_edges, split.test_edges]:
        if edges.size == 0:
            continue
        assert np.all(split.adj_train[edges[:, 0], edges[:, 1]] == 0)
