"""
Subgraph link-prediction pipeline for Prior-Informed Flow Matching (PIFM).

Provides training and inference entry points that operate on k-hop ego-net
subgraphs of a single large graph, enabling SGDM-style optimisation and
stitching-based predictions focused on masked edges.
"""

from __future__ import annotations

import os
import pickle
import yaml
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from pifm_sub import (
    LogitAveragingStitcher,
    SubgraphExpansionDataset,
    collate_subgraphs,
    laplacian_posenc,
    compute_node2vec_prior_batch,
    compute_node2vec_prior_single,
    init_global_node2vec_prior_from_adj, 
    compute_node2vec_scores_for_edges,
)
from pifm_sub.graphsage_prior import (
    init_global_graphsage_prior_from_adj,
    compute_graphsage_prior_batch,
    compute_graphsage_prior_single,
    compute_graphsage_scores_for_edges,
    init_global_graphsage_heart_prior_from_adj,
    compute_graphsage_heart_prior_batch,
    compute_graphsage_heart_prior_single,
    compute_graphsage_heart_scores_for_edges,
)
from utils.initialization import build_initial_A0_lp
from utils.tensor_utils import linear_coeffs, set_seed, sym_zero_diag_valid
from utils.evaluation import masked_upper_mse
from utils.denoiser import DenoiseNetworkA


@dataclass
class EdgeSplit:
    train_edges: np.ndarray
    val_edges: np.ndarray
    test_edges: np.ndarray
    adj_train: np.ndarray
    adj_full: np.ndarray


def default_edge_split_path(args) -> str:
    base = os.path.splitext(os.path.basename(args.single_graph_path))[0]
    # You can override via --edge_split_path if you add that to main.py
    custom = getattr(args, "edge_split_path", None)
    if custom:
        return custom
    return os.path.join(
        "outputs",
        "splits",
        f"{base}_seed{args.split_seed}_val{args.val_ratio}_test{args.test_ratio}.pkl",
    )


def save_edge_split(split: EdgeSplit, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(split, f)
    print(f"[Subgraph-LP] 💾 Saved edge split to {path}")


def load_edge_split(path: str) -> EdgeSplit:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, EdgeSplit):
        return obj
    # Backward-compatible: dict-like
    return EdgeSplit(
        train_edges=obj["train_edges"],
        val_edges=obj["val_edges"],
        test_edges=obj["test_edges"],
        adj_train=obj["adj_train"],
        adj_full=obj["adj_full"],
    )


def load_dataset_cfg_file(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    cfg_path = path.strip()
    if not cfg_path:
        return None
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Subgraph dataset cfg not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Dataset cfg must parse to a dict.")
    return data


def format_dataset_cfg(cfg: Optional[dict]) -> str:
    if not cfg:
        return "default k-hop BFS"
    ds = cfg.get("dataset", cfg)
    items = [
        f"method={ds.get('sampling_method')}",
        f"subgraph_size={ds.get('subgraph_size')}",
        f"rw={ds.get('per_node_samples_rw')}",
        f"ego={ds.get('per_node_samples_ego')}",
        f"unif={ds.get('per_node_samples_unif')}",
        f"ego_radius={ds.get('ego_sample_radius')}",
    ]
    return ", ".join(items)


def load_adjacency(path: str) -> np.ndarray:
    """Load an adjacency matrix from .npy or .pkl."""
    if path.endswith(".npy"):
        adj = np.load(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "toarray"):
            adj = obj.toarray()
        elif hasattr(obj, "todense"):
            adj = np.asarray(obj.todense())
        elif isinstance(obj, np.ndarray):
            adj = obj
        else:
            raise ValueError(f"Unsupported object in pickle: {type(obj)}")
    else:
        raise ValueError(f"Unsupported adjacency format: {path}")

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square; got shape {adj.shape}")
    adj = np.asarray(adj, dtype=np.float32)
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def split_edges(
    adj: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> EdgeSplit:
    """Perform an upper-triangular edge split into train/val/test sets."""
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    iu = np.triu_indices(adj.shape[0], k=1)
    pos_idx = np.where(adj[iu] > 0.5)[0]
    src = iu[0][pos_idx]
    dst = iu[1][pos_idx]
    edges = np.stack([src, dst], axis=1)

    rng = np.random.default_rng(seed)
    rng.shuffle(edges)

    n_total = edges.shape[0]
    n_val = int(round(n_total * val_ratio))
    n_test = int(round(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough edges for training after split.")

    train_edges = edges[:n_train]
    val_edges = edges[n_train : n_train + n_val]
    test_edges = edges[n_train + n_val :]

    def build_adj(edge_array: np.ndarray) -> np.ndarray:
        mat = np.zeros_like(adj, dtype=np.float32)
        if edge_array.size > 0:
            mat[edge_array[:, 0], edge_array[:, 1]] = 1.0
            mat[edge_array[:, 1], edge_array[:, 0]] = 1.0
        return mat

    adj_train = build_adj(train_edges)
    return EdgeSplit(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        adj_train=adj_train,
        adj_full=adj,
    )


def sample_non_edges(
    adj_full: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample non-edges (upper-triangular) without replacement."""
    iu = np.triu_indices(adj_full.shape[0], k=1)
    mask = adj_full[iu] < 0.5
    candidates = np.stack([iu[0][mask], iu[1][mask]], axis=1)
    if candidates.shape[0] < num_samples:
        raise ValueError("Not enough negative edges available for sampling.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(candidates.shape[0], size=num_samples, replace=False)
    return candidates[idx]


import numpy as np
from typing import Tuple



def make_global_masks_for_split(
    adj_full: np.ndarray,
    split_edges: np.ndarray,
    drop_p: float,
    neg_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build global edge & supervision masks for ONE split (train/val/test).

    Two regions only:
      - Masked (drop) region: random upper-tri pairs with probability drop_p.
      - Observed region: complement of the masked region.

    Returns:
      edge_mask_split       : [N,N], 1 where pairs are observed/anchored (not updated).
      supervision_mask_split: [N,N], 1 where pairs are masked/supervised (updated).
    """
    adj_full = np.asarray(adj_full, dtype=np.float32)
    N = adj_full.shape[0]
    if adj_full.shape[0] != adj_full.shape[1]:
        raise ValueError("adj_full must be square.")

    rng = np.random.default_rng(seed)

    edge_mask_split = np.zeros((N, N), dtype=np.float32)
    supervision_mask_split = np.zeros((N, N), dtype=np.float32)

    # Upper-tri view for decisions
    iu = np.triu_indices(N, k=1)
    drop_mask = rng.random(len(iu[0])) < drop_p

    # Masked region
    rows_drop = iu[0][drop_mask]
    cols_drop = iu[1][drop_mask]
    supervision_mask_split[rows_drop, cols_drop] = 1.0
    supervision_mask_split[cols_drop, rows_drop] = 1.0

    # Observed (anchored) region
    rows_keep = iu[0][~drop_mask]
    cols_keep = iu[1][~drop_mask]
    edge_mask_split[rows_keep, cols_keep] = 1.0
    edge_mask_split[cols_keep, rows_keep] = 1.0

    # no self-loops
    np.fill_diagonal(edge_mask_split, 0.0)
    np.fill_diagonal(supervision_mask_split, 0.0)

    return edge_mask_split, supervision_mask_split



class FeatureAdapter(nn.Module):
    """Map rich node features to the single-channel format expected by the denoiser."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def save_subgraph_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str = "outputs",
    filename: str = "loss_curve_subgraph.jpg",
) -> Optional[str]:
    """Plot train/val losses side-by-side and save to disk."""
    if not train_losses and not val_losses:
        print("[Subgraph-LP] ⚠️ No loss history available; skipping curve export.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    epochs_train = range(1, len(train_losses) + 1)
    epochs_val = range(1, len(val_losses) + 1)

    axes[0].plot(list(epochs_train), train_losses, marker="o", color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(list(epochs_val), val_losses, marker="o", color="#ff7f0e")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3, linestyle="--")

    # Auto-scale y-limits with padding and avoid scientific tick labels
    import matplotlib.ticker as mticker
    for ax, vals in ((axes[0], train_losses), (axes[1], val_losses)):
        ax.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        if vals:
            lo = min(vals)
            hi = max(vals)
            if lo == hi:
                pad = 0.1 * max(abs(lo), 1e-8)
            else:
                pad = max(0.05 * (hi - lo), 1e-8)
            ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("Subgraph LP Loss Curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Subgraph-LP] 🖼️ Saved loss curves → {output_path}")
    return output_path


from torch.utils.data import DataLoader
import numpy as np



def build_dataloaders(
    edge_split: EdgeSplit,
    args,
    device: torch.device,
):
    # Positional encodings (global; OK to compute from train graph or full graph)
    pe = laplacian_posenc(edge_split.adj_train, k=args.lap_pe_dim)

    N = edge_split.adj_full.shape[0]

    # ------------------------------------------------------------------
    # 1) Build per-split "full graphs" containing only that split's edges
    # ------------------------------------------------------------------
    # train: you already have this
    adj_train_only = np.asarray(edge_split.adj_train, dtype=np.float32)

    # val graph: only val_edges
    adj_val_only = np.zeros((N, N), dtype=np.float32)
    for (u, v) in edge_split.val_edges:
        u = int(u); v = int(v)
        adj_val_only[u, v] = 1.0
        adj_val_only[v, u] = 1.0

    # test graph: only test_edges
    adj_test_only = np.zeros((N, N), dtype=np.float32)
    for (u, v) in edge_split.test_edges:
        u = int(u); v = int(v)
        adj_test_only[u, v] = 1.0
        adj_test_only[v, u] = 1.0

    drop_p = float(getattr(args, "_subgraph_drop_p", getattr(args, "train_edge_drop_p", 0.1)))

    # ------------------------------------------------------------------
    # 2) Build per-split GLOBAL masks
    # ------------------------------------------------------------------
    edge_mask_train_full, sup_mask_train_full = make_global_masks_for_split(
        adj_full=edge_split.adj_full,
        split_edges=edge_split.train_edges,
        drop_p=drop_p,
        neg_ratio=args.subgraph_neg_ratio,
        seed=args.seed,
    )

    edge_centered = bool(getattr(args, "test_edge_centered_subgraphs", False))

    # Keep standard val/test masks even in edge-centered mode so hidden-edge metrics
    # have both positives and negatives. The edge-centered test dataset ignores these
    # masks when global_*_mask=None.
    edge_mask_val_full, sup_mask_val_full = make_global_masks_for_split(
        adj_full=edge_split.adj_full,
        split_edges=edge_split.val_edges,
        drop_p=drop_p,
        neg_ratio=args.subgraph_neg_ratio,
        seed=args.seed + 1,
    )

    edge_mask_test_full, sup_mask_test_full = make_global_masks_for_split(
        adj_full=edge_split.adj_full,
        split_edges=edge_split.test_edges,
        drop_p=drop_p,
        neg_ratio=args.subgraph_neg_ratio,
        seed=args.seed + 2,
    )

    # ------------------------------------------------------------------
    # 3) Datasets: each split sees its own graph + its own global masks
    # ------------------------------------------------------------------
    dataset_cfg = getattr(args, "_dataset_cfg_obj", None)
    train_edge_centered = bool(getattr(args, "train_edge_centered_subgraphs", False))

    train_dataset_cfg = dataset_cfg
    if train_edge_centered:
        # Disable SaGress for train; use edge-centered k-hop sampling/masking.
        train_dataset_cfg = None

    train_dataset = SubgraphExpansionDataset(
        adj_train_np=adj_train_only,      # ONLY train edges
        pe_global_np=pe,
        adj_full_np=edge_split.adj_full,  # ground truth
        seed_edges=edge_split.train_edges,
        split_tag="train",
        k=args.k_hop,
        max_nodes=args.max_nodes,
        target_coverage=args.target_coverage,
        drop_p=drop_p,
        seed=args.seed,
        resample=False,
        node_select_graph=getattr(args, "node_select_graph", "train"),
        global_edge_mask=None if train_edge_centered else edge_mask_train_full,
        global_supervision_mask=None if train_edge_centered else sup_mask_train_full,
        dataset_cfg=train_dataset_cfg,
        use_local_context=getattr(args, "use_local_context", True),
        edge_centered_mask_target=train_edge_centered,
    )

    # val_dataset = SubgraphExpansionDataset(
    #     adj_train_np=adj_val_only,        # ONLY val edges
    #     pe_global_np=pe,
    #     adj_full_np=edge_split.adj_full,
    #     seed_edges=edge_split.val_edges,
    #     split_tag="val",
    #     k=args.k_hop,
    #     max_nodes=args.max_nodes,
    #     target_coverage=args.target_coverage,
    #     drop_p=drop_p,
    #     seed=args.seed + 1,
    #     resample=False,
    #     node_select_graph=getattr(args, "node_select_graph", "full"),
    #     global_edge_mask=edge_mask_val_full,
    #     global_supervision_mask=sup_mask_val_full,
    #     dataset_cfg=dataset_cfg,
    #     use_local_context=getattr(args, "use_local_context", True),
    # )

    # test_dataset = SubgraphExpansionDataset(
    #     adj_train_np=adj_test_only,       # ONLY test edges
    #     pe_global_np=pe,
    #     adj_full_np=edge_split.adj_full,
    #     seed_edges=edge_split.test_edges,
    #     split_tag="test",
    #     k=args.k_hop,
    #     max_nodes=args.max_nodes,
    #     target_coverage=args.target_coverage,
    #     drop_p=drop_p,
    #     seed=args.seed + 2,
    #     resample=False,
    #     node_select_graph=getattr(args, "node_select_graph", "full"),
    #     global_edge_mask=edge_mask_test_full,
    #     global_supervision_mask=sup_mask_test_full,
    #     dataset_cfg=dataset_cfg,
    #     use_local_context=getattr(args, "use_local_context", True),
    # )
    
    
    dataset_cfg = getattr(args, "_dataset_cfg_obj", None)

    # Use train graph as the "sampling graph" for all splits
    adj_train_for_sampling = edge_split.adj_train

    val_dataset = SubgraphExpansionDataset(
        adj_train_np=adj_train_for_sampling,
        pe_global_np=pe,
        adj_full_np=edge_split.adj_full,
        seed_edges=edge_split.val_edges,
        split_tag="val",
        k=args.k_hop,
        max_nodes=args.max_nodes,
        target_coverage=getattr(args, "target_coverage", 1),
        drop_p=drop_p,
        seed=args.seed,
        resample=True,
        node_select_graph=getattr(args, "node_select_graph", "train"),
        global_edge_mask=edge_mask_val_full,
        global_supervision_mask=sup_mask_val_full,
        dataset_cfg=dataset_cfg,                      # keep SaGress for val if you want
        use_local_context=getattr(args, "use_local_context", True),
    )

    # For test, optionally override dataset_cfg to disable SaGress
    test_dataset_cfg = dataset_cfg
    if edge_centered:
        test_dataset_cfg = None   # => SubgraphExpansionDataset will call _build_epoch_nodes()

    test_dataset = SubgraphExpansionDataset(
        adj_train_np=adj_train_for_sampling,
        pe_global_np=pe,
        adj_full_np=edge_split.adj_full,
        seed_edges=edge_split.test_edges,
        split_tag="test",
        k=args.k_hop,
        max_nodes=args.max_nodes,
        target_coverage=getattr(args, "target_coverage", 1),
        drop_p=drop_p,
        seed=args.seed,
        resample=True,
        node_select_graph=getattr(args, "node_select_graph", "train"),
        global_edge_mask=None if edge_centered else edge_mask_test_full,
        global_supervision_mask=None if edge_centered else sup_mask_test_full,
        dataset_cfg=test_dataset_cfg,
        use_local_context=getattr(args, "use_local_context", True),
        edge_centered_mask_target=edge_centered,
    )


    # ------------------------------------------------------------------
    # 4) Dataloaders
    # ------------------------------------------------------------------
    train_bs = int(getattr(args, "batch_size", 1))
    eval_bs = getattr(args, "subgraph_eval_batch_size", None)
    if eval_bs is None:
        eval_bs = train_bs
    else:
        eval_bs = int(eval_bs)

    def make_loader(ds: SubgraphExpansionDataset, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_subgraphs,
        )

    train_loader = make_loader(train_dataset, shuffle=True,  batch_size=train_bs)
    val_loader   = make_loader(val_dataset,   shuffle=False, batch_size=eval_bs)
    test_loader  = make_loader(test_dataset,  shuffle=False, batch_size=eval_bs)

    loc_dim = 2 if getattr(args, "use_local_context", True) else 0
    feat_dim = max(1, pe.shape[1] + loc_dim)  # ensure at least one feature channel
    split_masks = {
        "train": {"edge": edge_mask_train_full, "sup": sup_mask_train_full},
        "val": {"edge": edge_mask_val_full, "sup": sup_mask_val_full},
        "test": {"edge": edge_mask_test_full, "sup": sup_mask_test_full},
    }
    return train_loader, val_loader, test_loader, feat_dim, split_masks


def run_test_inference(
    denoiser: DenoiseNetworkA,
    feature_adapter: Optional[nn.Module],
    device: torch.device,
    args,
    edge_split: EdgeSplit,
    test_loader: DataLoader,
    sup_masks: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """Run stitched inference on the existing test loader (test-only)."""
    prior_mode = getattr(args, "prior_init", "baseline")
    denoiser.eval()
    stitcher = LogitAveragingStitcher(edge_split.adj_full.shape[0])

    dataset = test_loader.dataset  # type: ignore[attr-defined]
    counts = [nodes.size for nodes in getattr(dataset, "epoch_nodes", [])]
    avg_nodes = np.mean(counts) if counts else float("nan")
    print(
        f"[Subgraph-LP] 🧪 Automatic test inference "
        f"(subgraphs: {len(counts)} | avg nodes ≈ {avg_nodes:.1f})"
    )

    test_dataset = test_loader.dataset  # type: ignore[attr-defined]
    adj_for_inference = test_dataset.adj_train   # adj_test_only

    with torch.no_grad():
        progress = tqdm(
            test_loader,
            desc="[Subgraph-LP] 🔁 Stitching test subgraphs",
            total=len(test_loader),
            dynamic_ncols=True,
            leave=False,
        )
        for batch in progress:
            process_subgraph_batch_for_inference(
                batch=batch,
                denoiser=denoiser,
                feature_adapter=feature_adapter,
                device=device,
                args=args,
                adj_train=adj_for_inference,
                stitcher=stitcher,
            )

    # Choose baseline scoring function (Node2Vec or LPFormer) for comparison
    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_structural = (getattr(args, "prior_init", "baseline") == "prior")

    baseline_edge_score_fn = None
    baseline_label = None

    if use_structural:
        if prior_type == "node2vec":
            baseline_edge_score_fn = compute_node2vec_scores_for_edges
            baseline_label = "Node2Vec"
        elif prior_type == "lpformer":
            try:
                from pifm_sub.LPFormerCodes.lpformer_prior import compute_lpformer_scores_for_edges
            except ImportError:
                baseline_edge_score_fn = None
                baseline_label = "LPFormer"
                print(
                    "[Subgraph-LP] ⚠️ LPFormer prior selected but pifm_sub.LPFormerCodes.lpformer_prior "
                    "could not be imported; skipping LPFormer baseline metrics."
                )
            else:
                baseline_edge_score_fn = compute_lpformer_scores_for_edges
                baseline_label = "LPFormer"
        elif prior_type == "graphsage":
            baseline_edge_score_fn = compute_graphsage_scores_for_edges
            baseline_label = "GraphSAGE"

    P_global = stitcher.finalize()
    metrics = compute_split_metrics(
        P_global,
        edge_split,
        args,
        n2v_edge_score_fn=baseline_edge_score_fn,
        sup_masks=sup_masks,
    )
    test_stats = metrics.get("test", {})

    # --- DEBUG: coverage of test positives and negatives ---
    counts = stitcher.counts  # (N,N) int32
    test_edges = edge_split.test_edges  # (E_test,2)
    cov_pos = (counts[test_edges[:, 0], test_edges[:, 1]] > 0)
    frac_pos_covered = float(cov_pos.mean())
    print(f"[DEBUG] fraction of TEST POS edges covered by any subgraph: {frac_pos_covered:.4f}")

    rng = np.random.default_rng(args.split_seed + 99)
    negatives = sample_non_edges(edge_split.adj_full, edge_split.test_edges.shape[0], seed=rng.integers(1e9))
    cnts_neg = counts[negatives[:, 0], negatives[:, 1]]
    print(
        "[DEBUG] fraction of negative edges ever seen in any subgraph:",
        np.mean(cnts_neg > 0),
    )

    stats = metrics.get("test", {})
    if stats:
        auc = stats.get("auc", float("nan"))
        ap = stats.get("ap", float("nan"))
        fpr = stats.get("fpr", float("nan"))
        fnr = stats.get("fnr", float("nan"))
        mrr = stats.get("mrr", float("nan"))
        line = (
            "[Subgraph-LP] 🧪 Test (PIFM): "
            f"AUC={auc:.4f} | AP={ap:.4f} | FPR={fpr:.4f} | FNR={fnr:.4f} | MRR={mrr:.4f}"
        )
        if "n2v_auc" in stats and baseline_label:
            n2v_auc = stats.get("n2v_auc", float("nan"))
            n2v_ap = stats.get("n2v_ap", float("nan"))
            n2v_fpr = stats.get("n2v_fpr", float("nan"))
            n2v_fnr = stats.get("n2v_fnr", float("nan"))
            n2v_mrr = stats.get("n2v_mrr", float("nan"))
            line += (
                f" || {baseline_label}: AUC={n2v_auc:.4f} | AP={n2v_ap:.4f} | "
                f"FPR={n2v_fpr:.4f} | FNR={n2v_fnr:.4f} | MRR={n2v_mrr:.4f}"
            )
        print(line)
        hidden_auc = stats.get("auc_hidden", float("nan"))
        if not np.isnan(hidden_auc):
            hidden_ap = stats.get("ap_hidden", float("nan"))
            hidden_fpr = stats.get("fpr_hidden", float("nan"))
            hidden_fnr = stats.get("fnr_hidden", float("nan"))
            hidden_mrr = stats.get("mrr_hidden", float("nan"))
            hidden_line = (
                "[Subgraph-LP] 🫥 Test (hidden edges): "
                f"AUC={hidden_auc:.4f} | AP={hidden_ap:.4f} | "
                f"FPR={hidden_fpr:.4f} | FNR={hidden_fnr:.4f} | MRR={hidden_mrr:.4f}"
            )
            if "n2v_auc_hidden" in stats and baseline_label:
                hidden_line += (
                    f" || {baseline_label}: AUC={stats.get('n2v_auc_hidden', float('nan')):.4f} | "
                    f"AP={stats.get('n2v_ap_hidden', float('nan')):.4f} | "
                    f"FPR={stats.get('n2v_fpr_hidden', float('nan')):.4f} | "
                    f"FNR={stats.get('n2v_fnr_hidden', float('nan')):.4f} | "
                    f"MRR={stats.get('n2v_mrr_hidden', float('nan')):.4f}"
                )
            print(hidden_line)
    else:
        print("[Subgraph-LP] 🧪 Test metrics: no test edges available")

    if stats:
        auc = stats.get("auc", float("nan"))
        ap = stats.get("ap", float("nan"))
        mrr = stats.get("mrr", float("nan"))
        print(f"[Subgraph-LP] 🧪 Test metrics: AUC={auc:.4f} | AP={ap:.4f} | MRR={mrr:.4f}")
    else:
        print("[Subgraph-LP] 🧪 Test metrics: no test edges available")

    setattr(args, "prior_init", prior_mode)
    return metrics


def train_subgraph_lp(args):
    """Train the denoiser on SGDM-style subgraph link-prediction batches."""
    if not args.single_graph_path:
        raise ValueError("--single_graph_path is required in subgraph link-prediction mode.")

    # ---- Prior selection -------------------------------------------------
    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_noise = bool(getattr(args, "hidden_gaussian_prior", False))

    use_n2v      = (prior_type == "node2vec")   and (not use_noise)
    use_lpformer = (prior_type == "lpformer")   and (not use_noise)
    use_sage     = (prior_type == "graphsage")  and (not use_noise)
    use_sage_heart = (prior_type == "graphsage_heart") and (not use_noise)

    use_structural_prior = use_n2v or use_lpformer or use_sage or use_sage_heart


    # prior_init now means: "prior" → some structural prior (N2V / LPFormer),
    #                       "baseline" → no structural prior (maybe Gaussian hidden)
    setattr(args, "prior_init", "prior" if use_structural_prior else "baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # --- Automatic checkpoint directory: outputs/<YYYYmmdd_HHMMSS>_<run_name>/ ---
    run_name = getattr(args, "run_name", None)

    if getattr(args, "ckpt_dir", None) is None and run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_root = os.path.join("outputs", f"{timestamp}_{run_name}")
        args.ckpt_dir = ckpt_root
        print(f"[Subgraph-LP] 💾 Checkpoints will be saved under: {args.ckpt_dir}")

    print("[Subgraph-LP] ✅ Starting training with subgraph link-prediction pipeline")
    print(f"[Subgraph-LP] 📄 Graph: {args.single_graph_path}")

    if use_n2v:
        prior_desc = "Node2Vec (global cached)"
    elif use_lpformer:
        prior_desc = "LPFormer (global model)"
    elif use_sage:
        prior_desc = "GraphSAGE (precomputed embeddings)"
    elif use_noise:
        prior_desc = "Gaussian prior only"
    elif use_sage_heart:
        prior_desc = "GraphSAGE-HEART (SAGEConv, precomputed embeddings)"
    else:
        prior_desc = "Zero fill (no structural prior)"


    print(f"[Subgraph-LP] ⚙️  Prior: {prior_desc}")
    print(
        f"[Subgraph-LP] ⚙️  k-hop={args.k_hop}, max_nodes={args.max_nodes}, "
        f"target_coverage={args.target_coverage}"
    )

    dataset_cfg = load_dataset_cfg_file(getattr(args, "subgraph_dataset_cfg", None))
    setattr(args, "_dataset_cfg_obj", dataset_cfg)
    if dataset_cfg:
        print(
            f"[Subgraph-LP] 📘 Subgraph sampler cfg ({args.subgraph_dataset_cfg}): "
            f"{format_dataset_cfg(dataset_cfg)}"
        )
    else:
        print("[Subgraph-LP] 📘 Subgraph sampler cfg: default k-hop BFS")

    drop_p = float(getattr(args, "train_edge_drop_p", 0.5))
    setattr(args, "_subgraph_drop_p", drop_p)
    print(f"[Subgraph-LP] 🎯 drop_p for edge supervision masks: {drop_p:.2f}")

    adj_full = load_adjacency(args.single_graph_path)
    split_path = default_edge_split_path(args)

    if os.path.exists(split_path):
        split = load_edge_split(split_path)
        print(
            f"[Subgraph-LP] 📂 Loaded existing edge split from {split_path} → "
            f"train: {split.train_edges.shape[0]} | "
            f"val: {split.val_edges.shape[0]} | "
            f"test: {split.test_edges.shape[0]}"
        )
    else:
        split = split_edges(adj_full, args.val_ratio, args.test_ratio, args.split_seed)
        print(
            "[Subgraph-LP] 📊 Created edge split → train: {} | val: {} | test: {}".format(
                split.train_edges.shape[0],
                split.val_edges.shape[0],
                split.test_edges.shape[0],
            )
        )
        save_edge_split(split, split_path)

    # ---- Initialize global prior (if any) --------------------------------
    if use_n2v:
        init_global_node2vec_prior_from_adj(split.adj_train, args)
    elif use_lpformer:
        from pifm_sub.LPFormerCodes.lpformer_prior import (
            init_global_lpformer_prior_from_adj,
            prepare_lpformer_prior_cache,
        )
        init_global_lpformer_prior_from_adj(split.adj_train, args)
    elif use_sage:
        init_global_graphsage_prior_from_adj(split.adj_train, args)
    elif use_sage_heart:
        init_global_graphsage_heart_prior_from_adj(split.adj_train, args)


    train_loader, val_loader, test_loader, feat_dim, split_masks = build_dataloaders(split, args, device)
    setattr(args, "_split_masks", split_masks)

    if use_lpformer:
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            prepare_lpformer_prior_cache(loader.dataset, split_name, args)

    # ---- DBG: show how many entries are actually supervised vs kept ----
    def _ut_count(mask: torch.Tensor, node_mask: torch.Tensor) -> int:
        N = mask.size(-1)
        ut = torch.triu(torch.ones(N, N, dtype=torch.bool, device=mask.device), diagonal=1)
        valid = ut & (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
        return int((mask.bool() & valid).sum().item())

    with torch.no_grad():
        for name, loader in [("test", test_loader)]:
            dataset = loader.dataset
            dataset.refresh_epoch()
            b = next(iter(loader))
            nm = b["node_mask"]
            em = b["edge_mask"]
            sm = b["supervision_mask"]
            print(
                f"[DEBUG] {name}: |Ω_keep|={_ut_count(em, nm)}  |Ω_sup|={_ut_count(sm, nm)}  "
                f"|Ω_keep ∩ Ω_sup|={_ut_count((em.bool() & sm.bool()).float(), nm)}"
            )

    # -------------------------------------------------------------------
    def _log_dataset_stats(name: str, loader) -> None:
        ds = loader.dataset  # type: ignore[attr-defined]
        counts = [int(nodes.size) for nodes in getattr(ds, "epoch_nodes", [])]
        avg_nodes = np.mean(counts) if counts else float("nan")
        print(
            f"[Subgraph-LP]   ↳ {name}: {len(counts)} subgraphs | avg nodes ≈ {avg_nodes:.1f}"
        )

    print("[Subgraph-LP] 🔍 Subgraph statistics (epoch 0 preview):")
    _log_dataset_stats("train", train_loader)
    _log_dataset_stats("val", val_loader)
    _log_dataset_stats("test", test_loader)

    max_node_num = args.max_nodes
    use_adapter = getattr(args, "feature_adapter", True)

    feature_adapter = FeatureAdapter(feat_dim).to(device) if use_adapter else None
    max_feat = 1 if feature_adapter else feat_dim

    denoiser = DenoiseNetworkA(
        max_feat_num=max_feat,
        max_node_num=max_node_num,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,
        num_heads=args.num_heads,
        conv=args.conv,
    ).to(device)

    params = list(denoiser.parameters())
    if feature_adapter is not None:
        params += list(feature_adapter.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    train_dataset = train_loader.dataset  # type: ignore[attr-defined]
    val_dataset = val_loader.dataset      # type: ignore[attr-defined]

    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
        setattr(args, "epoch", epoch)
        denoiser.train()
        train_dataset.refresh_epoch()
        counts_epoch = [nodes.size for nodes in getattr(train_dataset, "epoch_nodes", [])]
        avg_nodes_epoch = np.mean(counts_epoch) if counts_epoch else float("nan")
        print(
            f"[Subgraph-LP] 🔁 Epoch {epoch}/{args.epochs} "
            f"(subgraphs: {len(counts_epoch)} | avg nodes ≈ {avg_nodes_epoch:.1f})"
        )
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = subgraph_batch_loss(
                batch,
                denoiser,
                feature_adapter,
                device,
                args,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1
        train_loss = running_loss / max(1, steps)

        denoiser.eval()
        val_dataset.refresh_epoch()
        with torch.no_grad():
            val_loss = 0.0
            v_steps = 0
            for batch in val_loader:
                loss = subgraph_batch_loss(
                    batch,
                    denoiser,
                    feature_adapter,
                    device,
                    args,
                    train_mode=False,
                )
                val_loss += float(loss.item())
                v_steps += 1
        val_loss = val_loss / max(1, v_steps)
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        train_history.append(train_loss)
        val_history.append(val_loss)

        if getattr(args, "ckpt_dir", None) and (epoch % 500 == 0 or epoch == args.epochs):
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f"subgraph_epoch{epoch:04d}.pt")
            torch.save(denoiser.state_dict(), ckpt_path)
            print(f"[Subgraph-LP] 💾 Saved checkpoint → {ckpt_path}")

    loss_curve_output_dir = getattr(args, "ckpt_dir", None) or "outputs"
    loss_curve_path = save_subgraph_loss_curves(
        train_losses=train_history,
        val_losses=val_history,
        output_dir=loss_curve_output_dir,
        filename="loss_curve_subgraph.jpg",
    )

    test_metrics = run_test_inference(
        denoiser=denoiser,
        feature_adapter=feature_adapter,
        device=device,
        args=args,
        edge_split=split,
        test_loader=test_loader,
        sup_masks=split_masks,
    )

    return {
        "denoiser": denoiser,
        "feature_adapter": feature_adapter,
        "edge_split": split,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "device": device,
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "loss_curve_path": loss_curve_path,
        "test_metrics": test_metrics,
    }



def subgraph_batch_loss(
    batch: Dict[str, torch.Tensor],
    denoiser: DenoiseNetworkA,
    feature_adapter: Optional[nn.Module],
    device: torch.device,
    args,
    train_mode: bool = True,
) -> torch.Tensor:
    A_true = batch["A_true"].to(device)
    node_mask = batch["node_mask"].to(device)
    edge_mask = batch["edge_mask"].to(device)              # kept/context edges
    supervision_mask = batch["supervision_mask"].to(device)  # dropped edges to predict
    x_feat = batch["x_feat"].to(device)
    nodes_global = batch["nodes_global"].to(device)
    cached_lpformer_prior = batch.get("lpformer_prior")

    # Feature projection
    if feature_adapter is not None:
        x_proj = feature_adapter(x_feat)
    else:
        x_proj = x_feat
        
    node_mask_f = node_mask.float()
    x_proj = x_proj * node_mask_f.unsqueeze(-1)

    B = A_true.size(0)

    # --- Optional structural prior (Node2Vec or LPFormer) ----------------
    prior = None
    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_noise = bool(getattr(args, "hidden_gaussian_prior", False))

    if getattr(args, "prior_init", "baseline") == "prior" and not use_noise:
        if prior_type == "node2vec":
            prior = compute_node2vec_prior_batch(
                A_true=A_true,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        elif prior_type == "lpformer":
            if cached_lpformer_prior is not None:
                prior = cached_lpformer_prior.to(device)
            else:
                from pifm_sub.LPFormerCodes.lpformer_prior import compute_lpformer_prior_batch
                prior = compute_lpformer_prior_batch(
                    A_true=A_true,
                    edge_mask=edge_mask,
                    node_mask=node_mask,
                    nodes_global=nodes_global,
                    args=args,
                )
        elif prior_type == "graphsage":
            prior = compute_graphsage_prior_batch(
                A_true=A_true,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        elif prior_type == "graphsage_heart":
            prior = compute_graphsage_heart_prior_batch(
                A_true=A_true,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        else:
            prior = None



    # Build A0 using prior + observed edges
    A0 = build_initial_A0_lp(
        args,
        A_true=A_true,
        edge_mask=edge_mask,
        node_mask=node_mask,
        prior=prior,
    )

    # --- NEW: symmetric Gaussian prior on hidden entries (Ω = 1 - edge_mask) ---
    if getattr(args, "hidden_gaussian_prior", False):
        # Hidden entries = those not anchored by edge_mask (same notion as in inference)
        Omega_all = sym_zero_diag_valid(1.0 - edge_mask, node_mask)

        std = getattr(args, "hidden_gaussian_std", 0.25)
        noise = 0.5 + std * torch.randn_like(A0)

        # Make noise symmetric
        noise = 0.5 + std * torch.randn_like(A0)
        noise = 0.5 + std * torch.randn_like(A0)
        noise = (noise + noise.transpose(-1, -2)) / 2.0
        noise = noise.clamp(0.0, 1.0)

        A0 = A0 * (1.0 - Omega_all) + noise * Omega_all


    # Sample time and construct interpolation
    t = torch.rand(B, device=device)
    alpha, beta, _, _ = linear_coeffs(t)
    av = alpha.view(B, 1, 1)
    bv = beta.view(B, 1, 1)
    I_t = sym_zero_diag_valid(av * A0 + bv * A_true, node_mask)

    # Denoiser prediction
    inp = I_t.unsqueeze(1)
    b_pred = denoiser(x_proj, inp, node_mask, t)
    b_pred = sym_zero_diag_valid(b_pred, node_mask)

    # Supervision region: ONLY on intentionally dropped edges
    supervise = sym_zero_diag_valid(supervision_mask, node_mask)
    
    # ---- DBG-once: how many entries are supervised and how big the target is there? ----
    if not hasattr(args, "_dbg_printed"):
        with torch.no_grad():
            N = supervise.size(-1)
            ut = torch.triu(torch.ones(N, N, dtype=torch.bool, device=supervise.device), diagonal=1)
            valid = ut & (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
            sup_mask = (supervise > 0) & valid
            sup_cnt = int(sup_mask.sum().item())
            print(f"[DEBUG] supervised upper-tri entries this batch: {sup_cnt}")
        args._dbg_printed = True
    # --------------------------------------------------------------------


    # Apply mask to prediction
    b_pred = b_pred * supervise

    # Target is the residual A_true - A0, but loss only accumulates where supervise==1
    target = sym_zero_diag_valid(A_true - A0, node_mask)
    
    # ---- DBG-once: mean|target| on supervised entries ----
    if getattr(args, "_dbg_printed", False) and not getattr(args, "_dbg_target_printed", False):
        with torch.no_grad():
            N = supervise.size(-1)
            ut = torch.triu(torch.ones(N, N, dtype=torch.bool, device=supervise.device), diagonal=1)
            valid = ut & (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
            sup_mask = (supervise > 0) & valid
            denom = sup_mask.float().sum().clamp_min(1)
            mean_abs_target = (target.abs() * sup_mask.float()).sum() / denom
            print(f"[DEBUG] mean|target|@supervised = {mean_abs_target.item():.4f}")
        args._dbg_target_printed = True
    # --------------------------------------------------------------------


    loss = masked_upper_mse(b_pred, target, node_mask, supervise)
    return loss


def infer_subgraph_lp(args):
    if not args.single_graph_path:
        raise ValueError("--single_graph_path is required in subgraph link-prediction inference.")
    if not args.ckpt:
        raise ValueError("--ckpt must point to a trained model for inference.")

    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_noise = bool(getattr(args, "hidden_gaussian_prior", False))

    use_n2v        = (prior_type == "node2vec")        and (not use_noise)
    use_lpformer   = (prior_type == "lpformer")        and (not use_noise)
    use_sage       = (prior_type == "graphsage")       and (not use_noise)
    use_sage_heart = (prior_type == "graphsage_heart") and (not use_noise)
    use_structural = use_n2v or use_lpformer or use_sage or use_sage_heart


    setattr(args, "prior_init", "prior" if use_structural else "baseline")


    dataset_cfg = load_dataset_cfg_file(getattr(args, "subgraph_dataset_cfg", None))
    setattr(args, "_dataset_cfg_obj", dataset_cfg)
    if dataset_cfg:
        print(
            f"[Subgraph-LP] 📘 Subgraph sampler cfg ({args.subgraph_dataset_cfg}): "
            f"{format_dataset_cfg(dataset_cfg)}"
        )
    else:
        print("[Subgraph-LP] 📘 Subgraph sampler cfg: default k-hop BFS")

    if not hasattr(args, "_subgraph_drop_p"):
        setattr(args, "_subgraph_drop_p", float(getattr(args, "train_edge_drop_p", 0.5)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_full = load_adjacency(args.single_graph_path)
    split_path = default_edge_split_path(args)

    if os.path.exists(split_path):
        split = load_edge_split(split_path)
        print(
            f"[Subgraph-LP] 📂 Loaded existing edge split from {split_path} → "
            f"train: {split.train_edges.shape[0]} | "
            f"val: {split.val_edges.shape[0]} | "
            f"test: {split.test_edges.shape[0]}"
        )
    else:
        print("[Subgraph-LP] ⚠️ Edge split cache not found; recomputing with current args.")
        split = split_edges(adj_full, args.val_ratio, args.test_ratio, args.split_seed)
        save_edge_split(split, split_path)

    print("[Subgraph-LP] 🚀 Starting stitched inference")
    print(f"[Subgraph-LP] 📄 Graph: {args.single_graph_path}")
    
    if use_n2v:
        prior_desc = "Node2Vec (global cached)"
    elif use_lpformer:
        prior_desc = "LPFormer (global model)"
    elif use_sage:
        prior_desc = "GraphSAGE (precomputed embeddings)"
    elif use_noise:
        prior_desc = "Gaussian hidden prior only"
    elif use_sage_heart:
        prior_desc = "GraphSAGE-HEART (SAGEConv, precomputed embeddings)"
    else:
        prior_desc = "Zero fill (no structural prior)"

    print(f"[Subgraph-LP] ⚙️  Prior: {prior_desc}")


    print(
        "[Subgraph-LP] 📊 Edge split → train: {} | val: {} | test: {}".format(
            split.train_edges.shape[0], split.val_edges.shape[0], split.test_edges.shape[0]
        )
    )

    if use_n2v:
        init_global_node2vec_prior_from_adj(split.adj_train, args)
    elif use_lpformer:
        from pifm_sub.LPFormerCodes.lpformer_prior import (
            init_global_lpformer_prior_from_adj,
            prepare_lpformer_prior_cache,
        )
        init_global_lpformer_prior_from_adj(split.adj_train, args)
    elif use_sage:
        init_global_graphsage_prior_from_adj(split.adj_train, args)
    elif use_sage_heart:
        init_global_graphsage_heart_prior_from_adj(split.adj_train, args)

        
    train_loader, val_loader, test_loader, feat_dim, split_masks = build_dataloaders(split, args, device)
    setattr(args, "_split_masks", split_masks)

    if use_lpformer:
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            prepare_lpformer_prior_cache(loader.dataset, split_name, args)

    use_adapter = getattr(args, "feature_adapter", True)
    feature_adapter = FeatureAdapter(feat_dim).to(device) if use_adapter else None
    max_feat = 1 if feature_adapter else feat_dim

    denoiser = DenoiseNetworkA(
        max_feat_num=max_feat,
        max_node_num=args.max_nodes,
        nhid=args.hidden_dim,
        num_layers=args.num_layers,
        num_linears=args.num_linears,
        c_init=args.c_init,
        c_hid=args.c_hid,
        c_final=args.c_final,
        adim=args.hidden_dim,
        num_heads=args.num_heads,
        conv=args.conv,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    denoiser.load_state_dict(state)
    denoiser.eval()

    if feature_adapter is not None and hasattr(feature_adapter, "load_state_dict") and getattr(args, "adapter_ckpt", None):
        feature_adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))

    stitcher = LogitAveragingStitcher(adj_full.shape[0])

    rollout_state = None
    if getattr(args, "subgraph_traj_plots", False) and getattr(args, "test_edge_centered_subgraphs", False):
        base_dir = os.path.dirname(args.ckpt)
        roll_dir = os.path.join(base_dir, "rollouts")
        rollout_state = {
            "enabled": True,
            "dir": roll_dir,
            "max": int(getattr(args, "subgraph_traj_max_samples", 5)),
            "saved": 0,
            "interval": 10,
            "paths": [],
        }
        os.makedirs(roll_dir, exist_ok=True)

    with torch.no_grad():
        dataset = test_loader.dataset
        dataset.refresh_epoch()
        counts_epoch = [nodes.size for nodes in getattr(dataset, "epoch_nodes", [])]
        avg_nodes_epoch = np.mean(counts_epoch) if counts_epoch else float("nan")
        print(
            f"[Subgraph-LP] 🔁 Inference on test subgraphs "
            f"(count={len(counts_epoch)} | avg nodes ≈ {avg_nodes_epoch:.1f})"
        )

        # IMPORTANT: use the test dataset's own adjacency (adj_test_only),
        # not split.adj_train (which is the TRAIN graph)
        adj_for_inference = dataset.adj_train

        for batch in test_loader:
            process_subgraph_batch_for_inference(
                batch=batch,
                denoiser=denoiser,
                feature_adapter=feature_adapter,
                device=device,
                args=args,
                adj_train=adj_for_inference,
                stitcher=stitcher,
                rollout_state=rollout_state,
            )


    P_global = stitcher.finalize()

    # Baseline scorer (N2V / LPFormer) for comparison
    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_structural = (getattr(args, "prior_init", "baseline") == "prior")

    baseline_edge_score_fn = None
    baseline_label = None
    if use_structural:
        if prior_type == "node2vec":
            baseline_edge_score_fn = compute_node2vec_scores_for_edges
            baseline_label = "Node2Vec"
        elif prior_type == "lpformer":
            try:
                from pifm_sub.LPFormerCodes.lpformer_prior import compute_lpformer_scores_for_edges
            except ImportError:
                baseline_edge_score_fn = None
                baseline_label = "LPFormer"
                print(
                    "[Subgraph-LP] ⚠️ LPFormer prior selected but pifm_sub.LPFormerCodes.lpformer_prior "
                    "could not be imported; skipping LPFormer baseline metrics."
                )
            else:
                baseline_edge_score_fn = compute_lpformer_scores_for_edges
                baseline_label = "LPFormer"
        elif prior_type == "graphsage":
            baseline_edge_score_fn = compute_graphsage_scores_for_edges
            baseline_label = "GraphSAGE"
        elif prior_type == "graphsage_heart":
            baseline_edge_score_fn = compute_graphsage_heart_scores_for_edges
            baseline_label = "GraphSAGE-HEART"

    metrics = compute_split_metrics(
        P_global,
        split,
        args,
        n2v_edge_score_fn=baseline_edge_score_fn,
        sup_masks=split_masks,
    )
    stats = metrics.get("test", {})
    if np.isnan(stats.get("auc", float("nan"))):
        print("  - test: insufficient edges")
    else:
        mrr = stats.get("mrr", float("nan"))
        line = (
            f"  - test: PIFM AUC={stats['auc']:.4f} | AP={stats['ap']:.4f} "
            f"| FPR={stats['fpr']:.4f} | FNR={stats['fnr']:.4f} | MRR={mrr:.4f}"
        )
        if "n2v_auc" in stats and baseline_label:
            n2v_mrr = stats.get("n2v_mrr", float("nan"))
            line += (
                f" || {baseline_label} AUC={stats['n2v_auc']:.4f} | "
                f"AP={stats['n2v_ap']:.4f} | FPR={stats['n2v_fpr']:.4f} | "
                f"FNR={stats['n2v_fnr']:.4f} | MRR={n2v_mrr:.4f}"
            )
        print(line)

    if rollout_state and rollout_state.get("paths"):
        print("[Subgraph-LP] 🖼️ Saved rollout plots:")
        for p in rollout_state["paths"]:
            print(f"  - {p}")

    print("[Subgraph-LP] 📈 Metrics:")
    for split_name, stats in metrics.items():
        if np.isnan(stats.get("auc", float("nan"))):
            print(f"  - {split_name}: insufficient edges")
            continue

        mrr = stats.get("mrr", float("nan"))
        line = (
            f"  - {split_name}: "
            f"PIFM AUC={stats['auc']:.4f} | AP={stats['ap']:.4f} "
            f"| FPR={stats['fpr']:.4f} | FNR={stats['fnr']:.4f} | MRR={mrr:.4f}"
        )
        if "n2v_auc" in stats and baseline_label:
            n2v_mrr = stats.get("n2v_mrr", float("nan"))
            line += (
                f" || {baseline_label} AUC={stats['n2v_auc']:.4f} | "
                f"AP={stats['n2v_ap']:.4f} | FPR={stats['n2v_fpr']:.4f} | "
                f"FNR={stats['n2v_fnr']:.4f} | MRR={n2v_mrr:.4f}"
            )
        print(line)

        if "auc_hidden" in stats:
            hidden_auc = stats.get("auc_hidden", float("nan"))
            hidden_ap = stats.get("ap_hidden", float("nan"))
            hidden_fpr = stats.get("fpr_hidden", float("nan"))
            hidden_fnr = stats.get("fnr_hidden", float("nan"))
            hidden_mrr = stats.get("mrr_hidden", float("nan"))
            hidden_line = (
                f"    ↳ hidden edges: PIFM AUC={hidden_auc:.4f} | AP={hidden_ap:.4f} "
                f"| FPR={hidden_fpr:.4f} | FNR={hidden_fnr:.4f} | MRR={hidden_mrr:.4f}"
            )
            if "n2v_auc_hidden" in stats and baseline_label:
                hidden_line += (
                    f" || {baseline_label} AUC={stats.get('n2v_auc_hidden', float('nan')):.4f} | "
                    f"AP={stats.get('n2v_ap_hidden', float('nan')):.4f} | "
                    f"FPR={stats.get('n2v_fpr_hidden', float('nan')):.4f} | "
                    f"FNR={stats.get('n2v_fnr_hidden', float('nan')):.4f} | "
                    f"MRR={stats.get('n2v_mrr_hidden', float('nan')):.4f}"
                )
            print(hidden_line)



def _save_rollout_grid(frames: List[Tuple[str, np.ndarray]], gt: np.ndarray, out_path: str) -> None:
    """
    Save a grid of rollout snapshots plus ground-truth adjacency.
    frames: list of (label, A_np) for successive steps
    gt: ground-truth adjacency (np.ndarray)
    """
    import matplotlib.pyplot as plt  # local import to avoid overhead

    # We expect up to 12 panels (11 steps + GT). Use 3x4 grid plus a dedicated colorbar column.
    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.25, hspace=0.3)
    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
    cax = fig.add_subplot(gs[:, -1])
    vmax = 1.0
    vmin = 0.0

    panels = frames + [("GT", gt)]
    for ax, (label, mat) in zip(axes, panels):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    # Blank any leftover axes
    for ax in axes[len(panels):]:
        ax.axis("off")

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(f"Link-prediction rollout ({len(frames) - 1 if frames else 0} steps)", fontsize=10)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_subgraph_batch_for_inference(
    batch: Dict[str, torch.Tensor],
    denoiser: DenoiseNetworkA,
    feature_adapter: Optional[nn.Module],
    device: torch.device,
    args,
    adj_train: np.ndarray,
    stitcher: LogitAveragingStitcher,
    rollout_state: Optional[dict] = None,
) -> None:
    debug_eval_a0_only = False
    node_mask = batch["node_mask"].to(device)  # (B,maxn) bool
    edge_mask = batch["edge_mask"].to(device)  # (B,maxn,maxn)
    x_feat = batch["x_feat"].to(device)
    nodes_global = batch["nodes_global"].to(device)
    cached_lpformer_prior = batch.get("lpformer_prior")
    A_true_batch = batch.get("A_true")

    if feature_adapter is not None:
        x_proj = feature_adapter(x_feat)
    else:
        x_proj = x_feat
    node_mask_f = node_mask.float()
    x_proj = x_proj * node_mask_f.unsqueeze(-1)

    B, maxn = node_mask.shape
    dt = 1.0 / max(1, args.n_steps)

    valid_mask = (nodes_global >= 0) & node_mask
    n_nodes = valid_mask.sum(dim=1)

    edge_mask = edge_mask * node_mask_f.unsqueeze(1) * node_mask_f.unsqueeze(2)

    A_obs = torch.zeros(B, maxn, maxn, device=device, dtype=torch.float32)
    nodes_global_cpu = nodes_global.detach().cpu().numpy()
    for idx in range(B):
        n = int(n_nodes[idx].item())
        if n <= 0:
            continue
        nodes = nodes_global_cpu[idx, :n]
        sub = adj_train[np.ix_(nodes, nodes)]
        A_obs[idx, :n, :n] = torch.from_numpy(sub).to(device, dtype=torch.float32)

    prior = None
    prior_type = getattr(args, "subgraph_prior", "node2vec")
    use_noise = bool(getattr(args, "hidden_gaussian_prior", False))

    if getattr(args, "prior_init", "baseline") == "prior" and not use_noise:
        if prior_type == "node2vec":
            prior = compute_node2vec_prior_batch(
                A_true=A_obs,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        elif prior_type == "lpformer":
            if cached_lpformer_prior is not None:
                prior = cached_lpformer_prior.to(device)
            else:
                from pifm_sub.LPFormerCodes.lpformer_prior import compute_lpformer_prior_batch
                prior = compute_lpformer_prior_batch(
                    A_true=A_obs,
                    edge_mask=edge_mask,
                    node_mask=node_mask,
                    nodes_global=nodes_global,
                    args=args,
                )
        elif prior_type == "graphsage":
            prior = compute_graphsage_prior_batch(
                A_true=A_obs,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        elif prior_type == "graphsage_heart":
            prior = compute_graphsage_heart_prior_batch(
                A_true=A_obs,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
        else:
            prior = None



    A0 = build_initial_A0_lp(
        args,
        A_true=A_obs,
        edge_mask=edge_mask,
        node_mask=node_mask,
        prior=prior,
    )

    # Hidden mask Ω is the same notion as in training: "not observed / anchored"
    Omega = sym_zero_diag_valid(1.0 - edge_mask, node_mask)
    A_anchor = sym_zero_diag_valid(edge_mask * A_obs, node_mask)

    # --- NEW: symmetric Gaussian prior on hidden entries during inference ---
    if getattr(args, "hidden_gaussian_prior", False):
        std = getattr(args, "hidden_gaussian_std", 0.25)
        noise = 0.5 + std * torch.randn_like(A0)
        noise = (noise + noise.transpose(-1, -2)) / 2.0
        noise = noise.clamp(0.0, 1.0)

        A0 = A0 * (1.0 - Omega) + noise * Omega

    A = A0.clone()

    # Rollout capture setup
    capture_rollout = (
        rollout_state is not None
        and rollout_state.get("enabled", False)
        and rollout_state.get("saved", 0) < rollout_state.get("max", 0)
    )
    rollout_indices: List[int] = []
    rollout_buffers: List[Dict[str, List[Tuple[str, np.ndarray]]]] = []
    steps_to_capture: set[int] = set()
    if capture_rollout:
        interval = int(rollout_state.get("interval", 10))
        max_step_cap = min(int(rollout_state.get("max_step", args.n_steps)), args.n_steps)
        steps_to_capture = set(range(0, max_step_cap + 1, interval))
        steps_to_capture.add(args.n_steps)
        # Track samples in this batch until quota is met
        remaining = rollout_state["max"] - rollout_state.get("saved", 0)
        for idx in range(min(remaining, B)):
            rollout_indices.append(idx)
            rollout_buffers.append({"frames": []})

        # capture step 0
        if rollout_indices:
            A_np0 = A.detach().cpu().numpy()
            for local_idx, buf in zip(rollout_indices, rollout_buffers):
                n = int(n_nodes[local_idx].item())
                if n > 1:
                    buf["frames"].append((f"step 0", A_np0[local_idx, :n, :n].copy()))


    for step in range(args.n_steps):
        if Omega.numel() == 0:
            break
        t = torch.full((B,), step * dt, device=device)
        inp = A.unsqueeze(1)
        b_pred = denoiser(x_proj, inp, node_mask, t)
        b_pred = sym_zero_diag_valid(b_pred, node_mask)
        b_pred = b_pred * Omega

        A = A + dt * b_pred
        A = A.clamp(0.0, 1.0)
        A = A_anchor + Omega * A
        A = sym_zero_diag_valid(A, node_mask)

        if capture_rollout and steps_to_capture and (step + 1) in steps_to_capture:
            A_np = A.detach().cpu().numpy()
            for local_idx, buf in zip(rollout_indices, rollout_buffers):
                n = int(n_nodes[local_idx].item())
                if n > 1:
                    buf["frames"].append((f"step {step + 1}", A_np[local_idx, :n, :n].copy()))

    # # A_np = A.detach().cpu().numpy()
    # # for idx in range(B):
    # #     n = int(n_nodes[idx].item())
    # #     if n <= 1:
    # #         continue
    # #     nodes = nodes_global_cpu[idx, :n]
    # #     probs = A_np[idx, :n, :n]
    # #     stitcher.add_subgraph_probs(nodes, probs)
    
    # A_np = A0.detach().cpu().numpy()
    # for idx in range(B):
    #     n = int(n_nodes[idx].item())
    #     if n <= 1:
    #         continue
    #     nodes = nodes_global_cpu[idx, :n]
    #     probs = A_np[idx, :n, :n]
    #     stitcher.add_subgraph_probs(nodes, probs)
    # return

    if debug_eval_a0_only:
            A_to_use = A0     # just the prior (anchored by edge_mask)
    else:
        A_to_use = A      # full diffusion result

    # Save rollout plots if requested
    if capture_rollout and rollout_indices and rollout_state is not None:
        A_np_final = A_to_use.detach().cpu().numpy()
        gt_np = None
        if A_true_batch is not None:
            gt_np = A_true_batch.detach().cpu().numpy()
        save_dir = rollout_state.get("dir", "outputs")
        os.makedirs(save_dir, exist_ok=True)
        for local_idx, buf in zip(rollout_indices, rollout_buffers):
            if rollout_state["saved"] >= rollout_state["max"]:
                break
            n = int(n_nodes[local_idx].item())
            if n <= 1:
                continue
            frames = buf["frames"]
            frames.append((f"step {args.n_steps}", A_np_final[local_idx, :n, :n].copy()))
            gt_slice = gt_np[local_idx, :n, :n] if gt_np is not None else None
            if gt_slice is None:
                gt_slice = A_np_final[local_idx, :n, :n]
            out_path = os.path.join(save_dir, f"rollout_{rollout_state['saved']:03d}.png")
            _save_rollout_grid(frames, gt_slice, out_path)
            rollout_state["saved"] += 1
            rollout_state.setdefault("paths", []).append(out_path)

    A_np = A_to_use.detach().cpu().numpy()
    for idx in range(B):
        n = int(n_nodes[idx].item())
        if n <= 1:
            continue
        nodes = nodes_global_cpu[idx, :n]
        probs = A_np[idx, :n, :n]
        stitcher.add_subgraph_probs(nodes, probs)


def compute_split_metrics(
    P_global: np.ndarray,
    split: EdgeSplit,
    args,
    n2v_edge_score_fn=None,
    sup_masks: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    rng = np.random.default_rng(args.split_seed + 99)

    def scores_for_edges(edges: np.ndarray) -> np.ndarray:
        if edges.size == 0:
            return np.array([])
        return P_global[edges[:, 0], edges[:, 1]]

    def fpr_fnr(pos_scores: np.ndarray, neg_scores: np.ndarray, thresh: float = 0.5):
        if pos_scores.size == 0 or neg_scores.size == 0:
            return float("nan"), float("nan")
        y_true = np.concatenate(
            [np.ones_like(pos_scores), np.zeros_like(neg_scores)]
        )
        y_pred = np.concatenate([pos_scores, neg_scores])
        y_hat = (y_pred >= thresh).astype(np.int32)

        tp = np.sum((y_hat == 1) & (y_true == 1))
        fn = np.sum((y_hat == 0) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))
        tn = np.sum((y_hat == 0) & (y_true == 0))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        return float(fpr), float(fnr)

    def compute_mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
        """
        Compute MRR given:
        - pos_scores: shape [B]
        - neg_scores: shape [M]

        We treat the same negative score pool for every positive:
        - y_pred_pos: [B, 1]
        - y_pred_neg: [B, M]
        and use the "optimistic/pessimistic" tie-handling from your original code.
        """
        if pos_scores.size == 0 or neg_scores.size == 0:
            return float("nan")

        # Broadcast negatives for each positive
        y_pos = pos_scores.reshape(-1, 1)   # [B, 1]
        y_neg = neg_scores.reshape(1, -1)   # [1, M]

        optimistic_rank = (y_neg >= y_pos).sum(axis=1)
        pessimistic_rank = (y_neg > y_pos).sum(axis=1)
        ranks = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0  # [B]

        mrr = np.mean(1.0 / ranks.astype(np.float64))
        return float(mrr)


    for name, edges in [("val", split.val_edges), ("test", split.test_edges)]:
        if edges.size == 0:
            metrics[name] = {
                "auc": float("nan"),
                "ap": float("nan"),
                "fpr": float("nan"),
                "fnr": float("nan"),
            }
            if n2v_edge_score_fn is not None:
                metrics[name].update(
                    {
                        "n2v_auc": float("nan"),
                        "n2v_ap": float("nan"),
                        "n2v_fpr": float("nan"),
                        "n2v_fnr": float("nan"),
                    }
                )
            continue

        negatives = sample_non_edges(
            split.adj_full, edges.shape[0], seed=rng.integers(1e9)
        )

        # PIFM / stitched model
        pos_scores = scores_for_edges(edges)
        neg_scores = scores_for_edges(negatives)
        y_true = np.concatenate(
            [np.ones_like(pos_scores), np.zeros_like(neg_scores)]
        )
        y_pred = np.concatenate([pos_scores, neg_scores])

        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        fpr, fnr = fpr_fnr(pos_scores, neg_scores)
        mrr = compute_mrr(pos_scores, neg_scores) 

        split_metrics = {
            "auc": float(auc),
            "ap": float(ap),
            "fpr": fpr,
            "fnr": fnr,
            "mrr": mrr,
        }

        # Node2Vec / LPFormer baseline on the SAME edges / negatives
        if n2v_edge_score_fn is not None:
            n2v_pos = n2v_edge_score_fn(edges)
            n2v_neg = n2v_edge_score_fn(negatives)

            y_true_n2v = y_true  # same sets
            y_pred_n2v = np.concatenate([n2v_pos, n2v_neg])

            try:
                n2v_auc = roc_auc_score(y_true_n2v, y_pred_n2v)
                n2v_ap = average_precision_score(y_true_n2v, y_pred_n2v)
            except ValueError:
                n2v_auc, n2v_ap = float("nan"), float("nan")

            n2v_fpr, n2v_fnr = fpr_fnr(n2v_pos, n2v_neg)
            n2v_mrr = compute_mrr(n2v_pos, n2v_neg)

            split_metrics.update(
                {
                    "n2v_auc": float(n2v_auc),
                    "n2v_ap": float(n2v_ap),
                    "n2v_fpr": n2v_fpr,
                    "n2v_fnr": n2v_fnr,
                    "n2v_mrr": n2v_mrr,
                }
            )

            # === DEBUG: compare PIFM vs baseline on the SAME edges ===
            try:
                diff_pos = pos_scores - n2v_pos
                diff_neg = neg_scores - n2v_neg
                baseline_name = getattr(args, "subgraph_prior", "node2vec")
                print(
                    f"[DEBUG split={name}] mean abs diff vs {baseline_name} "
                    f"(pos): {np.mean(np.abs(diff_pos)):.4f} | "
                    f"(neg): {np.mean(np.abs(diff_neg)):.4f}"
                )
                if pos_scores.size > 1:
                    corr_pos = np.corrcoef(pos_scores, n2v_pos)[0, 1]
                else:
                    corr_pos = np.nan
                if neg_scores.size > 1:
                    corr_neg = np.corrcoef(neg_scores, n2v_neg)[0, 1]
                else:
                    corr_neg = np.nan
                print(
                    f"[DEBUG split={name}] corr(PIFM, {baseline_name}) "
                    f"pos={corr_pos:.4f}, neg={corr_neg:.4f}"
                )

                # print a few examples
                for i in range(min(5, edges.shape[0])):
                    e = edges[i]
                    print(
                        f"[DEBUG split={name}] edge {e}: "
                        f"PIFM={pos_scores[i]:.4f}, baseline={n2v_pos[i]:.4f}"
                    )
                for i in range(min(5, negatives.shape[0])):
                    e = negatives[i]
                    print(
                        f"[DEBUG split={name}] neg {e}: "
                        f"PIFM={neg_scores[i]:.4f}, baseline={n2v_neg[i]:.4f}"
                )
            except Exception as ex:
                print(f"[DEBUG split={name}] diff/corr debug failed: {ex}")

        # Hidden-edge-only metrics (test split): only evaluate on dropped edges
        if name == "test" and sup_masks is not None:
            sup_entry = sup_masks.get("test")
            sup_mask_test = None
            if isinstance(sup_entry, dict):
                sup_mask_test = sup_entry.get("sup")
            elif sup_entry is not None:
                sup_mask_test = sup_entry

            if sup_mask_test is not None and sup_mask_test.shape == split.adj_full.shape:
                # Evaluate strictly on the supervised (hidden) entries of the test mask
                iu = np.triu_indices(sup_mask_test.shape[0], k=1)
                sup_ut = sup_mask_test[iu] > 0.0
                if sup_ut.any():
                    rows = iu[0][sup_ut]
                    cols = iu[1][sup_ut]
                    hidden_edges_all = np.stack([rows, cols], axis=1)

                    # Split supervised pairs into true positives vs true negatives
                    is_pos = split.adj_full[rows, cols] > 0.5
                    pos_edges = hidden_edges_all[is_pos]
                    neg_edges = hidden_edges_all[~is_pos]

                    pos_scores_hidden = scores_for_edges(pos_edges) if pos_edges.size > 0 else np.array([])
                    neg_scores_hidden = scores_for_edges(neg_edges) if neg_edges.size > 0 else np.array([])

                    if pos_scores_hidden.size > 0 and neg_scores_hidden.size > 0:
                        y_true_hidden = np.concatenate(
                            [np.ones_like(pos_scores_hidden), np.zeros_like(neg_scores_hidden)]
                        )
                        y_pred_hidden = np.concatenate([pos_scores_hidden, neg_scores_hidden])

                        try:
                            auc_hidden = roc_auc_score(y_true_hidden, y_pred_hidden)
                            ap_hidden = average_precision_score(y_true_hidden, y_pred_hidden)
                        except ValueError:
                            auc_hidden, ap_hidden = float("nan"), float("nan")
                        fpr_hidden, fnr_hidden = fpr_fnr(pos_scores_hidden, neg_scores_hidden)
                        mrr_hidden = compute_mrr(pos_scores_hidden, neg_scores_hidden)
                    else:
                        auc_hidden = ap_hidden = fpr_hidden = fnr_hidden = mrr_hidden = float("nan")

                    split_metrics.update(
                        {
                            "auc_hidden": float(auc_hidden),
                            "ap_hidden": float(ap_hidden),
                            "fpr_hidden": fpr_hidden,
                            "fnr_hidden": fnr_hidden,
                            "mrr_hidden": mrr_hidden,
                            "hidden_pos_edges": int(is_pos.sum()),
                            "hidden_neg_edges": int((~is_pos).sum()),
                        }
                    )

                    if n2v_edge_score_fn is not None:
                        n2v_pos_hidden = n2v_edge_score_fn(pos_edges) if pos_edges.size > 0 else np.array([])
                        n2v_neg_hidden = n2v_edge_score_fn(neg_edges) if neg_edges.size > 0 else np.array([])

                        if n2v_pos_hidden.size > 0 and n2v_neg_hidden.size > 0:
                            y_pred_n2v_hidden = np.concatenate([n2v_pos_hidden, n2v_neg_hidden])
                            y_true_hidden_n2v = np.concatenate(
                                [np.ones_like(n2v_pos_hidden), np.zeros_like(n2v_neg_hidden)]
                            )
                            try:
                                n2v_auc_hidden = roc_auc_score(y_true_hidden_n2v, y_pred_n2v_hidden)
                                n2v_ap_hidden = average_precision_score(y_true_hidden_n2v, y_pred_n2v_hidden)
                            except ValueError:
                                n2v_auc_hidden, n2v_ap_hidden = float("nan"), float("nan")
                            n2v_fpr_hidden, n2v_fnr_hidden = fpr_fnr(n2v_pos_hidden, n2v_neg_hidden)
                            n2v_mrr_hidden = compute_mrr(n2v_pos_hidden, n2v_neg_hidden)
                        else:
                            n2v_auc_hidden = n2v_ap_hidden = n2v_fpr_hidden = n2v_fnr_hidden = n2v_mrr_hidden = float(
                                "nan"
                            )

                        split_metrics.update(
                            {
                                "n2v_auc_hidden": float(n2v_auc_hidden),
                                "n2v_ap_hidden": float(n2v_ap_hidden),
                                "n2v_fpr_hidden": n2v_fpr_hidden,
                                "n2v_fnr_hidden": n2v_fnr_hidden,
                                "n2v_mrr_hidden": n2v_mrr_hidden,
                            }
                        )
                else:
                    split_metrics.update(
                        {
                            "auc_hidden": float("nan"),
                            "ap_hidden": float("nan"),
                            "fpr_hidden": float("nan"),
                            "fnr_hidden": float("nan"),
                            "mrr_hidden": float("nan"),
                            "hidden_pos_edges": 0,
                        }
                    )

        metrics[name] = split_metrics


    return metrics



__all__ = [
    "train_subgraph_lp",
    "infer_subgraph_lp",
]
