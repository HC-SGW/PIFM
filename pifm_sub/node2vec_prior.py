"""
Global Node2Vec prior for the subgraph PIFM pipeline.

Design:
- Train ONE Node2Vec + logistic link predictor on the observed train graph
  (split.adj_train) BEFORE diffusion training.
- During diffusion (subgraph) training and inference, we only do cheap lookups:
  for each subgraph, gather global embeddings for `nodes_global` and score all
  pairs with the cached predictor to obtain a dense prior matrix.

This replaces the old per-subgraph / per-batch Node2Vec training, which was
extremely slow and conceptually misaligned with the PIFM prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import hashlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:  # pragma: no cover - import guard
    from torch_geometric.nn import Node2Vec
except ImportError:  # pragma: no cover
    Node2Vec = None
    
def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Config and simple link predictor
# =============================================================================


@dataclass
class Node2VecPriorConfig:
    embedding_dim: int
    walk_length: int
    walks_per_node: int
    context_size: int
    epochs: int
    lr: float
    batch_size: int
    neg_ratio: float
    clf_epochs: int
    clf_lr: float
    device: str = "auto"
    seed: int = 0


class _LinkPredictor(nn.Module):
    """
    Simple logistic link predictor on top of Hadamard products:

        score(u, v) = σ( wᵀ (z_u ⊙ z_v) + b )

    Mirrors the baseline used in baselines/train_node2vec_baseline.py.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        return self.linear(x).squeeze(-1)


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_device(cfg: Node2VecPriorConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(cfg.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _build_edge_index(pos_edges: np.ndarray,
                      num_nodes: int,
                      device: torch.device) -> torch.Tensor:
    """
    Build symmetric edge_index for Node2Vec from undirected edges.

    pos_edges: (E, 2) array with u < v.
    """
    if pos_edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    u = pos_edges[:, 0]
    v = pos_edges[:, 1]

    # undirected → add both (u,v) and (v,u)
    edges = np.concatenate(
        [np.stack([u, v], axis=1),
         np.stack([v, u], axis=1)],
        axis=0,
    )
    edge_index = torch.as_tensor(edges.T, dtype=torch.long, device=device)
    return edge_index


def _sample_negatives(num_nodes: int,
                      pos_edges: List[Tuple[int, int]],
                      neg_ratio: float,
                      seed: int) -> List[Tuple[int, int]]:
    """
    Uniform negative sampling from non-edges.
    """
    if neg_ratio <= 0 or not pos_edges:
        return []

    rng = np.random.default_rng(seed)
    pos_set = {(min(u, v), max(u, v)) for (u, v) in pos_edges}
    num_pos = len(pos_set)
    num_neg = int(num_pos * neg_ratio)
    if num_neg == 0:
        return []

    neg_edges = set()
    max_trials = num_neg * 50

    while len(neg_edges) < num_neg and max_trials > 0:
        max_trials -= 1
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in pos_set or (u, v) in neg_edges:
            continue
        neg_edges.add((u, v))

    return list(neg_edges)


def _train_node2vec_embedding(edge_index: torch.Tensor,
                              num_nodes: int,
                              cfg: Node2VecPriorConfig) -> torch.Tensor:
    """
    Train Node2Vec once on the (global) train graph.
    Returns embeddings of shape (num_nodes, D).
    """
    if Node2Vec is None:
        raise ImportError(
            "torch_geometric.nn.Node2Vec is required for Node2Vec prior but is not installed."
        )

    device = edge_index.device
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Clamp walk hyperparams to graph size
    walk_length = max(2, min(cfg.walk_length, max(2, num_nodes - 1)))
    context_size = max(1, min(cfg.context_size, walk_length))
    walks_per_node = max(1, cfg.walks_per_node)

    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=cfg.embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=1.0,
        q=1.0,
        num_nodes=num_nodes,
        sparse=True,
    ).to(device)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    loader = model.loader(batch_size=cfg.batch_size, shuffle=True , num_workers=0,generator=gen)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=cfg.lr)

    model.train()
    num_epochs = max(int(cfg.epochs), 1)
    for _ in range(num_epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device)).mean()
            loss.backward()
            optimizer.step()

    return model.embedding.weight.detach()  # (N, D)


def _train_link_predictor(
    emb: torch.Tensor,
    pos_edges: List[Tuple[int, int]],
    neg_edges: List[Tuple[int, int]],
    cfg: Node2VecPriorConfig,
    device: torch.device,
) -> _LinkPredictor:
    """
    Train a logistic regressor on top of Hadamard products of embeddings.
    Mirrors the baseline script's behavior.
    """
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    _seed_everything(cfg.seed)

    in_dim = emb.size(1)
    model = _LinkPredictor(in_dim).to(device)

    if not pos_edges or not neg_edges:
        # Not enough signal; leave near-zero, caller will fall back to A_obs where needed.
        return model

    # Build training tensors (support minibatching).
    def edge_to_feat(edges: List[Tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return torch.empty((0, in_dim), device=device)
        us = torch.tensor([u for (u, _) in edges], dtype=torch.long, device=device)
        vs = torch.tensor([v for (_, v) in edges], dtype=torch.long, device=device)
        x = emb[us] * emb[vs]  # (E, D)
        return x

    pos_x = edge_to_feat(pos_edges)
    neg_x = edge_to_feat(neg_edges)

    X = torch.cat([pos_x, neg_x], dim=0)
    y = torch.cat(
        [
            torch.ones(pos_x.size(0), device=device),
            torch.zeros(neg_x.size(0), device=device),
        ],
        dim=0,
    )

    if X.size(0) == 0:
        return model

    # Shuffle once
    perm = torch.randperm(X.size(0), device=device)
    X = X[perm]
    y = y[perm]

    batch_size = max(int(cfg.batch_size), 1)
    num_epochs = max(int(cfg.clf_epochs), 1)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.clf_lr)

    model.train()
    for _ in range(num_epochs):
        for start in range(0, X.size(0), batch_size):
            xb = X[start : start + batch_size]
            yb = y[start : start + batch_size]
            if xb.numel() == 0:
                continue
            optimizer.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            optimizer.step()

    return model


# =============================================================================
# Global cached prior state
# =============================================================================

_GLOBAL_N2V_STATE: Dict[str, Any] = {
    "ready": False,
    "emb": None,        # torch.FloatTensor on CPU, shape (N, D)
    "predictor": None,  # _LinkPredictor on some device
    "device": None,
    "num_nodes": 0,
}


def init_global_node2vec_prior_from_adj(
    adj_train: np.ndarray,
    args,
    verbose: bool = True,
) -> None:
    """
    Initialize the global Node2Vec prior from a train adjacency matrix.

    Call this ONCE before subgraph diffusion training / inference when
    using --subgraph_prior node2vec.

    - Uses only edges from adj_train (train split) → no leakage.
    - Trains Node2Vec on the full train graph.
    - Trains a logistic link predictor as in train_node2vec_baseline.py.
    - Caches (embeddings, predictor) for fast subgraph prior queries.
    """
    global _GLOBAL_N2V_STATE

    if _GLOBAL_N2V_STATE["ready"]:
        if verbose:
            print("[Node2VecPrior] ✅ Global Node2Vec prior already initialized; reusing cached state.")
        return

    if adj_train.ndim != 2 or adj_train.shape[0] != adj_train.shape[1]:
        raise ValueError("adj_train must be a square [N,N] numpy array.")

    N = adj_train.shape[0]
    # Upper-tri positive edges from train adjacency
    iu = np.triu_indices(N, k=1)
    mask = adj_train[iu] > 0.5
    us = iu[0][mask]
    vs = iu[1][mask]
    pos_edges: List[Tuple[int, int]] = [(int(u), int(v)) for u, v in zip(us, vs)]

    cfg = Node2VecPriorConfig(
        embedding_dim=int(getattr(args, "subgraph_n2v_dim", 32)),
        walk_length=int(getattr(args, "subgraph_n2v_walk_length", 8)),
        walks_per_node=int(getattr(args, "subgraph_n2v_walks_per_node", 4)),
        context_size=int(getattr(args, "subgraph_n2v_context_size", 4)),
        epochs=int(getattr(args, "subgraph_n2v_epochs", 15)),
        lr=float(getattr(args, "subgraph_n2v_lr", 1e-2)),
        batch_size=int(getattr(args, "subgraph_n2v_batch_size", 256)),
        neg_ratio=float(getattr(args, "subgraph_neg_ratio", 1.0)),
        clf_epochs=int(getattr(args, "subgraph_clf_epochs", 30)),
        clf_lr=float(getattr(args, "subgraph_clf_lr", 1e-2)),
        device=str(getattr(args, "subgraph_n2v_device", "auto")),
        seed=int(getattr(args, "seed", 0)),
    )

    device = _resolve_device(cfg)
    _seed_everything(cfg.seed)

    if not pos_edges:
        if verbose:
            print("[Node2VecPrior] ⚠️ No positive train edges; prior will be all zeros.")
        _GLOBAL_N2V_STATE.update(
            ready=True,
            emb=None,
            predictor=None,
            device=device,
            num_nodes=N,
        )
        return

    if verbose:
        print(f"[Node2VecPrior] 🔧 Training global Node2Vec on train graph "
              f"(N={N}, E={len(pos_edges)}, dim={cfg.embedding_dim}) on {device}.")

    pos_arr = np.array(pos_edges, dtype=np.int64)
    edge_index = _build_edge_index(pos_arr, N, device)

    # 1) Train Node2Vec embeddings
    emb = _train_node2vec_embedding(edge_index, N, cfg)

    # 2) Train logistic link predictor
    neg_edges = _sample_negatives(N, pos_edges, cfg.neg_ratio, cfg.seed)
    predictor = _train_link_predictor(emb.to(device), pos_edges, neg_edges, cfg, device)

    # Cache
    _GLOBAL_N2V_STATE.update(
        ready=True,
        emb=emb.detach().cpu(),
        predictor=predictor.to(device),
        device=device,
        num_nodes=N,
    )
    
    # === DEBUG: check compute_node2vec_prior_single vs compute_node2vec_scores_for_edges ===
    try:
        # pick a few random edges
        rng = np.random.default_rng(cfg.seed)
        N = adj_train.shape[0]
        test_pairs = []
        for _ in range(5):
            u = int(rng.integers(0, N))
            v = int(rng.integers(0, N))
            if u == v:
                v = (v + 1) % N
            test_pairs.append((u, v))
        arr = np.array(test_pairs, dtype=np.int64)

        # scores via baseline function
        baseline_scores = compute_node2vec_scores_for_edges(arr)

        # scores via subgraph prior API
        from .node2vec_prior import compute_node2vec_prior_single  # or adjust import if needed

        for (u, v), base_s in zip(test_pairs, baseline_scores):
            # build a minimal 2-node "subgraph"
            A_true = torch.zeros(2, 2, dtype=torch.float32)
            edge_mask = torch.zeros(2, 2, dtype=torch.float32)
            node_mask = torch.ones(2, dtype=torch.bool)
            nodes_global = torch.tensor([u, v], dtype=torch.long)

            prior_2 = compute_node2vec_prior_single(
                A_true=A_true,
                edge_mask=edge_mask,
                node_mask=node_mask,
                nodes_global=nodes_global,
                args=args,
            )
            s_sub = prior_2[0, 1].item()
            print(f"[DEBUG N2V] edge ({u},{v}): baseline={base_s:.4f}, subgraph_prior={s_sub:.4f}")
    except Exception as ex:
        print(f"[DEBUG N2V] error in sanity check: {ex}")


    if verbose:
        print("[Node2VecPrior] ✅ Global Node2Vec prior initialized and cached.")


# =============================================================================
# Public API used by main_sub.py
# =============================================================================
# --- NEW: local cache and tiny predictor ---
_N2V_SUBGRAPH_CACHE: Dict[str, torch.Tensor] = {}

class _TinyLinkPredictor(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(dim, 1, bias=True)
    def forward(self, hadamard_pairs: torch.Tensor) -> torch.Tensor:
        return self.lin(hadamard_pairs).squeeze(-1)

def _hash_key(nodes_np: np.ndarray, edge_mask_np: np.ndarray, cfg: Dict[str, int], seed: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(nodes_np.astype(np.int64).tobytes())
    # only upper triangle for stability
    tri = np.triu(edge_mask_np.astype(np.uint8), 1)
    h.update(tri.tobytes())
    for k in sorted(cfg.keys()):
        h.update(f"{k}={cfg[k]}".encode())
    h.update(f"seed={seed}".encode())
    return h.hexdigest()

def _edge_index_from_mask(mask_bool: torch.Tensor) -> torch.Tensor:
    # mask_bool: [n,n] with 1/True on observed edges (upper + lower)
    iu = torch.triu(mask_bool, diagonal=1)
    src, dst = torch.nonzero(iu, as_tuple=True)
    # undirected: add both directions
    e1 = torch.stack([src, dst], dim=0)
    e2 = torch.stack([dst, src], dim=0)
    return torch.cat([e1, e2], dim=1)  # [2, 2E]

@torch.no_grad()
def compute_node2vec_prior_batch(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
    # Masking is handled later in build_initial_A0_lp; we only score.
    del edge_mask

    device = A_true.device
    B, maxn, _ = A_true.shape
    priors = torch.zeros_like(A_true, device=device)

    state = _GLOBAL_N2V_STATE
    # If prior wasn't initialized, just return zeros (baseline A0 will be used).
    if not state.get("ready", False) or state.get("emb") is None or state.get("predictor") is None:
        return priors

    emb_global = state["emb"].to(device)           # [N, D], frozen
    predictor = state["predictor"].to(device).eval()  # frozen
    D = emb_global.size(1)

    for b in range(B):
        # how many valid nodes in this subgraph
        n = int(node_mask[b].sum().item())
        if n <= 1:
            continue

        g_nodes = nodes_global[b, :n].long()
        g_nodes = g_nodes[g_nodes >= 0]
        if g_nodes.numel() <= 1:
            continue

        z = emb_global[g_nodes]           # (n_sub, D)
        # Hadamard features for all pairs
        z_i = z.unsqueeze(1)              # (n_sub, 1, D)
        z_j = z.unsqueeze(0)              # (1, n_sub, D)
        feats = (z_i * z_j).reshape(-1, D)  # (n_sub*n_sub, D)

        logits = predictor(feats).view(z.size(0), z.size(0))
        probs = torch.sigmoid(logits)
        probs = torch.triu(probs, diagonal=1)
        probs = probs + probs.t()         # sym, zero diag
        priors[b, :z.size(0), :z.size(0)] = probs
        
        # --- DEBUG: compare with global edge scorer once ---
        if getattr(args, "debug_n2v_align", False) and not getattr(args, "_dbg_n2v_align_done", False):
            n_loc = z.size(0)
            if n_loc >= 2:
                # take a small set of local pairs
                edges_local = []
                for i in range(min(3, n_loc)):
                    for j in range(i + 1, min(3, n_loc)):
                        edges_local.append((int(i), int(j)))
                if edges_local:
                    g_nodes_np = g_nodes.cpu().numpy()
                    edges_global = np.array(
                        [[g_nodes_np[i], g_nodes_np[j]] for (i, j) in edges_local],
                        dtype=np.int64,
                    )
                    baseline = compute_node2vec_scores_for_edges(edges_global)
                    prior_vals = np.array([probs[i, j].item() for (i, j) in edges_local])
                    print("[DEBUG N2V align] edges_global:", edges_global)
                    print("[DEBUG N2V align] prior_vals :", prior_vals)
                    print("[DEBUG N2V align] baseline   :", baseline)
                    print("[DEBUG N2V align] max_abs_diff:",
                          float(np.max(np.abs(prior_vals - baseline))))
            args._dbg_n2v_align_done = True

    return priors



@torch.no_grad()
def compute_node2vec_prior_single(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    Single-sample wrapper matching the old signature.
    """
    if A_true.dim() == 2:
        A_true_b = A_true.unsqueeze(0)
        edge_mask_b = edge_mask.unsqueeze(0)
        node_mask_b = node_mask.unsqueeze(0)
        nodes_global_b = nodes_global.unsqueeze(0)
    else:
        A_true_b = A_true
        edge_mask_b = edge_mask
        node_mask_b = node_mask
        nodes_global_b = nodes_global

    prior_b = compute_node2vec_prior_batch(
        A_true=A_true_b,
        edge_mask=edge_mask_b,
        node_mask=node_mask_b,
        nodes_global=nodes_global_b,
        args=args,
    )
    return prior_b[0]


@torch.no_grad()
def compute_node2vec_scores_for_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given an array of global edges [[u,v], ...], return Node2Vec prior probs.

    Assumes init_global_node2vec_prior_from_adj(...) has been called.
    If the prior is not ready, returns 0.5 for all edges as a neutral baseline.
    """
    state = _GLOBAL_N2V_STATE
    if (
        not state["ready"]
        or state["emb"] is None
        or state["predictor"] is None
        or edges.size == 0
    ):
        return np.full(edges.shape[0], 0.5, dtype=np.float32)

    emb = state["emb"]
    predictor: _LinkPredictor = state["predictor"]
    device = predictor.linear.weight.device

    # Ensure numpy int64 → torch.long on correct device
    u = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    v = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)

    emb_t = emb.to(device)
    z_u = emb_t[u]
    z_v = emb_t[v]
    feats = z_u * z_v  # (E, D)

    logits = predictor(feats)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs



__all__ = [
    "init_global_node2vec_prior_from_adj",
    "compute_node2vec_prior_batch",
    "compute_node2vec_prior_single",
    "compute_node2vec_scores_for_edges",
]
