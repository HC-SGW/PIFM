# pifm_sub/graphsage_prior.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .aggregators import MeanAggregator
from .encoders import Encoder
from torch_geometric.nn import SAGEConv


# =============================================================================
# Utils
# =============================================================================

def _adj_to_adjlists(adj: np.ndarray) -> Dict[int, set]:
    """
    Convert a dense [N,N] adjacency matrix (0/1 or weighted) 
    into a GraphSAGE-style adjacency list: {i: set(neighs)}.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square [N,N]")

    N = adj.shape[0]
    adj_lists: Dict[int, set] = {}

    for i in range(N):
        neighs = np.nonzero(adj[i] > 0.5)[0]
        # Drop self-loops if present
        neighs = [int(j) for j in neighs if j != i]
        adj_lists[i] = set(neighs)

    return adj_lists


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _sample_negatives(
    num_nodes: int,
    pos_edges: List[Tuple[int, int]],
    neg_ratio: float,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Uniform negative sampling from non-edges. Same spirit as node2vec_prior.
    """
    if neg_ratio <= 0 or not pos_edges:
        return []

    rng = np.random.default_rng(seed)
    pos_set = {(min(u, v), max(u, v)) for (u, v) in pos_edges}
    num_pos = len(pos_set)
    num_neg = int(num_pos * neg_ratio)
    if num_neg == 0:
        return []

    neg_edges: set[Tuple[int, int]] = set()
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


# =============================================================================
# Config + small modules
# =============================================================================


@dataclass
class GraphSAGEPriorConfig:
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    epochs: int
    lr: float
    neg_ratio: float
    edge_batch_size: int
    device: str = "auto"
    seed: int = 0
    
@dataclass
class GraphSAGEHeartPriorConfig:
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    epochs: int
    lr: float
    neg_ratio: float
    dropout: float
    weight_decay: float
    device: str = "auto"
    seed: int = 0



class _LinkPredictor(nn.Module):
    """
    Same structure as Node2Vec prior:
        score(u, v) = σ( wᵀ (z_u ⊙ z_v) + b )
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class GraphSAGELayer(nn.Module):
    """
    Simple mean-aggregator GraphSAGE layer:

        h_i^{k+1} = σ( W_self h_i^k + W_neigh mean_{j in N(i)} h_j^k )
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # H: [N, Din]
        # A: [N, N] 0/1 adjacency (train graph)
        deg = A.sum(dim=1, keepdim=True)  # [N,1]
        deg = deg.clamp(min=1.0)

        neigh_agg = A @ H  # [N, Din]
        neigh_mean = neigh_agg / deg

        out = self.lin_self(H) + self.lin_neigh(neigh_mean)
        return F.relu(out)


class GraphSAGEEncoder(nn.Module):
    """
    Full-batch GraphSAGE encoder over the train graph.

    Input: num_nodes, adjacency A (train-only), learnable node embeddings
    Output: node embeddings of size [N, embedding_dim]
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        # Learnable base node features
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        layers: List[GraphSAGELayer] = []
        if num_layers <= 1:
            layers.append(GraphSAGELayer(hidden_dim, embedding_dim))
        else:
            layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            layers.append(GraphSAGELayer(hidden_dim, embedding_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: [N, N]
        H = self.node_emb.weight  # [N, hidden_dim]
        for layer in self.layers:
            H = layer(H, A)
        return H  # [N, embedding_dim]

class HeartSAGEEncoder(nn.Module):
    """
    HEART-style GraphSAGE encoder using torch_geometric.nn.SAGEConv.
    We use learnable node embeddings as input features.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        convs: List[SAGEConv] = []

        if num_layers <= 1:
            # Single layer: directly go to embedding_dim
            in_dim = embedding_dim
            self.node_emb = nn.Embedding(num_nodes, in_dim)
            convs.append(SAGEConv(in_dim, embedding_dim))
        else:
            # First + middle layers use hidden_dim, last goes to embedding_dim
            in_dim = hidden_dim
            self.node_emb = nn.Embedding(num_nodes, in_dim)
            convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                convs.append(SAGEConv(hidden_dim, hidden_dim))
            convs.append(SAGEConv(hidden_dim, embedding_dim))

        self.convs = nn.ModuleList(convs)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, Din]
        x = self.node_emb.weight
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # [N, embedding_dim]



# =============================================================================
# Global cached state (mirrors Node2Vec prior)
# =============================================================================


_GLOBAL_SAGE_STATE: Dict[str, Any] = {
    "ready": False,
    "emb": None,        # torch.FloatTensor on CPU, shape (N, D)
    "predictor": None,  # _LinkPredictor on some device
    "device": None,
    "num_nodes": 0,
}


_GLOBAL_SAGE_HEART_STATE: Dict[str, Any] = {
    "ready": False,
    "emb": None,        # torch.FloatTensor on CPU, shape (N, D)
    "predictor": None,  # _LinkPredictor on some device
    "device": None,
    "num_nodes": 0,
}


# =============================================================================
# Training: GraphSAGE encoder + link predictor on TRAIN graph only
# =============================================================================
def _train_graphsage_on_train_graph(
    adj_train: np.ndarray,
    pos_edges: List[Tuple[int, int]],
    cfg: GraphSAGEPriorConfig,
):
    """
    Train a GraphSAGE-simple style encoder + logistic link predictor on the TRAIN graph.

    - Uses GraphSAGE-simple's Encoder + MeanAggregator.
    - Input "features" are learnable embeddings (no external features).
    - Optimizes BCE over (pos, neg) edge scores on the train graph.
    """
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    N = adj_train.shape[0]
    adj_lists = _adj_to_adjlists(adj_train)

    # ----- GraphSAGE-simple encoder -----
    # In the original code, "features" is either a feature matrix or an nn.Embedding.
    # Here we use learnable node embeddings of dimension cfg.embedding_dim.
    feat_dim = cfg.embedding_dim
    features = nn.Embedding(N, feat_dim)
    if device.type == "cuda":
        features = features.cuda()

    aggregator = MeanAggregator(
        features=features,
        cuda=(device.type == "cuda"),
        gcn=False,          # set True if you want the GCN-style variant
    )

    # GraphSAGE-simple Encoder returns [embed_dim, len(nodes)]
    encoder = Encoder(
        features=features,
        feature_dim=feat_dim,
        embed_dim=cfg.embedding_dim,
        adj_lists=adj_lists,
        aggregator=aggregator,
        num_sample=10,         # neighbors per node (you can expose this in cfg)
        base_model=None,
        gcn=False,
        cuda=(device.type == "cuda"),
        feature_transform=False,
    )
    encoder = encoder.to(device)

    # ----- Link predictor over pairwise embeddings (same as your Node2Vec prior) -----
    predictor = _LinkPredictor(cfg.embedding_dim).to(device)

    # Encoder already owns aggregator + features, so this includes all node parameters
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Convert positive edges to tensor
    pos_edges_arr = np.array(pos_edges, dtype=np.int64)
    pos_edges_t = torch.from_numpy(pos_edges_arr).to(device)
    num_nodes = N

    for epoch in range(cfg.epochs):
        encoder.train()
        predictor.train()

        # Fresh negative sampling each epoch using your helper
        neg_edges = _sample_negatives(
            num_nodes=num_nodes,
            pos_edges=pos_edges,
            neg_ratio=cfg.neg_ratio,
            seed=cfg.seed + epoch,
        )
        if not neg_edges:
            # fallback: no negatives → skip this epoch
            continue
        neg_edges_arr = np.array(neg_edges, dtype=np.int64)
        neg_edges_t = torch.from_numpy(neg_edges_arr).to(device)

        opt.zero_grad()

        # GraphSAGE-simple encoder: pass a list of node ids
        all_nodes = list(range(num_nodes))
        z = encoder(all_nodes)        # [embed_dim, N]
        z = z.t()                     # [N, embed_dim]

        pos_score = predictor(z[pos_edges_t[:, 0]] * z[pos_edges_t[:, 1]]).view(-1)
        neg_score = predictor(z[neg_edges_t[:, 0]] * z[neg_edges_t[:, 1]]).view(-1)

        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score),
            ],
            dim=0,
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"[GraphSAGEPrior] epoch {epoch+1}/{cfg.epochs} loss={loss.item():.4f}")

    # Final embeddings (no grad) cached on CPU
    encoder.eval()
    with torch.no_grad():
        z_final = encoder(list(range(num_nodes))).t().cpu()   # [N, D]

    predictor_cpu = predictor.cpu()
    return z_final, predictor_cpu, device


def _train_graphsage_heart_on_train_graph(
    adj_train: np.ndarray,
    pos_edges: List[Tuple[int, int]],
    cfg: GraphSAGEHeartPriorConfig,
):
    """
    Train a HEART-style GraphSAGE encoder (SAGEConv stack) + logistic link predictor
    on the TRAIN graph.
    """
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    N = adj_train.shape[0]

    # Build edge_index from dense adjacency (train graph only)
    src, dst = np.nonzero(adj_train > 0.5)
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0),
        dtype=torch.long,
        device=device,
    )

    encoder = HeartSAGEEncoder(
        num_nodes=N,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    predictor = _LinkPredictor(cfg.embedding_dim).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.lr,
    )

    pos_edges_arr = np.array(pos_edges, dtype=np.int64)
    pos_edges_t = torch.from_numpy(pos_edges_arr).to(device)

    for epoch in range(cfg.epochs):
        encoder.train()
        predictor.train()

        neg_edges = _sample_negatives(
            num_nodes=N,
            pos_edges=pos_edges,
            neg_ratio=cfg.neg_ratio,
            seed=cfg.seed + epoch,
        )
        if not neg_edges:
            continue

        neg_edges_arr = np.array(neg_edges, dtype=np.int64)
        neg_edges_t = torch.from_numpy(neg_edges_arr).to(device)

        opt.zero_grad()

        # Full-graph embeddings via HEART SAGE
        z = encoder(edge_index)  # [N, D]

        pos_score = predictor(z[pos_edges_t[:, 0]] * z[pos_edges_t[:, 1]]).view(-1)
        neg_score = predictor(z[neg_edges_t[:, 0]] * z[neg_edges_t[:, 1]]).view(-1)

        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score),
            ],
            dim=0,
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"[GraphSAGEHeartPrior] epoch {epoch+1}/{cfg.epochs} loss={loss.item():.4f}")

    encoder.eval()
    with torch.no_grad():
        z_final = encoder(edge_index).cpu()  # [N, D]

    predictor_cpu = predictor.cpu()
    return z_final, predictor_cpu, device


# =============================================================================
# Public initialization API: called from main_sub.train_subgraph_lp / infer_subgraph_lp
# =============================================================================


def init_global_graphsage_prior_from_adj(
    adj_train: np.ndarray,
    args,
    verbose: bool = True,
) -> None:
    """
    Initialize global GraphSAGE prior from a TRAIN adjacency matrix.

    - Uses only adj_train (train split) → no leakage of val/test edges.
    - Trains GraphSAGE encoder + logistic link predictor on TRAIN edges.
    - Caches (embeddings, predictor) for fast subgraph prior queries.
    """
    global _GLOBAL_SAGE_STATE

    if _GLOBAL_SAGE_STATE.get("ready", False):
        if verbose:
            print("[GraphSAGEPrior] ✅ Global GraphSAGE prior already initialized; reusing cached state.")
        return

    if adj_train.ndim != 2 or adj_train.shape[0] != adj_train.shape[1]:
        raise ValueError("adj_train must be a square [N,N] numpy array.")

    N = adj_train.shape[0]

    # Extract positive train edges (upper-triangular)
    iu = np.triu_indices(N, k=1)
    mask = adj_train[iu] > 0.5
    us = iu[0][mask]
    vs = iu[1][mask]
    pos_edges: List[Tuple[int, int]] = [(int(u), int(v)) for u, v in zip(us, vs)]

    cfg = GraphSAGEPriorConfig(
        embedding_dim=int(getattr(args, "graphsage_dim", 64)),
        hidden_dim=int(getattr(args, "graphsage_hidden_dim", 64)),
        num_layers=int(getattr(args, "graphsage_layers", 2)),
        epochs=int(getattr(args, "graphsage_epochs", 1000)),
        lr=float(getattr(args, "graphsage_lr", 1e-2)),
        neg_ratio=float(getattr(args, "graphsage_neg_ratio", 1.0)),
        edge_batch_size=int(getattr(args, "graphsage_edge_batch_size", 4096)),
        device=str(getattr(args, "graphsage_device", "auto")),
        seed=int(getattr(args, "seed", 0)),
    )

    if not pos_edges:
        if verbose:
            print("[GraphSAGEPrior] ⚠️ No positive train edges; prior will be all zeros.")
        _GLOBAL_SAGE_STATE.update(
            ready=True,
            emb=None,
            predictor=None,
            device=_resolve_device(cfg.device),
            num_nodes=N,
        )
        return

    if verbose:
        print(
            f"[GraphSAGEPrior] 🔧 Training global GraphSAGE on train graph "
            f"(N={N}, E={len(pos_edges)}, dim={cfg.embedding_dim})"
        )

    emb_cpu, predictor, device = _train_graphsage_on_train_graph(adj_train, pos_edges, cfg)

    _GLOBAL_SAGE_STATE.update(
        ready=True,
        emb=emb_cpu,               # [N,D] on CPU
        predictor=predictor.to(device).eval(),
        device=device,
        num_nodes=N,
    )

    if verbose:
        print("[GraphSAGEPrior] ✅ Global GraphSAGE prior initialized and cached.")


def init_global_graphsage_heart_prior_from_adj(
    adj_train: np.ndarray,
    args,
    verbose: bool = True,
) -> None:
    """
    Initialize global GraphSAGE-HEART prior from TRAIN adjacency.
    Uses HEART-style SAGEConv encoder.
    """
    global _GLOBAL_SAGE_HEART_STATE

    if _GLOBAL_SAGE_HEART_STATE.get("ready", False):
        if verbose:
            print("[GraphSAGEHeartPrior] ✅ Global GraphSAGE-HEART prior already initialized; reusing cached state.")
        return

    if adj_train.ndim != 2 or adj_train.shape[0] != adj_train.shape[1]:
        raise ValueError("adj_train must be a square [N,N] numpy array.")

    N = adj_train.shape[0]

    # Extract positive TRAIN edges (upper-triangular)
    iu = np.triu_indices(N, k=1)
    mask = adj_train[iu] > 0.5
    us = iu[0][mask]
    vs = iu[1][mask]
    pos_edges: List[Tuple[int, int]] = [(int(u), int(v)) for u, v in zip(us, vs)]

    cfg = GraphSAGEHeartPriorConfig(
        embedding_dim=int(getattr(args, "graphsage_heart_dim", 128)),
        hidden_dim=int(getattr(args, "graphsage_heart_hidden_dim", 128)),
        num_layers=int(getattr(args, "graphsage_heart_layers", 2)),
        epochs=int(getattr(args, "graphsage_heart_epochs", 200)),
        lr=float(getattr(args, "graphsage_heart_lr", 1e-2)),
        neg_ratio=float(getattr(args, "graphsage_heart_neg_ratio", 1.0)),
        dropout=float(getattr(args, "graphsage_heart_dropout", 0.5)),
        weight_decay=float(getattr(args, "subgraph_sage_heart_weight_decay", 1e-4)),
        device=str(getattr(args, "graphsage_heart_device", "auto")),
        seed=int(getattr(args, "seed", 0)),
    )

    if not pos_edges:
        if verbose:
            print("[GraphSAGEHeartPrior] ⚠️ No positive train edges; prior will be all zeros.")
        _GLOBAL_SAGE_HEART_STATE.update(
            ready=True,
            emb=None,
            predictor=None,
            device=_resolve_device(cfg.device),
            num_nodes=N,
        )
        return

    if verbose:
        print(
            f"[GraphSAGEHeartPrior] 🔧 Training global GraphSAGE-HEART on train graph "
            f"(N={N}, E={len(pos_edges)}, dim={cfg.embedding_dim})"
        )

    emb_cpu, predictor, device = _train_graphsage_heart_on_train_graph(adj_train, pos_edges, cfg)

    _GLOBAL_SAGE_HEART_STATE.update(
        ready=True,
        emb=emb_cpu,               # [N,D] on CPU
        predictor=predictor.to(device).eval(),
        device=device,
        num_nodes=N,
    )

    if verbose:
        print("[GraphSAGEHeartPrior] ✅ Global GraphSAGE-HEART prior initialized and cached.")


# =============================================================================
# Subgraph prior API (mirrors node2vec_prior)
# =============================================================================


@torch.no_grad()
def compute_graphsage_prior_batch(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    Compute structural prior for a batch of subgraphs using cached GraphSAGE embeddings.

    A_true / edge_mask are not used directly here (masking handled later in build_initial_A0_lp),
    we only use node_mask + nodes_global to slice the global embeddings.
    """
    del edge_mask  # not needed here; we only score

    device = A_true.device
    B, maxn, _ = A_true.shape
    priors = torch.zeros_like(A_true, device=device)

    state = _GLOBAL_SAGE_STATE
    if (
        not state.get("ready", False)
        or state.get("emb") is None
        or state.get("predictor") is None
    ):
        return priors

    emb_global = state["emb"].to(device)                # [N,D]
    predictor: _LinkPredictor = state["predictor"].to(device).eval()
    D = emb_global.size(1)

    for b in range(B):
        n = int(node_mask[b].sum().item())
        if n <= 1:
            continue

        g_nodes = nodes_global[b, :n].long()
        g_nodes = g_nodes[g_nodes >= 0]
        if g_nodes.numel() <= 1:
            continue

        z = emb_global[g_nodes]  # [n,D]

        # All pair Hadamard
        z_i = z.unsqueeze(1)         # [n,1,D]
        z_j = z.unsqueeze(0)         # [1,n,D]
        feats = (z_i * z_j).reshape(-1, D)  # [n*n, D]

        logits = predictor(feats).view(z.size(0), z.size(0))
        probs = torch.sigmoid(logits)

        # Symmetric, zero diag
        probs = torch.triu(probs, diagonal=1)
        probs = probs + probs.t()

        priors[b, :z.size(0), :z.size(0)] = probs

    return priors


@torch.no_grad()
def compute_graphsage_heart_prior_batch(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    Same as compute_graphsage_prior_batch but using GraphSAGE-HEART embeddings.
    """
    del edge_mask

    device = A_true.device
    B, maxn, _ = A_true.shape
    priors = torch.zeros_like(A_true, device=device)

    state = _GLOBAL_SAGE_HEART_STATE
    if (
        not state.get("ready", False)
        or state.get("emb") is None
        or state.get("predictor") is None
    ):
        return priors

    emb_global = state["emb"].to(device)                # [N,D]
    predictor: _LinkPredictor = state["predictor"].to(device).eval()
    D = emb_global.size(1)

    for b in range(B):
        n = int(node_mask[b].sum().item())
        if n <= 1:
            continue

        g_nodes = nodes_global[b, :n].long()
        g_nodes = g_nodes[g_nodes >= 0]
        if g_nodes.numel() <= 1:
            continue

        z = emb_global[g_nodes]  # [n,D]

        z_i = z.unsqueeze(1)         # [n,1,D]
        z_j = z.unsqueeze(0)         # [1,n,D]
        feats = (z_i * z_j).reshape(-1, D)  # [n*n, D]

        logits = predictor(feats).view(z.size(0), z.size(0))
        probs = torch.sigmoid(logits)

        probs = torch.triu(probs, diagonal=1)
        probs = probs + probs.t()

        priors[b, :z.size(0), :z.size(0)] = probs

    return priors


@torch.no_grad()
def compute_graphsage_prior_single(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
    """
    Single-sample wrapper (same signature style as node2vec_prior_single).
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

    prior_b = compute_graphsage_prior_batch(
        A_true=A_true_b,
        edge_mask=edge_mask_b,
        node_mask=node_mask_b,
        nodes_global=nodes_global_b,
        args=args,
    )
    return prior_b[0]

@torch.no_grad()
def compute_graphsage_heart_prior_single(
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    nodes_global: torch.Tensor,
    args,
) -> torch.Tensor:
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

    prior_b = compute_graphsage_heart_prior_batch(
        A_true=A_true_b,
        edge_mask=edge_mask_b,
        node_mask=node_mask_b,
        nodes_global=nodes_global_b,
        args=args,
    )
    return prior_b[0]


@torch.no_grad()
def compute_graphsage_scores_for_edges(edges: np.ndarray) -> np.ndarray:
    """
    Baseline scorer for specific global edges, used in compute_split_metrics.

    Assumes init_global_graphsage_prior_from_adj(...) has been called.

    If not ready, returns 0.5 for all edges (neutral baseline).
    """
    state = _GLOBAL_SAGE_STATE
    if (
        not state.get("ready", False)
        or state.get("emb") is None
        or state.get("predictor") is None
        or edges.size == 0
    ):
        return np.full(edges.shape[0], 0.5, dtype=np.float32)

    emb = state["emb"]
    predictor: _LinkPredictor = state["predictor"]
    device = predictor.linear.weight.device

    u = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    v = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)

    emb_t = emb.to(device)
    z_u = emb_t[u]
    z_v = emb_t[v]
    feats = z_u * z_v  # [E,D]

    logits = predictor(feats)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


@torch.no_grad()
def compute_graphsage_heart_scores_for_edges(edges: np.ndarray) -> np.ndarray:
    """
    Baseline scorer for GraphSAGE-HEART on specific global edges.
    """
    state = _GLOBAL_SAGE_HEART_STATE
    if (
        not state.get("ready", False)
        or state.get("emb") is None
        or state.get("predictor") is None
        or edges.size == 0
    ):
        return np.full(edges.shape[0], 0.5, dtype=np.float32)

    emb = state["emb"]
    predictor: _LinkPredictor = state["predictor"]
    device = predictor.linear.weight.device

    u = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    v = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)

        # [E,D]
    emb_t = emb.to(device)
    z_u = emb_t[u]
    z_v = emb_t[v]
    feats = z_u * z_v

    logits = predictor(feats)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


__all__ = [
    "init_global_graphsage_prior_from_adj",
    "compute_graphsage_prior_batch",
    "compute_graphsage_prior_single",
    "compute_graphsage_scores_for_edges",
    "init_global_graphsage_heart_prior_from_adj",
    "compute_graphsage_heart_prior_batch",
    "compute_graphsage_heart_prior_single",
    "compute_graphsage_heart_scores_for_edges",
]
