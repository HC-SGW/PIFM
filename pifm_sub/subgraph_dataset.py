"""
Dataset utilities for SGDM-style subgraph expansion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

import os

from .context import build_local_context

from pifm_sub.samplers import sagress_sample_node_lists

from typing import Optional, Iterable, Tuple
import hashlib

def _neighbors_of(v: int, A: np.ndarray) -> np.ndarray:
    # Return 1-hop neighbors (excluding self)
    return np.flatnonzero(A[v] > 0.5)

import numpy as np
from typing import Tuple

def _k_hop_union(edge: Tuple[int, int], A: np.ndarray, k: int, max_nodes: int) -> np.ndarray:
    """
    Deterministic k-hop union around endpoints:
      - BFS by layers up to k
      - Iterate frontier and neighbors in sorted order
      - Clip by *discovery order* (keeps closest nodes first)
    """
    u, v = int(edge[0]), int(edge[1])
    N = A.shape[0]

    # Neighbors as sorted indices; threshold at 0.5 to treat A as binary
    def _nbrs(x: int) -> np.ndarray:
        return np.flatnonzero(A[x] > 0.5)

    visited = np.zeros(N, dtype=bool)
    order: list[int] = []

    # seed with {u, v} in sorted order
    frontier = [u, v]
    for x in sorted(frontier):
        if not visited[x]:
            visited[x] = True
            order.append(x)

    for _ in range(k):
        next_frontier: list[int] = []
        # process current frontier in sorted order for determinism
        for x in sorted(frontier):
            nbrs = np.sort(_nbrs(x))
            for y in nbrs:
                if not visited[y]:
                    visited[y] = True
                    order.append(y)
                    next_frontier.append(y)
                    if len(order) >= max_nodes:
                        return np.array(order[:max_nodes], dtype=np.int64)
        if not next_frontier:
            break
        frontier = next_frontier

    return np.array(order[:max_nodes], dtype=np.int64)

import numpy as np
from typing import Tuple

def make_global_masks(
    adj_full: np.ndarray,
    train_edges: np.ndarray,
    drop_p: float,
    neg_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build global edge_mask_full and supervision_mask_full on the FULL graph.

    - edge_mask_full[i,j] = 1 → this edge is kept as observed CONTEXT everywhere.
    - supervision_mask_full[i,j] = 1 → this pair is supervised (pos or neg) everywhere.
      (You will later intersect this with subgraphs.)

    Only TRAIN edges can become context or positive supervision.
    Negative supervision is sampled from true non-edges in adj_full.
    """
    adj_full = np.asarray(adj_full, dtype=np.float32)
    N = adj_full.shape[0]
    if adj_full.shape[0] != adj_full.shape[1]:
        raise ValueError("adj_full must be square.")

    train_edges = np.asarray(train_edges, dtype=np.int64)
    if train_edges.ndim != 2 or train_edges.shape[1] != 2:
        raise ValueError("train_edges must be shape [E,2].")

    # canonicalize undirected edges to (u < v) and dedup
    u = np.minimum(train_edges[:, 0], train_edges[:, 1])
    v = np.maximum(train_edges[:, 0], train_edges[:, 1])
    train_edges_uv = np.stack([u, v], axis=1)
    train_edges_uv = np.unique(train_edges_uv, axis=0)

    rng = np.random.default_rng(seed)

    edge_mask_full = np.zeros((N, N), dtype=np.float32)
    supervision_mask_full = np.zeros((N, N), dtype=np.float32)

    # --- POSITIVES: global “keep vs supervised” split for TRAIN edges ---
    if train_edges_uv.size > 0:
        keep_mask = rng.random(train_edges_uv.shape[0]) > drop_p
        kept_edges = train_edges_uv[keep_mask]
        dropped_edges = train_edges_uv[~keep_mask]

        # kept edges → context (edge_mask_full = 1)
        for (a, b) in kept_edges:
            edge_mask_full[a, b] = 1.0
            edge_mask_full[b, a] = 1.0

        # dropped edges → positive supervision (supervision_mask_full = 1)
        for (a, b) in dropped_edges:
            supervision_mask_full[a, b] = 1.0
            supervision_mask_full[b, a] = 1.0

    # --- NEGATIVES: global sampling from true non-edges in adj_full ---
    iu = np.triu_indices(N, k=1)
    is_edge_true = adj_full[iu] > 0.5         # all true edges (train+val+test)
    neg_candidates = np.where(~is_edge_true)[0]   # true non-edges

    num_pos_sup = int((supervision_mask_full[iu] > 0.0).sum())
    if num_pos_sup > 0 and neg_candidates.size > 0 and neg_ratio > 0:
        num_neg_sup = min(int(num_pos_sup * neg_ratio), neg_candidates.size)
        chosen = rng.choice(neg_candidates, size=num_neg_sup, replace=False)
        rows = iu[0][chosen]
        cols = iu[1][chosen]
        supervision_mask_full[rows, cols] = 1.0
        supervision_mask_full[cols, rows] = 1.0

    # no self-loops
    np.fill_diagonal(edge_mask_full, 0.0)
    np.fill_diagonal(supervision_mask_full, 0.0)

    return edge_mask_full, supervision_mask_full


class SubgraphExpansionDataset(Dataset):
    """Subgraph dataset supporting SaGress sampling via cfg or fallback k-hop BFS."""

    def __init__(
        self,
        adj_train_np: np.ndarray,
        pe_global_np: np.ndarray,
        *,
        adj_full_np: np.ndarray,
        seed_edges: np.ndarray,
        split_tag: str,
        k: int = 2,
        max_nodes: int = 256,
        target_coverage: int = 1,
        drop_p: float = 0.1,
        seed: int = 0,
        resample: bool = False,
        node_select_graph: str = "train",
        global_edge_mask: Optional[np.ndarray] = None,
        global_supervision_mask: Optional[np.ndarray] = None,
        dataset_cfg: Optional[dict] = None,
        use_local_context: bool = True,
        edge_centered_mask_target: bool = False,
    ):
        self.adj_train = np.asarray(adj_train_np, dtype=np.float32)
        self.adj_full = np.asarray(adj_full_np, dtype=np.float32)
        self.node_select_graph = str(node_select_graph)
        valid_graph_choices = ("full", "train", "val", "test")
        if self.node_select_graph not in valid_graph_choices:
            raise ValueError(f"node_select_graph must be one of {valid_graph_choices}.")

        # Optional SaGress config
        raw_cfg = dataset_cfg
        if isinstance(raw_cfg, dict) and "dataset" in raw_cfg:
            raw_cfg = raw_cfg["dataset"]
        self.dataset_cfg = raw_cfg
        self.use_sagress = raw_cfg is not None
        if self.use_sagress:
            required = [
                "sampling_method",
                "subgraph_size",
                "per_node_samples_rw",
                "per_node_samples_ego",
                "per_node_samples_unif",
                "ego_sample_radius",
            ]
            missing = [k for k in required if k not in raw_cfg]
            if missing:
                raise ValueError(f"dataset_cfg missing keys: {missing}")
            self.sampling_method = str(raw_cfg["sampling_method"]).strip()
            self.subgraph_size = int(raw_cfg["subgraph_size"])
            self.per_node_samples_rw = int(raw_cfg["per_node_samples_rw"])
            self.per_node_samples_ego = int(raw_cfg["per_node_samples_ego"])
            self.per_node_samples_unif = int(raw_cfg["per_node_samples_unif"])
            self.ego_sample_radius = int(raw_cfg["ego_sample_radius"])
        else:
            self.sampling_method = None
            self.subgraph_size = int(max_nodes)

        self.global_edge_mask = (
            None if global_edge_mask is None else np.asarray(global_edge_mask, dtype=np.float32)
        )
        self.global_supervision_mask = (
            None if global_supervision_mask is None else np.asarray(global_supervision_mask, dtype=np.float32)
        )
        if self.global_edge_mask is not None and self.global_edge_mask.shape != self.adj_full.shape:
            raise ValueError("global_edge_mask must match adjacency shape.")
        if self.global_supervision_mask is not None and self.global_supervision_mask.shape != self.adj_full.shape:
            raise ValueError("global_supervision_mask must match adjacency shape.")

        self.k = int(k)
        self.max_nodes = int(max_nodes)
        self.drop_p = float(drop_p)
        self.seed = int(seed)
        self.resample = bool(resample)
        self.split_tag = str(split_tag)
        self.target_coverage = int(target_coverage)
        self.use_local_context = bool(use_local_context)
        self.edge_centered_mask_target = bool(edge_centered_mask_target)

        seed_edges = np.asarray(seed_edges, dtype=np.int64)
        if seed_edges.ndim != 2 or seed_edges.shape[1] != 2:
            raise ValueError("seed_edges must be [M,2].")
        self.seed_edges = seed_edges
        self.epoch_seed_edges: List[Tuple[int, int]] = []
        self._edge_targets_local: List[Tuple[int, int]] = []

        self.pe = np.asarray(pe_global_np, dtype=np.float32)
        if self.pe.shape[0] != self.adj_train.shape[0]:
            raise ValueError("Global positional encodings must align with adjacency size.")

        if self.use_sagress:
            self.subgraph_node_lists = self._run_sagress_sampling()
            self.epoch_nodes = [np.asarray(nodes, dtype=np.int64) for nodes in self.subgraph_node_lists]
            # No specific target edges when using SaGress sampling.
            self.epoch_seed_edges = [(-1, -1)] * len(self.epoch_nodes)
            self._edge_targets_local = [(-1, -1)] * len(self.epoch_nodes)
            print(
                f"[SubgraphDataset] Generated {len(self.subgraph_node_lists)} subgraphs via {self.sampling_method}."
            )
        else:
            self.subgraph_node_lists = None
            self._build_epoch_nodes()

        self._lpformer_prior_files: Optional[List[Optional[str]]] = None
        self._lpformer_prior_cache_data: Optional[List[Optional[np.ndarray]]] = None

        digest = hashlib.sha1()
        for nodes in self.epoch_nodes:
            arr = np.asarray(nodes, dtype=np.int64)
            digest.update(arr.tobytes())
            digest.update(b"|")
        self._signature = digest.hexdigest()

    def _choose_sampling_adj(self) -> np.ndarray:
        # Default to train graph to avoid leaking held-out edges; allow "full" if explicitly requested.
        if self.node_select_graph == "full":
            return self.adj_full
        # "train", "val", "test" all map to the provided adj_train unless a separate split adjacency is added later.
        return self.adj_train

    def _run_sagress_sampling(self) -> List[List[int]]:
        adj_source = self._choose_sampling_adj()
        node_lists = sagress_sample_node_lists(
            adj_full=adj_source,
            sampling_method=self.sampling_method,
            subgraph_size=self.subgraph_size,
            per_node_samples_rw=self.per_node_samples_rw,
            per_node_samples_ego=self.per_node_samples_ego,
            num_uniform_samples=self.per_node_samples_unif,
            ego_radius=self.ego_sample_radius,
        )
        if not node_lists:
            raise RuntimeError("SaGress sampling produced no subgraphs.")
        return node_lists

    def _build_epoch_nodes(self) -> None:
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.seed_edges.shape[0])
        edges = self.seed_edges[order]
        A_for_bfs = self._choose_sampling_adj()
        epoch_nodes: List[np.ndarray] = []
        epoch_targets: List[Tuple[int, int]] = []
        epoch_seeds: List[Tuple[int, int]] = []
        for (u, v) in edges:
            nodes = _k_hop_union((int(u), int(v)), A_for_bfs, self.k, self.max_nodes)
            if nodes.size == 0:
                continue
            epoch_nodes.append(nodes)
            epoch_seeds.append((int(u), int(v)))
            try:
                u_idx = int(np.where(nodes == u)[0][0])
                v_idx = int(np.where(nodes == v)[0][0])
                epoch_targets.append((u_idx, v_idx))
            except Exception:
                epoch_targets.append((-1, -1))
        if not epoch_nodes:
            raise RuntimeError("No subgraphs could be constructed for this split.")
        self.epoch_nodes = epoch_nodes
        self.epoch_seed_edges = epoch_seeds
        self._edge_targets_local = epoch_targets

    def refresh_epoch(self) -> None:
        if not self.resample:
            return
        if self.use_sagress:
            self.subgraph_node_lists = self._run_sagress_sampling()
            self.epoch_nodes = [np.asarray(nodes, dtype=np.int64) for nodes in self.subgraph_node_lists]
            self.epoch_seed_edges = [(-1, -1)] * len(self.epoch_nodes)
            self._edge_targets_local = [(-1, -1)] * len(self.epoch_nodes)
        else:
            self._build_epoch_nodes()

    def __len__(self) -> int:
        return len(self.epoch_nodes)

    @property
    def signature(self) -> str:
        return self._signature

    def set_lpformer_prior_cache(self, file_paths: List[str]) -> None:
        if len(file_paths) != len(self.epoch_nodes):
            raise ValueError("Number of prior files must match dataset length.")
        self._lpformer_prior_files = list(file_paths)
        self._lpformer_prior_cache_data = [None] * len(file_paths)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        nodes = self.epoch_nodes[idx]
        if nodes.size == 0:
            raise RuntimeError("Encountered empty subgraph.")

        sub_full = self.adj_full[np.ix_(nodes, nodes)]
        sub_train = self.adj_train[np.ix_(nodes, nodes)]
        pe_sub = self.pe[nodes]
        if self.use_local_context:
            loc_ctx = build_local_context(sub_train)
        else:
            loc_ctx = np.zeros((sub_train.shape[0], 0), dtype=np.float32)

        if self.edge_centered_mask_target:
            edge_mask_kept = (sub_train > 0.5).astype(np.float32)
            supervision_mask = np.zeros_like(edge_mask_kept, dtype=np.float32)
            if idx < len(self._edge_targets_local):
                u_loc, v_loc = self._edge_targets_local[idx]
                n = edge_mask_kept.shape[0]
                if 0 <= u_loc < n and 0 <= v_loc < n:
                    edge_mask_kept[u_loc, v_loc] = 0.0
                    edge_mask_kept[v_loc, u_loc] = 0.0
                    supervision_mask[u_loc, v_loc] = 1.0
                    supervision_mask[v_loc, u_loc] = 1.0

            sample = {
                "nodes_global": nodes.astype(np.int64),
                "A_true": np.triu(sub_full, 1) + np.triu(sub_full, 1).T,
                "edge_mask": edge_mask_kept.astype(np.float32),
                "supervision_mask": supervision_mask.astype(np.float32),
                "pe_sub": pe_sub.astype(np.float32),
                "loc_ctx": loc_ctx.astype(np.float32),
            }

            if self._lpformer_prior_files:
                prior_path = self._lpformer_prior_files[idx]
                if prior_path and os.path.exists(prior_path):
                    cache = self._lpformer_prior_cache_data
                    mat = None
                    if cache is not None and cache[idx] is not None:
                        mat = cache[idx]
                    else:
                        mat = np.load(prior_path).astype(np.float32)
                        if cache is not None:
                            cache[idx] = mat
                    sample["lpformer_prior"] = mat

            return sample

        if self.global_edge_mask is not None and self.global_supervision_mask is not None:
            sub_edge_mask = self.global_edge_mask[np.ix_(nodes, nodes)]
            sub_sup_mask = self.global_supervision_mask[np.ix_(nodes, nodes)]
            edge_mask_kept = (sub_edge_mask > 0.5).astype(np.float32)
            supervision_mask = (sub_sup_mask > 0.5).astype(np.float32)
        else:
            if (self.global_edge_mask is not None) and (self.global_supervision_mask is not None):
                sub_edge_mask_full = self.global_edge_mask[np.ix_(nodes, nodes)]
                sub_sup_mask_full = self.global_supervision_mask[np.ix_(nodes, nodes)]
                sub_edge_mask_full = np.triu(sub_edge_mask_full, 1)
                sub_edge_mask_full = sub_edge_mask_full + sub_edge_mask_full.T
                sub_sup_mask_full = np.triu(sub_sup_mask_full, 1)
                sub_sup_mask_full = sub_sup_mask_full + sub_sup_mask_full.T
                edge_mask_kept = sub_edge_mask_full.astype(np.float32)
                supervision_mask = sub_sup_mask_full.astype(np.float32)
            else:
                rng = np.random.default_rng(self.seed + idx)
                N = sub_full.shape[0]
                iu = np.triu_indices(N, k=1)

                # Random drop over all upper-tri pairs
                drop_mask = rng.random(len(iu[0])) < self.drop_p

                edge_mask_kept = np.zeros_like(sub_full, dtype=np.float32)
                supervision_mask = np.zeros_like(sub_full, dtype=np.float32)

                if drop_mask.any():
                    rows = iu[0][drop_mask]; cols = iu[1][drop_mask]
                    supervision_mask[rows, cols] = 1.0
                if (~drop_mask).any():
                    rows = iu[0][~drop_mask]; cols = iu[1][~drop_mask]
                    edge_mask_kept[rows, cols] = 1.0

                edge_mask_kept = edge_mask_kept + edge_mask_kept.T
                supervision_mask = supervision_mask + supervision_mask.T
            edge_mask_kept = edge_mask_kept.astype(np.float32)
            supervision_mask = supervision_mask.astype(np.float32)

        sample = {
            "nodes_global": nodes.astype(np.int64),
            "A_true": np.triu(sub_full, 1) + np.triu(sub_full, 1).T,
            "edge_mask": edge_mask_kept.astype(np.float32),
            "supervision_mask": supervision_mask.astype(np.float32),
            "pe_sub": pe_sub.astype(np.float32),
            "loc_ctx": loc_ctx.astype(np.float32),
        }

        if self._lpformer_prior_files:
            prior_path = self._lpformer_prior_files[idx]
            if prior_path and os.path.exists(prior_path):
                cache = self._lpformer_prior_cache_data
                mat = None
                if cache is not None and cache[idx] is not None:
                    mat = cache[idx]
                else:
                    mat = np.load(prior_path).astype(np.float32)
                    if cache is not None:
                        cache[idx] = mat
                sample["lpformer_prior"] = mat

        return sample

def collate_subgraphs(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    Collate/dynamically pad a list of subgraph samples for batching.
    """
    if not batch:
        raise ValueError("Batch must contain at least one subgraph sample.")

    maxn = max(b["A_true"].shape[0] for b in batch)
    feat_dim_raw = batch[0]["pe_sub"].shape[1] + batch[0]["loc_ctx"].shape[1]
    feat_dim = max(1, feat_dim_raw)  # ensure at least one feature channel
    B = len(batch)

    A = torch.zeros(B, maxn, maxn, dtype=torch.float32)
    edge_mask = torch.zeros(B, maxn, maxn, dtype=torch.float32)
    supervision_mask = torch.zeros(B, maxn, maxn, dtype=torch.float32)
    node_mask = torch.zeros(B, maxn, dtype=torch.bool)
    x_feat = torch.zeros(B, maxn, feat_dim, dtype=torch.float32)
    nodes_global = torch.full((B, maxn), -1, dtype=torch.long)

    has_lpformer_prior = any("lpformer_prior" in sample for sample in batch)
    if has_lpformer_prior:
        lpformer_prior = torch.zeros(B, maxn, maxn, dtype=torch.float32)

    for i, sample in enumerate(batch):
        n = sample["A_true"].shape[0]
        A[i, :n, :n] = torch.from_numpy(sample["A_true"])
        edge_mask[i, :n, :n] = torch.from_numpy(sample["edge_mask"])
        supervision_mask[i, :n, :n] = torch.from_numpy(sample["supervision_mask"])
        node_mask[i, :n] = True

        xf = np.concatenate([sample["pe_sub"], sample["loc_ctx"]], axis=1)
        # Pad to match the allocated feature dimension if empty or short
        if xf.shape[1] == 0 and feat_dim == 1:
            xf = np.zeros((n, feat_dim), dtype=np.float32)
        elif xf.shape[1] < feat_dim:
            pad = np.zeros((n, feat_dim - xf.shape[1]), dtype=np.float32)
            xf = np.concatenate([xf, pad], axis=1)
        x_feat[i, :n] = torch.from_numpy(xf)
        nodes_global[i, :n] = torch.from_numpy(sample["nodes_global"]).long()

        if has_lpformer_prior and "lpformer_prior" in sample:
            lpformer_prior[i, :n, :n] = torch.from_numpy(sample["lpformer_prior"])

    batch_dict = {
        "A_true": A,
        "edge_mask": edge_mask,
        "supervision_mask": supervision_mask,
        "node_mask": node_mask,
        "x_feat": x_feat,
        "nodes_global": nodes_global,
    }
    if has_lpformer_prior:
        batch_dict["lpformer_prior"] = lpformer_prior

    return batch_dict
