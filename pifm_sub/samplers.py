"""
Sampling utilities for the subgraph expansion pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import os

import numpy as np
import random
import copy
from typing import List
import torch
import networkx as nx
from tqdm import tqdm
from typing import Union

def _use_progress_bar() -> bool:
    flag = os.environ.get("PIFM_SAGRESS_PROGRESS", "")
    return flag.strip().lower() in ("1", "true", "yes", "on")

def _progress(iterable, desc: str):
    if _use_progress_bar():
        total = None
        try:
            total = len(iterable)  # NodeView/range/list support len()
        except TypeError:
            pass
        miniters = 1
        if total:
            miniters = max(1, total // 25)  # about 25 updates max
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            dynamic_ncols=True,
            mininterval=1.5,
            miniters=miniters,
        )
    return iterable


def _ensure_numpy_adj(adj: np.ndarray) -> np.ndarray:
    """Convert an adjacency matrix to a NumPy array of dtype bool."""
    if hasattr(adj, "detach"):  # torch.Tensor
        adj_np = adj.detach().cpu().numpy()
    else:
        adj_np = np.asarray(adj)
    return adj_np.astype(bool, copy=False)


def k_hop_egonet(adj: np.ndarray, center: int, k: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build the node set for a k-hop ego-net rooted at `center`.
    """
    adj_np = _ensure_numpy_adj(adj)
    N = adj_np.shape[0]
    if center < 0 or center >= N:
        raise ValueError(f"Center node {center} out of bounds for graph with {N} nodes.")

    visited = set([int(center)])
    frontier = [int(center)]
    for _ in range(max(k, 0)):
        nxt = set()
        for u in frontier:
            nbrs = np.flatnonzero(adj_np[u])
            for v in nbrs:
                if v != u:
                    nxt.add(int(v))
        visited |= nxt
        frontier = list(nxt)
        if not frontier:
            break

    nodes = np.array(sorted(visited), dtype=np.int64)
    mapping = {int(g): i for i, g in enumerate(nodes)}
    return nodes, mapping


def capped_subsample(nodes: np.ndarray, max_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly subsample nodes down to `max_nodes` while preserving the first entry.
    """
    if max_nodes <= 0:
        raise ValueError("max_nodes must be > 0")
    if nodes.shape[0] <= max_nodes:
        return nodes
    if max_nodes == 1:
        return np.array([nodes[0]], dtype=nodes.dtype)

    center = nodes[0]
    remainder = nodes[1:]
    if remainder.size < max_nodes - 1:
        # If we do not have enough remainder nodes (possible due to duplicates),
        # fall back to sampling with replacement but ensure uniqueness via np.unique.
        sampled = remainder
    else:
        sampled = rng.choice(remainder, size=max_nodes - 1, replace=False)
    keep = np.concatenate(([center], sampled))
    keep = np.unique(keep.astype(nodes.dtype, copy=False))
    if keep.size < max_nodes:
        # pad with random picks (allow replacement); this scenario is rare.
        missing = max_nodes - keep.size
        extra = rng.choice(nodes[1:], size=missing, replace=True)
        keep = np.concatenate((keep, extra))
    return np.sort(keep.astype(nodes.dtype, copy=False))


class SubgraphCoverageSampler:
    """
    Iterates node ids to build k-hop ego-nets with a per-epoch coverage target.
    """

    def __init__(
        self,
        adj: np.ndarray,
        k: int = 2,
        max_nodes: int = 256,
        target_coverage: int = 2,
        seed: int = 0,
    ):
        self.adj = _ensure_numpy_adj(adj)
        self.k = int(k)
        self.max_nodes = int(max_nodes)
        self.target_coverage = int(target_coverage)
        if self.target_coverage <= 0:
            raise ValueError("target_coverage must be > 0")
        self.rng = np.random.default_rng(seed)
        self.N = self.adj.shape[0]

    def epoch(self) -> Iterable[np.ndarray]:
        seen = np.zeros(self.N, dtype=np.int32)
        order = self.rng.permutation(self.N)
        idx = 0
        while (seen < self.target_coverage).any():
            if idx >= len(order):
                idx = 0
            center = int(order[idx])
            idx += 1

            nodes, _ = k_hop_egonet(self.adj, center, self.k)
            if nodes.size == 0:
                continue

            # Reorder to place the center first so that capped_subsample keeps it.
            if nodes[0] != center:
                mask = nodes != center
                reordered = np.concatenate(([center], nodes[mask]))
            else:
                reordered = nodes

            nodes_capped = capped_subsample(reordered, self.max_nodes, self.rng)
            nodes_sorted = np.sort(nodes_capped.astype(np.int64, copy=False))
            seen[nodes_sorted] += 1
            yield nodes_sorted



def build_nx_from_adj(adj: Union[np.ndarray, torch.Tensor]) -> nx.Graph:
    """
    Convert a dense adjacency (NumPy or torch) into an undirected NetworkX graph.
    Treat >0.5 as an edge.
    """
    if isinstance(adj, torch.Tensor):
        if adj.device.type != "cpu":
            adj = adj.cpu()
        A = adj.numpy()
    else:
        A = np.asarray(adj)

    # symmetrize & binarize
    A = ((A + A.T) > 0.5).astype(np.int32)
    G = nx.from_numpy_array(A)  # undirected
    return G


def _random_walk(G: nx.Graph, start_node: int, length: int) -> List[int]:
    """Simple random walk of given length on an undirected NetworkX graph."""
    prev_node = start_node
    walk = [prev_node]
    for _ in range(length - 1):
        neighbors = list(G.neighbors(prev_node))
        if not neighbors:
            # isolated node: restart at a random node
            prev_node = random.choice(list(G.nodes()))
        else:
            prev_node = random.choice(neighbors)
        walk.append(prev_node)
    return walk


def _dedupe_preserve_order(nodes: List[int]) -> List[int]:
    """Remove duplicates but keep first occurrence (closer to SaGress behavior)."""
    seen = set()
    out = []
    for v in nodes:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def sagress_random_walk_samples(
    G: nx.Graph,
    per_node_samples: int,
    subgraph_size: int,
) -> List[List[int]]:
    """
    For each node in G, run 'per_node_samples' random walks of length 'subgraph_size'.
    Return a list of node-id lists (one per subgraph). Duplicates in a walk
    are removed (order preserved), so effective subgraph size can be <= subgraph_size.
    """
    all_samples = []
    for n in _progress(G.nodes(), "SaGress Random Walks"):
        for _ in range(per_node_samples):
            rw = _random_walk(G, n, subgraph_size)
            rw_unique = _dedupe_preserve_order(rw)
            all_samples.append(rw_unique)
    return all_samples


def sagress_uniform_samples(
    G: nx.Graph,
    num_samples: int,
    subgraph_size: int,
) -> List[List[int]]:
    """
    Uniformly sample 'subgraph_size' distinct nodes from the graph for each sample.
    (SaGress uses random.choices, but using distinct nodes is usually nicer for adjacency.)
    """
    n_nodes = G.number_of_nodes()
    node_ids = list(range(n_nodes))
    samples = []
    for _ in _progress(range(num_samples), "SaGress Uniform Samples"):
        # sample distinct nodes if possible
        if subgraph_size <= n_nodes:
            subs = random.sample(node_ids, k=subgraph_size)
        else:
            subs = random.choices(node_ids, k=subgraph_size)
        samples.append(subs)
    return samples


def sagress_ego_samples(
    G: nx.Graph,
    per_node_samples: int,
    subgraph_size: int,
    radius: int = 2,
) -> List[List[int]]:
    """
    Ego-graph sampling around each node with radius 'radius', shrinking by
    randomly burning nodes until size <= subgraph_size (same logic as SaGress).
    """
    all_samples = []
    G_undirected = G.to_undirected()
    for n in _progress(G.nodes(), "SaGress Ego Sampling"):
        ego_net = nx.ego_graph(G_undirected, n, radius=radius)
        initial_size = len(ego_net.nodes())
        for _ in range(per_node_samples):
            temp_ego = copy.deepcopy(ego_net)
            temp_size = initial_size
            final_sample = list(temp_ego.nodes())
            # burn nodes until we’re small enough
            while temp_size > subgraph_size:
                n_nodes_to_burn = int(temp_size / 2)
                candidates = list(temp_ego.nodes())
                if n in candidates:
                    candidates.remove(n)  # keep center
                if not candidates:
                    break
                burned_nodes = random.choices(candidates, k=n_nodes_to_burn)
                temp_ego.remove_nodes_from(burned_nodes)
                # largest connected component ∪ {n}
                if temp_ego.number_of_nodes() == 0:
                    break
                largest_cc = max(nx.connected_components(temp_ego), key=len)
                final_sample = list(set(largest_cc) | {n})
                temp_size = len(final_sample)
            all_samples.append(final_sample)
    return all_samples


def sagress_sample_node_lists(
    adj_full: torch.Tensor,
    sampling_method: str,
    subgraph_size: int,
    per_node_samples_rw: int = 30,
    per_node_samples_ego: int = 10,
    num_uniform_samples: int = 5000,
    ego_radius: int = 2,
) -> List[List[int]]:
    """
    High-level helper: given a full adjacency and a sampling method,
    return a list of subgraph node lists.
    """
    G = build_nx_from_adj(adj_full)

    if sampling_method == "random_walk":
        node_lists = sagress_random_walk_samples(G, per_node_samples_rw, subgraph_size)

    elif sampling_method == "ego":
        node_lists = sagress_ego_samples(G, per_node_samples_ego, subgraph_size, radius=ego_radius)

    elif sampling_method == "uniform":
        node_lists = sagress_uniform_samples(G, num_uniform_samples, subgraph_size)

    elif sampling_method == "mix":
        # Mimic LargeGraphDataset.mix
        lists_rw = sagress_random_walk_samples(G, per_node_samples_rw, subgraph_size)
        lists_unif = sagress_uniform_samples(G, num_uniform_samples, subgraph_size)
        lists_ego = sagress_ego_samples(G, per_node_samples_ego, subgraph_size, radius=ego_radius)
        node_lists = lists_rw + lists_unif + lists_ego

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")

    return node_lists
