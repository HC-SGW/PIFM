"""
PIFM subgraph utilities package.

Provides k-hop ego-net sampling, context builders, datasets, and stitching
helpers used by the subgraph expansion pipeline.
"""

from .samplers import SubgraphCoverageSampler, capped_subsample, k_hop_egonet
from .context import laplacian_posenc, build_local_context
from .subgraph_dataset import SubgraphExpansionDataset, collate_subgraphs
from .stitch import LogitAveragingStitcher
from .node2vec_prior import (
    Node2VecPriorConfig,
    compute_node2vec_prior_batch,
    compute_node2vec_prior_single,
    init_global_node2vec_prior_from_adj,
    compute_node2vec_scores_for_edges,
)

__all__ = [
    "SubgraphCoverageSampler",
    "capped_subsample",
    "k_hop_egonet",
    "laplacian_posenc",
    "build_local_context",
    "SubgraphExpansionDataset",
    "collate_subgraphs",
    "LogitAveragingStitcher",
    "Node2VecPriorConfig",
    "compute_node2vec_prior_batch",
    "compute_node2vec_prior_single",
    "init_global_node2vec_prior_from_adj",
    "compute_node2vec_scores_for_edges",
]
