"""
Stitching helpers to aggregate overlapping subgraph predictions.
"""

from __future__ import annotations

import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit transform."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LogitAveragingStitcher:
    """
    Aggregate edge probabilities from overlapping subgraphs via logit averaging.
    """

    def __init__(self, N: int):
        self.N = int(N)
        self.sum_logits = np.zeros((self.N, self.N), dtype=np.float64)
        self.counts = np.zeros((self.N, self.N), dtype=np.int32)

    def add_subgraph_probs(self, nodes: np.ndarray, P_sub: np.ndarray) -> None:
        if P_sub.shape[0] != P_sub.shape[1]:
            raise ValueError("Subgraph probability matrix must be square.")
        if P_sub.shape[0] != len(nodes):
            raise ValueError("Mismatch between probability matrix and node ids.")

        logits = logit(P_sub)
        idx = np.ix_(nodes, nodes)
        self.sum_logits[idx] += logits
        self.counts[idx] += 1

    def finalize(self) -> np.ndarray:
        mask = self.counts > 0
        logits = np.zeros_like(self.sum_logits, dtype=np.float64)
        logits[mask] = self.sum_logits[mask] / self.counts[mask]
        P = sigmoid(logits).astype(np.float32)
        np.fill_diagonal(P, 0.0)
        return P

