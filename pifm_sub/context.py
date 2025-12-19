"""
Global and local context feature builders for the subgraph pipeline.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _ensure_np_float(adj):
    if hasattr(adj, "detach"):  # torch.Tensor
        return adj.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(adj, dtype=np.float32)


def laplacian_posenc(adj, k: int = 8) -> np.ndarray:
    """
    Compute the top-k Laplacian positional encodings on the training graph.
    """
    adj_np = _ensure_np_float(adj)
    N = adj_np.shape[0]
    if k <= 0:
        return np.zeros((N, 0), dtype=np.float32)
    if N == 0:
        return np.zeros((0, k), dtype=np.float32)

    A = sp.csr_matrix(adj_np)
    deg = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(deg)
    L = D - A

    k_eff = min(k, max(N - 2, 1))
    jitter = sp.identity(N, format="csr") * 1e-6
    try:
        vals, vecs = spla.eigsh(L, k=k_eff, M=D, which="SM")
    except Exception:
        # For singular/ill-conditioned matrices fall back to regularised Laplacian.
        vals, vecs = spla.eigsh(L + jitter, k=k_eff, which="SM")

    PE = np.asarray(vecs, dtype=np.float32)
    if PE.shape[1] < k:
        pad = np.zeros((N, k - PE.shape[1]), dtype=np.float32)
        PE = np.concatenate([PE, pad], axis=1)
    return PE


def build_local_context(adj_sub: np.ndarray) -> np.ndarray:
    """
    Simple local context features (degree and normalised degree) for a subgraph.
    """
    adj_np = _ensure_np_float(adj_sub)
    degrees = adj_np.sum(axis=1, keepdims=True).astype(np.float32)
    max_possible = max(1.0, float(adj_np.shape[0] - 1))
    norm_deg = degrees / max_possible
    return np.concatenate([degrees, norm_deg], axis=1).astype(np.float32)
