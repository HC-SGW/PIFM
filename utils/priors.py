import os
import re
from typing import List, Sequence, Tuple

import numpy as np
import torch


def _pc1_01(Z: np.ndarray) -> np.ndarray:
    """Project onto first principal component and rescale to [0,1]."""
    Zc = Z - Z.mean(0, keepdims=True)
    v = np.random.randn(Zc.shape[1])
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(20):
        v = Zc.T @ (Zc @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    s = Zc @ v
    s -= s.min()
    rng = s.max() - s.min()
    if rng < 1e-12:
        return np.linspace(0, 1, Z.shape[0])
    return s / rng


def load_priors_from_npy_dir(
    npy_dir: str, masks: Sequence[torch.Tensor], args
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load per-graph raw reconstructed adjacency .npy files from `npy_dir`.
    Returns tensors aligned with provided masks and their 1D coordinates.
    """
    _ = args  # placeholder to preserve call signature
    if not os.path.isdir(npy_dir):
        raise FileNotFoundError(f"N2V prior dir not found: {npy_dir}")

    filenames = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
    if len(filenames) == 0:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    filenames_sorted = sorted(filenames)
    L = len(masks)
    priors = []
    z1d_list = []

    for idx in range(L):
        pattern_candidates = [
            f
            for f in filenames_sorted
            if re.search(rf"(?i)([_\-\./])g{idx}([_\-\.])", f)
            or re.search(rf"(?i)g{idx}[_\.\-]", f)
        ]
        if pattern_candidates:
            pick = pattern_candidates[0]
        elif len(filenames_sorted) == L:
            pick = filenames_sorted[idx]
        else:
            raise FileNotFoundError(
                f"Cannot find an .npy prior for graph index {idx} in {npy_dir}.\n"
                f"Found {len(filenames_sorted)} .npy files: {filenames_sorted}\n"
                "Either name files with `_g{idx}_` or supply exactly one .npy per graph."
            )

        path = os.path.join(npy_dir, pick)
        try:
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if len(keys) == 0:
                    raise ValueError(f"{path} is empty .npz")
                arr = arr[keys[0]]
            arr = np.array(arr, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load numpy file {path}: {e}")

        M_mask = masks[idx]
        M_mask_np = M_mask.detach().cpu().numpy() if torch.is_tensor(M_mask) else np.array(M_mask)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Loaded prior {path} is not square: shape {arr.shape}")
        if arr.shape != M_mask_np.shape:
            raise ValueError(
                f"Shape mismatch for prior {path}: prior {arr.shape} vs mask {M_mask_np.shape}. "
                "Ensure saved priors align with your masks/samples."
            )

        arr_sym = (arr + arr.T) / 2.0
        np.fill_diagonal(arr_sym, 0.0)
        arr_sym = np.clip(arr_sym, 0.0, 1.0)

        try:
            z1d = _pc1_01(arr_sym).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to compute z1d for {path}: {e}")

        priors.append(torch.from_numpy(arr_sym.astype(np.float32)))
        z1d_list.append(torch.from_numpy(z1d))

    return priors, z1d_list

def load_priors_from_lpformer(
    lpformer_prior,
    masks: List[torch.Tensor],
    nodes_global_list: List[torch.Tensor],
    args,
    chunk_size: int = 65536,
):
    """
    Use a global LPFormer prior to compute dense prior matrices for each subgraph.

    Inputs
    ------
    lpformer_prior:
        Object with .device and .score_edges(edge_index, test_set=True, chunk_size=...)
    masks:
        List of [n_i, n_i] float/bool tensors; 1 indicates entries where we want
        LPFormer scores (we'll use only the strict upper triangle, i<j, and then symmetrize).
    nodes_global_list:
        List of 1D tensors with global node IDs for each subgraph.
        nodes_global_list[b].shape[0] == masks[b].shape[0].

    Returns
    -------
    priors_list:
        List of FloatTensor[n_i, n_i] on CPU, symmetric with zero diagonal.
    aux_info:
        Dict with a few debug stats (num_scored_edges, num_subgraphs).
    """
    device = getattr(lpformer_prior, "device", torch.device("cpu"))

    all_u = []
    all_v = []
    batch_ids = []
    rows = []
    cols = []

    # Collect all edges (global indices) we want to score
    for b, (mask, nodes_global) in enumerate(zip(masks, nodes_global_list)):
        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask)
        else:
            mask_t = mask
        mask_t = mask_t.to(dtype=torch.bool)

        n = mask_t.size(0)
        if n <= 1:
            continue

        if isinstance(nodes_global, np.ndarray):
            nodes_g = torch.from_numpy(nodes_global)
        else:
            nodes_g = nodes_global
        nodes_g = nodes_g.to(dtype=torch.long)

        # Only score strict upper triangle to avoid duplicates
        ut = torch.triu(mask_t, diagonal=1)
        idx_i, idx_j = torch.nonzero(ut, as_tuple=True)
        if idx_i.numel() == 0:
            continue

        u = nodes_g[idx_i]  # global IDs
        v = nodes_g[idx_j]

        all_u.append(u)
        all_v.append(v)

        batch_ids.append(torch.full((u.size(0),), b, dtype=torch.long))
        rows.append(idx_i)
        cols.append(idx_j)

    # If nothing to score, just return zero matrices
    if not all_u:
        priors_list = []
        for mask in masks:
            n = int(mask.shape[0])
            priors_list.append(torch.zeros((n, n), dtype=torch.float32))
        aux = {"num_scored_edges": 0, "num_subgraphs": len(masks)}
        return priors_list, aux

    u_all = torch.cat(all_u, dim=0)
    v_all = torch.cat(all_v, dim=0)
    batch_ids_all = torch.cat(batch_ids, dim=0)
    rows_all = torch.cat(rows, dim=0)
    cols_all = torch.cat(cols, dim=0)

    edge_index = torch.stack([u_all, v_all], dim=0).to(device)

    # Single global call into LPFormer (it will chunk internally again if needed)
    scores_all = lpformer_prior.score_edges(
        edge_index,
        test_set=True,
        chunk_size=chunk_size,
    )
    scores_all = scores_all.detach().cpu().view(-1)

    # Build dense [n_i, n_i] priors per subgraph
    priors_list: List[torch.Tensor] = []
    for mask in masks:
        n = int(mask.shape[0])
        prior_mat = torch.zeros((n, n), dtype=torch.float32)
        priors_list.append(prior_mat)

    # Scatter scores into the appropriate matrices (symmetrically)
    for b, i, j, s in zip(
        batch_ids_all.tolist(),
        rows_all.tolist(),
        cols_all.tolist(),
        scores_all.tolist(),
    ):
        priors_list[b][i, j] = s
        priors_list[b][j, i] = s

    aux = {
        "num_scored_edges": int(scores_all.numel()),
        "num_subgraphs": len(masks),
    }
    return priors_list, aux



__all__ = ["load_priors_from_npy_dir", "load_priors_from_lpformer"]
