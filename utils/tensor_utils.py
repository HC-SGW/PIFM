import time
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_graphs(batch: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of adjacency matrices to the same size and return node masks."""
    batch = list(batch)
    max_n = max(g.size(0) for g in batch)
    B = len(batch)
    A_padded = torch.zeros(B, max_n, max_n, dtype=batch[0].dtype)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, A in enumerate(batch):
        n = A.size(0)
        A_padded[i, :n, :n] = A
        node_mask[i, :n] = True
    return A_padded, node_mask


def linear_coeffs(t: torch.Tensor):
    alpha = 1.0 - t
    beta = t
    alpha_dot = torch.full_like(t, -1.0)
    beta_dot = torch.ones_like(t)
    return alpha, beta, alpha_dot, beta_dot


def zero_diag_(M: torch.Tensor) -> torch.Tensor:
    M.diagonal().zero_()
    return M


def sym_zero_diag_valid(M: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Symmetrise, zero the diagonal, and mask invalid nodes."""
    if M.dim() == 2:
        nm = node_mask.to(M.dtype)
        pair = nm[:, None] * nm[None, :]
        M = M * pair
        ut = torch.triu(M, diagonal=1)
        M = ut + ut.T
        M.fill_diagonal_(0.0)
        return M * pair

    B, N, _ = M.shape
    nm = node_mask.to(M.dtype)
    pair = nm.unsqueeze(2) * nm.unsqueeze(1)
    M = M * pair

    ut_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=M.device), diagonal=1).unsqueeze(0)
    ut = M.masked_fill(~ut_mask, 0.0)
    M = ut + ut.transpose(1, 2)
    diag_mask = torch.eye(N, dtype=torch.bool, device=M.device).unsqueeze(0)
    M = M.masked_fill(diag_mask, 0.0)
    return M * pair


def add_masked_symmetric_noise(
    M: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    sigma: float,
    clip01: bool = True,
) -> torch.Tensor:
    """Add Gaussian noise on masked entries only, respecting symmetry and masks."""
    if sigma <= 0.0:
        return sym_zero_diag_valid(M, node_mask)

    unknown = (1.0 - edge_mask).to(M.dtype)
    unknown = sym_zero_diag_valid(unknown, node_mask)

    eps = torch.randn_like(M)
    eps = sym_zero_diag_valid(eps, node_mask)

    M_noisy = M + sigma * (eps * unknown)
    M_noisy = sym_zero_diag_valid(M_noisy, node_mask)
    if clip01:
        M_noisy.clamp_(0.0, 1.0)
    return M_noisy


def permute_square(A: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return A.index_select(0, p).index_select(1, p)


def invert_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


def si_gamma_coeffs(t: torch.Tensor, z_scale: float = 1.0, t_eps: float = 1e-3):
    """Stochastic interpolant coefficients with clamped time."""
    t = t.clamp(t_eps, 1.0 - t_eps)
    gamma = torch.sqrt(2.0 * t * (1.0 - t)) * z_scale
    gamma_dot = (1.0 - 2.0 * t) / torch.clamp(torch.sqrt(2.0 * t * (1.0 - t)), min=1e-12)
    gamma_dot = gamma_dot * z_scale
    return gamma, gamma_dot


def sample_graph_Z(
    B: int,
    N: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    Z = torch.randn(B, N, N, device=device)
    Z = sym_zero_diag_valid(Z, node_mask)
    return Z * (1.0 - edge_mask)


def upper_triu_mask_batched(node_mask: torch.Tensor) -> torch.Tensor:
    """Return [B,N,N] mask selecting valid-node upper-triangle entries."""
    B, N = node_mask.shape
    ut = torch.triu(torch.ones(N, N, dtype=torch.bool, device=node_mask.device), diagonal=1)
    ut = ut.unsqueeze(0).expand(B, -1, -1)
    pair = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    return ut & pair


def maybe_subsample(vec: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    """Randomly subsample vector rows if over `max_n` elements."""
    if vec.ndim == 2 and vec.shape[1] > 1:
        n = vec.shape[0]
    else:
        n = vec.size
    if n <= max_n:
        return vec
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return vec[idx] if vec.ndim == 1 else vec[idx, :]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


__all__ = [
    "set_seed",
    "collate_graphs",
    "linear_coeffs",
    "zero_diag_",
    "sym_zero_diag_valid",
    "add_masked_symmetric_noise",
    "permute_square",
    "invert_perm",
    "si_gamma_coeffs",
    "sample_graph_Z",
    "upper_triu_mask_batched",
    "maybe_subsample",
    "log",
]
