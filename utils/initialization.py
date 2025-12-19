from typing import Optional

import torch

from .tensor_utils import add_masked_symmetric_noise, sym_zero_diag_valid


def init_A0_gaussian_masked(
    A_anchor: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    mean: float = 0.5,
    var: float = 1.0,
    clip01: bool = True,
) -> torch.Tensor:
    """Fill masked region with Gaussian noise while keeping observed entries."""
    if var < 0.0:
        raise ValueError("init_gauss_var must be >= 0")
    sigma = float(var) ** 0.5

    U = torch.empty_like(A_anchor).normal_(mean=float(mean), std=sigma)
    U = sym_zero_diag_valid(U, node_mask)

    unknown = (1.0 - edge_mask).to(U.dtype)
    unknown = sym_zero_diag_valid(unknown, node_mask)

    A0 = edge_mask * A_anchor + unknown * U
    A0 = sym_zero_diag_valid(A0, node_mask)
    if clip01:
        A0.clamp_(0.0, 1.0)
    return A0


def init_gaussian_on_omega(
    Omega: torch.Tensor,
    node_mask: torch.Tensor,
    mean: float,
    var: float,
) -> torch.Tensor:
    """Sample Gaussian values on Ω (update region) and zero elsewhere."""
    anchor = torch.zeros_like(Omega)
    edge_mask = 1.0 - Omega
    return init_A0_gaussian_masked(anchor, node_mask, edge_mask, mean=mean, var=var, clip01=True)


def build_initial_A0(
    args,
    A_obs: torch.Tensor,
    node_mask: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    noise_std: float = 0.0,
    mean: Optional[float] = None,
    var: Optional[float] = None,
) -> torch.Tensor:
    """Construct initial A0 under the configured prior_init policy."""
    mode = getattr(args, "prior_init", "prior")
    if mode == "prior":
        if prior is None:
            raise ValueError('External prior required when prior_init="prior".')
        Omega = 1.0 - A_obs
        A0 = Omega * prior + A_obs
    elif mode == "gaussian":
        A0 = init_A0_gaussian_masked(
            A_anchor=A_obs,
            node_mask=node_mask,
            edge_mask=A_obs,
            mean=args.init_gauss_mean if mean is None else mean,
            var=args.init_gauss_var if var is None else var,
            clip01=True,
        )
    elif mode == "baseline":
        A0 = sym_zero_diag_valid(A_obs, node_mask)
    else:
        raise ValueError(f"Unsupported prior_init mode: {mode}")

    if noise_std > 0:
        A0 = add_masked_symmetric_noise(
            M=A0,
            node_mask=node_mask,
            edge_mask=A_obs,
            sigma=noise_std,
            clip01=True,
        )
    return A0


def build_initial_A0_lp(
    args,
    A_true: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Link-prediction initialization: keep observed entries from A_true and fill masked
    entries either from an external prior or Gaussian noise.
    """
    mode = getattr(args, "prior_init", "prior")
    anchor = sym_zero_diag_valid(edge_mask * A_true, node_mask)
    if mode == "prior":
        if prior is None:
            raise ValueError('External prior required when prior_init="prior".')
        prior = sym_zero_diag_valid(prior, node_mask)
        A0 = anchor + (1.0 - edge_mask) * prior
    elif mode == "gaussian":
        A0 = init_A0_gaussian_masked(
            A_anchor=anchor,
            node_mask=node_mask,
            edge_mask=edge_mask,
            mean=args.init_gauss_mean,
            var=args.init_gauss_var,
            clip01=True,
        )
    elif mode == "baseline":
        A0 = anchor.clone()
    else:
        raise ValueError(f"Unsupported prior_init mode: {mode}")
    return sym_zero_diag_valid(A0, node_mask)


__all__ = [
    "init_A0_gaussian_masked",
    "init_gaussian_on_omega",
    "build_initial_A0",
    "build_initial_A0_lp",
]
