"""Conformal helpers for frequency-calibrated logit prediction sets."""

from __future__ import annotations

import math

import torch


def nu_nonconformity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute A_kappa(x, y)=s_max-s_y-kappa/sqrt(n_y+alpha)."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape (N, V)")
    if target_ids.dim() != 1 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError("target_ids must have shape (N,)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    s_max = logits.max(dim=-1).values
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    freqs = token_freq_table.to(logits.device).float()[target_ids]
    return s_max - target_logits - kappa / torch.sqrt(freqs + alpha)


def conformal_quantile(scores: torch.Tensor, delta: float) -> float:
    """Split-conformal finite-sample quantile for target miscoverage delta."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if scores.numel() == 0:
        raise ValueError("scores must be non-empty")

    n = scores.numel()
    rank = min(math.ceil((n + 1) * (1 - delta)), n)
    sorted_scores = torch.sort(scores.flatten()).values
    return float(sorted_scores[rank - 1].item())


def conformal_nu_scores(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute A_kappa(x, i) for every token in the vocabulary."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape (N, V)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    s_max = logits.max(dim=-1, keepdim=True).values
    freqs = token_freq_table.to(logits.device).float().unsqueeze(0).expand_as(logits)
    return s_max - logits - kappa / torch.sqrt(freqs + alpha)


def conformal_nu_keep_mask(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    q_hat: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return S_nu(x)={i:A_kappa(x,i)<=q_hat}."""
    return conformal_nu_scores(logits, token_freq_table, kappa, alpha) <= q_hat
