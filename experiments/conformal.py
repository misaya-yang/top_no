"""Conformal helpers for frequency-calibrated logit prediction sets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GroupQuantile:
    """One auditable Mondrian calibration result."""

    group: int
    count: int
    q_hat: float
    finite: bool
    reason: str


def _validate_delta(delta: float) -> float:
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise ValueError("delta must be numeric")
    value = float(delta)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return value


def _validate_score_tensor(scores: torch.Tensor, *, nonempty: bool = True) -> None:
    if not isinstance(scores, torch.Tensor):
        raise ValueError("scores must be a torch.Tensor")
    if scores.is_complex():
        raise ValueError("scores must be real-valued")
    if nonempty and scores.numel() == 0:
        raise ValueError("scores must be non-empty")
    if torch.isnan(scores).any():
        raise ValueError("scores must not contain NaN")


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        raise ValueError("logits must have shape (N, V)")
    if logits.shape[1] == 0:
        raise ValueError("logits vocabulary dimension must be non-empty")
    if logits.is_complex() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite and real-valued")


def _validate_target_ids(logits: torch.Tensor, target_ids: torch.Tensor) -> None:
    if (
        not isinstance(target_ids, torch.Tensor)
        or target_ids.dim() != 1
        or target_ids.shape[0] != logits.shape[0]
        or target_ids.dtype == torch.bool
        or target_ids.is_floating_point()
        or target_ids.is_complex()
    ):
        raise ValueError("target_ids must be an integer tensor with shape (N,)")
    if ((target_ids < 0) | (target_ids >= logits.shape[1])).any():
        raise ValueError("target_ids contains an out-of-range token ID")


def margin_scores(logits: torch.Tensor) -> torch.Tensor:
    """Return the C-margin nonconformity score for every candidate token."""
    _validate_logits(logits)
    return logits.max(dim=-1, keepdim=True).values - logits


def margin_nonconformity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the C-margin score of each observed target token."""
    _validate_logits(logits)
    _validate_target_ids(logits, target_ids)
    maximum = logits.max(dim=-1).values
    targets = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return maximum - targets


def _validate_nu_inputs(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float,
) -> None:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or alpha <= 0
    ):
        raise ValueError("alpha must be positive")
    if (
        isinstance(kappa, bool)
        or not isinstance(kappa, (int, float))
        or not math.isfinite(float(kappa))
    ):
        raise ValueError("kappa must be a finite number")
    if not isinstance(token_freq_table, torch.Tensor) or token_freq_table.dim() != 1:
        raise ValueError("token_freq_table must be one-dimensional")
    if token_freq_table.numel() != logits.shape[1]:
        raise ValueError("token_freq_table must match the logits vocabulary")
    if token_freq_table.is_complex() or not torch.isfinite(token_freq_table).all():
        raise ValueError("token_freq_table must be finite and real-valued")
    if (token_freq_table < 0).any():
        raise ValueError("token_freq_table must be non-negative")


def nu_nonconformity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute A_kappa(x, y)=s_max-s_y-kappa/sqrt(n_y+alpha)."""
    _validate_logits(logits)
    _validate_target_ids(logits, target_ids)
    _validate_nu_inputs(logits, token_freq_table, kappa, alpha)
    frequencies = token_freq_table.to(logits.device).float()[target_ids]
    return margin_nonconformity(logits, target_ids) - kappa / torch.sqrt(
        frequencies + alpha
    )


def conformal_quantile(scores: torch.Tensor, delta: float) -> float:
    """Split-conformal finite-sample quantile for target miscoverage delta."""
    delta = _validate_delta(delta)
    _validate_score_tensor(scores)
    n = scores.numel()
    rank = math.ceil((n + 1) * (1 - delta))
    if rank > n:
        return float("inf")
    sorted_scores = torch.sort(scores.flatten()).values
    return float(sorted_scores[rank - 1].item())


def dither_scores(
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply explicit U[0,1) tie dithers without touching global RNG state.

    Callers own uniform generation so the same frozen score function can be
    applied to calibration targets and test candidates independent of batching.
    APS boundary uniforms are part of APS itself and should not be silently
    combined with this second randomization. MPS callers must explicitly move
    scores to CPU because MPS cannot represent the required float64 result.
    """
    _validate_score_tensor(scores, nonempty=False)
    if scores.device.type == "mps":
        raise ValueError(
            "MPS does not support float64 score dithering; move scores and "
            "uniforms to CPU explicitly"
        )
    if not isinstance(uniforms, torch.Tensor) or uniforms.shape != scores.shape:
        raise ValueError("uniforms must be a tensor with the same shape as scores")
    if uniforms.is_complex() or not torch.isfinite(uniforms).all():
        raise ValueError("uniforms must be finite values in [0, 1)")
    if ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError("uniforms must be finite values in [0, 1)")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("epsilon must be a positive finite number")
    base = scores.to(dtype=torch.float64)
    noise = uniforms.to(device=scores.device, dtype=torch.float64)
    return base + float(epsilon) * noise


def mondrian_quantiles(
    scores: torch.Tensor,
    groups: torch.Tensor,
    delta: float,
    *,
    expected_groups: Sequence[int] | None = None,
    min_bucket: int | None = None,
) -> tuple[GroupQuantile, ...]:
    """Calibrate groupwise thresholds, using +inf for absent or small groups."""
    delta = _validate_delta(delta)
    _validate_score_tensor(scores, nonempty=False)
    if (
        not isinstance(groups, torch.Tensor)
        or groups.dim() != 1
        or scores.dim() != 1
        or groups.shape != scores.shape
        or groups.dtype == torch.bool
        or groups.is_floating_point()
        or groups.is_complex()
    ):
        raise ValueError("groups must be a one-dimensional integer tensor matching scores")
    if (groups < 0).any():
        raise ValueError("groups must contain non-negative integer IDs")
    floor = math.ceil(5.0 / delta) if min_bucket is None else min_bucket
    if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
        raise ValueError("min_bucket must be a positive integer")

    observed = {int(value) for value in torch.unique(groups).tolist()}
    if expected_groups is None:
        if not observed:
            raise ValueError("expected_groups is required when calibration is empty")
        all_groups = observed
    else:
        normalized = set()
        for group in expected_groups:
            if isinstance(group, bool) or not isinstance(group, int) or group < 0:
                raise ValueError("expected_groups must contain non-negative integers")
            normalized.add(group)
        if not observed.issubset(normalized):
            raise ValueError("groups contains an ID absent from expected_groups")
        all_groups = normalized

    results = []
    aligned_groups = groups.to(device=scores.device)
    for group in sorted(all_groups):
        group_scores = scores[aligned_groups == group]
        count = group_scores.numel()
        if count == 0:
            q_hat = float("inf")
            reason = "absent"
        elif count < floor:
            q_hat = float("inf")
            reason = "below_min_bucket"
        else:
            q_hat = conformal_quantile(group_scores, delta)
            reason = "finite" if math.isfinite(q_hat) else "rank_exceeds_n"
        results.append(
            GroupQuantile(
                group=group,
                count=count,
                q_hat=q_hat,
                finite=math.isfinite(q_hat),
                reason=reason,
            )
        )
    return tuple(results)


def descending_order(logits: torch.Tensor) -> torch.Tensor:
    """Return probability-descending token order with token-ID-stable ties."""
    _validate_logits(logits)
    return torch.argsort(logits, dim=-1, descending=True, stable=True)


def _validate_order(logits: torch.Tensor, order: torch.Tensor) -> None:
    if (
        not isinstance(order, torch.Tensor)
        or order.shape != logits.shape
        or order.dtype == torch.bool
        or order.is_floating_point()
        or order.is_complex()
    ):
        raise ValueError("order must be an integer tensor matching logits")
    if ((order < 0) | (order >= logits.shape[1])).any():
        raise ValueError("each order row must be a token-ID permutation")
    seen = torch.zeros_like(order, dtype=torch.bool)
    seen.scatter_(-1, order, True)
    if not seen.all():
        raise ValueError("each order row must be a token-ID permutation")


def aps_scores(
    logits: torch.Tensor,
    *,
    order: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Return APS scores for an explicit total order and boundary uniforms.

    For token i, A(i)=sum_{j before i} p_j + u_i p_i. With all u_i=0,
    ``A(i) <= q`` is deterministic nucleus sampling with the crossing token.
    """
    _validate_logits(logits)
    _validate_order(logits, order)
    if not isinstance(uniforms, torch.Tensor) or uniforms.shape != logits.shape:
        raise ValueError("uniforms must be a tensor matching logits")
    if uniforms.is_complex() or not torch.isfinite(uniforms).all():
        raise ValueError("uniforms must be finite values in [0, 1)")
    if ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError("uniforms must be finite values in [0, 1)")

    order = order.to(device=logits.device)
    uniforms = uniforms.to(device=logits.device, dtype=logits.dtype)
    probabilities = torch.softmax(logits, dim=-1)
    sorted_probabilities = probabilities.gather(-1, order)
    cumulative_mass = sorted_probabilities.cumsum(dim=-1)
    prefix_mass = torch.zeros_like(cumulative_mass)
    prefix_mass[..., 1:] = cumulative_mass[..., :-1]
    sorted_uniforms = uniforms.gather(-1, order)
    sorted_scores = prefix_mass + sorted_uniforms * sorted_probabilities
    scores = torch.empty_like(sorted_scores)
    scores.scatter_(-1, order, sorted_scores)
    return scores


def aps_nonconformity(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    order: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Return the APS score of each observed target token."""
    _validate_logits(logits)
    _validate_target_ids(logits, target_ids)
    return aps_scores(
        logits,
        order=order,
        uniforms=uniforms,
    ).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def conformal_nu_scores(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute A_kappa(x, i) for every token in the vocabulary."""
    _validate_logits(logits)
    _validate_nu_inputs(logits, token_freq_table, kappa, alpha)

    margins = margin_scores(logits)
    freqs = token_freq_table.to(logits.device).float().unsqueeze(0).expand_as(logits)
    return margins - kappa / torch.sqrt(freqs + alpha)


def conformal_nu_keep_mask(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    q_hat: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return S_nu(x)={i:A_kappa(x,i)<=q_hat}."""
    return conformal_nu_scores(logits, token_freq_table, kappa, alpha) <= q_hat
