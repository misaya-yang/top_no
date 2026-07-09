"""Stable calibrated-method identities and tensor-only execution adapters.

This module is deliberately independent of dataset/model loading. It binds the
PR-2a conformal primitives to canonical registry keys without unblocking the
paper runner before the remaining methods, suffstats, and gate are complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from conformal import (
    GroupQuantile,
    aps_nonconformity,
    aps_scores,
    conformal_nu_scores,
    conformal_quantile,
    descending_order,
    dither_scores,
    logprob_nonconformity,
    logprob_scores,
    margin_nonconformity,
    margin_scores,
    mondrian_quantiles,
    nu_nonconformity,
)


METHOD_REGISTRY_VERSION = "icml2027-method-registry-v1"


@dataclass(frozen=True)
class MethodDefinition:
    key: str
    score_family: str
    conditioning_axis: str
    paper_role: str
    randomization_kind: str
    implemented: bool


@dataclass(frozen=True)
class MethodCalibration:
    registry_version: str
    method_key: str
    delta: float
    n_calibration: int
    q_hat: float | None
    group_quantiles: tuple[GroupQuantile, ...]
    params: tuple[tuple[str, float], ...]
    group_axis: str | None
    dither_epsilon: float | None
    min_bucket: int | None


_REGISTRY = (
    MethodDefinition("c_margin", "margin", "none", "null", "score_dither", True),
    MethodDefinition(
        "c_logprob", "log_probability", "none", "baseline", "score_dither", True
    ),
    MethodDefinition(
        "c_zmargin",
        "normalized_margin",
        "context",
        "baseline",
        "score_dither",
        False,
    ),
    MethodDefinition(
        "aps", "cumulative_mass", "none", "baseline", "aps_boundary", True
    ),
    MethodDefinition(
        "raps",
        "regularized_cumulative_mass",
        "none",
        "baseline",
        "aps_boundary",
        False,
    ),
    MethodDefinition(
        "ts_aps",
        "cumulative_mass",
        "temperature",
        "baseline",
        "aps_boundary",
        False,
    ),
    MethodDefinition(
        "cns", "cumulative_mass", "entropy", "baseline", "aps_boundary", False
    ),
    MethodDefinition(
        "entropy_mondrian_margin",
        "margin",
        "entropy",
        "conditioning_control",
        "score_dither",
        True,
    ),
    MethodDefinition(
        "frequency_mondrian_margin",
        "margin",
        "frequency",
        "frequency_family",
        "score_dither",
        True,
    ),
    MethodDefinition(
        "learned_h",
        "learned_reliability",
        "frequency",
        "frequency_family",
        "score_dither",
        False,
    ),
    MethodDefinition(
        "learned_g", "offset_margin", "frequency", "ablation", "score_dither", False
    ),
    MethodDefinition(
        "c_nu", "offset_margin", "frequency", "ablation", "score_dither", True
    ),
)

PAPER_REQUIRED_METHOD_KEYS = frozenset(
    {
        "c_margin",
        "c_logprob",
        "c_zmargin",
        "aps",
        "raps",
        "ts_aps",
        "cns",
        "entropy_mondrian_margin",
        "frequency_mondrian_margin",
        "learned_h",
    }
)

_BY_KEY = {item.key: item for item in _REGISTRY}
if len(_BY_KEY) != len(_REGISTRY):
    raise RuntimeError("method registry contains duplicate keys")


def method_registry() -> tuple[MethodDefinition, ...]:
    """Return the immutable, canonical method registry."""
    return _REGISTRY


def implemented_method_keys() -> set[str]:
    return {item.key for item in _REGISTRY if item.implemented}


def missing_paper_method_keys() -> set[str]:
    return set(PAPER_REQUIRED_METHOD_KEYS - implemented_method_keys())


def get_method_definition(method_key: str) -> MethodDefinition:
    try:
        return _BY_KEY[method_key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown method key: {method_key!r}") from exc


def _require_implemented(method_key: str) -> MethodDefinition:
    definition = get_method_definition(method_key)
    if not definition.implemented:
        raise NotImplementedError(
            f"registered method {method_key!r} is not implemented in this PR-2b slice"
        )
    return definition


def _require_uniforms(logits: torch.Tensor, uniforms: torch.Tensor | None) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or not isinstance(uniforms, torch.Tensor):
        raise ValueError("uniforms must be an explicit tensor matching logits")
    if uniforms.shape != logits.shape:
        raise ValueError("uniforms must be an explicit tensor matching logits")
    if uniforms.is_complex() or not torch.isfinite(uniforms).all():
        raise ValueError("uniforms must contain finite values in [0, 1)")
    if ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError("uniforms must contain finite values in [0, 1)")
    return uniforms


def _normalize_params(
    method_key: str,
    params: Mapping[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    supplied = {} if params is None else dict(params)
    required = {"kappa", "alpha"} if method_key == "c_nu" else set()
    if set(supplied) != required:
        raise ValueError(
            f"{method_key} params must contain exactly {sorted(required)!r}"
        )
    normalized = []
    for key in sorted(supplied):
        value = supplied[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{method_key} param {key} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{method_key} param {key} must be finite")
        if key == "alpha" and value <= 0.0:
            raise ValueError("c_nu param alpha must be positive")
        normalized.append((key, value))
    return tuple(normalized)


def _params_dict(calibration: MethodCalibration) -> dict[str, float]:
    return dict(calibration.params)


def _validate_groups(groups: torch.Tensor | None, *, expected_length: int) -> torch.Tensor:
    if (
        not isinstance(groups, torch.Tensor)
        or groups.dim() != 1
        or groups.shape[0] != expected_length
        or groups.dtype == torch.bool
        or groups.is_floating_point()
        or groups.is_complex()
    ):
        raise ValueError(
            f"groups must be a one-dimensional integer tensor of length {expected_length}"
        )
    if (groups < 0).any():
        raise ValueError("groups must contain non-negative IDs")
    return groups


def _target_scores(
    method_key: str,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    token_freq_table: torch.Tensor | None,
    params: dict[str, float],
    uniforms: torch.Tensor,
    dither_epsilon: float,
) -> torch.Tensor:
    if method_key == "aps":
        return aps_nonconformity(
            logits,
            target_ids,
            order=descending_order(logits),
            uniforms=uniforms,
        )
    if method_key == "c_nu":
        if token_freq_table is None:
            raise ValueError("c_nu requires token_freq_table")
        scores = nu_nonconformity(
            logits,
            target_ids,
            token_freq_table,
            kappa=params["kappa"],
            alpha=params["alpha"],
        )
    elif method_key == "c_logprob":
        scores = logprob_nonconformity(logits, target_ids)
    else:
        scores = margin_nonconformity(logits, target_ids)
    rows = torch.arange(target_ids.shape[0], device=target_ids.device)
    target_uniforms = uniforms.to(device=target_ids.device)[rows, target_ids]
    return dither_scores(scores, target_uniforms, epsilon=dither_epsilon)


def calibrate_method(
    method_key: str,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    delta: float,
    uniforms: torch.Tensor | None = None,
    token_freq_table: torch.Tensor | None = None,
    params: Mapping[str, float] | None = None,
    groups: torch.Tensor | None = None,
    expected_groups: Sequence[int] | None = None,
    min_bucket: int | None = None,
    dither_epsilon: float = 1e-6,
) -> MethodCalibration:
    """Calibrate one canonical method from explicit tensor inputs only."""
    definition = _require_implemented(method_key)
    uniforms = _require_uniforms(logits, uniforms)
    normalized_params = _normalize_params(method_key, params)
    scores = _target_scores(
        method_key,
        logits,
        target_ids,
        token_freq_table=token_freq_table,
        params=dict(normalized_params),
        uniforms=uniforms,
        dither_epsilon=dither_epsilon,
    )

    if definition.conditioning_axis in {"frequency", "entropy"} and method_key.endswith(
        "mondrian_margin"
    ):
        calibration_groups = _validate_groups(
            groups,
            expected_length=target_ids.shape[0],
        )
        if expected_groups is None:
            raise ValueError("expected_groups is required for Mondrian calibration")
        group_quantiles = mondrian_quantiles(
            scores,
            calibration_groups,
            delta,
            expected_groups=expected_groups,
            min_bucket=min_bucket,
        )
        effective_min_bucket = (
            math.ceil(5.0 / float(delta)) if min_bucket is None else min_bucket
        )
        q_hat = None
        group_axis = definition.conditioning_axis
    else:
        if groups is not None or expected_groups is not None or min_bucket is not None:
            raise ValueError(f"{method_key} does not accept Mondrian groups")
        q_hat = conformal_quantile(scores, delta)
        group_quantiles = ()
        group_axis = None
        effective_min_bucket = None

    return MethodCalibration(
        registry_version=METHOD_REGISTRY_VERSION,
        method_key=method_key,
        delta=float(delta),
        n_calibration=int(target_ids.numel()),
        q_hat=q_hat,
        group_quantiles=group_quantiles,
        params=normalized_params,
        group_axis=group_axis,
        dither_epsilon=None if method_key == "aps" else float(dither_epsilon),
        min_bucket=effective_min_bucket,
    )


def _candidate_scores(
    calibration: MethodCalibration,
    logits: torch.Tensor,
    *,
    token_freq_table: torch.Tensor | None,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    if calibration.method_key == "aps":
        return aps_scores(
            logits,
            order=descending_order(logits),
            uniforms=uniforms,
        )
    if calibration.method_key == "c_nu":
        if token_freq_table is None:
            raise ValueError("c_nu requires token_freq_table")
        params = _params_dict(calibration)
        scores = conformal_nu_scores(
            logits,
            token_freq_table,
            kappa=params["kappa"],
            alpha=params["alpha"],
        )
    elif calibration.method_key == "c_logprob":
        scores = logprob_scores(logits)
    else:
        scores = margin_scores(logits)
    if calibration.dither_epsilon is None:
        raise ValueError("non-APS calibration is missing dither_epsilon")
    return dither_scores(scores, uniforms, epsilon=calibration.dither_epsilon)


def _group_thresholds(
    calibration: MethodCalibration,
    groups: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    quantiles = {item.group: item.q_hat for item in calibration.group_quantiles}
    observed = {int(value) for value in torch.unique(groups).tolist()}
    unknown = sorted(observed - set(quantiles))
    if unknown:
        raise ValueError(
            f"prediction groups lack a registered calibration group: {unknown!r}"
        )
    thresholds = torch.empty(groups.shape, device=device, dtype=dtype)
    groups_device = groups.to(device=device)
    for group in observed:
        thresholds[groups_device == group] = quantiles[group]
    return thresholds


def prediction_set_mask(
    calibration: MethodCalibration,
    logits: torch.Tensor,
    *,
    uniforms: torch.Tensor | None = None,
    token_freq_table: torch.Tensor | None = None,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a prediction-set mask from a frozen method calibration."""
    if not isinstance(calibration, MethodCalibration):
        raise ValueError("calibration must be a MethodCalibration")
    if calibration.registry_version != METHOD_REGISTRY_VERSION:
        raise ValueError("method calibration registry_version mismatch")
    definition = _require_implemented(calibration.method_key)
    uniforms = _require_uniforms(logits, uniforms)
    scores = _candidate_scores(
        calibration,
        logits,
        token_freq_table=token_freq_table,
        uniforms=uniforms,
    )

    if calibration.group_axis is None:
        if groups is not None:
            raise ValueError(f"{calibration.method_key} does not accept prediction groups")
        if calibration.q_hat is None:
            raise ValueError("global method calibration is missing q_hat")
        return scores <= calibration.q_hat

    if definition.conditioning_axis == "frequency":
        candidate_groups = _validate_groups(groups, expected_length=logits.shape[1])
        thresholds = _group_thresholds(
            calibration,
            candidate_groups,
            device=scores.device,
            dtype=scores.dtype,
        ).unsqueeze(0)
    elif definition.conditioning_axis == "entropy":
        context_groups = _validate_groups(groups, expected_length=logits.shape[0])
        thresholds = _group_thresholds(
            calibration,
            context_groups,
            device=scores.device,
            dtype=scores.dtype,
        ).unsqueeze(-1)
    else:
        raise ValueError("unsupported group axis in method calibration")
    return scores <= thresholds
