"""Frozen frequency bucket policies for diagnostics and calibrated methods."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DIAGNOSTIC_BUCKET_KIND = "diagnostic-log10-v1"
METHOD_BUCKET_KIND = "method-true-token-mass-quantile-v1"
METHOD_BUCKET_POLICY_SCHEMA_VERSION = "icml2027-method-bucket-policy-v1"
DIAGNOSTIC_BUCKET_LABELS = (
    "B0:n=0",
    "B1:1-9",
    "B2:10-99",
    "B3:100-999",
    "B4:1000-9999",
    "B5:10000-99999",
    "B6:100000-999999",
    "B7:1000000-9999999",
    "B8:n>=1e7",
)
_DIAGNOSTIC_LOWER_BOUNDARIES = (
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
)


@dataclass(frozen=True)
class MethodBucketPolicy:
    schema_version: str
    bucket_kind: str
    fit_split: str
    ordering_source: str
    initial_bucket_count: int
    delta_grid: tuple[float, ...]
    floor_multiplier: int
    floor_reference_delta: float
    minimum_tune_targets_per_bucket: int
    equal_frequency_tie_policy: str
    initial_cut_policy: str
    unseen_policy: str
    underfloor_selection_policy: str
    merge_neighbor_policy: str
    min_final_bucket_count: int
    failure_policy: str


_METHOD_POLICY_FIELDS = {
    "schema_version",
    "bucket_kind",
    "fit_split",
    "ordering_source",
    "initial_bucket_count",
    "delta_grid",
    "floor_multiplier",
    "floor_reference_delta",
    "minimum_tune_targets_per_bucket",
    "equal_frequency_tie_policy",
    "initial_cut_policy",
    "unseen_policy",
    "underfloor_selection_policy",
    "merge_neighbor_policy",
    "min_final_bucket_count",
    "failure_policy",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_method_bucket_policy(policy: MethodBucketPolicy) -> None:
    if not isinstance(policy, MethodBucketPolicy):
        raise ValueError("policy must be a MethodBucketPolicy")
    expected_strings = {
        "schema_version": METHOD_BUCKET_POLICY_SCHEMA_VERSION,
        "bucket_kind": METHOD_BUCKET_KIND,
        "fit_split": "tune",
        "ordering_source": "pinned-d-freq-token-count",
        "equal_frequency_tie_policy": "never-split",
        "initial_cut_policy": "cumulative-true-token-mass-right-crossing",
        "unseen_policy": "merge-into-lowest-nonzero-frequency-bucket",
        "underfloor_selection_policy": "smallest-count-then-lower-frequency",
        "merge_neighbor_policy": "smaller-tune-mass-then-lower-frequency",
        "failure_policy": "reject-if-fewer-than-two-valid-buckets",
    }
    for field_name, expected in expected_strings.items():
        if getattr(policy, field_name) != expected:
            raise ValueError(f"unsupported method bucket {field_name}")
    for field_name in (
        "initial_bucket_count",
        "floor_multiplier",
        "minimum_tune_targets_per_bucket",
        "min_final_bucket_count",
    ):
        value = getattr(policy, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    if policy.initial_bucket_count != 8:
        raise ValueError("initial_bucket_count must equal the pre-registered K0=8")
    if policy.floor_multiplier != 5:
        raise ValueError("floor_multiplier must equal the pre-registered value 5")
    if policy.min_final_bucket_count != 2:
        raise ValueError("min_final_bucket_count must equal 2")
    expected_grid = (0.2, 0.1, 0.05, 0.02, 0.01)
    if policy.delta_grid != expected_grid:
        raise ValueError("delta_grid must equal the pre-registered paper grid")
    if (
        isinstance(policy.floor_reference_delta, bool)
        or not isinstance(policy.floor_reference_delta, (int, float))
        or not math.isfinite(float(policy.floor_reference_delta))
        or float(policy.floor_reference_delta) != min(expected_grid)
    ):
        raise ValueError("floor_reference_delta must be the strictest paper delta")
    expected_floor = math.ceil(
        policy.floor_multiplier / float(policy.floor_reference_delta)
    )
    if policy.minimum_tune_targets_per_bucket != expected_floor:
        raise ValueError("minimum_tune_targets_per_bucket does not match 5/delta")
    if not 2 <= policy.min_final_bucket_count <= policy.initial_bucket_count:
        raise ValueError("min_final_bucket_count must be between 2 and initial K")


def load_method_bucket_policy(path: Path) -> MethodBucketPolicy:
    try:
        payload = json.loads(
            Path(path).read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed to read method bucket policy") from exc
    if not isinstance(payload, dict) or set(payload) != _METHOD_POLICY_FIELDS:
        raise ValueError("method bucket policy has invalid fields")
    delta_grid = payload["delta_grid"]
    if not isinstance(delta_grid, list):
        raise ValueError("delta_grid must be a list")
    payload = dict(payload)
    payload["delta_grid"] = tuple(delta_grid)
    try:
        policy = MethodBucketPolicy(**payload)
    except TypeError as exc:
        raise ValueError("method bucket policy is malformed") from exc
    validate_method_bucket_policy(policy)
    return policy


def method_bucket_policy_sha256(policy: MethodBucketPolicy) -> str:
    validate_method_bucket_policy(policy)
    payload = asdict(policy)
    payload["delta_grid"] = list(policy.delta_grid)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def diagnostic_frequency_groups(token_counts: torch.Tensor) -> torch.Tensor:
    """Map D_freq token counts to the fixed Fable5 diagnostic B0..B8 bands.

    These interpretable log10 bands are for Phase-0 plots only. They are not
    the D_tune-fitted true-token-mass quantile groups used by Mondrian methods.
    """
    if (
        not isinstance(token_counts, torch.Tensor)
        or token_counts.dim() != 1
        or token_counts.numel() == 0
        or token_counts.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("token_counts must be a non-empty integer vector")
    if (token_counts < 0).any():
        raise ValueError("token_counts must be non-negative")
    boundaries = torch.tensor(
        _DIAGNOSTIC_LOWER_BOUNDARIES,
        dtype=torch.int64,
        device=token_counts.device,
    )
    return torch.bucketize(
        token_counts.to(torch.int64),
        boundaries,
        right=True,
    ).to(torch.int64)
