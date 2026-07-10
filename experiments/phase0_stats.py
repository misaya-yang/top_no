"""Streaming sufficient statistics for the Phase-0 margin/frequency pilot."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Collection, Sequence

import numpy as np
import torch

from frequency_buckets import DIAGNOSTIC_BUCKET_LABELS, diagnostic_frequency_groups


_N_FREQUENCY_GROUPS = len(DIAGNOSTIC_BUCKET_LABELS)
_SHIFT_GRID = np.arange(-2.0, 2.0001, 0.05)


@dataclass(frozen=True)
class GridSpec:
    margin_edges: tuple[float, ...]
    frequency_labels: tuple[str, ...] = DIAGNOSTIC_BUCKET_LABELS
    min_true_count: int = 20

    def __post_init__(self) -> None:
        if len(self.margin_edges) < 2 or not math.isinf(self.margin_edges[-1]):
            raise ValueError("margin_edges must end in an open-ended infinity bin")
        finite = self.margin_edges[:-1]
        if any(not math.isfinite(value) or value <= 0 for value in finite):
            raise ValueError("finite margin_edges must be positive and finite")
        if tuple(sorted(finite)) != finite or len(set(finite)) != len(finite):
            raise ValueError("margin_edges must be strictly increasing")
        if self.frequency_labels != DIAGNOSTIC_BUCKET_LABELS:
            raise ValueError("frequency_labels must use the frozen diagnostic policy")
        if (
            isinstance(self.min_true_count, bool)
            or not isinstance(self.min_true_count, int)
            or self.min_true_count <= 0
        ):
            raise ValueError("min_true_count must be a positive integer")

    @classmethod
    def default(cls, *, min_true_count: int = 20) -> "GridSpec":
        fine = tuple(float(value) for value in np.arange(0.25, 10.0001, 0.25))
        coarse = tuple(float(value) for value in np.arange(10.5, 20.0001, 0.5))
        return cls(fine + coarse + (float("inf"),), min_true_count=min_true_count)

    @property
    def n_margin_bins(self) -> int:
        return len(self.margin_edges)

    def margin_bin_indices(self, margins: torch.Tensor) -> torch.Tensor:
        if not isinstance(margins, torch.Tensor) or margins.is_complex():
            raise ValueError("margins must be a real torch.Tensor")
        if not torch.isfinite(margins).all():
            raise ValueError("margins must be finite")
        if (margins < 0).any():
            raise ValueError("margins must be non-negative")
        boundaries = torch.tensor(
            self.margin_edges[:-1],
            device=margins.device,
            dtype=margins.dtype,
        )
        return torch.bucketize(margins.contiguous(), boundaries, right=False)

    def margin_bin_centers(self) -> tuple[float, ...]:
        centers = []
        lower = 0.0
        for upper in self.margin_edges:
            if math.isinf(upper):
                centers.append(lower + 0.5)
            else:
                centers.append((lower + upper) / 2.0)
                lower = upper
        return tuple(centers)


@dataclass(frozen=True)
class DocumentGridStats:
    doc_id: str
    half: int
    num: torch.Tensor
    den: torch.Tensor
    perm_num: torch.Tensor
    perm_den: torch.Tensor
    n_positions: int


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return seed


def document_half(doc_id: str, *, seed: int) -> int:
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    seed = _validate_seed(seed)
    payload = f"icml2027-phase0-half-v1\x00{seed}\x00{doc_id}".encode()
    return hashlib.sha256(payload).digest()[0] & 1


def permuted_frequency_groups(groups: torch.Tensor, *, seed: int) -> torch.Tensor:
    seed = _validate_seed(seed)
    if (
        not isinstance(groups, torch.Tensor)
        or groups.dim() != 1
        or groups.dtype == torch.bool
        or groups.is_floating_point()
        or groups.is_complex()
    ):
        raise ValueError("groups must be a one-dimensional integer tensor")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cpu_groups = groups.detach().to(device="cpu", dtype=torch.long)
    order = torch.randperm(cpu_groups.numel(), generator=generator)
    return cpu_groups[order]


def _normalized_exclusions(values: Collection[int], vocab_size: int) -> tuple[int, ...]:
    normalized = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("excluded_token_ids must contain integers")
        if value < 0 or value >= vocab_size:
            raise ValueError("excluded_token_ids contains an out-of-range ID")
        normalized.add(value)
    return tuple(sorted(normalized))


def _empty_stats(doc_id: str, grid: GridSpec, seed: int) -> DocumentGridStats:
    shape = (grid.n_margin_bins, _N_FREQUENCY_GROUPS)
    zeros = torch.zeros(shape, dtype=torch.int64)
    return DocumentGridStats(
        doc_id=doc_id,
        half=document_half(doc_id, seed=seed),
        num=zeros.clone(),
        den=zeros.clone(),
        perm_num=zeros.clone(),
        perm_den=zeros.clone(),
        n_positions=0,
    )


def accumulate_document(
    doc_id: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    grid: GridSpec,
    excluded_token_ids: Collection[int],
    permutation_seed: int,
) -> DocumentGridStats:
    """Accumulate real and frequency-permuted grids for one document."""
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be GridSpec")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.dim() != 2
        or logits.is_complex()
    ):
        raise ValueError("logits must be a real two-dimensional tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain finite values")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dim() != 1
        or targets.shape[0] != logits.shape[0]
        or targets.dtype == torch.bool
        or targets.is_floating_point()
        or targets.is_complex()
    ):
        raise ValueError("targets must be a one-dimensional integer tensor matching logits")
    vocab_size = logits.shape[1]
    if ((targets < 0) | (targets >= vocab_size)).any():
        raise ValueError("targets contains an out-of-range token ID")
    groups_cpu = diagnostic_frequency_groups(token_counts)
    if groups_cpu.numel() != vocab_size:
        raise ValueError("token_counts must match logits vocabulary")
    exclusions = _normalized_exclusions(excluded_token_ids, vocab_size)
    permutation_seed = _validate_seed(permutation_seed)
    if logits.shape[0] == 0:
        return _empty_stats(doc_id, grid, permutation_seed)

    allowed_cpu = torch.ones(vocab_size, dtype=torch.bool)
    if exclusions:
        allowed_cpu[list(exclusions)] = False
    targets_cpu = targets.detach().to(device="cpu", dtype=torch.long)
    valid_rows_cpu = allowed_cpu[targets_cpu]
    if not valid_rows_cpu.any():
        return _empty_stats(doc_id, grid, permutation_seed)

    valid_rows = valid_rows_cpu.to(device=logits.device)
    working = logits[valid_rows].float()
    valid_targets = targets[valid_rows].long()
    margins = working.max(dim=-1, keepdim=True).values - working
    margin_bins = grid.margin_bin_indices(margins)
    allowed = allowed_cpu.to(device=logits.device)
    real_groups = groups_cpu.to(device=logits.device, dtype=torch.long)
    perm_groups = permuted_frequency_groups(
        groups_cpu,
        seed=permutation_seed,
    ).to(device=logits.device)

    def count_den(groups: torch.Tensor) -> torch.Tensor:
        flat = (
            margin_bins[:, allowed] * _N_FREQUENCY_GROUPS
            + groups[allowed].unsqueeze(0)
        ).reshape(-1)
        return torch.bincount(
            flat,
            minlength=grid.n_margin_bins * _N_FREQUENCY_GROUPS,
        ).reshape(grid.n_margin_bins, _N_FREQUENCY_GROUPS)

    rows = torch.arange(valid_targets.shape[0], device=logits.device)
    target_bins = margin_bins[rows, valid_targets]

    def count_num(groups: torch.Tensor) -> torch.Tensor:
        flat = target_bins * _N_FREQUENCY_GROUPS + groups[valid_targets]
        return torch.bincount(
            flat,
            minlength=grid.n_margin_bins * _N_FREQUENCY_GROUPS,
        ).reshape(grid.n_margin_bins, _N_FREQUENCY_GROUPS)

    return DocumentGridStats(
        doc_id=doc_id,
        half=document_half(doc_id, seed=permutation_seed),
        num=count_num(real_groups).to(device="cpu", dtype=torch.int64),
        den=count_den(real_groups).to(device="cpu", dtype=torch.int64),
        perm_num=count_num(perm_groups).to(device="cpu", dtype=torch.int64),
        perm_den=count_den(perm_groups).to(device="cpu", dtype=torch.int64),
        n_positions=int(valid_targets.numel()),
    )


def _checked_add(total: torch.Tensor, addition: torch.Tensor) -> None:
    if addition.shape != total.shape or addition.dtype != torch.int64:
        raise ValueError("statistics tensors must share shape and int64 dtype")
    if (addition < 0).any():
        raise ValueError("statistics counters must be non-negative")
    capacity = torch.iinfo(torch.int64).max - total
    if (addition > capacity).any():
        raise OverflowError("int64 statistics counter overflow")
    total.add_(addition)


def merge_document_stats(items: Sequence[DocumentGridStats]) -> dict[str, object]:
    if not items:
        raise ValueError("document statistics must be non-empty")
    first = items[0]
    shape = first.num.shape
    totals = {
        "num": torch.zeros(shape, dtype=torch.int64),
        "den": torch.zeros(shape, dtype=torch.int64),
        "perm_num": torch.zeros(shape, dtype=torch.int64),
        "perm_den": torch.zeros(shape, dtype=torch.int64),
        "half_num": torch.zeros((2, *shape), dtype=torch.int64),
        "half_den": torch.zeros((2, *shape), dtype=torch.int64),
        "half_perm_num": torch.zeros((2, *shape), dtype=torch.int64),
        "half_perm_den": torch.zeros((2, *shape), dtype=torch.int64),
    }
    seen = set()
    n_positions = 0
    for item in items:
        if not isinstance(item, DocumentGridStats):
            raise ValueError("items must contain DocumentGridStats")
        if item.doc_id in seen:
            raise ValueError(f"duplicate document statistics: {item.doc_id!r}")
        if item.half not in (0, 1):
            raise ValueError("document half must be zero or one")
        seen.add(item.doc_id)
        for key in ("num", "den", "perm_num", "perm_den"):
            value = getattr(item, key)
            _checked_add(totals[key], value)
            _checked_add(totals[f"half_{key}"][item.half], value)
        if item.n_positions < 0:
            raise ValueError("n_positions must be non-negative")
        n_positions += item.n_positions
    return {
        **totals,
        "n_positions": n_positions,
        "n_documents": len(items),
    }


def _fit_shift(
    centers: np.ndarray,
    group_log_rate: np.ndarray,
    group_weights: np.ndarray,
    group_valid: np.ndarray,
    ref_log_rate: np.ndarray,
    ref_valid: np.ndarray,
    *,
    low: float,
    high: float,
) -> float | None:
    group_mask = group_valid & (centers >= low) & (centers <= high)
    ref_mask = ref_valid & (centers >= low - 2.0) & (centers <= high + 2.0)
    if group_mask.sum() < 3 or ref_mask.sum() < 3:
        return None
    x = centers[group_mask]
    y = group_log_rate[group_mask]
    weights = group_weights[group_mask]
    ref_x = centers[ref_mask]
    ref_y = ref_log_rate[ref_mask]
    best: tuple[float, float] | None = None
    for shift in _SHIFT_GRID:
        aligned = np.interp(x - shift, ref_x, ref_y, left=np.nan, right=np.nan)
        overlap = np.isfinite(aligned)
        if overlap.sum() < 3:
            continue
        error = float(np.average((y[overlap] - aligned[overlap]) ** 2, weights=weights[overlap]))
        candidate = (error, float(shift))
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[1]


def analyze_grid(
    num: torch.Tensor,
    den: torch.Tensor,
    *,
    grid: GridSpec,
    reference_group: int | None = None,
) -> dict[str, object]:
    """Estimate horizontal reliability-curve shifts over margin [2, 12]."""
    expected = (grid.n_margin_bins, _N_FREQUENCY_GROUPS)
    if num.shape != expected or den.shape != expected:
        raise ValueError("num and den do not match the frozen grid")
    if num.dtype != torch.int64 or den.dtype != torch.int64:
        raise ValueError("num and den must use int64 counters")
    if (num < 0).any() or (den < 0).any() or (num > den).any():
        raise ValueError("num and den counters are inconsistent")
    num_np = num.cpu().numpy()
    den_np = den.cpu().numpy()
    valid = (num_np >= grid.min_true_count) & (den_np > 0)
    rates = np.full(num_np.shape, np.nan, dtype=np.float64)
    rates[valid] = num_np[valid] / den_np[valid]
    log_rates = np.log(rates)
    centers = np.asarray(grid.margin_bin_centers())
    window = (centers >= 2.0) & (centers <= 12.0)
    valid_counts = valid[window].sum(axis=0)
    eligible = np.flatnonzero(valid_counts >= 3)
    if eligible.size == 0:
        return {
            "informative": False,
            "reference_group": None,
            "shifts": {},
            "max_abs_shift": None,
            "rare_group": None,
            "rare_minus_reference_shift": None,
            "non_additive": False,
            "valid_cell_count": int(valid.sum()),
        }
    if reference_group is None:
        mass = den_np[window].sum(axis=0)
        reference_group = int(max(eligible, key=lambda group: (mass[group], group)))
    if reference_group not in eligible:
        raise ValueError("reference_group lacks three valid cells in margin [2, 12]")

    shifts: dict[str, float] = {str(reference_group): 0.0}
    subwindow_shifts: dict[int, tuple[float | None, float | None]] = {}
    for group in eligible:
        group = int(group)
        if group == reference_group:
            continue
        fitted = _fit_shift(
            centers,
            log_rates[:, group],
            den_np[:, group],
            valid[:, group],
            log_rates[:, reference_group],
            valid[:, reference_group],
            low=2.0,
            high=12.0,
        )
        if fitted is None:
            continue
        shifts[str(group)] = fitted
        subwindow_shifts[group] = (
            _fit_shift(
                centers,
                log_rates[:, group],
                den_np[:, group],
                valid[:, group],
                log_rates[:, reference_group],
                valid[:, reference_group],
                low=2.0,
                high=7.0,
            ),
            _fit_shift(
                centers,
                log_rates[:, group],
                den_np[:, group],
                valid[:, group],
                log_rates[:, reference_group],
                valid[:, reference_group],
                low=7.0,
                high=12.0,
            ),
        )
    non_reference = [int(group) for group in shifts if int(group) != reference_group]
    rare_group = min(non_reference) if non_reference else None
    non_additive = any(
        left is not None
        and right is not None
        and abs(left) >= 0.10
        and abs(right) >= 0.10
        and left * right < 0
        for left, right in subwindow_shifts.values()
    )
    return {
        "informative": bool(non_reference),
        "reference_group": reference_group,
        "shifts": shifts,
        "max_abs_shift": (
            max(abs(shifts[str(group)]) for group in non_reference)
            if non_reference
            else 0.0
        ),
        "rare_group": rare_group,
        "rare_minus_reference_shift": (
            shifts[str(rare_group)] if rare_group is not None else None
        ),
        "non_additive": non_additive,
        "valid_cell_count": int(valid.sum()),
    }
