#!/usr/bin/env python3
"""Summarize the four-cell Phase-0 pilot without overstating evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence


SUMMARY_SCHEMA_VERSION = "icml2027-phase0-summary-v1"
MEMO_SCHEMA_VERSION = "icml2027-phase0-decision-memo-v1"
EXPECTED_PAIRS = {
    ("qwen3b", "web"),
    ("qwen3b", "math"),
    ("qwen7b", "web"),
    ("qwen7b", "math"),
}


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sign(value: float, *, floor: float = 1e-9) -> int:
    if value > floor:
        return 1
    if value < -floor:
        return -1
    return 0


def _cell_record(cell: object) -> dict[str, object]:
    if not isinstance(cell, dict):
        raise ValueError("Phase-0 cell summary must be a JSON object")
    if cell.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported Phase-0 cell summary")
    if cell.get("evidence_grade") != "E-pilot" or cell.get("paper_citable") is not False:
        raise ValueError("Phase-0 cell has an invalid evidence grade")
    if cell.get("completion_status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("Phase-0 cell completion_status is invalid")
    analysis = cell.get("analysis")
    permutation = cell.get("permutation_analysis")
    halves = cell.get("half_analysis")
    if not isinstance(analysis, dict) or not isinstance(permutation, dict):
        raise ValueError("Phase-0 cell analysis is malformed")
    if not isinstance(halves, list) or len(halves) != 2 or not all(
        isinstance(item, dict) for item in halves
    ):
        raise ValueError("Phase-0 half analysis is malformed")
    informative = bool(analysis.get("informative")) and all(
        bool(item.get("informative")) for item in halves
    )
    if informative:
        effect = _finite_number(analysis.get("max_abs_shift"), "max_abs_shift")
        signed = _finite_number(
            analysis.get("rare_minus_reference_shift"),
            "rare_minus_reference_shift",
        )
        half_signed = tuple(
            _finite_number(item.get("rare_minus_reference_shift"), "half shift")
            for item in halves
        )
        perm_effect = _finite_number(
            permutation.get("max_abs_shift"),
            "permutation max_abs_shift",
        )
    else:
        effect = 0.0
        signed = 0.0
        half_signed = (0.0, 0.0)
        perm_effect = 0.0
    half_stable = (
        informative
        and _sign(signed) != 0
        and all(_sign(value) == _sign(signed) for value in half_signed)
    )
    for field in ("cell_key", "model_key", "domain_key"):
        if not isinstance(cell.get(field), str) or not cell[field]:
            raise ValueError(f"Phase-0 cell {field} is invalid")
    return {
        "cell_key": cell["cell_key"],
        "model_key": cell["model_key"],
        "domain_key": cell["domain_key"],
        "completion_status": cell["completion_status"],
        "informative": informative,
        "effect": effect,
        "signed_effect": signed,
        "direction": _sign(signed),
        "half_stable": half_stable,
        "permutation_effect": perm_effect,
        "permutation_separated": perm_effect < 0.5 * effect,
        "non_additive": bool(analysis.get("non_additive", False)),
        "n_documents": cell.get("n_documents"),
        "n_positions": cell.get("n_positions"),
    }


def summarize_cells(cells: Sequence[object]) -> dict[str, object]:
    records = [_cell_record(cell) for cell in cells]
    keys = [record["cell_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Phase-0 summaries contain duplicate cell keys")
    informative = [record for record in records if record["informative"]]
    pairs = {(item["model_key"], item["domain_key"]) for item in informative}
    directions = {item["direction"] for item in informative if item["direction"] != 0}
    domains = {item["domain_key"] for item in informative}
    models = {item["model_key"] for item in informative}
    cross_domain = len(domains) >= 2 and len(directions) == 1
    cross_scale = len(models) >= 2 and len(directions) == 1
    permutation_passes = sum(bool(item["permutation_separated"]) for item in informative)
    plan_a = (
        pairs == EXPECTED_PAIRS
        and all(float(item["effect"]) >= 0.30 for item in informative)
        and all(bool(item["half_stable"]) for item in informative)
        and cross_domain
        and cross_scale
        and permutation_passes >= 3
    )
    plan_b = (
        not plan_a
        and len(domains) >= 2
        and len(informative) >= 2
        and all(float(item["effect"]) < 0.30 for item in informative)
    )
    verdict = "PLAN_A_PILOT" if plan_a else "PLAN_B_PILOT" if plan_b else "INSUFFICIENT"
    return {
        "schema_version": MEMO_SCHEMA_VERSION,
        "evidence_grade": "E-pilot",
        "paper_citable": False,
        "verdict": verdict,
        "cell_count": len(records),
        "informative_cell_count": len(informative),
        "cross_domain_sign_agreement": cross_domain,
        "cross_scale_sign_agreement": cross_scale,
        "permutation_separation_count": permutation_passes,
        "non_additive_cells": [
            item["cell_key"] for item in informative if item["non_additive"]
        ],
        "cells": records,
        "next_action": (
            "run_full_preregistered_phase0"
            if verdict == "PLAN_A_PILOT"
            else "prepare_broad_margin_sufficiency_audit"
            if verdict == "PLAN_B_PILOT"
            else "collect_more_or_repair_pilot_evidence"
        ),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    summaries = []
    for path in sorted(root.glob("*/phase0_summary.json")):
        summaries.append(json.loads(path.read_text()))
    memo = summarize_cells(summaries)
    output = Path(args.output) if args.output else root / "decision_memo.json"
    _atomic_json(output, memo)
    print(json.dumps(memo, sort_keys=True))


if __name__ == "__main__":
    main()
