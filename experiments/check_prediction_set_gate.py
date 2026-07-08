#!/usr/bin/env python3
"""Gate downstream GPU experiments on prediction-set evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--target-coverage", type=float, default=0.95)
    p.add_argument("--coverage-tolerance", type=float, default=0.02)
    p.add_argument("--size-match-ratio", type=float, default=1.25)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def load_metrics(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def find_conformal_method(methods: dict[str, Any]) -> str:
    for name in methods:
        if name.startswith("conformal_nu"):
            return name
    raise RuntimeError("No conformal_nu method found in metrics")


def low_frequency_coverage(info: dict[str, Any]) -> float | None:
    bucket = info.get("bucket_summary", {}).get("1-2")
    if bucket and bucket.get("coverage") is not None:
        return float(bucket["coverage"])
    bucket = info.get("bucket_summary", {}).get("0")
    if bucket and bucket.get("coverage") is not None:
        return float(bucket["coverage"])
    return None


def evaluate_gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    methods = summary["methods"]
    conformal_name = find_conformal_method(methods)
    conformal = methods[conformal_name]
    min_coverage = args.target_coverage - args.coverage_tolerance

    baseline_names = [
        name for name in methods
        if name != conformal_name and any(prefix in name for prefix in ["top_p", "min_p", "fixed_margin", "top_nsigma"])
    ]

    covered_baselines = [
        name for name in baseline_names
        if float(methods[name]["coverage"]) >= min_coverage
    ]
    criterion_fixed_coverage = (
        float(conformal["coverage"]) >= min_coverage
        and bool(covered_baselines)
        and all(float(conformal["avg_set_size"]) < float(methods[name]["avg_set_size"]) for name in covered_baselines)
    )

    conformal_low = low_frequency_coverage(conformal)
    matched = []
    for name in baseline_names:
        baseline = methods[name]
        baseline_size = float(baseline["avg_set_size"])
        if baseline_size <= 0:
            continue
        ratio = max(float(conformal["avg_set_size"]), baseline_size) / min(float(conformal["avg_set_size"]), baseline_size)
        if ratio <= args.size_match_ratio:
            baseline_low = low_frequency_coverage(baseline)
            if baseline_low is None or conformal_low is None:
                continue
            matched.append({
                "name": name,
                "size_ratio": ratio,
                "low_freq_coverage_delta": conformal_low - baseline_low,
                "coverage_delta": float(conformal["coverage"]) - float(baseline["coverage"]),
            })

    criterion_matched_support = any(
        item["low_freq_coverage_delta"] > 0
        and item["coverage_delta"] >= -args.coverage_tolerance
        for item in matched
    )

    passed = criterion_fixed_coverage or criterion_matched_support
    return {
        "passed": passed,
        "conformal_method": conformal_name,
        "target_coverage": args.target_coverage,
        "coverage_tolerance": args.coverage_tolerance,
        "criterion_fixed_coverage": criterion_fixed_coverage,
        "criterion_matched_support": criterion_matched_support,
        "covered_baselines": covered_baselines,
        "matched_support_comparisons": matched,
        "conformal": {
            "coverage": conformal["coverage"],
            "avg_set_size": conformal["avg_set_size"],
            "low_frequency_coverage": conformal_low,
        },
    }


def main() -> None:
    args = parse_args()
    summary = load_metrics(args.metrics)
    report = evaluate_gate(summary, args)

    output = Path(args.output) if args.output else Path(args.metrics).with_name("prediction_set_gate.json")
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[prediction-set-gate] wrote {output}")
    if report["passed"]:
        print("[prediction-set-gate] PASS")
        return
    print("[prediction-set-gate] FAIL: revise the score before downstream generation runs")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
