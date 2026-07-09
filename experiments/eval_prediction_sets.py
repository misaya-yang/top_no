#!/usr/bin/env python3
"""Evaluate token-level prediction sets for truncation decoding.

This is the main experiment for the V2 thesis: compare candidate-set coverage
and efficiency on held-out next-token prediction positions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from conformal import conformal_quantile, nu_nonconformity
from freq_table import (
    FrequencyTableMetadata,
    load_frequency_table,
    runtime_tokenizer_identity,
    special_token_ids,
)
from protocol import effective_config_sha256, validate_protocol_inputs
from samplers import get_keep_mask


BUCKETS = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-10", 3, 10),
    ("11-100", 11, 100),
    (">100", 101, None),
]

SET_SIZE_HIST_BINS = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000]


@dataclass
class MethodStats:
    covered: int = 0
    total: int = 0
    set_size_sum: float = 0.0
    retained_mass_sum: float = 0.0
    set_sizes: list[int] = field(default_factory=list)
    bucket_total: dict[str, int] = field(default_factory=lambda: {name: 0 for name, _, _ in BUCKETS})
    bucket_covered: dict[str, int] = field(default_factory=lambda: {name: 0 for name, _, _ in BUCKETS})
    bucket_set_size_sum: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name, _, _ in BUCKETS})

    def update(
        self,
        keep: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
        target_freqs: torch.Tensor,
    ) -> None:
        rows = torch.arange(targets.shape[0], device=targets.device)
        covered = keep[rows, targets]
        set_sizes = keep.sum(dim=-1)
        retained_mass = (probs * keep).sum(dim=-1)

        self.covered += int(covered.sum().item())
        self.total += int(targets.numel())
        self.set_size_sum += float(set_sizes.float().sum().item())
        self.retained_mass_sum += float(retained_mass.float().sum().item())
        self.set_sizes.extend(int(x) for x in set_sizes.detach().cpu().tolist())

        freqs_cpu = target_freqs.detach().cpu()
        covered_cpu = covered.detach().cpu()
        set_sizes_cpu = set_sizes.detach().cpu()
        for name, low, high in BUCKETS:
            if high is None:
                mask = freqs_cpu >= low
            else:
                mask = (freqs_cpu >= low) & (freqs_cpu <= high)
            count = int(mask.sum().item())
            if count == 0:
                continue
            self.bucket_total[name] += count
            self.bucket_covered[name] += int(covered_cpu[mask].sum().item())
            self.bucket_set_size_sum[name] += float(set_sizes_cpu[mask].float().sum().item())

    def summary(self, config: dict[str, Any]) -> dict[str, Any]:
        coverage = self.covered / self.total if self.total else 0.0
        avg_set_size = self.set_size_sum / self.total if self.total else 0.0
        avg_retained_mass = self.retained_mass_sum / self.total if self.total else 0.0
        median_set_size = float(np.median(self.set_sizes)) if self.set_sizes else 0.0
        set_size_quantiles = {}
        if self.set_sizes:
            set_size_quantiles = {
                str(q): float(np.quantile(self.set_sizes, q / 100.0))
                for q in [10, 25, 50, 75, 90, 95, 99]
            }
        hist_counts = [0 for _ in range(len(SET_SIZE_HIST_BINS) + 1)]
        for size in self.set_sizes:
            idx = 0
            while idx < len(SET_SIZE_HIST_BINS) and size > SET_SIZE_HIST_BINS[idx]:
                idx += 1
            hist_counts[idx] += 1
        bucket_summary = {}
        for name, _, _ in BUCKETS:
            total = self.bucket_total[name]
            bucket_summary[name] = {
                "n": total,
                "coverage": self.bucket_covered[name] / total if total else None,
                "avg_set_size": self.bucket_set_size_sum[name] / total if total else None,
            }
        return {
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "median_set_size": median_set_size,
            "set_size_quantiles": set_size_quantiles,
            "set_size_histogram": {
                "upper_bounds": SET_SIZE_HIST_BINS,
                "counts": hist_counts,
            },
            "avg_retained_mass": avg_retained_mass,
            "bucket_summary": bucket_summary,
            "config": config,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--model-revision", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None,
                   choices=["wikitext", "c4", "local", "text_file"])
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--n-calibration", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--n-texts", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--position-batch-size", type=int, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None,
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--frequency-table", type=str, default=None)
    p.add_argument("--frequency-manifest", type=str, default=None)
    p.add_argument("--tune-manifest", type=str, default=None)
    p.add_argument("--calibration-manifest", type=str, default=None)
    p.add_argument("--test-manifest", type=str, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--allow-legacy-protocol",
        action="store_true",
        help=(
            "Run the pre-PR1 protocol for smoke/link tests only. The current "
            "protocol builds frequency counts from the loaded text pool and "
            "uses a sequential calibration/eval split, so its outputs are not "
            "paper evidence."
        ),
    )
    return p.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    config = {
        "model": "gpt2",
        "model_revision": None,
        "dataset": "wikitext",
        "split": "validation",
        "n_calibration": 64,
        "n_eval": 64,
        "n_texts": None,
        "max_length": 64,
        "batch_size": 2,
        "position_batch_size": 128,
        "kappa": 10.0,
        "alpha": 1.0,
        "delta": 0.05,
        "seed": 42,
        "dtype": "auto",
        "device": "auto",
        "output_dir": "./results/smoke_prediction_sets",
        "trust_remote_code": True,
        "allow_legacy_protocol": False,
        "frequency_table": None,
        "frequency_manifest": None,
        "tune_manifest": None,
        "calibration_manifest": None,
        "test_manifest": None,
    }
    if path:
        with open(path) as f:
            loaded = json.load(f)
        config.update(loaded)
    return config


def merge_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None and value is not False:
            config[key] = value
    if args.trust_remote_code:
        config["trust_remote_code"] = True
    if args.allow_legacy_protocol:
        config["allow_legacy_protocol"] = True
    return config


def assert_protocol_is_allowed(config: dict[str, Any]) -> dict[str, Any]:
    """Validate protocol inputs before allocating a dataset or model."""
    return validate_protocol_inputs(config)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(config: dict[str, Any], device: torch.device):
    dtype = resolve_dtype(config["dtype"], device)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config.get("model_revision"),
        trust_remote_code=config["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=config.get("model_revision"),
        dtype=dtype,
        trust_remote_code=config["trust_remote_code"],
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def validate_runtime_model_and_tokenizer(
    model,
    tokenizer,
    config: dict[str, Any],
) -> tuple[str, str | None]:
    """Bind runtime objects to the pinned model/tokenizer identity."""
    expected_revision = config.get("model_revision")
    model_revision = getattr(model.config, "_commit_hash", None)
    if expected_revision is not None and model_revision != expected_revision:
        raise ValueError(
            f"model_revision mismatch: runtime={model_revision!r} "
            f"expected={expected_revision!r}"
        )
    tokenizer_id, tokenizer_revision = runtime_tokenizer_identity(
        tokenizer,
        resolved_model_revision=model_revision,
    )
    if expected_revision is not None and tokenizer_revision != expected_revision:
        raise ValueError(
            f"tokenizer_revision mismatch: runtime={tokenizer_revision!r} "
            f"expected={expected_revision!r}"
        )
    vocab_size = int(model.config.vocab_size)
    if vocab_size <= 0:
        raise ValueError("model vocab_size must be positive")
    if hasattr(tokenizer, "__len__") and len(tokenizer) > vocab_size:
        raise ValueError(
            f"tokenizer size {len(tokenizer)} exceeds model vocab_size {vocab_size}"
        )
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is not None and input_embeddings.num_embeddings != vocab_size:
        raise ValueError(
            "input embedding size does not match model vocab_size: "
            f"embeddings={input_embeddings.num_embeddings} config={vocab_size}"
        )
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    output_size = getattr(output_embeddings, "out_features", vocab_size)
    if output_embeddings is not None and output_size != vocab_size:
        raise ValueError(
            "output embedding size does not match model vocab_size: "
            f"output={output_size} config={vocab_size}"
        )
    return tokenizer_id, tokenizer_revision


def estimate_n_texts(config: dict[str, Any]) -> int:
    if config.get("n_texts"):
        return int(config["n_texts"])
    target_positions = int(config["n_calibration"]) + int(config["n_eval"])
    usable_per_text = max(int(config["max_length"]) // 2, 8)
    return max(math.ceil(target_positions / usable_per_text) * 2, int(config["batch_size"]) * 4)


def load_texts(config: dict[str, Any]) -> list[str]:
    rng = np.random.RandomState(int(config["seed"]))
    n_texts = estimate_n_texts(config)
    dataset = config["dataset"]
    max_chars = int(config["max_length"]) * 8

    if dataset == "text_file":
        if not config.get("text_file"):
            raise ValueError("--text-file is required when dataset=text_file")
        lines = [
            line.strip()
            for line in Path(config["text_file"]).read_text().splitlines()
            if len(line.strip()) > 20
        ]
        if len(lines) < n_texts:
            raise RuntimeError(f"text_file has only {len(lines)} usable lines, need {n_texts}")
        rng.shuffle(lines)
        return [line[:max_chars] for line in lines[:n_texts]]

    if dataset == "local":
        from data_utils import load_text_samples
        return load_text_samples(n_texts, max_length=max_chars, seed=int(config["seed"]))

    if dataset == "wikitext":
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split=config["split"],
        )
        texts = [item["text"].strip() for item in ds if len(item["text"].strip()) > 20]
        if len(texts) < n_texts:
            raise RuntimeError(f"wikitext has only {len(texts)} usable rows, need {n_texts}")
        rng.shuffle(texts)
        return [text[:max_chars] for text in texts[:n_texts]]

    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split=config["split"], streaming=True)
        ds = ds.shuffle(seed=int(config["seed"]), buffer_size=10_000)
        texts = []
        for item in ds:
            text = item["text"].strip()
            if len(text) > 100:
                texts.append(text[:max_chars])
            if len(texts) >= n_texts:
                return texts
        raise RuntimeError(f"c4 stream ended with only {len(texts)} usable rows, need {n_texts}")

    raise ValueError(f"Unknown dataset: {dataset}")


def build_token_counts(tokenizer, texts: list[str], vocab_size: int, max_length: int, batch_size: int) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        mask = enc["attention_mask"].bool()
        valid_ids = ids[mask]
        counts += torch.bincount(valid_ids, minlength=vocab_size).float()
    return counts


def resolve_token_counts(
    tokenizer,
    model,
    config: dict[str, Any],
    texts: list[str],
) -> tuple[torch.Tensor, FrequencyTableMetadata | None]:
    """Load frozen counts, or use the explicit legacy smoke fallback."""
    if config.get("frequency_table"):
        tokenizer_id, tokenizer_revision = validate_runtime_model_and_tokenizer(
            model,
            tokenizer,
            config,
        )
        return load_frequency_table(
            Path(config["frequency_table"]),
            expected_model_id=config["model"],
            expected_tokenizer_id=tokenizer_id,
            expected_tokenizer_revision=tokenizer_revision,
            expected_vocab_size=model.config.vocab_size,
            expected_exclusion_token_ids=special_token_ids(tokenizer),
        )
    if not config.get("allow_legacy_protocol"):
        raise RuntimeError(
            "frequency_table is required outside explicit legacy smoke runs"
        )
    counts = build_token_counts(
        tokenizer,
        texts,
        vocab_size=model.config.vocab_size,
        max_length=int(config["max_length"]),
        batch_size=int(config["batch_size"]),
    )
    return counts, None


def default_methods(config: dict[str, Any], q_hat: float) -> list[dict[str, Any]]:
    kappa = float(config["kappa"])
    return [
        {"name": "top_k_50", "strategy": "top_k", "kwargs": {"k": 50}},
        {"name": "top_p_0.95", "strategy": "top_p", "kwargs": {"p": 0.95}},
        {"name": "min_p_0.05", "strategy": "min_p", "kwargs": {"p_min": 0.05}},
        {"name": "fixed_margin_3", "strategy": "fixed_margin", "kwargs": {"margin": 3.0}},
        {"name": "top_nsigma_2", "strategy": "top_nsigma", "kwargs": {"n_sigma": 2.0}},
        {"name": "nu_k10_m3", "strategy": "nu", "kwargs": {"kappa": kappa, "m0": 3.0}},
        {
            "name": f"conformal_nu_k{kappa:g}_delta{float(config['delta']):g}",
            "strategy": "conformal_nu",
            "kwargs": {"kappa": kappa, "q_hat": q_hat, "alpha": float(config["alpha"])},
        },
    ]


def batch_position_logits(model, tokenizer, texts: list[str], config: dict[str, Any], device: torch.device):
    max_length = int(config["max_length"])
    batch_size = int(config["batch_size"])
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        valid = (attention_mask[:, :-1] > 0) & (attention_mask[:, 1:] > 0)
        if not valid.any():
            continue
        yield logits[valid], targets[valid]


def keep_mask_for_method(logits: torch.Tensor, method: dict[str, Any], token_counts: torch.Tensor) -> torch.Tensor:
    strategy = method["strategy"]
    kwargs = dict(method["kwargs"])
    if strategy in {"nu", "conformal_nu", "nu_topp_floor", "nu_entropy", "nu_mathboost"}:
        kwargs["token_freq_table"] = token_counts
    return get_keep_mask(logits, strategy, **kwargs)


def collect_calibration_scores(
    model,
    tokenizer,
    texts: list[str],
    token_counts: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    needed = int(config["n_calibration"])
    chunks = []
    seen = 0
    token_counts_device = token_counts.to(device)
    for logits, targets in batch_position_logits(model, tokenizer, texts, config, device):
        take = min(needed - seen, targets.numel())
        if take <= 0:
            break
        scores = nu_nonconformity(
            logits[:take],
            targets[:take],
            token_counts_device,
            kappa=float(config["kappa"]),
            alpha=float(config["alpha"]),
        )
        chunks.append(scores.detach().cpu())
        seen += take
        if seen >= needed:
            break
    if seen < needed:
        raise RuntimeError(f"Only collected {seen} calibration positions, need {needed}")
    return torch.cat(chunks), seen


def evaluate_methods(
    model,
    tokenizer,
    texts: list[str],
    token_counts: torch.Tensor,
    methods: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, MethodStats], int]:
    stats = {method["name"]: MethodStats() for method in methods}
    needed = int(config["n_eval"])
    skip = int(config["n_calibration"])
    consumed = 0
    evaluated = 0
    pos_chunk_size = int(config["position_batch_size"])
    token_counts_device = token_counts.to(device)

    for logits, targets in batch_position_logits(model, tokenizer, texts, config, device):
        n = targets.numel()
        if consumed + n <= skip:
            consumed += n
            continue
        start = max(skip - consumed, 0)
        logits = logits[start:]
        targets = targets[start:]
        consumed += n

        if logits.shape[0] > needed - evaluated:
            logits = logits[:needed - evaluated]
            targets = targets[:needed - evaluated]

        for pos_start in range(0, targets.numel(), pos_chunk_size):
            pos_end = min(pos_start + pos_chunk_size, targets.numel())
            logits_chunk = logits[pos_start:pos_end]
            targets_chunk = targets[pos_start:pos_end]
            probs = F.softmax(logits_chunk.float(), dim=-1)
            target_freqs = token_counts_device[targets_chunk]

            for method in methods:
                keep = keep_mask_for_method(logits_chunk, method, token_counts_device)
                stats[method["name"]].update(keep, probs, targets_chunk, target_freqs)

        evaluated += int(targets.numel())
        if evaluated >= needed:
            break

    if evaluated < needed:
        raise RuntimeError(f"Only evaluated {evaluated} positions, need {needed}")
    return stats, evaluated


def plot_pareto(summary: dict[str, Any], output_dir: Path) -> None:
    methods = summary["methods"]
    names = list(methods)
    x = [methods[name]["avg_set_size"] for name in names]
    y = [methods[name]["coverage"] for name in names]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=70)
    for name, x_val, y_val in zip(names, x, y):
        plt.annotate(name, (x_val, y_val), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Average support size")
    plt.ylabel("True-token coverage")
    plt.title("Prediction-set coverage vs efficiency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_size_pareto.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = merge_args(load_config(args.config), args)
    protocol = assert_protocol_is_allowed(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config["device"])
    t0 = time.time()

    print(f"[prediction-sets] model={config['model']}")
    print(f"[prediction-sets] dataset={config['dataset']} split={config['split']}")
    print(f"[prediction-sets] device={device} dtype={config['dtype']}")

    texts = load_texts(config)
    print(f"[prediction-sets] loaded_texts={len(texts)}")
    model, tokenizer = load_model_and_tokenizer(config, device)
    token_counts, _ = resolve_token_counts(tokenizer, model, config, texts)
    print(f"[prediction-sets] vocab={model.config.vocab_size} nonzero_counts={(token_counts > 0).sum().item()}")

    cal_scores, n_cal = collect_calibration_scores(model, tokenizer, texts, token_counts, config, device)
    q_hat = conformal_quantile(cal_scores, float(config["delta"]))
    print(f"[prediction-sets] calibrated q_hat={q_hat:.6f} from n={n_cal}")

    methods = default_methods(config, q_hat)
    stats, n_eval = evaluate_methods(model, tokenizer, texts, token_counts, methods, config, device)

    summary = {
        "model": config["model"],
        "model_revision": config.get("model_revision"),
        "dataset": config["dataset"],
        "split": config["split"],
        "n_positions": n_eval,
        "n_calibration": n_cal,
        "n_eval": n_eval,
        "q_hat": q_hat,
        "kappa": float(config["kappa"]),
        "alpha": float(config["alpha"]),
        "delta": float(config["delta"]),
        "elapsed_sec": time.time() - t0,
        "config_sha256": effective_config_sha256(config),
        "protocol": protocol,
        "methods": {
            method["name"]: stats[method["name"]].summary(method["kwargs"])
            for method in methods
        },
    }
    with open(output_dir / "prediction_set_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_pareto(summary, output_dir)

    print("[prediction-sets] results:")
    for name, info in summary["methods"].items():
        print(
            f"  {name:28s} coverage={info['coverage']:.4f} "
            f"avg_size={info['avg_set_size']:.1f} mass={info['avg_retained_mass']:.4f}"
        )
    print(f"[prediction-sets] wrote {output_dir / 'prediction_set_metrics.json'}")
    print(f"[prediction-sets] wrote {output_dir / 'coverage_size_pareto.png'}")


if __name__ == "__main__":
    main()
