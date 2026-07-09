#!/usr/bin/env python3
"""Evaluate open-ended generation quality beyond Distinct-n."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from eval_prediction_sets import load_model_and_tokenizer, resolve_device
from eval_reasoning_self_consistency import conformal_kwargs_from_metrics
from freq_table import load_frequency_table_from_metrics, special_token_ids
from samplers import batch_generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--samples-per-prompt", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--prediction-set-metrics", type=str, default=None)
    p.add_argument("--compute-perplexity", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None,
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    config = {
        "model": "Qwen/Qwen2.5-3B",
        "datasets": ["writingprompts", "alpacaeval_creative"],
        "n_prompts": 300,
        "samples_per_prompt": 4,
        "batch_size": 8,
        "max_new_tokens": 256,
        "temperature": 0.9,
        "seed": 42,
        "dtype": "float16",
        "device": "cuda",
        "output_dir": "./results/openended_quality_qwen3b",
        "trust_remote_code": True,
        "prediction_set_metrics": "./results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json",
        "count_dataset": "wikitext",
        "count_split": "validation",
        "count_n_texts": 2000,
        "count_max_length": 256,
        "compute_perplexity": True,
        "perplexity_max_length": 256,
    }
    if path:
        with open(path) as f:
            config.update(json.load(f))
    return config


def merge_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None and value is not False:
            config[key] = value
    if args.compute_perplexity:
        config["compute_perplexity"] = True
    if args.trust_remote_code:
        config["trust_remote_code"] = True
    return config


def sample_prompts(prompts: list[str], n: int, seed: int) -> list[str]:
    if len(prompts) < n:
        raise RuntimeError(f"Dataset has only {len(prompts)} usable prompts, need {n}")
    rng = np.random.RandomState(seed)
    idx = np.arange(len(prompts))
    rng.shuffle(idx)
    return [prompts[int(i)] for i in idx[:n]]


def load_writingprompts(n: int, seed: int) -> list[str]:
    ds = load_dataset("euclaise/writingprompts", split="train")
    prompts = []
    for item in ds:
        prompt = item.get("prompt") or item.get("writingprompt")
        if prompt and len(prompt.strip()) > 20:
            prompts.append(prompt.strip())
        if len(prompts) >= n * 5:
            break
    return sample_prompts(prompts, n, seed)


def load_alpacaeval_creative(n: int, seed: int) -> list[str]:
    last_error: Exception | None = None
    for split in ["eval", "test", "validation", "train"]:
        try:
            ds = load_dataset("tatsu-lab/alpaca_eval", split=split, trust_remote_code=True)
            prompts = []
            creative_terms = ["write", "story", "poem", "creative", "imagine", "describe", "compose"]
            for item in ds:
                prompt = item.get("instruction") or item.get("prompt") or item.get("input")
                if not prompt:
                    continue
                if any(term in prompt.lower() for term in creative_terms):
                    prompts.append(prompt.strip())
                elif len(prompts) < n:
                    prompts.append(prompt.strip())
                if len(prompts) >= n * 5:
                    break
            return sample_prompts(prompts, n, seed)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load tatsu-lab/alpaca_eval from known splits: {last_error}")


def load_openended_prompts(name: str, n: int, seed: int) -> list[str]:
    if name == "writingprompts":
        return load_writingprompts(n, seed)
    if name == "alpacaeval_creative":
        return load_alpacaeval_creative(n, seed)
    raise ValueError(f"Unknown open-ended dataset: {name}")


def build_counts_for_strategies(model, tokenizer, config: dict[str, Any]) -> torch.Tensor:
    metrics_path = config.get("prediction_set_metrics")
    if not metrics_path:
        raise RuntimeError(
            "prediction_set_metrics with a frequency-table artifact is required; "
            "downstream runs may not rebuild counts"
        )
    counts, _ = load_frequency_table_from_metrics(
        Path(metrics_path),
        expected_model_id=config["model"],
        expected_tokenizer_id=config.get("tokenizer_id") or config["model"],
        expected_vocab_size=model.config.vocab_size,
        expected_exclusion_token_ids=special_token_ids(tokenizer),
    )
    return counts


def method_specs(config: dict[str, Any], token_counts: torch.Tensor) -> dict[str, dict[str, Any]]:
    methods = {
        "top_p_0.95": {"strategy": "top_p", "kwargs": {"p": 0.95}},
        "min_p_0.05": {"strategy": "min_p", "kwargs": {"p_min": 0.05}},
        "top_nsigma_2": {"strategy": "top_nsigma", "kwargs": {"n_sigma": 2.0}},
        "nu_k10_m3": {
            "strategy": "nu",
            "kwargs": {"token_freq_table": token_counts, "kappa": 10.0, "m0": 3.0},
        },
    }
    conformal = conformal_kwargs_from_metrics(config.get("prediction_set_metrics"))
    if conformal:
        methods["conformal_nu"] = {
            "strategy": "conformal_nu",
            "kwargs": {"token_freq_table": token_counts, **conformal},
        }
    return methods


def distinct_n(token_rows: list[list[int]], n: int) -> float:
    total = 0
    unique = set()
    for tokens in token_rows:
        if len(tokens) < n:
            continue
        for idx in range(len(tokens) - n + 1):
            total += 1
            unique.add(tuple(tokens[idx:idx + n]))
    return len(unique) / total if total else 0.0


def repetition_rate(token_rows: list[list[int]]) -> float:
    repeats = 0
    total = 0
    for tokens in token_rows:
        for idx in range(1, len(tokens)):
            total += 1
            repeats += int(tokens[idx] == tokens[idx - 1])
    return repeats / total if total else 0.0


def unique_token_ratio(token_rows: list[list[int]]) -> float:
    ratios = [len(set(tokens)) / len(tokens) for tokens in token_rows if tokens]
    return float(np.mean(ratios)) if ratios else 0.0


def ngram_counts(tokens: list[int], n: int) -> Counter[tuple[int, ...]]:
    return Counter(tuple(tokens[idx:idx + n]) for idx in range(max(len(tokens) - n + 1, 0)))


def sentence_bleu(candidate: list[int], references: list[list[int]], max_n: int = 4) -> float:
    if not candidate or not references:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_counts = ngram_counts(candidate, n)
        if not cand_counts:
            precisions.append(1e-8)
            continue
        max_ref: Counter[tuple[int, ...]] = Counter()
        for ref in references:
            ref_counts = ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                max_ref[gram] = max(max_ref[gram], count)
        overlap = sum(min(count, max_ref[gram]) for gram, count in cand_counts.items())
        precisions.append((overlap + 1.0) / (sum(cand_counts.values()) + 1.0))
    ref_len = min((len(ref) for ref in references), key=lambda size: abs(size - len(candidate)))
    bp = 1.0 if len(candidate) > ref_len else math.exp(1.0 - ref_len / max(len(candidate), 1))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def self_bleu(grouped_tokens: list[list[list[int]]]) -> float:
    scores = []
    for samples in grouped_tokens:
        for idx, candidate in enumerate(samples):
            refs = samples[:idx] + samples[idx + 1:]
            if refs:
                scores.append(sentence_bleu(candidate, refs))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluator_perplexity(model, tokenizer, texts: list[str], config: dict[str, Any]) -> float:
    device = next(model.parameters()).device
    losses = []
    batch_size = int(config["batch_size"])
    max_length = int(config["perplexity_max_length"])
    for start in range(0, len(texts), batch_size):
        batch = [text if text.strip() else tokenizer.eos_token for text in texts[start:start + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        outputs = model(**enc, labels=labels)
        losses.append(float(outputs.loss.detach().cpu().item()))
    return float(math.exp(np.mean(losses))) if losses else 0.0


def summarize_generation(
    model,
    tokenizer,
    grouped: list[list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    token_rows = [item["tokens"] for samples in grouped for item in samples]
    text_rows = [item["text"] for samples in grouped for item in samples]
    grouped_tokens = [[item["tokens"] for item in samples] for samples in grouped]
    summary = {
        "n_generations": len(token_rows),
        "avg_length": float(np.mean([len(tokens) for tokens in token_rows])) if token_rows else 0.0,
        "distinct_1": distinct_n(token_rows, 1),
        "distinct_2": distinct_n(token_rows, 2),
        "distinct_3": distinct_n(token_rows, 3),
        "self_bleu": self_bleu(grouped_tokens),
        "repetition_rate": repetition_rate(token_rows),
        "unique_token_ratio": unique_token_ratio(token_rows),
    }
    if config.get("compute_perplexity"):
        summary["external_lm_perplexity"] = evaluator_perplexity(model, tokenizer, text_rows, config)
        summary["perplexity_model"] = config["model"]
    return summary


def generate_grouped(
    model,
    tokenizer,
    prompts: list[str],
    method: dict[str, Any],
    config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    grouped = [[] for _ in prompts]
    for sample_idx in range(int(config["samples_per_prompt"])):
        torch.manual_seed(int(config["seed"]) + sample_idx)
        generated = batch_generate(
            model,
            tokenizer,
            prompts,
            max_new_tokens=int(config["max_new_tokens"]),
            batch_size=int(config["batch_size"]),
            strategy=method["strategy"],
            strategy_kwargs=method["kwargs"],
            temperature=float(config["temperature"]),
            max_prompt_length=256,
        )
        for idx, item in enumerate(generated):
            grouped[idx].append(item)
    return grouped


def main() -> None:
    args = parse_args()
    config = merge_args(load_config(args.config), args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    device = resolve_device(config["device"])
    print(f"[openended] model={config['model']} device={device}")
    model, tokenizer = load_model_and_tokenizer(config, device)
    token_counts = build_counts_for_strategies(model, tokenizer, config)
    methods = method_specs(config, token_counts)
    t0 = time.time()

    results: dict[str, Any] = {
        "model": config["model"],
        "datasets": {},
        "methods": list(methods),
        "samples_per_prompt": int(config["samples_per_prompt"]),
    }

    for dataset_name in config["datasets"]:
        print(f"[openended] dataset={dataset_name}")
        prompts = load_openended_prompts(dataset_name, int(config["n_prompts"]), int(config["seed"]))
        dataset_result = {}
        for method_name, method in methods.items():
            print(f"[openended] method={method_name}")
            grouped = generate_grouped(model, tokenizer, prompts, method, config)
            summary = summarize_generation(model, tokenizer, grouped, config)
            dataset_result[method_name] = summary
            print(
                f"  distinct_2={summary['distinct_2']:.4f} "
                f"self_bleu={summary['self_bleu']:.4f} "
                f"rep={summary['repetition_rate']:.4f}"
            )
        results["datasets"][dataset_name] = dataset_result

    results["elapsed_sec"] = time.time() - t0
    output_path = output_dir / "openended_quality_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[openended] wrote {output_path}")


if __name__ == "__main__":
    main()
