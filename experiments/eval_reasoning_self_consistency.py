#!/usr/bin/env python3
"""Evaluate decoding strategies with reasoning self-consistency metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from eval_prediction_sets import (
    load_model_and_tokenizer,
    resolve_device,
    validate_runtime_model_and_tokenizer,
)
from freq_table import load_frequency_table_from_metrics, special_token_ids
from samplers import batch_generate


ANSWER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--model-revision", type=str, default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--samples-per-question", type=int, default=None)
    p.add_argument("--n-questions", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--k-values", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--prediction-set-metrics", type=str, default=None)
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
        "model_revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "datasets": ["gsm8k", "math500", "svamp"],
        "samples_per_question": 16,
        "n_questions": 500,
        "batch_size": 8,
        "max_new_tokens": 256,
        "temperature": 0.8,
        "k_values": [1, 4, 8, 16],
        "seed": 42,
        "dtype": "float16",
        "device": "cuda",
        "output_dir": "./results/reasoning_self_consistency_qwen3b",
        "trust_remote_code": True,
        "prediction_set_metrics": "./results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json",
        "count_dataset": "wikitext",
        "count_split": "validation",
        "count_n_texts": 2000,
        "count_max_length": 256,
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
    if args.trust_remote_code:
        config["trust_remote_code"] = True
    return config


def extract_number(text: str) -> float | None:
    clean = text.replace(",", "")
    frac_matches = re.findall(r"\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}", clean)
    frac_matches += re.findall(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", clean)
    if frac_matches:
        num, den = frac_matches[-1]
        try:
            den_float = float(den)
            if den_float != 0:
                return float(num) / den_float
        except ValueError:
            pass
    matches = ANSWER_RE.findall(clean)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def normalize_text_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\\boxed", "")
    text = re.sub(r"[{}$]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def extract_answer(text: str) -> str | None:
    if "####" in text:
        tail = text.split("####")[-1]
        boxed = re.search(r"\\boxed\{([^{}]+)\}", tail)
        if boxed:
            return normalize_text_answer(boxed.group(1))
        number = extract_number(tail)
        return str(number) if number is not None else normalize_text_answer(tail)

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        return normalize_text_answer(boxed_matches[-1])

    final_match = re.search(r"(?:final answer|answer is|therefore)[^0-9+\-]*(.+)$", text, flags=re.I | re.S)
    if final_match:
        number = extract_number(final_match.group(1))
        if number is not None:
            return str(number)

    number = extract_number(text)
    return str(number) if number is not None else None


def answers_match(predicted: str | None, reference: str) -> bool:
    if predicted is None:
        return False
    pred_num = extract_number(predicted)
    ref_num = extract_number(reference)
    if pred_num is not None and ref_num is not None:
        return math.isclose(pred_num, ref_num, rel_tol=1e-6, abs_tol=1e-6)
    return normalize_text_answer(predicted) == normalize_text_answer(reference)


def load_gsm8k(n: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for item in ds:
        answer = item["answer"].split("####")[-1].strip()
        rows.append({"dataset": "gsm8k", "question": item["question"], "answer": answer})
    return sample_rows(rows, n, seed)


def load_math500(n: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for item in ds:
        question = item.get("problem") or item.get("question")
        answer = item.get("answer")
        if question and answer is not None:
            rows.append({"dataset": "math500", "question": question, "answer": str(answer)})
    return sample_rows(rows, n, seed)


def load_svamp(n: int, seed: int) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for dataset_name in ["ChilleD/SVAMP", "ChilleD/svamp", "Dahoas/svamp", "garrethlee/svamp"]:
        try:
            ds = load_dataset(dataset_name, split="test")
            rows = []
            for item in ds:
                body = item.get("Body") or item.get("body") or item.get("context") or ""
                question = item.get("Question") or item.get("question") or item.get("problem")
                answer = item.get("Answer") or item.get("answer") or item.get("result")
                if question and answer is not None:
                    full_question = f"{body.strip()} {question.strip()}".strip()
                    rows.append({"dataset": "svamp", "question": full_question, "answer": str(answer)})
            return sample_rows(rows, n, seed)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load SVAMP from known Hugging Face mirrors: {last_error}")


def sample_rows(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    if len(rows) < n:
        raise RuntimeError(f"Dataset has only {len(rows)} usable rows, need {n}")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    return [rows[int(i)] for i in indices[:n]]


def load_reasoning_dataset(name: str, n: int, seed: int) -> list[dict[str, str]]:
    if name == "gsm8k":
        return load_gsm8k(n, seed)
    if name == "math500":
        return load_math500(n, seed)
    if name == "svamp":
        return load_svamp(n, seed)
    raise ValueError(f"Unknown reasoning dataset: {name}")


def make_prompts(rows: list[dict[str, str]]) -> list[str]:
    return [
        "Solve the problem step by step. Put the final answer after ####.\n\n"
        f"Problem: {row['question']}\nSolution:"
        for row in rows
    ]


def build_counts_for_strategies(model, tokenizer, config: dict[str, Any]) -> torch.Tensor:
    metrics_path = config.get("prediction_set_metrics")
    if not metrics_path:
        raise RuntimeError(
            "prediction_set_metrics with a frequency-table artifact is required; "
            "downstream runs may not rebuild counts"
        )
    tokenizer_id, tokenizer_revision = validate_runtime_model_and_tokenizer(
        model,
        tokenizer,
        config,
    )
    counts, _ = load_frequency_table_from_metrics(
        Path(metrics_path),
        expected_model_id=config["model"],
        expected_model_revision=config.get("model_revision"),
        expected_tokenizer_id=tokenizer_id,
        expected_tokenizer_revision=tokenizer_revision,
        expected_vocab_size=model.config.vocab_size,
        expected_exclusion_token_ids=special_token_ids(tokenizer),
    )
    return counts


def conformal_kwargs_from_metrics(path: str | None) -> dict[str, float] | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        metrics = json.load(f)
    return {
        "kappa": float(metrics["kappa"]),
        "alpha": float(metrics["alpha"]),
        "q_hat": float(metrics["q_hat"]),
    }


def method_specs(config: dict[str, Any], token_counts: torch.Tensor) -> dict[str, dict[str, Any]]:
    methods = {
        "greedy": {"strategy": "greedy", "kwargs": {}},
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


def answer_entropy(answers: list[str | None]) -> float:
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return 0.0
    counts = Counter(valid)
    total = sum(counts.values())
    return float(-sum((count / total) * math.log(count / total + 1e-12) for count in counts.values()))


def evaluate_predictions(rows: list[dict[str, str]], predictions: list[list[str]], k_values: list[int]) -> dict[str, Any]:
    n = len(rows)
    extracted = [[extract_answer(text) for text in row_preds] for row_preds in predictions]
    correct = [
        [answers_match(answer, rows[idx]["answer"]) for answer in row_answers]
        for idx, row_answers in enumerate(extracted)
    ]

    metrics: dict[str, Any] = {"n_questions": n}
    for k in k_values:
        k_eff = min(k, len(predictions[0]))
        pass_count = 0
        maj_count = 0
        for idx in range(n):
            pass_count += int(any(correct[idx][:k_eff]))
            valid_answers = [answer for answer in extracted[idx][:k_eff] if answer is not None]
            if valid_answers:
                majority_answer = Counter(valid_answers).most_common(1)[0][0]
                maj_count += int(answers_match(majority_answer, rows[idx]["answer"]))
        metrics[f"pass@{k}"] = pass_count / n
        metrics[f"maj@{k}"] = maj_count / n

    metrics["acc@1"] = sum(int(row[0]) for row in correct) / n
    total_samples = sum(len(row) for row in extracted)
    invalid = sum(answer is None for row in extracted for answer in row)
    metrics["invalid_answer_rate"] = invalid / max(total_samples, 1)
    metrics["unique_answer_count"] = float(np.mean([
        len(set(answer for answer in row if answer is not None))
        for row in extracted
    ]))
    metrics["answer_entropy"] = float(np.mean([answer_entropy(row) for row in extracted]))
    return metrics


def generate_samples_for_method(
    model,
    tokenizer,
    prompts: list[str],
    method: dict[str, Any],
    config: dict[str, Any],
) -> list[list[str]]:
    samples_per_question = int(config["samples_per_question"])
    predictions = [[] for _ in prompts]
    rounds = 1 if method["strategy"] == "greedy" else samples_per_question

    for sample_idx in range(rounds):
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
            predictions[idx].append(item["text"])

    if rounds == 1 and samples_per_question > 1:
        predictions = [[row[0]] * samples_per_question for row in predictions]
    return predictions


def main() -> None:
    args = parse_args()
    config = merge_args(load_config(args.config), args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    device = resolve_device(config["device"])
    print(f"[reasoning-sc] model={config['model']} device={device}")
    model, tokenizer = load_model_and_tokenizer(config, device)
    token_counts = build_counts_for_strategies(model, tokenizer, config)
    methods = method_specs(config, token_counts)
    t0 = time.time()

    all_results: dict[str, Any] = {
        "model": config["model"],
        "model_revision": config.get("model_revision"),
        "datasets": {},
        "methods": list(methods),
        "samples_per_question": int(config["samples_per_question"]),
        "k_values": config["k_values"],
    }

    for dataset_name in config["datasets"]:
        print(f"[reasoning-sc] loading dataset={dataset_name}")
        rows = load_reasoning_dataset(dataset_name, int(config["n_questions"]), int(config["seed"]))
        prompts = make_prompts(rows)
        dataset_result = {}

        for method_name, method in methods.items():
            print(f"[reasoning-sc] dataset={dataset_name} method={method_name}")
            predictions = generate_samples_for_method(model, tokenizer, prompts, method, config)
            metrics = evaluate_predictions(rows, predictions, list(config["k_values"]))
            dataset_result[method_name] = metrics
            last_k = config["k_values"][-1]
            print(
                f"  acc@1={metrics['acc@1']:.4f} "
                f"pass@{last_k}={metrics[f'pass@{last_k}']:.4f} "
                f"maj@{last_k}={metrics[f'maj@{last_k}']:.4f} "
                f"invalid={metrics['invalid_answer_rate']:.4f}"
            )

        all_results["datasets"][dataset_name] = dataset_result

    all_results["elapsed_sec"] = time.time() - t0
    output_path = output_dir / "reasoning_self_consistency_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[reasoning-sc] wrote {output_path}")


if __name__ == "__main__":
    main()
