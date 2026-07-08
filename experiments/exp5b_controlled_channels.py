#!/usr/bin/env python3
"""Controlled channel probes for frequency-dependent logit sensitivity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_prediction_sets import build_token_counts, load_model_and_tokenizer, load_texts, resolve_device


BUCKETS = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-10", 3, 10),
    ("11-100", 11, 100),
    (">100", 101, None),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None,
                   choices=["wikitext", "c4", "local", "text_file"])
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--n-texts", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--channels", nargs="+", default=None)
    p.add_argument("--noise-sigma", type=float, default=None)
    p.add_argument("--n-perturbations", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
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
        "dataset": "wikitext",
        "split": "validation",
        "n_texts": 512,
        "max_length": 128,
        "batch_size": 4,
        "channels": ["hidden_noise", "dropout"],
        "noise_sigma": 0.01,
        "n_perturbations": 4,
        "seed": 42,
        "dtype": "float16",
        "device": "cuda",
        "output_dir": "./results/controlled_channels_qwen3b",
        "trust_remote_code": True,
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


def encode_batches(tokenizer, texts: list[str], config: dict[str, Any], device: torch.device):
    max_length = int(config["max_length"])
    batch_size = int(config["batch_size"])
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue
        yield input_ids, attention_mask


def bucket_stats(freqs: torch.Tensor, values: torch.Tensor) -> dict[str, Any]:
    freqs_cpu = freqs.detach().cpu()
    values_cpu = values.detach().float().cpu()
    out = {}
    for name, low, high in BUCKETS:
        if high is None:
            mask = freqs_cpu >= low
        else:
            mask = (freqs_cpu >= low) & (freqs_cpu <= high)
        if not mask.any():
            out[name] = {"n": 0, "mean": None, "variance": None}
            continue
        selected = values_cpu[mask]
        out[name] = {
            "n": int(selected.numel()),
            "mean": float(selected.mean().item()),
            "variance": float(selected.var(unbiased=False).item()) if selected.numel() > 1 else 0.0,
        }
    return out


def gather_target_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return logits[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def hidden_noise_channel(model, tokenizer, texts: list[str], token_counts: torch.Tensor, config: dict[str, Any], device: torch.device):
    model.eval()
    embedding = model.get_input_embeddings()
    all_freqs = []
    all_delta_abs = []
    sigma = float(config["noise_sigma"])
    n_perturb = int(config["n_perturbations"])

    for input_ids, attention_mask in encode_batches(tokenizer, texts, config, device):
        targets = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        clean_embeds = embedding(input_ids)
        clean_logits = model(inputs_embeds=clean_embeds, attention_mask=attention_mask).logits
        clean_target_logits = gather_target_logits(clean_logits.float(), targets)

        for _ in range(n_perturb):
            noise = sigma * torch.randn_like(clean_embeds)
            noisy_logits = model(inputs_embeds=clean_embeds + noise, attention_mask=attention_mask).logits
            noisy_target_logits = gather_target_logits(noisy_logits.float(), targets)
            delta = (noisy_target_logits - clean_target_logits).abs()
            all_delta_abs.append(delta[valid].detach().cpu())
            all_freqs.append(token_counts.to(device)[targets][valid].detach().cpu())

    freqs = torch.cat(all_freqs)
    delta_abs = torch.cat(all_delta_abs)
    return {
        "channel": "hidden_noise",
        "noise_sigma": sigma,
        "n_perturbations": n_perturb,
        "overall_mean_abs_delta": float(delta_abs.mean().item()),
        "bucket_stats": bucket_stats(freqs, delta_abs),
    }


@torch.no_grad()
def dropout_channel(model, tokenizer, texts: list[str], token_counts: torch.Tensor, config: dict[str, Any], device: torch.device):
    model.train()
    all_freqs = []
    all_variances = []
    n_perturb = int(config["n_perturbations"])

    for input_ids, attention_mask in encode_batches(tokenizer, texts, config, device):
        targets = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        samples = []
        for _ in range(n_perturb):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            samples.append(gather_target_logits(logits.float(), targets).detach().cpu())
        stacked = torch.stack(samples, dim=0)
        variances = stacked.var(dim=0, unbiased=False)
        all_variances.append(variances[valid.cpu()])
        all_freqs.append(token_counts.to(device)[targets][valid].detach().cpu())

    model.eval()
    freqs = torch.cat(all_freqs)
    variances = torch.cat(all_variances)
    return {
        "channel": "dropout",
        "n_perturbations": n_perturb,
        "overall_target_logit_variance": float(variances.mean().item()),
        "bucket_stats": bucket_stats(freqs, variances),
    }


def plot_results(results: dict[str, Any], output_dir: Path) -> None:
    channels = results["channels"]
    x = np.arange(len(BUCKETS))
    width = min(0.8 / max(len(channels), 1), 0.25)

    plt.figure(figsize=(10, 5.5))
    for idx, (channel_name, info) in enumerate(channels.items()):
        values = []
        for bucket, _, _ in BUCKETS:
            mean = info["bucket_stats"][bucket]["mean"]
            values.append(np.nan if mean is None else max(float(mean), 1e-12))
        offset = (idx - (len(channels) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=channel_name)
    plt.xticks(x, [name for name, _, _ in BUCKETS])
    plt.yscale("log")
    plt.xlabel("Target-token frequency bucket")
    plt.ylabel("Channel sensitivity metric (log)")
    plt.title("Controlled channel sensitivity by token frequency")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "controlled_channel_frequency_sensitivity.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    config = merge_args(load_config(args.config), args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    device = resolve_device(config["device"])
    print(f"[controlled-channels] model={config['model']} device={device}")
    texts = load_texts(config)
    model, tokenizer = load_model_and_tokenizer(config, device)
    token_counts = build_token_counts(
        tokenizer,
        texts,
        vocab_size=model.config.vocab_size,
        max_length=int(config["max_length"]),
        batch_size=int(config["batch_size"]),
    )

    t0 = time.time()
    channel_results = {}
    if "hidden_noise" in config["channels"]:
        print("[controlled-channels] running hidden_noise")
        channel_results["hidden_noise"] = hidden_noise_channel(model, tokenizer, texts, token_counts, config, device)
    if "dropout" in config["channels"]:
        print("[controlled-channels] running dropout")
        channel_results["dropout"] = dropout_channel(model, tokenizer, texts, token_counts, config, device)

    results = {
        "model": config["model"],
        "dataset": config["dataset"],
        "split": config.get("split"),
        "n_texts": len(texts),
        "max_length": int(config["max_length"]),
        "elapsed_sec": time.time() - t0,
        "channels": channel_results,
        "claim_boundary": (
            "This probes controlled perturbation sensitivity. It does not identify the real LLM noise channel."
        ),
    }
    output_path = output_dir / "controlled_channel_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, output_dir)
    print(f"[controlled-channels] wrote {output_path}")
    print(f"[controlled-channels] wrote {output_dir / 'controlled_channel_frequency_sensitivity.png'}")


if __name__ == "__main__":
    main()
