#!/usr/bin/env python3
"""
ν-sampling parameter sweep — BATCHED GPU version.
Batch multiple prompts together to maximize RTX 5090 utilization.
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_model_and_tokenizer, load_gsm8k_passages, load_creative_passages


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_truncation(logits, strategy, token_freq_table=None, **kwargs):
    """Apply truncation strategy to batched logits. logits: (B, V)"""
    if strategy == "top_nsigma":
        n_sigma = kwargs.get("n_sigma", 2.0)
        sigma = logits.std(dim=-1, keepdim=True)
        s_max = logits.max(dim=-1, keepdim=True).values
        keep = (s_max - logits) <= n_sigma * sigma
        logits[~keep] = float("-inf")
    elif strategy == "nu":
        kappa = kwargs.get("kappa", 10.0)
        m0 = kwargs.get("m0", 3.0)
        s_max = logits.max(dim=-1, keepdim=True).values
        n_i = token_freq_table.to(logits.device).float().unsqueeze(0).expand_as(logits)
        margin = m0 + kappa / torch.sqrt(n_i + 1)
        keep = (s_max - logits) <= margin
        logits[~keep] = float("-inf")
    return logits


@torch.no_grad()
def batch_generate(model, tokenizer, prompt_texts, max_new_tokens, batch_size,
                   strategy, strategy_kwargs, temperature=1.0):
    """Batch generation with KV cache. Returns list of generated token lists."""
    device = next(model.parameters()).device
    all_generated = []

    # Process in batches
    for batch_start in range(0, len(prompt_texts), batch_size):
        batch_prompts = prompt_texts[batch_start:batch_start + batch_size]
        B = len(batch_prompts)

        # Tokenize with padding
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=80).to(device)
        input_ids = enc["input_ids"]        # (B, T_prompt)
        attention_mask = enc["attention_mask"]  # (B, T_prompt)
        prompt_lens = attention_mask.sum(dim=-1).cpu().tolist()

        # Initialize generated sequences
        generated = input_ids.clone()  # (B, T_prompt)
        gen_mask = attention_mask.clone()  # (B, T_prompt)

        # KV cache
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                # First step: process full prompt
                outputs = model(input_ids=generated, attention_mask=gen_mask,
                                use_cache=True)
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]  # (B, V)
            else:
                # Subsequent steps: only the last token
                outputs = model(input_ids=generated[:, -1:],
                                attention_mask=gen_mask,
                                past_key_values=past_key_values,
                                use_cache=True)
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]  # (B, V)

            # Temperature
            next_logits = next_logits / temperature

            # Apply truncation (modifies in-place)
            next_logits = apply_truncation(next_logits.clone(), strategy,
                                           **strategy_kwargs)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            # Handle all-inf rows (fallback to untruncated)
            for b in range(B):
                if probs[b].sum() < 0.5:
                    fallback = F.softmax(outputs.logits[b, -1, :] / temperature, dim=-1)
                    probs[b] = fallback

            next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            gen_mask = torch.cat([gen_mask, torch.ones(B, 1, device=device, dtype=gen_mask.dtype)], dim=-1)

        # Extract generated tokens (remove prompt)
        for b in range(B):
            plen = prompt_lens[b]
            gen_tokens = generated[b, plen:].cpu().tolist()
            all_generated.append(gen_tokens)

    return all_generated


def compute_metrics(gen_tokens):
    """Compute metrics from generated token list."""
    metrics = {"length": len(gen_tokens)}
    if len(gen_tokens) > 1:
        repeats = sum(1 for i in range(1, len(gen_tokens))
                      if gen_tokens[i] == gen_tokens[i-1])
        metrics["repetition_rate"] = repeats / (len(gen_tokens) - 1)
    else:
        metrics["repetition_rate"] = 0
    for n in [1, 2, 3]:
        if len(gen_tokens) >= n:
            ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
            metrics[f"distinct-{n}"] = len(set(ngrams)) / max(len(ngrams), 1)
        else:
            metrics[f"distinct-{n}"] = 0
    if len(gen_tokens) >= 3:
        trigrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens)-2)]
        tri_counts = Counter(trigrams)
        repeated_tris = sum(1 for c in tri_counts.values() if c > 1)
        metrics["trigram_repeat_frac"] = repeated_tris / max(len(tri_counts), 1)
    else:
        metrics["trigram_repeat_frac"] = 0
    metrics["unique_token_ratio"] = len(set(gen_tokens)) / max(len(gen_tokens), 1)
    return metrics


def run_strategy(model, tokenizer, prompts_by_label, strategy, strategy_kwargs,
                 max_new_tokens, batch_size, temperature, seed, name):
    """Run a strategy on all prompts, batched."""
    results = {}
    for label, prompt_texts in prompts_by_label.items():
        torch.manual_seed(seed)
        all_gen = batch_generate(
            model, tokenizer, prompt_texts, max_new_tokens, batch_size,
            strategy, strategy_kwargs, temperature
        )
        metrics_list = [compute_metrics(g) for g in all_gen]
        results[label] = {
            "repetition_rate": float(np.mean([m["repetition_rate"] for m in metrics_list])),
            "distinct-1": float(np.mean([m["distinct-1"] for m in metrics_list])),
            "distinct-2": float(np.mean([m["distinct-2"] for m in metrics_list])),
            "distinct-3": float(np.mean([m["distinct-3"] for m in metrics_list])),
            "trigram_repeat_frac": float(np.mean([m["trigram_repeat_frac"] for m in metrics_list])),
        }

    avg_rep = (results["factual"]["repetition_rate"] + results["creative"]["repetition_rate"]) / 2
    avg_d2 = (results["factual"]["distinct-2"] + results["creative"]["distinct-2"]) / 2
    avg_tri = (results["factual"]["trigram_repeat_frac"] + results["creative"]["trigram_repeat_frac"]) / 2
    print(f"  {name:25s}: rep={avg_rep:.4f}  d2={avg_d2:.4f}  tri={avg_tri:.4f}")
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"
    print(f"[nu-sweep] Device: {device}")
    print(f"[nu-sweep] Batch size: {args.batch_size}")

    print("[nu-sweep] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)

    print("[nu-sweep] Building token frequency table...")
    from data_utils import load_text_samples
    texts = load_text_samples(2000, max_length=4096, seed=args.seed)
    all_text = " ".join(texts)
    all_token_ids = tokenizer.encode(all_text, add_special_tokens=False)
    token_counts = torch.zeros(model.config.vocab_size, dtype=torch.float32)
    for tid in all_token_ids:
        if tid < model.config.vocab_size:
            token_counts[tid] += 1

    print("[nu-sweep] Preparing prompts...")
    factual = load_gsm8k_passages(n=args.n_prompts)
    creative = load_creative_passages(n=args.n_prompts)
    prompts_by_label = {
        "factual": [p[:200] for p in factual[:args.n_prompts]],
        "creative": [p[:200] for p in creative[:args.n_prompts]],
    }
    total = sum(len(v) for v in prompts_by_label.values())
    print(f"  Total prompts: {total}")

    all_results = {}
    t0 = time.time()

    # Baseline: top-nσ=2
    print("\n[nu-sweep] === Baseline ===")
    all_results["top_nsigma_2"] = run_strategy(
        model, tokenizer, prompts_by_label, "top_nsigma", {"n_sigma": 2.0},
        args.max_new_tokens, args.batch_size, args.temperature, args.seed, "top_nsigma_2"
    )

    # Sweep
    kappa_values = [5, 10, 20]
    m0_values = [1.0, 3.0, 5.0]

    print("\n[nu-sweep] === ν-sampling sweep ===")
    for m0 in m0_values:
        for kappa in kappa_values:
            name = f"nu_k{kappa}_m{int(m0)}"
            all_results[name] = run_strategy(
                model, tokenizer, prompts_by_label, "nu",
                {"token_freq_table": token_counts, "kappa": float(kappa), "m0": m0},
                args.max_new_tokens, args.batch_size, args.temperature, args.seed, name
            )

    elapsed = time.time() - t0
    print(f"\n[nu-sweep] All done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("ν-SAMPLING PARAMETER SWEEP (Batched GPU)")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Rep Rate':>10} {'Distinct-2':>12} {'Tri Rep':>10}")
    print("-" * 80)

    best_name = None
    best_score = float("inf")

    for name, s in all_results.items():
        avg_rep = (s["factual"]["repetition_rate"] + s["creative"]["repetition_rate"]) / 2
        avg_d2 = (s["factual"]["distinct-2"] + s["creative"]["distinct-2"]) / 2
        avg_tri = (s["factual"]["trigram_repeat_frac"] + s["creative"]["trigram_repeat_frac"]) / 2
        score = avg_rep * 10 + avg_tri * 5 - avg_d2
        if score < best_score and name != "top_nsigma_2":
            best_score = score
            best_name = name
        print(f"{name:<25} {avg_rep:>10.4f} {avg_d2:>12.4f} {avg_tri:>10.4f}")

    print("-" * 80)
    if best_name:
        print(f"Best ν config: {best_name}")

    # ── Heatmap ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, metric in enumerate(["repetition_rate", "distinct-2", "trigram_repeat_frac"]):
        ax = axes[ax_idx]
        data = np.zeros((len(m0_values), len(kappa_values)))
        for i, m0 in enumerate(m0_values):
            for j, k in enumerate(kappa_values):
                name = f"nu_k{k}_m{int(m0)}"
                s = all_results[name]
                data[i, j] = (s["factual"][metric] + s["creative"][metric]) / 2

        cmap = "RdYlGn" if metric == "distinct-2" else "RdYlGn_r"
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(kappa_values)))
        ax.set_xticklabels(kappa_values)
        ax.set_yticks(range(len(m0_values)))
        ax.set_yticklabels(m0_values)
        ax.set_xlabel("κ (frequency penalty)", fontsize=12)
        ax.set_ylabel("m₀ (base margin)", fontsize=12)
        titles = {"repetition_rate": "Token Repetition (↓ better)",
                   "distinct-2": "Distinct-2 (↑ better)",
                   "trigram_repeat_frac": "Trigram Repeat (↓ better)"}
        ax.set_title(titles[metric], fontsize=12)
        for i in range(len(m0_values)):
            for j in range(len(kappa_values)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig4b_nu_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save
    with open(f"{args.output_dir}/exp4b_nu_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[nu-sweep] Saved to {args.output_dir}/exp4b_nu_sweep_results.json")
    print(f"[nu-sweep] Figure: {args.output_dir}/fig4b_nu_sweep.png")
    print("[nu-sweep] Done!")


if __name__ == "__main__":
    main()
