#!/usr/bin/env python3
"""
Experiment 4: Real Decoding Comparison + ν-sampling
====================================================
Protocol A.2: Compare certified truncation against top-p / min-p / top-nσ.
Also implements and tests ν-sampling (the paper's new rule).

Strategies:
  (a) Certified: keep i iff s_max - s_i ≤ m* (theory-derived margin)
  (b) top-p=0.95: nucleus sampling
  (c) min-p=0.05: keep i iff p_i/p_max ≥ 0.05
  (d) top-nσ (n=2): keep i iff s_i ≥ s_max - 2σ
  (e) ν-sampling: keep i iff s_max - s_i ≤ m₀ + κ/√(n_i+1)
  (f) Greedy: argmax baseline
"""
import argparse, json, os, time, re
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_model_and_tokenizer, load_gsm8k_passages, load_creative_passages
from samplers import apply_truncation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Truncation Strategies
# ══════════════════════════════════════════════════════════════

def truncate_certified(logits, margin=5.0):
    """Certified truncation: keep tokens within margin of max logit."""
    s_max = logits.max(dim=-1, keepdim=True).values
    keep = (s_max - logits) <= margin
    logits_masked = logits.clone()
    logits_masked[~keep] = float("-inf")
    return logits_masked


def truncate_top_p(logits, p=0.95):
    """Nucleus (top-p) sampling."""
    return apply_truncation(logits, "top_p", p=p)


def truncate_min_p(logits, p_min=0.05):
    """min-p sampling: keep tokens with p_i/p_max ≥ p_min."""
    probs = F.softmax(logits, dim=-1)
    p_max = probs.max(dim=-1, keepdim=True).values
    keep = probs >= p_min * p_max
    logits_masked = logits.clone()
    logits_masked[~keep] = float("-inf")
    return logits_masked


def truncate_top_nsigma(logits, n_sigma=2.0):
    """top-nσ: keep tokens within n*std of max logit."""
    return apply_truncation(logits, "top_nsigma", n_sigma=n_sigma)


def truncate_nu_sampling(logits, token_freq_table, kappa=2.0, m0=3.0):
    """
    ν-sampling: frequency-dependent margin.
    Keep i iff s_max - s_i ≤ m₀ + κ/√(n_i + 1)
    Lower-frequency tokens receive a larger uncertainty margin.
    """
    return apply_truncation(
        logits,
        "nu",
        token_freq_table=token_freq_table,
        kappa=kappa,
        m0=m0,
    )


# ══════════════════════════════════════════════════════════════
#  Generation + Metrics
# ══════════════════════════════════════════════════════════════

def generate_with_strategy(model, tokenizer, prompt_ids, max_new_tokens,
                           strategy_fn, strategy_kwargs, temperature=1.0):
    """Generate text using a specific truncation strategy."""
    device = next(model.parameters()).device
    generated = prompt_ids.clone()
    past_key_values = None
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=generated)
            else:
                outputs = model(input_ids=generated[:, -1:],
                                past_key_values=past_key_values)
            past_key_values = outputs.past_key_values

        raw_logits = outputs.logits[:, -1, :]

        # Apply truncation strategy
        logits_truncated = strategy_fn(raw_logits, **strategy_kwargs)

        # Sample from truncated distribution
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probs = F.softmax(logits_truncated / temperature, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            raise RuntimeError("Invalid truncated distribution; refusing silent fallback.")

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return generated


def compute_metrics(tokenizer, prompt_ids, generated_ids, reference_texts=None):
    """Compute text quality and diversity metrics."""
    # Decode generated tokens (excluding prompt)
    gen_tokens = generated_ids[0][prompt_ids.shape[1]:].cpu().tolist()
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_tokens:
        gen_tokens = gen_tokens[:gen_tokens.index(tokenizer.eos_token_id)]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    metrics = {}

    # Length
    metrics["length"] = len(gen_tokens)

    # Repetition rate: fraction of tokens that are repeats of the previous token
    if len(gen_tokens) > 1:
        repeats = sum(1 for i in range(1, len(gen_tokens))
                      if gen_tokens[i] == gen_tokens[i-1])
        metrics["repetition_rate"] = repeats / (len(gen_tokens) - 1)
    else:
        metrics["repetition_rate"] = 0

    # Distinct-n: fraction of unique n-grams
    for n in [1, 2, 3]:
        if len(gen_tokens) >= n:
            ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
            metrics[f"distinct-{n}"] = len(set(ngrams)) / max(len(ngrams), 1)
        else:
            metrics[f"distinct-{n}"] = 0

    # Trigram repetition: fraction of trigrams appearing more than once
    if len(gen_tokens) >= 3:
        trigrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens)-2)]
        tri_counts = Counter(trigrams)
        repeated_tris = sum(1 for c in tri_counts.values() if c > 1)
        metrics["trigram_repeat_frac"] = repeated_tris / max(len(tri_counts), 1)
    else:
        metrics["trigram_repeat_frac"] = 0

    # Unique token ratio
    metrics["unique_token_ratio"] = len(set(gen_tokens)) / max(len(gen_tokens), 1)

    metrics["text"] = gen_text[:500]  # Store first 500 chars for inspection

    return metrics


# ══════════════════════════════════════════════════════════════
#  Main Experiment
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"

    # ── Load model ──
    print("[exp4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)

    # ── Build token frequency table from local corpus ──
    print("[exp4] Building token frequency table...")
    from data_utils import load_text_samples, tokenize_batch
    texts = load_text_samples(2000, max_length=4096, seed=args.seed)
    all_text = " ".join(texts)
    all_token_ids = tokenizer.encode(all_text, add_special_tokens=False)
    token_counts = torch.zeros(model.config.vocab_size, dtype=torch.float32)
    for tid in all_token_ids:
        if tid < model.config.vocab_size:
            token_counts[tid] += 1
    print(f"  Corpus tokens: {len(all_token_ids)}, unique: {len(set(all_token_ids))}")

    # ── Prepare prompts ──
    print("[exp4] Preparing prompts...")
    factual_passages = load_gsm8k_passages(n=args.n_prompts)
    creative_passages = load_creative_passages(n=args.n_prompts)
    all_prompts = [
        ("factual", p[:200]) for p in factual_passages[:args.n_prompts]
    ] + [
        ("creative", p[:200]) for p in creative_passages[:args.n_prompts]
    ]

    # ── Define strategies ──
    strategies = {
        "greedy": {
            "fn": lambda logits, **kw: logits,  # no truncation, use argmax
            "kwargs": {},
            "sample": False,
        },
        "top_p_0.95": {
            "fn": truncate_top_p,
            "kwargs": {"p": 0.95},
            "sample": True,
        },
        "min_p_0.05": {
            "fn": truncate_min_p,
            "kwargs": {"p_min": 0.05},
            "sample": True,
        },
        "top_nsigma_2": {
            "fn": truncate_top_nsigma,
            "kwargs": {"n_sigma": 2.0},
            "sample": True,
        },
        "certified_m5": {
            "fn": truncate_certified,
            "kwargs": {"margin": 5.0},
            "sample": True,
        },
        "nu_sampling": {
            "fn": truncate_nu_sampling,
            "kwargs": {"token_freq_table": token_counts, "kappa": 2.0, "m0": 3.0},
            "sample": True,
        },
    }

    # ── Run experiment ──
    all_results = {name: {"factual": [], "creative": []} for name in strategies}
    t0 = time.time()

    for strat_name, strat_config in strategies.items():
        print(f"\n[exp4] Strategy: {strat_name}")
        strategy_fn = strat_config["fn"]
        strategy_kwargs = strat_config["kwargs"]
        use_sampling = strat_config["sample"]

        for label, prompt in all_prompts:
            # Encode prompt
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=80).to(device)
            prompt_ids = enc["input_ids"]

            if use_sampling:
                torch.manual_seed(args.seed)
                gen_ids = generate_with_strategy(
                    model, tokenizer, prompt_ids, args.max_new_tokens,
                    strategy_fn, strategy_kwargs, temperature=args.temperature
                )
            else:
                # Greedy generation
                with torch.no_grad():
                    out = model.generate(
                        prompt_ids, max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                gen_ids = out

            metrics = compute_metrics(tokenizer, prompt_ids, gen_ids)
            all_results[strat_name][label].append(metrics)

        # Print summary
        for label in ["factual", "creative"]:
            items = all_results[strat_name][label]
            avg_rep = np.mean([m["repetition_rate"] for m in items])
            avg_d1 = np.mean([m["distinct-1"] for m in items])
            avg_d2 = np.mean([m["distinct-2"] for m in items])
            avg_tri = np.mean([m["trigram_repeat_frac"] for m in items])
            print(f"  [{label:8s}] rep={avg_rep:.4f}  d1={avg_d1:.4f}  "
                  f"d2={avg_d2:.4f}  tri_rep={avg_tri:.4f}")

    elapsed = time.time() - t0
    print(f"\n[exp4] All strategies done in {elapsed:.1f}s")

    # ── Aggregate results ──
    summary = {}
    for strat_name in strategies:
        summary[strat_name] = {}
        for label in ["factual", "creative"]:
            items = all_results[strat_name][label]
            summary[strat_name][label] = {
                "n": len(items),
                "avg_length": float(np.mean([m["length"] for m in items])),
                "repetition_rate": float(np.mean([m["repetition_rate"] for m in items])),
                "distinct-1": float(np.mean([m["distinct-1"] for m in items])),
                "distinct-2": float(np.mean([m["distinct-2"] for m in items])),
                "distinct-3": float(np.mean([m["distinct-3"] for m in items])),
                "trigram_repeat_frac": float(np.mean([m["trigram_repeat_frac"] for m in items])),
                "unique_token_ratio": float(np.mean([m["unique_token_ratio"] for m in items])),
            }

    # ── Plot comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    strat_names = list(strategies.keys())
    colors = ["gray", "blue", "green", "orange", "red", "purple"]

    # Top-left: Repetition rate
    ax = axes[0, 0]
    x = np.arange(len(strat_names))
    width = 0.35
    fact_reps = [summary[s]["factual"]["repetition_rate"] for s in strat_names]
    crea_reps = [summary[s]["creative"]["repetition_rate"] for s in strat_names]
    ax.bar(x - width/2, fact_reps, width, label="Factual", color="forestgreen")
    ax.bar(x + width/2, crea_reps, width, label="Creative", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Repetition Rate", fontsize=12)
    ax.set_title("Token Repetition Rate (lower = better)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: Distinct-2
    ax = axes[0, 1]
    fact_d2 = [summary[s]["factual"]["distinct-2"] for s in strat_names]
    crea_d2 = [summary[s]["creative"]["distinct-2"] for s in strat_names]
    ax.bar(x - width/2, fact_d2, width, label="Factual", color="forestgreen")
    ax.bar(x + width/2, crea_d2, width, label="Creative", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Distinct-2", fontsize=12)
    ax.set_title("Bigram Diversity (higher = better)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Trigram repetition
    ax = axes[1, 0]
    fact_tri = [summary[s]["factual"]["trigram_repeat_frac"] for s in strat_names]
    crea_tri = [summary[s]["creative"]["trigram_repeat_frac"] for s in strat_names]
    ax.bar(x - width/2, fact_tri, width, label="Factual", color="forestgreen")
    ax.bar(x + width/2, crea_tri, width, label="Creative", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Trigram Repeat Fraction", fontsize=12)
    ax.set_title("Phrase-Level Repetition (lower = better)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Quality-Diversity scatter
    ax = axes[1, 1]
    for i, s in enumerate(strat_names):
        rep = (summary[s]["factual"]["repetition_rate"] +
               summary[s]["creative"]["repetition_rate"]) / 2
        div = (summary[s]["factual"]["distinct-2"] +
               summary[s]["creative"]["distinct-2"]) / 2
        ax.scatter(rep, div, s=150, c=colors[i], zorder=5)
        ax.annotate(s, (rep, div), fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("Average Repetition Rate (lower = better)", fontsize=12)
    ax.set_ylabel("Average Distinct-2 (higher = better)", fontsize=12)
    ax.set_title("Quality-Diversity Trade-off", fontsize=13)
    ax.grid(True, alpha=0.3)
    # Ideal is bottom-right (low rep, high diversity)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0.05, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig4_decoding_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save results ──
    save_data = {
        "model": args.model,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "summary": summary,
        "sample_texts": {
            strat: {
                label: [m["text"] for m in all_results[strat][label][:3]]
                for label in ["factual", "creative"]
            }
            for strat in strategies
        },
    }
    with open(f"{args.output_dir}/exp4_decoding_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n[exp4] Results saved to {args.output_dir}/exp4_decoding_results.json")
    print(f"[exp4] Figure saved to {args.output_dir}/fig4_decoding_comparison.png")

    # ── Print comparison table ──
    print("\n" + "=" * 80)
    print("DECODING STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<18} {'Rep Rate':>10} {'Distinct-1':>12} {'Distinct-2':>12} {'Tri Rep':>10}")
    print("-" * 80)
    for s in strat_names:
        avg_rep = (summary[s]["factual"]["repetition_rate"] +
                   summary[s]["creative"]["repetition_rate"]) / 2
        avg_d1 = (summary[s]["factual"]["distinct-1"] +
                  summary[s]["creative"]["distinct-1"]) / 2
        avg_d2 = (summary[s]["factual"]["distinct-2"] +
                  summary[s]["creative"]["distinct-2"]) / 2
        avg_tri = (summary[s]["factual"]["trigram_repeat_frac"] +
                   summary[s]["creative"]["trigram_repeat_frac"]) / 2
        print(f"{s:<18} {avg_rep:>10.4f} {avg_d1:>12.4f} {avg_d2:>12.4f} {avg_tri:>10.4f}")

    print("[exp4] Done!")


if __name__ == "__main__":
    main()
