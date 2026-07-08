#!/usr/bin/env python3
"""
Experiment 6: Cross-Model + Cross-Condition Ablation Study
============================================================
Systematic ablation across:
  - Model size: Qwen2.5-3B vs 7B
  - Temperature: {0.5, 0.8, 1.0, 1.5}
  - Sequence length: {100, 200, 500}
  - σ₀ and c parameters for synthetic channel

Validates that core findings generalize beyond specific model/config choices.
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_model_and_tokenizer, load_text_samples, tokenize_batch, free_model
from samplers import batch_generate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"])
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(gen_tokens):
    metrics = {"length": len(gen_tokens)}
    if len(gen_tokens) > 1:
        repeats = sum(1 for i in range(1, len(gen_tokens)) if gen_tokens[i] == gen_tokens[i-1])
        metrics["rep_rate"] = repeats / (len(gen_tokens) - 1)
    else:
        metrics["rep_rate"] = 0
    for n in [1, 2, 3]:
        if len(gen_tokens) >= n:
            ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
            metrics[f"distinct-{n}"] = len(set(ngrams)) / max(len(ngrams), 1)
        else:
            metrics[f"distinct-{n}"] = 0
    if len(gen_tokens) >= 3:
        trigrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens)-2)]
        tri_counts = Counter(trigrams)
        repeated = sum(1 for c in tri_counts.values() if c > 1)
        metrics["tri_rep"] = repeated / max(len(tri_counts), 1)
    else:
        metrics["tri_rep"] = 0
    metrics["vocab_richness"] = len(set(gen_tokens)) / max(len(gen_tokens), 1)
    return metrics


# ══════════════════════════════════════════════════════════════
#  Ablation 2: Synthetic Channel Parameter Sweep
# ══════════════════════════════════════════════════════════════

def ablation_synthetic_params(device):
    """Sweep σ₀ and c in the synthetic channel, verify c_fit > 0 robustly."""
    print("\n═══ Ablation: Synthetic Channel Parameter Sweep ═══")
    from scipy.optimize import curve_fit

    # Parameters to sweep
    sigma0_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    c_values = [10, 50, 100, 200, 500]

    # Simulated data
    n_tokens = 5000
    np.random.seed(42)
    # Zipf-like frequencies
    ranks = np.arange(1, 1001)
    freqs = 1.0 / ranks
    freqs = freqs / freqs.sum() * n_tokens
    freqs = freqs.astype(int)
    freqs = np.maximum(freqs, 1)

    results = []
    for sigma0 in sigma0_values:
        for c in c_values:
            # Generate synthetic residuals
            residuals = []
            n_per_token = []
            for i, n_i in enumerate(freqs):
                sigma_i = np.sqrt(sigma0**2 + c / n_i)
                r = np.random.normal(0, sigma_i, max(n_i, 1))
                residuals.extend(r.tolist())
                n_per_token.extend([n_i] * len(r))

            residuals = np.array(residuals)
            n_per_token = np.array(n_per_token)

            # Bin and fit
            log_n = np.log10(n_per_token + 1)
            n_bins = 20
            bin_edges = np.linspace(log_n.min(), log_n.max() + 0.01, n_bins + 1)
            bin_centers, bin_vars = [], []
            for b in range(n_bins):
                mask = (log_n >= bin_edges[b]) & (log_n < bin_edges[b + 1])
                if mask.sum() < 10:
                    continue
                bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
                bin_vars.append(np.var(residuals[mask]))

            bin_centers = np.array(bin_centers)
            bin_vars = np.array(bin_vars)

            def var_model(n, s0_sq, c_fit):
                return s0_sq + c_fit / (n + 1e-8)

            try:
                popt, _ = curve_fit(var_model, bin_centers, bin_vars,
                                    p0=[sigma0**2, c], maxfev=5000,
                                    bounds=([0, -np.inf], [np.inf, np.inf]))
                s0_fit, c_fit = popt
                # R²
                pred = var_model(bin_centers, s0_fit, c_fit)
                ss_res = np.sum((bin_vars - pred)**2)
                ss_tot = np.sum((bin_vars - np.mean(bin_vars))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
            except Exception:
                s0_fit, c_fit, r2 = 0, 0, 0

            results.append({
                "sigma0": sigma0, "c": c,
                "sigma0_fit": float(s0_fit), "c_fit": float(c_fit),
                "c_positive": c_fit > 0,
                "c_ratio": float(c_fit / c) if c > 0 else 0,
                "r2": float(r2),
            })

    # Print summary
    print(f"  {'σ₀':>6} {'c':>6} {'c_fit':>8} {'c_fit/c':>8} {'R²':>6} {'c>0':>5}")
    print("  " + "-" * 50)
    for r in results:
        mark = "✓" if r["c_positive"] else "✗"
        print(f"  {r['sigma0']:>6.2f} {r['c']:>6.0f} {r['c_fit']:>8.2f} "
              f"{r['c_ratio']:>8.3f} {r['r2']:>6.3f} {mark:>5}")

    n_pass = sum(1 for r in results if r["c_positive"])
    print(f"\n  c > 0 in {n_pass}/{len(results)} configurations")

    return results


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"
    temperatures = [0.5, 0.8, 1.0, 1.5]
    seq_lengths = [100, 200, 500]
    strategies = {
        "top_p_0.95": {"strategy": "top_p", "kwargs": {"p": 0.95}},
        "top_nsigma_2": {"strategy": "top_nsigma", "kwargs": {"n_sigma": 2.0}},
        "nu_k10_m3": {"strategy": "nu", "kwargs": {"kappa": 10.0, "m0": 3.0}},
    }

    # ── Ablation 1: Cross-model × temperature × strategy ──
    all_decoding_results = {}

    # Prepare prompts
    CREATIVE_PROMPTS = [
        "Write a short story about a robot learning to paint. The robot",
        "Describe a mysterious island that appears only during full moons. The island",
        "Write about a chef who discovers that their cooking can time-travel. The chef",
        "Describe a world where music has visible colors and shapes. In this world",
        "Write about a librarian who finds a book that writes itself. The book",
        "Describe a city built entirely inside a giant tree. The city",
        "Write about a child who can talk to shadows. One day the shadows",
        "Describe a garden where the flowers tell secrets. The garden",
        "Write about an old clock that counts backwards. When it reaches zero",
        "Describe a mountain that moves one step closer every night. The mountain",
        "Write a story about a painter whose portraits come alive at midnight.",
        "Describe a forest where the trees remember everyone who has passed through.",
        "Write about a musician who plays a violin made of starlight.",
        "Describe a river that flows upward into the sky.",
        "Write about a door that opens to a different place each time.",
        "Describe a library where the books rearrange themselves by mood.",
        "Write about a cat that collects lost memories from the street.",
        "Describe a lighthouse that guides dreams instead of ships.",
        "Write about a tailor who sews constellations into coats.",
        "Describe a tea shop where each blend reveals a different future.",
        "Write about a bridge that only appears when two people miss each other.",
        "Describe a mirror that shows who you were in another life.",
        "Write about a garden where plants grow letters instead of flowers.",
        "Describe a train station where time runs at different speeds on each platform.",
        "Write about a baker whose bread rises only when someone tells the truth.",
        "Describe an umbrella that protects from bad memories instead of rain.",
        "Write about a compass that points toward the nearest adventure.",
        "Describe a bookshop where the stories leak into the real world.",
        "Write about a window that looks out onto different centuries.",
        "Describe a piano that plays the emotions of whoever sits at it.",
    ]
    prompts = CREATIVE_PROMPTS[:args.n_prompts]

    for model_name in args.models:
        model_short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_name, dtype=torch.float16)

        # Build token frequency table
        texts = load_text_samples(2000, max_length=4096, seed=args.seed)
        all_text = " ".join(texts)
        all_token_ids = tokenizer.encode(all_text, add_special_tokens=False)
        token_counts = torch.zeros(model.config.vocab_size, dtype=torch.float32)
        for tid in all_token_ids:
            if tid < model.config.vocab_size:
                token_counts[tid] += 1

        # Update nu kwargs with token table
        strategies["nu_k10_m3"]["kwargs"]["token_freq_table"] = token_counts

        model_results = {}

        # ── Temperature ablation ──
        for temp in temperatures:
            print(f"\n  Temperature = {temp}")
            temp_results = {}
            for strat_name, strat_config in strategies.items():
                torch.manual_seed(args.seed)
                gen = batch_generate(
                    model, tokenizer, prompts, 200, args.batch_size,
                    strat_config["strategy"], strat_config["kwargs"], temp,
                    return_dict=False
                )
                metrics_list = [compute_metrics(g) for g in gen]
                agg = {k: float(np.mean([m[k] for m in metrics_list]))
                       for k in metrics_list[0]}
                temp_results[strat_name] = agg
                print(f"    {strat_name:15s}: d2={agg['distinct-2']:.4f}  "
                      f"rep={agg['rep_rate']:.4f}  tri={agg['tri_rep']:.4f}")

            model_results[f"temp_{temp}"] = temp_results

        # ── Sequence length ablation (at temp=1.0) ──
        print(f"\n  Sequence length ablation (T=1.0)")
        for seq_len in seq_lengths:
            len_results = {}
            for strat_name, strat_config in strategies.items():
                torch.manual_seed(args.seed)
                gen = batch_generate(
                    model, tokenizer, prompts, seq_len, args.batch_size,
                    strat_config["strategy"], strat_config["kwargs"], 1.0,
                    return_dict=False
                )
                metrics_list = [compute_metrics(g) for g in gen]
                agg = {k: float(np.mean([m[k] for m in metrics_list]))
                       for k in metrics_list[0]}
                len_results[strat_name] = agg
                print(f"    L={seq_len:3d} {strat_name:15s}: d2={agg['distinct-2']:.4f}  "
                      f"rep={agg['rep_rate']:.4f}")

            model_results[f"seqlen_{seq_len}"] = len_results

        all_decoding_results[model_short] = model_results
        free_model(model)

    # ── Ablation 2: Synthetic channel parameter sweep ──
    synth_results = ablation_synthetic_params(device)

    # ══════════════════════════════════════════════════════════
    #  Analysis & Plotting
    # ══════════════════════════════════════════════════════════

    # ── Figure 1: Temperature × Strategy heatmap per model ──
    n_models = len(all_decoding_results)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for m_idx, (model_name, m_results) in enumerate(all_decoding_results.items()):
        strat_names = list(strategies.keys())

        for metric_idx, (metric, title) in enumerate([
            ("distinct-2", "Distinct-2 (↑)"),
            ("rep_rate", "Rep Rate (↓)"),
            ("tri_rep", "Trigram Rep (↓)")
        ]):
            ax = axes[m_idx, metric_idx]
            data = np.zeros((len(temperatures), len(strat_names)))
            for i, temp in enumerate(temperatures):
                key = f"temp_{temp}"
                for j, sn in enumerate(strat_names):
                    data[i, j] = m_results[key][sn][metric]

            cmap = "RdYlGn" if metric == "distinct-2" else "RdYlGn_r"
            im = ax.imshow(data, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(strat_names)))
            ax.set_xticklabels([s.replace("_", "\n") for s in strat_names], fontsize=8)
            ax.set_yticks(range(len(temperatures)))
            ax.set_yticklabels(temperatures)
            ax.set_xlabel("Strategy", fontsize=10)
            ax.set_ylabel("Temperature", fontsize=10)
            ax.set_title(f"{model_name}: {title}", fontsize=11)
            for i in range(len(temperatures)):
                for j in range(len(strat_names)):
                    ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig6a_temp_strategy_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Sequence length × Strategy ──
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for m_idx, (model_name, m_results) in enumerate(all_decoding_results.items()):
        strat_names = list(strategies.keys())

        for metric_idx, (metric, title) in enumerate([
            ("distinct-2", "Distinct-2"),
            ("tri_rep", "Trigram Repeat")
        ]):
            ax = axes[m_idx, metric_idx]
            for j, sn in enumerate(strat_names):
                vals = []
                for sl in seq_lengths:
                    key = f"seqlen_{sl}"
                    vals.append(m_results[key][sn][metric])
                ax.plot(seq_lengths, vals, "o-", label=sn, markersize=8, lw=2)
            ax.set_xlabel("Generation length L", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"{model_name}: {title} vs Length", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig6b_seqlen_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: Synthetic channel c_fit heatmap ──
    sigma0_vals = sorted(set(r["sigma0"] for r in synth_results))
    c_vals = sorted(set(r["c"] for r in synth_results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: c_fit/c ratio
    ax = axes[0]
    ratio_data = np.zeros((len(sigma0_vals), len(c_vals)))
    for i, s0 in enumerate(sigma0_vals):
        for j, c in enumerate(c_vals):
            match = [r for r in synth_results if r["sigma0"] == s0 and r["c"] == c]
            if match:
                ratio_data[i, j] = match[0]["c_ratio"]

    im = ax.imshow(ratio_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels(c_vals)
    ax.set_yticks(range(len(sigma0_vals)))
    ax.set_yticklabels(sigma0_vals)
    ax.set_xlabel("True c", fontsize=12)
    ax.set_ylabel("σ₀", fontsize=12)
    ax.set_title("c_fit / c_true (1.0 = perfect recovery)", fontsize=12)
    for i in range(len(sigma0_vals)):
        for j in range(len(c_vals)):
            ax.text(j, i, f"{ratio_data[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Right: c > 0 pass/fail
    ax2 = axes[1]
    pass_data = np.zeros((len(sigma0_vals), len(c_vals)))
    for i, s0 in enumerate(sigma0_vals):
        for j, c in enumerate(c_vals):
            match = [r for r in synth_results if r["sigma0"] == s0 and r["c"] == c]
            if match:
                pass_data[i, j] = 1 if match[0]["c_positive"] else 0

    im2 = ax2.imshow(pass_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(c_vals)))
    ax2.set_xticklabels(c_vals)
    ax2.set_yticks(range(len(sigma0_vals)))
    ax2.set_yticklabels(sigma0_vals)
    ax2.set_xlabel("True c", fontsize=12)
    ax2.set_ylabel("σ₀", fontsize=12)
    ax2.set_title("c_fit > 0? (green=pass, red=fail)", fontsize=12)
    for i in range(len(sigma0_vals)):
        for j in range(len(c_vals)):
            mark = "✓" if pass_data[i, j] > 0.5 else "✗"
            ax2.text(j, i, mark, ha="center", va="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig6c_synth_param_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    #  Summary Table
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Cross-model comparison at temp=1.0, L=200
    print("\n── Cross-Model × Strategy (T=1.0, L=200) ──")
    print(f"{'Model':<20} {'Strategy':<18} {'D-2':>8} {'Rep':>8} {'Tri Rep':>8} {'Vocab':>8}")
    print("-" * 80)
    for model_name, m_results in all_decoding_results.items():
        key = "temp_1.0"
        for sn in strategies:
            r = m_results[key][sn]
            print(f"{model_name:<20} {sn:<18} {r['distinct-2']:>8.4f} "
                  f"{r['rep_rate']:>8.4f} {r['tri_rep']:>8.4f} {r['vocab_richness']:>8.4f}")

    # Check: does ν-sampling win on BOTH models?
    for model_name, m_results in all_decoding_results.items():
        key = "temp_1.0"
        d2_vals = {sn: m_results[key][sn]["distinct-2"] for sn in strategies}
        best = max(d2_vals, key=d2_vals.get)
        print(f"\n  {model_name}: Best D-2 = {best} ({d2_vals[best]:.4f})")

    # Synthetic channel robustness
    n_pass = sum(1 for r in synth_results if r["c_positive"])
    print(f"\n── Synthetic Channel Robustness ──")
    print(f"  c > 0 in {n_pass}/{len(synth_results)} configs "
          f"({100*n_pass/len(synth_results):.0f}%)")

    # ── Save ──
    save_data = {
        "decoding": {
            model: {
                cond: {
                    strat: metrics
                    for strat, metrics in conds.items()
                }
                for cond, conds in m_results.items()
            }
            for model, m_results in all_decoding_results.items()
        },
        "synthetic_ablation": synth_results,
        "n_synth_pass": n_pass,
        "n_synth_total": len(synth_results),
    }
    with open(f"{args.output_dir}/exp6_ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp6] Results saved to {args.output_dir}/exp6_ablation_results.json")
    print("[exp6] Done!")


if __name__ == "__main__":
    main()
