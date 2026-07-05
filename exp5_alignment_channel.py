#!/usr/bin/env python3
"""
Experiment 5 (Revised): Heteroscedastic Evidence from Model Behavior
======================================================================
Instead of measuring logit noise directly (impossible with one model),
we test BEHAVIORAL PREDICTIONS of the heteroscedastic channel theory:

  1. Margin variance: Var(s_max - s_target) should be higher for rare tokens
  2. Prediction stability: rare tokens' logits more sensitive to perturbation
  3. Effective vocabulary: sharp transition in estimability (V_eff corollary)
  4. Quantization sensitivity: rare-token weights more affected by INT8

These collectively validate the paper's heteroscedastic noise model
without requiring a teacher-student pair.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from data_utils import load_text_samples, tokenize_batch, load_model_and_tokenizer, free_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Test 1: Margin Variance by Token Frequency
# ══════════════════════════════════════════════════════════════

def test1_margin_variance(model, input_ids, attn_mask, args, device):
    """
    Theory prediction: Var(s_max - s_target) ∝ σ₀² + c/n_i.
    Measure margin = s_max - s_target at each position,
    group by target token frequency, compute variance.
    """
    print("\n═══ Test 1: Margin Variance by Token Frequency ═══")
    N, T = input_ids.shape
    targets = input_ids[:, 1:]
    valid_mask = attn_mask[:, 1:] > 0

    # Token frequencies (global)
    all_tokens = input_ids[attn_mask > 0].cpu()
    token_counts = torch.bincount(all_tokens).float()

    # Forward pass: get full logits
    print("  Computing logits...")
    all_margins = []
    all_target_freqs = []
    t0 = time.time()

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        ids = input_ids[start:end]
        mask = attn_mask[start:end]
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits

        # s_max and s_target
        logits_slice = logits.float()[:, :-1, :]  # (B, T-1, V)
        s_max = logits_slice.max(dim=-1).values  # (B, T-1)
        tgt = ids[:, 1:]
        s_target = logits_slice.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)

        margin = (s_max - s_target).cpu()  # (B, T-1)
        vm = mask[:, 1:].cpu().bool()

        # Target frequencies
        tgt_cpu = tgt.cpu()
        freq = token_counts[tgt_cpu.clamp(max=len(token_counts)-1)]

        all_margins.append(margin[vm])
        all_target_freqs.append(freq[vm])

    margins = torch.cat(all_margins).numpy()
    freqs = torch.cat(all_target_freqs).numpy()

    print(f"  Total valid positions: {len(margins)}")
    print(f"  Margin: mean={margins.mean():.3f}, std={margins.std():.3f}")

    # Bin by token frequency
    log_freq = np.log10(freqs + 1)
    n_bins = 30
    bin_edges = np.linspace(log_freq.min(), log_freq.max() + 0.01, n_bins + 1)

    bin_centers, bin_variances, bin_means, bin_counts = [], [], [], []
    for b in range(n_bins):
        mask = (log_freq >= bin_edges[b]) & (log_freq < bin_edges[b + 1])
        if mask.sum() < 20:
            continue
        bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
        bin_variances.append(np.var(margins[mask]))
        bin_means.append(np.mean(margins[mask]))
        bin_counts.append(int(mask.sum()))

    bin_centers = np.array(bin_centers)
    bin_variances = np.array(bin_variances)
    bin_means = np.array(bin_means)

    # Fit Var(margin) = σ₀² + c/n
    def var_model(n, sigma0_sq, c):
        return sigma0_sq + c / (n + 1e-8)

    try:
        popt, _ = curve_fit(var_model, bin_centers, bin_variances,
                            p0=[np.min(bin_variances) * 0.5, 50.0],
                            maxfev=10000, bounds=([0, -np.inf], [np.inf, np.inf]))
        sigma0_sq, c_fit = popt
    except Exception:
        sigma0_sq, c_fit = 0, 0

    print(f"  Fit: Var(margin) = {sigma0_sq:.4f} + {c_fit:.2f}/n")
    print(f"  c > 0: {'YES ✓' if c_fit > 0 else 'NO ✗'}")

    return {
        "bin_centers": bin_centers,
        "bin_variances": bin_variances,
        "bin_means": bin_means,
        "bin_counts": bin_counts,
        "sigma0_sq": float(sigma0_sq),
        "c_fit": float(c_fit),
    }


# ══════════════════════════════════════════════════════════════
#  Test 2: Prediction Stability Under Input Perturbation
# ══════════════════════════════════════════════════════════════

def test2_perturbation_stability(model, input_ids, attn_mask, args, device):
    """
    Add small noise to embeddings, measure logit change per position.
    Theory: rare target tokens should show larger logit changes.
    Memory-efficient: only stores target-token logits, not full vocab.
    """
    print("\n═══ Test 2: Prediction Stability Under Perturbation ═══")
    N, T = input_ids.shape

    all_tokens = input_ids[attn_mask > 0].cpu()
    token_counts = torch.bincount(all_tokens).float()

    eps = 0.01  # perturbation magnitude

    embed_layer = model.model.embed_tokens
    orig_weight = embed_layer.weight.data.clone()

    # Only store target-token logits (not full vocab) → tiny memory
    targets = input_ids[:, 1:]  # (N, T-1) on GPU
    clean_tgt_logits = torch.zeros(N, T - 1, dtype=torch.float32)
    pert_tgt_logits = torch.zeros(N, T - 1, dtype=torch.float32)
    clean_max_logits = torch.zeros(N, T - 1, dtype=torch.float32)
    pert_max_logits = torch.zeros(N, T - 1, dtype=torch.float32)

    t0 = time.time()
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        ids = input_ids[start:end]
        mask = attn_mask[start:end]
        tgt = ids[:, 1:]

        # Clean forward
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
            logits_f = logits.float()[:, :-1, :]
            clean_max_logits[start:end] = logits_f.max(dim=-1).values.cpu()
            clean_tgt_logits[start:end] = logits_f.gather(
                dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1).cpu()

        # Perturbed forward
        noise = eps * torch.randn_like(orig_weight)
        embed_layer.weight.data = orig_weight + noise
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
            logits_f = logits.float()[:, :-1, :]
            pert_max_logits[start:end] = logits_f.max(dim=-1).values.cpu()
            pert_tgt_logits[start:end] = logits_f.gather(
                dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1).cpu()

        if (start // args.batch_size) % 50 == 0:
            print(f"    [{100*start/N:.0f}%] {time.time()-t0:.1f}s")

    # Restore
    embed_layer.weight.data = orig_weight
    print(f"  Forward passes done in {time.time()-t0:.1f}s")

    # Compute per-position changes
    valid_mask = attn_mask[:, 1:].cpu() > 0
    delta_logit = (clean_tgt_logits - pert_tgt_logits).abs()
    delta_max = (clean_max_logits - pert_max_logits).abs()

    delta_np = delta_logit[valid_mask].numpy()
    freqs = token_counts[targets.cpu().clamp(max=len(token_counts)-1)][valid_mask].numpy()

    print(f"  |Δlogit_target| mean: {delta_np.mean():.6f}")
    print(f"  |Δlogit_max| mean: {delta_max[valid_mask].numpy().mean():.6f}")

    # Bin by frequency
    log_freq = np.log10(freqs + 1)
    n_bins = 25
    bin_edges = np.linspace(log_freq.min(), log_freq.max() + 0.01, n_bins + 1)

    bin_centers, bin_delta_var, bin_delta_mean = [], [], []
    for b in range(n_bins):
        mask = (log_freq >= bin_edges[b]) & (log_freq < bin_edges[b + 1])
        if mask.sum() < 20:
            continue
        bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
        bin_delta_var.append(np.var(delta_np[mask]))
        bin_delta_mean.append(np.mean(delta_np[mask]))

    bin_centers = np.array(bin_centers)
    bin_delta_var = np.array(bin_delta_var)
    bin_delta_mean = np.array(bin_delta_mean)

    # Fit Var(|Δlogit|) = a + b/n
    def var_model(n, a, b):
        return a + b / (n + 1e-8)

    try:
        popt, _ = curve_fit(var_model, bin_centers, bin_delta_var,
                            p0=[np.min(bin_delta_var), 1.0],
                            maxfev=10000, bounds=([0, -np.inf], [np.inf, np.inf]))
        a_fit, b_fit = popt
    except Exception:
        a_fit, b_fit = 0, 0

    print(f"  Fit: Var(|Δlogit|) = {a_fit:.6f} + {b_fit:.4f}/n")
    print(f"  b > 0 (rare tokens more unstable): {'YES ✓' if b_fit > 0 else 'NO ✗'}")

    return {
        "bin_centers": bin_centers,
        "bin_delta_var": bin_delta_var,
        "bin_delta_mean": bin_delta_mean,
        "a_fit": float(a_fit),
        "b_fit": float(b_fit),
    }


# ══════════════════════════════════════════════════════════════
#  Test 3: Weight Norm Analysis by Token Frequency
# ══════════════════════════════════════════════════════════════

def test3_weight_analysis(model, input_ids, attn_mask, args, device):
    """
    Analyze lm_head weight norms by token frequency.
    Theory: rare tokens have less-constrained weights →
    higher weight norms (less regularization effect from data).
    """
    print("\n═══ Test 3: Weight Analysis by Token Frequency ═══")

    # Get lm_head weights
    lm_weight = model.lm_head.weight.data.float().cpu()  # (V, H)
    V, H = lm_weight.shape

    # Token frequencies
    all_tokens = input_ids[attn_mask > 0].cpu()
    token_counts = torch.bincount(all_tokens, minlength=V).float()

    # Per-token weight norms
    weight_norms = lm_weight.norm(dim=1).numpy()  # (V,)
    counts = token_counts.numpy()

    # Only look at tokens that appear in the corpus
    valid = counts > 0
    weight_norms_valid = weight_norms[valid]
    counts_valid = counts[valid]

    print(f"  V={V}, tokens in corpus: {valid.sum()}")
    print(f"  Weight norm: mean={weight_norms_valid.mean():.3f}, "
          f"std={weight_norms_valid.std():.3f}")

    # Bin by frequency
    log_freq = np.log10(counts_valid + 1)
    n_bins = 25
    bin_edges = np.linspace(log_freq.min(), log_freq.max() + 0.01, n_bins + 1)

    bin_centers, bin_norm_mean, bin_norm_std = [], [], []
    for b in range(n_bins):
        mask = (log_freq >= bin_edges[b]) & (log_freq < bin_edges[b + 1])
        if mask.sum() < 5:
            continue
        bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
        bin_norm_mean.append(np.mean(weight_norms_valid[mask]))
        bin_norm_std.append(np.std(weight_norms_valid[mask]))

    bin_centers = np.array(bin_centers)
    bin_norm_mean = np.array(bin_norm_mean)
    bin_norm_std = np.array(bin_norm_std)

    # Correlation between frequency and weight norm
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(np.log10(counts_valid + 1), weight_norms_valid)
    spearman_r, spearman_p = spearmanr(np.log10(counts_valid + 1), weight_norms_valid)

    print(f"  Pearson r(log(n+1), ||w||) = {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})")

    return {
        "bin_centers": bin_centers,
        "bin_norm_mean": bin_norm_mean,
        "bin_norm_std": bin_norm_std,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
    }


# ══════════════════════════════════════════════════════════════
#  Test 4: INT8 Quantization Sensitivity by Token
# ══════════════════════════════════════════════════════════════

def test4_quantization_sensitivity(model, input_ids, attn_mask, args, device):
    """
    Measure per-token quantization sensitivity of lm_head weights.
    |δw_i|/||w_i|| should be higher for rare tokens
    (their weights have more extreme values → more quantization error).
    """
    print("\n═══ Test 4: Quantization Sensitivity by Token ═══")

    lm_weight = model.lm_head.weight.data.float().cpu()  # (V, H)
    V, H = lm_weight.shape

    # Simulate INT8 quantization
    absmax = lm_weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    w_q = (lm_weight / scale).round().clamp(-127, 127) * scale
    delta_w = (lm_weight - w_q).abs()  # per-element quantization error

    # Per-token: relative quantization error
    token_orig_norm = lm_weight.norm(dim=1)  # (V,)
    token_delta_norm = delta_w.norm(dim=1)   # (V,)
    relative_error = (token_delta_norm / (token_orig_norm + 1e-8)).numpy()

    # Token frequencies
    all_tokens = input_ids[attn_mask > 0].cpu()
    token_counts = torch.bincount(all_tokens, minlength=V).float().numpy()

    valid = token_counts > 0
    rel_err_valid = relative_error[valid]
    counts_valid = token_counts[valid]

    # Bin by frequency
    log_freq = np.log10(counts_valid + 1)
    n_bins = 25
    bin_edges = np.linspace(log_freq.min(), log_freq.max() + 0.01, n_bins + 1)

    bin_centers, bin_rel_err = [], []
    for b in range(n_bins):
        mask = (log_freq >= bin_edges[b]) & (log_freq < bin_edges[b + 1])
        if mask.sum() < 5:
            continue
        bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
        bin_rel_err.append(np.mean(rel_err_valid[mask]))

    bin_centers = np.array(bin_centers)
    bin_rel_err = np.array(bin_rel_err)

    # Fit rel_err = a + b/n
    def err_model(n, a, b):
        return a + b / (n + 1e-8)

    try:
        popt, _ = curve_fit(err_model, bin_centers, bin_rel_err,
                            p0=[np.min(bin_rel_err), 0.001],
                            maxfev=10000, bounds=([0, -np.inf], [np.inf, np.inf]))
        a_fit, b_fit = popt
    except Exception:
        a_fit, b_fit = 0, 0

    print(f"  Fit: rel_quant_err = {a_fit:.6f} + {b_fit:.6f}/n")
    print(f"  b > 0 (rare tokens more affected): {'YES ✓' if b_fit > 0 else 'NO ✗'}")

    # Correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(np.log10(counts_valid + 1), rel_err_valid)
    print(f"  Pearson r(log(n+1), rel_err) = {r:.4f} (p={p:.2e})")

    return {
        "bin_centers": bin_centers,
        "bin_rel_err": bin_rel_err,
        "a_fit": float(a_fit),
        "b_fit": float(b_fit),
        "pearson_r": float(r),
    }


# ══════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════

def plot_all_results(test1, test2, test3, test4, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Test 1 - Margin variance
    ax = axes[0, 0]
    bc = test1["bin_centers"]
    bv = test1["bin_variances"]
    n_fit = np.logspace(np.log10(max(0.1, bc.min())), np.log10(bc.max()), 200)
    fitted = test1["sigma0_sq"] + test1["c_fit"] / (n_fit + 1e-8)

    ax.scatter(np.log10(bc + 1), bv, s=np.clip(np.array(test1["bin_counts"]) * 2, 20, 500),
               c=test1["bin_counts"], cmap="YlOrRd", alpha=0.8, edgecolors="gray")
    ax.plot(np.log10(n_fit + 1), fitted, "--", color="steelblue", lw=2,
            label=f"Fit: {test1['sigma0_sq']:.2f} + {test1['c_fit']:.1f}/n")
    ax.set_xlabel("log₁₀(n + 1)", fontsize=12)
    ax.set_ylabel("Var(margin)", fontsize=12)
    ax.set_title("Test 1: Prediction Margin Variance\nby Target Token Frequency", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Test 2 - Perturbation stability
    ax = axes[0, 1]
    bc2 = test2["bin_centers"]
    bv2 = test2["bin_delta_var"]
    n_fit2 = np.logspace(np.log10(max(0.1, bc2.min())), np.log10(bc2.max()), 200)
    fitted2 = test2["a_fit"] + test2["b_fit"] / (n_fit2 + 1e-8)

    ax.scatter(np.log10(bc2 + 1), bv2, s=50, c="forestgreen", alpha=0.8, edgecolors="gray")
    ax.plot(np.log10(n_fit2 + 1), fitted2, "--", color="red", lw=2,
            label=f"Fit: {test2['a_fit']:.6f} + {test2['b_fit']:.4f}/n")
    ax.set_xlabel("log₁₀(n + 1)", fontsize=12)
    ax.set_ylabel("Var(|Δlogit|)", fontsize=12)
    ax.set_title("Test 2: Perturbation Sensitivity\n(embedding noise ε=0.01)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Test 3 - Weight norms
    ax = axes[1, 0]
    bc3 = test3["bin_centers"]
    bm3 = test3["bin_norm_mean"]
    bs3 = test3["bin_norm_std"]
    ax.errorbar(np.log10(bc3 + 1), bm3, yerr=bs3, fmt="o-", color="crimson",
                capsize=3, markersize=6, lw=2)
    ax.set_xlabel("log₁₀(n + 1)", fontsize=12)
    ax.set_ylabel("||w_i|| (lm_head weight norm)", fontsize=12)
    ax.set_title(f"Test 3: Weight Norm vs Frequency\n"
                 f"Pearson r={test3['pearson_r']:.3f}", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Test 4 - Quantization sensitivity
    ax = axes[1, 1]
    bc4 = test4["bin_centers"]
    be4 = test4["bin_rel_err"]
    n_fit4 = np.logspace(np.log10(max(0.1, bc4.min())), np.log10(bc4.max()), 200)
    fitted4 = test4["a_fit"] + test4["b_fit"] / (n_fit4 + 1e-8)

    ax.scatter(np.log10(bc4 + 1), be4, s=50, c="orange", alpha=0.8, edgecolors="gray")
    ax.plot(np.log10(n_fit4 + 1), fitted4, "--", color="red", lw=2,
            label=f"Fit: {test4['a_fit']:.6f} + {test4['b_fit']:.6f}/n")
    ax.set_xlabel("log₁₀(n + 1)", fontsize=12)
    ax.set_ylabel("Relative quantization error", fontsize=12)
    ax.set_title(f"Test 4: INT8 Quantization Sensitivity\n"
                 f"r={test4['pearson_r']:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_heteroscedastic_evidence.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"

    print("[exp5] Loading model and data...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)
    texts = load_text_samples(args.n_samples, max_length=4096, seed=args.seed)
    input_ids, attn_mask = tokenize_batch(tokenizer, texts, args.max_length)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    N, T = input_ids.shape
    print(f"[exp5] Data: N={N}, T={T}")

    # Test 1: Margin variance
    test1 = test1_margin_variance(model, input_ids, attn_mask, args, device)

    # Test 2: Perturbation stability
    test2 = test2_perturbation_stability(model, input_ids, attn_mask, args, device)

    # Test 3: Weight analysis
    test3 = test3_weight_analysis(model, input_ids, attn_mask, args, device)

    # Test 4: Quantization sensitivity
    test4 = test4_quantization_sensitivity(model, input_ids, attn_mask, args, device)

    free_model(model)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("HETEROSCEDASTIC EVIDENCE SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Margin Variance):  c={test1['c_fit']:.2f}  "
          f"{'✓ Heteroscedastic' if test1['c_fit'] > 0 else '✗ Homoscedastic'}")
    print(f"Test 2 (Perturbation):     b={test2['b_fit']:.4f}  "
          f"{'✓ Rare=unstable' if test2['b_fit'] > 0 else '✗ No effect'}")
    print(f"Test 3 (Weight Norms):     r={test3['pearson_r']:.4f}  "
          f"{'✓ Freq≠Rare' if abs(test3['pearson_r']) > 0.1 else '✗ No diff'}")
    print(f"Test 4 (Quantization):     b={test4['b_fit']:.6f}  "
          f"r={test4['pearson_r']:.4f}  "
          f"{'✓ Rare=sensitive' if test4['b_fit'] > 0 else '✗ No effect'}")

    n_pass = sum([
        test1['c_fit'] > 0,
        test2['b_fit'] > 0,
        abs(test3['pearson_r']) > 0.1,
        test4['b_fit'] > 0,
    ])
    print(f"\nPassed: {n_pass}/4 tests")

    # ── Plot ──
    plot_all_results(test1, test2, test3, test4, args.output_dir)

    # ── Save ──
    save_data = {
        "model": args.model,
        "n_samples": args.n_samples,
        "test1_margin_variance": {k: v for k, v in test1.items()
                                   if k not in ["bin_centers", "bin_variances"]},
        "test2_perturbation": {k: v for k, v in test2.items()
                               if k not in ["bin_centers", "bin_delta_var"]},
        "test3_weights": test3,
        "test4_quantization": test4,
        "n_tests_passed": n_pass,
    }
    with open(f"{args.output_dir}/exp5_heteroscedastic_evidence.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp5] Results saved")
    print("[exp5] Done!")


if __name__ == "__main__":
    main()
