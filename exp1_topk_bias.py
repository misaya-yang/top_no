#!/usr/bin/env python3
"""
Experiment 1: Top-K Bias Verification + K* Scaling
====================================================
Tests Theorem X.1: Top-K reference with bias, optimal K*, and the
falsification prediction that K* ∝ σ²/a² (log-log slope = 2 for σ).

Sections:
  A. Real-model bias vs corrected theory  σ√(2·ln(eV/K))
  B. Zipf slope estimation from head logits
  C. Synthetic Zipf+Gaussian K* sweep  (Protocol A.1)
  D. Coverage probability  (Protocol A.1)
  E. Falsification: log-log K* vs σ slope  (Protocol A.3)
  F. Correlated noise rank ablation  (Reviewer-2 attack #1)
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from data_utils import load_text_samples, tokenize_batch, load_model_and_tokenizer, free_model

K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Theorem X.1: corrected theory ──

def theory_bias(v, k):
    """Selection bias upper bound: σ√(2·ln(e·V/K))"""
    return np.sqrt(2 * np.log(np.e * v / k))


def theory_bias_lower(v, k):
    """Asymptotic bias (tied configuration): σ√(2·ln(V/K))"""
    return np.sqrt(2 * np.log(v / k))


def W_objective(K, v, sigma, a, delta):
    """
    Full W(K,δ) from Corollary 2:
    W = σ√(2·ln(eV/K)) + a·ln(K) + 2σ√(2·ln(1/δ)/K)
    """
    K = max(K, 1)
    bias = sigma * np.sqrt(2 * np.log(np.e * v / K))
    hetero = a * np.log(K)
    fluct = 2 * sigma * np.sqrt(2 * np.log(1 / delta) / K)
    return bias + hetero + fluct


def compute_Kstar_theory(v, sigma, a, delta):
    """
    K₀ = 2σ²·ln(1/δ) / a²  and  corrected K*.
    Returns (K0, Kstar, Kstar_corr).
    """
    K0 = 2 * sigma**2 * np.log(1 / delta) / a**2
    K0 = max(K0, 1)
    denom = a * np.sqrt(2 * np.log(np.e * v / K0))
    if denom > sigma:
        Kstar_corr = K0 * (1 - sigma / denom) ** (-2)
    else:
        Kstar_corr = K0
    # Numerical minimizer of W over K ∈ [1, V]
    res = minimize_scalar(lambda k: W_objective(k, v, sigma, a, delta),
                          bounds=(1, v), method="bounded")
    Kstar_num = res.x
    return K0, Kstar_num, Kstar_corr


# ══════════════════════════════════════════════════════════════
#  Section A: Real-model bias vs corrected theory
# ══════════════════════════════════════════════════════════════

def section_a_real_bias(model, tokenizer, args, device):
    """Measure Top-K bias on real data, compare to corrected theory."""
    print("\n═══ Section A: Real-Model Top-K Bias ═══")

    texts = load_text_samples(args.n_samples, max_length=4096, seed=args.seed)
    input_ids, attn_mask = tokenize_batch(tokenizer, texts, args.max_length)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    N, T = input_ids.shape
    vocab_size = model.config.vocab_size
    print(f"  N={N}, T={T}, V={vocab_size}")

    # Accumulators
    bias_sum = {k: torch.zeros(T, dtype=torch.float32, device=device) for k in K_VALUES}
    bias_sq_sum = {k: torch.zeros(T, dtype=torch.float32, device=device) for k in K_VALUES}
    sigma_sum = torch.zeros(T, dtype=torch.float32, device=device)
    count = torch.zeros(T, dtype=torch.float32, device=device)

    # Also collect top-50 logits per position for Zipf fitting
    top50_values = []
    top50_sample_count = 0

    t0 = time.time()
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        ids = input_ids[start:end]
        mask = attn_mask[start:end]

        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits

        logits_f32 = logits.float()
        sigma = logits_f32.std(dim=-1)
        s_max = logits_f32.max(dim=-1).values

        for k in K_VALUES:
            actual_k = min(k, vocab_size)
            topk_vals = logits_f32.topk(actual_k, dim=-1).values
            s_topk_mean = topk_vals.mean(dim=-1)
            bias = s_topk_mean - s_max
            bias_sum[k] += bias.sum(dim=0)
            bias_sq_sum[k] += (bias ** 2).sum(dim=0)

        sigma_sum += sigma.sum(dim=0)
        count += mask.float().sum(dim=0)

        # Collect top-50 for Zipf fitting (first 200 samples)
        if top50_sample_count < 200:
            take = min(200 - top50_sample_count, end - start)
            top50 = logits_f32[:take].topk(min(50, vocab_size), dim=-1).values
            top50_values.append(top50.cpu())
            top50_sample_count += take

        if (start // args.batch_size) % 50 == 0:
            print(f"  [{100*start/N:.0f}%] {time.time()-t0:.1f}s")

    print(f"  Inference done in {time.time()-t0:.1f}s")

    valid = count > 0

    # Compute normalized bias per K
    results_a = {}
    print(f"\n{'K':>6} {'E[bias/σ]':>12} {'theory+e':>10} {'theory':>10} {'MSE(+e)':>10}")
    for k in K_VALUES:
        mean_bias = bias_sum[k][valid] / count[valid]
        mean_sigma = sigma_sum[valid] / count[valid]
        norm_bias = mean_bias / (mean_sigma + 1e-8)
        avg = norm_bias.mean().item()
        std = norm_bias.std().item()

        theo_corrected = -theory_bias(vocab_size, k)
        theo_naive = -theory_bias_lower(vocab_size, k)

        results_a[k] = {
            "empirical": avg, "std": std,
            "theory_corrected": theo_corrected,
            "theory_naive": theo_naive,
            "mse_corrected": (avg - theo_corrected) ** 2,
        }
        print(f"{k:6d} {avg:12.4f} {theo_corrected:10.4f} {theo_naive:10.4f} "
              f"{(avg - theo_corrected)**2:10.4f}")

    # ── Optimal K* (empirical MSE) ──
    best_k_emp, best_mse_emp = None, float("inf")
    for k in K_VALUES:
        mean_bias_pos = bias_sum[k][valid] / count[valid]
        mean_sigma_pos = sigma_sum[valid] / count[valid]
        bias_sq = ((mean_bias_pos / (mean_sigma_pos + 1e-8)) ** 2).mean().item()
        e_bias_sq = bias_sq_sum[k][valid] / count[valid]
        var_per_pos = e_bias_sq - mean_bias_pos ** 2
        variance = (var_per_pos / (mean_sigma_pos ** 2 + 1e-8)).mean().item()
        mse = bias_sq + variance
        if mse < best_mse_emp:
            best_mse_emp = mse
            best_k_emp = k

    print(f"\n  ★ Empirical K* = {best_k_emp}  (MSE = {best_mse_emp:.6f})")

    # Plot A
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ks = list(results_a.keys())
    emp = [results_a[k]["empirical"] for k in ks]
    emp_std = [results_a[k]["std"] for k in ks]

    k_range = np.logspace(np.log10(1), np.log10(100), 200)
    theo_curve_corrected = -np.sqrt(2 * np.log(np.e * vocab_size / k_range))
    theo_curve_naive = -np.sqrt(2 * np.log(vocab_size / k_range))

    ax.errorbar(ks, emp, yerr=emp_std, fmt="o-", color="crimson",
                capsize=4, markersize=8, lw=2, label="Empirical E[bias/σ]")
    ax.plot(k_range, theo_curve_corrected, "--", color="steelblue", lw=2,
            label=r"Theorem 1: $-\sqrt{2\ln(eV/K)}$")
    ax.plot(k_range, theo_curve_naive, ":", color="gray", lw=1.5,
            label=r"Naive: $-\sqrt{2\ln(V/K)}$ (missing $e$)")
    ax.set_xscale("log")
    ax.set_xlabel("K", fontsize=13)
    ax.set_ylabel("Normalized Bias (bias / σ)", fontsize=13)
    ax.set_title(f"Top-K Bias vs Corrected Theory (V={vocab_size})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: MSE comparison
    ax2 = axes[1]
    mses_corr = [results_a[k]["mse_corrected"] for k in ks]
    mses_naive = [(results_a[k]["empirical"] - results_a[k]["theory_naive"])**2
                  for k in ks]
    x_pos = np.arange(len(ks))
    ax2.bar(x_pos - 0.2, mses_corr, 0.4, color="steelblue", label="Corrected (+e)")
    ax2.bar(x_pos + 0.2, mses_naive, 0.4, color="gray", alpha=0.5, label="Naive")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("K", fontsize=13)
    ax2.set_ylabel("MSE (empirical − theory)²", fontsize=13)
    ax2.set_title("Theory Fit: Corrected vs Naive", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig1a_topk_bias.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results_a, best_k_emp, top50_values, vocab_size


# ══════════════════════════════════════════════════════════════
#  Section B: Zipf slope estimation
# ══════════════════════════════════════════════════════════════

def section_b_zipf_fit(top50_values, vocab_size):
    """Estimate Zipf slope a from top-K logit gaps."""
    print("\n═══ Section B: Zipf Slope Estimation ═══")

    all_top50 = torch.cat(top50_values, dim=0)  # (n_samples*T, 50)
    # Average across all positions to get mean top-K values
    mean_topk = all_top50.mean(dim=0).numpy().flatten()  # (50,)
    K_fit = len(mean_topk)
    print(f"  mean_topk shape: {mean_topk.shape}, K_fit={K_fit}")

    # Estimate a: q_{(1)} - q_{(k)} ≈ a·ln(k)
    ranks = np.arange(1, K_fit + 1, dtype=np.float64)
    gaps = (mean_topk[0] - mean_topk).astype(np.float64)  # ≥ 0
    log_k = np.log(ranks)

    # OLS fit: gaps = a * ln(k) + intercept
    slope, intercept = np.polyfit(log_k, gaps, 1)
    a_hat = float(slope)
    intercept = float(intercept)

    print(f"  Estimated Zipf slope: a = {a_hat:.4f}")
    print(f"  (q_{{(1)}} - q_{{(k)}} ≈ {a_hat:.3f} · ln(k) + {intercept:.3f})")

    # Also estimate σ (average logit std)
    sigma_hat = all_top50.std(dim=-1).mean().item()
    print(f"  Estimated σ (logit std): {sigma_hat:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, gaps, "o", color="crimson", markersize=4, label="Empirical gaps")
    ax.plot(ranks, a_hat * log_k + intercept, "--", color="steelblue", lw=2,
            label=f"Fit: a={a_hat:.3f}")
    ax.set_xlabel("Rank k", fontsize=12)
    ax.set_ylabel(r"$q_{(1)} - q_{(k)}$", fontsize=12)
    ax.set_title(f"Zipf Envelope Fit (â={a_hat:.3f})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args_global.output_dir}/fig1b_zipf_fit.png", dpi=150, bbox_inches="tight")
    plt.close()

    return a_hat, sigma_hat


# ══════════════════════════════════════════════════════════════
#  Section C: Synthetic K* sweep (Protocol A.1)
# ══════════════════════════════════════════════════════════════

def section_c_synthetic_kstar(vocab_size_model, a_hat, sigma_hat):
    """
    Synthetic Zipf + Gaussian: sweep (a, σ), measure empirical K*,
    compare to theoretical K* = 2σ²ln(1/δ)/a².
    Protocol A.1 specifies v=32,000 for synthetic experiments.
    """
    print("\n═══ Section C: Synthetic K* Sweep ═══")

    v = 32000  # Protocol A.1: v = 32,000
    print(f"  Synthetic vocab v={v} (Protocol A.1)")
    a_values = [0.5, 1.0, 2.0]
    sigma_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    delta = 0.05
    n_trials = 5000

    results = []

    for a in a_values:
        for sigma in sigma_values:
            # Generate Zipf true scores
            ranks = np.arange(1, v + 1)
            q = -a * np.log(ranks)

            # Monte Carlo trials
            best_ks = []
            for _ in range(n_trials):
                noise = sigma * np.random.randn(v)
                s = q + noise
                s_sorted = np.sort(s)[::-1]
                q_max = q[0]

                cum_s = np.cumsum(s_sorted)
                k_range = np.arange(1, min(101, v + 1))
                topk_means = cum_s[k_range - 1] / k_range
                errors = np.abs(topk_means - q_max)
                best_k = k_range[np.argmin(errors)]
                best_ks.append(best_k)

            emp_kstar = np.median(best_ks)
            K0, theo_kstar, theo_kstar_corr = compute_Kstar_theory(
                v, sigma, a, delta
            )

            results.append({
                "a": a, "sigma": sigma,
                "empirical_kstar": float(emp_kstar),
                "K0": float(K0),
                "theory_kstar": float(theo_kstar),
                "theory_kstar_corr": float(theo_kstar_corr),
                "ratio": float(emp_kstar / max(theo_kstar, 1)),
            })
            print(f"  a={a:.1f} σ={sigma:.2f}: "
                  f"emp={emp_kstar:.0f}  theo={theo_kstar:.1f}  "
                  f"corr={theo_kstar_corr:.1f}  ratio={emp_kstar/max(theo_kstar,1):.2f}")

    # Plot: K* heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: log-log K* vs σ for each a
    ax = axes[0]
    for a in a_values:
        subset = [r for r in results if r["a"] == a]
        sigmas = [r["sigma"] for r in subset]
        emp_ks = [r["empirical_kstar"] for r in subset]
        theo_ks = [r["theory_kstar"] for r in subset]

        ax.loglog(sigmas, emp_ks, "o-", label=f"a={a} empirical", markersize=7)
        ax.loglog(sigmas, theo_ks, "--", alpha=0.5, label=f"a={a} theory")

        # Fit log-log slope
        log_sig = np.log(sigmas)
        log_k = np.log(np.maximum(emp_ks, 1))
        if len(log_sig) > 1:
            slope = np.polyfit(log_sig, log_k, 1)[0]
            print(f"  Log-log slope for a={a}: {slope:.2f} (target: 2.0)")

    ax.set_xlabel("σ", fontsize=13)
    ax.set_ylabel("K*", fontsize=13)
    ax.set_title("Falsification: K* ∝ σ² (slope=2 on log-log)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Right: ratio empirical/theoretical
    ax2 = axes[1]
    for a in a_values:
        subset = [r for r in results if r["a"] == a]
        sigmas = [r["sigma"] for r in subset]
        ratios = [r["ratio"] for r in subset]
        ax2.plot(sigmas, ratios, "o-", label=f"a={a}", markersize=7)

    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Target ratio=1")
    ax2.axhspan(0.5, 2.0, alpha=0.1, color="green", label="Acceptable [0.5, 2.0]")
    ax2.set_xlabel("σ", fontsize=13)
    ax2.set_ylabel("Empirical K* / Theoretical K*", fontsize=13)
    ax2.set_title("K* Prediction Accuracy (Protocol A.1 target)", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args_global.output_dir}/fig1c_synthetic_kstar.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════
#  Section D: Coverage probability (Protocol A.1)
# ══════════════════════════════════════════════════════════════

def section_d_coverage(vocab_size):
    """
    Verify Theorem 1 interval coverage on synthetic data.
    For each trial, check if s̄_(K) falls within the theorem's interval.
    """
    print("\n═══ Section D: Coverage Probability ═══")

    a, sigma = 1.0, 1.0
    delta_values = [0.01, 0.05, 0.10, 0.20]
    n_trials = 10000
    ranks = np.arange(1, vocab_size + 1)
    q = -a * np.log(ranks)
    q_max = q[0]
    K_test = 10
    Delta_K = a * np.log(K_test)  # Zipf envelope bound for K=10

    coverage_results = {}

    for delta in delta_values:
        covered_count = 0
        for _ in range(n_trials):
            noise = sigma * np.random.randn(vocab_size)
            s = q + noise
            s_sorted = np.sort(s)[::-1]
            s_topk_mean = s_sorted[:K_test].mean()

            # Theorem 1 interval:
            # q_max - Δ_K - σ√(2ln(1/δ)/K) ≤ s̄_(K) ≤ q_max + σ√(2ln(eV/K)) + σ√(2ln(1/δ)/K)
            lower = q_max - Delta_K - sigma * np.sqrt(2 * np.log(1/delta) / K_test)
            upper = q_max + sigma * np.sqrt(2 * np.log(np.e * vocab_size / K_test)) + \
                    sigma * np.sqrt(2 * np.log(1/delta) / K_test)

            if lower <= s_topk_mean <= upper:
                covered_count += 1

        cov = covered_count / n_trials
        target = 1 - 2 * delta
        coverage_results[delta] = {"coverage": cov, "target": target}
        print(f"  δ={delta:.2f}: coverage={cov:.4f}  target≥{target:.2f}  "
              f"{'✓' if cov >= target else '✗'}")

    return coverage_results


# ══════════════════════════════════════════════════════════════
#  Section E: Falsification test (Protocol A.3)
# ══════════════════════════════════════════════════════════════

def section_e_falsification(synthetic_results):
    """
    Protocol A.3: If empirical K* doesn't scale as σ²/a²
    (log-log slope ≠ 2 for σ at fixed a), Theorem X.1(b) is rejected.
    """
    print("\n═══ Section E: Falsification Test ═══")

    a_values = sorted(set(r["a"] for r in synthetic_results))
    all_slopes = []

    for a in a_values:
        subset = [r for r in synthetic_results if r["a"] == a]
        sigmas = np.array([r["sigma"] for r in subset])
        emp_ks = np.array([r["empirical_kstar"] for r in subset])

        # Filter out K*=1 (floor effect)
        valid = emp_ks > 1
        if valid.sum() < 2:
            continue

        log_sig = np.log(sigmas[valid])
        log_k = np.log(emp_ks[valid])
        slope, intercept = np.polyfit(log_sig, log_k, 1)
        all_slopes.append(slope)

        # 95% CI via bootstrap
        n_boot = 1000
        boot_slopes = []
        n_pts = len(log_sig)
        for _ in range(n_boot):
            idx = np.random.choice(n_pts, n_pts, replace=True)
            if len(np.unique(idx)) < 2:
                continue
            s, _ = np.polyfit(log_sig[idx], log_k[idx], 1)
            boot_slopes.append(s)
        ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

        rejected = not (ci_low <= 2.0 <= ci_high)
        print(f"  a={a:.1f}: slope={slope:.3f}  "
              f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]  "
              f"{'REJECTED' if rejected else 'not rejected'}")

    mean_slope = np.mean(all_slopes) if all_slopes else 0
    verdict = "PASS" if abs(mean_slope - 2.0) < 0.5 else "FAIL"
    print(f"\n  Mean slope across a values: {mean_slope:.3f} → {verdict}")
    return all_slopes


# ══════════════════════════════════════════════════════════════
#  Section F: Correlated noise rank ablation
# ══════════════════════════════════════════════════════════════

def section_f_correlated_noise(vocab_size):
    """
    Reviewer-2 attack #1: test bias under low-rank correlated noise.
    ε = W·δ where W is V×r, δ is r-dim Gaussian.
    """
    print("\n═══ Section F: Correlated Noise Rank Ablation ═══")

    sigma = 1.0
    a = 1.0
    ranks = np.arange(1, vocab_size + 1)
    q = -a * np.log(ranks)
    n_trials = 5000
    rank_values = [1, 5, 20, 100, 500, vocab_size]

    results = []
    for r in rank_values:
        # Generate projection matrix W: V×r, orthonormal columns
        W = np.random.randn(vocab_size, r)
        W, _ = np.linalg.qr(W)
        W *= np.sqrt(vocab_size / r)  # normalize so ‖W‖_F² = V

        biases = []
        for _ in range(n_trials):
            delta = sigma * np.random.randn(r)
            noise = W @ delta / np.sqrt(vocab_size)  # normalize
            s = q + noise
            s_sorted = np.sort(s)[::-1]
            # Top-K bias for K=10
            K = 10
            topk_mean = s_sorted[:K].mean()
            s_max = s_sorted[0]
            biases.append(topk_mean - s_max)

        avg_bias = np.mean(biases)
        theory_iid = -sigma * np.sqrt(2 * np.log(np.e * vocab_size / 10))
        results.append({
            "rank": r,
            "avg_bias": float(avg_bias),
            "theory_iid": float(theory_iid),
            "ratio": float(avg_bias / theory_iid) if theory_iid != 0 else 0,
        })
        print(f"  rank={r:6d}: bias={avg_bias:.4f}  "
              f"theory_iid={theory_iid:.4f}  ratio={avg_bias/theory_iid:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks_plot = [r["rank"] for r in results]
    biases_plot = [abs(r["avg_bias"]) for r in results]
    theory_line = abs(results[0]["theory_iid"])

    ax.semilogx(ranks_plot, biases_plot, "o-", color="crimson", markersize=8, lw=2,
                label="|Empirical bias|")
    ax.axhline(theory_line, color="steelblue", linestyle="--", lw=2,
               label=f"IID theory: {theory_line:.3f}")
    ax.set_xlabel("Noise rank r", fontsize=13)
    ax.set_ylabel("|Bias| (Top-K mean − max)", fontsize=13)
    ax.set_title("Correlated Noise Ablation (Reviewer-2 Attack #1)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args_global.output_dir}/fig1f_correlated_noise.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    global args_global
    args = parse_args()
    args_global = args
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)
    vocab_size = model.config.vocab_size
    print(f"[exp1] V={vocab_size}")

    # A: Real-model bias
    results_a, best_k_emp, top50_values, V = section_a_real_bias(
        model, tokenizer, args, device
    )

    # B: Zipf slope
    a_hat, sigma_hat = section_b_zipf_fit(top50_values, vocab_size)

    # Free model before synthetic experiments
    free_model(model)

    # C: Synthetic K* sweep
    synthetic_results = section_c_synthetic_kstar(vocab_size, a_hat, sigma_hat)

    # D: Coverage probability
    coverage = section_d_coverage(vocab_size)

    # E: Falsification test
    slopes = section_e_falsification(synthetic_results)

    # F: Correlated noise
    corr_results = section_f_correlated_noise(vocab_size)

    # ── Compute theory K* with estimated parameters ──
    delta = 0.05
    K0, Kstar, Kstar_corr = compute_Kstar_theory(vocab_size, sigma_hat, a_hat, delta)
    print(f"\n═══ Summary ═══")
    print(f"  Estimated: a={a_hat:.3f}, σ={sigma_hat:.3f}")
    print(f"  Theory K₀ = {K0:.1f}")
    print(f"  Theory K* (numerical) = {Kstar:.1f}")
    print(f"  Theory K* (corrected) = {Kstar_corr:.1f}")
    print(f"  Empirical K* = {best_k_emp}")

    # ── Save all results ──
    save_data = {
        "vocab_size": vocab_size,
        "model": args.model,
        "a_hat": a_hat,
        "sigma_hat": sigma_hat,
        "empirical_Kstar": best_k_emp,
        "theory_Kstar": float(Kstar),
        "theory_Kstar_corr": float(Kstar_corr),
        "K0": float(K0),
        "section_a": {str(k): v for k, v in results_a.items()},
        "section_c": synthetic_results,
        "section_d": {str(k): v for k, v in coverage.items()},
        "section_e_slopes": slopes,
        "section_f": corr_results,
    }
    with open(f"{args.output_dir}/exp1_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp1] All results saved to {args.output_dir}/exp1_results.json")
    print("[exp1] Done!")


if __name__ == "__main__":
    main()
