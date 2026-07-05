#!/usr/bin/env python3
"""
GPU-accelerated synthetic experiments for Exp 1 (Sections C, D, E, F).
Protocol A.1: v = 32,000. All Monte Carlo trials batched on GPU.
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── Import theory functions from exp1 ──
import sys
sys.path.insert(0, os.path.dirname(__file__))
from exp1_topk_bias import (
    theory_bias, theory_bias_lower, W_objective,
    compute_Kstar_theory, K_VALUES,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Section C: Synthetic K* Sweep (GPU-batched)
# ══════════════════════════════════════════════════════════════

def section_c_synthetic_kstar(device):
    """
    Synthetic Zipf + Gaussian: sweep (a, σ), measure empirical K*,
    compare to theoretical K* = 2σ²ln(1/δ)/a².
    Protocol A.1: v=32,000.
    """
    print("\n═══ Section C: Synthetic K* Sweep (GPU) ═══")
    v = 32000
    print(f"  v={v} (Protocol A.1)")

    a_values = [0.5, 1.0, 2.0]
    # Pick σ so K₀ = 2σ²ln(1/δ)/a² lands in [3, 500] for clean slope fit
    # δ=0.05 → ln(1/δ)≈3.0
    sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    delta = 0.05
    n_trials = 5000
    K_max = 1000  # increased from 100 to avoid ceiling

    # Pre-compute ranks for all trials (shared across parameters)
    ranks_t = torch.arange(1, v + 1, device=device, dtype=torch.float32)

    results = []
    t0 = time.time()

    for a in a_values:
        # Zipf true scores: q_k = -a * ln(k)
        q = -a * torch.log(ranks_t)  # (v,)

        for sigma in sigma_values:
            # Skip if theory K0 is way out of range
            K0_check = 2 * sigma**2 * np.log(1/delta) / a**2
            if K0_check > 800 or K0_check < 1:  # keep K₀ in useful range
                continue

            # Generate ALL noise at once: (n_trials, v)
            noise = sigma * torch.randn(n_trials, v, device=device)
            s = q.unsqueeze(0) + noise  # (n_trials, v)

            # Only need top-K_max values (much faster than full sort)
            s_topk, _ = s.topk(K_max, dim=-1)  # (n_trials, K_max)

            # Cumulative sum for top-K means
            cum_s = s_topk.cumsum(dim=-1)  # (n_trials, K_max)
            k_range = torch.arange(1, K_max + 1, device=device)
            topk_means = cum_s / k_range.unsqueeze(0).float()  # (n_trials, K_max)

            # Error: |topk_mean - q_max|
            q_max = q[0]
            errors = (topk_means - q_max).abs()  # (n_trials, K_max)

            # Best K per trial
            best_k_per_trial = errors.argmin(dim=-1) + 1  # (n_trials,), 1-indexed
            emp_kstar = best_k_per_trial.float().median().item()

            # Theory (cap theo at v)
            K0, theo_kstar, theo_kstar_corr = compute_Kstar_theory(v, sigma, a, delta)
            theo_kstar = min(theo_kstar, v)

            results.append({
                "a": a, "sigma": sigma,
                "empirical_kstar": float(emp_kstar),
                "K0": float(K0),
                "theory_kstar": float(theo_kstar),
                "theory_kstar_corr": float(min(theo_kstar_corr, v)),
                "ratio": float(emp_kstar / max(theo_kstar, 1)),
            })
            print(f"  a={a:.2f} σ={sigma:.1f}: "
                  f"emp={emp_kstar:.0f}  theo={theo_kstar:.1f}  "
                  f"corr={min(theo_kstar_corr, v):.1f}  "
                  f"ratio={emp_kstar/max(theo_kstar,1):.2f}")

    elapsed = time.time() - t0
    print(f"  Section C done in {elapsed:.1f}s")

    # ── Log-log slopes (using K₀ as theory reference) ──
    for a in a_values:
        subset = [r for r in results if r["a"] == a]
        sigmas = [r["sigma"] for r in subset]
        emp_ks = [r["empirical_kstar"] for r in subset]
        theo_ks = [r["K0"] for r in subset]
        # Only use points in the valid range
        valid = [(e > 1 and e < 999 and t > 1 and t < 800)
                 for e, t in zip(emp_ks, theo_ks)]
        if sum(valid) >= 3:
            log_sig = np.log([s for s, v in zip(sigmas, valid) if v])
            log_k = np.log([e for e, v in zip(emp_ks, valid) if v])
            slope = np.polyfit(log_sig, log_k, 1)[0]
            print(f"  Log-log slope for a={a}: {slope:.2f} (target: 2.0)")
        else:
            print(f"  Log-log slope for a={a}: insufficient valid points")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for a in a_values:
        subset = [r for r in results if r["a"] == a]
        sigmas = [r["sigma"] for r in subset]
        emp_ks = [r["empirical_kstar"] for r in subset]
        theo_ks = [r["theory_kstar"] for r in subset]
        ax.loglog(sigmas, emp_ks, "o-", label=f"a={a} empirical", markersize=7)
        ax.loglog(sigmas, theo_ks, "--", alpha=0.5, label=f"a={a} theory")
    ax.set_xlabel("σ", fontsize=13)
    ax.set_ylabel("K*", fontsize=13)
    ax.set_title("Falsification: K* ∝ σ² (slope=2 on log-log)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

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
#  Section D: Coverage Probability (GPU-batched)
# ══════════════════════════════════════════════════════════════

def section_d_coverage(device):
    """Verify Theorem X.1 interval coverage on synthetic data."""
    print("\n═══ Section D: Coverage Probability (GPU) ═══")

    v = 32000
    a, sigma = 1.0, 1.0
    delta_values = [0.01, 0.05, 0.10, 0.20]
    n_trials = 10000
    K_test = 10
    Delta_K = a * np.log(K_test)

    ranks_t = torch.arange(1, v + 1, device=device, dtype=torch.float32)
    q = -a * torch.log(ranks_t)
    q_max = q[0].item()

    coverage_results = {}
    t0 = time.time()

    for delta in delta_values:
        # Batch: (n_trials, v)
        noise = sigma * torch.randn(n_trials, v, device=device)
        s = q.unsqueeze(0) + noise

        # Sort descending, take top-K mean
        s_sorted, _ = s.sort(dim=-1, descending=True)
        s_topk_mean = s_sorted[:, :K_test].mean(dim=-1)  # (n_trials,)

        # Theorem interval
        lower = q_max - Delta_K - sigma * np.sqrt(2 * np.log(1/delta) / K_test)
        upper = q_max + sigma * np.sqrt(2 * np.log(np.e * v / K_test)) + \
                sigma * np.sqrt(2 * np.log(1/delta) / K_test)

        covered = ((s_topk_mean >= lower) & (s_topk_mean <= upper)).sum().item()
        cov = covered / n_trials
        target = 1 - 2 * delta
        coverage_results[delta] = {"coverage": cov, "target": target}
        mark = "✓" if cov >= target else "✗"
        print(f"  δ={delta:.2f}: coverage={cov:.4f}  target≥{target:.2f}  {mark}")

    print(f"  Section D done in {time.time()-t0:.1f}s")
    return coverage_results


# ══════════════════════════════════════════════════════════════
#  Section E: Falsification Test (Protocol A.3)
# ══════════════════════════════════════════════════════════════

def section_e_falsification(synthetic_results):
    """Log-log slope of K* vs σ at fixed a. Target: slope = 2."""
    print("\n═══ Section E: Falsification Test ═══")

    a_values = sorted(set(r["a"] for r in synthetic_results))
    all_slopes = []

    for a in a_values:
        subset = [r for r in synthetic_results if r["a"] == a]
        sigmas = np.array([r["sigma"] for r in subset])
        emp_ks = np.array([r["empirical_kstar"] for r in subset])
        # Use K₀ (leading-order theory) for slope comparison
        theo_ks = np.array([r["K0"] for r in subset])

        # Filter out floor (K*=1) and ceiling (K*=K_max) effects
        valid = (emp_ks > 1) & (emp_ks < 999) & (theo_ks > 1) & (theo_ks < 800)
        if valid.sum() < 3:
            print(f"  a={a:.2f}: too few valid points ({valid.sum()}), skipping")
            continue

        log_sig = np.log(sigmas[valid])
        log_k = np.log(emp_ks[valid])
        slope, intercept = np.polyfit(log_sig, log_k, 1)
        all_slopes.append(slope)

        # Bootstrap CI
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
    print(f"\n  Mean slope: {mean_slope:.3f} → {verdict}")
    return all_slopes


# ══════════════════════════════════════════════════════════════
#  Section F: Correlated Noise Rank Ablation (GPU-batched)
# ══════════════════════════════════════════════════════════════

def section_f_correlated_noise(device):
    """Reviewer-2 attack #1: test bias under low-rank correlated noise."""
    print("\n═══ Section F: Correlated Noise Rank Ablation (GPU) ═══")

    v = 32000
    sigma = 1.0
    a = 1.0
    n_trials = 5000
    rank_values = [1, 5, 20, 100, 500]
    K = 10

    ranks_t = torch.arange(1, v + 1, device=device, dtype=torch.float32)
    q = -a * torch.log(ranks_t)

    results = []
    t0 = time.time()

    for r in rank_values:
        # Generate projection matrix W: (v, r), orthonormal columns
        W_np = np.random.randn(v, r)
        W_np, _ = np.linalg.qr(W_np)
        W_np *= np.sqrt(v / r)
        W = torch.from_numpy(W_np).float().to(device)  # (v, r)

        # Batch: delta (n_trials, r), noise = W @ delta.T / sqrt(v)
        delta = sigma * torch.randn(n_trials, r, device=device)  # (n_trials, r)
        noise = delta @ W.T / np.sqrt(v)  # (n_trials, v)
        s = q.unsqueeze(0) + noise  # (n_trials, v)

        # Sort, top-K mean
        s_sorted, _ = s.sort(dim=-1, descending=True)
        topk_mean = s_sorted[:, :K].mean(dim=-1)
        s_max = s_sorted[:, 0]
        biases = (topk_mean - s_max).cpu().numpy()

        avg_bias = float(np.mean(biases))
        theory_iid = float(-sigma * np.sqrt(2 * np.log(np.e * v / K)))
        results.append({
            "rank": r,
            "avg_bias": avg_bias,
            "theory_iid": theory_iid,
            "ratio": avg_bias / theory_iid if theory_iid != 0 else 0,
        })
        print(f"  rank={r:6d}: bias={avg_bias:.4f}  "
              f"theory_iid={theory_iid:.4f}  ratio={avg_bias/theory_iid:.3f}")

    print(f"  Section F done in {time.time()-t0:.1f}s")

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[exp1-synth] Device: {device}")
    if torch.cuda.is_available():
        print(f"[exp1-synth] GPU: {torch.cuda.get_device_name(0)}")

    # Section C: Synthetic K* sweep
    synthetic_results = section_c_synthetic_kstar(device)

    # Section D: Coverage probability
    coverage = section_d_coverage(device)

    # Section E: Falsification
    slopes = section_e_falsification(synthetic_results)

    # Section F: Correlated noise
    corr_results = section_f_correlated_noise(device)

    # ── Save ──
    save_data = {
        "v": 32000,
        "section_c": synthetic_results,
        "section_d": {str(k): v for k, v in coverage.items()},
        "section_e_slopes": [float(s) for s in slopes],
        "section_f": corr_results,
    }
    with open(f"{args.output_dir}/exp1_synth_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp1-synth] Saved to {args.output_dir}/exp1_synth_results.json")
    print("[exp1-synth] Done!")


if __name__ == "__main__":
    main()
