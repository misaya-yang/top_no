#!/usr/bin/env python3
"""
Experiment 1B: Vocabulary Size Ablation — Finite-V Correction
===============================================================
Addresses the K* log-log slope deviation (2.5-3.5 vs theoretical 2.0).
Tests whether the deviation is a finite-vocabulary artifact by varying V:

  V ∈ {2000, 5000, 10000, 32000, 100000}

Theory: as V → ∞, slope → 2.0.
If slope decreases monotonically with V, the deviation is confirmed
as a finite-sample artifact (not a theoretical failure).
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_kstar_for_v(V, a_values, sigma_values, delta, n_trials, K_max, device):
    """
    Run synthetic K* experiment for a specific vocabulary size V.
    Returns dict of {a: [(sigma, emp_kstar, theo_kstar), ...]}.
    """
    ranks_t = torch.arange(1, V + 1, device=device, dtype=torch.float32)
    results = {a: [] for a in a_values}

    for a in a_values:
        q = -a * torch.log(ranks_t)
        q_max = q[0].item()

        for sigma in sigma_values:
            K0_check = 2 * sigma**2 * np.log(1 / delta) / a**2
            if K0_check > K_max * 0.8 or K0_check < 1:
                continue

            # Batch Monte Carlo
            noise = sigma * torch.randn(n_trials, V, device=device)
            s = q.unsqueeze(0) + noise

            # Top-K_max
            s_topk, _ = s.topk(min(K_max, V), dim=-1)
            cum_s = s_topk.cumsum(dim=-1)
            k_range = torch.arange(1, s_topk.shape[1] + 1, device=device)
            topk_means = cum_s / k_range.unsqueeze(0).float()

            errors = (topk_means - q_max).abs()
            best_k = errors.argmin(dim=-1) + 1
            emp_kstar = best_k.float().median().item()

            # Theory K*
            K0 = 2 * sigma**2 * np.log(1 / delta) / a**2

            results[a].append({
                "sigma": sigma,
                "emp_kstar": float(emp_kstar),
                "K0": float(K0),
            })

    return results


def fit_slope(results_for_a):
    """Fit log-log slope of K* vs σ."""
    sigmas = [r["sigma"] for r in results_for_a]
    emp_ks = [r["emp_kstar"] for r in results_for_a]
    theo_ks = [r["K0"] for r in results_for_a]

    valid = [(e > 1 and t > 1 and e < 999 and t < 999)
             for e, t in zip(emp_ks, theo_ks)]

    if sum(valid) < 3:
        return None, None

    log_sig = np.log([s for s, v in zip(sigmas, valid) if v])
    log_k = np.log([e for e, v in zip(emp_ks, valid) if v])

    slope, intercept = np.polyfit(log_sig, log_k, 1)
    return float(slope), float(intercept)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[exp1b] Device: {device}")

    # ── Parameters ──
    V_values = [2000, 5000, 10000, 32000, 100000]
    a_values = [0.5, 1.0, 2.0]
    sigma_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    delta = 0.05
    n_trials = 3000
    K_max = 1000

    print(f"\n[exp1b] Vocabulary sizes: {V_values}")
    print(f"[exp1b] a values: {a_values}")
    print(f"[exp1b] n_trials: {n_trials}, K_max: {K_max}")

    # ── Run for each V ──
    all_results = {}
    all_slopes = {}

    t0 = time.time()
    for V in V_values:
        print(f"\n═══ V = {V} ═══")
        results_V = compute_kstar_for_v(
            V, a_values, sigma_values, delta, n_trials, K_max, device
        )
        all_results[V] = results_V

        slopes_V = {}
        for a in a_values:
            slope, intercept = fit_slope(results_V[a])
            if slope is not None:
                slopes_V[a] = slope
                deviation = slope - 2.0
                print(f"  a={a:.1f}: slope={slope:.3f} (deviation from 2.0: {deviation:+.3f})")
            else:
                print(f"  a={a:.1f}: insufficient data")

        all_slopes[V] = slopes_V

    elapsed = time.time() - t0
    print(f"\n[exp1b] All V done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Analysis: slope vs V ──
    print("\n═══ Slope Convergence Analysis ══")
    print(f"{'V':>8}  {'slope(a=0.5)':>14}  {'slope(a=1.0)':>14}  {'slope(a=2.0)':>14}  {'mean':>8}")
    print("-" * 70)

    convergence_data = []
    for V in V_values:
        slopes = []
        row = f"{V:>8}"
        for a in a_values:
            s = all_slopes[V].get(a)
            if s is not None:
                row += f"  {s:>14.3f}"
                slopes.append(s)
            else:
                row += f"  {'N/A':>14}"
        mean_slope = np.mean(slopes) if slopes else 0
        row += f"  {mean_slope:>8.3f}"
        print(row)
        convergence_data.append({"V": V, "mean_slope": float(mean_slope),
                                 "slopes": all_slopes[V]})

    # ── Fit convergence rate ──
    valid_conv = [(d["V"], d["mean_slope"]) for d in convergence_data if d["mean_slope"] > 0]
    if len(valid_conv) >= 3:
        Vs = np.array([v[0] for v in valid_conv], dtype=float)
        slopes = np.array([v[1] for v in valid_conv])
        # Fit: slope(V) = 2.0 + A / V^α
        try:
            from scipy.optimize import curve_fit
            def convergence_model(V, A, alpha):
                return 2.0 + A / (V ** alpha)
            popt, _ = curve_fit(convergence_model, Vs, slopes, p0=[1000, 0.5])
            A_fit, alpha_fit = popt
            extrapolated = convergence_model(1e8, A_fit, alpha_fit)
            print(f"\n  Convergence fit: slope(V) = 2.0 + {A_fit:.1f} / V^{alpha_fit:.3f}")
            print(f"  Extrapolated slope (V→∞): {extrapolated:.4f}")
        except Exception:
            A_fit, alpha_fit = 0, 0
            extrapolated = 2.0

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left: K* vs σ for all V (one panel per a)
    ax = axes[0]
    colors_V = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for V, color in zip(V_values, colors_V):
        for a in [1.0]:  # just a=1 for clarity
            subset = all_results[V].get(a, [])
            sigmas = [r["sigma"] for r in subset]
            emp_ks = [r["emp_kstar"] for r in subset]
            theo_ks = [r["K0"] for r in subset]
            ax.loglog(sigmas, emp_ks, "o-", label=f"V={V} (emp)", color=color,
                      markersize=5, alpha=0.8)
    # Theory line
    sigmas_plot = np.linspace(0.3, 10, 100)
    K0_plot = 2 * sigmas_plot**2 * np.log(1/delta) / 1.0**2
    ax.loglog(sigmas_plot, K0_plot, "k--", lw=2, alpha=0.5, label="Theory K₀ ∝ σ²")
    ax.set_xlabel("σ", fontsize=13)
    ax.set_ylabel("K*", fontsize=13)
    ax.set_title("K* vs σ (a=1.0) for different V", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Middle: Slope vs V
    ax2 = axes[1]
    for a in a_values:
        Vs_plot = []
        slopes_plot = []
        for V in V_values:
            s = all_slopes[V].get(a)
            if s is not None:
                Vs_plot.append(V)
                slopes_plot.append(s)
        if Vs_plot:
            ax2.semilogx(Vs_plot, slopes_plot, "o-", label=f"a={a}", markersize=8, lw=2)

    ax2.axhline(2.0, color="black", linestyle="--", lw=2, label="Theory: slope=2.0")
    ax2.axhspan(1.5, 2.5, alpha=0.1, color="green", label="±0.5 band")
    ax2.set_xlabel("Vocabulary size V", fontsize=13)
    ax2.set_ylabel("Log-log slope of K* vs σ", fontsize=13)
    ax2.set_title("Slope Convergence: Finite-V → Asymptotic", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Right: deviation from 2.0 vs 1/V
    ax3 = axes[2]
    for a in a_values:
        Vs_inv = []
        deviations = []
        for V in V_values:
            s = all_slopes[V].get(a)
            if s is not None:
                Vs_inv.append(1.0 / V)
                deviations.append(s - 2.0)
        if Vs_inv:
            ax3.plot(Vs_inv, deviations, "o-", label=f"a={a}", markersize=8, lw=2)
            # Linear fit
            if len(Vs_inv) >= 3:
                coef = np.polyfit(Vs_inv, deviations, 1)
                x_fit = np.linspace(0, max(Vs_inv) * 1.1, 100)
                ax3.plot(x_fit, np.polyval(coef, x_fit), "--", alpha=0.5)

    ax3.axhline(0, color="black", linestyle="--", lw=2, label="No deviation")
    ax3.set_xlabel("1/V", fontsize=13)
    ax3.set_ylabel("Slope deviation from 2.0", fontsize=13)
    ax3.set_title("Deviation ∝ 1/V? (x-intercept = asymptote)", fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig1b_vocab_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──
    save_data = {
        "V_values": V_values,
        "a_values": a_values,
        "sigma_values": sigma_values,
        "delta": delta,
        "n_trials": n_trials,
        "slopes": {
            str(V): {str(a): s for a, s in slopes.items()}
            for V, slopes in all_slopes.items()
        },
        "convergence": convergence_data,
    }
    with open(f"{args.output_dir}/exp1b_vocab_ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp1b] Results saved to {args.output_dir}/exp1b_vocab_ablation_results.json")
    print("[exp1b] Done!")


if __name__ == "__main__":
    main()
