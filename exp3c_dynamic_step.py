#!/usr/bin/env python3
"""
Experiment 3C+: Dynamic Step Size for ACI Margin Adaptation
=============================================================
Addresses the 9.4% coverage gap in Exp 3 Section C.
Tests three step-size variants for the ACI update:

  Original:  u_{t+1} = u_t + η · (error_t - β)          [fixed η]
  Decay:     u_{t+1} = u_t + η_t · (error_t - β)         [η_t = η₀/√(1+γt)]
  Momentum:  u_{t+1} = u_t + η · (error_t - β) + μ·v_t   [with momentum]
             v_{t+1} = μ·v_t + η · (error_t - β)

Expected: decay/momentum reduce coverage gap from ~9.4% to <5%.
"""
import argparse, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pinball_loss(margin, D_t, beta):
    """β-pinball loss: asymmetric absolute error."""
    residual = D_t - margin
    return np.where(residual >= 0, beta * residual, (beta - 1) * residual)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_aci_variant(lambda_hat, D_t, beta=0.9, eta=0.1, variant="fixed",
                    gamma=0.01, mu=0.9, w=50, m0=0.0, alpha_grid=None):
    """
    Run ACI with a specific step-size variant.

    variant: "fixed" | "decay" | "momentum"
    """
    T = len(lambda_hat)
    if alpha_grid is None:
        alpha_grid = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    N_alpha = len(alpha_grid)

    learning_rate = np.sqrt(8 * np.log(N_alpha) / max(T, 1))
    weights = np.ones(N_alpha) / N_alpha

    u = 0.0
    v = 0.0  # momentum velocity
    margins = np.zeros(T)
    errors = np.zeros(T)
    cumulative_losses = {a: 0.0 for a in alpha_grid}
    algo_losses = np.zeros(T)

    for t in range(T):
        # Feedforward
        window_start = max(0, t - w + 1)
        Lambda_t = np.sum(np.maximum(lambda_hat[window_start:t+1], 0))

        # EWA mixture
        alpha_t = np.sum(weights * alpha_grid)

        # Posted margin
        m_t = m0 + alpha_t * Lambda_t + u
        m_t = max(m_t, 0)
        m_t = min(m_t, 100)
        margins[t] = m_t

        # Observe error
        errors[t] = 1.0 if D_t[t] > m_t else 0.0

        # Step size computation
        if variant == "fixed":
            eta_t = eta
        elif variant == "decay":
            eta_t = eta / np.sqrt(1 + gamma * t)
        elif variant == "momentum":
            eta_t = eta
        else:
            eta_t = eta

        # ACI update with variant
        signal = errors[t] - beta
        if variant == "momentum":
            v = mu * v + eta_t * signal
            u = u + v
        else:
            u = u + eta_t * signal

        # EWA update
        losses = np.array([
            pinball_loss(m0 + a * Lambda_t + u, D_t[t], beta)
            for a in alpha_grid
        ])
        algo_losses[t] = losses[np.argmax(weights)]
        for i, a in enumerate(alpha_grid):
            cumulative_losses[a] += losses[i]
        weights *= np.exp(-learning_rate * losses)
        weights /= weights.sum()

    # Best fixed loss
    best_fixed_loss = min(cumulative_losses.values())
    cum_algo_loss = np.cumsum(algo_losses)
    regret = cum_algo_loss - best_fixed_loss * np.arange(1, T + 1) / T

    return margins, errors, regret, cum_algo_loss


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load real λ̂ from exp3 results (or simulate) ──
    exp3_path = f"{args.output_dir}/exp3_results.json"
    if os.path.exists(exp3_path):
        with open(exp3_path) as f:
            exp3_data = json.load(f)
        print("[exp3c+] Loaded exp3 results")
    else:
        exp3_data = None
        print("[exp3c+] No exp3 results found, using simulated data")

    # ── Simulate realistic λ̂ and D_t ──
    # Use a longer sequence for better statistics
    T_total = 5000
    np.random.seed(args.seed)

    # Generate λ̂: bimodal (contractive + expansive)
    lambda_hat = np.zeros(T_total)
    # Contractive mode: centered at -1
    mode1 = np.random.normal(-1.0, 0.5, T_total)
    # Expansive mode: centered at +2
    mode2 = np.random.normal(2.0, 1.0, T_total)
    # Random switching
    regime = np.random.choice([0, 1], T_total, p=[0.6, 0.4])
    lambda_hat = np.where(regime == 0, mode1, mode2)

    # D_t: margin needed (90th percentile of recent λ̂)
    D_t = np.zeros(T_total)
    window = 20
    for t in range(T_total):
        start = max(0, t - window)
        D_t[t] = np.percentile(lambda_hat[start:t+1], 90) if t > 0 else lambda_hat[0]

    # Add non-stationarity: burst → normal → calm
    burst_end = T_total // 4
    calm_start = 3 * T_total // 4
    D_t[:burst_end] *= 2.5       # burst: high demand
    D_t[burst_end:calm_start] *= 1.0  # normal
    D_t[calm_start:] *= 0.2      # calm: low demand

    beta = 0.9
    eta = 0.1

    # ── Run all variants ──
    print("\n═══ ACI Step Size Variants ═══")
    variants = {
        "fixed (η=0.1)": {"variant": "fixed", "eta": 0.1, "gamma": 0, "mu": 0},
        "decay (η=0.1, γ=0.005)": {"variant": "decay", "eta": 0.1, "gamma": 0.005, "mu": 0},
        "decay-fast (η=0.1, γ=0.02)": {"variant": "decay", "eta": 0.1, "gamma": 0.02, "mu": 0},
        "momentum (η=0.05, μ=0.9)": {"variant": "momentum", "eta": 0.05, "gamma": 0, "mu": 0.9},
        "momentum (η=0.05, μ=0.95)": {"variant": "momentum", "eta": 0.05, "gamma": 0, "mu": 0.95},
    }

    results = {}
    for name, config in variants.items():
        margins, errors, regret, cum_loss = run_aci_variant(
            lambda_hat, D_t, beta=beta,
            eta=config["eta"], variant=config["variant"],
            gamma=config["gamma"], mu=config["mu"]
        )
        err_rate = errors.mean()
        gap = abs(err_rate - beta)
        mean_margin = margins.mean()

        # Compute coverage gap per phase
        gap_burst = abs(errors[:burst_end].mean() - beta)
        gap_normal = abs(errors[burst_end:calm_start].mean() - beta)
        gap_calm = abs(errors[calm_start:].mean() - beta)

        results[name] = {
            "error_rate": float(err_rate),
            "gap": float(gap),
            "gap_burst": float(gap_burst),
            "gap_normal": float(gap_normal),
            "gap_calm": float(gap_calm),
            "mean_margin": float(mean_margin),
            "margins": margins,
            "errors": errors,
            "regret": regret,
            "cum_loss": cum_loss,
        }
        print(f"  {name:30s}: err={err_rate:.3f}  gap={gap:.3f}  "
              f"burst_gap={gap_burst:.3f}  calm_gap={gap_calm:.3f}")

    # ── Find best variant ──
    best_name = min(results, key=lambda k: results[k]["gap"])
    best_gap = results[best_name]["gap"]
    print(f"\n  Best: {best_name} with gap={best_gap:.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ["gray", "blue", "cyan", "green", "limegreen"]

    # Top-left: running error rate
    ax = axes[0, 0]
    window_smooth = 100
    for (name, res), color in zip(results.items(), colors):
        running = np.convolve(res["errors"],
                            np.ones(window_smooth)/window_smooth, mode='valid')
        ax.plot(running, label=name, color=color, alpha=0.8, lw=1.5)
    ax.axhline(beta, color="black", linestyle="--", lw=2, label=f"Target β={beta}")
    ax.axvline(burst_end, color="red", linestyle=":", alpha=0.5)
    ax.axvline(calm_start, color="blue", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Error rate (windowed)", fontsize=12)
    ax.set_title("Coverage: Running Error Rate by Variant", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Top-right: coverage gap comparison (bar chart)
    ax = axes[0, 1]
    names = list(results.keys())
    gaps = [results[n]["gap"] for n in names]
    bar_colors = ["red" if g > 0.05 else "green" for g in gaps]
    bars = ax.barh(names, gaps, color=bar_colors, alpha=0.7)
    ax.axvline(0.05, color="orange", linestyle="--", lw=2, label="5% target")
    ax.axvline(0.02, color="green", linestyle="--", lw=2, label="2% target")
    ax.set_xlabel("|Error Rate - β|", fontsize=12)
    ax.set_title("Coverage Gap by Variant (lower = better)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, gap in zip(bars, gaps):
        ax.text(gap + 0.002, bar.get_y() + bar.get_height()/2,
                f"{gap:.3f}", va="center", fontsize=9)

    # Bottom-left: margins over time (best 3 variants)
    ax = axes[1, 0]
    top3 = sorted(results.items(), key=lambda x: x[1]["gap"])[:3]
    for (name, res), color in zip(top3, ["gray", "green", "blue"]):
        ax.plot(res["margins"], label=name, color=color, alpha=0.6, lw=1)
    ax.axvline(burst_end, color="red", linestyle=":", alpha=0.5, label="Burst end")
    ax.axvline(calm_start, color="blue", linestyle=":", alpha=0.5, label="Calm start")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Posted margin m_t", fontsize=12)
    ax.set_title("Margin Adaptation (Top 3 Variants)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: phase-wise gap breakdown
    ax = axes[1, 1]
    phases = ["Burst", "Normal", "Calm"]
    phase_keys = ["gap_burst", "gap_normal", "gap_calm"]
    x = np.arange(len(phases))
    width = 0.15
    for i, (name, res) in enumerate(list(results.items())[:4]):
        gaps_phase = [res[k] for k in phase_keys]
        ax.bar(x + i * width, gaps_phase, width, label=name[:20],
               alpha=0.7, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(phases)
    ax.set_ylabel("|Error Rate - β|", fontsize=12)
    ax.set_title("Coverage Gap by Phase", fontsize=13)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig3c_dynamic_step.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──
    save_data = {
        "beta": beta,
        "eta": eta,
        "T_total": T_total,
        "variants": {
            name: {
                "error_rate": r["error_rate"],
                "gap": r["gap"],
                "gap_burst": r["gap_burst"],
                "gap_normal": r["gap_normal"],
                "gap_calm": r["gap_calm"],
                "mean_margin": r["mean_margin"],
            }
            for name, r in results.items()
        },
        "best_variant": best_name,
        "best_gap": best_gap,
    }
    with open(f"{args.output_dir}/exp3c_dynamic_step_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[exp3c+] Results saved to {args.output_dir}/exp3c_dynamic_step_results.json")
    print("[exp3c+] Done!")


if __name__ == "__main__":
    main()
