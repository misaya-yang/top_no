#!/usr/bin/env python3
"""
Experiment 2 (Revised): Identifiability via Synthetic Heteroscedastic Channel
=============================================================================
Instead of a real teacher-student pair (where architecture gap dominates),
we construct a KNOWN heteroscedastic channel:

  teacher logits s*_i  →  student logits s_i = s*_i + ε_i
  where ε_i ~ N(0, σ₀² + c/n_i)  and n_i = token frequency

This directly tests the paper's claims:
  Theorem 3': Var(r) = σ₀² + c/n  (residual variance vs frequency)
  Protocol B.1: truncated-KL converges, full-KL does not
  Theorem 4: tail non-estimability
  Theorem 5: full-KL minimax inconsistency
  Corollary: V_eff sharp transition
"""
import argparse, json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from data_utils import load_text_samples, tokenize_batch, load_model_and_tokenizer, free_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    # Noise parameters (KNOWN ground truth)
    p.add_argument("--sigma0", type=float, default=0.1,
                   help="Base noise σ₀ (architecture-independent floor)")
    p.add_argument("--c-param", type=float, default=100.0,
                   help="Frequency-dependent noise coefficient c")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Section A: Residual Variance vs Token Frequency
# ══════════════════════════════════════════════════════════════

def section_a_residuals(input_ids, attn_mask, teacher_logits, student_logits,
                        sigma0_true, c_true, output_dir):
    """Verify Var(r) = σ₀² + c/n with known ground truth."""
    print("\n═══ Section A: Residual Variance vs Token Frequency ═══")
    print(f"  Ground truth: σ₀²={sigma0_true**2:.4f}, c={c_true:.2f}")

    N, Tm1 = teacher_logits.shape
    residuals = teacher_logits - student_logits  # (N, T-1)
    targets = input_ids[:, 1:]

    # Token frequencies
    all_tokens = input_ids[attn_mask > 0]
    token_counts = torch.bincount(all_tokens)

    # Per-position frequency
    valid_mask = attn_mask[:, 1:] > 0
    n_i = token_counts[targets[valid_mask]].float().numpy()
    r_i = residuals[valid_mask].numpy()
    targets_valid = targets[valid_mask].numpy()

    print(f"  Valid positions: {len(n_i)}")
    print(f"  Unique tokens: {len(np.unique(targets_valid))}")

    # Bin by log-frequency
    log_n = np.log10(n_i + 1)
    n_bins = 30
    bin_edges = np.linspace(log_n.min(), log_n.max() + 0.01, n_bins + 1)
    bin_centers, bin_variances, bin_counts = [], [], []

    for b in range(n_bins):
        mask = (log_n >= bin_edges[b]) & (log_n < bin_edges[b + 1])
        if mask.sum() < 10:
            continue
        bin_centers.append(10 ** ((bin_edges[b] + bin_edges[b + 1]) / 2) - 1)
        bin_variances.append(np.var(r_i[mask]))
        bin_counts.append(int(mask.sum()))

    bin_centers = np.array(bin_centers)
    bin_variances = np.array(bin_variances)
    bin_counts = np.array(bin_counts)

    # Fit Var(r) = σ₀² + c/n
    def var_model(n, sigma0_sq, c):
        return sigma0_sq + c / (n + 1e-8)

    try:
        popt, _ = curve_fit(var_model, bin_centers, bin_variances,
                            p0=[sigma0_true**2, c_true], maxfev=10000)
        sigma0_sq_fit, c_fit = popt
    except Exception:
        sigma0_sq_fit, c_fit = 0.0, 0.0

    print(f"  Fit: Var(r) = {sigma0_sq_fit:.4f} + {c_fit:.2f}/n")
    print(f"  Ground truth:     {sigma0_true**2:.4f} + {c_true:.2f}/n")
    print(f"  Recovery error: σ₀² err={abs(sigma0_sq_fit - sigma0_true**2):.4f}, "
          f"c err={abs(c_fit - c_true):.2f}")

    # Also compute per-token residual variance (direct verification)
    unique_tokens = np.unique(targets_valid)
    per_token_var = []
    per_token_n = []
    for tok in unique_tokens:
        tok_mask = targets_valid == tok
        if tok_mask.sum() >= 5:
            per_token_var.append(np.var(r_i[tok_mask]))
            per_token_n.append(n_i[tok_mask][0])  # all same frequency

    per_token_var = np.array(per_token_var)
    per_token_n = np.array(per_token_n)

    # Sort by frequency and plot
    sort_idx = np.argsort(per_token_n)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: binned analysis
    ax = axes[0]
    ax.scatter(np.log10(bin_centers + 1), bin_variances,
               s=np.clip(bin_counts / 100, 20, 500),
               c=bin_counts, cmap="YlOrRd", alpha=0.8, edgecolors="gray", lw=0.5)
    plt.colorbar(ax.collections[0], ax=ax, label="Bin count")

    n_fit = np.logspace(np.log10(max(0.1, bin_centers.min())),
                         np.log10(bin_centers.max() + 1), 200)
    ax.plot(np.log10(n_fit + 1), var_model(n_fit, sigma0_sq_fit, c_fit),
            "--", color="steelblue", lw=2,
            label=f"Fit: σ₀²={sigma0_sq_fit:.3f} + {c_fit:.1f}/n")
    ax.plot(np.log10(n_fit + 1), var_model(n_fit, sigma0_true**2, c_true),
            ":", color="green", lw=2,
            label=f"Truth: σ₀²={sigma0_true**2:.3f} + {c_true:.1f}/n")
    ax.axvspan(0, np.log10(11), alpha=0.15, color="red",
               label="IG explosion zone (n < 10)")
    ax.set_xlabel("log₁₀(n + 1)", fontsize=13)
    ax.set_ylabel("Var(r)", fontsize=13)
    ax.set_title("Residual Variance vs Token Frequency\n(Synthetic Heteroscedastic Channel)",
                 fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: per-token scatter
    ax2 = axes[1]
    ax2.scatter(per_token_n[sort_idx], per_token_var[sort_idx],
                s=10, alpha=0.3, color="steelblue")
    ax2.plot(n_fit, var_model(n_fit, sigma0_sq_fit, c_fit),
             "--", color="red", lw=2, label="Fitted curve")
    ax2.plot(n_fit, var_model(n_fit, sigma0_true**2, c_true),
             ":", color="green", lw=2, label="Ground truth")
    ax2.set_xscale("log")
    ax2.set_xlabel("Token frequency n", fontsize=13)
    ax2.set_ylabel("Var(r_i)", fontsize=13)
    ax2.set_title("Per-Token Residual Variance", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_residuals_synth.png", dpi=150, bbox_inches="tight")
    plt.close()

    return sigma0_sq_fit, c_fit, bin_centers, bin_variances


# ══════════════════════════════════════════════════════════════
#  Section B: n-Sweep + Convergence (GPU-batched where possible)
# ══════════════════════════════════════════════════════════════

def section_b_nsweep(input_ids, attn_mask, teacher_logits, base_noise,
                     token_counts_global, sigma0, c_param, output_dir, device,
                     seed=42):
    """
    Measure ESTIMATION ERROR convergence: for each n, draw multiple subsamples,
    estimate σ² from each, measure variance of estimates across subsamples.
    Theory: Var(σ̂²) ∝ 1/n for both full and truncated,
    but full-KL has additional bias from rare tokens → MSE doesn't converge.
    """
    print("\n═══ Section B: n-Sweep + Convergence (estimation error) ═══")

    N_max, Tm1 = teacher_logits.shape
    targets = input_ids[:N_max, 1:]
    valid_mask = attn_mask[:N_max, 1:] > 0
    global_counts = token_counts_global.float()

    n_values = [50, 100, 200, 500, 1000, 2000]
    n_values = [n for n in n_values if n <= N_max]
    n_replicates = 20  # number of independent subsamples per n

    results_b = []
    zero_counts = []

    # True σ² (average over all positions with global frequencies)
    all_n = global_counts[targets[:N_max].clamp(max=len(global_counts)-1)].float()
    all_n = all_n.clamp(min=1)
    true_sigma_sq = (sigma0**2 + c_param / all_n)[valid_mask].mean().item()
    print(f"  True average σ² = {true_sigma_sq:.4f}")

    for n_sub in n_values:
        full_estimates = []
        trunc_estimates = {5: [], 10: [], 20: []}
        zero_fracs = []

        for rep in range(n_replicates):
            torch.manual_seed(seed + rep * 1000 + n_sub)
            np.random.seed(seed + rep * 1000 + n_sub)
            indices = np.sort(np.random.choice(N_max, n_sub, replace=False))

            sub_ids = input_ids[indices]
            sub_mask = attn_mask[indices]
            sub_targets = sub_ids[:, 1:]
            sub_vm = valid_mask[indices]
            sub_tokens = sub_ids[sub_mask > 0]
            sub_counts = torch.bincount(sub_tokens)

            # Zero-count check
            sub_target_flat = sub_targets[sub_vm]
            if len(sub_target_flat) > 0:
                z = (sub_counts[sub_target_flat] == 0).sum().item()
                zero_fracs.append(z / max(len(sub_target_flat), 1))
            else:
                zero_fracs.append(0)

            # Generate fresh noise with GLOBAL frequencies
            sub_teacher = teacher_logits[indices]
            sub_n = global_counts[sub_targets.clamp(max=len(global_counts)-1)].float().clamp(min=1)
            sigma_pos = torch.sqrt(sigma0**2 + c_param / sub_n)
            sub_noise = sigma_pos * torch.randn_like(sub_teacher)

            residuals = sub_noise  # the injected noise IS the residual
            sub_vm_bool = sub_vm.bool() if sub_vm.dtype != torch.bool else sub_vm

            # (a) Full estimate: mean squared residual over all valid positions
            all_res = residuals[sub_vm_bool]
            if len(all_res) > 0:
                full_est = (all_res ** 2).mean().item()
            else:
                full_est = 0
            full_estimates.append(full_est)

            # (b) Truncated estimates: only high-frequency tokens
            for thresh in [5, 10, 20]:
                global_freq = global_counts[sub_targets.clamp(max=len(global_counts)-1)]
                keep = (global_freq >= thresh) & sub_vm_bool
                if keep.sum() > 0:
                    trunc_res = residuals[keep]
                    trunc_est = (trunc_res ** 2).mean().item()
                else:
                    trunc_est = 0
                trunc_estimates[thresh].append(trunc_est)

        # Compute variance of estimates across replicates
        full_var = np.var(full_estimates)
        full_mse = np.mean([(e - true_sigma_sq)**2 for e in full_estimates])
        avg_zero = np.mean(zero_fracs)
        zero_counts.append({"n": n_sub, "zero_frac": avg_zero})

        for thresh in [5, 10, 20]:
            trunc_var = np.var(trunc_estimates[thresh])
            trunc_mse = np.mean([(e - true_sigma_sq)**2 for e in trunc_estimates[thresh]])
            results_b.append({
                "n": n_sub, "c": thresh,
                "full_var": full_var,
                "full_mse": full_mse,
                "trunc_var": trunc_var,
                "trunc_mse": trunc_mse,
            })

        print(f"  n={n_sub:4d}: full_MSE={full_mse:.4f}  "
              f"full_Var={full_var:.4f}  zeros={avg_zero:.3f}")

    # ── Plot convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ns = sorted(set(r["n"] for r in results_b))

    # Plot MSE of full vs truncated estimates
    full_mses = [r["full_mse"] for r in results_b if r["c"] == 5]
    if full_mses:
        ax.loglog(ns, full_mses, "s--", color="red", label="Full (all tokens) MSE", markersize=6)

    for thresh in [5, 10, 20]:
        trunc_mses = [r["trunc_mse"] for r in results_b if r["c"] == thresh]
        valid_pairs = [(n, m) for n, m in zip(ns, trunc_mses) if m > 0]
        if valid_pairs:
            vns, vms = zip(*valid_pairs)
            ax.loglog(vns, vms, "o-", label=f"Truncated (freq≥{thresh}) MSE", markersize=6)

    # Reference slopes
    if ns:
        ref_ns = np.array(ns, dtype=float)
        ref_ms = np.array(full_mses, dtype=float)
        if len(ref_ms) > 0 and ref_ms[0] > 0:
            ref_line_m1 = ref_ms[0] * (ref_ns[0] / ref_ns) ** 1.0
            ax.loglog(ref_ns, ref_line_m1, ":", color="gray", alpha=0.5,
                      label="Reference: n⁻¹")

    ax.set_xlabel("n (subsample size)", fontsize=13)
    ax.set_ylabel("MSE of σ² estimate", fontsize=13)
    ax.set_title("Convergence: MSE of Full vs Truncated σ² Estimate", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Right: zero-count instability
    ax2 = axes[1]
    zc_ns = [z["n"] for z in zero_counts]
    zc_fracs = [z["zero_frac"] for z in zero_counts]
    ax2.plot(zc_ns, zc_fracs, "o-", color="crimson", markersize=8, lw=2)
    ax2.set_xlabel("n (subsample size)", fontsize=13)
    ax2.set_ylabel("Fraction of zero-count targets", fontsize=13)
    ax2.set_title("Full-KL Instability: Zero-Frequency Hits", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2b_nsweep_synth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fit log-log slope for MSE convergence
    for thresh in [5, 10, 20]:
        trunc_mses = np.array([r["trunc_mse"] for r in results_b if r["c"] == thresh])
        ns_arr = np.array(ns, dtype=float)
        valid = (trunc_mses > 0) & (ns_arr > 0)
        if valid.sum() >= 2:
            slope, _ = np.polyfit(np.log(ns_arr[valid]), np.log(trunc_mses[valid]), 1)
            print(f"  Truncated MSE (freq≥{thresh}) log-log slope: {slope:.3f} "
                  f"(target: -1.0)")
    # Full MSE slope
    full_mses = np.array([r["full_mse"] for r in results_b if r["c"] == 5])
    ns_arr = np.array(ns, dtype=float)
    valid = (full_mses > 0) & (ns_arr > 0)
    if valid.sum() >= 2:
        slope, _ = np.polyfit(np.log(ns_arr[valid]), np.log(full_mses[valid]), 1)
        print(f"  Full MSE log-log slope: {slope:.3f} (target: -1.0)")

    return results_b, zero_counts


# ══════════════════════════════════════════════════════════════
#  Section D: Two-Point Demonstration (Theorem 5)
# ══════════════════════════════════════════════════════════════

def section_d_twopoint(input_ids, attn_mask, output_dir):
    """
    Construct T^A, T^B identical on V_eff but with different tails.
    Verify permutation test cannot distinguish (p > 0.3) while
    full-KL differs by ≥ Λ.
    """
    print("\n═══ Section D: Two-Point Demonstration ═══")

    all_tokens = input_ids[attn_mask > 0]
    N_total = all_tokens.shape[0]
    token_counts = torch.bincount(all_tokens).float()
    V = token_counts.shape[0]

    T_base = (token_counts / N_total).numpy()
    n = N_total

    # Tail tokens: T_i ≤ 1/(4n)
    tail_threshold = 1.0 / (4 * n)
    tail_mask = (T_base > 0) & (T_base <= tail_threshold)
    tail_indices = np.where(tail_mask)[0]
    head_indices = np.where(~tail_mask)[0]

    tail_mass = T_base[tail_indices].sum()
    n_tail = len(tail_indices)

    print(f"  V={V}, n={n}, tail tokens={n_tail}, tail mass={tail_mass:.6f}")

    if n_tail < 10:
        print("  Too few tail tokens. Using synthetic tail.")
        # Add synthetic tail tokens
        n_synth_tail = 500
        tail_indices = np.arange(V, V + n_synth_tail)
        V_extended = V + n_synth_tail
        T_base_ext = np.zeros(V_extended)
        T_base_ext[:V] = T_base
        # Give tail tokens tiny probabilities
        tail_p = 1e-6
        T_base_ext[V:] = tail_p
        T_base_ext = T_base_ext / T_base_ext.sum()
        T_base = T_base_ext
        V = V_extended
        n_tail = n_synth_tail
        head_indices = np.where(~((np.arange(V) >= V - n_synth_tail)))[0]

    # Construct T^A and T^B
    eta = min(tail_mass if tail_mass > 0 else 1e-4, 1.0 / (9 * n))
    n_redist = min(n_tail, 100)
    selected = tail_indices[:n_redist]

    T_A = T_base.copy()
    T_B = T_base.copy()

    # T^A: uniform on selected tail
    T_A[selected] = eta / n_redist
    # T^B: exponential decay on selected tail
    half = n_redist // 2
    T_B[selected[:half]] = eta / half
    decay = np.exp(-np.arange(n_redist - half) * 2.0)
    decay /= decay.sum()
    remaining = eta - T_B[selected[:half]].sum()
    T_B[selected[half:]] = remaining * decay

    # Normalize
    T_A = T_A / T_A.sum()
    T_B = T_B / T_B.sum()

    # Full-KL difference (Λ)
    p_star = T_base.copy()
    p_star[p_star == 0] = 1e-15
    kl_A = np.sum(p_star * np.log(p_star / np.maximum(T_A, 1e-15)))
    kl_B = np.sum(p_star * np.log(p_star / np.maximum(T_B, 1e-15)))
    Lambda = abs(kl_A - kl_B)

    # Truncated KL (head only) — only where all distributions are > 0
    head_valid = (p_star[head_indices] > 0) & (T_A[head_indices] > 0) & (T_B[head_indices] > 0)
    hi = head_indices[head_valid]
    trunc_kl_A = np.sum(p_star[hi] * np.log(p_star[hi] / T_A[hi]))
    trunc_kl_B = np.sum(p_star[hi] * np.log(p_star[hi] / T_B[hi]))

    print(f"  Full-KL(T^A) = {kl_A:.4f}")
    print(f"  Full-KL(T^B) = {kl_B:.4f}")
    print(f"  Λ = |KL_A - KL_B| = {Lambda:.4f}")
    print(f"  |Δ trunc-KL| = {abs(trunc_kl_A - trunc_kl_B):.6f} (should be ≈ 0)")

    # Generate samples and permutation test
    np.random.seed(42)
    samples_A = np.random.choice(V, size=n, p=T_A)
    samples_B = np.random.choice(V, size=n, p=T_B)

    counts_A = np.bincount(samples_A, minlength=V)
    counts_B = np.bincount(samples_B, minlength=V)
    test_stat = float(np.sum((counts_A - counts_B) ** 2) / (counts_A + counts_B + 1).sum())

    # Permutation test
    combined = np.concatenate([samples_A, samples_B])
    n_perm = 1000
    perm_stats = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        pA = combined[:n]
        pB = combined[n:]
        cA = np.bincount(pA, minlength=V)
        cB = np.bincount(pB, minlength=V)
        stat = float(np.sum((cA - cB) ** 2) / (cA + cB + 1).sum())
        perm_stats.append(stat)

    p_value = np.mean(np.array(perm_stats) >= test_stat)
    print(f"  Permutation test p-value: {p_value:.3f} (target > 0.3)")

    # Hellinger distance
    h2 = 0.5 * np.sum((np.sqrt(T_A) - np.sqrt(T_B)) ** 2)
    h2_n = n * h2
    tv_bound = np.sqrt(2 * h2_n)
    print(f"  h²(T^A, T^B) = {h2:.6f}")
    print(f"  n·h² = {h2_n:.4f}")
    print(f"  TV bound = {tv_bound:.4f} (target ≤ 0.48)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    sort_idx = np.argsort(T_base)[::-1][:200]
    ax.plot(T_A[sort_idx], label="T^A", alpha=0.7)
    ax.plot(T_B[sort_idx], label="T^B", alpha=0.7, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Token rank (top 200)", fontsize=13)
    ax.set_ylabel("Probability (log scale)", fontsize=13)
    ax.set_title("Two-Point Construction: T^A vs T^B", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(np.array(perm_stats), bins=50, color="steelblue", alpha=0.7,
             edgecolor="white", density=True)
    ax2.axvline(test_stat, color="red", linewidth=2,
                label=f"Observed (p={p_value:.3f})")
    ax2.set_xlabel("Test statistic", fontsize=13)
    ax2.set_ylabel("Density", fontsize=13)
    ax2.set_title(f"Permutation Test (p={p_value:.3f}, target > 0.3)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2d_twopoint_synth.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "Lambda": Lambda,
        "p_value": p_value,
        "h2": float(h2),
        "tv_bound": float(tv_bound),
        "trunc_kl_diff": float(abs(trunc_kl_A - trunc_kl_B)),
    }


# ══════════════════════════════════════════════════════════════
#  Section E: V_eff Corollary
# ══════════════════════════════════════════════════════════════

def section_e_corollary(input_ids, attn_mask, output_dir):
    """Verify V_eff sharp transition between sufficiency and necessity thresholds."""
    print("\n═══ Section E: V_eff Corollary ═══")

    all_tokens = input_ids[attn_mask > 0]
    N_total = all_tokens.shape[0]
    token_counts = torch.bincount(all_tokens).float().numpy()
    V = len(token_counts)
    delta = 0.05
    n = N_total

    sufficiency_thresh = np.log(V / delta) / n
    necessity_thresh = 1.0 / (4 * n)

    print(f"  n={n}, V={V}")
    print(f"  Sufficiency threshold: {sufficiency_thresh:.6f}")
    print(f"  Necessity threshold:   {necessity_thresh:.6f}")

    thresholds = np.logspace(
        np.log10(max(necessity_thresh * 0.1, 1e-8)),
        np.log10(sufficiency_thresh * 100 + 1e-8),
        50,
    )

    v_eff_sizes = []
    for thresh in thresholds:
        v_eff = (token_counts >= thresh * n).sum()
        v_eff_sizes.append(int(v_eff))

    # Relative estimation error
    non_zero = token_counts[token_counts > 0]
    T_vals = non_zero / n
    rel_errors = np.sqrt((1 - T_vals) / (n * T_vals))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.loglog(thresholds, v_eff_sizes, color="crimson", lw=2)
    ax.axvline(sufficiency_thresh, color="green", linestyle="--", lw=2,
               label=f"Sufficiency: log(V/δ)/n = {sufficiency_thresh:.2e}")
    ax.axvline(necessity_thresh, color="red", linestyle="--", lw=2,
               label=f"Necessity: 1/(4n) = {necessity_thresh:.2e}")
    ax.axvspan(necessity_thresh, sufficiency_thresh, alpha=0.1, color="orange",
               label="Transition zone")
    ax.set_xlabel("Threshold (probability)", fontsize=13)
    ax.set_ylabel("V_eff (tokens above threshold)", fontsize=13)
    ax.set_title("Effective Vocabulary: Sharp Transition", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    ax2 = axes[1]
    ax2.loglog(T_vals, rel_errors, ".", color="steelblue", alpha=0.3, markersize=2)
    ax2.axvline(necessity_thresh, color="red", linestyle="--", lw=2,
                label=f"1/(4n) = {necessity_thresh:.2e}")
    T_plot = np.logspace(np.log10(T_vals.min()), np.log10(T_vals.max()), 200)
    ax2.loglog(T_plot, 1 / np.sqrt(n * T_plot), "--", color="gray", alpha=0.5,
               label="Theory: 1/√(nT)")
    ax2.set_xlabel("Token probability T_i", fontsize=13)
    ax2.set_ylabel("Relative estimation error", fontsize=13)
    ax2.set_title("Estimability vs Token Probability", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2e_corollary_synth.png", dpi=150, bbox_inches="tight")
    plt.close()

    v_eff_suff = (token_counts >= sufficiency_thresh * n).sum()
    v_eff_nec = (token_counts >= necessity_thresh * n).sum()
    print(f"  V_eff at sufficiency: {v_eff_suff}")
    print(f"  V_eff at necessity:   {v_eff_nec}")

    return {
        "sufficiency_thresh": float(sufficiency_thresh),
        "necessity_thresh": float(necessity_thresh),
        "v_eff_sufficiency": int(v_eff_suff),
        "v_eff_necessity": int(v_eff_nec),
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"

    # ── Load model and data ──
    print("[exp2-synth] Loading model and data...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)
    texts = load_text_samples(args.n_samples, max_length=4096, seed=args.seed)
    input_ids, attn_mask = tokenize_batch(tokenizer, texts, args.max_length)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    N, T = input_ids.shape
    print(f"[exp2-synth] Data: N={N}, T={T}")

    # ── Compute teacher logits (clean) ──
    print("[exp2-synth] Computing teacher (clean) logits...")
    teacher_logits = torch.zeros(N, T - 1, dtype=torch.float32)
    t0 = time.time()
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        ids = input_ids[start:end]
        mask = attn_mask[start:end]
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        # Gather target-token logits
        targets = ids[:, 1:]
        gathered = logits.float()[:, :-1, :].gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        teacher_logits[start:end] = gathered.cpu()
        if (start // args.batch_size) % 50 == 0:
            print(f"  [{100*start/N:.0f}%] {time.time()-t0:.1f}s")
    print(f"  Teacher logits done in {time.time()-t0:.1f}s")

    # ── Compute token frequencies ──
    all_tokens = input_ids[attn_mask > 0].cpu()
    token_counts = torch.bincount(all_tokens)

    # ── Inject heteroscedastic noise ──
    print(f"\n[exp2-synth] Injecting heteroscedastic noise: "
          f"σ₀={args.sigma0}, c={args.c_param}")
    targets = input_ids[:, 1:].cpu()
    valid_mask = attn_mask[:, 1:].cpu() > 0

    # Per-position noise std: σ(n_i) = √(σ₀² + c/n_i)
    n_per_pos = token_counts[targets.clamp(max=len(token_counts)-1)].float()
    n_per_pos = n_per_pos.clamp(min=1)
    sigma_per_pos = torch.sqrt(args.sigma0**2 + args.c_param / n_per_pos)

    # Generate noise
    noise = sigma_per_pos * torch.randn_like(teacher_logits)
    noise[~valid_mask] = 0  # no noise on padding

    student_logits = teacher_logits + noise
    base_noise = noise.clone()

    print(f"  Noise stats: mean={noise[valid_mask].mean():.4f}, "
          f"std={noise[valid_mask].std():.4f}")
    print(f"  σ₀²={args.sigma0**2:.4f}, c={args.c_param:.2f}")
    print(f"  Expected mean Var ≈ {args.sigma0**2 + args.c_param / n_per_pos[valid_mask].mean():.4f}")

    # Free model
    free_model(model)

    # ── Section A: Residual variance ──
    sigma0_sq_fit, c_fit, _, _ = section_a_residuals(
        input_ids.cpu(), attn_mask.cpu(), teacher_logits, student_logits,
        args.sigma0, args.c_param, args.output_dir
    )

    # ── Section B: n-Sweep convergence ──
    results_b, zero_counts = section_b_nsweep(
        input_ids.cpu(), attn_mask.cpu(), teacher_logits, base_noise,
        token_counts, args.sigma0, args.c_param, args.output_dir, device,
        seed=args.seed
    )

    # ── Section D: Two-point ──
    twopoint = section_d_twopoint(input_ids.cpu(), attn_mask.cpu(), args.output_dir)

    # ── Section E: V_eff corollary ──
    corollary = section_e_corollary(input_ids.cpu(), attn_mask.cpu(), args.output_dir)

    # ── Save ──
    save_data = {
        "model": args.model,
        "noise_params": {"sigma0": args.sigma0, "c": args.c_param},
        "section_a": {"sigma0_sq_fit": sigma0_sq_fit, "c_fit": c_fit,
                      "sigma0_sq_true": args.sigma0**2, "c_true": args.c_param},
        "section_b": results_b,
        "section_b_zero_counts": zero_counts,
        "section_d": twopoint,
        "section_e": corollary,
    }
    with open(f"{args.output_dir}/exp2_synth_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp2-synth] Results saved to {args.output_dir}/exp2_synth_results.json")
    print("[exp2-synth] Done!")


if __name__ == "__main__":
    main()
