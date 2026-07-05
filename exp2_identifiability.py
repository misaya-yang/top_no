#!/usr/bin/env python3
"""
Experiment 2: Identifiability Gap + Effective Vocabulary + Tail Impossibility
===============================================================================
Tests Theorems X.2: Head estimation on V_eff (Thm 3), tail non-estimability
(Thm 4), full-KL minimax-inconsistency (Thm 5), and the corollary.

Sections:
  A. Teacher-student residual analysis  (original + Var(r)=σ₀²+c/n)
  B. Subsample n-sweep + full-KL vs truncated-KL convergence  (Protocol B.1)
  C. Full-KL instability (zero-count) metric
  D. Two-point demonstration  (Protocol B.2 / Thm 4, 5)
  E. V_eff corollary: sharp transition at 1/(4n)
"""
import argparse, os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

from data_utils import load_text_samples, tokenize_batch, load_model_and_tokenizer, free_model

TEACHER_CKPT = "./results/exp2_teacher_logits.pt"
STUDENT_CKPT = "./results/exp2_student_logits.pt"
DATA_CKPT = "./results/exp2_data.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--student", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Phase 1/2: Teacher / Student forward passes
# ══════════════════════════════════════════════════════════════

def run_forward_save_logits(model, input_ids, attn_mask, output_path,
                            batch_size, label):
    """Save target-token logits and full-vocab logits (for KL computation)."""
    device = next(model.parameters()).device
    N, T = input_ids.shape
    V = model.config.vocab_size

    # Save target logits (N, T-1) as float16 — compact (~1MB)
    target_logits = torch.zeros(N, T - 1, dtype=torch.float16)
    # For truncated-KL, compute top-100 log-probs on the fly (not full V).
    # This is sufficient for KL estimation on V_eff tokens.
    TOP_K_LOGPROBS = 100
    n_kl_samples = min(200, N)
    top_logprobs = torch.zeros(n_kl_samples, T - 1, TOP_K_LOGPROBS, dtype=torch.float16)
    top_indices = torch.zeros(n_kl_samples, T - 1, TOP_K_LOGPROBS, dtype=torch.int32)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = input_ids[start:end].to(device)
        mask = attn_mask[start:end].to(device)

        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits

        logits_f32 = logits.float()
        targets = ids[:, 1:]
        gathered = logits_f32[:, :-1, :].gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        target_logits[start:end] = gathered.cpu().half()

        # Save top-K log-probs for KL samples (memory-efficient KL estimation)
        if start < n_kl_samples:
            kl_end = min(end, n_kl_samples)
            kl_batch = kl_end - start
            log_probs = torch.nn.functional.log_softmax(
                logits_f32[:kl_batch, :-1, :], dim=-1
            )
            top_lp, top_idx = log_probs.topk(TOP_K_LOGPROBS, dim=-1)
            top_logprobs[start:kl_end] = top_lp.cpu().half()
            top_indices[start:kl_end] = top_idx.cpu().int()

        if (start // batch_size) % 50 == 0:
            pct = 100 * start / N
            print(f"  [{label}] {pct:5.1f}%  {time.time()-t0:.1f}s")

    torch.save({
        "target_logits": target_logits,
        "top_logprobs": top_logprobs,
        "top_indices": top_indices,
        "n_kl_samples": n_kl_samples,
        "top_k": TOP_K_LOGPROBS,
    }, output_path)
    sz = os.path.getsize(output_path) / 1e6
    print(f"  [{label}] Saved {output_path}  ({sz:.0f}MB)")


# ══════════════════════════════════════════════════════════════
#  Section A: Residual variance vs frequency (original)
# ══════════════════════════════════════════════════════════════

def section_a_residuals(input_ids, attn_mask, output_dir):
    """Original analysis: Var(r) = σ₀² + c/n fitted on binned data."""
    print("\n═══ Section A: Residual Variance vs Token Frequency ═══")

    t_data = torch.load(TEACHER_CKPT, weights_only=True)
    s_data = torch.load(STUDENT_CKPT, weights_only=True)
    teacher_logits = t_data["target_logits"].float()
    student_logits = s_data["target_logits"].float()

    N, Tm1 = teacher_logits.shape
    residuals = teacher_logits - student_logits
    targets = input_ids[:, 1:]

    # Mask out padding
    valid_mask = attn_mask[:, 1:] > 0
    all_tokens = input_ids[attn_mask > 0]
    token_counts = torch.bincount(all_tokens)

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
                            p0=[0.1, 10.0], maxfev=10000)
        sigma0_sq, c_param = popt
    except Exception:
        sigma0_sq, c_param = 0.0, 0.0

    print(f"  Fit: Var(r) = {sigma0_sq:.4f} + {c_param:.2f}/n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.scatter(np.log10(bin_centers + 1), bin_variances,
               s=np.clip(bin_counts / 100, 20, 500),
               c=bin_counts, cmap="YlOrRd", alpha=0.8, edgecolors="gray", lw=0.5)
    plt.colorbar(ax.collections[0], ax=ax, label="Bin count")

    n_fit = np.logspace(np.log10(max(0.1, bin_centers.min())),
                         np.log10(bin_centers.max() + 1), 200)
    ax.plot(np.log10(n_fit + 1), var_model(n_fit, sigma0_sq, c_param),
            "--", color="steelblue", lw=2,
            label=f"σ₀²={sigma0_sq:.3f} + {c_param:.1f}/n")
    ax.axvspan(0, np.log10(11), alpha=0.15, color="red",
               label="IG explosion zone (n < 10)")
    ax.set_xlabel("log₁₀(n + 1)", fontsize=13)
    ax.set_ylabel("Var(r)", fontsize=13)
    ax.set_title("Residual Variance vs Token Frequency", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return sigma0_sq, c_param, bin_centers, bin_variances, fig, axes


# ══════════════════════════════════════════════════════════════
#  Section B: Subsample n-sweep + convergence rates (Protocol B.1)
# ══════════════════════════════════════════════════════════════

def section_b_nsweep(input_ids, attn_mask, output_dir):
    """
    Subsample n from available data, estimate σ² (θ̂) by:
      (a) full-KL: variance of teacher-student residuals over ALL positions
      (b) truncated-KL: variance only over positions with frequent tokens (V_eff)
    Measure convergence rate on log-log axes.
    """
    print("\n═══ Section B: n-Sweep + Full-KL vs Truncated-KL ═══")

    t_data = torch.load(TEACHER_CKPT, weights_only=True)
    s_data = torch.load(STUDENT_CKPT, weights_only=True)

    teacher_logits = t_data["target_logits"].float()  # (N, T-1)
    student_logits = s_data["target_logits"].float()
    N_max = teacher_logits.shape[0]
    Tm1 = teacher_logits.shape[1]

    # Residuals as the basis for σ² estimation
    residuals = teacher_logits - student_logits  # (N_max, T-1)

    # Token frequencies from full dataset
    all_tokens = input_ids[attn_mask > 0]
    token_counts_full = torch.bincount(all_tokens)
    targets = input_ids[:N_max, 1:]
    valid_mask = attn_mask[:N_max, 1:] > 0

    # n-sweep: subsample sizes (Protocol B.1 target: 10^4..10^8, limited by data)
    c_values = [1.0, 2.0, 3.0]
    n_values = [5, 10, 20, 50, 100, 200]
    n_values = [n for n in n_values if n <= N_max]

    results_b = []
    zero_counts = []

    for n_sub in n_values:
        indices = np.random.choice(N_max, n_sub, replace=False)
        indices = np.sort(indices)

        # Recompute token frequencies on subsample
        sub_ids = input_ids[indices]
        sub_mask = attn_mask[indices]
        sub_tokens = sub_ids[sub_mask > 0]
        sub_counts = torch.bincount(sub_tokens)
        sub_targets = sub_ids[:, 1:]
        sub_valid = sub_mask[:, 1:] > 0
        sub_target_flat = sub_targets[sub_valid]

        # Zero-count instability metric
        zeros = (sub_counts[sub_target_flat] == 0).sum().item()
        total = sub_target_flat.shape[0]
        zero_frac = zeros / max(total, 1)
        zero_counts.append({"n": n_sub, "zeros": zeros, "total": total,
                            "zero_frac": zero_frac})

        # Get residuals for subsample
        sub_residuals = residuals[indices]  # (n_sub, T-1)
        sub_vm = valid_mask[indices] if max(indices) < valid_mask.shape[0] else valid_mask[:n_sub]

        # (a) Full-KL θ̂ estimate: Var(residuals) over ALL valid positions
        all_res = sub_residuals[sub_vm]
        full_var_estimate = all_res.var().item() if len(all_res) > 0 else 0

        # (b) Truncated-KL θ̂ estimates for different c
        for c in c_values:
            threshold = max(1, int(n_sub ** (1 - c / 3)))  # v^{-c} ≈ n^{-(c/log_v(n))}
            pos_freqs = sub_counts[sub_targets[sub_valid]]
            keep = pos_freqs >= threshold

            if keep.sum() > 0:
                trunc_res = sub_residuals[sub_vm][keep]
                trunc_var = trunc_res.var().item()
            else:
                trunc_var = 0

            v_eff_size = (sub_counts >= threshold).sum().item()
            results_b.append({
                "n": n_sub, "c": c,
                "full_var": full_var_estimate,
                "trunc_var": trunc_var,
                "v_eff": v_eff_size,
            })

        print(f"  n={n_sub:4d}: zeros={zero_frac:.3f}  "
              f"full_σ²={full_var_estimate:.4f}")

    # ── Plot convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ns = sorted(set(r["n"] for r in results_b))
    for c in c_values:
        trunc_vars = [r["trunc_var"] for r in results_b if r["c"] == c]
        valid_pairs = [(n, v) for n, v in zip(ns, trunc_vars) if v > 0]
        if valid_pairs:
            vns, vvs = zip(*valid_pairs)
            ax.loglog(vns, vvs, "o-", label=f"Truncated (c={c})", markersize=6)

    full_vars = [r["full_var"] for r in results_b if r["c"] == c_values[0]]
    valid_full = [(n, v) for n, v in zip(ns, full_vars) if v > 0]
    if valid_full:
        fns, fvs = zip(*valid_full)
        ax.loglog(fns, fvs, "s--", color="red", label="Full (all tokens)", markersize=6)
        # Reference -1/2 slope line
        ref_ns = np.array(fns, dtype=float)
        ref_vs = np.array(fvs, dtype=float)
        ref_line = ref_vs[0] * (ref_ns[0] / ref_ns) ** 0.5
        ax.loglog(ref_ns, ref_line, ":", color="gray", alpha=0.5,
                  label="Reference: n^{-1/2}")

    ax.set_xlabel("n (subsample size)", fontsize=13)
    ax.set_ylabel("σ² estimate (Var of residuals)", fontsize=13)
    ax.set_title("Convergence: Full vs Truncated σ² Estimate", fontsize=14)
    ax.legend(fontsize=10)
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
    plt.savefig(f"{output_dir}/fig2b_nsweep_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fit log-log slope for truncated σ²
    for c in c_values:
        trunc_vars = np.array([r["trunc_var"] for r in results_b if r["c"] == c])
        ns_arr = np.array(ns, dtype=float)
        valid = (trunc_vars > 0) & (ns_arr > 0)
        if valid.sum() >= 2:
            slope, _ = np.polyfit(np.log(ns_arr[valid]), np.log(trunc_vars[valid]), 1)
            print(f"  Truncated σ² (c={c}) log-log slope: {slope:.3f} "
                  f"(target: -1.0 for variance, -0.5 for σ)")

    return results_b, zero_counts


# ══════════════════════════════════════════════════════════════
#  Section D: Two-point demonstration (Protocol B.2 / Thm 4, 5)
# ══════════════════════════════════════════════════════════════

def section_d_twopoint(input_ids, attn_mask, output_dir):
    """
    Construct T^A, T^B as in Theorem 5: identical on V_eff but different
    tail profiles. Verify a permutation test cannot distinguish them
    (p > 0.3) while full-KL plug-ins differ by ≥ Λ.
    """
    print("\n═══ Section D: Two-Point Demonstration ═══")

    # Use token counts from data to construct realistic T^A, T^B
    all_tokens = input_ids[attn_mask > 0]
    N_total = all_tokens.shape[0]
    token_counts = torch.bincount(all_tokens).float()
    V = token_counts.shape[0]

    # Base distribution: empirical frequencies
    T_base = token_counts / N_total
    T_base = T_base.numpy()

    # n draws (simulate sample budget)
    n = N_total

    # Identify tail tokens: T_i ≤ 1/(4n)
    tail_threshold = 1.0 / (4 * n)
    tail_mask = (T_base > 0) & (T_base <= tail_threshold)
    tail_indices = np.where(tail_mask)[0]
    head_indices = np.where(~tail_mask)[0]

    # Tail mass
    tail_mass = T_base[tail_indices].sum()
    n_tail = len(tail_indices)

    print(f"  V={V}, n={n}, tail tokens={n_tail}, tail mass={tail_mass:.6f}")

    if n_tail < 10:
        print("  Too few tail tokens for two-point construction. Skipping.")
        return None

    # Construct T^A and T^B:
    # Identical on head, redistribute tail mass differently
    eta = min(tail_mass, 1.0 / (9 * n))  # η ≤ 1/(9n) as in Theorem 5

    T_A = T_base.copy()
    T_B = T_base.copy()

    # Redistribute η mass among tail tokens differently
    # T^A: uniform on tail
    # T^B: concentrate on half the tail tokens (exponentially smaller on others)
    n_redist = min(n_tail, 100)
    selected = tail_indices[:n_redist]

    # T^A: spread η uniformly
    T_A[selected] = eta / n_redist
    T_A[tail_indices[n_redist:]] = (tail_mass - eta) / max(len(tail_indices) - n_redist, 1)

    # T^B: concentrate η on first half, exponentially decay on second half
    half = n_redist // 2
    T_B[selected[:half]] = eta / half
    # Second half: very small
    remaining_mass = tail_mass - eta
    decay = np.exp(-np.arange(n_redist - half) * 2.0)
    decay /= decay.sum()
    T_B[selected[half:]] = remaining_mass * decay
    T_B[tail_indices[n_redist:]] = 1e-12

    # Normalize both to sum to 1
    T_A = T_A / T_A.sum()
    T_B = T_B / T_B.sum()

    # Compute full-KL difference (the Λ gap)
    # KL = Σ p* · log(p*/T)
    # Use teacher as p* (approximate with T_base)
    p_star = T_base.copy()
    p_star[p_star == 0] = 1e-15

    kl_A = np.sum(p_star * np.log(p_star / np.maximum(T_A, 1e-15)))
    kl_B = np.sum(p_star * np.log(p_star / np.maximum(T_B, 1e-15)))
    Lambda = abs(kl_A - kl_B)
    print(f"  Full-KL(T^A) = {kl_A:.4f}")
    print(f"  Full-KL(T^B) = {kl_B:.4f}")
    print(f"  Λ = |KL_A - KL_B| = {Lambda:.4f}")

    # Truncated KL (head only): should be identical
    trunc_kl_A = np.sum(p_star[head_indices] *
                        np.log(p_star[head_indices] / T_A[head_indices]))
    trunc_kl_B = np.sum(p_star[head_indices] *
                        np.log(p_star[head_indices] / T_B[head_indices]))
    print(f"  Truncated-KL(T^A) = {trunc_kl_A:.4f}")
    print(f"  Truncated-KL(T^B) = {trunc_kl_B:.4f}")
    print(f"  |Δ trunc-KL| = {abs(trunc_kl_A - trunc_kl_B):.6f} (should be ≈ 0)")

    # Generate samples from T^A and T^B
    np.random.seed(42)
    samples_A = np.random.choice(V, size=n, p=T_A)
    samples_B = np.random.choice(V, size=n, p=T_B)

    # Permutation test: can we distinguish the two sample sets?
    # Count frequency vectors and compare
    counts_A = np.bincount(samples_A, minlength=V)
    counts_B = np.bincount(samples_B, minlength=V)
    test_stat_observed = np.sum((counts_A - counts_B) ** 2) / (counts_A + counts_B + 1)

    # Permutation test (1000 permutations)
    combined = np.concatenate([samples_A, samples_B])
    n_perm = 1000
    perm_stats = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_A = combined[:n]
        perm_B = combined[n:]
        cA = np.bincount(perm_A, minlength=V)
        cB = np.bincount(perm_B, minlength=V)
        perm_stats.append(np.sum((cA - cB) ** 2) / (cA + cB + 1))

    p_value = np.mean(np.array(perm_stats) >= test_stat_observed)
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

    # Left: T^A vs T^B (sorted by T_base frequency)
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

    # Right: permutation test histogram
    ax2 = axes[1]
    ax2.hist(perm_stats, bins=50, color="steelblue", alpha=0.7, edgecolor="white",
             density=True)
    ax2.axvline(test_stat_observed, color="red", linewidth=2,
                label=f"Observed (p={p_value:.3f})")
    ax2.set_xlabel("Test statistic", fontsize=13)
    ax2.set_ylabel("Density", fontsize=13)
    ax2.set_title(f"Permutation Test (p={p_value:.3f}, target > 0.3)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2d_twopoint.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "Lambda": Lambda,
        "p_value": p_value,
        "h2": float(h2),
        "tv_bound": float(tv_bound),
        "trunc_kl_diff": float(abs(trunc_kl_A - trunc_kl_B)),
    }


# ══════════════════════════════════════════════════════════════
#  Section E: V_eff corollary verification
# ══════════════════════════════════════════════════════════════

def section_e_corollary(input_ids, attn_mask, output_dir):
    """
    Verify the corollary: tokens above v^{-c} ≈ log(V/δ)/n are jointly
    estimable; tokens below 1/(4n) are individually non-estimable.
    Plot the sharp transition.
    """
    print("\n═══ Section E: V_eff Corollary ═══")

    all_tokens = input_ids[attn_mask > 0]
    N_total = all_tokens.shape[0]
    token_counts = torch.bincount(all_tokens).float().numpy()
    V = len(token_counts)
    delta = 0.05
    n = N_total

    # Critical thresholds
    sufficiency_thresh = np.log(V / delta) / n  # v^{-c} ≈ log(V/δ)/n
    necessity_thresh = 1.0 / (4 * n)            # Theorem 4: T_i ≤ 1/(4n)

    print(f"  n={n}, V={V}")
    print(f"  Sufficiency threshold: {sufficiency_thresh:.6f}")
    print(f"  Necessity threshold:   {necessity_thresh:.6f}")

    # Compute V_eff for a range of thresholds
    thresholds = np.logspace(
        np.log10(max(necessity_thresh * 0.1, 1e-8)),
        np.log10(sufficiency_thresh * 100 + 1e-8),
        50,
    )

    v_eff_sizes = []
    for thresh in thresholds:
        v_eff = (token_counts >= thresh * n).sum()
        v_eff_sizes.append(int(v_eff))

    # Per-token estimation error (proxy: variance of frequency estimate)
    # For each token, Var(hat_T_i) = T_i(1-T_i)/n ≈ T_i/n
    # Relative error: std(hat_T_i) / T_i = sqrt((1-T_i)/(n*T_i))
    non_zero = token_counts[token_counts > 0]
    T_vals = non_zero / n
    rel_errors = np.sqrt((1 - T_vals) / (n * T_vals))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: V_eff vs threshold with critical lines
    ax = axes[0]
    ax.loglog(thresholds, v_eff_sizes, color="crimson", lw=2)
    ax.axvline(sufficiency_thresh, color="green", linestyle="--", lw=2,
               label=f"Sufficiency: log(V/δ)/n = {sufficiency_thresh:.2e}")
    ax.axvline(necessity_thresh, color="red", linestyle="--", lw=2,
               label=f"Necessity: 1/(4n) = {necessity_thresh:.2e}")
    ax.axvspan(necessity_thresh, sufficiency_thresh, alpha=0.1, color="orange",
               label="Transition zone (log factors)")
    ax.set_xlabel("Threshold (probability)", fontsize=13)
    ax.set_ylabel("V_eff (tokens above threshold)", fontsize=13)
    ax.set_title("Effective Vocabulary: Sharp Transition (Corollary)", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # Right: relative estimation error vs token probability
    ax2 = axes[1]
    ax2.loglog(T_vals, rel_errors, ".", color="steelblue", alpha=0.3, markersize=2)
    ax2.axvline(necessity_thresh, color="red", linestyle="--", lw=2,
                label=f"1/(4n) = {necessity_thresh:.2e}")
    # Theoretical curve: sqrt(1/(n*T))
    T_plot = np.logspace(np.log10(T_vals.min()), np.log10(T_vals.max()), 200)
    ax2.loglog(T_plot, 1 / np.sqrt(n * T_plot), "--", color="gray", alpha=0.5,
               label="Theory: 1/√(nT)")
    ax2.set_xlabel("Token probability T_i", fontsize=13)
    ax2.set_ylabel("Relative estimation error", fontsize=13)
    ax2.set_title("Estimability vs Token Probability", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2e_corollary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # V_eff at sufficiency threshold
    v_eff_suff = (token_counts >= sufficiency_thresh * n).sum()
    v_eff_nec = (token_counts >= necessity_thresh * n).sum()
    print(f"  V_eff at sufficiency: {v_eff_suff}")
    print(f"  V_eff at necessity:   {v_eff_nec}")
    print(f"  Transition width:     {v_eff_nec - v_eff_suff} tokens")

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

    # ── Data preparation ──
    if os.path.exists(DATA_CKPT):
        print("[exp2] Loading cached data")
        data = torch.load(DATA_CKPT, weights_only=True)
        input_ids, attn_mask = data["input_ids"], data["attn_mask"]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = load_text_samples(args.n_samples, max_length=4096, seed=args.seed)
        input_ids, attn_mask = tokenize_batch(tokenizer, texts, args.max_length)
        torch.save({"input_ids": input_ids, "attn_mask": attn_mask}, DATA_CKPT)

    N, T = input_ids.shape
    print(f"[exp2] Data: N={N}, T={T}")

    # ── Phase 1: Teacher ──
    if not os.path.exists(TEACHER_CKPT):
        print("\n═══ Phase 1: Teacher ═══")
        model, _ = load_model_and_tokenizer(args.teacher, dtype=torch.float16)
        run_forward_save_logits(model, input_ids, attn_mask,
                                TEACHER_CKPT, args.batch_size, "teacher")
        free_model(model)

    # ── Phase 2: Student ──
    if not os.path.exists(STUDENT_CKPT):
        print("\n═══ Phase 2: Student ═══")
        model, _ = load_model_and_tokenizer(args.student, dtype=torch.float16)
        run_forward_save_logits(model, input_ids, attn_mask,
                                STUDENT_CKPT, args.batch_size, "student")
        free_model(model)

    # ── Section A: Residual variance ──
    sigma0_sq, c_param, _, _, fig_a, axes_a = section_a_residuals(
        input_ids, attn_mask, args.output_dir
    )

    # ── Section B: n-sweep convergence ──
    results_b, zero_counts = section_b_nsweep(
        input_ids, attn_mask, args.output_dir
    )

    # ── Section D: Two-point demonstration ──
    twopoint = section_d_twopoint(input_ids, attn_mask, args.output_dir)

    # ── Section E: V_eff corollary ──
    corollary = section_e_corollary(input_ids, attn_mask, args.output_dir)

    # ── Add V_eff panel to Section A figure ──
    thresholds = list(range(1, 101))
    all_tokens = input_ids[attn_mask > 0]
    token_counts = torch.bincount(all_tokens).float().numpy()
    targets = input_ids[:, 1:]
    valid_mask = attn_mask[:, 1:] > 0
    unique_tokens = np.unique(targets[valid_mask].numpy())
    all_freqs = token_counts[unique_tokens.astype(int)]

    v_eff_list = []
    ig_bounds = []
    t_data = torch.load(TEACHER_CKPT, weights_only=True)
    s_data = torch.load(STUDENT_CKPT, weights_only=True)
    residuals = (t_data["target_logits"].float() - s_data["target_logits"].float())
    r_flat = residuals[valid_mask].numpy()
    n_flat = token_counts[targets[valid_mask].numpy().astype(int)]

    for thresh in thresholds:
        v_eff = int((all_freqs >= thresh).sum())
        v_eff_list.append(v_eff)
        mask = n_flat >= thresh
        ig_bounds.append(float(np.var(r_flat[mask])) if mask.sum() > 0 else 0.0)

    ax2 = axes_a[1]
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(thresholds, v_eff_list, color="crimson", lw=2,
                   label="V_eff")
    l2, = ax2_twin.plot(thresholds, ig_bounds, color="steelblue", lw=2,
                        label="IG bound (Var)")
    ax2.set_xlabel("Frequency threshold", fontsize=13)
    ax2.set_ylabel("V_eff", color="crimson", fontsize=13)
    ax2_twin.set_ylabel("IG bound", color="steelblue", fontsize=13)
    ax2.set_title("Effective Vocab vs Identifiability Gap", fontsize=14)
    ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig2_identifiability_gap.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save ──
    save_data = {
        "teacher": args.teacher,
        "student": args.student,
        "n_samples": N,
        "max_length": T,
        "section_a": {"sigma0_sq": sigma0_sq, "c_param": c_param},
        "section_b": results_b,
        "section_b_zero_counts": zero_counts,
        "section_d": twopoint,
        "section_e": corollary,
    }
    with open(f"{args.output_dir}/exp2_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp2] All results saved to {args.output_dir}/exp2_results.json")
    print("[exp2] Done!")


if __name__ == "__main__":
    main()
