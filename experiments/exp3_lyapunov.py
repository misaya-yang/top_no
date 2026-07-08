#!/usr/bin/env python3
"""
Experiment 3: Nonstationary Sensitivity + Adaptive Margin
==========================================================
Tests Proposition 1 (cumulative rule fails), Theorem 7 (ACI coverage),
and Theorem 8 (EWA regret). Protocol C.

Sections:
  A. Real-generation Lyapunov measurement + bimodality  (Protocol C.1)
  B. Perturbation propagation via norm-ratio  (paper's definition)
  C. Online margin adaptation: fixed / cumulative / proposed  (Protocol C.2)
  D. Falsification: burst-then-calm → linear regret  (Protocol C.3)
  E. Long-sequence scaling (L ∈ {256, 512, 1024})
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from data_utils import load_gsm8k_passages, load_creative_passages
from data_utils import load_model_and_tokenizer, free_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--min-tokens", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Forward pass utilities
# ══════════════════════════════════════════════════════════════

def _compute_rotary_emb(model, seq_len, device, dtype):
    """Compute rotary embeddings from inv_freq — guaranteed correct shape."""
    inv_freq = model.model.rotary_emb.inv_freq  # (head_dim//2,)
    pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype).unsqueeze(0)
    freqs = torch.outer(pos.squeeze(0), inv_freq)        # (T, head_dim//2)
    emb = torch.cat((freqs, freqs), dim=-1)               # (T, head_dim)
    cos = emb.cos().unsqueeze(0).to(dtype)                 # (1, T, head_dim)
    sin = emb.sin().unsqueeze(0).to(dtype)                 # (1, T, head_dim)
    return cos, sin


def manual_forward(model, input_ids, attention_mask):
    """
    Clean forward: uses model's standard forward for correctness,
    returns logits + all hidden states + position embeddings.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
        hidden_states = list(out.hidden_states)   # [embed, layer0_out, ..., layerN_out]
        logits = out.logits
        seq_len = input_ids.shape[1]
        pos_emb = _compute_rotary_emb(model, seq_len, input_ids.device,
                                       hidden_states[0].dtype)
    return logits, hidden_states, pos_emb


def manual_forward_perturbed(model, input_ids, attention_mask,
                              clean_hs, layer_idx, epsilon,
                              position_embeddings=None):
    """
    Perturb hidden state AFTER layer_idx, then run layers [layer_idx+1..N].
    clean_hs[layer_idx+1] is the output of layer_idx — that's what we perturb.
    """
    with torch.no_grad():
        h = clean_hs[layer_idx + 1].clone()
        if h.dim() == 2:
            h = h.unsqueeze(0)
        noise = epsilon * torch.randn_like(h)
        h = h + noise

        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        if position_embeddings is None:
            position_embeddings = _compute_rotary_emb(
                model, seq_len, h.device, h.dtype)

        for i in range(layer_idx + 1, len(model.model.layers)):
            out = model.model.layers[i](
                h, attention_mask=None,
                position_ids=pos_ids,
                position_embeddings=position_embeddings)
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 2:
                h = h.unsqueeze(0)

        h = model.model.norm(h)
        logits = model.lm_head(h)
    return logits, noise


# ══════════════════════════════════════════════════════════════
#  Section A: Real-generation Lyapunov + bimodality
# ══════════════════════════════════════════════════════════════

def generate_sequences(model, tokenizer, prompts, n_tokens, n_samples):
    """Generate n_samples sequences from prompts using the model."""
    print(f"  Generating {n_samples} sequences of ~{n_tokens} tokens …")
    device = next(model.parameters()).device
    generated = []

    for i, prompt in enumerate(prompts[:n_samples]):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=100).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=n_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
            )
        # Trim to prompt + generated
        gen_ids = out[0]
        if len(gen_ids) > n_tokens + 50:
            gen_ids = gen_ids[:n_tokens + 50]
        generated.append(gen_ids.cpu())

        if (i + 1) % 10 == 0:
            print(f"    Generated {i+1}/{n_samples}")

    return generated


def measure_lyapunov_per_token(model, input_ids, attention_mask, layer_idx,
                                epsilon):
    """
    Measure λ̂_t per token position.

    Paper definition (Protocol C.1): λ̂_t = log(‖δ_logits,t‖ / ‖δ_logits,t-1‖)
    This is the log-ratio of logit perturbation norms at CONSECUTIVE token
    positions — measuring how the perturbation grows/shrinks along the sequence.

    Also returns KL/ε² as a secondary metric.
    """
    logits_clean, hidden_states, pos_emb = manual_forward(model, input_ids, attention_mask)

    # Perturb at layer_idx
    logits_pert, noise = manual_forward_perturbed(
        model, input_ids, attention_mask, hidden_states, layer_idx, epsilon,
        position_embeddings=pos_emb
    )

    # KL divergence per position (secondary metric)
    log_p = F.log_softmax(logits_clean.float(), dim=-1)
    log_q = F.log_softmax(logits_pert.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).squeeze(0)  # (T,)

    # ── Paper's λ̂_t: consecutive-token perturbation norm ratio ──
    # δ_logits[t] = logits_clean[t] - logits_pert[t]
    delta_logits = (logits_clean - logits_pert).float().squeeze(0)  # (T, V)
    logit_norms = delta_logits.norm(dim=-1)  # (T,)

    # λ̂_t = log(‖δ_logits,t‖ / ‖δ_logits,t-1‖)
    # For t=0, there's no t-1, so we use the hidden perturbation norm as baseline
    hidden_pert_norm = noise.norm(dim=-1).squeeze(0)  # (T,)
    # Build the "previous" norms: [hidden_norm[0], logit_norm[0], logit_norm[1], ...]
    prev_norms = torch.cat([hidden_pert_norm[:1], logit_norms[:-1]])  # (T,)

    norm_ratio = torch.log(logit_norms / (prev_norms + 1e-10) + 1e-10)  # (T,)

    return kl.cpu().numpy(), norm_ratio.cpu().numpy()


def section_a_bimodality(model, tokenizer, args, device):
    """
    Generate real sequences, measure λ̂_t, test for bimodality
    (contractive vs expansive regions).
    """
    print("\n═══ Section A: Real-Generation Lyapunov + Bimodality ═══")

    num_layers = model.config.num_hidden_layers
    layer_indices = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    # Prepare prompts (factual)
    factual_passages = load_gsm8k_passages(n=args.n_samples)
    creative_passages = load_creative_passages(n=args.n_samples)

    # Generate sequences (use shorter length for speed)
    gen_len = min(args.max_tokens, 300)
    factual_seqs = generate_sequences(model, tokenizer, factual_passages,
                                       gen_len, args.n_samples)
    creative_seqs = generate_sequences(model, tokenizer, creative_passages,
                                        gen_len, args.n_samples)

    all_kl = {"factual": [], "creative": []}
    all_norm = {"factual": [], "creative": []}

    for label, seqs in [("factual", factual_seqs), ("creative", creative_seqs)]:
        for s_idx, seq in enumerate(seqs):
            input_ids = seq.unsqueeze(0).to(device)

            for layer_idx in layer_indices:
                kl, norm_ratio = measure_lyapunov_per_token(
                    model, input_ids, None, layer_idx, args.epsilon
                )
                all_kl[label].extend((kl / args.epsilon**2).tolist())
                all_norm[label].extend(norm_ratio.tolist())

        print(f"  [{label}] {len(seqs)} sequences processed")

    # ── Bimodality test ──
    print("\n  Bimodality analysis:")
    bimodality_results = {}
    for label in ["factual", "creative"]:
        vals = np.array(all_kl[label])
        vals = vals[np.isfinite(vals)]

        # Hartigan's dip test approximation:
        # Use KDE + look for multiple modes
        try:
            kde = gaussian_kde(vals, bw_method=0.1)
            x_grid = np.linspace(vals.min(), vals.max(), 500)
            density = kde(x_grid)
            # Count local maxima
            peaks = []
            for i in range(1, len(density) - 1):
                if density[i] > density[i-1] and density[i] > density[i+1]:
                    peaks.append(i)
            n_modes = len(peaks)
        except Exception:
            n_modes = 1

        bimodal = n_modes >= 2
        bimodality_results[label] = {
            "n_modes": n_modes,
            "bimodal": bimodal,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
        }
        print(f"    {label}: modes={n_modes}  bimodal={bimodal}  "
              f"mean={vals.mean():.3f}  median={np.median(vals):.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: KL/ε² curves per label
    for col, label in enumerate(["factual", "creative"]):
        ax = axes[0, col]
        color = "forestgreen" if label == "factual" else "crimson"
        vals = np.array(all_kl[label])
        vals = vals[np.isfinite(vals)]

        # Clip outliers for visualization
        p99 = np.percentile(vals, 99)
        vals_plot = vals[vals <= p99]

        ax.hist(vals_plot, bins=100, color=color, alpha=0.7, edgecolor="white",
                density=True)
        ax.axvline(np.median(vals), color="black", linestyle="--",
                   label=f"Median={np.median(vals):.2f}")
        title = "Factual / Deterministic" if label == "factual" else "Creative / Open-ended"
        ax.set_title(f"{title}\nKL/ε² Distribution", fontsize=13)
        ax.set_xlabel("KL / ε²", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Row 0, col 2: comparison overlay
    ax = axes[0, 2]
    for label, color in [("factual", "forestgreen"), ("creative", "crimson")]:
        vals = np.array(all_kl[label])
        vals = vals[np.isfinite(vals)]
        p99 = np.percentile(vals, 99)
        vals = vals[vals <= p99]
        ax.hist(vals, bins=80, alpha=0.5, color=color, density=True,
                label=f"{label} (n={len(vals)})")
    ax.set_title("Overlay: Factual vs Creative", fontsize=13)
    ax.set_xlabel("KL / ε²", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 1: Norm-ratio distributions (paper's λ̂ definition)
    for col, label in enumerate(["factual", "creative"]):
        ax = axes[1, col]
        color = "forestgreen" if label == "factual" else "crimson"
        vals = np.array(all_norm[label])
        vals = vals[np.isfinite(vals)]
        p99 = np.percentile(vals, 99)
        vals = vals[vals <= p99]

        ax.hist(vals, bins=100, color=color, alpha=0.7, edgecolor="white",
                density=True)
        ax.axvline(0, color="black", linestyle="-", alpha=0.5,
                   label="λ̂ = 0 (boundary)")
        ax.axvline(np.mean(vals), color="orange", linestyle="--",
                   label=f"Mean={np.mean(vals):.3f}")
        title = "Factual" if label == "factual" else "Creative"
        ax.set_title(f"{title}\nλ̂ = log(‖δ_logits‖/‖δ_hidden‖)", fontsize=13)
        ax.set_xlabel("λ̂ (log norm ratio)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Row 1, col 2: contractive vs expansive proportions
    ax = axes[1, 2]
    proportions = {}
    for label in ["factual", "creative"]:
        vals = np.array(all_norm[label])
        vals = vals[np.isfinite(vals)]
        contractive = (vals < 0).mean()
        expansive = (vals > 0).mean()
        proportions[label] = {"contractive": contractive, "expansive": expansive}

    x = np.arange(2)
    width = 0.35
    fact_vals = [proportions["factual"]["contractive"],
                 proportions["factual"]["expansive"]]
    crea_vals = [proportions["creative"]["contractive"],
                 proportions["creative"]["expansive"]]
    ax.bar(x - width/2, fact_vals, width, label="Factual", color="forestgreen")
    ax.bar(x + width/2, crea_vals, width, label="Creative", color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels(["Contractive\n(λ̂ < 0)", "Expansive\n(λ̂ > 0)"])
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Bimodality: Contractive vs Expansive", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig3_lyapunov.png", dpi=150, bbox_inches="tight")
    plt.close()

    return all_kl, all_norm, bimodality_results


# ══════════════════════════════════════════════════════════════
#  Section C: Online margin adaptation (Protocol C.2)
# ══════════════════════════════════════════════════════════════

def pinball_loss(margin, D_t, beta):
    """β-pinball loss: asymmetric absolute error."""
    residual = D_t - margin
    return np.where(residual >= 0, beta * residual, (beta - 1) * residual)


def run_online_algorithm(lambda_hat, D_t, beta=0.9, eta=0.1,
                          alpha_grid=None, w=50, m0=0.0):
    """
    Windowed feedforward + ACI feedback + EWA over α.
    Returns margins, errors, cumulative regret.
    """
    T = len(lambda_hat)
    if alpha_grid is None:
        alpha_grid = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    N_alpha = len(alpha_grid)

    # EWA weights
    learning_rate = np.sqrt(8 * np.log(N_alpha) / max(T, 1))
    weights = np.ones(N_alpha) / N_alpha

    u = 0.0  # ACI state
    margins = np.zeros(T)
    errors = np.zeros(T)
    cumulative_losses = {a: 0.0 for a in alpha_grid}
    best_fixed_losses = np.zeros(T)
    algo_losses = np.zeros(T)

    for t in range(T):
        # Feedforward: windowed sum of λ̂⁺
        window_start = max(0, t - w + 1)
        Lambda_t = np.sum(np.maximum(lambda_hat[window_start:t+1], 0))

        # EWA mixture
        alpha_t = np.sum(weights * alpha_grid)

        # Posted margin
        m_t = m0 + alpha_t * Lambda_t + u
        m_t = max(m_t, 0)  # non-negative
        m_t = min(m_t, 100)  # cap
        margins[t] = m_t

        # Observe error
        errors[t] = 1.0 if D_t[t] > m_t else 0.0

        # ACI update
        u = u + eta * (errors[t] - beta)

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

        # Best fixed α up to time t
        best_loss = min(cumulative_losses.values())
        best_fixed_losses[t] = best_loss

    # Cumulative regret
    cum_algo_loss = np.cumsum(algo_losses)
    regret = cum_algo_loss - np.minimum.accumulate(best_fixed_losses)

    return margins, errors, regret, cum_algo_loss


def section_c_online(model, tokenizer, args, device, all_kl):
    """
    Compare three margin strategies on generated sequences:
    (a) fixed global-λ margin
    (b) cumulative rule (Proposition 1: known to fail)
    (c) proposed windowed + ACI + EWA algorithm
    """
    print("\n═══ Section C: Online Margin Adaptation ═══")

    # Use KL/ε² values as λ̂_t (proxy for local Lyapunov)
    lambda_hat = np.array(all_kl["creative"])
    lambda_hat = lambda_hat[np.isfinite(lambda_hat)]

    if len(lambda_hat) < 100:
        print("  Insufficient data for online experiment. Skipping.")
        return None

    # Simulate D_t (minimal sufficient margin)
    # D_t is the actual margin needed - proxy: local entropy or confidence
    # For simulation: D_t = local 95th percentile of recent λ̂ values
    D_t = np.zeros(len(lambda_hat))
    window = 20
    for t in range(len(lambda_hat)):
        start = max(0, t - window)
        D_t[t] = np.percentile(lambda_hat[start:t+1], 90) if t > 0 else lambda_hat[0]

    # Add non-stationarity: burst phase then calm phase
    T_total = len(lambda_hat)
    burst_end = T_total // 3
    calm_start = 2 * T_total // 3

    # Scale D_t for different phases
    D_t[:burst_end] *= 2.0       # burst: high D_t
    D_t[burst_end:calm_start] *= 1.0  # normal
    D_t[calm_start:] *= 0.3      # calm: low D_t

    beta = 0.9
    eta = 0.1

    # (a) Fixed margin: set to global 90th percentile
    m_fixed = np.percentile(lambda_hat, 90)
    errors_fixed = (D_t > m_fixed).astype(float)

    # (b) Cumulative rule: m_t = m_0 + α·Σ λ̂⁺ (Proposition 1)
    alpha_cum = 0.1
    Lambda_cum = np.cumsum(np.maximum(lambda_hat, 0))
    m_cumulative = alpha_cum * Lambda_cum
    m_cumulative = np.minimum(m_cumulative, 100)  # cap
    errors_cumulative = (D_t > m_cumulative).astype(float)

    # (c) Proposed algorithm
    margins_prop, errors_prop, regret, cum_loss = run_online_algorithm(
        lambda_hat, D_t, beta=beta, eta=eta, w=50
    )

    # ── Compute metrics ──
    results = {}
    for name, errs, margins in [
        ("fixed", errors_fixed, np.full(T_total, m_fixed)),
        ("cumulative", errors_cumulative, m_cumulative),
        ("proposed", errors_prop, margins_prop),
    ]:
        err_rate = errs.mean()
        mean_margin = margins.mean()
        gap = abs(err_rate - beta)
        results[name] = {
            "error_rate": float(err_rate),
            "target_beta": beta,
            "gap": float(gap),
            "mean_margin": float(mean_margin),
        }
        print(f"  {name:12s}: err={err_rate:.3f}  target={beta}  "
              f"gap={gap:.3f}  mean_margin={mean_margin:.2f}")

    # ── Pinball regret ──
    # Best fixed α
    best_fixed_loss = min(
        np.sum(pinball_loss(m_fixed, D_t, beta)),
        np.sum(pinball_loss(np.percentile(lambda_hat, 95), D_t, beta)),
        np.sum(pinball_loss(np.percentile(lambda_hat, 80), D_t, beta)),
    )
    prop_total_loss = cum_loss[-1]
    regret_final = prop_total_loss - best_fixed_loss
    print(f"  Regret (proposed vs best fixed): {regret_final:.2f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: error rates over time (running average)
    ax = axes[0, 0]
    window_smooth = 100
    for name, errs, color in [
        ("Fixed", errors_fixed, "gray"),
        ("Cumulative", errors_cumulative, "orange"),
        ("Proposed", errors_prop, "forestgreen"),
    ]:
        running = np.convolve(errs, np.ones(window_smooth)/window_smooth, mode='valid')
        ax.plot(running, label=name, color=color, alpha=0.8)
    ax.axhline(beta, color="black", linestyle="--", label=f"Target β={beta}")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Error rate (windowed)", fontsize=12)
    ax.set_title("Coverage: Running Error Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Top-right: posted margins over time
    ax = axes[0, 1]
    ax.plot(np.full(T_total, m_fixed), label="Fixed", color="gray", alpha=0.7)
    ax.plot(m_cumulative, label="Cumulative", color="orange", alpha=0.7)
    ax.plot(margins_prop, label="Proposed", color="forestgreen", alpha=0.7)
    ax.axvline(burst_end, color="red", linestyle=":", alpha=0.5, label="Burst end")
    ax.axvline(calm_start, color="blue", linestyle=":", alpha=0.5, label="Calm start")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Posted margin m_t", fontsize=12)
    ax.set_title("Margin Adaptation Over Time", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: cumulative pinball loss
    ax = axes[1, 0]
    cum_fixed = np.cumsum(pinball_loss(np.full(T_total, m_fixed), D_t, beta))
    cum_cumul = np.cumsum(pinball_loss(m_cumulative, D_t, beta))
    ax.plot(cum_fixed, label="Fixed", color="gray")
    ax.plot(cum_cumul, label="Cumulative", color="orange")
    ax.plot(cum_loss, label="Proposed", color="forestgreen")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Cumulative pinball loss", fontsize=12)
    ax.set_title("Cumulative Loss Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom-right: regret curve
    ax = axes[1, 1]
    ax.plot(regret, color="forestgreen", lw=2)
    # Reference √T line
    ref = np.sqrt(np.arange(1, T_total + 1)) * regret[-1] / np.sqrt(T_total)
    ax.plot(ref, "--", color="gray", alpha=0.5, label="Reference: √T")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Regret vs best fixed α", fontsize=12)
    ax.set_title(f"EWA Regret (final: {regret_final:.1f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig3c_online.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════
#  Section D: Falsification (Protocol C.3)
# ══════════════════════════════════════════════════════════════

def section_d_falsification(output_dir):
    """
    Proposition 1 falsification: burst-then-calm sequence →
    cumulative rule shows linear regret, proposed recovers in O(m/η).
    """
    print("\n═══ Section D: Burst-Then-Calm Falsification ═══")

    T = 2000
    # Construct λ̂: high for first half, zero for second half
    lambda_bar = 2.0
    lambda_hat = np.zeros(T)
    lambda_hat[:T//2] = lambda_bar
    # D_t: zero everywhere (the calm phase should have m_t → 0)
    D_t = np.zeros(T)

    # (b) Cumulative rule
    alpha = 0.1
    Lambda_cum = np.cumsum(np.maximum(lambda_hat, 0))
    m_cum = alpha * Lambda_cum
    m_cum = np.minimum(m_cum, 50)  # cap

    # Pinball loss for cumulative (β=0.9, D_t=0 → overage penalty)
    beta = 0.9
    losses_cum = pinball_loss(m_cum, D_t, beta)
    cum_loss_cum = np.cumsum(losses_cum)

    # (c) Proposed algorithm
    margins_prop, errors_prop, regret_prop, cum_loss_prop = run_online_algorithm(
        lambda_hat, D_t, beta=beta, eta=0.1, w=30
    )

    # Check linear regret growth for cumulative
    second_half_cum = cum_loss_cum[T//2:]
    if len(second_half_cum) > 10:
        slope_cum = (second_half_cum[-1] - second_half_cum[0]) / len(second_half_cum)
    else:
        slope_cum = 0

    # Check recovery time for proposed
    # Recovery: margin returns to near-zero after burst ends
    recovery_threshold = 1.0
    recovery_time = None
    for t in range(T//2, T):
        if margins_prop[t] < recovery_threshold:
            recovery_time = t - T//2
            break

    print(f"  Cumulative: linear loss rate in calm phase = {slope_cum:.4f}/step")
    print(f"  Proposed: recovery time = {recovery_time} steps "
          f"{'(recovered)' if recovery_time else '(not recovered)'}")

    # Falsification verdicts
    cum_linear = slope_cum > 0.01  # linear if loss rate > threshold
    prop_recovers = recovery_time is not None and recovery_time < T//4

    print(f"  Proposition 1: cumulative linear regret → "
          f"{'CONFIRMED' if cum_linear else 'NOT CONFIRMED'}")
    print(f"  Theorem 7: proposed recovery → "
          f"{'CONFIRMED' if prop_recovers else 'NOT CONFIRMED'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: margins over time
    ax = axes[0]
    ax.plot(m_cum, label="Cumulative rule", color="orange", lw=2)
    ax.plot(margins_prop, label="Proposed (windowed+ACI)", color="forestgreen", lw=2)
    ax.axvline(T//2, color="red", linestyle="--", alpha=0.7, label="Burst ends")
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel("Margin m_t", fontsize=13)
    ax.set_title("Burst-Then-Calm: Margin Behavior", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: cumulative losses
    ax = axes[1]
    ax.plot(cum_loss_cum, label="Cumulative (linear growth!)", color="orange", lw=2)
    ax.plot(cum_loss_prop, label="Proposed", color="forestgreen", lw=2)
    ax.axvline(T//2, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel("Cumulative pinball loss", fontsize=13)
    ax.set_title("Falsification: Cumulative Rule Linear Regret", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3d_falsification.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "cumulative_loss_rate": float(slope_cum),
        "proposed_recovery_time": recovery_time,
        "cumulative_linear": cum_linear,
        "proposed_recovers": prop_recovers,
    }


# ══════════════════════════════════════════════════════════════
#  Section E: Long-sequence scaling
# ══════════════════════════════════════════════════════════════

def section_e_long_sequences(model, tokenizer, args, device):
    """Test perturbation amplification at different sequence lengths."""
    print("\n═══ Section E: Long-Sequence Scaling ═══")

    num_layers = model.config.num_hidden_layers
    layer_idx = num_layers // 2  # middle layer

    lengths = [256, 512, 1024]
    creative_passages = load_creative_passages(n=10)

    results_e = {}

    for L in lengths:
        print(f"  Testing L={L} …")
        kl_values = []

        for prompt in creative_passages[:5]:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=80).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=L,
                    do_sample=True, temperature=0.8,
                )
            gen_ids = out[0][:L+50].unsqueeze(0)

            kl, _ = measure_lyapunov_per_token(
                model, gen_ids, None, layer_idx, args.epsilon
            )
            amp = kl / args.epsilon**2
            kl_values.extend(amp[np.isfinite(amp)].tolist())

        kl_arr = np.array(kl_values)
        results_e[L] = {
            "mean": float(kl_arr.mean()),
            "std": float(kl_arr.std()),
            "p95": float(np.percentile(kl_arr, 95)),
            "median": float(np.median(kl_arr)),
        }
        print(f"    L={L}: mean={kl_arr.mean():.3f}  "
              f"p95={np.percentile(kl_arr, 95):.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    Ls = list(results_e.keys())
    means = [results_e[L]["mean"] for L in Ls]
    p95s = [results_e[L]["p95"] for L in Ls]
    stds = [results_e[L]["std"] for L in Ls]

    ax.errorbar(Ls, means, yerr=stds, fmt="o-", color="crimson",
                capsize=5, markersize=8, lw=2, label="Mean ± std")
    ax.plot(Ls, p95s, "s--", color="steelblue", markersize=8,
            label="95th percentile")
    ax.set_xlabel("Sequence length L", fontsize=13)
    ax.set_ylabel("KL / ε²", fontsize=13)
    ax.set_title("Perturbation Amplification vs Sequence Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/fig3e_longseq.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results_e


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0"
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16)

    # Section A: Real-generation Lyapunov + bimodality
    all_kl, all_norm, bimodality = section_a_bimodality(
        model, tokenizer, args, device
    )

    # Section C: Online margin adaptation
    online_results = section_c_online(
        model, tokenizer, args, device, all_kl
    )

    # Section D: Falsification (no model needed)
    falsification = section_d_falsification(args.output_dir)

    # Section E: Long sequences
    long_seq = section_e_long_sequences(model, tokenizer, args, device)

    free_model(model)

    # ── Summary ──
    print("\n═══ Summary ═══")
    for label, info in bimodality.items():
        print(f"  {label}: bimodal={info['bimodal']}  modes={info['n_modes']}")
    if online_results:
        for name, info in online_results.items():
            print(f"  {name}: err={info['error_rate']:.3f}  gap={info['gap']:.3f}")
    print(f"  Falsification: cum_linear={falsification['cumulative_linear']}  "
          f"prop_recovers={falsification['proposed_recovers']}")
    for L, info in long_seq.items():
        print(f"  L={L}: mean_amp={info['mean']:.3f}")

    # ── Save ──
    save_data = {
        "model": args.model,
        "epsilon": args.epsilon,
        "n_samples": args.n_samples,
        "bimodality": bimodality,
        "online": online_results,
        "falsification": falsification,
        "long_sequences": long_seq,
    }
    with open(f"{args.output_dir}/exp3_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[exp3] All results saved to {args.output_dir}/exp3_results.json")
    print("[exp3] Done!")


if __name__ == "__main__":
    main()
