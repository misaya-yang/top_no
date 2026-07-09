"""Shared decoding samplers for experiment scripts.

The truncation rules operate on raw model logits. Temperature is applied only
after the candidate set is selected, so logit-space margin rules keep a stable
raw-logit interpretation across temperatures.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


LEGACY_NU_STRATEGIES = {"nu_topp_floor", "nu_entropy", "nu_mathboost"}


def _require_token_freq(token_freq_table: torch.Tensor | None, strategy: str) -> torch.Tensor:
    if token_freq_table is None:
        raise ValueError(f"{strategy} requires token_freq_table")
    return token_freq_table


def _require_legacy_strategy_enabled(strategy: str, kwargs: dict[str, Any]) -> None:
    if not kwargs.get("legacy", False):
        raise ValueError(
            f"{strategy} is a deprecated legacy strategy. Pass legacy=True only "
            "when reproducing archived experiments; do not use it in paper pipelines."
        )


def _top_p_keep_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Return the standard nucleus keep mask, including the crossing token."""
    if p <= 0:
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, logits.argmax(dim=-1, keepdim=True), True)
        return keep
    if p >= 1:
        return torch.ones_like(logits, dtype=torch.bool)

    probs = F.softmax(logits, dim=-1)
    sorted_idx = torch.argsort(logits, dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, sorted_idx)
    cum_probs = sorted_probs.cumsum(dim=-1)
    remove = cum_probs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_keep = ~remove
    keep = torch.zeros_like(sorted_keep)
    keep.scatter_(-1, sorted_idx, sorted_keep)
    return keep


def _nu_margin(
    logits: torch.Tensor,
    token_freq_table: torch.Tensor,
    kappa: float,
    m0: float,
) -> torch.Tensor:
    n_i = token_freq_table.to(logits.device).float().unsqueeze(0).expand_as(logits)
    return m0 + kappa / torch.sqrt(n_i + 1)


def get_keep_mask(
    logits: torch.Tensor,
    strategy: str,
    token_freq_table: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Return the boolean candidate-set mask for batched raw logits.

    `nu_topp_floor`, `nu_entropy`, and `nu_mathboost` are archived repair
    variants from the retired hypothesis-testing framing. They require
    `legacy=True` and should not appear in current paper pipelines.
    """
    if strategy == "greedy":
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, logits.argmax(dim=-1, keepdim=True), True)

    elif strategy == "top_k":
        k = max(int(kwargs.get("k", 50)), 1)
        k = min(k, logits.shape[-1])
        top_values, _ = logits.topk(k, dim=-1)
        threshold = top_values[..., -1:].expand_as(logits)
        keep = logits >= threshold

    elif strategy == "top_p":
        keep = _top_p_keep_mask(logits, kwargs.get("p", 0.95))

    elif strategy == "min_p":
        p_min = kwargs.get("p_min", 0.05)
        if (
            isinstance(p_min, bool)
            or not isinstance(p_min, (int, float))
            or not math.isfinite(float(p_min))
            or p_min > 1
        ):
            raise ValueError("p_min must be a finite number no greater than 1")
        if p_min <= 0:
            keep = torch.ones_like(logits, dtype=torch.bool)
        else:
            s_max = logits.max(dim=-1, keepdim=True).values
            keep = (s_max - logits) <= -math.log(float(p_min))

    elif strategy == "fixed_margin":
        margin = kwargs.get("margin", 3.0)
        s_max = logits.max(dim=-1, keepdim=True).values
        keep = (s_max - logits) <= margin

    elif strategy == "top_nsigma":
        n_sigma = kwargs.get("n_sigma", 2.0)
        sigma = logits.std(dim=-1, keepdim=True)
        s_max = logits.max(dim=-1, keepdim=True).values
        keep = (s_max - logits) <= n_sigma * sigma

    elif strategy == "nu":
        freqs = _require_token_freq(token_freq_table, strategy)
        s_max = logits.max(dim=-1, keepdim=True).values
        margin = _nu_margin(logits, freqs, kwargs.get("kappa", 10.0), kwargs.get("m0", 3.0))
        keep = (s_max - logits) <= margin

    elif strategy == "conformal_nu":
        freqs = _require_token_freq(token_freq_table, strategy)
        if "q_hat" not in kwargs:
            raise ValueError("conformal_nu requires q_hat")
        kappa = kwargs.get("kappa", 10.0)
        alpha = kwargs.get("alpha", 1.0)
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        s_max = logits.max(dim=-1, keepdim=True).values
        n_i = freqs.to(logits.device).float().unsqueeze(0).expand_as(logits)
        nonconformity = s_max - logits - kappa / torch.sqrt(n_i + alpha)
        keep = nonconformity <= float(kwargs["q_hat"])

    elif strategy == "nu_topp_floor":
        _require_legacy_strategy_enabled(strategy, kwargs)
        freqs = _require_token_freq(token_freq_table, strategy)
        s_max = logits.max(dim=-1, keepdim=True).values
        margin = _nu_margin(logits, freqs, kwargs.get("kappa", 10.0), kwargs.get("m0", 3.0))
        nu_keep = (s_max - logits) <= margin
        topp_keep = _top_p_keep_mask(logits, kwargs.get("p", 0.95))
        keep = nu_keep | topp_keep

    elif strategy == "nu_entropy":
        _require_legacy_strategy_enabled(strategy, kwargs)
        freqs = _require_token_freq(token_freq_table, strategy)
        kappa = kwargs.get("kappa", 10.0)
        m0 = kwargs.get("m0", 3.0)
        s_max = logits.max(dim=-1, keepdim=True).values

        logits_centered = logits - s_max
        log_probs = logits_centered - torch.logsumexp(logits_centered, dim=-1, keepdim=True)
        probs = log_probs.exp()
        safe_log_probs = torch.where(probs > 1e-10, log_probs, torch.zeros_like(log_probs))
        entropy = -(probs * safe_log_probs).sum(dim=-1, keepdim=True).clamp(min=0)
        log_eff_support = torch.log(torch.tensor(100.0, device=logits.device))
        entropy_ratio = (entropy / log_eff_support).clamp(0.05, 1.0)

        margin = _nu_margin(logits, freqs, kappa * entropy_ratio, m0)
        keep = (s_max - logits) <= margin

    elif strategy == "nu_mathboost":
        _require_legacy_strategy_enabled(strategy, kwargs)
        freqs = _require_token_freq(token_freq_table, strategy)
        math_freq = kwargs.get("math_freq_table")
        if math_freq is None:
            raise ValueError("nu_mathboost requires math_freq_table")
        combined_freq = torch.maximum(freqs.to(logits.device).float(), math_freq.to(logits.device).float())
        s_max = logits.max(dim=-1, keepdim=True).values
        margin = _nu_margin(logits, combined_freq, kwargs.get("kappa", 10.0), kwargs.get("m0", 3.0))
        keep = (s_max - logits) <= margin

    else:
        raise ValueError(f"Unknown truncation strategy: {strategy}")

    return keep


def apply_truncation(
    logits: torch.Tensor,
    strategy: str,
    token_freq_table: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Apply a truncation rule to batched raw logits of shape (B, V)."""
    logits = logits.clone()
    keep = get_keep_mask(logits, strategy, token_freq_table=token_freq_table, **kwargs)
    logits[~keep] = float("-inf")
    return logits


def sample_next_tokens(
    raw_logits: torch.Tensor,
    strategy: str,
    strategy_kwargs: dict[str, Any],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample one token per row after raw-logit truncation."""
    if strategy == "greedy":
        return raw_logits.argmax(dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    truncated_logits = apply_truncation(raw_logits, strategy, **strategy_kwargs)
    probs = F.softmax(truncated_logits / temperature, dim=-1)
    row_sums = probs.sum(dim=-1)
    if not torch.isfinite(probs).all() or torch.any(row_sums <= 0):
        raise RuntimeError(
            f"Invalid truncated distribution for strategy={strategy}; "
            "top token should always survive, so this indicates a sampler bug."
        )
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def batch_generate(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    max_new_tokens: int,
    batch_size: int,
    strategy: str,
    strategy_kwargs: dict[str, Any],
    temperature: float = 1.0,
    max_prompt_length: int = 100,
    return_dict: bool = True,
) -> list[dict[str, Any]] | list[list[int]]:
    """EOS-aware batched KV-cache generation with left padding.

    Returned token lists contain only newly generated tokens before EOS; prompt
    padding and after-EOS tokens are excluded from metrics.
    """
    device = next(model.parameters()).device
    all_results: list[dict[str, Any]] | list[list[int]] = []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    for batch_start in range(0, len(prompt_texts), batch_size):
        batch_prompts = prompt_texts[batch_start:batch_start + batch_size]
        bsz = len(batch_prompts)
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        generated = enc["input_ids"].clone()
        gen_mask = enc["attention_mask"].clone()
        past_key_values = None
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        new_token_ids: list[list[int]] = [[] for _ in range(bsz)]
        stopped_eos = [False for _ in range(bsz)]

        for step in range(max_new_tokens):
            if step == 0:
                position_ids = gen_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(gen_mask == 0, 0)
                outputs = model(
                    input_ids=generated,
                    attention_mask=gen_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )
            else:
                position_ids = gen_mask.long().sum(dim=-1, keepdim=True) - 1
                position_ids = position_ids.clamp(min=0)
                outputs = model(
                    input_ids=generated[:, -1:],
                    attention_mask=gen_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            raw_logits = outputs.logits[:, -1, :]

            active = ~finished
            next_tokens = sample_next_tokens(raw_logits, strategy, strategy_kwargs, temperature)
            next_tokens = torch.where(
                active.unsqueeze(-1),
                next_tokens,
                torch.full_like(next_tokens, pad_token_id),
            )

            for row in range(bsz):
                if not active[row]:
                    continue
                token_id = int(next_tokens[row, 0].item())
                if eos_token_id is not None and token_id == eos_token_id:
                    stopped_eos[row] = True
                else:
                    new_token_ids[row].append(token_id)

            if eos_token_id is not None:
                finished |= active & next_tokens.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

            append_mask = active.to(dtype=gen_mask.dtype).unsqueeze(-1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            gen_mask = torch.cat([gen_mask, append_mask], dim=-1)

        if return_dict:
            for row, ids in enumerate(new_token_ids):
                all_results.append({
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                    "tokens": ids,
                    "n_generated_tokens": len(ids),
                    "stopped_eos": stopped_eos[row],
                })
        else:
            all_results.extend(new_token_ids)

    return all_results
