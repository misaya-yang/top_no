# Fable5 Analysis

Source: user-provided Fable5 review pasted on 2026-07-09.

## 1. Is the framing sound and novel enough?

The framing is sound. The novelty, as currently stated, is not.

Viewing truncation decoding as token-level prediction-set construction is correct and clean, but it is not ours. Conformal Nucleus Sampling (Ravfogel et al., ACL Findings 2023) already treats top-p sets as prediction sets and split-calibrates them for target coverage, stratified by entropy bins. Non-Exchangeable Conformal Language Generation (Ulmer et al., EACL Findings 2024) built token-level conformal sets for generation and dealt with exchangeability. Conformal Language Modeling (Quach et al., ICLR 2024) covers sequence level, and a December 2025 arXiv paper is squarely on next-token conformal prediction sets balancing coverage against set efficiency. There is also a TACL survey of conformal prediction for NLP.

Therefore the novelty budget rests on the frequency-indexed margin. The unification `min-p = fixed margin`, `top-nsigma = context-scaled margin`, and `nu = token-feature margin` is useful exposition, not a theorem. Conformal coverage itself is free: split conformal gives target marginal coverage for any measurable score. The real contribution must be either better efficiency or better conditional/frequency-bucket coverage.

## 2. Is the conformal/coverage argument valid?

The mechanics are mostly fine, but the claims need repair:

- Exchangeability: token positions within a document are dependent. The theorem-grade fix is document-level splitting and one randomly sampled position per document for calibration/evaluation.
- Frequency counts: token counts must be estimated from a corpus disjoint from calibration and evaluation, so they are fixed side information.
- Kappa selection: `kappa` needs a dedicated tuning split. Choosing it from calibration/eval feedback voids the guarantee.
- Marginal vs conditional: split conformal gives marginal coverage, not low-frequency conditional coverage.

The most dangerous missing baseline is frequency-Mondrian conformal on the plain margin score `s_max - s_i`, stratified by frequency bucket. If conformal-nu does not beat this on efficiency, frequency is better injected through calibration groups than through the nu score.

## 3. Is the score pointed the right way?

The smoke result with average support around 14,000 tokens is likely structural rather than a small hyperparameter accident. The vocabulary is dominated by rare tokens, so a rule that gives a margin bonus to low-frequency tokens can retain huge parts of the tail.

The method implicitly assumes:

```text
At fixed logit margin, low-frequency tokens are true more often than high-frequency tokens.
```

This needs direct testing. The key diagnostic is a `(margin x frequency)` reliability plot: bucket tokens by logit margin and token frequency, then measure empirical true-token rate per cell. This decides the sign of the frequency term and whether the project premise exists.

## 4. Experimental plan gaps

The current plan is not yet sufficient for ICML/NeurIPS/ICLR:

1. Baselines must be conformalized too: conformal top-p, conformal plain margin (`kappa=0`), frequency-Mondrian plain margin, and conformalized epsilon/eta/typical-family baselines.
2. The gate must compare calibrated methods against calibrated methods at matched `delta`, not calibrated nu against uncalibrated baselines.
3. One model and one dataset is too weak. Need multiple model families, a scale axis, and multiple domains because token frequency is tokenizer-dependent.
4. Main metrics should be coverage-size Pareto, Pareto AUC, size-stratified coverage, and per-frequency-bucket coverage at matched support size.
5. Downstream generation is secondary and should not imply token-level coverage directly improves generation quality.

## 5. Lethal reviewer attacks

1. Conformal Nucleus Sampling already did conformal top-p in 2023; if nu loses to `kappa=0`, there is no contribution.
2. Coverage attainment is not evidence because conformal coverage is generic.
3. Why not Mondrian conformal by frequency bucket?
4. Where is the empirical evidence that logits are frequency-miscalibrated?
5. Exchangeability under within-document dependence.
6. Frequency-table and `kappa` leakage.
7. Pathological support sizes.
8. Tokenizer dependence of `n_i`.
9. Missing epsilon/eta/typical baselines.
10. Legacy claims about noise-channel identification, mathboost, GSM8K, and Distinct-n.

## Narrower defensible paper

Suggested title:

```text
Do Language Model Logits Need Frequency Correction? A Conformal Audit of Truncation Sampling.
```

Suggested contribution stack:

1. Framework/lemma: common truncation rules can be written as support-construction/logit-margin rules and split-calibrated, with a precise document-level split protocol.
2. Measurement: systematic measurement of frequency-conditional coverage failures across model families, scales, tokenizers, and domains.
3. Method conditional on data: whichever of nu-margin with empirically correct sign or frequency-Mondrian calibration wins the Pareto comparison.

This framing is robust even if the naive "widen margins for rare tokens" intuition fails.

## Prioritized action plan

1. Run the `(margin x frequency)` reliability diagnostic on GPT-2 and one mid-size model before the GPU queue.
2. Implement conformal plain margin (`kappa=0`), conformal top-p, and frequency-Mondrian plain margin.
3. Fix the protocol: four disjoint splits for frequency-count corpus, `kappa` tuning, conformal calibration, and evaluation; document-level split; one-position-per-document theorem-grade calibration/eval.
4. Rewrite the gate to compare calibrated methods against calibrated methods at matched `delta`, with pass criteria on Pareto dominance and per-bucket coverage at matched size.
5. Then run the expanded matrix across at least three model families, a scale axis, and at least three domains, with epsilon/eta/typical baselines conformalized identically.
6. Delete or demote all legacy final-report claims, noise-channel identification, mathboost, GSM8K/Distinct-n superiority, and possibly the word "Sampling" if the contribution is prediction sets.

## Bottom line

The direction is sound, but the current differentiator is one score term whose sign is questionable and whose prior-art neighborhood is crowded. The project becomes credible only after the reliability diagnostic and Mondrian comparison.
