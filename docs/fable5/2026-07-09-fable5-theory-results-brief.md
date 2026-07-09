# Fable5 Theory and Results Review Brief

Date: 2026-07-09

Repository: `/Users/misaya.yanghejazfs.com.au/misaya_project/top_no`

Current branch: `codex/calibrated-support-testing`

Current local commit: `5ba0458 Add calibrated prediction set experiment pipeline`

Remote status: this commit has not been pushed yet because GitHub credentials were unavailable on this machine.

## How To Use This Brief

This document is intended to be pasted into a strong external model session for a deep paper/theory/results review. The goal is not to get praise. The goal is to identify whether the paper can become a credible ICML/NeurIPS/ICLR-level submission, what claims are defensible today, what claims must be weakened, and which experiments/theorems should be attacked first.

Ask the reviewer model to act as:

- a skeptical ICML area chair,
- a theory reviewer for decoding / conformal prediction / statistical learning,
- an experimental ML reviewer who checks whether evidence actually supports the claim,
- and a collaborator who proposes a stronger, narrower paper if the current one is overbroad.

## One-Line Project Summary

The project is being reframed from a heuristic "nu-sampling beats top-p/top-nsigma" decoding paper into:

```text
nu-Sampling: Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

The current core idea is:

```text
Truncation decoding can be viewed as token-level prediction-set construction over next-token logits. Standard decoding rules define candidate token sets by cardinality, probability mass, or global logit margins. nu introduces token-frequency calibrated margins, and split conformal calibration can turn this into a target-coverage prediction-set method.
```

## What Fable5 Should Review First

Read in this order:

1. `docs/paper/CLAIM_STACK.md`
2. `docs/paper/RELATED_WORK_POSITIONING.md`
3. `docs/paper/EXPERIMENT_MAINLINE.md`
4. `docs/reports/REPRODUCIBILITY_AUDIT.md`
5. `docs/superpowers/plans/2026-07-08-calibrated-support-testing-roadmap.md`
6. `experiments/samplers.py`
7. `experiments/conformal.py`
8. `experiments/eval_prediction_sets.py`
9. `experiments/check_prediction_set_gate.py`
10. `results/smoke_prediction_sets/prediction_set_metrics.json`
11. Legacy only: `docs/reports/FINAL_EXPERIMENT_REPORT.md`

Important: `FINAL_EXPERIMENT_REPORT.md` is not paper-ready evidence. It predates the sampler/generation fixes. Treat it as a historical artifact and source of hypotheses, not as reliable final results.

## Current Code Architecture

### Shared Decoding Surface

Main file:

```text
experiments/samplers.py
```

It now centralizes truncation/generation behavior:

- `get_keep_mask(logits, strategy, token_freq_table=None, **kwargs)`
- `apply_truncation(logits, strategy, token_freq_table=None, **kwargs)`
- `sample_next_tokens(raw_logits, strategy, strategy_kwargs, temperature)`
- `batch_generate(...)`

Implemented candidate-set strategies:

- `greedy`
- `top_k`
- `top_p`
- `min_p`
- `fixed_margin`
- `top_nsigma`
- `nu`
- `conformal_nu`
- `nu_topp_floor`
- `nu_entropy`
- `nu_mathboost`

Important implementation corrections already made:

- top-p keeps the crossing token in the standard nucleus set.
- Truncation is applied on raw logits; temperature is applied only after truncation.
- Invalid truncated distributions raise an error instead of silently falling back to full softmax.
- Batched generation uses left padding, explicit `position_ids`, EOS-aware stopping, and generated-token-only metrics.

### Conformal Helpers

Main file:

```text
experiments/conformal.py
```

Core functions:

- `nu_nonconformity(logits, target_ids, token_freq_table, kappa, alpha=1.0)`
- `conformal_quantile(scores, delta)`
- `conformal_nu_scores(logits, token_freq_table, kappa, alpha=1.0)`
- `conformal_nu_keep_mask(logits, token_freq_table, kappa, q_hat, alpha=1.0)`

### Main Evaluation Pipeline

Main file:

```text
experiments/eval_prediction_sets.py
```

Purpose:

Evaluate token-level prediction sets before making downstream generation claims.

Outputs:

- `prediction_set_metrics.json`
- `coverage_size_pareto.png`
- plus derived plots from `experiments/plot_prediction_sets.py`

Metrics:

- true-token coverage,
- average support size,
- median support size,
- support-size quantiles,
- support-size histogram,
- average retained probability mass,
- frequency-bucket coverage,
- frequency-bucket average support size.

Main methods:

- `top_k_50`
- `top_p_0.95`
- `min_p_0.05`
- `fixed_margin_3`
- `top_nsigma_2`
- `nu_k10_m3`
- `conformal_nu_k{kappa}_delta{delta}`

### Downstream Pipelines Now Exist But Are Not Yet Paper-Run

Reasoning self-consistency:

```text
experiments/eval_reasoning_self_consistency.py
configs/reasoning_self_consistency_qwen3b.json
scripts/run_reasoning_self_consistency_qwen3b.sh
```

Reports:

- `acc@1`
- `pass@4`, `pass@8`, `pass@16`
- `maj@4`, `maj@8`, `maj@16`
- invalid answer rate
- unique answer count
- answer entropy

Datasets:

- GSM8K
- MATH-500
- SVAMP

Open-ended quality:

```text
experiments/eval_openended_quality.py
configs/openended_quality_qwen3b.json
scripts/run_openended_quality_qwen3b.sh
```

Reports:

- Distinct-n as surface diversity only,
- self-BLEU,
- repetition rate,
- evaluator LM perplexity,
- length-normalized unique token ratio.

Controlled channels:

```text
experiments/exp5b_controlled_channels.py
configs/controlled_channels_qwen3b.json
scripts/run_controlled_channels_qwen3b.sh
```

Implemented channel probes:

- hidden/input-embedding Gaussian perturbation,
- dropout ensemble target-logit variance.

Not implemented yet:

- int8/int4 quantization residual channel,
- bootstrap-model channel,
- robust multi-model channel evidence.

## Mathematical Framing To Audit

Let `s_i(x)` denote the raw next-token logit for token `i` at context `x`. Let:

```text
s_max(x) = max_j s_j(x)
```

The retained token set is:

```text
S(x) subset of vocabulary V
```

Decoding samples from:

```text
p_tilde_i = exp(s_i / T) 1[i in S(x)] / sum_j exp(s_j / T) 1[j in S(x)]
```

### Existing Rules As Support Constructors

Top-k:

```text
S_topk(x) = k largest-logit tokens
```

Top-p:

```text
S_topp(x) = smallest sorted set whose cumulative probability mass exceeds p
```

min-p:

```text
p_i >= alpha_p * p_max
```

Since softmax ratios cancel the denominator:

```text
p_i / p_max = exp(s_i - s_max)
```

So:

```text
min-p with alpha_p is equivalent to s_max - s_i <= -log(alpha_p)
```

For `alpha_p = 0.05`, the margin is about:

```text
-log(0.05) ~= 2.996
```

This is why `min_p_0.05` and `fixed_margin_3` should be nearly equivalent. This is unit-tested.

top-nsigma:

```text
s_max - s_i <= n_sigma * std_j(s_j)
```

This is a context-global logit margin.

nu:

```text
s_max - s_i <= m0 + kappa / sqrt(n_i + 1)
```

where `n_i` is a token frequency count from a corpus/frequency table.

Interpretation correction:

- Lower-frequency tokens have smaller `n_i`.
- Therefore `kappa / sqrt(n_i + 1)` is larger.
- Therefore the current nu rule widens the retained margin for low-frequency tokens.
- It is an uncertainty-aware conservative recall rule for low-frequency tokens, not a rare-token penalty.

### Conformal nu Nonconformity

The V2 method target is:

```text
A_kappa(x, i) = s_max(x) - s_i(x) - kappa / sqrt(n_i + alpha)
```

Calibration on held-out true next tokens:

```text
scores_t = A_kappa(x_t, y_t)
q_hat = Quantile_{1-delta}(scores_t)
```

Retain at test time:

```text
S_nu(x) = { i : A_kappa(x, i) <= q_hat }
```

Split-conformal quantile implementation:

```text
rank = ceil((n + 1) * (1 - delta))
q_hat = sorted_scores[min(rank, n) - 1]
```

Fable5 should audit whether the paper can claim finite-sample marginal coverage under the actual data-generation/evaluation protocol:

- Are next-token positions exchangeable enough under the chosen split?
- Does reusing the same text pool to build token counts leak information?
- Is token frequency `n_i` treated as fixed side information or estimated from calibration/eval text?
- Does corpus frequency estimation affect conformal validity?
- Does choosing `kappa` on the same data break the coverage claim unless a separate tuning split is used?
- Is the claim conditional coverage, marginal coverage, or empirical coverage?

## Current Results Status

### Current Reliable Local Smoke

The only post-fix model-backed result committed from the new pipeline is a local smoke test:

```text
model: gpt2
dataset: wikitext validation
device: MPS
n_calibration: 64
n_eval: 64
```

It is a link test, not paper evidence.

Smoke metrics:

| Method | Coverage | Avg Support | Median Support | Retained Mass |
|---|---:|---:|---:|---:|
| `top_k_50` | 0.6562 | 50.0 | 50.0 | 0.7069 |
| `top_p_0.95` | 0.9375 | 2638.6 | 1072.0 | 0.9537 |
| `min_p_0.05` | 0.5625 | 18.7 | 10.0 | 0.6200 |
| `fixed_margin_3` | 0.5625 | 18.8 | 10.0 | 0.6205 |
| `top_nsigma_2` | 0.8750 | 443.1 | 158.0 | 0.8314 |
| `nu_k10_m3` | 0.9219 | 14887.0 | 10577.0 | 0.9967 |
| `conformal_nu_k10_delta0.05` | 0.8906 | 13982.4 | 9570.0 | 0.9960 |

Smoke gate result:

```text
FAIL, expected for this tiny local smoke.
```

Interpretation:

- This verifies that the code path runs end-to-end.
- It does not support the paper claim.
- In fact, the smoke result shows nu/conformal-nu can be extremely inefficient with current hyperparameters on small GPT-2/WikiText smoke.
- This is a warning sign that the score and calibration need serious scrutiny before expensive downstream experiments.

### Legacy Result Status

Legacy results under `results/` and `docs/reports/FINAL_EXPERIMENT_REPORT.md` include claims like:

- Top-K bias coverage and Slepian/correlated-noise evidence.
- Synthetic heteroscedastic channel recovery with `c_fit > 0`.
- n-sweep convergence slope near `-1`.
- two-point indistinguishability and V_eff transition.
- Lyapunov / adaptive margin experiments.
- old nu decoding comparisons and old downstream GSM8K/creative metrics.

But these are not final evidence because:

- downstream generation used old sampler/generation code,
- old top-p handling was wrong,
- old batch generation had padding/position issues,
- old scripts had synthetic fallback behavior,
- old reports overclaimed "identified noise channel",
- old mathboost interpretation was partially backwards.

Use these legacy results only as:

- historical motivation,
- hypothesis generators,
- theorem/protocol sketches to salvage,
- examples of claims that must be weakened or rerun.

## Main Decision Gate Before GPU Spending

The current GPU queue runs:

```bash
PYTHON_BIN=python bash scripts/run_icml2027_gpu_queue.sh
```

The queue executes:

1. Qwen2.5-3B prediction-set evaluation.
2. Prediction-set figures.
3. Decision gate.
4. Reasoning self-consistency only if gate passes.
5. Open-ended quality only if gate passes.
6. Controlled channels only if gate passes.

Gate file:

```text
experiments/check_prediction_set_gate.py
```

Gate logic:

Proceed only if either:

```text
At fixed target coverage, conformal-nu has lower average support size than strong baselines.
```

or:

```text
At matched support size, conformal-nu improves low-frequency bucket coverage without degrading overall coverage.
```

Fable5 should audit whether this gate is too weak, too strong, or misaligned with the paper's intended claim.

## Current Hypothesis

The strongest version of the paper is likely not:

```text
nu-sampling improves all downstream generation metrics.
```

The stronger and safer version is:

```text
Token truncation can be audited as prediction-set construction. nu provides a frequency-indexed logit-margin score that can be split-calibrated for target true-token coverage. The right first-order evidence is coverage/efficiency, especially by token-frequency bucket. Downstream generation should be secondary and gated by prediction-set evidence.
```

## Likely Reviewer Attacks

### Attack 1: This Is Just min-p With A Frequency-Dependent Margin

This is partly true. The response may be:

- min-p is a fixed logit-margin special case,
- top-nsigma is a context-global margin,
- nu is a token-wise margin,
- conformal calibration turns the margin into a target-coverage prediction set.

Fable5 should decide whether this is enough novelty.

### Attack 2: Token Frequency Is A Weak Proxy

The paper currently assumes frequency relates to uncertainty / sensitivity / estimability. Evidence is mixed:

- synthetic channel evidence supports the estimator machinery,
- controlled-channel scripts now exist but Qwen3B results are not run,
- real hidden-state/quantization/dropout evidence is not yet paper-ready.

Fable5 should identify what theoretical assumption is actually needed:

```text
Var(error_i | x) or sensitivity_i decreases with n_i
```

or a weaker operational claim:

```text
n_i is a useful side-information feature for calibrating support sets.
```

### Attack 3: Conformal Prediction Validity May Not Hold

Possible issues:

- next-token samples are dependent within documents,
- calibration/eval positions may not be exchangeable,
- token counts may be estimated from the same corpus,
- model logits are deterministic but data contexts are dependent,
- hyperparameter selection may use evaluation feedback.

Fable5 should propose a clean split protocol:

```text
frequency-count corpus / kappa tuning split / conformal calibration split / final evaluation split
```

and state which coverage theorem applies.

### Attack 4: Support Sets Are Too Large

In the smoke run, nu/conformal-nu retained huge supports. If Qwen3B behaves similarly, the method fails as an efficiency improvement.

Fable5 should attack:

- sign of the nonconformity term,
- whether `- kappa / sqrt(n_i + alpha)` is the right direction,
- whether there should be two channels: recall uncertainty vs reliability prior,
- whether low-frequency tokens should get wider margin, narrower margin, or a calibrated learned feature,
- whether support-size regularization should be part of tuning.

### Attack 5: Existing Baselines Are Incomplete

Current V2 prediction-set baselines include top-k/top-p/min-p/fixed-margin/top-nsigma/nu/conformal-nu.

Missing or not first-class yet:

- typical sampling,
- epsilon sampling,
- eta sampling / desmoothing,
- Mirostat,
- locally typical variants,
- possibly contrastive or speculative decoding baselines if relevant.

Fable5 should decide which baselines are mandatory for an ICML-level decoding paper.

### Attack 6: Downstream Metrics Can Distract From The Core Claim

The old project overemphasized Distinct/repetition and toy GSM8K numbers. The new plan demotes them.

Fable5 should decide whether the paper should:

- be purely about prediction-set coverage/efficiency,
- include downstream generation only as secondary evidence,
- or abandon downstream generation entirely until prediction-set evidence is strong.

## Questions Fable5 Should Answer

### Theory

1. Is the proposed nonconformity score

```text
A_kappa(x, i) = s_max(x) - s_i(x) - kappa / sqrt(n_i + alpha)
```

the right score for the stated goal?

2. Should the sign of the frequency term be reversed for some claims?

3. Should there be two separate frequency effects?

```text
recall uncertainty: low-frequency tokens need wider margins
reliability/prior: low-frequency tokens may be less likely true completions
```

4. Can the method be expressed as conformal prediction with fixed side information?

5. What assumptions are required for finite-sample marginal coverage?

6. Is there any theorem worth proving beyond direct conformal validity?

7. Can the paper claim novelty if min-p is a fixed-margin special case and nu is a feature-dependent margin?

8. What is the cleanest theorem statement for:

```text
min-p = fixed logit margin
top-nsigma = context-global logit margin
nu = token-wise margin
conformal-nu = calibrated prediction set
```

### Experiments

1. Is prediction-set coverage/efficiency the correct main experiment?

2. Is the decision gate sufficient?

3. What dataset/model matrix is needed before paper submission?

Current intended first run:

```text
Qwen/Qwen2.5-3B on WikiText validation
n_calibration=5000
n_eval=20000
```

4. Should calibration/eval use WikiText, C4, OpenWebText, Pile, or domain-specific corpora?

5. How should token frequency counts be built to avoid leakage?

6. What are the mandatory baselines?

7. What is the right support-efficiency metric?

Options:

- average support size,
- median support size,
- retained probability mass,
- expected set size at target coverage,
- area under coverage-size Pareto curve,
- low-frequency bucket coverage at matched support.

8. If conformal-nu keeps huge supports, what ablation should be run first?

### Paper Positioning

1. Is the title too broad?

Current primary:

```text
nu-Sampling: Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

2. Should "sampling" be removed from the title if the core contribution is prediction sets?

Possible alternative:

```text
Frequency-Calibrated Logit Prediction Sets for Language Model Decoding
```

3. Should the old "hypothesis testing" framing be retained or replaced by "prediction sets"?

4. Should the paper target theory track, main conference empirical paper, or workshop first?

5. What contribution stack is defensible if Qwen3B prediction-set results are weak?

## What Not To Claim

Do not claim:

- "we identified the real LLM noise channel,"
- "nu beats top-p on GSM8K,"
- "legacy JSON results are paper-ready,"
- "mathboost rescues low-frequency math tokens by widening their margin,"
- "Distinct-n proves quality,"
- "coverage is conformal-guaranteed without stating exchangeability/splitting assumptions."

## What Can Be Claimed Today

The current code supports these limited claims:

- The repository now has a reproducible sampler core.
- top-p crossing-token behavior is fixed and tested.
- min-p and fixed logit margin equivalence is implemented and tested.
- nu can be represented as a token-frequency-dependent logit-margin rule.
- conformal-nu can be evaluated as a split-calibrated prediction-set method.
- a full Qwen3B experiment queue exists, but has not yet produced paper-ready results.
- local GPT-2 smoke validates the pipeline but does not support the paper claim.

## Recommended Review Output From Fable5

Please ask Fable5 to produce:

1. **Theory audit**

   - exact theorem statements that are valid,
   - assumptions needed,
   - proof sketch or proof failure,
   - whether the nonconformity score is well-motivated.

2. **Claim-stack rewrite**

   - strongest defensible paper title,
   - 3 contribution bullets,
   - 1 paragraph abstract,
   - claims to delete.

3. **Experiment audit**

   - mandatory baselines,
   - required data splits,
   - leakage risks,
   - which metrics go in the main table,
   - what result would falsify the project.

4. **Reviewer attack memo**

   - top 10 reviewer objections,
   - how to fix each,
   - whether the project is currently ICML-submission-ready, workshop-ready, or needs re-scoping.

## Commands Available In The Repo

Local smoke:

```bash
bash scripts/run_prediction_sets_smoke.sh
```

GPU prediction-set run:

```bash
PYTHON_BIN=python bash scripts/run_prediction_sets_qwen3b.sh
```

Plot prediction-set figures:

```bash
PYTHON_BIN=python bash scripts/run_prediction_set_plots.sh
```

Gate:

```bash
python experiments/check_prediction_set_gate.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Full GPU queue:

```bash
PYTHON_BIN=python bash scripts/run_icml2027_gpu_queue.sh
```

Lightweight verification:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m compileall experiments tests
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
for script in scripts/*.sh; do bash -n "$script"; done
git diff --check
```

## Current Local Verification

On 2026-07-08, the following passed locally:

- Python compile over `experiments` and `tests`.
- 14 unit tests.
- shell syntax check for all `scripts/*.sh`.
- GPT-2/WikiText prediction-set smoke on MPS.
- prediction-set plot generation from smoke metrics.
- gate script generation of an expected smoke failure report.

## Current State Summary For The Reviewer

The project is promising only if it becomes disciplined:

```text
Main claim: calibrated support construction, not generic decoding superiority.
Main evidence: true-token coverage vs support-size efficiency, not Distinct-n.
Main risk: the nu score may keep supports too large or rely on weak frequency assumptions.
Main theoretical issue: conformal validity and frequency side-information need a clean split protocol.
Main experimental issue: Qwen3B paper-level results have not been run after fixes.
```

The most useful Fable5 output would be a hard-nosed answer to:

```text
Is frequency-calibrated logit prediction sets a real publishable contribution, and what exact theorem/experiment package would make it credible?
```
