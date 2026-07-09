# Frequency-Offset Margin Rules: Refined Theory and Experimental Program

**Target venue:** ICML 2027 (expected abstract deadline ~late January 2027, based on ICML 2026 cadence: abstract Jan 23, full paper Jan 28 — confirm when the CFP posts)

**Date:** 2026-07-09

**Status:** Research plan. Supersedes the claim framing in `docs/paper/CLAIM_STACK.md`; consistent with, and stricter than, the gate philosophy in `experiments/check_prediction_set_gate.py`.

---

## Contents

- §0. Executive summary and the one-sentence thesis
- Part I — Refined theoretical framework
  - §1.1 Setup and notation
  - §1.2 Lemma 1: truncation rules are margin rules (the unification, done right)
  - §1.3 Proposition 2: what conformal calibration gives for free — and what it doesn't
  - §1.4 Proposition 4: the optimality lens — exactly when frequency can help
  - §1.5 The method space: frequency-offset margin rules (nu is a special case, not the method)
  - §1.6 Proposition 3: group-conditional validity via frequency-Mondrian calibration
  - §1.7 From token-level to sequence-level: what the guarantee does and does not transfer
  - §1.8 The split protocol and exchangeability (theorem-grade vs. practice-grade)
  - §1.9 Theory claims to make, and claims to never make
- Part II — Experimental program
  - §2.1 Phase 0: the decisive diagnostic (run this before anything else)
  - §2.2 Phase 1: the calibrated Pareto benchmark
  - §2.3 Phase 2: conditional coverage, scale, and tokenizer science
  - §2.4 Phase 3 (gated): downstream generation
  - §2.5 The redesigned decision gate and pre-registered falsification criteria
  - §2.6 Engineering, compute, and statistical-inference notes
- Part III — Paper plan for ICML 2027
  - §3.1 Two claim stacks: Plan A (method paper) and Plan B (audit paper)
  - §3.2 Related-work differentiation map
  - §3.3 Reviewer objection → mitigation table
  - §3.4 Timeline working back from the deadline
- §4. Immediate two-week sprint checklist
- References

---

## §0. Executive summary and the one-sentence thesis

The project's current weakness is that it treats "conformal calibration of a nu score" as the contribution. Conformal coverage is free for any score, so the contribution must be located elsewhere. This plan relocates it.

**The one-sentence thesis:**

```text
Logit margin is the score implicitly used by every practical truncation rule;
whether token frequency carries additional signal beyond margin is a measurable
property of the model (frequency-modulated miscalibration), and if it exists,
frequency-offset margin rules — of which nu, frequency-Mondrian calibration,
and a learned offset are three instantiations — provably dominate margin-only
rules on the coverage/size frontier, with group-conditional coverage guarantees
available at no extra assumption.
```

Three structural changes from the current framing:

1. **The empirical premise is promoted to Figure 1.** The method can only work if, at fixed logit margin, the probability that a candidate token is the true next token varies with token frequency (§1.4, Assumption A). This is directly estimable at trivial cost (§2.1). Everything else is gated on this measurement. Note that the two live hypotheses point in *opposite directions*: the desmoothing view of truncation sampling (Hewitt et al., 2022) predicts rare tokens are *over*-weighted at fixed margin (offset should shrink their margins), while the estimability/recall-uncertainty view behind nu predicts they are *under*-resolved (offset should widen them). Your GPT-2 smoke blow-up is weak evidence for the desmoothing side. Either answer is publishable; assuming the answer is not.

2. **nu is demoted from "the method" to "one parametric ansatz" inside a principled family.** The family is offset margin rules `A(x,i) = m_i(x) - g(n_i)`: nu is `g(n) = kappa/sqrt(n+alpha)` (kappa of either sign), frequency-Mondrian calibration is a piecewise-constant `g` with per-bucket guarantees, and the plug-in feature-restricted oracle is a nonparametrically learned `g`. This absorbs the strongest reviewer attack ("why not Mondrian?") into the paper's own structure.

3. **The paper survives a null result.** If the Phase 0 diagnostic shows margin is a sufficient score (no frequency signal), the paper pivots to a calibration audit of truncation rules (Plan B, §3.1) — still a defensible ICML submission, and a much stronger workshop paper than a forced method paper with unconvincing wins.

---

# Part I — Refined theoretical framework

## §1.1 Setup and notation

Fix a language model with vocabulary `V`, `|V| = v`. For context `x`, the model produces logits `s(x) in R^v`, softmax probabilities `p_i(x)`, and we define the **logit margin**

```text
m_i(x) = s_max(x) - s_i(x) >= 0,     s_max(x) = max_j s_j(x).
```

Since `p_i(x)/p_max(x) = exp(-m_i(x))`, margin ordering equals probability ordering per context.

A **token frequency table** assigns each token id a count `n_i >= 0`, computed from a corpus `D_freq` that is fixed before any calibration or evaluation data is touched (§1.8). A **bucket function** `b: V -> {1..K}` partitions the vocabulary by frequency (e.g., log-count bands; unseen tokens `n_i = 0` are their own bucket).

A **truncation rule** is a set-valued map `S(x) ⊆ V`; decoding samples from the renormalized (optionally temperature-scaled) softmax restricted to `S(x)`. The evaluation population is (context, next-token) pairs `(X, Y)` drawn from a corpus distribution; the model is deterministic given `X`, so all randomness is in the data.

Two objectives, in tension:

```text
Coverage(S)  = P( Y in S(X) )
Size(S)      = E[ |S(X)| ]        (and its distribution, not just the mean)
```

The paper's object of study is the coverage–size frontier of families of rules, and the *conditional* refinement `P(Y in S(X) | b(Y) = k)` per frequency bucket.

## §1.2 Lemma 1: truncation rules are margin rules (the unification, done right)

Call a rule **monotone** if for every `x`, whenever `i in S(x)` and `s_j(x) >= s_i(x)`, also `j in S(x)` (retained sets are downward-closed in margin).

**Lemma 1 (representation).** A truncation rule is monotone if and only if there exists a context-dependent threshold `tau(x) in [0, +inf]` such that, up to ties at the threshold,

```text
S(x) = { i : m_i(x) <= tau(x) }.
```

All standard rules are monotone, differing only in how `tau(x)` is chosen:

| Rule | Threshold `tau(x)` | Character |
|---|---|---|
| greedy | `0` | degenerate |
| min-p(`a`) | `-log a` | **fixed** margin (constant across contexts) |
| top-k | k-th smallest margin at `x` | rank-based |
| top-p | margin of the nucleus crossing token at `x` | mass-based |
| top-nsigma | `n * std_j(s_j(x))` | context-scale-normalized |
| epsilon-sampling | `log p_max(x) - log eps` | absolute-probability-based |

The nu rule `m_i(x) <= m0 + kappa/sqrt(n_i + alpha)` with `kappa != 0` is the **only non-monotone rule in the zoo**: it can retain a lower-probability token while excluding a higher-probability one, when the former is rarer. Define the enclosing family of **offset margin rules**:

```text
S_g(x) = { i : m_i(x) - g(n_i) <= tau },      g: counts -> R.
```

Monotone-with-global-threshold rules are `g ≡ 0`; nu is `g(n) = kappa/sqrt(n+alpha)`; frequency-Mondrian (§1.6) is piecewise-constant `g` with per-piece calibrated `tau`; the learned method (§1.5) fits `g` nonparametrically.

**Status of Lemma 1:** true, easy, and *expository*. It is scaffolding, not a contribution on its own — min-p ≡ fixed margin is two lines (already unit-tested in `experiments/samplers.py`'s test suite). Its value is that it makes the paper's question sharp: *is the margin a sufficient statistic for constructing supports, or does token identity (via frequency) carry additional signal?*

**Proof obligations:** trivial (downward-closed sets in a totally ordered family are sublevel sets; handle ties by randomized or inclusive tie-breaking and say which you use — the top-p crossing-token fix you already made is exactly an inclusive tie/crossing convention).

## §1.3 Proposition 2: what conformal calibration gives for free — and what it doesn't

**Proposition 2 (split-conformal validity; standard).** Let `A(x, i)` be any nonconformity score satisfying:

- **(A1) Fixed score.** `A` — including the frequency table `n_i`, the offset `g`, kappa, alpha, bucket edges, and every other hyperparameter — is fixed independently of the calibration and test data (built from `D_freq` and `D_tune` only; §1.8).
- **(A2) Exchangeability.** Calibration pairs `(X_1,Y_1),...,(X_n,Y_n)` and the test pair `(X,Y)` are exchangeable.

Calibrate `q_hat` as the `ceil((n+1)(1-delta))`-th smallest of `{A(X_t, Y_t)}` (your implemented formula) and set `S(x) = { i : A(x,i) <= q_hat }`. Then

```text
P( Y in S(X) ) >= 1 - delta,
```

and if the calibration scores are almost surely distinct, also `P(Y in S(X)) <= 1 - delta + 1/(n+1)`. (Papadopoulos et al. 2002; Vovk et al. 2005; Lei et al. 2018.)

Three consequences you must internalize and the paper must state explicitly:

1. **Coverage attainment is never evidence for a score.** A random score achieves the same marginal coverage. Any table showing "conformal-nu reaches 90% coverage" is content-free unless paired with set size. All comparisons in the paper are *at matched coverage* or on the *coverage–size frontier* (calibrated method vs. calibrated method). This also invalidates the current gate's comparison of conformal-nu against *uncalibrated* baselines (§2.5).

2. **The guarantee is marginal, not conditional.** `P(Y in S)` averaged over everything. Per-frequency-bucket coverage — the story the paper wants to tell — is *not* implied and must come from Mondrian calibration (§1.6) or be demonstrated empirically.

3. **The guarantee is exactly as strong as (A1) and (A2).** Frequency counts leaking from calibration text breaks (A1). Kappa tuned on calibration feedback breaks (A1). Token positions pooled across shared documents break (A2) in a specific, fixable way (§1.8). The paper states the theorem with these assumptions in the statement, not in a footnote.

## §1.4 Proposition 4: the optimality lens — exactly when frequency can help

This is the theoretical heart of the refined paper. It converts "is nu a good idea?" from an opinion into a measurable property of the model.

**Feature-restricted rules.** Consider rules whose retain/drop decision for candidate `i` at context `x` may depend only on a feature vector `phi(x, i)` — e.g., `phi = (m_i(x))` for margin-only rules, or `phi = (m_i(x), n_i)` for offset rules. Define the **pair-level hit rate**

```text
h(phi0) = P( Y = i  |  phi(X, i) = phi0 )
```

informally: among all (context, candidate-token) pairs whose features equal `phi0`, the fraction in which the candidate is in fact the realized next token. (Formally, define via the pair measure that weights each `(X, i)` pair equally and use a regular conditional probability; ties and null sets handled in the appendix.)

**Proposition 4 (feature-restricted Neyman–Pearson).**

- (i) Within the class of `phi`-measurable rules, the coverage–size frontier is traced by superlevel sets `{ phi : h(phi) >= t }`. (Retaining pairs in decreasing order of conditional hit probability is optimal; this is the classification-oracle result of Sadinle–Lei–Wasserman 2019 restricted to the sigma-algebra of `phi`.)
- (ii) With `phi = (m)`: margin-threshold rules are frontier-optimal among margin-only rules iff `h` is monotone (decreasing) in `m` — empirically guaranteed in practice, so calibrated min-p is already optimal *within its class*.
- (iii) With `phi = (m, n)`: the `(m,n)`-frontier **strictly dominates** the `m`-frontier at some coverage levels **iff** `h(m, n)` is not almost-everywhere a function of `m` alone (up to order-preserving reparametrization). Otherwise the frontiers coincide and every frequency-offset rule is weakly dominated.

**Assumption A (frequency-modulated margin miscalibration).** There is a non-null region of margins where `h(m, n)` genuinely varies with `n` at fixed `m`.

Consequences:

- **Assumption A is the paper's entire license to exist as a method paper.** No Assumption A ⇒ calibrated min-p (C-margin) is unbeatable within this design space ⇒ pivot to Plan B.
- **The sign question answers itself.** If at fixed margin rare tokens hit *more* often, `g` should widen their margins (`kappa > 0`, the current nu sign). If they hit *less* often — the desmoothing prediction, and what your smoke blow-up hints at — then `kappa < 0` and the support-size problem fixes itself simultaneously.
- **The optimal score is an estimand, not an ansatz.** The frontier-optimal `(m,n)`-rule thresholds `h(m,n)` itself. So the principled method is: estimate `h_hat(m, n)` on a tuning split (2-D binned regression / isotonic-in-m per bucket), use `A(x,i) = -h_hat(m_i(x), n_i)` (or the induced additive offset `g_hat`), then conformalize on a fresh calibration split for finite-sample validity. nu's `kappa/sqrt(n+alpha)` is now a parametric special case you can *test* against the fitted `g_hat` — if the fitted offset doesn't look like an inverse square root, that is a finding, not a failure.
- **Proof obligations:** (i) is a rearrangement/NP argument, appendix-length; (ii)–(iii) are corollaries. Be careful with: ties, atoms in the distribution of `h(phi)`, and the exact sense of "strictly dominates" (dominance at some coverage levels, equality elsewhere). None of this is deep; all of it must be exact.

**Why the smoke test blew up — the arithmetic.** With `kappa = 10`, an unseen token (`n = 0`, `alpha = 1`) receives an offset of `10/sqrt(1) = 10` nats of extra margin, versus ~`0.01` for a token seen a million times. Ten nats of margin relaxation admits essentially the entire tail of a 50k-vocabulary softmax (`e^10 ≈ 22,000×` probability-ratio slack). Average support ~14,000 on GPT-2 is therefore not a hyperparameter accident; it is the parametric form doing exactly what it says at that scale. Any plausible fitted `|g|` will be well under ~1–2 nats. This paragraph, in some form, belongs in the paper: it shows you understand your own method's failure mode quantitatively.

## §1.5 The method space: frequency-offset margin rules

The paper's method section should present one family with three instantiations, ordered by increasing flexibility, each conformalized identically (Prop 2) so that all differences are attributable to the score:

| Instantiation | Offset `g(n)` | Free parameters | Guarantee | Role in paper |
|---|---|---|---|---|
| **C-nu** | `kappa / sqrt(n + alpha)` | `kappa in R` (either sign!), `alpha` | marginal (Prop 2) | parametric ansatz; interpretable |
| **Mondrian-margin** | piecewise constant per bucket (implicit: per-bucket `q_hat_k`) | bucket edges `K` | **per-bucket** (Prop 3) | strongest guarantee; the baseline that must be beaten or joined |
| **Learned-g** | `g_hat` fit on `D_tune` (isotonic / binned, from `h_hat`) | smoothing choices | marginal (Prop 2) | plug-in feature-restricted oracle; expected best frontier |

Design notes:

- **Let kappa be signed.** The current codebase hard-assumes the widening direction. `conformal.py` should take `kappa in R`; the tuning split decides the sign. This is a one-line change that removes the project's largest embedded assumption.
- **Mondrian-margin ≙ offset rule.** Per-bucket thresholds `m_i <= q_hat_{b(i)}` are exactly `m_i - g(n_i) <= 0` with `g = q_hat_{b(i)}` piecewise constant. So "why not just Mondrian?" is answered structurally: Mondrian *is* a member of the family, with the strongest guarantee, and the empirical question is whether smooth offsets (nu / learned-g) buy efficiency over the piecewise-constant one.
- **Context-normalization is an orthogonal axis.** `A(x,i) = m_i(x)/sigma(s(x)) - g(n_i)` combines the top-nsigma idea with the frequency offset. Include as an ablation (one extra row), not a headline.
- **Everything else in the literature also lives in this map** — see §3.2 — which is precisely what makes the framing publishable as a *framework*: APS is conformalized top-p, CNS is entropy-Mondrian APS, RAPS is APS with a rank penalty, C-margin is calibrated min-p, and this paper contributes the token-identity (frequency) axis that none of them use.

## §1.6 Proposition 3: group-conditional validity via frequency-Mondrian calibration

**Proposition 3 (label-conditional / Mondrian validity; standard).** Partition `V` into buckets `B_1..B_K` by the fixed frequency table. For each `k`, calibrate `q_hat_k` on the calibration scores whose *true* token falls in `B_k` (that subset has size `n_k`), with the same order-statistic rule. Set `S(x) = { i : A(x,i) <= q_hat_{b(i)} }`. Under (A1) and per-bucket exchangeability,

```text
P( Y in S(X) | b(Y) = k ) >= 1 - delta    for every bucket k.
```

(Vovk 2012 label-conditional conformal; Sadinle et al. 2019.)

Practical fine print the paper must include:

- **Finite-sample cost.** Bucket `k` needs `n_k >= ceil((1-delta)(n_k+1))` to yield a finite threshold — effectively `n_k >~ 1/delta`. Rare-token buckets receive calibration points in proportion to their *token mass*, not their type count, so define buckets by **mass quantiles of the true-token distribution** (e.g., K = 5–8 log-count bands merged to satisfy `n_k >= 50` at `delta = 0.05`), and report every `n_k`.
- **Unseen-token bucket.** `n_i = 0` tokens have no calibration examples by construction (they never appear as true tokens in a corpus that resembles `D_freq`... unless domain-shifted). Handle explicitly: merge into the rarest bucket or use the marginal `q_hat` as fallback; state which.
- **The variance story.** Per-bucket thresholds on ~50–200 points are noisy; report per-bucket coverage with clustered bootstrap CIs (§2.6). Smooth offsets (learned-g) can be read as a variance-reduction device relative to Mondrian — a legitimate selling point if the data support it.

## §1.7 From token-level to sequence-level: what transfers

Be surgical about this or a reviewer will be:

- The guarantee is about **teacher-forced next-token coverage** on corpus text: `P(Y in S(X)) >= 1 - delta` where `Y` is the corpus continuation. It is an audit statement about the support constructor, full stop.
- For a length-`T` continuation, the union bound gives `P(all T corpus tokens covered) >= 1 - T*delta`. This is why **efficiency at small delta (0.01–0.05) is the practically relevant regime** — set your delta grid accordingly, and expect the interesting method separation there.
- For *generation* there is no "true token," so nothing is guaranteed. The honest transfer story: truncation exists to cut probability-inflated junk while retaining plausible continuations; a support constructor with a better coverage/size frontier on corpus text is a better-audited implementation of that goal. Downstream metrics (Phase 3) are then *corroboration*, never the claim. Write this paragraph into the paper nearly verbatim; it defuses the "coverage ≠ quality" objection by conceding it upfront.

## §1.8 The split protocol and exchangeability

Four disjoint resources, all split at the **document level**, manifests hashed and committed:

```text
D_freq   →  token counts n_i                 (fixed first; never touched again)
D_tune   →  h_hat, g_hat, kappa, alpha, bucket edges, any model selection
D_cal    →  conformal quantiles q_hat (or q_hat_k) ONLY
D_test   →  all reported numbers; touched once per frozen config
```

Rules:

1. **Frequency counts.** Two defensible choices, and the contrast is itself an ablation: (a) *pretraining-aligned counts* — for open-data models (Pythia/The Pile, OLMo/Dolma) use the actual pretraining corpus counts; this is the scientifically interesting version ("does the model under/over-trust tokens it saw rarely *in training*?"); (b) *domain counts* from a large held-out corpus disjoint from cal/test. Either satisfies (A1); mixing counts into `D_cal`/`D_test` text does not.
2. **Exchangeability, theorem-grade.** Documents i.i.d. from the corpus distribution; sample **one position uniformly per document** for `D_cal` and for the theorem-grade rows of `D_test`. Then calibration and test pairs are i.i.d., (A2) holds exactly, and Prop 2/3 apply with no asterisks.
3. **Exchangeability, practice-grade.** Pooling all positions per document is standard and sample-efficient but breaks (A2) — two positions in one document are dependent, so the `n+1` scores are not exchangeable. Report pooled results as the *empirical* main tables with document-clustered bootstrap CIs, and show in an appendix that one-per-document and pooled coverage agree (they will, to within CI). Optionally cite and apply two-layer/hierarchical conformal constructions (Dunn–Wasserman–Ramdas-style random-effects prediction sets) for a pooled guarantee; do not let this become a rabbit hole.
4. **Corpus consequence.** WikiText-103 validation has only ~60 articles — one-per-document dies there. Use a many-document corpus (C4 validation-style slices: tens of thousands of documents) as the primary corpus; keep WikiText as a secondary with paragraph-level units if you must, and say so.
5. **Selection discipline.** Kappa/bucket/g choices live in `D_tune`. If you report a small grid of configs on `D_test`, nominate one primary config *before* looking, or apply a selection correction; otherwise the coverage claim quietly dies. Pre-register the primary config in the repo (commit the frozen JSON before the run — cheap and reviewer-visible).
6. **Distribution shift.** Calibrate on domain A, test on domain B as a *robustness* experiment (expect degradation; weighted/non-exchangeable conformal à la Barber et al. 2023 is the optional patch). Never present shifted-domain coverage as guaranteed.

## §1.9 Theory claims to make, and claims to never make

Make exactly these, stated with their assumptions inline:

- **Lemma 1** (representation/unification) — expository.
- **Proposition 2** (marginal validity of conformalized offset rules under (A1)+(A2)) — inherited, cited, not claimed as novel.
- **Proposition 3** (per-frequency-bucket validity of Mondrian-margin) — inherited, cited; its *application* to frequency buckets is the paper's.
- **Proposition 4** (feature-restricted frontier; strict dominance iff Assumption A) — the paper's theoretical contribution, with full proof in appendix.

Never claim:

- "Guaranteed coverage" without (A1)/(A2) stated in the same breath; any *conditional* coverage from Prop 2; anything about generation quality following from coverage; that conformal calibration itself is novel; identified noise channels, mathboost, Lyapunov margins, or any legacy `FINAL_EXPERIMENT_REPORT.md` result (per the brief, those predate the sampler fixes and stay dead).

---

# Part II — Experimental program

Ordering principle: **cheapest decisive experiment first.** Phase 0 costs a few GPU-hours and determines whether the method paper exists. Prediction-set experiments (Phases 0–2) need only forward passes over corpus text — a single forward pass over a document yields logits at *every* position — so they are nearly free; there is no compute excuse for a thin matrix. Generation (Phase 3) is the only expensive phase and is gated.

## §2.1 Phase 0: the decisive diagnostic

**Question:** Does `h(m, n)` — the pair-level hit rate — depend on frequency at fixed margin (Assumption A), and in which direction?

**Estimator.** Stream over `D_tune` positions. For each position, bin **all** `v` candidate tokens into a 2-D grid: margin bins (e.g., 120 bins over [0, 25] nats, finer near 0) × frequency buckets (e.g., 8 log-count bands incl. `n=0`). Accumulate:

- `DEN[m_bin, b] +=` count of candidates in cell (a vectorized scatter-add per position);
- `NUM[m_bin, b] += 1` for the cell of the realized true token.

Then `h_hat = NUM / DEN` per cell. This is one pass, no logit storage, GPU-trivial (`~50k positions × v` scatter-adds).

**Outputs.**

1. **Figure-1 candidate:** `log h_hat` vs. margin, one curve per frequency bucket, with document-clustered bootstrap bands. Under the null (margin sufficient), curves coincide. Under desmoothing, rare-token curves sit *below* common-token curves at fixed margin. Under the nu hypothesis, above.
2. **The implied optimal offset:** for each bucket, the horizontal shift `g*(b)` that best aligns its curve with the reference bucket — this *is* the fitted `g_hat`, previews kappa's sign and magnitude, and can be compared against the `kappa/sqrt(n+alpha)` shape.
3. **A test:** stratified permutation test (permute bucket labels within margin bins, cluster-respecting) or a logistic GLM `1{i=Y} ~ spline(m) * spline(log(n+1))` with document-clustered SEs; report the interaction effect size (max pairwise |Δ log-odds| at fixed margin) with CI.

**Where:** GPT-2-small (sanity, matches your smoke), Pythia-1.4B and Pythia-6.9B (fixed tokenizer, public Pile counts), Qwen2.5-3B (your queue model). Corpora: C4-val slice + OpenWebMath slice.

**Decision rule (pre-registered):**

```text
PLAN A (method paper): the interaction CI excludes zero, same sign,
  |Δ log-odds| >= 0.3 somewhere in the practically relevant margin range
  (m in [2, 12] nats), on >= 2 models and >= 2 domains.
PLAN B (audit paper): otherwise.
```

Either outcome produces Figure 1. This phase costs roughly a day of engineering plus < 5 GPU-hours.

## §2.2 Phase 1: the calibrated Pareto benchmark

**Methods (rows).** All conformalized with the identical Prop-2 pipeline; monotone baselines first, offset methods second; uncalibrated classics last for context.

| # | Method | Score `A(x,i)` | Notes |
|---|---|---|---|
| 1 | C-margin | `m_i(x)` | calibrated min-p; **the null method every offset must beat** |
| 2 | C-logprob | `-log p_i(x)` | calibrated epsilon-sampling |
| 3 | C-zmargin | `m_i(x)/std(s(x))` | calibrated top-nsigma |
| 4 | APS | cum. mass above `i` | calibrated top-p (Romano et al. 2020) |
| 5 | RAPS | APS + `lam*(rank_i - k0)+` | size-regularized (Angelopoulos et al. 2021) |
| 6 | CNS | entropy-Mondrian APS | Ravfogel et al. 2023, the direct predecessor — **mandatory** |
| 7 | C-nu(kappa) | `m_i - kappa/sqrt(n_i+alpha)` | kappa signed, tuned on `D_tune` |
| 8 | Mondrian-margin | `m_i` w/ per-bucket `q_hat_k` | per-bucket guarantee (Prop 3) |
| 9 | Learned-g | `m_i - g_hat(log n_i)` | plug-in oracle from Phase 0's `h_hat` |
| 10 | Mondrian + learned-g | combo | if 8 and 9 both help |
| 11 | ablation | `m_i/std - g_hat(log n_i)` | context-norm × offset interaction |
| — | uncalibrated top-p/min-p/top-k/top-nsigma/eta/typical | swept hyperparameters | empirical curves for context only |

**Delta grid:** `{0.2, 0.1, 0.05, 0.02, 0.01}` — the small end matters (§1.7).

**Models (tokenizer and scale diversity is the point):**

- Pythia 410M / 1.4B / 6.9B — fixed tokenizer + public Pile counts → clean scale trends;
- OLMo-2-7B — fully open pretraining data → exact pretraining-aligned `n_i`, the leakage-proof flagship;
- Qwen2.5-3B (152k vocab), Llama-3.1-8B (128k), Gemma-2-9B (256k) — tokenizer/vocab-size variation.

Minimum credible submission: Pythia ladder + OLMo-2 + two of the closed-data trio. Your current "Qwen-3B only" plan is one row of this and would not survive review alone.

**Corpora:** C4-val (primary; document-rich), OpenWebMath, code (The Stack / CodeParrot val slice), WikiText-103 (secondary, paragraph units), GSM8K prompts (shift experiment only). `n_cal ≈ 5k` one-per-document (plus pooled variant), `n_test >= 100k` positions across `>= 2k` documents.

**Metrics (columns).** Marginal coverage w/ clustered CI; mean, median, p90, p99 set size; retained mass; **log-size Pareto AUC** across the delta grid (headline); per-bucket coverage and per-bucket size at matched marginal coverage; **CovGap** = max bucket undercoverage; size-stratified coverage (SSC). Headline tables at `delta in {0.1, 0.05}`; full grid in appendix.

**The three comparisons that decide the paper:**

1. Best offset method (7–10) vs. **C-margin (1)** — does frequency beat the null?
2. Best offset method vs. **CNS (6) and RAPS (5)** — does token-identity conditioning beat context conditioning and size regularization?
3. **C-nu (7) vs. Mondrian-margin (8) vs. Learned-g (9)** — is the parametric ansatz worth anything beyond its own family?

## §2.3 Phase 2: conditional coverage, scale, and tokenizer science

This phase is what elevates the paper from a benchmark to a finding, and it reuses Phase 0/1 artifacts:

- **Scale trend:** does the Phase-0 interaction effect shrink with model size (Pythia 410M → 6.9B)? Either answer is a headline: "frequency miscalibration is a small-model artifact" or "persists at scale."
- **Tokenizer transfer:** same underlying text, models with different vocabularies — does the effect track *token* frequency (tokenizer-specific) or *word* frequency (language-level)? Reviewers will ask; nobody has measured it.
- **Counts provenance ablation:** pretraining-aligned counts (OLMo/Pythia) vs. domain counts — which carries the signal?
- **Domain interaction:** is the effect larger on math/code (where the nu/mathboost intuition originated) than on web text?
- **Shift robustness:** calibrate C4 → test GSM8K/code; report honest degradation curves for all methods.

## §2.4 Phase 3 (gated): downstream generation

Run only if the §2.5 gate passes. Sampling from calibrated supports at matched delta across methods; temperature applied after truncation (already fixed in `samplers.py`).

- **Reasoning:** your existing self-consistency pipeline (GSM8K, MATH-500, SVAMP; acc@1, maj@k, pass@k) — adequate as-is, but power it properly: >= 500 problems, >= 3 seeds, report seed-level dispersion.
- **Open-ended:** add **MAUVE** as the primary open-ended metric (drop Distinct-n from the main text; keep repetition rate and self-BLEU as secondary). An LM-judge pairwise win-rate is acceptable as tertiary with all the usual caveats stated.
- Frame all of it as corroboration of the audit claim (§1.7), occupying at most ~1 page + appendix.

## §2.5 The redesigned decision gate and pre-registered falsification criteria

Replace the current gate (which compares conformal-nu against *uncalibrated* baselines — an apples-to-oranges test that can pass for reasons unrelated to nu) with calibrated-vs-calibrated criteria, thresholds committed to the repo before the Qwen/OLMo runs:

```text
G1 (efficiency): some offset method improves log-size Pareto AUC over the best
    monotone calibrated baseline (rows 1–6) by >= 5%, with document-clustered
    bootstrap 95% CI excluding 0, on >= 2 models × >= 2 domains.

G2 (conditional): at matched marginal coverage (±0.5pp) and matched mean size
    (±5%), an offset method cuts CovGap by >= 50% relative to C-margin and APS.

PASS = G1 or G2  →  Phase 3 unlocked; Plan A confirmed.
```

**Falsifiers (write them down now, in the repo):**

- Phase 0 interaction ≈ 0 across models/domains → Assumption A false → **Plan B**, stop all nu work.
- Learned-g ≈ Mondrian ≈ C-margin frontiers within CI → frequency signal exists but is not exploitable → Plan B with the diagnostic as the finding.
- C-nu dominated by Mondrian/learned-g everywhere → drop the nu ansatz specifically; the paper proceeds on the family (this is a *renaming*, not a failure — do not cling to the nu brand).

## §2.6 Engineering, compute, and statistical-inference notes

- **Compute reality check.** Phases 0–2 are forward passes: ~2k docs × ~2k tokens = ~4M tokens per (model, corpus) — minutes-to-an-hour per pair on one A100, well under 100 GPU-hours for the whole matrix. Phase 3 (e.g., 16 samples × 500 problems × 512 tokens × ~6 methods × 3 models) is the real spend; budget it only after the gate.
- **The sufficient-statistic trick (build this first).** For any *binned* offset rule, all metrics are computable from, per position: (a) the `(margin-bin × bucket)` count matrix (int16, ~200×8), and (b) the true token's `(margin, bucket, rank, cum-mass)`. Persist these (~0.5 GB per 100k positions) and every offset-rule ablation — any `g`, any kappa, any delta — replays from cache with **zero** additional model passes. APS/RAPS/CNS need (b) plus an entropy scalar. This makes the ablation space essentially free and is itself a nice reproducibility artifact to release.
- **Numerics:** work in raw logits (fp32 accumulation for margins), never re-softmax truncated sets before measuring retained mass; you already raise on invalid truncated distributions — keep that.
- **Inference:** every coverage/size/CovGap number carries a document-clustered bootstrap CI (positions within a document are dependent; naive binomial CIs will be anti-conservatively narrow). Multiple models/domains: report per-cell, no pooling across models.
- **Tests to add to the suite:** (1) synthetic exchangeable-data coverage sanity (empirical coverage within binomial band of `1 - delta` over many resamples); (2) Mondrian per-bucket coverage on synthetic; (3) equivalence tests APS ≡ conformal-top-p and C-margin ≡ calibrated-min-p on random logits; (4) a leakage tripwire that fails CI if `D_freq`/`D_tune`/`D_cal`/`D_test` document-ID manifests intersect.
- **Repro:** frozen config JSONs (incl. kappa, bucket edges) committed pre-run; split manifests content-hashed; one command per table.

---

# Part III — Paper plan for ICML 2027

## §3.1 Two claim stacks

**Plan A — method paper** (if Phase 0 finds the effect and the gate passes):

```text
Title: Frequency-Offset Margin Rules: Conformal Prediction Sets for
       Language Model Truncation

C1. A representation lemma unifying truncation rules as margin rules, and a
    method space (offset margin rules) that contains min-p, top-nsigma, APS/
    top-p, CNS, Mondrian calibration, and nu as special cases.
C2. A feature-restricted optimality theorem: frequency-offset rules strictly
    dominate margin-only rules iff the model exhibits frequency-modulated
    margin miscalibration — plus the first direct measurement of that
    miscalibration across model scale, tokenizer, and domain.
C3. A calibrated coverage/size benchmark showing offset rules improve the
    Pareto frontier and (via Mondrian) deliver per-frequency-bucket coverage
    guarantees; gated downstream corroboration on reasoning and open-ended
    generation.
```

**Plan B — audit paper** (if the effect is null or unexploitable):

```text
Title: Is Logit Margin Enough? A Conformal Audit of Truncation Sampling

C1. Same lemma and method-space map (the framework stands on its own).
C2. A large-scale measurement showing margin is (approximately) a sufficient
    score: at fixed margin, token frequency carries little exploitable signal
    across scales, tokenizers, and domains — adjudicating between the
    desmoothing and estimability hypotheses.
C3. Practical guidance: calibrated min-p ≈ the frontier; per-bucket coverage
    audit of deployed truncation defaults; released audit toolkit + cached
    sufficient statistics.
```

Plan A is a main-conference paper if the effect is clean. Plan B is a credible main-conference submission only with an exceptional measurement section (breadth + the scale/tokenizer science of §2.3); otherwise it is a strong workshop paper and a foundation for the next idea. Decide after Phase 0, not before. In both plans, "nu-Sampling" leaves the title — the brand carries no content and invites the "just min-p with a tweak" reading.

## §3.2 Related-work differentiation map

One table in the paper kills five reviewer objections at once ("how is this different from X?"):

| Work | Set constructor | Conditioning | Guarantee | What this paper adds |
|---|---|---|---|---|
| min-p / top-p / top-k / top-nsigma / eta-typical | margin/mass thresholds | context geometry only | none | calibration + the frequency axis |
| Hewitt et al. 2022 (eta/desmoothing) | prob. threshold | entropy | none | *tests* their tail-overweighting prediction directly (Fig 1) |
| Ravfogel et al. 2023 (CNS) | conformal top-p | entropy bins (context) | marginal per bin | token-identity (frequency) conditioning; margin scores; group guarantees on *label* buckets |
| Quach et al. 2024 (Conformal LM) | sequence-level sets | — | sequence risk | orthogonal level of analysis |
| Ulmer et al. 2024 (non-exch. CLG) | kNN-weighted token sets | neighborhood | approx./weighted | fixed side-information feature; exact split validity; efficiency focus |
| VACP (arXiv 2512.22682) | semantic masking + temp. scoring | semantic mask | marginal | frequency feature; offset-rule family; Mondrian per-bucket validity; scale/tokenizer science |
| Romano/Angelopoulos (APS/RAPS) | cum-mass scores | rank penalty | marginal | these become baselines 4–5 inside the map |

Read VACP closely before writing a word of related work — it is the nearest recent neighbor and reviewers will know it.

## §3.3 Reviewer objection → mitigation table

| # | Objection | Mitigation (already built into this plan) |
|---|---|---|
| 1 | "CNS did conformal decoding sets in 2023" | §3.2 map; CNS is baseline row 6; contribution is the frequency axis + Prop 4 |
| 2 | "Coverage is free; your tables show coverage" | all comparisons at matched coverage / frontier AUC (§2.2); coverage-only claims banned (§1.3) |
| 3 | "Why not Mondrian on frequency buckets?" | Mondrian is *inside* the method family (row 8) with Prop 3; question becomes empirical |
| 4 | "If logits are calibrated, probability ordering is optimal — you can't win" | conceded and formalized as Prop 4(iii); Assumption A measured in Fig 1 before any method claim |
| 5 | "Tokens aren't exchangeable" | one-per-document theorem-grade protocol + pooled empirical with clustered CIs (§1.8) |
| 6 | "Frequency table / kappa leak" | four-way split, pre-registered configs, leakage tripwire test (§1.8, §2.6) |
| 7 | "Your own smoke test shows absurd supports" | diagnosed quantitatively (§1.4 arithmetic); signed kappa + learned-g fix; smoke never cited as evidence |
| 8 | "n_i is tokenizer-dependent, arbitrary" | tokenizer-transfer experiment (§2.3) turns the bug into a study |
| 9 | "Missing eta/typical/RAPS baselines" | rows 2, 5, and uncalibrated sweep row (§2.2) |
| 10 | "Token coverage says nothing about generation" | conceded upfront (§1.7); downstream demoted to gated corroboration |

## §3.4 Timeline (working back from ~Jan 22/28, 2027)

| Window | Milestone | Kill criteria checked |
|---|---|---|
| Jul W2–W3 2026 | Protocol freeze (splits, buckets, gate thresholds committed); signed-kappa + Mondrian + learned-g implemented; sufficient-statistic cache | — |
| Jul W3–W4 | **Phase 0** on GPT-2 / Pythia-1.4B / Qwen-3B × 2 domains | **Plan A vs. Plan B decision** |
| Aug W1–W2 | Phase 1 core: 3 models × 2 domains, full method table | Gate G1/G2 first read |
| Aug W3–Sep W2 | Full Phase 1 matrix + Phase 2 (scale/tokenizer/provenance) | falsifiers re-checked |
| Sep | Intro + theory + Fig 1 drafted (writable regardless of outcome); appendix proofs of Lemma 1 / Prop 4; optional NeurIPS-2026-workshop dry-run submission of the diagnostic | — |
| Oct | Phase 3 if gated in; ablations; internal red-team pass using §3.3 as the checklist | gate final |
| Nov | Full draft; external feedback (advisor + one conformal-literate reader); repro package | — |
| Dec | Revision; figures to camera quality; claims audit against §1.9 ban list | — |
| Jan 2027 | Buffer; abstract + full submission | — |

Slack analysis: the plan reaches a submittable Plan-B paper by ~October even if every method result is null; Plan A adds Phase 3 and stronger claims on top of the same skeleton. The single schedule risk is delaying Phase 0 — it gates everything and costs almost nothing.

---

# §4. Immediate two-week sprint checklist

1. Allow signed kappa in `experiments/conformal.py`; add Mondrian-margin and learned-g constructors (~150 LOC total against your existing interfaces).
2. Implement the Phase-0 scatter-add histogram and the sufficient-statistic cache (§2.6) in `experiments/eval_prediction_sets.py` or a sibling module.
3. Build the four-way document-level split manifests for C4-val + OpenWebMath; commit hashes; add the leakage tripwire unit test.
4. Fetch/compute frequency tables: Pile counts (Pythia), Dolma counts (OLMo-2), plus a domain-count table from a disjoint C4 shard.
5. Run Phase 0 on GPT-2, Pythia-1.4B, Qwen2.5-3B × C4/OpenWebMath. Make the Figure-1 plot. **Decide Plan A vs. Plan B.**
6. Rewrite `check_prediction_set_gate.py` to the §2.5 criteria; commit thresholds before any Qwen/OLMo Phase-1 run.
7. Read VACP (arXiv 2512.22682) end-to-end and write the §3.2 row from the actual paper, not the abstract.
8. Delete or quarantine legacy claims per §1.9 (mathboost, noise channel, GSM8K wins) so no draft ever inherits them.

---

# References

- Ravfogel, Goldberg, Goldberger. *Conformal Nucleus Sampling.* Findings of ACL 2023. https://arxiv.org/abs/2305.02633
- Quach et al. *Conformal Language Modeling.* ICLR 2024. https://arxiv.org/abs/2306.10193
- Ulmer et al. *Non-Exchangeable Conformal Language Generation with Nearest Neighbors.* Findings of EACL 2024. https://arxiv.org/abs/2402.00707
- *Conformal Prediction Sets for Next-Token Prediction in LLMs (VACP).* arXiv:2512.22682, Dec 2025. https://arxiv.org/abs/2512.22682
- Campos et al. *Conformal Prediction for NLP: A Survey.* TACL 2024. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00715/125278
- Barber, Candès, Ramdas, Tibshirani. *Conformal Prediction Beyond Exchangeability.* Ann. Stat. 2023. https://arxiv.org/abs/2202.13415
- Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Noncoverage.* JASA 2019.
- Romano, Sesia, Candès. *Classification with Valid and Adaptive Coverage (APS).* NeurIPS 2020.
- Angelopoulos, Bates, Malik, Jordan. *Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS).* ICLR 2021.
- Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML 2012. (label-conditional / Mondrian)
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive Inference for Regression.* JASA 2018.
- Dunn, Wasserman, Ramdas. *Distribution-Free Prediction Sets for Two-Layer Hierarchical Models.* (pooled-position guarantee option, §1.8)
- Hewitt, Manning, Liang. *Truncation Sampling as Language Model Desmoothing (eta-sampling).* Findings of EMNLP 2022.
- Meister et al. *Locally Typical Sampling.* TACL 2023.
- Nguyen et al. *Turning Up the Heat: Min-p Sampling.* ICLR 2025. https://arxiv.org/abs/2407.01082
- *Top-nsigma: Not All Logits Are You Need.* 2024. https://arxiv.org/abs/2411.07641
- Biderman et al. *Pythia: A Suite for Analyzing LLMs Across Training and Scaling.* ICML 2023.
- OLMo team. *OLMo 2 / Dolma.* Allen Institute for AI. (open pretraining counts)
- ICML 2026 dates (cadence anchor for ICML 2027): https://icml.cc/Conferences/2026/Dates



