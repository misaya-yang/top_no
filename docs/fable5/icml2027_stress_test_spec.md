# Stress Test → Implementation-Ready Specification

**Object under review:** "Frequency-Offset Margin Rules: Refined Theory and Experimental Program" (2026-07-09), cited below as PLAN §x.y.

**Reviewer stance:** strict ICML PC / theory referee / experimental auditor. This document does not rewrite the PLAN; it hardens it. Where the PLAN is wrong or underspecified, this document overrides it. Everything here is written to be executable by a coding agent and defensible in a rebuttal.

**Date:** 2026-07-09

---

## Verdict summary (read this first)

| Item | Verdict | Action |
|---|---|---|
| Lemma 1 | Correct, trivial, expository | Keep, ≤ half page main text; fix tie conventions |
| Prop 2 | Correct, standard, not novel | Keep as cited proposition; proof = citation + 5 lines appendix |
| Prop 3 | Correct, standard machinery, novel *application* | Keep in main text; add random-`n_k` and infinite-threshold fine print |
| Prop 4 | Correct **after repairs** (feature-class lattice, randomization, strictness lemma) | Promote to Theorem 1; it is a *lens*, not a deep result — position accordingly |
| Phase 0 | Right idea, underspecified estimator and test | Full executable spec in §2; **n_tune raised to ≥300k positions** |
| Method family | One redundancy (learned-g vs learned-h unclear), one gap | learned-h (2-D) is the general method; g-additive is an ablation; C-nu demoted to ansatz paragraph |
| Baselines | **One mandatory baseline missing** (entropy-Mondrian margin control), one cheap standard missing (TS+APS) | Added in §4; gate rewritten with exact thresholds |
| Splits | Sound, but near-duplicate leakage and pretraining-contamination unhandled | MinHash dedup across split boundaries; contamination stated as limitation + fresh-text robustness slice |
| Matrix | Too big for a rented-GPU budget as written | Three tiers in §6; MVP ≈ 20 GPU-hrs |
| Paper identity | PLAN hedges between two papers | Decided in §7: audit-first hybrid, one title per Phase-0 outcome |
| Gate | Directionally right, numerically vague, missing a confound control | Rewritten in §4.3 with pass/fail numbers |

The single most important correction in this document: **without an entropy-conditioned control baseline, every positive result the PLAN could produce is confounded** — "conditioning on frequency helps" is indistinguishable from "conditioning on anything helps." See §4.1.

---

# §1. Theory hardening

## 1.1 Lemma 1 (representation of truncation rules)

**Verdict: correct, trivial, expository. Keep — it earns its space by making the question sharp, not by being a result.**

Required repairs before it appears in a paper:

1. **Tie convention.** As stated ("downward-closed in margin"), top-k and top-p are *not* monotone: with tied logits, top-k with `k=1` and two argmaxes splits a tie; nucleus sort order can split ties at the boundary. State the lemma for rules that cannot distinguish tokens with equal logits, and note that top-k/top-p satisfy it under inclusive tie-breaking (your top-p crossing-token fix is exactly this) or randomized tie-breaking. One sentence + one footnote.
2. **Threshold range.** `tau(x) ∈ [0, +∞]`, with `+∞` = no truncation, `0` = greedy (up to argmax ties).
3. **Temperature covariance.** Margins scale as `1/T` under temperature `T` applied to raw logits; margin-threshold rules are equivariant (threshold scales accordingly). One remark; prevents a "what about temperature?" review question. Your pipeline applies temperature after truncation, so the audit is temperature-free by construction — say so.

**Proof obligation:** three lines (sublevel sets of a total preorder). Main text statement, proof inline or appendix.

## 1.2 Proposition 2 (split-conformal marginal validity)

**Verdict: correct, standard, zero novelty — and that is fine. Cite Papadopoulos et al. 2002 / Vovk et al. 2005 / Lei et al. 2018; do not present as a contribution.**

Exact assumptions that must appear *in the proposition statement*, not prose:

- (A1) The score `A` — including the frequency table, `g`/`kappa`/`alpha`, bucket edges, and any fitted component — is measurable and fixed independently of `D_cal ∪ D_test` (trained on `D_freq ∪ D_tune` only).
- (A2) The `n` calibration pairs and the test pair are exchangeable.
- Quantile rule: `q_hat = ceil((n+1)(1-delta))`-th order statistic (already implemented); coverage `≥ 1−delta`; upper bound `≤ 1−delta+1/(n+1)` **only under almost-surely distinct scores** — with a discrete score component (buckets) ties are possible, so claim the upper bound only if you verify near-continuity or use smoothed conformal. Cheap fix: add `U ~ Unif(0, 1e-6)` dithering to scores; then claim both bounds cleanly. Specify the dithering in the paper.

**Proof obligation:** none beyond citation; 5-line appendix recap for self-containedness.

## 1.3 Proposition 3 (frequency-Mondrian / label-conditional validity)

**Verdict: correct, standard machinery (Vovk 2012), and the *application to frequency buckets* is where the paper's guarantee novelty lives. Keep in main text.**

Required fine print (statement-level, reviewers will check):

1. **Random per-bucket counts.** `n_k = #{t : b(Y_t)=k}` is random. Label-conditional conformal is valid with random `n_k` (condition on bucket membership; within-bucket pairs exchangeable under i.i.d. sampling). State it; don't hand-wave.
2. **Infinite thresholds.** If `n_k < ceil(1/delta) − 1`, `q_hat_k = +∞` and bucket `k` retains the entire vocabulary. This is *valid but vacuous*. The paper must report all `n_k` and define buckets by **true-token mass quantiles on `D_tune`** (PLAN §1.6 already says this) with a hard floor: merge buckets until `n_k ≥ 5/delta` (e.g., ≥100 at `delta=0.05`).
3. **Marginal implication.** Bucket-conditional validity at level `delta` implies marginal validity at `delta` (average of conditionals). One line.
4. **Impossibility preemption.** Fully `X`-conditional coverage is impossible distribution-free (Vovk 2012; Lei & Wasserman 2014; Foygel Barber et al. 2021). Label-bucket conditioning over a *finite fixed partition* is exactly what remains achievable. Put this sentence in the paper; it converts a likely objection into evidence you know the literature.

**Proof obligation:** citation + short appendix proof (per-bucket rank argument).

## 1.4 Proposition 4 (feature-restricted Neyman–Pearson) — the load-bearing item

**Verdict: correct in spirit; as drafted in PLAN §1.4 it has three fixable gaps. After repair it is a clean, moderately novel *adaptation* of least-ambiguous-classifier logic (Sadinle–Lei–Wasserman 2019) to feature-restricted token-level rules. It is strong enough to be the paper's organizing theorem for an empirical/method paper. It is NOT strong enough to carry a theory-track submission. Position the paper's center of mass on measurement + guarantees + benchmark, with Theorem 1 as the lens.**

### Gap 1: the feature-class lattice is being blurred

PLAN §1.4(ii) says "calibrated min-p is optimal within its class." True — but top-p/APS, top-nsigma, and CNS are **not in the `(m)` class**: their thresholds depend on context statistics (cumulative mass, `std(s(x))`, entropy). The honest structure is a lattice of feature classes:

```text
(m)  ⊂  (m, n)                       [frequency axis — this paper]
(m)  ⊂  (m, c(x))                    [context axis — top-p, top-nsigma, CNS]
(m, n)  vs  (m, c(x))                [INCOMPARABLE — neither nests the other]
(m, n, c(x))                         [join — where Mondrian×entropy controls live]
```

**Theorem 1 must only claim: `(m,n)` strictly dominates `(m)` iff Assumption A.** It licenses *nothing* about `(m,n)` vs `(m, entropy)`. That comparison is empirical, which is exactly why the entropy-conditioned control baseline (§4.1) is mandatory. Draw this lattice as a small figure in the paper; it is the cleanest related-work map you have.

### Gap 2: exact statement and objects

Specify, in the appendix, all of:

- **Pair measure.** `mu` = law of `phi(X, I)` with `X ~` corpus distribution, `I ~ Unif(V)` independent. Hit function `h(phi0) = P(Y = I | phi(X,I) = phi0)` via regular conditional probability. Then `Coverage(D) = v·E_mu[h·1_D]`, `Size(D) = v·mu(D)` for a retain-region `D` in feature space.
- **Randomized rules.** The frontier is traced by superlevel sets `{h ≥ t}` *with randomized tie-breaking at the boundary* (otherwise atoms of `h` leave gaps). Without this the "frontier" claim is false in discrete settings.
- **(i) NP optimality** within `phi`-measurable randomized rules: standard Neyman–Pearson argument.
- **(ii) Monotonicity remark.** In the `(m)` class, optimal sets are superlevel sets of `h_m(m) = E[h | m]`; they are *threshold* rules iff `h_m` is nonincreasing — an empirical fact to verify and display, not assume (Phase 0 gives it for free).
- **(iii) Strictness lemma.** `E[h_{(m,n)} | m] = h_m` (tower property; refinement is mean-preserving). Prove: frontiers coincide for all size budgets iff `h_{(m,n)} = h_m` mu-a.e.; if they differ on a positive-measure set, there exists a budget where the `(m,n)` rule strictly wins (take `t` separating the disagreeing values; compare NP sets at matched size). This is the only proof requiring actual care — write it fully. Optional footnote: this is Blackwell sufficiency of `m` for the retain/drop decision problem.
- **Corollary (additive-offset representability).** The `(m,n)`-optimal rule is an *additive-offset* margin rule `m − g(n) ≤ tau` for all levels iff `h` has shift structure `h(m,n) = rho(m − g(n))` with `rho` decreasing. Otherwise additive `g` is itself suboptimal within `(m,n)` and the general method is thresholding `h_hat(m,n)` directly. **This corollary matters:** it cleanly separates the general method (learned-h) from the interpretable ablation (learned-g additive) and predicts in advance when they diverge.

### Gap 3: estimand relativity

`h` is a property of the pair **(model, deployment corpus)**, not of the model alone. Different domains have different `h` — that is a feature (Phase 2 domain analysis) but must be stated, or a reviewer will ask "is this miscalibration of the model or of your corpus choice?" Answer in the paper: both, and that is the correct object — truncation rules are deployed against a text distribution.

### What to remove or refuse to prove

- **No sequence-level theorem.** The union bound remark (PLAN §1.7) is one sentence; any attempt at a sequence-level guarantee drags you into Quach et al. territory and out of scope.
- **No beyond-exchangeability theory.** Cite Barber et al. 2023 for the shift experiment's interpretation; do not derive weights.
- **No "two-channel" (recall vs. reliability) formalism** from the original brief. Phase 0's `h` surface *is* the resolution of that question; a separate formalism would be decoration.

---

# §2. Phase 0 diagnostic — executable specification

**Purpose:** estimate `h(m, b)` and decide Plan A vs. Plan B. Everything below is implementable today against `experiments/` with no new model code.

## 2.1 Candidate-token population (exact)

- **Documents:** from `D_tune` (§5), shuffled by seeded hash. **Positions:** all `t` with ≥16 context tokens, predicting token `y` at `t+1`; stride 1; skip positions whose target is BOS/PAD; **include** EOS as a target and as a candidate.
- **Candidates:** for each retained position, all `i ∈ V_gen` = vocabulary minus PAD/BOS and any never-generable control tokens (frozen exclusion list per tokenizer, committed to the repo). `|V_gen| = v_gen`.
- **Pairs population:** `{(position, i) : i ∈ V_gen}`. One position contributes `v_gen` denominator counts and exactly 1 numerator count.
- **Sample size:** `N_pos ≥ 300,000` positions across `≥ 3,000` documents per (model, domain). Rationale (power): rare-bucket true tokens are ~1–3% of positions; a log-odds contrast of 0.3 with ~1/sqrt(NUM) noise needs `NUM ≈ 100+` per (coarse margin window × bucket); 300k positions delivers that in the m ∈ [2,12] window for all but the zero-count bucket. The PLAN's implied ~50k is underpowered for exactly the buckets that matter — this is a correction.

## 2.2 Grids

- **Margin bins (fine grid, fixed, committed):** edges `0 : 0.25 : 10` (40 bins), `10 : 0.5 : 20` (20 bins), `(20, ∞]` overflow. Raw logits, fp32 margins, no temperature.
- **Frequency buckets (diagnostic):** `B0: n=0`; `B1: 1–9`; `B2: 10–99`; `B3: 10^2–10^3`; … `B8: ≥10^7` (log10 bands, count from `D_freq`). Interpretable; used for Figure 1. (Mondrian *methods* use mass-quantile buckets instead — different object, PLAN §1.6; do not conflate them in code: two bucket tables.)

## 2.3 Counts, estimator, smoothing

Per position, one vectorized pass:

```text
DEN[mbin(m_i), b(i)] += 1     for all i in V_gen      (scatter-add on GPU)
NUM[mbin(m_y), b(y)] += 1     for the realized y
h_hat = NUM / DEN             (elementwise; no smoothing in the estimator)
```

- **No additive smoothing** inside `h_hat` (it biases exactly the sparse cells under study). Sparsity is handled by *masking and coarsening*, not smoothing: a cell enters plots/tests only if `NUM ≥ 20` after coarsening margin bins within-bucket (greedy merge of adjacent fine bins until the floor is met). Report both the fine masked heatmap and coarsened curves.
- For the *method* (learned-h/learned-g), fit on the same counts but with monotone-in-m isotonic regression per bucket + light `(NUM+0.5)/(DEN+1)` stabilization — the method may smooth; the *diagnostic* may not. Two different functions in code.

## 2.4 Class-imbalance guardrails (how this plot lies, and how to stop it)

1. **Never compare marginals.** "Rare tokens hit less overall" is trivially true (they occur less). The only licensed comparison is *within a margin bin*. Enforce in code: the plotting/test API takes margin-conditional slices only.
2. **Simpson's-lemma check.** Within-bin margin distributions still differ across buckets inside a 0.25-nat bin (residual confounding). Verify: recompute the bucket contrast using within-cell mean margins as a covariate (logistic GLM below); if the contrast moves by >20%, narrow the bins.
3. **Per-domain analysis only.** Never pool C4 and OpenWebMath in one `h` surface — rare tokens concentrate in math/code, so pooling manufactures a frequency effect out of a domain effect. Pooling across *documents within* a domain is fine.
4. **Duplicate suppression.** Near-duplicate documents create pseudo-replication that shrinks CIs falsely → MinHash dedup at split construction (§5.3).
5. **Cluster-aware uncertainty.** All CIs by document-level bootstrap (resample documents with replacement, `B=1000`, percentile intervals on `log h_hat`). Within-document token dependence makes naive binomial CIs anti-conservative; do not report them.

## 2.5 Test statistic and decision rule (pre-registered, exact)

- **Primary statistic:** for each bucket `b`, the *aligned offset* `g*(b)` = the horizontal shift (in nats) minimizing squared distance between bucket-`b`'s coarsened `log h` curve and the reference bucket `B_ref` (highest-mass bucket) over the window `m ∈ [2, 12]`. Effect size `Δ = max_b |g*(b) − g*(B_ref)|` over buckets with ≥3 valid cells in-window.
- **Secondary (model-based) statistic:** logistic GLM on pair-subsampled data — `1{i=y} ~ cs(m, df=6) + C(bucket) + cs(m):C(bucket)` with document-clustered SEs; report the max |bucket main-effect + interaction| in log-odds over the window. (Pair subsampling: keep all numerator pairs, subsample denominator pairs at rate `r ≈ 10^-3` with weight `1/r` — full pair-level GLM is 10^9 rows otherwise.)
- **Decision (commit to repo before running):**

```text
PLAN A (effect exists):  Δ ≥ 0.30 nats with doc-bootstrap 95% CI excluding 0.10,
                         same sign of (g*(rare) − g*(common)) on ≥2 models and
                         ≥2 domains, and GLM secondary statistic agrees in sign.
PLAN B (no usable effect): otherwise.
NON-ADDITIVE FLAG:       if g*(b) estimates are sign-inconsistent across the
                         margin window (curves cross), additive g is wrong —
                         learned-h (2-D) becomes the only method variant carried
                         forward; C-nu and learned-g move to appendix.
```

## 2.6 Interpretation of the three outcomes

- **(a) Rare tokens hit MORE at fixed margin** (`g*(rare) > 0`): the original nu sign. Method = widen rare margins, bounded by fitted `g_hat` (expect ≤ ~1 nat — the PLAN's kappa=10 arithmetic stands as the cautionary tale). Efficiency risk is real (tail admission): the gate (§4.3) decides. Narrative: "models under-trust rare tokens they can't estimate."
- **(b) Rare tokens hit LESS at fixed margin** (`g*(rare) < 0`): the desmoothing prediction (Hewitt et al. 2022) — and what the GPT-2 smoke blow-up weakly suggests. Method = *shrink* rare-token margins; efficiency improves mechanically; Mondrian still guarantees rare-bucket coverage `1−delta`, so the "we abandon rare tokens" objection is answered by construction. Narrative: "the softmax tail is smoothing noise; frequency tells you where." This is the *easier* paper to defend: efficiency and the guarantee point the same way.
- **(c) No effect** (`Δ < 0.3` or non-replicating): margin is (approximately) frequency-sufficient. Plan B audit paper (§7): the finding is that calibrated min-p is frontier-optimal in the achievable class and per-bucket coverage of standard defaults is already near-uniform — a real, citable negative result with the same Figure 1.

**Figure-1 hygiene:** fit/tune on `D_tune`; regenerate the displayed figure on `D_test` at the very end from the frozen grids, so the paper's figure is not the tuning artifact.

---

# §3. Method family clarification

## 3.1 Triage (decisive)

| Method | Status | Reason |
|---|---|---|
| **learned-h** (threshold `−h_hat(m,n)`, conformalized) | **ESSENTIAL — the paper's general method** | It is the plug-in feature-restricted oracle; by Theorem 1 it is the right object. Everything else in the family is a constrained version of it. |
| **Mondrian-margin (frequency buckets)** | **ESSENTIAL — the paper's guarantee** | Only member with per-bucket validity (Prop 3). Even if it loses on efficiency, it anchors the "guarantees" column of every table. |
| **C-margin** | **ESSENTIAL — the null** | The `(m)`-class optimum; every family member is measured against it. |
| learned-g (additive `m − g_hat(n)`) | Recommended ablation | Interpretable; the Corollary (§1.4) predicts when it matches learned-h. If NON-ADDITIVE FLAG trips, appendix only. |
| C-nu (signed kappa) | **Demoted: one paragraph + one table row as "parametric ansatz"** | Keep for lineage and interpretability; expect it to lose to learned-g/h. Its defeat is a *finding* ("the inverse-sqrt ansatz is mis-shaped; the data prefer …"), reported in one sentence. |
| `m/std(s(x)) − g(n)` (context-norm × offset) | Optional, appendix | Tests axis interaction; cut first under space pressure. |

## 3.2 Title/body decision on "nu"

Remove "nu" from the title (already done in PLAN) **and from all section headings and method names in tables**. It survives as: (i) the ansatz paragraph, (ii) a footnote acknowledging the project's origin. Reviewers punish branded methods that lose their own comparisons; they reward papers that subsume their initial idea into a principled family and report which member wins.

## 3.3 If Mondrian or learned-h beats C-nu (expected)

Nothing breaks — this is the anticipated outcome and the paper is *pre-structured* for it: the contribution is the family + the measurement + the guarantee, and the empirical section simply reports the family's internal ranking. The only forbidden move is quietly dropping C-nu from tables after it loses; report it, one row, with the shape mismatch shown against fitted `g_hat` (that comparison plot is genuinely informative).

## 3.4 If learned-h barely beats C-margin but Mondrian-margin matches learned-h

Then the efficiency story is thin and the guarantee story is the paper: frequency-Mondrian delivers per-bucket coverage at ≈no efficiency cost, while marginal methods silently undercover rare buckets by X pp (measured). That is a legitimate Plan-A-minor variant; the gate (§4.3) encodes it as passing G2 while failing G1.

---

# §4. Baseline and gate audit

## 4.1 Baseline verdict: one mandatory addition, one cheap standard addition

The PLAN's rows 1–6 (C-margin, C-logprob, C-zmargin, APS, RAPS, CNS) are necessary but **not sufficient**. Two additions:

1. **Entropy-Mondrian margin [MANDATORY — confound control].** Mondrian calibration of the *margin* score with buckets on context entropy quantiles (a context-feature analogue of the frequency-Mondrian method). Without it, any G2-style win of frequency-Mondrian is confounded: "conditioning helps" vs. "conditioning *on frequency* helps" are indistinguishable, and CNS (entropy × APS) is not a clean control because it also changes the base score from margin to cumulative mass. With it, you get a 2×2: {margin, APS} × {entropy-, frequency-}conditioning. This addition is cheap (same code path as frequency-Mondrian with a different bucket function) and closes the paper's largest inferential hole.
2. **TS+APS [cheap standard].** Temperature-scale the logits on `D_tune` (NLL-optimal T), then APS. This is the "did you try basic recalibration first?" reviewer question, answered for ~30 LOC. If TS+APS closes most of the gap the frequency story claims, you need to know *now*.

Explicitly **not** implemented (cite only, one line each in related work): Ulmer et al. kNN-weighted sets (heavy, different assumption regime), Mirostat (online control, not a prediction set), speculative/contrastive decoding (different problem). C-typical (`|log p_i + H(p)|` score) is a nice-to-have appendix row if time permits; rank it last.

**Final mandatory table (10 calibrated rows + uncalibrated sweep):**

```text
Monotone / context axis:  C-margin | C-logprob | C-zmargin | APS | RAPS | TS+APS | CNS | entropy-Mondrian-margin
Frequency axis (family):  frequency-Mondrian-margin | learned-h    (+ learned-g, C-nu as ablation rows)
Context for plots:        uncalibrated top-p / min-p / top-k / top-nsigma / eta / typical sweeps
```

## 4.2 Metric definitions (frozen)

- **Realized coverage** with document-clustered bootstrap 95% CI (`B=1000`).
- **Set-size distribution:** mean, median, p90, p99 (report all four; methods with equal means can differ 100× at p99, and p99 is what decoding latency feels).
- **Pareto AUC (headline):** for the delta grid `{0.2, 0.1, 0.05, 0.02, 0.01}`, plot `(realized coverage, log10 mean size)`; linearly interpolate onto the coverage grid `[0.90, 0.99]` (step 0.005); AUC = mean of interpolated `log10` sizes — lower is better. Methods whose realized-coverage range does not span the grid are extrapolation-masked, never extrapolated.
- **CovGap:** `max_k (1 − delta − coverage_k)_+` over frequency buckets (mass-quantile buckets, `n_k ≥ 100` enforced), at each delta.
- **Bucket size profile:** mean size per bucket at matched marginal coverage — this is where "rare-token abandonment" would show; report it so no reviewer has to ask.
- **Retained mass:** secondary, appendix (it is optimizable by degenerate huge sets, hence never a headline).

## 4.3 The gate, rewritten (final; commit before any paper-grade GPU run)

Preconditions for evaluating the gate at all: `n_test ≥ 100k` positions, `≥ 2k` documents, per-domain; frozen primary configs; leakage tripwire green.

```text
G1 (efficiency win):
  at delta in {0.10, 0.05}:  mean size ratio (best frequency-family member /
  best non-frequency calibrated baseline incl. entropy-Mondrian-margin and
  TS+APS) ≤ 0.90, at realized coverage within ±0.5pp of the baseline's,
  with doc-clustered bootstrap 95% CI of the ratio entirely below 1.00,
  replicated on ≥2 model families × ≥2 domains.
  AND Pareto-AUC improvement ≥ 0.02 log10 units on the same cells.

G2 (conditional-coverage win):
  at matched marginal coverage (±0.5pp) and mean size inflation ≤ +10%:
  CovGap(frequency-Mondrian) ≤ 0.5 × CovGap(C-margin)  AND
  CovGap(frequency-Mondrian) ≤ 0.8 × CovGap(entropy-Mondrian-margin),
  at delta = 0.05, replicated on ≥2 models × ≥2 domains.

PASS = G1 or G2   → unlock Phase 3 (downstream generation) and Plan A claims.
G2-only PASS      → Plan A-minor: guarantee-led paper (§3.4); Phase 3 optional.
FAIL              → Plan B; no Phase 3; no method superiority claims anywhere.
```

The two changes from PLAN §2.5: the comparison set for G1 now *includes the new controls* (a win that evaporates against TS+APS or entropy conditioning is not a win), and G2 now requires beating the entropy-conditioned control specifically, not just C-margin.

---

# §5. Data split and validity protocol

## 5.1 The four-way split, implementation-exact

```text
D_freq:  disjoint C4-train shard(s) by URL hash (plus Pile counts for Pythia,
         Dolma counts for OLMo-2 as pretraining-aligned variants). Volume
         target ≥ 5e9 tokens for stable tail counts; counts table frozen and
         content-hashed first.
D_tune:  40% of eval-corpus documents   (Phase 0 h_hat, g_hat, kappa, buckets,
                                         TS temperature, all model selection)
D_cal:   25% of eval-corpus documents   (conformal quantiles ONLY)
D_test:  35% of eval-corpus documents   (reported numbers; one pass per frozen
                                         config)
```

Assignment rule: `split = hash(url_or_doc_id + global_salt) mod 100` with fixed bands — deterministic, reproducible, and committed as a manifest of doc-ID → split (content-hashed). Any change of salt or bands = a new protocol version, logged.

## 5.2 One-position-per-document calibration (theorem-grade rows)

- For each `D_cal` document: `t* = Unif{16, …, len−1}` drawn with `rng(seed = hash(doc_id + salt_cal))`; the calibration score is computed at `t*` only. Same construction on `D_test` for the guarantee-check table. Result: calibration and test pairs are i.i.d. draws from the same (document, position) law ⇒ (A2) exact ⇒ Props 2–3 hold with no asterisk.
- Pooled-position variants (stride 4 to bound storage) are the *empirical* main tables. The paper labels every table row `[G]` (guarantee-grade) or `[E]` (empirical); coverage claims in prose reference `[G]` rows only. This labeling discipline is cheap and disarms the exchangeability objection completely.
- Appendix check: `[G]` vs `[E]` coverage agreement within CI (expected; if not, the domain has pathological document structure — investigate before publishing anything).

## 5.3 Leakage risks, ranked, with mitigations

1. **Near-duplicate documents straddling splits** (C4 is full of them): MinHash-LSH dedup (shingle 13-grams, Jaccard ≥ 0.8 → same cluster; keep one representative per cluster, assign *clusters* to splits, not documents). Without this, `D_cal`/`D_test` dependence quietly invalidates (A2) and shrinks every CI. **Blocking** for paper-grade runs.
2. **Frequency counts touching eval text:** structurally prevented by §5.1; enforced by the tripwire test (§8).
3. **Kappa/bucket/temperature tuned on feedback from `D_cal`/`D_test`:** prevented by freezing `configs/primary_*.json` (content-hash in the run log) before first `D_test` read; the runner refuses to evaluate a config whose hash is absent from the pre-registered list.
4. **Pretraining contamination** (model has memorized eval text — C4 overlaps Dolma; WikiText is everywhere): cannot be fully removed for open models. Mitigate: (i) state as a limitation; (ii) one robustness slice of demonstrably post-cutoff text (e.g., a 2026 news/wiki crawl slice) for the flagship model; (iii) note the audit question is still well-posed on seen text — `h` is a property of (model, corpus) — but generalization claims soften.
5. **Cross-domain reuse of `D_tune`-fitted `g_hat`:** fits are per-domain; the cross-domain transfer of `g_hat` is an explicit Phase-2 experiment, never silent.

## 5.4 Which results may claim finite-sample marginal coverage

Only: `[G]`-rows, frozen primary config, dithered scores, per (model, domain) cell — Prop 2 for global methods, Prop 3 per-bucket for Mondrian methods. Everything else — pooled positions, shifted domains, any post-hoc config — is empirical and must be labeled so. No exceptions, including in the abstract.

---

# §6. Experimental matrix under a rented-GPU budget

Prediction-set phases are forward-pass-only; the sufficient-statistics cache (PLAN §2.6) makes every method/delta/g ablation a **zero-GPU replay**. Budget numbers below assume one rented A100-80GB (or 2×4090 with bf16 + device_map); they are deliberately conservative.

## 6.1 Tier 0 — run first (≈ 15–20 GPU-hrs total)

| Step | Models | Domains | Positions | Est. GPU-hrs |
|---|---|---|---|---|
| Phase 0 | GPT-2-large (local sanity), Pythia-1.4B, Qwen2.5-3B | C4-val, OpenWebMath | 300k/pair | ~6 |
| Phase 1 essential (8 mandatory baselines + 2 family) via suffstats cache | Pythia-6.9B, Qwen2.5-3B | C4-val, OpenWebMath | 100k test + 5k `[G]` cal | ~8 |
| Gate evaluation | — (replay) | — | — | 0 |

Decision output: Plan A vs. B, gate first read, NON-ADDITIVE FLAG status. **Nothing else is allowed on the rented GPU until this tier's memo is written.**

## 6.2 Tier 1 — minimum viable ICML matrix (≈ +40–60 GPU-hrs)

- Models: + **OLMo-2-7B** (exact Dolma counts — the leakage-proof flagship; promote to headline model), + Llama-3.1-8B (tokenizer axis), + Pythia-410M & 2.8B (scale trend, cheap).
- Domains: + code (The Stack val slice). Keep WikiText-103 out of the main tables (60-document corpus fails §5.2); it may appear as an appendix curiosity only.
- Phase 2 analyses (scale, tokenizer transfer, counts provenance, shift C4→GSM8K/code): all replay/forward-light.
- This tier is submission-viable: 5 model families / scale ladder × 3 domains × 10 calibrated methods with `[G]` and `[E]` rows.

## 6.3 Tier 2 — full matrix (only if Tier 1 is clean; ≈ +80–150 GPU-hrs)

- - Gemma-2-9B (256k vocab extreme of the tokenizer axis), fresh post-cutoff text slice (§5.3.4), delta grid extension to 0.005 (long-sequence regime), Phase 3 downstream **iff gate passed**: self-consistency (500 problems × 16 samples × 3 seeds) + open-ended with MAUVE — this is the single most expensive line item (~60–100 GPU-hrs alone); it buys corroboration, not the claim. If budget forces a choice between Phase 3 and the tokenizer axis, **keep the tokenizer axis**.

## 6.4 Fallback workshop matrix (if the rented budget collapses, ≈ 10 GPU-hrs)

GPT-2-large + Pythia-1.4B + Qwen2.5-3B, C4 + OpenWebMath, Phase 0 + the 8-baseline essential Phase 1, no Phase 3 — submitted to a UQ/reliable-ML workshop as the measurement paper. This exact artifact is also the ideal ICML pilot, so no work is wasted.

**Do not run yet, at any tier:** controlled channels (`exp5b` — cut from the paper entirely; it belongs to the old noise-channel story), any ≥13B model, Mirostat-style online methods, any generation experiment pre-gate.

---

# §7. Paper shape decision

## 7.1 Identity: audit-first hybrid (decided)

The paper is a **measurement paper with a method payoff**, not a method paper with a motivating plot. Rationale: (i) Figure 1 (the `h` surface) is novel, cheap, and unfalsifiable-by-reviewers — it is the part no one can take away; (ii) the method family's fate is genuinely uncertain until Phase 0; (iii) ICML has repeatedly rewarded "careful audit + principled fix" papers, and punishes "new sampler beats baselines" papers — the graveyard of decoding heuristics is large.

## 7.2 Title (one per Phase-0 outcome, decided now so writing can start)

- **Outcome (a) or (b) + gate pass:** *"Is Margin Enough? Frequency-Aware Conformal Prediction Sets for Language-Model Truncation"*
- **Outcome (b) specifically, if the desmoothing sign dominates the story:** *"Calibrated Tail Pruning: Token Frequency Locates the Noise in Language-Model Logits"* (secondary option; only if effects are large and uniform)
- **Outcome (c):** *"Is Margin Enough? A Conformal Audit of Truncation Sampling"* (Plan B)

## 7.3 Final claim stack (Plan A wording; Plan B strikes C3′ and softens C2′)

```text
C1′  A representation lemma and feature-class lattice unifying truncation rules
     as margin rules, with conformal calibration giving finite-sample coverage
     for every member under a stated, exactly-implemented split protocol. [G]
C2′  The first measurement of frequency-modulated margin miscalibration
     h(m, n) across model scale, tokenizer, and domain — adjudicating between
     the desmoothing and estimability hypotheses, with a feature-restricted
     optimality theorem making h the exact criterion for whether frequency
     can help. [Theorem 1 + Figure 1]
C3′  Frequency-aware members of the family (Mondrian-margin, learned-h)
     improve the coverage/size Pareto frontier and/or per-bucket coverage over
     margin-, probability-, and entropy-conditioned calibrated baselines,
     with per-frequency-bucket finite-sample guarantees; downstream generation
     corroborates under the pre-registered gate. [E + G]
```

**What makes it ICML-worthy:** a decision-relevant question every LLM deployment implicitly answers (what to truncate), a new measurement instrument with a matching optimality criterion, guarantees that survive review because their assumptions are implemented rather than assumed, and full replay-level reproducibility (suffstats cache published).

**What forces a downgrade:** Phase-0 effect `Δ < 0.3` nats or non-replicating across models/domains (→ Plan B); Plan B *and* per-bucket coverage of standard defaults turning out already near-uniform (the audit finds nothing broken → workshop); gate failing on all cells at Tier 1 with only single-model wins (→ workshop or a hardened negative-result paper, which is still submittable to ICML's journal-to-conference track or a strong workshop).

---

# §8. Implementation tasks for Codex (ordered; two-week sprint)

## 8.1 Engineering checklist (strict order; PR-sized units)

```text
T1  experiments/conformal.py
    - kappa: float (signed); alpha param unchanged
    - add mondrian_quantiles(scores, groups, delta, min_bucket=ceil(5/delta))
    - add score dithering U(0,1e-6) behind flag --dither (default on)
T2  experiments/freq_table.py  [NEW]
    - build/load counts from D_freq shards; two tables per model:
      pretraining-aligned (Pile/Dolma) and domain (C4-train shard)
    - frozen exclusion list per tokenizer (PAD/BOS/control ids)
    - emits content-hash; all downstream configs reference the hash
T3  experiments/splits.py  [NEW]
    - MinHash-LSH dedup (13-gram shingles, J≥0.8) → cluster manifest
    - hash-banded cluster→split assignment (40/25/35), salted; manifest JSON
    - one-position-per-doc sampler rng(seed=hash(doc_id+salt))
T4  experiments/suffstats.py  [NEW]
    - per position: (margin-bin × freq-bucket) int16 count matrix +
      true-token record (margin, bucket, rank, cum-mass, entropy scalar)
    - writer (streaming, sharded .npz) + replay evaluator for ALL methods
T5  experiments/phase0_reliability.py  [NEW]
    - NUM/DEN accumulation, masking/coarsening, aligned-offset g*(b),
      doc-bootstrap CIs, GLM secondary test, Figure-1 plots, decision memo JSON
T6  experiments/methods.py  [REFACTOR from samplers.py scoring paths]
    - registry: c_margin, c_logprob, c_zmargin, aps, raps, ts_aps, cns,
      mondrian_margin(freq|entropy buckets), learned_h, learned_g, c_nu
    - every method consumes suffstats replay OR live logits via one interface
T7  experiments/check_prediction_set_gate.py  [REWRITE to §4.3]
    - preconditions, G1/G2 with clustered-bootstrap CIs, [G]/[E] labeling,
      machine-readable PASS/G2-only/FAIL verdict
T8  configs/: phase0_{model}_{domain}.json, primary_{model}_{domain}.json
    (frozen, content-hashed), gate_thresholds.json (pre-registered)
T9  scripts/: run_phase0.sh, run_phase1_tier0.sh, run_gate.sh — each idempotent,
    each refusing to run if manifests/tripwire tests are stale
```

## 8.2 Tests that must be green before any paper-grade GPU spend

```text
U1  test_conformal_synthetic.py    exchangeable synthetic data: empirical
                                   coverage within binomial band of 1−delta
                                   over ≥200 resamples (marginal + Mondrian)
U2  test_equivalences.py           APS ≡ conformal top-p; C-margin ≡ calibrated
                                   min-p; C-nu(kappa=0) ≡ C-margin, on random
                                   logits (exact, up to dithering tolerance)
U3  test_tripwire.py               D_freq/D_tune/D_cal/D_test doc-cluster
                                   manifests pairwise disjoint; runner aborts
                                   on intersection or on unregistered config hash
U4  test_suffstats_replay.py       replay metrics == direct-from-logits metrics
                                   within fp tolerance on a 500-position fixture
U5  test_one_per_doc.py            sampler determinism + uniformity; [G]-row
                                   pipeline produces exactly one score per doc
U6  test_phase0_stats.py           aligned-offset recovery on synthetic h with
                                   a known injected shift (±0.5 nats) and null
                                   (0 nats) — no false positives at 95% CI
```

## 8.3 Two-week sprint with stop conditions

```text
D1–D3    T1–T4 + U1/U2/U4.            STOP if U1 coverage sanity fails → fix
                                       before anything else exists.
D4–D5    T3 manifests for C4/OpenWebMath + counts tables (T2) + U3/U5.
D6–D8    T5; Phase 0 on GPT-2-large + Pythia-1.4B (local/cheap GPU) + U6.
         STOP if curves are seed/split-unstable (rerun with disjoint D_tune
         halves; g*(b) must agree within CI) → diagnose before renting.
D9       Rent GPU: Phase 0 on Qwen2.5-3B + Pythia-6.9B × 2 domains.
         Write the decision memo: PLAN A/B + sign + NON-ADDITIVE FLAG.
D10–D12  T6/T7; Tier-0 Phase 1 (suffstats pass on Pythia-6.9B, Qwen-3B ×
         C4, OpenWebMath); gate replay.
         STOP if any method's realized coverage deviates from 1−delta beyond
         CI on [G] rows → protocol bug, halt all conclusions.
D13      Gate verdict + Tier-1 go/no-go + budget re-estimate.
D14      Write-up of Phase 0/Tier 0 (this becomes the paper's §4 draft and,
         verbatim, the workshop fallback). Freeze primary configs for Tier 1.
```

**Standing stop conditions (any time):** tripwire red; `[G]` coverage violation; suffstats replay mismatch; Phase-0 sign flipping between models — each halts GPU spend until a written diagnosis lands in `docs/reports/`.

---

## Final word

The PLAN survives this stress test structurally: the family framing, the four-way split, the gate philosophy, and the Phase-0-first ordering are right. What it lacked was adversarial specificity — the entropy-conditioned control, the TS+APS sanity baseline, the feature-class lattice that keeps Theorem 1 honest, dedup before exchangeability claims, `[G]`/`[E]` labeling, powered sample sizes, and numeric pass/fail lines. Those are now specified. The project's fate rests on one number — the Phase-0 effect size `Δ` — and the plan now gets to that number inside two weeks for under twenty GPU-hours, with a publishable exit on every branch.


