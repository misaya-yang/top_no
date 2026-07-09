# Repo Reconciliation Addendum (v1.1)

**Date:** 2026-07-09

**What this is:** an update to the two prior documents — the PLAN ("Frequency-Offset Margin Rules: Refined Theory and Experimental Program") and the SPEC ("Stress Test → Implementation-Ready Specification") — after direct inspection of the now-pushed `main` branch of `github.com/misaya-yang/top_no`. The prior documents were written against the project brief's *description* of the code; this addendum reconciles them with the code itself.

**Headline:** the analysis and plan survive; the *statuses* change. Two items the SPEC treated as risks are now **verified defects** in pushed code, one SPEC task turns out to be already half-done, and the repo's public narrative contradicts the current definition of the idea. Nothing in the theory program, Phase-0 statistical design, gate criteria, tier structure, or paper-shape decision changes.

---

## §A. What remains in force, unchanged

The following SPEC sections stand exactly as written and are not repeated here:

- SPEC §1 (theory hardening: Lemma 1 / Prop 2 / Prop 3 / Theorem 1 repairs and proof obligations)
- SPEC §2 (Phase 0 executable diagnostic: population, grids, estimator, guardrails, decision rule, 300k-position floor)
- SPEC §3 (method triage: learned-h / Mondrian-margin / C-margin essential; C-nu demoted to ansatz)
- SPEC §4.1–4.3 (baseline set incl. entropy-Mondrian control and TS+APS; frozen metric definitions; the G1/G2 gate criteria)
- SPEC §5 (four-way split targets, one-position-per-document, dedup, [G]/[E] labeling)
- SPEC §6 (Tier 0/1/2 matrices and budgets) and §7 (paper identity, titles, claim stack)

This addendum supersedes **SPEC §8 (implementation tasks and sprint)** with repo-verified statuses and a PR-by-PR plan, and adds one new work item (PR-0, narrative alignment) that the SPEC did not contain because the drift was not visible until the push.

---

## §B. Assessment changes after code inspection

| Prior status (PLAN/SPEC) | Verified status on `main` | Consequence |
|---|---|---|
| "Frequency counts *may* leak from eval text" (risk) | **CONFIRMED DEFECT:** `build_token_counts()` in `experiments/eval_prediction_sets.py` accumulates counts over the *entire loaded text pool*, including calibration and eval positions | (A1) violated → no run from this pipeline can claim conformal validity; also corrupts the frequency buckets (see below) |
| "Calibration/eval positions may not be exchangeable" (risk) | **CONFIRMED DEFECT:** positions are consumed *sequentially* — first `n_calibration` positions of the stream calibrate, the next `n_eval` evaluate. This is a temporal split (early documents vs. later documents), not an exchangeable one | (A2) violated by construction, independent of the document-dependence issue; coverage numbers from this pipeline are biased in an unknown direction |
| "Gate compares calibrated vs. uncalibrated" (inference from brief) | **CONFIRMED:** gate matches methods by substring (`conformal_nu*` vs. names containing `top_p`/`min_p`/`fixed_margin`/`top_nsigma`), hardcoded `target_coverage=0.95, tol=0.02`, size-ratio ≤ 1.25 | Gate can pass or fail for reasons unrelated to the score; SPEC §4.3 rewrite confirmed necessary, now with exact anchors |
| "kappa must be generalized to signed" (SPEC T1) | **HALF-DONE ALREADY:** no sign assertion exists anywhere; negative kappa runs today. But defaults are `kappa=10.0, m0=3.0` — the exact configuration that produced 14k-token supports — and nothing tunes kappa | T1 shrinks to: dithering + `mondrian_quantiles` + tuning path + safer defaults |
| Frequency buckets assumed corpus-scale | **ARTIFACT CONFIRMED:** eval buckets are `0, 1-2, 3-10, 11-100, >100` — count bands this tiny only make sense because counts come from the small eval pool (the leakage bug). Corpus-scale counts need log10 bands to ≥10^7 | Bucket tables must be rebuilt when PR-1 lands; all existing bucket-coverage numbers are doubly invalid |
| Repo narrative assumed transitional | **THREE-WAY DRIFT CONFIRMED:** README still titles the project "Truncation Sampling as Hypothesis Testing"; `CLAIM_STACK.md` still leads with "nu-Sampling: Frequency-Calibrated Logit Prediction Sets"; the current definition (offset-margin family, measurement-first, nu as ansatz) exists only outside the repo | New PR-0: narrative alignment. Anyone (including a coding agent) executing from the repo today would build the superseded idea |
| Qwen3B queue possibly already run | **NOT RUN / NOT COMMITTED** (results path 404s; only the GPT-2 smoke is committed) | Correct call — and the gate must stay closed until PR-1..3 merge |

Verified-correct items (no action, for the record): the conformal quantile `rank = min(ceil((n+1)*(1-delta)), n)`; top-p crossing-token retention; truncation on raw logits with temperature after; error-on-invalid truncated distributions; `min_p`/`fixed_margin` implemented (equivalence testable); `conformal_nu` requiring an explicit `q_hat`.

---

## §C. Scorecard: SPEC §8 tasks vs. verified repo state

| Task | Status on `main` | Notes |
|---|---|---|
| T1 `conformal.py` upgrades | **PARTIAL** | signed kappa incidentally works; missing: dithering, `mondrian_quantiles`, kappa/g tuning entry point |
| T2 `freq_table.py` | **MISSING — and its absence is the P0 leakage bug** | counts currently built inline from eval pool |
| T3 `splits.py` | **MISSING — and its absence is the P0 exchangeability bug** | sequential stream split in place today |
| T4 `suffstats.py` | MISSING | |
| T5 `phase0_reliability.py` | MISSING | |
| T6 methods registry | MISSING | 7 hardcoded methods; no APS/RAPS/CNS/TS+APS/entropy-Mondrian/learned-h; legacy `nu_topp_floor`/`nu_entropy`/`nu_mathboost` still first-class strategies |
| T7 gate rewrite | NOT STARTED (old gate confirmed live) | |
| T8 frozen configs | PARTIAL | run configs exist; no `primary_*.json` / `gate_thresholds.json` pre-registration |
| T9 idempotent scripts w/ staleness refusal | PARTIAL | scripts exist; no manifest/tripwire enforcement |
| U1–U6 test suite | **NONE PRESENT** | existing 14 unit tests cover sampler mechanics only |

---

## §D. Revised implementation plan: five PRs, strict order

Each PR is merge-blocked on its listed tests. No GPU rental before PR-3 is merged.

### PR-0 — Narrative alignment (½ day; do first, it prevents wasted agent work)

- README: replace the "Truncation Sampling as Hypothesis Testing" framing with the current one-liner:
  ```text
  Frequency-offset margin rules: measuring whether token frequency carries
  information beyond the logit margin, and calibrating truncation into
  prediction sets with finite-sample (per-bucket) coverage.
  ```
- `docs/paper/CLAIM_STACK.md`: replace the claim stack with SPEC §7.3 (C1′–C3′) and both conditional titles; delete "nu-Sampling…" as primary; move the old stack to `docs/paper/ARCHIVE_CLAIM_STACK_v0.md`.
- Commit the PLAN and SPEC documents into `docs/paper/` so the repo, not a chat log, is the source of truth.
- Mark `nu_topp_floor`, `nu_entropy` as deprecated in `samplers.py` docstrings; gate `nu_mathboost` behind an explicit `--legacy` flag (it is on the do-not-claim list and must not be reachable from paper pipelines).
- Add a README "results status" note: the committed GPT-2 smoke is a pipeline link-test whose protocol is superseded; no committed result is paper evidence.

### PR-1 — Kill the two P0 defects (2–3 days)

- **`experiments/freq_table.py` (new):** counts from a `D_freq` source only (C4-train shard by URL-hash; optional Pile/Dolma pretraining-aligned tables). Emits content-hashed artifact `{model_tokenizer}_{source}_counts.pt` + JSON sidecar (source manifest hash, exclusion list). `eval_prediction_sets.py` loses `build_token_counts()`; it *loads* a table and refuses to run without one whose source manifest is disjoint from eval manifests.
- **`experiments/splits.py` (new):** MinHash-LSH dedup (13-gram shingles, J ≥ 0.8) → cluster manifest; salted hash-band assignment of *clusters* to `D_tune/D_cal/D_test` (40/25/35); seeded one-position-per-document sampler for `[G]` rows; stride-4 pooled sampler for `[E]` rows. `eval_prediction_sets.py`'s sequential skip/take logic is deleted, not bypassed.
- **Bucket tables rebuilt (two, per SPEC §2.2):** diagnostic log10 bands `B0..B8`; method-side mass-quantile buckets with `n_k ≥ ceil(5/delta)` floor. The `0/1-2/3-10/11-100/>100` scheme is removed.
- **Tests:** U3 (tripwire: pairwise-disjoint manifests; runner aborts on intersection or unregistered config hash), U5 (one-per-doc determinism/uniformity), plus a regression test asserting `eval_prediction_sets.py` raises if handed a counts table whose manifest intersects eval docs.

### PR-2 — Conformal core + methods registry (2–3 days)

- Finish T1: `mondrian_quantiles(scores, groups, delta, min_bucket)`, score dithering `U(0,1e-6)` (default on), kappa/g tuning entry point reading `D_tune` only; change `nu` strategy defaults from `kappa=10.0, m0=3.0` to config-required (no silent defaults that reproduce the 14k-support failure).
- T6 registry (`experiments/methods.py`): `c_margin, c_logprob, c_zmargin, aps, raps, ts_aps, cns, mondrian_margin(freq|entropy), learned_h, learned_g, c_nu` — one interface over live logits or suffstats replay. Gate and eval identify methods by registry key, never by name substring.
- T4 `suffstats.py` writer + replay evaluator.
- **Tests:** U1 (synthetic exchangeable coverage, marginal + Mondrian), U2 (APS ≡ conformal-top-p; C-margin ≡ calibrated min-p; C-nu(κ=0) ≡ C-margin), U4 (replay ≡ direct within fp tolerance).

### PR-3 — Gate rewrite (1 day)

- Replace `check_prediction_set_gate.py` logic with SPEC §4.3 verbatim: preconditions (n_test ≥ 100k, ≥ 2k docs, tripwire green, frozen configs), G1/G2 with doc-clustered bootstrap CIs, comparison set = calibrated baselines *including* `entropy-Mondrian-margin` and `ts_aps`, three-way verdict `PASS / G2-only / FAIL`, `[G]`/`[E]` labels in the report JSON.
- `configs/gate_thresholds.json` committed in the same PR (pre-registration).

### PR-4 — Phase 0 (2–3 days, then the decisive runs)

- T5 `phase0_reliability.py` per SPEC §2 (NUM/DEN scatter-add, masking/coarsening, aligned-offset `g*(b)`, doc-bootstrap CIs, GLM secondary, decision-memo JSON). Test U6 (synthetic shift recovery ±0.5 nats; no false positive on null).
- Runs: GPT-2-large + Pythia-1.4B locally, then Qwen2.5-3B + Pythia-6.9B on rented GPU × {C4-val, OpenWebMath}, 300k positions each. Output: the Plan A/B decision memo.

### Updated sprint (relative days; D1 = first working day)

```text
D1        PR-0 merged.
D2–D4     PR-1 (P0 fixes) + U3/U5 green.
D5–D7     PR-2 + U1/U2/U4 green.        STOP if U1 coverage sanity fails.
D8        PR-3 merged; thresholds frozen.
D9–D11    PR-4 code + local Phase 0.     STOP if g*(b) unstable across
                                          disjoint D_tune halves.
D12       Rented GPU: Phase 0 on Qwen-3B + Pythia-6.9B; decision memo.
D13–D14   Tier-0 Phase 1 via suffstats on 2 models × 2 domains; gate verdict;
          Tier-1 go/no-go and budget re-estimate.
```

Net schedule effect vs. SPEC §8.3: zero — the half-done T1 offsets the new PR-0.

---

## §E. Standing rules re-affirmed for the current repo

1. **No paper-grade GPU run until PR-1..3 are merged and U1–U5 are green.** The committed pipeline, run today, produces numbers that are invalid twice over (A1 and A2), and the current gate could bless them.
2. **No committed result on `main` is citable** — including the smoke run — until regenerated under the new protocol; the README must say so (PR-0).
3. **Legacy strategies stay out of paper pipelines.** `nu_mathboost` behind `--legacy`; `nu_topp_floor`/`nu_entropy` deprecated; none appear in the methods registry.
4. **Everything else** — theory obligations, Phase-0 statistics, baselines including the entropy-conditioned control, metric definitions, tiers, titles, claim stack, timeline to ICML 2027 — proceeds per PLAN/SPEC without modification.