# Final Decision and Paper Draft v0.1

**Project:** top_no → **CASPR** (Calibrated Success-Preserving Rollout Pruning)
**Role of this document:** binding adjudication of the two competing proposals (`topno_gpt5.6pro.md`, "Report A"; `topno_deep_review_20260710.md`, "Report B"), followed by the first full draft of the single resulting paper.
**Date:** 2026-07-10 (Asia/Shanghai). **Author:** final research lead.
**Status of every claim in this document:** no experimental results exist yet for the new method; every number attached to the new method below is a design target, a feasibility estimate, or a preregistered gate — never a result. No proof beyond the sketches given is claimed complete.

---

# Part 0 — Verification ledger

Both reports were treated as competing proposals, not as authorities. I re-verified their load-bearing citations by web search on 2026-07-10. Labels used throughout this document:

- **[V]** — verified today (arXiv/OpenReview/official page located and consistent with the report's description).
- **[C]** — pre-mid-2025 literature I am confident of from training data; not re-verified today.
- **[R]** — reported by one or both reviews; not independently located today; treated as plausible but at-risk.
- **[U]** — could not be located today, or located with material discrepancies; treated as unverified.

| Item | Status | Notes from today's check |
|---|---|---|
| min-p critique, arXiv:2506.13681 | **[V]** | Real. Current title "Turning Down the Heat: A Critical Analysis of Min-p Sampling in Language Models" (Report B cited an earlier title variant). |
| DeepConf, arXiv:2508.15260 | **[V]** | Real; Meta; windowed-confidence rollout filtering, heuristic, no distribution-free guarantee. |
| Certaindex, arXiv:2412.20993 | **[V]** | Real; retitled "Efficiently Scaling LLM Reasoning with Certaindex"; systems scheduler, no guarantee. |
| Conformal Thinking, arXiv:2602.03814 | **[V]** | Real — **and now published at ICML 2026** (Apple ML page + author announcement). Single-trajectory dual-threshold stopping with distribution-free risk control. Both reports listed it as a preprint; its status has upgraded, which matters for our novelty margin. |
| CROP, arXiv:2605.30085 | **[V]** | Real; "Conformal Certification of Reasoning Trace Prefixes". |
| Truncation Blind Spot, arXiv:2603.18482 | **[V]** | Real; under review. |
| ORCA, arXiv:2604.01170 | **[V]** | Real; "Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning". Crowds the *online* calibration cell. |
| Local Branch Routing, arXiv:2606.25354 | **[V]** | Real; trained router over local lookahead branches; no statistical guarantee. |
| ATTS, arXiv:2509.15148 | **[V]** | Real; ICLR 2026; conformal machinery inside speculative/asynchronous test-time scaling. |
| p-less Sampling, arXiv:2509.23234 | **[V]** | Real; ICLR 2026 (OpenReview). Confirms static-sampler space is not formally closed, but does not change the verdict below. |
| Anytime-Valid Conformal Risk Control, arXiv:2602.04364 | **[V]** | Found incidentally; directly relevant machinery for our *open* online extension (e-process route), and a mild scoop risk for anyone building the online cell. |
| Early stopping via confidence dynamics, arXiv:2604.04930 | **[V]** | Real. |
| "July 7, 2026 recall-controlled cascade" (Report A) | **[U]** | Could not locate. Report A's closest-competitor claim is therefore unverified. The *idea* of a recall-style contract stands on its own math (we adopt and generalize it), but the concurrent-work risk must be re-swept at submission. |
| REFRAIN / ReASC / SAT / "confidence-weighted token set cover" / "dynamic value-thresholded abstention (ICML 2026)" (Report A) | **[R/U]** | Not located today; Report A cites no arXiv IDs for these. Treated as at-risk baselines: included in the baseline list conditionally on locating them. |
| Min-k Sampling (ACL 2026), VACP arXiv:2512.22682, CWWI (NeurIPS 2025 wksp), ARC-Decode, "When Is a Draft Accepted?" arXiv:2606.30265, Conformal Policy Control arXiv:2603.02196 | **[R]** | Consistent across one or both reports; not individually re-verified today; none is load-bearing for the chosen method. |
| CRC (arXiv:2208.02814, ICLR 2024), Learn-then-Test (Angelopoulos et al. 2021), RCPS (Bates et al. 2021), split/Mondrian conformal (Vovk et al.), ACI (Gibbs & Candès 2021), CLM (Quach et al., ICLR 2024), conditional-validity impossibility (Foygel Barber, Candès, Ramdas, Tibshirani 2021), betting confidence sequences (Waudby-Smith & Ramdas), constrained-online impossibility (Mannor, Tsitsiklis & Yu 2009), eta-sampling (Hewitt et al. 2022), BAT (Finlayson et al., ICLR 2024), min-p (ICLR 2025), speculative decoding (Leviathan et al.; Chen et al. 2023), FR-Spec (2025), semantic entropy (Kuhn et al.; Farquhar et al. 2024), ESC / adaptive self-consistency (2023–24), compute-optimal TTS (Snell et al. 2024), s1 / L1 / TALE, DeepSeek-R1-Distill (2025-01), Qwen3 (2025-04), MATH-500, GPQA-Diamond (198 items), AIME (30 problems/year) | **[C]** | Standard machinery and assets; CRC's arXiv record confirmed incidentally today. |

Two factual discrepancies between the reports, resolved: (i) Report B's claim that Conformal Thinking was "preprint, single trajectory" was correct in February but is now stale — it is an ICML 2026 paper, which *strengthens* the case that only the population/set-valued cell remains open. (ii) Report A's AIME'24+'25 problem count is 60 (2×30), not the 120 implied by Report B's matrix; the design below uses 60.

---

# Part I — Adjudication

## I.1 What the Phase-0 evidence genuinely establishes, and what it invalidates

I re-derived the conclusions from the pilot statistics as described by both reports (the two reports agree on the raw facts; their descriptions of `phase0_stats.py`, the shift grid, and the four cell summaries are mutually consistent, and I checked the internal arithmetic of each critique).

**Established.**
1. The measurement instrument (margin-conditional hit-rate surfaces $h(m,n)$ with document-clustered inference, tune/calibration/test hygiene, fail-closed gates) runs end-to-end and produces stable *directional* reads. This is infrastructure-grade evidence, not method evidence.
2. All four model×domain cells show a **negative** signed rare-vs-reference shift: at fixed margin, rarer-bucket tokens hit *less*. Whatever the effect ultimately is, it has the desmoothing sign (Hewitt et al. 2022 [C]), which is the *opposite* of the implemented ν rule with positive κ. The original rule was pointed the wrong way. This is the single most decision-relevant Phase-0 fact.

**Invalidated or unidentified.**
1. The math-cell magnitudes (−2.0) sit exactly on the boundary of the ±2-nat search grid: they are censored statistics, not effect estimates. Nothing about "a 2-nat effect" is established.
2. Both web cells changed their data-selected reference bucket between halves; a contrast with a moving zero point does not estimate a stable quantity. The web effect (−0.2/−0.3) is unidentified.
3. The single unrestricted vocabulary permutation destroys morphology, script, token identity, internal-unigram structure, and frequency simultaneously; it cannot attribute the signal to *external* frequency. Occupancy was not matched (6 vs 3 valid groups), so even its negative-control role is compromised.
4. Consequently: Phase-0 does **not** establish a frequency effect, does **not** establish its absence, and **cannot** support any frequency-first method. It licenses exactly one methodological conclusion — the original frequency-offset sampler is dead as a headline — and one demand: any future use of frequency must survive internal-unigram, morphology, entropy, rank, and token-identity controls.

Both reports reach this verdict; I concur after independent checking. Two structural points both reports also make, which I verified as deductions and adopt as constraints on the final paper: token-level gold coverage is weakly connected to generation utility, and post-softmax set size is not production compute. Any successor method must attach its decision to a consumer that pays in tokens, FLOPs, or latency.

## I.2 The strict audit of the two proposed risk contracts

This is the decisive technical issue, so I treat it first and in full.

### I.2.1 Report A's contract is vacuous under rarity and adversely aligned with cost minimization

Report A's central object is the unconditional (marginal) catastrophic-pruning loss
$$L(S,Z)=\mathbf 1\{\max_{b\in\mathcal B}Z_b=1 \ \wedge\ \max_{b\in S}Z_b=0\},\qquad \text{contract } \mathbb E[L]\le\alpha .$$
Write $E$ for the event $\{\max_b Z_b=1\}$ ("the bank is solvable") and $p_E=\Pr(E)$. Three elementary facts, each fatal at a different difficulty level:

1. **Vacuity.** $\mathbb E[L]\le p_E$ for *every* policy, including "retain nothing." On any distribution with $p_E\le\alpha$ — e.g., a 7–8B model on AIME-class problems at small bank width — every policy is feasible and the contract certifies garbage.
2. **Difficulty-dependent dilution.** $\mathbb E[L]=p_E\cdot\Pr(\text{retained fails}\mid E)$. A marginal contract at level $\alpha$ is a conditional contract at level $\alpha/p_E$: at $p_E=0.25$ and $\alpha=0.05$, one is licensed to destroy the winners of 20% of all solvable instances. The contract's meaning silently degrades exactly where compute matters most.
3. **Adverse selection under cost minimization.** In a mixture population with a subgroup $G$ (say, hard problems with long traces) of mass $w_G$ and conditional solvability $p_{E|G}$, any policy is marginal-feasible while failing *every* solvable instance of $G$ whenever $w_G\,p_{E|G}\le\alpha$. Because hard problems have the longest continuations, they offer the largest per-instance savings — so the cost-minimizing feasible policy is *actively pushed* toward sacrificing precisely the solvable-hard instances. This is not a corner case; it is the optimizer's gradient direction.

Report A itself lists "conditional branch survival among full-bank-solvable examples" as a *secondary metric*, but its theorem, its calibration procedure, and its headline are all marginal. As stated, the contract fails the strictness test.

### I.2.2 Report B's contract is non-vacuous but noisy, selection-entangled, and its online spine rests on an unresolved (and, as proposed, incorrect) repair

Report B's per-round loss is disagreement with the full-budget reference, $\ell_t(\theta)=\mathbf 1\{A_\theta(Q_t)\neq A_{\text{full}}(Q_t)\}$. This is not vacuous under rarity (good), and it is replayable (good). But:

1. **Both-wrong noise.** On instances the reference gets wrong, $A_{\text{full}}$ is close to an arbitrary draw from the model's error distribution; a cheaper policy will disagree with it often and harmlessly. The contract burns risk budget on meaningless disagreement — worst exactly on hard benchmarks, where most instances are both-wrong.
2. **Penalized improvement.** If $A_\theta$ is right and $A_{\text{full}}$ wrong, the loss fires. A contract that punishes beneficial disagreement is the wrong shape.
3. **Preservation/selection entanglement.** The loss cannot distinguish "we killed every good branch" from "a good branch survived and the vote picked a bad one" — the two failures the user of a pruning method most needs to tell apart.
4. **The online spine rests on an open problem, and the proposed repair (i) does not work as stated.** Report B honestly flags that vote-based $\ell_t(\theta)$ is non-monotone in the aggressiveness knob. But its candidate repair — "count any vote-composition change as a loss; mechanically monotone for nested kill rules" — fails under either reading of its ambiguous phrasing: read as "the vote *outcome* changed relative to full," it is not monotone (under nested kills the survivor set shrinks monotonically, but the outcome-change indicator can toggle *off* as more branches die — remove a wrong-majority clique and the vote returns to the reference answer); read as "the survivor *composition* changed in any potentially vote-relevant way," it is monotone but so conservative that essentially any kill of a reference-agreeing branch scores as a loss, destroying the efficiency the scheduler exists to buy. The object that is simultaneously monotone, computable, and minimally conservative exists — the running-max envelope $\tilde\ell_\theta=\max_{\theta'\preceq\theta}\ell_{\theta'}$ along the nested path, the *least* monotone majorant of the realized loss on that path — and I adopt it in the open-problem section, with its conservatism priced empirically. This matters: it means Report B's Theorem B is one unproved lemma plus one broken-or-vacuous lemma away from holding.
5. **Overclaimed regret clause.** Theorem B claims, for adversarial streams, simultaneous long-run risk control, budget tracking, *and* regret against the best feasible-in-hindsight policy. For adversarially chosen constraint/loss sequences this collides with the classical impossibility of Mannor, Tsitsiklis & Yu (2009) [C] unless the comparator or stream class is restricted; the cited machinery (Mahdavi et al.; Yu & Neely) is for fixed or stochastic constraints, not adversarial ones. The validity (risk-tracking) clause is plausible; the regret clause as stated is not safe.

### I.2.3 The repaired contract, and the guarantee taxonomy the paper must state

Define the **success-conditional preservation risk** of a budgeted policy $\pi_\lambda$ relative to the frozen full-budget reference $\pi_{\text{full}}$:
$$R(\lambda)\;=\;\Pr\big(\pi_\lambda \text{ answers incorrectly}\;\big|\;\pi_{\text{full}} \text{ answers correctly}\big).$$
For unique-final-answer tasks this equals one-sided disagreement restricted to reference-success instances, so it inherits Report B's replayability while eliminating both-wrong noise and the improvement penalty; and it is exactly the non-vacuous version of Report A's survival object, lifted from "a viable branch survives" to "the win is preserved end-to-end," with the survival event retained as the diagnostic decomposition term. The taxonomy the paper commits to:

- **Marginal** ($\Pr(E\wedge\text{fail})\le\alpha$): vacuous under rarity, adversely selecting. Rejected; kept only as a negative exhibit (Proposition 1 + an empirical demonstration on real banks).
- **Success-conditional population** ($R(\lambda)\le\alpha$): non-gameable in the above senses; achievable distribution-free with finite-sample certificates by calibrating on the (observable-on-banks) reference-success subpopulation. **This is the paper's contract.**
- **Group-conditional (Mondrian)**: achievable per pre-declared, deployment-measurable groups; the strongest honest answer to residual within-population tilting.
- **Per-instance conditional**: impossible distribution-free in any nontrivial form (adaptation of the conditional-validity impossibility of Foygel Barber et al. 2021 [C]); stated as such.
- **Sequential/online under drift**: genuinely open; formulated with two proof routes and preregistered fallbacks, not claimed.

What the contract does and does not promise under the four stress conditions named in the mandate: under **stochastic continuations**, the guarantee is marginal over the frozen sampler's continuation randomness (via a replay-closure lemma), never per-realization; under **adaptive tree growth** (spawning new branches mid-run), nothing is guaranteed — the method is pruning-only by design, because spawning breaks replay closure; under **policy-induced distribution shift**, nothing shifts in single-turn tasks (the prompt stream is exogenous; pruning cannot alter it), and multi-turn agentic feedback is explicitly out of scope; under **censored outcomes**, calibration is uncensored by construction (banks materialize every branch), deployment is censored but the guarantee needs no deployment-time observation of the reference — monitoring uses randomized full-bank audits at rate $\rho$.

## I.3 Which proposal is stronger, and the verdict

**Report B is the stronger proposal**, on three grounds. First, its bank-and-replay experimental architecture *dissolves* the obstacle Report A names as its own hardest unresolved obligation: with every branch materialized to completion on calibration data, there are no censored counterfactuals at calibration time, and any deterministic multi-checkpoint schedule can be evaluated exactly by replay — so the "adaptive multiplicity / censored feedback" problem Report A could not solve simply does not arise for the offline contract. Second, its literature sweep is verifiably more reliable: today's checks confirmed Report B's key citations nearly verbatim, while several of Report A's closest-competitor claims (the July-7 cascade, REFRAIN/ReASC/SAT, "dynamic value-thresholded abstention at ICML 2026") could not be located. Third, it correctly identifies that the certified object must be attached to a consumer that pays in tokens and that the single-trajectory cell is occupied — a judgment now *stronger* than Report B knew, since Conformal Thinking has upgraded to ICML 2026.

**But Report B's chosen spine must be rejected.** Its central theorem is the online coupled-control result, which (i) rests on the non-monotonicity obligation it admits is unresolved, and whose proposed repair I showed above to be incorrect as stated; (ii) contains a regret clause in tension with a classical impossibility; (iii) uses the disagreement loss that fails the strictness audit; and (iv) targets the online cell, which verified 2026 work (ORCA; Anytime-Valid CRC) is crowding fastest. A paper whose load-bearing theorem is in that state is not the right commitment.

**Report A contributes** the correct decision problem (set-valued pruning of a branch bank — richer than stop/continue), the survival risk object (which becomes our diagnostic decomposition term), the elementary submodularity/top-$k$ structure (used as motivation for score-ranked nested families, never sold as theory), and the three-verification-regime task design. **Report A as a whole is rejected** because its central contract is vacuous/gameable (§I.2.1), its guarantee is confined to frozen checkpoints with no viable answer to deployment censoring except the machinery Report B already provides, and its remaining theorem content is inherited.

**The verdict** is therefore a synthesis that is still one method: Report A's decision problem and preservation object, with the contract repaired to success-conditional form; Report B's bank/replay/audit architecture, calibration machinery, and preregistration discipline; the certified offline population result as the spine; the online extension demoted to an honestly-flagged open program with a corrected surrogate and two proof routes. Token frequency appears nowhere in the headline: it is one preregistered rung of the signal-ablation ladder, kept only if it clears confound controls, dropped without ceremony otherwise.

## I.4 Closest existing work and the precise remaining novelty gap

As of today's sweep, the occupied cells are: **single-trajectory stopping with distribution-free risk control** (Conformal Thinking, ICML 2026 [V]); **prefix-level certification** (CROP [V]); **response-count/budget selection with LTT at whole-response level** (Conformal Language Modeling, ICLR 2024 [C]); **heuristic population-level filtering and scheduling** (DeepConf [V], Certaindex [V], ESC/adaptive-consistency [C]); **learned branch routing without guarantees** (Local Branch Routing [V]); **conformal machinery inside speculative test-time scaling** (ATTS, ICLR 2026 [V]); **online calibration under shift for reasoning** (ORCA [V]); **anytime-valid CRC machinery** (arXiv:2602.04364 [V]); **instance-level think/skip gating with risk control** (CWWI [R]).

The unoccupied composite — checked against everything located today — is:

> **Set-valued, mid-trajectory pruning of a parallel rollout population, calibrated end-to-end over multi-checkpoint kill-and-commit schedules on fully materialized banks, under a success-conditional (non-vacuous, non-gameable) preservation contract, with an explicit survival-vs-selection decomposition and a deployment-equivalence lemma.**

Each component exists somewhere; the composed object does not, to the best of a verification-heavy sweep. The gap is narrow: Conformal Thinking's group is one increment (parallel branches) away; the unverified recall-cascade [U] may already hold the contract class at episode level. The paper must therefore (a) be precise that its contract-class analysis (Proposition 1 and the taxonomy) is part of the contribution, not just the method; and (b) carry a mandatory submission-time re-sweep gate (§II.7.5).

## I.5 Inherited machinery versus potentially new claims

**Inherited, to be cited and never sold as new:** split conformal and Clopper–Pearson/binomial UCBs; Learn-then-Test and fixed-sequence testing; conformal risk control; RCPS; Mondrian/group conditioning; ACI and its telescoping argument; constrained online convex optimization; e-processes/betting confidence sequences; submodularity of coverage functions and top-$k$ optimality under conditional independence; coupling arguments.

**Potentially new (each honestly scoped):**
1. **The contract-class analysis for pruning** — the vacuity/dilution/adverse-selection propositions (elementary mathematics, but apparently unstated in this literature and decision-critical), together with an *empirical demonstration* that a marginally-calibrated cost-minimizer concentrates its failures on solvable-hard instances. New in context; trivial in technique; high decision value.
2. **The replay-closure lemma** — distributional equivalence of deployment-time pruning and bank replay under explicit conditions (conditional independence of branch continuations, prefix-measurable signals, deterministic aggregation, frozen sampler). Small, load-bearing, easy to get subtly wrong, and the thing that makes end-to-end schedule calibration legitimate. New in context.
3. **End-to-end conditional LTT over multi-checkpoint schedules on materialized banks** — inherited statistics on a new decision surface; the honest label is "new application surface, not new statistics."
4. **The monotone-envelope surrogate** for online tracking (running-max along the nested path), correcting Report B's repair — proposed with its conservatism price defined and measurable; its coupled online analysis is **open**.
5. **The online success-conditional tracking theorem** (ratio target with audited feedback under drift) — **genuinely open**; formulated at full strength with two proof routes and a preregistered fallback; explicitly not claimed.

## I.6 Why the rejected proposals should not be the final paper

**Report A as the final paper** would submit a theorem whose contract a competent reviewer can void with one sentence ("your guarantee is implied by task hardness"), whose optimizer provably drifts toward sacrificing the most valuable instances, whose sequential story is admitted-open with no mechanism, and whose remaining formal content is standard LTT plus elementary submodularity. Its own §9 self-assessment ("low to moderate" with only the easy theorem) is accurate.

**Report B as the final paper** would stake the spine on an online theorem that is currently unprovable as stated (one open lemma, one incorrect lemma, one impossibility-adjacent clause), measured in a loss that wastes budget on both-wrong noise and punishes improvement, in the sub-cell (online calibration) where verified 2026 activity is densest. Its architecture is the best thing in either report; its headline bet is the worst.

The synthesis keeps what each got right and puts the certificate where it can actually be proven this year.

## I.7 The binding decision

One paper. Method name **CASPR** — Calibrated Success-Preserving Rollout Pruning. One central risk object — the success-conditional preservation risk $R(\lambda)$. One algorithm — nested kill-and-commit schedules over a parallel rollout population, calibrated once on materialized banks by conditional Learn-then-Test with fixed-sequence testing, deployed frozen, monitored by randomized audits. One theorem program — the preservation-contract program: vacuity and adverse selection of marginal contracts (proved), finite-sample success-conditional validity end-to-end (proved modulo routine write-out), group-conditional validity and per-instance impossibility (adaptations), online maintenance under drift (open, formulated, two routes, preregistered fallback). One experimental plan — a ~15-GPU-hour decisive pilot with preregistered pass/kill gates, then a 150–300-GPU-hour full matrix. Token frequency: demoted to one ablation rung with a preregistered survival threshold. Everything else in both reports is dropped.

---

# Part II — Paper draft v0.1

## Title

**Don't Prune the Win: Success-Conditional Risk Control for Parallel Test-Time Reasoning**

**Method name (stable):** **CASPR** — **Ca**librated **S**uccess-**P**reserving **R**ollout pruning.
(Backup title, if the imperative register is judged too informal at submission: *"Success-Conditional Risk Control for Pruning Parallel Reasoning Rollouts."*)

## Abstract

Parallel test-time scaling — best-of-$N$, self-consistency, and their confidence-weighted descendants — spends most of its tokens completing rollouts that end up discarded. Heuristic filters such as windowed-confidence early killing recover much of this compute, but come with no contract: nothing bounds how often the filter destroys the very rollouts that would have produced the correct answer. We first show that the seemingly natural contract for such a bound — "with probability at least $1-\alpha$, it does not happen that the full rollout population succeeds while the pruned one fails" — is *vacuous on hard distributions* (it is implied by task hardness alone whenever full-budget success is rarer than $\alpha$) and *adversely selecting* under cost minimization (the cheapest compliant policy preferentially sacrifices solvable hard instances). We propose instead the **success-conditional preservation contract**: among instances that the full-budget procedure would have answered correctly, the pruned procedure answers correctly with probability at least $1-\alpha$. We give **CASPR**, a training-free controller that kills and early-commits members of a parallel rollout population at token-level checkpoints using prefix-measurable signals, and whose multi-checkpoint schedule is calibrated *end-to-end* — avoiding per-checkpoint error composition entirely — by conditional Learn-then-Test on fully materialized rollout banks, where every counterfactual outcome is observed by construction. A replay-closure lemma transfers the calibrated guarantee to deployment, where pruned branches are censored; a decomposition separates the two distinct failure modes (no correct rollout survives, versus a correct rollout survives and the aggregator discards it); Mondrian variants give group-conditional control, and we show nontrivial per-instance control is impossible distribution-free. We formulate — but do not claim — the online extension that would maintain the contract under distribution drift, identify the exact obstruction (non-monotonicity of vote-based losses in the pruning knob), and propose a monotone-envelope surrogate whose price is measurable on banks. The experimental program is bank-first: all calibration, baselines, and ablations are zero-GPU replays over released banks, with preregistered pass/kill gates, on two model families and three verification regimes. [Results: to be run; this draft contains the preregistered targets and gates, not outcomes.]

## 1. Introduction

### 1.1 The compute problem

Reasoning-tuned language models buy accuracy with tokens. The dominant test-time recipe is embarrassingly parallel: sample $N$ long chains of thought, then aggregate — majority vote, confidence-weighted vote, or verifier selection. The recipe is effective and simple, and it is extraordinarily wasteful: most sampled chains are redundant with the eventual consensus or doomed from early on, and the cost of the recipe is dominated by completing them anyway. A now-substantial line of work exploits this slack heuristically. Early-stopping self-consistency truncates the population when a vote stabilizes; DeepConf kills rollouts whose windowed token-confidence sags; serving-layer schedulers such as Certaindex allocate compute across requests by certainty proxies; routers learn which local branch to commit to. These methods report large token savings — often 40–80% — at small measured accuracy cost, on the benchmarks where they were tuned.

What none of them provides is a contract. The user of a pruning heuristic has no handle on the quantity that actually frightens them: *how often does pruning throw away the win?* — how often would the full-budget system have answered correctly, while the pruned system, on the same problem, answers incorrectly? Without a bound on that quantity, every reported savings number is an average over a silent redistribution of failures, and nothing prevents the redistribution from concentrating exactly where extra compute was most valuable. Single-trajectory risk control has recently arrived — Conformal Thinking (ICML 2026) calibrates stop-versus-continue thresholds for one chain with distribution-free guarantees — but the population-level action space, where the tokens actually are, remains contract-free.

### 1.2 Why the obvious contract fails

The obvious formalization — bound the probability that "the full population contains a success but the pruned subset does not" — is the one a first draft writes down, and it is broken in a way that matters. Its expectation is capped by the probability that the full population succeeds at all. On distributions where full-budget success is rare (frontier benchmarks, small models, low $N$), the bound is implied by hardness alone: *every* policy satisfies it, including the one that deletes everything. Between the extremes it silently rescales — a nominal $\alpha$ becomes a conditional tolerance of $\alpha/\Pr(\text{full-budget success})$ — so the same certificate means something four times weaker on a benchmark half as solvable. Worst, when a calibrated cost-minimizer optimizes against the marginal constraint over a mixed population, the slack is spent where per-instance savings are largest: on hard, long-trace, *solvable* instances. The contract does not merely under-protect them; it steers the optimizer toward sacrificing them. Section 3 states these observations as Proposition 1 and the experimental plan includes their empirical demonstration on real rollout banks.

The repair is conditioning. The **success-conditional preservation contract** bounds
$$\Pr\big(\text{pruned system wrong}\ \big|\ \text{full-budget system right}\big)\ \le\ \alpha,$$
a guarantee whose meaning is invariant to task hardness: it promises, uniformly over benchmark difficulty, that at most an $\alpha$-fraction *of the wins you would have had* are forfeited. It cannot be satisfied by hardness, cannot be gamed by netting improvements against losses, does not punish the pruned system for beating the reference, and — because "the full-budget system is right" is an event observable on calibration data where every rollout is materialized — it is calibratable with finite-sample tools.

### 1.3 Where this project comes from: the failure of the frequency-offset approach

This paper's protocol discipline descends from a project that failed, and the failure is instructive enough to recount. The predecessor method, ν-sampling, was a static truncation rule that *widened* the admissible logit margin for low-corpus-frequency tokens, on the hypothesis that rare tokens are systematically over-pruned. Its preregistered Phase-0 pilot (two Qwen2.5 scales × web/math, margin-conditional hit-rate surfaces with clustered inference) produced, in all four cells, a signed effect of the *opposite* direction — at fixed margin, rare-bucket tokens were *less* often correct, the desmoothing sign — with math-cell magnitudes censored at the boundary of the ±2-nat search grid, web-cell contrasts unstable across data halves due to a data-selected reference bucket, and a permutation control too coarse to attribute anything to external frequency rather than morphology, token identity, or the model's internal unigram prior. The project's own gates correctly returned INSUFFICIENT, and the method was retired.

Three durable lessons shape the present design. First, *sign before magnitude*: a preregistered directional read killed a method before a benchmark could flatter it. Second, *a signal is not a method*: even had the frequency effect been real, next-token coverage has no consumer that pays in compute; prediction-set size is not FLOPs. Third, *side information must survive its confounders and change a calibrated decision*: token frequency accordingly appears in this paper only as one preregistered rung of a signal-ablation ladder — behind windowed confidence, entropy, length, cross-branch agreement, and the model's own internal unigram prior — retained only if it improves the certified cost frontier by a preregistered margin, and dropped otherwise. The headline object of this paper does not mention frequency, and nothing below depends on it.

### 1.4 Contributions

1. **A contract-class analysis for rollout pruning** (§3, §5.1): marginal preservation contracts are vacuous under rarity and adversely selecting under cost minimization (Proposition 1); success-conditional contracts repair both defects; group-conditional versions are achievable, and nontrivial per-instance versions are impossible distribution-free (Theorem 4 and Remark 2). We demonstrate the adverse-selection mechanism empirically on real banks.
2. **CASPR** (§4): a training-free kill-and-commit controller over parallel rollout populations, whose *entire multi-checkpoint schedule* is calibrated end-to-end on materialized rollout banks by conditional Learn-then-Test with fixed-sequence testing — no per-checkpoint composition, no counterfactual censoring at calibration — with a replay-closure lemma (Lemma 1) transferring the guarantee to censored deployment, and a decomposition (Proposition 3) separating survival failure from selection failure.
3. **A preregistered, bank-first empirical program** (§6–§7): released rollout banks over two model families and three verification regimes on which every calibration, baseline, and ablation is a zero-GPU deterministic replay; validity ("money") plots of realized conditional risk against nominal $\alpha$; the measured *price of the certificate* against oracle-tuned heuristics; and a signal audit with preregistered demotion rules. [All results pending; targets and gates in §7.]
4. **The online frontier, honestly** (§5.5): the drift-maintenance extension formulated at full strength; the exact obstruction (non-monotone vote losses) identified; a corrected monotone-envelope surrogate proposed with a measurable price; two proof routes sketched; nothing claimed.

## 2. Related work and novelty statement

*(Citation hygiene: tags [V]/[C]/[R]/[U] are this draft's verification labels per Part 0; they will be replaced by ordinary citations at submission after the mandatory re-sweep of §7.5.)*

**Heuristic population control.** Early-stopping and adaptive self-consistency [C] stop sampling when votes stabilize; DeepConf [V] kills low-confidence rollouts online with large token savings; Certaindex [V] schedules serving compute by certainty proxies; CarBoN [R] calibrates logits for best-of-$N$; Local Branch Routing [V] trains a router over lookahead branches. These define the action space and the savings headroom, and provide no distribution-free guarantee. CASPR's relationship to DeepConf is deliberate and precise: we adopt its windowed-confidence signal as the default kill score, then wrap the *schedule* in a certificate; DeepConf (offline- and oracle-tuned) is the primary heuristic frontier against which the price of that certificate is measured.

**Risk-controlled test-time compute.** Conformal Thinking (ICML 2026) [V] calibrates dual stop/futility thresholds for a *single* trajectory; CROP [V] certifies reasoning-trace prefixes against step labels; CWWI [R] gates think-versus-skip per instance; Conformal Language Modeling [C] calibrates how many *complete* responses to sample and filter; ATTS [V] applies conformal machinery inside speculative test-time scaling; ORCA [V] adapts reasoning calibration online via test-time training; anytime-valid conformal risk control [V] supplies sequential machinery. None of these takes the set-valued, mid-trajectory action — which rollouts of a running population to kill, when to commit — and none states, let alone controls, a success-conditional preservation risk. The two nearest misses are Conformal Thinking (right guarantee style, single-trajectory action space, marginal-style contract) and CLM (population action, whole-response granularity, pre-long-CoT setting, marginal contract).

**Novelty statement.** To the best of a verification-heavy sweep dated 2026-07-10 (Part 0), no prior or concurrent work provides: (i) finite-sample, distribution-free control of a **success-conditional** preservation risk (ii) for **set-valued kill-and-commit actions over parallel rollout populations** (iii) calibrated **end-to-end over multi-checkpoint schedules** on **fully materialized banks** with a **deployment-equivalence lemma** replacing per-checkpoint composition, together with (iv) a **contract-class analysis** showing the marginal alternative is vacuous and adversely selecting. Components (statistical machinery, signals, the pruning action itself) are individually known; the composed object and the contract analysis are, to our knowledge, new. Known risks to this statement: an unlocated report of a "recall-controlled cascade" for agent episodes [U] would, if real and population-level, force a sharper delta (our set-valued action, end-to-end schedule calibration, decomposition, and vacuity analysis would remain); the Conformal Thinking group is one increment away. §7.5 preregisters the re-sweep.

**Everything inherited.** Split conformal calibration and exact binomial upper confidence bounds [C]; Learn-then-Test and fixed-sequence testing [C]; conformal risk control [C]/[V]; RCPS [C]; Mondrian conditioning [C]; ACI [C]; e-processes and betting confidence sequences [C]/[V]; the impossibility of nontrivial distribution-free instance-conditional validity [C]; submodularity of coverage functions and top-$k$ optimality under conditional independence (used only as motivation for score-ranked nested families) [C]; constrained-online-learning impossibility [C]. We claim none of this.

## 3. Problem formulation

### 3.1 Instances, banks, and the reference protocol

Fix a task distribution $\mathcal P$ over problems $X$ with verifiable unique answers $y(X)$ (exact-match math, hidden unit tests, or single-key multiple choice; the imperfect-judge regime is treated as a limitation in §8). Fix a **generation protocol** $\mathcal G=(\text{model } M,\ \text{sampler parameters }\sigma,\ N,\ T_{\max})$, frozen before calibration: given $X$, protocol $\mathcal G$ launches $N$ independent rollouts $b\in[N]$ from $M(\cdot\mid X;\sigma)$, each run to natural completion or $T_{\max}$ tokens.

An **instance** is the pair
$$W=(X,\ \mathcal T),\qquad \mathcal T=\{(\tau_b, a_b, z_b, \mathrm{len}_b)\}_{b=1}^{N},$$
where $\tau_b$ is the full token trace of rollout $b$ with its per-token log-probabilities, $a_b$ its extracted final answer, $z_b=\mathbf 1\{a_b=y(X)\}\in\{0,1\}$ its graded outcome, and $\mathrm{len}_b$ its length. We call a materialized $W$ a **bank instance**: on calibration data every rollout is completed and graded, so *nothing is censored at calibration time*. Instances $W_1,\dots,W_n,W_{n+1}$ are assumed i.i.d. (problem draw plus generation randomness); this is Assumption **(A1)**.

The **reference policy** $\pi_{\mathrm{full}}$ completes all $N$ rollouts and aggregates with a fixed, deterministic rule $\mathrm{AGG}$ (default: confidence-weighted majority over extracted answers, ties broken by mean trace confidence then by branch index). Write $A_{\mathrm{full}}(W)=\mathrm{AGG}(\{(a_b,\cdot)\}_{b\le N})$ and define the **reference-success event**
$$E(W)=\mathbf 1\{A_{\mathrm{full}}(W)=y(X)\},\qquad p_E=\Pr(E=1).$$
$E$ is a deterministic function of the instance; it is observed on banks and censored at deployment.

### 3.2 Budgeted policies

A **budgeted policy** $\pi_\lambda$ observes the population as it generates, at token-level checkpoints $t\in\{\Delta,2\Delta,\dots\}$, and takes two kinds of set-valued action:

- **Kill**: permanently stop a running rollout (its continuation is never generated);
- **Commit**: stop the entire instance and output an answer aggregated from the rollouts that have *finished* (produced an answer) so far.

Decisions may depend only on **prefix-measurable signals**: functions of the tokens generated so far by currently-alive rollouts, the answers and statistics of already-finished rollouts, and the prompt (Assumption **(A2)**). The default kill score is DeepConf-style windowed confidence (mean token log-probability over the trailing $w$ tokens of the rollout's own prefix); the ablation ladder (§6.6) adds entropy, length, cross-branch agreement, an optional process reward model, and the audited frequency rung. The policy family $\Lambda$ is a finite, preregistered grid of **nested kill-and-commit schedules** (§4.2). Each $\pi_\lambda$ outputs an answer $A_\lambda(W)$ and spends a token budget $\mathrm{tok}_\lambda(W)$; the reference spends $\mathrm{tok}_{\mathrm{full}}(W)=\sum_b \mathrm{len}_b$.

Because signals are prefix-measurable and the aggregator is deterministic, $A_\lambda(W)$ and $\mathrm{tok}_\lambda(W)$ are *deterministic functions of the bank instance*: any schedule can be evaluated exactly, after the fact, by replaying the bank (§4.3). This single design choice is what removes calibration-time censoring.

### 3.3 The risk and compute objectives

Define the per-instance loss and the two risks
$$\ell_\lambda(W)=\mathbf 1\{A_\lambda(W)\neq y(X)\},\qquad
R_{\mathrm{marg}}(\lambda)=\Pr\big(E=1 \wedge \ell_\lambda=1\big),\qquad
R(\lambda)=\Pr\big(\ell_\lambda=1 \,\big|\, E=1\big),$$
and the compute functional
$$C(\lambda)=\mathbb E\!\left[\frac{\mathrm{tok}_\lambda(W)}{\mathrm{tok}_{\mathrm{full}}(W)}\right]\in(0,1].$$
The **CASPR program** is
$$\min_{\lambda\in\Lambda}\ \widehat C(\lambda)
\quad\text{subject to}\quad
\Pr\big(R(\hat\lambda)\le\alpha\big)\ge 1-\delta\ \text{over the calibration draw},$$
i.e., cost minimization among policies *certified* at success-conditional level $\alpha$ with calibration confidence $1-\delta$. The central risk object of the paper is $R(\lambda)$, the **success-conditional preservation risk**: the fraction of reference wins the budgeted policy forfeits.

**Why not the marginal risk.** Proposition 1 (§5.1) shows $R_{\mathrm{marg}}(\lambda)\le p_E$ identically, so a marginal contract at level $\alpha$ is (i) vacuous whenever $p_E\le\alpha$, (ii) equivalent to a conditional contract at the hardness-dependent level $\alpha/p_E$, and (iii) adversely selecting: over mixtures, the cost-minimizing marginally-feasible policy may fail *all* solvable instances of any subgroup $G$ with $w_G\,p_{E|G}\le\alpha$, and the cost gradient points at exactly those subgroups. The marginal risk is reported in experiments only as a negative exhibit.

**Why not the net accuracy drop.** The alternative contract $\mathbb E[\mathbf 1\{A_{\mathrm{full}}\text{ right}\}-\mathbf 1\{A_\lambda \text{ right}\}]\le\alpha$ permits *churn*: large conditional losses offset by wins on other instances, hiding a redistribution of failures. Our contract implies a net-drop bound (the drop is at most $p_E\,R(\lambda)\le\alpha$, and is smaller whenever the pruned policy wins instances the reference loses) but not conversely; we therefore control $R$ and *report* the net drop.

**Why condition on $E$ rather than on bank solvability.** The event $E^{\mathrm{sol}}=\{\exists b: z_b=1\}$ ("some rollout is correct") satisfies $E\subseteq E^{\mathrm{sol}}$ (with a unique correct answer, an aggregator over rollout answers can only be right if some rollout is). Conditioning on $E$ makes the contract *end-to-end* — it protects realized wins, which is what a user experiences — while $E^{\mathrm{sol}}$-conditioning protects only the existence of a winning branch that the aggregator might still discard. We adopt $E$ for the contract and use $E^{\mathrm{sol}}$ inside the diagnostic decomposition (Proposition 3), which is precisely the separation of *preservation of a viable branch* from *final answer selection*.

### 3.4 The guarantee taxonomy and the scope ledger

The contract classes, ordered by strength, with their status in this paper:

| Class | Definition | Status |
|---|---|---|
| Marginal | $R_{\mathrm{marg}}(\lambda)\le\alpha$ | Rejected: vacuous under rarity, adversely selecting (Prop. 1); shown empirically. |
| Success-conditional (population) | $R(\lambda)\le\alpha$ | **The contract.** Achieved with finite-sample certificates (Thm. 2). |
| Group-conditional (Mondrian) | $R_g(\lambda)\le\alpha$ for each pre-declared, deployment-measurable group $g$ | Achieved per group at the price of per-group sample size (Thm. 4). |
| Instance-conditional | $\Pr(\ell_\lambda=1\mid E=1, X=x)\le\alpha$ for a.e. $x$ | Impossible distribution-free in nontrivial form (Remark 2). |
| Sequential / drift | time-average conditional risk $\le\alpha+o(1)$ on non-exchangeable streams | **Open** (§5.5); formulated, not claimed. |

**Scope ledger — what is and is not guaranteed.**

- *Stochastic continuations.* Rollout outcomes are random under $\mathcal G$; the guarantee is **marginal over generation randomness** (the instance $W$ includes it). No per-realization statement — "the same winning trace survives" — is made or true.
- *Adaptive tree growth.* **Not covered.** CASPR is pruning-and-commit only. Spawning new branches conditioned on survivors changes the answer distribution in ways not materialized in any bank, breaking Lemma 1. (Spawning also cannot be smuggled in as "harmless": it alters cost and the aggregate answer.) Extension requires tree-structured banks and is future work.
- *Policy-induced distribution shift.* Two levels, resolved differently. *Within an instance*, later checkpoint states do depend on earlier kills — but the loss is defined for the entire composed schedule and evaluated by exact replay, so no per-checkpoint calibration (and hence no within-instance shift correction) is ever needed; this is what dissolves Report A's "policy-induced covariate shift" obstacle for the offline contract. *Across instances*, single-turn prompt streams are exogenous: pruning cannot change which problems arrive. Multi-turn agentic settings, where today's output is tomorrow's prompt, are **not covered**.
- *Censored outcomes.* Calibration is censoring-free by construction (banks). At deployment, killed rollouts and the reference outcome are censored; the guarantee needs no deployment-time observation (it was purchased at calibration and transfers via Lemma 1 under (A1)–(A5)). *Monitoring* the realized risk requires randomized full-budget audits at rate $\rho$ (§4.4).
- *Distribution shift between calibration and deployment.* **Not covered** by Theorem 2. Partially addressed by Mondrian groups (Thm. 4), measured by transfer cells (§6.5), and targeted by the open online extension (§5.5).
- *Verifier noise.* All guarantees are relative to the grader's labels; a weak judge yields a contract about judge-approval, not truth.
- *Reference definition.* The contract is relative to $\pi_{\mathrm{full}}$ at the same $(N,\sigma,T_{\max})$; choosing $N$ is upstream (complementary to CLM-style budget selection [C]).

## 4. Method: CASPR

### 4.1 Signals

For rollout $b$ alive at checkpoint $t$: windowed confidence $s^{\mathrm{conf}}_b(t)$ = mean token log-probability over the trailing $w$ tokens (default $w=512$; tune-fixed); optional rungs: trailing-window entropy, length-so-far, and (for the audit ladder only) internal-unigram-residualized and corpus-frequency features of the prefix. For the population: the multiset $D(t)$ of answers already produced by finished rollouts, with weights $u_b$ = mean trace confidence; the **commit statistic** $V(t)$ = weighted vote share of the leading answer in $D(t)$. All signals are prefix-measurable (A2).

### 4.2 The policy family: nested kill-and-commit schedules

A policy $\lambda=(\kappa, v)\in\Lambda$ acts at each checkpoint $t=\Delta,2\Delta,\dots$ (default $\Delta=512$):

1. **Commit test.** If $|D(t)|\ge 2$ and $V(t)\ge v$: abort all alive rollouts and output $\mathrm{AGG}(D(t))$.
2. **Kill rule.** Among alive rollouts, kill those with $s^{\mathrm{conf}}_b(t)$ below the $\kappa$-quantile of the currently-alive scores — **except** the current argmax, which is never killed (the *floor*: at least one rollout always survives to completion, so the policy always answers).

If no commit fires, the policy ends when all surviving rollouts finish; output $\mathrm{AGG}(D(\infty))$. The grid is $\kappa\in\{0,0.1,\dots,0.5\}\times v\in\{0.55,0.65,\dots,0.95,\text{off}\}$ plus optional per-checkpoint ramps $\kappa(t)$, $|\Lambda|\approx 40$–$200$, preregistered. The family is **nested in aggressiveness** along either coordinate (larger $\kappa$ kills supersets given identical score streams; larger $1/v$ commits earlier), which fixed-sequence testing exploits for power; *no result below requires monotonicity of the risk in $\lambda$* (vote flips make it non-monotone in general — §5.5). The reference $\lambda_{\mathrm{full}}=(\kappa{=}0, v{=}\text{off})$ is always in $\Lambda$ and is feasible by construction ($R(\lambda_{\mathrm{full}})=0$), so calibration never returns an empty certified set. Motivation for score-ranked nested kills (not a theorem we sell): under conditional independence of rollout outcomes given the prompt and equal costs, retaining the top-$k$ by success probability is optimal among size-$k$ retentions [C]; ranked-by-score nested families are the policy-family shape this fact recommends.

### 4.3 Exact replay on banks

Given a bank instance $W$ and any $\lambda$, `REPLAY` simulates the deployment timeline deterministically by token index: rollouts finish when $t\ge\mathrm{len}_b$; signals are computed on stored prefixes; kills mask stored suffixes so they are never read (A2 enforced mechanically); commits truncate cost. Outputs: $A_\lambda(W)$, $\ell_\lambda(W)$, $\mathrm{tok}_\lambda(W)$, the finisher set $F_\lambda(W)$ (rollouts that produced an answer available to the aggregator), and the survival indicator $S_\lambda(W)=\mathbf 1\{\exists b\in F_\lambda(W): z_b=1\}$. Replay is exact, so **calibration involves no counterfactual estimation whatsoever**: the "censored feedback" obstruction dissolves at calibration time by design.

```
Algorithm 1 — BANKGEN (once per model×task cell; the only GPU stage)
Input: problems D, protocol G = (M, σ, N, T_max), stride Δ, window w
for x in D:
    launch N i.i.d. rollouts from M(·|x; σ); stream to completion or T_max
    store per-token logprobs (top-20) and entropies; extract answer a_b; grade z_b
    emit bank instance W = (x, {(τ_b, a_b, z_b, len_b)}_{b=1..N})
Output: bank corpus; content-addressed, revision-pinned, split-manifested
```

```
Algorithm 2 — REPLAY(λ = (κ, v), W)   # deterministic, CPU-only
alive ← [N]; finished ← ∅; spent_b ← 0 ∀b
for t = Δ, 2Δ, ...:
    move {b ∈ alive : len_b ≤ t} to finished            # natural completions
    if |finished| ≥ 2 and voteshare(finished) ≥ v:      # COMMIT
        abort alive at t; break
    for b in alive: s_b ← window_conf(τ_b[1..t], w)     # own-prefix signal
    kill ← {b ∈ alive : s_b < quantile_κ({s_b})} \ {argmax_b s_b}   # FLOOR
    alive ← alive \ kill                                 # suffixes of killed b are masked
    if alive = ∅: break
â ← AGG(finished);  ℓ ← 1{â ≠ y(x)}
tok_λ ← Σ_b (tokens generated for b under this schedule)
S ← 1{∃ b ∈ finished : z_b = 1}
return (â, ℓ, tok_λ, S)
```

### 4.4 Calibration and deployment

```
Algorithm 3 — CALIBRATE (CPU-only)
Input: calibration banks {W_i}_{i≤n}, grid Λ, target (α, δ),
       FST order λ(1), λ(2), ... fixed on the TUNE split
       (safest-first: ascending tune-estimated savings)
E_i ← 1{REPLAY(λ_full, W_i).ℓ = 0}                       # reference success
n₁ ← #{i : E_i = 1}
for j = 1, 2, ... :                                       # fixed-sequence testing
    F_j ← Σ_{i: E_i=1} REPLAY(λ(j), W_i).ℓ                # conditional failures
    U_j ← ClopperPearsonUCB(F_j, n₁; level 1−δ)
    if U_j > α: break                                     # stop at first failure
Λ_cert ← {λ(1..j−1)} ∪ {λ_full}
λ̂ ← argmin_{λ ∈ Λ_cert}  mean_i tok_λ(W_i)/tok_full(W_i)
return λ̂  (report Û_λ̂, ĉ(λ̂), and the full certified frontier)
```

```
Algorithm 4 — DEPLOY(λ̂, x)
run Algorithm 2's control flow generatively (tokens produced on demand);
with probability ρ (audit): additionally complete ALL rollouts to T_max,
    grade, and log the full loss surface {ℓ_λ(W)}_{λ∈Λ} for monitoring
    and any scheduled recalibration (§5.5 fallback)
```

Variant certifications, same machinery: **Bonferroni-LTT** (test all $\lambda$ at level $\delta/|\Lambda|$; no ordering needed; less powerful); **Mondrian** (run Algorithm 3 within each pre-declared group; Thm. 4). Deployment never observes $E$; the certificate does not require it to (Lemma 1). Audits exist for *monitoring and maintenance*, not for the validity of Theorem 2.

## 5. Theory: the preservation-contract program

Assumptions, collected. **(A1)** Instances $W_1,\dots,W_n,W_{n+1}$ i.i.d. (problem draw × generation randomness under frozen $\mathcal G$). **(A2)** Policy signals are prefix-measurable: at any checkpoint they are functions only of tokens generated so far by alive rollouts, statistics of finished rollouts, and the prompt. **(A3)** Rollout continuations are conditionally independent across rollouts given the prompt and each rollout's own prefix (i.i.d. parallel sampling; no cross-rollout attention or shared state). **(A4)** Aggregation and tie-breaking are deterministic; the answer-extraction function is fixed. **(A5)** $\Lambda$ is finite and, together with the FST order, fixed before touching calibration data (tune split only).

### 5.1 Proposition 1 (Vacuity and adverse selection of the marginal contract)

*Let $E$ be the reference-success event, $p_E=\Pr(E)$, and $R_{\mathrm{marg}}(\lambda)=\Pr(E\wedge \ell_\lambda)$. Then:*

*(i) For every policy $\lambda$ (including degenerate ones), $R_{\mathrm{marg}}(\lambda)\le p_E$. Hence if $p_E\le\alpha$, every policy satisfies the marginal contract.*

*(ii) $R_{\mathrm{marg}}(\lambda)=p_E\,R(\lambda)$: the marginal contract at level $\alpha$ is the success-conditional contract at level $\alpha/p_E$.*

*(iii) Let $\mathcal P$ be a mixture of groups $g$ with weights $w_g$ and conditional reference-success rates $p_{E|g}$, and let $r_g(\lambda)=\Pr(\ell_\lambda\mid E,g)$. The marginal contract constrains only $\sum_g w_g\,p_{E|g}\,r_g(\lambda)\le\alpha$; in particular $r_G(\lambda)=1$ is feasible for any group $G$ with $w_G\,p_{E|G}\le\alpha$. Moreover, if the achievable expected savings per unit of constrained risk mass $w_g\,p_{E|g}\,r_g$ is (weakly) maximal on $G$ — the typical situation when $G$ collects the longest-trace problems, whose continuations are the most expensive to preserve — then among marginally-feasible policies the cost minimizer weakly prefers saturating $r_G$ before spending risk budget anywhere else.*

**Proof.** (i)–(ii): $\{E\wedge\ell_\lambda\}\subseteq E$ and the definition of conditional probability. (iii): the constraint is linear in $(r_g)_g$ with coefficients $w_g p_{E|g}$; feasibility of the vertex with $r_G=1$, $r_{g\neq G}=0$ is the stated inequality. For the last clause: reallocating constraint mass $w_g p_{E|g}\,\mathrm dr_g \to w_G p_{E|G}\,\mathrm dr_G$ preserves feasibility by construction and weakly increases expected savings whenever savings-per-unit-mass is maximal on $G$; iterate to the saturated vertex. $\square$

*Status: elementary; new in context; also demonstrated empirically (§6.7, exhibit X1) by running Bonferroni-LTT against $R_{\mathrm{marg}}$ on real banks and showing the selected policy concentrates conditional failures on the hardest solvable stratum.*

**Remark 1 (relation to net accuracy drop).** $\mathrm{Acc}(\pi_{\mathrm{full}})-\mathrm{Acc}(\pi_\lambda)=R_{\mathrm{marg}}(\lambda)-\Pr(\neg E\wedge \neg\ell_\lambda)\le p_E\,R(\lambda)$. So the success-conditional contract implies a net-drop bound of $\alpha\,p_E$; the converse fails (churn).

### 5.2 Lemma 1 (Replay closure: deployment equals bank replay)

*Under (A2)–(A4) and deployment generation from the same frozen $\mathcal G$ (same model revision, sampler parameters, and stopping rules as BANKGEN), for every $\lambda\in\Lambda$ the joint law of $(A_\lambda,\ \mathrm{tok}_\lambda,\ S_\lambda)$ produced by Algorithm 4 on a fresh problem $X\sim\mathcal P$ equals the law of $\mathrm{REPLAY}(\lambda, W)$ with $W\sim\mathcal P\times\mathcal G$.*

**Proof sketch.** Couple the two processes on a common probability space by generating, for each rollout, the full continuation stream that BANKGEN would produce; (A3) makes these $N$ streams mutually independent given the prompt and unaffected by other rollouts' kill status. Deployment under $\lambda$ is then a *measurable revelation schedule* over this coupled object: at each checkpoint it reads exactly the prefixes REPLAY reads (A2), takes the same actions (same $\lambda$, deterministic rules, (A4)), and never reads a killed rollout's suffix — whose distribution is therefore irrelevant, not merely unobserved. By induction over checkpoints the two processes traverse identical action sequences on the coupled realization, hence produce identical outputs; marginalizing the coupling gives equality in law. $\square$

*Status: short, load-bearing, new in context. It is the license for calibrating a **schedule** (a multi-checkpoint object) as if it were a one-shot classifier, and the reason no per-checkpoint error composition, union bound over checkpoints, or off-policy correction appears anywhere in the offline theory. Failure modes of its assumptions are real and listed: batch-level interactions that couple rollouts (violates A3), signals peeking at killed suffixes or at wall-clock (violates A2), sampler or revision drift between bank generation and deployment (violates the frozen-$\mathcal G$ premise). The experiment plan verifies the lemma end-to-end by comparing deployed and replayed answer distributions on a held-out slice (§6.7, exhibit X2).*

### 5.3 Theorem 2 (Finite-sample success-conditional validity of CASPR)

*Under (A1)–(A5), let $n_1=\#\{i\le n: E(W_i)=1\}$, and for each $\lambda$ let $F_\lambda=\sum_{i:E_i=1}\ell_\lambda(W_i)$. Let $U_\lambda$ be the exact Clopper–Pearson upper $(1-\delta)$ confidence bound for a binomial proportion given $(F_\lambda, n_1)$. Let $\Lambda_{\mathrm{cert}}$ be the certified set returned by fixed-sequence testing in the preregistered order (stop at the first $\lambda$ with $U_\lambda>\alpha$), always augmented with $\lambda_{\mathrm{full}}$, and let $\hat\lambda=\arg\min_{\lambda\in\Lambda_{\mathrm{cert}}}\widehat C(\lambda)$. Then*
$$\Pr\big(R(\hat\lambda)\le\alpha\big)\ \ge\ 1-\delta,$$
*where the probability is over the calibration draw (and holds conditionally on $n_1\ge1$). The same statement holds for the Bonferroni variant with $U_\lambda$ at level $1-\delta/|\Lambda|$ and $\Lambda_{\mathrm{cert}}=\{\lambda: U_\lambda\le\alpha\}\cup\{\lambda_{\mathrm{full}}\}$. By Lemma 1, $R(\hat\lambda)$ is simultaneously the deployment-time conditional risk.*

**Proof sketch.** Conditional on $n_1$, the reference-success calibration instances are i.i.d. draws from $\mathcal P(\cdot\mid E=1)$ (E is a deterministic instance functional, so conditioning selects an i.i.d. subpopulation), and for fixed $\lambda$, $F_\lambda\mid n_1\sim\mathrm{Binomial}(n_1, R(\lambda))$; Clopper–Pearson gives $\Pr(R(\lambda)>U_\lambda\mid n_1)\le\delta$ pointwise. Fixed-sequence control: let $\lambda(j^\ast)$ be the *first* policy in the preregistered order with $R>\alpha$ (if none, nothing to prove; note $R(\lambda_{\mathrm{full}})=0$ so the augmentation is harmless). Any erroneous certification requires certifying $\lambda(j^\ast)$, since testing stops at the first failure and $\lambda(j^\ast)$ precedes every later true violator; and $\Pr(U_{\lambda(j^\ast)}\le\alpha)\le\Pr(U_{\lambda(j^\ast)}<R(\lambda(j^\ast)))\le\delta$. Hence with probability $\ge1-\delta$ every certified policy has $R\le\alpha$; $\hat\lambda$ is certified; done. Selection by estimated cost among certified policies is unrestricted — this is the Learn-then-Test logic [C], applied to a conditional subpopulation and a schedule-valued family. $\square$

*Status: inherited statistics (split binomial calibration + LTT/FST), new decision surface (multi-checkpoint schedules; success-conditional subpopulation; replay-exact losses). The one nonstandard ingredient is that the conditioning event is a functional of the reference policy inside the same family — handled by observing $E$ on banks and never at deployment. What remains to write out for the paper: the measure-theoretic footnote for conditioning on $\{n_1=m\}$ jointly with FST (routine), and the degenerate cases $n_1\in\{0\}$ (return $\lambda_{\mathrm{full}}$; contract vacuous but explicit).*

**Power accounting (design, not theorem).** With zero conditional failures, the CP bound certifies $\alpha\approx \ln(1/\delta)/n_1$ ($\approx 2.3/n_1$ at $\delta=0.1$): $n_1=237$ certifies $\alpha=0.01$; $\alpha=0.05$ at $n_1\approx230$ tolerates roughly 6–7 conditional failures. Hard benchmarks with small $n_1$ are *test-only* cells by design (§6.5). This arithmetic is what makes MATH-500-scale calibration realistic and AIME-scale calibration impossible — stated openly rather than hidden.

### 5.4 Proposition 3 (Survival–selection decomposition) and Theorem 4 (Mondrian)

**Proposition 3.** With $S_\lambda$ the survival indicator (a finisher with $z_b=1$ exists), the conditional risk splits as
$$R(\lambda)=\underbrace{\Pr\big(S_\lambda=0\mid E=1\big)}_{\text{preservation failure: no viable finisher}}\;+\;\underbrace{\Pr\big(S_\lambda=1,\ \ell_\lambda=1\mid E=1\big)}_{\text{selection failure: viable finisher discarded}},$$
both terms exactly computable by replay on banks. *Proof: the two events partition $\{\ell_\lambda=1\}$, since $S_\lambda=0\Rightarrow\ell_\lambda=1$ under unique answers.* $\square$ — This is the formal separation of *branch preservation* from *answer selection* demanded of the method: CASPR's certificate constrains the sum; the decomposition attributes it, tells the practitioner whether to spend on gentler kill schedules (first term) or a better aggregator (second), and is the honest answer to "you preserved a winner and then voted it away."

**Theorem 4 (group-conditional validity).** Let $G(W)\in\{1,\dots,K\}$ be a pre-declared grouping measurable at deployment decision time (prompt features and/or first-checkpoint signals; e.g., benchmark ID, length bin, first-checkpoint vote-entropy bin). Running Algorithm 3 within each group $g$ (its own $n_{1,g}$, FST, and $\hat\lambda_g$) yields $\Pr(R_g(\hat\lambda_g)\le\alpha)\ge1-\delta$ for each $g$, where $R_g(\lambda)=\Pr(\ell_\lambda=1\mid E=1, G=g)$; simultaneous validity over all $K$ groups at $1-\delta$ by allocating $\delta/K$, or reported per-group. *Proof: Theorem 2 within each i.i.d. subpopulation.* $\square$ — Mondrian control is the practical answer to residual within-population tilting left open by Proposition 1(iii): groups aligned with cost heterogeneity (length/difficulty bins) remove the optimizer's incentive to concentrate failures where savings are largest.

**Remark 2 (instance-conditional impossibility).** No nontrivial distribution-free guarantee of the form $\Pr(\ell_\lambda=1\mid E=1,X=x)\le\alpha$ a.e. is achievable: conditioning on a continuous $X$, any procedure achieving it for all distributions must be trivial (refuse to prune) on atomless regions. This adapts the conditional-validity impossibility of Foygel Barber, Candès, Ramdas & Tibshirani (2021) [C]; the adaptation (from conditional coverage of prediction sets to conditional control of a binary policy loss) is routine but must be written out — flagged as an owed appendix, not assumed.

### 5.5 The hardest unresolved obligation: maintaining the contract online under drift

Everything above is a *static-distribution* promise. The deployment reality both source proposals worried about — benchmark mix drifting, difficulty ramps, model-usage shifts — needs a sequential statement. We formulate the strongest version we believe is provable, identify exactly what is missing, and preregister the fallback. **Nothing in this subsection is claimed as a result.**

**Setting.** Instances arrive as a stream $t=1,2,\dots$, not exchangeable. The deployed knob $\lambda_t\in\Lambda$ (totally ordered by aggressiveness along the preregistered nested path) may be updated online. With probability $\rho$, round $t$ is an **audit**: the full bank is materialized (all $N$ rollouts completed), which — by replay — reveals the *entire* loss surface $\{\ell_\lambda(W_t)\}_{\lambda\in\Lambda}$ and $E_t$; non-audit rounds reveal only realized cost. Importance-weighted estimates: $\hat\ell_t(\lambda)=\frac{\mathrm{audit}_t}{\rho}\,\ell_\lambda(W_t)\,E_t$, $\hat e_t=\frac{\mathrm{audit}_t}{\rho}E_t$.

**Target.** With probability $\ge1-\delta$ over audit randomness, for any stream,
$$\frac{\sum_{t\le T} E_t\,\ell_{\lambda_t}(W_t)}{\max\!\big(1,\sum_{t\le T}E_t\big)}\ \le\ \alpha+O\!\Big(\tfrac{1}{\sqrt{\rho T}}\Big),$$
i.e., time-average *success-conditional* risk tracking — the ratio form, not the joint form, so that the contract's meaning does not drift with the stream's hardness (the lesson of Proposition 1, carried into the online setting).

**The obstruction, stated exactly.** ACI-style tracking [C] updates a scalar knob against the realized loss and controls the time-average by a bounded-potential telescoping argument. That argument needs the realized loss to respond monotonically to the knob. Here $\ell_\lambda(W)$ is **not monotone in $\lambda$**: killing one more rollout can flip a majority vote *back to* the reference answer, so aggressiveness can lower the realized loss non-monotonically along the nested path. This is the load-bearing open problem. It was correctly flagged in one source proposal, but the repair proposed there ("any vote-composition change counts as a loss, mechanically monotone under nested kills") fails under either reading: as vote-*outcome* change it is still non-monotone (the indicator toggles off as more rollouts die); as vote-relevant *composition* change it is monotone but near-vacuous, charging a loss for almost any kill of a reference-agreeing rollout.

**Route A (monotone envelope; our proposed repair).** Define the running-max envelope along the nested path,
$$\tilde\ell_\lambda(W)=\max_{\lambda'\preceq\lambda}\ell_{\lambda'}(W),$$
which is monotone by construction, upper-bounds the realized loss ($\lambda\preceq\lambda$), is the **least** monotone majorant of $\ell$ on the path (any monotone $g\ge\ell$ satisfies $g(\lambda)\ge\max_{\lambda'\preceq\lambda}\ell_{\lambda'}(W)$), and is *computable on every audited round* (audits materialize the full surface). Track $\tilde\ell$ with the ACI-style update $\lambda_{t+1}=\Pi_\Lambda[\lambda_t+\eta(\alpha\,\hat e_t-\hat{\tilde\ell}_t(\lambda_t))]$. Conjectured statement: for any stream, with probability $\ge1-\delta$, the time-average of $E_t\tilde\ell_{\lambda_t}$ is at most $\alpha\,\overline{E}_T+O(1/(\eta T))+O(\eta)+O(\sqrt{\log(1/\delta)/(\rho T)})$, hence the realized conditional risk (dominated by the envelope) is likewise controlled. Proof plan: bounded-iterate telescoping for the deterministic part (needs the monotone response the envelope was built to supply — the step that must be written carefully, since the envelope is monotone in $\lambda$ per instance but the update responds to a stochastic estimate); Azuma for the $1/\rho$-bounded martingale differences; a self-normalized step for the moving ratio target. *Open items:* the telescoping under IW noise with a ratio target; and the **envelope price** $\mathbb E[\tilde\ell_\lambda-\ell_\lambda]$, an empirical quantity (how often do votes flip back?) measurable on banks *before* any theory is written — preregistered as exhibit X6, with the honest possibility that the price is large and Route A is efficient only for commit-dominant schedules.

**Route B (anytime-valid certificates per knob).** Maintain, per $\lambda$ on the audited conditional stream, an e-process/betting confidence sequence [C]/[V] for $R(\lambda)$; deploy the most aggressive $\lambda$ whose anytime-valid upper bound is $\le\alpha$; switch only at audit epochs. Valid without monotonicity under exchangeable segments (and cleanly composable with the anytime-valid CRC toolkit [V]); degrades under genuine drift to a detect-and-reset scheme whose regret is unanalyzed. *Open item:* drift robustness beyond piecewise exchangeability.

**What we will not claim.** No adversarial-stream regret against the best-feasible-in-hindsight policy: for adversarial loss/constraint sequences this collides with the impossibility of Mannor, Tsitsiklis & Yu (2009) [C]; any regret clause must assume stochastic or slowly-drifting streams, and we decline to headline one. **Preregistered fallback** if neither route closes: block-wise scheduled recalibration from audit batches (Theorem 2 applied per block under piecewise exchangeability), plus a published *drift failure analysis* of fixed calibrated policies — a finding either way, per gate K4 (§7.4).

## 6. Experimental design

The program is **bank-first**: one GPU pass per model×task cell materializes everything; every number in the paper thereafter — calibration, baselines, ablations, negative exhibits, online simulations — is a deterministic CPU replay over released banks. This is the predecessor repository's suffstats-and-replay discipline lifted from token level to trajectory level.

### 6.1 Models

Two open reasoning families, frozen revisions, plus one scale point:
- **DeepSeek-R1-Distill-Qwen-7B** [C] — primary; long-CoT, well-characterized on math.
- **Qwen3-8B (thinking mode)** [C] — second family/lineage.
- *Stretch:* one ~32B-class reasoning model (QwQ-class) on one benchmark, for a scale note.
One cell will additionally attach an open process reward model as an optional signal rung and as the PRM-top-$k$ baseline; the main method must not require it.

### 6.2 Tasks: three verification regimes

- **Exact-match math:** MATH-500 (500 problems); AIME 2024+2025 (60 problems, held out as a *test-only* hard cell — §5.3's power accounting makes calibrating on it impossible, and we say so rather than hide it).
- **Executable code:** a LiveCodeBench slice (~150 problems, post-model-cutoff window to limit contamination), hidden unit tests as grader.
- **Multiple-choice science:** GPQA-Diamond (198), single-key grading; the weak-signal regime (answers extractable early, votes noisy).
Contamination hygiene: decontamination checks against each model's known cutoff; AIME'25 and the LCB window chosen post-cutoff where possible; MinHash dedup tooling reused from the predecessor repo.

### 6.3 Banks

Per model×task: $N=32$ rollouts/problem at temperature and top-$p$ fixed to each model's recommended reasoning settings, $T_{\max}$ 8k (MATH), 16k (AIME/GPQA/LCB); per-token top-20 logprobs and entropies stored; answers extracted by fixed regexes/executors; outcomes graded. Sub-banks $N\in\{8,16\}$ are prefix-subsets of the same banks (no regeneration); because rollouts are i.i.d., the pilot's $N{=}16$ MATH bank is reused verbatim as the first half of the full program's $N{=}32$ bank, and split manifests persist from pilot to full program so no problem ever changes role (tune/calibration/test) after first being touched. A 100-problem slice per cell is regenerated with 3 seeds for bank-stochasticity error bars. Checkpoint stride $\Delta=512$; window $w$, extraction rules, and all schedule hyperparameters fixed on tune splits. Storage ≈ 30–60 GB compressed. Banks, replayer, manifests, and gate configs are the released artifact.

### 6.4 Splits

Problem-level four-way splits, stratified, manifest-pinned, reused-forbidden across roles: **tune** (signals, FST order, grid pruning), **calibration** (Algorithm 3 only), **test** (one-shot realized-risk and cost reporting), **oracle-audit** (deployment-equivalence check X2). AIME is test-only. Transfer cells: calibrate on MATH-500, test on GPQA/LCB (and each permutation), reported as *out-of-contract* diagnostics — the certificate does not extend there, which is the point of showing them.

### 6.5 Calibration-feasibility accounting (design honesty)

| Cell (7B, $N{=}16$, plausible) | cal problems | plausible $p_E$ | $n_1$ | feasible $\alpha$ |
|---|---|---|---|---|
| MATH-500 | 250 | ~0.90–0.95 | ~225–240 | 0.02–0.05 comfortably |
| GPQA-Diamond | 99 | ~0.5–0.6 | ~50–60 | ~0.10; pool for less |
| LiveCodeBench slice | 75 | ~0.4–0.6 | ~30–45 | test-only or pooled |
| AIME'24+'25 | — | ~0.3–0.6 | — | test-only |

($p_E$ entries are planning estimates from published pass@1/cons@k ranges, to be replaced by measured values; the table's *structure* — which cells can calibrate at which $\alpha$ — is itself reported in the paper.) Pooled calibration across benchmarks uses benchmark-ID Mondrian groups (Thm. 4).

### 6.6 Baselines, metrics, ablations

**Baselines** (all replayed on identical banks, tuned on tune splits; oracle-tuned variants additionally get test-set-optimal thresholds, reported as oracles): full-bank reference; static best-of-$k$, $k\in\{1,2,4,8,16\}$; random pruning at matched realized cost; DeepConf-low/high [V] (offline + oracle-tuned; primary heuristic frontier); entropy-threshold kill; ESC / adaptive self-consistency [C]; single-trajectory conformal stopping per rollout (Conformal-Thinking-style [V], adapted to our banks; their contract, our action space); CWWI-style instance gate [R, if locatable]; PRM top-$k$ at checkpoints (where PRM attached); Certaindex-style index policy [V]; marginal-contract CASPR (the Prop.-1 negative exhibit, not a competitor).

**Metrics.** Primary: realized success-conditional risk on test with exact binomial CIs, at each nominal $\alpha\in\{0.01,0.02,0.05,0.10\}$ ($\delta=0.1$) — the **validity plot** (nominal vs realized, the money plot); certified and realized token fraction $C(\lambda)$; absolute accuracy vs reference (net drop, per Remark 1). Secondary: survival-vs-selection decomposition (Prop. 3); marginal risk (negative exhibit); per-group risks under Mondrian; cost-risk Pareto frontiers with baselines at matched *realized* risk; **price of certificate** = cost gap between $\hat\lambda$ and the best oracle-tuned heuristic at equal realized risk; wall-clock and KV-peak (§6.8); $n_1$ per cell; envelope price $\mathbb E[\tilde\ell-\ell]$ (X6).

**Uncertainty reporting.** Problem-level cluster bootstrap (1,000 resamples) for cost/accuracy/frontiers; exact CP intervals for all risks; 3-seed bank slice for generation stochasticity; all CIs at 90%, preregistered; every plot carries $n$ and $n_1$.

**Ablations.** Signal ladder (cumulative rungs): (1) windowed confidence only → (2) +entropy → (3) +length → (4) +cross-branch agreement → (5) +PRM → (6) +internal-unigram/morphology features → (7) +external corpus frequency → (8) +residualized frequency (external minus internal-unigram prediction). *Preregistered frequency demotion rule:* rungs 7–8 are retained in the paper only if they improve certified cost at fixed $\alpha$ by ≥2% relative, replicated in ≥2 model×task cells, after rung 6 — the exact confound discipline the Phase-0 post-mortem demands; otherwise they are reported in one sentence and dropped. Also: commit knob off (kill-only) and kill off (commit-only); stride $\Delta\in\{256,512,1024\}$; floor $\in\{1,2\}$; $N\in\{8,16,32\}$; aggregator $\in$ {majority, confidence-weighted, PRM-select}; Mondrian on/off.

### 6.7 Preregistered exhibits

**X1 (adverse selection, empirical Prop. 1):** calibrate against $R_{\mathrm{marg}}$, plot conditional failure rates by difficulty stratum; prediction: failures concentrate in the hardest solvable stratum, and disappear under the conditional contract at matched savings. **X2 (replay closure):** deploy $\hat\lambda$ generatively on the oracle-audit split; compare deployed vs replayed answer/cost distributions (prediction: statistically indistinguishable; any gap indicts A2/A3 violations in the serving stack). **X3 (validity plot)**, **X4 (Pareto + certificate price)**, **X5 (decomposition)**, **X6 (envelope price)**, **X7 (transfer/drift diagnostics)**: as defined above. Exploratory only (no headline): online-stream simulations of Routes A/B over reordered banks (i.i.d. shuffle; benchmark-ordered drift; difficulty ramp) at audit rates $\rho\in\{0.05,0.1,0.2\}$.

### 6.8 Wall-clock evaluation

Token fractions are not latency. A vLLM-based serving replica (fixed hardware, e.g. one A100-80GB; greedy batching of the rollout population per problem) measures end-to-end per-problem latency and aggregate throughput for $\pi_{\mathrm{full}}$ vs $\pi_{\hat\lambda}$ on a 100-problem subsample × 3 repetitions per cell, plus peak KV occupancy. Hypothesis to test, not assume: population pruning improves latency *superlinearly* relative to tokens saved when stragglers dominate the batch critical path (kills and early commits truncate exactly the stragglers). Reported with hardware/config manifests; no cross-hardware extrapolation.

## 7. Pilot, full matrix, budget, and preregistered gates

### 7.1 Smallest decisive pilot (P1)

**Scope:** DeepSeek-R1-Distill-Qwen-7B; MATH-500 ($N=16$, $T_{\max}$ 8k, banks for all 500; 125 tune / 250 calibration / 125 test) plus AIME'25 (30 problems, test-only); core grid $|\Lambda|=42$; baselines: full, best-of-$k$, random, DeepConf-offline, DeepConf-oracle, ESC.
**Cost:** ~25–30M generated tokens ≈ **10–20 A100-hours** (batched vLLM), then CPU replay. Wall-clock bench deferred to full paper.
**Deliverables:** X1–X5 on one cell; measured $p_E$, $n_1$; the envelope-price measurement X6.

**Preregistered pilot gates (all evaluated one-shot on the pilot test split):**
- **P-V (validity):** realized conditional failure count of $\hat\lambda(\alpha{=}0.05)$ within the exact binomial 95% acceptance region for $\alpha=0.05$ at the realized $n_{1,\mathrm{test}}$. *Fail ⇒ protocol or theory bug; full stop until diagnosed in writing.*
- **P-H (headroom):** certified token savings at $\alpha=0.05$ ≥ **25%** vs full $N=16$. *Fail ⇒ the certified-efficiency thesis dies on this action space; pivot to the certificate-price audit paper (gate K3's exit), do not proceed to full banks.*
- **P-P (price):** oracle-tuned DeepConf at matched realized risk saves < **2×** our certified savings. *Fail ⇒ reframe: the paper becomes "what a guarantee costs at test time" (still bank-first, same artifact).*
- **P-N (power):** $n_1\ge150$ on the calibration split. *Fail ⇒ resize splits or drop to $\alpha=0.1$ before full banks.*
- **P-S (separation, diagnostic):** decomposition X5 attributes ≥70% of conditional failures to one nameable term. *No kill; informs whether schedule or aggregator work dominates next.*

### 7.2 Full experiment matrix

| Axis | Levels |
|---|---|
| Models | R1-Distill-7B; Qwen3-8B-thinking; stretch 32B×1 benchmark |
| Tasks | MATH-500; AIME'24+'25 (test-only); GPQA-Diamond; LCB slice |
| $N$ | 8, 16, 32 (sub-banked) |
| Contract | $\alpha\in\{0.01,0.02,0.05,0.10\}$, $\delta=0.1$; Mondrian {benchmark, length-bin, first-checkpoint-entropy bin} |
| Policies | $|\Lambda|\approx200$ nested $(\kappa(t),v)$ schedules |
| Baselines | §6.6 list |
| Ablations | §6.6 list incl. frequency rungs 7–8 with demotion rule |
| Exhibits | X1–X7 + exploratory online simulations |
| Seeds | 3-seed 100-problem slice per cell |

### 7.3 Compute budget (planning estimates, not results)

Bank generation: 2 models × (~900 problems) × 32 rollouts × ~4–6k avg tokens ≈ 250–350M tokens ≈ **80–150 A100-80GB-hours** batched; 3-seed slices +15%; stretch 32B cell +60–100 h; serving benchmark ~20 h. **Total ≈ 150–300 A100-hours**, front-loaded; all analysis thereafter CPU. Pilot is ~10% of the total and gates the rest.

### 7.4 Full-paper preregistered pass/kill gates

- **K1 (headroom):** ≥30% token savings at $\alpha=0.05$ with realized validity, on ≥2 model×task cells ⇒ method paper proceeds. Else ⇒ certificate-price audit paper.
- **K2 (validity):** any calibrated policy violating its realized-risk acceptance region on a clean in-distribution test cell ⇒ full stop, written diagnosis, no submission until resolved.
- **K3 (price):** if oracle-tuned heuristics dominate every certified point by >2× cost at equal realized risk ⇒ the headline becomes the measured price of certification (an honest, publishable finding on the same artifact).
- **K4 (online):** if neither Route A nor B yields a provable theorem by the theory deadline (6 weeks after pilot pass), ship offline-only + drift failure analysis; the online section remains "open problem," as drafted.
- **K5 (envelope):** if X6's envelope price exceeds 30% relative cost at matched risk, Route A is declared inefficient and the online program is Route B or fallback only.
- **K6 (frequency):** rungs 7–8 fail the demotion rule ⇒ frequency is dropped from the paper except one sentence in the signal audit. (Symmetrically: passing does not promote it to the headline.)

### 7.5 Submission-time re-sweep gate (mandatory)

Before any submission: re-verify the [R]/[U] items (the recall-cascade claim; REFRAIN/ReASC/SAT; value-thresholded abstention; CWWI's venue; VACP), re-check Conformal Thinking's camera-ready for population extensions, and search anew for set-valued/preservation-contract work (including anything building on anytime-valid CRC [V]). If the composed object of §2 is occupied by then, the delta narrows to: contract-class analysis + end-to-end schedule calibration + decomposition + banks; the introduction is rewritten accordingly *before* submission, not after review.

## 8. Limitations and failure modes

1. **Reference-relative, one-sided.** CASPR certifies preservation of the reference's wins; it makes no absolute-accuracy promise and no promise about problems the reference fails. A better-than-reference pruned policy earns no extra credit from the contract (it is reported, not certified).
2. **Population-level, not per-instance.** Proposition 1's mixture critique applies *within* any population we fail to stratify; Mondrian groups mitigate exactly as far as pre-declared, deployment-measurable groups go; Remark 2 says no method can do better distribution-free. A user with one problem gets a population promise.
3. **Frozen-protocol dependence.** The certificate binds $(M,\sigma,N,T_{\max})$, model revision, extraction rules, and serving stack (A2/A3). Sampler upgrades, revision bumps, or batch-coupled inference void it; X2 is the tripwire.
4. **Calibration hunger on hard tasks.** $n_1\alpha\gtrsim\ln(1/\delta)$ is unforgiving: AIME-class cells cannot be calibrated at small $\alpha$ with public problem counts — they are test-only here. This is a fact about the contract's honesty, and it will be presented as such, but it limits claimed coverage.
5. **Verifier dependence.** Guarantees are relative to graders. The GPQA cell probes the weak-signal regime, but true weak-judge settings (open-ended generation) are out of scope.
6. **No spawning, no agents.** Adaptive branch growth and multi-turn feedback loops break Lemma 1; the method is pruning/commit-only over a fixed initial population, and says so.
7. **Online maintenance is open.** §5.5 is a program with two routes and a fallback, not a theorem; under drift, the shipped guarantee is per-block recalibration or nothing.
8. **Scoop risk.** Conformal Thinking's group is one step away; the [U] recall-cascade may exist; §7.5 is the mitigation, not a solution.
9. **Wall-clock transfer.** Token savings need not survive arbitrary serving stacks; we measure one configuration honestly and decline to extrapolate.

## 9. The final claim stack (what the submission will assert)

1. **[Formulation/negative]** Marginal preservation contracts for test-time pruning are vacuous under task hardness and adversely selecting under cost minimization — proved (Prop. 1) and demonstrated on real banks (X1). Success-conditional contracts repair both; group-conditional versions are the strongest distribution-free-achievable refinement (Thm. 4, Remark 2).
2. **[Method/validity]** CASPR controls the success-conditional preservation risk at user-specified $(\alpha,\delta)$ with finite samples, end-to-end over multi-checkpoint kill-and-commit schedules, with calibration free of counterfactual censoring (banks) and deployment covered by replay closure (Lemma 1, Thm. 2), and with certified failures attributable to preservation vs selection (Prop. 3). Statistical machinery inherited from LTT/CP; the decision surface, conditioning event, closure lemma, and decomposition are the contribution.
3. **[Empirical]** On two model families and three verification regimes: realized validity across the $\alpha$ grid (X3); ≥30% certified token savings at $\alpha=0.05$ on ≥2 cells (K1 target — asserted only if achieved); the measured price of the certificate against oracle-tuned heuristic frontiers (X4); survival/selection attribution (X5); wall-clock and memory on a fixed serving replica (§6.8). Every number replayable from released banks without a GPU.
4. **[Open program]** The online drift-maintenance problem formulated with the exact obstruction (non-monotone vote losses), a corrected monotone-envelope surrogate with measured price (X6), two proof routes, and a preregistered fallback — explicitly labeled open.
   **Not asserted anywhere:** absolute-accuracy gains; per-instance guarantees; guarantees under spawning, agentic feedback, or distribution shift; any frequency headline; any online theorem (unless K4's deadline is beaten, in which case claim 4 upgrades and says so).

## 10. Reuse from the current repository vs new construction

**Direct reuse (~25–35% of code):** conformal/binomial utilities (finite-sample ranks, exact CP bounds) → the certification core; four-way split manifests, content-addressed artifacts, model/tokenizer revision pinning, MinHash dedup → bank hygiene; PASS/FAIL gate architecture with machine-readable decision memos → gates P-*/K-*; clustered-bootstrap reporting; the GLM/isotonic signal-audit instrument → the §6.6 ladder (unit of analysis changed from token position to problem×schedule); frequency tables → rungs 7–8 only.
**Reuse as concepts (~60–70% of protocol):** fail-closed runners; preregistration discipline; [G]/[E] evidence grading (retained verbatim in the paper's reporting); tune/cal/test role separation; suffstats-then-replay architecture (the banks *are* trajectory-level suffstats).
**Built new:** BANKGEN harness (vLLM streaming with per-token top-20 logprob capture, answer extraction, graders per regime); the deterministic replayer with suffix masking; the schedule grid + FST calibrator; baseline reimplementations (DeepConf/ESC/CT-style/Certaindex-style); the serving replica bench; online-simulation harness.
**Explicitly retired:** the ν score and name; $h(m,n)$ as a scientific centerpiece; next-token prediction-set suffstats and benchmarks; all legacy generation results; Phase-0 numbers in any evidentiary role (they appear only in §1.3 as motivation, labeled as a failed pilot).

---

## Appendix A — Design constants (preregistered defaults)

$\Delta=512$; $w=512$; floor $=1$; $\mathrm{AGG}$ = confidence-weighted majority with deterministic tie-breaks; $\delta=0.1$; CP intervals exact; FST order = ascending tune-estimated savings; grid as §4.2; all frozen before calibration data is touched, changes only via a written amendment before the affected run.

## Appendix B — Owed proofs and write-outs (tracked obligations)

1. Measure-theoretic footnote: joint conditioning on $n_1$ with FST selection (routine).
2. Remark 2's adaptation of the conditional-validity impossibility to policy losses (routine but owed).
3. Lemma 1 full proof with the explicit coupling construction and the batching-caveat discussion.
4. Route A/B analyses (open; K4 deadline applies).
5. Verification that confidence-weighted AGG with tie-breaks is a.s. unambiguous under the stored precision (implementation-level, but the determinism claim depends on it).

## Sources

**Uploaded proposals adjudicated:** `topno_gpt5.6pro.md` (Report A); `topno_deep_review_20260710.md` (Report B).
**Key external items verified 2026-07-10:** [Conformal Thinking, arXiv:2602.03814](https://arxiv.org/abs/2602.03814) ([ICML 2026 per Apple ML](https://machinelearning.apple.com/research/conformal-thinking-risk-control)) · [DeepConf, arXiv:2508.15260](https://arxiv.org/abs/2508.15260) · [Certaindex, arXiv:2412.20993](https://arxiv.org/abs/2412.20993) · [min-p critical analysis, arXiv:2506.13681](https://arxiv.org/abs/2506.13681) · [CROP, arXiv:2605.30085](https://arxiv.org/abs/2605.30085) · [Truncation Blind Spot, arXiv:2603.18482](https://arxiv.org/abs/2603.18482) · [ORCA, arXiv:2604.01170](https://arxiv.org/abs/2604.01170) · [Local Branch Routing, arXiv:2606.25354](https://arxiv.org/abs/2606.25354) · [ATTS, arXiv:2509.15148](https://arxiv.org/abs/2509.15148) · [p-less Sampling, arXiv:2509.23234](https://arxiv.org/abs/2509.23234) · [Anytime-Valid Conformal Risk Control, arXiv:2602.04364](https://arxiv.org/abs/2602.04364) · [Conformal Risk Control, arXiv:2208.02814](https://arxiv.org/abs/2208.02814). Items tagged [C]/[R]/[U] per Part 0.
