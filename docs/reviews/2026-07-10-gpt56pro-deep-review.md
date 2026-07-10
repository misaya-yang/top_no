# Executive verdict

Area-chair decision on the current project: reject as a method paper; preserve it as unusually strong audit and protocol infrastructure.

There is still meaningful innovation space in heuristic or model-weight-free decoding in 2026. ICLR 2026 gave an oral to p-less Sampling, and ACL 2026 published Min-k Sampling as a long paper. Those results show that static sampling is not categorically exhausted. But the bar has moved: a new rule now needs a robust invariant, unusually decisive evaluation, or direct systems relevance—not merely another scalar threshold with plausible intuition. (开放评审)

For this repository specifically:

- Abandon ν-sampling as the centerpiece and probably abandon the name.
- Do not resume the existing broad next-token prediction-set benchmark.
- Retain external frequency only as an optional, residualized side feature.
- The strongest ambitious pivot is toward risk-budgeted branch survival for test-time search: allocate compute so that pruning is calibrated against the downstream event "all potentially successful continuations were eliminated."
- The lower-risk fallback is a rigorous multi-tokenizer audit of whether external frequency survives internal-unigram, morphology, entropy, rank, and token-identity controls.

I use three labels below:

- **Verified**: directly supported by the repository or primary literature.
- **Deduction**: follows mathematically or methodologically from the verified evidence.
- **Proposal**: a new research direction; not an established result.

---

## 1. Review of the current repository

### 1.1 What the project has done well

**Verified.** The repository has responded correctly to the original failure:

- It explicitly retires the frequency-dependent noise-channel identification story.
- It treats conformal coverage as inherited machinery rather than novelty.
- It distinguishes general learned (h(m,n)), additive (g(n)), and frequency-Mondrian calibration.
- It recognizes that a frequency interaction does not imply horizontal-shift structure.
- It has repaired nucleus crossing-token behavior, logit/temperature ordering, padding, EOS handling, silent fallback, and several split/leakage issues.
- It is deliberately fail-closed for paper-grade runs.
- It labels the latest pilot E-pilot, paper_citable=false, and INSUFFICIENT. (GitHub)

This is good research behavior. The protocol, audit trail, and willingness to kill claims are more valuable than most of the original method code.

The feature-restricted formulation is also mathematically clean:

$$
h_{\mathcal G}=\mathbb E[\mathbf 1\{Y=I\}\mid \mathcal G],
$$

with retain regions given by upper level sets of $h_{\mathcal G}$. The nested-feature argument correctly says that $(m,n)$ can weakly improve on $m$, while refusing to claim that it beats richer context features. The repository also correctly states that additive offsets require a single-index structure such as

$$
h(m,n)=\rho(m-g(n)).
$$

That is a useful organizing lens, but it is an adaptation of least-ambiguous set classification/Neyman–Pearson reasoning, not a deep new theorem. (GitHub)

### 1.2 What the Phase-0 pilot actually says

The signed rare-versus-reference shifts recorded in the decision memo are:

| Cell | Signed shift | Half-stable | Separated from permutation |
|------|-------------|-------------|---------------------------|
| Qwen2.5-3B math | -2.0 | yes | yes |
| Qwen2.5-3B web | -0.2 | no | yes |
| Qwen2.5-7B math | -2.0 | yes | no |
| Qwen2.5-7B web | -0.3 | no | yes |

All four directions are negative. The implemented original ν rule with positive $\kappa$ does the opposite: it widens the admissible margin for lower-frequency tokens. Thus, to the extent the pilot indicates anything, it points away from the original rare-token-conservatism story and toward pruning or desmoothing of rare-tail logits. (GitHub)

There are three additional reasons not to overinterpret the table.

**First**, the shift search in phase0_stats.py is restricted to $[-2,2]$. Both math cells hit exactly $-2.0$, and the 7B-math permutation hits magnitude $2.0$. Those are boundary-censored values, not identified effect magnitudes. They say that the fitted objective preferred an edge of the search range; they do not establish a two-nat shift. (GitHub)

**Second**, both web cells changed their chosen reference group across halves. A contrast whose anchor changes is not estimating the same simple quantity in the two halves. This is more serious than an ordinary wide confidence interval.

**Third**, the single vocabulary permutation preserves bucket cardinalities but destroys morphology, token length, script, whitespace behavior, output-head geometry, model-internal unigram, and semantic/token clusters simultaneously. It is therefore a weak negative control for the claim "external frequency itself matters." A structure-preserving conditional randomization test is needed.

**Bottom line**: the pilot is compatible with a real external-frequency effect, a token-identity/morphology effect, a model-unigram effect, or an unstable reference artifact. It does not distinguish them. The repository's own pause decision is correct. (GitHub)

### 1.3 The deeper problem is not merely insufficient power

Even a perfectly executed positive $h(m,n)$ study would leave four major scientific gaps.

#### A. Gold next-token coverage is weakly connected to generation utility

**Deduction.** The observed corpus token is one acceptable continuation, not the unique desirable action. Dropping it can be harmless if a semantic equivalent remains; retaining it can be useless if its continuation has low task value.

Consequently,

$$
\Pr(Y_t\in S_t)
$$

is not generally the decision-relevant risk. For reasoning, search, constrained generation, or speculative execution, useful risks are closer to:

- probability of eliminating every successful branch;
- task-accuracy loss relative to a full rollout bank;
- probability of violating a formal or semantic constraint;
- draft–target disagreement;
- expected verifier regret;
- probability that no semantically equivalent continuation remains.

This is the main reason the project should move beyond next-token prediction sets.

#### B. Prediction-set size is not production inference cost

**Deduction.** In ordinary autoregressive decoding, the transformer and usually the complete vocabulary head have already been evaluated before a truncation mask is applied. Reducing a support from 10,000 tokens to 100 tokens generally does not reduce transformer FLOPs and often does not materially reduce latency.

Set size is a legitimate statistical efficiency metric. It is not a production-compute metric unless the method changes something upstream, such as:

- vocabulary-head computation;
- speculative candidate generation;
- branch expansion;
- constrained-search state expansion;
- communication or memory traffic;
- number or length of rollouts.

FR-Spec is instructive precisely because it couples token frequency to vocabulary-space compression inside speculative execution rather than claiming that a smaller post-softmax set is itself a serving speedup. (ACL Anthology)

#### C. Frequency is currently standing in for too many latent variables

At minimum, a convincing frequency result must control for:

$$
\begin{aligned}
&\text{margin, rank, probability, entropy, logit scale},\\
&\text{token byte length, character class, whitespace, script, case},\\
&\text{model-internal context-averaged unigram probability},\\
&\text{output embedding norm/bias or equivalent head statistics},\\
&\text{token semantic or embedding cluster},\\
&\text{tokenizer segmentation and alternative-tokenization behavior}.
\end{aligned}
$$

A token-specific empirical-Bayes intercept is also a necessary baseline. Otherwise, frequency may merely be a low-dimensional proxy for token identity.

#### D. The theoretical object is sound but too narrow

The feature-class frontier theorem says when side information can help a retain/drop decision under the chosen candidate-pair measure. It does not establish that retain/drop is the right downstream action, that frequency is the best side information, or that support cardinality is the correct cost.

That theorem is appropriate for an audit paper. It cannot by itself carry an ICML/NeurIPS/ICLR method paper.

---

## 2. Decoding and inference landscape through July 10, 2026

### 2.1 Static and locally adaptive truncation

The static/truncation line remains active. Eta sampling interprets truncation as desmoothing and makes its threshold entropy-dependent; locally typical sampling selects tokens by deviation from context information content. More recently, min-p, Top-$n\sigma$, p-less Sampling, and Min-k have proposed relative-probability, normalized-logit, hyperparameter-free, or local sorted-logit criteria. P-less is an ICLR 2026 oral and Min-k is an ACL 2026 long paper. A 2025 preprint critically re-evaluating min-p also illustrates how fragile sampler conclusions can be to baselines and evaluation design. (ACL Anthology)

**Occupied territory**: another context-local boundary determined by entropy, normalized logits, relative probabilities, or a "tail cliff."

**Remaining space**: a static sampler can still succeed, but only with a much stronger invariant or measurement result than "token frequency appears related to reliability."

### 2.2 Online and uncertainty-adaptive decoding

Mirostat already framed decoding as online feedback control of realized surprise/perplexity. Adaptive Decoding and subsequent uncertainty-aware methods vary candidate sets or model-combination weights according to confidence or entropy. UCD, for example, dynamically adjusts contrastive-model contributions at each step rather than using one global weight. (arXiv)

**Occupied territory**: "use entropy or uncertainty to adjust a threshold/temperature/weight dynamically."

**Remaining space**: decisions tied to a calibrated downstream loss, especially under delayed feedback or branch dependence.

### 2.3 Conformal and risk-aware generation

Conformal Nucleus Sampling already calibrated next-token nucleus sets; Non-Exchangeable Conformal Language Generation addressed weighted/nonexchangeable token-level settings; Conformal Language Modeling moved to response-level validity. Conformal risk control supplies the generic machinery for controlling bounded downstream losses. None of these makes frequency-aware scores novel by itself. (ACL Anthology)

The 2026 frontier has moved further:

- **ATTS**, accepted at ICLR 2026, combines statistical rejection control, speculative execution, and asynchronous parallel/sequential test-time scaling.
- **ORCA**, an April 2026 preprint, calibrates reasoning-time sampling under distribution shift using test-time adaptation and reports substantial compute savings.
- **CATS** is an ICLR 2026 workshop oral, not a main-conference paper; it chooses lower test-time scaling modes while controlling expected accuracy loss.
- **VACP** is a 2025 preprint specifically on efficient next-token conformal prediction sets. (arXiv)

**Occupied territory**: "use conformal calibration to choose a sampling budget or prediction set."

**Remaining space**: a new loss, action granularity, or sequential composition problem—not another application of split conformal to a new score.

### 2.4 Long-reasoning and test-time compute control

Compute-optimal test-time scaling already allocates search effort according to problem difficulty. By ACL 2026, the space includes:

- **REFRAIN**: adaptive early stopping with a sliding-window UCB controller;
- **ReASC**: confidence-weighted adaptive self-consistency;
- **SAT**: step-level transitions among reasoning modes using a process reward model;
- confidence-weighted token set cover for pruning parallel hypotheses;
- dynamic value-thresholded abstention, accepted at ICML 2026;
- a July 7, 2026 preprint giving an episode-level recall-controlled cascade for aborting doomed agent trajectories. (开放评审)

**Local Branch Routing**, a June 2026 preprint, goes even closer to branch-level control: it constructs local lookahead trees and trains a router to commit to a subtree. (arXiv)

**Occupied territory**: generic early stopping, adaptive numbers of samples, value-threshold abstention, and learned branch routing.

**Remaining space**: calibrated selection among multiple concurrent branches, with a global guarantee against eliminating every viable path and an explicit compute-reallocation policy. That window is real but narrow.

### 2.5 Speculative decoding

Exact speculative decoding preserves the target distribution while reducing target-model calls. Subsequent work has expanded draft structures, acceptance policies, and vocabulary compression. FR-Spec already uses token frequency to compress large draft vocabularies while preserving the final target distribution; ATTS couples speculative decoding to test-time scaling and statistical rejection control. (Proceedings of Machine Learning Research)

**Occupied territory**: frequency-ranked draft-vocabulary restriction, exact acceptance, learned/adaptive drafter selection, and asynchronous scaling.

**Remaining space**: deliberately approximate speculation with a task-level risk budget, or joint branch-value/acceptance control. This is a systems-heavy program.

### 2.6 Semantic and constrained decoding

DOMINO provides efficient subword-aligned constrained generation for formal languages; sequential Monte Carlo has been used for syntactic and semantic control. These methods operate on structured constraints and path weights rather than a scalar truncation threshold. (Proceedings of Machine Learning Research)

Semantic entropy and Kernel Language Entropy move uncertainty from surface tokens toward equivalence classes or similarities among complete responses. This is a strong warning against treating raw token frequency as the natural uncertainty representation. (Nature)

**Occupied territory**: formal constraint satisfaction and semantic clustering of output uncertainty.

**Remaining space**: calibrated, tokenizer-invariant event sets that drive actual search expansion.

### 2.7 Landscape synthesis

**Verified conclusion**: the field is not out of decoding ideas.

**Deduction**: "training-free" is no longer a sufficient novelty claim. A credible 2026 project needs at least one of:

- a new decision-relevant risk;
- a new action space, such as branches or semantic events;
- an online or anytime guarantee;
- a robust invariance across tokenizers/domains;
- measurable serving benefits;
- a decisive negative measurement that rules out a widely assumed signal.

The current static frequency threshold has none of these yet.

---

## 3. Five stronger research theses

### Thesis A — Risk-budgeted branch survival for test-time search

#### 1. Central scientific question

At a reasoning checkpoint with candidate branches $\mathcal B$, can a post-hoc controller retain the cheapest subset $S\subseteq\mathcal B$ while controlling

$$
\Pr\left(
\text{the full branch bank contains a successful continuation, but }S\text{ contains none}
\right)?
$$

This replaces gold next-token coverage with baseline-relative catastrophic-pruning risk.

#### 2. What would be genuinely new

The closest works now include adaptive self-consistency, hypothesis set cover, dynamic abstention, the recall-controlled agent-abort cascade, ORCA, and Local Branch Routing. The novelty cannot be "prune low-confidence traces."

It would instead be:

- a set-valued branch action, not a stop/continue action for one trace;
- a downstream viability loss, not lexical coverage, rejection rate, or confidence;
- explicit global risk allocation across branches and checkpoints;
- compute reallocation from pruned branches to surviving or newly sampled branches;
- no foundation-model finetuning—only a tune/calibration controller.

The July 2026 recall-controlled abort paper explicitly names reallocation of aborted compute to retries as a next step, so this opportunity may close quickly. (arXiv)

#### 3. Possible theoretical contribution

For fixed checkpoints, define $Z_b\in\{0,1\}$ as whether branch $b$, under a frozen continuation and verifier protocol, yields a valid solution. Define

$$
L(S,Z) = \mathbf 1\left\{
\max_{b\in\mathcal B} Z_b=1 \;\land\;
\max_{b\in S}Z_b=0
\right\}.
$$

The objective is

$$
\min_{\pi}\; \mathbb E[C(\pi(X))]
\quad
\text{subject to}
\quad
\mathbb E[L(\pi(X),Z)]\le \alpha.
$$

Two useful results are available:

- $f_x(S)=\Pr(\exists b\in S:Z_b=1\mid X=x,\mathcal B)$ is monotone submodular because it is an expected coverage function.
- A finite policy family can be selected using standard learn-then-test/conformal risk-control bounds so that its catastrophic-pruning risk is at most $\alpha$ with calibration confidence $1-\beta$.

Under conditionally independent Bernoulli branch success and equal costs, top-$k$ branch success probabilities are optimal. With dependence and unequal costs, this becomes a submodular cover problem.

These are adaptations of standard risk control and submodular optimization—not new generic theory.

The genuinely difficult theorem would cover adaptive tree growth, censored counterfactual outcomes, and predictable risk allocation across policy-induced nodes.

#### 4. Supporting or refuting empirical signature

**Support**:

- calibrated catastrophic-pruning loss at $\alpha\in\{0.01,0.05,0.10\}$;
- at least 20–30% fewer continuation tokens or target FLOPs at less than 0.5 percentage-point task-accuracy loss;
- gains beyond PRM top-$k$, entropy, adaptive self-consistency, weighted set cover, dynamic abstention, and learned routing;
- nontrivial benefit from modeling branch dependence/diversity rather than ranking branches independently.

**Refute**:

- confidence or PRM top-$k$ already lies on the same frontier;
- branch-success labels are too unstable to calibrate;
- wall-clock overhead eliminates token savings;
- calibration transfers poorly across tasks or search policies.

#### 5. Smallest decisive experiment and cost

One 7–8B model; MATH-500 and one code benchmark; 1,000 problems total; eight partial branches per problem at one frozen checkpoint; 256–512 continuation tokens per branch; exact-answer/unit-test outcomes.

Approximately 5–10 million generated tokens, or roughly 8–25 A100-80GB-equivalent hours, depending heavily on batching and serving implementation.

The decisive comparison is not against full self-consistency alone. It is against PRM/confidence top-$k$, weighted set cover, and a tuned early-abort baseline.

#### 6. Main reviewer attack

"Your guarantee is merely a binomial or conformal risk bound over a finite policy family, while the interesting branch values were obtained with expensive full counterfactual rollouts. The deployed adaptive search no longer has exchangeable examples or observed outcomes for pruned branches."

This attack is correct unless the adaptive/censored-feedback problem is solved.

---

### Thesis B — Tokenizer-invariant hierarchical event sets

#### 1. Central scientific question

Can decoding construct calibrated supports over canonical events—byte strings, AST actions, grammar states, mathematical operators, or semantic equivalence classes—rather than tokenizer-specific tokens?

#### 2. What would be genuinely new

Formal constrained decoding already maps token sequences to grammar states, while semantic-uncertainty work clusters completed responses. The proposed contribution would be a calibrated, hierarchical event set used to decide which partial continuations to expand, with invariance across tokenizers.

I did not find a published 2024–July 2026 method that cleanly combines:

- tokenizer pushforward probabilities;
- calibrated hierarchical event coverage;
- coarse-to-fine search expansion;
- comparison of equivalent events across tokenizers.

#### 3. Possible theoretical contribution

Let $\phi_T$ map token sequences under tokenizer $T$ to a canonical event space $\mathcal E$. For an event $e$,

$$
P_T(e\mid x) = \sum_{z:\phi_T(z)=e}P_T(z\mid x).
$$

Under exact aggregation and tokenizers representing the same canonical byte-string process, prediction sets constructed solely from the pushforward distribution $P_T\circ\phi_T^{-1}$ are invariant to tokenizer refinement.

A hierarchical version could minimize expansion cost:

$$
\min_{S\subseteq \mathcal E}
\mathbb E[c(S)]
\quad\text{s.t.}\quad
\Pr(E^\star\in S)\ge 1-\alpha.
$$

Conformal calibration is standard. The pushforward-invariance lemma is straightforward measure theory. The harder contribution is an efficient approximation with bounded mass error when enumerating tokenizations is impossible.

#### 4. Supporting or refuting empirical signature

**Support**:

- materially smaller cross-tokenizer variance in set coverage and search cost;
- event sets transfer between two tokenizers on identical source text;
- better support efficiency than token-level APS/margin sets;
- improved constrained-search or branch-expansion latency.

**Refute**:

- exact event aggregation is computationally prohibitive;
- approximations destroy coverage;
- event definitions are task-specific;
- grammar-constrained decoding already gives the same practical benefit.

#### 5. Smallest decisive experiment and cost

Start with two exact event spaces:

- JSON/grammar production actions;
- code or arithmetic AST actions.

Use two tokenizers and one or two 7B-class models on 50,000–100,000 decision points. Estimated cost: 5–15 A100 hours, plus significant engineering for tokenization tries and event aggregation.

#### 6. Main reviewer attack

"The claimed tokenizer invariance holds only because you defined an exact canonical event map in tasks with formal grammars. It does not generalize to natural-language semantics, where equivalence classes are model- or judge-dependent."

That likely limits the first paper to structured generation.

---

### Thesis C — Hierarchical empirical-Bayes side information instead of raw frequency

#### 1. Central scientific question

Is external token frequency useful because it estimates a latent lexical prior, and does it add anything after accounting for a model's own implicit unigram distribution and tokenizer morphology?

#### 2. What would be genuinely new

The current project treats one external count $n_i$ as fixed side information. A stronger program would estimate a latent token/event prior from several noisy views:

- external-corpus count;
- context-averaged model probability;
- domain-specific count;
- token morphology and byte structure;
- embedding or semantic cluster;
- tokenizer lineage.

The scientific target becomes residual external information, not raw frequency:

$$
r_i = \log(n_i+\epsilon) - \mathbb E[
\log(n_i+\epsilon)
\mid
u_i^{\text{model}},\;\text{morphology}_i,\;\text{cluster}_i
].
$$

#### 3. Possible theoretical contribution

One possible hierarchy is

$$
n_{i,d}\mid \theta_i,\delta_d \sim \operatorname{Poisson}
\left(N_d e^{\theta_i+\delta_{i,d}}\right),
$$

with

$$
\theta_i\mid c(i)\sim \mathcal N(\mu_{c(i)},\tau^2_{c(i)}).
$$

The posterior predictive prior becomes a feature for value or reliability estimation.

Under a correctly specified hierarchy and Bayes loss, posterior decisions minimize integrated Bayes risk; shrinkage can dominate raw counts in integrated estimation risk. This is standard empirical Bayes, not a new theorem.

A more interesting result would characterize when external counts improve a calibrated decision frontier after conditioning on the model-internal prior:

$$
V_{(m,u,r)}(s)>V_{(m,u)}(s)
$$

if and only if the refined conditional value differs in a decision-relevant neighborhood of the operating boundary—not merely somewhere in the distribution.

That is an adaptation of the repository's existing feature-frontier lens.

#### 4. Supporting or refuting empirical signature

**Support**:

- external residual $r_i$ improves held-out log loss or branch-value prediction after all controls;
- gains transfer across domains or tokenizers better than raw counts;
- posterior shrinkage is especially useful for rare tokens;
- calibrated support or branch policies improve at matched risk.

**Refute**:

- model-internal unigram and morphology absorb the complete effect;
- residual gains are tiny or tokenizer-specific;
- token-specific shrinkage is a stronger and simpler baseline.

#### 5. Smallest decisive experiment and cost

Use existing Phase-0 sufficient statistics where possible; add context-averaged unigram logits and token morphology. Test Qwen 3B/7B on web and math with cross-fitting and structure-preserving randomizations.

If logits are cached: mostly CPU plus under 2 GPU hours.
If new forward passes are needed: approximately 3–8 A100 hours.

This should be performed before any new frequency-centered method.

#### 6. Main reviewer attack

"This is a careful feature-engineering and empirical-Bayes audit. The theory is standard, and the external frequency effect disappears once the model's own lexical prior is included."

That failure would still be a valuable negative result, but not a top-tier method paper.

---

### Thesis D — Online risk control under domain drift and delayed feedback

#### 1. Central scientific question

Can a decoding controller maintain a target task-risk/compute frontier as the deployment stream shifts among web, math, code, tools, and languages?

#### 2. What would be genuinely new

Plain adaptive thresholds are already covered by ACI-style work, and ORCA directly targets online reasoning calibration under shift. Therefore the novelty would need to be:

- sparse or delayed task feedback rather than immediate labels;
- simultaneous control of risk and compute;
- explicit change-point or nonstationary regret analysis;
- branch- or trajectory-level actions.

ORCA makes a generic "online conformal reasoning" proposal substantially less novel. (arXiv)

#### 3. Possible theoretical contribution

One could optimize

$$
\sum_{t=1}^T C_t(\pi_t)
\quad\text{subject to}\quad
\frac1T\sum_{t=1}^T L_t(\pi_t)\le \alpha
$$

using an online primal-dual controller, with delayed observations of $L_t$.

A plausible result would combine:

- a long-run risk-violation bound;
- dynamic regret against the best time-varying policy sequence;
- dependence on policy path length or number of distribution changes.

This would be an adaptation of ACI, online convex optimization, and delayed-feedback results. It is only theoretically interesting if the actual decoding action and feedback process create a nonstandard problem. (NeurIPS Proceedings)

#### 4. Supporting or refuting empirical signature

**Support**:

- risk recovers rapidly after web→math→code shifts;
- lower cumulative compute than static or periodically recalibrated controllers;
- robust behavior with delayed, sparse, or noisy verifier labels.

**Refute**:

- offline rolling recalibration performs equally well;
- feedback is unavailable at deployment;
- risk guarantees reduce to average coverage while allowing severe transient failures.

#### 5. Smallest decisive experiment and cost

Replay a frozen stream with abrupt and gradual domain shifts using cached logits or rollouts. Approximately 2–6 A100 hours, or nearly zero GPU cost if all scores and outcomes are cached.

#### 6. Main reviewer attack

"This is generic ACI or online primal-dual learning applied to another score. The LLM-specific contribution is thin."

I consider this a weak standalone thesis unless attached to branch survival or speculation.

---

### Thesis E — Risk-controlled approximate speculative decoding

#### 1. Central scientific question

Can speculative acceptance be relaxed to improve speed while controlling a task-level or sequence-distribution risk, rather than insisting on exact target-distribution equivalence?

#### 2. What would be genuinely new

Exact speculative decoding, FR-Spec, and ATTS already cover exactness, frequency-ranked vocabulary compression, and statistically controlled asynchronous scaling. A new contribution would need to permit approximation deliberately and expose an explicit user-selected risk/throughput frontier. (ACL Anthology)

Frequency could predict draft–target mismatch, but it should be one feature among:

- target–draft margin disagreement;
- entropy;
- draft confidence;
- token morphology;
- residualized external frequency;
- recent acceptance history.

#### 3. Possible theoretical contribution

Let $P_t$ be the target conditional kernel and $\widetilde P_t$ the relaxed speculative kernel. If

$$
\operatorname{TV}(P_t,\widetilde P_t)\le \epsilon_t
$$

uniformly along relevant histories, standard coupling gives

$$
\operatorname{TV}(P_{1:T},\widetilde P_{1:T}) \le \sum_{t=1}^T\epsilon_t.
$$

Thus a sequence-level approximation budget can be allocated across positions.

Alternatively, for a bounded task loss $\ell$, choose an acceptance policy by conformal risk control:

$$
\min_\pi \mathbb E[\text{latency}(\pi)]
\quad\text{s.t.}\quad
\mathbb E[\ell(\widetilde Y)-\ell(Y_{\text{target}})]\le \alpha.
$$

Both are adaptations of standard coupling and risk control. The novelty would be the practical controller and measured systems frontier.

#### 4. Supporting or refuting empirical signature

**Support**:

- 1.3–2× end-to-end throughput gain over strong exact speculative baselines;
- calibrated task degradation below the selected risk level;
- stable benefits across target–draft pairs and domains;
- residual frequency predicts mismatch beyond target/draft logits.

**Refute**:

- sequence TV bounds are too loose to permit useful relaxation;
- task-level calibration fails to transfer;
- optimized exact systems already match the throughput;
- kernel and synchronization costs dominate.

#### 5. Smallest decisive experiment and cost

One 7–14B target, one 1–3B draft, three domains, and an offline replay of exact target/draft distributions. Algorithmic pilot: 10–30 A100 hours.

A paper-grade systems evaluation with optimized serving kernels would likely require well over 100 GPU hours and substantial systems engineering.

#### 6. Main reviewer attack

"You bought speed by changing the target distribution, your quality tests are underpowered, and your guarantee is either vacuous TV accounting or task-specific empirical calibration."

This can be a strong systems paper, but it is less aligned with the current repository.

---

## 4. Comparative ranking

My subjective assessment:

| Thesis | Scientific novelty | Pilot feasibility | Theory ceiling | Top-conference potential |
|--------|-------------------|-------------------|----------------|-------------------------|
| A. Branch-survival search | high | medium-high | medium-high if sequential problem solved | strongest |
| B. Tokenizer-invariant event sets | high | medium-low | medium | strong but task-specific |
| C. Hierarchical residual frequency | medium | high | low-medium | good audit/fallback |
| D. Online drift controller | low-medium | high | medium | crowded by ORCA/ACI |
| E. Approximate speculation | medium-high | medium-low | medium | strong systems route |

The strongest direction is A, but only in its multi-branch/global-risk form. A single-trace early-exit paper is already occupied.

---

## 5. Recommended research program

### Working title

**Preserve One Good Path: Risk-Budgeted Branching for Test-Time Language-Model Search**

### One-sentence thesis

At each reasoning checkpoint, retain the cheapest subset of candidate branches whose calibrated probability of eliminating every successful continuation is below a user-specified risk budget.

### Three defensible contributions

#### Contribution 1 — A decision-relevant risk object

Define and benchmark catastrophic branch-pruning loss:

$$
L(S,Z) = \mathbf 1\left\{
\text{full bank has a successful branch and retained bank has none}
\right\}.
$$

This is baseline-relative task degradation, not token coverage, entropy, or lexical diversity.

#### Contribution 2 — A calibrated cost-constrained branch selector

Build nested or finite branch-selection policies using:

- continuation-value estimates;
- branch uncertainty;
- semantic/diversity dependence;
- cost estimates;
- optional residualized token-frequency features.

Select a policy under an explicit risk constraint using established learn-then-test or conformal risk-control machinery.

#### Contribution 3 — Global risk allocation and real systems evaluation

Extend from one fixed checkpoint to multiple checkpoints with a risk budget allocated across the tree, and demonstrate reductions in generated tokens, target FLOPs, wall latency, and memory—not merely support cardinality.

### Main theorem statement

**Theorem — Fixed-checkpoint calibrated branch preservation**

Let

$$
W=(X,\mathcal B,\{Z_b,c_b\}_{b\in\mathcal B})
$$

denote one problem instance, where:

- $X$ is the prompt and checkpoint state;
- $\mathcal B$ is a finite branch bank generated by a frozen proposal protocol;
- $Z_b\in\{0,1\}$ indicates whether completing branch $b$ under a frozen continuation-and-verification protocol yields a valid solution;
- $c_b>0$ is its continuation cost.

Assume $W_1,\ldots,W_n,W_{n+1}$ are exchangeable.

Let $\{S_\lambda\}_{\lambda\in\Lambda}$ be a finite family of branch-selection policies fixed independently of the calibration set. Define

$$
L_\lambda(W) = \mathbf 1
\left\{
\max_{b\in\mathcal B}Z_b=1
\;\land\;
\max_{b\in S_\lambda(X,\mathcal B)}Z_b=0
\right\},
$$

and

$$
R(\lambda)=\mathbb E[L_\lambda(W)].
$$

For each $\lambda$, construct a simultaneous one-sided upper confidence bound $U_n(\lambda)$ satisfying

$$
\Pr\left(
\forall \lambda\in\Lambda:
R(\lambda)\le U_n(\lambda)
\right)
\ge 1-\beta.
$$

Choose

$$
\widehat\lambda \in \arg\min_{\lambda:U_n(\lambda)\le\alpha} \widehat C(\lambda).
$$

Then

$$
\Pr\left(R(\widehat\lambda)\le\alpha\right)\ge 1-\beta.
$$

Moreover, for each $x$,

$$
f_x(S) = \Pr\left(\max_{b\in S}Z_b=1\mid X=x,\mathcal B\right)
$$

is normalized, monotone, and submodular, and

$$
\mathbb E[L(S,Z)\mid X=x,\mathcal B] = f_x(\mathcal B)-f_x(S).
$$

If, conditional on $X=x,\mathcal B$, the $Z_b$ are independent Bernoulli variables with probabilities $r_b(x)$ and all branch costs are equal, then among subsets of cardinality $k$, the risk-minimizing set consists of the $k$ largest $r_b(x)$.

### Novelty status of this theorem

- The simultaneous risk-selection part is a standard learn-then-test/conformal-risk-control adaptation.
- The submodularity fact follows because union-of-success events form a coverage function.
- The top-$k$ result under conditional independence is elementary.

Do not sell these components as new generic theory. The contribution is the branch-survival decision problem, its operationalization, and—if achieved—the adaptive tree extension.

### Proof sketch

For fixed $Z$, the function

$$
g_Z(S)=\mathbf 1\{S\cap \{b:Z_b=1\}\neq\emptyset\}
$$

is a set-coverage function, hence monotone and submodular. Conditional expectation preserves both properties, giving the result for $f_x$.

Because the event "some retained branch succeeds" is a subset of "some full-bank branch succeeds,"

$$
\Pr(\text{full succeeds, retained fails}\mid x) = f_x(\mathcal B)-f_x(S).
$$

Under conditional independence,

$$
f_x(S) = 1-\prod_{b\in S}(1-r_b(x)).
$$

For fixed cardinality, this is maximized by the largest $r_b$, equivalently the largest $-\log(1-r_b)$.

For risk calibration, simultaneous validity of $U_n(\lambda)$ implies that every policy declared feasible has true risk at most $\alpha$ on the simultaneous-validity event. Data-dependent selection among those policies therefore preserves the bound. Exact binomial bounds with Bonferroni correction are sufficient for a first theorem; sharper conformal-risk-control bounds can replace them.

For predetermined checkpoints $t=1,\ldots,T$, if each catastrophic-pruning event has probability at most $\alpha_t$, a standard union bound gives total path-loss probability at most $\sum_t\alpha_t$. That composition result alone is not novel or sufficient for adaptive trees.

### Hardest unresolved proof obligation

The real system generates and visits nodes adaptively. That creates three coupled failures:

1. **Policy-induced covariate shift**: later checkpoint states depend on earlier pruning.
2. **Censored counterfactual outcomes**: $Z_b$ is not observed for branches that were not continued.
3. **Adaptive multiplicity**: the number and identity of pruning decisions are random.

A publishable flagship result needs one of:

- a randomized logging policy plus off-policy risk estimation;
- a full-tree offline calibration corpus with a clean argument for deployment transfer;
- always-valid confidence sequences/e-processes for adaptive branch decisions;
- a conservative sequential guarantee using predictable risk budgets and valid conditional bounds.

Without this, the formal guarantee applies only to frozen branch banks at fixed checkpoints. That version is valid but probably insufficient for a top-tier main-track paper given the 2026 competition.

---

## 6. Minimum experiment matrix for a top-conference submission

### Models

At minimum:

- two independent model families;
- one 4–8B scale and one 12–32B scale;
- frozen revisions and decoding implementations;
- at least one model with a usable process/value signal and one without it.

A Qwen-family pair plus a Gemma/Llama-family pair would be adequate.

### Tasks

Three distinct verification regimes:

1. **Math**: exact final-answer checking; MATH/AIME-style.
2. **Code**: hidden or held-out unit tests.
3. **Weak-verifier reasoning**: science or multi-hop QA with an imperfect judge/PRM.

The third regime is necessary to expose whether gains depend entirely on perfect outcome verification.

### Branch construction

- widths $B\in\{4,8,16\}$;
- early, middle, and late checkpoints;
- independent rollout bank and local lookahead tree;
- continuation budgets of at least two lengths;
- frozen proposal temperatures and seeds.

### Required baselines

- full branch bank;
- random pruning;
- sequence likelihood/margin/entropy top-$k$;
- PRM or value top-$k$;
- semantic-diversity selection;
- standard self-consistency;
- adaptive self-consistency/ReASC;
- weighted token set cover;
- REFRAIN or equivalent early stopping;
- dynamic value-threshold abstention;
- the July 2026 recall-controlled cascade where applicable;
- ORCA-style calibrated sampling;
- Local Branch Routing as a learned-routing reference, if reproducible. (ACL Anthology)

### Feature ablations

1. logits only;
2. logits plus entropy/rank;
3. plus semantic diversity;
4. plus PRM/value;
5. plus hidden-state probe;
6. plus model-internal unigram and morphology;
7. plus external frequency;
8. plus residualized/hierarchical external frequency.

The frequency contribution is the difference between 6 and 7/8—not between raw margin and frequency.

### Metrics

**Primary**:

- catastrophic-pruning risk;
- final task accuracy;
- target-model FLOPs or forward passes;
- generated tokens;
- wall latency at controlled batch/serving conditions;
- peak KV memory;
- risk–compute Pareto frontier.

**Secondary**:

- conditional branch survival among full-bank-solvable examples;
- calibration curves;
- risk under domain shift;
- selector error given that a good branch survives;
- branch diversity and dependence.

It is important to separate:

$$
\text{branch availability failure}
\quad\text{from}\quad
\text{final selector/verifier failure}.
$$

Otherwise the method can preserve a good branch without ever choosing it.

### Splits and evidence

- Tune, calibration, and test split by problem/source, not generated trajectory.
- Full counterfactual branch outcomes on the calibration and a designated oracle-test subset.
- A second deployment-like test in which only selected branches are evaluated.
- Frozen policy family before calibration.
- Cluster bootstrap by problem.
- Multiple search-policy seeds.
- Explicit [G] and [E] labeling can be retained from the current repository.

### Approximate compute

A credible full submission likely requires:

- 50–120 million generated target-model tokens;
- roughly 150–400 A100-80GB-equivalent hours;
- additional engineering for cached KV branching and wall-clock benchmarking.

These are planning estimates, not hardware-independent forecasts.

---

## 7. Explicit kill criteria

The project should stop or downgrade if any of the following occurs.

1. **Calibration failure**: fixed-checkpoint catastrophic-pruning risk exceeds $\alpha$ outside the preregistered tolerance on any clean in-distribution test cell.
2. **No material compute gain**: less than 20% target-token/FLOP reduction or less than 10–15% wall-latency reduction at at most 0.5 percentage-point accuracy loss.
3. **Baseline collapse**: gains disappear against PRM top-$k$, dynamic abstention, recall-controlled abort, ORCA, or weighted set cover.
4. **No multi-cell replication**: benefit does not replicate on at least two model families and two task types.
5. **Counterfactual impracticality**: obtaining branch-success labels costs more than the method can plausibly save, with no path to randomized logging or weak supervision.
6. **Sequential-theory failure**: the guarantee cannot be extended beyond frozen checkpoints without unrealistic full-tree enumeration. In this case, submit only the fixed-checkpoint measurement paper.
7. **Systems failure**: token savings do not translate into wall-clock or memory improvements under a realistic batched serving implementation.
8. **Frequency-specific kill**: after controlling for model unigram, morphology, tokenizer, entropy, rank, and token identity, external frequency contributes less than approximately 1–2% relative compute improvement and has no stable predictive lift. Drop frequency immediately; do not kill the branch framework solely for this reason.

---

## 8. How much of the repository can be reused

### High-value reuse

The following concepts and components remain useful:

- fail-closed protocol;
- immutable artifacts and hashes;
- model and tokenizer revision binding;
- deterministic split manifests;
- cross-corpus duplicate checks;
- tune/calibration/test role separation;
- clustered bootstrap;
- machine-readable gates and decision memos;
- [G] versus [E] evidence distinction;
- shared model-generation and sampler plumbing.

### Moderate reuse

- conformal quantile and risk-bound utilities;
- Mondrian/group calibration abstractions;
- method registry design;
- gate-evidence schemas, after changing the data unit from token position to problem/checkpoint/branch bank;
- frequency tables as one feature source;
- document/problem clustered inference.

### Low or no reuse

- the ν score and its name;
- current $h(m,n)$ as the central scientific object;
- next-token mean support size as the main objective;
- current prediction-set suffstat schema;
- all legacy generation results;
- Phase-0 evidence as method evidence.

My estimate:

- **Direct code reuse**: approximately 20–35%.
- **Protocol and research-engineering reuse**: approximately 60–70%.
- **Reusable empirical evidence**: effectively 0% for the new method; the Phase-0 pilot is only motivation for stringent frequency controls.

The repository should be forked conceptually rather than incrementally extended. Keeping the same paper directory and method naming would create anchoring pressure.

---

## 9. Realistic assessment of the recommended direction

### Novelty

- Problem/object novelty: strong.
- Generic theoretical novelty: modest.
- Potential sequential-theory novelty: meaningful, if adaptive censoring and risk composition are genuinely solved.
- Empirical novelty: strong if the project releases counterfactual branch-survival data and demonstrates real systems savings.

The latest literature makes the novelty window narrow. Dynamic abstention is already at ICML 2026, ORCA calibrates reasoning sampling, Local Branch Routing performs learned subtree selection, and the July 7 recall-controlled cascade controls success retention across episode gates. The paper must therefore be about multi-branch set selection and global path survival, not "calibrated early stopping." (arXiv)

### Feasibility

- Fixed-checkpoint pilot: good.
- Full branch-bank paper: moderate.
- Adaptive-tree theory and deployment: difficult.
- Dependence on external frequency: unnecessary and currently doubtful.

### Top-conference potential

A strong result could be competitive if it contains all three:

1. a clean downstream risk guarantee;
2. clear superiority over the 2026 adaptive-compute baselines;
3. substantial wall-clock or FLOP savings on multiple tasks and model families.

A paper containing only fixed-checkpoint CRC plus one math model would likely be a Findings/workshop paper.

My qualitative assessment:

- current static frequency-offset paper: very low top-conference probability;
- branch-survival paper with only the easy theorem: low to moderate;
- branch-survival paper with adaptive-tree treatment and real systems gains: credible top-conference candidate.

---

## 10. Lower-risk fallback

### Working title

**Does External Frequency Survive Its Confounders? A Multi-Tokenizer Audit of Language-Model Tail Reliability**

### Thesis

External corpus frequency should be credited only for predictive information that remains after conditioning on confidence, model-internal lexical priors, morphology, token identity, and tokenizer structure.

### Core study

For each candidate token or canonical event, estimate incremental out-of-sample value in stages:

$$
\begin{aligned}
\mathcal F_0 &= (m,\operatorname{rank},p,H,\sigma_{\text{logit}}),\\
\mathcal F_1 &= \mathcal F_0+\text{morphology/bytes/script/whitespace},\\
\mathcal F_2 &= \mathcal F_1+\text{model-internal unigram/head features},\\
\mathcal F_3 &= \mathcal F_2+\text{token/embedding cluster},\\
\mathcal F_4 &= \mathcal F_3+\text{external frequency}.
\end{aligned}
$$

Evaluate whether $\mathcal F_4$ improves:

- cross-fitted candidate log loss or deviance;
- conditional mutual-information estimates with uncertainty;
- calibrated coverage–size frontiers;
- frequency-bucket coverage;
- continuation or branch-value prediction.

Use conditional randomization controls that permute external frequency only within strata defined by internal unigram, morphology, token length, script, and embedding cluster. A single unrestricted vocabulary permutation is not sufficient.

### Required negative-result discipline

A null paper needs:

- equivalence bounds, not merely nonsignificant $p$-values;
- a detectable-effect floor;
- same-text cross-tokenizer analyses;
- multiple model families and domains;
- fixed external corpora;
- no downstream sampler claim unless a frontier improvement survives all controls.

### Compute and reuse

- 3 model scales/families × web/math/code;
- approximately 10–30 A100 hours if logits can be efficiently cached;
- substantially less for analyses using the preserved Qwen pilot data, though new internal-unigram and morphology artifacts will be required;
- roughly 60–75% of the current protocol/code concepts are reusable.

### Publication potential

A well-powered finding that external frequency is fully absorbed by the model's internal unigram and tokenizer morphology would be a credible:

- ACL/EMNLP Findings paper;
- uncertainty/reliable-ML workshop paper;
- thesis chapter;
- public audit benchmark.

A replicated positive residual effect would give a much better foundation for a later method than the current Phase-0 result.

---

## Final recommendation

Do not restart the current ν/frequency-offset method paper. The pilot did not merely fail a gate; the original rule's sign is disfavored, the strongest effects are boundary-censored, the web contrasts are unstable, the control is weak, and the target risk is too detached from generation utility and production cost.

The underlying insight can still survive in a broader form:

> Side information is useful only when it changes a calibrated, downstream decision—not because it correlates with next-token correctness.

The ambitious program should therefore make branch viability, global path survival, and compute allocation the scientific center. Frequency becomes one auditable feature and is removed without ceremony if model-internal priors or morphology absorb it.

The fallback should be the confounder-complete frequency audit. That is substantially safer, scientifically honest, and much more likely to produce a durable contribution than finishing the current static-threshold roadmap.
