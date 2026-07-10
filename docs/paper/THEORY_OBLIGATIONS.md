# Theory Obligations and Claim Boundaries

This note is the paper-facing contract for what the current framework proves,
what it inherits from conformal prediction, and what must be established by
data. It uses the active frequency-offset margin framing; the retired
hypothesis-testing/noise-channel story is out of scope.

## 1. Candidate-token probability space

Let `X` be a context, `Y` its observed next token, and `v=|V|`. Independently
draw `I ~ Uniform(V)` and let `mu` be the probability law of `(X,Y,I)`. Write
`R=1{Y=I}` and define raw-logit margin

```text
m_i(X) = max_j s_j(X) - s_i(X).
```

For a possibly randomized selector `a(X,I) in [0,1]`,

```text
expected size = v E_mu[a(X,I)]
coverage      = v E_mu[R a(X,I)] = P(Y in S(X)).
```

Without this measure, phrases such as “probability a token is correct at fixed
margin” are ambiguous and can silently weight tokens or contexts incorrectly.

## 2. Algebraic equivalences (exact, but not novel)

### Lemma A — min-p is a margin rule

At temperature `T>0`,

```text
p_i(T) / p_max(T) = exp(-(m_i/T)).
```

Therefore min-p with ratio `alpha_p` is exactly

```text
m_i <= -T log(alpha_p).
```

The repository constructs supports from raw logits before sampling
temperature, so its current min-p/C-margin equivalence uses `T=1` at support
construction. This is an algebraic identity, not a contribution.

### Lemma B — C-logprob is calibrated epsilon sampling

With `A_logp(X,i)=-log p_i(X)`, the set `A_logp<=q` is exactly

```text
p_i(X) >= exp(-q).
```

Unlike a raw margin threshold, this score depends on the full context
normalizer. It is a mandatory context-sensitive calibrated baseline.

### Lemma C — C-zmargin is calibrated top-nsigma

With sample standard deviation `sigma_s(X)` and the explicit convention that a
zero-variance row has all-zero scores,

```text
A_z(X,i) = m_i(X) / sigma_s(X).
```

Then `A_z<=q` is exactly top-nsigma with `n_sigma=q`. The score is invariant to
positive affine logit transforms `s -> a*s+b`; raw C-margin scales by `a`.

### Tie convention

Any statement that a support rule is purely margin-monotone must say whether
equal-logit tokens are included together. Top-k can split ties unless inclusive
or randomized tie handling is specified; with a fixed order it is rank
measurable. Top-p/APS additionally depend on the context's cumulative
probability-mass profile, plus a stable total order and APS boundary uniform.
None of these is automatically in the pure single-candidate margin feature
class, and keeping the crossing token does not remove equal-logit distinctions.

## 3. Standard split-conformal guarantee

Let a score `A` be fixed using only `D_freq` and `D_tune`. Given `n`
exchangeable calibration scores, define

```text
k = ceil((n+1)(1-delta)).
q_hat = kth smallest calibration score, or +inf when k>n.
S(X) = {i : A(X,i) <= q_hat}.
```

Under exchangeability of calibration/test pairs,

```text
P(Y in S(X)) >= 1-delta.
```

This is inherited machinery and must be cited, not sold as novelty. The score
may be C-margin, C-logprob, C-zmargin, APS/RAPS, a frequency offset, or a learned
function; conformal validity alone does not prefer one.

The explicit dither/boundary uniforms are part of the score definition. Their
construction must be fixed independently of `D_cal`/`D_test` outcomes and must
be applied symmetrically to calibration targets and test candidates. A score
path that differs by dtype, reduction order, or tie rule between calibration
and prediction is not the same conformal procedure.

## 4. Frequency-Mondrian guarantee

Let `b(i)` be a finite token-frequency partition fixed from `D_freq` and
`D_tune`. Calibrate one threshold per true-token bucket:

```text
q_hat_k from {A(X_t,Y_t) : b(Y_t)=k}
S(X) = {i : A(X,i) <= q_hat_{b(i)}}.
```

Let `n_k` be the random number of calibration labels in bucket `k`. Conditional
on the complete calibration membership vector (and therefore on `n_k=r`),
within-bucket exchangeability gives, for every `r`,

```text
P(Y* in S(X*) | b(Y*)=k, n_k=r) >= 1-delta.
```

If the finite-sample rank is unavailable, `q_hat_k=+inf` and conditional
coverage is one. Averaging over random `n_k` gives the advertised bucket
guarantee. The repository's `ceil(5/delta)` floor is a pre-registered variance
safeguard stronger than the minimum rank requirement, not a new theorem.

Diagnostic B0..B8 log-count bands and method true-token-mass buckets are
different artifacts. Substituting the diagnostic bands into Mondrian/CovGap
silently changes the protocol.

## 5. Feature-restricted optimality lens

For feature sigma-field `G=sigma(phi(X,I))`, define

```text
h_G = E_mu[R | G].
```

For normalized size budget `s`, define

```text
V_G(s) = sup {E_mu[R a] : a is G-measurable, 0<=a<=1, E_mu[a]<=s}.
```

A Neyman-Pearson/rearrangement argument implies that an upper level set of
`h_G`, with boundary randomization at atoms, attains this frontier.

Apply this to the nested feature classes

```text
phi_0 = m
phi_1 = (m,n)
phi_2 = (m,n,c(X)).
```

For nested sigma-fields `G0 subset G1`, `V_G1(s)>=V_G0(s)` for every budget.
The complete frontiers agree for all budgets iff
`E[R|G1]=E[R|G0]` `mu`-almost surely. Strict improvement at one chosen budget
needs more than nonconstant conditional variation: a sufficient condition is
that the fine posterior has positive conditional mass on both sides of every
coarse optimal boundary at that budget (the strict hinge/Jensen case), with
appropriate handling of atoms.

Finding `h(m,n)` nonconstant in `n` is therefore not enough to promise a
practical improvement at the relevant operating point. Conversely,
`h_(m,n)=h_m` almost surely only rules out frequency gains inside this
retain/drop feature class. It does not prove margin sufficient relative to
richer context features, nor prove fixed margin/min-p frontier-optimal unless
`h_m` is also nonincreasing in `m`. Any empirical null needs adequate power.

This theorem is a lens for the empirical program, not evidence that frequency
helps a particular model/domain.

## 6. When an additive frequency offset is adequate

The general `(m,n)` oracle thresholds `h(m,n)`. An additive score

```text
A_g(m,n) = m - g(n)
```

represents the full `(m,n)` oracle frontier for every size budget iff, up to
`mu`-null sets, there is a `g` and a nonincreasing `rho` such that

```text
h(m,n) = rho(m-g(n))
```

with boundary randomization allowed at flats/atoms. Equality at one operating
level only requires that one oracle superlevel set equal one `m-g(n)` sublevel
set; it does not establish the full shift representation.

Consequences:

- signed inverse-square-root C-nu is only a parametric ansatz;
- learned-g is an interpretable constrained ablation;
- learned-h is the general frequency-feature method;
- frequency-Mondrian is a piecewise threshold member with the strongest
  per-bucket guarantee.

A population-oracle gap between learned-h and the best additive learned-g would
refute the shift representation. A replicated, held-out significant empirical
win is evidence against it, but finite-sample estimation, regularization, and
tuning error prevent treating one observed win as a proof of failure.

## 7. Evidence needed before each claim

| Claim | Minimum evidence |
|---|---|
| marginal coverage | frozen `[G]` score/config and split-conformal test |
| per-frequency-bucket coverage | frozen `[G]` frequency-Mondrian artifact and every `n_k` |
| frequency helps beyond margin | margin-conditional Phase-0 interaction with clustered uncertainty and replication |
| efficiency improvement | calibrated-vs-calibrated matched-coverage frontier with paired cluster bootstrap |
| downstream generation value | only after the preregistered G1/G2 gate |
| margin is approximately frequency-sufficient within the studied `(m,n)` class/populations | preregistered equivalence bound and detectable-effect floor across model scale/tokenizer/domain |

`[E]` pooled-position rows are empirical diagnostics. They must never be mixed
with `[G]` rows in calibration, bootstrap, or prose guarantees.

## 8. Open proof and writing obligations

Before paper submission:

1. State the candidate-token measure explicitly in the feature-optimality
   theorem and handle randomized boundary selection.
2. Prove the nested-class weak dominance and give sufficient conditions for
   strictness without claiming an iff stronger than the assumptions allow.
3. State the additive-shift representation with exact monotonicity/tie
   conditions.
4. State random `n_k`, unavailable ranks, and `+inf` in the Mondrian
   proposition itself.
5. Keep temperature claims score-specific. If logits and calibration scores
   are both divided by the same `T`, C-margin `q_hat` scales by `1/T` and its
   support is unchanged; a frozen unscaled threshold does not have that
   invariance. C-zmargin itself is invariant. C-logprob/APS generally change
   nonlinearly, and an additive `g` must be scaled or refitted consistently.
6. Cite conformal and Mondrian guarantees as prior machinery. The paper's
   contribution is the frequency-axis measurement, method family, conditional
   audit, and any replicated empirical finding.
