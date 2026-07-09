# PR-2a Conformal Core Design

## Scope

This slice implements only mathematical primitives whose contracts can be
closed independently of the future methods registry, tuning artifacts, and
sufficient-statistic schema:

- exact finite-sample split-conformal quantiles;
- groupwise/Mondrian quantiles with auditable vacuous fallbacks;
- explicit score dithering;
- C-margin score helpers;
- APS scores with explicit total order and boundary uniforms;
- equivalence tests against existing min-p, top-p, and C-nu behavior.

It does not wire these methods into the paper runner. Nonlegacy execution
continues to stop at `blocked_pending_pr2_pr3`.

## Finite-sample quantile

For `n` calibration scores and miscoverage `delta`, define

```text
k = ceil((n + 1) * (1 - delta)).
```

If `k <= n`, the threshold is the exact `k`-th order statistic with no
interpolation. If `k = n + 1`, the threshold is positive infinity. Clamping the
rank to `n` is invalid: for example, `n=4, delta=0.1` would return the
calibration maximum and cover only 4/5 exchangeable ranks instead of at least
0.9. Positive infinity retains the complete vocabulary and is valid but
vacuous.

NaN scores are rejected. Infinite score values remain legal because they are
also the explicit conservative threshold representation.

## Mondrian calibration

`mondrian_quantiles` returns one immutable `GroupQuantile` per sorted group ID,
including its calibration count, threshold, finite flag, and reason. Missing
expected groups and groups below the paper floor receive positive infinity.
The default floor is `ceil(5 / delta)`; callers may lower it for diagnostics,
but the base `n+1` rank rule still applies.

The core never merges buckets. Merge direction requires ordered bucket edges,
a policy fitted on `D_tune`, and a frozen artifact; group labels alone are
insufficient to make that decision safely.

## Randomization contracts

`dither_scores(A, U)` computes `A + 1e-6 U` in float64 for explicit
`U in [0,1)`. The helper does not draw from global RNG. The later runner must
derive frozen uniforms from stable document/position/token identities and apply
the same randomized score definition to calibration targets and test
candidates. Dithering calibration alone would break exchangeability. MPS is
rejected with a controlled error because it cannot represent float64; callers
must explicitly move cached scores and uniforms to CPU rather than silently
losing the dither in float32.

APS has its own boundary randomization and does not silently add score dither:

```text
prefix_i = sum_{j before i in order} p_j
A_APS(i; u_i) = prefix_i + u_i * p_i.
```

Both the descending token permutation and uniforms are explicit inputs. The
provided default order helper uses probability/logit descending order with
stable token-ID ties. With all `u_i=0`, `A_APS <= q` is exactly the repository's
deterministic top-p rule including the crossing token. General uniforms define
randomized-boundary APS and are not claimed equivalent to deterministic top-p.

## Algebraic equivalences

- C-margin keeps `s_max - s_i <= q`, exactly min-p with
  `p_min = exp(-q)`. The shared min-p sampler evaluates this relation directly
  in logit space so fp16 probability underflow cannot admit zero-probability
  tail tokens.
- C-nu at `kappa=0` is exactly C-margin.
- Deterministic APS (`u=0`) is crossing-token top-p under one shared explicit
  order.

These are implementation invariants, not novelty claims.

## Deferred work

Learned-h/g, TS+APS, RAPS, CNS, frequency/entropy bucket construction,
automatic bucket merging, suffstats/replay, evaluator dispatch, gate logic, and
default runner randomization remain separate PR-2b/PR-3 work. A premature
registry would freeze interfaces before those data and randomness contracts are
specified.
