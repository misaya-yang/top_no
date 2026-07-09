# PR-2b Calibrated Method Registry Validation

Validation target: `codex/pr2b-margin-registry`.

## Scope and stop condition

This slice introduces canonical identities for the complete planned method
matrix and tensor-only calibration/prediction adapters for five methods:

- `c_margin`;
- `aps`;
- signed `c_nu` with explicit `kappa` and `alpha`;
- `frequency_mondrian_margin` with candidate-token (`V`) groups;
- `entropy_mondrian_margin` with context-row (`N`) groups.

The other mandatory paper baselines remain registered as unavailable. Protocol
validation reports their exact keys and still raises `blocked_pending_pr2_pr3`.
This slice therefore cannot launch or label a run as paper-grade.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
git diff --check                                          PASS
python3 -m unittest discover tests                        161/161 PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         161 run, 160 PASS,
                                                         1 MPS-only skip
```

The ten new method tests cover canonical registry membership, explicit
implementation status, unknown/unavailable method rejection, C-margin/min-p
equivalence, C-nu/C-margin equality at `kappa=0`, signed-kappa isolation, APS
and deterministic top-p equivalence, finite-sample `+inf`, both Mondrian group
axes, vacuous absent groups, and malformed randomness/group inputs.

## RTX 5090 real-vocabulary tensor smoke

A synthetic-logit smoke used Qwen2.5-7B's real vocabulary width (`V=152064`)
on CUDA. It calibrated all five implemented registry methods from 19 rows and
constructed two test masks. Every mask had shape `(2, 152064)` and completed
without loading model weights. Example mean set sizes were 147,944 for
C-margin, 147,722.5 for signed C-nu, 148,154 for APS, 149,685 for frequency
Mondrian, and 149,836.5 for entropy Mondrian.

The large supports are expected from only 19 synthetic calibration rows and
are not empirical findings. This is strictly a device, vocabulary-shape, and
finite-sample execution smoke; it is not paper evidence.

Independent review found no Critical/Important issues. It separately checked
calibration/test dither symmetry, APS boundary-only randomization, the `V` vs
`N` Mondrian axes, unknown/vacuous groups, finite-sample `+inf`, and agreement
between the mandatory registry and the frozen Fable5 method matrix.
