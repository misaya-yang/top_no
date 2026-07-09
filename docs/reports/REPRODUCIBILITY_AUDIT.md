# Reproducibility Audit Notes

## 2026-07-08 Sampler And Generation Fixes

This audit applies the highest-priority implementation fixes from the repository
review. The goal is to make future reruns defensible before any ICML-facing
claim is made from the downstream decoding experiments.

Implemented:

- Shared sampler implementation in `experiments/samplers.py`.
- Standard nucleus sampling: top-p now keeps the crossing token in the minimal
  cumulative-mass set.
- Raw-logit truncation before temperature scaling for logit-space rules.
- Left-padded batch generation so `outputs.logits[:, -1, :]` corresponds to the
  real prompt boundary for every row.
- EOS-aware stopping and generated-token-only metric inputs.
- No silent fallback to untruncated softmax; invalid truncated distributions now
  raise an error.
- Matched random seed reset for Exp4C creative generation across strategies.
- Unit tests for top-p crossing-token behavior and nu margin direction.

Interpretation correction:

The current nu rule is:

```text
keep(i) iff s_max - s_i <= m0 + kappa / sqrt(n_i + 1)
```

This margin is larger for lower-frequency tokens. Therefore the current rule is
an uncertainty-aware conservative retention rule for low-frequency tokens, not a
rare-token penalty. Domain frequency updates such as `max(general, math)` make
domain-frequent tokens more estimated and contract their uncertainty margin; they
do not "rescue" low-frequency tokens by widening the margin.

Legacy result status:

All JSON and figure artifacts currently under `results/` were generated before
these sampler and generation fixes. They should be treated as historical
debugging artifacts until Exp4C, Exp6, and Exp7 are rerun.

Remaining high-priority work:

- Rerun downstream experiments after these fixes.
- Replace synthetic GSM8K fallback claims with real benchmark runs or clearly
  label them as toy arithmetic fallback.
- Add validation sweeps for top-p, top-nsigma, min-p, typical, eta/epsilon, and
  nu variants before test-set reporting.
- Add quality metrics beyond Distinct-n for open-ended generation.
- Strengthen channel evidence beyond synthetic-channel estimator recovery.

## 2026-07-09 Repo Reconciliation Addendum

Fable5's pushed-`main` reconciliation upgraded two earlier risks to confirmed
protocol defects:

- `experiments/eval_prediction_sets.py` builds token-frequency counts from the
  same loaded text pool later used for calibration/evaluation. This leaks
  evaluation text into the frequency side information and invalidates any
  conformal-validity claim from that runner.
- The runner consumes calibration positions first and evaluation positions next
  from the same shuffled stream. This is a sequential split rather than a
  document-level exchangeable split.

Until these are fixed, committed prediction-set outputs are link tests only.
The runner now refuses paper-grade execution unless `allow_legacy_protocol=true`
is explicitly set for smoke tests.
