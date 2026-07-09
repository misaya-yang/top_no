# ICML 2027 Research State — 2026-07-10

## Executive state

Current `main`: `c89884f8ae04817539116d48ae625b6ea8bec90f`.

The repository now has an audited, fail-closed foundation for the active paper
framing, **Frequency-Offset Margin Rules for Calibrated Language Model
Decoding**. It does not yet have a paper-grade result. All existing legacy
prediction-set numbers remain link/smoke evidence only.

The central falsifiable question is unchanged:

```text
At fixed logit margin, does pinned token frequency carry enough additional
information to improve calibrated coverage/size behavior?
```

Coverage itself is standard split-conformal machinery. The paper's novelty
budget remains the frequency diagnostic, the offset/Mondrian/learned family,
and any replicated efficiency or conditional-coverage finding.

## Completed protocol rounds

1. **PR-1a..1d — provenance and split validity**
   - immutable frequency artifacts;
   - exact document text binding;
   - deterministic cluster-level tune/cal/test splits;
   - one-position-per-document `[G]` selection;
   - threshold-complete cross-corpus near-duplicate receipt.
2. **PR-1e — EOS-aware pinned frequency builder** (`b7cda35`)
   - fixed raw-text tokenization policy;
   - special-token filtering plus one synthetic EOS per document;
   - pinned offline Hub repo/revision validation;
   - real Qwen2.5-7B tokenizer artifact smoke.
3. **PR-2a — conformal core** (`46d497b`)
   - exact finite-sample rank, including required `+inf`;
   - explicit score dithering and APS boundary uniforms;
   - C-margin/C-nu/APS primitives and equivalence tests;
   - auditable Mondrian thresholds/counts/reasons.
4. **PR-2b — canonical method registry** (`f2ad7e8`)
   - stable method keys and implementation status;
   - one tensor-only calibration/prediction contract;
   - paper execution reports exact missing mandatory keys.
5. **PR-2c — per-position gate evidence** (`ab573b0`)
   - content-addressed `[G]`/`[E]` artifacts;
   - per-document/cluster rows for paired bootstrap;
   - calibration, PR-1, config, tuning, randomization, and test-row binding;
   - strict finite-sample rank/reason reconstruction and registry-content hash.
6. **PR-2d — bucket separation and preregistration** (`b254a69`)
   - exact diagnostic B0..B8 log-count bands;
   - separate method-side true-token-mass bucket kind;
   - v1 method policy freezes K0=8, no split of equal-frequency ties,
     unseen handling, strictest-delta floor 500, deterministic merging, and
     fail-closed minimum bucket count.
7. **PR-2e/2f — mandatory calibrated baselines** (`7c8e86f`, `c89884f`)
   - stable C-logprob with chunked target-only log-normalizer;
   - stable C-zmargin with shared chunked sample-std statistics;
   - exact epsilon-set and top-nsigma equivalences at zero dither.

Every major round received an independent Critical/Important review. Review
counterexamples were converted into regression tests before merge.

## Verification snapshot

Local main:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python3 -m unittest discover tests                        183/183 PASS
git diff --check                                          PASS
```

RTX 5090 server:

```text
python -m unittest discover tests                         183 run, 182 PASS,
                                                         1 MPS-only skip
```

Real-width CUDA probes used `V=152064` and checked all implemented registry
paths. C-logprob preserved `log(V)` for a constant fp16 row at offset 60,000.
C-zmargin target/full scores were bitwise identical across the full vocabulary.
These are numerical/shape tests, not scientific results.

## Server and model state

At the final audit:

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; idle, 29 C;
- scratch disk: 50 GB total, 26 GB used, 25 GB available;
- no experiment Python process active;
- Qwen2.5-3B cached at revision
  `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` (about 5.8 GB);
- Qwen2.5-7B cached at revision
  `d149729398750b98c0af14eb82c78cfe92750796` (about 15 GB);
- GPT-2 cached at revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

The existing Qwen2.5-7B legacy local-text smoke is explicitly noncitable. Its
large-support behavior is useful only as a failure-mode diagnostic.

## Remaining hard blockers

Paper-grade prediction-set execution must stay blocked until all of the
following are complete:

1. Mandatory registry methods: `raps`, `ts_aps`, `cns`, and `learned_h`.
2. A method-bucket builder that binds exact `D_tune` target rows, the pinned
   `D_freq` frequency artifact, and the committed policy hash.
3. A sharded suffstats writer/replayer with direct/replay agreement tests.
4. Main-runner integration using independently bound calibration/test
   documents and `[G]` forwards; no sequential legacy skip/take path.
5. Per-method tuning artifacts built from `D_tune` only.
6. PR-3 paired document-cluster bootstrap, frozen gate thresholds, mandatory
   comparator completeness, and `PASS / G2-only / FAIL / INELIGIBLE` handling.
7. A production `D_freq` corpus/table. The two-document frequency artifact is
   only a functional smoke; Fable5's target is corpus scale.

## Strict next sequence

1. Implement the bound method-bucket builder and artifact tests.
2. Implement suffstats schema/writer/replay before another large model pass.
3. Add RAPS, tune-only temperature scaling plus APS, CNS, and learned-h.
4. Wire the `[G]` calibration/test forwards to emit gate evidence v2.
5. Implement and freeze PR-3; keep downstream generation locked.
6. Build a production frequency table and run Phase 0 first.
7. Only if Phase 0 is stable across tune halves, run the calibrated Pareto
   matrix. Only if G1 or G2 passes, spend GPU budget on downstream generation.

This ordering protects both possible papers: a frequency-method paper if the
effect is real and exploitable, or a broad conformal audit if margin is already
nearly sufficient.

