# AGENTS.md

## Project Context

This repository supports an ICML 2027 paper draft, "Truncation Sampling as
Hypothesis Testing". Treat experiment code, figure names, JSON result names, and
CLI argument shapes as part of the paper workflow unless a task explicitly asks
to change them.

## Repository Layout

- `experiments/`: experiment scripts and shared helpers. Keep cross-experiment
  utilities here unless they become a reusable package.
- `experiments/samplers.py`: shared sampler and batch generation implementation
  for downstream decoding experiments.
- `tests/`: lightweight correctness checks for sampler behavior.
- `scripts/`: shell entrypoints for running experiment suites from the repository
  root.
- `results/`: generated figures, JSON results, and large transient checkpoints.
  Do not hand-edit result artifacts when a script should regenerate them.
- `docs/reports/`: experiment reports and paper-facing notes.

## Working Rules

- Read relevant files before editing and match the existing experiment style.
- Make the smallest change that solves the task. Avoid drive-by refactors,
  renames, or formatting-only rewrites.
- Preserve existing public surfaces: CLI flags, output filenames, result schema,
  and figure names.
- Treat `results/` artifacts as legacy unless they were regenerated after the
  latest sampler/generation changes.
- Do not add dependencies unless the task requires it; if adding one, explain why
  and update `requirements.txt`.
- Keep secrets, API keys, tokens, and local credentials out of files, logs, and
  responses.

## Running Experiments

- Main suite: `bash scripts/run_all.sh`
- Supplementary suite: `bash scripts/run_supplementary.sh`
- Single experiment:

```bash
PYTHONPATH="$PWD/experiments" python3 experiments/<experiment>.py --output-dir ./results
```

Most full experiments are GPU/model-cache dependent. The suite scripts default
to offline Hugging Face mode for the current GPU server workflow.

## Local Environment

- Preferred local Torch smoke-test environment:
  `/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python`
- Verified on 2026-07-08: Python 3.12.12, torch 2.9.1, CUDA unavailable, MPS
  available.
- Use this environment for lightweight sampler tests on this Mac:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python -m unittest discover tests
```

The full model experiments still require the expected model cache and enough
GPU/MPS memory; do not infer paper numbers from smoke tests.

## Verification

Before calling work done, run the strongest check that fits the change:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover tests
```

For behavior changes, also run the smallest relevant experiment or a reduced
sample-size command and report exactly what was run.
