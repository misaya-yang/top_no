# top_no

Research repository for the ICML 2027 draft:
**Truncation Sampling as Hypothesis Testing**.

## Layout

- `experiments/`: runnable experiment scripts and shared experiment utilities.
- `scripts/`: suite-level shell entrypoints.
- `results/`: generated figures and JSON result artifacts.
- `docs/reports/`: experiment writeups and summary reports.
- `requirements.txt`: Python runtime dependencies.
- `AGENTS.md`: working instructions for repository agents and contributors.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full model-backed experiments expect the relevant Hugging Face model cache to
exist locally. The suite scripts default to offline Hugging Face mode for the GPU
server environment.

## Run

Run the main experiment suite:

```bash
bash scripts/run_all.sh
```

Run supplementary experiments:

```bash
bash scripts/run_supplementary.sh
```

Run one experiment directly from the repository root:

```bash
PYTHONPATH="$PWD/experiments" python3 experiments/exp1_topk_bias.py --output-dir ./results
```

## Lightweight Verification

These checks verify the repository structure and syntax without rerunning the
expensive GPU experiments:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
```
