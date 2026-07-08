# Figure Plan

## Mainline Figure 1: Coverage vs Support Size

Source:

```bash
python experiments/eval_prediction_sets.py --config configs/prediction_sets_qwen3b.json
python experiments/plot_prediction_sets.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Output:

- `results/prediction_sets_qwen3b_wikitext/prediction_set_coverage_efficiency.png`

Claim it supports:

> Frequency-calibrated prediction sets should improve the coverage/efficiency tradeoff over fixed support or fixed probability-mass truncation.

## Mainline Figure 2: Coverage by Token Frequency

Output:

- `results/prediction_sets_qwen3b_wikitext/prediction_set_bucket_coverage.png`

Claim it supports:

> The nu margin targets low-frequency false exclusions; this must show up as frequency-bucket coverage, not just overall coverage.

## Mainline Figure 3: Retained Mass and Support Distribution

Output:

- `results/prediction_sets_qwen3b_wikitext/prediction_set_distribution_summary.png`

Claim it supports:

> Support-size improvements are meaningful only if retained mass and set-size quantiles do not hide pathological tails.

## Decision Gate

Run:

```bash
python experiments/check_prediction_set_gate.py --metrics results/prediction_sets_qwen3b_wikitext/prediction_set_metrics.json
```

Proceed to downstream generation only if the gate passes. If it fails, revise the nonconformity score before spending GPU time on reasoning or open-ended quality.
