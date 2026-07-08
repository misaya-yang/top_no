# Controlled Channel Evidence Plan

The current paper-safe claim is narrow:

> Frequency is a useful proxy for token-wise logit sensitivity across controlled perturbation channels.

Do not claim that the real LLM noise channel has been identified.

## Implemented Channels

Run:

```bash
python experiments/exp5b_controlled_channels.py --config configs/controlled_channels_qwen3b.json
```

Outputs:

- `results/controlled_channels_qwen3b/controlled_channel_metrics.json`
- `results/controlled_channels_qwen3b/controlled_channel_frequency_sensitivity.png`

Implemented probes:

- `hidden_noise`: add Gaussian noise to input embeddings and measure target-logit absolute change by target-token frequency bucket.
- `dropout`: run a dropout ensemble in training mode and measure target-logit variance by target-token frequency bucket.

## Not Yet Implemented

Quantization residuals should be added only with an explicit dependency decision. The clean version is:

```text
compare fp16/fp32 logits against int8/int4 logits, bucket residuals by target-token frequency, and report whether rare-token residuals are consistently larger.
```

Do not add `bitsandbytes` or quantization-specific dependencies silently.
