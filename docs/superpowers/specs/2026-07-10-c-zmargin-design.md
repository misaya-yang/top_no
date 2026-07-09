# PR-2f C-zmargin Baseline Design

C-zmargin uses `A(x,i)=(s_max-s_i)/std(s(x))`, with the sample standard
deviation (`V-1` denominator) used by the existing top-nsigma sampler. Thus a
calibrated threshold is exactly calibrated top-nsigma at zero dither.

The implementation centers each row before variance accumulation for shift
stability. Zero-variance and one-token rows receive all-zero scores, retaining
their tied vocabulary. Low-precision inputs accumulate in float32 and float64
is preserved. Target calibration makes two 4096-token passes for mean/variance
and does not materialize a full fp32 `(N,V)` working tensor.

This implements one mandatory context-normalized baseline without changing the
runner, gate, or paper-grade stop condition.

