# Related Work Positioning

Use decoding rules as support-construction baselines:

- `top-k`: cardinality constraint.
- `top-p`: cumulative probability-mass constraint.
- `typical`: local information-rate criterion.
- `Mirostat`: sequence-level perplexity control.
- `eta` / desmoothing: probability-space support recovery.
- `min-p`: fixed logit-margin special case.
- `top-nsigma`: context-global logit-margin special case.
- `nu`: token-wise frequency-indexed logit-margin score; calibration is a separate step.

## Novelty Boundary

Claims already occupied by prior work:

- Conformal top-p and entropy-conditioned next-token prediction sets:
  Conformal Nucleus Sampling (Ravfogel et al., Findings ACL 2023).
- Non-exchangeable, retrieval-weighted token-level conformal generation: Ulmer
  et al. (Findings EACL 2024).
- Sequence/response-level conformal generation: Quach et al. (ICLR 2024).
- Next-token conformal coverage/efficiency as a topic: VACP
  (`arXiv:2512.22682`, preprint; its masking and tuning validity require
  independent reproduction before use as evidence).
- Token-clustered conditional conformal prediction: Ding et al. (NeurIPS 2023)
  supplies the generic clustering baseline; frequency-Mondrian is not a new
  conformal theorem.

The defensible novelty target is narrower: to our knowledge, prior work has not
systematically measured whether **external-corpus token frequency adds
information after conditioning on logit margin**, nor mapped the resulting
coverage-size frontier across model scale, tokenizer, and domain. Learned-`h`
is an oracle-approximation tool for that question, not a new generic
set-classification theorem.

## Positioning Sentence

```text
Rather than proposing another heuristic truncation threshold, we evaluate truncation as prediction-set construction: the central object is the retained token set, measured by true-token coverage and support efficiency before any downstream generation metric is claimed.
```

## Baseline Priority

The first table should compare calibrated methods against calibrated methods.
Uncalibrated truncation rules are useful diagnostics, but not enough for the
paper claim:

- C-margin, C-logprob, and C-zmargin.
- APS and RAPS.
- TS+APS with temperature selected on `D_tune` only.
- CNS and entropy-Mondrian margin as context-conditioning controls.
- Frequency-Mondrian margin.
- Ding-style score-distribution/token-clustered conformal prediction as a
  generic token-identity control.
- General learned-`h(m,n)` and additive learned-`g`, reported separately.
- Signed C-nu and a log-frequency offset as parametric ablations.
- VACP reproduction with mask and temperature frozen before `D_cal`.
- Uncalibrated top-p/min-p/top-k/top-nsigma/eta/typical sweeps for context only.

Do not lead with Distinct-n. It is a downstream surface-diversity metric, not the core claim. Do not present conformal coverage alone as evidence for frequency-aware decoding; coverage is generic once a valid split-conformal score is chosen.

## Verified Primary Sources

- Ravfogel, Goldberg, and Goldberger (2023), [Conformal Nucleus Sampling](https://aclanthology.org/2023.findings-acl.3/).
- Ulmer et al. (2024), [Non-Exchangeable Conformal Language Generation](https://aclanthology.org/2024.findings-eacl.129/).
- Quach et al. (2024), [Conformal Language Modeling](https://openreview.net/forum?id=pzUhfQ74c5).
- Ding et al. (2023), [Class-conditional conformal prediction with clustered classes](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html).
- Kotla and Kotla (2025), [Conformal Prediction Sets for Next-Token Prediction in Large Language Models](https://arxiv.org/abs/2512.22682) (preprint; treat reported efficiency and validity as unverified until independently reproduced).
- Sadinle, Lei, and Wasserman (2019), [Least Ambiguous Set-Valued Classifiers With Bounded Error Levels](https://doi.org/10.1080/01621459.2017.1395341).
