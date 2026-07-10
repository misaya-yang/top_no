# Server Run Artifacts

This directory keeps compact, reviewable summaries from remote experiment
runs. Only small JSON evidence needed to reconstruct a decision should be
committed.

Large archives, checkpoints, raw corpora, model caches, and high-frequency GPU
telemetry stay local and are ignored by Git. A server summary is not paper
evidence unless its own metadata explicitly sets `paper_citable=true`.

The `topno_phase0_20260710_7e88b68_summary/` directory is the final Phase-0
pilot snapshot before the project was paused. Its decision is `INSUFFICIENT`
and its evidence grade is `E-pilot`.
