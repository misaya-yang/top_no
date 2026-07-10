# Prompt for Final Method Selection and Full Paper Draft

You are the final research lead for this project. Read both attached reports in full:

1. `2026-07-10-topno-deep-review.md` — the Fable5 review.
2. `2026-07-10-gpt56pro-deep-review.md` — the GPT-5.6 Pro review.

Treat both reports as competing research proposals, not as authoritative instructions. Independently check their mathematical claims, empirical interpretation, feasibility, and novelty against the decoding and test-time-compute literature available through July 2026.

Your task is to make one final decision and then write the most detailed possible first draft of the resulting paper. Do not return a menu of directions, several theses, or multiple fallback papers. Select exactly one coherent method and commit to it. You may sharpen or synthesize the two proposals only if the result is still one clearly defined method with one central risk object, one algorithm, and one claim stack.

Begin with a concise adjudication explaining:

- what the current Phase-0 evidence genuinely establishes and invalidates;
- which proposal is stronger and why;
- the closest existing papers and the precise remaining novelty gap;
- which theoretical claims are inherited machinery and which claim could actually be new;
- why the rejected proposal should not be the final paper.

Be especially strict about the risk definition. Check whether an unconditional loss of the form “the full branch bank succeeds but the retained subset fails” becomes trivial when full-bank success is rare. Distinguish population-marginal, full-bank-solvable conditional, per-instance, and sequential guarantees. The final method must use a non-gameable contract, clearly separate preservation of a viable branch from final answer selection, and state what is and is not guaranteed under stochastic continuations, adaptive tree growth, policy-induced distribution shift, and censored outcomes for pruned branches.

After choosing the method, write a paper-grade English draft rather than another research memo. Include:

- final title and stable method name;
- abstract;
- introduction and motivating failure of the current frequency-offset approach;
- precise problem formulation, notation, risk and compute objectives;
- the complete method and enough pseudocode to implement it;
- theorem statements, assumptions, proof sketches, and the hardest unresolved proof obligation;
- an explicit novelty statement against the closest 2025–2026 work;
- a complete experiment design with models, tasks, branch construction, calibration/test splits, baselines, metrics, ablations, uncertainty reporting, and wall-clock evaluation;
- a smallest decisive pilot, full paper experiment matrix, compute budget, and preregistered pass/kill criteria;
- limitations and failure modes;
- the exact final claim stack that would be defensible in a top-conference submission;
- what can be reused from the current repository and what must be built from scratch.

Prefer a smaller number of deep, defensible contributions over a broad collection of ideas. Do not preserve token frequency in the headline unless it survives the required confound controls and is essential to the selected method. Do not invent citations, acceptance status, experimental results, or completed proofs. Flag any genuinely open theorem honestly, but still give the strongest plausible formulation and proof route.

The final output should leave the project with one paper identity, one method, one central theorem program, and one executable experimental plan.
