# Paper-Grade Protocol Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable external frequency-table artifacts and fail-closed
provenance checks so a conformal threshold can never be reused with silently
rebuilt or mismatched token counts.

**Architecture:** `experiments/splits.py` owns canonical document manifests and
intersection tripwires. `experiments/freq_table.py` owns canonical count-tensor
hashing, artifact I/O, compatibility validation, and upstream-metrics reference
loading. The prediction-set evaluator records provenance but stays
`blocked_pending_pr1b`; downstream generation loads the exact referenced
artifact or refuses to run.

**Tech Stack:** Python 3.9+, PyTorch, standard-library `dataclasses`, `hashlib`,
`json`, `pathlib`, and `unittest`. No new dependency.

## Global Constraints

- Preserve the current CLI flags, output filenames, result schema fields, and
  `build_token_counts()` import surface unless this plan explicitly adds fields.
- Legacy smoke remains available only with `allow_legacy_protocol=true` and is
  always labeled non-citable.
- Paper-grade execution remains blocked until PR-1b, PR-2, and PR-3 complete.
- Protocol validation must happen before model allocation whenever possible.
- Do not modify generated files under `results/`.
- Use TDD for each behavioral change and commit only after the full repository
  verification passes.

---

### Task 1: Canonical document manifests and disjointness tripwire

**Files:**
- Create: `experiments/splits.py`
- Create: `tests/test_protocol_manifests.py`

**Interfaces:**
- Produces: `ManifestDocument`, `DocumentManifest`, `manifest_sha256()`,
  `save_manifest()`, `load_manifest()`, and `assert_pairwise_disjoint()`.
- Consumes: JSON manifest paths from evaluator configuration and the frequency
  artifact's `source_manifest_sha256`.

- [x] **Step 1: Write manifest hashing and intersection tests**

Create `tests/test_protocol_manifests.py` with tests that construct manifests in
temporary directories, verify deterministic hashes under reordered documents,
round-trip through JSON, and reject collisions independently on `doc_id`,
`content_sha256`, and `cluster_id`.

```python
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from splits import (
    DocumentManifest,
    ManifestDocument,
    assert_pairwise_disjoint,
    load_manifest,
    manifest_sha256,
    save_manifest,
)


def manifest(role: str, suffix: str) -> DocumentManifest:
    return DocumentManifest(
        protocol_version="icml2027-pr1a",
        role=role,
        source="fixture",
        documents=(
            ManifestDocument(
                doc_id=f"doc-{suffix}",
                content_sha256=f"content-{suffix}",
                cluster_id=f"cluster-{suffix}",
            ),
        ),
    )


class ProtocolManifestTests(unittest.TestCase):
    def test_hash_is_independent_of_document_order(self):
        first = manifest("cal", "a").documents[0]
        second = manifest("cal", "b").documents[0]
        a = DocumentManifest("icml2027-pr1a", "cal", "fixture", (first, second))
        b = DocumentManifest("icml2027-pr1a", "cal", "fixture", (second, first))
        self.assertEqual(manifest_sha256(a), manifest_sha256(b))

    def test_round_trip_preserves_hash(self):
        original = manifest("freq", "freq")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_manifest(original, Path(tmp) / "freq.json")
            loaded = load_manifest(path)
        self.assertEqual(loaded, original)
        self.assertEqual(manifest_sha256(loaded), manifest_sha256(original))

    def test_disjoint_manifests_pass(self):
        assert_pairwise_disjoint({
            "freq": manifest("freq", "f"),
            "tune": manifest("tune", "u"),
            "cal": manifest("cal", "c"),
            "test": manifest("test", "t"),
        })

    def test_doc_id_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument("doc-same", "other-content", "other-cluster")
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))
        with self.assertRaisesRegex(ValueError, "doc_id"):
            assert_pairwise_disjoint({"freq": left, "test": right})

    def test_content_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument("other-doc", "content-same", "other-cluster")
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))
        with self.assertRaisesRegex(ValueError, "content_sha256"):
            assert_pairwise_disjoint({"freq": left, "test": right})

    def test_cluster_intersection_fails(self):
        left = manifest("freq", "same")
        right_doc = ManifestDocument("other-doc", "other-content", "cluster-same")
        right = DocumentManifest("icml2027-pr1a", "test", "fixture", (right_doc,))
        with self.assertRaisesRegex(ValueError, "cluster_id"):
            assert_pairwise_disjoint({"freq": left, "test": right})


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the focused tests and confirm the missing-module failure**

Run:

```bash
python3 -m unittest tests.test_protocol_manifests -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'splits'`.

- [x] **Step 3: Implement the manifest contract**

Create `experiments/splits.py`. Canonical JSON uses sorted documents and
`sort_keys=True, separators=(",", ":")`. Reject blank identifiers, duplicate
identifiers within one manifest, wrapper hash mismatch, and pairwise
intersections. `save_manifest()` writes a wrapper containing
`manifest_sha256` and `manifest`.

- [x] **Step 4: Run the focused tests**

Run:

```bash
python3 -m unittest tests.test_protocol_manifests -v
```

Expected: 6 tests pass.

---

### Task 2: Immutable frequency-table artifacts

**Files:**
- Create: `experiments/freq_table.py`
- Create: `tests/test_freq_table.py`

**Interfaces:**
- Consumes: canonical `DocumentManifest` hash, count tensor, model/tokenizer
  identity, vocabulary size, and exclusion IDs.
- Produces: `FrequencyTableMetadata`, `counts_sha256()`,
  `make_frequency_table_metadata()`, `save_frequency_table()`,
  and `load_frequency_table()`.

- [x] **Step 1: Write artifact round-trip and tamper tests**

Create tests using `torch.tensor([4, 0, 7], dtype=torch.int64)` and a one-document
frequency manifest. Verify deterministic sidecar filenames, exact round-trip,
and fatal failures for modified tensor data, modified metadata, model mismatch,
tokenizer mismatch, vocabulary mismatch, and exclusion mismatch. Skip the file
only when Torch is unavailable, matching the repository's current test style.

- [x] **Step 2: Run tests and confirm the missing-module failure**

Run:

```bash
python3 -m unittest tests.test_freq_table -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'freq_table'`.

- [x] **Step 3: Implement canonical hashing and safe I/O**

Implementation requirements:

```python
PROTOCOL_VERSION = "icml2027-pr1a"

@dataclass(frozen=True)
class FrequencyTableMetadata:
    protocol_version: str
    model_id: str
    tokenizer_id: str
    tokenizer_revision: str | None
    vocab_size: int
    counts_dtype: str
    counts_sha256: str
    source_manifest_sha256: str
    exclusion_token_ids: tuple[int, ...]
    num_documents: int
    num_tokens: int
```

Canonicalize counts to contiguous CPU `int64`. Hash a UTF-8 header
`torch.int64:<numel>\n` followed by NumPy C-order bytes. Artifact identity is the
SHA-256 of canonical metadata JSON. The sidecar wrapper is:

```json
{
  "artifact_id": "<metadata sha256>",
  "counts_file": "<artifact_id>.counts.pt",
  "metadata": {
    "protocol_version": "icml2027-pr1a",
    "model_id": "Qwen/Qwen2.5-3B",
    "tokenizer_id": "Qwen/Qwen2.5-3B",
    "tokenizer_revision": null,
    "vocab_size": 151936,
    "counts_dtype": "torch.int64",
    "counts_sha256": "<tensor sha256>",
    "source_manifest_sha256": "<manifest sha256>",
    "exclusion_token_ids": [151643, 151644],
    "num_documents": 10000,
    "num_tokens": 5000000000
  }
}
```

Use `torch.load(path, map_location="cpu", weights_only=True)`, with a
`TypeError` fallback for older Torch that does not accept `weights_only`.
Validate the wrapper artifact ID, sidecar filename, relative counts filename,
tensor shape/dtype/integrality/non-negativity/hash, expected identity fields,
and that every exclusion ID is in range and has count zero.

- [x] **Step 4: Run artifact tests**

Run:

```bash
python3 -m unittest tests.test_freq_table -v
```

Expected: all artifact tests pass.

---

### Task 3: Evaluator protocol validation and provenance

**Files:**
- Create: `experiments/protocol.py`
- Modify: `experiments/eval_prediction_sets.py`
- Create: `tests/test_prediction_protocol.py`

**Interfaces:**
- Consumes: optional `frequency_table`, `frequency_manifest`, `tune_manifest`,
  `calibration_manifest`, and `test_manifest` config paths.
- Produces: `validate_protocol_inputs(config) -> dict[str, Any]`,
  `effective_config_sha256(config) -> str`, and result JSON `protocol` metadata.
  The first two functions live in the lightweight `protocol.py` boundary so
  their fail-closed behavior is testable without importing model/dataset/plot
  dependencies.

- [x] **Step 1: Write protocol-state tests**

Tests must assert:

```python
legacy = validate_protocol_inputs({"allow_legacy_protocol": True})
self.assertEqual(legacy["evidence_grade"], "legacy-smoke")

with self.assertRaisesRegex(RuntimeError, "frequency_table"):
    validate_protocol_inputs({"allow_legacy_protocol": False})
```

With four valid manifest fixtures and a frequency sidecar whose source hash
matches `D_freq`, validation must raise a final
`blocked_pending_pr1b` error. Replace `load_model_and_tokenizer` with a mock and
assert it was not called, proving the failure happens before model allocation.
An intersecting manifest and a source-hash mismatch must fail with their
specific invariant names.

- [x] **Step 2: Run the focused tests and observe failure**

Run:

```bash
python3 -m unittest tests.test_prediction_protocol -v
```

Expected: import or assertion failures because protocol validation is not yet
implemented.

- [x] **Step 3: Add config fields and fail-closed validation**

Extend defaults and CLI with optional paths. Keep
`--allow-legacy-protocol`. `validate_protocol_inputs()` behaves as follows:

1. legacy flag true: return protocol version `legacy-pre-pr1`, evidence grade
   `legacy-smoke`, and never imply disjointness;
2. non-legacy with any missing artifact/manifest path: raise naming all missing
   fields;
3. load all manifests, require roles `freq/tune/cal/test`, check pairwise
   disjointness, and compare the frequency sidecar's source manifest hash;
4. after these pass, raise that PR-1b document-aware sampling is still missing.

Add `effective_config_sha256()` using canonical JSON. In legacy `main()`, record
a `protocol` block and `config_sha256`. When an external frequency table is
supplied for a legacy smoke, load it after tokenizer/model creation through
`load_frequency_table()`; otherwise call the preserved legacy
`build_token_counts()` path. Record the exact artifact reference when used.

- [x] **Step 4: Run the protocol and existing smoke-helper tests**

Run:

```bash
python3 -m unittest tests.test_prediction_protocol tests.test_conformal tests.test_samplers -v
```

Expected: all applicable tests pass; environment-dependent tests may retain
their existing explicit skip.

---

### Task 4: Downstream consumers refuse score-identity drift

**Files:**
- Modify: `experiments/eval_reasoning_self_consistency.py`
- Modify: `experiments/eval_openended_quality.py`
- Modify: `tests/test_prediction_protocol.py`

**Interfaces:**
- Consumes: upstream `prediction_set_metrics.json` containing
  `protocol.frequency_table.metadata_path`, `artifact_id`, and `counts_sha256`.
- Produces: the exact validated count tensor used for upstream calibration.

- [x] **Step 1: Add downstream mismatch tests**

Add fixtures for an upstream metrics JSON. Assert that missing artifact
reference, missing file, and hash mismatch each raise before `load_texts()` can
be called. Assert a matching artifact returns exactly the saved counts.

- [x] **Step 2: Run the test and observe the current silent rebuild**

Run:

```bash
python3 -m unittest tests.test_prediction_protocol -v
```

Expected: new tests fail because both downstream scripts still call
`build_token_counts()` over new WikiText text.

- [x] **Step 3: Replace implicit rebuilding with the shared artifact loader**

Change both `build_counts_for_strategies()` functions:

```python
def build_counts_for_strategies(model, tokenizer, config):
    metrics_path = config.get("prediction_set_metrics")
    if not metrics_path:
        raise RuntimeError(
            "prediction_set_metrics with a frequency-table artifact is required; "
            "downstream runs may not rebuild counts"
        )
    counts, _ = load_frequency_table_from_metrics(
        Path(metrics_path),
        expected_model_id=config["model"],
        expected_tokenizer_id=config["model"],
        expected_vocab_size=model.config.vocab_size,
        expected_exclusion_token_ids=special_token_ids(tokenizer),
    )
    return counts
```

Implement `special_token_ids(tokenizer)` in `freq_table.py` by collecting all
integer IDs in `tokenizer.all_special_ids`, sorting, and deduplicating. Implement
`load_frequency_table_from_metrics()` to load the metrics reference, validate
the sidecar through `load_frequency_table()`, and compare the recorded artifact
ID and counts hash. Remove unused `load_texts`/`build_token_counts` imports from
these two scripts only.

- [x] **Step 4: Re-run downstream protocol tests**

Run:

```bash
python3 -m unittest tests.test_prediction_protocol tests.test_eval_helpers -v
```

Expected: all applicable tests pass.

---

### Task 5: Align the executable queue with the current claim stack

**Files:**
- Modify: `scripts/run_icml2027_gpu_queue.sh`
- Modify: `docs/paper/EXPERIMENT_MAINLINE.md`
- Modify: `docs/paper/RELATED_WORK_POSITIONING.md`
- Modify: `docs/paper/CLAIM_STACK.md`

**Interfaces:**
- Produces: a five-stage gated queue with controlled channels removed and a
  claim stack that distinguishes general learned-`h` from additive offsets.

- [x] **Step 1: Add a shell regression assertion**

Run before editing:

```bash
rg -n "Controlled channels|exp5b_controlled_channels" \
  scripts/run_icml2027_gpu_queue.sh docs/paper/EXPERIMENT_MAINLINE.md
```

Expected: matches demonstrate stale mainline entries.

- [x] **Step 2: Make the smallest narrative corrections**

- Remove controlled channels from the active one-shot queue and renumber it
  `[1/5]` through `[5/5]`.
- Mark Stage 1 commands as target post-PR-3 entrypoints, not currently runnable
  paper commands.
- Add the mandatory calibrated rows RAPS, TS+APS, CNS,
  entropy-Mondrian-margin, frequency-Mondrian-margin, and learned-h to related
  work positioning.
- Expand the formal object in `CLAIM_STACK.md`: general learned-h uses
  `A_h(x,i)=-h_hat(m_i,n_i)`; additive `m_i-g(n_i)` is a restricted,
  interpretable family that is oracle-shaped only under
  `h(m,n)=rho(m-g(n))`.
- State that a null frequency effect does not by itself prove C-margin is
  margin-class optimal; Phase 0 must also check monotonicity of `h_m(m)` and
  report a power bound.

- [x] **Step 3: Verify stale queue references are gone**

Run:

```bash
bash -n scripts/run_icml2027_gpu_queue.sh
! rg -n "exp5b_controlled_channels" scripts/run_icml2027_gpu_queue.sh
```

Expected: shell syntax passes and the queue has no controlled-channel command.

---

### Task 6: Full verification, server validation, and major-round commit

**Files:**
- Modify only files already listed in Tasks 1–5.

**Interfaces:**
- Produces: one reviewed, tested, pushed PR-1a code round.

- [ ] **Step 1: Run local repository checks**

Run:

```bash
python3 -m compileall experiments
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover tests -v
git diff --check
```

Expected: compile and shell checks exit 0; all installed-dependency tests pass;
environment-dependent skips remain explicit.

- [ ] **Step 2: Review the complete diff for scope and contracts**

Run:

```bash
git status --short
git diff --stat
git diff -- experiments/freq_table.py experiments/splits.py \
  experiments/eval_prediction_sets.py \
  experiments/eval_reasoning_self_consistency.py \
  experiments/eval_openended_quality.py tests scripts docs/paper
```

Expected: no generated result artifacts, dependency additions, unrelated
formatting, secret material, or paper-grade unblock.

- [ ] **Step 3: Validate on the RTX 5090 server**

After pushing the code commit, fast-forward `/root/neurips2027`, activate
`/root/autodl-tmp/venvs/neurips2027`, set
`HF_HOME=/root/autodl-tmp/huggingface`, and run the same checks. Then run:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
name = "Qwen/Qwen2.5-3B"
config = AutoConfig.from_pretrained(name, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
print(config.model_type, config.vocab_size, len(tokenizer))
PY
```

Expected: local-only loading succeeds and vocabulary identity is available for
frequency artifact validation. Do not launch a paper-grade experiment.

- [ ] **Step 4: Commit and push the major round**

Run:

```bash
git add experiments tests scripts docs/paper \
  docs/superpowers/plans/2026-07-10-paper-grade-protocol-foundation.md
git commit -m "Add frequency provenance protocol foundation"
git push origin main
```

Expected: commit succeeds and `origin/main` advances. Record both the design
commit and implementation commit hashes in the progress update.
