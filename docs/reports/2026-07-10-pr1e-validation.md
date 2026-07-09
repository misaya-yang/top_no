# PR-1e Frequency Builder Validation

Validation target: `codex/pr1e-frequency-builder`, including independent-review
repairs after the initial `5a5c5e7` implementation commit.

## Automated checks

Local macOS:

```text
python3 -m compileall experiments                         PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python3 -m unittest discover tests                        151/151 PASS
```

RTX 5090 server:

```text
python -m compileall experiments                          PASS
for script in scripts/*.sh; do bash -n "$script"; done   PASS
python -m unittest discover tests                         151 run, 150 PASS,
                                                         1 MPS-only skip
```

## Real Qwen2.5-7B tokenizer smoke

The server loaded only the pinned offline Qwen2.5-7B config and tokenizer at
revision `d149729398750b98c0af14eb82c78cfe92750796`. It built and reloaded a
frequency artifact from two exact manifest-bound temporary documents without
loading model weights.

```json
{
  "artifact_id": "65d96388ed2f9b44d1885d13415618a878058e22f2e92a04e4663a37bac190c2",
  "config_vocab_size": 152064,
  "counts_length": 152064,
  "tokenizer_length": 151665,
  "eos_token_id": 151643,
  "eos_count": 2,
  "eos_excluded": false,
  "excluded_special_ids": 13,
  "num_documents": 2,
  "num_tokens": 29
}
```

This confirms why the builder uses model config vocabulary size rather than
tokenizer length and verifies the one-EOS-per-document policy with the real
tokenizer. The two-document artifact is a functional smoke and is not paper
evidence or a production frequency table.

Independent review found and drove three fail-closed repairs: runtime expected
EOS IDs now reject bool/out-of-range values before equality comparison, and the
CLI requires a 40-hex requested revision plus independently resolved matching
commit hashes from both config and tokenizer. The CLI also rejects mutable local
model/tokenizer paths, including directories shaped like cached Hub snapshots,
so they cannot masquerade as a pinned repo revision. After repair, the reviewer
found no remaining Critical/Important issues and passed the full local suite.
