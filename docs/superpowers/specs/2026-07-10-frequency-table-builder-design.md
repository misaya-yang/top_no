# PR-1e Frequency-Table Builder Design

## Problem

PR-1a could validate immutable count artifacts but could not produce one from a
frozen corpus. Its exclusion helper also zeroed every special token, including
EOS. Because raw documents are encoded with no implicit special tokens, merely
removing EOS from exclusions would still leave EOS frequency at zero and place
the stopping token in the unseen bucket.

## Frozen tokenization policy

The v2 artifact identity binds:

```text
raw-no-specials-filter-specials-plus-one-eos-per-doc-v1
```

For each exact manifest-bound document:

1. encode raw text with special-token insertion, truncation, and padding off;
2. remove every special/control ID produced from the raw body, including a
   literal EOS marker;
3. add exactly one synthetic EOS boundary;
4. add no BOS/PAD and never concatenate across documents.

EOS inclusion wins if another tokenizer role aliases the same ID. The artifact
must satisfy `counts[eos_token_id] == num_documents`; all other special IDs are
canonical exclusions. The schema version, policy, EOS ID, counts hash,
tokenizer/revision, and source-manifest hash are all content-addressed.

## Offline builder

The CLI binds a `freq` manifest to exact JSONL content, then loads pinned
`AutoConfig` and `AutoTokenizer` snapshots with `local_files_only=True`. Config
`vocab_size` sizes the count vector because tokenizer length may be smaller than
the language-model output vocabulary. The CLI never loads model weights and
prints no corpus text.

The current counter processes one document at a time and retains only its token
IDs plus the CPU int64 count vector. It is suitable for audited small/single
shard artifacts. A paper-scale table still needs deterministic shard receipts,
partial count hashes, resume support, and an int64 reducer.

## Safety boundary

Old sidecars without v2 schema/policy/EOS fields fail closed. Runtime loaders
compare the artifact EOS ID with the tokenizer EOS ID. PR-2/PR-3 remain blocked;
this change does not authorize paper-grade experiments or claim that a small
smoke corpus is pretraining-representative.
