# PR-1e Frequency-Table Builder Plan

- [x] Bind tokenization policy, frequency schema, and EOS ID in metadata.
- [x] Filter raw special IDs and count exactly one EOS per document.
- [x] Require EOS count to equal manifest document count on save/load.
- [x] Compare runtime and artifact EOS IDs.
- [x] Add an offline config+tokenizer-only CLI over exact manifest/JSONL input.
- [x] Use model config vocabulary size and keep model weights unloaded.
- [x] Add fake-tokenizer and mock-offline-CLI tests.
- [ ] Run a real pinned Qwen tokenizer smoke on the server.
- [ ] Obtain independent review, merge, and push.
