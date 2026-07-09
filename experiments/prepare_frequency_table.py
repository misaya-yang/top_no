#!/usr/bin/env python3
"""Build a pinned frequency table from a frozen D_freq manifest and JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from cross_corpus import bind_frequency_documents
from freq_table import (
    count_frequency_tokens,
    frequency_exclusion_token_ids,
    load_frequency_table_metadata,
    make_frequency_table_metadata,
    runtime_tokenizer_identity,
    save_frequency_table,
)
from splits import manifest_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-jsonl", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("revision must be a pinned 40-character lowercase commit SHA")
    tokenizer_id = args.tokenizer_id or args.model_id
    common = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(args.model_id, **common)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **common)
    config_revision = getattr(config, "_commit_hash", None)
    if config_revision != args.revision:
        raise ValueError(
            "model revision mismatch: "
            f"config={config_revision!r} requested={args.revision!r}"
        )
    tokenizer_init = getattr(tokenizer, "init_kwargs", None)
    tokenizer_revision = (
        tokenizer_init.get("_commit_hash")
        if isinstance(tokenizer_init, dict)
        else None
    )
    if tokenizer_revision != args.revision:
        raise ValueError(
            "tokenizer revision mismatch: "
            f"tokenizer={tokenizer_revision!r} requested={args.revision!r}"
        )
    resolved_tokenizer_id, resolved_revision = runtime_tokenizer_identity(
        tokenizer,
    )
    if resolved_tokenizer_id != tokenizer_id:
        raise ValueError(
            "tokenizer_id mismatch: "
            f"runtime={resolved_tokenizer_id!r} requested={tokenizer_id!r}"
        )
    if resolved_revision != args.revision:
        raise ValueError("tokenizer revision did not resolve to requested commit")
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise ValueError("model config vocab_size must be an integer")
    if len(tokenizer) > vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model vocab_size")

    manifest, documents = bind_frequency_documents(
        Path(args.source_manifest),
        Path(args.document_jsonl),
    )
    counts = count_frequency_tokens(
        tokenizer,
        documents,
        vocab_size=vocab_size,
    )
    metadata = make_frequency_table_metadata(
        counts,
        model_id=args.model_id,
        tokenizer_id=resolved_tokenizer_id,
        tokenizer_revision=resolved_revision,
        source_manifest_sha256=manifest_sha256(manifest),
        exclusion_token_ids=frequency_exclusion_token_ids(tokenizer),
        num_documents=len(documents),
        eos_token_id=tokenizer.eos_token_id,
    )
    sidecar = save_frequency_table(counts, metadata, Path(args.output_dir))
    _, artifact_id, _ = load_frequency_table_metadata(sidecar)
    print(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "eos_token_id": metadata.eos_token_id,
                "num_documents": metadata.num_documents,
                "num_tokens": metadata.num_tokens,
                "sidecar": str(sidecar.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
