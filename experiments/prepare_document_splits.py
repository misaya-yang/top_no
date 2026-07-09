#!/usr/bin/env python3
"""Build deterministic tune/cal/test manifests from frozen source JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splits import (
    build_split_artifacts,
    load_source_documents_jsonl,
    save_split_artifacts,
    split_receipt_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--cluster-namespace-sha256", default=None)
    parser.add_argument("--global-salt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shingle-size", type=int, default=13)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--minhash-seed", type=int, default=1729)
    parser.add_argument("--num-perm", type=int, default=100)
    parser.add_argument("--lsh-bands", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_split_artifacts(
        load_source_documents_jsonl(Path(args.input_jsonl)),
        source=args.source,
        source_snapshot_sha256=args.source_snapshot_sha256,
        cluster_namespace_sha256=args.cluster_namespace_sha256,
        global_salt=args.global_salt,
        shingle_size=args.shingle_size,
        jaccard_threshold=args.jaccard_threshold,
        minhash_seed=args.minhash_seed,
        num_perm=args.num_perm,
        lsh_bands=args.lsh_bands,
    )
    receipt_path = save_split_artifacts(result, Path(args.output_dir))
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": split_receipt_sha256(result.receipt),
                "num_input_documents": result.receipt.num_input_documents,
                "num_clusters": result.receipt.num_clusters,
                "role_counts": {
                    role: len(manifest.documents)
                    for role, manifest in result.manifests.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
