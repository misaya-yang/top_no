#!/usr/bin/env python3
"""Create a recomputable zero-overlap receipt for paper-protocol corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_corpus import (
    audit_cross_corpus,
    cross_corpus_receipt_sha256,
    save_cross_corpus_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency-manifest", required=True)
    parser.add_argument("--frequency-document-jsonl", required=True)
    parser.add_argument("--evaluation-split-receipt", required=True)
    parser.add_argument("--evaluation-document-jsonl", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_cross_corpus(
        frequency_manifest_path=Path(args.frequency_manifest),
        frequency_document_jsonl=Path(args.frequency_document_jsonl),
        evaluation_split_receipt_path=Path(args.evaluation_split_receipt),
        evaluation_document_jsonl=Path(args.evaluation_document_jsonl),
    )
    if audit.receipt.verdict != "pass":
        summary = {
            "error": "cross-corpus near-duplicate audit failed",
            "match_count": audit.receipt.match_count,
            "matches": [
                {
                    "frequency_doc_id": match.frequency_doc_id,
                    "evaluation_doc_id": match.evaluation_doc_id,
                    "intersection_size": match.intersection_size,
                    "union_size": match.union_size,
                }
                for match in audit.matches
            ],
        }
        raise SystemExit(json.dumps(summary, sort_keys=True))

    output = save_cross_corpus_audit(audit, Path(args.output))
    print(
        json.dumps(
            {
                "artifact_id": cross_corpus_receipt_sha256(audit.receipt),
                "candidate_pair_count": audit.receipt.candidate_pair_count,
                "exact_comparison_count": audit.receipt.exact_comparison_count,
                "match_count": audit.receipt.match_count,
                "output": str(output.resolve()),
                "verdict": audit.receipt.verdict,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
