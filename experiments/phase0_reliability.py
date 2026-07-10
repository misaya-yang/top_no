#!/usr/bin/env python3
"""Run one provenance-bound Phase-0 margin/frequency pilot cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Collection, Iterable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from document_store import BoundDocument, bind_split_documents
from freq_table import (
    frequency_exclusion_token_ids,
    load_frequency_table,
    load_frequency_table_metadata,
    runtime_tokenizer_identity,
)
from paper_runtime import (
    configure_paper_runtime,
    runtime_receipt_artifact_id,
    save_runtime_receipt,
)
from phase0_stats import (
    DocumentGridStats,
    GridSpec,
    accumulate_document,
    analyze_grid,
    merge_document_stats,
)
from protocol import validate_phase0_inputs
from splits import ManifestDocument, SelectedPosition, pooled_positions


CHECKPOINT_SCHEMA_VERSION = "icml2027-phase0-checkpoint-v1"
SUMMARY_SCHEMA_VERSION = "icml2027-phase0-summary-v1"
MATRIX_SCHEMA_VERSION = "icml2027-phase0-two-hour-matrix-v1"


@dataclass(frozen=True)
class DocumentLogitsRows:
    doc_id: str
    logits: torch.Tensor
    targets: torch.Tensor
    selections: tuple[SelectedPosition, ...]


@dataclass(frozen=True)
class ConsumeResult:
    status: str
    document_stats: tuple[DocumentGridStats, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class PreflightCell:
    cell: dict[str, object]
    model: dict[str, object]
    domain: dict[str, object]
    protocol_config: dict[str, object]
    protocol_receipt: dict[str, object]
    documents: tuple[BoundDocument, ...]
    tokenizer: object
    token_counts: torch.Tensor
    exclusion_token_ids: tuple[int, ...]
    frequency_artifact_id: str
    model_weight_paths: tuple[Path, ...]


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalized_exclusions(values: Collection[int], vocab_size: int) -> tuple[int, ...]:
    result = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("excluded_target_ids must contain integers")
        if value < 0 or value >= vocab_size:
            raise ValueError("excluded_target_ids contains an out-of-range token ID")
        result.add(value)
    return tuple(sorted(result))


def iter_document_logits(
    model,
    tokenizer,
    documents: Sequence[BoundDocument],
    *,
    device: torch.device,
    max_length: int,
    min_context: int,
    stride: int,
    batch_size: int,
    excluded_target_ids: Collection[int],
) -> Iterable[DocumentLogitsRows]:
    """Yield stride-selected causal rows from full document forwards."""
    max_length = _require_positive_int(max_length, "max_length")
    stride = _require_positive_int(stride, "stride")
    batch_size = _require_positive_int(batch_size, "batch_size")
    if isinstance(min_context, bool) or not isinstance(min_context, int) or min_context < 1:
        raise ValueError("min_context must be a positive integer")
    ordered = tuple(sorted(documents, key=lambda item: item.doc_id))
    if not ordered or any(not isinstance(item, BoundDocument) for item in ordered):
        raise ValueError("documents must contain BoundDocument values")
    if {item.role for item in ordered} != {"tune"}:
        raise ValueError("Phase-0 documents must come from the tune manifest")
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("Phase-0 causal extraction requires right padding")

    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        encoded = tokenizer(
            [item.text for item in batch],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer returned malformed batched tensors")
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = output.logits
        if logits.dim() != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError("model returned malformed causal logits")
        exclusions = _normalized_exclusions(excluded_target_ids, logits.shape[-1])
        for row_index, document in enumerate(batch):
            length = int(attention_mask[row_index].sum().item())
            token_ids = tuple(input_ids[row_index, :length].detach().cpu().tolist())
            manifest_document = ManifestDocument(
                document.doc_id,
                document.content_sha256,
                document.cluster_id,
            )
            selections = pooled_positions(
                manifest_document,
                token_ids,
                excluded_target_ids=exclusions,
                min_context=min_context,
                stride=stride,
            )
            if not selections:
                continue
            target_indices = torch.tensor(
                [item.target_index for item in selections],
                device=device,
                dtype=torch.long,
            )
            yield DocumentLogitsRows(
                doc_id=document.doc_id,
                logits=logits[row_index, target_indices - 1, :],
                targets=input_ids[row_index, target_indices],
                selections=selections,
            )


def consume_document_rows(
    rows: Iterable[DocumentLogitsRows],
    *,
    token_counts: torch.Tensor,
    grid: GridSpec,
    excluded_token_ids: Collection[int],
    permutation_seed: int,
    wall_seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> ConsumeResult:
    if not isinstance(wall_seconds, (int, float)) or wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    started = clock()
    elapsed = 0.0
    stats = []
    status = "COMPLETE"
    for row in rows:
        elapsed = clock() - started
        if elapsed >= wall_seconds:
            status = "PARTIAL"
            break
        stats.append(
            accumulate_document(
                row.doc_id,
                row.logits,
                row.targets,
                token_counts,
                grid=grid,
                excluded_token_ids=excluded_token_ids,
                permutation_seed=permutation_seed,
            )
        )
    return ConsumeResult(status, tuple(stats), elapsed)


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, *, expected_identity: str) -> dict[str, object]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError("failed to load Phase-0 checkpoint") from exc
    if not isinstance(payload, dict):
        raise ValueError("Phase-0 checkpoint must contain a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Phase-0 checkpoint schema mismatch")
    if payload.get("identity_sha256") != expected_identity:
        raise ValueError("Phase-0 checkpoint identity mismatch")
    return payload


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _load_matrix(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported Phase-0 matrix config")
    return payload


def _by_key(values: object, name: str) -> dict[str, dict[str, object]]:
    if not isinstance(values, list):
        raise ValueError(f"matrix {name} must be a list")
    result = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("key"), str):
            raise ValueError(f"matrix {name} contains a malformed entry")
        key = value["key"]
        if key in result:
            raise ValueError(f"matrix {name} contains duplicate key {key!r}")
        result[key] = value
    return result


def _relative_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a non-empty relative path")
    path = (root / value).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"{name} escapes the Phase-0 data root")
    return path


def _frequency_sidecar(data_root: Path, directory: object) -> Path:
    root = _relative_path(data_root, directory, "frequency_dir")
    matches = sorted(root.glob("[0-9a-f]" * 64 + ".json"))
    if len(matches) != 1:
        raise ValueError(
            f"frequency_dir must contain exactly one content-addressed sidecar: {root}"
        )
    return matches[0]


def _validate_cached_model_weights(model_id: str, revision: str) -> tuple[Path, ...]:
    common = {
        "revision": revision,
        "local_files_only": True,
        "_raise_exceptions_for_missing_entries": False,
        "_raise_exceptions_for_connection_errors": False,
    }
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = cached_file(model_id, index_name, **common)
        if index_path is None:
            continue
        try:
            shard_paths, _ = get_checkpoint_shard_files(
                model_id,
                index_path,
                revision=revision,
                local_files_only=True,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("model weight shards are not fully cached") from exc
        resolved = tuple(Path(path).resolve() for path in shard_paths)
        if resolved and all(path.is_file() and path.stat().st_size > 0 for path in resolved):
            return resolved
        raise ValueError("model weight shards are not fully cached")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight_path = cached_file(model_id, filename, **common)
        if weight_path is None:
            continue
        resolved = Path(weight_path).resolve()
        if resolved.is_file() and resolved.stat().st_size > 0:
            return (resolved,)
    raise ValueError("model weights are not fully cached")


def preflight_cell(matrix_path: Path, cell_key: str, data_root: Path) -> PreflightCell:
    matrix = _load_matrix(matrix_path)
    models = _by_key(matrix.get("models"), "models")
    domains = _by_key(matrix.get("domains"), "domains")
    cells = _by_key(matrix.get("cells"), "cells")
    try:
        cell = cells[cell_key]
        model = models[cell["model_key"]]
        domain = domains[cell["domain_key"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown or malformed Phase-0 cell: {cell_key!r}") from exc
    frequency_sidecar = _frequency_sidecar(data_root, model.get("frequency_dir"))
    paths = {
        key: _relative_path(data_root, domain[key], key)
        for key in (
            "document_jsonl",
            "tune_manifest",
            "calibration_manifest",
            "test_manifest",
            "split_receipt",
            "cross_corpus_receipt",
        )
    }
    paths["frequency_manifest"] = _relative_path(
        data_root, matrix["frequency_manifest"], "frequency_manifest"
    )
    paths["frequency_document_jsonl"] = _relative_path(
        data_root,
        matrix["frequency_document_jsonl"],
        "frequency_document_jsonl",
    )
    protocol_config = {
        "allow_legacy_protocol": False,
        "model_revision": model["revision"],
        "frequency_table": str(frequency_sidecar),
        **{key: str(value) for key, value in paths.items()},
        "calibration_position_salt": matrix["calibration_position_salt"],
        "test_position_salt": matrix["test_position_salt"],
    }
    protocol_receipt = validate_phase0_inputs(protocol_config)
    bound = bind_split_documents(
        paths["split_receipt"],
        paths["document_jsonl"],
        configured_manifests={
            "tune": paths["tune_manifest"],
            "cal": paths["calibration_manifest"],
            "test": paths["test_manifest"],
        },
    )
    common = {
        "revision": model["revision"],
        "local_files_only": True,
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    config = AutoConfig.from_pretrained(model["model_id"], **common)
    tokenizer = AutoTokenizer.from_pretrained(model["model_id"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    resolved_model_revision = getattr(config, "_commit_hash", None)
    if resolved_model_revision != model["revision"]:
        raise ValueError("cached model config revision mismatch")
    tokenizer_id, tokenizer_revision = runtime_tokenizer_identity(
        tokenizer,
        resolved_model_revision=resolved_model_revision,
    )
    model_weight_paths = _validate_cached_model_weights(
        str(model["model_id"]),
        str(model["revision"]),
    )
    token_counts, _ = load_frequency_table(
        frequency_sidecar,
        expected_model_id=model["model_id"],
        expected_tokenizer_id=tokenizer_id,
        expected_tokenizer_revision=tokenizer_revision,
        expected_vocab_size=int(config.vocab_size),
        expected_exclusion_token_ids=frequency_exclusion_token_ids(tokenizer),
        expected_eos_token_id=tokenizer.eos_token_id,
    )
    _, artifact_id, _ = load_frequency_table_metadata(frequency_sidecar)
    return PreflightCell(
        cell=cell,
        model=model,
        domain=domain,
        protocol_config=protocol_config,
        protocol_receipt=protocol_receipt,
        documents=bound.for_role("tune"),
        tokenizer=tokenizer,
        token_counts=token_counts,
        exclusion_token_ids=frequency_exclusion_token_ids(tokenizer),
        frequency_artifact_id=artifact_id,
        model_weight_paths=model_weight_paths,
    )


def _stats_payload(item: DocumentGridStats) -> dict[str, object]:
    return asdict(item)


def _stats_from_payload(value: object) -> DocumentGridStats:
    if not isinstance(value, dict):
        raise ValueError("checkpoint document statistics are malformed")
    return DocumentGridStats(**value)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, path)


def run_cell(
    preflight: PreflightCell,
    *,
    output_dir: Path,
    wall_seconds: float,
    created_by_commit: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", created_by_commit) is None:
        raise ValueError("created_by_commit must be a pinned 40-hex commit")
    runtime = configure_paper_runtime(
        seed=int(preflight.cell["seed"]),
        created_by_commit=created_by_commit,
        require_cuda=True,
    )
    runtime_dir = output_dir / "runtime"
    runtime_path = save_runtime_receipt(runtime, runtime_dir)
    runtime_id = runtime_receipt_artifact_id(runtime)
    identity_payload = {
        "cell": preflight.cell,
        "model": preflight.model,
        "domain": preflight.domain,
        "protocol": preflight.protocol_receipt,
        "commit": created_by_commit,
        "runtime_artifact_id": runtime_id,
        "frequency_artifact_id": preflight.frequency_artifact_id,
    }
    identity = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
    checkpoint_path = output_dir / "checkpoint.pt"
    stats: list[DocumentGridStats] = []
    processed = set()
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, expected_identity=identity)
        stats = [_stats_from_payload(item) for item in checkpoint["document_stats"]]
        processed = {item.doc_id for item in stats}

    model = AutoModelForCausalLM.from_pretrained(
        preflight.model["model_id"],
        revision=preflight.model["revision"],
        local_files_only=True,
        trust_remote_code=bool(preflight.model.get("trust_remote_code", False)),
        dtype=torch.float16,
    ).to(torch.device("cuda"))
    model.eval()
    documents = tuple(item for item in preflight.documents if item.doc_id not in processed)
    grid = GridSpec.default(min_true_count=int(preflight.cell["min_true_count"]))
    started = time.monotonic()
    max_positions = int(preflight.cell["max_positions"])
    status = "COMPLETE"
    rows = iter_document_logits(
        model,
        preflight.tokenizer,
        documents,
        device=torch.device("cuda"),
        max_length=int(preflight.cell["max_length"]),
        min_context=int(preflight.cell["min_context"]),
        stride=int(preflight.cell["stride"]),
        batch_size=int(preflight.cell["batch_size"]),
        excluded_target_ids=preflight.exclusion_token_ids,
    )
    positions = sum(item.n_positions for item in stats)
    for row in rows:
        if time.monotonic() - started >= wall_seconds:
            status = "PARTIAL"
            break
        remaining = max_positions - positions
        if remaining <= 0:
            break
        logits = row.logits[:remaining]
        targets = row.targets[:remaining]
        item = accumulate_document(
            row.doc_id,
            logits,
            targets,
            preflight.token_counts,
            grid=grid,
            excluded_token_ids=preflight.exclusion_token_ids,
            permutation_seed=int(preflight.cell["seed"]),
        )
        stats.append(item)
        positions += item.n_positions
        save_checkpoint(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity_sha256": identity,
                "document_stats": [_stats_payload(value) for value in stats],
                "processed_doc_ids": [value.doc_id for value in stats],
            },
        )
    if not stats:
        raise RuntimeError("Phase-0 cell produced no eligible positions")
    merged = merge_document_stats(stats)
    analysis = analyze_grid(merged["num"], merged["den"], grid=grid)
    permutation_analysis = analyze_grid(
        merged["perm_num"], merged["perm_den"], grid=grid
    )
    half_analysis = [
        analyze_grid(merged["half_num"][half], merged["half_den"][half], grid=grid)
        for half in (0, 1)
    ]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evidence_grade": "E-pilot",
        "paper_citable": False,
        "cell_key": preflight.cell["key"],
        "model_key": preflight.cell["model_key"],
        "domain_key": preflight.cell["domain_key"],
        "model_id": preflight.model["model_id"],
        "model_revision": preflight.model["revision"],
        "created_by_commit": created_by_commit,
        "runtime_artifact_id": runtime_id,
        "runtime_receipt_path": str(runtime_path.resolve()),
        "frequency_artifact_id": preflight.frequency_artifact_id,
        "protocol": preflight.protocol_receipt,
        "identity_sha256": identity,
        "completion_status": status,
        "n_documents": merged["n_documents"],
        "n_positions": merged["n_positions"],
        "stride": preflight.cell["stride"],
        "elapsed_seconds": time.monotonic() - started,
        "analysis": analysis,
        "permutation_analysis": permutation_analysis,
        "half_analysis": half_analysis,
    }
    _atomic_json(output_dir / "phase0_summary.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wall-seconds", type=float, default=1500.0)
    parser.add_argument("--created-by-commit", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = preflight_cell(Path(args.config), args.cell, Path(args.data_root))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "cell": args.cell,
                    "status": "PREFLIGHT_PASS",
                    "documents": len(preflight.documents),
                    "frequency_artifact_id": preflight.frequency_artifact_id,
                    "model_revision": preflight.model["revision"],
                    "model_weight_shards": len(preflight.model_weight_paths),
                },
                sort_keys=True,
            )
        )
        return
    if args.created_by_commit is None:
        raise ValueError("--created-by-commit is required outside preflight")
    summary = run_cell(
        preflight,
        output_dir=Path(args.output_root) / args.cell,
        wall_seconds=args.wall_seconds,
        created_by_commit=args.created_by_commit,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
