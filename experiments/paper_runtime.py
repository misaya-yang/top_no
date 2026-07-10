"""Fail-closed deterministic runtime receipts for citable CUDA experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch


RUNTIME_RECEIPT_SCHEMA_VERSION = "icml2027-runtime-receipt-v1"
RUNTIME_POLICY_VERSION = "icml2027-deterministic-cuda-v1"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class RuntimePolicy:
    seed: int
    python_hash_seed: str
    cublas_workspace_config: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool


@dataclass(frozen=True)
class RuntimeEnvironment:
    python_version: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    cuda_available: bool
    device_names: tuple[str, ...]
    device_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeReceipt:
    schema_version: str
    policy_version: str
    created_by_commit: str
    policy: RuntimePolicy
    environment: RuntimeEnvironment


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}]")
    return seed


def _validate_commit(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("created_by_commit must be a pinned 40-hex commit")
    return value


def _require_nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_policy(policy: RuntimePolicy) -> None:
    if not isinstance(policy, RuntimePolicy):
        raise ValueError("policy must be RuntimePolicy")
    seed = _validate_seed(policy.seed)
    if policy.python_hash_seed != str(seed):
        raise ValueError("python_hash_seed must exactly match seed")
    if policy.cublas_workspace_config != _CUBLAS_WORKSPACE_CONFIG:
        raise ValueError(
            f"cublas_workspace_config must be {_CUBLAS_WORKSPACE_CONFIG!r}"
        )
    expected = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for field_name, expected_value in expected.items():
        actual = getattr(policy, field_name)
        if not isinstance(actual, bool) or actual is not expected_value:
            raise ValueError(f"{field_name} must be {expected_value}")


def _validate_environment(
    environment: RuntimeEnvironment,
    *,
    require_cuda: bool,
) -> None:
    if not isinstance(environment, RuntimeEnvironment):
        raise ValueError("environment must be RuntimeEnvironment")
    _require_nonempty(environment.python_version, "python_version")
    _require_nonempty(environment.torch_version, "torch_version")
    if not isinstance(environment.cuda_available, bool):
        raise ValueError("cuda_available must be boolean")
    if not isinstance(environment.device_names, tuple) or not isinstance(
        environment.device_capabilities, tuple
    ):
        raise ValueError("device identity fields must be tuples")
    if len(environment.device_names) != len(environment.device_capabilities):
        raise ValueError("device names and capabilities must have equal length")
    if require_cuda and not environment.cuda_available:
        raise ValueError("paper runtime receipt requires CUDA")
    if environment.cuda_available:
        _require_nonempty(environment.cuda_version, "cuda_version")
        if (
            isinstance(environment.cudnn_version, bool)
            or not isinstance(environment.cudnn_version, int)
            or environment.cudnn_version <= 0
        ):
            raise ValueError("cudnn_version must be a positive integer")
        if not environment.device_names:
            raise ValueError("CUDA runtime must contain at least one device")
    elif environment.device_names or environment.device_capabilities:
        raise ValueError("non-CUDA runtime cannot contain CUDA devices")
    for name in environment.device_names:
        _require_nonempty(name, "device_name")
    for capability in environment.device_capabilities:
        if (
            not isinstance(capability, str)
            or re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None
        ):
            raise ValueError("device_capability must use major.minor format")
    if environment.cuda_version is not None:
        _require_nonempty(environment.cuda_version, "cuda_version")
    if environment.cudnn_version is not None and (
        isinstance(environment.cudnn_version, bool)
        or not isinstance(environment.cudnn_version, int)
        or environment.cudnn_version <= 0
    ):
        raise ValueError("cudnn_version must be a positive integer or None")


def validate_runtime_receipt(
    receipt: RuntimeReceipt,
    *,
    require_cuda: bool = True,
) -> None:
    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be boolean")
    if not isinstance(receipt, RuntimeReceipt):
        raise ValueError("receipt must be RuntimeReceipt")
    if receipt.schema_version != RUNTIME_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported runtime receipt schema_version")
    if receipt.policy_version != RUNTIME_POLICY_VERSION:
        raise ValueError("unsupported runtime policy_version")
    _validate_commit(receipt.created_by_commit)
    _validate_policy(receipt.policy)
    _validate_environment(receipt.environment, require_cuda=require_cuda)


def _receipt_payload(receipt: RuntimeReceipt) -> dict[str, object]:
    return asdict(receipt)


def runtime_receipt_artifact_id(receipt: RuntimeReceipt) -> str:
    validate_runtime_receipt(receipt)
    return hashlib.sha256(_canonical_json(_receipt_payload(receipt))).hexdigest()


def save_runtime_receipt(receipt: RuntimeReceipt, output_dir: Path) -> Path:
    artifact_id = runtime_receipt_artifact_id(receipt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{artifact_id}.json"
    wrapper = {"artifact_id": artifact_id, "receipt": _receipt_payload(receipt)}
    path.write_bytes(_canonical_json(wrapper) + b"\n")
    return path


def _require_keys(payload: object, expected: set[str], field_name: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{field_name} has invalid fields")
    return payload


def load_runtime_receipt(path: Path) -> tuple[RuntimeReceipt, str]:
    path = Path(path)
    try:
        wrapper = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("failed to read runtime receipt") from exc
    wrapper = _require_keys(
        wrapper,
        {"artifact_id", "receipt"},
        "runtime receipt wrapper",
    )
    raw = _require_keys(
        wrapper["receipt"],
        {item.name for item in fields(RuntimeReceipt)},
        "runtime receipt",
    )
    policy_payload = _require_keys(
        raw["policy"],
        {item.name for item in fields(RuntimePolicy)},
        "runtime policy",
    )
    environment_payload = _require_keys(
        raw["environment"],
        {item.name for item in fields(RuntimeEnvironment)},
        "runtime environment",
    )
    try:
        environment = RuntimeEnvironment(
            **{
                **environment_payload,
                "device_names": tuple(environment_payload["device_names"]),
                "device_capabilities": tuple(
                    environment_payload["device_capabilities"]
                ),
            }
        )
        receipt = RuntimeReceipt(
            schema_version=raw["schema_version"],
            policy_version=raw["policy_version"],
            created_by_commit=raw["created_by_commit"],
            policy=RuntimePolicy(**policy_payload),
            environment=environment,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime receipt contains invalid field types") from exc
    artifact_id = runtime_receipt_artifact_id(receipt)
    if wrapper["artifact_id"] != artifact_id:
        raise ValueError("artifact_id mismatch for runtime receipt")
    if path.name != f"{artifact_id}.json":
        raise ValueError("artifact_id filename mismatch for runtime receipt")
    return receipt, artifact_id


def _current_policy(seed: int) -> RuntimePolicy:
    return RuntimePolicy(
        seed=seed,
        python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
    )


def _current_environment() -> RuntimeEnvironment:
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        count = torch.cuda.device_count()
        names = tuple(torch.cuda.get_device_name(index) for index in range(count))
        capabilities = tuple(
            ".".join(str(value) for value in torch.cuda.get_device_capability(index))
            for index in range(count)
        )
    else:
        names = ()
        capabilities = ()
    return RuntimeEnvironment(
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        cuda_available=cuda_available,
        device_names=names,
        device_capabilities=capabilities,
    )


def assert_current_runtime(
    receipt: RuntimeReceipt,
    *,
    require_cuda: bool = True,
) -> None:
    validate_runtime_receipt(receipt, require_cuda=require_cuda)
    actual_policy = _current_policy(receipt.policy.seed)
    actual_environment = _current_environment()
    if actual_policy != receipt.policy or actual_environment != receipt.environment:
        raise ValueError("current runtime does not match the frozen runtime receipt")


def configure_paper_runtime(
    *,
    seed: int,
    created_by_commit: str,
    require_cuda: bool = True,
) -> RuntimeReceipt:
    """Configure strict determinism before CUDA initialization and capture it."""
    seed = _validate_seed(seed)
    created_by_commit = _validate_commit(created_by_commit)
    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be boolean")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            "PYTHONHASHSEED must equal seed in the process launch environment"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' in the process launch environment"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("paper runtime must be configured before CUDA initialization")
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("paper runtime requires an available CUDA device")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    receipt = RuntimeReceipt(
        schema_version=RUNTIME_RECEIPT_SCHEMA_VERSION,
        policy_version=RUNTIME_POLICY_VERSION,
        created_by_commit=created_by_commit,
        policy=_current_policy(seed),
        environment=_current_environment(),
    )
    validate_runtime_receipt(receipt, require_cuda=require_cuda)
    assert_current_runtime(receipt, require_cuda=require_cuda)
    return receipt
