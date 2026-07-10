import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

runtime_spec = importlib.util.find_spec("paper_runtime")
paper_runtime = importlib.import_module("paper_runtime") if runtime_spec else None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class PaperRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(paper_runtime)

    def receipt(self):
        return paper_runtime.RuntimeReceipt(
            schema_version=paper_runtime.RUNTIME_RECEIPT_SCHEMA_VERSION,
            policy_version=paper_runtime.RUNTIME_POLICY_VERSION,
            created_by_commit="a" * 40,
            policy=paper_runtime.RuntimePolicy(
                seed=42,
                python_hash_seed="42",
                cublas_workspace_config=":4096:8",
                deterministic_algorithms=True,
                deterministic_warn_only=False,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
            ),
            environment=paper_runtime.RuntimeEnvironment(
                python_version="3.12.12",
                torch_version="2.9.1+cu128",
                cuda_version="12.8",
                cudnn_version=91002,
                cuda_available=True,
                device_names=("NVIDIA GeForce RTX 5090",),
                device_capabilities=("12.0",),
            ),
        )

    def test_content_addressed_round_trip_and_tamper_rejection(self):
        receipt = self.receipt()
        with tempfile.TemporaryDirectory() as tmp:
            path = paper_runtime.save_runtime_receipt(receipt, Path(tmp))
            loaded, artifact_id = paper_runtime.load_runtime_receipt(path)

            self.assertEqual(loaded, receipt)
            self.assertEqual(
                artifact_id,
                paper_runtime.runtime_receipt_artifact_id(receipt),
            )
            self.assertEqual(path.name, f"{artifact_id}.json")

            payload = json.loads(path.read_text())
            payload["receipt"]["policy"]["deterministic_algorithms"] = False
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "deterministic_algorithms"):
                paper_runtime.load_runtime_receipt(path)

    def test_receipt_rejects_non_cuda_and_noncanonical_policy(self):
        receipt = self.receipt()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            paper_runtime.validate_runtime_receipt(
                replace(
                    receipt,
                    environment=replace(
                        receipt.environment,
                        cuda_available=False,
                        cuda_version=None,
                        cudnn_version=None,
                        device_names=(),
                        device_capabilities=(),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "cublas_workspace_config"):
            paper_runtime.validate_runtime_receipt(
                replace(
                    receipt,
                    policy=replace(
                        receipt.policy, cublas_workspace_config=":16:8"
                    ),
                )
            )

    def test_configuration_fails_before_mutation_when_launch_env_is_missing(self):
        prior_mode = torch.get_deterministic_debug_mode()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
                paper_runtime.configure_paper_runtime(
                    seed=42,
                    created_by_commit="a" * 40,
                    require_cuda=False,
                )
        self.assertEqual(torch.get_deterministic_debug_mode(), prior_mode)

        with patch.dict(os.environ, {"PYTHONHASHSEED": "42"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                paper_runtime.configure_paper_runtime(
                    seed=42,
                    created_by_commit="a" * 40,
                    require_cuda=False,
                )
        self.assertEqual(torch.get_deterministic_debug_mode(), prior_mode)

    def test_configuration_refuses_cuda_initialized_process(self):
        launch_env = {
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        with patch.dict(os.environ, launch_env, clear=True), patch.object(
            torch.cuda, "is_initialized", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "before CUDA initialization"):
                paper_runtime.configure_paper_runtime(
                    seed=42,
                    created_by_commit="a" * 40,
                    require_cuda=False,
                )

    def test_cpu_configuration_sets_and_verifies_every_runtime_switch(self):
        launch_env = {
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        prior_debug = torch.get_deterministic_debug_mode()
        prior_benchmark = torch.backends.cudnn.benchmark
        prior_cudnn_deterministic = torch.backends.cudnn.deterministic
        prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            with patch.dict(os.environ, launch_env, clear=True):
                receipt = paper_runtime.configure_paper_runtime(
                    seed=42,
                    created_by_commit="a" * 40,
                    require_cuda=False,
                )
                paper_runtime.assert_current_runtime(receipt, require_cuda=False)

                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(
                    torch.is_deterministic_algorithms_warn_only_enabled()
                )
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(receipt.environment.cuda_available)

                with self.assertRaisesRegex(ValueError, "current runtime"):
                    paper_runtime.assert_current_runtime(
                        replace(
                            receipt,
                            environment=replace(
                                receipt.environment,
                                torch_version="different",
                            ),
                        ),
                        require_cuda=False,
                    )
        finally:
            torch.set_deterministic_debug_mode(prior_debug)
            torch.backends.cudnn.benchmark = prior_benchmark
            torch.backends.cudnn.deterministic = prior_cudnn_deterministic
            torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32

    def test_seed_and_commit_are_strictly_validated(self):
        launch_env = {
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        invalid = (
            ({"seed": True, "created_by_commit": "a" * 40}, "seed"),
            ({"seed": -1, "created_by_commit": "a" * 40}, "seed"),
            ({"seed": 2**32, "created_by_commit": "a" * 40}, "seed"),
            ({"seed": 42, "created_by_commit": "main"}, "created_by_commit"),
        )
        with patch.dict(os.environ, launch_env, clear=True):
            for kwargs, message in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, message
                ):
                    paper_runtime.configure_paper_runtime(
                        require_cuda=False,
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
