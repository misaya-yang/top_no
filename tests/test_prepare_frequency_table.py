import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

try:
    import torch
    import prepare_frequency_table  # noqa: E402
    from freq_table import load_frequency_table  # noqa: E402
    from splits import (  # noqa: E402
        DocumentManifest,
        ManifestDocument,
        SourceDocument,
        content_sha256,
        save_manifest,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class PrepareFrequencyTableTests(unittest.TestCase):
    def test_cli_rejects_branch_names_and_missing_resolved_hashes(self):
        base = [
            "prepare_frequency_table.py",
            "--document-jsonl",
            "unused.jsonl",
            "--source-manifest",
            "unused-manifest.json",
            "--model-id",
            "fixture/model",
            "--revision",
            "main",
            "--output-dir",
            "unused-output",
        ]
        with patch("sys.argv", base), patch.object(
            prepare_frequency_table.AutoConfig,
            "from_pretrained",
        ) as config_load:
            with self.assertRaisesRegex(ValueError, "40-character.*commit"):
                prepare_frequency_table.main()
        config_load.assert_not_called()

        revision = "b" * 40

        class MissingConfigHash:
            vocab_size = 6
            _commit_hash = None

        class MissingTokenizerHash:
            name_or_path = "fixture/model"
            init_kwargs = {}

        pinned = list(base)
        pinned[pinned.index("main")] = revision
        with patch("sys.argv", pinned), patch.object(
            prepare_frequency_table.AutoConfig,
            "from_pretrained",
            return_value=MissingConfigHash(),
        ), patch.object(
            prepare_frequency_table.AutoTokenizer,
            "from_pretrained",
            return_value=MissingTokenizerHash(),
        ):
            with self.assertRaisesRegex(ValueError, "model revision mismatch"):
                prepare_frequency_table.main()

        class PinnedConfig:
            vocab_size = 6
            _commit_hash = revision

        with patch("sys.argv", pinned), patch.object(
            prepare_frequency_table.AutoConfig,
            "from_pretrained",
            return_value=PinnedConfig(),
        ), patch.object(
            prepare_frequency_table.AutoTokenizer,
            "from_pretrained",
            return_value=MissingTokenizerHash(),
        ):
            with self.assertRaisesRegex(ValueError, "tokenizer revision mismatch"):
                prepare_frequency_table.main()

    def test_offline_cli_builds_loadable_eos_aware_artifact_without_model_weights(self):
        revision = "a" * 40

        class Config:
            vocab_size = 6
            _commit_hash = revision

        class Tokenizer:
            name_or_path = "fixture/model"
            init_kwargs = {"_commit_hash": revision}
            eos_token_id = 5
            all_special_ids = [0, 5]

            def __len__(self):
                return 6

            @staticmethod
            def encode(text, add_special_tokens, truncation, padding):
                if (add_special_tokens, truncation, padding) != (False, False, False):
                    raise AssertionError("unexpected tokenization policy")
                return {"alpha": [0, 1, 2, 5], "beta": [2, 3, 2]}[text]

        documents = (
            SourceDocument("doc-a", "alpha"),
            SourceDocument("doc-b", "beta"),
        )
        manifest = DocumentManifest(
            protocol_version="icml2027-pr1a",
            role="freq",
            source="fixture",
            documents=tuple(
                ManifestDocument(
                    item.doc_id,
                    content_sha256(item.text),
                    hashlib.sha256(item.doc_id.encode()).hexdigest(),
                )
                for item in documents
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = save_manifest(manifest, root / "manifest.json")
            jsonl = root / "documents.jsonl"
            jsonl.write_text(
                "\n".join(
                    json.dumps({"doc_id": item.doc_id, "text": item.text})
                    for item in reversed(documents)
                )
                + "\n"
            )
            output = root / "table"
            argv = [
                "prepare_frequency_table.py",
                "--document-jsonl",
                str(jsonl),
                "--source-manifest",
                str(manifest_path),
                "--model-id",
                "fixture/model",
                "--revision",
                revision,
                "--output-dir",
                str(output),
            ]
            stdout = io.StringIO()
            with patch("sys.argv", argv), patch.object(
                prepare_frequency_table.AutoConfig,
                "from_pretrained",
                return_value=Config(),
            ) as config_load, patch.object(
                prepare_frequency_table.AutoTokenizer,
                "from_pretrained",
                return_value=Tokenizer(),
            ) as tokenizer_load, redirect_stdout(stdout):
                prepare_frequency_table.main()

            payload = json.loads(stdout.getvalue())
            sidecar = Path(payload["sidecar"])
            counts, metadata = load_frequency_table(
                sidecar,
                expected_model_id="fixture/model",
                expected_tokenizer_id="fixture/model",
                expected_tokenizer_revision=revision,
                expected_vocab_size=6,
                expected_exclusion_token_ids=(0,),
                expected_eos_token_id=5,
            )

        self.assertEqual(counts.tolist(), [0, 1, 3, 1, 0, 2])
        self.assertEqual(metadata.num_documents, 2)
        self.assertEqual(metadata.eos_token_id, 5)
        self.assertEqual(payload["num_tokens"], 7)
        for mocked in (config_load, tokenizer_load):
            self.assertTrue(mocked.call_args.kwargs["local_files_only"])
            self.assertEqual(mocked.call_args.kwargs["revision"], revision)


if __name__ == "__main__":
    unittest.main()
