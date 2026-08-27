import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from capture_glm53_hidden_suite import (
    _finalize_window,
    _load_panel_role,
    _load_suite_token_ids,
    _validate_chunks,
)
from export_glm53_lm_head import main as export_head_main
from import_glm53_teacher_logits import main as import_teacher_main
from kld_common import canonical_json_sha256, sha256_file
from replay_glm53_hidden_logits import main as replay_hidden_main


class Glm53CaptureTests(unittest.TestCase):
    def test_selection_role_loads_directly_from_sealed_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            panel_dir = Path(temporary)
            arrays = panel_dir / "arrays"
            arrays.mkdir()
            windows = []
            for role, index in (("fit", 0), ("selection", 0), ("selection", 1)):
                window_id = f"{role}-{index:04d}"
                token_path = arrays / f"{window_id}.tokens.npy"
                np.save(
                    token_path,
                    np.arange(index, index + 5, dtype=np.int32),
                    allow_pickle=False,
                )
                windows.append(
                    {
                        "window_id": window_id,
                        "role": role,
                        "prediction_positions": 4,
                        "token_ids_sha256": sha256_file(token_path),
                    }
                )
            (panel_dir / "panel.json").write_text(
                json.dumps(
                    {
                        "schema": "quant-pipeline.glm53-token-panel.v1",
                        "sealed_corpus_sha256": "c" * 64,
                        "windows": windows,
                    }
                )
            )

            _, _, records = _load_panel_role(panel_dir, "selection")
            self.assertEqual([record["index"] for record in records], [0, 1])
            self.assertEqual(
                [record["window_id"] for record in records],
                ["selection-0000", "selection-0001"],
            )
            self.assertEqual(_load_suite_token_ids(panel_dir, records[1]).numel(), 5)

    def test_teacher_import_references_external_token_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            arrays = dataset / "arrays"
            logits_dir = dataset / "logits"
            output = root / "capture"
            arrays.mkdir(parents=True)
            logits_dir.mkdir()
            records = []
            for index in range(2):
                token_ids = np.arange(index, index + 5, dtype=np.int32)
                token_path = arrays / f"final-{index:04d}.tokens.npy"
                np.save(token_path, token_ids, allow_pickle=False)
                logit_path = logits_dir / f"window-{index:04d}.safetensors"
                save_file(
                    {"logits": torch.randn(4, 7, dtype=torch.float32)},
                    str(logit_path),
                )
                records.append(
                    {
                        "window_id": f"final-{index:04d}",
                        "domain": "synthetic",
                        "document_id": f"doc-{index}",
                        "path": str(logit_path.relative_to(dataset)),
                        "bytes": logit_path.stat().st_size,
                        "sha256": sha256_file(logit_path),
                        "prediction_positions": 4,
                        "token_ids_sha256": sha256_file(token_path),
                    }
                )
            manifest = {
                "schema": "quant-pipeline.glm53-bf16-teacher-logits-dataset.v1",
                "source_model": "synthetic/glm53",
                "model_revision": "5" * 40,
                "vocab_size": 7,
                "dataset_sha256": "d" * 64,
                "teacher_capture_receipt_sha256": "c" * 64,
                "token_panel_receipt_sha256": "p" * 64,
                "logit_files": records,
            }
            (dataset / "dataset-manifest.json").write_text(json.dumps(manifest))
            (dataset / "config.json").write_text("{}\n")
            self.assertEqual(
                import_teacher_main(
                    [
                        "--dataset-dir",
                        str(dataset),
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            imported = json.loads((output / "manifest.json").read_text())
            self.assertEqual(imported["storage_dtype"], "float32")
            self.assertEqual(len(imported["windows"]), 2)
            self.assertIn("input_ids_file", imported["windows"][0])
            self.assertNotIn("input_ids", load_file(output / "logits_0000.safetensors"))

    def test_hidden_finalize_head_export_and_f32_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw" / "request"
            raw.mkdir(parents=True)
            first = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            second = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
            for start, end, tensor in ((0, 3, first), (3, 5, second)):
                save_file(
                    {"hidden_states": tensor},
                    str(raw / f"hidden.rows-{start:06d}-{end:06d}.safetensors"),
                    metadata={"semantic_point": "after_final_rmsnorm_before_lm_head"},
                )
            chunks = _validate_chunks(raw, expected_rows=5, expected_width=4)
            hidden_dir = root / "hidden"
            hidden_dir.mkdir()
            token_ids = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64)
            hidden_path = hidden_dir / "hidden_0000.safetensors"
            record = _finalize_window(
                chunks,
                output_path=hidden_path,
                token_ids=token_ids,
                index=0,
                hidden_width=4,
            )
            hidden_manifest = {
                "schema": "quant-toolkit.prefill-hidden.v1",
                "role": "candidate",
                "run_label": "synthetic",
                "model": "synthetic/glm53",
                "storage_dtype": "bfloat16",
                "hidden_width": 4,
                "token_sha256": canonical_json_sha256(
                    [record["input_ids_canonical_sha256"]]
                ),
                "runtime_manifest": "runtime.json",
                "runtime_manifest_file_sha256": "r" * 64,
                "windows": [record],
            }
            hidden_manifest["manifest_sha256"] = canonical_json_sha256(hidden_manifest)
            (hidden_dir / "manifest.json").write_text(
                json.dumps(hidden_manifest, indent=2, sort_keys=True) + "\n"
            )

            model_dir = root / "model"
            model_dir.mkdir()
            weight = torch.randn(7, 4, dtype=torch.bfloat16)
            shard_path = model_dir / "model-00001-of-00001.safetensors"
            save_file({"lm_head.weight": weight}, str(shard_path))
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"lm_head.weight": shard_path.name}})
            )
            (model_dir / "config.json").write_text("{}\n")
            head_dir = root / "head"
            self.assertEqual(
                export_head_main(
                    [
                        "--model-dir",
                        str(model_dir),
                        "--model-revision",
                        "5" * 40,
                        "--output-dir",
                        str(head_dir),
                        "--hidden-width",
                        "4",
                        "--vocab-size",
                        "7",
                    ]
                ),
                0,
            )

            logits_dir = root / "logits"
            self.assertEqual(
                replay_hidden_main(
                    [
                        "--hidden-dir",
                        str(hidden_dir),
                        "--lm-head-dir",
                        str(head_dir),
                        "--output-dir",
                        str(logits_dir),
                        "--device",
                        "cpu",
                        "--row-chunk",
                        "2",
                    ]
                ),
                0,
            )
            replayed = load_file(logits_dir / "logits_0000.safetensors")
            expected_hidden = torch.cat((first, second), dim=0)[:4]
            expected = F.linear(expected_hidden, weight).float()
            torch.testing.assert_close(replayed["logits"], expected)
            torch.testing.assert_close(replayed["input_ids"], token_ids.int())
            self.assertEqual(replayed["logits"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
