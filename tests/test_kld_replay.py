import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_captured_prefill_kld import main as compare_captured_main
from kld_common import (
    canonical_json_sha256,
    sha256_file,
    summarize_kld,
    tokenwise_kld,
)
from replay_captured_prefill_kld import main as replay_captured_main
from replay_prefill_kld import main as replay_main


class KldReplayTests(unittest.TestCase):
    def test_capture_once_compare_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_dir = root / "reference-capture"
            candidate_dir = root / "candidate-capture"
            report_dir = root / "report"
            replay_dir = root / "replay"
            reference_dir.mkdir()
            candidate_dir.mkdir()

            generator = torch.Generator().manual_seed(5310)
            reference = torch.randn(7, 29, generator=generator, dtype=torch.float32)
            candidate = reference + 0.08 * torch.randn(
                7, 29, generator=generator, dtype=torch.float32
            )
            input_ids = torch.tensor([2, 3, 5, 7, 11, 13, 17, 19], dtype=torch.int32)

            def write_capture(
                directory, logits, role, run_label, external_tokens=False
            ):
                path = directory / "logits_0.safetensors"
                tensors = {"logits": logits}
                record = {
                    "index": 0,
                    "file": path.name,
                }
                if external_tokens:
                    token_path = directory / "input_ids_0.npy"
                    np.save(token_path, input_ids.numpy(), allow_pickle=False)
                    record.update(
                        {
                            "input_ids_file": token_path.name,
                            "input_ids_file_sha256": sha256_file(token_path),
                            "input_ids_canonical_sha256": canonical_json_sha256(
                                input_ids.to(torch.int64).tolist()
                            ),
                        }
                    )
                else:
                    tensors["input_ids"] = input_ids
                save_file(tensors, str(path))
                record["sha256"] = sha256_file(path)
                token_sha = canonical_json_sha256(input_ids.to(torch.int64).tolist())
                manifest = {
                    "schema": "quant-toolkit.prefill-logits.v1",
                    "role": role,
                    "run_label": run_label,
                    "model": f"synthetic-{role}",
                    "model_revision": "5" * 40,
                    "storage_dtype": "float32",
                    "token_sha256": token_sha,
                    "engine_request": {"kv_cache_dtype": "bfloat16"},
                    "windows": [record],
                }
                manifest["manifest_sha256"] = canonical_json_sha256(manifest)
                (directory / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )

            write_capture(
                reference_dir,
                reference,
                "canonical",
                "bf16-tp8-bf16-kv",
                external_tokens=True,
            )
            write_capture(candidate_dir, candidate, "candidate", "fp8-tp4-bf16-kv")
            self.assertEqual(
                compare_captured_main(
                    [
                        "--reference-logits",
                        str(reference_dir),
                        "--candidate-logits",
                        str(candidate_dir),
                        "--output-dir",
                        str(report_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                replay_captured_main(
                    [
                        "--reference-logits",
                        str(reference_dir),
                        "--candidate-logits",
                        str(candidate_dir),
                        "--report",
                        str(report_dir),
                        "--output-dir",
                        str(replay_dir),
                    ]
                ),
                0,
            )
            summary = json.loads((report_dir / "summary.json").read_text())
            verification = json.loads((replay_dir / "verification.json").read_text())
            self.assertEqual(summary["schema"], "quant-toolkit.captured-prefill-kld.v1")
            self.assertEqual(
                verification["schema"],
                "quant-toolkit.captured-prefill-kld-verification.v1",
            )
            self.assertIn("p99_9", summary["aggregate"]["kld_nats"])
            self.assertIn("p99_9", verification["aggregate"]["kld_nats"])
            self.assertLessEqual(verification["max_tokenwise_kld_difference"], 1e-12)

    def test_full_artifact_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_dir = root / "reference"
            candidate_dir = root / "candidate"
            replay_dir = root / "replay"
            reference_dir.mkdir()
            candidate_dir.mkdir()

            generator = torch.Generator().manual_seed(5300)
            reference = torch.randn(5, 23, generator=generator, dtype=torch.float32)
            candidate = reference + 0.05 * torch.randn(
                5, 23, generator=generator, dtype=torch.float32
            )
            input_ids = torch.tensor([17, 19, 23, 29, 31, 37], dtype=torch.int32)

            reference_path = reference_dir / "logits_0.safetensors"
            candidate_path = candidate_dir / "candidate_logits_0.safetensors"
            save_file(
                {"logits": reference, "input_ids": input_ids}, str(reference_path)
            )
            save_file(
                {"logits": candidate, "input_ids": input_ids}, str(candidate_path)
            )

            reference_manifest = {
                "schema": "quant-toolkit.prefill-logits.v1",
                "storage_dtype": "float32",
                "windows": [
                    {
                        "index": 0,
                        "file": reference_path.name,
                        "sha256": sha256_file(reference_path),
                    }
                ],
            }
            reference_manifest["manifest_sha256"] = canonical_json_sha256(
                reference_manifest
            )
            (reference_dir / "manifest.json").write_text(
                json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n"
            )

            kld, reference_top1, candidate_top1 = tokenwise_kld(reference, candidate)
            window = summarize_kld(kld, reference_top1, candidate_top1)
            window.update(
                {
                    "index": 0,
                    "candidate_file": candidate_path.name,
                    "candidate_file_sha256": sha256_file(candidate_path),
                }
            )
            tokenwise_path = candidate_dir / "tokenwise.safetensors"
            save_file(
                {
                    "kld_nats": kld,
                    "kld_bits": kld / math.log(2.0),
                    "reference_top1": reference_top1,
                    "candidate_top1": candidate_top1,
                },
                str(tokenwise_path),
            )
            candidate_summary = {
                "schema": "quant-toolkit.prefill-kld.v1",
                "candidate_storage_dtype": "float32",
                "reference_manifest": reference_manifest,
                "windows": [window],
                "aggregate": summarize_kld(kld, reference_top1, candidate_top1),
                "tokenwise_file": tokenwise_path.name,
                "tokenwise_file_sha256": sha256_file(tokenwise_path),
            }
            candidate_summary["summary_sha256"] = canonical_json_sha256(
                candidate_summary
            )
            (candidate_dir / "summary.json").write_text(
                json.dumps(candidate_summary, indent=2, sort_keys=True) + "\n"
            )

            self.assertEqual(
                replay_main(
                    [
                        "--reference-logits",
                        str(reference_dir),
                        "--candidate-logits",
                        str(candidate_dir),
                        "--output-dir",
                        str(replay_dir),
                    ]
                ),
                0,
            )
            verification = json.loads((replay_dir / "verification.json").read_text())
            self.assertEqual(
                verification["schema"],
                "quant-toolkit.prefill-kld-verification.v1",
            )
            self.assertLessEqual(verification["max_tokenwise_kld_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
