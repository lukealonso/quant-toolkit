import json
import tempfile
import unittest
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file
from truncate_glm53_routed_evidence import main


class Glm53RoutedRecoveryTests(unittest.TestCase):
    def test_quarantines_corrupt_suffix_and_reseals_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "evidence"
            windows = evidence / "windows"
            windows.mkdir(parents=True)
            records = []
            for index in range(3):
                window_id = f"fit-{index:04d}"
                window_dir = windows / window_id
                window_dir.mkdir()
                layer_path = window_dir / "layer-0003.safetensors"
                layer_path.write_bytes(f"valid-{index}".encode())
                records.append(
                    {
                        "window_id": window_id,
                        "bytes": layer_path.stat().st_size,
                        "layers": [
                            {
                                "layer_id": 3,
                                "file": layer_path.name,
                                "sha256": sha256_file(layer_path),
                            }
                        ],
                    }
                )
            (windows / "fit-0001" / "layer-0003.safetensors").write_bytes(b"corrupt")
            manifest = {
                "schema": "quant-toolkit.glm53-routed-evidence.v1",
                "windows": records,
                "total_bytes": sum(record["bytes"] for record in records),
            }
            manifest["manifest_sha256"] = canonical_json_sha256(manifest)
            (evidence / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            quarantine = root / "quarantine"
            receipt = root / "receipt.json"
            self.assertEqual(
                main(
                    [
                        "--evidence-dir",
                        str(evidence),
                        "--first-invalid",
                        "fit-0001",
                        "--quarantine-dir",
                        str(quarantine),
                        "--reason",
                        "test crash",
                        "--receipt",
                        str(receipt),
                    ]
                ),
                0,
            )
            repaired = json.loads((evidence / "manifest.json").read_text())
            expected_seal = repaired.pop("manifest_sha256")
            self.assertEqual(expected_seal, canonical_json_sha256(repaired))
            self.assertEqual(
                [window["window_id"] for window in repaired["windows"]],
                ["fit-0000"],
            )
            self.assertTrue((windows / "fit-0000").is_dir())
            self.assertFalse((windows / "fit-0001").exists())
            self.assertTrue((quarantine / "fit-0001").is_dir())
            self.assertTrue((quarantine / "fit-0002").is_dir())
            recovery = json.loads(receipt.read_text())
            self.assertEqual(recovery["first_invalid_mismatched_layers"], [3])


if __name__ == "__main__":
    unittest.main()
