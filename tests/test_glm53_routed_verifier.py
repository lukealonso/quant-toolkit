import json
import tempfile
import unittest
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file
from verify_glm53_routed_evidence import main


class Glm53RoutedVerifierTests(unittest.TestCase):
    def test_verifies_all_bound_files_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "fit"
            panel = root / "panel"
            (evidence / "windows" / "fit-0000").mkdir(parents=True)
            (panel / "arrays").mkdir(parents=True)
            runtime = root / "runtime.json"
            runtime.write_text("{}\n", encoding="utf-8")
            layer = evidence / "windows" / "fit-0000" / "layer-0003.safetensors"
            layer.write_bytes(b"layer evidence")
            token = panel / "arrays" / "fit-0000.tokens.npy"
            token.write_bytes(b"token evidence")
            manifest = {
                "schema": "quant-toolkit.glm53-routed-evidence.v1",
                "runtime_manifest": "/run/runtime.json",
                "runtime_manifest_file_sha256": sha256_file(runtime),
                "routed_layers": [3],
                "windows": [
                    {
                        "window_id": "fit-0000",
                        "token_file": "/panel/arrays/fit-0000.tokens.npy",
                        "token_file_sha256": sha256_file(token),
                        "bytes": layer.stat().st_size,
                        "layers": [
                            {
                                "layer_id": 3,
                                "file": layer.name,
                                "bytes": layer.stat().st_size,
                                "sha256": sha256_file(layer),
                            }
                        ],
                    }
                ],
                "total_bytes": layer.stat().st_size,
            }
            manifest["manifest_sha256"] = canonical_json_sha256(manifest)
            (evidence / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            receipt = root / "verification.json"

            self.assertEqual(
                main(
                    [
                        "--evidence-dir",
                        str(evidence),
                        "--run-root",
                        str(root),
                        "--panel-dir",
                        str(panel),
                        "--expected-windows",
                        "1",
                        "--workers",
                        "2",
                        "--receipt",
                        str(receipt),
                    ]
                ),
                0,
            )
            result = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(result["evidence_file_count"], 1)
            self.assertEqual(result["panel_file_count"], 1)
            self.assertEqual(result["verified_evidence_bytes"], layer.stat().st_size)


if __name__ == "__main__":
    unittest.main()
