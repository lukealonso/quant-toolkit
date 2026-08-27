import json
import tempfile
import unittest
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file
from record_glm53_routed_resume import main


class Glm53RoutedResumeTests(unittest.TestCase):
    def test_binds_verified_contiguous_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "fit"
            evidence.mkdir()
            runtime_path = root / "runtime.json"
            previous_tool = root / "previous.py"
            resume_tool = root / "resume.py"
            verification_path = root / "verification.json"
            previous_tool.write_text("old\n", encoding="utf-8")
            resume_tool.write_text("new\n", encoding="utf-8")
            runtime = {
                "tooling": {"capture_tool_sha256": sha256_file(previous_tool)}
            }
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            manifest = {
                "schema": "quant-toolkit.glm53-routed-evidence.v1",
                "runtime_manifest_file_sha256": sha256_file(runtime_path),
                "windows": [{"window_id": "fit-0004"}],
            }
            manifest["manifest_sha256"] = canonical_json_sha256(manifest)
            manifest_path = evidence / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verification = {
                "schema": "quant-toolkit.glm53-routed-evidence-verification.v1",
                "manifest_sha256": manifest["manifest_sha256"],
            }
            verification["receipt_sha256"] = canonical_json_sha256(verification)
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            receipt_path = root / "resume.json"

            self.assertEqual(
                main(
                    [
                        "--evidence-dir",
                        str(evidence),
                        "--runtime-manifest",
                        str(runtime_path),
                        "--previous-tool",
                        str(previous_tool),
                        "--resume-tool",
                        str(resume_tool),
                        "--migration-verification",
                        str(verification_path),
                        "--first-window",
                        "fit-0005",
                        "--last-window",
                        "fit-0009",
                        "--storage-root",
                        "/run",
                        "--transport",
                        "NFS 4.2",
                        "--storage-backend",
                        "ZFS",
                        "--reason",
                        "test migration",
                        "--quant-toolkit-commit",
                        "deadbeef",
                        "--receipt",
                        str(receipt_path),
                    ]
                ),
                0,
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["completed_last_window"], "fit-0004")
            self.assertEqual(receipt["resume_first_window"], "fit-0005")
            self.assertFalse(receipt["capture_semantics_changed"])


if __name__ == "__main__":
    unittest.main()
