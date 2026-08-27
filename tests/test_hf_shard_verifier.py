import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from verify_hf_shards import main as verify_main


class HfShardVerifierTests(unittest.TestCase):
    def test_local_hash_verification_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            shards = []
            for index, payload in enumerate((b"glm" * 17, b"flash" * 23), start=1):
                name = f"model-{index:05d}-of-00002.safetensors"
                (snapshot / name).write_bytes(payload)
                shards.append(
                    {
                        "path": name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            manifest = {
                "schema": "quant-toolkit.hf-lfs-shard-manifest.v1",
                "repo": "synthetic/glm",
                "revision": "5" * 40,
                "shard_count": len(shards),
                "total_shard_bytes": sum(item["bytes"] for item in shards),
                "shards": shards,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            report_path = root / "verification.json"
            arguments = [
                "--manifest",
                str(manifest_path),
                "--directory",
                str(snapshot),
                "--output",
                str(report_path),
            ]
            self.assertEqual(verify_main(arguments), 0)
            self.assertEqual(verify_main(arguments), 0)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["verified_count"], 2)
            self.assertEqual(report["verified_bytes"], manifest["total_shard_bytes"])
            self.assertEqual(
                [item["observed_sha256"] for item in report["verified"]],
                [item["sha256"] for item in shards],
            )

    def test_size_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            name = "model-00001-of-00001.safetensors"
            (snapshot / name).write_bytes(b"wrong")
            manifest = {
                "schema": "quant-toolkit.hf-lfs-shard-manifest.v1",
                "repo": "synthetic/glm",
                "revision": "5" * 40,
                "shard_count": 1,
                "total_shard_bytes": 7,
                "shards": [
                    {
                        "path": name,
                        "bytes": 7,
                        "sha256": hashlib.sha256(b"correct").hexdigest(),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                verify_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--directory",
                        str(snapshot),
                        "--mode",
                        "size",
                        "--output",
                        str(root / "verification.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
