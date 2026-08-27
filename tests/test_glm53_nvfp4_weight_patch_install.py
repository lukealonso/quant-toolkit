import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from install_glm53_nvfp4_weight_patches import _hardlink_clone, _rewrite_shards


class Glm53Nvfp4WeightPatchInstallTests(unittest.TestCase):
    def test_hardlink_clone_then_atomic_shard_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            shard = baseline / "model-00001-of-00001.safetensors"
            save_file(
                {
                    "replace": torch.tensor([1.0, 2.0]),
                    "preserve": torch.tensor([3.0, 4.0]),
                },
                shard,
            )
            patch = root / "layer-0003.safetensors"
            save_file({"replace": torch.tensor([5.0, 6.0])}, patch)

            _hardlink_clone(baseline, candidate)
            cloned = candidate / shard.name
            self.assertEqual(shard.stat().st_ino, cloned.stat().st_ino)
            records = _rewrite_shards(
                candidate,
                {"replace": shard.name, "preserve": shard.name},
                {"replace": patch},
            )
            self.assertEqual(len(records), 1)
            self.assertNotEqual(shard.stat().st_ino, cloned.stat().st_ino)
            observed = load_file(cloned)
            torch.testing.assert_close(observed["replace"], torch.tensor([5.0, 6.0]))
            torch.testing.assert_close(observed["preserve"], torch.tensor([3.0, 4.0]))
            original = load_file(shard)
            torch.testing.assert_close(original["replace"], torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
