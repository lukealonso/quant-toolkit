import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from install_glm53_precision_reserve import (
    _rewrite_precision_reserve,
    _sidecars,
    _tensor_inventory,
)


class Glm53PrecisionReserveInstallTests(unittest.TestCase):
    def test_restores_bf16_weight_and_removes_nvfp4_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            candidate = root / "candidate"
            source.mkdir()
            candidate.mkdir()
            key = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
            sidecars = _sidecars(key)
            shard = "model.safetensors"
            save_file({key: torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}, source / shard)
            save_file(
                {
                    key: torch.arange(4, dtype=torch.uint8).reshape(2, 2),
                    sidecars[0]: torch.ones(1, dtype=torch.float8_e4m3fn),
                    sidecars[1]: torch.ones(1, dtype=torch.float32),
                    sidecars[2]: torch.ones(1, dtype=torch.float32),
                    "preserved": torch.tensor([7.0], dtype=torch.bfloat16),
                },
                candidate / shard,
            )
            source_map = {key: shard}
            candidate_map = {
                key: shard,
                sidecars[0]: shard,
                sidecars[1]: shard,
                sidecars[2]: shard,
                "preserved": shard,
            }

            records = _rewrite_precision_reserve(
                source, candidate, source_map, candidate_map, {key}
            )

            observed = load_file(candidate / shard)
            self.assertEqual(set(observed), {key, "preserved"})
            self.assertEqual(observed[key].dtype, torch.bfloat16)
            torch.testing.assert_close(
                observed[key], torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
            )
            self.assertEqual(set(candidate_map), {key, "preserved"})
            self.assertEqual(records[0]["restored_bf16_weights"], 1)
            self.assertEqual(records[0]["removed_nvfp4_sidecars"], 3)
            self.assertEqual(_tensor_inventory(candidate, candidate_map), (2, 18))


if __name__ == "__main__":
    unittest.main()
