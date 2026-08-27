import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from kld_common import canonical_json_sha256, sha256_file
from replay_glm53_routed_moments import replay_moments


class Glm53RoutedMomentTests(unittest.TestCase):
    def test_float64_weighted_gram_matches_direct_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window_dir = root / "windows" / "fit-0000"
            window_dir.mkdir(parents=True)
            hidden = torch.tensor(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                dtype=torch.bfloat16,
            )
            ids = torch.tensor([[1, 0], [0, 2], [2, 1], [1, 2]], dtype=torch.int16)
            weights = torch.tensor(
                [[2.0, 0.5], [1.5, 1.0], [0.25, 2.25], [1.25, 1.25]],
                dtype=torch.float32,
            )
            layer_path = window_dir / "layer-0003.safetensors"
            save_file(
                {
                    "hidden_states": hidden,
                    "router_logits": torch.zeros(4, 3),
                    "topk_ids": ids,
                    "topk_weights": weights,
                },
                layer_path,
            )
            layer_record = {
                "layer_id": 3,
                "file": layer_path.name,
                "sha256": sha256_file(layer_path),
                "bytes": layer_path.stat().st_size,
            }
            manifest = {
                "schema": "quant-toolkit.glm53-routed-evidence.v1",
                "role": "fit",
                "routed_layers": [3],
                "hidden_width": 3,
                "num_experts": 3,
                "windows": [{"window_id": "fit-0000", "layers": [layer_record]}],
            }
            manifest["manifest_sha256"] = canonical_json_sha256(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest))

            tensors, receipt = replay_moments(
                root,
                layer_id=3,
                expert_id=1,
                powers=(0, 1, 2),
                columns=(0, 3),
                device="cpu",
            )
            selected_x = hidden[[0, 2, 3]].double()
            selected_w = torch.tensor([2.0, 2.25, 1.25], dtype=torch.float64)
            for power in (0, 1, 2):
                w = torch.ones_like(selected_w) if power == 0 else selected_w.pow(power)
                expected = selected_x.T @ (selected_x * w.unsqueeze(-1))
                torch.testing.assert_close(tensors[f"gram_p{power}"], expected)
            self.assertEqual(receipt["routed_rows"], 3)


if __name__ == "__main__":
    unittest.main()
