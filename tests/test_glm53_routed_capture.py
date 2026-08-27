import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from capture_glm53_routed_suite import EXPECTED_ROUTED_LAYERS, _finalize_pass


class Glm53RoutedCaptureTests(unittest.TestCase):
    def test_finalize_requires_and_validates_all_routed_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_pass = root / "raw" / "pass-00000000"
            raw_pass.mkdir(parents=True)
            rows = 4
            hidden_width = 4
            num_experts = 6
            top_k = 2
            for layer_id in EXPECTED_ROUTED_LAYERS:
                ids = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 3]], dtype=torch.int16)
                weights = torch.tensor(
                    [[1.0, 1.5], [0.5, 2.0], [1.25, 1.25], [2.25, 0.25]],
                    dtype=torch.float32,
                )
                save_file(
                    {
                        "hidden_states": torch.randn(rows, hidden_width).bfloat16(),
                        "router_logits": torch.randn(rows, num_experts).float(),
                        "topk_ids": ids,
                        "topk_weights": weights,
                    },
                    raw_pass / f"layer-{layer_id:04d}.safetensors",
                    metadata={
                        "layer_id": str(layer_id),
                        "hidden_semantic_point": (
                            "input_to_routed_moe_before_sequence_parallel_chunk"
                        ),
                        "router_logit_semantic_point": "pre_sigmoid_gate_linear_output",
                        "route_semantic_point": "exact_vllm_router_selection",
                    },
                )

            destination = root / "windows" / "fit-0000"
            records = _finalize_pass(
                raw_pass,
                destination,
                token_ids=torch.arange(rows).numpy(),
                hidden_width=hidden_width,
                num_experts=num_experts,
                top_k=top_k,
            )
            self.assertEqual(len(records), len(EXPECTED_ROUTED_LAYERS))
            self.assertFalse(raw_pass.exists())
            self.assertTrue(destination.is_dir())
            self.assertTrue(all(record["sha256"] for record in records))


if __name__ == "__main__":
    unittest.main()
