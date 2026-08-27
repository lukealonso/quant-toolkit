import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from augment_glm53_nvfp4_reservoirs import collect_route_map


class Glm53Nvfp4ConditionalReservoirTests(unittest.TestCase):
    def test_route_map_replays_sample_selection_and_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hidden = torch.tensor(
                [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16
            )
            ids = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=torch.int16)
            weights = torch.tensor(
                [[2.0, 0.5], [1.25, 1.25], [0.75, 1.75], [1.5, 1.0]],
                dtype=torch.float32,
            )
            evidence = root / "layer.safetensors"
            save_file(
                {
                    "hidden_states": hidden,
                    "topk_ids": ids,
                    "topk_weights": weights,
                },
                evidence,
            )
            samples = torch.stack(
                (
                    hidden[[0, 2]],
                    hidden[[0, 1]],
                    hidden[[1, 2]],
                )
            )
            reservoir = root / "reservoir.safetensors"
            save_file(
                {
                    "samples": samples,
                    "sampled_counts": torch.full((3,), 2, dtype=torch.int64),
                },
                reservoir,
            )
            result = collect_route_map(
                evidence_paths=[evidence],
                reservoir_path=reservoir,
                num_experts=3,
                samples_per_expert=2,
                hidden_size=2,
            )
            torch.testing.assert_close(
                result["route_weights"],
                torch.tensor([[2.0, 1.75], [0.5, 1.25], [1.25, 0.75]]),
            )
            torch.testing.assert_close(
                result["source_row_indices"],
                torch.tensor([[0, 2], [0, 1], [1, 2]], dtype=torch.int32),
            )
            self.assertTrue((result["source_file_indices"] == 0).all())

    def test_route_map_fails_closed_on_sample_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "layer.safetensors"
            save_file(
                {
                    "hidden_states": torch.ones(1, 2, dtype=torch.bfloat16),
                    "topk_ids": torch.tensor([[0]], dtype=torch.int16),
                    "topk_weights": torch.tensor([[1.0]], dtype=torch.float32),
                },
                evidence,
            )
            reservoir = root / "reservoir.safetensors"
            save_file(
                {
                    "samples": torch.zeros(1, 1, 2, dtype=torch.bfloat16),
                    "sampled_counts": torch.ones(1, dtype=torch.int64),
                },
                reservoir,
            )
            with self.assertRaisesRegex(RuntimeError, "reservoir replay mismatch"):
                collect_route_map(
                    evidence_paths=[evidence],
                    reservoir_path=reservoir,
                    num_experts=1,
                    samples_per_expert=1,
                    hidden_size=2,
                )


if __name__ == "__main__":
    unittest.main()
