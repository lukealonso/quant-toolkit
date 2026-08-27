import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from optimize_glm53_nvfp4_routed_weights import (
    _weighted_coordinate_second_moment,
    choose_complete_expert_combination,
)


class Glm53Nvfp4RoutedWeightOptimizationTests(unittest.TestCase):
    def test_route_power_two_coordinate_moment(self):
        rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        routes = torch.tensor([1.0, 2.0])
        observed = _weighted_coordinate_second_moment(rows, routes, 2)
        expected = torch.tensor(
            [(1.0 + 4.0 * 9.0) / 5.0, (4.0 + 4.0 * 16.0) / 5.0]
        )
        torch.testing.assert_close(observed, expected)

    def test_complete_function_can_select_optimized_tuple(self):
        x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        source_gate = torch.eye(2)
        source_up = torch.eye(2)
        source_down = torch.eye(2)
        source_z = torch.nn.functional.silu(x @ source_gate.T) * (x @ source_up.T)
        bad = torch.zeros_like(source_gate)
        choice, scores = choose_complete_expert_combination(
            x=x,
            source_down_inputs=source_z,
            route_weights=torch.ones(2),
            source_down_weight=source_down,
            gate_choices=(bad, source_gate),
            up_choices=(bad, source_up),
            down_choices=(bad, source_down),
            swiglu_limit=10.0,
        )
        self.assertEqual(choice, (1, 1, 1))
        self.assertEqual(scores[-1], 0.0)

    def test_stock_tuple_wins_ties_deterministically(self):
        identity = torch.eye(2)
        x = torch.ones(2, 2)
        source_z = torch.nn.functional.silu(x) * x
        choice, _scores = choose_complete_expert_combination(
            x=x,
            source_down_inputs=source_z,
            route_weights=torch.ones(2),
            source_down_weight=identity,
            gate_choices=(identity, identity),
            up_choices=(identity, identity),
            down_choices=(identity, identity),
            swiglu_limit=10.0,
        )
        self.assertEqual(choice, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
