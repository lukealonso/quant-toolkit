import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from kld_common import dense_prompt_logits, summarize_kld, tokenwise_kld
from replay_prefill_kld import independent_tokenwise_kld


class KldCommonTests(unittest.TestCase):
    def test_identical_logits_are_zero(self):
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]])
        values, reference_top1, candidate_top1 = tokenwise_kld(logits, logits)
        self.assertTrue(torch.allclose(values, torch.zeros_like(values), atol=1e-12))
        summary = summarize_kld(values, reference_top1, candidate_top1)
        self.assertEqual(summary["direction"], "KL(reference||candidate)")
        self.assertEqual(summary["top1_agreement"], 1.0)
        self.assertEqual(summary["kld_nats"]["p99_9"], 0.0)

    def test_known_bernoulli_kld_and_bits(self):
        reference = torch.log(torch.tensor([[0.75, 0.25]], dtype=torch.float64))
        candidate = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
        values, reference_top1, candidate_top1 = tokenwise_kld(reference, candidate)
        expected_nats = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
        self.assertAlmostEqual(float(values[0]), expected_nats, places=12)
        summary = summarize_kld(values, reference_top1, candidate_top1)
        self.assertAlmostEqual(
            summary["kld_bits"]["mean"], expected_nats / math.log(2.0), places=12
        )

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            tokenwise_kld(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_dense_prompt_logits_from_patched_runtime(self):
        raw = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        output = SimpleNamespace(prompt_logits=raw)
        dense, are_log_probs = dense_prompt_logits(output, True, 2, 4)
        self.assertFalse(are_log_probs)
        self.assertTrue(torch.equal(dense, raw[:2, :4]))

    def test_dense_prompt_logits_from_flat_runtime(self):
        prompt_logprobs = SimpleNamespace(
            start_indices=[0, 0, 3],
            end_indices=[0, 3, 6],
            token_ids=[0, 1, 2, 0, 1, 2],
            logprobs=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        )
        output = SimpleNamespace(prompt_logprobs=prompt_logprobs)
        dense, are_log_probs = dense_prompt_logits(output, False, 2, 3)
        self.assertTrue(are_log_probs)
        expected = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        self.assertTrue(torch.equal(dense, expected))

    def test_independent_numpy_replay_matches_torch_producer(self):
        generator = torch.Generator().manual_seed(53)
        reference = torch.randn(17, 257, generator=generator, dtype=torch.float32)
        candidate = reference + 0.1 * torch.randn(
            17, 257, generator=generator, dtype=torch.float32
        )
        produced, reference_top1, candidate_top1 = tokenwise_kld(
            reference, candidate, chunk_rows=3
        )
        replayed, replay_reference_top1, replay_candidate_top1 = (
            independent_tokenwise_kld(
                reference.numpy(), candidate.numpy(), chunk_rows=5
            )
        )
        self.assertTrue(np.allclose(produced.numpy(), replayed, rtol=0.0, atol=1e-12))
        self.assertTrue(np.array_equal(reference_top1.numpy(), replay_reference_top1))
        self.assertTrue(np.array_equal(candidate_top1.numpy(), replay_candidate_top1))


if __name__ == "__main__":
    unittest.main()
