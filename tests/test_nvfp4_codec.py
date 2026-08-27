import unittest

import torch
from nvfp4_codec import (
    decode_nvfp4,
    encode_nvfp4,
    optimize_block_scales,
    pack_fp4,
    quantize_nvfp4_with_scales,
    unpack_fp4,
)


class Nvfp4CodecTests(unittest.TestCase):
    def test_pack_order_and_codebook(self):
        values = torch.tensor(
            [
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            ],
            dtype=torch.float32,
        )
        packed = pack_fp4(values)
        self.assertEqual(
            packed.tolist(), [[0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE]]
        )
        torch.testing.assert_close(unpack_fp4(packed), values)

    def test_encode_decode_is_deterministic_and_finite(self):
        weight = torch.linspace(-0.25, 0.25, 64, dtype=torch.float32).reshape(4, 16)
        first = encode_nvfp4(weight)
        second = encode_nvfp4(weight)
        for left, right in zip(first, second, strict=True):
            self.assertTrue(torch.equal(left, right))
        decoded = decode_nvfp4(*first)
        self.assertEqual(decoded.shape, weight.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_secondary_scale_override_is_preserved(self):
        weight = torch.linspace(-0.1, 0.1, 32, dtype=torch.float32).reshape(2, 16)
        override = torch.tensor(1.0e-4, dtype=torch.float32)
        packed, block_scale, scale_2 = encode_nvfp4(weight, scale_2=override)
        self.assertTrue(torch.equal(scale_2, override))
        self.assertTrue(
            torch.isfinite(decode_nvfp4(packed, block_scale, scale_2)).all()
        )

    def test_block_scale_search_never_worsens_diagonal_objective(self):
        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(5, 32, generator=generator)
        coordinate_weights = torch.linspace(0.25, 2.0, 32)
        baseline = encode_nvfp4(weight)
        baseline_decoded = decode_nvfp4(*baseline)
        optimized = optimize_block_scales(
            weight,
            factors=(0.75, 0.875, 1.0, 1.125, 1.25),
            coordinate_weights=coordinate_weights,
        )
        optimized_decoded = optimized[3]
        baseline_loss = (
            (baseline_decoded - weight).square() * coordinate_weights
        ).sum()
        optimized_loss = (
            (optimized_decoded - weight).square() * coordinate_weights
        ).sum()
        self.assertLessEqual(float(optimized_loss), float(baseline_loss))

    def test_block_scale_search_preserves_baseline_on_ties(self):
        weight = torch.zeros(2, 32, dtype=torch.float32)
        _, baseline_scale, _ = encode_nvfp4(weight)
        packed, optimized_scale, scale_2, decoded = optimize_block_scales(
            weight,
            factors=(0.75, 1.0, 1.25),
        )
        self.assertTrue(torch.equal(optimized_scale, baseline_scale))
        self.assertTrue(torch.equal(decoded, weight))
        self.assertTrue(
            torch.equal(decode_nvfp4(packed, optimized_scale, scale_2), decoded)
        )

    def test_explicit_scale_quantization_matches_baseline(self):
        generator = torch.Generator().manual_seed(11)
        weight = torch.randn(4, 32, generator=generator)
        expected_packed, block_scale, scale_2 = encode_nvfp4(weight)
        actual_packed, actual_decoded = quantize_nvfp4_with_scales(
            weight, block_scale, scale_2
        )
        self.assertTrue(torch.equal(actual_packed, expected_packed))
        self.assertTrue(
            torch.equal(actual_decoded, decode_nvfp4(*encode_nvfp4(weight)))
        )

    def test_rejects_invalid_layout(self):
        with self.assertRaises(ValueError):
            encode_nvfp4(torch.ones(3, 15))


if __name__ == "__main__":
    unittest.main()
