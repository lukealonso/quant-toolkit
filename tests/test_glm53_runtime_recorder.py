import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from record_glm53_candidate_runtime import _runtime_fields


class Glm53RuntimeRecorderTests(unittest.TestCase):
    def test_runtime_fields_come_from_live_command(self):
        command = [
            "/model",
            "--tensor-parallel-size",
            "4",
            "--decode-context-parallel-size",
            "1",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.95",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "2048",
            "--no-enable-prefix-caching",
            "--enforce-eager",
        ]

        self.assertEqual(
            _runtime_fields(command),
            {
                "tensor_parallel_size": 4,
                "decode_context_parallel_size": 1,
                "kv_cache_requested": "fp8",
                "kv_cache_effective": "fp8_ds_mla",
                "max_model_len": 4096,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 2048,
                "gpu_memory_utilization": 0.95,
                "prefix_caching": False,
                "enforce_eager": True,
            },
        )

    def test_prefix_cache_and_non_eager_are_detected(self):
        command = [
            "/model",
            "--tensor-parallel-size",
            "4",
            "--decode-context-parallel-size",
            "1",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            "524288",
            "--max-num-seqs",
            "30",
            "--max-num-batched-tokens",
            "8192",
            "--enable-prefix-caching",
        ]

        fields = _runtime_fields(command)
        self.assertTrue(fields["prefix_caching"])
        self.assertFalse(fields["enforce_eager"])
        self.assertEqual(fields["max_num_seqs"], 30)

    def test_missing_required_runtime_flag_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "--kv-cache-dtype"):
            _runtime_fields(
                [
                    "/model",
                    "--tensor-parallel-size",
                    "4",
                    "--decode-context-parallel-size",
                    "1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
