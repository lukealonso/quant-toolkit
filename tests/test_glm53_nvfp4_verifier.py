import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from verify_glm53_nvfp4_checkpoint import main as verify_main


class Glm53Nvfp4VerifierTests(unittest.TestCase):
    def _write_checkpoint(self, directory: Path, tensors: dict, config: dict) -> None:
        directory.mkdir()
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, directory / shard)
        index = {
            "metadata": {},
            "weight_map": {key: shard for key in sorted(tensors)},
        }
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index) + "\n"
        )
        (directory / "config.json").write_text(json.dumps(config) + "\n")

    def _fixtures(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        candidate = root / "candidate"
        config = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "first_k_dense_replace": 1,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
            },
        }
        source_tensors = {
            "model.language_model.norm.weight": torch.ones(16),
            "model.language_model.layers.0.hc_attn_base": torch.ones(2),
            "model.language_model.layers.0.self_attn.q_conv1d.weight": torch.ones((2, 4)),
            "model.language_model.layers.0.self_attn.k_conv1d.weight": torch.ones((2, 4)),
            "model.language_model.layers.0.self_attn.v_conv1d.weight": torch.ones((2, 4)),
        }
        for layer in (1, 2):
            for expert in range(2):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = (
                        f"model.language_model.layers.{layer}.mlp.experts."
                        f"{expert}.{projection}.weight"
                    )
                    source_tensors[key] = torch.ones((16, 16), dtype=torch.bfloat16)
        self._write_checkpoint(source, source_tensors, config)

        candidate_tensors = {
            "model.language_model.layers.0.attn_hc.base": torch.ones(2),
            "model.language_model.layers.0.self_attn.conv1d.weight": torch.ones((6, 4)),
        }
        for key, tensor in source_tensors.items():
            if key.startswith("model.language_model.layers.0."):
                continue
            if ".layers.1.mlp.experts." not in key:
                candidate_tensors[key] = tensor
                continue
            prefix = key.removesuffix(".weight")
            candidate_tensors[key] = torch.zeros((16, 8), dtype=torch.uint8)
            candidate_tensors[prefix + ".weight_scale"] = torch.ones(
                (16, 1), dtype=torch.float8_e4m3fn
            )
            candidate_tensors[prefix + ".weight_scale_2"] = torch.tensor(0.25)
            candidate_tensors[prefix + ".input_scale"] = torch.tensor(0.125)
        candidate_config = dict(config)
        candidate_config["quantization_config"] = {"quant_algo": "NVFP4"}
        self._write_checkpoint(candidate, candidate_tensors, candidate_config)
        return source, candidate

    def test_exact_routed_coverage_and_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, candidate = self._fixtures(root)
            report_path = root / "report.json"
            self.assertEqual(
                verify_main(
                    [
                        "--source-model",
                        str(source),
                        "--candidate-model",
                        str(candidate),
                        "--source-revision",
                        "b" * 40,
                        "--candidate-name",
                        "synthetic",
                        "--output",
                        str(report_path),
                    ]
                ),
                0,
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["coverage"]["quantized_weight_tensors"], 6)
            self.assertEqual(report["coverage"]["quantized_parameters"], 1536)
            self.assertTrue(report["coverage"]["gate_up_weight_scale_2_tied"])
            self.assertTrue(report["coverage"]["gate_up_input_scale_tied"])
            self.assertEqual(
                report["coverage"]["input_scale_storage"],
                "static_f32_per_expert_projection",
            )
            self.assertEqual(len(report["report_sha256"]), 64)

    def test_untied_gate_up_scale_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, candidate = self._fixtures(root)
            shard = candidate / "model-00001-of-00001.safetensors"
            from safetensors.torch import load_file

            tensors = load_file(shard)
            key = "model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale_2"
            tensors[key] = torch.tensor(0.5)
            save_file(tensors, shard)
            with self.assertRaisesRegex(ValueError, "not tied"):
                verify_main(
                    [
                        "--source-model",
                        str(source),
                        "--candidate-model",
                        str(candidate),
                        "--source-revision",
                        "b" * 40,
                        "--candidate-name",
                        "synthetic",
                        "--output",
                        str(root / "report.json"),
                    ]
                )

    def test_missing_w4a4_input_scale_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, candidate = self._fixtures(root)
            shard = candidate / "model-00001-of-00001.safetensors"
            from safetensors.torch import load_file

            tensors = load_file(shard)
            tensors.pop(
                "model.language_model.layers.1.mlp.experts.0.down_proj.input_scale"
            )
            save_file(tensors, shard)
            index_path = candidate / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"] = {key: shard.name for key in sorted(tensors)}
            index_path.write_text(json.dumps(index) + "\n")
            with self.assertRaisesRegex(ValueError, "keyset"):
                verify_main(
                    [
                        "--source-model",
                        str(source),
                        "--candidate-model",
                        str(candidate),
                        "--source-revision",
                        "b" * 40,
                        "--candidate-name",
                        "synthetic",
                        "--output",
                        str(root / "report.json"),
                    ]
                )

    def test_bf16_precision_reserve_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, candidate = self._fixtures(root)
            from safetensors.torch import load_file

            source_tensors = load_file(source / "model-00001-of-00001.safetensors")
            candidate_shard = candidate / "model-00001-of-00001.safetensors"
            tensors = load_file(candidate_shard)
            for expert in range(2):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = (
                        "model.language_model.layers.1.mlp.experts."
                        f"{expert}.{projection}.weight"
                    )
                    tensors[key] = source_tensors[key]
                    prefix = key.removesuffix(".weight")
                    for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
                        tensors.pop(prefix + suffix)
            save_file(tensors, candidate_shard)
            index_path = candidate / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"] = {key: candidate_shard.name for key in sorted(tensors)}
            index_path.write_text(json.dumps(index) + "\n")
            config_path = candidate / "config.json"
            config = json.loads(config_path.read_text())
            config["quantization_config"]["ignore"] = [
                "model.language_model.layers.1.mlp.experts*"
            ]
            config_path.write_text(json.dumps(config) + "\n")
            report_path = root / "report.json"

            self.assertEqual(
                verify_main(
                    [
                        "--source-model",
                        str(source),
                        "--candidate-model",
                        str(candidate),
                        "--source-revision",
                        "b" * 40,
                        "--candidate-name",
                        "synthetic-reserve",
                        "--bf16-reserve-layers",
                        "1",
                        "--output",
                        str(report_path),
                    ]
                ),
                0,
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["coverage"]["quantized_weight_tensors"], 0)
            self.assertEqual(report["coverage"]["bf16_reserve_weight_tensors"], 6)
            self.assertEqual(report["coverage"]["bf16_reserve_layers"], [1])


if __name__ == "__main__":
    unittest.main()
