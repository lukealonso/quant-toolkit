import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from streaming_loader import (
    StreamingModelLoader,
    _checkpoint_key_to_model_key,
    _glm5_next_conv_target_key,
)


class _Leaf(nn.Module):
    pass


def _meta_parameter(shape):
    return nn.Parameter(
        torch.empty(shape, device="meta", dtype=torch.bfloat16),
        requires_grad=False,
    )


def _synthetic_glm_layer():
    layer = _Leaf()
    layer.attn_hc = _Leaf()
    layer.attn_hc.fn = _meta_parameter((2, 3))
    layer.attn_hc.base = _meta_parameter((2,))
    layer.attn_hc.scale = _meta_parameter((1,))
    layer.ffn_hc = _Leaf()
    layer.ffn_hc.fn = _meta_parameter((2, 3))
    layer.ffn_hc.base = _meta_parameter((2,))
    layer.ffn_hc.scale = _meta_parameter((1,))
    layer.self_attn = _Leaf()
    layer.self_attn.forget_gate = _Leaf()
    layer.self_attn.forget_gate.f_a_proj = _Leaf()
    layer.self_attn.forget_gate.f_a_proj.weight = _meta_parameter((2, 3))
    layer.self_attn.forget_gate.f_b_proj = _Leaf()
    layer.self_attn.forget_gate.f_b_proj.weight = _meta_parameter((4, 2))
    layer.self_attn.forget_gate.dt_bias = _meta_parameter((4,))
    layer.self_attn.forget_gate.A_log = _meta_parameter((2,))
    layer.self_attn.conv1d = _Leaf()
    layer.self_attn.conv1d.weight = _meta_parameter((6, 1, 2))
    return layer


def test_glm53_checkpoint_key_renames_match_transformers_table():
    prefix = "model.language_model.layers.3."
    assert _checkpoint_key_to_model_key(prefix + "hc_attn_fn") == prefix + "attn_hc.fn"
    assert _checkpoint_key_to_model_key(prefix + "hc_ffn_scale") == prefix + "ffn_hc.scale"
    assert _checkpoint_key_to_model_key(prefix + "self_attn.f_a_proj.weight") == (
        prefix + "self_attn.forget_gate.f_a_proj.weight"
    )
    assert _checkpoint_key_to_model_key(prefix + "self_attn.A_log") == (
        prefix + "self_attn.forget_gate.A_log"
    )
    assert _glm5_next_conv_target_key(prefix + "self_attn.q_conv1d.weight") == (
        prefix + "self_attn.conv1d.weight",
        "q",
    )


def test_glm53_layer_materialization_renames_and_fuses_qkv_conv():
    prefix = "model.language_model.layers.0."
    tensors = {
        prefix + "hc_attn_fn": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        prefix + "hc_attn_base": torch.arange(2, dtype=torch.float32),
        prefix + "hc_attn_scale": torch.ones(1),
        prefix + "hc_ffn_fn": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10,
        prefix + "hc_ffn_base": torch.arange(2, dtype=torch.float32) + 10,
        prefix + "hc_ffn_scale": torch.ones(1) * 2,
        prefix + "self_attn.f_a_proj.weight": torch.ones((2, 3)),
        prefix + "self_attn.f_b_proj.weight": torch.ones((4, 2)) * 2,
        prefix + "self_attn.dt_bias": torch.arange(4, dtype=torch.float32),
        prefix + "self_attn.A_log": torch.arange(2, dtype=torch.float32),
        prefix + "self_attn.q_conv1d.weight": torch.ones((2, 1, 2)),
        prefix + "self_attn.k_conv1d.weight": torch.ones((2, 1, 2)) * 2,
        prefix + "self_attn.v_conv1d.weight": torch.ones((2, 1, 2)) * 3,
    }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shard = "model.safetensors"
        save_file(tensors, root / shard)
        loader = StreamingModelLoader.__new__(StreamingModelLoader)
        loader.snapshot_dir = str(root)
        loader.weight_map = {key: shard for key in tensors}
        loader._layer_prefix = "model.language_model.layers."
        layer = _synthetic_glm_layer()

        loader._materialize_layer_module(layer, 0, "cpu")

        assert not any(t.device.type == "meta" for t in layer.parameters())
        assert layer.attn_hc.base.dtype == torch.float32
        assert layer.self_attn.forget_gate.A_log.dtype == torch.float32
        assert torch.equal(layer.attn_hc.fn, tensors[prefix + "hc_attn_fn"])
        assert torch.equal(
            layer.self_attn.forget_gate.f_b_proj.weight,
            tensors[prefix + "self_attn.f_b_proj.weight"],
        )
        expected_conv = torch.cat(
            [
                tensors[prefix + "self_attn.q_conv1d.weight"],
                tensors[prefix + "self_attn.k_conv1d.weight"],
                tensors[prefix + "self_attn.v_conv1d.weight"],
            ],
            dim=0,
        )
        assert torch.equal(layer.self_attn.conv1d.weight, expected_conv)
