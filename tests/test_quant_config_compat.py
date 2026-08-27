import copy

import pytest

from quant_config_compat import compose_quant_config


OVERRIDES = {
    "*weight_quantizer": {"enable": False},
    "*input_quantizer": {"enable": False},
    "model.layers.*.mlp.experts.gate_proj.*.weight_quantizer": {"enable": True},
    "model.layers.*.mlp.experts.gate_proj.*.input_quantizer": {"enable": True},
    "*lm_head*": {"enable": False},
}


def test_compose_legacy_mapping_is_fail_closed_and_quantile_enabled():
    base = {
        "algorithm": "max",
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
            "*input_quantizer": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
        },
    }
    original = copy.deepcopy(base)

    result = compose_quant_config(base, OVERRIDES, calibration_method="quantile")

    assert base == original
    assert result["algorithm"] == "quantile"
    assert result["quant_cfg"]["*weight_quantizer"] == {"enable": False}
    assert result["quant_cfg"]["*input_quantizer"] == {"enable": False}
    assert result["quant_cfg"][
        "model.layers.*.mlp.experts.gate_proj.*.weight_quantizer"
    ]["block_sizes"] == {-1: 16}
    expert_input = result["quant_cfg"][
        "model.layers.*.mlp.experts.gate_proj.*.input_quantizer"
    ]
    assert expert_input["calibrator"] == "quantile"
    assert expert_input["num_bits"] == (2, 1)


def test_compose_ordered_rules_preserves_precedence_and_full_cfg():
    base = {
        "algorithm": "max",
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
            },
            {
                "quantizer_name": "*input_quantizer",
                "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
            },
            {"quantizer_name": "*lm_head*", "enable": False},
        ],
    }
    original = copy.deepcopy(base)

    result = compose_quant_config(base, OVERRIDES, calibration_method="max")

    assert base == original
    rules = result["quant_cfg"]
    assert rules[-5:] == [
        {"quantizer_name": "*weight_quantizer", "enable": False},
        {"quantizer_name": "*input_quantizer", "enable": False},
        {
            "quantizer_name": "model.layers.*.mlp.experts.gate_proj.*.weight_quantizer",
            "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
        },
        {
            "quantizer_name": "model.layers.*.mlp.experts.gate_proj.*.input_quantizer",
            "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
        },
        {"quantizer_name": "*lm_head*", "enable": False},
    ]


def test_ordered_rules_reject_unavailable_legacy_quantile_path():
    with pytest.raises(RuntimeError, match="does not provide.*QuantileCalibrator"):
        compose_quant_config(
            {
                "algorithm": "max",
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*input_quantizer",
                        "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
                    },
                ],
            },
            {"target.input_quantizer": {"enable": True}},
            calibration_method="quantile",
        )


def test_missing_format_rule_fails_closed():
    with pytest.raises(KeyError, match="input_quantizer"):
        compose_quant_config(
            {"algorithm": "max", "quant_cfg": []},
            {"target.input_quantizer": {"enable": True}},
            calibration_method="max",
        )
