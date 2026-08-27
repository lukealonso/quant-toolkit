"""Compose ModelOpt quantization configs across legacy and ordered rule schemas."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def _find_base_cfg(quant_cfg, quantizer_name: str):
    if isinstance(quant_cfg, Mapping):
        try:
            return copy.deepcopy(quant_cfg[quantizer_name])
        except KeyError as exc:
            raise KeyError(f"base quantization config has no {quantizer_name!r} rule") from exc

    if isinstance(quant_cfg, list):
        for entry in reversed(quant_cfg):
            if entry.get("quantizer_name") != quantizer_name:
                continue
            if "cfg" not in entry:
                raise KeyError(
                    f"base quantization rule {quantizer_name!r} has no quantizer cfg"
                )
            return copy.deepcopy(entry["cfg"])
        raise KeyError(f"base quantization config has no {quantizer_name!r} rule")

    raise TypeError(
        "ModelOpt quant_cfg must be either a mapping or an ordered list, "
        f"not {type(quant_cfg).__name__}"
    )


def _base_rule_for(pattern: str) -> str:
    if pattern.endswith("weight_quantizer"):
        return "*weight_quantizer"
    if pattern.endswith("input_quantizer"):
        return "*input_quantizer"
    raise ValueError(f"cannot infer a base quantizer rule for enabled pattern {pattern!r}")


def compose_quant_config(
    base_config: Mapping,
    overrides: Mapping[str, Mapping],
    *,
    calibration_method: str,
):
    """Return a new ModelOpt config with ordered, fail-closed overrides.

    ModelOpt <=0.45 used a mapping from wildcard to quantizer attributes.
    ModelOpt 0.46 uses an ordered list of rules with explicit
    ``quantizer_name`` and ``cfg`` fields.  This function preserves the same
    deny-all/selective-enable policy in either representation.  Exact enabled
    input rules receive the quantile calibrator when requested; attaching it
    only to a later-disabled broad input rule would silently leave experts on
    max calibration.
    """

    config = copy.deepcopy(base_config)
    quant_cfg = config.get("quant_cfg")
    if not isinstance(quant_cfg, (Mapping, list)):
        raise TypeError("base ModelOpt config has an unsupported quant_cfg schema")
    if calibration_method == "quantile" and isinstance(quant_cfg, list):
        raise RuntimeError(
            "this ordered-rule ModelOpt build does not provide the legacy "
            "QuantileCalibrator or quantile calibration algorithm; use a sealed "
            "max-calibration recipe or pin a verified legacy ModelOpt runtime"
        )

    # Keep an immutable lookup copy. Ordered overrides may repeat a base rule
    # with enable=False, and the format-bearing rule must remain discoverable.
    base_rules = copy.deepcopy(quant_cfg)

    if calibration_method == "quantile":
        config["algorithm"] = "quantile"

    for pattern, requested in overrides.items():
        requested = copy.deepcopy(requested)
        selectively_enabled = requested == {"enable": True}
        if selectively_enabled:
            cfg = _find_base_cfg(base_rules, _base_rule_for(pattern))
            if calibration_method == "quantile" and pattern.endswith("input_quantizer"):
                cfg["calibrator"] = "quantile"

        if isinstance(quant_cfg, Mapping):
            quant_cfg[pattern] = cfg if selectively_enabled else requested
        else:
            entry = {"quantizer_name": pattern}
            if selectively_enabled:
                entry["cfg"] = cfg
            else:
                entry.update(requested)
            quant_cfg.append(entry)

    return config
