#!/usr/bin/env python3
"""Install and seal a static-global NVFP4 input-scale shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open


ROUTED_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
SCALE_FILE = "model-inputscales.safetensors"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".input_scale"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--scale-receipt", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    candidate = args.candidate_model.resolve()
    index_path = candidate / "model.safetensors.index.json"
    config_path = candidate / "config.json"
    index = _load_json(index_path)
    config = _load_json(config_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("candidate index has no weight_map object")

    routed_weights = sorted(
        key
        for key in weight_map
        if ROUTED_WEIGHT_RE.fullmatch(key)
        and key.removesuffix(".weight") + ".weight_scale" in weight_map
        and key.removesuffix(".weight") + ".weight_scale_2" in weight_map
    )
    expected_scales = {_input_key(key) for key in routed_weights}
    with safe_open(args.scales, framework="pt", device="cpu") as handle:
        actual_scales = set(handle.keys())
        if actual_scales != expected_scales:
            missing = sorted(expected_scales - actual_scales)
            extra = sorted(actual_scales - expected_scales)
            raise ValueError(
                "input-scale keyset mismatch: "
                f"expected={len(expected_scales)} actual={len(actual_scales)} "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        values: dict[str, float] = {}
        for key in sorted(actual_scales):
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32 or tensor.numel() != 1:
                raise ValueError(f"input scale is not scalar F32: {key}")
            value = float(tensor.item())
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"input scale is not finite and positive: {key}")
            values[key] = value

    for key in routed_weights:
        match = ROUTED_WEIGHT_RE.fullmatch(key)
        assert match is not None
        layer, expert, projection = match.groups()
        if projection != "gate_proj":
            continue
        gate = values[_input_key(key)]
        up_key = (
            f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            "up_proj.input_scale"
        )
        if gate != values[up_key]:
            raise ValueError(
                f"gate/up input scales differ: layer={layer} expert={expert}"
            )

    destination = candidate / SCALE_FILE
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing scale shard: {destination}")
    temporary = destination.with_name(f".{destination.name}.incomplete")
    shutil.copy2(args.scales, temporary)
    os.replace(temporary, destination)
    for key in expected_scales:
        weight_map[key] = SCALE_FILE
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = sum(
        (candidate / shard).stat().st_size for shard in set(weight_map.values())
    )
    _write_json_atomic(index_path, index)

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise TypeError("candidate config has no quantization_config object")
    quant_config["quant_algo"] = "NVFP4"
    for group in quant_config.get("config_groups", {}).values():
        activations = group.get("input_activations")
        if isinstance(activations, dict):
            activations["dynamic"] = True
    _write_json_atomic(config_path, config)

    source_receipt = _load_json(args.scale_receipt)
    receipt = {
        "schema": "quant-toolkit.install-nvfp4-input-scales.v1",
        "candidate": str(candidate),
        "input_scale_tensors": len(expected_scales),
        "input_scale_shard": {
            "path": SCALE_FILE,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        },
        "source_scale_receipt": {
            "path": str(args.scale_receipt.resolve()),
            "file_sha256": _sha256(args.scale_receipt),
            "receipt_sha256": source_receipt.get("receipt_sha256"),
        },
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(index_path),
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    _write_json_atomic(args.receipt, receipt)
    print(json.dumps({"event": "nvfp4_input_scales_installed", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
