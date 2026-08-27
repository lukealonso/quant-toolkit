#!/usr/bin/env python3
"""Create a sealed multiplicative variant of GLM routed-expert input scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SCALE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.input_scale$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scales", type=Path, required=True)
    parser.add_argument("--output-scales", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gate-up-factor", type=float, required=True)
    parser.add_argument("--down-factor", type=float, required=True)
    parser.add_argument("--gate-up-cap-quantile", type=float, default=1.0)
    parser.add_argument("--down-cap-quantile", type=float, default=1.0)
    args = parser.parse_args(argv)

    for label, factor in (
        ("gate/up", args.gate_up_factor),
        ("down", args.down_factor),
    ):
        if not math.isfinite(factor) or factor <= 0:
            parser.error(f"{label} factor must be finite and positive")
    for label, quantile in (
        ("gate/up", args.gate_up_cap_quantile),
        ("down", args.down_cap_quantile),
    ):
        if not math.isfinite(quantile) or not 0 < quantile <= 1:
            parser.error(f"{label} cap quantile must be in (0, 1]")
    for destination in (args.output_scales, args.receipt):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite: {destination}")

    source = load_file(args.source_scales)
    scaled_values: dict[str, tuple[int, int, str, float]] = {}
    for key, tensor in source.items():
        match = SCALE_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected input-scale key: {key}")
        if tensor.dtype != torch.float32 or tensor.numel() != 1:
            raise ValueError(f"input scale is not scalar F32: {key}")
        layer, expert, projection = match.groups()
        value = float(tensor.item())
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"input scale is not finite and positive: {key}")
        factor = args.down_factor if projection == "down_proj" else args.gate_up_factor
        scaled_value = float((tensor * factor).item())
        if not math.isfinite(scaled_value) or scaled_value <= 0:
            raise ValueError(f"scaled input scale is invalid: {key}")
        scaled_values[key] = (int(layer), int(expert), projection, scaled_value)

    group_values: dict[tuple[int, str], list[float]] = {}
    for _key, (layer, _expert, projection, value) in scaled_values.items():
        # Gate and up are tied, so count only gate once in their shared cap.
        if projection == "up_proj":
            continue
        group = "down" if projection == "down_proj" else "gate_up"
        group_values.setdefault((layer, group), []).append(value)
    cap_quantiles = {
        "gate_up": args.gate_up_cap_quantile,
        "down": args.down_cap_quantile,
    }
    caps = {
        identity: float(
            torch.quantile(
                torch.tensor(values, dtype=torch.float64),
                cap_quantiles[identity[1]],
            )
        )
        for identity, values in group_values.items()
    }

    output: dict[str, torch.Tensor] = {}
    projections: dict[tuple[int, int], dict[str, float]] = {}
    clipped_by_group = {"gate_up": 0, "down": 0}
    for key, (layer, expert, projection, value) in scaled_values.items():
        group = "down" if projection == "down_proj" else "gate_up"
        capped = min(value, caps[(layer, group)])
        if capped < value:
            clipped_by_group[group] += 1
        output[key] = torch.tensor(capped, dtype=torch.float32)
        projections.setdefault((layer, expert), {})[projection] = float(output[key])

    for identity, values in projections.items():
        if set(values) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"incomplete expert input scales: {identity}")
        if values["gate_proj"] != values["up_proj"]:
            raise ValueError(f"gate/up scales are not tied: {identity}")

    args.output_scales.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_scales.with_name(f".{args.output_scales.name}.incomplete")
    save_file(dict(sorted(output.items())), temporary)
    os.replace(temporary, args.output_scales)

    receipt = {
        "schema": "quant-toolkit.glm53-nvfp4-input-scale-rescale.v1",
        "source": {
            "path": str(args.source_scales.resolve()),
            "bytes": args.source_scales.stat().st_size,
            "sha256": _sha256(args.source_scales),
        },
        "method": {
            "gate_up_factor": args.gate_up_factor,
            "down_factor": args.down_factor,
            "gate_up_cap_quantile": args.gate_up_cap_quantile,
            "down_cap_quantile": args.down_cap_quantile,
            "operation": (
                "float32 scalar multiplication followed by per-layer "
                "expert-maximum quantile cap"
            ),
            "clipped_tensors": clipped_by_group,
            "layer_caps": [
                {"layer": layer, "projection_group": group, "cap": cap}
                for (layer, group), cap in sorted(caps.items())
            ],
        },
        "topology": {
            "experts": len(projections),
            "input_scale_tensors": len(output),
            "gate_up_tied": True,
        },
        "output": {
            "path": str(args.output_scales.resolve()),
            "bytes": args.output_scales.stat().st_size,
            "sha256": _sha256(args.output_scales),
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = args.receipt.with_name(f".{args.receipt.name}.incomplete")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_receipt, args.receipt)
    print(json.dumps({"event": "nvfp4_input_scales_rescaled", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
