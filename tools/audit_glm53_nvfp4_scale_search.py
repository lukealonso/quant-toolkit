#!/usr/bin/env python3
"""Audit exact deployable NVFP4 block-scale search on one GLM expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import nvfp4_codec
import torch
from kld_common import canonical_json_sha256, sha256_file
from nvfp4_codec import decode_nvfp4, encode_nvfp4, optimize_block_scales
from safetensors import safe_open
from safetensors.torch import save_file

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SCHEMA = "quant-toolkit.glm53-nvfp4-scale-search-audit.v1"


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    payload = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError(f"checkpoint has no weight map: {model_dir}")
    return weight_map


def _load_tensors(
    model_dir: Path, weight_map: dict[str, str], keys: list[str]
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    tensors = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=(0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.125, 1.1875, 1.25),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source_model = args.source_model.resolve()
    weight_map = _load_weight_map(source_model)
    prefixes = {
        projection: (
            f"model.language_model.layers.{args.layer}.mlp.experts."
            f"{args.expert}.{projection}"
        )
        for projection in PROJECTIONS
    }
    keys = [prefix + ".weight" for prefix in prefixes.values()]
    source = _load_tensors(source_model, weight_map, keys)
    device = torch.device(args.device)
    weights = {key: value.to(device=device) for key, value in source.items()}
    shared_gate_up_scale_2 = (
        max(
            weights[prefixes[projection] + ".weight"].to(torch.float32).abs().amax()
            for projection in ("gate_proj", "up_proj")
        )
        / (6.0 * 448.0)
    ).reshape(())

    artifact_tensors = {}
    records = []
    for projection in PROJECTIONS:
        prefix = prefixes[projection]
        weight = weights[prefix + ".weight"]
        scale_2_override = (
            shared_gate_up_scale_2 if projection in ("gate_proj", "up_proj") else None
        )
        baseline = encode_nvfp4(weight, scale_2=scale_2_override)
        baseline_decoded = decode_nvfp4(*baseline)
        optimized = optimize_block_scales(
            weight,
            scale_2=scale_2_override,
            factors=tuple(args.factors),
        )
        packed, block_scale, scale_2, optimized_decoded = optimized
        source_f32 = weight.to(torch.float32)
        baseline_error = baseline_decoded - source_f32
        optimized_error = optimized_decoded - source_f32
        baseline_sse = float(baseline_error.square().sum().cpu())
        optimized_sse = float(optimized_error.square().sum().cpu())
        artifact_tensors[prefix + ".weight"] = packed.cpu().clone()
        artifact_tensors[prefix + ".weight_scale"] = block_scale.cpu().clone()
        artifact_tensors[prefix + ".weight_scale_2"] = scale_2.cpu().clone()
        records.append(
            {
                "projection": projection,
                "source_key": prefix + ".weight",
                "source_shard": weight_map[prefix + ".weight"],
                "source_shape": list(weight.shape),
                "source_dtype": str(weight.dtype),
                "source_tensor_sha256": _tensor_sha256(weight),
                "packed_shape": list(packed.shape),
                "block_scale_shape": list(block_scale.shape),
                "secondary_scale": float(scale_2.cpu()),
                "baseline_mse": baseline_sse / weight.numel(),
                "optimized_mse": optimized_sse / weight.numel(),
                "relative_sse_reduction": (
                    (baseline_sse - optimized_sse) / baseline_sse
                    if baseline_sse
                    else 0.0
                ),
                "changed_block_scales": int((block_scale != baseline[1]).sum().cpu()),
                "total_block_scales": block_scale.numel(),
                "changed_packed_bytes": int((packed != baseline[0]).sum().cpu()),
                "total_packed_bytes": packed.numel(),
                "baseline_decoded_sha256": _tensor_sha256(baseline_decoded),
                "optimized_decoded_sha256": _tensor_sha256(optimized_decoded),
                "optimized_packed_sha256": _tensor_sha256(packed),
                "optimized_block_scale_sha256": _tensor_sha256(block_scale),
            }
        )

    gate_scale_2 = artifact_tensors[prefixes["gate_proj"] + ".weight_scale_2"]
    up_scale_2 = artifact_tensors[prefixes["up_proj"] + ".weight_scale_2"]
    if not torch.equal(gate_scale_2, up_scale_2):
        raise RuntimeError("optimized gate/up secondary scales are not tied")

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = args.artifact.with_suffix(args.artifact.suffix + ".tmp")
    save_file(artifact_tensors, str(temporary_artifact))
    os.replace(temporary_artifact, args.artifact)
    receipt = {
        "schema": SCHEMA,
        "source_model": str(source_model),
        "source_index_sha256": sha256_file(
            source_model / "model.safetensors.index.json"
        ),
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
        "codec_file_sha256": sha256_file(Path(nvfp4_codec.__file__).resolve()),
        "layer": args.layer,
        "expert": args.expert,
        "device": str(device),
        "factors": list(args.factors),
        "objective": "unweighted exact decoded weight squared error",
        "runtime_format": "packed-e2m1-plus-fp8-e4m3fn-block-scale-plus-f32-secondary-scale",
        "gate_up_secondary_scale_tied": True,
        "artifact": str(args.artifact.resolve()),
        "artifact_bytes": args.artifact.stat().st_size,
        "artifact_sha256": sha256_file(args.artifact),
        "projections": records,
        "result": "PASS",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
