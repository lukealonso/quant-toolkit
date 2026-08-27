#!/usr/bin/env python3
"""Prove that the local NVFP4 encoder reproduces a GLM ModelOpt export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from nvfp4_codec import decode_nvfp4, encode_nvfp4
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SCHEMA = "quant-toolkit.glm53-nvfp4-codec-audit.v1"


def _load_index(model_dir: Path) -> dict[str, str]:
    payload = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"checkpoint has no weight map: {model_dir}")
    return weight_map


def _load_tensors(
    model_dir: Path, weight_map: dict[str, str], keys: list[str]
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    result = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                result[key] = handle.get_tensor(key)
    return result


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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source_model = args.source_model.resolve()
    candidate_model = args.candidate_model.resolve()
    source_map = _load_index(source_model)
    candidate_map = _load_index(candidate_model)
    prefixes = {
        projection: (
            f"model.language_model.layers.{args.layer}.mlp.experts."
            f"{args.expert}.{projection}"
        )
        for projection in PROJECTIONS
    }
    source_keys = [prefix + ".weight" for prefix in prefixes.values()]
    source_tensors = _load_tensors(source_model, source_map, source_keys)
    shared_gate_up_scale_2 = (
        max(
            source_tensors[prefixes[projection] + ".weight"]
            .to(torch.float32)
            .abs()
            .amax()
            for projection in ("gate_proj", "up_proj")
        )
        / (6.0 * 448.0)
    ).reshape(())

    records = []
    for projection in PROJECTIONS:
        prefix = prefixes[projection]
        weight_key = prefix + ".weight"
        scale_key = prefix + ".weight_scale"
        scale_2_key = prefix + ".weight_scale_2"
        source = source_tensors[weight_key]
        candidate = _load_tensors(
            candidate_model, candidate_map, [weight_key, scale_key, scale_2_key]
        )
        encoded_weight, encoded_scale, encoded_scale_2 = encode_nvfp4(
            source,
            scale_2=(
                shared_gate_up_scale_2
                if projection in ("gate_proj", "up_proj")
                else None
            ),
        )
        comparisons = {
            "packed_weight_exact": torch.equal(encoded_weight, candidate[weight_key]),
            "block_scale_exact": torch.equal(encoded_scale, candidate[scale_key]),
            "secondary_scale_exact": torch.equal(
                encoded_scale_2, candidate[scale_2_key]
            ),
        }
        if not all(comparisons.values()):
            raise RuntimeError(
                f"local codec does not reproduce {weight_key}: {comparisons}"
            )
        decoded = decode_nvfp4(
            candidate[weight_key], candidate[scale_key], candidate[scale_2_key]
        )
        records.append(
            {
                "projection": projection,
                "source_key": weight_key,
                "source_shard": source_map[weight_key],
                "candidate_shard": candidate_map[weight_key],
                "source_shape": list(source.shape),
                "source_dtype": str(source.dtype),
                "packed_shape": list(candidate[weight_key].shape),
                "comparisons": comparisons,
                "source_tensor_sha256": _tensor_sha256(source),
                "packed_tensor_sha256": _tensor_sha256(candidate[weight_key]),
                "block_scale_tensor_sha256": _tensor_sha256(candidate[scale_key]),
                "secondary_scale_tensor_sha256": _tensor_sha256(
                    candidate[scale_2_key]
                ),
                "decoded_tensor_sha256": _tensor_sha256(decoded),
                "reconstruction_mse": float(
                    torch.mean((decoded - source.to(torch.float32)).square())
                ),
            }
        )

    receipt = {
        "schema": SCHEMA,
        "source_model": str(source_model),
        "candidate_model": str(candidate_model),
        "layer": args.layer,
        "expert": args.expert,
        "gate_up_secondary_scale_tied": True,
        "projections": records,
        "result": "PASS",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"event": "glm53_nvfp4_codec_audit_complete", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
