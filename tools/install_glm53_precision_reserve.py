#!/usr/bin/env python3
"""Restore selected routed GLM expert layers to BF16 in an NVFP4 clone."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from install_glm53_nvfp4_weight_patches import (
    _fsync_directory,
    _hardlink_clone,
    _load_json,
    _write_json_atomic,
)
from kld_common import canonical_json_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMA = "quant-toolkit.glm53-nvfp4-precision-reserve-install.v1"
WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def _sidecars(weight: str) -> tuple[str, str, str]:
    prefix = weight.removesuffix(".weight")
    return (
        prefix + ".weight_scale",
        prefix + ".weight_scale_2",
        prefix + ".input_scale",
    )


def _rewrite_precision_reserve(
    source: Path,
    candidate: Path,
    source_map: dict[str, str],
    candidate_map: dict[str, str],
    weights: set[str],
) -> list[dict]:
    removals = {sidecar for weight in weights for sidecar in _sidecars(weight)}
    absent = sorted((weights | removals) - set(candidate_map))
    if absent:
        raise RuntimeError(f"candidate reserve keyset is incomplete: {absent[:8]}")
    if missing := sorted(weights - set(source_map)):
        raise RuntimeError(f"source reserve weights are missing: {missing[:8]}")

    changed_by_shard: dict[str, set[str]] = defaultdict(set)
    for key in weights | removals:
        changed_by_shard[candidate_map[key]].add(key)

    records = []
    with ExitStack() as stack:
        source_handles = {
            shard: stack.enter_context(
                safe_open(source / shard, framework="pt", device="cpu")
            )
            for shard in sorted({source_map[key] for key in weights})
        }
        for shard in sorted(changed_by_shard):
            path = candidate / shard
            original_mode = path.stat().st_mode & 0o777
            temporary = path.with_name(f".{path.name}.incomplete")
            shard_changes = changed_by_shard[shard]
            shard_weights = shard_changes & weights
            shard_removals = shard_changes & removals
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensors = {
                    key: handle.get_tensor(key)
                    for key in handle.keys()
                    if key not in shard_removals
                }
            for key in shard_weights:
                tensors[key] = source_handles[source_map[key]].get_tensor(key)
            save_file(tensors, temporary)
            os.chmod(temporary, original_mode)
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            _fsync_directory(candidate)
            records.append(
                {
                    "shard": shard,
                    "bytes": path.stat().st_size,
                    "restored_bf16_weights": len(shard_weights),
                    "removed_nvfp4_sidecars": len(shard_removals),
                    "sha256": sha256_file(path),
                }
            )
            print(
                f"shard={shard} restored={len(shard_weights)} "
                f"removed={len(shard_removals)}",
                flush=True,
            )
    for key in removals:
        candidate_map.pop(key)
    return records


def _tensor_inventory(model: Path, weight_map: dict[str, str]) -> tuple[int, int]:
    count = 0
    total = 0
    for shard in sorted(set(weight_map.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                dtype = handle.get_slice(key).get_dtype()
                widths = {
                    "BOOL": 1,
                    "U8": 1,
                    "I8": 1,
                    "F8_E4M3": 1,
                    "F8_E5M2": 1,
                    "I16": 2,
                    "U16": 2,
                    "F16": 2,
                    "BF16": 2,
                    "I32": 4,
                    "U32": 4,
                    "F32": 4,
                    "I64": 8,
                    "U64": 8,
                    "F64": 8,
                }
                elements = 1
                for extent in shape:
                    elements *= int(extent)
                count += 1
                total += elements * widths[dtype]
    return count, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    source = args.source_model.resolve()
    baseline = args.baseline_model.resolve()
    candidate = args.candidate_model.resolve()
    layers = sorted(set(args.layers))
    if not layers or any(layer < 3 or layer > 44 for layer in layers):
        parser.error("reserve layers must be unique main routed layers 3..44")

    source_index_path = source / "model.safetensors.index.json"
    baseline_index_path = baseline / "model.safetensors.index.json"
    source_index = _load_json(source_index_path)
    baseline_index = _load_json(baseline_index_path)
    source_map = source_index["weight_map"]
    candidate_map = dict(baseline_index["weight_map"])
    weights = {
        key
        for key in source_map
        if (match := WEIGHT_RE.fullmatch(key)) and int(match.group(1)) in layers
    }
    expected_weights = len(layers) * 288 * 3
    if len(weights) != expected_weights:
        raise RuntimeError(
            f"reserve weight coverage mismatch: {len(weights)} != {expected_weights}"
        )

    _hardlink_clone(baseline, candidate)
    try:
        changed_shards = _rewrite_precision_reserve(
            source, candidate, source_map, candidate_map, weights
        )
        candidate_index = _load_json(candidate / "model.safetensors.index.json")
        candidate_index["weight_map"] = dict(sorted(candidate_map.items()))
        tensor_count, tensor_bytes = _tensor_inventory(candidate, candidate_map)
        candidate_index.setdefault("metadata", {})["total_size"] = tensor_bytes
        _write_json_atomic(candidate / "model.safetensors.index.json", candidate_index)

        config = _load_json(candidate / "config.json")
        ignore = config["quantization_config"].setdefault("ignore", [])
        for layer in layers:
            pattern = f"model.language_model.layers.{layer}.mlp.experts*"
            if pattern not in ignore:
                ignore.append(pattern)
        ignore.sort()
        _write_json_atomic(candidate / "config.json", config)
    except Exception:
        failure = candidate.with_name(candidate.name + ".failed-install")
        if not failure.exists():
            os.replace(candidate, failure)
            _fsync_directory(failure.parent)
        raise

    baseline_count, baseline_bytes = _tensor_inventory(
        baseline, baseline_index["weight_map"]
    )
    receipt = {
        "schema": SCHEMA,
        "source": {
            "directory": str(source),
            "revision": args.source_revision,
            "config_sha256": sha256_file(source / "config.json"),
            "index_sha256": sha256_file(source_index_path),
        },
        "baseline": {
            "directory": str(baseline),
            "config_sha256": sha256_file(baseline / "config.json"),
            "index_sha256": sha256_file(baseline_index_path),
            "tensor_count": baseline_count,
            "tensor_bytes": baseline_bytes,
        },
        "candidate": {
            "directory": str(candidate),
            "config_sha256": sha256_file(candidate / "config.json"),
            "index_sha256": sha256_file(
                candidate / "model.safetensors.index.json"
            ),
            "tensor_count": tensor_count,
            "tensor_bytes": tensor_bytes,
            "added_tensor_bytes": tensor_bytes - baseline_bytes,
        },
        "reserve": {
            "layers": layers,
            "bf16_weight_tensors": len(weights),
            "bf16_parameters": 0,
            "removed_nvfp4_sidecars": len(weights) * 3,
            "remaining_nvfp4_weight_tensors": 36_288 - len(weights),
        },
        "changed_shards": changed_shards,
        "result": "PASS",
    }
    # Source routed weights are all BF16; their exact parameter count follows
    # directly from the restored tensor headers.
    parameters = 0
    for shard in sorted({source_map[key] for key in weights}):
        selected = {key for key in weights if source_map[key] == shard}
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for key in selected:
                elements = 1
                for extent in handle.get_slice(key).get_shape():
                    elements *= int(extent)
                parameters += elements
    receipt["reserve"]["bf16_parameters"] = parameters
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    output = args.output if args.output.is_absolute() else candidate / args.output
    _write_json_atomic(output, receipt)
    print(
        json.dumps(
            {
                "event": "glm53_precision_reserve_installed",
                "candidate": str(candidate),
                "receipt": str(output.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
