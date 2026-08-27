#!/usr/bin/env python3
"""Install sealed routed-expert NVFP4 layer patches into a hardlink clone."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "quant-toolkit.glm53-nvfp4-weight-patch-install.v1"
LAYER_SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-layer.v1"
ROUTED_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _sidecars(weight_key: str) -> tuple[str, str, str]:
    prefix = weight_key.removesuffix(".weight")
    return weight_key, prefix + ".weight_scale", prefix + ".weight_scale_2"


def _hardlink_clone(source: Path, candidate: Path) -> None:
    if candidate.exists():
        raise FileExistsError(f"candidate already exists: {candidate}")
    shutil.copytree(source, candidate, copy_function=os.link, symlinks=True)
    _fsync_directory(candidate.parent)


def _load_patch_inventory(
    patch_dir: Path,
    layers: list[int],
    expected_keys: set[str],
) -> tuple[dict[str, Path], list[dict]]:
    key_sources: dict[str, Path] = {}
    records = []
    for layer in layers:
        patch_path = patch_dir / f"layer-{layer:04d}.safetensors"
        receipt_path = patch_dir / f"layer-{layer:04d}.receipt.json"
        if not patch_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"missing layer patch or receipt: layer={layer}")
        receipt = _load_json(receipt_path)
        body = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("schema") != LAYER_SCHEMA:
            raise RuntimeError(f"unexpected layer receipt schema: {receipt_path}")
        if receipt.get("receipt_sha256") != canonical_json_sha256(body):
            raise RuntimeError(f"layer receipt seal mismatch: {receipt_path}")
        if receipt.get("patch_sha256") != sha256_file(patch_path):
            raise RuntimeError(f"layer patch hash mismatch: {patch_path}")
        with safe_open(patch_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            metadata = handle.metadata() or {}
        if metadata.get("schema") != LAYER_SCHEMA or metadata.get("layer") != str(
            layer
        ):
            raise RuntimeError(f"layer patch metadata mismatch: {patch_path}")
        for key in keys:
            if key in key_sources:
                raise RuntimeError(f"duplicate patched tensor key: {key}")
            key_sources[key] = patch_path
        records.append(
            {
                "layer": layer,
                "patch": str(patch_path.resolve()),
                "patch_bytes": patch_path.stat().st_size,
                "patch_sha256": receipt["patch_sha256"],
                "receipt": str(receipt_path.resolve()),
                "receipt_file_sha256": sha256_file(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "decoded_reconstruction_sha256": receipt[
                    "decoded_reconstruction_sha256"
                ],
                "relative_sse_reduction": receipt["relative_sse_reduction"],
            }
        )
    actual_keys = set(key_sources)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            "weight patch keyset mismatch: "
            f"expected={len(expected_keys)} actual={len(actual_keys)} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return key_sources, records


def _rewrite_shards(
    candidate: Path,
    weight_map: dict[str, str],
    key_sources: dict[str, Path],
) -> list[dict]:
    replacements_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in key_sources:
        replacements_by_shard[weight_map[key]].append(key)
    records = []
    with ExitStack() as stack:
        patch_handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in sorted(set(key_sources.values()))
        }
        for index, shard in enumerate(sorted(replacements_by_shard), 1):
            shard_path = candidate / shard
            temporary = shard_path.with_name(f".{shard_path.name}.incomplete")
            replacement_keys = set(replacements_by_shard[shard])
            with safe_open(shard_path, framework="pt", device="cpu") as source:
                tensors = {
                    key: (
                        patch_handles[key_sources[key]].get_tensor(key)
                        if key in replacement_keys
                        else source.get_tensor(key)
                    )
                    for key in source.keys()
                }
            if not replacement_keys <= set(tensors):
                raise RuntimeError(f"replacement key absent from shard: {shard}")
            save_file(tensors, temporary)
            descriptor = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, shard_path)
            _fsync_directory(candidate)
            records.append(
                {
                    "shard": shard,
                    "bytes": shard_path.stat().st_size,
                    "replaced_tensors": len(replacement_keys),
                    "sha256": sha256_file(shard_path),
                }
            )
            print(
                f"shard={index}/{len(replacements_by_shard)} name={shard} "
                f"replaced={len(replacement_keys)}",
                flush=True,
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline = args.baseline_model.resolve()
    candidate = args.candidate_model.resolve()
    index_path = baseline / "model.safetensors.index.json"
    config_path = baseline / "config.json"
    index = _load_json(index_path)
    config = _load_json(config_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("baseline index has no weight_map object")
    text_config = config["text_config"]
    first_layer = int(text_config["first_k_dense_replace"])
    num_layers = int(text_config["num_hidden_layers"])
    layers = list(range(first_layer, num_layers))
    routed_weights = {
        key
        for key in weight_map
        if (match := ROUTED_WEIGHT_RE.fullmatch(key))
        and first_layer <= int(match.group(1)) < num_layers
        and key.removesuffix(".weight") + ".weight_scale" in weight_map
        and key.removesuffix(".weight") + ".weight_scale_2" in weight_map
    }
    expected_keys = {
        sidecar for key in routed_weights for sidecar in _sidecars(key)
    }
    key_sources, patch_records = _load_patch_inventory(
        args.patch_dir, layers, expected_keys
    )

    _hardlink_clone(baseline, candidate)
    try:
        shard_records = _rewrite_shards(candidate, weight_map, key_sources)
    except Exception:
        failure = candidate.with_name(candidate.name + ".failed-install")
        if not failure.exists():
            os.replace(candidate, failure)
            _fsync_directory(failure.parent)
        raise

    if sha256_file(candidate / "config.json") != sha256_file(config_path):
        raise RuntimeError("candidate config changed during weight patch install")
    if sha256_file(candidate / "model.safetensors.index.json") != sha256_file(
        index_path
    ):
        raise RuntimeError("candidate index changed during weight patch install")
    receipt = {
        "schema": SCHEMA,
        "baseline": {
            "directory": str(baseline),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
        },
        "candidate": {
            "directory": str(candidate),
            "config_sha256": sha256_file(candidate / "config.json"),
            "index_sha256": sha256_file(
                candidate / "model.safetensors.index.json"
            ),
        },
        "coverage": {
            "layers": layers,
            "routed_weights": len(routed_weights),
            "replaced_tensors": len(key_sources),
            "expected_routed_weights": 36_288,
            "expected_replaced_tensors": 108_864,
        },
        "patches": patch_records,
        "rewritten_shards": shard_records,
        "unchanged_input_scale_sha256": sha256_file(
            candidate / "model-inputscales.safetensors"
        ),
        "result": "PASS",
    }
    if receipt["coverage"]["routed_weights"] != 36_288 or len(key_sources) != 108_864:
        raise RuntimeError("GLM routed NVFP4 patch coverage is not canonical")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    output = args.output if args.output.is_absolute() else candidate / args.output
    _write_json_atomic(output, receipt)
    print(
        json.dumps(
            {
                "event": "glm53_nvfp4_weight_patches_installed",
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
