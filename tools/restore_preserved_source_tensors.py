#!/usr/bin/env python3
"""Restore protected exported tensors whose dtype/shape drifted from source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from verify_glm53_nvfp4_checkpoint import (
    canonical_source_info,
    read_checkpoint,
    source_routed_keys,
)
from streaming_loader import _checkpoint_key_to_model_key, _glm5_next_conv_target_key


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_model).resolve()
    candidate_dir = Path(args.candidate_model).resolve()
    source_config = _load_json(source_dir / "config.json")
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("Source config has no text_config object")

    source_index_path = source_dir / "model.safetensors.index.json"
    candidate_index_path = candidate_dir / "model.safetensors.index.json"
    source_index = _load_json(source_index_path)
    candidate_index = _load_json(candidate_index_path)
    source_weight_map = source_index.get("weight_map")
    candidate_weight_map = candidate_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not isinstance(candidate_weight_map, dict):
        raise TypeError("Source and candidate indexes must contain weight_map objects")

    _unused, source_info, _scalars, _files = read_checkpoint(source_dir)
    _unused, candidate_info, _scalars, _files = read_checkpoint(candidate_dir)
    canonical_info = canonical_source_info(source_info, text_config)
    selected = set(source_routed_keys(source_info, text_config))
    protected_prefix = (
        f"model.language_model.layers.{int(text_config['num_hidden_layers'])}."
    )

    raw_keys_by_canonical: dict[str, list[str]] = defaultdict(list)
    for raw_key in source_weight_map:
        if raw_key.startswith(protected_prefix):
            canonical = raw_key
        else:
            canonical = _checkpoint_key_to_model_key(raw_key)
            if _glm5_next_conv_target_key(canonical) is not None:
                continue
        raw_keys_by_canonical[canonical].append(raw_key)

    mismatches = []
    for key, expected in canonical_info.items():
        if key in selected:
            continue
        actual = candidate_info.get(key)
        if actual is None:
            raise KeyError(f"Candidate is missing protected tensor: {key}")
        if actual["dtype"] != expected["dtype"] or actual["shape"] != expected["shape"]:
            raw_keys = raw_keys_by_canonical.get(key, [])
            if len(raw_keys) != 1:
                raise RuntimeError(
                    f"Protected mismatch does not have one source tensor: {key} -> {raw_keys}"
                )
            mismatches.append((key, raw_keys[0]))
    if not mismatches:
        raise RuntimeError("No protected dtype/shape mismatches found")

    replacements = {}
    source_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for candidate_key, raw_key in mismatches:
        source_groups[source_weight_map[raw_key]].append((candidate_key, raw_key))
    for shard_name, entries in source_groups.items():
        with safe_open(source_dir / shard_name, framework="pt", device="cpu") as handle:
            for candidate_key, raw_key in entries:
                replacements[candidate_key] = handle.get_tensor(raw_key)

    candidate_groups: dict[str, list[str]] = defaultdict(list)
    for candidate_key, _raw_key in mismatches:
        candidate_groups[candidate_weight_map[candidate_key]].append(candidate_key)

    changed_files = []
    for shard_name, keys in sorted(candidate_groups.items()):
        shard_path = candidate_dir / shard_name
        tensors = load_file(shard_path)
        for key in keys:
            tensors[key] = replacements[key]
        temporary = shard_path.with_name(f".{shard_path.name}.incomplete")
        save_file(tensors, temporary)
        os.replace(temporary, shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor_slice = handle.get_slice(key)
                expected = canonical_info[key]
                if (
                    tensor_slice.get_dtype() != expected["dtype"]
                    or list(tensor_slice.get_shape()) != expected["shape"]
                ):
                    raise RuntimeError(f"Restored tensor header still mismatches: {key}")
        changed_files.append({"path": shard_name, "bytes": shard_path.stat().st_size})

    metadata = candidate_index.setdefault("metadata", {})
    metadata["total_size"] = sum(
        (candidate_dir / shard_name).stat().st_size
        for shard_name in set(candidate_weight_map.values())
    )
    _write_json_atomic(candidate_index_path, candidate_index)

    restored_keys = sorted(key for key, _raw_key in mismatches)
    receipt = {
        "schema": "quant-toolkit.restore-preserved-source-tensors.v1",
        "source": {
            "directory": str(source_dir),
            "revision": args.source_revision,
            "index_sha256": _sha256_file(source_index_path),
        },
        "candidate": {
            "directory": str(candidate_dir),
            "index_sha256": _sha256_file(candidate_index_path),
        },
        "restored_tensor_count": len(restored_keys),
        "restored_keyset_sha256": _canonical_sha256(restored_keys),
        "changed_shards": changed_files,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_json_atomic(Path(args.receipt).resolve(), receipt)
    print(json.dumps({"event": "preserved_tensors_restored", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
