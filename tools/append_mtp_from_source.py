#!/usr/bin/env python3
"""Append protected source-precision MTP tensors to an exported checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from export_hf import MTP_SHARD_NAME, _merge_mtp_weights, _rename_shards


COPY_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_model).resolve()
    candidate_dir = Path(args.candidate_model).resolve()
    source_index_path = source_dir / "model.safetensors.index.json"
    candidate_index_path = candidate_dir / "model.safetensors.index.json"
    source_index = _load_json(source_index_path)
    candidate_index = _load_json(candidate_index_path)
    source_weight_map = source_index.get("weight_map")
    weight_map = candidate_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not isinstance(weight_map, dict):
        raise TypeError("Source and candidate indexes must contain weight_map objects")
    if (candidate_dir / MTP_SHARD_NAME).exists():
        raise FileExistsError(f"Candidate already has {MTP_SHARD_NAME}")

    prefixes = ["mtp.", *args.prefix]
    selected = sorted(
        key for key in source_weight_map if any(key.startswith(prefix) for prefix in prefixes)
    )
    if not selected:
        raise RuntimeError(f"No source tensors match prefixes: {prefixes}")
    overlap = sorted(set(selected) & set(weight_map))
    if overlap:
        raise RuntimeError(f"Candidate already contains {len(overlap)} selected tensor(s)")

    _merge_mtp_weights(
        source_dir,
        candidate_dir,
        weight_map,
        shard_idx=1,
        total_tensors=len(weight_map),
        extra_prefixes=args.prefix,
    )
    _rename_shards(candidate_dir, weight_map, total_shards=0)

    copied = []
    for name in COPY_NAMES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, candidate_dir / name)
            copied.append(name)

    mtp_path = candidate_dir / MTP_SHARD_NAME
    receipt = {
        "schema": "quant-toolkit.append-mtp-from-source.v1",
        "source": {
            "directory": str(source_dir),
            "revision": args.source_revision,
            "index_sha256": _sha256_file(source_index_path),
        },
        "candidate": {
            "directory": str(candidate_dir),
            "index_sha256": _sha256_file(candidate_index_path),
        },
        "prefixes": prefixes,
        "appended_tensor_count": len(selected),
        "appended_keyset_sha256": _canonical_sha256(selected),
        "mtp_shard": {
            "path": MTP_SHARD_NAME,
            "bytes": mtp_path.stat().st_size,
            "sha256": _sha256_file(mtp_path),
        },
        "copied_metadata": copied,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_json_atomic(Path(args.receipt).resolve(), receipt)
    print(json.dumps({"event": "mtp_appended", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
