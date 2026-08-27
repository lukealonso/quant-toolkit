#!/usr/bin/env python3
"""Merge and verify distributed GLM NVFP4 optimization receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file


PART_SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-optimization.v1"
LAYER_SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-layer.v1"
SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-optimization-merged.v1"


def _load_sealed(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise RuntimeError(f"unexpected receipt schema: {path}")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != canonical_json_sha256(body):
        raise RuntimeError(f"receipt seal mismatch: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-receipts", nargs="+", required=True, type=Path)
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    parts = []
    layer_ids = []
    for path in args.part_receipts:
        part = _load_sealed(path, PART_SCHEMA)
        parts.append(
            {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "receipt_sha256": part["receipt_sha256"],
                "layers": part["topology"]["layers"],
                "proxy": part["proxy"],
            }
        )
        layer_ids.extend(int(layer) for layer in part["topology"]["layers"])
    if len(layer_ids) != len(set(layer_ids)):
        raise RuntimeError("distributed optimizer receipts contain duplicate layers")
    expected_layers = list(range(3, 45))
    if sorted(layer_ids) != expected_layers:
        raise RuntimeError(
            f"distributed optimizer layer coverage mismatch: {sorted(layer_ids)}"
        )

    layer_records = []
    choice_counts: dict[str, int] = {}
    for layer in expected_layers:
        patch_path = args.patch_dir / f"layer-{layer:04d}.safetensors"
        receipt_path = args.patch_dir / f"layer-{layer:04d}.receipt.json"
        receipt = _load_sealed(receipt_path, LAYER_SCHEMA)
        if receipt.get("layer") != layer:
            raise RuntimeError(f"layer receipt identity mismatch: {receipt_path}")
        patch_sha256 = sha256_file(patch_path)
        if patch_sha256 != receipt.get("patch_sha256"):
            raise RuntimeError(f"layer patch hash mismatch: {patch_path}")
        for label, count in receipt["choice_counts"].items():
            choice_counts[label] = choice_counts.get(label, 0) + int(count)
        layer_records.append(
            {
                "layer": layer,
                "patch": str(patch_path.resolve()),
                "patch_bytes": patch_path.stat().st_size,
                "patch_sha256": patch_sha256,
                "patch_tensors": receipt["patch_tensors"],
                "receipt": str(receipt_path.resolve()),
                "receipt_file_sha256": sha256_file(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "decoded_reconstruction_sha256": receipt[
                    "decoded_reconstruction_sha256"
                ],
                "baseline_complete_sse": receipt["baseline_complete_sse"],
                "selected_complete_sse": receipt["selected_complete_sse"],
                "relative_sse_reduction": receipt["relative_sse_reduction"],
            }
        )

    baseline_sse = sum(item["baseline_complete_sse"] for item in layer_records)
    selected_sse = sum(item["selected_complete_sse"] for item in layer_records)
    patch_tensors = sum(int(item["patch_tensors"]) for item in layer_records)
    patch_bytes = sum(int(item["patch_bytes"]) for item in layer_records)
    experts = sum(choice_counts.values())
    if patch_tensors != 108_864 or experts != 12_096:
        raise RuntimeError(
            "canonical GLM optimization coverage mismatch: "
            f"patch_tensors={patch_tensors} experts={experts}"
        )
    receipt = {
        "schema": SCHEMA,
        "parts": parts,
        "coverage": {
            "layers": expected_layers,
            "experts": experts,
            "routed_weight_tensors": 36_288,
            "patch_tensors": patch_tensors,
            "patch_bytes": patch_bytes,
        },
        "proxy": {
            "objective": (
                "route-weighted complete expert output SSE over deterministic "
                "conditional reservoirs"
            ),
            "stock_included": True,
            "baseline_complete_sse": baseline_sse,
            "selected_complete_sse": selected_sse,
            "relative_sse_reduction": (
                (baseline_sse - selected_sse) / baseline_sse
                if baseline_sse
                else 0.0
            ),
            "choice_counts": dict(sorted(choice_counts.items())),
        },
        "layers": layer_records,
        "result": "PASS",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_json_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "event": "glm53_nvfp4_optimization_receipts_merged",
                "receipt": str(args.output.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                **receipt["coverage"],
                **receipt["proxy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
