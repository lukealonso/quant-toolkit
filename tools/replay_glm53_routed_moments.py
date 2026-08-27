#!/usr/bin/env python3
"""Replay exact float64 routed activation moments from sealed GLM evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from kld_common import canonical_json_sha256, sha256_file
from safetensors.torch import load_file, save_file

EVIDENCE_SCHEMA = "quant-toolkit.glm53-routed-evidence.v1"
MOMENT_SCHEMA = "quant-toolkit.glm53-routed-moments.v1"


def _verify_manifest(manifest: dict) -> None:
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError(f"unexpected routed-evidence schema: {manifest.get('schema')}")
    if expected != canonical_json_sha256(body):
        raise RuntimeError("routed-evidence manifest seal mismatch")


def _parse_columns(value: str, hidden_width: int) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("columns must be START:END") from exc
    if start < 0 or end <= start or end > hidden_width:
        raise argparse.ArgumentTypeError(
            f"columns must satisfy 0 <= START < END <= {hidden_width}"
        )
    return start, end


def replay_moments(
    evidence_dir: Path,
    *,
    layer_id: int,
    expert_id: int,
    powers: tuple[int, ...],
    columns: tuple[int, int],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    manifest_path = evidence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(manifest)
    if layer_id not in manifest["routed_layers"]:
        raise RuntimeError(f"layer {layer_id} is not a routed layer")
    if expert_id < 0 or expert_id >= int(manifest["num_experts"]):
        raise RuntimeError(f"expert {expert_id} is out of range")
    if not powers or any(power not in (0, 1, 2) for power in powers):
        raise RuntimeError("route-weight powers must be selected from 0, 1, 2")

    start, end = columns
    width = end - start
    compute_device = torch.device(device)
    grams = {
        power: torch.zeros((width, width), dtype=torch.float64, device=compute_device)
        for power in powers
    }
    sums = {
        power: torch.zeros(width, dtype=torch.float64, device=compute_device)
        for power in powers
    }
    weight_sums = {power: 0.0 for power in powers}
    routed_rows = 0
    source_records = []

    for window in manifest["windows"]:
        layer_record = next(
            (item for item in window["layers"] if int(item["layer_id"]) == layer_id),
            None,
        )
        if layer_record is None:
            raise RuntimeError(f"window {window['window_id']} is missing layer {layer_id}")
        layer_path = evidence_dir / "windows" / window["window_id"] / layer_record["file"]
        if sha256_file(layer_path) != layer_record["sha256"]:
            raise RuntimeError(f"routed layer hash mismatch: {layer_path}")
        tensors = load_file(layer_path, device="cpu")
        topk_ids = tensors["topk_ids"]
        matches = topk_ids == expert_id
        if bool((matches.sum(dim=-1) > 1).any()):
            raise RuntimeError(f"duplicate expert selection within a token: {layer_path}")
        selected = matches.any(dim=-1)
        count = int(selected.sum())
        if not count:
            continue
        route_weights = (tensors["topk_weights"] * matches).sum(dim=-1)[selected]
        x = tensors["hidden_states"][selected, start:end].to(
            device=compute_device,
            dtype=torch.float64,
        )
        route_weights = route_weights.to(device=compute_device, dtype=torch.float64)
        routed_rows += count
        for power in powers:
            weights = torch.ones_like(route_weights) if power == 0 else route_weights.pow(power)
            sums[power].add_((x * weights.unsqueeze(-1)).sum(dim=0))
            weight_sums[power] += float(weights.sum().cpu())
            weighted_x = x * torch.sqrt(weights).unsqueeze(-1)
            grams[power].add_(weighted_x.T @ weighted_x)
        source_records.append(
            {
                "window_id": window["window_id"],
                "layer_file_sha256": layer_record["sha256"],
                "routed_rows": count,
            }
        )

    if routed_rows == 0:
        raise RuntimeError(f"no rows routed to layer={layer_id} expert={expert_id}")
    output_tensors = {}
    for power in powers:
        gram = grams[power].cpu().contiguous()
        if not torch.equal(gram, gram.T):
            # GEMM can differ by a final rounding bit across the triangle. Seal
            # the mathematically symmetric representation used downstream.
            gram = (gram + gram.T) * 0.5
        if bool((torch.diagonal(gram) < 0).any()) or not torch.isfinite(gram).all():
            raise RuntimeError(f"invalid p={power} routed Gram matrix")
        output_tensors[f"gram_p{power}"] = gram
        output_tensors[f"sum_p{power}"] = sums[power].cpu().contiguous()
        output_tensors[f"weight_sum_p{power}"] = torch.tensor(
            [weight_sums[power]], dtype=torch.float64
        )

    receipt = {
        "schema": MOMENT_SCHEMA,
        "evidence_manifest": str(manifest_path.resolve()),
        "evidence_manifest_file_sha256": sha256_file(manifest_path),
        "evidence_manifest_sha256": manifest["manifest_sha256"],
        "role": manifest["role"],
        "layer_id": layer_id,
        "expert_id": expert_id,
        "route_weight_powers": list(powers),
        "columns": [start, end],
        "storage_dtype": "float64",
        "routed_rows": routed_rows,
        "source_windows": source_records,
    }
    return output_tensors, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--power", type=int, choices=(0, 1, 2), action="append")
    parser.add_argument("--columns", default="0:4096")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads((args.evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    hidden_width = int(manifest["hidden_width"])
    columns = _parse_columns(args.columns, hidden_width)
    powers = tuple(sorted(set(args.power or (0, 1, 2))))
    tensors, receipt = replay_moments(
        args.evidence_dir.resolve(),
        layer_id=args.layer,
        expert_id=args.expert,
        powers=powers,
        columns=columns,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    save_file(
        tensors,
        str(temporary),
        metadata={
            "schema": MOMENT_SCHEMA,
            "layer_id": str(args.layer),
            "expert_id": str(args.expert),
            "columns": f"{columns[0]}:{columns[1]}",
        },
    )
    os.replace(temporary, args.output)
    receipt["moment_file"] = args.output.name
    receipt["moment_file_sha256"] = sha256_file(args.output)
    receipt["moment_file_bytes"] = args.output.stat().st_size
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_receipt, receipt_path)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
