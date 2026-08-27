#!/usr/bin/env python3
"""Capture sealed GLM-5.3 routed-MoE evidence from a cstech endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
import torch
from kld_common import canonical_json_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import load_file

SCHEMA = "quant-toolkit.glm53-routed-evidence.v1"
PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
EXPECTED_ROUTED_LAYERS = tuple(range(3, 45))
EXPECTED_KEYS = {
    "hidden_states",
    "router_logits",
    "topk_ids",
    "topk_weights",
}


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _seal_manifest(path: Path, manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = canonical_json_sha256(body)
    _write_json_atomic(path, body)
    manifest.clear()
    manifest.update(body)


def _verify_manifest(manifest: dict) -> None:
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected != canonical_json_sha256(body):
        raise RuntimeError("routed-evidence manifest seal mismatch")


def _load_panel(panel_dir: Path, role: str) -> tuple[Path, dict, list[dict]]:
    panel_path = panel_dir / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("schema") != PANEL_SCHEMA:
        raise RuntimeError(f"unexpected token-panel schema: {panel.get('schema')}")
    records = [record for record in panel.get("windows", []) if record.get("role") == role]
    if not records:
        raise RuntimeError(f"token panel contains no {role!r} windows")
    return panel_path, panel, records


def _load_tokens(panel_dir: Path, record: dict) -> tuple[Path, np.ndarray]:
    token_path = panel_dir / "arrays" / f"{record['window_id']}.tokens.npy"
    if sha256_file(token_path) != record["token_ids_sha256"]:
        raise RuntimeError(f"token file hash mismatch: {token_path}")
    token_ids = np.load(token_path, allow_pickle=False)
    if token_ids.ndim != 1 or token_ids.dtype.kind not in "iu":
        raise RuntimeError(f"token IDs must be a one-dimensional integer array: {token_path}")
    if token_ids.shape[0] != int(record["prediction_positions"]) + 1:
        raise RuntimeError(f"token count mismatch: {token_path}")
    return token_path, token_ids.astype(np.int64, copy=False)


def _validate_layer(
    layer_path: Path,
    *,
    layer_id: int,
    rows: int,
    hidden_width: int,
    num_experts: int,
    top_k: int,
) -> dict:
    with safe_open(layer_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != EXPECTED_KEYS:
            raise RuntimeError(f"unexpected tensors in {layer_path}: {list(handle.keys())}")
        expected = {
            "hidden_states": ("BF16", [rows, hidden_width]),
            "router_logits": ("F32", [rows, num_experts]),
            "topk_ids": ("I16", [rows, top_k]),
            "topk_weights": ("F32", [rows, top_k]),
        }
        for key, (dtype, shape) in expected.items():
            tensor = handle.get_slice(key)
            if tensor.get_dtype() != dtype or tensor.get_shape() != shape:
                raise RuntimeError(
                    f"invalid {key} in {layer_path}: "
                    f"dtype={tensor.get_dtype()} shape={tensor.get_shape()}"
                )
        metadata = handle.metadata() or {}
        if metadata.get("layer_id") != str(layer_id):
            raise RuntimeError(f"layer metadata mismatch: {layer_path}")
        if metadata.get("hidden_semantic_point") != (
            "input_to_routed_moe_before_sequence_parallel_chunk"
        ):
            raise RuntimeError(f"hidden semantic point mismatch: {layer_path}")
        if metadata.get("router_logit_semantic_point") != (
            "pre_sigmoid_gate_linear_output"
        ):
            raise RuntimeError(f"router semantic point mismatch: {layer_path}")
        if metadata.get("route_semantic_point") != "exact_vllm_router_selection":
            raise RuntimeError(f"route semantic point mismatch: {layer_path}")

    tensors = load_file(layer_path, device="cpu")
    ids = tensors["topk_ids"]
    weights = tensors["topk_weights"]
    if int(ids.min()) < 0 or int(ids.max()) >= num_experts:
        raise RuntimeError(f"expert ID out of range: {layer_path}")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise RuntimeError(f"invalid route weights: {layer_path}")
    route_weight_sum = weights.sum(dim=-1)
    if not torch.allclose(
        route_weight_sum,
        torch.full_like(route_weight_sum, 2.5),
        rtol=2e-5,
        atol=2e-5,
    ):
        raise RuntimeError(f"route weights do not sum to routed scale 2.5: {layer_path}")
    return {
        "layer_id": layer_id,
        "file": layer_path.name,
        "bytes": layer_path.stat().st_size,
        "sha256": sha256_file(layer_path),
    }


def _finalize_pass(
    raw_pass: Path,
    destination: Path,
    *,
    token_ids: np.ndarray,
    hidden_width: int,
    num_experts: int,
    top_k: int,
) -> list[dict]:
    observed = sorted(
        int(path.stem.removeprefix("layer-"))
        for path in raw_pass.glob("layer-*.safetensors")
    )
    if tuple(observed) != EXPECTED_ROUTED_LAYERS:
        raise RuntimeError(
            f"expected routed layers {EXPECTED_ROUTED_LAYERS}; captured {tuple(observed)}"
        )
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite finalized routed evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for layer_id in EXPECTED_ROUTED_LAYERS:
        layer_path = raw_pass / f"layer-{layer_id:04d}.safetensors"
        layer_fd = os.open(layer_path, os.O_RDONLY)
        try:
            os.fsync(layer_fd)
        finally:
            os.close(layer_fd)
    _fsync_directory(raw_pass)
    os.replace(raw_pass, destination)
    _fsync_directory(destination.parent)
    try:
        return [
            _validate_layer(
                destination / f"layer-{layer_id:04d}.safetensors",
                layer_id=layer_id,
                rows=token_ids.shape[0],
                hidden_width=hidden_width,
                num_experts=num_experts,
                top_k=top_k,
            )
            for layer_id in EXPECTED_ROUTED_LAYERS
        ]
    except Exception:
        failed = destination.with_name(destination.name + ".failed-validation")
        if not failed.exists():
            os.replace(destination, failed)
            _fsync_directory(failed.parent)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="GLM-5.3-Flash-BF16")
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("fit", "conditional-fit"),
        required=True,
    )
    parser.add_argument("--raw-capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hidden-width", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    args = parser.parse_args(argv)

    panel_dir = args.panel_dir.resolve()
    panel_path, panel, role_records = _load_panel(panel_dir, args.role)
    stop = len(role_records) if args.stop_window is None else args.stop_window
    selected = role_records[args.start_window : stop]
    if not selected:
        parser.error("selected window range is empty")

    raw_dir = args.raw_capture_dir.resolve()
    output_dir = args.output_dir.resolve()
    windows_dir = output_dir / "windows"
    raw_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.runtime_manifest.resolve()
    runtime_sha256 = sha256_file(runtime_path)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_manifest(manifest)
    else:
        manifest = {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "run_name": args.run_name,
            "role": args.role,
            "model": args.model,
            "panel_schema": panel["schema"],
            "panel_file_sha256": sha256_file(panel_path),
            "sealed_corpus_sha256": panel["sealed_corpus_sha256"],
            "runtime_manifest": str(runtime_path),
            "runtime_manifest_file_sha256": runtime_sha256,
            "routed_layers": list(EXPECTED_ROUTED_LAYERS),
            "hidden_width": args.hidden_width,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "route_weight_powers": [0, 1, 2],
            "cross_coordinate_representation": (
                "exact_factorized_rows; reconstruct X^T diag(route_weight^p) X in float64"
            ),
            "windows": [],
        }
        _seal_manifest(manifest_path, manifest)

    fixed_bindings = {
        "role": args.role,
        "panel_file_sha256": sha256_file(panel_path),
        "runtime_manifest_file_sha256": runtime_sha256,
        "hidden_width": args.hidden_width,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
    }
    for key, expected in fixed_bindings.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"resumable manifest binding changed for {key}")

    completed = {record["window_id"]: record for record in manifest["windows"]}
    for record in selected:
        window_id = record["window_id"]
        token_path, token_ids = _load_tokens(panel_dir, record)
        destination = windows_dir / window_id
        if window_id in completed:
            if not destination.is_dir():
                raise RuntimeError(f"manifest window is missing: {destination}")
            print(f"{window_id}: already captured", flush=True)
            continue

        before = {path.name for path in raw_dir.glob("pass-*") if path.is_dir()}
        request = {
            "model": args.model,
            "prompt": token_ids.tolist(),
            "max_tokens": 1,
            "temperature": 0,
            "seed": 1,
            "ignore_eos": True,
        }
        started = time.monotonic()
        response = requests.post(args.url, json=request, timeout=args.timeout)
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        after = {path.name for path in raw_dir.glob("pass-*") if path.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(
                f"expected exactly one complete routed pass for {window_id}; got {created}"
            )
        raw_pass = raw_dir / created[0]
        layer_records = _finalize_pass(
            raw_pass,
            destination,
            token_ids=token_ids,
            hidden_width=args.hidden_width,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        payload = response.json()
        window_record = {
            "window_id": window_id,
            "document_id": record["document_id"],
            "domain": record["domain"],
            "token_file": str(token_path),
            "token_file_sha256": record["token_ids_sha256"],
            "tokens": int(token_ids.shape[0]),
            "elapsed_seconds": elapsed,
            "request_id": payload.get("id"),
            "bytes": sum(item["bytes"] for item in layer_records),
            "layers": layer_records,
        }
        manifest["windows"].append(window_record)
        manifest["windows"].sort(key=lambda item: item["window_id"])
        manifest["total_bytes"] = sum(item["bytes"] for item in manifest["windows"])
        _seal_manifest(manifest_path, manifest)
        completed[window_id] = window_record
        print(json.dumps(window_record, sort_keys=True), flush=True)

    if any(path.is_dir() for path in raw_dir.glob("pass-*")):
        leftovers = sorted(path.name for path in raw_dir.glob("pass-*") if path.is_dir())
        raise RuntimeError(f"unclaimed raw routed capture directories remain: {leftovers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
