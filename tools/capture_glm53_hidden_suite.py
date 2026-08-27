#!/usr/bin/env python3
"""Capture GLM-5.3 final-normalized hidden states for a sealed token suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
import torch
from kld_common import canonical_json_sha256, canonical_token_ids_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HIDDEN_SCHEMA = "quant-toolkit.prefill-hidden.v1"
LOGIT_SCHEMA = "quant-toolkit.prefill-logits.v1"
TEACHER_DATASET_SCHEMA = "quant-pipeline.glm53-bf16-teacher-logits-dataset.v1"
TOKEN_PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
CHUNK_RE = re.compile(r"hidden\.rows-(\d+)-(\d+)\.safetensors$")


def _verify_manifest_seal(manifest: dict, label: str) -> None:
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected != canonical_json_sha256(body):
        raise RuntimeError(f"{label} manifest seal mismatch")


def _write_manifest(path: Path, manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = canonical_json_sha256(body)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    manifest.clear()
    manifest.update(body)


def _record_index(record: dict) -> int:
    for key in ("index", "context_index", "window_index"):
        if key in record:
            return int(record[key])
    raise RuntimeError(f"suite record has no index: {record}")


def _load_suite_token_ids(suite_dir: Path, record: dict) -> torch.Tensor:
    if record.get("input_ids_file"):
        path = suite_dir / record["input_ids_file"]
        expected_file_sha256 = record.get("input_ids_file_sha256")
        if not expected_file_sha256 or sha256_file(path) != expected_file_sha256:
            raise RuntimeError(f"suite token file hash mismatch: {path}")
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            token_ids = torch.from_numpy(np.asarray(array))
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            token_ids = torch.tensor(payload, dtype=torch.int64)
        elif path.suffix == ".safetensors":
            token_ids = load_file(path)["input_ids"]
        else:
            raise RuntimeError(f"unsupported suite token file: {path}")
    elif record.get("token_file"):
        path = suite_dir / record["token_file"]
        if path.suffix == ".npy":
            token_ids = torch.from_numpy(np.asarray(np.load(path, allow_pickle=False)))
        else:
            token_ids = torch.tensor(
                json.loads(path.read_text(encoding="utf-8")), dtype=torch.int64
            )
    elif record.get("file"):
        path = suite_dir / record["file"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            if "input_ids" not in set(handle.keys()):
                raise RuntimeError(f"suite record has no input_ids: {path}")
            token_ids = handle.get_tensor("input_ids")
    else:
        raise RuntimeError("suite record does not identify an input-token artifact")

    if (
        token_ids.ndim != 1
        or token_ids.dtype == torch.bool
        or token_ids.is_floating_point()
    ):
        raise RuntimeError("suite token IDs must be a one-dimensional integer array")
    observed = canonical_token_ids_sha256(token_ids)
    expected = record.get("input_ids_canonical_sha256") or record.get(
        "token_ids_json_sha256"
    )
    if expected and observed != expected:
        raise RuntimeError("suite canonical token hash mismatch")
    return token_ids.to(torch.int64)


def _load_panel_role(panel_dir: Path, role: str) -> tuple[Path, dict, list[dict]]:
    panel_path = panel_dir / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel.get("schema") != TOKEN_PANEL_SCHEMA:
        raise RuntimeError(f"unexpected token-panel schema: {panel.get('schema')}")
    source_records = [
        record for record in panel.get("windows", []) if record.get("role") == role
    ]
    if not source_records:
        raise RuntimeError(f"token panel contains no {role!r} windows")
    records = []
    for index, source in enumerate(source_records):
        record = dict(source)
        record["index"] = index
        record["input_ids_file"] = f"arrays/{source['window_id']}.tokens.npy"
        record["input_ids_file_sha256"] = source["token_ids_sha256"]
        records.append(record)
    return panel_path, panel, records


def _validate_chunks(
    request_dir: Path, *, expected_rows: int, expected_width: int
) -> list[tuple[int, int, Path]]:
    chunks = []
    for path in request_dir.glob("hidden.rows-*.safetensors"):
        match = CHUNK_RE.fullmatch(path.name)
        if match:
            chunks.append((int(match.group(1)), int(match.group(2)), path))
    chunks.sort()
    next_row = 0
    for start, end, path in chunks:
        if start != next_row or end <= start:
            raise RuntimeError(
                f"non-contiguous hidden chunks: expected {next_row}, got [{start}, {end})"
            )
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["hidden_states"]:
                raise RuntimeError(f"unexpected hidden tensor keys: {path}")
            tensor = handle.get_slice("hidden_states")
            if tensor.get_dtype() != "BF16":
                raise RuntimeError(f"hidden capture is not BF16: {path}")
            if tensor.get_shape() != [end - start, expected_width]:
                raise RuntimeError(f"hidden capture shape mismatch: {path}")
            metadata = handle.metadata() or {}
            if metadata.get("semantic_point") != "after_final_rmsnorm_before_lm_head":
                raise RuntimeError(f"hidden semantic point is not sealed: {path}")
        next_row = end
    if next_row != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} contiguous hidden rows; captured {next_row}"
        )
    return chunks


def _finalize_window(
    chunks: list[tuple[int, int, Path]],
    *,
    output_path: Path,
    token_ids: torch.Tensor,
    index: int,
    hidden_width: int,
) -> dict:
    captured = torch.cat(
        [load_file(path)["hidden_states"] for _, _, path in chunks], dim=0
    )
    scored_rows = token_ids.numel() - 1
    hidden_states = captured[:scored_rows].contiguous()
    if hidden_states.dtype != torch.bfloat16 or list(hidden_states.shape) != [
        scored_rows,
        hidden_width,
    ]:
        raise RuntimeError(f"invalid finalized hidden tensor for window {index}")
    canonical_token_sha256 = canonical_token_ids_sha256(token_ids)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(
        {
            "hidden_states": hidden_states,
            "input_ids": token_ids.to(torch.int32).contiguous(),
        },
        str(temporary),
        metadata={
            "semantic_point": "after_final_rmsnorm_before_lm_head",
            "input_ids_canonical_sha256": canonical_token_sha256,
            "window_index": str(index),
        },
    )
    os.replace(temporary, output_path)
    return {
        "index": index,
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "positions": scored_rows,
        "hidden_width": hidden_width,
        "input_ids_canonical_sha256": canonical_token_sha256,
    }


def _validate_finalized(path: Path, record: dict) -> None:
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"finalized hidden hash mismatch: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"hidden_states", "input_ids"}:
            raise RuntimeError(f"unexpected finalized hidden keys: {path}")
        hidden = handle.get_slice("hidden_states")
        tokens = handle.get_slice("input_ids")
        if hidden.get_dtype() != "BF16" or hidden.get_shape() != [
            int(record["positions"]),
            int(record["hidden_width"]),
        ]:
            raise RuntimeError(f"invalid finalized hidden tensor: {path}")
        if tokens.get_shape() != [int(record["positions"]) + 1]:
            raise RuntimeError(f"invalid finalized input IDs: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="GLM-5.3-Flash-BF16")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite-manifest", type=Path)
    source.add_argument(
        "--panel-dir",
        type=Path,
        help="Sealed token-panel directory; pair with --panel-role.",
    )
    parser.add_argument(
        "--panel-role",
        choices=("fit", "conditional-fit", "selection", "confirmation", "final"),
    )
    parser.add_argument(
        "--token-panel-dir",
        type=Path,
        help=(
            "Published panel.json/arrays directory. Required only when the suite "
            "manifest is Brandon's teacher dataset manifest."
        ),
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hidden-width", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--stop-window", type=int)
    args = parser.parse_args(argv)

    if args.panel_dir is not None:
        if args.panel_role is None:
            parser.error("--panel-role is required with --panel-dir")
        suite_dir = args.panel_dir.resolve()
        suite_path, suite, records = _load_panel_role(suite_dir, args.panel_role)
    else:
        if args.panel_role is not None:
            parser.error("--panel-role is valid only with --panel-dir")
        suite_path = args.suite_manifest.resolve()
        suite_dir = suite_path.parent
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if suite.get("schema") == LOGIT_SCHEMA:
            _verify_manifest_seal(suite, "suite")
        if suite.get("schema") == TEACHER_DATASET_SCHEMA:
            if args.token_panel_dir is None:
                parser.error(
                    "--token-panel-dir is required for a teacher dataset manifest"
                )
            suite_dir = args.token_panel_dir.resolve()
            records = []
            for source_record in suite.get("logit_files", []):
                record = dict(source_record)
                record["index"] = len(records)
                record["input_ids_file"] = (
                    f"arrays/{source_record['window_id']}.tokens.npy"
                )
                record["input_ids_file_sha256"] = source_record["token_ids_sha256"]
                records.append(record)
        else:
            records = suite.get("windows", suite.get("contexts"))
    if not isinstance(records, list) or not records:
        raise RuntimeError("suite manifest contains no windows")
    suite_token_sha256 = suite.get("token_sha256")
    if not suite_token_sha256:
        suite_token_sha256 = canonical_json_sha256(
            [
                canonical_token_ids_sha256(_load_suite_token_ids(suite_dir, record))
                for record in records
            ]
        )
    stop = len(records) if args.stop_window is None else args.stop_window
    selected = records[args.start_window : stop]
    if not selected:
        parser.error("window range is empty")

    runtime_path = args.runtime_manifest.resolve()
    runtime_sha256 = sha256_file(runtime_path)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_manifest_seal(manifest, "hidden capture")
    else:
        manifest = {
            "schema": HIDDEN_SCHEMA,
            "role": "candidate",
            "run_label": args.run_name,
            "created_utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "hidden_width": args.hidden_width,
            "storage_dtype": "bfloat16",
            "semantic_point": "after_final_rmsnorm_before_lm_head",
            "suite_role": args.panel_role,
            "suite_manifest_file_sha256": sha256_file(suite_path),
            "token_sha256": suite_token_sha256,
            "runtime_manifest": str(runtime_path),
            "runtime_manifest_file_sha256": runtime_sha256,
            "windows": [],
        }
        _write_manifest(manifest_path, manifest)
    if manifest.get("runtime_manifest_file_sha256") != runtime_sha256:
        raise RuntimeError("runtime manifest changed during resumable capture")
    if manifest.get("suite_manifest_file_sha256") != sha256_file(suite_path):
        raise RuntimeError("suite manifest changed during resumable capture")

    completed = {int(item["index"]): item for item in manifest["windows"]}
    for suite_record in selected:
        index = _record_index(suite_record)
        token_ids = _load_suite_token_ids(suite_dir, suite_record)
        output_path = args.output_dir / f"hidden_{index:04d}.safetensors"
        if index in completed:
            _validate_finalized(output_path, completed[index])
            print(f"window {index:04d}: already captured", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"unrecorded hidden output exists: {output_path}")

        before = {item.name for item in args.capture_dir.iterdir() if item.is_dir()}
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
        after = {item.name for item in args.capture_dir.iterdir() if item.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(
                f"expected one raw capture directory for window {index}; got {created}"
            )
        request_dir = args.capture_dir / created[0]
        chunks = _validate_chunks(
            request_dir,
            expected_rows=token_ids.numel(),
            expected_width=args.hidden_width,
        )
        record = _finalize_window(
            chunks,
            output_path=output_path,
            token_ids=token_ids,
            index=index,
            hidden_width=args.hidden_width,
        )
        payload = response.json()
        record.update(
            {
                "elapsed_seconds": elapsed,
                "request_id": payload.get("id"),
                "raw_capture_directory": str(request_dir),
                "raw_chunk_files": [path.name for _, _, path in chunks],
                "source_window_id": suite_record.get("window_id"),
                "source_domain": suite_record.get("domain"),
            }
        )
        manifest["windows"].append(record)
        manifest["windows"].sort(key=lambda item: int(item["index"]))
        manifest["total_bytes"] = sum(
            int(item["bytes"]) for item in manifest["windows"]
        )
        _write_manifest(manifest_path, manifest)
        completed[index] = record
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
