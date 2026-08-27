#!/usr/bin/env python3
"""Replay a sealed GLM-5.3 hidden capture through the shared BF16 LM head."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from kld_common import canonical_json_sha256, canonical_token_ids_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HIDDEN_SCHEMA = "quant-toolkit.prefill-hidden.v1"
HEAD_SCHEMA = "quant-toolkit.glm53-lm-head.v1"
LOGIT_SCHEMA = "quant-toolkit.prefill-logits.v1"


def _verify_seal(document: dict, field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != canonical_json_sha256(body):
        raise RuntimeError(f"{label} seal mismatch")


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


def _configure_compute(device: torch.device) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
    return {
        "device": str(device),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "bf16_reduced_precision_reduction": False,
        "deterministic_algorithms": device.type == "cuda",
        "projection": "torch.nn.functional.linear",
        "output_cast": "float32",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dir", type=Path, required=True)
    parser.add_argument("--lm-head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--row-chunk", type=int, default=64)
    args = parser.parse_args(argv)
    if args.row_chunk < 1:
        parser.error("row chunk must be positive")

    hidden_dir = args.hidden_dir.resolve()
    head_dir = args.lm_head_dir.resolve()
    output_dir = args.output_dir.resolve()
    hidden_manifest_path = hidden_dir / "manifest.json"
    head_manifest_path = head_dir / "manifest.json"
    hidden_manifest = json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
    head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
    if hidden_manifest.get("schema") != HIDDEN_SCHEMA:
        raise RuntimeError("unsupported hidden-capture schema")
    if head_manifest.get("schema") != HEAD_SCHEMA:
        raise RuntimeError("unsupported LM-head schema")
    _verify_seal(hidden_manifest, "manifest_sha256", "hidden capture")
    _verify_seal(head_manifest, "manifest_sha256", "LM head")

    head_path = head_dir / head_manifest["file"]
    if sha256_file(head_path) != head_manifest["file_sha256"]:
        raise RuntimeError("LM-head file hash mismatch")
    head_cpu = load_file(head_path)[head_manifest["tensor_key"]]
    expected_shape = [int(value) for value in head_manifest["shape"]]
    if head_cpu.dtype != torch.bfloat16 or list(head_cpu.shape) != expected_shape:
        raise RuntimeError("LM-head tensor identity mismatch")
    vocab_size, hidden_width = expected_shape

    device = torch.device(args.device)
    compute = _configure_compute(device)
    head = head_cpu.to(device)
    del head_cpu

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_seal(manifest, "manifest_sha256", "logit replay")
    else:
        manifest = {
            "schema": LOGIT_SCHEMA,
            "role": hidden_manifest.get("role", "candidate"),
            "run_label": hidden_manifest.get("run_label"),
            "model": hidden_manifest.get("model"),
            "model_revision": hidden_manifest.get("model_revision"),
            "storage_dtype": "float32",
            "token_sha256": hidden_manifest.get("token_sha256"),
            "source_hidden_manifest_file_sha256": sha256_file(hidden_manifest_path),
            "source_hidden_manifest_sha256": hidden_manifest["manifest_sha256"],
            "lm_head_manifest_file_sha256": sha256_file(head_manifest_path),
            "lm_head_manifest_sha256": head_manifest["manifest_sha256"],
            "engine_request": {
                "hidden_runtime_manifest": hidden_manifest.get("runtime_manifest"),
                "hidden_runtime_manifest_file_sha256": hidden_manifest.get(
                    "runtime_manifest_file_sha256"
                ),
                "lm_head_source_model_revision": head_manifest["source_model_revision"],
                "replay_compute": compute,
            },
            "windows": [],
        }
        _write_manifest(manifest_path, manifest)
    if manifest.get("source_hidden_manifest_file_sha256") != sha256_file(
        hidden_manifest_path
    ):
        raise RuntimeError("hidden manifest changed during resumable replay")
    if manifest.get("lm_head_manifest_file_sha256") != sha256_file(head_manifest_path):
        raise RuntimeError("LM-head manifest changed during resumable replay")

    completed = {int(item["index"]): item for item in manifest["windows"]}
    for hidden_record in hidden_manifest["windows"]:
        index = int(hidden_record["index"])
        hidden_path = hidden_dir / hidden_record["file"]
        if sha256_file(hidden_path) != hidden_record["sha256"]:
            raise RuntimeError(f"hidden artifact hash mismatch: {hidden_path}")
        output_path = output_dir / f"logits_{index:04d}.safetensors"
        if index in completed:
            if sha256_file(output_path) != completed[index]["sha256"]:
                raise RuntimeError(f"replayed logit hash mismatch: {output_path}")
            print(f"window {index:04d}: already replayed", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"unrecorded replay output exists: {output_path}")

        with safe_open(hidden_path, framework="pt", device="cpu") as handle:
            hidden_states = handle.get_tensor("hidden_states")
            input_ids = handle.get_tensor("input_ids")
        positions = int(hidden_record["positions"])
        if hidden_states.dtype != torch.bfloat16 or list(hidden_states.shape) != [
            positions,
            hidden_width,
        ]:
            raise RuntimeError(f"hidden tensor shape/dtype mismatch: {hidden_path}")
        if list(input_ids.shape) != [positions + 1]:
            raise RuntimeError(f"hidden token alignment mismatch: {hidden_path}")
        canonical_token_sha256 = canonical_token_ids_sha256(input_ids)
        if canonical_token_sha256 != hidden_record["input_ids_canonical_sha256"]:
            raise RuntimeError(f"hidden token hash mismatch: {hidden_path}")

        logits = torch.empty((positions, vocab_size), dtype=torch.float32, device="cpu")
        started = time.monotonic()
        with torch.inference_mode():
            for row_start in range(0, positions, args.row_chunk):
                row_end = min(row_start + args.row_chunk, positions)
                source = hidden_states[row_start:row_end].to(device)
                projected = F.linear(source, head).float()
                logits[row_start:row_end].copy_(projected.to("cpu"))
                del source, projected
        elapsed = time.monotonic() - started
        temporary = output_path.with_suffix(output_path.suffix + ".incomplete")
        save_file(
            {"logits": logits, "input_ids": input_ids.to(torch.int32).contiguous()},
            str(temporary),
            metadata={
                "semantic_point": "shared_bf16_lm_head_output",
                "input_ids_canonical_sha256": canonical_token_sha256,
                "source_hidden_file_sha256": hidden_record["sha256"],
            },
        )
        os.replace(temporary, output_path)
        record = {
            "index": index,
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "positions": positions,
            "vocab_size": vocab_size,
            "input_ids_canonical_sha256": canonical_token_sha256,
            "source_hidden_file_sha256": hidden_record["sha256"],
            "elapsed_seconds": elapsed,
        }
        manifest["windows"].append(record)
        manifest["windows"].sort(key=lambda item: int(item["index"]))
        manifest["total_bytes"] = sum(
            int(item["bytes"]) for item in manifest["windows"]
        )
        _write_manifest(manifest_path, manifest)
        completed[index] = record
        print(json.dumps(record, sort_keys=True), flush=True)
        del hidden_states, input_ids, logits
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
