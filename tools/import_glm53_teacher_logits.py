#!/usr/bin/env python3
"""Seal Brandon's GLM-5.3 BF16 teacher logits as a reusable KLD capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from kld_common import (
    canonical_json_sha256,
    canonical_token_ids_sha256,
    sha256_file,
)
from safetensors import safe_open

DATASET_SCHEMA = "quant-pipeline.glm53-bf16-teacher-logits-dataset.v1"
CAPTURE_SCHEMA = "quant-toolkit.prefill-logits.v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _link(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--token-panel-dir",
        type=Path,
        help="Directory containing panel.json and arrays/ (defaults to dataset dir).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", default="bf16-transformers-ep4-eager-no-cache")
    parser.add_argument(
        "--link-mode", choices=("symlink", "hardlink"), default="symlink"
    )
    parser.add_argument(
        "--verify-logits",
        choices=("sha256", "size"),
        default="sha256",
        help="Full SHA-256 is required for a publishable import.",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir.resolve()
    panel_dir = (args.token_panel_dir or dataset_dir).resolve()
    output_dir = args.output_dir.resolve()
    dataset_manifest_path = dataset_dir / "dataset-manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    if dataset_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported GLM-5.3 teacher dataset schema")

    logit_records = dataset_manifest.get("logit_files")
    if not isinstance(logit_records, list) or not logit_records:
        raise ValueError("teacher dataset manifest contains no logit files")

    preflight: list[dict] = []
    canonical_token_hashes: list[str] = []
    for index, source_record in enumerate(logit_records):
        window_id = str(source_record["window_id"])
        logit_path = dataset_dir / source_record["path"]
        token_path = panel_dir / "arrays" / f"{window_id}.tokens.npy"
        if not logit_path.is_file():
            raise FileNotFoundError(logit_path)
        if not token_path.is_file():
            raise FileNotFoundError(
                f"Missing published token payload {token_path}; the receipt/hash alone "
                "cannot reproduce KLD."
            )

        expected_logit_bytes = int(source_record["bytes"])
        if logit_path.stat().st_size != expected_logit_bytes:
            raise RuntimeError(f"teacher logit size mismatch: {logit_path}")
        observed_logit_sha256 = (
            sha256_file(logit_path)
            if args.verify_logits == "sha256"
            else source_record["sha256"]
        )
        if (
            args.verify_logits == "sha256"
            and observed_logit_sha256 != source_record["sha256"]
        ):
            raise RuntimeError(f"teacher logit hash mismatch: {logit_path}")

        with safe_open(logit_path, framework="np") as handle:
            if list(handle.keys()) != ["logits"]:
                raise RuntimeError(f"unexpected tensor keys in {logit_path}")
            logit_slice = handle.get_slice("logits")
            expected_shape = [
                int(source_record["prediction_positions"]),
                int(dataset_manifest["vocab_size"]),
            ]
            if logit_slice.get_dtype() != "F32":
                raise RuntimeError(f"teacher logits are not F32: {logit_path}")
            if logit_slice.get_shape() != expected_shape:
                raise RuntimeError(f"teacher logit shape mismatch: {logit_path}")

        observed_token_file_sha256 = sha256_file(token_path)
        if observed_token_file_sha256 != source_record["token_ids_sha256"]:
            raise RuntimeError(f"token payload hash mismatch: {token_path}")
        token_ids = np.load(token_path, allow_pickle=False)
        if token_ids.ndim != 1 or not np.issubdtype(token_ids.dtype, np.integer):
            raise RuntimeError(f"invalid token array: {token_path}")
        if token_ids.size != int(source_record["prediction_positions"]) + 1:
            raise RuntimeError(f"token/logit row alignment mismatch: {token_path}")
        canonical_token_sha256 = canonical_token_ids_sha256(
            torch.from_numpy(np.asarray(token_ids))
        )
        canonical_token_hashes.append(canonical_token_sha256)
        preflight.append(
            {
                "index": index,
                "source": source_record,
                "logit_path": logit_path,
                "logit_sha256": observed_logit_sha256,
                "token_path": token_path,
                "token_file_sha256": observed_token_file_sha256,
                "token_canonical_sha256": canonical_token_sha256,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")

    windows = []
    for item in preflight:
        index = item["index"]
        logit_name = f"logits_{index:04d}.safetensors"
        token_name = f"input_ids_{index:04d}.npy"
        _link(item["logit_path"], output_dir / logit_name, args.link_mode)
        _link(item["token_path"], output_dir / token_name, args.link_mode)
        source_record = item["source"]
        windows.append(
            {
                "index": index,
                "window_id": source_record["window_id"],
                "domain": source_record.get("domain"),
                "document_id": source_record.get("document_id"),
                "file": logit_name,
                "bytes": int(source_record["bytes"]),
                "sha256": item["logit_sha256"],
                "positions": int(source_record["prediction_positions"]),
                "vocab_size": int(dataset_manifest["vocab_size"]),
                "input_ids_file": token_name,
                "input_ids_file_sha256": item["token_file_sha256"],
                "input_ids_canonical_sha256": item["token_canonical_sha256"],
            }
        )

    backend_path = dataset_dir / "backend.json"
    plan_path = dataset_dir / "plan.json"
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "role": "canonical",
        "run_label": args.run_label,
        "model": dataset_manifest["source_model"],
        "model_revision": dataset_manifest["model_revision"],
        "storage_dtype": "float32",
        "token_sha256": canonical_json_sha256(canonical_token_hashes),
        "source_dataset_manifest_file_sha256": sha256_file(dataset_manifest_path),
        "source_dataset_sha256": dataset_manifest["dataset_sha256"],
        "source_teacher_capture_receipt_sha256": dataset_manifest[
            "teacher_capture_receipt_sha256"
        ],
        "source_token_panel_receipt_sha256": dataset_manifest[
            "token_panel_receipt_sha256"
        ],
        "engine_request": {
            "backend": _read_json(backend_path) if backend_path.is_file() else None,
            "plan": _read_json(plan_path) if plan_path.is_file() else None,
        },
        "link_mode": args.link_mode,
        "logit_verification": args.verify_logits,
        "windows": windows,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    temporary = output_dir / ".manifest.json.incomplete"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "event": "glm53_teacher_capture_imported",
                "windows": len(windows),
                "positions": sum(item["positions"] for item in windows),
                "manifest_sha256": manifest["manifest_sha256"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
