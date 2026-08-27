#!/usr/bin/env python3
"""Compare two sealed dense-prefill logit captures without loading a model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from kld_common import (
    canonical_json_sha256,
    load_capture_input_ids,
    sha256_file,
    summarize_kld,
    tokenwise_kld,
)
from safetensors.torch import load_file, save_file

CAPTURE_SCHEMA = "quant-toolkit.prefill-logits.v1"
REPORT_SCHEMA = "quant-toolkit.captured-prefill-kld.v1"


def _load_capture(directory: Path, label: str) -> tuple[dict, Path]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported {label} capture schema")
    expected_seal = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not expected_seal or canonical_json_sha256(body) != expected_seal:
        raise ValueError(f"{label} manifest seal mismatch")
    if manifest.get("storage_dtype") != "float32":
        raise ValueError(f"{label} capture must store float32 logits")
    if not manifest.get("windows"):
        raise ValueError(f"{label} capture contains no windows")
    return manifest, manifest_path


def _records_by_index(manifest: dict, label: str) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for record in manifest["windows"]:
        index = int(record["index"])
        if index in records:
            raise ValueError(f"duplicate {label} window index {index}")
        records[index] = record
    return records


def _capture_identity(directory: Path, manifest: dict, manifest_path: Path) -> dict:
    return {
        "directory": str(directory),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "model": manifest.get("model"),
        "model_revision": manifest.get("model_revision"),
        "run_label": manifest.get("run_label"),
        "role": manifest.get("role"),
        "storage_dtype": manifest.get("storage_dtype"),
        "engine_request": manifest.get("engine_request"),
        "token_sha256": manifest.get("token_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--candidate-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument(
        "--compute-dtype", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args(argv)

    if args.chunk_rows < 1:
        parser.error("chunk rows must be positive")
    reference_dir = Path(args.reference_logits).resolve()
    candidate_dir = Path(args.candidate_logits).resolve()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")

    reference_manifest, reference_manifest_path = _load_capture(
        reference_dir, "reference"
    )
    candidate_manifest, candidate_manifest_path = _load_capture(
        candidate_dir, "candidate"
    )
    if reference_manifest.get("token_sha256") != candidate_manifest.get("token_sha256"):
        raise ValueError("capture token-stream identity mismatch")

    reference_records = _records_by_index(reference_manifest, "reference")
    candidate_records = _records_by_index(candidate_manifest, "candidate")
    if reference_records.keys() != candidate_records.keys():
        raise ValueError("capture window-index sets differ")

    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    all_kld = []
    all_reference_top1 = []
    all_candidate_top1 = []
    windows = []
    started = time.time()
    for index in sorted(reference_records):
        reference_record = reference_records[index]
        candidate_record = candidate_records[index]
        reference_path = reference_dir / reference_record["file"]
        candidate_path = candidate_dir / candidate_record["file"]
        reference_sha = sha256_file(reference_path)
        candidate_sha = sha256_file(candidate_path)
        if reference_sha != reference_record.get("sha256"):
            raise ValueError(f"reference logit hash mismatch: {reference_path}")
        if candidate_sha != candidate_record.get("sha256"):
            raise ValueError(f"candidate logit hash mismatch: {candidate_path}")

        reference_tensors = load_file(reference_path)
        candidate_tensors = load_file(candidate_path)
        reference_input_ids = load_capture_input_ids(
            reference_dir, reference_record, reference_tensors
        )
        candidate_input_ids = load_capture_input_ids(
            candidate_dir, candidate_record, candidate_tensors
        )
        for label, tensors, input_ids in (
            ("reference", reference_tensors, reference_input_ids),
            ("candidate", candidate_tensors, candidate_input_ids),
        ):
            expected_tokens = int(tensors["logits"].shape[0]) + 1
            if input_ids.numel() != expected_tokens:
                raise ValueError(
                    f"{label} token/logit row alignment mismatch in window {index}"
                )
        if not torch.equal(
            reference_input_ids.to(torch.int64), candidate_input_ids.to(torch.int64)
        ):
            raise ValueError(f"input token mismatch in window {index}")
        kld, reference_top1, candidate_top1 = tokenwise_kld(
            reference_tensors["logits"],
            candidate_tensors["logits"],
            chunk_rows=args.chunk_rows,
            compute_dtype=compute_dtype,
        )
        window = summarize_kld(kld, reference_top1, candidate_top1)
        window.update(
            {
                "index": index,
                "reference_file": reference_record["file"],
                "reference_file_sha256": reference_sha,
                "candidate_file": candidate_record["file"],
                "candidate_file_sha256": candidate_sha,
                "input_ids_sha256": canonical_json_sha256(
                    reference_input_ids.to(torch.int64).tolist()
                ),
            }
        )
        windows.append(window)
        all_kld.append(kld)
        all_reference_top1.append(reference_top1)
        all_candidate_top1.append(candidate_top1)

    combined_kld = torch.cat(all_kld)
    combined_reference_top1 = torch.cat(all_reference_top1)
    combined_candidate_top1 = torch.cat(all_candidate_top1)
    aggregate = summarize_kld(
        combined_kld, combined_reference_top1, combined_candidate_top1
    )
    tokenwise_path = destination / "tokenwise.safetensors"
    tokenwise_temporary = destination / ".tokenwise.safetensors.incomplete"
    save_file(
        {
            "kld_nats": combined_kld,
            "kld_bits": combined_kld / math.log(2.0),
            "reference_top1": combined_reference_top1,
            "candidate_top1": combined_candidate_top1,
        },
        str(tokenwise_temporary),
    )
    os.replace(tokenwise_temporary, tokenwise_path)

    result = {
        "schema": REPORT_SCHEMA,
        "direction": "KL(reference||candidate)",
        "reference_capture": _capture_identity(
            reference_dir, reference_manifest, reference_manifest_path
        ),
        "candidate_capture": _capture_identity(
            candidate_dir, candidate_manifest, candidate_manifest_path
        ),
        "compute_dtype": args.compute_dtype,
        "aggregate": aggregate,
        "windows": windows,
        "tokenwise_file": tokenwise_path.name,
        "tokenwise_file_bytes": tokenwise_path.stat().st_size,
        "tokenwise_file_sha256": sha256_file(tokenwise_path),
        "elapsed_sec": time.time() - started,
    }
    result["summary_sha256"] = canonical_json_sha256(result)
    summary_path = destination / "summary.json"
    summary_temporary = destination / ".summary.json.incomplete"
    summary_temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary_path)
    print(
        json.dumps(
            {
                "event": "captured_prefill_kld_complete",
                "summary": str(summary_path),
                "aggregate": aggregate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
