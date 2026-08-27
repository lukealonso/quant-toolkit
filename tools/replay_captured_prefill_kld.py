#!/usr/bin/env python3
"""Independently replay a KLD report made from two sealed logit captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

CAPTURE_SCHEMA = "quant-toolkit.prefill-logits.v1"
REPORT_SCHEMA = "quant-toolkit.captured-prefill-kld.v1"
VERIFICATION_SCHEMA = "quant-toolkit.captured-prefill-kld-verification.v1"


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_token_ids_sha256(token_ids: np.ndarray) -> str:
    if token_ids.ndim != 1:
        raise ValueError(f"input token IDs must be rank 1, got {token_ids.shape}")
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise ValueError(f"input token IDs must be integers, got {token_ids.dtype}")
    return canonical_json_sha256(token_ids.astype(np.int64).tolist())


def _load_capture_input_ids(
    directory: Path,
    record: dict,
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    if "input_ids" in tensors:
        token_ids = tensors["input_ids"]
    else:
        relative_path = record.get("input_ids_file")
        if not relative_path:
            raise ValueError(
                "capture window has neither embedded nor external input_ids"
            )
        path = directory / relative_path
        expected_file_sha256 = record.get("input_ids_file_sha256")
        if not expected_file_sha256:
            raise ValueError(f"external input IDs have no file hash: {path}")
        if sha256_file(path) != expected_file_sha256:
            raise ValueError(f"external input-token hash mismatch: {path}")
        if path.suffix == ".npy":
            token_ids = np.load(path, allow_pickle=False)
        elif path.suffix == ".safetensors":
            external_tensors = load_file(str(path))
            if "input_ids" not in external_tensors:
                raise ValueError(f"external token artifact has no input_ids: {path}")
            token_ids = external_tensors["input_ids"]
        else:
            raise ValueError(f"unsupported external token artifact: {path}")

    observed_canonical_sha256 = _canonical_token_ids_sha256(token_ids)
    expected_canonical_sha256 = record.get("input_ids_canonical_sha256")
    if (
        expected_canonical_sha256
        and observed_canonical_sha256 != expected_canonical_sha256
    ):
        raise ValueError("canonical input-token identity mismatch")
    return token_ids


def _verify_seal(document: dict, field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != canonical_json_sha256(body):
        raise ValueError(f"{label} seal mismatch")


def _load_capture(directory: Path, label: str) -> tuple[dict, Path]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported {label} capture schema")
    _verify_seal(manifest, "manifest_sha256", f"{label} manifest")
    if manifest.get("storage_dtype") != "float32":
        raise ValueError(f"{label} capture must store float32 logits")
    return manifest, manifest_path


def _records_by_index(manifest: dict, label: str) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for record in manifest.get("windows", []):
        index = int(record["index"])
        if index in records:
            raise ValueError(f"duplicate {label} window index {index}")
        records[index] = record
    if not records:
        raise ValueError(f"{label} capture contains no windows")
    return records


def independent_tokenwise_kld(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    chunk_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KL(reference || candidate) using explicit float64 NumPy math."""
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference and candidate logits must both be rank 2")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"logit shape mismatch: reference={reference.shape} "
            f"candidate={candidate.shape}"
        )
    if chunk_rows < 1:
        raise ValueError("chunk rows must be positive")

    result = np.empty(reference.shape[0], dtype=np.float64)
    reference_top1 = np.empty(reference.shape[0], dtype=np.int64)
    candidate_top1 = np.empty(reference.shape[0], dtype=np.int64)
    for start in range(0, len(result), chunk_rows):
        stop = min(start + chunk_rows, len(result))
        target = np.asarray(reference[start:stop], dtype=np.float64)
        observed = np.asarray(candidate[start:stop], dtype=np.float64)
        if not np.isfinite(target).all() or not np.isfinite(observed).all():
            raise ValueError(f"non-finite dense logits in rows [{start}, {stop})")
        target_shift = target - np.max(target, axis=1, keepdims=True)
        observed_shift = observed - np.max(observed, axis=1, keepdims=True)
        target_partition = np.sum(np.exp(target_shift), axis=1, keepdims=True)
        observed_partition = np.sum(np.exp(observed_shift), axis=1, keepdims=True)
        target_probability = np.exp(target_shift) / target_partition
        target_log_probability = target_shift - np.log(target_partition)
        observed_log_probability = observed_shift - np.log(observed_partition)
        result[start:stop] = np.sum(
            target_probability * (target_log_probability - observed_log_probability),
            axis=1,
        )
        reference_top1[start:stop] = np.argmax(target, axis=1)
        candidate_top1[start:stop] = np.argmax(observed, axis=1)
    return result, reference_top1, candidate_top1


def _stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_9": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def _compare_scalar(
    observed: float, expected: float, tolerance: float, label: str
) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{label} differs: replay={observed:.17g} "
            f"producer={expected:.17g} tolerance={tolerance:.3g}"
        )


def _verify_capture_identity(
    identity: dict, manifest: dict, manifest_path: Path, label: str
) -> None:
    if identity.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError(f"report references a different {label} manifest")
    if identity.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise ValueError(f"report {label} manifest-file identity mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--candidate-logits", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)

    reference_dir = Path(args.reference_logits).resolve()
    candidate_dir = Path(args.candidate_logits).resolve()
    report_dir = Path(args.report).resolve()
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
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported captured-prefill KLD report schema")
    _verify_seal(summary, "summary_sha256", "KLD summary")
    _verify_capture_identity(
        summary["reference_capture"],
        reference_manifest,
        reference_manifest_path,
        "reference",
    )
    _verify_capture_identity(
        summary["candidate_capture"],
        candidate_manifest,
        candidate_manifest_path,
        "candidate",
    )

    reference_records = _records_by_index(reference_manifest, "reference")
    candidate_records = _records_by_index(candidate_manifest, "candidate")
    report_records = {int(record["index"]): record for record in summary["windows"]}
    if not (
        reference_records.keys() == candidate_records.keys() == report_records.keys()
    ):
        raise ValueError("capture/report window-index sets differ")

    all_values = []
    all_reference_top1 = []
    all_candidate_top1 = []
    windows = []
    for index in sorted(reference_records):
        reference_record = reference_records[index]
        candidate_record = candidate_records[index]
        report_record = report_records[index]
        reference_path = reference_dir / reference_record["file"]
        candidate_path = candidate_dir / candidate_record["file"]
        reference_sha = sha256_file(reference_path)
        candidate_sha = sha256_file(candidate_path)
        if reference_sha != reference_record.get("sha256"):
            raise ValueError(f"reference logit hash mismatch: {reference_path}")
        if candidate_sha != candidate_record.get("sha256"):
            raise ValueError(f"candidate logit hash mismatch: {candidate_path}")
        if reference_sha != report_record.get("reference_file_sha256"):
            raise ValueError(f"report reference logit identity mismatch: {index}")
        if candidate_sha != report_record.get("candidate_file_sha256"):
            raise ValueError(f"report candidate logit identity mismatch: {index}")

        reference_tensors = load_file(str(reference_path))
        candidate_tensors = load_file(str(candidate_path))
        reference_input_ids = _load_capture_input_ids(
            reference_dir, reference_record, reference_tensors
        )
        candidate_input_ids = _load_capture_input_ids(
            candidate_dir, candidate_record, candidate_tensors
        )
        for label, tensors, input_ids in (
            ("reference", reference_tensors, reference_input_ids),
            ("candidate", candidate_tensors, candidate_input_ids),
        ):
            expected_tokens = int(tensors["logits"].shape[0]) + 1
            if input_ids.size != expected_tokens:
                raise ValueError(
                    f"{label} token/logit row alignment mismatch in window {index}"
                )
        if not np.array_equal(
            reference_input_ids.astype(np.int64),
            candidate_input_ids.astype(np.int64),
        ):
            raise ValueError(f"input token mismatch in window {index}")
        input_ids_sha = _canonical_token_ids_sha256(reference_input_ids)
        if input_ids_sha != report_record.get("input_ids_sha256"):
            raise ValueError(f"report input-token identity mismatch: {index}")

        values, reference_top1, candidate_top1 = independent_tokenwise_kld(
            reference_tensors["logits"],
            candidate_tensors["logits"],
            chunk_rows=args.chunk_rows,
        )
        window = {
            "index": index,
            "positions": int(values.size),
            "kld_nats": _stats(values),
            "top1_agreement": float(np.mean(reference_top1 == candidate_top1)),
            "reference_file_sha256": reference_sha,
            "candidate_file_sha256": candidate_sha,
        }
        for key, observed in window["kld_nats"].items():
            _compare_scalar(
                observed,
                float(report_record["kld_nats"][key]),
                args.absolute_tolerance,
                f"window {index} kld_nats.{key}",
            )
        _compare_scalar(
            window["top1_agreement"],
            float(report_record["top1_agreement"]),
            args.absolute_tolerance,
            f"window {index} top1 agreement",
        )
        windows.append(window)
        all_values.append(values)
        all_reference_top1.append(reference_top1)
        all_candidate_top1.append(candidate_top1)

    values = np.concatenate(all_values)
    reference_top1 = np.concatenate(all_reference_top1)
    candidate_top1 = np.concatenate(all_candidate_top1)
    bits = values / math.log(2.0)
    tokenwise_path = report_dir / summary["tokenwise_file"]
    if sha256_file(tokenwise_path) != summary.get("tokenwise_file_sha256"):
        raise ValueError("producer tokenwise artifact hash mismatch")
    producer = load_file(str(tokenwise_path))
    max_kld_difference = float(np.max(np.abs(values - producer["kld_nats"])))
    max_bits_difference = float(np.max(np.abs(bits - producer["kld_bits"])))
    if max_kld_difference > args.absolute_tolerance:
        raise RuntimeError("independent tokenwise KLD differs from producer")
    if max_bits_difference > args.absolute_tolerance:
        raise RuntimeError("independent tokenwise KLD bits differ from producer")
    if not np.array_equal(reference_top1, producer["reference_top1"]):
        raise RuntimeError("independent reference top-1 IDs differ from producer")
    if not np.array_equal(candidate_top1, producer["candidate_top1"]):
        raise RuntimeError("independent candidate top-1 IDs differ from producer")

    aggregate_nats = _stats(values)
    aggregate_bits = _stats(bits)
    top1_agreement = float(np.mean(reference_top1 == candidate_top1))
    for key, observed in aggregate_nats.items():
        _compare_scalar(
            observed,
            float(summary["aggregate"]["kld_nats"][key]),
            args.absolute_tolerance,
            f"aggregate kld_nats.{key}",
        )
    for key, observed in aggregate_bits.items():
        _compare_scalar(
            observed,
            float(summary["aggregate"]["kld_bits"][key]),
            args.absolute_tolerance,
            f"aggregate kld_bits.{key}",
        )
    _compare_scalar(
        top1_agreement,
        float(summary["aggregate"]["top1_agreement"]),
        args.absolute_tolerance,
        "aggregate top1 agreement",
    )

    independent_path = destination / "independent_tokenwise.safetensors"
    independent_temporary = destination / ".independent_tokenwise.incomplete"
    save_file(
        {
            "kld_nats": values,
            "kld_bits": bits,
            "reference_top1": reference_top1,
            "candidate_top1": candidate_top1,
        },
        str(independent_temporary),
    )
    os.replace(independent_temporary, independent_path)
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "direction": "KL(reference||candidate)",
        "reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "report_file_sha256": sha256_file(summary_path),
        "report_sha256": summary["summary_sha256"],
        "producer_tokenwise_file_sha256": summary["tokenwise_file_sha256"],
        "independent_tokenwise_file": independent_path.name,
        "independent_tokenwise_file_sha256": sha256_file(independent_path),
        "max_tokenwise_kld_difference": max_kld_difference,
        "max_tokenwise_bits_difference": max_bits_difference,
        "absolute_tolerance": args.absolute_tolerance,
        "aggregate": {
            "positions": int(values.size),
            "kld_nats": aggregate_nats,
            "kld_bits": aggregate_bits,
            "top1_agreement": top1_agreement,
        },
        "windows": windows,
    }
    verification["verification_sha256"] = canonical_json_sha256(verification)
    verification_path = destination / "verification.json"
    verification_temporary = destination / ".verification.json.incomplete"
    verification_temporary.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(verification_temporary, verification_path)
    print(
        json.dumps(
            {
                "event": "captured_prefill_kld_replay_complete",
                "verification": str(verification_path),
                "aggregate": verification["aggregate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
