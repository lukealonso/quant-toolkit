#!/usr/bin/env python3
"""Independently replay a dense-prefill KLD report from saved logits.

This verifier deliberately uses NumPy and an explicit log-sum-exp formula,
not the PyTorch producer implementation in kld_common.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


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


def independent_tokenwise_kld(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    chunk_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KL(reference || candidate) using an explicit NumPy formula."""
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference and candidate logits must both be rank 2")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"logit shape mismatch: reference={reference.shape} "
            f"candidate={candidate.shape}"
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

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


def _verify_seal(document: dict, field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != canonical_json_sha256(body):
        raise ValueError(f"{label} seal mismatch")


def _stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_9": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def _compare_scalar(observed: float, expected: float, tolerance: float, label: str):
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{label} differs: replay={observed:.17g} "
            f"producer={expected:.17g} tolerance={tolerance:.3g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--candidate-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)

    reference_dir = Path(args.reference_logits).resolve()
    candidate_dir = Path(args.candidate_logits).resolve()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")

    reference_manifest_path = reference_dir / "manifest.json"
    candidate_summary_path = candidate_dir / "summary.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    candidate_summary = json.loads(candidate_summary_path.read_text())
    if reference_manifest.get("schema") != "quant-toolkit.prefill-logits.v1":
        raise ValueError("unsupported reference manifest schema")
    if candidate_summary.get("schema") != "quant-toolkit.prefill-kld.v1":
        raise ValueError("unsupported candidate summary schema")
    _verify_seal(reference_manifest, "manifest_sha256", "reference manifest")
    _verify_seal(candidate_summary, "summary_sha256", "candidate summary")
    embedded_reference = candidate_summary.get("reference_manifest", {})
    if (
        embedded_reference.get("manifest_sha256")
        != reference_manifest["manifest_sha256"]
    ):
        raise ValueError("candidate summary references a different teacher manifest")
    if reference_manifest.get("storage_dtype") != "float32":
        raise ValueError("independent NumPy replay requires float32 reference logits")
    if candidate_summary.get("candidate_storage_dtype") != "float32":
        raise ValueError("independent NumPy replay requires float32 candidate logits")

    reference_records = {
        int(record["index"]): record for record in reference_manifest["windows"]
    }
    all_values = []
    all_reference_top1 = []
    all_candidate_top1 = []
    window_verifications = []
    for candidate_record in candidate_summary["windows"]:
        index = int(candidate_record["index"])
        if index not in reference_records:
            raise ValueError(f"candidate window {index} has no reference window")
        reference_record = reference_records[index]
        reference_path = reference_dir / reference_record["file"]
        candidate_path = candidate_dir / candidate_record["candidate_file"]
        if sha256_file(reference_path) != reference_record.get("sha256"):
            raise ValueError(f"reference logit hash mismatch: {reference_path}")
        if sha256_file(candidate_path) != candidate_record.get("candidate_file_sha256"):
            raise ValueError(f"candidate logit hash mismatch: {candidate_path}")
        reference_tensors = load_file(str(reference_path))
        candidate_tensors = load_file(str(candidate_path))
        if not np.array_equal(
            reference_tensors["input_ids"], candidate_tensors["input_ids"]
        ):
            raise ValueError(f"input token mismatch in window {index}")
        values, reference_top1, candidate_top1 = independent_tokenwise_kld(
            reference_tensors["logits"],
            candidate_tensors["logits"],
            chunk_rows=args.chunk_rows,
        )
        all_values.append(values)
        all_reference_top1.append(reference_top1)
        all_candidate_top1.append(candidate_top1)
        window_verifications.append(
            {
                "index": index,
                "positions": int(values.size),
                "reference_file_sha256": reference_record["sha256"],
                "candidate_file_sha256": candidate_record["candidate_file_sha256"],
                "kld_nats": _stats(values),
                "top1_agreement": float(np.mean(reference_top1 == candidate_top1)),
            }
        )
        for key, observed in window_verifications[-1]["kld_nats"].items():
            _compare_scalar(
                observed,
                float(candidate_record["kld_nats"][key]),
                args.absolute_tolerance,
                f"window {index} kld_nats.{key}",
            )
        _compare_scalar(
            window_verifications[-1]["top1_agreement"],
            float(candidate_record["top1_agreement"]),
            args.absolute_tolerance,
            f"window {index} top1 agreement",
        )

    if len(window_verifications) != len(reference_records):
        raise ValueError("candidate/reference window count mismatch")

    values = np.concatenate(all_values)
    reference_top1 = np.concatenate(all_reference_top1)
    candidate_top1 = np.concatenate(all_candidate_top1)
    tokenwise_path = candidate_dir / candidate_summary["tokenwise_file"]
    if sha256_file(tokenwise_path) != candidate_summary.get("tokenwise_file_sha256"):
        raise ValueError("producer tokenwise artifact hash mismatch")
    producer = load_file(str(tokenwise_path))
    max_kld_difference = float(np.max(np.abs(values - producer["kld_nats"])))
    if max_kld_difference > args.absolute_tolerance:
        raise RuntimeError(
            f"independent tokenwise KLD differs from producer by "
            f"{max_kld_difference:.17g}"
        )
    if not np.array_equal(reference_top1, producer["reference_top1"]):
        raise RuntimeError("independent reference top-1 IDs differ from producer")
    if not np.array_equal(candidate_top1, producer["candidate_top1"]):
        raise RuntimeError("independent candidate top-1 IDs differ from producer")
    bits = values / math.log(2.0)
    max_bits_difference = float(np.max(np.abs(bits - producer["kld_bits"])))
    if max_bits_difference > args.absolute_tolerance:
        raise RuntimeError("independent tokenwise KLD bits differ from producer")

    aggregate_nats = _stats(values)
    aggregate_bits = _stats(bits)
    top1_agreement = float(np.mean(reference_top1 == candidate_top1))
    producer_aggregate = candidate_summary["aggregate"]
    for key, observed in aggregate_nats.items():
        _compare_scalar(
            observed,
            float(producer_aggregate["kld_nats"][key]),
            args.absolute_tolerance,
            f"aggregate kld_nats.{key}",
        )
    for key, observed in aggregate_bits.items():
        _compare_scalar(
            observed,
            float(producer_aggregate["kld_bits"][key]),
            args.absolute_tolerance,
            f"aggregate kld_bits.{key}",
        )
    _compare_scalar(
        top1_agreement,
        float(producer_aggregate["top1_agreement"]),
        args.absolute_tolerance,
        "aggregate top1 agreement",
    )

    independent_path = destination / "independent_tokenwise.safetensors"
    save_file(
        {
            "kld_nats": values,
            "kld_bits": bits,
            "reference_top1": reference_top1,
            "candidate_top1": candidate_top1,
        },
        str(independent_path),
    )
    verification = {
        "schema": "quant-toolkit.prefill-kld-verification.v1",
        "direction": "KL(reference||candidate)",
        "reference_manifest_file_sha256": sha256_file(reference_manifest_path),
        "reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "candidate_summary_file_sha256": sha256_file(candidate_summary_path),
        "candidate_summary_sha256": candidate_summary["summary_sha256"],
        "producer_tokenwise_file_sha256": candidate_summary["tokenwise_file_sha256"],
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
        "windows": window_verifications,
    }
    verification["verification_sha256"] = canonical_json_sha256(verification)
    verification_path = destination / "verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"event": "independent_kld_replay_complete", **verification["aggregate"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
