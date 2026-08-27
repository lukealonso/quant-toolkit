#!/usr/bin/env python3
"""Quarantine a corrupt routed-evidence suffix and reseal its manifest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import kld_common
from kld_common import canonical_json_sha256, sha256_file

SCHEMA = "quant-toolkit.glm53-routed-evidence.v1"
RECEIPT_SCHEMA = "quant-toolkit.glm53-routed-evidence-recovery.v1"


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _verify_manifest(manifest: dict) -> None:
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected evidence schema: {manifest.get('schema')}")
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected != canonical_json_sha256(body):
        raise RuntimeError("routed-evidence manifest seal mismatch")


def _window_hash_mismatches(evidence_dir: Path, window: dict) -> list[int]:
    mismatches = []
    for layer in window["layers"]:
        path = evidence_dir / "windows" / window["window_id"] / layer["file"]
        if not path.is_file() or sha256_file(path) != layer["sha256"]:
            mismatches.append(int(layer["layer_id"]))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--first-invalid", required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    evidence_dir = args.evidence_dir.resolve()
    manifest_path = evidence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(manifest)
    original_manifest_sha256 = manifest["manifest_sha256"]
    window_ids = [window["window_id"] for window in manifest["windows"]]
    try:
        invalid_index = window_ids.index(args.first_invalid)
    except ValueError as exc:
        raise RuntimeError(
            f"first invalid window is not in the manifest: {args.first_invalid}"
        ) from exc
    if invalid_index == 0:
        raise RuntimeError("refusing to truncate every evidence window")

    retained = manifest["windows"][:invalid_index]
    removed = manifest["windows"][invalid_index:]
    last_retained_mismatches = _window_hash_mismatches(evidence_dir, retained[-1])
    if last_retained_mismatches:
        raise RuntimeError(
            f"last retained window is corrupt: {retained[-1]['window_id']} "
            f"layers={last_retained_mismatches}"
        )
    first_invalid_mismatches = _window_hash_mismatches(evidence_dir, removed[0])
    if not first_invalid_mismatches:
        raise RuntimeError(
            f"first invalid window still matches its hashes: {args.first_invalid}"
        )

    quarantine_dir = args.quarantine_dir.resolve()
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for window in removed:
        source = evidence_dir / "windows" / window["window_id"]
        destination = quarantine_dir / window["window_id"]
        if not source.is_dir():
            raise RuntimeError(f"evidence window is missing: {source}")
        if destination.exists():
            raise RuntimeError(f"quarantine destination already exists: {destination}")
    for window in removed:
        os.replace(
            evidence_dir / "windows" / window["window_id"],
            quarantine_dir / window["window_id"],
        )

    recovery_event = {
        "recovered_utc": datetime.now(UTC).isoformat(),
        "original_manifest_sha256": original_manifest_sha256,
        "first_invalid_window": args.first_invalid,
        "first_invalid_mismatched_layers": first_invalid_mismatches,
        "last_retained_window": retained[-1]["window_id"],
        "removed_windows": [window["window_id"] for window in removed],
        "quarantine_dir": str(quarantine_dir),
        "reason": args.reason,
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
        "kld_common_file_sha256": sha256_file(Path(kld_common.__file__).resolve()),
    }
    manifest["windows"] = retained
    manifest["total_bytes"] = sum(window["bytes"] for window in retained)
    manifest.setdefault("recovery_events", []).append(recovery_event)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json_atomic(manifest_path, manifest)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "evidence_dir": str(evidence_dir),
        "manifest_file_sha256": sha256_file(manifest_path),
        "final_manifest_sha256": manifest["manifest_sha256"],
        **recovery_event,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
