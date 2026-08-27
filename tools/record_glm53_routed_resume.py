#!/usr/bin/env python3
"""Seal the exact boundary and runtime facts for a routed-capture resume."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from kld_common import canonical_json_sha256, sha256_file

EVIDENCE_SCHEMA = "quant-toolkit.glm53-routed-evidence.v1"
VERIFICATION_SCHEMA = "quant-toolkit.glm53-routed-evidence-verification.v1"
RECEIPT_SCHEMA = "quant-toolkit.glm53-routed-capture-resume.v1"


def _verify_seal(payload: dict, key: str, label: str) -> None:
    expected = payload.get(key)
    body = {name: value for name, value in payload.items() if name != key}
    if expected != canonical_json_sha256(body):
        raise RuntimeError(f"{label} seal mismatch")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _window_number(window_id: str) -> tuple[str, int]:
    prefix, separator, number = window_id.rpartition("-")
    if not separator or not number.isdigit():
        raise RuntimeError(f"unexpected window ID: {window_id}")
    return prefix, int(number)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--previous-tool", type=Path, required=True)
    parser.add_argument("--resume-tool", type=Path, required=True)
    parser.add_argument("--migration-verification", type=Path, required=True)
    parser.add_argument("--first-window", required=True)
    parser.add_argument("--last-window", required=True)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--storage-backend", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--quant-toolkit-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    evidence_dir = args.evidence_dir.resolve()
    manifest_path = evidence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError(f"unexpected evidence schema: {manifest.get('schema')}")
    _verify_seal(manifest, "manifest_sha256", "evidence manifest")
    if not manifest.get("windows"):
        raise RuntimeError("evidence manifest contains no completed windows")

    previous_window = manifest["windows"][-1]["window_id"]
    previous_prefix, previous_number = _window_number(previous_window)
    first_prefix, first_number = _window_number(args.first_window)
    last_prefix, last_number = _window_number(args.last_window)
    if previous_prefix != first_prefix or first_prefix != last_prefix:
        raise RuntimeError("resume window prefixes differ")
    if first_number != previous_number + 1 or last_number < first_number:
        raise RuntimeError(
            f"resume range is not contiguous after {previous_window}: "
            f"{args.first_window}..{args.last_window}"
        )

    runtime_path = args.runtime_manifest.resolve()
    runtime_sha256 = sha256_file(runtime_path)
    if runtime_sha256 != manifest["runtime_manifest_file_sha256"]:
        raise RuntimeError("runtime manifest no longer matches evidence binding")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    previous_tool_sha256 = sha256_file(args.previous_tool.resolve())
    if previous_tool_sha256 != runtime["tooling"]["capture_tool_sha256"]:
        raise RuntimeError("previous capture tool no longer matches runtime binding")

    verification_path = args.migration_verification.resolve()
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise RuntimeError("unexpected migration-verification schema")
    _verify_seal(verification, "receipt_sha256", "migration verification")
    if verification["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("migration verification binds a different evidence manifest")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "evidence_dir": str(evidence_dir),
        "pre_resume_manifest_file_sha256": sha256_file(manifest_path),
        "pre_resume_manifest_sha256": manifest["manifest_sha256"],
        "completed_window_count": len(manifest["windows"]),
        "completed_last_window": previous_window,
        "resume_first_window": args.first_window,
        "resume_last_window": args.last_window,
        "runtime_manifest_file_sha256": runtime_sha256,
        "previous_capture_tool_sha256": previous_tool_sha256,
        "resume_capture_tool_sha256": sha256_file(args.resume_tool.resolve()),
        "capture_semantics_changed": False,
        "capture_tool_change": "durability fsync only",
        "migration_verification_file_sha256": sha256_file(verification_path),
        "migration_verification_receipt_sha256": verification["receipt_sha256"],
        "storage_root": args.storage_root,
        "transport": args.transport,
        "storage_backend": args.storage_backend,
        "reason": args.reason,
        "quant_toolkit_commit": args.quant_toolkit_commit,
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
