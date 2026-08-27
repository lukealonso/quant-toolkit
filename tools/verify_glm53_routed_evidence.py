#!/usr/bin/env python3
"""Verify every file in a sealed GLM-5.3 routed-evidence manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from kld_common import canonical_json_sha256, sha256_file

SCHEMA = "quant-toolkit.glm53-routed-evidence.v1"
RECEIPT_SCHEMA = "quant-toolkit.glm53-routed-evidence-verification.v1"


def _verify_seal(manifest: dict) -> None:
    expected = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected evidence schema: {manifest.get('schema')}")
    if expected != canonical_json_sha256(body):
        raise RuntimeError("routed-evidence manifest seal mismatch")


def _container_path(root: Path, value: str, expected_root: str) -> Path:
    source = PurePosixPath(value)
    parts = source.parts[1:] if source.is_absolute() else source.parts
    if not parts or parts[0] != expected_root or ".." in parts:
        raise RuntimeError(f"unexpected /{expected_root} path in manifest: {value}")
    return root.joinpath(*parts[1:])


def _verify_file(item: tuple[Path, int, str]) -> tuple[int, str]:
    path, expected_bytes, expected_sha256 = item
    if not path.is_file():
        raise RuntimeError(f"manifest file is missing: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"file size mismatch: {path}: expected={expected_bytes} observed={observed_bytes}"
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"file hash mismatch: {path}: expected={expected_sha256} "
            f"observed={observed_sha256}"
        )
    return observed_bytes, observed_sha256


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--panel-dir", type=Path)
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    evidence_dir = args.evidence_dir.resolve()
    run_root = (args.run_root or evidence_dir.parent).resolve()
    manifest_path = evidence_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_seal(manifest)
    windows = manifest.get("windows", [])
    if not windows:
        raise RuntimeError("evidence manifest contains no windows")
    if args.expected_windows is not None and len(windows) != args.expected_windows:
        raise RuntimeError(
            f"window count mismatch: expected={args.expected_windows} observed={len(windows)}"
        )

    runtime_path = _container_path(
        run_root, manifest["runtime_manifest"], expected_root="run"
    )
    if sha256_file(runtime_path) != manifest["runtime_manifest_file_sha256"]:
        raise RuntimeError(f"runtime manifest hash mismatch: {runtime_path}")

    routed_layers = [int(layer_id) for layer_id in manifest["routed_layers"]]
    if len(set(routed_layers)) != len(routed_layers):
        raise RuntimeError("manifest routed_layers contains duplicates")
    window_ids = [window["window_id"] for window in windows]
    if len(set(window_ids)) != len(window_ids):
        raise RuntimeError("manifest window IDs contain duplicates")
    if window_ids != sorted(window_ids):
        raise RuntimeError("manifest windows are not sorted by window_id")

    windows_dir = evidence_dir / "windows"
    observed_window_ids = {
        path.name for path in windows_dir.iterdir() if path.is_dir()
    }
    if observed_window_ids != set(window_ids):
        raise RuntimeError(
            "window directory set mismatch: "
            f"missing={sorted(set(window_ids) - observed_window_ids)} "
            f"unexpected={sorted(observed_window_ids - set(window_ids))}"
        )

    files: list[tuple[Path, int, str]] = []
    panel_files: list[tuple[Path, int, str]] = []
    aggregate_bytes = 0
    for window in windows:
        window_dir = windows_dir / window["window_id"]
        layers = window["layers"]
        layer_ids = [int(layer["layer_id"]) for layer in layers]
        if layer_ids != routed_layers:
            raise RuntimeError(
                f"routed layer set/order mismatch: {window['window_id']}"
            )
        expected_files = [layer["file"] for layer in layers]
        if len(set(expected_files)) != len(expected_files):
            raise RuntimeError(f"duplicate layer file: {window['window_id']}")
        observed_files = {path.name for path in window_dir.iterdir() if path.is_file()}
        if observed_files != set(expected_files):
            raise RuntimeError(
                f"layer file set mismatch: {window['window_id']}: "
                f"missing={sorted(set(expected_files) - observed_files)} "
                f"unexpected={sorted(observed_files - set(expected_files))}"
            )
        window_bytes = sum(int(layer["bytes"]) for layer in layers)
        if window_bytes != int(window["bytes"]):
            raise RuntimeError(f"window byte total mismatch: {window['window_id']}")
        aggregate_bytes += window_bytes
        files.extend(
            (window_dir / layer["file"], int(layer["bytes"]), layer["sha256"])
            for layer in layers
        )
        if args.panel_dir is not None:
            token_path = _container_path(
                args.panel_dir.resolve(), window["token_file"], expected_root="panel"
            )
            panel_files.append((token_path, token_path.stat().st_size, window["token_file_sha256"]))

    if aggregate_bytes != int(manifest["total_bytes"]):
        raise RuntimeError(
            f"manifest byte total mismatch: expected={manifest['total_bytes']} "
            f"observed={aggregate_bytes}"
        )

    all_files = files + panel_files
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        verified = []
        for index, result in enumerate(executor.map(_verify_file, all_files), start=1):
            verified.append(result)
            if index % 1000 == 0 or index == len(all_files):
                print(f"verified {index}/{len(all_files)} files", file=sys.stderr, flush=True)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "verified_utc": datetime.now(UTC).isoformat(),
        "evidence_dir": str(evidence_dir),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_manifest_file_sha256": manifest["runtime_manifest_file_sha256"],
        "window_count": len(windows),
        "routed_layer_count": len(routed_layers),
        "evidence_file_count": len(files),
        "panel_file_count": len(panel_files),
        "verified_evidence_bytes": sum(item[0] for item in verified[: len(files)]),
        "workers": args.workers,
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
