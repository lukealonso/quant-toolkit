#!/usr/bin/env python3
"""Verify a local or SSH-hosted snapshot against a pinned HF shard manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import subprocess
import time
from pathlib import Path

MANIFEST_SCHEMA = "quant-toolkit.hf-lfs-shard-manifest.v1"
REPORT_SCHEMA = "quant-toolkit.hf-lfs-shard-verification.v1"


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _ssh_args(host: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "Compression=no",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=300",
        "-o",
        "ControlPath=/tmp/quant-toolkit-hf-%C",
        host,
    ]


def remote_size(host: str, path: str) -> int:
    command = f"wc -c < {shlex.quote(path)}"
    result = subprocess.run(
        [*_ssh_args(host), command],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def remote_sha256(host: str, path: str, chunk_bytes: int) -> tuple[str, int]:
    command = f"dd if={shlex.quote(path)} bs={chunk_bytes} status=none"
    process = subprocess.Popen(
        [*_ssh_args(host), command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to open SSH hash stream")
    digest = hashlib.sha256()
    size = 0
    while chunk := process.stdout.read(chunk_bytes):
        digest.update(chunk)
        size += len(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"remote read failed for {path}: exit={returncode}: {stderr.strip()}"
        )
    return digest.hexdigest(), size


def _write_report(path: Path, report: dict) -> None:
    report = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_json_sha256(report)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_resume(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported verification report schema")
    if not expected or canonical_json_sha256(body) != expected:
        raise ValueError("verification report seal mismatch")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--mode", choices=("size", "sha256"), default="sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args(argv)

    if args.chunk_mib < 1:
        parser.error("chunk MiB must be positive")
    chunk_bytes = args.chunk_mib * 1024 * 1024
    manifest_path = Path(args.manifest).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported HF shard manifest schema")
    if len(manifest.get("shards", [])) != int(manifest.get("shard_count", -1)):
        raise ValueError("manifest shard count mismatch")
    if sum(int(item["bytes"]) for item in manifest["shards"]) != int(
        manifest.get("total_shard_bytes", -1)
    ):
        raise ValueError("manifest total byte count mismatch")

    target = {
        "kind": "ssh" if args.remote_host else "local",
        "host": args.remote_host,
        "directory": args.directory,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if output_path.exists():
        report = _load_resume(output_path)
        if report.get("manifest_file_sha256") != manifest_file_sha256:
            raise ValueError("resume report uses a different shard manifest")
        if report.get("target") != target or report.get("mode") != args.mode:
            raise ValueError("resume report target or mode mismatch")
    else:
        report = {
            "schema": REPORT_SCHEMA,
            "manifest": {
                "repo": manifest["repo"],
                "revision": manifest["revision"],
                "shard_count": manifest["shard_count"],
                "total_shard_bytes": manifest["total_shard_bytes"],
            },
            "manifest_file_sha256": manifest_file_sha256,
            "target": target,
            "mode": args.mode,
            "chunk_bytes": chunk_bytes,
            "verified": [],
            "complete": False,
        }

    verified = {item["path"]: item for item in report["verified"]}
    started = time.time()
    for expected in manifest["shards"]:
        name = str(expected["path"])
        if name in verified:
            item = verified[name]
            if (
                int(item["bytes"]) != int(expected["bytes"])
                or item.get("expected_sha256") != expected["sha256"]
                or (
                    args.mode == "sha256"
                    and item.get("observed_sha256") != expected["sha256"]
                )
            ):
                raise ValueError(f"invalid resumed verification row: {name}")
            continue

        item_started = time.time()
        if args.remote_host:
            path = posixpath.join(args.directory, name)
            if args.mode == "sha256":
                observed_sha256, observed_bytes = remote_sha256(
                    args.remote_host, path, chunk_bytes
                )
            else:
                observed_bytes = remote_size(args.remote_host, path)
                observed_sha256 = None
        else:
            path = Path(args.directory) / name
            if args.mode == "sha256":
                observed_sha256, observed_bytes = sha256_file(path, chunk_bytes)
            else:
                observed_bytes = path.stat().st_size
                observed_sha256 = None

        if observed_bytes != int(expected["bytes"]):
            raise RuntimeError(
                f"size mismatch for {name}: observed={observed_bytes} "
                f"expected={expected['bytes']}"
            )
        if args.mode == "sha256" and observed_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: observed={observed_sha256} "
                f"expected={expected['sha256']}"
            )
        row = {
            "path": name,
            "bytes": observed_bytes,
            "expected_sha256": expected["sha256"],
            "observed_sha256": observed_sha256,
            "elapsed_sec": time.time() - item_started,
        }
        report["verified"].append(row)
        report["verified_count"] = len(report["verified"])
        report["verified_bytes"] = sum(
            int(item["bytes"]) for item in report["verified"]
        )
        report["last_update_elapsed_sec"] = time.time() - started
        _write_report(output_path, report)
        print(
            json.dumps(
                {
                    "event": "hf_shard_verified",
                    "target": target,
                    "mode": args.mode,
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report["verified_count"] = len(report["verified"])
    report["verified_bytes"] = sum(int(item["bytes"]) for item in report["verified"])
    report["complete"] = (
        report["verified_count"] == manifest["shard_count"]
        and report["verified_bytes"] == manifest["total_shard_bytes"]
    )
    report["last_update_elapsed_sec"] = time.time() - started
    _write_report(output_path, report)
    print(
        json.dumps(
            {
                "event": "hf_shard_verification_complete",
                "output": str(output_path),
                "target": target,
                "mode": args.mode,
                "verified_count": report["verified_count"],
                "verified_bytes": report["verified_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
