#!/usr/bin/env python3
"""Make a sealed direct paired comparison of two GLM KLD reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

SCHEMA = "quant-toolkit.glm53-paired-candidate-comparison.v1"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(directory: Path) -> tuple[Path, dict]:
    path = directory / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = summary.get("summary_sha256")
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if expected != _canonical_sha256(body):
        raise RuntimeError(f"summary seal mismatch: {path}")
    tokenwise = directory / summary["tokenwise_file"]
    if _file_sha256(tokenwise) != summary["tokenwise_file_sha256"]:
        raise RuntimeError(f"tokenwise hash mismatch: {tokenwise}")
    return tokenwise, summary


def _load_tokenwise(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="numpy") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def compare_reports(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline_file, baseline_summary = _load_summary(baseline_dir)
    candidate_file, candidate_summary = _load_summary(candidate_dir)
    baseline = _load_tokenwise(baseline_file)
    candidate = _load_tokenwise(candidate_file)

    if (
        baseline_summary["reference_capture"]["manifest_sha256"]
        != candidate_summary["reference_capture"]["manifest_sha256"]
    ):
        raise RuntimeError("candidate reports use different reference captures")
    baseline_windows = baseline_summary["windows"]
    candidate_windows = candidate_summary["windows"]
    if [row["input_ids_sha256"] for row in baseline_windows] != [
        row["input_ids_sha256"] for row in candidate_windows
    ]:
        raise RuntimeError("candidate reports use different token windows")
    if baseline["kld_nats"].shape != candidate["kld_nats"].shape:
        raise RuntimeError("candidate tokenwise shapes differ")

    baseline_kld = baseline["kld_nats"].astype(np.float64)
    candidate_kld = candidate["kld_nats"].astype(np.float64)
    delta = candidate_kld - baseline_kld
    positions_per_window = [int(row["positions"]) for row in baseline_windows]
    if len(set(positions_per_window)) != 1:
        raise RuntimeError("nonuniform windows are not supported")
    window_width = positions_per_window[0]
    expected_positions = len(positions_per_window) * window_width
    if baseline_kld.size != expected_positions:
        raise RuntimeError("window positions do not match tokenwise tensor")
    baseline_window = baseline_kld.reshape(-1, window_width).mean(axis=1)
    candidate_window = candidate_kld.reshape(-1, window_width).mean(axis=1)
    window_delta = candidate_window - baseline_window

    reference_top1 = baseline["reference_top1"]
    if not np.array_equal(reference_top1, candidate["reference_top1"]):
        raise RuntimeError("candidate reports have different reference top-1")
    baseline_correct = baseline["candidate_top1"] == reference_top1
    candidate_correct = candidate["candidate_top1"] == reference_top1

    baseline_mean = float(np.mean(baseline_kld))
    candidate_mean = float(np.mean(candidate_kld))
    result = {
        "schema": SCHEMA,
        "direction": "candidate minus baseline; negative favors candidate",
        "baseline": {
            "report": str(baseline_dir.resolve()),
            "summary_file_sha256": _file_sha256(baseline_dir / "summary.json"),
            "summary_sha256": baseline_summary["summary_sha256"],
            "tokenwise_file_sha256": baseline_summary["tokenwise_file_sha256"],
            "run_label": baseline_summary["candidate_capture"]["run_label"],
            "mean_kld_nats": baseline_mean,
            "top1_agreement": float(np.mean(baseline_correct)),
        },
        "candidate": {
            "report": str(candidate_dir.resolve()),
            "summary_file_sha256": _file_sha256(candidate_dir / "summary.json"),
            "summary_sha256": candidate_summary["summary_sha256"],
            "tokenwise_file_sha256": candidate_summary["tokenwise_file_sha256"],
            "run_label": candidate_summary["candidate_capture"]["run_label"],
            "mean_kld_nats": candidate_mean,
            "top1_agreement": float(np.mean(candidate_correct)),
        },
        "paired": {
            "positions": int(delta.size),
            "mean_kld_delta_nats": float(np.mean(delta)),
            "relative_mean_kld_delta": float(
                (candidate_mean - baseline_mean) / baseline_mean
            ),
            "token_positions_candidate_better": int(np.sum(delta < 0)),
            "token_positions_equal": int(np.sum(delta == 0)),
            "token_positions_candidate_worse": int(np.sum(delta > 0)),
            "token_fraction_candidate_better": float(np.mean(delta < 0)),
            "window_count": len(positions_per_window),
            "windows_candidate_better": int(np.sum(window_delta < 0)),
            "windows_equal": int(np.sum(window_delta == 0)),
            "windows_candidate_worse": int(np.sum(window_delta > 0)),
            "window_mean_kld_delta_nats": _quantiles(window_delta),
            "baseline_only_top1_correct": int(
                np.sum(baseline_correct & ~candidate_correct)
            ),
            "candidate_only_top1_correct": int(
                np.sum(candidate_correct & ~baseline_correct)
            ),
        },
        "windows": [
            {
                "index": int(baseline_windows[index]["index"]),
                "input_ids_sha256": baseline_windows[index]["input_ids_sha256"],
                "baseline_mean_kld_nats": float(baseline_window[index]),
                "candidate_mean_kld_nats": float(candidate_window[index]),
                "mean_kld_delta_nats": float(window_delta[index]),
            }
            for index in range(len(positions_per_window))
        ],
    }
    result["comparison_sha256"] = _canonical_sha256(result)
    return result


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--quant-toolkit-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = compare_reports(args.baseline_report, args.candidate_report)
    result.pop("comparison_sha256")
    result["tooling"] = {
        "quant_toolkit_commit": args.quant_toolkit_commit,
        "tool_sha256": _file_sha256(Path(__file__).resolve()),
    }
    result["comparison_sha256"] = _canonical_sha256(result)
    _write_atomic(args.output, result)
    print(json.dumps({"event": "glm53_candidate_reports_compared", **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
