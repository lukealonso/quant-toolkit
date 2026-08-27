#!/usr/bin/env python3
"""Seal route weights and BF16 conditional down inputs for GLM reservoirs.

The input reservoirs intentionally contain only routed hidden rows.  This pass
replays their deterministic selection against the sealed evidence, verifies
every selected row byte-for-byte, records its route weight and source
coordinates, then evaluates the immutable BF16 gate/up projections to retain
the conditional SwiGLU state consumed by each down projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from kld_common import canonical_json_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import load_file, save_file


SCHEMA = "quant-toolkit.glm53-nvfp4-conditional-reservoirs.v1"
ROUTE_MAP_SCHEMA = "quant-toolkit.glm53-nvfp4-route-map.v1"
CONDITIONAL_SCHEMA = "quant-toolkit.glm53-nvfp4-conditional-state.v1"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_safetensors_atomic(
    tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    save_file(tensors, temporary, metadata=metadata)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _layer_key(layer: int, expert: int, projection: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.weight"
    )


def _evidence_files(
    evidence_root: Path, roles: list[str], layer: int
) -> tuple[list[Path], list[dict]]:
    records: list[tuple[str, Path]] = []
    manifests = []
    for role in roles:
        manifest_path = evidence_root / role / "manifest.json"
        manifest = _load_json(manifest_path)
        body = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        if manifest.get("manifest_sha256") != canonical_json_sha256(body):
            raise RuntimeError(
                f"routed-evidence manifest seal mismatch: {manifest_path}"
            )
        manifests.append(
            {
                "role": role,
                "path": str(manifest_path.resolve()),
                "file_sha256": sha256_file(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "windows": len(manifest.get("windows", [])),
            }
        )
        for window in manifest.get("windows", []):
            window_id = window["window_id"]
            layer_record = next(
                item for item in window["layers"] if int(item["layer_id"]) == layer
            )
            layer_path = (
                evidence_root / role / "windows" / window_id / layer_record["file"]
            )
            if not layer_path.is_file():
                raise FileNotFoundError(layer_path)
            order = hashlib.sha256(f"{role}:{window_id}".encode()).hexdigest()
            records.append((order, layer_path))
    return [path for _order, path in sorted(records)], manifests


def collect_route_map(
    *,
    evidence_paths: list[Path],
    reservoir_path: Path,
    num_experts: int,
    samples_per_expert: int,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Reconstruct and verify the exact source of each reservoir sample."""
    reservoir = load_file(reservoir_path, device="cpu")
    samples = reservoir["samples"]
    sampled_counts = reservoir["sampled_counts"].to(torch.int64)
    expected_shape = (num_experts, samples_per_expert, hidden_size)
    if tuple(samples.shape) != expected_shape:
        raise RuntimeError(
            "invalid reservoir sample shape: "
            f"{tuple(samples.shape)} != {expected_shape}"
        )
    if not torch.equal(
        sampled_counts, torch.full_like(sampled_counts, samples_per_expert)
    ):
        raise RuntimeError("reservoir sampled counts are incomplete")

    route_weights = torch.empty(num_experts, samples_per_expert, dtype=torch.float32)
    source_file_indices = torch.empty(
        num_experts, samples_per_expert, dtype=torch.int32
    )
    source_row_indices = torch.empty(
        num_experts, samples_per_expert, dtype=torch.int32
    )
    collected = torch.zeros(num_experts, dtype=torch.int64)

    for file_index, path in enumerate(evidence_paths):
        with safe_open(path, framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor("hidden_states")
            topk_ids = handle.get_tensor("topk_ids").to(torch.int64)
            topk_weights = handle.get_tensor("topk_weights").to(torch.float32)
        if tuple(hidden.shape[1:]) != (hidden_size,):
            raise RuntimeError(
                f"unexpected hidden shape in {path}: {tuple(hidden.shape)}"
            )
        need = (collected < samples_per_expert).nonzero().flatten().tolist()
        for expert in need:
            rows = (topk_ids == expert).any(dim=1).nonzero().flatten()
            missing = samples_per_expert - int(collected[expert])
            take = min(missing, int(rows.numel()))
            if not take:
                continue
            chosen_rows = rows[:take]
            start = int(collected[expert])
            stop = start + take
            if not torch.equal(samples[expert, start:stop], hidden[chosen_rows]):
                raise RuntimeError(
                    f"reservoir replay mismatch: expert={expert} file={path}"
                )
            matches = topk_ids[chosen_rows] == expert
            selected_weights = (topk_weights[chosen_rows] * matches).sum(dim=-1)
            if not torch.isfinite(selected_weights).all() or bool(
                (selected_weights <= 0).any()
            ):
                raise RuntimeError(
                    f"invalid selected route weight: expert={expert} file={path}"
                )
            route_weights[expert, start:stop].copy_(selected_weights)
            source_file_indices[expert, start:stop].fill_(file_index)
            source_row_indices[expert, start:stop].copy_(chosen_rows.to(torch.int32))
            collected[expert] += take
        if bool((collected == samples_per_expert).all()):
            break

    if not torch.equal(collected, sampled_counts):
        missing = (collected != sampled_counts).nonzero().flatten().tolist()
        raise RuntimeError(f"route-map replay incomplete for experts: {missing[:16]}")
    return {
        "route_weights": route_weights,
        "source_file_indices": source_file_indices,
        "source_row_indices": source_row_indices,
    }


def _validate_existing(
    path: Path,
    *,
    schema: str,
    layer: int,
    reservoir_sha256: str,
    route_map_sha256: str | None = None,
) -> dict | None:
    if not path.is_file():
        return None
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = {
        "schema": schema,
        "layer": str(layer),
        "reservoir_sha256": reservoir_sha256,
    }
    if route_map_sha256 is not None:
        expected["route_map_sha256"] = route_map_sha256
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing artifact provenance mismatch: {path}")
    return {
        "layer": layer,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "resumed": True,
    }


def _build_conditional_state(
    *,
    layer: int,
    device: str,
    source_model: Path,
    weight_map: dict[str, str],
    reservoir_path: Path,
    route_map_path: Path,
    output_path: Path,
    num_experts: int,
    samples_per_expert: int,
    intermediate_size: int,
    swiglu_limit: float,
) -> dict:
    reservoir_sha256 = sha256_file(reservoir_path)
    route_map_sha256 = sha256_file(route_map_path)
    resumed = _validate_existing(
        output_path,
        schema=CONDITIONAL_SCHEMA,
        layer=layer,
        reservoir_sha256=reservoir_sha256,
        route_map_sha256=route_map_sha256,
    )
    if resumed is not None:
        print(f"layer={layer} conditional_resume device={device}", flush=True)
        return resumed

    torch.cuda.set_device(device)
    samples = load_file(reservoir_path, device="cpu")["samples"]
    route_map = load_file(route_map_path, device="cpu")
    conditional = torch.empty(
        num_experts,
        samples_per_expert,
        intermediate_size,
        dtype=torch.bfloat16,
    )
    needed_shards = {
        weight_map[_layer_key(layer, expert, projection)]
        for expert in range(num_experts)
        for projection in ("gate_proj", "up_proj")
    }
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(source_model / shard, framework="pt", device="cpu")
            )
            for shard in needed_shards
        }
        with torch.inference_mode():
            for expert in range(num_experts):
                x = samples[expert].to(device)
                gate_key = _layer_key(layer, expert, "gate_proj")
                up_key = _layer_key(layer, expert, "up_proj")
                gate_weight = (
                    handles[weight_map[gate_key]].get_tensor(gate_key).to(device)
                )
                up_weight = handles[weight_map[up_key]].get_tensor(up_key).to(device)
                gate = F.linear(x, gate_weight).clamp(max=swiglu_limit)
                up = F.linear(x, up_weight).clamp(
                    min=-swiglu_limit, max=swiglu_limit
                )
                value = F.silu(gate) * up
                if value.dtype != torch.bfloat16:
                    value = value.to(torch.bfloat16)
                conditional[expert].copy_(value.cpu())
                del x, gate_weight, up_weight, gate, up, value

    tensors = {
        "conditional_down_inputs": conditional,
        "route_weights": route_map["route_weights"],
        "source_file_indices": route_map["source_file_indices"],
        "source_row_indices": route_map["source_row_indices"],
    }
    _save_safetensors_atomic(
        tensors,
        output_path,
        {
            "schema": CONDITIONAL_SCHEMA,
            "layer": str(layer),
            "reservoir_sha256": reservoir_sha256,
            "route_map_sha256": route_map_sha256,
            "activation": "silu(clamp_max(gate,10))*clamp(up,-10,10)",
            "projection_math": "torch.bfloat16 source weights and inputs on CUDA",
        },
    )
    print(f"layer={layer} conditional_complete device={device}", flush=True)
    return {
        "layer": layer,
        "path": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "resumed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--reservoir-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--roles", nargs="+", default=["fit", "conditional-fit"])
    parser.add_argument("--samples-per-expert", type=int, default=256)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument(
        "--phase",
        choices=("all", "route-maps", "conditional"),
        default="all",
        help="Run both phases, only route-map replay, or only CUDA replay.",
    )
    args = parser.parse_args(argv)

    source_model = args.source_model.resolve()
    config_path = source_model / "config.json"
    index_path = source_model / "model.safetensors.index.json"
    config = _load_json(config_path)["text_config"]
    weight_map = _load_json(index_path)["weight_map"]
    layers = list(
        range(int(config["first_k_dense_replace"]), int(config["num_hidden_layers"]))
    )
    num_experts = int(config["n_routed_experts"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    swiglu_limit = float(config["swiglu_limit"])
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if args.phase != "route-maps" and not devices:
        raise ValueError("--devices must name at least one CUDA device")
    if args.samples_per_expert <= 0 or not math.isfinite(
        float(args.samples_per_expert)
    ):
        raise ValueError("--samples-per-expert must be positive")

    route_dir = args.output_dir / "route-maps"
    conditional_dir = args.output_dir / "conditional-states"
    route_records = []
    evidence_manifests = None
    for layer in layers:
        reservoir_path = args.reservoir_dir / f"layer-{layer:04d}.safetensors"
        if not reservoir_path.is_file():
            raise FileNotFoundError(reservoir_path)
        reservoir_sha256 = sha256_file(reservoir_path)
        route_path = route_dir / f"layer-{layer:04d}.safetensors"
        resumed = _validate_existing(
            route_path,
            schema=ROUTE_MAP_SCHEMA,
            layer=layer,
            reservoir_sha256=reservoir_sha256,
        )
        paths, manifests = _evidence_files(args.evidence_root, args.roles, layer)
        if evidence_manifests is None:
            evidence_manifests = manifests
        if resumed is not None:
            print(f"layer={layer} route_map_resume", flush=True)
            route_records.append(resumed)
            continue
        tensors = collect_route_map(
            evidence_paths=paths,
            reservoir_path=reservoir_path,
            num_experts=num_experts,
            samples_per_expert=args.samples_per_expert,
            hidden_size=hidden_size,
        )
        _save_safetensors_atomic(
            tensors,
            route_path,
            {
                "schema": ROUTE_MAP_SCHEMA,
                "layer": str(layer),
                "reservoir_sha256": reservoir_sha256,
                "selection": "sha256(role:window_id), then first routed rows",
                "route_weight": "sum(topk_weights where topk_ids equals expert)",
            },
        )
        route_records.append(
            {
                "layer": layer,
                "path": str(route_path.resolve()),
                "bytes": route_path.stat().st_size,
                "sha256": sha256_file(route_path),
                "resumed": False,
            }
        )
        print(f"layer={layer} route_map_complete", flush=True)

    def worker(device_index: int) -> list[dict]:
        records = []
        for layer in layers[device_index :: len(devices)]:
            records.append(
                _build_conditional_state(
                    layer=layer,
                    device=devices[device_index],
                    source_model=source_model,
                    weight_map=weight_map,
                    reservoir_path=(
                        args.reservoir_dir / f"layer-{layer:04d}.safetensors"
                    ),
                    route_map_path=route_dir / f"layer-{layer:04d}.safetensors",
                    output_path=conditional_dir / f"layer-{layer:04d}.safetensors",
                    num_experts=num_experts,
                    samples_per_expert=args.samples_per_expert,
                    intermediate_size=intermediate_size,
                    swiglu_limit=swiglu_limit,
                )
            )
        return records

    conditional_records = []
    if args.phase != "route-maps":
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for records in executor.map(worker, range(len(devices))):
                conditional_records.extend(records)

    receipt = {
        "schema": SCHEMA,
        "source": {
            "directory": str(source_model),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
        },
        "evidence_manifests": evidence_manifests,
        "method": {
            "phase": args.phase,
            "roles": args.roles,
            "samples_per_expert": args.samples_per_expert,
            "reservoir_selection_verified_byte_exact": True,
            "route_weight": "sum(topk_weights where topk_ids equals expert)",
            "conditional_state": (
                "BF16 source gate/up CUDA replay through GLM clamped SwiGLU"
            ),
        },
        "topology": {
            "layers": layers,
            "experts_per_layer": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "route_maps": sorted(route_records, key=lambda item: item["layer"]),
        "conditional_states": sorted(
            conditional_records, key=lambda item: item["layer"]
        ),
        "tools": {
            "augment_sha256": sha256_file(Path(__file__).resolve()),
        },
        "result": "PASS" if args.phase != "route-maps" else "PASS_ROUTE_MAPS",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_json_atomic(args.receipt, receipt)
    print(
        json.dumps(
            {
                "event": "glm53_nvfp4_conditional_reservoirs_complete",
                "receipt": str(args.receipt.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
