#!/usr/bin/env python3
"""Derive static global NVFP4 expert activation scales from routed BF16 evidence.

Gate/up scales use the exact routed-input maximum over every requested fit role.
Down-projection scales replay a deterministic per-expert reservoir through the
source BF16 gate/up projections and GLM-5.3's clamped SwiGLU.  The resulting
small scale shard can be tuned independently of the immutable packed weights.
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
from safetensors import safe_open
from safetensors.torch import load_file, save_file


NVFP4_GLOBAL_DENOMINATOR = 6.0 * 448.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _layer_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{suffix}"
    )


def _evidence_files(
    evidence_root: Path,
    roles: list[str],
    layer: int,
) -> tuple[list[Path], list[dict]]:
    records: list[tuple[str, Path]] = []
    manifests: list[dict] = []
    for role in roles:
        manifest_path = evidence_root / role / "manifest.json"
        manifest = _load_json(manifest_path)
        manifests.append(
            {
                "role": role,
                "path": str(manifest_path),
                "file_sha256": _sha256(manifest_path),
                "manifest_sha256": manifest.get("manifest_sha256"),
                "windows": len(manifest.get("windows", [])),
            }
        )
        for window in manifest.get("windows", []):
            window_id = window["window_id"]
            layer_record = next(
                item for item in window["layers"] if int(item["layer_id"]) == layer
            )
            path = evidence_root / role / "windows" / window_id / layer_record["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            # Hash-order windows to make the finite reservoir deterministic but
            # corpus-wide rather than biased toward panel order.
            order = hashlib.sha256(f"{role}:{window_id}".encode()).hexdigest()
            records.append((order, path))
    return [path for _order, path in sorted(records)], manifests


def _build_reservoir(
    *,
    evidence_root: Path,
    roles: list[str],
    layer: int,
    num_experts: int,
    hidden_size: int,
    samples_per_expert: int,
    output: Path,
) -> dict:
    paths, _manifests = _evidence_files(evidence_root, roles, layer)
    samples = torch.empty(
        num_experts, samples_per_expert, hidden_size, dtype=torch.bfloat16
    )
    sampled_counts = torch.zeros(num_experts, dtype=torch.int64)
    routed_counts = torch.zeros(num_experts, dtype=torch.int64)
    input_amax = torch.zeros(num_experts, dtype=torch.float32)

    for file_index, path in enumerate(paths, 1):
        with safe_open(path, framework="pt", device="cpu") as handle:
            hidden = handle.get_tensor("hidden_states")
            topk_ids = handle.get_tensor("topk_ids").to(torch.int64)
        if tuple(hidden.shape[1:]) != (hidden_size,):
            raise ValueError(f"unexpected hidden shape in {path}: {tuple(hidden.shape)}")

        flat_ids = topk_ids.reshape(-1)
        routed_counts += torch.bincount(flat_ids, minlength=num_experts)
        row_amax = hidden.abs().to(torch.float32).amax(dim=1)
        routed_amax = row_amax[:, None].expand_as(topk_ids).reshape(-1)
        input_amax.scatter_reduce_(
            0, flat_ids, routed_amax, reduce="amax", include_self=True
        )

        need = (sampled_counts < samples_per_expert).nonzero().flatten().tolist()
        for expert in need:
            rows = (topk_ids == expert).any(dim=1).nonzero().flatten()
            missing = samples_per_expert - int(sampled_counts[expert])
            take = min(missing, int(rows.numel()))
            if take:
                start = int(sampled_counts[expert])
                samples[expert, start : start + take].copy_(hidden[rows[:take]])
                sampled_counts[expert] += take

        if file_index % 32 == 0 or file_index == len(paths):
            print(
                f"layer={layer} evidence={file_index}/{len(paths)} "
                f"reservoir_min={int(sampled_counts.min())}",
                flush=True,
            )

    if (routed_counts == 0).any():
        missing = (routed_counts == 0).nonzero().flatten().tolist()
        raise RuntimeError(f"layer {layer} has unrouted experts: {missing[:16]}")
    if (sampled_counts < samples_per_expert).any():
        missing = (sampled_counts < samples_per_expert).nonzero().flatten().tolist()
        raise RuntimeError(f"layer {layer} has incomplete reservoirs: {missing[:16]}")
    if not torch.isfinite(input_amax).all() or (input_amax <= 0).any():
        raise RuntimeError(f"layer {layer} has invalid routed input maxima")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    save_file(
        {
            "input_amax": input_amax,
            "routed_counts": routed_counts,
            "sampled_counts": sampled_counts,
            "samples": samples,
        },
        temporary,
    )
    os.replace(temporary, output)
    return {
        "layer": layer,
        "file": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "routed_count_min": int(routed_counts.min()),
        "routed_count_max": int(routed_counts.max()),
        "input_amax_min": float(input_amax.min()),
        "input_amax_max": float(input_amax.max()),
    }


def _compute_layer_scales(
    *,
    layer: int,
    device: str,
    source_model: Path,
    weight_map: dict[str, str],
    reservoir_path: Path,
    num_experts: int,
    swiglu_limit: float,
    gate_up_headroom: float,
    down_headroom: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    torch.cuda.set_device(device)
    reservoir = load_file(reservoir_path)
    samples = reservoir["samples"]
    counts = reservoir["sampled_counts"]
    input_amax = reservoir["input_amax"].to(torch.float32)
    down_amax = torch.zeros(num_experts, dtype=torch.float32)

    needed_shards = {
        weight_map[_layer_key(layer, expert, projection, "weight")]
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
                count = int(counts[expert])
                x = samples[expert, :count].to(device, non_blocking=False)
                gate_key = _layer_key(layer, expert, "gate_proj", "weight")
                up_key = _layer_key(layer, expert, "up_proj", "weight")
                gate_weight = handles[weight_map[gate_key]].get_tensor(gate_key).to(device)
                up_weight = handles[weight_map[up_key]].get_tensor(up_key).to(device)
                gate = F.linear(x, gate_weight).clamp(max=swiglu_limit)
                up = F.linear(x, up_weight).clamp(
                    min=-swiglu_limit, max=swiglu_limit
                )
                intermediate = F.silu(gate) * up
                down_amax[expert] = intermediate.abs().to(torch.float32).amax().cpu()
                del x, gate_weight, up_weight, gate, up, intermediate

    if not torch.isfinite(down_amax).all() or (down_amax <= 0).any():
        raise RuntimeError(f"layer {layer} has invalid down-projection input maxima")

    gate_up_scale = input_amax * gate_up_headroom / NVFP4_GLOBAL_DENOMINATOR
    down_scale = down_amax * down_headroom / NVFP4_GLOBAL_DENOMINATOR
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(num_experts):
        for projection in ("gate_proj", "up_proj"):
            tensors[_layer_key(layer, expert, projection, "input_scale")] = (
                gate_up_scale[expert].clone()
            )
        tensors[_layer_key(layer, expert, "down_proj", "input_scale")] = (
            down_scale[expert].clone()
        )

    stats = {
        "layer": layer,
        "device": device,
        "gate_up_amax_min": float(input_amax.min()),
        "gate_up_amax_max": float(input_amax.max()),
        "down_amax_min": float(down_amax.min()),
        "down_amax_max": float(down_amax.max()),
    }
    print(f"layer={layer} scales_complete device={device}", flush=True)
    return tensors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-scales", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--reservoir-dir", required=True, type=Path)
    parser.add_argument("--roles", nargs="+", default=["fit", "conditional-fit"])
    parser.add_argument("--samples-per-expert", type=int, default=64)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--gate-up-headroom", type=float, default=1.05)
    parser.add_argument("--down-headroom", type=float, default=1.20)
    args = parser.parse_args()

    source_config_path = args.source_model / "config.json"
    source_index_path = args.source_model / "model.safetensors.index.json"
    source_config = _load_json(source_config_path)
    text_config = source_config["text_config"]
    weight_map = _load_json(source_index_path)["weight_map"]
    first_layer = int(text_config["first_k_dense_replace"])
    num_layers = int(text_config["num_hidden_layers"])
    layers = list(range(first_layer, num_layers))
    num_experts = int(text_config["n_routed_experts"])
    hidden_size = int(text_config["hidden_size"])
    swiglu_limit = float(text_config["swiglu_limit"])
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")
    for value in (
        args.gate_up_headroom,
        args.down_headroom,
        args.samples_per_expert,
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("headrooms and samples-per-expert must be positive")

    reservoir_receipts = []
    for layer in layers:
        reservoir_path = args.reservoir_dir / f"layer-{layer:04d}.safetensors"
        if reservoir_path.is_file():
            print(f"layer={layer} reservoir_resume {reservoir_path}", flush=True)
            reservoir_receipts.append(
                {
                    "layer": layer,
                    "file": str(reservoir_path),
                    "bytes": reservoir_path.stat().st_size,
                    "sha256": _sha256(reservoir_path),
                    "resumed": True,
                }
            )
        else:
            reservoir_receipts.append(
                _build_reservoir(
                    evidence_root=args.evidence_root,
                    roles=args.roles,
                    layer=layer,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    samples_per_expert=args.samples_per_expert,
                    output=reservoir_path,
                )
            )

    def device_worker(device_index: int):
        result_tensors: dict[str, torch.Tensor] = {}
        result_stats = []
        for layer in layers[device_index :: len(devices)]:
            layer_tensors, layer_stats = _compute_layer_scales(
                layer=layer,
                device=devices[device_index],
                source_model=args.source_model,
                weight_map=weight_map,
                reservoir_path=args.reservoir_dir / f"layer-{layer:04d}.safetensors",
                num_experts=num_experts,
                swiglu_limit=swiglu_limit,
                gate_up_headroom=args.gate_up_headroom,
                down_headroom=args.down_headroom,
            )
            result_tensors.update(layer_tensors)
            result_stats.append(layer_stats)
        return result_tensors, result_stats

    scale_tensors: dict[str, torch.Tensor] = {}
    layer_stats = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for tensors, stats in executor.map(device_worker, range(len(devices))):
            scale_tensors.update(tensors)
            layer_stats.extend(stats)

    expected = len(layers) * num_experts * 3
    if len(scale_tensors) != expected:
        raise RuntimeError(
            f"input-scale keyset incomplete: actual={len(scale_tensors)} expected={expected}"
        )
    args.output_scales.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_scales.with_name(f".{args.output_scales.name}.incomplete")
    save_file(dict(sorted(scale_tensors.items())), temporary)
    os.replace(temporary, args.output_scales)

    _files, evidence_manifests = _evidence_files(
        args.evidence_root, args.roles, layers[0]
    )
    receipt = {
        "schema": "quant-toolkit.glm53-nvfp4-input-scales.v1",
        "source": {
            "directory": str(args.source_model.resolve()),
            "config_sha256": _sha256(source_config_path),
            "index_sha256": _sha256(source_index_path),
        },
        "evidence_manifests": evidence_manifests,
        "method": {
            "gate_up": "exact routed BF16 input amax over all evidence windows",
            "down": "BF16 source gate/up replay over deterministic routed reservoir",
            "activation": "silu(clamp_max(gate,10))*clamp(up,-10,10)",
            "nvfp4_global_denominator": NVFP4_GLOBAL_DENOMINATOR,
            "samples_per_expert": args.samples_per_expert,
            "gate_up_headroom": args.gate_up_headroom,
            "down_headroom": args.down_headroom,
        },
        "topology": {
            "layers": layers,
            "experts_per_layer": num_experts,
            "input_scale_tensors": len(scale_tensors),
        },
        "reservoirs": reservoir_receipts,
        "layer_stats": sorted(layer_stats, key=lambda item: item["layer"]),
        "output": {
            "path": str(args.output_scales.resolve()),
            "bytes": args.output_scales.stat().st_size,
            "sha256": _sha256(args.output_scales),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    _write_json_atomic(args.receipt, receipt)
    print(json.dumps({"event": "glm53_nvfp4_input_scales_complete", **receipt["output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
