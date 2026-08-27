#!/usr/bin/env python3
"""Optimize exact GLM routed-expert NVFP4 bytes with complete-function replay.

For each expert, stock ModelOpt and nearby exact FP8 block-scale candidates are
constructed from immutable BF16 weights.  All eight stock/optimized choices
across gate, up, and down are evaluated through the complete clamped-SwiGLU
expert using route-weighted conditional reservoirs.  Stock is always one of
the choices, so an accepted per-expert patch cannot regress this proxy.
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

import nvfp4_codec
import torch
import torch.nn.functional as F
from kld_common import canonical_json_sha256, sha256_file
from nvfp4_codec import decode_nvfp4, encode_nvfp4, optimize_block_scales
from safetensors import safe_open
from safetensors.torch import load_file, save_file


SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-optimization.v1"
LAYER_SCHEMA = "quant-toolkit.glm53-nvfp4-routed-weight-layer.v1"
DEFAULT_FACTORS = (0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.125, 1.1875, 1.25)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _save_safetensors_atomic(
    tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    save_file(dict(sorted(tensors.items())), temporary, metadata=metadata)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _weight_key(layer: int, expert: int, projection: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.weight"
    )


def _weighted_coordinate_second_moment(
    rows: torch.Tensor, route_weights: torch.Tensor, power: int
) -> torch.Tensor:
    if power not in (0, 1, 2):
        raise ValueError("route power must be 0, 1, or 2")
    values = rows.to(torch.float32)
    routes = route_weights.to(device=values.device, dtype=torch.float32)
    weights = torch.ones_like(routes) if power == 0 else routes.pow(power)
    denominator = weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
    return (values.square() * weights.unsqueeze(-1)).sum(dim=0) / denominator


def choose_complete_expert_combination(
    *,
    x: torch.Tensor,
    source_down_inputs: torch.Tensor,
    route_weights: torch.Tensor,
    source_down_weight: torch.Tensor,
    gate_choices: tuple[torch.Tensor, torch.Tensor],
    up_choices: tuple[torch.Tensor, torch.Tensor],
    down_choices: tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float,
) -> tuple[tuple[int, int, int], list[float]]:
    """Return the lowest-SSE stock/optimized projection combination."""
    x_f32 = x.to(torch.float32)
    route_f32 = route_weights.to(device=x.device, dtype=torch.float32)
    reference = F.linear(
        source_down_inputs.to(torch.float32), source_down_weight.to(torch.float32)
    )
    scores = []
    combinations = []
    with torch.inference_mode():
        for gate_choice in range(2):
            for up_choice in range(2):
                gate = F.linear(x_f32, gate_choices[gate_choice]).clamp(
                    max=swiglu_limit
                )
                up = F.linear(x_f32, up_choices[up_choice]).clamp(
                    min=-swiglu_limit, max=swiglu_limit
                )
                intermediate = F.silu(gate) * up
                for down_choice in range(2):
                    observed = F.linear(intermediate, down_choices[down_choice])
                    error = (observed - reference) * route_f32.unsqueeze(-1)
                    scores.append(float(error.square().sum(dtype=torch.float64).cpu()))
                    combinations.append((gate_choice, up_choice, down_choice))
    best_index = min(range(len(scores)), key=lambda index: (scores[index], index))
    return combinations[best_index], scores


def _validate_existing_layer(
    *,
    patch_path: Path,
    receipt_path: Path,
    layer: int,
    source_index_sha256: str,
    reservoir_sha256: str,
    conditional_sha256: str,
    codec_sha256: str,
    factors: tuple[float, ...],
) -> dict | None:
    if not patch_path.exists() and not receipt_path.exists():
        return None
    if not patch_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete resumed layer artifact: layer={layer}")
    receipt = _load_json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_json_sha256(body):
        raise RuntimeError(f"layer receipt seal mismatch: {receipt_path}")
    expected = {
        "layer": layer,
        "source_index_sha256": source_index_sha256,
        "reservoir_sha256": reservoir_sha256,
        "conditional_sha256": conditional_sha256,
        "codec_sha256": codec_sha256,
        "factors": list(factors),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"resumed layer provenance mismatch: {key} layer={layer}"
            )
    if receipt.get("patch_sha256") != sha256_file(patch_path):
        raise RuntimeError(f"resumed layer patch hash mismatch: {patch_path}")
    receipt["resumed"] = True
    print(f"layer={layer} optimization_resume", flush=True)
    return receipt


def _optimize_layer(
    *,
    layer: int,
    device: str,
    source_model: Path,
    weight_map: dict[str, str],
    reservoir_path: Path,
    conditional_path: Path,
    output_dir: Path,
    num_experts: int,
    samples_per_expert: int,
    swiglu_limit: float,
    route_power: int,
    factors: tuple[float, ...],
    source_index_sha256: str,
    codec_sha256: str,
) -> dict:
    patch_path = output_dir / f"layer-{layer:04d}.safetensors"
    receipt_path = output_dir / f"layer-{layer:04d}.receipt.json"
    reservoir_sha256 = sha256_file(reservoir_path)
    conditional_sha256 = sha256_file(conditional_path)
    resumed = _validate_existing_layer(
        patch_path=patch_path,
        receipt_path=receipt_path,
        layer=layer,
        source_index_sha256=source_index_sha256,
        reservoir_sha256=reservoir_sha256,
        conditional_sha256=conditional_sha256,
        codec_sha256=codec_sha256,
        factors=factors,
    )
    if resumed is not None:
        return resumed

    torch.cuda.set_device(device)
    reservoir = load_file(reservoir_path, device="cpu")
    conditional = load_file(conditional_path, device="cpu")
    samples = reservoir["samples"]
    down_inputs = conditional["conditional_down_inputs"]
    route_weights = conditional["route_weights"]
    expected_samples = (num_experts, samples_per_expert)
    if tuple(samples.shape[:2]) != expected_samples:
        raise RuntimeError(f"sample topology mismatch for layer={layer}")
    if tuple(down_inputs.shape[:2]) != expected_samples:
        raise RuntimeError(f"conditional topology mismatch for layer={layer}")
    if tuple(route_weights.shape) != expected_samples:
        raise RuntimeError(f"route-weight topology mismatch for layer={layer}")

    needed_shards = {
        weight_map[_weight_key(layer, expert, projection)]
        for expert in range(num_experts)
        for projection in PROJECTIONS
    }
    patch_tensors: dict[str, torch.Tensor] = {}
    records = []
    decoded_digest = hashlib.sha256()
    baseline_total = 0.0
    selected_total = 0.0
    choice_counts: dict[str, int] = {}

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(source_model / shard, framework="pt", device="cpu")
            )
            for shard in needed_shards
        }
        with torch.inference_mode():
            for expert in range(num_experts):
                keys = {
                    projection: _weight_key(layer, expert, projection)
                    for projection in PROJECTIONS
                }
                weights = {
                    projection: handles[weight_map[keys[projection]]]
                    .get_tensor(keys[projection])
                    .to(device)
                    for projection in PROJECTIONS
                }
                x = samples[expert].to(device)
                z = down_inputs[expert].to(device)
                routes = route_weights[expert].to(device)
                x_moment = _weighted_coordinate_second_moment(x, routes, route_power)
                z_moment = _weighted_coordinate_second_moment(z, routes, route_power)

                shared_scale_2 = (
                    torch.maximum(
                        weights["gate_proj"].to(torch.float32).abs().amax(),
                        weights["up_proj"].to(torch.float32).abs().amax(),
                    )
                    / (nvfp4_codec.FP4_MAX * nvfp4_codec.FP8_MAX)
                ).reshape(())
                baselines = {
                    "gate_proj": encode_nvfp4(
                        weights["gate_proj"], scale_2=shared_scale_2
                    ),
                    "up_proj": encode_nvfp4(
                        weights["up_proj"], scale_2=shared_scale_2
                    ),
                    "down_proj": encode_nvfp4(weights["down_proj"]),
                }
                optimized = {
                    "gate_proj": optimize_block_scales(
                        weights["gate_proj"],
                        scale_2=shared_scale_2,
                        factors=factors,
                        coordinate_weights=x_moment,
                    ),
                    "up_proj": optimize_block_scales(
                        weights["up_proj"],
                        scale_2=shared_scale_2,
                        factors=factors,
                        coordinate_weights=x_moment,
                    ),
                    "down_proj": optimize_block_scales(
                        weights["down_proj"],
                        factors=factors,
                        coordinate_weights=z_moment,
                    ),
                }
                baseline_decoded = {
                    projection: decode_nvfp4(*baselines[projection])
                    for projection in PROJECTIONS
                }
                optimized_decoded = {
                    projection: optimized[projection][3]
                    for projection in PROJECTIONS
                }
                choice, scores = choose_complete_expert_combination(
                    x=x,
                    source_down_inputs=z,
                    route_weights=routes,
                    source_down_weight=weights["down_proj"],
                    gate_choices=(
                        baseline_decoded["gate_proj"],
                        optimized_decoded["gate_proj"],
                    ),
                    up_choices=(
                        baseline_decoded["up_proj"],
                        optimized_decoded["up_proj"],
                    ),
                    down_choices=(
                        baseline_decoded["down_proj"],
                        optimized_decoded["down_proj"],
                    ),
                    swiglu_limit=swiglu_limit,
                )
                baseline_sse = scores[0]
                selected_index = choice[0] * 4 + choice[1] * 2 + choice[2]
                selected_sse = scores[selected_index]
                if selected_sse > baseline_sse:
                    raise RuntimeError(
                        "stock-inclusive proxy regressed: "
                        f"layer={layer} expert={expert}"
                    )
                baseline_total += baseline_sse
                selected_total += selected_sse
                choice_label = "".join(str(value) for value in choice)
                choice_counts[choice_label] = choice_counts.get(choice_label, 0) + 1

                for projection, selected in zip(PROJECTIONS, choice, strict=True):
                    if selected:
                        packed, block_scale, scale_2, decoded = optimized[projection]
                    else:
                        packed, block_scale, scale_2 = baselines[projection]
                        decoded = baseline_decoded[projection]
                    prefix = keys[projection].removesuffix(".weight")
                    patch_tensors[keys[projection]] = packed.cpu().contiguous()
                    patch_tensors[prefix + ".weight_scale"] = (
                        block_scale.cpu().contiguous()
                    )
                    patch_tensors[prefix + ".weight_scale_2"] = (
                        scale_2.cpu().reshape(()).contiguous()
                    )
                    decoded_digest.update(
                        decoded.cpu()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )

                records.append(
                    {
                        "expert": expert,
                        "choice_gate_up_down": choice_label,
                        "baseline_complete_sse": baseline_sse,
                        "selected_complete_sse": selected_sse,
                        "relative_sse_reduction": (
                            (baseline_sse - selected_sse) / baseline_sse
                            if baseline_sse
                            else 0.0
                        ),
                    }
                )
                del (
                    weights,
                    x,
                    z,
                    routes,
                    x_moment,
                    z_moment,
                    baselines,
                    optimized,
                    baseline_decoded,
                    optimized_decoded,
                )
                if (expert + 1) % 16 == 0 or expert + 1 == num_experts:
                    print(
                        f"layer={layer} expert={expert + 1}/{num_experts} "
                        f"relative_sse_reduction="
                        f"{(baseline_total - selected_total) / baseline_total:.8f}",
                        flush=True,
                    )

    expected_tensors = num_experts * len(PROJECTIONS) * 3
    if len(patch_tensors) != expected_tensors:
        raise RuntimeError(
            f"layer patch keyset incomplete: {len(patch_tensors)} != {expected_tensors}"
        )
    metadata = {
        "schema": LAYER_SCHEMA,
        "layer": str(layer),
        "source_index_sha256": source_index_sha256,
        "reservoir_sha256": reservoir_sha256,
        "conditional_sha256": conditional_sha256,
        "codec_sha256": codec_sha256,
        "factors": json.dumps(list(factors), separators=(",", ":")),
        "route_power": str(route_power),
        "runtime_format": (
            "packed-e2m1-plus-fp8-e4m3fn-block-scale-plus-f32-secondary-scale"
        ),
    }
    _save_safetensors_atomic(patch_tensors, patch_path, metadata)
    receipt = {
        "schema": LAYER_SCHEMA,
        "layer": layer,
        "device": device,
        "source_index_sha256": source_index_sha256,
        "reservoir_sha256": reservoir_sha256,
        "conditional_sha256": conditional_sha256,
        "codec_sha256": codec_sha256,
        "factors": list(factors),
        "route_power": route_power,
        "objective": (
            "route-weighted complete expert output SSE over deterministic "
            "conditional reservoir"
        ),
        "stock_included": True,
        "choice_counts": dict(sorted(choice_counts.items())),
        "baseline_complete_sse": baseline_total,
        "selected_complete_sse": selected_total,
        "relative_sse_reduction": (
            (baseline_total - selected_total) / baseline_total
            if baseline_total
            else 0.0
        ),
        "experts": records,
        "patch_tensors": len(patch_tensors),
        "patch_bytes": patch_path.stat().st_size,
        "patch_sha256": sha256_file(patch_path),
        "decoded_reconstruction_sha256": decoded_digest.hexdigest(),
        "resumed": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_json_atomic(receipt_path, receipt)
    print(f"layer={layer} optimization_complete device={device}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--reservoir-dir", required=True, type=Path)
    parser.add_argument("--conditional-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--samples-per-expert", type=int, default=256)
    parser.add_argument("--route-power", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    args = parser.parse_args(argv)

    source_model = args.source_model.resolve()
    config_path = source_model / "config.json"
    index_path = source_model / "model.safetensors.index.json"
    text_config = _load_json(config_path)["text_config"]
    weight_map = _load_json(index_path)["weight_map"]
    all_layers = list(
        range(
            int(text_config["first_k_dense_replace"]),
            int(text_config["num_hidden_layers"]),
        )
    )
    layers = all_layers if args.layers is None else args.layers
    if (
        not layers
        or len(set(layers)) != len(layers)
        or not set(layers) <= set(all_layers)
    ):
        raise ValueError("--layers must be a nonempty unique subset of routed layers")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")
    factors = tuple(float(value) for value in args.factors)
    if not factors or 1.0 not in factors or any(
        not math.isfinite(value) or value <= 0 for value in factors
    ):
        raise ValueError("--factors must be positive, finite, and contain 1.0")

    num_experts = int(text_config["n_routed_experts"])
    swiglu_limit = float(text_config["swiglu_limit"])
    source_index_sha256 = sha256_file(index_path)
    codec_sha256 = sha256_file(Path(nvfp4_codec.__file__).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def worker(device_index: int) -> list[dict]:
        records = []
        for layer in layers[device_index :: len(devices)]:
            records.append(
                _optimize_layer(
                    layer=layer,
                    device=devices[device_index],
                    source_model=source_model,
                    weight_map=weight_map,
                    reservoir_path=(
                        args.reservoir_dir / f"layer-{layer:04d}.safetensors"
                    ),
                    conditional_path=(
                        args.conditional_dir / f"layer-{layer:04d}.safetensors"
                    ),
                    output_dir=args.output_dir,
                    num_experts=num_experts,
                    samples_per_expert=args.samples_per_expert,
                    swiglu_limit=swiglu_limit,
                    route_power=args.route_power,
                    factors=factors,
                    source_index_sha256=source_index_sha256,
                    codec_sha256=codec_sha256,
                )
            )
        return records

    layer_records = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for records in executor.map(worker, range(len(devices))):
            layer_records.extend(records)
    baseline_sse = sum(record["baseline_complete_sse"] for record in layer_records)
    selected_sse = sum(record["selected_complete_sse"] for record in layer_records)
    receipt = {
        "schema": SCHEMA,
        "source": {
            "directory": str(source_model),
            "config_sha256": sha256_file(config_path),
            "index_sha256": source_index_sha256,
        },
        "evidence": {
            "reservoir_dir": str(args.reservoir_dir.resolve()),
            "conditional_dir": str(args.conditional_dir.resolve()),
            "samples_per_expert": args.samples_per_expert,
            "factorized_rows_preserve_cross_coordinate_terms": True,
            "route_weight_power": args.route_power,
        },
        "method": {
            "factors": list(factors),
            "stock_included": True,
            "combinations_per_expert": 8,
            "selection_unit": "complete routed expert gate/up/down tuple",
            "runtime_format": (
                "packed-e2m1-plus-fp8-e4m3fn-block-scale-plus-f32-secondary-scale"
            ),
        },
        "topology": {
            "layers": layers,
            "experts_per_layer": num_experts,
            "patch_tensors": sum(record["patch_tensors"] for record in layer_records),
        },
        "proxy": {
            "baseline_complete_sse": baseline_sse,
            "selected_complete_sse": selected_sse,
            "relative_sse_reduction": (
                (baseline_sse - selected_sse) / baseline_sse
                if baseline_sse
                else 0.0
            ),
        },
        "layers": sorted(layer_records, key=lambda record: record["layer"]),
        "tools": {
            "optimizer_sha256": sha256_file(Path(__file__).resolve()),
            "codec_sha256": codec_sha256,
        },
        "result": "PASS",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_json_atomic(args.receipt, receipt)
    print(
        json.dumps(
            {
                "event": "glm53_nvfp4_routed_weight_optimization_complete",
                "receipt": str(args.receipt.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                **receipt["proxy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
