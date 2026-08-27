#!/usr/bin/env python3
"""Verify and seal GLM-5.3 routed-expert NVFP4 checkpoint coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_loader import (  # noqa: E402
    _checkpoint_key_to_model_key,
    _glm5_next_conv_target_key,
)

SCHEMA = "quant-toolkit.glm53-nvfp4-coverage.v1"
BLOCK_SIZE = 16
ROUTED_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def tensor_nbytes(info: dict) -> int:
    try:
        element_bytes = DTYPE_BYTES[info["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {info['dtype']}") from exc
    return numel(tuple(info["shape"])) * element_bytes


def read_checkpoint(
    directory: Path,
    *,
    scalar_keys: set[str] | None = None,
) -> tuple[dict, dict[str, dict], dict[str, float], list[dict]]:
    index_path = directory / "model.safetensors.index.json"
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")

    keys_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise TypeError("checkpoint weight_map keys and values must be strings")
        keys_by_shard[shard].add(key)

    info: dict[str, dict] = {}
    scalars: dict[str, float] = {}
    files: list[dict] = []
    scalar_keys = scalar_keys or set()
    for shard, indexed_keys in sorted(keys_by_shard.items()):
        path = directory / shard
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
        files.append({"path": shard, "bytes": path.stat().st_size})
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != indexed_keys:
                missing = sorted(indexed_keys - actual_keys)
                extra = sorted(actual_keys - indexed_keys)
                raise ValueError(
                    f"index/shard key mismatch for {path}: "
                    f"missing={missing[:8]} extra={extra[:8]}"
                )
            for key in sorted(actual_keys):
                tensor_slice = handle.get_slice(key)
                info[key] = {
                    "dtype": tensor_slice.get_dtype(),
                    "shape": list(tensor_slice.get_shape()),
                    "shard": shard,
                }
                if key in scalar_keys:
                    tensor = handle.get_tensor(key).reshape(-1)
                    if tensor.numel() != 1:
                        raise ValueError(f"expected scalar tensor: {key}")
                    scalars[key] = float(tensor.float().item())

    if set(info) != set(weight_map):
        raise AssertionError("checkpoint header scan did not cover the complete index")
    return index, info, scalars, files


def source_routed_keys(source_info: dict[str, dict], text_config: dict) -> list[str]:
    first_sparse_layer = int(text_config["first_k_dense_replace"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["n_routed_experts"])
    selected: list[str] = []
    observed: dict[tuple[int, str], set[int]] = defaultdict(set)
    for key in source_info:
        match = ROUTED_WEIGHT_RE.fullmatch(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        if first_sparse_layer <= layer < num_hidden_layers:
            selected.append(key)
            observed[(layer, match.group("proj"))].add(int(match.group("expert")))

    expected_experts = set(range(num_experts))
    for layer in range(first_sparse_layer, num_hidden_layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            actual = observed.get((layer, projection), set())
            if actual != expected_experts:
                raise ValueError(
                    "source routed-expert topology mismatch: "
                    f"layer={layer} projection={projection} "
                    f"expected={num_experts} actual={len(actual)}"
                )
    expected_count = (num_hidden_layers - first_sparse_layer) * num_experts * 3
    if len(selected) != expected_count:
        raise AssertionError("selected routed tensor count does not match topology")
    return sorted(selected)


def sidecar_keys(weight_key: str) -> tuple[str, str]:
    prefix = weight_key.removesuffix(".weight")
    return (
        prefix + ".weight_scale",
        prefix + ".weight_scale_2",
    )


def input_scale_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".input_scale"


def canonical_source_info(source_info: dict[str, dict], text_config: dict) -> dict[str, dict]:
    """Map raw checkpoint tensors to the pinned Transformers model keyspace."""
    protected_prefix = (
        f"model.language_model.layers.{int(text_config['num_hidden_layers'])}."
    )
    canonical: dict[str, dict] = {}
    fused: dict[str, dict[str, dict]] = defaultdict(dict)

    for raw_key, info in source_info.items():
        if raw_key.startswith(protected_prefix):
            mapped = raw_key
        else:
            mapped = _checkpoint_key_to_model_key(raw_key)
            glm_conv = _glm5_next_conv_target_key(mapped)
            if glm_conv is not None:
                target, part = glm_conv
                fused[target][part] = info
                continue
        if mapped in canonical:
            raise ValueError(f"checkpoint key mapping collision: {raw_key} -> {mapped}")
        canonical[mapped] = dict(info)

    required_parts = {"conv1d.weight": {"q", "k", "v"}}
    for target, parts in fused.items():
        suffix = next(
            (name for name in required_parts if target.endswith(name)),
            None,
        )
        if suffix is None or set(parts) != required_parts[suffix]:
            raise ValueError(
                f"incomplete source fusion for {target}: parts={sorted(parts)}"
            )
        ordered = [parts[name] for name in sorted(parts)]
        dtypes = {item["dtype"] for item in ordered}
        trailing_shapes = {tuple(item["shape"][1:]) for item in ordered}
        if len(dtypes) != 1 or len(trailing_shapes) != 1:
            raise ValueError(f"incompatible source fusion tensors for {target}")
        canonical[target] = {
            "dtype": ordered[0]["dtype"],
            "shape": [sum(item["shape"][0] for item in ordered), *ordered[0]["shape"][1:]],
            "shard": "<canonical-fusion>",
        }
    return canonical


def validate_quantized_tensor(
    key: str,
    source: dict,
    candidate_info: dict[str, dict],
) -> None:
    source_shape = tuple(source["shape"])
    if len(source_shape) != 2 or source_shape[-1] % BLOCK_SIZE:
        raise ValueError(
            f"unsupported source routed weight shape for {key}: {source_shape}"
        )
    packed = candidate_info[key]
    expected_packed_shape = (*source_shape[:-1], source_shape[-1] // 2)
    if packed["dtype"] != "U8" or tuple(packed["shape"]) != expected_packed_shape:
        raise ValueError(
            f"invalid NVFP4 packed weight {key}: "
            f"dtype={packed['dtype']} shape={packed['shape']} "
            f"expected=U8/{expected_packed_shape}"
        )

    weight_scale_key, weight_scale_2_key = sidecar_keys(key)
    expected_scale_shape = (*source_shape[:-1], source_shape[-1] // BLOCK_SIZE)
    weight_scale = candidate_info[weight_scale_key]
    if (
        weight_scale["dtype"] != "F8_E4M3"
        or tuple(weight_scale["shape"]) != expected_scale_shape
    ):
        raise ValueError(
            f"invalid NVFP4 block scale {weight_scale_key}: "
            f"dtype={weight_scale['dtype']} shape={weight_scale['shape']} "
            f"expected=F8_E4M3/{expected_scale_shape}"
        )
    scalar = candidate_info[weight_scale_2_key]
    if scalar["dtype"] != "F32" or numel(tuple(scalar["shape"])) != 1:
        raise ValueError(
            f"invalid NVFP4 scalar {weight_scale_2_key}: "
            f"dtype={scalar['dtype']} shape={scalar['shape']}"
        )


def write_report(path: Path, report: dict) -> None:
    report = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_json_sha256(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16-reserve-layers", type=int, nargs="*", default=[])
    parser.add_argument(
        "--hash-candidate-shards",
        action="store_true",
        help="Include full SHA-256 hashes for every candidate tensor shard.",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source_model).resolve()
    candidate_dir = Path(args.candidate_model).resolve()
    source_config_path = source_dir / "config.json"
    candidate_config_path = candidate_dir / "config.json"
    source_config = load_json(source_config_path)
    candidate_config = load_json(candidate_config_path)
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("source GLM-5.3 config has no text_config object")
    architectures = source_config.get("architectures", [])
    if "Glm5NextForConditionalGeneration" not in architectures:
        raise ValueError("source checkpoint is not GLM-5.3-Flash")

    quant_config = candidate_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise TypeError("candidate config has no quantization_config object")
    quant_algo = str(quant_config.get("quant_algo", "")).upper()
    if quant_algo not in {"NVFP4", "W4A16_NVFP4"}:
        raise ValueError(
            "candidate quantization_config must declare NVFP4 or W4A16_NVFP4"
        )
    requires_input_scales = quant_algo == "NVFP4"

    _source_index, source_info, _unused, source_files = read_checkpoint(source_dir)
    selected = source_routed_keys(source_info, text_config)
    reserve_layers = sorted(set(args.bf16_reserve_layers))
    first_sparse = int(text_config["first_k_dense_replace"])
    last_main = int(text_config["num_hidden_layers"]) - 1
    if any(layer < first_sparse or layer > last_main for layer in reserve_layers):
        parser.error(
            f"BF16 reserve layers must be main routed layers {first_sparse}..{last_main}"
        )
    reserved = {
        key
        for key in selected
        if int(ROUTED_WEIGHT_RE.fullmatch(key).group("layer")) in reserve_layers
    }
    quantized = [key for key in selected if key not in reserved]
    quantized_set = set(quantized)
    ignored = quant_config.get("ignore", [])
    for layer in reserve_layers:
        pattern = f"model.language_model.layers.{layer}.mlp.experts*"
        if pattern not in ignored:
            raise ValueError(f"BF16 reserve is absent from quant ignore list: {pattern}")
    canonical_info = canonical_source_info(source_info, text_config)
    scalar_keys = {sidecar_keys(key)[1] for key in quantized}
    if requires_input_scales:
        scalar_keys.update(input_scale_key(key) for key in quantized)
    _candidate_index, candidate_info, scalars, candidate_files = read_checkpoint(
        candidate_dir,
        scalar_keys=scalar_keys,
    )

    expected_candidate_keys = set(canonical_info)
    for key in quantized:
        expected_candidate_keys.update(sidecar_keys(key))
        if requires_input_scales:
            expected_candidate_keys.add(input_scale_key(key))
    actual_candidate_keys = set(candidate_info)
    if actual_candidate_keys != expected_candidate_keys:
        missing = sorted(expected_candidate_keys - actual_candidate_keys)
        extra = sorted(actual_candidate_keys - expected_candidate_keys)
        raise ValueError(
            "candidate tensor keyset is not routed-expert-only NVFP4: "
            f"missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:8]} first_extra={extra[:8]}"
        )

    for key, source_tensor in canonical_info.items():
        if key in quantized_set:
            validate_quantized_tensor(key, source_tensor, candidate_info)
        elif (
            candidate_info[key]["dtype"] != source_tensor["dtype"]
            or candidate_info[key]["shape"] != source_tensor["shape"]
        ):
            raise ValueError(f"preserved tensor dtype/shape changed: {key}")

    for key in quantized:
        _weight_scale, weight_scale_2 = sidecar_keys(key)
        value = scalars[weight_scale_2]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"non-positive or non-finite quant scalar: {weight_scale_2}"
            )
        activation_scale = input_scale_key(key)
        if requires_input_scales:
            info = candidate_info[activation_scale]
            if info["dtype"] != "F32" or numel(tuple(info["shape"])) != 1:
                raise ValueError(
                    f"invalid NVFP4 activation scalar {activation_scale}: "
                    f"dtype={info['dtype']} shape={info['shape']}"
                )
            activation_value = scalars[activation_scale]
            if not math.isfinite(activation_value) or activation_value <= 0.0:
                raise ValueError(
                    f"non-positive or non-finite activation scalar: {activation_scale}"
                )

    first_sparse_layer = int(text_config["first_k_dense_replace"])
    num_hidden_layers = int(text_config["num_hidden_layers"])
    num_experts = int(text_config["n_routed_experts"])
    for layer in range(first_sparse_layer, num_hidden_layers):
        if layer in reserve_layers:
            continue
        for expert in range(num_experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            gate = scalars[prefix + ".gate_proj.weight_scale_2"]
            up = scalars[prefix + ".up_proj.weight_scale_2"]
            if gate != up:
                raise ValueError(
                    "gate/up weight_scale_2 is not tied: "
                    f"layer={layer} expert={expert} gate={gate} up={up}"
                )
            if requires_input_scales:
                gate_input = scalars[input_scale_key(prefix + ".gate_proj.weight")]
                up_input = scalars[input_scale_key(prefix + ".up_proj.weight")]
                if gate_input != up_input:
                    raise ValueError(
                        "gate/up input_scale is not tied: "
                        f"layer={layer} expert={expert} "
                        f"gate={gate_input} up={up_input}"
                    )

    source_tensor_bytes = sum(tensor_nbytes(value) for value in source_info.values())
    selected_params = sum(
        numel(tuple(source_info[key]["shape"])) for key in quantized
    )
    candidate_tensor_bytes = sum(
        tensor_nbytes(value) for value in candidate_info.values()
    )
    preserved_tensor_bytes = sum(
        tensor_nbytes(canonical_info[key])
        for key in canonical_info
        if key not in quantized_set
    )
    expected_candidate_tensor_bytes = (
        preserved_tensor_bytes
        + selected_params // 2
        + selected_params // BLOCK_SIZE
        + len(quantized) * 4
        + (len(quantized) * 4 if requires_input_scales else 0)
    )
    if candidate_tensor_bytes != expected_candidate_tensor_bytes:
        raise ValueError(
            "candidate tensor-byte total disagrees with exact NVFP4 layout: "
            f"actual={candidate_tensor_bytes} expected={expected_candidate_tensor_bytes}"
        )

    if args.hash_candidate_shards:
        for item in candidate_files:
            item["sha256"] = sha256_file(candidate_dir / item["path"])

    selected_key_sha256 = canonical_json_sha256(quantized)
    report = {
        "schema": SCHEMA,
        "source": {
            "directory": str(source_dir),
            "revision": args.source_revision,
            "config_sha256": sha256_file(source_config_path),
            "index_sha256": sha256_file(source_dir / "model.safetensors.index.json"),
            "tensor_count": len(source_info),
            "tensor_bytes": source_tensor_bytes,
            "shards": source_files,
        },
        "candidate": {
            "name": args.candidate_name,
            "directory": str(candidate_dir),
            "config_sha256": sha256_file(candidate_config_path),
            "index_sha256": sha256_file(candidate_dir / "model.safetensors.index.json"),
            "tensor_count": len(candidate_info),
            "tensor_bytes": candidate_tensor_bytes,
            "shards_hashed": args.hash_candidate_shards,
            "shards": candidate_files,
        },
        "coverage": {
            "first_sparse_layer": first_sparse_layer,
            "last_main_layer": num_hidden_layers - 1,
            "protected_mtp_layer": num_hidden_layers,
            "experts_per_layer": num_experts,
            "quantized_weight_tensors": len(quantized),
            "quantized_parameters": selected_params,
            "source_parameters": sum(
                numel(tuple(value["shape"])) for value in source_info.values()
            ),
            "quantized_parameter_fraction": selected_params
            / sum(numel(tuple(value["shape"])) for value in source_info.values()),
            "selected_keyset_sha256": selected_key_sha256,
            "bf16_reserve_layers": reserve_layers,
            "bf16_reserve_weight_tensors": len(reserved),
            "block_size": BLOCK_SIZE,
            "packed_weight_dtype": "U8",
            "block_scale_dtype": "F8_E4M3",
            "global_scale_dtype": "F32",
            "activation_mode": (
                "w4a4_dynamic_group_static_global"
                if requires_input_scales
                else "w4a16_bf16"
            ),
            "input_scale_storage": (
                "static_f32_per_expert_projection"
                if requires_input_scales
                else "absent"
            ),
            "gate_up_weight_scale_2_tied": True,
            "gate_up_input_scale_tied": requires_input_scales,
            "unexpected_quantized_tensors": 0,
        },
    }
    write_report(Path(args.output).resolve(), report)
    print(
        json.dumps({"event": "glm53_nvfp4_checkpoint_verified", **report["coverage"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
