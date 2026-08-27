#!/usr/bin/env python3
"""Export and seal the shared BF16 GLM-5.3 language-model head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from kld_common import canonical_json_sha256, sha256_file
from safetensors import safe_open
from safetensors.torch import save_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-key", default="lm_head.weight")
    parser.add_argument("--hidden-width", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=154880)
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index.get("weight_map", {}).get(args.tensor_key)
    if not shard_name:
        raise RuntimeError(f"tensor is absent from checkpoint index: {args.tensor_key}")
    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        if args.tensor_key not in set(handle.keys()):
            raise RuntimeError(
                f"indexed tensor is absent from shard: {args.tensor_key}"
            )
        weight = handle.get_tensor(args.tensor_key)
    expected_shape = [args.vocab_size, args.hidden_width]
    if weight.dtype != torch.bfloat16 or list(weight.shape) != expected_shape:
        raise RuntimeError(
            f"expected BF16 LM head {expected_shape}; got {weight.dtype} "
            f"{list(weight.shape)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_path = output_dir / "lm-head.safetensors"
    temporary = output_dir / ".lm-head.safetensors.incomplete"
    save_file(
        {"weight": weight.contiguous()},
        str(temporary),
        metadata={
            "source_tensor_key": args.tensor_key,
            "source_model_revision": args.model_revision,
        },
    )
    os.replace(temporary, output_path)
    manifest = {
        "schema": "quant-toolkit.glm53-lm-head.v1",
        "source_model_directory": str(model_dir),
        "source_model_revision": args.model_revision,
        "source_index_file_sha256": sha256_file(index_path),
        "source_config_file_sha256": sha256_file(config_path),
        "source_shard": shard_name,
        "source_shard_sha256": sha256_file(shard_path),
        "source_tensor_key": args.tensor_key,
        "tensor_key": "weight",
        "dtype": "bfloat16",
        "shape": expected_shape,
        "file": output_path.name,
        "file_bytes": output_path.stat().st_size,
        "file_sha256": sha256_file(output_path),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_temporary = output_dir / ".manifest.json.incomplete"
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
