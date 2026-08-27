"""Streaming layer-by-layer HF checkpoint export for quantized models.

Forked from modelopt's unified_export_hf.py to support models that don't fit
in CPU RAM (e.g. GLM-5 at ~1.5TB BF16). Instead of materializing the entire
state dict at once, we iterate over top-level modules, strip accelerate hooks,
move to CPU, extract state dict, and write safetensors shards incrementally.
"""

import collections.abc
import gc
import json
import re
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.layer_utils import (
    get_expert_linear_names,
    is_moe,
    set_expert_quantizer_amax,
)
from modelopt.torch.export.quant_utils import (
    get_quant_config,
    postprocess_state_dict,
)
from modelopt.torch.export.unified_export_hf import (
    _process_quantized_modules,
    requantize_resmooth_fused_llm_layers,
)


MAX_SHARD_SIZE = 5 * 1024**3  # 5 GB per shard.
SCALE_SHARD_NAME = "model-inputscales.safetensors"
MTP_SHARD_NAME = "model-mtp.safetensors"

_EXPERT_PROJ_RE = re.compile(
    r"^(.*\.experts)\.(gate_proj|up_proj|down_proj)\.(\d+)\.(.+)$"
)


def _remap_expert_key_to_checkpoint(key: str) -> str:
    """Reverse the _QuantFusedExperts module-tree layout back to checkpoint format.

    Module tree (projection-first): experts.gate_proj.0.weight
    Checkpoint  (expert-first):     experts.0.gate_proj.weight
    """
    m = _EXPERT_PROJ_RE.match(key)
    if m:
        prefix, proj, idx, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{prefix}.{idx}.{proj}.{suffix}"
    return key


def _handle_moe_expert_quantizers(model: nn.Module) -> None:
    """Handle input quantizers of MoE experts that were not calibrated.

    Mirrors the logic from _export_transformers_checkpoint lines 646-697.
    """
    for _, sub_module in model.named_modules():
        if is_moe(sub_module) and hasattr(sub_module, "experts"):
            expert_linear_names = get_expert_linear_names(sub_module)
            for linear_name in expert_linear_names:
                if "QuantDbrxExperts" in type(sub_module.experts).__name__:
                    experts_mlp = sub_module.experts.mlp
                    if hasattr(experts_mlp, linear_name):
                        linear_modulelist = getattr(experts_mlp, linear_name)
                        if hasattr(linear_modulelist, "__iter__"):
                            set_expert_quantizer_amax(
                                modules=list(linear_modulelist),
                                quantizer_attrs=["input_quantizer"],
                            )
                elif "QuantGptOssExperts" in type(sub_module.experts).__name__:
                    gpt_oss_linear_names = ["gate_up_proj", "down_proj"]
                    for linear_name in gpt_oss_linear_names:
                        if hasattr(sub_module.experts, linear_name):
                            linear_module = getattr(sub_module.experts, linear_name)
                            if hasattr(linear_module, "input_quantizer"):
                                set_expert_quantizer_amax(
                                    modules=[linear_module],
                                    quantizer_attrs=["input_quantizer"],
                                )
                elif hasattr(sub_module.experts, "gate_proj") and isinstance(
                    getattr(sub_module.experts, "gate_proj", None), nn.ModuleList
                ):
                    # _QuantFusedExperts (GLM5): per-projection ModuleLists
                    # gate_proj[i], up_proj[i], down_proj[i] are quantized nn.Linear
                    # Force device=cpu — storage GPUs are packed with weights and
                    # can't allocate even small tensors for the internal torch.stack.
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj_list = getattr(sub_module.experts, proj_name, None)
                        if proj_list is not None and isinstance(proj_list, nn.ModuleList):
                            set_expert_quantizer_amax(
                                modules=list(proj_list),
                                quantizer_attrs=["input_quantizer", "weight_quantizer"],
                                device=torch.device("cpu"),
                            )
                elif isinstance(sub_module.experts, collections.abc.Iterable):
                    try:
                        set_expert_quantizer_amax(
                            modules=[
                                getattr(expert, linear_name)
                                for expert in sub_module.experts
                            ],
                            quantizer_attrs=["input_quantizer", "weight_quantizer"],
                        )
                    except AttributeError as e:
                        expert_types = [
                            type(expert).__name__ for expert in sub_module.experts
                        ]
                        raise AttributeError(
                            f"Failed to access attribute '{linear_name}' on experts. "
                            f"MoE module type: {type(sub_module).__name__}, "
                            f"Expert types: {expert_types}, "
                            f"Expected linear names: {expert_linear_names}. "
                            f"Original error: {e}"
                        ) from e
                else:
                    raise NotImplementedError(
                        f"MoE model with experts type "
                        f"'{type(sub_module.experts).__name__}' is not supported."
                    )


def _strip_hooks(module: nn.Module) -> None:
    """Remove accelerate hooks from a single module and all its children."""
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return
    remove_hook_from_module(module, recurse=True)


def _tensor_size(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _flush_shard(
    shard: dict[str, torch.Tensor],
    shard_idx: int,
    export_dir: Path,
    weight_map: dict[str, str],
) -> int:
    """Write one safetensors shard to disk and update weight_map. Returns next shard_idx."""
    fname = f"model-{shard_idx:05d}.safetensors"
    print(f"  Writing shard {fname} ({len(shard)} tensors, "
          f"{sum(_tensor_size(v) for v in shard.values()) / 1e9:.2f} GB)")
    save_file(shard, str(export_dir / fname))
    for key in shard:
        weight_map[key] = fname
    return shard_idx + 1


def _enumerate_top_level_modules(model: nn.Module):
    """Yield (prefix, module) pairs for every top-level piece of the model.

    This produces the decoder layers one at a time, plus the embedding, norm,
    and lm_head modules. The prefixes are the dotted names as they appear in
    the full model state dict (e.g. "model.layers.0", "model.embed_tokens").

    Handles both standard CausalLM (model.model.layers) and VL models where
    layers are nested under model.model.language_model.layers.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError(
            "Expected model to have a .model attribute (standard HF CausalLM layout)."
        )

    lm = getattr(inner, "language_model", None)
    if lm is not None:
        # VL model: layers are inside model.model.language_model.
        for name, child in inner.named_children():
            if name == "language_model":
                continue
            yield f"model.{name}", child

        layers = getattr(lm, "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                yield f"model.language_model.layers.{i}", layer

        for name, child in lm.named_children():
            if name == "layers":
                continue
            yield f"model.language_model.{name}", child
    else:
        # Standard CausalLM: layers directly on model.model.
        layers = getattr(inner, "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                yield f"model.layers.{i}", layer

        for name, child in inner.named_children():
            if name == "layers":
                continue
            yield f"model.{name}", child

    # Top-level children of the outer model that are not .model (lm_head, ...).
    for name, child in model.named_children():
        if name == "model":
            continue
        yield name, child


def export_hf(
    model: nn.Module,
    export_dir: str | Path,
    dtype: torch.dtype | None = None,
    prepare_fn: Any | None = None,
    extra_mtp_prefixes: list[str] | None = None,
    preserve_remote_code: bool = False,
    source_dir: str | Path | None = None,
) -> None:
    """Export a quantized HF model to safetensors, streaming one layer at a time.

    This avoids materializing the full state dict in CPU RAM. For each top-level
    module we: process quantized weights in-place, strip accelerate hooks, move
    to CPU, extract its state dict slice, and flush to a shard file.

    Args:
        prepare_fn: Optional callback called after pre-export processing but
            before the per-layer export loop. Used by StreamingModelLoader to
            remove streaming hooks and install materialization callbacks at the
            right point (after requantize_resmooth's dummy forward pass).
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _resolve_source_dir(model, source_dir)

    if dtype is None:
        dtype = model.config.torch_dtype

    # --- Pre-export processing (all on GPU, hooks still intact) ---
    _handle_moe_expert_quantizers(model)
    resmooth_target = model
    _patched_arch = False
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        resmooth_target = model.model.language_model
        if getattr(resmooth_target.config, "architectures", None) is None:
            resmooth_target.config.architectures = []
            _patched_arch = True
    # Skipped: requantize_resmooth_fused_llm_layers does a full dummy forward
    # then fuses gate/up amaxes and AWQ pre_quant_scales. For NVFP4 without
    # AWQ/SVDQuant this is all no-ops (resmooth_only returns early for MoE,
    # pre_quant_scale is None, gate/up tying already done in quantize.py).
    # Calling it on streaming models OOMs because the dummy forward + amax
    # stacking keeps too much on GPU 0 simultaneously.
    if _patched_arch:
        del resmooth_target.config.architectures
    quant_config = get_quant_config(model)
    hf_quant_config = convert_hf_quant_config_format(quant_config) if quant_config else None

    kv_cache_max_bound = 448
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    # Transition from streaming execution to export mode. Must happen after
    # pre-processing (which may do dummy forward passes needing streaming hooks)
    # but before the per-layer loop (which needs materialization callbacks).
    if prepare_fn is not None:
        prepare_fn(model)

    # --- Streaming layer-by-layer export ---
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    scale_shard: dict[str, torch.Tensor] = {}  # input_scale tensors in dedicated shard
    shard_size = 0
    shard_idx = 1
    total_tensors = 0

    print("\nStreaming export — processing layers...")
    for prefix, module in _enumerate_top_level_modules(model):
        print(f"  Processing {prefix}...")

        # Bring everything to CPU before processing.  Meta/disk layers use
        # a materialization callback; GPU-stored layers just .to("cpu").
        materialize_fn = getattr(module, "_streaming_materialize", None)
        if materialize_fn is not None:
            materialize_fn(module)
            delattr(module, "_streaming_materialize")
        else:
            _strip_hooks(module)
            module.to("cpu")
            torch.cuda.empty_cache()

        # Pack quantized weights in-place for this subtree (all on CPU now).
        _process_quantized_modules(module, dtype)

        # Extract this module's state dict with correct full-model key names.
        # input_scale tensors are collected separately into a dedicated shard
        # so they can be updated without touching the large weight shards.
        for local_key, tensor in module.state_dict().items():
            full_key = _remap_expert_key_to_checkpoint(f"{prefix}.{local_key}")
            cpu_tensor = tensor.detach().contiguous().cpu()
            t_size = _tensor_size(cpu_tensor)

            if full_key.endswith(".input_scale"):
                scale_shard[full_key] = cpu_tensor.clone()
            else:
                shard[full_key] = cpu_tensor
                shard_size += t_size

            total_tensors += 1

            if shard_size >= MAX_SHARD_SIZE:
                shard_idx = _flush_shard(shard, shard_idx, export_dir, weight_map)
                shard.clear()
                shard_size = 0

        # Free this module's parameters to reclaim CPU RAM.
        for param in module.parameters():
            param.data = torch.empty(0)
        for buf_name, buf in module.named_buffers():
            buf.data = torch.empty(0)
        gc.collect()

    # Flush remaining weight tensors.
    if shard:
        shard_idx = _flush_shard(shard, shard_idx, export_dir, weight_map)
        shard.clear()

    total_shards = shard_idx - 1

    # Flush input_scale tensors into a dedicated file (not a numbered shard).
    # This file is excluded from postprocess/rename so it keeps its fixed name,
    # making it trivial to update scales without touching weight shards.
    if scale_shard:
        scale_size = sum(_tensor_size(v) for v in scale_shard.values())
        scale_path = export_dir / SCALE_SHARD_NAME
        save_file(scale_shard, str(scale_path))
        for key in scale_shard:
            weight_map[key] = SCALE_SHARD_NAME
        print(f"  Writing {SCALE_SHARD_NAME} ({len(scale_shard)} tensors, "
              f"{scale_size / 1e3:.0f} KB)")
        scale_shard.clear()

    print(f"\nWrote {total_tensors} tensors across {total_shards + 1} files.")

    # Merge MTP (multi-token prediction) weights from source checkpoint if present.
    shard_idx, total_tensors = _merge_mtp_weights(
        source_dir, export_dir, weight_map, shard_idx, total_tensors,
        extra_prefixes=extra_mtp_prefixes or [],
    )

    # Postprocess: filter quantizer metadata keys and rename kv cache scales.
    # We need to do this on the weight_map keys. The actual tensor data has
    # already been written, so we re-read shards that need changes.
    _postprocess_shards(export_dir, weight_map, kv_cache_max_bound, kv_cache_format)

    # Rename shards to final HF naming (model-00001-of-NNNNN.safetensors).
    _rename_shards(export_dir, weight_map, total_shards)

    # --- Save config, tokenizer, and quant metadata ---
    _save_model_metadata(
        model,
        export_dir,
        hf_quant_config,
        preserve_remote_code=preserve_remote_code,
        source_dir=source_dir,
    )

    print(f"Export complete. Saved to: {export_dir}")


def _resolve_source_dir(model: nn.Module, source_dir: str | Path | None) -> Path:
    if source_dir is not None:
        resolved = Path(source_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Model source directory does not exist: {resolved}")
        return resolved

    name_or_path = model.config._name_or_path
    if Path(name_or_path).is_dir():
        return Path(name_or_path).resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            name_or_path,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
            local_files_only=True,
        )
    )


def _merge_mtp_weights(
    source_dir: Path,
    export_dir: Path,
    weight_map: dict[str, str],
    shard_idx: int,
    total_tensors: int,
    extra_prefixes: list[str] | None = None,
) -> tuple[int, int]:
    """Copy MTP weights unquantized from the source checkpoint into our export.

    Keys starting with "mtp." are always included. Additional prefixes can be
    supplied via extra_prefixes for models that store MTP weights elsewhere
    (e.g. GLM-5 uses model.layers.78. for its next-token prediction head).
    """
    from safetensors.torch import load_file

    src_index_path = source_dir / "model.safetensors.index.json"
    if not src_index_path.exists():
        print("  No source index found, skipping MTP merge.")
        return shard_idx, total_tensors

    with open(src_index_path) as f:
        src_wm = json.load(f)["weight_map"]

    all_prefixes = ["mtp."] + (extra_prefixes or [])
    mtp_keys = [k for k in src_wm if any(k.startswith(p) for p in all_prefixes)]
    # Don't re-merge keys already exported by the main loop.
    mtp_keys = [k for k in mtp_keys if k not in weight_map]
    if not mtp_keys:
        print("  No MTP keys in source checkpoint, skipping.")
        return shard_idx, total_tensors

    src_shards_needed = sorted(set(src_wm[k] for k in mtp_keys))
    print(f"\nMerging {len(mtp_keys)} MTP weights from {len(src_shards_needed)} source shard(s)...")

    mtp_shard: dict[str, torch.Tensor] = {}
    mtp_size = 0
    for src_shard_name in src_shards_needed:
        src_data = load_file(str(source_dir / src_shard_name))
        for key in mtp_keys:
            if src_wm[key] == src_shard_name and key in src_data:
                t = src_data[key].contiguous()
                mtp_shard[key] = t
                mtp_size += _tensor_size(t)
                total_tensors += 1

    if mtp_shard:
        mtp_path = export_dir / MTP_SHARD_NAME
        if mtp_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing MTP shard: {mtp_path}")
        save_file(mtp_shard, str(mtp_path))
        for key in mtp_shard:
            weight_map[key] = MTP_SHARD_NAME
        print(
            f"  Wrote {MTP_SHARD_NAME}: {len(mtp_shard)} tensors "
            f"({mtp_size / 1e9:.2f} GB)"
        )
        mtp_shard.clear()

    return shard_idx, total_tensors


def _postprocess_shards(
    export_dir: Path,
    weight_map: dict[str, str],
    kv_cache_max_bound: float,
    kv_cache_format: str | None,
) -> None:
    """Apply postprocess_state_dict logic to already-written shards.

    Re-reads each shard, filters/renames keys, and rewrites if changed.
    """
    from safetensors.torch import load_file

    shard_files = set(weight_map.values())
    new_weight_map: dict[str, str] = {}

    for shard_file in sorted(shard_files):
        path = export_dir / shard_file
        shard_data = load_file(str(path))

        processed = postprocess_state_dict(
            shard_data, kv_cache_max_bound, kv_cache_format
        )

        # Drop dummy KV cache scales/biases from modelopt.
        # These are all 1.0 when only quantizing expert MLPs (not KV cache),
        # and the key naming convention differs between sglang and vLLM,
        # causing spurious warnings in both frameworks.
        processed = {
            k: v for k, v in processed.items()
            if not any(s in k for s in (".k_scale", ".v_scale", ".k_bias", ".v_bias"))
        }

        if set(processed.keys()) != set(shard_data.keys()):
            save_file(processed, str(path))
            print(f"  Postprocessed {shard_file}: "
                  f"{len(shard_data)} -> {len(processed)} keys")

        for key in processed:
            new_weight_map[key] = shard_file

    weight_map.clear()
    weight_map.update(new_weight_map)


def _rename_shards(
    export_dir: Path,
    weight_map: dict[str, str],
    total_shards: int,
) -> None:
    """Rename shards to the standard HF format: model-00001-of-NNNNN.safetensors."""
    del total_shards
    old_names = sorted(
        name
        for name in set(weight_map.values())
        if name not in {SCALE_SHARD_NAME, MTP_SHARD_NAME}
    )
    total_shards = len(old_names)
    rename_map: dict[str, str] = {}

    for i, old_name in enumerate(old_names, 1):
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old_name != new_name:
            (export_dir / old_name).rename(export_dir / new_name)
            rename_map[old_name] = new_name

    # Update weight_map with final names.
    for key in list(weight_map.keys()):
        old_shard = weight_map[key]
        if old_shard in rename_map:
            weight_map[key] = rename_map[old_shard]

    # Write the index file.
    index = {
        "metadata": {"total_size": sum(
            (export_dir / f).stat().st_size
            for f in set(weight_map.values())
        )},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(export_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Wrote model.safetensors.index.json ({len(weight_map)} entries)")


def _save_model_metadata(
    model: nn.Module,
    export_dir: Path,
    hf_quant_config: dict | None,
    preserve_remote_code: bool = False,
    source_dir: Path | None = None,
) -> None:
    """Save config.json (with quantization_config), generation_config, and tokenizer files."""
    # Save config.json from the model's config object.
    config = model.config
    config_dict = config.to_dict()

    # Strip auto_map for models already in upstream transformers. Remote-code
    # models such as MiMo need these entries plus the copied Python files.
    if not preserve_remote_code:
        config_dict.pop("auto_map", None)

    # Ensure architectures is set. Nested text_config may have None; derive
    # from the outer model class (e.g. Qwen3_5MoeForConditionalGeneration).
    if not config_dict.get("architectures"):
        config_dict["architectures"] = [type(model).__name__]

    if hf_quant_config is not None:
        config_dict["quantization_config"] = hf_quant_config

    with open(export_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # Save hf_quant_config.json for backward compatibility.
    if hf_quant_config is not None:
        with open(export_dir / "hf_quant_config.json", "w") as f:
            json.dump(hf_quant_config, f, indent=4)

    # Save generation_config if present.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        gen_config = model.generation_config.to_dict()
        with open(export_dir / "generation_config.json", "w") as f:
            json.dump(gen_config, f, indent=4)

    # Copy tokenizer and chat template files from the source model.
    import shutil
    if source_dir is None:
        source_dir = _resolve_source_dir(model, None)
    copy_patterns = [
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "special_tokens_map.json", "tokenizer.model", "added_tokens.json",
        "processor_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json",
        "merges.txt", "vocab.json",
    ]
    if preserve_remote_code:
        copy_patterns.extend([
            "configuration_mimo_v2.py", "modeling_mimo_v2.py",
        ])
    copied = []
    for name in copy_patterns:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)
            copied.append(name)

    if preserve_remote_code:
        for src in sorted(source_dir.glob("*.py")):
            if src.name in copied:
                continue
            shutil.copy2(src, export_dir / src.name)
            copied.append(src.name)

    if preserve_remote_code:
        for dirname in ("audio_tokenizer",):
            src = source_dir / dirname
            if src.exists():
                dst = export_dir / dirname
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                copied.append(dirname)

    print(f"  Wrote config.json, generation_config.json, and copied: {', '.join(copied)}")
