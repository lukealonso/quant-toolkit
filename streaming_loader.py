"""Streaming model loader for large MoE models.

Replaces accelerate's device_map="auto" with a simple, predictable system:
- GPU 0 is the sole execution device; it may optionally store overflow layers
  within an explicit capacity budget.
- GPUs 1-N and CPU are dumb weight storage
- Disk-resident layers load directly from safetensors on demand (no re-serialization)

Every decoder layer runs on GPU 0 via forward hooks that copy weights in,
execute, then free. Activations stay on GPU 0 throughout.
"""

import gc
import json
import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _parse_gib(s):
    s = s.strip()
    for suffix in ("GiB", "gib"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)])
    for suffix in ("MiB", "mib"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) / 1024
    for suffix in ("GB", "gb"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * 1000**3 / 1024**3
    for suffix in ("MB", "mb"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * 1000**2 / 1024**3
    return float(s)


def _extract_layer_idx(key):
    """Extract layer index from a parameter key containing 'layers.N.'."""
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _detect_layer_prefix(weight_map):
    """Detect the key prefix before 'layers.N.' from checkpoint keys.

    Returns e.g. 'model.layers.' or 'model.language_model.layers.' depending
    on whether the checkpoint is a text-only or VL model.
    """
    candidates = defaultdict(set)
    for key in weight_map:
        m = re.match(r"^(.*?layers\.)\d+\.", key)
        if m:
            candidates[m.group(1)].add(int(key[m.end(1):].split(".", 1)[0]))
    if candidates:
        # Some multimodal side towers also contain modules named layers.N
        # (MiMo audio does this). The main decoder has by far the largest
        # layer-index set, so use that as the streaming layer prefix.
        return max(candidates.items(), key=lambda item: len(item[1]))[0]
    return "model.layers."


def _checkpoint_key_to_model_key(key: str) -> str:
    """Apply the checkpoint-to-model key renames needed by streaming loads.

    Transformers handles these through conversion_mapping.py during normal
    from_pretrained loads. The streaming loader reads safetensors directly, so
    it needs the relevant MiniMax M3-VL conversion rules locally.
    """
    if key.startswith("language_model.lm_head"):
        key = "lm_head" + key[len("language_model.lm_head"):]
    elif key.startswith("language_model.model."):
        key = "model.language_model." + key[len("language_model.model."):]
    elif key.startswith("vision_tower.vision_model.embeddings.patch_embedding."):
        key = "model.vision_tower.embeddings.proj." + key[
            len("vision_tower.vision_model.embeddings.patch_embedding.") :
        ]
    elif key.startswith("vision_tower.vision_model.encoder.layers."):
        key = "model.vision_tower.layers." + key[len("vision_tower.vision_model.encoder.layers.") :]
    elif key.startswith("vision_tower.vision_model.pre_layrnorm."):
        key = "model.vision_tower.pre_layrnorm." + key[
            len("vision_tower.vision_model.pre_layrnorm.") :
        ]
    elif key.startswith("multi_modal_projector.linear_1."):
        key = "model.multi_modal_projector.linear_1." + key[len("multi_modal_projector.linear_1.") :]
    elif key.startswith("multi_modal_projector.linear_2."):
        key = "model.multi_modal_projector.linear_2." + key[len("multi_modal_projector.linear_2.") :]
    elif key.startswith("patch_merge_mlp.linear_1."):
        key = "model.multi_modal_projector.merge_linear_1." + key[len("patch_merge_mlp.linear_1.") :]
    elif key.startswith("patch_merge_mlp.linear_2."):
        key = "model.multi_modal_projector.merge_linear_2." + key[len("patch_merge_mlp.linear_2.") :]

    key = re.sub(
        r"\.language_model\.layers\.(\d+)\.block_sparse_moe\.experts\.",
        r".language_model.layers.\1.mlp.experts.",
        key,
    )
    key = re.sub(
        r"\.language_model\.layers\.(\d+)\.block_sparse_moe\.shared_experts\.",
        r".language_model.layers.\1.mlp.shared_experts.",
        key,
    )
    key = re.sub(
        r"\.language_model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$",
        r".language_model.layers.\1.mlp.gate.weight",
        key,
    )
    key = re.sub(
        r"\.language_model\.layers\.(\d+)\.block_sparse_moe\.e_score_correction_bias$",
        r".language_model.layers.\1.mlp.gate.e_score_correction_bias",
        key,
    )

    key = key.replace(".self_attn.index_q_proj.", ".self_attn.indexer.q_proj.")
    key = key.replace(".self_attn.index_k_proj.", ".self_attn.indexer.k_proj.")
    key = key.replace(".self_attn.index_q_norm.", ".self_attn.indexer.q_norm.")
    key = key.replace(".self_attn.index_k_norm.", ".self_attn.indexer.k_norm.")

    # GLM-5.3 (transformers model_type=glm5_next) checkpoint conversions.
    # Keep these byte-for-byte equivalent to the pinned Transformers
    # conversion_mapping.py rules. The streaming loader bypasses
    # from_pretrained(), so otherwise these tensors remain on meta.
    key = re.sub(r"\.self_attn\.f_a_proj\.", ".self_attn.forget_gate.f_a_proj.", key)
    key = re.sub(r"\.self_attn\.f_b_proj\.", ".self_attn.forget_gate.f_b_proj.", key)
    key = re.sub(r"\.self_attn\.dt_bias$", ".self_attn.forget_gate.dt_bias", key)
    key = re.sub(r"\.self_attn\.A_log$", ".self_attn.forget_gate.A_log", key)
    key = re.sub(r"\.hc_attn_fn$", ".attn_hc.fn", key)
    key = re.sub(r"\.hc_attn_base$", ".attn_hc.base", key)
    key = re.sub(r"\.hc_attn_scale$", ".attn_hc.scale", key)
    key = re.sub(r"\.hc_ffn_fn$", ".ffn_hc.fn", key)
    key = re.sub(r"\.hc_ffn_base$", ".ffn_hc.base", key)
    key = re.sub(r"\.hc_ffn_scale$", ".ffn_hc.scale", key)

    key = re.sub(r"\.mlp\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
    key = re.sub(r"\.mlp\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
    key = re.sub(r"\.mlp\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
    return key


def _dense_gate_up_target_key(key: str) -> tuple[str, str] | None:
    """Return (target_key, part) for dense gate/up weights that need concat."""
    m = re.match(r"^(.*\.mlp(?:\.shared_experts)?)\.(gate_proj|up_proj)\.weight$", key)
    if not m:
        return None
    part = "gate" if m.group(2) == "gate_proj" else "up"
    return f"{m.group(1)}.gate_up_proj.weight", part


def _glm5_next_conv_target_key(key: str) -> tuple[str, str] | None:
    """Return the fused GLM-5.3 conv1d target and q/k/v source part."""
    m = re.match(r"^(.*\.self_attn)\.([qkv])_conv1d\.weight$", key)
    if not m:
        return None
    return f"{m.group(1)}.conv1d.weight", m.group(2)


def _expert_first_to_projection_first_key(key: str) -> str | None:
    m = re.match(
        r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
        key,
    )
    if not m:
        return None
    prefix, idx, proj = m.group(1), m.group(2), m.group(3)
    return f"{prefix}.{proj}.{idx}.weight"


def _load_checkpoint_tensors(snapshot_dir, weight_map, raw_keys, device):
    tensors = {}
    by_shard = defaultdict(list)
    for raw_key in raw_keys:
        by_shard[weight_map[raw_key]].append(raw_key)
    for shard_file, keys in by_shard.items():
        shard_path = os.path.join(snapshot_dir, shard_file)
        with safe_open(shard_path, framework="pt", device=str(device)) as f:
            available = set(f.keys())
            for raw_key in keys:
                if raw_key in available:
                    tensors[raw_key] = f.get_tensor(raw_key)
    return tensors


def _resolve_snapshot_dir(model_id):
    """Resolve a HF model ID to the local snapshot directory."""
    if os.path.isdir(model_id):
        return model_id
    # Standard HF cache layout
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    snapshots = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snapshots):
        # Use the most recent snapshot
        entries = sorted(os.listdir(snapshots))
        if entries:
            return os.path.join(snapshots, entries[-1])
    raise FileNotFoundError(
        f"Cannot resolve snapshot dir for {model_id}. "
        f"Pass a local path or ensure the model is cached."
    )


def _iter_safetensors_files(snapshot_dir):
    for name in sorted(os.listdir(snapshot_dir)):
        if name.endswith(".safetensors") and name != "model_mtp.safetensors":
            yield name


def _build_weight_map_from_shards(snapshot_dir):
    """Build a weight map from safetensors metadata.

    Some repos publish an index whose shard names differ from the actual files
    in cache. Scanning safetensors keys is metadata-only and avoids depending
    on those stale shard filenames.
    """
    weight_map = {}
    for shard_file in _iter_safetensors_files(snapshot_dir):
        shard_path = os.path.join(snapshot_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard_file
    return weight_map


def _device_sort_key(device):
    device = str(device)
    if device.startswith("cuda:"):
        return (0, int(device.split(":", 1)[1]))
    if device == "cpu":
        return (1, 0)
    if device == "meta":
        return (2, 0)
    return (3, device)


def _format_indices(indices):
    indices = sorted(indices)
    if not indices:
        return ""

    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


class StreamingModelLoader:
    """Load and manage a large model with streaming execution on GPU 0."""

    def __init__(
        self,
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_capacity_gib=None,
        gpu0_storage_capacity_gib=0,
        cpu_capacity_gib=200,
        host_staged_device_moves=False,
        attention_implementation=None,
        cuda_staging_device=None,
        cuda_staging_source_device=None,
        cuda_staging_reserve_gib=0,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.gpu0_storage_capacity_gib = gpu0_storage_capacity_gib
        self.cpu_capacity_gib = cpu_capacity_gib
        self.host_staged_device_moves = host_staged_device_moves
        self.attention_implementation = attention_implementation
        self.num_gpus = torch.cuda.device_count()
        self.cuda_staging_device = self._normalize_cuda_staging_device(cuda_staging_device)
        self.cuda_staging_source_device = self._normalize_cuda_staging_source_device(cuda_staging_source_device)
        self.cuda_staging_reserve_gib = cuda_staging_reserve_gib

        if gpu_capacity_gib is None:
            # Auto-detect from first GPU
            self.gpu_capacity_gib = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
        else:
            self.gpu_capacity_gib = gpu_capacity_gib

        self.snapshot_dir = _resolve_snapshot_dir(model_id)
        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map = index["weight_map"]
        missing_shards = {
            shard
            for shard in set(self.weight_map.values())
            if not os.path.exists(os.path.join(self.snapshot_dir, shard))
        }
        if missing_shards:
            print(
                "  Rebuilding safetensors weight map from local shards "
                f"({len(missing_shards)} indexed shard names missing)"
            )
            self.weight_map = _build_weight_map_from_shards(self.snapshot_dir)
        self._layer_prefix = _detect_layer_prefix(self.weight_map)

        # Will be populated during load_model
        self.storage_map = {}  # layer_idx -> device string
        self.layer_tensor_sizes = {}  # layer_idx -> {relative tensor name -> bytes}
        self.layer_tensor_storage = {}  # layer_idx -> {relative tensor name -> device string}
        self.layer_split_primary_device = {}  # layer_idx -> fallback storage device
        self._hook_handles = []
        self._model_state_keys = None

    def _normalize_cuda_staging_device(self, device):
        if device is None:
            return None
        staging = torch.device(device)
        if staging.type != "cuda" or staging.index is None:
            raise ValueError(f"cuda_staging_device must be explicit CUDA device like 'cuda:1', got {device!r}")
        if staging.index == 0:
            raise ValueError("cuda_staging_device must not be cuda:0; GPU 0 is the streaming execution device")
        if staging.index >= self.num_gpus:
            raise ValueError(
                f"cuda_staging_device {staging} is outside the visible CUDA device range "
                f"0-{self.num_gpus - 1}"
            )
        return staging

    def _normalize_cuda_staging_source_device(self, device):
        if device is None:
            return None
        source = torch.device(device)
        if source.type != "cuda" or source.index is None:
            raise ValueError(f"cuda_staging_source_device must be explicit CUDA device like 'cuda:9', got {device!r}")
        if source.index == 0:
            raise ValueError("cuda_staging_source_device must not be cuda:0; GPU 0 is the streaming execution device")
        if source.index >= self.num_gpus:
            raise ValueError(
                f"cuda_staging_source_device {source} is outside the visible CUDA device range "
                f"0-{self.num_gpus - 1}"
            )
        if self.cuda_staging_device is not None and source == self.cuda_staging_device:
            raise ValueError("cuda_staging_source_device must differ from cuda_staging_device")
        return source

    def load_model(self, register_moe_fn=None, model_cls=None):
        """Load the model with streaming hooks. Returns the model.

        Args:
            register_moe_fn: Optional callable to register MoE quantization
                patches before model creation (e.g. register_glm5_moe_for_quantization).
            model_cls: Optional explicit model class (e.g. Qwen3_5MoeForCausalLM).
                When provided and the config is a VL composite with text_config,
                the text_config is extracted automatically.
        """
        if register_moe_fn is not None:
            register_moe_fn()

        cls = model_cls or AutoModelForCausalLM
        print(f"Loading config from {self.model_id}...")
        if cls is AutoModelForCausalLM:
            config = AutoConfig.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
        else:
            config = cls.config_class.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
        if self.attention_implementation is not None:
            if self.attention_implementation == "minimax_m3_flex":
                from models.minimax_m3_flex import register_minimax_m3_flex_attention

                register_minimax_m3_flex_attention()
            config._attn_implementation = self.attention_implementation

        print(f"Creating model on meta device via {cls.__name__}...")
        config_kwargs = {}
        if self.attention_implementation is not None:
            config_kwargs["attn_implementation"] = self.attention_implementation
        if cls is AutoModelForCausalLM:
            with init_empty_weights():
                model = cls.from_config(
                    config,
                    torch_dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code,
                    **config_kwargs,
                )
        else:
            with init_empty_weights():
                model = cls._from_config(config, dtype=self.dtype, **config_kwargs)

        # Compute per-layer sizes from meta model
        self.layer_tensor_sizes = self._compute_layer_tensor_sizes(model)
        layer_sizes = {
            layer_idx: sum(tensor_sizes.values())
            for layer_idx, tensor_sizes in self.layer_tensor_sizes.items()
        }
        self.storage_map = self._compute_storage_map(layer_sizes)
        self._print_storage_summary(layer_sizes)

        # Materialize permanent modules on GPU 0
        print("\nLoading permanent modules to GPU 0 (embed, norm, lm_head)...")
        self._materialize_permanent_modules(model)

        # Materialize GPU/CPU-assigned layers
        print("Loading layers to storage devices...")
        for layer_idx, device in sorted(self.storage_map.items()):
            if device == "meta":
                continue
            print(f"  Layer {layer_idx} -> {device}")
            if device == "split":
                self._materialize_layer_split(model, layer_idx)
            else:
                self._materialize_layer(model, layer_idx, device)

        meta_count = sum(1 for d in self.storage_map.values() if d == "meta")
        if meta_count:
            print(f"  {meta_count} layers remain on meta (will load from safetensors on demand)")

        # Monkey-patch _QuantFusedExperts._setup for lazy meta handling
        self._patch_quant_fused_experts_setup()

        # Install streaming hooks
        self._install_hooks(model)

        # Report VRAM usage
        print(f"\nVRAM after loading:")
        for i in range(self.num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {alloc:.1f} / {total:.1f} GiB ({total - alloc:.1f} GiB free)")

        return model

    @staticmethod
    def _get_layers(model):
        """Find the decoder layer list, handling VL model nesting."""
        m = model.model
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            return m.language_model.layers
        raise AttributeError(
            f"Cannot find decoder layers on {type(m).__name__}. "
            f"Expected .layers or .language_model.layers"
        )

    def _compute_layer_tensor_sizes(self, model):
        """Compute tensor storage sizes in bytes for each decoder layer."""
        layer_tensor_sizes = {}
        for i, layer in enumerate(self._get_layers(model)):
            tensor_sizes = {}
            for name, p in layer.named_parameters():
                tensor_sizes[name] = p.numel() * p.element_size()
            for name, b in layer.named_buffers():
                tensor_sizes[name] = b.numel() * b.element_size()
            layer_tensor_sizes[i] = tensor_sizes
        return layer_tensor_sizes

    def _compute_storage_map(self, layer_sizes):
        """Assign layers to storage devices, splitting a layer when useful.

        Whole-layer placement is preferred because it is simpler and keeps most
        transfers contiguous. If a whole layer cannot fit on any single storage
        GPU, try placing its individual tensors into the aggregate slack across
        GPUs 1-N before spilling to GPU 0, CPU, or disk.
        """
        storage_map = {}
        self.layer_tensor_storage = {}
        self.layer_split_primary_device = {}
        # Reserve headroom per storage GPU for CUDA context, driver memory,
        # and PyTorch allocator fragmentation from loading many small tensors.
        gpu_headroom = 0
        default_gpu_capacity = self.gpu_capacity_gib * 1024**3 - gpu_headroom
        gpu0_capacity = self.gpu0_storage_capacity_gib * 1024**3
        cpu_capacity = self.cpu_capacity_gib * 1024**3
        gpu_capacities = {
            gpu_idx: self._storage_gpu_capacity_bytes(gpu_idx, default_gpu_capacity)
            for gpu_idx in range(1, self.num_gpus)
        }

        # GPU 0 is reserved for execution. Pack GPUs 1-N first, then use the
        # explicit GPU-0 overflow budget if one was provided.
        current_gpu = 1
        gpu_used = {gpu_idx: 0 for gpu_idx in range(1, self.num_gpus)}
        gpu0_used = 0
        cpu_used = 0

        for i in sorted(layer_sizes.keys()):
            size = layer_sizes[i]

            placed = False
            while current_gpu < self.num_gpus:
                gpu_capacity = gpu_capacities[current_gpu]
                if gpu_used[current_gpu] + size <= gpu_capacity:
                    storage_map[i] = f"cuda:{current_gpu}"
                    gpu_used[current_gpu] += size
                    placed = True
                    break
                else:
                    current_gpu += 1

            if not placed:
                split_storage = self._try_split_layer_across_storage_gpus(
                    i, gpu_used, gpu_capacities
                )
                if split_storage is not None:
                    storage_map[i] = "split"
                    self.layer_tensor_storage[i] = split_storage
                    self.layer_split_primary_device[i] = self._primary_split_device(i)
                    placed = True

            if not placed:
                if gpu0_used + size <= gpu0_capacity:
                    storage_map[i] = "cuda:0"
                    gpu0_used += size
                    placed = True

            if not placed:
                if cpu_used + size <= cpu_capacity:
                    storage_map[i] = "cpu"
                    cpu_used += size
                else:
                    storage_map[i] = "meta"

        return storage_map

    def _storage_gpu_capacity_bytes(self, gpu_idx, default_capacity):
        if (
            self.cuda_staging_device is not None
            and gpu_idx == self.cuda_staging_device.index
        ):
            return max(
                0,
                default_capacity - self.cuda_staging_reserve_gib * 1024**3,
            )
        return default_capacity

    def _try_split_layer_across_storage_gpus(self, layer_idx, gpu_used, gpu_capacities):
        """Balanced tensor placement into existing storage GPU slack.

        Large tensors start on the least-used viable GPUs to avoid creating a
        near-OOM storage device. Small tensors then prefer devices already used
        by this layer, keeping the runtime copy fan-in low when possible.
        """
        tensor_sizes = self.layer_tensor_sizes.get(layer_idx, {})
        if not tensor_sizes or self.num_gpus <= 1:
            return None

        trial_used = dict(gpu_used)
        placement = {}
        layer_gpus = []
        for name, size in sorted(tensor_sizes.items(), key=lambda item: item[1], reverse=True):
            candidates = []
            if layer_gpus:
                candidates = [
                    gpu_idx
                    for gpu_idx in layer_gpus
                    if gpu_capacities[gpu_idx] - trial_used[gpu_idx] >= size
                ]
            if not candidates:
                candidates = [
                    gpu_idx
                    for gpu_idx in range(1, self.num_gpus)
                    if gpu_capacities[gpu_idx] - trial_used[gpu_idx] >= size
                ]
            if not candidates:
                return None

            if any(gpu_idx in layer_gpus for gpu_idx in candidates):
                # Pack later/smaller tensors into this layer's existing shards.
                best_gpu = min(
                    candidates,
                    key=lambda gpu_idx: gpu_capacities[gpu_idx] - trial_used[gpu_idx] - size,
                )
            else:
                # Balance new large shards across the storage GPUs.
                best_gpu = max(
                    candidates,
                    key=lambda gpu_idx: gpu_capacities[gpu_idx] - trial_used[gpu_idx],
                )
            if best_gpu is None:
                return None
            if best_gpu not in layer_gpus:
                layer_gpus.append(best_gpu)
            placement[name] = f"cuda:{best_gpu}"
            trial_used[best_gpu] += size

        gpu_used.update(trial_used)
        return placement

    def _primary_split_device(self, layer_idx):
        by_device = defaultdict(int)
        for name, device in self.layer_tensor_storage.get(layer_idx, {}).items():
            by_device[device] += self.layer_tensor_sizes.get(layer_idx, {}).get(name, 0)
        if not by_device:
            return "cpu"
        return max(by_device.items(), key=lambda item: item[1])[0]

    def _print_storage_summary(self, layer_sizes):
        """Print a summary of layer placement."""
        by_device_layers = defaultdict(list)
        by_device_split_layers = defaultdict(set)
        by_device_bytes = defaultdict(int)
        split_layers = []
        for idx, device in sorted(self.storage_map.items()):
            if device == "split":
                split_layers.append(idx)
                for name, tensor_device in self.layer_tensor_storage.get(idx, {}).items():
                    by_device_split_layers[tensor_device].add(idx)
                    by_device_bytes[tensor_device] += self.layer_tensor_sizes[idx].get(name, 0)
                continue

            by_device_layers[device].append(idx)
            by_device_bytes[device] += layer_sizes[idx]

        print(f"\nStorage map ({len(self.storage_map)} layers):")
        all_devices = sorted(by_device_bytes.keys(), key=_device_sort_key)
        for device in all_devices:
            parts = []
            whole = by_device_layers.get(device, [])
            split = sorted(by_device_split_layers.get(device, []))
            if whole:
                parts.append(f"layers [{_format_indices(whole)}]")
            if split:
                parts.append(f"split [{_format_indices(split)}]")
            total_gib = by_device_bytes[device] / 1024**3
            print(f"  {device}: {', '.join(parts)} ({total_gib:.1f} GiB)")

        if split_layers:
            print("  split layer details:")
            for idx in split_layers:
                by_device = defaultdict(int)
                for name, device in self.layer_tensor_storage.get(idx, {}).items():
                    by_device[device] += self.layer_tensor_sizes[idx].get(name, 0)
                parts = [
                    f"{device} {size / 1024**3:.1f} GiB"
                    for device, size in sorted(by_device.items(), key=lambda item: _device_sort_key(item[0]))
                ]
                print(f"    layer {idx}: {', '.join(parts)}")

    def _get_layer_param_keys(self, layer_idx):
        """Get all checkpoint keys belonging to a specific layer."""
        prefix = f"{self._layer_prefix}{layer_idx}."
        return [k for k in self.weight_map if k.startswith(prefix)]

    def _get_permanent_param_keys(self):
        """Get checkpoint keys for non-layer modules (embed, norm, lm_head)."""
        decoder_layer_re = re.compile(rf"^{re.escape(self._layer_prefix)}\d+\.")
        return [k for k in self.weight_map if not decoder_layer_re.match(k)]

    def _materialize_permanent_modules(self, model):
        """Load embed_tokens, norm, rotary_emb, lm_head to GPU 0.

        Checkpoint keys that don't exist on the model (e.g. MTP weights in
        VL checkpoints) are silently skipped.
        """
        model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
        all_keys = self._get_permanent_param_keys()
        keys = all_keys
        direct_or_mapped = sum(
            1
            for k in all_keys
            if k in model_keys or _checkpoint_key_to_model_key(k) in model_keys
        )
        skipped = len(all_keys) - direct_or_mapped
        if skipped:
            print(f"  Skipping {skipped} checkpoint keys not in model (e.g. MTP)")
        self._materialize_params(model, keys, "cuda:0")

        # .to(cuda:0) on all non-layer submodules to catch registered buffers
        # created during __init__ (rotary inv_freq, vision pos embeds, etc.)
        # that aren't in the checkpoint and are still on meta/cpu.
        m = model.model if hasattr(model, "model") else model
        for name, child in m.named_children():
            if name == "layers":
                continue
            if hasattr(m, "language_model") and name == "language_model":
                for lm_name, lm_child in child.named_children():
                    if lm_name != "layers":
                        _zero_materialize_meta_tensors(
                            lm_child, "cuda:0", f"{name}.{lm_name}"
                        )
                        lm_child.to("cuda:0")
                continue
            _zero_materialize_meta_tensors(child, "cuda:0", name)
            child.to("cuda:0")
        for name, child in model.named_children():
            if name == "model":
                continue
            _zero_materialize_meta_tensors(child, "cuda:0", name)
            child.to("cuda:0")

    def _materialize_layer(self, model, layer_idx, device):
        """Load all parameters for a layer from safetensors to the given device."""
        keys = self._get_layer_param_keys(layer_idx)
        self._materialize_params(model, keys, device)

    def _materialize_layer_split(self, model, layer_idx):
        """Load one layer with individual tensors placed on storage GPUs."""
        keys = self._get_layer_param_keys(layer_idx)
        raw_prefix = f"{self._layer_prefix}{layer_idx}."
        model_prefix = _checkpoint_key_to_model_key(raw_prefix)
        target_devices_by_key = {
            f"{model_prefix}{relative_name}": device
            for relative_name, device in self.layer_tensor_storage[layer_idx].items()
        }
        self._materialize_params(
            model,
            keys,
            "cpu",
            target_devices_by_key=target_devices_by_key,
        )

    def _materialize_params(self, model, param_keys, device, target_devices_by_key=None):
        """Load parameters from safetensors and place into model.

        Handles the checkpoint-to-model key mismatch for MoE experts:
        checkpoint has per-expert keys (experts.{i}.gate_proj.weight) that must
        be fused into 3D parameters (experts.gate_up_proj, experts.down_proj).
        """
        target_devices_by_key = target_devices_by_key or {}

        def target_device_for(model_key):
            return target_devices_by_key.get(model_key, device)

        model_keys = self._get_model_state_keys(model)

        # Separate expert keys from regular keys. Only fuse per-expert
        # checkpoint weights when the in-memory model actually has fused
        # parameters. MiMo already uses experts.{i}.{proj}.weight modules;
        # quantized streaming layers use projection-first experts.{proj}.{i}.
        expert_pattern = re.compile(
            r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        regular_entries = []
        dense_gate_up_groups = defaultdict(dict)
        glm_conv_groups = defaultdict(dict)
        # expert_groups[prefix] = {expert_idx: {proj_name: checkpoint_key}}
        expert_groups = defaultdict(lambda: defaultdict(dict))

        for raw_key in param_keys:
            key = _checkpoint_key_to_model_key(raw_key)

            dense_gate_up = _dense_gate_up_target_key(key)
            if dense_gate_up is not None:
                target_key, part = dense_gate_up
                if target_key in model_keys:
                    dense_gate_up_groups[target_key][part] = raw_key
                    continue

            glm_conv = _glm5_next_conv_target_key(key)
            if glm_conv is not None:
                target_key, part = glm_conv
                if target_key in model_keys:
                    glm_conv_groups[target_key][part] = raw_key
                    continue

            m = expert_pattern.match(key)
            projection_first_key = _expert_first_to_projection_first_key(key)
            if key not in model_keys and not m and projection_first_key not in model_keys:
                continue
            if m:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                if (
                    f"{prefix}.gate_up_proj" in model_keys
                    or f"{prefix}.down_proj" in model_keys
                ):
                    expert_groups[prefix][idx][proj] = raw_key
                elif projection_first_key in model_keys:
                    regular_entries.append((raw_key, projection_first_key))
                elif key in model_keys:
                    regular_entries.append((raw_key, key))
            else:
                regular_entries.append((raw_key, key))

        # Load regular (non-expert) keys directly
        by_shard = defaultdict(list)
        for raw_key, model_key in regular_entries:
            by_shard[self.weight_map[raw_key]].append((raw_key, model_key))

        for shard_file, entries in by_shard.items():
            shard_path = os.path.join(self.snapshot_dir, shard_file)
            load_device = "cpu" if target_devices_by_key else str(device)
            with safe_open(shard_path, framework="pt", device=load_device) as f:
                available = set(f.keys())
                for raw_key, model_key in entries:
                    if raw_key in available:
                        tensor = f.get_tensor(raw_key)
                        target_device = target_device_for(model_key)
                        set_module_tensor_to_device(
                            model,
                            model_key,
                            target_device,
                            value=tensor,
                            dtype=tensor.dtype,
                        )

        # Fuse dense MiniMax M3 gate/up weights into gate_up_proj.
        for target_key, parts in dense_gate_up_groups.items():
            if "gate" not in parts or "up" not in parts:
                raise RuntimeError(
                    f"Missing dense gate/up source weight for {target_key}: {sorted(parts)}"
                )
            tensors = _load_checkpoint_tensors(
                self.snapshot_dir,
                self.weight_map,
                [parts["gate"], parts["up"]],
                "cpu",
            )
            gate = tensors[parts["gate"]]
            up = tensors[parts["up"]]
            gate_up = torch.cat([gate, up], dim=0)
            target_device = target_device_for(target_key)
            set_module_tensor_to_device(
                model,
                target_key,
                target_device,
                value=gate_up,
                dtype=gate_up.dtype,
            )
            del gate, up, gate_up, tensors

        # Fuse GLM-5.3 KDA q/k/v depthwise convolution weights in the same
        # q,k,v order used by Transformers' authoritative conversion mapping.
        for target_key, parts in glm_conv_groups.items():
            if set(parts) != {"q", "k", "v"}:
                raise RuntimeError(
                    f"Missing GLM-5.3 q/k/v conv source weight for {target_key}: "
                    f"{sorted(parts)}"
                )
            tensors = _load_checkpoint_tensors(
                self.snapshot_dir,
                self.weight_map,
                [parts[part] for part in ("q", "k", "v")],
                "cpu",
            )
            fused = torch.cat([tensors[parts[part]] for part in ("q", "k", "v")], dim=0)
            target_device = target_device_for(target_key)
            set_module_tensor_to_device(
                model,
                target_key,
                target_device,
                value=fused,
                dtype=fused.dtype,
            )
            del fused, tensors

        # Fuse per-expert keys into 3D parameters.
        # Always fuse on CPU to avoid massive temporary GPU memory spikes —
        # accumulating 768 individual expert tensors + stacked + fused can be
        # 4-5x the final layer size, causing OOM on the target GPU.
        for prefix, experts_dict in expert_groups.items():
            num_experts = max(experts_dict.keys()) + 1

            # Collect all expert shard files we'll need
            expert_shard_keys = defaultdict(list)
            for idx in range(num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    ck = experts_dict.get(idx, {}).get(proj)
                    if ck:
                        expert_shard_keys[self.weight_map[ck]].append((idx, proj, ck))

            # Load expert tensors to CPU regardless of target device
            gate_tensors = [None] * num_experts
            up_tensors = [None] * num_experts
            down_tensors = [None] * num_experts

            for shard_file, entries in expert_shard_keys.items():
                shard_path = os.path.join(self.snapshot_dir, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for idx, proj, ck in entries:
                        t = f.get_tensor(ck)
                        if proj == "gate_proj":
                            gate_tensors[idx] = t
                        elif proj == "up_proj":
                            up_tensors[idx] = t
                        else:
                            down_tensors[idx] = t

            # Fuse on CPU: gate [N,I,H] + up [N,I,H] -> gate_up [N,2I,H]
            gate_stacked = torch.stack(gate_tensors)  # [N, I, H]
            up_stacked = torch.stack(up_tensors)      # [N, I, H]
            gate_up = torch.cat([gate_stacked, up_stacked], dim=1)  # [N, 2I, H]
            del gate_stacked, up_stacked, gate_tensors, up_tensors

            # Move fused results to their target devices. Split storage may
            # place the large gate/up and down expert tensors on different GPUs.
            gate_up_key = f"{prefix}.gate_up_proj"
            gate_up_device = target_device_for(gate_up_key)
            set_module_tensor_to_device(
                model,
                gate_up_key,
                gate_up_device,
                value=gate_up,
                dtype=gate_up.dtype,
            )
            del gate_up

            # Fuse down on CPU: [N, H, I]
            down_stacked = torch.stack(down_tensors)
            del down_tensors
            down_key = f"{prefix}.down_proj"
            down_device = target_device_for(down_key)
            set_module_tensor_to_device(
                model,
                down_key,
                down_device,
                value=down_stacked,
                dtype=down_stacked.dtype,
            )
            del down_stacked
            gc.collect()

    def _patch_quant_fused_experts_setup(self):
        """Monkey-patch _QuantFusedExperts._setup to handle meta tensors lazily."""
        from moe_registry import _QuantFusedExperts

        if getattr(_QuantFusedExperts, "_streaming_patch_applied", False):
            return

        original_setup = _QuantFusedExperts._setup

        def patched_setup(self_expert):
            if self_expert.gate_up_proj.device == torch.device("meta"):
                # Meta device — create per-expert Linear structure on meta,
                # actual data will be loaded by the streaming hook on demand.
                I = self_expert.intermediate_dim
                H = self_expert.hidden_dim

                with init_empty_weights():
                    gate_proj = nn.ModuleList(
                        [nn.Linear(H, I, bias=False) for _ in range(self_expert.num_experts)]
                    )
                    up_proj = nn.ModuleList(
                        [nn.Linear(H, I, bias=False) for _ in range(self_expert.num_experts)]
                    )
                    down_proj = nn.ModuleList(
                        [nn.Linear(I, H, bias=False) for _ in range(self_expert.num_experts)]
                    )

                delattr(self_expert, "gate_up_proj")
                delattr(self_expert, "down_proj")
                self_expert.gate_proj = gate_proj
                self_expert.up_proj = up_proj
                self_expert.down_proj = down_proj
                self_expert._needs_lazy_unfuse = True
            else:
                original_setup(self_expert)
                self_expert._needs_lazy_unfuse = False

        _QuantFusedExperts._setup = patched_setup
        _QuantFusedExperts._streaming_patch_applied = True

    def _install_hooks(self, model):
        """Register streaming forward hooks on each decoder layer."""
        layers = self._get_layers(model)
        for i, layer in enumerate(layers):
            device = self.storage_map[i]
            hook = LayerStreamingHook(
                layer_idx=i,
                storage_device=device,
                loader=self,
            )
            h1 = layer.register_forward_pre_hook(hook.pre_forward)
            h2 = layer.register_forward_hook(hook.post_forward)
            self._hook_handles.extend([h1, h2])
        print(f"Installed streaming hooks on {len(layers)} layers")

    def remove_hooks(self):
        """Remove all streaming hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def prepare_export(self, model):
        """Prepare model for export by removing streaming hooks and installing
        export-mode materialization callbacks on meta layers."""
        self.remove_hooks()

        for i, layer in enumerate(self._get_layers(model)):
            device = self.storage_map[i]
            if device == "meta":
                # Attach a callback that export_hf will check before .to("cpu")
                layer._streaming_materialize = lambda mod, idx=i: (
                    self._materialize_layer_for_export(model, mod, idx)
                )

    def _materialize_layer_for_export(self, model, module, layer_idx):
        """Load a meta layer's weights for export (to CPU, with expert unfusing)."""
        del model
        self._materialize_layer_module(module, layer_idx, "cpu")
        _calibrate_meta_weight_amaxes(module)

    def _materialize_layer_module(self, module, layer_idx, device):
        """Load one decoder layer into an already-created module tree."""
        keys = self._get_layer_param_keys(layer_idx)
        raw_prefix = f"{self._layer_prefix}{layer_idx}."
        model_prefix = _checkpoint_key_to_model_key(raw_prefix)
        module_keys = set(dict(module.named_parameters()).keys()) | set(dict(module.named_buffers()).keys())

        regular_entries = []
        dense_gate_up_groups = defaultdict(dict)
        glm_conv_groups = defaultdict(dict)

        for raw_key in keys:
            model_key = _checkpoint_key_to_model_key(raw_key)
            if not model_key.startswith(model_prefix):
                continue

            dense_gate_up = _dense_gate_up_target_key(model_key)
            if dense_gate_up is not None:
                target_key, part = dense_gate_up
                if target_key.startswith(model_prefix):
                    target_relative = target_key[len(model_prefix):]
                    if target_relative in module_keys:
                        dense_gate_up_groups[target_relative][part] = raw_key
                        continue

            glm_conv = _glm5_next_conv_target_key(model_key)
            if glm_conv is not None:
                target_key, part = glm_conv
                if target_key.startswith(model_prefix):
                    target_relative = target_key[len(model_prefix):]
                    if target_relative in module_keys:
                        glm_conv_groups[target_relative][part] = raw_key
                        continue

            relative = model_key[len(model_prefix):]
            projection_first_key = _expert_first_to_projection_first_key(model_key)
            if projection_first_key is not None and projection_first_key.startswith(model_prefix):
                projection_first_relative = projection_first_key[len(model_prefix):]
                if projection_first_relative in module_keys:
                    relative = projection_first_relative

            if relative not in module_keys:
                continue
            regular_entries.append((raw_key, relative))

        by_shard = defaultdict(list)
        for raw_key, relative in regular_entries:
            by_shard[self.weight_map[raw_key]].append((raw_key, relative))

        for shard_file, entries in by_shard.items():
            shard_path = os.path.join(self.snapshot_dir, shard_file)
            with safe_open(shard_path, framework="pt", device=str(device)) as f:
                available = set(f.keys())
                for raw_key, relative in entries:
                    if raw_key not in available:
                        continue
                    tensor = f.get_tensor(raw_key)
                    _assign_tensor_to_module(module, relative, tensor, device)

        for target_relative, parts in dense_gate_up_groups.items():
            if "gate" not in parts or "up" not in parts:
                raise RuntimeError(
                    f"Missing dense gate/up source weight for {target_relative}: {sorted(parts)}"
                )
            tensors = _load_checkpoint_tensors(
                self.snapshot_dir,
                self.weight_map,
                [parts["gate"], parts["up"]],
                device,
            )
            gate_up = torch.cat([tensors[parts["gate"]], tensors[parts["up"]]], dim=0)
            _assign_tensor_to_module(module, target_relative, gate_up, device)
            del gate_up, tensors

        for target_relative, parts in glm_conv_groups.items():
            if set(parts) != {"q", "k", "v"}:
                raise RuntimeError(
                    f"Missing GLM-5.3 q/k/v conv source weight for {target_relative}: "
                    f"{sorted(parts)}"
                )
            tensors = _load_checkpoint_tensors(
                self.snapshot_dir,
                self.weight_map,
                [parts[part] for part in ("q", "k", "v")],
                device,
            )
            fused = torch.cat([tensors[parts[part]] for part in ("q", "k", "v")], dim=0)
            _assign_tensor_to_module(module, target_relative, fused, device)
            del fused, tensors

    def _get_model_state_keys(self, model):
        if self._model_state_keys is None:
            self._model_state_keys = (
                set(dict(model.named_parameters()).keys())
                | set(dict(model.named_buffers()).keys())
            )
        return self._model_state_keys


def _remap_expert_key(module, relative_key):
    """Remap checkpoint expert key to match module tree after _QuantFusedExperts._setup.

    Checkpoint format: mlp.experts.{idx}.{proj}.weight
    Module format:     mlp.experts.{proj}.{idx}.weight
    """
    m = re.match(r"(mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", relative_key)
    if m and _has_projection_first_experts(module):
        prefix, idx, proj = m.group(1), m.group(2), m.group(3)
        return f"{prefix}.{proj}.{idx}.weight"
    return relative_key


def _has_projection_first_experts(module):
    mlp = getattr(module, "mlp", None)
    experts = getattr(mlp, "experts", None)
    return (
        experts is not None
        and hasattr(experts, "gate_proj")
        and isinstance(getattr(experts, "gate_proj", None), nn.ModuleList)
    )


def _module_has_state_key(module, relative_key):
    return (
        relative_key in dict(module.named_parameters()).keys()
        or relative_key in dict(module.named_buffers()).keys()
    )


def _assign_tensor_to_module(module, relative_key, tensor, device):
    """Assign a tensor to a module parameter by dotted relative key.

    Keys should already be remapped via _remap_expert_key if needed.
    Walks the module tree to find the target parameter.
    """
    parts = relative_key.split(".")
    target = module
    for part in parts[:-1]:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)
    param_name = parts[-1]
    old = getattr(target, param_name)
    if isinstance(old, nn.Parameter):
        target._parameters[param_name] = nn.Parameter(
            tensor.to(device=device), requires_grad=False
        )
    else:
        setattr(target, param_name, tensor.to(device=device))


def _zero_materialize_meta_tensors(module, device, prefix=""):
    """Materialize missing permanent tensors before moving a meta-built module.

    Streaming construction creates the whole model under init_empty_weights().
    Checkpoint-backed tensors are assigned explicitly, but tensors absent from
    the checkpoint, such as MiMo's visual merger biases, remain on meta. Plain
    module.to(device) cannot copy those, so initialize only the still-meta
    tensors to zero and leave already-loaded checkpoint tensors untouched.
    """
    for name, param in list(module.named_parameters()):
        if param.device.type != "meta":
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name == "audio_encoder.input_local_transformer.embed_tokens.weight":
            # The upstream MiMo remote code explicitly ignores this missing
            # weight, and the audio encoder calls Qwen2Model via inputs_embeds.
            value = torch.zeros(param.shape, dtype=param.dtype, device=device)
            set_module_tensor_to_device(module, name, device, value=value)
            continue
        if not name.endswith(".bias") and name != "bias":
            raise RuntimeError(
                "Permanent module tensor is still on meta after checkpoint load: "
                f"{full_name}. Refusing to zero-initialize a non-bias parameter."
            )
        value = torch.zeros(param.shape, dtype=param.dtype, device=device)
        set_module_tensor_to_device(module, name, device, value=value)
    for name, buf in list(module.named_buffers()):
        if buf.device.type != "meta":
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        rotary_value = _init_rotary_buffer(module, name, device)
        if rotary_value is not None:
            set_module_tensor_to_device(module, name, device, value=rotary_value)
            continue
        if buf.numel() != 0:
            raise RuntimeError(
                "Permanent module buffer is still on meta after checkpoint load: "
                f"{full_name}. Add explicit initialization instead of silently "
                "zeroing it."
            )
        value = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
        set_module_tensor_to_device(module, name, device, value=value)


def _init_rotary_buffer(module, buffer_name, device):
    if not (
        buffer_name.endswith("inv_freq")
        or buffer_name.endswith("original_inv_freq")
    ):
        return None

    parts = buffer_name.split(".")
    target = module
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else getattr(target, part)

    if parts[-1] == "original_inv_freq":
        inv_freq = getattr(target, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.device.type != "meta":
            return inv_freq.detach().clone().to(device=device)

    if hasattr(target, "rope_init_fn") and hasattr(target, "config"):
        inv_freq, attention_scaling = target.rope_init_fn(target.config, device)
        if hasattr(target, "attention_scaling"):
            target.attention_scaling = attention_scaling
        return inv_freq.to(device=device)

    if hasattr(target, "compute_default_rope_parameters") and hasattr(target, "config"):
        inv_freq, attention_scaling = target.compute_default_rope_parameters(target.config, device)
        if hasattr(target, "attention_scaling"):
            target.attention_scaling = attention_scaling
        return inv_freq.to(device=device)

    old = getattr(target, parts[-1])
    if type(target).__name__ == "MiMoVisionRotaryEmbedding":
        dim = old.numel() * 2
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        return inv_freq

    return None


def _calibrate_meta_weight_amaxes(module):
    """Recompute weight amaxes that ModelOpt initialized from meta weights.

    Disk-backed layers are still meta when ModelOpt runs its global weight-only
    calibration pass.  Once their immutable weights are materialized, replace
    those placeholder amaxes using the same max-calibration routine ModelOpt
    uses for ordinary resident weights.
    """
    from modelopt.torch.quantization.model_calib import max_calibrate
    from modelopt.torch.quantization.nn import TensorQuantizer

    calibrated = 0
    for child in module.modules():
        quantizer = getattr(child, "weight_quantizer", None)
        weight = getattr(child, "weight", None)
        if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
            continue
        amax = getattr(quantizer, "_amax", None)
        if not isinstance(amax, torch.Tensor) or amax.device.type != "meta":
            continue
        if not isinstance(weight, torch.Tensor) or weight.device.type == "meta":
            raise RuntimeError(
                "Cannot calibrate a lazy weight amax without materialized weight data"
            )
        calibrator = getattr(quantizer, "_calibrator", None)
        if calibrator is None or not hasattr(calibrator, "reset"):
            raise RuntimeError("Lazy weight amax requires a resettable ModelOpt calibrator")
        calibrator.reset()
        # TensorQuantizer's setter copies into an existing buffer.  A copy into
        # a meta buffer remains meta, so unregister the placeholder first and
        # let load_calib_amax create a real buffer on the weight device.
        delattr(quantizer, "_amax")
        max_calibrate(
            quantizer,
            lambda current, value=weight: current(value),
            distributed_sync=False,
        )
        resolved = getattr(quantizer, "_amax", None)
        if not isinstance(resolved, torch.Tensor) or resolved.device.type == "meta":
            raise RuntimeError("Lazy weight amax remained meta after calibration")
        calibrated += 1
    return calibrated


class LayerStreamingHook:
    """Forward hooks that stream a decoder layer's weights to GPU 0 for execution."""

    def __init__(self, layer_idx, storage_device, loader):
        self.layer_idx = layer_idx
        self.storage_device = storage_device
        self.loader = loader

    def pre_forward(self, module, args):
        """Copy or load layer weights to GPU 0 before forward pass."""
        if self.storage_device == "meta":
            self._load_from_disk(module)
            calibrated = _calibrate_meta_weight_amaxes(module)
            if calibrated:
                print(
                    f"  Layer {self.layer_idx}: calibrated {calibrated} "
                    "lazy weight amax(es)"
                )
            # _load_from_disk only loads checkpoint weights. After mtq.quantize
            # attaches TensorQuantizer modules, their buffers (_amax, _pre_quant_scale,
            # etc.) live on CPU (preserved there by _unload_to_meta). We need to
            # move them to GPU 0 for the forward pass.
            self._move_quantizer_state(module, "cuda:0")
        else:
            self._copy_to_gpu0(module)

    def post_forward(self, module, args, output):
        """Move weights back to storage after forward pass."""
        if self.storage_device == "meta":
            self._unload_to_meta(module)
        else:
            self._copy_to_storage(module)

        # Ensure output is on GPU 0
        if isinstance(output, torch.Tensor) and output.device != torch.device("cuda:0"):
            return self._move_tensor(output, "cuda:0")
        if isinstance(output, tuple):
            return tuple(
                self._move_tensor(t, "cuda:0")
                if isinstance(t, torch.Tensor) and t.device != torch.device("cuda:0")
                else t
                for t in output
            )
        return output

    def _move_tensor(self, tensor, device, keep_intermediates=None):
        """Move tensors while controlling CUDA peer mappings.

        Large streaming copies between storage GPUs and GPU 0 can exhaust CUDA
        peer mapping resources on 10-GPU systems. The optional CUDA staging
        device acts as a trampoline only for the configured source GPU, so GPUs
        that can peer with GPU 0 still copy directly.
        """
        target = torch.device(device)
        if (
            self.loader.cuda_staging_device is not None
            and tensor.device.type == "cuda"
            and target.type == "cuda"
            and tensor.device != target
        ):
            staging = self.loader.cuda_staging_device
            source = self.loader.cuda_staging_source_device
            if source is not None:
                if tensor.device == source and target.index == 0:
                    staged = tensor.to(staging, non_blocking=True)
                    if keep_intermediates is not None:
                        keep_intermediates.append(staged)
                    return staged.to(target, non_blocking=True)
                if tensor.device.index == 0 and target == source:
                    staged = tensor.to(staging, non_blocking=True)
                    if keep_intermediates is not None:
                        keep_intermediates.append(staged)
                    return staged.to(target, non_blocking=True)
            elif tensor.device == staging or target == staging:
                return tensor.to(target, non_blocking=True)
            elif tensor.device.index == 0 or target.index == 0:
                staged = tensor.to(staging, non_blocking=True)
                if keep_intermediates is not None:
                    keep_intermediates.append(staged)
                return staged.to(target, non_blocking=True)
        if (
            self.loader.host_staged_device_moves
            and tensor.device.type == "cuda"
            and target.type == "cuda"
            and tensor.device != target
        ):
            return tensor.to("cpu").to(target)
        return tensor.to(target, non_blocking=True)

    def _copy_to_gpu0(self, module):
        """Move all params and buffers from storage to GPU 0."""
        target = torch.device("cuda:0")
        for param in module.parameters():
            if param.device != target:
                self._move_owner_data(param, target)
        for buf in module.buffers():
            if buf.device != target:
                self._move_owner_data(buf, target)

    def _copy_to_storage(self, module):
        """Move all params and buffers back to their storage device."""
        if self.storage_device == "split":
            self._copy_to_split_storage(module)
            return

        device = self.storage_device
        target = torch.device(device)
        for param in module.parameters():
            if param.device != target:
                self._move_owner_data(param, target)
        for buf in module.buffers():
            if buf.device != target:
                self._move_owner_data(buf, target)

    def _copy_to_split_storage(self, module):
        for name, param in module.named_parameters():
            device = self._storage_device_for_name(name)
            if param.device != torch.device(device):
                self._move_owner_data(param, device)
        for name, buf in module.named_buffers():
            device = self._storage_device_for_name(name)
            if buf.device != torch.device(device):
                self._move_owner_data(buf, device)

    def _move_owner_data(self, tensor_owner, device):
        intermediates = []
        data = self._move_tensor(tensor_owner.data, device, intermediates)
        self._synchronize_move_devices([data], intermediates)
        tensor_owner.data = data
        intermediates.clear()

    def _synchronize_move_devices(self, tensors, intermediates):
        sync_devices = set()
        if self.loader.cuda_staging_device is not None and intermediates:
            sync_devices.add(self.loader.cuda_staging_device.index)
        for tensor in tensors:
            if tensor.device.type == "cuda":
                sync_devices.add(tensor.device.index)
        for tensor in intermediates:
            if tensor.device.type == "cuda":
                sync_devices.add(tensor.device.index)
        for device_idx in sorted(idx for idx in sync_devices if idx is not None):
            torch.cuda.synchronize(device_idx)

    def _storage_device_for_name(self, name):
        tensor_storage = self.loader.layer_tensor_storage.get(self.layer_idx, {})
        if name in tensor_storage:
            return tensor_storage[name]

        expert_remaps = (
            ("mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj"),
            ("mlp.experts.gate_proj.", "mlp.experts.gate_up_proj"),
            ("mlp.experts.up_proj.", "mlp.experts.gate_up_proj"),
            ("mlp.experts.down_proj.", "mlp.experts.down_proj"),
        )
        for prefix, original_name in expert_remaps:
            if name.startswith(prefix) and original_name in tensor_storage:
                return tensor_storage[original_name]

        for original_name, device in tensor_storage.items():
            parent_name = original_name.rsplit(".", 1)[0]
            if parent_name and name.startswith(parent_name + "."):
                return device

        return self.loader.layer_split_primary_device.get(self.layer_idx, "cpu")

    def _move_quantizer_state(self, module, device):
        """Move all TensorQuantizer buffers and params to the given device.

        After mtq.quantize, each quantizable module has TensorQuantizer children
        with calibration buffers (_amax, _pre_quant_scale, _scale, etc.).
        These must be on the same device as the computation.
        """
        for name, child in module.named_modules():
            cls_name = type(child).__name__
            if "Quantizer" not in cls_name:
                continue
            target = torch.device(device)
            for buf in child.buffers(recurse=False):
                if buf.numel() > 0 and buf.device != target:
                    buf.data = self._move_tensor(buf.data, device)
            for param in child.parameters(recurse=False):
                if param.numel() > 0 and param.device != target:
                    param.data = self._move_tensor(param.data, device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(int(str(device).split(":")[1]))

    def _load_from_disk(self, module):
        """Load layer from safetensors directly to GPU 0.

        After mtq.quantize, expert modules have per-expert nn.Linear structure.
        The loader maps source checkpoint keys into that module tree, including
        MiniMax M3's block_sparse_moe/w1-w3-w2 names.
        """
        self.loader._materialize_layer_module(module, self.layer_idx, "cuda:0")

    def _unload_to_meta(self, module):
        """Free GPU 0 memory for disk-backed layers.

        Resets weight parameter data to empty meta tensors but preserves
        quantizer buffers (like _amax) on CPU so calibration data survives.
        """
        for name, child in module.named_modules():
            # Preserve TensorQuantizer state by moving to CPU instead of meta
            cls_name = type(child).__name__
            last_part = name.rsplit(".", 1)[-1] if name else ""
            if "Quantizer" in cls_name or last_part.endswith("quantizer"):
                for buf_name, buf in child.named_buffers(recurse=False):
                    if buf.device != torch.device("cpu"):
                        buf.data = buf.data.to("cpu")
                for p_name, param in child.named_parameters(recurse=False):
                    if param.device != torch.device("cpu"):
                        param.data = param.data.to("cpu")
                continue

            # Reset regular parameters to empty CPU tensors (frees GPU memory).
            # Can't use meta device here — PyTorch won't assign meta to CUDA .data.
            for p_name, param in child.named_parameters(recurse=False):
                param.data = torch.empty(0, dtype=param.dtype)

            for buf_name, buf in child.named_buffers(recurse=False):
                buf.data = torch.empty(0, dtype=buf.dtype)

        gc.collect()
        torch.cuda.empty_cache()
