"""Streaming model loader for large MoE models.

Replaces accelerate's device_map="auto" with a simple, predictable system:
- GPU 0 is the sole execution device (no weights stored, only activations)
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
    for suffix in ("GiB", "GB", "gib", "gb"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)])
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
    for key in weight_map:
        m = re.match(r"^(.*?layers\.)\d+\.", key)
        if m:
            return m.group(1)
    return "model.layers."


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


class StreamingModelLoader:
    """Load and manage a large model with streaming execution on GPU 0."""

    def __init__(
        self,
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        gpu_capacity_gib=None,
        cpu_capacity_gib=200,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.cpu_capacity_gib = cpu_capacity_gib

        self.num_gpus = torch.cuda.device_count()
        if gpu_capacity_gib is None:
            # Auto-detect per-GPU capacity
            self.gpu_capacities_gib = [
                torch.cuda.get_device_properties(i).total_memory / 1024**3
                for i in range(self.num_gpus)
            ]
        else:
            self.gpu_capacities_gib = [gpu_capacity_gib] * self.num_gpus

        self.snapshot_dir = _resolve_snapshot_dir(model_id)
        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map = index["weight_map"]
        self._layer_prefix = _detect_layer_prefix(self.weight_map)

        # Will be populated during load_model
        self.storage_map = {}  # layer_idx -> device string
        self._hook_handles = []

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

        print(f"Loading config from {self.model_id}...")
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )

        cls = model_cls or AutoModelForCausalLM
        print(f"Creating model on meta device via {cls.__name__}...")
        if cls is AutoModelForCausalLM:
            with init_empty_weights():
                model = cls.from_config(
                    config,
                    torch_dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code,
                )
        else:
            with init_empty_weights():
                model = cls._from_config(config, dtype=self.dtype)

        # Compute per-layer sizes from meta model
        layer_sizes = self._compute_layer_sizes(model)
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

    def _compute_layer_sizes(self, model):
        """Compute parameter size in bytes for each decoder layer."""
        layer_sizes = {}
        for i, layer in enumerate(self._get_layers(model)):
            total = 0
            for p in layer.parameters():
                total += p.numel() * p.element_size()
            layer_sizes[i] = total
        return layer_sizes

    def _compute_storage_map(self, layer_sizes):
        """Assign each layer to a storage device: cuda:1-N, cpu, or meta (disk)."""
        storage_map = {}
        cpu_capacity = self.cpu_capacity_gib * 1024**3

        # GPU 0 is reserved for execution — start packing from GPU 1
        current_gpu = 1
        gpu_used = 0
        cpu_used = 0

        for i in sorted(layer_sizes.keys()):
            size = layer_sizes[i]

            placed = False
            while current_gpu < self.num_gpus:
                # Reserve ~4% headroom for CUDA context, driver memory,
                # and PyTorch allocator fragmentation.
                gpu_total = self.gpu_capacities_gib[current_gpu] * 1024**3
                gpu_capacity = gpu_total * 0.96
                if gpu_used + size <= gpu_capacity:
                    storage_map[i] = f"cuda:{current_gpu}"
                    gpu_used += size
                    placed = True
                    break
                else:
                    current_gpu += 1
                    gpu_used = 0

            if not placed:
                if cpu_used + size <= cpu_capacity:
                    storage_map[i] = "cpu"
                    cpu_used += size
                else:
                    storage_map[i] = "meta"

        return storage_map

    def _print_storage_summary(self, layer_sizes):
        """Print a summary of layer placement."""
        by_device = defaultdict(list)
        for idx, device in sorted(self.storage_map.items()):
            by_device[device].append(idx)

        print(f"\nStorage map ({len(self.storage_map)} layers):")
        for device in sorted(by_device.keys(), key=str):
            indices = by_device[device]
            total_gib = sum(layer_sizes[i] for i in indices) / 1024**3
            if len(indices) <= 6:
                idx_str = ", ".join(str(i) for i in indices)
            else:
                idx_str = f"{indices[0]}-{indices[-1]}"
            print(f"  {device}: layers [{idx_str}] ({total_gib:.1f} GiB)")

    def _get_layer_param_keys(self, layer_idx):
        """Get all checkpoint keys belonging to a specific layer."""
        prefix = f"{self._layer_prefix}{layer_idx}."
        return [k for k in self.weight_map if k.startswith(prefix)]

    def _get_permanent_param_keys(self):
        """Get checkpoint keys for non-layer modules (embed, norm, lm_head)."""
        return [k for k in self.weight_map if _extract_layer_idx(k) is None]

    def _materialize_permanent_modules(self, model):
        """Load embed_tokens, norm, rotary_emb, lm_head to GPU 0.

        Checkpoint keys that don't exist on the model (e.g. MTP weights in
        VL checkpoints) are silently skipped.
        """
        model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
        all_keys = self._get_permanent_param_keys()
        keys = [k for k in all_keys if k in model_keys]
        skipped = len(all_keys) - len(keys)
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
                        lm_child.to("cuda:0")
                continue
            child.to("cuda:0")
        for name, child in model.named_children():
            if name == "model":
                continue
            child.to("cuda:0")

    def _materialize_layer(self, model, layer_idx, device):
        """Load all parameters for a layer from safetensors to the given device."""
        keys = self._get_layer_param_keys(layer_idx)
        self._materialize_params(model, keys, device)

    def _materialize_params(self, model, param_keys, device):
        """Load parameters from safetensors and place into model.

        Handles the checkpoint-to-model key mismatch for MoE experts:
        checkpoint has per-expert keys (experts.{i}.gate_proj.weight) that must
        be fused into 3D parameters (experts.gate_up_proj, experts.down_proj).
        """
        # Separate expert keys from regular keys
        expert_pattern = re.compile(
            r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        regular_keys = []
        # expert_groups[prefix] = {expert_idx: {proj_name: checkpoint_key}}
        expert_groups = defaultdict(lambda: defaultdict(dict))

        for key in param_keys:
            m = expert_pattern.match(key)
            if m:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                expert_groups[prefix][idx][proj] = key
            else:
                regular_keys.append(key)

        # Load regular (non-expert) keys directly
        by_shard = defaultdict(list)
        for key in regular_keys:
            by_shard[self.weight_map[key]].append(key)

        for shard_file, keys in by_shard.items():
            shard_path = os.path.join(self.snapshot_dir, shard_file)
            with safe_open(shard_path, framework="pt", device=str(device)) as f:
                available = set(f.keys())
                for key in keys:
                    if key in available:
                        tensor = f.get_tensor(key)
                        set_module_tensor_to_device(
                            model, key, device, value=tensor
                        )

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

            # Move fused result to target device
            set_module_tensor_to_device(
                model, f"{prefix}.gate_up_proj", device, value=gate_up
            )
            del gate_up

            # Fuse down on CPU: [N, H, I]
            down_stacked = torch.stack(down_tensors)
            del down_tensors
            set_module_tensor_to_device(
                model, f"{prefix}.down_proj", device, value=down_stacked
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
        keys = self._get_layer_param_keys(layer_idx)
        by_shard = defaultdict(list)
        for key in keys:
            by_shard[self.weight_map[key]].append(key)

        prefix = f"{self._layer_prefix}{layer_idx}."
        for shard_file, shard_keys in by_shard.items():
            shard_path = os.path.join(self.snapshot_dir, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                available = set(f.keys())
                for key in shard_keys:
                    if key not in available:
                        continue
                    tensor = f.get_tensor(key)
                    relative = key[len(prefix):]
                    relative = _remap_expert_key(relative)
                    _assign_tensor_to_module(module, relative, tensor, "cpu")


def _remap_expert_key(relative_key):
    """Remap checkpoint expert key to match module tree after _QuantFusedExperts._setup.

    Checkpoint format: mlp.experts.{idx}.{proj}.weight
    Module format:     mlp.experts.{proj}.{idx}.weight
    """
    m = re.match(r"(mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", relative_key)
    if m:
        prefix, idx, proj = m.group(1), m.group(2), m.group(3)
        return f"{prefix}.{proj}.{idx}.weight"
    return relative_key


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
            return output.to("cuda:0")
        if isinstance(output, tuple):
            return tuple(
                t.to("cuda:0") if isinstance(t, torch.Tensor) and t.device != torch.device("cuda:0") else t
                for t in output
            )
        return output

    def _copy_to_gpu0(self, module):
        """Move all params and buffers from storage to GPU 0."""
        for param in module.parameters():
            if param.device != torch.device("cuda:0"):
                param.data = param.data.to("cuda:0", non_blocking=True)
        for buf in module.buffers():
            if buf.device != torch.device("cuda:0"):
                buf.data = buf.data.to("cuda:0", non_blocking=True)
        torch.cuda.synchronize(0)

    def _copy_to_storage(self, module):
        """Move all params and buffers back to their storage device."""
        dev = self.storage_device
        for param in module.parameters():
            if param.device != torch.device(dev):
                param.data = param.data.to(dev, non_blocking=True)
        for buf in module.buffers():
            if buf.device != torch.device(dev):
                buf.data = buf.data.to(dev, non_blocking=True)

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
                    buf.data = buf.data.to(device, non_blocking=True)
            for param in child.parameters(recurse=False):
                if param.numel() > 0 and param.device != target:
                    param.data = param.data.to(device, non_blocking=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(int(str(device).split(":")[1]))

    def _load_from_disk(self, module):
        """Load layer from safetensors directly to GPU 0.

        After mtq.quantize, expert modules have per-expert nn.Linear structure
        (gate_proj[i], up_proj[i], down_proj[i]) which matches the checkpoint
        key format directly. We just need to map the checkpoint path to the
        module tree path.
        """
        keys = self.loader._get_layer_param_keys(self.layer_idx)
        by_shard = defaultdict(list)
        for key in keys:
            by_shard[self.loader.weight_map[key]].append(key)

        prefix = f"{self.loader._layer_prefix}{self.layer_idx}."
        for shard_file, shard_keys in by_shard.items():
            shard_path = os.path.join(self.loader.snapshot_dir, shard_file)
            with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                available = set(f.keys())
                for key in shard_keys:
                    if key not in available:
                        continue
                    tensor = f.get_tensor(key)
                    relative = key[len(prefix):]
                    # Checkpoint: mlp.experts.{i}.gate_proj.weight
                    # Module tree (after _QuantFusedExperts._setup):
                    #   mlp.experts.gate_proj.{i}.weight
                    # Remap: experts.N.proj.weight -> experts.proj.N.weight
                    relative = _remap_expert_key(relative)
                    _assign_tensor_to_module(module, relative, tensor, "cuda:0")

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
