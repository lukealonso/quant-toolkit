import argparse
import copy
import gc
import json
import os
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer
from safetensors.torch import save_file
import modelopt.torch.quantization as mtq
import logging

from models import load_config, AVAILABLE_MODELS
from quant_config_compat import compose_quant_config
from models.mimo_v25_visual import (
    build_mimo_processor,
    precompute_mimo_visual_embeds_for_batches,
    replace_mimo_image_pixels,
)
from models.mimo_v25_media import (
    audio_codes_from_mels,
    audio_placeholder_count_from_codes,
    ensure_mimo_audio_tokenizer,
    expand_audio_placeholders,
    load_audio_mel,
    mimo_video_processor_kwargs,
    normalize_processor_inputs,
    pad_audio_codes_to_group_boundary,
    video_audio_source,
    video_uses_audio,
)
from token_panel import build_token_panel_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS,
                    help="Model config to use.")
parser.add_argument("--model-id", default=None,
                    help="Override the default HuggingFace model ID or local path.")
parser.add_argument("--export-dir", required=True)
parser.add_argument("--calib-config", default=None,
                    help="TOML file describing calibration datasets and parameters.")
parser.add_argument("--data-dir", default="data",
                    help="Base directory for relative dataset paths in the TOML.")
parser.add_argument("--calib-jsonl", default=None,
                    help="Single calibration JSONL (shorthand; ignored if --calib-config given).")
parser.add_argument("--calib-limit", type=int, default=192)
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--batch-tokens", type=int, default=128 * 1024,
                    help="Token budget for auto-computing batch_size when a dataset sets max_len.")
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--cpu-capacity", type=str, default="200GiB")
parser.add_argument("--streaming-gpu-capacity", type=str, default=None,
                    help="Streaming planner capacity for each storage GPU 1-N. Default: physical GPU memory.")
parser.add_argument("--streaming-gpu0-storage-capacity", type=str, default="0GiB",
                    help="Optional streaming layer-storage budget on execution GPU 0, e.g. 48GiB.")
parser.add_argument("--save-amax", type=str, default=None,
                    help="Save calibration amax values to this safetensors file.")
parser.add_argument("--skip-export", action="store_true",
                    help="Skip model export (amax-only calibration run).")
parser.add_argument(
    "--post-quant-smoke",
    action="store_true",
    help=(
        "Calibrate weights without a model forward, then run exactly one configured batch "
        "after quantization is finalized. Intended for dynamic-block NVFP4 streaming smoke tests."
    ),
)
streaming_group = parser.add_mutually_exclusive_group()
streaming_group.add_argument("--streaming", dest="streaming", action="store_true",
                             help="Force streaming loader. Default: use model config.")
streaming_group.add_argument("--no-streaming", dest="streaming", action="store_false",
                             help="Disable streaming loader, overriding the model config.")
parser.add_argument("--device-map", default=None,
                    help="Transformers device_map for non-streaming loads. Defaults to sequential for MiniMax-M3, auto otherwise.")
parser.add_argument("--max-memory-per-gpu", default=None,
                    help="Optional Accelerate max_memory cap for every visible GPU, e.g. 88GiB.")
parser.add_argument("--max-memory-cpu", default=None,
                    help="Optional Accelerate max_memory cap for CPU offload, e.g. 256GiB.")
parser.add_argument("--gpu-headroom-gib", type=float, default=None,
                    help="GPU memory to leave free when deriving max_memory. Defaults to 7 GiB for MiniMax-M3.")
parser.add_argument("--host-staged-device-moves", choices=["auto", "yes", "no"], default="auto",
                    help="Stage cross-GPU moves through CPU. For streaming, auto enables this only with more than 9 visible GPUs.")
parser.add_argument("--cuda-staging-device", default=None,
                    help="For streaming, CUDA trampoline device used by --cuda-staging-source-device transfers to/from cuda:0, e.g. cuda:8.")
parser.add_argument("--cuda-staging-source-device", default=None,
                    help="Storage CUDA device whose transfers to/from cuda:0 should relay through --cuda-staging-device, e.g. cuda:9.")
parser.add_argument("--cuda-staging-reserve", default="0GiB",
                    help="Reduce the streaming planner capacity on --cuda-staging-device by this amount.")
parser.add_argument("--attn-implementation", default="auto",
                    help="Transformers attention backend. For MiniMax-M3, auto resolves to sdpa; use minimax_m3_flex to try the custom FlexAttention path.")
parser.add_argument("--floor-amaxes", action="store_true",
                    help="Floor sparse expert amaxes to median/10 of their peer group.")
parser.add_argument("--resume-amax", type=str, default=None,
                    help="Load amax checkpoint and resume calibration from where it left off.")
parser.add_argument("--resume-batch", type=int, default=0,
                    help="Skip batches before this number (1-indexed). Use with --resume-amax.")
parser.add_argument("--calib-method", default="max", choices=["max", "quantile"],
                    help="Calibration algorithm. 'quantile' uses P2 streaming quantile estimation.")
parser.add_argument("--save-quantiles", type=str, default=None,
                    help="Save quantile estimates to this JSON file (quantile calibration only).")
parser.set_defaults(streaming=None)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Resolve calibration datasets.
# ---------------------------------------------------------------------------

def load_calib_datasets(args):
    """Return list of dicts with keys: path, limit, batch_size, optional max_len.

    Also applies [calibration] overrides from the TOML to args if present.
    """
    if args.calib_config:
        with open(args.calib_config, "rb") as f:
            toml_cfg = tomllib.load(f)
        datasets = toml_cfg.get("dataset", [])
        if not datasets:
            parser.error(f"No [[dataset]] entries in {args.calib_config}")
        batch_tokens = args.batch_tokens
        for i, ds in enumerate(datasets):
            if "path" not in ds:
                parser.error(f"dataset[{i}] missing 'path' in {args.calib_config}")
            if not os.path.isabs(ds["path"]):
                ds["path"] = os.path.join(args.data_dir, ds["path"])
            if "batch_size" not in ds:
                if "max_len" in ds:
                    ds["batch_size"] = max(1, batch_tokens // ds["max_len"])
                else:
                    ds["batch_size"] = args.batch_size

        # [calibration] section overrides CLI defaults.
        calib_sec = toml_cfg.get("calibration", {})
        if "method" in calib_sec:
            args.calib_method = calib_sec["method"]
        if "quantiles" in calib_sec:
            args.quantiles = calib_sec["quantiles"]

        return datasets

    if not args.calib_jsonl:
        parser.error("Provide either --calib-config or --calib-jsonl")
    return [{
        "path": args.calib_jsonl,
        "limit": args.calib_limit,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
    }]


calib_datasets = load_calib_datasets(args)
print(f"\nCalibration plan: {len(calib_datasets)} dataset(s)")
for i, ds in enumerate(calib_datasets):
    lim = ds.get('limit', 'all')
    max_len = ds.get("max_len")
    max_len_str = max_len if max_len is not None else "dynamic"
    print(f"  [{i+1}] {ds['path']}  (limit={lim}, batch={ds['batch_size']}, maxlen={max_len_str})")


# ---------------------------------------------------------------------------
# Load model config and model.
# ---------------------------------------------------------------------------

cfg = load_config(args.model)
MODEL_ID = args.model_id or cfg.model_id
TRUST_REMOTE = cfg.trust_remote_code
PROCESSOR_TRUST_REMOTE = cfg.get_processor_trust_remote_code()
use_streaming = args.streaming if args.streaming is not None else cfg.streaming
ATTN_IMPLEMENTATION = None
if args.attn_implementation != "auto":
    ATTN_IMPLEMENTATION = args.attn_implementation
elif args.model == "minimax_m3":
    ATTN_IMPLEMENTATION = "sdpa"
if ATTN_IMPLEMENTATION == "minimax_m3_flex":
    from models.minimax_m3_flex import register_minimax_m3_flex_attention

    register_minimax_m3_flex_attention()
if ATTN_IMPLEMENTATION is not None:
    print(f"Using attention implementation: {ATTN_IMPLEMENTATION}")

cfg.register_moe()
qcfg = compose_quant_config(
    mtq.NVFP4_DEFAULT_CFG,
    cfg.get_all_quant_overrides(),
    calibration_method=args.calib_method,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

has_mm = any(ds.get("multimodal", False) for ds in calib_datasets)
processor = None
if has_mm:
    from transformers import AutoProcessor
    from PIL import Image
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=PROCESSOR_TRUST_REMOTE,
        )
        print(f"Loaded multimodal processor for {MODEL_ID}")
    except OSError:
        if args.model != "mimo_v25":
            raise
        print(f"Deferring MiMo processor construction until model config is loaded for {MODEL_ID}")


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


def _max_memory_map(per_gpu: str | None, cpu: str | None) -> dict | None:
    max_memory = {}
    if per_gpu is not None:
        for idx in range(torch.cuda.device_count()):
            max_memory[idx] = per_gpu
    if cpu is not None:
        max_memory["cpu"] = cpu
    return max_memory or None


def _default_per_gpu_memory(headroom_gib: float | None) -> str | None:
    if args.model != "minimax_m3":
        return None
    if headroom_gib is None:
        headroom_gib = 7.0
    if not torch.cuda.is_available() or headroom_gib <= 0:
        return None
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    usable_gib = max(1, int(total_gib - headroom_gib))
    return f"{usable_gib}GiB"


def _patch_accelerate_host_staged_device_moves():
    try:
        import accelerate.hooks as accelerate_hooks
        from accelerate.utils import operations as accelerate_ops
    except ImportError:
        return

    if getattr(accelerate_hooks, "_quantize_host_staged_send", False):
        return

    original_send_to_device = accelerate_ops.send_to_device

    def host_staged_send_to_device(value, device, non_blocking=False, skip_keys=None):
        target = torch.device(device) if not isinstance(device, torch.device) else device
        if torch.is_tensor(value):
            if value.device.type == "cuda" and target.type == "cuda" and value.device != target:
                return value.to("cpu").to(target)
            return original_send_to_device(value, device, non_blocking=non_blocking, skip_keys=skip_keys)
        if isinstance(value, tuple):
            return tuple(
                host_staged_send_to_device(item, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for item in value
            )
        if isinstance(value, list):
            return [
                host_staged_send_to_device(item, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for item in value
            ]
        if isinstance(value, Mapping):
            if isinstance(skip_keys, str):
                skip = {skip_keys}
            else:
                skip = set(skip_keys or [])
            return type(value)(
                {
                    key: item
                    if key in skip
                    else host_staged_send_to_device(item, device, non_blocking=non_blocking, skip_keys=skip_keys)
                    for key, item in value.items()
                }
            )
        return value

    accelerate_ops.send_to_device = host_staged_send_to_device
    accelerate_hooks.send_to_device = host_staged_send_to_device
    accelerate_hooks._quantize_host_staged_send = True


model_cls = cfg.get_model_cls()

if use_streaming:
    from streaming_loader import StreamingModelLoader

    print(f"Loading model from {MODEL_ID} with streaming loader...")
    streaming_cuda_staging_device = args.cuda_staging_device
    streaming_host_stage_moves = args.host_staged_device_moves == "yes" or (
        args.host_staged_device_moves == "auto"
        and torch.cuda.device_count() > 9
        and streaming_cuda_staging_device is None
    )
    if streaming_cuda_staging_device is not None:
        if args.cuda_staging_source_device:
            print(
                "Using CUDA-staged streaming layer moves "
                f"{args.cuda_staging_source_device} -> {streaming_cuda_staging_device} -> cuda:0"
            )
        else:
            print(f"Using CUDA-staged streaming layer moves through {streaming_cuda_staging_device}")
    if streaming_host_stage_moves:
        print("Using host-staged streaming layer moves")
    loader = StreamingModelLoader(
        model_id=MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE,
        gpu_capacity_gib=(
            _parse_gib(args.streaming_gpu_capacity)
            if args.streaming_gpu_capacity is not None else None
        ),
        gpu0_storage_capacity_gib=_parse_gib(args.streaming_gpu0_storage_capacity),
        cpu_capacity_gib=_parse_gib(args.cpu_capacity),
        host_staged_device_moves=streaming_host_stage_moves,
        attention_implementation=ATTN_IMPLEMENTATION,
        cuda_staging_device=streaming_cuda_staging_device,
        cuda_staging_source_device=args.cuda_staging_source_device,
        cuda_staging_reserve_gib=_parse_gib(args.cuda_staging_reserve),
    )
    model = loader.load_model(model_cls=model_cls)
else:
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {MODEL_ID} onto GPUs...")
    loader = None
    cls = model_cls or AutoModelForCausalLM
    device_map = args.device_map or ("sequential" if args.model == "minimax_m3" else "auto")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": TRUST_REMOTE,
        "device_map": device_map,
    }
    if ATTN_IMPLEMENTATION is not None:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    per_gpu_memory = args.max_memory_per_gpu or _default_per_gpu_memory(args.gpu_headroom_gib)
    max_memory = _max_memory_map(per_gpu_memory, args.max_memory_cpu)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
        print(f"Using max_memory={max_memory}")
    host_stage_moves = args.host_staged_device_moves == "yes" or (
        args.host_staged_device_moves == "auto" and args.model == "minimax_m3"
    )
    if host_stage_moves:
        _patch_accelerate_host_staged_device_moves()
        print("Using host-staged cross-GPU activation moves")
    print(f"Using device_map={device_map}")
    model = cls.from_pretrained(
        MODEL_ID,
        **model_kwargs,
    )
if args.model == "minimax_m3" and ATTN_IMPLEMENTATION is not None:
    from models.minimax_m3_flex import assert_attention_implementation

    assert_attention_implementation(model, ATTN_IMPLEMENTATION)

if has_mm and processor is None and getattr(getattr(model, "config", None), "model_type", None) == "mimo_v2":
    processor = build_mimo_processor(tokenizer, model.config)
    print(f"Built MiMo multimodal processor for {MODEL_ID}")
if has_mm and processor is not None and getattr(processor, "chat_template", None) is None:
    processor.chat_template = tokenizer.chat_template

# Dtype distribution before quantization.
print(f"\n{'='*60}")
print("Data type distribution BEFORE quantization:")
dtype_stats = {}
total_params = 0
for name, param in model.named_parameters():
    dtype = str(param.dtype)
    if dtype not in dtype_stats:
        dtype_stats[dtype] = {"count": 0, "size_bytes": 0}
    dtype_stats[dtype]["count"] += 1
    dtype_stats[dtype]["size_bytes"] += param.numel() * param.element_size()
    total_params += param.numel()

for dtype, stats in sorted(dtype_stats.items()):
    print(f"  {dtype:<20} {stats['count']:>6} tensors, {stats['size_bytes']/1e9:>8.2f} GB")
print(
    f"  {'TOTAL':<20} {sum(s['count'] for s in dtype_stats.values()):>6} tensors, "
    f"{sum(s['size_bytes'] for s in dtype_stats.values())/1e9:>8.2f} GB"
)
print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Build calibration batches from all datasets.
# ---------------------------------------------------------------------------

text_chat_template_counts = defaultdict(int)


def _apply_text_chat_template(messages):
    final_role = messages[-1].get("role") if messages else None
    if final_role == "assistant":
        text_chat_template_counts["assistant_continuation"] += 1
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    text_chat_template_counts["generation_prompt"] += 1
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_prompts(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            if "messages" in j:
                try:
                    yield _apply_text_chat_template(j["messages"])
                except Exception:
                    texts = [m["content"] for m in j["messages"] if m.get("role") == "user"]
                    if texts:
                        text_chat_template_counts["fallback_user_text"] += 1
                        yield " ".join(texts)
            elif "prompt" in j or "text" in j:
                text_chat_template_counts["raw_prompt"] += 1
                yield j.get("prompt") or j.get("text")


def _tokenize_batch(texts, max_len):
    """Tokenize a batch of text, using processor if available (VL models)."""
    kwargs = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if max_len is not None:
        kwargs.update({
            "truncation": True,
            "max_length": max_len,
        })
    if processor is not None:
        batch = processor(**kwargs)
        batch = _normalize_processor_batch(batch)
        # Text-only batches need explicit position_ids to bypass the VL
        # model's compute_3d_position_ids, which fails without image tokens.
        if not _has_visual_inputs(batch):
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand_as(batch["input_ids"])
        return batch
    return tokenizer(**kwargs)


def build_batches(prompts, max_len, batch_size):
    buf = []
    for p in prompts:
        buf.append(p)
        if len(buf) == batch_size:
            yield _tokenize_batch(buf, max_len)
            buf = []
    if buf:
        yield _tokenize_batch(buf, max_len)


def _messages_with_answer_prefix(messages):
    messages = copy.deepcopy(messages)
    if messages and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content")
        if not content:
            messages[-1]["content"] = "Answer:"
        elif isinstance(content, str) and not content.startswith("Answer:"):
            messages[-1]["content"] = f"Answer: {content}"
    else:
        messages.append({"role": "assistant", "content": "Answer:"})
    return messages


def _apply_mm_chat_template(messages):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
    if _is_mimo_v2_model():
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def _has_visual_inputs(batch):
    return any(
        key in batch
        for key in (
            "pixel_values",
            "image_embeds",
            "pixel_values_videos",
            "video_pixel_values",
            "video_embeds",
        )
    )


def _is_mimo_v2_model():
    return getattr(getattr(model, "config", None), "model_type", None) == "mimo_v2"


def _is_minimax_m3_vl_model():
    return getattr(getattr(model, "config", None), "model_type", None) == "minimax_m3_vl"


def _normalize_processor_batch(batch):
    if _is_mimo_v2_model():
        return normalize_processor_inputs(batch)
    return batch


def _video_processor_kwargs():
    if _is_mimo_v2_model():
        return mimo_video_processor_kwargs()
    if _is_minimax_m3_vl_model():
        # The published M3 processor merges video kwargs with do_resize=False,
        # but its patch packing requires H/W to be aligned to patch*merge.
        # It also uses every frame for path inputs unless sampling is enabled,
        # which can create huge video prompts that are easy to truncate.
        return {
            "do_resize": True,
            "do_sample_frames": True,
            "fps": 0.5,
            "max_pixels": 224 * 224,
        }
    return {}


def _config_value(name, default=None):
    value = getattr(model.config, name, None)
    if value is not None:
        return value
    processor_config = getattr(model.config, "processor_config", None) or {}
    if isinstance(processor_config, dict):
        return processor_config.get(name, default)
    return getattr(processor_config, name, default)


def _media_ref(part, media_type):
    return part.get(media_type) or part.get("path") or part.get("file") or part.get("url")


def _resolve_media_path(ref, dataset_path):
    path = Path(ref).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        Path(args.data_dir) / path,
        Path(dataset_path).parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _load_tensor_value(value, dataset_path, tensor_name=None):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return torch.tensor(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported tensor value {type(value).__name__}")

    path = Path(_resolve_media_path(value, dataset_path))
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            if tensor_name and tensor_name in obj:
                return obj[tensor_name]
            tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if tensors:
                return tensors[0]
        raise ValueError(f"No tensor found in {path}")
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(path, device="cpu")
        if tensor_name and tensor_name in tensors:
            return tensors[tensor_name]
        return tensors[sorted(tensors)[0]]
    if suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    if suffix == ".json":
        with open(path) as f:
            return torch.tensor(json.load(f))

    raise ValueError(f"Unsupported tensor file type: {path}")


def _ensure_mimo_audio_tokenizer():
    return ensure_mimo_audio_tokenizer(
        model=model,
        model_id=MODEL_ID,
        fallback_model_id=cfg.model_id,
        log_file=sys.stdout,
    )


def _load_audio_mel(audio_path):
    return load_audio_mel(audio_path, _ensure_mimo_audio_tokenizer())


def _audio_codes_from_mels(mels):
    _ensure_mimo_audio_tokenizer()
    return audio_codes_from_mels(model, mels)


def _prepare_audio_part(part, dataset_path):
    if "audio_embeds" in part:
        embeds = _load_tensor_value(part["audio_embeds"], dataset_path, tensor_name=part.get("tensor_name"))
        if embeds.dim() != 2:
            raise ValueError(f"audio_embeds must be 2D [N, H], got {tuple(embeds.shape)}")
        return {"embeds": embeds, "placeholder_count": embeds.shape[0]}

    if "audio_codes" in part:
        codes = _load_tensor_value(part["audio_codes"], dataset_path, tensor_name=part.get("tensor_name")).long()
        if codes.dim() == 1:
            codes = codes.unsqueeze(-1)
        if codes.dim() != 2:
            raise ValueError(f"audio_codes must be 2D [T, C], got {tuple(codes.shape)}")
        codes = pad_audio_codes_to_group_boundary(codes, model.config)
        return {"codes": codes, "placeholder_count": audio_placeholder_count_from_codes(codes, model.config)}

    if "audio_mels" in part:
        mel = _load_tensor_value(part["audio_mels"], dataset_path, tensor_name=part.get("tensor_name")).float()
        codes = pad_audio_codes_to_group_boundary(_audio_codes_from_mels([mel])[0], model.config)
        return {"codes": codes, "placeholder_count": audio_placeholder_count_from_codes(codes, model.config)}

    audio_ref = _media_ref(part, "audio")
    if audio_ref:
        mel = _load_audio_mel(_resolve_media_path(audio_ref, dataset_path))
        codes = pad_audio_codes_to_group_boundary(_audio_codes_from_mels([mel])[0], model.config)
        return {"codes": codes, "placeholder_count": audio_placeholder_count_from_codes(codes, model.config)}

    return None


def iter_mm_samples(path, limit=None):
    """Yield multimodal samples from JSONL with image, video, and audio parts."""
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            messages = j.get("messages", [])
            processed_messages = copy.deepcopy(messages)
            images = []
            videos = []
            audios = []
            for msg in processed_messages:
                content = msg.get("content", [])
                if isinstance(content, str):
                    continue
                new_content = []
                for part in content:
                    new_content.append(part)
                    if part.get("type") == "image":
                        img_path = _media_ref(part, "image")
                        if img_path:
                            images.append(Image.open(_resolve_media_path(img_path, path)).convert("RGB"))
                    elif part.get("type") == "video":
                        video_path = _media_ref(part, "video")
                        if video_path:
                            resolved_video_path = _resolve_media_path(video_path, path)
                            videos.append(resolved_video_path)
                            if _is_mimo_v2_model() and video_uses_audio(part, resolved_video_path):
                                audio_ref = video_audio_source(part, resolved_video_path)
                                audio_part = {"type": "audio", "audio": audio_ref}
                                audio = _prepare_audio_part(audio_part, path)
                                if audio is not None:
                                    audios.append(audio)
                                    # SGLang turns video-with-audio into both VIDEO and AUDIO
                                    # modalities. Keep the prompt modal tokens aligned with
                                    # that by adding an audio part next to the video part.
                                    new_content.append(audio_part)
                    elif part.get("type") == "audio":
                        audio = _prepare_audio_part(part, path)
                        if audio is not None:
                            audios.append(audio)
                msg["content"] = new_content
            if images or videos or audios:
                yield {
                    "messages": processed_messages,
                    "images": images,
                    "videos": videos,
                    "audios": audios,
                }


def build_mm_batches(samples, max_len, batch_size):
    """Build multimodal batches using the processor."""
    buf_texts = []
    buf_images = []
    buf_videos = []
    buf_audio_codes = []
    buf_audio_embeds = []
    buf_audio_counts = []

    def flush():
        if not buf_texts:
            return None
        if buf_audio_codes and buf_audio_embeds:
            raise ValueError("A single multimodal batch cannot mix audio_codes and audio_embeds")

        kwargs = {
            "text": buf_texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if max_len is not None:
            kwargs.update({
                "truncation": True,
                "max_length": max_len,
            })
        if buf_images:
            kwargs["images"] = buf_images
        if buf_videos:
            kwargs["videos"] = buf_videos
            kwargs.update(_video_processor_kwargs())
        batch = processor(**kwargs)
        batch = _normalize_processor_batch(batch)
        if _is_mimo_v2_model() and buf_images:
            replace_mimo_image_pixels(batch, buf_images, model.config)
        if buf_audio_codes:
            batch["audio_codes"] = torch.cat(buf_audio_codes, dim=0).long()
        if buf_audio_embeds:
            batch["audio_embeds"] = torch.cat(buf_audio_embeds, dim=0)
        if buf_audio_counts:
            audio_token_id = _config_value("audio_token_id")
            if audio_token_id is None:
                raise ValueError("Audio calibration data requires model.config.audio_token_id")
            expected = sum(buf_audio_counts)
            actual = int((batch["input_ids"] == int(audio_token_id)).sum().item())
            if actual != expected:
                raise ValueError(
                    "Audio placeholder mismatch: "
                    f"tokenized prompt has {actual} audio token(s), but audio payload expects {expected}"
                )
        if not _has_visual_inputs(batch):
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand_as(batch["input_ids"])
        return batch

    for sample in samples:
        text = _apply_mm_chat_template(_messages_with_answer_prefix(sample["messages"]))
        audio_counts = [audio["placeholder_count"] for audio in sample["audios"]]
        if audio_counts:
            text = expand_audio_placeholders(text, audio_counts)
        buf_texts.append(text)
        buf_images.extend(sample["images"])
        buf_videos.extend(sample["videos"])
        buf_audio_counts.extend(audio_counts)
        for audio in sample["audios"]:
            if "codes" in audio:
                buf_audio_codes.append(audio["codes"])
            elif "embeds" in audio:
                buf_audio_embeds.append(audio["embeds"])
        if len(buf_texts) == batch_size:
            yield flush()
            buf_texts = []
            buf_images = []
            buf_videos = []
            buf_audio_codes = []
            buf_audio_embeds = []
            buf_audio_counts = []
    if buf_texts:
        yield flush()


# Pre-build all batches, tagged with dataset index for logging.
all_batches = []
for ds_idx, ds in enumerate(calib_datasets):
    if ds.get("format") == "token_panel":
        role = ds.get("role")
        if not role:
            parser.error(f"token_panel dataset[{ds_idx}] requires a role")
        ds_batches = build_token_panel_batches(
            ds["path"],
            role=role,
            limit=ds.get("limit"),
            max_len=ds.get("max_len"),
            batch_size=ds["batch_size"],
        )
    elif ds.get("multimodal", False):
        ds_batches = list(build_mm_batches(
            iter_mm_samples(ds["path"], limit=ds.get("limit")),
            max_len=ds.get("max_len"),
            batch_size=ds["batch_size"],
        ))
    else:
        ds_batches = list(build_batches(
            iter_prompts(ds["path"], limit=ds.get("limit")),
            max_len=ds.get("max_len"),
            batch_size=ds["batch_size"],
        ))
    print(f"  Dataset [{ds_idx+1}]: {len(ds_batches)} batches")
    all_batches.extend((ds_idx, b) for b in ds_batches)

print(f"  Total: {len(all_batches)} batches across {len(calib_datasets)} dataset(s)")
if text_chat_template_counts:
    counts = ", ".join(f"{key}={value}" for key, value in sorted(text_chat_template_counts.items()))
    print(f"  Text prompt formats: {counts}")
if _is_mimo_v2_model():
    converted = precompute_mimo_visual_embeds_for_batches(
        MODEL_ID,
        model.config,
        all_batches,
    )
    if converted:
        print(f"Precomputed MiMo visual embeddings for {converted} calibration batch tensor(s)")
if getattr(model, "audio_tokenizer", None) is not None:
    # Audio media has already been converted to compact codes in all_batches.
    # Keep the tokenizer sidecar out of ModelOpt wrapping and checkpoint export.
    model.audio_tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

model.eval()
for p in model.parameters():
    p.requires_grad_(False)
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Calibration forward loop.
# ---------------------------------------------------------------------------

amax_ckpt_path = os.path.join(args.export_dir, "amax_checkpoint.safetensors")
quantile_ckpt_path = os.path.join(args.export_dir, "quantile_checkpoint.json")
os.makedirs(args.export_dir, exist_ok=True)


def _save_amax_checkpoint(m, batch_num):
    from modelopt.torch.quantization.nn import TensorQuantizer
    amaxes = {}
    for name, mod in m.named_modules():
        if isinstance(mod, TensorQuantizer):
            cal = getattr(mod, "_calibrator", None)
            if cal is not None and getattr(cal, "_calib_amax", None) is not None:
                v = cal._calib_amax.detach().clone().cpu()
                amaxes[name] = v.reshape(1) if v.dim() == 0 else v
    save_file(amaxes, amax_ckpt_path)
    print(f"    [checkpoint] {len(amaxes)} amaxes saved after batch {batch_num}")


def _save_quantile_checkpoint(m, batch_num):
    from modelopt.torch.quantization.calib.quantile import save_quantile_data
    n_saved = save_quantile_data(m, quantile_ckpt_path)
    print(f"    [checkpoint] {n_saved} quantile estimates saved after batch {batch_num}")


def _restore_amax(m, path):
    from safetensors.torch import load_file
    from modelopt.torch.quantization.nn import TensorQuantizer
    saved = load_file(path)
    restored = 0
    for name, mod in m.named_modules():
        if name in saved and isinstance(mod, TensorQuantizer):
            cal = getattr(mod, "_calibrator", None)
            if cal is not None:
                v = saved[name]
                if v.shape == torch.Size([1]):
                    v = v.squeeze(0)
                device = cal._calib_amax.device if cal._calib_amax is not None else "cuda:0"
                cal._calib_amax = v.to(device)
                restored += 1
    print(f"Restored {restored}/{len(saved)} calibrator amaxes from {path}")


_MIMO_ALLOWED_QUANTIZER_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.(?:weight_quantizer|input_quantizer)$"
)

_MINIMAX_M3_ALLOWED_QUANTIZER_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_proj|up_proj|down_proj)\.\d+\."
    r"(?:weight_quantizer|input_quantizer)$"
)

_GLM53_FLASH_ALLOWED_QUANTIZER_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_proj|up_proj|down_proj)\.\d+\."
    r"(?:weight_quantizer|input_quantizer)$"
)

def _validate_enabled_quantizers(m):
    allowed_re = None
    label = None
    if args.model == "mimo_v25":
        allowed_re = _MIMO_ALLOWED_QUANTIZER_RE
        label = "MiMo"
    elif args.model == "minimax_m3":
        allowed_re = _MINIMAX_M3_ALLOWED_QUANTIZER_RE
        label = "MiniMax M3"
    elif args.model == "glm5_3_flash":
        allowed_re = _GLM53_FLASH_ALLOWED_QUANTIZER_RE
        label = "GLM-5.3-Flash"

    if allowed_re is None:
        return

    enabled = []
    bad = []
    for name, mod in m.named_modules():
        is_enabled = getattr(mod, "is_enabled", None)
        if is_enabled is None:
            continue
        if bool(is_enabled):
            enabled.append(name)
            if not allowed_re.match(name):
                bad.append(name)

    if bad:
        sample = "\n    ".join(bad[:32])
        raise RuntimeError(
            f"{label} quantization is configured for routed expert MLPs only, "
            f"but found {len(bad)} enabled non-expert quantizer(s):\n    {sample}"
        )
    if not enabled:
        raise RuntimeError(f"{label} quantization has no enabled routed expert MLP quantizers")
    print(f"  {label} enabled quantizers: {len(enabled)} routed expert MLP quantizer(s)")


def _override_quantile_levels(m):
    """Override quantile levels on all QuantileCalibrators if specified in config."""
    if not hasattr(args, "quantiles"):
        return
    from modelopt.torch.quantization.calib.quantile import QuantileCalibrator, P2QuantileEstimator
    count = 0
    for name, mod in m.named_modules():
        cal = getattr(mod, "_calibrator", None)
        if isinstance(cal, QuantileCalibrator):
            cal._quantile_probs = list(args.quantiles)
            cal._estimators = {p: P2QuantileEstimator(p) for p in cal._quantile_probs}
            count += 1
    if count:
        print(f"  Overrode quantile levels to {args.quantiles} on {count} calibrators")


def forward_loop(m):
    input_device = next(m.parameters()).device
    resume = args.resume_batch

    _validate_enabled_quantizers(m)
    _override_quantile_levels(m)

    if args.resume_amax:
        _restore_amax(m, args.resume_amax)
        print(f"Resuming from batch {resume + 1}")

    print(f"\nCalibration: {len(all_batches)} batches across {len(calib_datasets)} dataset(s)...")
    cur_ds = -1
    for i, (ds_idx, batch) in enumerate(all_batches, 1):
        if i <= resume:
            continue
        if ds_idx != cur_ds:
            cur_ds = ds_idx
            ds = calib_datasets[ds_idx]
            print(f"\n  --- Dataset [{ds_idx+1}]: {os.path.basename(ds['path'])} "
                  f"(batch={ds['batch_size']}, maxlen={ds.get('max_len', 'dynamic')}) ---")
        print(f"  Batch {i}/{len(all_batches)}...")
        kwargs = {
            k: v.to(input_device, non_blocking=True)
            for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            outputs = m(**kwargs, use_cache=False)
        del outputs, kwargs
        gc.collect()
        torch.cuda.empty_cache()
        _save_amax_checkpoint(m, i)
        if args.calib_method == "quantile":
            _save_quantile_checkpoint(m, i)
    print("Calibration complete.")


def post_quant_smoke(m):
    if len(all_batches) != 1:
        raise RuntimeError(
            f"--post-quant-smoke requires exactly one configured batch, got {len(all_batches)}"
        )
    _validate_enabled_quantizers(m)
    input_device = next(m.parameters()).device
    _, batch = all_batches[0]
    kwargs = {
        key: value.to(input_device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    print("\nPost-quant smoke: 1 batch with finalized quantizers...")
    with torch.no_grad():
        outputs = m(**kwargs, use_cache=False)
    del outputs, kwargs
    gc.collect()
    torch.cuda.empty_cache()
    print("Post-quant smoke complete.")


# ---------------------------------------------------------------------------
# Quantize.
# ---------------------------------------------------------------------------

print(f"\nQuantizing with NVFP4 (model={args.model}, calib={args.calib_method})...")
calibration_loop = None if args.post_quant_smoke else forward_loop
model = mtq.quantize(model, qcfg, calibration_loop)
if args.post_quant_smoke:
    post_quant_smoke(model)
print(f"{'='*60}")

if args.save_quantiles:
    from modelopt.torch.quantization.calib.quantile import save_quantile_data
    os.makedirs(os.path.dirname(os.path.abspath(args.save_quantiles)), exist_ok=True)
    n_saved = save_quantile_data(model, args.save_quantiles)
    print(f"Saved quantile data for {n_saved} quantizers to {args.save_quantiles}")


# ---------------------------------------------------------------------------
# Post-calibration: diagnostic + optional amax flooring.
# ---------------------------------------------------------------------------

print("\nCalibration amax diagnostic:")
zero_amax_count = 0
nonzero_amax_count = 0
nan_amax_count = 0
sample_lines = []
for name, mod in model.named_modules():
    if not hasattr(mod, "gate_proj") or not isinstance(getattr(mod, "gate_proj", None), nn.ModuleList):
        continue
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj_list = getattr(mod, proj_name, None)
        if proj_list is None:
            continue
        for i, expert_linear in enumerate(proj_list):
            for qname in ("weight_quantizer", "input_quantizer"):
                q = getattr(expert_linear, qname, None)
                if q is None or not hasattr(q, "_amax"):
                    continue
                amax = q._amax
                if torch.isnan(amax).any():
                    nan_amax_count += 1
                elif (amax == 0).all():
                    zero_amax_count += 1
                else:
                    nonzero_amax_count += 1
                if len(sample_lines) < 12 and i < 3:
                    sample_lines.append(
                        f"  {name}.{proj_name}[{i}].{qname}._amax = "
                        f"{amax.flatten()[:4].tolist()} (device={amax.device})"
                    )

for line in sample_lines:
    print(line)
print(f"\n  Summary: {nonzero_amax_count} nonzero, {zero_amax_count} zero, {nan_amax_count} NaN")
if zero_amax_count > 0:
    print(f"  WARNING: {zero_amax_count} quantizers have zero amax (lost during offload?)")
if nan_amax_count > 0:
    print(f"  WARNING: {nan_amax_count} quantizers have NaN amax")
print(f"{'='*60}")


# Floor sparse expert amaxes: experts with amax < median/10 of their
# peer group get pulled up to median/10. Prevents NaN from tight scales
# fitted to a handful of calibration samples.
if args.floor_amaxes:
    _EXPERT_AMAX_RE = re.compile(
        r"^(?P<prefix>.*\.experts)\."
        r"(?:(?P<idx1>\d+)\.(?P<proj1>gate_proj|up_proj|down_proj)"
        r"|(?P<proj2>gate_proj|up_proj|down_proj)\.(?P<idx2>\d+))"
        r"\.(?P<qtype>input_quantizer|weight_quantizer)$"
    )
    groups = defaultdict(dict)
    for name, mod in model.named_modules():
        m = _EXPERT_AMAX_RE.match(name)
        if m and hasattr(mod, "_amax"):
            prefix = m.group("prefix")
            proj = m.group("proj1") or m.group("proj2")
            expert_idx = int(m.group("idx1") or m.group("idx2"))
            qtype = m.group("qtype")
            groups[(prefix, proj, qtype)][expert_idx] = mod

    floored = 0
    for (prefix, proj, qtype), experts in groups.items():
        vals = sorted(m._amax.float().item() for m in experts.values() if m._amax.item() > 0)
        if not vals:
            continue
        median = vals[len(vals) // 2]
        threshold = median / 10
        for idx, mod in experts.items():
            if mod._amax.item() < threshold:
                mod._amax.fill_(threshold)
                floored += 1

    print(f"Floored {floored} sparse expert amaxes to median/10 ({len(groups)} groups)")


# ---------------------------------------------------------------------------
# Tie gate/up projection weight quantizer amaxes for fused w13 export.
# ---------------------------------------------------------------------------

def _tie_pair(gq, uq):
    if gq is None or uq is None:
        return False
    if not hasattr(gq, "_amax") or not hasattr(uq, "_amax"):
        return False
    shared = torch.max(gq._amax, uq._amax)
    gq._amax.copy_(shared)
    uq._amax.copy_(shared)
    return True


tied = 0
for name, mod in model.named_modules():
    if hasattr(mod, "gate_proj") and hasattr(mod, "up_proj"):
        if isinstance(mod.gate_proj, nn.ModuleList):
            for i in range(len(mod.gate_proj)):
                if _tie_pair(
                    getattr(mod.gate_proj[i], "weight_quantizer", None),
                    getattr(mod.up_proj[i], "weight_quantizer", None),
                ):
                    tied += 1
        elif isinstance(mod.gate_proj, nn.Linear):
            if _tie_pair(
                getattr(mod.gate_proj, "weight_quantizer", None),
                getattr(mod.up_proj, "weight_quantizer", None),
            ):
                tied += 1
    elif hasattr(mod, "w1") and hasattr(mod, "w3"):
        if _tie_pair(
            getattr(mod.w1, "weight_quantizer", None),
            getattr(mod.w3, "weight_quantizer", None),
        ):
            tied += 1
print(f"Tied gate/up weight_quantizer amax for {tied} experts.")


# ---------------------------------------------------------------------------
# Save amaxes / export.
# ---------------------------------------------------------------------------

def _collect_amax(model):
    amaxes = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "_amax"):
            amaxes[name] = mod._amax.detach().cpu()
    return amaxes


if args.save_amax:
    amaxes = _collect_amax(model)
    tensors = {k: v.reshape(1) if v.dim() == 0 else v for k, v in amaxes.items()}
    os.makedirs(os.path.dirname(args.save_amax), exist_ok=True)
    save_file(tensors, args.save_amax)
    zero_count = sum(1 for v in tensors.values() if (v == 0).all())
    nan_count = sum(1 for v in tensors.values() if torch.isnan(v).any())
    print(f"Saved {len(tensors)} amax values to {args.save_amax}")
    if zero_count:
        print(f"  WARNING: {zero_count} amaxes are all-zero (uncalibrated)")
    if nan_count:
        print(f"  WARNING: {nan_count} amaxes contain NaN")

if args.skip_export:
    print("\nSkipping export (--skip-export).")
else:
    from export_hf import export_hf

    print("\nExporting quantized model to HF format...")
    prepare_fn = loader.prepare_export if loader is not None else None
    source_dir = loader.snapshot_dir if loader is not None else None
    export_hf(model, export_dir=args.export_dir, prepare_fn=prepare_fn,
              extra_mtp_prefixes=cfg.extra_mtp_prefixes,
              preserve_remote_code=cfg.preserve_remote_code,
              source_dir=source_dir)
    print(f"Quantized model exported to {args.export_dir}")
