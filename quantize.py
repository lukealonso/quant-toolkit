import argparse
import copy
import gc
import json
import os
import re
import tomllib
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoTokenizer
from safetensors.torch import save_file
import modelopt.torch.quantization as mtq
import logging

from models import load_config, AVAILABLE_MODELS

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
                    help="Token budget for auto-computing batch_size per dataset (batch_size = batch_tokens // max_len).")
parser.add_argument("--max-len", type=int, default=4096)
parser.add_argument("--cpu-capacity", type=str, default="200GiB")
parser.add_argument("--save-amax", type=str, default=None,
                    help="Save calibration amax values to this safetensors file.")
parser.add_argument("--skip-export", action="store_true",
                    help="Skip model export (amax-only calibration run).")
parser.add_argument("--streaming", action="store_true", default=None,
                    help="Force streaming loader. Default: use model config.")
parser.add_argument("--floor-amaxes", action="store_true",
                    help="Floor sparse expert amaxes to median/10 of their peer group.")
parser.add_argument("--resume-amax", type=str, default=None,
                    help="Load amax checkpoint and resume calibration from where it left off.")
parser.add_argument("--resume-batch", type=int, default=0,
                    help="Skip batches before this number (1-indexed). Use with --resume-amax.")
parser.add_argument("--microbatch", type=int, default=None,
                    help="Split calibration batches into microbatches of this size to limit GPU 0 activation memory.")
parser.add_argument("--calib-method", default="max", choices=["max", "quantile"],
                    help="Calibration algorithm. 'quantile' uses P2 streaming quantile estimation.")
parser.add_argument("--save-quantiles", type=str, default=None,
                    help="Save quantile estimates to this JSON file (quantile calibration only).")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Resolve calibration datasets.
# ---------------------------------------------------------------------------

def load_calib_datasets(args):
    """Return list of dicts with keys: path, limit, batch_size, max_len.

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
            ds.setdefault("max_len", 4096)
            if "batch_size" not in ds:
                ds["batch_size"] = max(1, batch_tokens // ds["max_len"])

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
    print(f"  [{i+1}] {ds['path']}  (limit={lim}, batch={ds['batch_size']}, maxlen={ds['max_len']})")


# ---------------------------------------------------------------------------
# Load model config and model.
# ---------------------------------------------------------------------------

cfg = load_config(args.model)
MODEL_ID = args.model_id or cfg.model_id
TRUST_REMOTE = cfg.trust_remote_code
use_streaming = args.streaming if args.streaming is not None else cfg.streaming

cfg.register_moe()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

has_mm = any(ds.get("multimodal", False) for ds in calib_datasets)
processor = None
if has_mm:
    from transformers import AutoProcessor
    from PIL import Image
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
    print(f"Loaded multimodal processor for {MODEL_ID}")


def _parse_gib(s):
    s = s.strip()
    for suffix in ("GiB", "GB", "gib", "gb"):
        if s.endswith(suffix):
            return float(s[: -len(suffix)])
    return float(s)


model_cls = cfg.get_model_cls()

if use_streaming:
    from streaming_loader import StreamingModelLoader

    print(f"Loading model from {MODEL_ID} with streaming loader...")
    loader = StreamingModelLoader(
        model_id=MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE,
        cpu_capacity_gib=_parse_gib(args.cpu_capacity),
    )
    model = loader.load_model(model_cls=model_cls)
else:
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {MODEL_ID} onto GPUs...")
    loader = None
    cls = model_cls or AutoModelForCausalLM
    model = cls.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=TRUST_REMOTE,
        device_map="auto",
    )

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

def _apply_chat_template(messages):
    tmpl = processor if processor is not None else tokenizer
    return tmpl.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def iter_prompts(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            if "messages" in j:
                try:
                    yield _apply_chat_template(j["messages"])
                except Exception:
                    texts = [m["content"] for m in j["messages"] if m.get("role") == "user"]
                    if texts:
                        yield " ".join(texts)
            elif "prompt" in j or "text" in j:
                yield j.get("prompt") or j.get("text")


def _tokenize_batch(texts, max_len):
    """Tokenize a batch of text, using processor if available (VL models)."""
    if processor is not None:
        batch = processor(text=texts, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
        # Text-only batches need explicit position_ids to bypass the VL
        # model's compute_3d_position_ids, which fails without image tokens.
        if "pixel_values" not in batch:
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand_as(batch["input_ids"])
        return batch
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len)


def build_batches(prompts, max_len, batch_size):
    buf = []
    for p in prompts:
        buf.append(p)
        if len(buf) == batch_size:
            yield _tokenize_batch(buf, max_len)
            buf = []
    if buf:
        yield _tokenize_batch(buf, max_len)


def iter_mm_samples(path, limit=None):
    """Yield (messages, [PIL.Image]) from multimodal JSONL."""
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            j = json.loads(line)
            messages = j.get("messages", [])
            images = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, str):
                    continue
                for part in content:
                    if part.get("type") == "image":
                        img_path = part.get("image", "")
                        if img_path:
                            images.append(Image.open(img_path).convert("RGB"))
            if images:
                yield messages, images


def build_mm_batches(samples, max_len, batch_size):
    """Build multimodal batches using the processor."""
    buf_texts = []
    buf_images = []
    for messages, images in samples:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        buf_texts.append(text)
        buf_images.extend(images)
        if len(buf_texts) == batch_size:
            yield processor(
                text=buf_texts, images=buf_images, padding=True,
                truncation=True, max_length=max_len, return_tensors="pt",
            )
            buf_texts = []
            buf_images = []
    if buf_texts:
        yield processor(
            text=buf_texts, images=buf_images, padding=True,
            truncation=True, max_length=max_len, return_tensors="pt",
        )


# Pre-build all batches, tagged with dataset index for logging.
all_batches = []
for ds_idx, ds in enumerate(calib_datasets):
    if ds.get("multimodal", False):
        ds_batches = list(build_mm_batches(
            iter_mm_samples(ds["path"], limit=ds.get("limit")),
            max_len=ds["max_len"],
            batch_size=ds["batch_size"],
        ))
    else:
        ds_batches = list(build_batches(
            iter_prompts(ds["path"], limit=ds.get("limit")),
            max_len=ds["max_len"],
            batch_size=ds["batch_size"],
        ))
    print(f"  Dataset [{ds_idx+1}]: {len(ds_batches)} batches")
    all_batches.extend((ds_idx, b) for b in ds_batches)

print(f"  Total: {len(all_batches)} batches across {len(calib_datasets)} dataset(s)")

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


def forward_loop(m):
    input_device = next(m.parameters()).device
    resume = args.resume_batch

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
                  f"(batch={ds['batch_size']}, maxlen={ds['max_len']}) ---")
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
    print("Calibration complete.")


# ---------------------------------------------------------------------------
# Quantize.
# ---------------------------------------------------------------------------

qcfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
for pattern, override in cfg.get_all_quant_overrides().items():
    qcfg["quant_cfg"][pattern] = override

if args.calib_method == "quantile":
    qcfg["algorithm"] = "quantile"
    calib_cfg = {"type": "quantile"}
    if hasattr(args, "quantiles"):
        calib_cfg["quantiles"] = args.quantiles
    qcfg["quant_cfg"]["*input_quantizer"]["calibrator"] = calib_cfg

print(f"\nQuantizing with NVFP4 (model={args.model}, calib={args.calib_method})...")
model = mtq.quantize(model, qcfg, forward_loop)
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
    export_hf(model, export_dir=args.export_dir, prepare_fn=prepare_fn)
    print(f"Quantized model exported to {args.export_dir}")
