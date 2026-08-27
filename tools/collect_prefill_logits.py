#!/usr/bin/env python3
"""Capture dense BF16 teacher prefill logits from a vLLM runtime.

The output is reusable: every shard stores the exact input token IDs beside
the logits, so candidate scoring does not retokenize or reconstruct a corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from kld_common import (
    canonical_json_sha256,
    dense_prompt_logits,
    sampling_params_for_prompt_logits,
    sha256_file,
)
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def load_source_text(args, tokenizer) -> tuple[str, dict]:
    if args.input_jsonl:
        source_path = Path(args.input_jsonl).resolve()
        source_bytes = source_path.read_bytes()
        rows = []
        with source_path.open() as handle:
            for line in handle:
                item = json.loads(line)
                if "messages" in item:
                    rows.append(
                        tokenizer.apply_chat_template(
                            item["messages"],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    )
                else:
                    value = item.get("text") or item.get("prompt")
                    if value:
                        rows.append(value)
        return "\n\n".join(rows), {
            "kind": "jsonl",
            "path": str(source_path),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(rows),
        }

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets or pass --input-jsonl for reference capture"
        ) from exc
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
    )
    rows = [str(row[args.text_field]) for row in dataset if row.get(args.text_field)]
    rows = [row for row in rows if row.strip()]
    return "\n\n".join(rows), {
        "kind": "huggingface_dataset",
        "repo": args.dataset,
        "config": args.dataset_config,
        "split": args.dataset_split,
        "revision": args.dataset_revision,
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "rows": len(rows),
    }


def llm_kwargs(args):
    values = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_logprobs": -1,
    }
    for key in (
        "tokenizer",
        "revision",
        "tokenizer_revision",
        "attention_backend",
        "distributed_executor_backend",
    ):
        value = getattr(args, key)
        if value:
            values[key] = value
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        values["quantization"] = args.quantization
    values.update(json.loads(args.llm_extra_json))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--role",
        choices=("canonical", "candidate"),
        default="canonical",
        help="Semantic capture role; it does not change inference or storage.",
    )
    parser.add_argument(
        "--run-label",
        help="Stable operator label for the weight/topology/KV policy being captured.",
    )
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument(
        "--dataset-revision",
        default="b08601e04326c79dfdd32d625aee71d232d685c3",
        help="Immutable dataset revision (default pins Salesforce/wikitext).",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--storage-dtype",
        choices=("bfloat16", "float32"),
        default="float32",
        help="Keep float32 for publishable KLD and independent NumPy replay.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="Use auto/BF16 for a clean teacher; pass fp8 for deployment KLD.",
    )
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--attention-backend")
    parser.add_argument("--distributed-executor-backend", choices=("mp", "ray"))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--llm-extra-json", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.context_length < 2 or args.stride < 1 or args.max_windows < 1:
        parser.error("context length, stride, and max windows must be positive")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"output directory is not empty: {destination}")

    tokenizer_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    text, source = load_source_text(args, tokenizer)
    needed = args.context_length + (args.max_windows - 1) * args.stride
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=needed,
    )
    if len(token_ids) < args.context_length:
        raise RuntimeError(
            f"source produced {len(token_ids)} tokens, need {args.context_length}"
        )

    params, supports_prompt_logits = sampling_params_for_prompt_logits(SamplingParams)
    engine_request = llm_kwargs(args)
    model = LLM(**engine_request)
    storage_dtype = (
        torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    )
    manifest = {
        "schema": "quant-toolkit.prefill-logits.v1",
        "role": args.role,
        "run_label": args.run_label,
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer": tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "source": source,
        "context_length": args.context_length,
        "stride": args.stride,
        "token_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "first_16_token_ids": token_ids[:16],
        "storage_dtype": args.storage_dtype,
        "engine_request": engine_request,
        "prompt_logits_mode": (
            "return_prompt_logits" if supports_prompt_logits else "flat_prompt_logprobs"
        ),
        "windows": [],
    }
    started = time.time()
    for index, start in enumerate(range(0, len(token_ids), args.stride)):
        if index >= args.max_windows:
            break
        window = token_ids[start : start + args.context_length]
        if len(window) != args.context_length:
            break
        prompt: TokensPrompt = {
            "prompt_token_ids": window,
            "target_token_ids": window[1:],
        }
        output = model.generate([prompt], sampling_params=params)[0]
        logits, are_log_probs = dense_prompt_logits(
            output,
            supports_prompt_logits,
            len(window) - 1,
            int(model.llm_engine.model_config.get_vocab_size()),
        )
        path = destination / f"logits_{index}.safetensors"
        temporary = destination / f".{path.name}.incomplete"
        save_file(
            {
                "logits": logits.to(storage_dtype),
                "input_ids": torch.tensor(window, dtype=torch.int32),
            },
            str(temporary),
        )
        os.replace(temporary, path)
        manifest["windows"].append(
            {
                "index": index,
                "start": start,
                "file": path.name,
                "positions": len(window) - 1,
                "vocab_size": logits.shape[1],
                "stored_values_are_log_probs": are_log_probs,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        print(
            json.dumps(
                {
                    "event": "prefill_capture_window_written",
                    "role": args.role,
                    **manifest["windows"][-1],
                }
            ),
            flush=True,
        )
        del output, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not manifest["windows"]:
        raise RuntimeError("no complete windows were captured")
    manifest["elapsed_sec"] = time.time() - started
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    with open(destination / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "event": "prefill_capture_complete",
                "role": args.role,
                "windows": len(manifest["windows"]),
                "output_dir": str(destination),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
