"""Shared dense-prefill KLD helpers.

The comparison direction is always KL(reference || candidate). Values are
reported in nats and bits; bits are nats / ln(2).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_token_ids_sha256(token_ids: torch.Tensor) -> str:
    """Hash a one-dimensional token array independent of its integer width."""
    if token_ids.ndim != 1:
        raise ValueError(
            f"input token IDs must be rank 1, got {tuple(token_ids.shape)}"
        )
    if token_ids.dtype == torch.bool or token_ids.is_floating_point():
        raise ValueError(f"input token IDs must be integers, got {token_ids.dtype}")
    return canonical_json_sha256(token_ids.to(torch.int64).tolist())


def load_capture_input_ids(
    directory: str | Path,
    record: dict,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Load embedded or externally sealed input IDs for a capture window."""
    if "input_ids" in tensors:
        token_ids = tensors["input_ids"]
    else:
        relative_path = record.get("input_ids_file")
        if not relative_path:
            raise ValueError(
                "capture window has neither embedded nor external input_ids"
            )
        path = Path(directory) / relative_path
        expected_file_sha256 = record.get("input_ids_file_sha256")
        if not expected_file_sha256:
            raise ValueError(f"external input IDs have no file hash: {path}")
        if sha256_file(path) != expected_file_sha256:
            raise ValueError(f"external input-token hash mismatch: {path}")
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"input token IDs must be integers: {path}")
            token_ids = torch.from_numpy(np.asarray(array))
        elif path.suffix == ".safetensors":
            from safetensors.torch import load_file

            external_tensors = load_file(path)
            if "input_ids" not in external_tensors:
                raise ValueError(f"external token artifact has no input_ids: {path}")
            token_ids = external_tensors["input_ids"]
        else:
            raise ValueError(f"unsupported external token artifact: {path}")

    observed_canonical_sha256 = canonical_token_ids_sha256(token_ids)
    expected_canonical_sha256 = record.get("input_ids_canonical_sha256")
    if (
        expected_canonical_sha256
        and observed_canonical_sha256 != expected_canonical_sha256
    ):
        raise ValueError("canonical input-token identity mismatch")
    return token_ids


def sampling_params_for_prompt_logits(SamplingParams):
    """Build parameters for patched or flat-full-logprob vLLM runtimes."""
    supports_prompt_logits = (
        "return_prompt_logits" in inspect.signature(SamplingParams).parameters
    )
    if supports_prompt_logits:
        params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            return_prompt_logits=True,
            detokenize=False,
        )
    else:
        params = SamplingParams(
            prompt_logprobs=-1,
            flat_logprobs=True,
            max_tokens=1,
            detokenize=False,
        )
    return params, supports_prompt_logits


def dense_prompt_logits(output, supports_prompt_logits: bool, npos: int, vocab: int):
    """Return dense CPU logits/log-probabilities for prediction positions."""
    if supports_prompt_logits:
        raw = output.prompt_logits
        if raw is None:
            raise RuntimeError("vLLM returned no prompt_logits")
        if raw.shape[0] < npos or raw.shape[1] < vocab:
            raise ValueError(
                f"prompt logits shape {tuple(raw.shape)} cannot satisfy "
                f"required {(npos, vocab)}"
            )
        return raw[:npos, :vocab].detach().to("cpu", copy=True), False

    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    dense = torch.full((npos, vocab), float("-inf"), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for pos in range(npos):
            src_pos = pos + 1
            start = prompt_logprobs.start_indices[src_pos]
            end = prompt_logprobs.end_indices[src_pos]
            ids = torch.as_tensor(
                prompt_logprobs.token_ids[start:end], dtype=torch.long
            )
            values = torch.as_tensor(
                prompt_logprobs.logprobs[start:end], dtype=torch.float32
            )
            valid = (ids >= 0) & (ids < vocab)
            dense[pos, ids[valid]] = values[valid]
    else:
        for pos in range(npos):
            for token_id, value in prompt_logprobs[pos + 1].items():
                token_id = int(token_id)
                if 0 <= token_id < vocab:
                    dense[pos, token_id] = float(
                        value.logprob if hasattr(value, "logprob") else value
                    )
    return dense, True


def tokenwise_kld(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_rows: int = 8,
    compute_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute tokenwise KL(reference || candidate) and both top-1 IDs."""
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference and candidate logits must both be rank 2")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"logit shape mismatch: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    values = []
    ref_top1 = []
    candidate_top1 = []
    for start in range(0, reference.shape[0], chunk_rows):
        end = min(reference.shape[0], start + chunk_rows)
        ref = reference[start:end].to(compute_dtype)
        cand = candidate[start:end].to(compute_dtype)
        if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
            raise ValueError(f"non-finite dense logits in rows [{start}, {end})")
        log_ref = F.log_softmax(ref, dim=-1)
        log_candidate = F.log_softmax(cand, dim=-1)
        values.append(
            F.kl_div(
                log_candidate,
                log_ref,
                reduction="none",
                log_target=True,
            )
            .sum(dim=-1)
            .to(torch.float64)
        )
        ref_top1.append(ref.argmax(dim=-1).to(torch.int64))
        candidate_top1.append(cand.argmax(dim=-1).to(torch.int64))
    return (
        torch.cat(values),
        torch.cat(ref_top1),
        torch.cat(candidate_top1),
    )


def summarize_kld(
    values_nats: torch.Tensor,
    ref_top1: torch.Tensor,
    candidate_top1: torch.Tensor,
) -> dict:
    if values_nats.numel() == 0:
        raise ValueError("cannot summarize an empty KLD tensor")
    values_nats = values_nats.to(torch.float64)
    values_bits = values_nats / math.log(2.0)

    def stats(values):
        return {
            "mean": float(values.mean()),
            "median": float(torch.quantile(values, 0.5)),
            "p95": float(torch.quantile(values, 0.95)),
            "p99": float(torch.quantile(values, 0.99)),
            "p99_9": float(torch.quantile(values, 0.999)),
            "max": float(values.max()),
        }

    return {
        "direction": "KL(reference||candidate)",
        "positions": int(values_nats.numel()),
        "kld_nats": stats(values_nats),
        "kld_bits": stats(values_bits),
        "top1_agreement": float((ref_top1 == candidate_top1).double().mean()),
    }
