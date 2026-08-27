# SPDX-License-Identifier: Apache-2.0
"""Environment-gated tensors for offline distribution-fidelity evaluation."""

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

_routed_capture_lock = threading.Lock()
_routed_capture_sequence = 0
_routed_capture_directory: Path | None = None
_routed_capture_last_layer = -1
_routed_capture_step_active = False


def _is_capture_rank() -> bool:
    """Return whether this is the single writer in the TP group."""
    return get_tp_group().is_first_rank


def _request_directory(root: str, req_id: str) -> Path:
    safe_req_id = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in req_id
    )
    request_directory = Path(root) / safe_req_id
    request_directory.mkdir(parents=True, exist_ok=True)
    return request_directory


def _save_tensor(
    tensor: torch.Tensor,
    *,
    output_path: Path,
    key: str,
    metadata: dict[str, str],
) -> None:
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite distribution capture: {output_path}")
    from safetensors.torch import save_file

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file({key: tensor}, str(temporary_path), metadata=metadata)
    os.replace(temporary_path, output_path)


def _next_routed_capture_directory(root: str) -> Path:
    global _routed_capture_sequence

    capture_root = Path(root)
    capture_root.mkdir(parents=True, exist_ok=True)
    while True:
        output = capture_root / f"pass-{_routed_capture_sequence:08d}"
        _routed_capture_sequence += 1
        try:
            output.mkdir()
        except FileExistsError:
            continue
        return output


def set_glm53_routed_capture_step_active(active: bool) -> None:
    """Arm routed evidence only for real scheduler steps, never warmups."""
    global _routed_capture_step_active
    _routed_capture_step_active = bool(active)


def capture_glm53_routed_evidence(
    moe: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> None:
    """Capture full-token GLM MoE inputs and exact vLLM routing decisions.

    GLM's sequence-parallel MoE forward chunks tokens before routing. This hook
    intentionally runs immediately before that chunk and only on TP rank zero,
    where the complete replicated input is still available. It uses the same
    gate and router objects as the serving forward, then stores the factorized
    routed evidence needed to replay full cross-coordinate moments offline.
    Capture launches must use eager execution and exactly one request per step.
    """
    capture_root = os.environ.get("VLLM_GLM53_ROUTED_CAPTURE_DIR")
    if not capture_root or not _routed_capture_step_active or not _is_capture_rank():
        return

    experts = getattr(moe, "experts", None)
    router = getattr(experts, "router", None)
    layer_id = getattr(experts, "layer_id", None)
    if router is None or layer_id is None:
        raise RuntimeError("GLM routed capture requires a MoERunner with a router and layer_id")
    layer_id = int(layer_id)

    global _routed_capture_directory, _routed_capture_last_layer
    with _routed_capture_lock:
        if (
            _routed_capture_directory is None
            or layer_id <= _routed_capture_last_layer
        ):
            _routed_capture_directory = _next_routed_capture_directory(capture_root)
        request_directory = _routed_capture_directory
        _routed_capture_last_layer = layer_id

    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "GLM routed capture requires two-dimensional native BF16 MoE inputs; "
            f"got shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype}"
        )

    # GateLinear returns (router_logits, optional bias). The router call is the
    # exact serving implementation, including sigmoid scoring, correction bias,
    # grouped selection, renormalization, and routed_scaling_factor.
    with torch.no_grad():
        router_logits, _ = moe.gate(hidden_states)
        topk_weights, topk_ids = router.select_experts(
            hidden_states,
            router_logits,
        )

    tensors = {
        "hidden_states": hidden_states.detach().to(device="cpu").contiguous(),
        "router_logits": router_logits.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous(),
        "topk_ids": topk_ids.detach().to(
            device="cpu", dtype=torch.int16
        ).contiguous(),
        "topk_weights": topk_weights.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous(),
    }
    output_path = request_directory / f"layer-{layer_id:04d}.safetensors"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite routed capture: {output_path}")
    from safetensors.torch import save_file

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(
        tensors,
        str(temporary_path),
        metadata={
            "layer_id": str(layer_id),
            "hidden_semantic_point": "input_to_routed_moe_before_sequence_parallel_chunk",
            "router_logit_semantic_point": "pre_sigmoid_gate_linear_output",
            "route_semantic_point": "exact_vllm_router_selection",
            "route_weight_dtype": "float32",
        },
    )
    os.replace(temporary_path, output_path)
    logger.info(
        "Saved GLM routed evidence layer=%d rows=%d to %s",
        layer_id,
        hidden_states.shape[0],
        output_path,
    )


def capture_pre_lm_head_prompt_hidden_states(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_batch: Any,
    prompt_lens: np.ndarray,
) -> None:
    """Store final-normalized prompt rows without evaluating the LM head."""
    capture_root = os.environ.get("VLLM_KLD_HIDDEN_CAPTURE_DIR")
    if not capture_root or not _is_capture_rank():
        return

    normalize = getattr(model, "compute_pre_lm_head_hidden_states", None)
    if normalize is None:
        raise RuntimeError(
            "VLLM_KLD_HIDDEN_CAPTURE_DIR requires a model that exposes "
            "compute_pre_lm_head_hidden_states"
        )

    for batch_index, request_id in enumerate(input_batch.req_ids):
        if not input_batch.is_prefilling_np[batch_index]:
            continue
        state_index = int(input_batch.idx_mapping_np[batch_index])
        row_start = int(input_batch.num_computed_prefill_tokens_np[batch_index])
        prompt_rows = int(prompt_lens[state_index])
        scheduled_rows = int(input_batch.num_scheduled_tokens[batch_index])
        row_end = min(row_start + scheduled_rows, prompt_rows)
        if row_start >= row_end:
            continue

        query_offset = int(input_batch.query_start_loc_np[batch_index])
        row_count = row_end - row_start
        request_hidden_states = hidden_states[
            query_offset : query_offset + row_count
        ]
        normalized = normalize(request_hidden_states)
        if normalized.dtype != torch.bfloat16:
            raise RuntimeError(
                "Pre-LM-head capture requires native BF16 states; got "
                f"{normalized.dtype} for request {request_id}"
            )

        output_path = _request_directory(capture_root, request_id) / (
            f"hidden.rows-{row_start:06d}-{row_end:06d}.safetensors"
        )
        normalized_cpu = normalized.detach().to(device="cpu").contiguous()
        _save_tensor(
            normalized_cpu,
            output_path=output_path,
            key="hidden_states",
            metadata={
                "request_id": request_id,
                "row_start": str(row_start),
                "row_end": str(row_end),
                "semantic_point": "after_final_rmsnorm_before_lm_head",
            },
        )
        logger.info(
            "Saved pre-LM-head rows [%d, %d) shape=%s to %s",
            row_start,
            row_end,
            tuple(normalized_cpu.shape),
            output_path,
        )
