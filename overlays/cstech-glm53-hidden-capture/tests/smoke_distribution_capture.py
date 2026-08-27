#!/usr/bin/env python3
"""CPU-only build smoke test for the environment-gated capture module."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from vllm.v1.worker.gpu import distribution_capture


class IdentityFinalNorm(torch.nn.Module):
    def compute_pre_lm_head_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states


class FakeRouter:
    def select_experts(self, hidden_states, router_logits):
        del hidden_states
        weights, ids = torch.topk(router_logits.sigmoid(), k=2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.float(), ids.int()


class FakeExperts:
    layer_id = 3
    router = FakeRouter()


class FakeGate(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states.float() @ torch.arange(12).reshape(4, 3).float(), None


class FakeMoe(torch.nn.Module):
    experts = FakeExperts()
    gate = FakeGate()


def make_batch(computed: int, scheduled: int) -> SimpleNamespace:
    return SimpleNamespace(
        idx_mapping_np=np.array([0]),
        is_prefilling_np=np.array([True]),
        num_computed_prefill_tokens_np=np.array([computed]),
        num_scheduled_tokens=np.array([scheduled]),
        query_start_loc_np=np.array([0, scheduled]),
        req_ids=["capture/request"],
    )


with tempfile.TemporaryDirectory() as temporary_directory:
    os.environ["VLLM_KLD_HIDDEN_CAPTURE_DIR"] = temporary_directory
    distribution_capture._is_capture_rank = lambda: True
    model = IdentityFinalNorm()

    first = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, first, make_batch(0, 2), np.array([5])
    )
    second = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, second, make_batch(2, 3), np.array([5])
    )

    request_directory = Path(temporary_directory) / "capture_request"
    first_path = request_directory / "hidden.rows-000000-000002.safetensors"
    second_path = request_directory / "hidden.rows-000002-000005.safetensors"
    torch.testing.assert_close(load_file(first_path)["hidden_states"], first)
    torch.testing.assert_close(load_file(second_path)["hidden_states"], second)

with tempfile.TemporaryDirectory() as temporary_directory:
    os.environ["VLLM_GLM53_ROUTED_CAPTURE_DIR"] = temporary_directory
    distribution_capture._is_capture_rank = lambda: True
    distribution_capture.set_glm53_routed_capture_step_active(True)
    hidden = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    distribution_capture.capture_glm53_routed_evidence(FakeMoe(), hidden)
    routed_path = (
        Path(temporary_directory) / "pass-00000000" / "layer-0003.safetensors"
    )
    routed = load_file(routed_path)
    assert set(routed) == {
        "hidden_states",
        "router_logits",
        "topk_ids",
        "topk_weights",
    }
    torch.testing.assert_close(routed["hidden_states"], hidden)
    assert routed["router_logits"].shape == (5, 3)
    assert routed["topk_ids"].dtype == torch.int16
    assert routed["topk_weights"].dtype == torch.float32
