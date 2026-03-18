"""ModelOpt quantization support for MoE models with fused 3D expert weights.

Supports:
  - MiniMaxM2 (built-in transformers): MiniMaxM2Experts
  - GLM-4 MoE (glm4_moe): Glm4MoeMoE + Glm4MoeNaiveMoe
  - GLM-5 / glm_moe_dsa (remote code): GlmMoeDsaMoE + GlmMoeDsaNaiveMoe
  - Qwen3.5 MoE: Qwen3_5MoeSparseMoeBlock + Qwen3_5MoeExperts

All models use natural top-k routing during calibration. Expert coverage is
achieved through a large, diverse calibration dataset rather than forcing
all-expert activation, which biases downstream scales.
"""
import torch
import torch.nn as nn
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import _QuantSparseMoe


def patch_glm5_attention_indexer():
    """Patch GLM-5 attention indexer layout to match checkpoint keys/shapes."""
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention
    except ImportError:
        print("⚠ glm_moe_dsa attention class not found, skipping indexer patch")
        return

    if getattr(GlmMoeDsaAttention, "_glm5_indexer_patch_applied", False):
        return

    original_init = GlmMoeDsaAttention.__init__

    def patched_init(self, config, layer_idx):
        original_init(self, config, layer_idx)

        if hasattr(self, "indexer"):
            return

        index_n_heads = getattr(config, "index_n_heads", self.num_heads)
        index_head_dim = getattr(config, "index_head_dim", self.qk_head_dim)

        for name in ("wq_b", "wk", "k_norm", "weights_proj"):
            if hasattr(self, name):
                delattr(self, name)

        self.indexer = nn.Module()
        self.indexer.wq_b = nn.Linear(config.q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.indexer.wk = nn.Linear(config.hidden_size, index_head_dim, bias=config.attention_bias)
        self.indexer.k_norm = nn.LayerNorm(index_head_dim)
        self.indexer.weights_proj = nn.Linear(config.hidden_size, index_n_heads, bias=False)

    GlmMoeDsaAttention.__init__ = patched_init
    GlmMoeDsaAttention._glm5_indexer_patch_applied = True
    print("✓ Patched GLM-5 attention indexer layout")


# ---------------------------------------------------------------------------
# Shared: unfuse 3D expert params into per-expert nn.Linear.
# ---------------------------------------------------------------------------

class _QuantFusedExperts(QuantModule):
    """Unfuse 3D expert params into per-expert nn.Linear for quantization."""

    def _setup(self):
        from accelerate import init_empty_weights

        dtype, device = self.gate_up_proj.dtype, self.gate_up_proj.device
        I = self.intermediate_dim
        H = self.hidden_dim

        def _copy_weight(module, weight):
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().data.to(dtype=dtype, device=device)

        with init_empty_weights():
            gate_proj = nn.ModuleList(
                [nn.Linear(H, I, bias=False) for _ in range(self.num_experts)]
            )
            up_proj = nn.ModuleList(
                [nn.Linear(H, I, bias=False) for _ in range(self.num_experts)]
            )
            down_proj = nn.ModuleList(
                [nn.Linear(I, H, bias=False) for _ in range(self.num_experts)]
            )

        for idx in range(self.num_experts):
            _copy_weight(gate_proj[idx], self.gate_up_proj[idx, :I, :])
            _copy_weight(up_proj[idx], self.gate_up_proj[idx, I:, :])
            _copy_weight(down_proj[idx], self.down_proj[idx])

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = self.gate_proj[expert_idx](current_state)
            up = self.up_proj[expert_idx](current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


# ---------------------------------------------------------------------------
# GLM-4 MoE (glm4_moe).
# ---------------------------------------------------------------------------

class _QuantGlm4MoeMoE(_QuantSparseMoe):
    @property
    def num_experts(self):
        return self.n_routed_experts

    def forward(self, hidden_states):
        return super(_QuantSparseMoe, self).forward(hidden_states)


def register_glm4_7_moe_for_quantization():
    """Register GLM-4 MoE (glm4_moe) classes with modelopt."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeMoE,
            Glm4MoeNaiveMoe,
        )
    except ImportError:
        print("⚠ glm4_moe not in transformers, skipping GLM-4 registration")
        return

    if QuantModuleRegistry.get(Glm4MoeMoE) is None:
        QuantModuleRegistry.register(
            {Glm4MoeMoE: "Glm4MoeMoE"}
        )(_QuantGlm4MoeMoE)

    if QuantModuleRegistry.get(Glm4MoeNaiveMoe) is None:
        QuantModuleRegistry.register(
            {Glm4MoeNaiveMoe: "Glm4MoeNaiveMoe"}
        )(_QuantFusedExperts)

    print("✓ Registered GLM-4 MoE for quantization")


# ---------------------------------------------------------------------------
# GLM-5 (glm_moe_dsa).
# ---------------------------------------------------------------------------

class _QuantGlmMoeDsaMoE(_QuantSparseMoe):
    @property
    def num_experts(self):
        return self.n_routed_experts

    def forward(self, hidden_states):
        return super(_QuantSparseMoe, self).forward(hidden_states)


def register_glm5_moe_for_quantization():
    """Register GLM-5 (glm_moe_dsa) MoE classes with modelopt."""
    patch_glm5_attention_indexer()
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaMoE,
            GlmMoeDsaNaiveMoe,
        )
    except ImportError:
        print("⚠ glm_moe_dsa not in transformers, skipping GLM-5 registration")
        return

    if QuantModuleRegistry.get(GlmMoeDsaMoE) is None:
        QuantModuleRegistry.register(
            {GlmMoeDsaMoE: "GlmMoeDsaMoE"}
        )(_QuantGlmMoeDsaMoE)

    if QuantModuleRegistry.get(GlmMoeDsaNaiveMoe) is None:
        QuantModuleRegistry.register(
            {GlmMoeDsaNaiveMoe: "GlmMoeDsaNaiveMoe"}
        )(_QuantFusedExperts)

    print("✓ Registered GLM-5 MoE for quantization")


# ---------------------------------------------------------------------------
# Qwen3.5 MoE.
# ---------------------------------------------------------------------------

class _QuantQwen35MoeSparseMoeBlock(_QuantSparseMoe):
    @property
    def num_experts(self):
        return self.experts.num_experts

    def forward(self, hidden_states):
        return super(_QuantSparseMoe, self).forward(hidden_states)


def register_qwen35_moe_for_quantization():
    """Register Qwen3.5 MoE (qwen3_5_moe) classes with modelopt."""
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeSparseMoeBlock,
            Qwen3_5MoeExperts,
        )
    except ImportError:
        print("⚠ qwen3_5_moe not in transformers, skipping Qwen3.5 registration")
        return

    if QuantModuleRegistry.get(Qwen3_5MoeSparseMoeBlock) is None:
        QuantModuleRegistry.register(
            {Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"}
        )(_QuantQwen35MoeSparseMoeBlock)

    if QuantModuleRegistry.get(Qwen3_5MoeExperts) is None:
        QuantModuleRegistry.register(
            {Qwen3_5MoeExperts: "Qwen3_5MoeExperts"}
        )(_QuantFusedExperts)

    print("✓ Registered Qwen3.5 MoE for quantization")
