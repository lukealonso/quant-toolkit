from .base import ModelQuantConfig


class _Glm53FlashConfig(ModelQuantConfig):
    def get_model_cls(self):
        from transformers import Glm5NextForConditionalGeneration

        return Glm5NextForConditionalGeneration

    def register_moe(self):
        from moe_registry import register_glm5_next_moe_for_quantization

        register_glm5_next_moe_for_quantization()


Glm53FlashConfig = _Glm53FlashConfig(
    model_id="zai-org/GLM-5.3-Flash-BF16",
    streaming=True,
    extra_quant_overrides={
        # Start from a fail-closed configuration. GLM-5.3 adds KDA linear
        # attention, NoPE sparse attention/indexers, hyper-connections, vision,
        # and an MTP layer. Only routed expert linears are enabled below.
        "*weight_quantizer": {"enable": False},
        "*input_quantizer": {"enable": False},
        "model.language_model.layers.*.mlp.experts.gate_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.gate_proj.*.input_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.up_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.up_proj.*.input_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.down_proj.*.weight_quantizer": {"enable": True},
        "model.language_model.layers.*.mlp.experts.down_proj.*.input_quantizer": {"enable": True},
        "*[kv]_bmm_quantizer": {"enable": False},
        "*visual*": {"enable": False},
        "*self_attn*": {"enable": False},
        "*indexer*": {"enable": False},
        "*embed_tokens*": {"enable": False},
        "*lm_head*": {"enable": False},
        "*shared_expert*": {"enable": False},
        "*shared_experts*": {"enable": False},
        "*mlp.gate*": {"enable": False},
        "*attn_hc*": {"enable": False},
        "*ffn_hc*": {"enable": False},
    },
    # Transformers intentionally ignores the speculative layer. Preserve it
    # from the BF16 source until it receives its own measured NVFP4 pass.
    extra_mtp_prefixes=["model.language_model.layers.45."],
)
