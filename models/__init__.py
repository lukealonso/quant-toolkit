from .glm4_7 import Glm4_7Config
from .glm5 import Glm5Config
from .minimax_m25 import MinimaxM25Config
from .qwen3_5_122b import Qwen35_122BConfig
from .qwen3_5_moe import Qwen35MoeConfig
from .qwen3_5_moe_noshared import Qwen35MoeNoSharedConfig

_CONFIGS = {
    "glm4_7": Glm4_7Config,
    "glm5": Glm5Config,
    "minimax_m25": MinimaxM25Config,
    "qwen3_5_122b": Qwen35_122BConfig,
    "qwen3_5_moe": Qwen35MoeConfig,
    "qwen3_5_moe_noshared": Qwen35MoeNoSharedConfig,
}

AVAILABLE_MODELS = list(_CONFIGS.keys())


def load_config(name: str):
    if name not in _CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {AVAILABLE_MODELS}")
    return _CONFIGS[name]
