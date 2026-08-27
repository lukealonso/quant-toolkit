from .glm5 import Glm5Config
from .glm5_1 import Glm51Config
from .glm5_2 import Glm52Config
from .glm5_3_flash import Glm53FlashConfig
from .mimo_v25 import MimoV25Config
from .minimax_m25 import MinimaxM25Config
from .minimax_m27 import MinimaxM27Config
from .minimax_m3 import MinimaxM3Config
from .qwen3_5_122b import Qwen35_122BConfig
from .qwen3_5_moe import Qwen35MoeConfig
from .qwen3_5_moe_noshared import Qwen35MoeNoSharedConfig

_CONFIGS = {
    "glm5": Glm5Config,
    "glm5_1": Glm51Config,
    "glm5_2": Glm52Config,
    "glm5_3_flash": Glm53FlashConfig,
    "mimo_v25": MimoV25Config,
    "minimax_m25": MinimaxM25Config,
    "minimax_m27": MinimaxM27Config,
    "minimax_m3": MinimaxM3Config,
    "qwen3_5_122b": Qwen35_122BConfig,
    "qwen3_5_moe": Qwen35MoeConfig,
    "qwen3_5_moe_noshared": Qwen35MoeNoSharedConfig,
}

AVAILABLE_MODELS = list(_CONFIGS.keys())


def load_config(name: str):
    if name not in _CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {AVAILABLE_MODELS}")
    return _CONFIGS[name]
