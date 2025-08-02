import torch
import enum
import dataclasses
from typing import Optional, Sequence

@dataclasses.dataclass
class QwenConfig:
    vocab_size: int = 151936
    max_position_embeddings: int = 131072
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    hidden_size: int = 1536
    intermediate_size: int = 8960
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tokenizer: Optional[str] = None
    use_bias: bool = True
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    initializer_range: float = 0.02
    max_window_layers: int = 28
    rope_theta: float = 1000000.0
    sliding_window_size: int = 131072
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""

def get_config_for_1_5b_v2(tokenizer: Optional[str] = None, dtype: str = 'bfloat16') -> QwenConfig:
    return QwenConfig(dtype=dtype, tokenizer=tokenizer)
