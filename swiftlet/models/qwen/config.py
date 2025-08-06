import torch
import enum
import dataclasses
from typing import Optional, Sequence

@dataclasses.dataclass
class QwenConfig:
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
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
    max_window_layers: int = 21
    rope_theta: float = 1000000.0
    sliding_window_size: int = 32768
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"
    quant: bool = False
    quant_type: str = '' # The type of quantization used, e.g., 'int8', 'int4'.
    use_cache: bool = True
    use_sliding_window: bool = False

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""

def get_config_for_1_5b_v2(tokenizer: Optional[str] = None, dtype: str = 'bfloat16') -> QwenConfig:
    return QwenConfig(dtype=dtype, tokenizer=tokenizer)

def get_config_for_7b_v2(tokenizer: Optional[str] = None, dtype: str = 'bfloat16') -> QwenConfig:
    return QwenConfig(
        vocab_size=152064,
        max_position_embeddings=131072,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_size=3584,
        intermediate_size=18944,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tokenizer=tokenizer,
        use_bias=True,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        initializer_range=0.02,
        max_window_layers=28,
        rope_theta=1000000.0,
        sliding_window_size=131072,
        tie_word_embeddings=False,
        dtype=dtype,
        use_cache=True,
        use_sliding_window=False
    )