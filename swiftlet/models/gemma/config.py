# Portion of the code is adapted from https://github.com/google/gemma_pytorch/blob/main/gemma/siglip_vision/config.py
# This file is under the Apache License 2.0

import torch
import enum
import dataclasses
from typing import Optional, Sequence
from swiftlet.models.gemma.image_preprocessor import preprocessor

class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2
    GEMMA_3 = 3

@dataclasses.dataclass
class SiglipVisionModelConfig:
  """Returns the model config for the vision model of Gemma 3 andPaliGemma."""
  # The number of transformer encoder blocks in the siglip encoder model.
  num_hidden_layers: int = 27
  # The dimension of the embedding.
  embedding_dim: int = 1152
  # Whether to use bias in the 2D conv embedding layer.
  embedding_use_bias: bool = True
  # The number of channels in the input images.
  input_channels: int = 3
  # The input image size.
  image_size: int = preprocessor.DEFAULT_IMAGE_SIZE
  # Kernel size of 2D convolution layer.
  conv2d_patch_size = 14
  # The number of attention heads used in the attention layers of the model.
  num_attention_heads: int = 16
  # The number of head dimensions.
  head_dim: int = 72
  # Clarify: is num_key_value same as num_query_groups?
  num_key_value_heads: int = 16
  # The number of query groups for implementing attention.
  num_query_groups: int = 16
  # Clarify: usecase of this field is not clear.
  qkv_use_bias: bool = True
  # Clarify: usecase of this field is not clear.
  output_proj_use_bias: bool = True
  # The dimension of the MLP representations.
  intermediate_size: int = 4304
  # The epsilon used by the layer normalization layers.
  layer_norm_eps: float = 1e-6
  # Clarify: identify if the dtype varies for the siglip vision model.
  dtype: str = 'bfloat16'
  # Whether a quantized version of the model is used.
  quant: bool = False
  # The sequence length of the encoding.
  encoding_sequence_length: int = 256


@dataclasses.dataclass
class GemmaConfig:
    architecture: Architecture = Architecture.GEMMA_1
    vocab_size: int = 256000
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_size: int = 3072
    intermediate_size: int = 24576
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    dtype: str = 'bfloat16'
    quant: bool = False
    tokenizer: Optional[str] = None
    attn_types: Optional[Sequence[AttentionType]] = None
    sliding_window_size: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    attn_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[int] = None
    use_pre_ffw_norm: bool = False
    use_post_ffw_norm: bool = False
    rope_wave_length: dict[AttentionType, int] | None = None
    use_qk_norm: bool = False

    vision_config: SiglipVisionModelConfig | None = None
    rope_scaling_factor: int| None = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""

def get_config_for_1b(tokenizer: str, dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=26,
      num_attention_heads=4,
      num_key_value_heads=1,
      hidden_size=1152,
      intermediate_size=6912,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=512,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=32_768,
      tokenizer=tokenizer,
      use_qk_norm=True,
      vision_config=None,
  )

def get_config_for_2b(tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
    return GemmaConfig(
        dtype=dtype,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384,
        tokenizer=tokenizer,
    )

def get_config_for_2b_v2(tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
    return GemmaConfig(
        dtype=dtype,
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=2304,
        intermediate_size=9216,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 13,
        sliding_window_size=4096,
        tokenizer=tokenizer,
    )

def get_config_for_4b(tokenizer: str, dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=34,
      num_attention_heads=8,
      num_key_value_heads=4,
      hidden_size=2560,
      intermediate_size=10240,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=1024,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      tokenizer=tokenizer,
      use_qk_norm=True,
      vision_config=SiglipVisionModelConfig(),
      rope_scaling_factor=8,
  )

def get_config_for_7b(tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
    return GemmaConfig(dtype=dtype, tokenizer=tokenizer)

def get_config_for_9b(tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
    return GemmaConfig(
        dtype=dtype,
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=3584,
        intermediate_size=14336,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 21,
        sliding_window_size=4096,
        tokenizer=tokenizer,
    )

def get_config_for_12b(tokenizer: str, dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=48,
      num_attention_heads=16,
      num_key_value_heads=8,
      hidden_size=3840,
      intermediate_size=3840 * 8 // 2,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=1024,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=131_072,
      tokenizer=tokenizer,
      use_qk_norm=True,
      vision_config=SiglipVisionModelConfig(),
      rope_scaling_factor=8,
  )

def get_config_for_27b(tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_2,
      num_hidden_layers=46,
      num_attention_heads=32,
      num_key_value_heads=16,
      hidden_size=4608,
      intermediate_size=36864,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      final_logit_softcapping=30.0,
      attn_logit_softcapping=50.0,
      head_dim=128,
      attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 23,
      sliding_window_size=4096,
      query_pre_attn_scalar=144,  # hidden_size / num_attention_heads
      tokenizer=tokenizer,
  )

def get_config_for_27b_v3(tokenizer: str, dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=62,
      num_attention_heads=32,
      num_key_value_heads=16,
      hidden_size=5376,
      intermediate_size=5376 * 8 // 2,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=128,
      query_pre_attn_scalar=5376 // 32,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=1024,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=131_072,
      tokenizer=tokenizer,
      use_qk_norm=True,
      vision_config=SiglipVisionModelConfig(),
      rope_scaling_factor=8,
  )

def get_gemma_config(variant: str, tokenizer: str, dtype: str = 'bfloat16') -> GemmaConfig:
   if variant == '1b':
       return get_config_for_1b(tokenizer, dtype)
   elif variant == '2b':
       return get_config_for_2b(tokenizer, dtype)
   elif variant == '2b_v2':
       return get_config_for_2b_v2(tokenizer, dtype)
   elif variant == '4b':
        return get_config_for_4b(tokenizer, dtype)
   elif variant == '7b':
        return get_config_for_7b(tokenizer, dtype)
   elif variant == '9b':
        return get_config_for_9b(tokenizer, dtype)
   elif variant == '12b':
        return get_config_for_12b(tokenizer, dtype)
   elif variant == '27b':
        return get_config_for_27b(tokenizer, dtype)
   elif variant == '27b_v3':
        return get_config_for_27b_v3(tokenizer, dtype)
   else:
       raise ValueError(f"Unknown Gemma variant: {variant}. Supported variants are: "
                        "'1b', '2b', '2b_v2', '4b', '7b', '9b', '12b', '27b', '27b_v3'.")