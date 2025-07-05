import torch
from torch import nn
from typing import Tuple, List, Mapping, Union, Sequence, Any
import swiftlet.models.gemma.config as gemma_config
from swiftlet.models.gemma import tokenizer
from swiftlet.models.gemma.model import (
    RMSNorm,
    GemmaAttention,
    GemmaMLP,
    Sampler,
    precompute_freqs_cis,
)
from swiftlet.kernels.pretrained_model import PreTrainedModel
from swiftlet.kernels.text_generation import TextGeneration
from swiftlet.models.gemma import tokenizer
from swiftlet.kernels.embedding import Embedding


class Gemma2Block(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type,
        )

        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
            quant_type=config.quant_type,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2Model(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.global_attn = GemmaAttention(
            config=config, attn_type=gemma_config.AttentionType.GLOBAL
        )
        self.local_attn = GemmaAttention(
            config=config, attn_type=gemma_config.AttentionType.LOCAL_SLIDING
        )
        self.attn_type = [self.local_attn, self.global_attn] * (config.num_hidden_layers // 2)

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                raise ValueError(
                    "Gemma architecture is not supported in this version. Use GemmaModel instead."
                )
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2Block(config, attn_type))
            elif config.architecture == gemma_config.Architecture.GEMMA_3:
                raise ValueError(
                    "Gemma 3 architecture is not supported in this version. Use Gemma3Model instead."
                )
            else:
                raise ValueError(f"Unsupported architecture: {config.architecture}")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma2ForCausalLM(nn.Module, PreTrainedModel, TextGeneration):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        custom_patterns = None
    ):
        super().__init__()
        PreTrainedModel.__init__(self, custom_patterns)
        
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = Gemma2Model(config)
        self.sampler = Sampler(vocab_size, config)

        self.quant = config.quant
        self.qaunt_type = config.quant_type

        self._register_freqs_cis("freqs_cis", head_dim, max_seq_len)

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000
    ):
        self.register_buffer(
            name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
        )

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        freqs_cis = {}

        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            raise ValueError(
                "Gemma 3 architecture is not supported in this version. Use Gemma3ForCausalLM instead."
            )
        else:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.freqs_cis.index_select(0, input_positions)
            )
            freqs_cis[gemma_config.AttentionType.GLOBAL] = self.freqs_cis.index_select(
                0, input_positions
            )

        kv_write_indices = input_positions

        hidden_states = self.embedder(
            input_token_ids
        )  # [batch_size, input_len, hidden_size]
        normalizer = torch.tensor(
            self.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.embedder.weight

        # if self.config.quant:
        #     embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(
        #         -1
        #     )
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits