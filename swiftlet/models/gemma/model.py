import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Mapping, List, Union, Optional, Any, Sequence
import swiftlet.models.gemma.config as gemma_config
from swiftlet.kernels.pretrained_model import PreTrainedModel
from swiftlet.kernels.text_generation import TextGeneration
from swiftlet.models.gemma import tokenizer
from swiftlet.kernels.embedding import Embedding
from swiftlet.kernels.rmsnorm import RMSNorm
from swiftlet.kernels.linear import Linear
from swiftlet.kernels.rope import precompute_freqs_cis, apply_rotary_emb


# This is taken from https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L28
class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(
            probs, num_samples=1, replacement=True
        ).squeeze(dim=-1)
        return next_token_ids, logits


class GemmaAttention(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=config.quant,
            quant_type=config.quant_type,
            bias=config.use_bias,
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size, quant=config.quant, quant_type=config.quant_type, bias=config.use_bias
        )

        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = (
            hidden_states_shape  # [batch_size, input_len, hidden_size]
        )

        qkv = self.qkv_proj(
            hidden_states
        )  # [batch_size, input_len, (num_heads + 2 * num_kv_heads) * head_dim]
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(
            batch_size, -1, self.num_heads, self.head_dim
        )  # [batch_size, input_len, n_heads, head_dim]
        xk = xk.view(
            batch_size, -1, self.num_kv_heads, self.head_dim
        )  # [batch_size, input_len, n_local_kv_heads, head_dim]
        xv = xv.view(
            batch_size, -1, self.num_kv_heads, self.head_dim
        )  # [batch_size, input_len, n_local_kv_heads, head_dim]

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Apply positional encoding
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        q = xq.transpose(1, 2)  # [batch_size, n_heads, input_len, head_dim]
        k = key.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        v = value.transpose(1, 2)  # [batch size, n_heads, seq_len, head_dim]

        q.mul_(self.scaling)
        attn_score = torch.matmul(
            q, k.transpose(2, 3)
        )  # [batch_size, n_heads, input_len, head_dim] @ [batch_size, n_heads, head_dim, seq_len] = [batch_size, n_heads, input_len, seq_len]

        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            attn_score = attn_score / self.attn_logit_softcapping
            attn_score = torch.tanh(attn_score)
            attn_score = attn_score * self.attn_logit_softcapping

        attn_score = attn_score + mask
        attn_score = attn_score.softmax(
            dim=-1
        )  # [batch_size, n_heads, input_len, seq_len]

        output = torch.matmul(
            attn_score, v
        )  # [batch_size, n_heads, input_len, seq_len] @ [batch size, n_heads, seq_len, head_dim] = [batch_size, n_heads, input_len, head_dim]
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        )  # [batch_size, input_len, n_heads * head_dim]

        output = self.o_proj(output)  # [batch_size, input_len, hidden_size]

        return output


class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: bool, quant_type: str):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant, quant_type, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, quant, quant_type, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, quant, quant_type, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        outputs = self.down_proj(gate * up)
        return outputs


class GemmaBlock(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()

        self.attn_type = gemma_config.AttentionType.GLOBAL

        self.self_attn = GemmaAttention(config=config, attn_type=self.attn_type)

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
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaBlock(config))
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                raise ValueError(
                    "Gemma 2 architecture is not supported in this version. Use Gemma2Model instead."
                )
            elif config.architecture == gemma_config.Architecture.GEMMA_3:
                raise ValueError(
                    "Gemma 3 architecture is not supported in this version. Use Gemma3TextModel instead."
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
                ffreqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module, PreTrainedModel, TextGeneration):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        custom_patterns = None,
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
        self.model = GemmaModel(config)
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