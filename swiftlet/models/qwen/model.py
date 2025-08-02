import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Mapping, List, Union, Optional
import swiftlet.models.qwen.config as qwen_config
from swiftlet.kernels.linear import Linear
from swiftlet.kernels.rope import precompute_freqs_cis, apply_rotary_emb
from swiftlet.kernels.rmsnorm import RMSNorm
from swiftlet.kernels.embedding import Embedding
from swiftlet.kernels.pretrained_model import PreTrainedModel
from swiftlet.kernels.text_generation import TextGeneration


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: qwen_config.QwenConfig):
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


class QwenAttention(nn.Module):
    def __init__(
        self,
        config: qwen_config.QwenConfig,
        attn_type,
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

        self.use_bias = config.use_bias
        self.scaling = self.head_dim**-0.5
        self.attn_dropout_prob = config.attention_dropout

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=False,
            bias=self.use_bias,
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, self.hidden_size, quant=False, bias=False
        )

        self.attn_dropout = nn.Dropout(p=self.attn_dropout_prob)

        self.sliding_window_size = config.sliding_window_size
        self.use_sliding_window = config.use_sliding_window
        self.attn_type = attn_type

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

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        q.mul_(self.scaling)
        attn_score = torch.matmul(q, k.transpose(2, 3))

        if (
            self.sliding_window_size is not None
            and self.attn_type == "LOCAL"
            and local_mask is not None
            and self.use_sliding_window
            and local_mask is not None
        ):
            mask = local_mask

        attn_score = attn_score + mask
        attn_score = torch.softmax(attn_score, dim=-1)

        attn_score = self.attn_dropout(attn_score)

        output = torch.matmul(attn_score, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)

        output = self.o_proj(output)

        return output


class QwenMLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_act: str, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant=False, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, quant=False, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, quant=False, bias=False)

        if hidden_act.lower() == "silu":
            self.act_fn = F.silu
        elif hidden_act.lower() in ("gelu", "gelu_tanh"):
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.act_fn(gate)
        up = self.up_proj(x)
        outputs = self.down_proj(gate * up)
        return outputs


class QwenBlock(nn.Module):
    def __init__(self, config: qwen_config.QwenConfig):
        super().__init__()

        self.attn_type = "LOCAL" if config.use_sliding_window else "GLOBAL"
        self.self_attn = QwenAttention(
            config=config,
            attn_type=self.attn_type,
        )

        self.mlp = QwenMLP(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            intermediate_size=config.intermediate_size,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, esp=config.rms_norm_eps)
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


class QwenModel(nn.Module):
    def __init__(self, config: qwen_config.QwenConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_window_layers = config.max_window_layers
        self.use_sliding_window = config.use_sliding_window

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(QwenBlock(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if self.use_sliding_window and i >= self.max_window_layers:
                layer.attn_type = "LOCAL"
            else:
                layer.attn_type = "GLOBAL"
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenForCausalLM(nn.Module, PreTrainedModel, TextGeneration):
    def __init__(
        self,
        tokenizer,
        config: qwen_config.QwenConfig,
        custom_patterns=None,
    ):
        super().__init__()
        PreTrainedModel.__init__(self, custom_patterns)

        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer
        self.embedder = Embedding(vocab_size, config.hidden_size, quant=False)
        self.model = QwenModel(config)
        self.sampler = Sampler(vocab_size, config)

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

        freqs_cis["LOCAL"] = (
            self.freqs_cis.index_select(0, input_positions)
        )
        freqs_cis["GLOBAL"] = self.freqs_cis.index_select(
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

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        return next_tokens, logits
