import os
import gc
import json
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from typing import Tuple, Mapping, List, Union, Optional, Any, Sequence
import swiftlet.models.gemma.config as gemma_config
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


class GemmaForCausalLM(nn.Module):
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)
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

        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(
                -1
            )
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (
                batch_size,
                max_seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full(
            (batch_size, max_seq_len), self.tokenizer.pad_id, dtype=torch.int64
        )
        input_token_ids_tensor = torch.full(
            (batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64
        )
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, : len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len]
            )
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(
            device
        )
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(
            torch.float
        )
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        local_mask_tensor = (
            mask_tensor
            + torch.tril(
                torch.full(
                    (1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device
                ),
                diagonal=-self.config.sliding_window_size,
            )
            if self.config.sliding_window_size
            else None
        )
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = (
            local_mask_tensor.index_select(2, input_positions_tensor)
            if local_mask_tensor is not None
            else None
        )
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = (
            None
            if not temperature
            else torch.FloatTensor([temperature] * batch_size).to(device)
        )
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(
                dim=1
            )
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(
                dim=1
            )
            output_token_ids = torch.where(
                curr_prompt_mask, curr_token_ids, next_token_ids
            ).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = (
                local_mask_tensor.index_select(2, input_positions_tensor)
                if local_mask_tensor is not None
                else None
            )
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[
                len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len
            ]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

    def from_pretrained(self, model_path: str, map_location="cpu"):

        def _collect_safetensors_files(path):
            if os.path.isfile(path) and path.endswith(".safetensors"):
                return [path]
            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.isdir(path) and os.path.isfile(idx):
                with open(idx, "r", encoding="utf-8") as f:
                    index = json.load(f)
                return sorted(
                    os.path.join(path, shard) for shard in set(index["weight_map"].values())
                )
            if os.path.isdir(path):
                return sorted([
                    os.path.join(path, f)
                    for f in os.listdir(path) if f.endswith(".safetensors")
                ])
            return []

        def _map_gemma3_keys(hf_key):
            """Map HuggingFace Gemma3 keys to your custom implementation keys."""
            
            # Handle keys that already have 'model.' prefix
            if hf_key.startswith("model."):
                stripped_key = hf_key[6:]  # Remove "model." prefix
                
                # Embedding layer mapping
                if stripped_key == "embed_tokens.weight":
                    return "embedder.weight"
                
                # Output layer mapping
                if stripped_key == "norm.weight":
                    return "model.norm.weight"
                
                # Layer-specific mappings
                if stripped_key.startswith("layers."):
                    parts = stripped_key.split(".")
                    layer_idx = parts[1]
                    
                    # Input layer norm
                    if stripped_key.endswith("input_layernorm.weight"):
                        return f"model.layers.{layer_idx}.input_layernorm.weight"
                    
                    # Post attention layer norm
                    if stripped_key.endswith("post_attention_layernorm.weight"):
                        return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    
                    # Attention projections - ADD .linear for quantized layers
                    if "self_attn" in stripped_key:
                        # Query and Key norm mappings
                        if stripped_key.endswith("q_norm.weight"):
                            return f"model.layers.{layer_idx}.self_attn.query_norm.weight"
                        elif stripped_key.endswith("k_norm.weight"):
                            return f"model.layers.{layer_idx}.self_attn.key_norm.weight"
                        # Individual Q, K, V projections - ADD .linear for quantized layers
                        elif stripped_key.endswith("q_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.q_proj.linear.weight"
                        elif stripped_key.endswith("k_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.k_proj.linear.weight"
                        elif stripped_key.endswith("v_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.v_proj.linear.weight"
                        elif stripped_key.endswith("o_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.o_proj.linear.weight"
                    
                    # MLP projections - ADD .linear for quantized layers
                    if "mlp" in stripped_key:
                        if stripped_key.endswith("gate_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.gate_proj.linear.weight"
                        elif stripped_key.endswith("up_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.up_proj.linear.weight"
                        elif stripped_key.endswith("down_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.down_proj.linear.weight"
            
            # Handle keys without 'model.' prefix (original HF format)
            else:
                # Embedding layer mapping
                if hf_key == "embed_tokens.weight":
                    return "embedder.weight"
                
                # Output layer mapping
                if hf_key == "norm.weight":
                    return "model.norm.weight"
                
                # Layer-specific mappings
                if hf_key.startswith("layers."):
                    parts = hf_key.split(".")
                    layer_idx = parts[1]
                    
                    # Input layer norm
                    if hf_key.endswith("input_layernorm.weight"):
                        return f"model.layers.{layer_idx}.input_layernorm.weight"
                    
                    # Post attention layer norm
                    if hf_key.endswith("post_attention_layernorm.weight"):
                        return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    
                    # Attention projections - ADD .linear for quantized layers
                    if "self_attn" in hf_key:
                        # Query and Key norm mappings
                        if hf_key.endswith("q_norm.weight"):
                            return f"model.layers.{layer_idx}.self_attn.query_norm.weight"
                        elif hf_key.endswith("k_norm.weight"):
                            return f"model.layers.{layer_idx}.self_attn.key_norm.weight"
                        # Individual Q, K, V projections - ADD .linear for quantized layers
                        elif hf_key.endswith("q_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.q_proj.linear.weight"
                        elif hf_key.endswith("k_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.k_proj.linear.weight"
                        elif hf_key.endswith("v_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.v_proj.linear.weight"
                        elif hf_key.endswith("o_proj.weight"):
                            return f"model.layers.{layer_idx}.self_attn.o_proj.linear.weight"
                    
                    # MLP projections - ADD .linear for quantized layers
                    if "mlp" in hf_key:
                        if hf_key.endswith("gate_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.gate_proj.linear.weight"
                        elif hf_key.endswith("up_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.up_proj.linear.weight"
                        elif hf_key.endswith("down_proj.weight"):
                            return f"model.layers.{layer_idx}.mlp.down_proj.linear.weight"
            
            # If no mapping found, return original key
            return hf_key

        def _combine_qkv_weights(raw_weights):
            """Combine separate Q, K, V weights into QKV projection if needed."""
            combined_weights = {}
            qkv_groups = {}
            
            print(f"QKV weights to process: {list(raw_weights.keys())[:5]}...")
            
            # Group Q, K, V weights by layer
            for key, weight in raw_weights.items():
                if "self_attn" in key and any(proj in key for proj in ["q_proj", "k_proj", "v_proj"]):
                    # Extract layer number - handle both formats
                    if key.startswith("model.layers."):
                        parts = key.split(".")
                        layer_idx = parts[2]  # model.layers.X.self_attn...
                    else:
                        parts = key.split(".")
                        layer_idx = parts[1]  # layers.X.self_attn...
                    
                    if layer_idx not in qkv_groups:
                        qkv_groups[layer_idx] = {}
                    
                    if "q_proj.weight" in key:
                        qkv_groups[layer_idx]['q'] = weight
                        print(f"Found Q proj for layer {layer_idx}: {weight.shape}")
                    elif "k_proj.weight" in key:
                        qkv_groups[layer_idx]['k'] = weight
                        print(f"Found K proj for layer {layer_idx}: {weight.shape}")
                    elif "v_proj.weight" in key:
                        qkv_groups[layer_idx]['v'] = weight
                        print(f"Found V proj for layer {layer_idx}: {weight.shape}")
            
            # Combine Q, K, V weights for each layer
            combined_count = 0
            for layer_idx, qkv_dict in qkv_groups.items():
                if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
                    # Concatenate Q, K, V weights along the output dimension (dim=0)
                    qkv_combined = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
                    # FIXED: Add .linear for quantized layers
                    combined_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.linear.weight"
                    combined_weights[combined_key] = qkv_combined
                    combined_count += 1
                else:
                    print(f"⚠️ Incomplete QKV set for layer {layer_idx}: {list(qkv_dict.keys())}")
            
            print(f"Successfully combined QKV weights for {combined_count} layers")
            return combined_weights

        def _remap_keys_and_combine(raw):
            """Remap keys and handle special cases like QKV combination."""
            # First pass: basic key remapping
            remapped = {}
            unmapped_qkv = {}
            
            print(f"Sample raw keys: {list(raw.keys())[:10]}")
            
            for k, v in raw.items():
                # Convert to float32 if needed
                if v.dtype == torch.float16:
                    v = v.to(torch.float32)
                
                # Skip the malformed key
                if k == "model.layers.layers.self_attn.qkv_proj.weight":
                    print(f"⚠️ Skipping malformed key: {k}")
                    continue
                
                # Check if this is a QKV weight that needs special handling
                if ("self_attn" in k and any(proj in k for proj in ["q_proj", "k_proj", "v_proj"]) 
                    and not k.endswith("qkv_proj.weight")):
                    # This is a separate Q/K/V projection - might need combining
                    unmapped_qkv[k] = v
                    continue
                
                new_key = _map_gemma3_keys(k)
                if new_key and new_key != k:
                    remapped[new_key] = v
                elif new_key:
                    remapped[k] = v
            
            # Handle QKV combination if needed
            if unmapped_qkv:
                print(f"Processing {len(unmapped_qkv)} separate QKV weights")
                qkv_combined = _combine_qkv_weights(unmapped_qkv)
                remapped.update(qkv_combined)
            
            return remapped

        def _load_safetensors_file(filepath, device):
            """Load a single safetensors file."""
            # Use load_file directly - it's simpler and more reliable
            return load_safetensors(filepath, device=device)

        # ——— Try safetensors first ———
        safefiles = _collect_safetensors_files(model_path)
        if safefiles:
            print(f"Found {len(safefiles)} safetensors file(s)")
            raw = {}
            for f in safefiles:
                print(f"Loading {f}")
                raw.update(_load_safetensors_file(f, map_location))

            if not raw:
                raise RuntimeError(f"Found safetensors files but no tensors were loaded from {model_path!r}")

            print(f"Loaded {len(raw)} raw tensors from safetensors")
            
            # Apply key remapping and combinations
            sd = _remap_keys_and_combine(raw)
            
            # Add any missing positional embeddings if your model needs them
            if hasattr(self, 'local_freqs_cis') and 'local_freqs_cis' not in sd:
                print("⚠️ local_freqs_cis not found in checkpoint, using model's initialized values")
            if hasattr(self, 'global_freqs_cis') and 'global_freqs_cis' not in sd:
                print("⚠️ global_freqs_cis not found in checkpoint, using model's initialized values")
            
            # Load state dict
            missing, unexpected = self.load_state_dict(sd, strict=False)
            
            print(f"✅ Loaded {len(sd) - len(unexpected)} tensors from safetensors")
            if missing:
                print(f"⚠️ Missing keys ({len(missing)}): {missing[:5]} {'...' if len(missing) > 5 else ''}")
                # Print first few missing keys for debugging
                for key in missing[:10]:
                    print(f"   Missing: {key}")
            if unexpected:
                print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]} {'...' if len(unexpected) > 5 else ''}")
                # Print first few unexpected keys for debugging
                for key in unexpected[:10]:
                    print(f"   Unexpected: {key}")
            
            return

        # ——— Fallback: single-file PyTorch .ckpt/.bin ———
        if os.path.isfile(model_path):
            ckpt = torch.load(model_path, map_location=map_location)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"✅ Loaded PyTorch checkpoint, missing={len(missing)}, unexpected={len(unexpected)}")
            return

        # ——— Fallback: sharded PyTorch folder ———
        idx2 = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isdir(model_path) and os.path.isfile(idx2):
            with open(idx2, "r", encoding="utf-8") as f:
                index = json.load(f)
            all_sd = {}
            for shard in set(index["weight_map"].values()):
                part = torch.load(os.path.join(model_path, shard), map_location=map_location)
                part_sd = part.get("model_state_dict", part)
                all_sd.update(part_sd)
                del part, part_sd
                gc.collect()
            missing, unexpected = self.load_state_dict(all_sd, strict=False)
            print(f"✅ Loaded sharded PyTorch checkpoint, missing={len(missing)}, unexpected={len(unexpected)}")
            return

        raise FileNotFoundError(f"No checkpoint found at '{model_path}'")