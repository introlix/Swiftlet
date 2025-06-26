import os
import json
import gc
import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors
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


class Gemma2ForCausalLM(nn.Module):
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
        self.model = Gemma2Model(config)
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

    def from_pretrained(self, model_path: str):
        """Load the model state from a given path."""

        # For safetensors
        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            sd = load_safetensors(model_path, device="cpu")
            self.load_state_dict(sd, strict=False)
            return

        # for safetensors shards + index
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isdir(model_path) and os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            # The index maps tensor names to shard filenames
            shards = set(index["weight_map"].values())
            for shard in shards:
                shard_path = os.path.join(model_path, shard)
                sd = load_safetensors(shard_path, device="cpu")
                self.load_state_dict(sd, strict=False)
                del sd
                gc.collect()
            return

        # For PyTorch .bin files
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path,
                    mmap=True,
                    weights_only=True,
                )["model_state_dict"],
                strict=False,
            )

            return

        # For PyTorch shards + index
        index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isdir(model_path) and os.path.isfile(index_file):
            index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(
                    shard_path, map_location="cpu", weights_only=True
                )
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()
            return


