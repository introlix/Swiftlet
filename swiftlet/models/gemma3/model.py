import os
import json
import gc
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from typing import Tuple, List, Mapping, Union, Sequence, Any
import swiftlet.models.gemma.config as gemma_config
from swiftlet.models.gemma import tokenizer
from swiftlet.models.gemma.model import (
    RMSNorm,
    Linear,
    Sampler,
    precompute_freqs_cis,
    GemmaMLP,
    GemmaAttention,
)
from swiftlet.models.gemma import tokenizer
from swiftlet.kernels.embedding import Embedding
from swiftlet.kernels.siglip_vision import siglip_vision_model
from swiftlet.models.gemma3 import gemma3_preprocessor


class Gemma3Block(nn.Module):
    def __init__(
        self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType
    ):
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


class Gemma3Model(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                raise ValueError(
                    "Gemma architecture is not supported in this version. Use GemmaModel instead."
                )
            elif config.architecture == gemma_config.Architecture.GEMMA_2:
                raise ValueError(
                    "Gemma 3 architecture is not supported in this version. Use Gemma3Model instead."
                )
            elif config.architecture == gemma_config.Architecture.GEMMA_3:
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma3Block(config, attn_type))
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


class Gemma3ForCausalLM(nn.Module):
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
        self.model = Gemma3Model(config)
        self.sampler = Sampler(vocab_size, config)

        if config.rope_wave_length is None:
            raise ValueError("rope_wave_length must be provided for Gemma3.")

        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }

        for attn_type, name in [
            (gemma_config.AttentionType.LOCAL_SLIDING, "local_freqs_cis"),
            (gemma_config.AttentionType.GLOBAL, "global_freqs_cis"),
        ]:
            theta = rope_lengths.get(attn_type, defaults[attn_type])
            self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

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
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.local_freqs_cis.index_select(0, input_positions)
            )
            freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.global_freqs_cis.index_select(0, input_positions)
            )
        else:
            raise ValueError(
                "Gemma 3 architecture is not supported in this version. Use Gemma3ForCausalLM instead."
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
            
            # Embedding layer mapping
            if hf_key == "embed_tokens.weight":
                return "embedder.weight"
            
            # Output layer mapping
            if hf_key == "norm.weight":
                return "norm.weight"  # Adjust if your final norm has different name
            
            # Layer-specific mappings
            if hf_key.startswith("layers."):
                # Extract layer number
                parts = hf_key.split(".")
                layer_idx = parts[1]
                
                # Input layer norm
                if hf_key.endswith("input_layernorm.weight"):
                    return f"model.layers.{layer_idx}.input_layernorm.weight"
                
                # Post attention layer norm
                if hf_key.endswith("post_attention_layernorm.weight"):
                    return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                
                # Attention projections
                if "self_attn" in hf_key:
                    if hf_key.endswith("q_proj.weight"):
                        # For QKV combined projection, you might need to handle this differently
                        return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                    elif hf_key.endswith("k_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                    elif hf_key.endswith("v_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                    elif hf_key.endswith("o_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                    
                    # If your implementation uses combined QKV projection
                    if any(x in hf_key for x in ["q_proj", "k_proj", "v_proj"]):
                        # You might need to combine these - see _combine_qkv_weights below
                        return None  # Handle specially
                
                # MLP projections
                if "mlp" in hf_key:
                    if hf_key.endswith("gate_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                    elif hf_key.endswith("up_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.up_proj.weight"
                    elif hf_key.endswith("down_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            # If no mapping found, return original key
            return hf_key

        def _combine_qkv_weights(raw_weights):
            """Combine separate Q, K, V weights into QKV projection if needed."""
            combined_weights = {}
            qkv_groups = {}
            
            # Group Q, K, V weights by layer
            for key, weight in raw_weights.items():
                if "self_attn" in key and any(proj in key for proj in ["q_proj", "k_proj", "v_proj"]):
                    # Extract layer number
                    layer_match = key.split(".")
                    if len(layer_match) >= 2:
                        layer_idx = layer_match[1]
                        if layer_idx not in qkv_groups:
                            qkv_groups[layer_idx] = {}
                        
                        if "q_proj.weight" in key:
                            qkv_groups[layer_idx]['q'] = weight
                        elif "k_proj.weight" in key:
                            qkv_groups[layer_idx]['k'] = weight
                        elif "v_proj.weight" in key:
                            qkv_groups[layer_idx]['v'] = weight
                else:
                    combined_weights[key] = weight
            
            # Combine Q, K, V weights for each layer
            for layer_idx, qkv_dict in qkv_groups.items():
                if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
                    # Concatenate Q, K, V weights
                    qkv_combined = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
                    combined_weights[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"] = qkv_combined
            
            return combined_weights

        def _remap_keys_and_combine(raw):
            """Remap keys and handle special cases like QKV combination."""
            # First pass: basic key remapping
            remapped = {}
            unmapped_qkv = {}
            
            for k, v in raw.items():
                # Convert to float32 if needed
                if v.dtype == torch.float16:
                    v = v.to(torch.float32)
                
                # Check if this is a QKV weight that needs special handling
                if ("self_attn" in k and any(proj in k for proj in ["q_proj", "k_proj", "v_proj"]) 
                    and hasattr(self, 'model') and hasattr(self.model, 'layers')):
                    # Check if your model uses combined QKV projection
                    layer_idx = k.split(".")[1] if "layers." in k else "0"
                    try:
                        # Try to access the layer to see if it has qkv_proj
                        sample_layer = self.model.layers[0] if hasattr(self.model, 'layers') else None
                        if sample_layer and hasattr(sample_layer.self_attn, 'qkv_proj'):
                            unmapped_qkv[k] = v
                            continue
                    except:
                        pass
                
                new_key = _map_gemma3_keys(k)
                if new_key and new_key != k:
                    remapped[new_key] = v
                elif new_key:
                    remapped[k] = v
            
            # Handle QKV combination if needed
            if unmapped_qkv:
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




# This is taken from https://github.com/google/gemma_pytorch/blob/main/gemma/gemma3_model.py#L30
class Gemma3ForMultimodalLM(nn.Module):
    """Gemma3 model for multimodal causal LM."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_3
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.text_token_embedder = Embedding(
            vocab_size, config.hidden_size, config.quant
        )
        self.model = Gemma3Model(config)
        self.sampler = Sampler(vocab_size, config)

        if config.vision_config is None:
            raise ValueError("vision_config must be provided for Gemma3.")
        self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(
            config.vision_config
        )
        # transformer/embedder/mm_soft_embedding_norm
        self.mm_soft_embedding_norm = RMSNorm(
            config.vision_config.embedding_dim, eps=config.rms_norm_eps
        )
        # transformer/embedder/mm_input_projection
        self.mm_input_projection = Linear(
            config.vision_config.embedding_dim, config.hidden_size, config.quant
        )

        if config.rope_wave_length is None:
            raise ValueError("rope_wave_length must be provided for Gemma3.")
        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
        self._register_freqs_cis(
            "local_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.LOCAL_SLIDING,
                defaults[gemma_config.AttentionType.LOCAL_SLIDING],
            ),
        )
        self._register_freqs_cis(
            "global_freqs_cis",
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.GLOBAL,
                defaults[gemma_config.AttentionType.GLOBAL],
            ),
            rope_scaling_factor=config.rope_scaling_factor,
        )

    def _register_freqs_cis(
        self,
        name: str,
        head_dim: int,
        max_seq_len: int,
        theta: int = 10_000,
        rope_scaling_factor: int = 1,
    ):
        self.register_buffer(
            name,
            precompute_freqs_cis(
                head_dim,
                max_seq_len * 2,
                theta=theta,
                rope_scaling_factor=rope_scaling_factor,
            ),
        )

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,  # B x L
        image_patches: torch.Tensor,  # B x N x C x H x W (3x896x896)
        image_presence_mask: torch.Tensor,  # B x N
        input_positions: torch.Tensor,
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
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
            self.local_freqs_cis.index_select(0, input_positions)
        )
        freqs_cis[gemma_config.AttentionType.GLOBAL] = (
            self.global_freqs_cis.index_select(0, input_positions)
        )
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(
            self.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = hidden_states * normalizer
        if image_patches is not None and self.config.vision_config is not None:
            # the input has images
            B, N, C, H, W = image_patches.shape
            # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
            flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
            image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
            image_embeddings = self.mm_soft_embedding_norm(
                image_embeddings
            )  # (B*N) x U x D
            image_embeddings = self.mm_input_projection(
                image_embeddings
            )  # (B*N) x U x model_dim
            hidden_states = self.populate_image_embeddings(
                hidden_states.clone(),
                image_embeddings.clone(),
                input_token_ids.clone(),
                image_presence_mask.clone(),
            )

        kv_write_indices = input_positions

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
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

    def populate_image_embeddings(
        self,
        hidden_states: torch.Tensor,  # B x L x model_dim
        image_embeddings: torch.Tensor,  # (B*N) x U x model_dim
        input_token_ids: torch.Tensor,  # B x L
        image_presence_mask: torch.Tensor,  # B x N
    ):
        batch_size, seq_len, model_dim = hidden_states.shape
        # Step 1 of 2: Fetch valid image embeddings
        # flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(
            image_presence_mask.flatten(), as_tuple=False
        ).squeeze()
        # num_valid_images x model_dim
        valid_image_embeddings = image_embeddings.index_select(
            0, valid_image_embeddings_indices
        )

        # Step 2 of 2: Replace image embeddings at right places.
        image_placeholder_mask = (
            input_token_ids == self.tokenizer.image_token_placeholder_id
        )
        image_placeholder_indices = (
            image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()
        )

        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(
            -1, self.config.hidden_size
        )
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
        batch_size = input_ids.shape[0]
        causal_mask = torch.tril(
            torch.ones(
                (batch_size, 1, sequence_length, sequence_length),
                dtype=torch.bool,
                device=input_ids.device,
            )
        )
        image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
        # Pad the mask to the left with 0. This is to make sure the boundary
        # detection works correctly. Boundary (starting index of image patch) is
        # detected when the value changes from 0 to 1.
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        # Find the boundary (starting index) of the image tokens patch.
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        # Number the boundary.
        # boundary:
        # [[False, False,  True, False, False],
        #  [False,  True, False,  True, False]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [0, 1, 1, 2, 2]]
        numbered_boundary = torch.cumsum(boundary, dim=-1)

        # image_token_mask:
        # [[False, False,  True,  True, False],
        #  [True,  True, False,  True, True]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [1, 1, 1, 2, 2]]
        # q_block_indices:
        # [[0, 0, 1, 1, 0],
        #  [1, 1, 0, 2, 2]]
        q_block_indices = image_token_mask * numbered_boundary
        kv_block_indices = q_block_indices
        # Test the equality of vertical and horizontal numbered patches
        # to create the bidirectional mask.
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        # The upper triangular matrix's diagonal is shifted by sliding window size
        # before doing logical 'and' with attention mask. This is to make sure the
        # local attention is within the sliding window.
        local_mask = torch.logical_and(
            attention_mask,
            torch.triu(
                torch.ones(
                    (1, 1, sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                diagonal=-(self.config.sliding_window_size - 1),
            ),
        )
        return attention_mask, local_mask

    def generate(
        self,
        prompts: Sequence[Sequence[Union[str, Image.Image]]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model."""
        # Inference only.
        processing_result = gemma3_preprocessor.tokenize_raw_input(
            self.tokenizer, prompts, self.config, output_len, device
        )
        batch_size = processing_result["batch_size"]
        user_input_token_ids = processing_result["user_input_token_ids"]
        image_batch = processing_result["image_batch"]
        min_prompt_len = processing_result["min_prompt_len"]
        max_prompt_len = processing_result["max_prompt_len"]
        total_seq_len = processing_result["max_seq_len"]
        image_presence_mask = processing_result["image_presence_mask"]

        # Create attention mask.
        min_dtype = torch.finfo(self.dtype).min
        if self.config.sliding_window_size is None:
            raise ValueError("gemma 3 model requires sliding_window size")
        boolean_mask, local_boolean_mask = self.create_attention_mask(
            user_input_token_ids, total_seq_len
        )
        mask_tensor = torch.where(
            boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)
        ).contiguous()
        local_mask_tensor = torch.where(
            local_boolean_mask,
            0,
            torch.tensor(min_dtype, dtype=torch.float32, device=device),
        ).contiguous()

        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (
                batch_size,
                total_seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        input_token_ids_tensor = torch.full(
            (batch_size, min_prompt_len),
            self.tokenizer.pad_id,
            dtype=torch.int64,
            device=device,
        )
        token_ids_tensor = user_input_token_ids.to(device)
        for i in range(batch_size):
            p = user_input_token_ids[i]
            input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

        input_positions_tensor = torch.arange(
            0, min_prompt_len, dtype=torch.int64, device=device
        )
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(
            2, input_positions_tensor
        )
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = (
            None
            if not temperature
            else torch.FloatTensor([temperature] * batch_size).to(device)
        )
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(total_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                image_patches=image_batch,
                image_presence_mask=image_presence_mask,
                input_positions=input_positions_tensor,
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
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1
            image_batch = None
            image_presence_mask = None

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            output = tokens
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]
            results.append(self.tokenizer.decode(output))

        return results

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
            
            # Embedding layer mapping
            if hf_key == "embed_tokens.weight":
                return "embedder.weight"
            
            # Output layer mapping
            if hf_key == "norm.weight":
                return "norm.weight"  # Adjust if your final norm has different name
            
            # Layer-specific mappings
            if hf_key.startswith("layers."):
                # Extract layer number
                parts = hf_key.split(".")
                layer_idx = parts[1]
                
                # Input layer norm
                if hf_key.endswith("input_layernorm.weight"):
                    return f"model.layers.{layer_idx}.input_layernorm.weight"
                
                # Post attention layer norm
                if hf_key.endswith("post_attention_layernorm.weight"):
                    return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                
                # Attention projections
                if "self_attn" in hf_key:
                    if hf_key.endswith("q_proj.weight"):
                        # For QKV combined projection, you might need to handle this differently
                        return f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                    elif hf_key.endswith("k_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                    elif hf_key.endswith("v_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                    elif hf_key.endswith("o_proj.weight"):
                        return f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                    
                    # If your implementation uses combined QKV projection
                    if any(x in hf_key for x in ["q_proj", "k_proj", "v_proj"]):
                        # You might need to combine these - see _combine_qkv_weights below
                        return None  # Handle specially
                
                # MLP projections
                if "mlp" in hf_key:
                    if hf_key.endswith("gate_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                    elif hf_key.endswith("up_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.up_proj.weight"
                    elif hf_key.endswith("down_proj.weight"):
                        return f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            # If no mapping found, return original key
            return hf_key

        def _combine_qkv_weights(raw_weights):
            """Combine separate Q, K, V weights into QKV projection if needed."""
            combined_weights = {}
            qkv_groups = {}
            
            # Group Q, K, V weights by layer
            for key, weight in raw_weights.items():
                if "self_attn" in key and any(proj in key for proj in ["q_proj", "k_proj", "v_proj"]):
                    # Extract layer number
                    layer_match = key.split(".")
                    if len(layer_match) >= 2:
                        layer_idx = layer_match[1]
                        if layer_idx not in qkv_groups:
                            qkv_groups[layer_idx] = {}
                        
                        if "q_proj.weight" in key:
                            qkv_groups[layer_idx]['q'] = weight
                        elif "k_proj.weight" in key:
                            qkv_groups[layer_idx]['k'] = weight
                        elif "v_proj.weight" in key:
                            qkv_groups[layer_idx]['v'] = weight
                else:
                    combined_weights[key] = weight
            
            # Combine Q, K, V weights for each layer
            for layer_idx, qkv_dict in qkv_groups.items():
                if 'q' in qkv_dict and 'k' in qkv_dict and 'v' in qkv_dict:
                    # Concatenate Q, K, V weights
                    qkv_combined = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
                    combined_weights[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"] = qkv_combined
            
            return combined_weights

        def _remap_keys_and_combine(raw):
            """Remap keys and handle special cases like QKV combination."""
            # First pass: basic key remapping
            remapped = {}
            unmapped_qkv = {}
            
            for k, v in raw.items():
                # Convert to float32 if needed
                if v.dtype == torch.float16:
                    v = v.to(torch.float32)
                
                # Check if this is a QKV weight that needs special handling
                if ("self_attn" in k and any(proj in k for proj in ["q_proj", "k_proj", "v_proj"]) 
                    and hasattr(self, 'model') and hasattr(self.model, 'layers')):
                    # Check if your model uses combined QKV projection
                    layer_idx = k.split(".")[1] if "layers." in k else "0"
                    try:
                        # Try to access the layer to see if it has qkv_proj
                        sample_layer = self.model.layers[0] if hasattr(self.model, 'layers') else None
                        if sample_layer and hasattr(sample_layer.self_attn, 'qkv_proj'):
                            unmapped_qkv[k] = v
                            continue
                    except:
                        pass
                
                new_key = _map_gemma3_keys(k)
                if new_key and new_key != k:
                    remapped[new_key] = v
                elif new_key:
                    remapped[k] = v
            
            # Handle QKV combination if needed
            if unmapped_qkv:
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
