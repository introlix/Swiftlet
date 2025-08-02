import os
import json
import gc
import torch
import re
import warnings
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher


class UniversalParameterMapper:
    """
    Universal parameter mapper that automatically handles various naming conventions
    for multiple LLM architectures including Gemma2, Qwen, and others.
    """

    def __init__(self, custom_patterns=None):
        # Comprehensive transformation patterns for multiple architectures
        self.base_patterns = [
            # ================================
            # BASIC WRAPPER PATTERNS
            # ================================
            # Handle model prefixes
            (r"^module\.", ""),  # Remove DataParallel wrapper
            (r"^model\.", ""),  # Remove model wrapper
            (r"^", "model."),  # Add model prefix
            
            # ================================
            # ATTENTION LAYER PATTERNS
            # ================================
            # Standard self-attention patterns (bidirectional)
            (r"(.*)\.self_attn\.q_proj\.linear\.(weight|bias)", r"\1.self_attn.q_proj.\2"),
            (r"(.*)\.self_attn\.k_proj\.linear\.(weight|bias)", r"\1.self_attn.k_proj.\2"),
            (r"(.*)\.self_attn\.v_proj\.linear\.(weight|bias)", r"\1.self_attn.v_proj.\2"),
            (r"(.*)\.self_attn\.o_proj\.linear\.(weight|bias)", r"\1.self_attn.o_proj.\2"),
            (r"(.*)\.self_attn\.q_proj\.(weight|bias)", r"\1.self_attn.q_proj.linear.\2"),
            (r"(.*)\.self_attn\.k_proj\.(weight|bias)", r"\1.self_attn.k_proj.linear.\2"),
            (r"(.*)\.self_attn\.v_proj\.(weight|bias)", r"\1.self_attn.v_proj.linear.\2"),
            (r"(.*)\.self_attn\.o_proj\.(weight|bias)", r"\1.self_attn.o_proj.linear.\2"),
            
            # Global attention patterns for Gemma2 (bidirectional)
            (r"(.*)\.global_attn\.q_proj\.linear\.(weight|bias)", r"\1.global_attn.q_proj.\2"),
            (r"(.*)\.global_attn\.k_proj\.linear\.(weight|bias)", r"\1.global_attn.k_proj.\2"),
            (r"(.*)\.global_attn\.v_proj\.linear\.(weight|bias)", r"\1.global_attn.v_proj.\2"),
            (r"(.*)\.global_attn\.o_proj\.linear\.(weight|bias)", r"\1.global_attn.o_proj.\2"),
            (r"(.*)\.global_attn\.q_proj\.(weight|bias)", r"\1.global_attn.q_proj.linear.\2"),
            (r"(.*)\.global_attn\.k_proj\.(weight|bias)", r"\1.global_attn.k_proj.linear.\2"),
            (r"(.*)\.global_attn\.v_proj\.(weight|bias)", r"\1.global_attn.v_proj.linear.\2"),
            (r"(.*)\.global_attn\.o_proj\.(weight|bias)", r"\1.global_attn.o_proj.linear.\2"),
            
            # Quantized layer patterns (using '.layer' submodule)
            (r"(.*)\.self_attn\.q_proj\.layer\.(weight|bias)", r"\1.self_attn.q_proj.\2"),
            (r"(.*)\.self_attn\.k_proj\.layer\.(weight|bias)", r"\1.self_attn.k_proj.\2"),
            (r"(.*)\.self_attn\.v_proj\.layer\.(weight|bias)", r"\1.self_attn.v_proj.\2"),
            (r"(.*)\.self_attn\.o_proj\.layer\.(weight|bias)", r"\1.self_attn.o_proj.\2"),
            (r"(.*)\.self_attn\.q_proj\.(weight|bias)", r"\1.self_attn.q_proj.layer.\2"),
            (r"(.*)\.self_attn\.k_proj\.(weight|bias)", r"\1.self_attn.k_proj.layer.\2"),
            (r"(.*)\.self_attn\.v_proj\.(weight|bias)", r"\1.self_attn.v_proj.layer.\2"),
            (r"(.*)\.self_attn\.o_proj\.(weight|bias)", r"\1.self_attn.o_proj.layer.\2"),
            
            # Global attention quantized patterns
            (r"(.*)\.global_attn\.q_proj\.layer\.(weight|bias)", r"\1.global_attn.q_proj.\2"),
            (r"(.*)\.global_attn\.k_proj\.layer\.(weight|bias)", r"\1.global_attn.k_proj.\2"),
            (r"(.*)\.global_attn\.v_proj\.layer\.(weight|bias)", r"\1.global_attn.v_proj.\2"),
            (r"(.*)\.global_attn\.o_proj\.layer\.(weight|bias)", r"\1.global_attn.o_proj.\2"),
            (r"(.*)\.global_attn\.q_proj\.(weight|bias)", r"\1.global_attn.q_proj.layer.\2"),
            (r"(.*)\.global_attn\.k_proj\.(weight|bias)", r"\1.global_attn.k_proj.layer.\2"),
            (r"(.*)\.global_attn\.v_proj\.(weight|bias)", r"\1.global_attn.v_proj.layer.\2"),
            (r"(.*)\.global_attn\.o_proj\.(weight|bias)", r"\1.global_attn.o_proj.layer.\2"),
            
            # ================================
            # QWEN-SPECIFIC ATTENTION PATTERNS
            # ================================
            # Qwen attention patterns (c_attn is combined QKV, c_proj is output)
            (r"(.*)\.attn\.c_attn\.(weight|bias)", r"\1.self_attn.qkv_proj.\2"),
            (r"(.*)\.attn\.c_proj\.(weight|bias)", r"\1.self_attn.o_proj.\2"),
            (r"(.*)\.self_attn\.qkv_proj\.(weight|bias)", r"\1.attn.c_attn.\2"),
            (r"(.*)\.self_attn\.o_proj\.(weight|bias)", r"\1.attn.c_proj.\2"),
            
            # Additional Qwen attention variations
            (r"(.*)\.attention\.(weight|bias)", r"\1.self_attn.\2"),
            (r"(.*)\.self_attn\.(weight|bias)", r"\1.attention.\2"),
            (r"(.*)\.attn\.(weight|bias)", r"\1.self_attn.\2"),
            (r"(.*)\.self_attn\.(weight|bias)", r"\1.attn.\2"),
            
            # ================================
            # QKV COMBINATION PATTERNS
            # ================================
            # Standard QKV patterns (all layer types)
            (r"(.*)\.qkv_proj\.linear\.(weight|bias)", r"\1.qkv_proj.\2"),
            (r"(.*)\.qkv_proj\.(weight|bias)", r"\1.qkv_proj.linear.\2"),
            (r"(.*)\.qkv_proj\.layer\.(weight|bias)", r"\1.qkv_proj.\2"),
            (r"(.*)\.qkv_proj\.(weight|bias)", r"\1.qkv_proj.layer.\2"),
            
            # Global attention QKV patterns
            (r"(.*)\.global_attn\.qkv_proj\.linear\.(weight|bias)", r"\1.global_attn.qkv_proj.\2"),
            (r"(.*)\.global_attn\.qkv_proj\.(weight|bias)", r"\1.global_attn.qkv_proj.linear.\2"),
            (r"(.*)\.global_attn\.qkv_proj\.layer\.(weight|bias)", r"\1.global_attn.qkv_proj.\2"),
            (r"(.*)\.global_attn\.qkv_proj\.(weight|bias)", r"\1.global_attn.qkv_proj.layer.\2"),
            
            # ================================
            # MLP/FFN PATTERNS
            # ================================
            # Standard MLP patterns (bidirectional)
            (r"(.*)\.mlp\.gate_proj\.linear\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.up_proj\.linear\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.down_proj\.linear\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.gate_proj.linear.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.up_proj.linear.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.down_proj.linear.\2"),
            
            # Quantized MLP patterns
            (r"(.*)\.mlp\.gate_proj\.layer\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.up_proj\.layer\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.down_proj\.layer\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.gate_proj.layer.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.up_proj.layer.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.down_proj.layer.\2"),
            
            # Qwen MLP patterns (w1=gate, w2=down, w3=up)
            (r"(.*)\.mlp\.w1\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.w2\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.w3\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.w1.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.w2.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.w3.\2"),
            
            # Alternative MLP naming
            (r"(.*)\.feed_forward\.w1\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.feed_forward\.w2\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.feed_forward\.w3\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            
            # ================================
            # EMBEDDING PATTERNS
            # ================================
            # Token embeddings
            (r"embed_tokens\.weight", "embedder.weight"),
            (r"embedder\.weight", "embed_tokens.weight"),
            (r"word_embeddings\.weight", "embedder.weight"),
            (r"token_embeddings\.weight", "embedder.weight"),
            (r"transformer\.wte\.weight", "model.embed_tokens.weight"),
            (r"model\.embed_tokens\.weight", "transformer.wte.weight"),
            (r"embeddings\.word_embeddings\.weight", "model.embed_tokens.weight"),
            
            # Position embeddings
            (r"embeddings\.position_embeddings\.weight", "model.embed_positions.weight"),
            (r"transformer\.wpe\.weight", "model.embed_positions.weight"),
            
            # ================================
            # FREQUENCY/ROTARY EMBEDDING PATTERNS
            # ================================
            # Frequency embeddings for Gemma2/RoPE
            (r"^freqs_cis$", "model.freqs_cis"),
            (r"^model\.freqs_cis$", "freqs_cis"),
            (r"^rope\.freqs_cis$", "freqs_cis"),
            (r"^freqs_cis$", "rope.freqs_cis"),
            (r"^rotary_emb\.inv_freq$", "freqs_cis"),
            (r"^freqs_cis$", "rotary_emb.inv_freq"),
            (r"^embed_positions\.inv_freq$", "freqs_cis"),
            
            # ================================
            # NORMALIZATION PATTERNS
            # ================================
            # Query/Key normalization
            (r"(.*)\.query_norm\.weight", r"\1.q_norm.weight"),
            (r"(.*)\.key_norm\.weight", r"\1.k_norm.weight"),
            (r"(.*)\.q_norm\.weight", r"\1.query_norm.weight"),
            (r"(.*)\.k_norm\.weight", r"\1.key_norm.weight"),
            
            # Layer normalization variations
            (r"(.*)\.input_layernorm\.weight", r"\1.input_layer_norm.weight"),
            (r"(.*)\.input_layer_norm\.weight", r"\1.input_layernorm.weight"),
            (r"(.*)\.post_attention_layernorm\.weight", r"\1.post_attn_norm.weight"),
            (r"(.*)\.post_attn_norm\.weight", r"\1.post_attention_layernorm.weight"),
            (r"(.*)\.layer_norm\.weight", r"\1.layernorm.weight"),
            (r"(.*)\.layernorm\.weight", r"\1.layer_norm.weight"),
            
            # Qwen-specific normalization
            (r"(.*)\.ln_1\.weight", r"\1.input_layernorm.weight"),
            (r"(.*)\.ln_2\.weight", r"\1.post_attention_layernorm.weight"),
            (r"(.*)\.input_layernorm\.weight", r"\1.ln_1.weight"),
            (r"(.*)\.post_attention_layernorm\.weight", r"\1.ln_2.weight"),
            
            # Final/output normalization
            (r"^norm\.weight", "model.norm.weight"),
            (r"^model\.norm\.weight", "norm.weight"),
            (r"final_layer_norm\.weight", "norm.weight"),
            (r"transformer\.ln_f\.weight", "model.norm.weight"),
            (r"model\.norm\.weight", "transformer.ln_f.weight"),
            
            # ================================
            # LAYER INDEXING PATTERNS
            # ================================
            # Layer indexing variations
            (r"layers\.(\d+)\.", r"layer.\1."),
            (r"layer\.(\d+)\.", r"layers.\1."),
            (r"h\.(\d+)\.", r"layers.\1."),
            (r"transformer\.h\.(\d+)\.", r"model.layers.\1."),
            (r"model\.layers\.(\d+)\.", r"transformer.h.\1."),
            (r"blocks\.(\d+)\.", r"layers.\1."),
            (r"decoder\.layers\.(\d+)\.", r"model.layers.\1."),
            
            # ================================
            # OUTPUT HEAD PATTERNS
            # ================================
            # Language model head
            (r"lm_head\.weight", "model.lm_head.weight"),
            (r"model\.lm_head\.weight", "lm_head.weight"),
            (r"output\.weight", "lm_head.weight"),
            (r"classifier\.weight", "lm_head.weight"),
            
            # ================================
            # ARCHITECTURE-SPECIFIC PATTERNS
            # ================================
            # Handle different attention naming conventions
            (r"(.*)\.attention\.", r"\1.self_attn."),
            (r"(.*)\.self_attn\.", r"\1.attention."),
            (r"(.*)\.attn\.", r"\1.self_attn."),
            (r"(.*)\.self_attn\.", r"\1.attn."),
            
            # Handle different MLP naming conventions
            (r"(.*)\.feed_forward\.", r"\1.mlp."),
            (r"(.*)\.mlp\.", r"\1.feed_forward."),
            (r"(.*)\.ffn\.", r"\1.mlp."),
            (r"(.*)\.mlp\.", r"\1.ffn."),
        ]

        # Add custom patterns if provided
        self.patterns = self.base_patterns.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern), replacement) for pattern, replacement in self.patterns
        ]

    def generate_key_variants(self, key: str) -> List[str]:
        """Generate all possible variants of a parameter key"""
        variants = {key}  # Start with original

        # Apply all transformation patterns iteratively
        for _ in range(3):  # Multiple iterations to catch chain transformations
            new_variants = set()
            for pattern_re, replacement in self.compiled_patterns:
                for variant in variants:
                    if pattern_re.search(variant):
                        new_variant = pattern_re.sub(replacement, variant)
                        new_variants.add(new_variant)
            variants.update(new_variants)

        # Add common prefix/suffix variations
        additional_variants = set()
        for variant in list(variants):
            # Module wrapper variations
            if variant.startswith("module."):
                additional_variants.add(variant[7:])
            else:
                additional_variants.add("module." + variant)

            # Model wrapper variations
            if variant.startswith("model."):
                additional_variants.add(variant[6:])
            elif not variant.startswith("module."):
                additional_variants.add("model." + variant)

        variants.update(additional_variants)
        return list(variants)

    def calculate_similarity_score(self, key1: str, key2: str) -> float:
        """Calculate structural similarity between parameter names"""
        # Basic string similarity
        base_score = SequenceMatcher(None, key1, key2).ratio()

        # Structural similarity (component matching)
        parts1 = key1.split(".")
        parts2 = key2.split(".")

        # Penalize very different lengths
        len_diff = abs(len(parts1) - len(parts2))
        len_penalty = min(len_diff * 0.1, 0.3)

        # Bonus for matching endings (weight/bias matching is important)
        ending_bonus = 0
        if parts1 and parts2:
            if parts1[-1] == parts2[-1]:
                ending_bonus = 0.2
            elif parts1[-1] in ["weight", "bias"] and parts2[-1] in ["weight", "bias"]:
                ending_bonus = 0.1

        # Bonus for matching layer numbers
        layer_bonus = 0
        layer_match1 = re.search(r"\.(\d+)\.", key1)
        layer_match2 = re.search(r"\.(\d+)\.", key2)
        if (
            layer_match1
            and layer_match2
            and layer_match1.group(1) == layer_match2.group(1)
        ):
            layer_bonus = 0.1

        return base_score + ending_bonus + layer_bonus - len_penalty

    def find_best_matches(
        self,
        source_keys: List[str],
        target_keys: List[str],
        source_dict: Dict,
        target_dict: Dict,
    ) -> Dict[str, str]:
        """Find the best parameter mappings using multiple strategies"""
        target_keys_set = set(target_keys)
        mappings = {}
        used_targets = set()

        for source_key in source_keys:
            if source_key in used_targets:
                continue

            best_match = None
            best_score = 0

            # Strategy 1: Try all generated variants
            variants = self.generate_key_variants(source_key)
            for variant in variants:
                if variant in target_keys_set and variant not in used_targets:
                    # Check shape compatibility
                    if (
                        source_key in source_dict
                        and variant in target_dict
                        and source_dict[source_key].shape == target_dict[variant].shape
                    ):
                        score = 1.0  # Perfect match
                        if score > best_score:
                            best_score = score
                            best_match = variant

            # Strategy 2: Fuzzy matching if no exact variant match
            if not best_match:
                for target_key in target_keys_set:
                    if target_key in used_targets:
                        continue

                    # Check shape compatibility first
                    if (
                        source_key in source_dict
                        and target_key in target_dict
                        and source_dict[source_key].shape
                        == target_dict[target_key].shape
                    ):

                        score = self.calculate_similarity_score(source_key, target_key)
                        if (
                            score > best_score and score > 0.7
                        ):  # Threshold for fuzzy matching
                            best_score = score
                            best_match = target_key

            if best_match:
                mappings[source_key] = best_match
                used_targets.add(best_match)

        return mappings

    def handle_qkv_combination(
        self, source_dict: Dict, target_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Handle QKV weight and bias combination automatically for all attention types."""
        qkv_combinations = {}
        qkv_groups = defaultdict(dict)

        # Group Q, K, V weights and biases by layer for all attention types
        attention_types = ["self_attn", "global_attn", "attn", "attention"]
        projection_types = ["q_proj", "k_proj", "v_proj", "query", "key", "value"]
        
        for key, tensor in source_dict.items():
            # Check if this is an attention-related parameter
            has_attention = any(attn_type in key for attn_type in attention_types)
            has_projection = any(proj in key for proj in projection_types)
            
            if has_attention and has_projection:
                # Extract layer number and attention type
                layer_match = re.search(r"layers?\.(\d+)\.", key)
                if not layer_match:
                    layer_match = re.search(r"h\.(\d+)\.", key)  # Qwen style
                if not layer_match:
                    continue
                    
                layer_num = layer_match.group(1)
                
                # Determine attention type
                if "global_attn" in key:
                    attn_type = "global_attn"
                elif "self_attn" in key:
                    attn_type = "self_attn"
                elif "attention" in key:
                    attn_type = "attention"
                else:
                    attn_type = "attn"
                
                layer_id = f"{layer_num}.{attn_type}"

                # Handle both weights and biases
                if key.endswith(".weight"):
                    if any(q_name in key for q_name in ["q_proj", "query"]):
                        qkv_groups[layer_id]["q_weight"] = (key, tensor)
                    elif any(k_name in key for k_name in ["k_proj", "key"]):
                        qkv_groups[layer_id]["k_weight"] = (key, tensor)
                    elif any(v_name in key for v_name in ["v_proj", "value"]):
                        qkv_groups[layer_id]["v_weight"] = (key, tensor)
                elif key.endswith(".bias"):
                    if any(q_name in key for q_name in ["q_proj", "query"]):
                        qkv_groups[layer_id]["q_bias"] = (key, tensor)
                    elif any(k_name in key for k_name in ["k_proj", "key"]):
                        qkv_groups[layer_id]["k_bias"] = (key, tensor)
                    elif any(v_name in key for v_name in ["v_proj", "value"]):
                        qkv_groups[layer_id]["v_bias"] = (key, tensor)

        # Check if target expects combined QKV
        target_expects_qkv = any("qkv_proj" in key or "c_attn" in key for key in target_dict.keys())

        if target_expects_qkv:
            for layer_id, qkv_dict in qkv_groups.items():
                # Parse layer_id to get layer number and attention type
                layer_num, attn_type = layer_id.split(".")

                # Process weights
                if (
                    "q_weight" in qkv_dict
                    and "k_weight" in qkv_dict
                    and "v_weight" in qkv_dict
                ):
                    q_key, q_tensor = qkv_dict["q_weight"]
                    k_key, k_tensor = qkv_dict["k_weight"]
                    v_key, v_tensor = qkv_dict["v_weight"]

                    combined_weight = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

                    # Generate potential target keys for different architectures
                    potential_keys = [
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.weight",
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.linear.weight",
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.layer.weight",
                        f"transformer.h.{layer_num}.attn.c_attn.weight",
                        f"layers.{layer_num}.{attn_type}.qkv_proj.weight",
                    ]
                    
                    # Generate all variants for each potential key
                    all_variants = []
                    for base_key in potential_keys:
                        all_variants.extend(self.generate_key_variants(base_key))

                    for qkv_key in all_variants:
                        if qkv_key in target_dict:
                            if combined_weight.shape == target_dict[qkv_key].shape:
                                qkv_combinations[qkv_key] = combined_weight
                                break  # Found a match, move to next layer

                # Process biases (if they exist in source)
                if (
                    "q_bias" in qkv_dict
                    and "k_bias" in qkv_dict
                    and "v_bias" in qkv_dict
                ):
                    q_bias_key, q_bias_tensor = qkv_dict["q_bias"]
                    k_bias_key, k_bias_tensor = qkv_dict["k_bias"]
                    v_bias_key, v_bias_tensor = qkv_dict["v_bias"]

                    combined_bias = torch.cat([q_bias_tensor, k_bias_tensor, v_bias_tensor], dim=0)

                    # Generate potential bias target keys
                    potential_bias_keys = [
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.bias",
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.linear.bias",
                        f"model.layers.{layer_num}.{attn_type}.qkv_proj.layer.bias",
                        f"transformer.h.{layer_num}.attn.c_attn.bias",
                        f"layers.{layer_num}.{attn_type}.qkv_proj.bias",
                    ]
                    
                    # Generate all variants for each potential bias key
                    all_bias_variants = []
                    for base_key in potential_bias_keys:
                        all_bias_variants.extend(self.generate_key_variants(base_key))

                    for qkv_bias_key in all_bias_variants:
                        if qkv_bias_key in target_dict:
                            if combined_bias.shape == target_dict[qkv_bias_key].shape:
                                qkv_combinations[qkv_bias_key] = combined_bias
                                break  # Found a match, move to next layer

        return qkv_combinations


class PreTrainedModel:
    """
    Universal pretrained model class with automatic parameter mapping
    for multiple LLM architectures including Gemma2, Qwen, and others.
    """

    def __init__(self, custom_patterns=None):
        self.mapper = UniversalParameterMapper(custom_patterns)

    def from_pretrained(
        self, model_path: str, map_location="cpu", strict=False, verbose=True
    ):
        """
        Load model with automatic parameter mapping

        Args:
            model_path: Path to model checkpoint
            map_location: Device to load tensors to
            strict: Whether to enforce strict parameter matching
            verbose: Whether to print detailed loading information
        """

        def _collect_safetensors_files(path):
            """Collect all safetensors files from a path"""
            if os.path.isfile(path) and path.endswith(".safetensors"):
                return [path]
            
            if os.path.isdir(path):
                safetensors_files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.endswith(".safetensors")
                ]
                if len(safetensors_files) == 1:
                    return safetensors_files

            idx = os.path.join(path, "model.safetensors.index.json")
            if os.path.isdir(path) and os.path.isfile(idx):
                with open(idx, "r", encoding="utf-8") as f:
                    index = json.load(f)
                return sorted(
                    os.path.join(path, shard)
                    for shard in set(index["weight_map"].values())
                )

            if os.path.isdir(path):
                return sorted(
                    [
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if f.endswith(".safetensors")
                    ]
                )

            return []

        def _load_safetensors_file(filepath, device):
            """Load a single safetensors file"""
            return load_safetensors(filepath, device=device)

        def _smart_parameter_mapping(source_dict, target_dict):
            """Apply smart parameter mapping with multiple strategies"""

            # Convert to float32 if needed
            processed_source = {}
            for k, v in source_dict.items():
                if v.dtype == torch.float16:
                    v = v.to(torch.float32)
                processed_source[k] = v

            # Strategy 1: Handle QKV combination
            qkv_combinations = self.mapper.handle_qkv_combination(
                processed_source, target_dict
            )

            # Strategy 2: Find parameter mappings
            source_keys = list(processed_source.keys())
            target_keys = list(target_dict.keys())

            # Remove QKV components that were combined
            if qkv_combinations:
                combined_layers = defaultdict(set)
                for qkv_key in qkv_combinations.keys():
                    layer_match = re.search(r"layers?\.(\d+)\.", qkv_key)
                    if not layer_match:
                        layer_match = re.search(r"h\.(\d+)\.", qkv_key)
                    if layer_match:
                        layer_num = layer_match.group(1)
                        # Determine attention type from the combined key
                        if "global_attn" in qkv_key:
                            attn_type = "global_attn"
                        elif "self_attn" in qkv_key:
                            attn_type = "self_attn"
                        elif "attention" in qkv_key:
                            attn_type = "attention"
                        else:
                            attn_type = "attn"
                        combined_layers[layer_num].add(attn_type)

                # Filter out individual Q, K, V components for combined layers
                filtered_source_keys = []
                for key in source_keys:
                    layer_match = re.search(r"layers?\.(\d+)\.", key)
                    if not layer_match:
                        layer_match = re.search(r"h\.(\d+)\.", key)
                    
                    if layer_match:
                        layer_num = layer_match.group(1)
                        # Check if this layer's attention was combined
                        attn_was_combined = False
                        for attn_type in combined_layers.get(layer_num, set()):
                            if attn_type in key and any(proj in key for proj in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                                attn_was_combined = True
                                break
                        
                        if attn_was_combined:
                            continue  # Skip individual Q, K, V for combined layers
                    
                    filtered_source_keys.append(key)
                source_keys = filtered_source_keys

            mappings = self.mapper.find_best_matches(
                source_keys, target_keys, processed_source, target_dict
            )

            # Strategy 3: Combine all mappings
            final_state_dict = {}

            # Add QKV combinations
            final_state_dict.update(qkv_combinations)

            # Add mapped parameters
            for source_key, target_key in tqdm(
                mappings.items(), desc="Mapping parameters", disable=not verbose
            ):
                final_state_dict[target_key] = processed_source[source_key]

            mapped_count = len(final_state_dict)
            total_source = len(source_dict)
            total_target = len(target_dict)

            success_rate = (
                (mapped_count / total_source) * 100 if total_source > 0 else 0
            )
            coverage_rate = (
                (mapped_count / total_target) * 100 if total_target > 0 else 0
            )

            # Find unmapped keys
            mapped_source_keys = set(mappings.keys())
            if qkv_combinations:
                # Add QKV source keys that were combined
                for qkv_target_key in qkv_combinations.keys():
                    layer_match = re.search(r"layers?\.(\d+)\.", qkv_target_key)
                    if not layer_match:
                        layer_match = re.search(r"h\.(\d+)\.", qkv_target_key)
                    if layer_match:
                        layer_num = layer_match.group(1)
                        for source_key in source_dict.keys():
                            if (
                                f"layers.{layer_num}." in source_key
                                or f"layer.{layer_num}." in source_key
                                or f"h.{layer_num}." in source_key
                            ) and any(
                                proj in source_key
                                for proj in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]
                            ):
                                mapped_source_keys.add(source_key)

            unmapped_source = set(source_dict.keys()) - mapped_source_keys
            unmapped_target = set(target_dict.keys()) - set(final_state_dict.keys())

            if verbose:
                print(f"✅ Parameter mapping completed:")
                print(f"   Successfully mapped: {mapped_count}")
                print(f"   Success rate: {success_rate:.1f}% ({mapped_count}/{total_source})")
                print(f"   Target coverage: {coverage_rate:.1f}% ({mapped_count}/{total_target})")

                if qkv_combinations:
                    print(f"   QKV combinations created: {len(qkv_combinations)}")

                if unmapped_source:
                    print(
                        f"   Unmapped source keys ({len(unmapped_source)}): {list(unmapped_source)[:3]}{'...' if len(unmapped_source) > 3 else ''}"
                    )

                if unmapped_target:
                    print(
                        f"   Missing target keys ({len(unmapped_target)}): {list(unmapped_target)[:3]}{'...' if len(unmapped_target) > 3 else ''}"
                    )

            return final_state_dict, list(unmapped_target), list(unmapped_source)

        # Get target model state dict
        target_state_dict = self.state_dict()

        # Try safetensors first
        safefiles = _collect_safetensors_files(model_path)
        if safefiles:
            if verbose:
                print(f"📁 Loading from {len(safefiles)} safetensors file(s)")

            raw_weights = {}
            for f in tqdm(
                safefiles, desc="Loading safetensors files", disable=not verbose
            ):
                raw_weights.update(_load_safetensors_file(f, map_location))

            if not raw_weights:
                raise RuntimeError(
                    f"Found safetensors files but no tensors were loaded from {model_path!r}"
                )

            if verbose:
                print(f"📊 Source parameters: {len(raw_weights)}")
                print(f"📊 Target parameters: {len(target_state_dict)}")

            # Apply smart parameter mapping
            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                raw_weights, target_state_dict
            )

            # Load the mapped state dict
            try:
                self.load_state_dict(mapped_state_dict, strict=strict)
                if verbose:
                    print("✅ Model loaded successfully!")

            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load model state dict: {e}")
                else:
                    if verbose:
                        print(f"⚠️  Warning: Some parameters could not be loaded: {e}")

            return

        # Fallback: PyTorch checkpoint
        if os.path.isfile(model_path):
            if verbose:
                print(f"📁 Loading PyTorch checkpoint: {model_path}")

            ckpt = torch.load(model_path, map_location=map_location)
            source_dict = (
                ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            )

            if verbose:
                print(f"📊 Source parameters: {len(source_dict)}")
                print(f"📊 Target parameters: {len(target_state_dict)}")

            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                source_dict, target_state_dict
            )

            self.load_state_dict(mapped_state_dict, strict=strict)
            if verbose:
                print("✅ Model loaded successfully!")
            return

        # Fallback: Sharded PyTorch
        idx_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isdir(model_path) and os.path.isfile(idx_path):
            if verbose:
                print(f"📁 Loading sharded PyTorch checkpoint from: {model_path}")

            with open(idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            all_weights = {}
            shards = sorted(set(index["weight_map"].values()))
            
            for shard in tqdm(shards, desc="Loading shards", disable=not verbose):
                shard_path = os.path.join(model_path, shard)

                part = torch.load(shard_path, map_location=map_location)
                part_dict = part.get("model_state_dict", part)
                all_weights.update(part_dict)

                del part, part_dict
                gc.collect()

            if verbose:
                print(f"📊 Source parameters: {len(all_weights)}")
                print(f"📊 Target parameters: {len(target_state_dict)}")

            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                all_weights, target_state_dict
            )

            self.load_state_dict(mapped_state_dict, strict=strict)
            if verbose:
                print("✅ Model loaded successfully!")
            return

        raise FileNotFoundError(f"No checkpoint found at '{model_path}'")