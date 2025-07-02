import os
import json
import gc
import torch
import re
import warnings
from safetensors.torch import load_file as load_safetensors
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher


class AutoParameterMapper:
    """
    Advanced parameter mapper that automatically handles various naming conventions
    without requiring manual mapping functions for each architecture change.
    """

    def __init__(self, custom_patterns=None):
        # Base transformation patterns - extensible and flexible
        self.base_patterns = [
            # Handle model prefixes
            (r"^module\.", ""),  # Remove DataParallel wrapper
            (r"^model\.", ""),  # Remove model wrapper
            (r"^", "model."),  # Add model prefix
            # --- Standard Linear Layer Patterns ---
            # Attention layer patterns (bidirectional)
            (
                r"(.*)\.self_attn\.q_proj\.linear\.(weight|bias)",
                r"\1.self_attn.q_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.k_proj\.linear\.(weight|bias)",
                r"\1.self_attn.k_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.v_proj\.linear\.(weight|bias)",
                r"\1.self_attn.v_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.o_proj\.linear\.(weight|bias)",
                r"\1.self_attn.o_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.q_proj\.(weight|bias)",
                r"\1.self_attn.q_proj.linear.\2",
            ),
            (
                r"(.*)\.self_attn\.k_proj\.(weight|bias)",
                r"\1.self_attn.k_proj.linear.\2",
            ),
            (
                r"(.*)\.self_attn\.v_proj\.(weight|bias)",
                r"\1.self_attn.v_proj.linear.\2",
            ),
            (
                r"(.*)\.self_attn\.o_proj\.(weight|bias)",
                r"\1.self_attn.o_proj.linear.\2",
            ),
            # MLP/FFN patterns (bidirectional)
            (r"(.*)\.mlp\.gate_proj\.linear\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.up_proj\.linear\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.down_proj\.linear\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.gate_proj.linear.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.up_proj.linear.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.down_proj.linear.\2"),
            # --- NEW: Patterns for Quantized Layers (e.g., using a '.layer' submodule) ---
            # Attention layer patterns for '.layer' submodule (bidirectional)
            (
                r"(.*)\.self_attn\.q_proj\.layer\.(weight|bias)",
                r"\1.self_attn.q_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.k_proj\.layer\.(weight|bias)",
                r"\1.self_attn.k_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.v_proj\.layer\.(weight|bias)",
                r"\1.self_attn.v_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.o_proj\.layer\.(weight|bias)",
                r"\1.self_attn.o_proj.\2",
            ),
            (
                r"(.*)\.self_attn\.q_proj\.(weight|bias)",
                r"\1.self_attn.q_proj.layer.\2",
            ),
            (
                r"(.*)\.self_attn\.k_proj\.(weight|bias)",
                r"\1.self_attn.k_proj.layer.\2",
            ),
            (
                r"(.*)\.self_attn\.v_proj\.(weight|bias)",
                r"\1.self_attn.v_proj.layer.\2",
            ),
            (
                r"(.*)\.self_attn\.o_proj\.(weight|bias)",
                r"\1.self_attn.o_proj.layer.\2",
            ),
            # MLP/FFN patterns for '.layer' submodule (bidirectional)
            (r"(.*)\.mlp\.gate_proj\.layer\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.up_proj\.layer\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.down_proj\.layer\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.gate_proj.layer.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.up_proj.layer.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.down_proj.layer.\2"),
            # --- QKV Combination Patterns (for all layer types) ---
            (r"(.*)\.qkv_proj\.linear\.(weight|bias)", r"\1.qkv_proj.\2"),
            (r"(.*)\.qkv_proj\.(weight|bias)", r"\1.qkv_proj.linear.\2"),
            (r"(.*)\.qkv_proj\.layer\.(weight|bias)", r"\1.qkv_proj.\2"),  # NEW
            (r"(.*)\.qkv_proj\.(weight|bias)", r"\1.qkv_proj.layer.\2"),  # NEW
            # --- Other Common Patterns ---
            # Embedding patterns
            (r"embed_tokens\.weight", "embedder.weight"),
            (r"embedder\.weight", "embed_tokens.weight"),
            (r"word_embeddings\.weight", "embedder.weight"),
            (r"token_embeddings\.weight", "embedder.weight"),
            # Normalization patterns
            (r"(.*)\.query_norm\.weight", r"\1.q_norm.weight"),
            (r"(.*)\.key_norm\.weight", r"\1.k_norm.weight"),
            (r"(.*)\.q_norm\.weight", r"\1.query_norm.weight"),
            (r"(.*)\.k_norm\.weight", r"\1.key_norm.weight"),
            # Layer norm variations
            (r"(.*)\.input_layernorm\.weight", r"\1.input_layer_norm.weight"),
            (r"(.*)\.post_attention_layernorm\.weight", r"\1.post_attn_norm.weight"),
            (r"(.*)\.layer_norm\.weight", r"\1.layernorm.weight"),
            (r"(.*)\.layernorm\.weight", r"\1.layer_norm.weight"),
            # Output/final norm
            (r"^norm\.weight", "model.norm.weight"),
            (r"^model\.norm\.weight", "norm.weight"),
            (r"final_layer_norm\.weight", "norm.weight"),
            # Handle layer indexing variations
            (r"layers\.(\d+)\.", r"layer.\1."),
            (r"layer\.(\d+)\.", r"layers.\1."),
            (r"h\.(\d+)\.", r"layers.\1."),
            (r"transformer\.h\.(\d+)\.", r"model.layers.\1."),
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

        # Apply all transformation patterns
        for pattern_re, replacement in self.compiled_patterns:
            for variant in list(variants):
                if pattern_re.search(variant):
                    new_variant = pattern_re.sub(replacement, variant)
                    variants.add(new_variant)

        # Add common prefix/suffix variations
        additional_variants = set()
        for variant in variants:
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
        """Handle QKV weight combination automatically."""
        qkv_combinations = {}
        qkv_groups = defaultdict(dict)

        # Group Q, K, V weights by layer
        for key, tensor in source_dict.items():
            if "self_attn" in key and any(
                proj in key for proj in ["q_proj", "k_proj", "v_proj"]
            ):
                layer_match = re.search(r"layers?\.(\d+)\.", key)
                if layer_match:
                    layer_id = layer_match.group(1)

                    # Ensure we are handling weights only for concatenation
                    if key.endswith(".weight"):
                        if "q_proj" in key:
                            qkv_groups[layer_id]["q_weight"] = (key, tensor)
                        elif "k_proj" in key:
                            qkv_groups[layer_id]["k_weight"] = (key, tensor)
                        elif "v_proj" in key:
                            qkv_groups[layer_id]["v_weight"] = (key, tensor)
                    elif key.endswith(".bias"):
                        if "q_proj" in key:
                            qkv_groups[layer_id]["q_bias"] = (key, tensor)
                        elif "k_proj" in key:
                            qkv_groups[layer_id]["k_bias"] = (key, tensor)
                        elif "v_proj" in key:
                            qkv_groups[layer_id]["v_bias"] = (key, tensor)

        # Check if target expects combined QKV
        target_expects_qkv = any("qkv_proj" in key for key in target_dict.keys())

        if target_expects_qkv:
            for layer_id, qkv_dict in qkv_groups.items():
                # Process weights
                if (
                    "q_weight" in qkv_dict
                    and "k_weight" in qkv_dict
                    and "v_weight" in qkv_dict
                ):
                    q_key, q_tensor = qkv_dict["q_weight"]
                    k_key, k_tensor = qkv_dict["k_weight"]
                    v_key, v_tensor = qkv_dict["v_weight"]

                    combined_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

                    # Dynamically generate potential target keys using our patterns
                    base_qkv_key = f"model.layers.{layer_id}.self_attn.qkv_proj.weight"
                    qkv_key_variants = self.generate_key_variants(base_qkv_key)

                    for qkv_key in qkv_key_variants:
                        if qkv_key in target_dict:
                            if combined_tensor.shape == target_dict[qkv_key].shape:
                                qkv_combinations[qkv_key] = combined_tensor
                                break  # Found a match, move to next layer

                # Process biases (if they exist)
                if (
                    "q_bias" in qkv_dict
                    and "k_bias" in qkv_dict
                    and "v_bias" in qkv_dict
                ):
                    # (Logic for combining biases is similar if needed)
                    pass

        return qkv_combinations


class PreTrainedModel:
    """
    A class representing some utility functions for pretrained models.
    """

    def __init__(self, custom_patterns=None):
        self.mapper = AutoParameterMapper(custom_patterns)

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
            if verbose:
                print(f"🔄 Starting smart parameter mapping...")
                print(f"   Source parameters: {len(source_dict)}")
                print(f"   Target parameters: {len(target_dict)}")

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
            if qkv_combinations and verbose:
                print(f"   Combined QKV weights: {len(qkv_combinations)}")

            # Strategy 2: Find parameter mappings
            source_keys = list(processed_source.keys())
            target_keys = list(target_dict.keys())

            # Remove QKV components that were combined
            if qkv_combinations:
                combined_layers = set()
                for qkv_key in qkv_combinations.keys():
                    layer_match = re.search(r"layers?\.(\d+)\.", qkv_key)
                    if layer_match:
                        combined_layers.add(layer_match.group(1))

                # Filter out individual Q, K, V components for combined layers
                filtered_source_keys = []
                for key in source_keys:
                    layer_match = re.search(r"layers?\.(\d+)\.", key)
                    if layer_match and layer_match.group(1) in combined_layers:
                        if any(proj in key for proj in ["q_proj", "k_proj", "v_proj"]):
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
            for source_key, target_key in mappings.items():
                final_state_dict[target_key] = processed_source[source_key]

            # Calculate statistics
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
                    if layer_match:
                        layer_id = layer_match.group(1)
                        for source_key in source_dict.keys():
                            if (
                                f"layers.{layer_id}." in source_key
                                or f"layers.{layer_id}." in source_key
                            ) and any(
                                proj in source_key
                                for proj in ["q_proj", "k_proj", "v_proj"]
                            ):
                                mapped_source_keys.add(source_key)

            unmapped_source = set(source_dict.keys()) - mapped_source_keys
            unmapped_target = set(target_dict.keys()) - set(final_state_dict.keys())

            if verbose:
                print(f"✅ Parameter mapping completed:")
                print(f"   Successfully mapped: {mapped_count}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Target coverage: {coverage_rate:.1f}%")

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
                print(f"📁 Found {len(safefiles)} safetensors file(s)")

            raw_weights = {}
            for f in safefiles:
                if verbose:
                    print(f"   Loading {os.path.basename(f)}")
                raw_weights.update(_load_safetensors_file(f, map_location))

            if not raw_weights:
                raise RuntimeError(
                    f"Found safetensors files but no tensors were loaded from {model_path!r}"
                )

            # Apply smart parameter mapping
            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                raw_weights, target_state_dict
            )

            # Load the mapped state dict
            try:
                self.load_state_dict(mapped_state_dict, strict=strict)

                if verbose:
                    print(f"✅ Successfully loaded model from safetensors")
                    if not strict and (missing_keys or unmapped_keys):
                        if missing_keys:
                            print(
                                f"⚠️  Missing {len(missing_keys)} parameters (using model defaults)"
                            )
                        if unmapped_keys:
                            print(
                                f"⚠️  {len(unmapped_keys)} source parameters were not used"
                            )

            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load model state dict: {e}")
                else:
                    warnings.warn(f"Some parameters could not be loaded: {e}")

            return

        # Fallback: PyTorch checkpoint
        if os.path.isfile(model_path):
            if verbose:
                print(f"📁 Loading PyTorch checkpoint: {model_path}")

            ckpt = torch.load(model_path, map_location=map_location)
            source_dict = (
                ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            )

            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                source_dict, target_state_dict
            )

            self.load_state_dict(mapped_state_dict, strict=strict)
            if verbose:
                print(f"✅ Successfully loaded PyTorch checkpoint")
            return

        # Fallback: Sharded PyTorch
        idx_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.isdir(model_path) and os.path.isfile(idx_path):
            if verbose:
                print(f"📁 Loading sharded PyTorch checkpoint from: {model_path}")

            with open(idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            all_weights = {}
            for shard in set(index["weight_map"].values()):
                shard_path = os.path.join(model_path, shard)
                if verbose:
                    print(f"   Loading shard: {shard}")

                part = torch.load(shard_path, map_location=map_location)
                part_dict = part.get("model_state_dict", part)
                all_weights.update(part_dict)

                del part, part_dict
                gc.collect()

            mapped_state_dict, missing_keys, unmapped_keys = _smart_parameter_mapping(
                all_weights, target_state_dict
            )

            self.load_state_dict(mapped_state_dict, strict=strict)
            if verbose:
                print(f"✅ Successfully loaded sharded PyTorch checkpoint")
            return

        raise FileNotFoundError(f"No checkpoint found at '{model_path}'")
