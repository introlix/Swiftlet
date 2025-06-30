import os
import json
import gc
import torch
from safetensors.torch import load_file as load_safetensors

class GemmaModelLoader:
    """
    A class to load the Gemma model from a specified path.
    """
    def from_pretrained(self, model_path: str, map_location="cpu", quant: bool = False, quant_dtype: str = None):

        # if quant == True and quant_dtype is None:
        #     raise ValueError("If quant is True, quant_dtype must be specified (e.g., 'int8', 'int4', 'fp4')")
        
        # if quant == False and quant_dtype is not None:
        #     quant = True

        # if quant:
        #     self.quant = True
        #     self.quant_dtype = quant_dtype
        

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