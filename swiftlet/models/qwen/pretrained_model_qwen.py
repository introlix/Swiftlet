import os
import json
import torch
import gc
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from contextlib import contextmanager


class PreTrainedModel:
    def __init__(self):
        pass

    @contextmanager
    def _memory_efficient_loading(self):
        """Context manager for memory-efficient loading"""
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            yield
        finally:
            # Clean up after loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def convert_hf_to_your_model(self, hf_state: dict) -> dict:
        """Memory-efficient conversion from HF Qwen2 to your model structure"""
        new_state = {}
        
        try:
            # 1) Embeddings - Use reference instead of clone to save memory
            embed_weight = hf_state['model.embed_tokens.weight']
            new_state['embedder.weight'] = embed_weight
            # Share the same tensor reference instead of cloning
            new_state['lm_head.weight'] = embed_weight  # Same reference, not a copy
            
            # 2) Get number of layers from the state dict
            layer_nums = set()
            for key in hf_state.keys():
                if 'model.layers.' in key:
                    layer_num = key.split('model.layers.')[1].split('.')[0]
                    if layer_num.isdigit():
                        layer_nums.add(int(layer_num))
            
            num_layers = max(layer_nums) + 1 if layer_nums else 0
            print(f"Detected {num_layers} layers")
            
            # 3) Process each layer with memory cleanup
            for i in range(num_layers):
                hf_prefix = f'model.layers.{i}'
                your_prefix = f'model.layers.{i}'
                
                # === ATTENTION ===
                # QKV combination - use direct reference to avoid extra memory
                q_weight = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
                k_weight = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
                v_weight = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
                
                # Concatenate and immediately delete references to save memory
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                new_state[f'{your_prefix}.self_attn.qkv_proj.linear.weight'] = qkv_weight
                
                # QKV bias (if exists)
                q_bias_key = f'{hf_prefix}.self_attn.q_proj.bias'
                if q_bias_key in hf_state:
                    q_bias = hf_state[q_bias_key]
                    k_bias = hf_state[f'{hf_prefix}.self_attn.k_proj.bias']
                    v_bias = hf_state[f'{hf_prefix}.self_attn.v_proj.bias']
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    new_state[f'{your_prefix}.self_attn.qkv_proj.linear.bias'] = qkv_bias
                
                # Output projection - direct reference
                new_state[f'{your_prefix}.self_attn.o_proj.linear.weight'] = hf_state[f'{hf_prefix}.self_attn.o_proj.weight']
                if f'{hf_prefix}.self_attn.o_proj.bias' in hf_state:
                    new_state[f'{your_prefix}.self_attn.o_proj.linear.bias'] = hf_state[f'{hf_prefix}.self_attn.o_proj.bias']
                
                # === MLP === - direct references
                new_state[f'{your_prefix}.mlp.gate_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.gate_proj.weight']
                new_state[f'{your_prefix}.mlp.up_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.up_proj.weight']
                new_state[f'{your_prefix}.mlp.down_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.down_proj.weight']
                
                # MLP bias (if exists)
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    bias_key = f'{hf_prefix}.mlp.{proj}.bias'
                    if bias_key in hf_state:
                        new_state[f'{your_prefix}.mlp.{proj}.linear.bias'] = hf_state[bias_key]
                
                # === LAYER NORMS === - direct references
                new_state[f'{your_prefix}.input_layernorm.weight'] = hf_state[f'{hf_prefix}.input_layernorm.weight']
                new_state[f'{your_prefix}.post_attention_layernorm.weight'] = hf_state[f'{hf_prefix}.post_attention_layernorm.weight']
                
                # Periodic garbage collection for large models
                if i % 5 == 0:  # Every 5 layers
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 4) Final norm - direct reference
            new_state['model.norm.weight'] = hf_state['model.norm.weight']
            
            return new_state
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            # Clean up on error
            del new_state
            gc.collect()
            raise

    def from_pretrained(self, model_path, map_location='cpu', strict=False, verbose=True, 
                       max_memory_gb=None, device_map=None):
        """Load HF model with memory-efficient conversion"""
        
        # Force CPU loading for large models if not specified
        if max_memory_gb and max_memory_gb > 8:
            map_location = 'cpu'
            if verbose:
                print(f"Loading large model ({max_memory_gb}GB) to CPU first")
        
        def _collect_safetensors_files(path):
            if os.path.isfile(path) and path.endswith('.safetensors'):
                return [path]
            if os.path.isdir(path):
                # Check for index file
                idx_path = os.path.join(path, 'model.safetensors.index.json')
                if os.path.isfile(idx_path):
                    with open(idx_path, 'r') as f:
                        idx_data = json.load(f)
                    return sorted(set(os.path.join(path, v) for v in idx_data['weight_map'].values()))
                # Get all safetensors files
                return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors')])
            return []
        
        with self._memory_efficient_loading():
            # Get your model's target state dict
            target_state = self.state_dict()
            
            # Try loading safetensors first
            safetensor_files = _collect_safetensors_files(model_path)
            if safetensor_files:
                if verbose:
                    print(f"Loading {len(safetensor_files)} safetensors files...")
                
                hf_state = {}
                
                # Load files one by one for large models to avoid memory spikes
                for i, file_path in enumerate(tqdm(safetensor_files, desc='Loading', disable=not verbose)):
                    try:
                        file_state = load_safetensors(file_path, device=map_location)
                        hf_state.update(file_state)
                        
                        # Periodic cleanup for large models
                        if (i + 1) % 3 == 0:  # Every 3 files
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        # Clean up partial state
                        del hf_state
                        gc.collect()
                        raise
                
                # Convert HF state to your model format
                try:
                    converted_state = self.convert_hf_to_your_model(hf_state)
                    
                    # Clean up original HF state to free memory
                    del hf_state
                    gc.collect()
                    
                    if verbose:
                        print(f"Converted {len(converted_state)} parameters")
                        missing_keys = set(target_state.keys()) - set(converted_state.keys())
                        unexpected_keys = set(converted_state.keys()) - set(target_state.keys())
                        if missing_keys:
                            print(f"Missing keys: {sorted(list(missing_keys))}")
                        if unexpected_keys:
                            print(f"Unexpected keys: {sorted(list(unexpected_keys))}")
                    
                    # Load the converted state
                    self.load_state_dict(converted_state, strict=strict)
                    
                    # Clean up converted state
                    del converted_state
                    gc.collect()
                    
                    return self
                    
                except Exception as e:
                    print(f"Error during conversion: {e}")
                    if 'hf_state' in locals():
                        del hf_state
                    if 'converted_state' in locals():
                        del converted_state
                    gc.collect()
                    raise
            
            # Fallback to PyTorch files (with similar memory management)
            pytorch_file = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.isfile(pytorch_file):
                if verbose:
                    print("Loading pytorch_model.bin...")
                
                try:
                    checkpoint = torch.load(pytorch_file, map_location=map_location)
                    hf_state = checkpoint.get('model_state_dict', checkpoint)
                    del checkpoint  # Free memory immediately
                    
                    converted_state = self.convert_hf_to_your_model(hf_state)
                    del hf_state
                    gc.collect()
                    
                    self.load_state_dict(converted_state, strict=strict)
                    del converted_state
                    gc.collect()
                    
                    return self
                    
                except Exception as e:
                    print(f"Error loading pytorch_model.bin: {e}")
                    # Cleanup
                    for var in ['checkpoint', 'hf_state', 'converted_state']:
                        if var in locals():
                            del locals()[var]
                    gc.collect()
                    raise
            
            # Try sharded PyTorch files (with memory management)
            shard_files = [f for f in os.listdir(model_path) if f.startswith('pytorch_model-') and f.endswith('.bin')]
            if shard_files:
                if verbose:
                    print(f"Loading {len(shard_files)} PyTorch shard files...")
                
                hf_state = {}
                try:
                    for shard_file in tqdm(sorted(shard_files), desc='Loading shards', disable=not verbose):
                        checkpoint = torch.load(os.path.join(model_path, shard_file), map_location=map_location)
                        hf_state.update(checkpoint.get('model_state_dict', checkpoint))
                        del checkpoint  # Free immediately
                        gc.collect()
                    
                    converted_state = self.convert_hf_to_your_model(hf_state)
                    del hf_state
                    gc.collect()
                    
                    self.load_state_dict(converted_state, strict=strict)
                    del converted_state
                    gc.collect()
                    
                    return self
                    
                except Exception as e:
                    print(f"Error loading sharded files: {e}")
                    # Cleanup
                    for var in ['checkpoint', 'hf_state', 'converted_state']:
                        if var in locals():
                            del locals()[var]
                    gc.collect()
                    raise
        
        raise FileNotFoundError(f"No valid checkpoint found at {model_path}")

    def load_in_8bit(self, model_path, **kwargs):
        """Load model with 8-bit quantization (assumes model already has quantized layers)"""
        try:
            import bitsandbytes as bnb
            
            if 'verbose' not in kwargs:
                kwargs['verbose'] = True
                
            if kwargs.get('verbose'):
                print("Loading model with 8-bit quantization...")
            
            # Load with memory-efficient loading, assuming quantization is handled in model architecture
            return self.from_pretrained(model_path, **kwargs)
            
        except ImportError:
            print("bitsandbytes not available. Loading normally to CPU.")
            return self.from_pretrained(model_path, map_location='cpu', **kwargs)
        except Exception as e:
            print(f"Error during 8-bit loading: {e}")
            print("Falling back to normal loading...")
            return self.from_pretrained(model_path, map_location='cpu', **kwargs)

    def load_in_4bit(self, model_path, **kwargs):
        """Load model with 4-bit quantization (assumes model already has quantized layers)"""
        try:
            import bitsandbytes as bnb
            
            if 'verbose' not in kwargs:
                kwargs['verbose'] = True
                
            if kwargs.get('verbose'):
                print("Loading model with 4-bit quantization...")
            
            # Load with memory-efficient loading, assuming quantization is handled in model architecture
            return self.from_pretrained(model_path, **kwargs)
            
        except ImportError:
            print("bitsandbytes not available. Loading normally to CPU.")
            return self.from_pretrained(model_path, map_location='cpu', **kwargs)
        except Exception as e:
            print(f"Error during 4-bit loading: {e}")
            print("Falling back to normal loading...")
            return self.from_pretrained(model_path, map_location='cpu', **kwargs)