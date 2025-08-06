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
                       max_memory_gb=None, device_map=None, low_cpu_mem_usage=True):
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
            # Memory monitoring
            def log_memory(stage=""):
                if verbose:
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        print(f"Memory usage {stage}: {mem.percent:.1f}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")
                    except ImportError:
                        # Fallback memory monitoring without psutil
                        import os
                        try:
                            # Linux/Unix memory check
                            with open('/proc/meminfo', 'r') as f:
                                meminfo = f.read()
                                for line in meminfo.split('\n'):
                                    if 'MemTotal:' in line:
                                        total_kb = int(line.split()[1])
                                    elif 'MemAvailable:' in line:
                                        avail_kb = int(line.split()[1])
                                used_gb = (total_kb - avail_kb) / 1024**2
                                total_gb = total_kb / 1024**2
                                print(f"Memory usage {stage}: {used_gb:.1f}GB/{total_gb:.1f}GB")
                        except:
                            print(f"Memory check {stage}: psutil not available, install with 'pip install psutil' for detailed monitoring")
                    
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3
                        gpu_max = torch.cuda.max_memory_allocated() / 1024**3
                        print(f"GPU memory: {gpu_mem:.1f}GB (max: {gpu_max:.1f}GB)")
            
            log_memory("before loading")
            
            # Get your model's target state dict
            target_state = self.state_dict()
            
            # Try loading safetensors first
            safetensor_files = _collect_safetensors_files(model_path)
            if safetensor_files:
                if verbose:
                    print(f"Loading {len(safetensor_files)} safetensors files...")
                
                hf_state = {}
                
                # For very large models, use ultra low memory mode
                if low_cpu_mem_usage and len(safetensor_files) > 2:
                    if verbose:
                        print("Using ultra low memory loading mode...")
                    
                    # Process and load each file individually to minimize peak memory
                    for i, file_path in enumerate(tqdm(safetensor_files, desc='Loading', disable=not verbose)):
                        try:
                            log_memory(f"loading file {i+1}/{len(safetensor_files)}")
                            
                            # Load one file at a time
                            file_state = load_safetensors(file_path, device=map_location)
                            
                            # Convert this portion immediately
                            partial_converted = self._convert_partial_state(file_state, i, verbose)
                            
                            # Load converted weights directly into model to avoid storing in memory
                            self._load_partial_state_dict(partial_converted, strict=False, verbose=verbose)
                            
                            # Clean up immediately
                            del file_state, partial_converted
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            raise
                    
                    log_memory("after ultra low memory loading")
                    return self
                
                else:
                    # Standard loading with periodic cleanup
                    for i, file_path in enumerate(tqdm(safetensor_files, desc='Loading', disable=not verbose)):
                        try:
                            file_state = load_safetensors(file_path, device=map_location)
                            hf_state.update(file_state)
                            
                            # More frequent cleanup for large models
                            if (i + 1) % 2 == 0:  # Every 2 files instead of 3
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            # Clean up partial state
                            del hf_state
                            gc.collect()
                            raise
                
                log_memory("after loading files")
                
                # Convert HF state to your model format
                try:
                    converted_state = self.convert_hf_to_your_model(hf_state)
                    
                    # Clean up original HF state immediately
                    del hf_state
                    gc.collect()
                    log_memory("after conversion")
                    
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
                    log_memory("after loading into model")
                    
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

    def _convert_partial_state(self, file_state: dict, file_index: int, verbose: bool) -> dict:
        """Convert a portion of HF state for ultra low memory loading"""
        converted = {}
        
        for hf_key, tensor in file_state.items():
            # Convert keys based on patterns
            if 'model.embed_tokens.weight' in hf_key:
                converted['embedder.weight'] = tensor
                converted['lm_head.weight'] = tensor  # Share reference
            elif 'model.norm.weight' in hf_key:
                converted['model.norm.weight'] = tensor
            elif 'model.layers.' in hf_key:
                # Extract layer number
                parts = hf_key.split('.')
                layer_idx = parts[2]
                
                # Convert attention weights
                if 'self_attn.q_proj.weight' in hf_key:
                    # Store temporarily for QKV concatenation
                    converted[f'_temp_q_{layer_idx}'] = tensor
                elif 'self_attn.k_proj.weight' in hf_key:
                    converted[f'_temp_k_{layer_idx}'] = tensor
                elif 'self_attn.v_proj.weight' in hf_key:
                    converted[f'_temp_v_{layer_idx}'] = tensor
                elif 'self_attn.o_proj.weight' in hf_key:
                    new_key = hf_key.replace('self_attn.o_proj.weight', 'self_attn.o_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'self_attn.o_proj.bias' in hf_key:
                    new_key = hf_key.replace('self_attn.o_proj.bias', 'self_attn.o_proj.linear.bias')
                    converted[new_key] = tensor
                # QKV bias handling
                elif 'self_attn.q_proj.bias' in hf_key:
                    converted[f'_temp_q_bias_{layer_idx}'] = tensor
                elif 'self_attn.k_proj.bias' in hf_key:
                    converted[f'_temp_k_bias_{layer_idx}'] = tensor
                elif 'self_attn.v_proj.bias' in hf_key:
                    converted[f'_temp_v_bias_{layer_idx}'] = tensor
                # MLP layers
                elif 'mlp.gate_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.gate_proj.weight', 'mlp.gate_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'mlp.up_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.up_proj.weight', 'mlp.up_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'mlp.down_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.down_proj.weight', 'mlp.down_proj.linear.weight')
                    converted[new_key] = tensor
                # MLP bias
                elif 'mlp.gate_proj.bias' in hf_key:
                    new_key = hf_key.replace('mlp.gate_proj.bias', 'mlp.gate_proj.linear.bias')
                    converted[new_key] = tensor
                elif 'mlp.up_proj.bias' in hf_key:
                    new_key = hf_key.replace('mlp.up_proj.bias', 'mlp.up_proj.linear.bias')
                    converted[new_key] = tensor
                elif 'mlp.down_proj.bias' in hf_key:
                    new_key = hf_key.replace('mlp.down_proj.bias', 'mlp.down_proj.linear.bias')
                    converted[new_key] = tensor
                # Layer norms
                elif 'input_layernorm.weight' in hf_key:
                    converted[hf_key] = tensor
                elif 'post_attention_layernorm.weight' in hf_key:
                    converted[hf_key] = tensor
        
        return converted

    def _load_partial_state_dict(self, partial_state: dict, strict: bool, verbose: bool):
        """Load a partial state dict into the model"""
        # Handle QKV concatenation for any complete sets
        qkv_layers = {}
        qkv_bias_layers = {}
        
        # Collect QKV weights and biases
        for key in list(partial_state.keys()):
            if key.startswith('_temp_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    if 'bias' in key:
                        # Handle bias: _temp_q_bias_0 -> q, bias, 0
                        qkv_type = parts[1]  # q, k, or v
                        layer_idx = parts[3]  # layer index
                        
                        if layer_idx not in qkv_bias_layers:
                            qkv_bias_layers[layer_idx] = {}
                        qkv_bias_layers[layer_idx][qkv_type] = partial_state.pop(key)
                    else:
                        # Handle weight: _temp_q_0 -> q, 0
                        qkv_type = parts[1]  # q, k, or v
                        layer_idx = parts[2]  # layer index
                        
                        if layer_idx not in qkv_layers:
                            qkv_layers[layer_idx] = {}
                        qkv_layers[layer_idx][qkv_type] = partial_state.pop(key)
        
        # Create QKV concatenations for complete sets
        for layer_idx, qkv_dict in qkv_layers.items():
            if all(k in qkv_dict for k in ['q', 'k', 'v']):
                qkv_weight = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
                partial_state[f'model.layers.{layer_idx}.self_attn.qkv_proj.linear.weight'] = qkv_weight
        
        # Create QKV bias concatenations for complete sets
        for layer_idx, qkv_bias_dict in qkv_bias_layers.items():
            if all(k in qkv_bias_dict for k in ['q', 'k', 'v']):
                qkv_bias = torch.cat([qkv_bias_dict['q'], qkv_bias_dict['k'], qkv_bias_dict['v']], dim=0)
                partial_state[f'model.layers.{layer_idx}.self_attn.qkv_proj.linear.bias'] = qkv_bias
        
        # Load what we can into the model
        model_state = self.state_dict()
        loaded_count = 0
        
        for key, tensor in partial_state.items():
            if key in model_state:
                try:
                    model_state[key].copy_(tensor)
                    loaded_count += 1
                    if verbose and loaded_count % 10 == 0:  # Log every 10th parameter
                        print(f"Loaded {loaded_count} parameters...")
                except Exception as e:
                    if verbose:
                        print(f"Failed to load {key}: {e}")
            else:
                if verbose:
                    print(f"Key not found in model: {key}")

    def load_with_swap(self, model_path, swap_dir="/tmp/model_swap", **kwargs):
        """Ultra low memory loading using disk swap for very large models"""
        import tempfile
        import pickle
        
        os.makedirs(swap_dir, exist_ok=True)
        
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
        
        print("Using disk swap loading for ultra large model...")
        
        # Load and immediately save to disk
        safetensor_files = _collect_safetensors_files(model_path)
        temp_files = []
        
        try:
            for i, file_path in enumerate(tqdm(safetensor_files, desc='Processing to swap')):
                # Load one file
                file_state = load_safetensors(file_path, device='cpu')
                
                # Convert portion
                converted = self._convert_partial_state(file_state, i, kwargs.get('verbose', True))
                del file_state
                
                # Save to temporary file
                temp_file = os.path.join(swap_dir, f"temp_state_{i}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(converted, f)
                temp_files.append(temp_file)
                
                del converted
                gc.collect()
            
            # Now load from swap files one by one
            for temp_file in tqdm(temp_files, desc='Loading from swap'):
                with open(temp_file, 'rb') as f:
                    partial_state = pickle.load(f)
                
                self._load_partial_state_dict(partial_state, False, kwargs.get('verbose', True))
                del partial_state
                gc.collect()
        
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        print("Swap loading complete!")
        return self

    def _convert_partial_state(self, file_state: dict, file_index: int, verbose: bool) -> dict:
        """Convert a portion of HF state for ultra low memory loading"""
        converted = {}
        
        for hf_key, tensor in file_state.items():
            # Convert keys based on patterns
            if 'model.embed_tokens.weight' in hf_key:
                converted['embedder.weight'] = tensor
                converted['lm_head.weight'] = tensor  # Share reference
            elif 'model.norm.weight' in hf_key:
                converted['model.norm.weight'] = tensor
            elif 'model.layers.' in hf_key:
                # Extract layer number
                parts = hf_key.split('.')
                layer_idx = parts[2]
                
                # Convert attention weights
                if 'self_attn.q_proj.weight' in hf_key:
                    # We need to collect Q, K, V for concatenation
                    # For now, store them separately and handle in a second pass
                    converted[f'_temp_q_{layer_idx}'] = tensor
                elif 'self_attn.k_proj.weight' in hf_key:
                    converted[f'_temp_k_{layer_idx}'] = tensor
                elif 'self_attn.v_proj.weight' in hf_key:
                    converted[f'_temp_v_{layer_idx}'] = tensor
                elif 'self_attn.o_proj.weight' in hf_key:
                    new_key = hf_key.replace('self_attn.o_proj.weight', 'self_attn.o_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'mlp.gate_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.gate_proj.weight', 'mlp.gate_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'mlp.up_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.up_proj.weight', 'mlp.up_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'mlp.down_proj.weight' in hf_key:
                    new_key = hf_key.replace('mlp.down_proj.weight', 'mlp.down_proj.linear.weight')
                    converted[new_key] = tensor
                elif 'input_layernorm.weight' in hf_key:
                    converted[hf_key] = tensor
                elif 'post_attention_layernorm.weight' in hf_key:
                    converted[hf_key] = tensor
        
        return converted

    def _load_partial_state_dict(self, partial_state: dict, strict: bool, verbose: bool):
        """Load a partial state dict into the model"""
        # Handle QKV concatenation for any complete sets
        qkv_layers = {}
        for key in list(partial_state.keys()):
            if key.startswith('_temp_'):
                parts = key.split('_')
                qkv_type = parts[1]  # q, k, or v
                layer_idx = parts[2]
                
                if layer_idx not in qkv_layers:
                    qkv_layers[layer_idx] = {}
                qkv_layers[layer_idx][qkv_type] = partial_state.pop(key)
        
        # Create QKV concatenations for complete sets
        for layer_idx, qkv_dict in qkv_layers.items():
            if all(k in qkv_dict for k in ['q', 'k', 'v']):
                qkv_weight = torch.cat([qkv_dict['q'], qkv_dict['k'], qkv_dict['v']], dim=0)
                partial_state[f'model.layers.{layer_idx}.self_attn.qkv_proj.linear.weight'] = qkv_weight
        
        # Load what we can
        model_state = self.state_dict()
        for key, tensor in partial_state.items():
            if key in model_state:
                try:
                    model_state[key].copy_(tensor)
                    if verbose:
                        print(f"Loaded: {key}")
                except Exception as e:
                    if verbose:
                        print(f"Failed to load {key}: {e}")

    def load_with_swap(self, model_path, swap_dir="/tmp/model_swap", **kwargs):
        """Ultra low memory loading using disk swap for very large models"""
        import tempfile
        import pickle
        
        os.makedirs(swap_dir, exist_ok=True)
        
        print("Using disk swap loading for ultra large model...")
        
        # Load and immediately save to disk
        safetensor_files = self._collect_safetensors_files(model_path)
        temp_files = []
        
        try:
            for i, file_path in enumerate(tqdm(safetensor_files, desc='Processing to swap')):
                # Load one file
                file_state = load_safetensors(file_path, device='cpu')
                
                # Convert portion
                converted = self._convert_partial_state(file_state, i, kwargs.get('verbose', True))
                del file_state
                
                # Save to temporary file
                temp_file = os.path.join(swap_dir, f"temp_state_{i}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(converted, f)
                temp_files.append(temp_file)
                
                del converted
                gc.collect()
            
            # Now load from swap files one by one
            for temp_file in tqdm(temp_files, desc='Loading from swap'):
                with open(temp_file, 'rb') as f:
                    partial_state = pickle.load(f)
                
                self._load_partial_state_dict(partial_state, False, kwargs.get('verbose', True))
                del partial_state
                gc.collect()
        
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        print("Swap loading complete!")
        return self

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