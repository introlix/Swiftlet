import os
import json
import torch
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors


class PreTrainedModel:
    def __init__(self):
        pass

    def convert_hf_to_your_model(self, hf_state: dict) -> dict:
        """Hardcoded conversion from HF Qwen2 to your model structure"""
        new_state = {}
        
        # 1) Embeddings - Set both embedder.weight and lm_head.weight
        embed_weight = hf_state['model.embed_tokens.weight']
        new_state['embedder.weight'] = embed_weight
        new_state['lm_head.weight'] = embed_weight.clone()  # Separate copy for lm_head
        
        # 2) Get number of layers from the state dict
        layer_nums = set()
        for key in hf_state.keys():
            if 'model.layers.' in key:
                layer_num = key.split('model.layers.')[1].split('.')[0]
                if layer_num.isdigit():
                    layer_nums.add(int(layer_num))
        
        num_layers = max(layer_nums) + 1 if layer_nums else 0
        print(f"Detected {num_layers} layers")
        
        # 3) Process each layer
        for i in range(num_layers):
            hf_prefix = f'model.layers.{i}'
            your_prefix = f'model.layers.{i}'
            
            # === ATTENTION ===
            # QKV combination (Q, K, V → qkv_proj.linear)
            q_weight = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
            k_weight = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
            v_weight = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
            new_state[f'{your_prefix}.self_attn.qkv_proj.linear.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            # QKV bias (if exists)
            q_bias_key = f'{hf_prefix}.self_attn.q_proj.bias'
            if q_bias_key in hf_state:
                q_bias = hf_state[q_bias_key]
                k_bias = hf_state[f'{hf_prefix}.self_attn.k_proj.bias']
                v_bias = hf_state[f'{hf_prefix}.self_attn.v_proj.bias']
                new_state[f'{your_prefix}.self_attn.qkv_proj.linear.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
            
            # Output projection
            new_state[f'{your_prefix}.self_attn.o_proj.linear.weight'] = hf_state[f'{hf_prefix}.self_attn.o_proj.weight']
            if f'{hf_prefix}.self_attn.o_proj.bias' in hf_state:
                new_state[f'{your_prefix}.self_attn.o_proj.linear.bias'] = hf_state[f'{hf_prefix}.self_attn.o_proj.bias']
            
            # === MLP ===
            new_state[f'{your_prefix}.mlp.gate_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.gate_proj.weight']
            new_state[f'{your_prefix}.mlp.up_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.up_proj.weight']
            new_state[f'{your_prefix}.mlp.down_proj.linear.weight'] = hf_state[f'{hf_prefix}.mlp.down_proj.weight']
            
            # MLP bias (if exists)
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                bias_key = f'{hf_prefix}.mlp.{proj}.bias'
                if bias_key in hf_state:
                    new_state[f'{your_prefix}.mlp.{proj}.linear.bias'] = hf_state[bias_key]
            
            # === LAYER NORMS ===
            new_state[f'{your_prefix}.input_layernorm.weight'] = hf_state[f'{hf_prefix}.input_layernorm.weight']
            new_state[f'{your_prefix}.post_attention_layernorm.weight'] = hf_state[f'{hf_prefix}.post_attention_layernorm.weight']
        
        # 4) Final norm
        new_state['model.norm.weight'] = hf_state['model.norm.weight']
        
        return new_state

    def from_pretrained(self, model_path, map_location='cpu', strict=False, verbose=True):
        """Load HF model with hardcoded conversion"""
        
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
        
        # Get your model's target state dict
        target_state = self.state_dict()
        
        # Try loading safetensors first
        safetensor_files = _collect_safetensors_files(model_path)
        if safetensor_files:
            if verbose:
                print(f"Loading {len(safetensor_files)} safetensors files...")
            
            hf_state = {}
            for file_path in tqdm(safetensor_files, desc='Loading', disable=not verbose):
                hf_state.update(load_safetensors(file_path, device=map_location))
            
            # Convert HF state to your model format
            converted_state = self.convert_hf_to_your_model(hf_state)
            
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
            return self
        
        # Fallback to PyTorch files
        pytorch_file = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.isfile(pytorch_file):
            if verbose:
                print("Loading pytorch_model.bin...")
            
            checkpoint = torch.load(pytorch_file, map_location=map_location)
            hf_state = checkpoint.get('model_state_dict', checkpoint)
            
            converted_state = self.convert_hf_to_your_model(hf_state)
            self.load_state_dict(converted_state, strict=strict)
            return self
        
        # Try sharded PyTorch files
        shard_files = [f for f in os.listdir(model_path) if f.startswith('pytorch_model-') and f.endswith('.bin')]
        if shard_files:
            if verbose:
                print(f"Loading {len(shard_files)} PyTorch shard files...")
            
            hf_state = {}
            for shard_file in tqdm(sorted(shard_files), desc='Loading shards', disable=not verbose):
                checkpoint = torch.load(os.path.join(model_path, shard_file), map_location=map_location)
                hf_state.update(checkpoint.get('model_state_dict', checkpoint))
            
            converted_state = self.convert_hf_to_your_model(hf_state)
            self.load_state_dict(converted_state, strict=strict)
            return self
        
        raise FileNotFoundError(f"No valid checkpoint found at {model_path}")