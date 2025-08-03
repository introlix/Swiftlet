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
    def __init__(self, custom_patterns=None):
        # Enhanced base patterns with better Qwen2 support
        self.base_patterns = [
            # ================================
            # BASIC WRAPPER PATTERNS
            # ================================
            # Handle model prefixes
            (r"^module\.", ""),  # Remove DataParallel wrapper
            (r"^model\.", ""),  # Remove model wrapper
            (r"^", "model."),  # Add model prefix
            
            # ================================
            # EMBEDDING PATTERNS (Enhanced for your structure)
            # ================================
            # Token embeddings - Handle your embedder structure
            (r"^model\.embed_tokens\.weight$", "embedder.weight"),
            (r"^embedder\.weight$", "model.embed_tokens.weight"),
            (r"^embed_tokens\.weight$", "embedder.weight"),
            (r"^word_embeddings\.weight$", "embedder.weight"),
            (r"^token_embeddings\.weight$", "embedder.weight"),
            (r"^transformer\.wte\.weight$", "embedder.weight"),
            (r"^embeddings\.word_embeddings\.weight$", "embedder.weight"),
            
            # ================================
            # ATTENTION LAYER PATTERNS (Enhanced for your .linear structure)
            # ================================
            # Your model uses .linear submodules, HF doesn't
            # Convert FROM HF TO your structure
            (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", r"model.layers.\1.self_attn.q_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", r"model.layers.\1.self_attn.k_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", r"model.layers.\1.self_attn.v_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", r"model.layers.\1.self_attn.o_proj.linear.weight"),
            
            # Convert FROM your structure TO HF (reverse direction)
            (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.linear\.weight$", r"model.layers.\1.self_attn.q_proj.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.linear\.weight$", r"model.layers.\1.self_attn.k_proj.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.linear\.weight$", r"model.layers.\1.self_attn.v_proj.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.linear\.weight$", r"model.layers.\1.self_attn.o_proj.weight"),
            
            # Handle bias terms similarly
            (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$", r"model.layers.\1.self_attn.q_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$", r"model.layers.\1.self_attn.k_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$", r"model.layers.\1.self_attn.v_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$", r"model.layers.\1.self_attn.o_proj.linear.bias"),
            
            # ================================
            # MLP PATTERNS (Enhanced for your .linear structure)
            # ================================
            # Convert FROM HF TO your structure
            (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"model.layers.\1.mlp.gate_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"model.layers.\1.mlp.up_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"model.layers.\1.mlp.down_proj.linear.weight"),
            
            # Convert FROM your structure TO HF
            (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.linear\.weight$", r"model.layers.\1.mlp.gate_proj.weight"),
            (r"^model\.layers\.(\d+)\.mlp\.up_proj\.linear\.weight$", r"model.layers.\1.mlp.up_proj.weight"),
            (r"^model\.layers\.(\d+)\.mlp\.down_proj\.linear\.weight$", r"model.layers.\1.mlp.down_proj.weight"),
            
            # Handle MLP bias terms
            (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.bias$", r"model.layers.\1.mlp.gate_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.mlp\.up_proj\.bias$", r"model.layers.\1.mlp.up_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.mlp\.down_proj\.bias$", r"model.layers.\1.mlp.down_proj.linear.bias"),
            
            # ================================
            # QKV COMBINATION PATTERNS (Enhanced)
            # ================================
            # Handle combined QKV projections for your structure
            (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$", r"model.layers.\1.self_attn.qkv_proj.linear.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.linear\.weight$", r"model.layers.\1.self_attn.qkv_proj.weight"),
            (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.bias$", r"model.layers.\1.self_attn.qkv_proj.linear.bias"),
            (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.linear\.bias$", r"model.layers.\1.self_attn.qkv_proj.bias"),
            
            # ================================
            # NORMALIZATION PATTERNS
            # ================================
            # Layer normalization - these typically don't change for Qwen2
            (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"model.layers.\1.input_layernorm.weight"),
            (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"model.layers.\1.post_attention_layernorm.weight"),
            
            # Final/output normalization
            (r"^model\.norm\.weight$", "model.norm.weight"),
            (r"^norm\.weight$", "model.norm.weight"),
            (r"^final_layer_norm\.weight$", "model.norm.weight"),
            
            # ================================
            # QWEN-SPECIFIC ATTENTION PATTERNS
            # ================================
            # Qwen attention patterns (c_attn is combined QKV, c_proj is output)
            (r"(.*)\.attn\.c_attn\.(weight|bias)", r"\1.self_attn.qkv_proj.linear.\2"),
            (r"(.*)\.attn\.c_proj\.(weight|bias)", r"\1.self_attn.o_proj.linear.\2"),
            (r"(.*)\.self_attn\.qkv_proj\.linear\.(weight|bias)", r"\1.attn.c_attn.\2"),
            (r"(.*)\.self_attn\.o_proj\.linear\.(weight|bias)", r"\1.attn.c_proj.\2"),
            
            # ================================
            # ADDITIONAL GENERIC PATTERNS
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
            
            # Standard MLP patterns (bidirectional)
            (r"(.*)\.mlp\.gate_proj\.linear\.(weight|bias)", r"\1.mlp.gate_proj.\2"),
            (r"(.*)\.mlp\.up_proj\.linear\.(weight|bias)", r"\1.mlp.up_proj.\2"),
            (r"(.*)\.mlp\.down_proj\.linear\.(weight|bias)", r"\1.mlp.down_proj.\2"),
            (r"(.*)\.mlp\.gate_proj\.(weight|bias)", r"\1.mlp.gate_proj.linear.\2"),
            (r"(.*)\.mlp\.up_proj\.(weight|bias)", r"\1.mlp.up_proj.linear.\2"),
            (r"(.*)\.mlp\.down_proj\.(weight|bias)", r"\1.mlp.down_proj.linear.\2"),
            
            # ================================
            # OUTPUT HEAD PATTERNS (Enhanced for tied embeddings)
            # ================================
            # Language model head - Handle tied embeddings for Qwen2
            (r"^lm_head\.weight$", "lm_head.weight"),
            (r"^model\.lm_head\.weight$", "lm_head.weight"),
            (r"^output\.weight$", "lm_head.weight"),
            (r"^classifier\.weight$", "lm_head.weight"),
            
            # Handle tied embeddings (embeddings used as lm_head)
            (r"^model\.embed_tokens\.weight$", "lm_head.weight"),
            (r"^embedder\.weight$", "lm_head.weight"),
            (r"^embed_tokens\.weight$", "lm_head.weight"),
            
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
        ]
        
        self.patterns = self.base_patterns.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        self.compiled_patterns = [
            (re.compile(p), r) for p, r in self.patterns
        ]

    def generate_key_variants(self, key: str) -> List[str]:
        variants = {key}
        for _ in range(3):
            new_variants = set()
            for pattern_re, replacement in self.compiled_patterns:
                for v in variants:
                    if pattern_re.search(v):
                        new_variants.add(pattern_re.sub(replacement, v))
            variants.update(new_variants)
        # prefix/suffix expansions
        extras = set()
        for v in variants:
            extras.add(v)
            if v.startswith("module."):
                extras.add(v[7:])
            else:
                extras.add("module." + v)
            if v.startswith("model."):
                extras.add(v[6:])
            else:
                extras.add("model." + v)
        variants.update(extras)
        return list(variants)

    def calculate_similarity_score(self, key1: str, key2: str) -> float:
        base = SequenceMatcher(None, key1, key2).ratio()
        parts1, parts2 = key1.split('.'), key2.split('.')
        len_penalty = min(abs(len(parts1)-len(parts2))*0.1, 0.3)
        end_bonus = 0.2 if parts1[-1]==parts2[-1] else (0.1 if parts1[-1] in ['weight','bias'] and parts2[-1] in ['weight','bias'] else 0)
        match1 = re.search(r"\.(\d+)\.", key1);
        match2 = re.search(r"\.(\d+)\.", key2)
        layer_bonus = 0.1 if match1 and match2 and match1.group(1)==match2.group(1) else 0
        return base + end_bonus + layer_bonus - len_penalty

    def find_best_matches(
        self, source_keys: List[str], target_keys: List[str],
        source_dict: Dict, target_dict: Dict
    ) -> Dict[str, str]:
        target_set = set(target_keys)
        used = set()
        mappings = {}
        for src in source_keys:
            best, best_score = None, 0
            # exact via variants
            for var in self.generate_key_variants(src):
                if var in target_set and var not in used:
                    if source_dict[src].shape==target_dict[var].shape:
                        best, best_score = var, 1.0
                        break
            # fuzzy
            if not best:
                for tgt in target_keys:
                    if tgt in used: continue
                    if source_dict[src].shape==target_dict[tgt].shape:
                        score = self.calculate_similarity_score(src, tgt)
                        if score>0.7 and score>best_score:
                            best, best_score = tgt, score
            if best:
                mappings[src] = best
                used.add(best)
        return mappings

    def handle_qkv_combination(
        self, source_dict: Dict, target_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Enhanced QKV combination handling for your specific structure"""
        combos = {}
        groups = defaultdict(dict)
        
        # Collect Q, K, V projections from source
        for k, t in source_dict.items():
            if any(at in k for at in ['self_attn','global_attn','attn','attention']) and any(pj in k for pj in ['q_proj','k_proj','v_proj','query','key','value']):
                m = re.search(r"(?:layers|h)\.(\d+)\.", k)
                if not m: continue
                ln = m.group(1)
                att = 'global_attn' if 'global_attn' in k else ('self_attn' if 'self_attn' in k else ('attention' if 'attention' in k else 'attn'))
                lid = f"{ln}.{att}"
                if k.endswith('.weight'):
                    if any(x in k for x in ['q_proj','query']): groups[lid]['q_weight']=(k,t)
                    if any(x in k for x in ['k_proj','key']):   groups[lid]['k_weight']=(k,t)
                    if any(x in k for x in ['v_proj','value']): groups[lid]['v_weight']=(k,t)
                if k.endswith('.bias'):
                    if any(x in k for x in ['q_proj','query']): groups[lid]['q_bias']=(k,t)
                    if any(x in k for x in ['k_proj','key']):   groups[lid]['k_bias']=(k,t)
                    if any(x in k for x in ['v_proj','value']): groups[lid]['v_bias']=(k,t)
        
        # Check if target has combined QKV projections
        if any('qkv_proj' in k or 'c_attn' in k for k in target_dict):
            for lid, qd in groups.items():
                ln, att = lid.split('.')
                
                # Combine weights (Q, K, V order)
                if 'q_weight' in qd and 'k_weight' in qd and 'v_weight' in qd:
                    qk, kk, vk = (qd['q_weight'][1], qd['k_weight'][1], qd['v_weight'][1])
                    cat = torch.cat([qk, kk, vk], dim=0)
                    
                    # Try various candidate keys for your structure
                    candidates = [
                        f"model.layers.{ln}.{att}.qkv_proj.linear.weight",
                        f"model.layers.{ln}.{att}.qkv_proj.weight",
                        f"model.layers.{ln}.attn.c_attn.weight",
                        f"transformer.h.{ln}.attn.c_attn.weight",
                    ]
                    
                    for c in candidates:
                        for var in self.generate_key_variants(c):
                            if var in target_dict and target_dict[var].shape == cat.shape:
                                combos[var] = cat
                                break
                
                # Combine biases similarly
                if 'q_bias' in qd and 'k_bias' in qd and 'v_bias' in qd:
                    qb, kb, vb = (qd['q_bias'][1], qd['k_bias'][1], qd['v_bias'][1])
                    cat = torch.cat([qb, kb, vb], dim=0)
                    
                    candidates = [
                        f"model.layers.{ln}.{att}.qkv_proj.linear.bias",
                        f"model.layers.{ln}.{att}.qkv_proj.bias",
                        f"model.layers.{ln}.attn.c_attn.bias",
                        f"transformer.h.{ln}.attn.c_attn.bias",
                    ]
                    
                    for c in candidates:
                        for var in self.generate_key_variants(c):
                            if var in target_dict and target_dict[var].shape == cat.shape:
                                combos[var] = cat
                                break
        
        return combos


class PreTrainedModel:
    def __init__(self, custom_patterns=None):
        self.mapper = UniversalParameterMapper(custom_patterns)

    def split_qkv_if_needed(self, source_dict):
        """Split combined QKV projections if your model expects separate Q, K, V"""
        updates = {}
        for k in list(source_dict):
            t = source_dict[k]
            tag = '.weight' if k.endswith('.weight') else ('.bias' if k.endswith('.bias') else None)
            if tag and (('attn.c_attn' in k or '.qkv_proj' in k) and t.ndim in (1,2)):
                n = t.shape[0]
                if n % 3: continue
                h = n // 3
                parts = list(t.split(h, dim=0))
                
                # Determine the base prefix for separate projections
                if 'attn.c_attn' in k:
                    base_prefix = k.replace('attn.c_attn', 'self_attn')
                elif '.qkv_proj' in k:
                    base_prefix = k.replace('.qkv_proj', '.self_attn')
                else:
                    continue
                
                base_prefix = base_prefix.rsplit('.', 1)[0]
                
                # Add .linear if your structure requires it
                for p, name in zip(parts, ['q_proj', 'k_proj', 'v_proj']):
                    # Try with .linear first (for your structure)
                    key_with_linear = f"{base_prefix}.{name}.linear{tag}"
                    key_without_linear = f"{base_prefix}.{name}{tag}"
                    updates[key_with_linear] = p
                    updates[key_without_linear] = p
                
                del source_dict[k]
        
        return updates

    def from_pretrained(self, model_path, map_location='cpu', strict=False, verbose=True):
        def _collect_safetensors_files(path):
            if os.path.isfile(path) and path.endswith('.safetensors'): 
                return [path]
            if os.path.isdir(path):
                idx = os.path.join(path, 'model.safetensors.index.json')
                if os.path.isfile(idx):
                    idxd = json.load(open(idx, 'r'))
                    return sorted(set(os.path.join(path, v) for v in idxd['weight_map'].values()))
                return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors'))
            return []
        
        def _smart_mapping(src, tgt):
            if verbose:
                print(f"Source keys: {len(src)}, Target keys: {len(tgt)}")
                print("Sample source keys:", list(src.keys())[:5])
                print("Sample target keys:", list(tgt.keys())[:5])
            
            # Handle tied embeddings for lm_head
            if 'lm_head.weight' in tgt and 'lm_head.weight' not in src:
                if 'model.embed_tokens.weight' in src:
                    # Copy embedding weights to lm_head (your model has separate lm_head, HF uses tied)
                    src['lm_head.weight'] = src['model.embed_tokens.weight'].clone()
                    if verbose:
                        print("Copying tied embeddings: model.embed_tokens.weight → lm_head.weight")
                elif 'embedder.weight' in src:
                    src['lm_head.weight'] = src['embedder.weight'].clone()
                    if verbose:
                        print("Copying tied embeddings: embedder.weight → lm_head.weight")
            
            # Handle QKV splitting first
            splits = self.split_qkv_if_needed(src)
            final = {**splits}
            
            # Handle QKV combination (in case source has separate Q,K,V but target wants combined)
            combos = self.mapper.handle_qkv_combination(src, tgt)
            final.update(combos)
            
            # Convert to float32 if needed
            proc = {k: (v.to(torch.float32) if v.dtype == torch.float16 else v) for k, v in src.items()}
            
            # Find parameter mappings
            maps = self.mapper.find_best_matches(list(proc), list(tgt), proc, tgt)
            for s, t in maps.items(): 
                final[t] = proc[s]
            
            if verbose: 
                print(f"Mapped {len(final)}/{len(src)} source → {len(final)}/{len(tgt)} target")
                missing_in_target = set(tgt.keys()) - set(final.keys())
                if missing_in_target:
                    print(f"Missing in target: {sorted(list(missing_in_target))[:10]}...")
                unused_from_source = set(src.keys()) - {k for k in src.keys() if any(final.get(tk) is src[k] for tk in final.keys())}
                if unused_from_source:
                    print(f"Unused from source: {sorted(list(unused_from_source))[:10]}...")
            
            return final

        # Get target state dict from your model
        target = self.state_dict()
        
        # Try safetensors first
        safefiles = _collect_safetensors_files(model_path)
        if safefiles:
            raw = {}
            for f in tqdm(safefiles, desc='Loading safetensors', disable=not verbose): 
                raw.update(load_safetensors(f, map_location))
            mapped = _smart_mapping(raw, target)
            self.load_state_dict(mapped, strict=strict)
            return self
        
        # Fallback to PyTorch .bin files
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.isfile(bin_path):
            ckpt = torch.load(bin_path, map_location)
            src = ckpt.get('model_state_dict', ckpt)
            mapped = _smart_mapping(src, target)
            self.load_state_dict(mapped, strict=strict)
            return self
        
        # Try sharded PyTorch files
        shard_files = [f for f in os.listdir(model_path) if f.startswith('pytorch_model-') and f.endswith('.bin')]
        if shard_files:
            raw = {}
            for f in tqdm(sorted(shard_files), desc='Loading PyTorch shards', disable=not verbose):
                ckpt = torch.load(os.path.join(model_path, f), map_location)
                raw.update(ckpt.get('model_state_dict', ckpt))
            mapped = _smart_mapping(raw, target)
            self.load_state_dict(mapped, strict=strict)
            return self
        
        raise FileNotFoundError(f"No checkpoint found at {model_path}")