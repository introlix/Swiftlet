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
        # (Paste the full base_patterns list here)
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
        combos = {}
        groups = defaultdict(dict)
        for k,t in source_dict.items():
            if any(at in k for at in ['self_attn','global_attn','attn','attention']) and any(pj in k for pj in ['q_proj','k_proj','v_proj','query','key','value']):
                m = re.search(r"(?:layers|h)\.(\d+)\.",k)
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
        if any('qkv_proj' in k or 'c_attn' in k for k in target_dict):
            for lid,qd in groups.items():
                ln,att = lid.split('.')
                # weights
                if 'q_weight' in qd and 'k_weight' in qd and 'v_weight' in qd:
                    qk,vk,nk=(qd['q_weight'][1],qd['k_weight'][1],qd['v_weight'][1])
                    cat = torch.cat([qk,nk,vk],dim=0)
                    cand = [
                        f"model.layers.{ln}.{att}.qkv_proj.weight",
                        f"transformer.h.{ln}.attn.c_attn.weight",
                    ]
                    for c in cand:
                        for var in self.generate_key_variants(c):
                            if var in target_dict and target_dict[var].shape==cat.shape:
                                combos[var]=cat
                                break
                # biases similar...
                if 'q_bias' in qd and 'k_bias' in qd and 'v_bias' in qd:
                    qb,kb,vb=(qd['q_bias'][1],qd['k_bias'][1],qd['v_bias'][1])
                    cat = torch.cat([qb,kb,vb],dim=0)
                    cand = [
                        f"model.layers.{ln}.{att}.qkv_proj.bias",
                        f"transformer.h.{ln}.attn.c_attn.bias",
                    ]
                    for c in cand:
                        for var in self.generate_key_variants(c):
                            if var in target_dict and target_dict[var].shape==cat.shape:
                                combos[var]=cat
                                break
        return combos


class PreTrainedModel:
    def __init__(self, custom_patterns=None):

        self.mapper = UniversalParameterMapper(custom_patterns)

    def split_qkv_if_needed(self, source_dict):
        updates = {}
        for k in list(source_dict):
            t = source_dict[k]
            tag = '.weight' if k.endswith('.weight') else ('.bias' if k.endswith('.bias') else None)
            if tag and (('attn.c_attn' in k or '.qkv_proj' in k) and t.ndim in (1,2)):
                n = t.shape[0]
                if n%3: continue
                h = n//3
                parts = list(t.split(h,dim=0))
                pref = k.rsplit('.',2)[0]+'.self_attn'
                for p,name in zip(parts,['q_proj','k_proj','v_proj']): updates[f"{pref}.{name}{tag}"]=p
                del source_dict[k]
        return updates

    def from_pretrained(self, model_path, map_location='cpu', strict=False, verbose=True):
        def _collect_safetensors_files(path):
            if os.path.isfile(path) and path.endswith('.safetensors'): return [path]
            if os.path.isdir(path):
                idx=os.path.join(path,'model.safetensors.index.json')
                if os.path.isfile(idx):
                    idxd=json.load(open(idx,'r'))
                    return sorted(set(os.path.join(path,v) for v in idxd['weight_map'].values()))
                return sorted(os.path.join(path,f) for f in os.listdir(path) if f.endswith('.safetensors'))
            return []
        def _smart_mapping(src, tgt):
            splits=self.split_qkv_if_needed(src)
            final = {**splits}
            combos=self.mapper.handle_qkv_combination(src,tgt)
            final.update(combos)
            proc={k:(v.to(torch.float32) if v.dtype==torch.float16 else v) for k,v in src.items()}
            maps=self.mapper.find_best_matches(list(proc),list(tgt),proc,tgt)
            for s,t in maps.items(): final[t]=proc[s]
            if verbose: print(f"Mapped {len(final)}/{len(src)} → {len(final)}/{len(tgt)}")
            return final

        target = self.state_dict()
        safefiles = _collect_safetensors_files(model_path)
        if safefiles:
            raw={}
            for f in tqdm(safefiles,desc='Load safetensors',disable=not verbose): raw.update(load_safetensors(f,map_location))
            mapped_state_dict, missing_keys, unmapped_keys = _smart_mapping(raw, target)
            print("❗️ Missing TARGET keys:", missing_keys)
            print("❓ Unexpected SOURCE keys:", unmapped_keys)
            self.load_state_dict(mapped_state_dict, strict=strict)
            return
        # PyTorch .bin and shards fallback
        bin_path=os.path.join(model_path,'pytorch_model.bin')
        if os.path.isfile(bin_path):
            ckpt=torch.load(bin_path,map_location)
            src=ckpt.get('model_state_dict',ckpt)
            mapped=_smart_mapping(src,target)
            self.load_state_dict(mapped,strict=strict)
            return
        raise FileNotFoundError(f"No checkpoint at {model_path}")