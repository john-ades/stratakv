import torch
from typing import Optional, Tuple, Dict, Any, List
try:
    from transformers.cache_utils import DynamicCache as BaseCache
except ImportError:
    # Fallback if transformers is not installed or too old
    class BaseCache:
        pass

import threading

from .core.config import StrataKVConfig
from .tiers.tier0_sink import Tier0Sink
from .tiers.tier1_recent import Tier1Recent
from .tiers.tier2_latent import Tier2Latent
from .compression.transmla import TransMLACruncher
from .clustering.buffer import AbitClusterBuffer
from .tiers.tier3_sonic import Tier3Sonic

class StrataKVCache(BaseCache):
    """
    StrataKVCache manages the tiered cascading cache sequence for all layers 
    during an LLM forward pass.
    """
    def __init__(self, config: StrataKVConfig):
        super().__init__()
        self.config = config
        self._lock = threading.Lock()
        
        # We hold lists of tiers per layer index
        self._tier0_sinks: List[Optional[Tier0Sink]] = []
        self._tier1_recents: List[Optional[Tier1Recent]] = []
        self._tier2_latents: List[Optional[Tier2Latent]] = []
        self._tier3_buffers: List[Optional[AbitClusterBuffer]] = []
        self._tier3_sonics: List[Optional[Tier3Sonic]] = []
        
        self.seen_tokens = 0
        
    def _ensure_initialized(self, layer_idx: int):
        # Dynamically extend lists to support up to layer_idx
        while len(self._tier0_sinks) <= layer_idx:
            if self.config.enable_tier0:
                self._tier0_sinks.append(Tier0Sink(self.config.tier0_size, len(self._tier0_sinks)))
            else:
                self._tier0_sinks.append(None)
                
            if self.config.enable_tier1:
                self._tier1_recents.append(Tier1Recent(self.config.tier1_size, len(self._tier1_recents)))
            else:
                self._tier1_recents.append(None)
                
            if self.config.enable_tier2:
                self._tier2_latents.append(Tier2Latent(self.config.tier2_size, len(self._tier2_latents)))
            else:
                self._tier2_latents.append(None)
                
            if self.config.enable_tier3:
                self._tier3_buffers.append(AbitClusterBuffer(self.config))
                self._tier3_sonics.append(Tier3Sonic(self.config.tier3_size, len(self._tier3_sonics)))
            else:
                self._tier3_buffers.append(None)
                self._tier3_sonics.append(None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new key/value states.
        Then, reconstructs the full kv-cache for this layer to return to the model attention mechanism.
        """
        with self._lock:
            self._ensure_initialized(layer_idx)
            
            # 1) Pass tokens through Tier 0 Sink first
            evicted_keys = key_states
            evicted_values = value_states
            
            t0 = self._tier0_sinks[layer_idx]
            if t0 is not None and evicted_keys is not None:
                evicted_keys, evicted_values = t0.push(evicted_keys, evicted_values)
                
            # 2) Pass whatever spills out into Tier 1 Sliding Window
            t1 = self._tier1_recents[layer_idx]
            dropped_keys, dropped_values = None, None
            if t1 is not None and evicted_keys is not None:
                # We push evicted from T0 into T1
                dropped_keys, dropped_values = t1.push(evicted_keys, evicted_values)
                
            # 3) Crunch tokens from Tier 1 into Tier 2 via TransMLA
            t2 = self._tier2_latents[layer_idx]
            cruncher = None
            if cache_kwargs is not None:
                cruncher = cache_kwargs.get("strata_cruncher", None)
                
            if t2 is not None and cruncher is not None and dropped_keys is not None and dropped_values is not None:
                # Ensure matrices are on the right device and dtype
                cruncher.to(device=dropped_keys.device, dtype=dropped_keys.dtype)
                
                # Crunch to latent C_kv and positional K_rope
                c_kv, k_rope = cruncher(dropped_keys, dropped_values)
                
                # Evicted from T2 are just discarded from memory entirely (or moved to Tier 3)
                # For now, baseline T2 discards them
                t2.push(c_kv, k_rope)
                
                # Phase 1: Push latents into the Tier 3 ABIT buffer to detect semantic boundaries
                t3_buf = self._tier3_buffers[layer_idx]
                t3_sonic = self._tier3_sonics[layer_idx]
                sonic_cruncher = None
                if cache_kwargs is not None:
                    sonic_cruncher = cache_kwargs.get("sonic_cruncher", None)
                    
                if t3_buf is not None:
                    sealed_clusters = t3_buf.push(c_kv, k_rope)
                    
                    # Phase 3: Orthogonal Sequence Compression (SONIC Cruncher)
                    if t3_sonic is not None and sonic_cruncher is not None and sealed_clusters:
                        sonic_cruncher.to(device=dropped_keys.device, dtype=dropped_keys.dtype)
                        batch_size = key_states.shape[0]
                        for cluster in sealed_clusters:
                            k = getattr(self.config, "tier3_k", 4)
                            # Compress the sequence
                            c_nexus = sonic_cruncher(cluster.c_kv, k)
                            # Phase 2: Positional binding, expand Medoid K_rope
                            k_rope_expanded = cluster.expand_medoid_k_rope(k)
                            t3_sonic.push(c_nexus, k_rope_expanded, batch_idx=cluster.batch_idx, batch_size=batch_size)
                
            # 4) Reconstruct the view of the full cache for this layer
            all_k = []
            all_v = []
            
            if t0 is not None:
                k0, v0 = t0.get_cache()
                if k0 is not None:
                    all_k.append(k0)
                    all_v.append(v0)
                    
            if t1 is not None:
                k1, v1 = t1.get_cache()
                if k1 is not None:
                    all_k.append(k1)
                    all_v.append(v1)
        
        
        
        # If no tiers are enabled (unlikely), just return incoming
        if len(all_k) == 0:
            return key_states, value_states
            
        full_k = torch.cat(all_k, dim=2)
        full_v = torch.cat(all_v, dim=2)
        
        if layer_idx == 0:
            # Update seen tokens only once per layer cycle
            # This is technically `seq_len` returned to the model max, 
            # but wait, the model expects cache length to be total length if DynamicCache.
            # We can track the internal size.
            pass
            
        return full_k, full_v
        
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self._tier0_sinks) <= layer_idx:
            return 0
        l = 0
        t0 = self._tier0_sinks[layer_idx]
        if t0 and t0.k_cache is not None:
            l += t0.k_cache.shape[2]
            
        t1 = self._tier1_recents[layer_idx]
        if t1 and t1.k_cache is not None:
            l += t1.k_cache.shape[2]
            
        t2 = self._tier2_latents[layer_idx]
        if t2 and t2.k_cache is not None:
            seq_dim = 1 if t2.k_cache.dim() == 3 else 2
            l += t2.k_cache.shape[seq_dim]
            
        return l
        
    def get_max_length(self) -> Optional[int]:
        cap = 0
        if self.config.enable_tier0: cap += self.config.tier0_size
        if self.config.enable_tier1: cap += self.config.tier1_size
        if self.config.enable_tier2: cap += self.config.tier2_size
        return cap
        
    def get_tier2_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_kv_t2, K_rope_t2) for the given layer.
        """
        if len(self._tier2_latents) <= layer_idx:
            return None, None
        t2 = self._tier2_latents[layer_idx]
        if t2 is None:
            return None, None
        return t2.get_cache()
        
    def get_tier3_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_nexus_t3, K_rope_t3) for the given layer.
        """
        if len(self._tier3_sonics) <= layer_idx:
            return None, None
        t3 = self._tier3_sonics[layer_idx]
        if t3 is None:
            return None, None
        return t3.get_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Used in beam search. Not implemented for baseline."""
        raise NotImplementedError("Beam search not supported yet in StrataKVCache")
