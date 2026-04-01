Here is a high-level, phased engineering strategy to stack Tier 3 (SONIC) orthogonally into `stratakv-torch` for LLaMA.

---

### Phase 1: Dynamic Semantic Boundary Detection (Abit Clustering)
Instead of waiting for an arbitrary `<|im_end|>` token, Tier 3 will passively monitor the semantic drift of tokens sequentially to naturally determine boundaries.

1. **The Streaming Cluster Buffer**: Create an intermediate `AbitClusterBuffer` inside `StrataKVCache`. As older tokens age out of the Tier 1 sliding window and are crunched into Tier 2 latents, they drop into this buffer.
2. **Online Evaluation**: For every new latent token ($C_{kv}$) that enters the buffer, evaluate it using your `abit` clustering logic. The algorithm tracks the semantic centroid and looks for variance spikes or distribution shifts.
3. **Triggering the Cut**: Once `abit` identifies a semantic boundary (i.e., the distance exceeds the adaptive threshold, indicating a topic or thought shift), the buffer is "sealed" as a contiguous **Cluster Segment**. This variable-length segment is instantly dispatched to the Tier 3 Cruncher.

### Phase 2: Saliency Anchoring & The Medoid Token (Positional Binding)
Once `abit` seals a Cluster Segment, you must generate the spatial anchor for the upcoming Nexus tokens *before* compressing the raw data. 

1. **Centroid & Medoid Extraction**: Calculate the pairwise distance of every latent token in the sealed cluster. The token that minimizes the average distance to all other tokens in the cluster is selected as the **Medoid Token** (the semantic "center of mass" representing the core concept).
2. **Positional Harvesting**: Extract the decoupled positional embedding tensor ($K_{rope}$) of this specific Medoid token. 
3. **RoPE Injection**: Instead of using SONIC's naive additive "turn embedding", explicitly map the Medoid's $K_{rope}$ to your $K$ Nexus tokens. When the LLM queries its memory later, the entire cluster of Nexus tokens will appear spatially localized exactly where the core concept occurred, maintaining precise positional synergy with the rest of the context.

### Phase 3: Orthogonal Sequence Compression (The SONIC Cruncher)
You are compressing the sequence ($N$ tokens $\rightarrow$ $K$ Nexus tokens) independent of how Tier 2 compresses the channel. 

1. **Nexus Base Initialization**: In `src/compression/sonic.py`, create a `SonicCruncher(nn.Module)`. Initialize a learnable base embedding $e_{base}$ for the Nexus tokens, projected to the latent $r_{kv}$ dimension.
2. **The Compression Forward Pass**: 
   * Perform an attention-weighted aggregation (as defined in the SONIC Information Bottleneck) where the $K$ Nexus base tokens act as Queries, and the sealed $N$-length TransMLA latents ($C_{kv}$) act as Keys/Values.
   * The result is $K$ highly dense Nexus Latents ($C_{nexus}$) that encapsulate the entire `abit` cluster.
3. **Cache Update (`Tier3Sonic`)**: 
   * Append the new $C_{nexus}$ and the Medoid $K_{rope}$ to `src/tiers/tier3_sonic.py`.
   * The original $N$ latent tokens of the cluster are permanently deleted from Tier 2 memory.

### Phase 4: Tri-Path Hybrid Attention (The Inference Forward Pass)
Because your Tier 3 Nexus tokens are just mathematically aggregated Tier 2 TransMLA latents, **they share the exact same latent dimensional space.** This makes the inference forward pass incredibly elegant. You don't need a separate decompression matrix for Tier 3!

Modify `_strata_llama_attention_forward` in `src/models/llama/modeling_llama.py`:
1. **Concatenation**: Because Tier 2 and Tier 3 share the $r_{kv}$ target rank and the decoupled $K_{rope}$ format, simply concatenate them along the sequence dimension:
   ```python
   C_kv_latent = torch.cat([C_kv_t3, C_kv_t2], dim=1)
   K_rope_latent = torch.cat([K_rope_t3, K_rope_t2], dim=2)
   ```
2. **Hierarchical Visibility Masking**: Apply SONIC's masking rule: Drop causal masking between Nexus tokens. All Tier 3 Nexus tokens should form a fully connected, bidirectionally visible memory graph, while standard T1/T2 tokens obey standard causal masking.
3. **Single Absorb Pass**: Pass the combined latent cache through the existing `TransMLAAbsorber` to compute `scores_deep`.
4. **Global Merge**: Compute the attention scores across Tier 1 (Dense) and the concatenated Latent Tiers. Apply a global softmax, and late-decompress the combined latent values using the existing TransMLA $W_{UV}$ matrix.

### Phase 5: Dynamic Budget Training & Curriculum Healing
Just like TransMLA, SONIC is a learning-based framework. The projection weights must be trained to actively aggregate and retrieve information via the Medoid anchor.

1. **Create `scripts/tier3/tier3_healing.py`**: Freeze LLaMA's base weights and your healed TransMLA matrices. Only unfreeze the parameters inside the `SonicCruncher`.
2. **Information Bottleneck Reconstruction ($\mathcal{L}_{Recon}$)**: Apply a reconstruction loss forcing the $K$ Medoid-anchored Nexus tokens to accurately reconstruct the TransMLA $C_{kv}$ representations of the dropped `abit` cluster body.
3. **End-to-End Distillation**: Train against a full-context uncompressed teacher model ($\mathcal{L}_{KD}$) and add a penalty if the model ignores the Nexus tokens during autoregressive generation ($\mathcal{L}_{Reg}$).
4. **Adaptive Budgeting (The $K$ Variable)**: During training, randomly alter the `abit` clustering variance threshold (resulting in varying cluster sizes like 10 tokens vs. 1000 tokens) and randomly sample the Nexus budget $K \in \{1, 2, 4\}$. This prevents the model from overfitting to a specific cluster size and enables fluid inference-time memory scaling (e.g., instructing the model to drop $K$ to $1$ when VRAM is critically low).