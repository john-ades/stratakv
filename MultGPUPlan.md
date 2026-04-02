roadmap for the multi-GPU phase:

### Phase 1: Upgrade to Distributed Training (`Accelerate`)
Currently, your healing scripts (`tier2_healing.py` and `tier3_healing.py`) rely on `device_map="auto"`. While this is fine for inference, during training on a multi-GPU node (e.g., 4x or 8x A100s/H100s), it defaults to naive **Pipeline Parallelism** (putting layer 1 on GPU 0, layer 2 on GPU 1, etc.). This causes severe cross-GPU communication bottlenecks and terrible compute utilization.

*   **The Fix:** You already have `accelerate` in your `pyproject.toml`. Wrap your `HealingTrainer` and `Tier3HealingTrainer` inside an `Accelerator` object. Because you are freezing the massive base LLaMA weights and only training the tiny TransMLA/SONIC projection matrices, **Distributed Data Parallel (DDP)** will be lightning fast and easily fit into VRAM.
*   **Implementation:**
    ```python
    from accelerate import Accelerator
    accelerator = Accelerator()
    
    # Inside your trainer setup:
    self.model, self.optimizer, dataloader = accelerator.prepare(
        self.model, self.optimizer, dataloader
    )
    
    # In train_step, replace loss.backward() with:
    accelerator.backward(loss)
    ```

### Phase 2: Fix the Tier 3 Dataset Strategy (Crucial Literature Insight)
In `tier3_healing.py`, you currently default to `HuggingFaceFW/fineweb-edu`. While this is the exact right dataset for Tier 2 TransMLA healing (which requires ~6 Billion tokens of generic text to align the PCA projections), **it is the wrong dataset for Tier 3 SONIC.**
*   **The Insight:** The SONIC paper explicitly states that sequence compression thrives on *multi-turn dialogues with topic shifts*. In fact, the authors achieved convergence in just one hour on 4x A40 GPUs using a highly specific dataset of only **4,949 synthetic multi-turn dialogue samples**.
*   **The Fix:** ABIT clustering and Nexus tokens need to learn how to aggregate topics and dialogue turns. You need to swap `fineweb-edu` for a multi-turn conversational dataset (like `HuggingFaceH4/ultrachat_200k` or a synthetic equivalent). This will teach the `SonicCruncher` how to accurately utilize the Medoid spatial anchors to retrieve historical facts across distinct conversational boundaries.

### Phase 3: Build a Quantitative Benchmarking Suite
To prove StrataKV actually works, you need to measure its degradation against the uncompressed base model. Create a `scripts/evaluate/` directory containing the exact benchmarks used by the authors to prove your Tri-Path Attention isn't hallucinating:
*   **Needle In A Haystack (NIAH):** Inject a specific fact at token 1,000, pad the prompt to 64,000+ tokens (forcing the fact deep into Tier 3 Nexus storage), and ask the model to retrieve it. This validates your Phase 2 Medoid `expand_medoid_k_rope` logic.
*   **MTBench101 & SafeDialBench:** The SONIC paper heavily relies on these multi-turn benchmarks to prove that clustered Nexus tokens survive long conversation horizons better than naive eviction methods like H2O or StreamingLLM.
*   **CoreRes (Coreference Resolution):** A synthetic task to test if an entity introduced in Turn 1 can be accurately referenced by a pronoun in Turn 20 using only Tier 3 memory.

### Phase 4: VRAM & Throughput Profiling (The Selling Point)
The entire premise of StrataKV is infinite context bounded strictly by VRAM limits. You need hard data to prove that your architecture flattens the KV-cache scaling curve.
*   **VRAM Footprint Profiling:** Create a `profile_memory.py` script tracking `torch.cuda.max_memory_allocated()` as context grows from 4k $\rightarrow$ 32k $\rightarrow$ 128k tokens. You want to plot the linear VRAM growth of a standard LLaMA vs. the sub-linear/flattened growth of StrataKV as tokens dump into the dense $K$-budget Tier 3.
*   **Latency Metrics:** Track **TTFT** (Time To First Token) and **ITL** (Inter-Token Latency). Because your Tier 2 and Tier 3 operations use *late-decompression* (only expanding the 1D output vector `out_t2 = out_t2_latent @ W_UV`), your ITL should be drastically faster than a baseline model once the context exceeds 10k tokens.

### Phase 5: The Final Boss (Triton Kernels & vLLM Integration)
In `modeling_llama.py`, your `hybrid_attention` uses native PyTorch operations (`torch.cat`, `matmul`). This is perfect for eager-mode research and proving the math, but standard PyTorch attention will not yield the **10.6x inference speedup** advertised in the TransMLA paper due to memory bandwidth bottlenecks.
*   **The Next Engineering Leap:** Real acceleration happens when you bypass memory bandwidth limits. Because TransMLA effectively transforms a standard GQA model into an MLA model, your ultimate deployment goal should be formatting your healed `.pt` matrices to match DeepSeek's MLA layout. You can then load the model into **vLLM** or **SGLang** to utilize their highly optimized, custom Triton/CUDA kernels for the "Absorb" PagedAttention operation.

### Summary of your immediate next commands on the server:
1. Run `pip install accelerate flash-attn wandb`.
2. Run `accelerate config` to map your multi-GPU topology.
3. Run full-scale Tier 2 extraction to get real base matrices: 
   ```bash
   python scripts/tier2/tier2_extraction.py -m meta-llama/Llama-3.2-1B-Instruct -n 2000 -s 4096 -r 128 -o outputs/llama_t2_base_full.pt
   ```
4. Launch distributed Tier 2 Healing (using `fineweb-edu` for ~1B-6B tokens):
   ```bash
   accelerate launch scripts/tier2/tier2_healing.py -p outputs/llama_t2_base_full.pt -o outputs/llama_t2_healed.pt --seq-len 8192
   ```
5. Launch distributed Tier 3 Healing (using a **multi-turn** dataset):
   ```bash
   accelerate launch scripts/tier3/tier3_healing.py --dataset HuggingFaceH4/ultrachat_200k -p outputs/llama_t2_healed.pt -o outputs/llama_t3_sonic_healed.pt
   ```