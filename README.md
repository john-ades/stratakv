# StrataKV
**Scalable Tiered Representation Architecture for Token Attention**

StrataKV is designed to be trained on top of a base model, utilizing already powerful open model weights as a starting point.

**Goal:**
- Utilize already trained base models.
- Enable extremely long-term LLM inference on bounded hardware.

**Challenge:**
- Balance information-theoretic boundaries of a cascading memory cache to maximally extend an LLM's context.
StrataKV is an adaptive, multi-tiered caching architecture designed to intelligently manage and preserve the long-term memory of LLM agents. 
The system utilizes a cascading pipeline where high-priority context—such as the system prompt and the most recent N tokens—is preserved in an uncompressed L1 cache, while older KV tokens are progressively pushed into deeper tiers. 
Within these deeper strata, the framework continuously "crunches" the data using customizable, orthogonal compression techniques, such as sequence, channel, and precision compression. 
By allowing researchers to seamlessly swap and experiment with different combinations of these methods over time, the architecture provides a highly scalable way to maximize context density, reduce memory footprint, and solve the long-horizon memory bottlenecks of autonomous agents.

## Current Plan

Tier 0:
    system prompt / attention sinks
    full precision / quality

Tier 1: 
    l1 cache for n recent tokens
    full precision / quality

Tier 2:
    TransMLA compression 
    full precision / compressed channels / extracted rope info

Tier 3: 
    SONIC compression
    full precision / compressed sequence into nexus tokens

    * drop after tier 3 full


### Features
- [x] Tiered KV caching system
- [x] Dimension Compression via [TransMLA](https://github.com/MuLabPKU/TransMLA)
- [x] Sequence Compression via [SONIC](https://arxiv.org/abs/2601.21927v1)


### Future Explorations + Tiers
- [ ] Precision Quantization Compression via [TurboQuant](https://arxiv.org/abs/2504.19874)
- [ ] Token Eviction via [SideQuest](https://arxiv.org/abs/2602.22603)
- [ ] Token Retrieval via [MemArt](https://openreview.net/forum?id=YolJOZOGhI)
- [ ] Latent Continuous Thought via [Coconut](https://arxiv.org/abs/2412.06769)
- [ ] Latent Tool Calling 
- [ ] Latent Semantic Knowledge Mapping via [Engram](https://arxiv.org/pdf/2601.07372)
- [ ] Lightweight Online Learning (tbd) 
- [ ] Extendable Multimodal Modules (tbd)
- [ ] Mixture of Expert Support
- [ ] Offline World Model Learning (tbd...trying to take advantage of recent World Model research) 
- [ ] Efficient Multi-Agent Cache Sharing/Communication via [Agent Vector Protocol](https://github.com/VectorArc/avp-python)

## Running Experiments

To train and distill StrataKV locally or on a remote GPU cluster, we provide a unified fault-tolerant execution script. This orchestration pipeline coordinates Phase 1 extraction, Phase 2 TransMLA Curriculum Healing, and Phase 3 SONIC Distillation.

### Setup and Configuration

1. **Install Dependencies:**
   This project uses `uv` for rapid, deterministic dependency management. Install the environment:
   ```bash
   uv sync
   ```

2. **Configure Accelerate Environment:**
   Before running the experiment across multiple GPUs, configure the distributed topology. On your target machine or head node, run:
   ```bash
   uv run accelerate config
   ```
   *Follow the prompts, configure the amount of distributed processes, and enable `bf16` precision.*

3. **Authenticate Services:**
   We strictly rely on Weights & Biases for telemetry logging and remote state checkpointing (necessary for spot instance persistence). You also need a Hugging Face token to download the base models and datasets. Authenticate and set your environment variables:
   ```bash
   export WANDB_ENTITY="your-wandb-entity"
   export WANDB_PROJECT="your-wandb-project"
   export HF_TOKEN="your-hf-token"
   
   uv run wandb login
   uv run huggingface-cli login
   ```

### Execution

Because the pipeline natively patches attention distributions and loss functions across devices via PyTorch Distributed Data Parallelism (DDP), you MUST launch it using `accelerate`:

```bash
uv run accelerate launch scripts/run_experiment.py
```

### Spot Instance Fault Tolerance

The pipeline is inherently built to survive unmanaged, low-cost spot instance preemptions and terminations. 
- **W&B Remote Checkpointing:** Every 500 steps, the active `accelerate` trainer compresses and uploads the local checkpoint snapshot (layer weights, clustered GPU buffers, optimizer states, random topologies) directly into your active W&B run as a flagged `checkpoint` artifact.
- **Auto-Resumption:** Should the process crash mid-phase, any replacement spot instance spun up utilizing the identical `accelerate launch` command will interrogate W&B, fetch your `latest` checkpoint for the specific phase, and silently orchestrate dataloader synchronization via `accelerator.skip_first_batches`. The architecture guarantees lossless training continuity horizontally without duplication vulnerabilities.

### Evaluation

Once `run_experiment.py` finishes Phase 3 successfully, it exports the target dimension and sequence-constrained components automatically as a standard PyTorch dictionary artifact (e.g. `outputs/experiment_01/healed_t3_sonic.pt`). 

The script additionally patches local memory contexts logically into massive batches, preventing VRAM OOMs during long-context probing queries. Simply integrate the `StrataKVConfig` definitions inside standard language model test beds (NeedleBench, RULER, MTBench) as documented explicitly in the script's post-training console dump!
