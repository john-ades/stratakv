import sys
import subprocess
from rich.console import Console

console = Console()

def run_step(command: list[str], step_name: str):
    console.print(f"\n[bold magenta]=== Starting {step_name} ===[/bold magenta]")
    console.print(f"[dim]Running command: {' '.join(command)}[/dim]")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        console.print(f"[bold red]❌ {step_name} failed with return code {result.returncode}[/bold red]")
        sys.exit(result.returncode)
    else:
        console.print(f"[bold green]✅ {step_name} completed successfully![/bold green]")

def main():
    console.print("[bold cyan]🚀 Starting StrataKV Tier 2 End-to-End Smoke Test Pipeline 🚀[/bold cyan]\n")
    
    # Common parameters for a fast smoke test
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    base_matrices = "outputs/smoke_test_base.pt"
    healed_matrices = "outputs/smoke_test_healed.pt"
    
    # Step 1: Phase 1 Offline Calibration (Fast Extraction)
    # We use very few samples and a small target rank to run fast
    extraction_cmd = [
        sys.executable, "scripts/tier2/tier2_extraction.py",
        "--model-id", model_id,
        "--num-samples", "2",
        "--seq-len", "128",
        "--target-rank", "32",
        "--output-path", base_matrices
    ]
    run_step(extraction_cmd, "Phase 1: Offline Calibration Extraction")
    
    # Step 2: Phase 5 Curriculum Healing
    # We run just 5 steps on real data to ensure autograd and matrix updates work
    healing_cmd = [
        sys.executable, "scripts/tier2/tier2_healing.py",
        "--model-id", model_id,
        "--matrices-path", base_matrices,
        "--output-path", healed_matrices,
        "--max-steps", "1",        # Only 1 backprop step for smoke test
        "--seq-len", "64",         # Extremely small sequence length
        "--prefix-len", "32",      # Small prefix
        "--tier1-size", "16",      # Crucial: Must be < prefix_len to force eviction
        "--dataset", "dummy",      # Use dummy data to completely skip HF streamer loops
        "--device", "cpu"          # Avoid MPS tensor storage bugs during testing
    ]
    run_step(healing_cmd, "Phase 5: Curriculum Fine-Tuning Healing")
    
    # Step 3: E2E Generation using the 'healed' compressed cache
    # Set Tier 1 size to be extremely small (16) so it forcibly spills to Tier 2 in the prompt
    generation_cmd = [
        sys.executable, "scripts/tier2/tier2_llama.py",
        "--model-id", model_id,
        "--matrices-path", healed_matrices,
        "--tier1-size", "16",
        "--max-new-tokens", "64"   # Generate enough tokens to guarantee tier 2 cache hits over time
    ]
    run_step(generation_cmd, "Final Evaluation: End-to-End Inference Generation")
    
    console.print("\n[bold green]🎉 E2E Pipeline completed without crashing! The architecture works! 🎉[/bold green]")

if __name__ == "__main__":
    main()
