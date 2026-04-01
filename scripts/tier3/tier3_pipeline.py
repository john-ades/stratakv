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
    console.print("[bold cyan]🚀 Starting StrataKV Tier 3 End-to-End Smoke Test Pipeline 🚀[/bold cyan]\n")
    
    # Common parameters for a fast smoke test
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    base_matrices = "outputs/smoke_test_base.pt"
    t2_healed_matrices = "outputs/smoke_test_t2_healed.pt"
    t3_healed_matrices = "outputs/smoke_test_t3_sonic_healed.pt"
    
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
    run_step(extraction_cmd, "Phase 1: Tier 2 Offline Calibration Extraction")
    
    # Step 2: Phase 5 Curriculum Healing for Tier 2
    healing_t2_cmd = [
        sys.executable, "scripts/tier2/tier2_healing.py",
        "--model-id", model_id,
        "--matrices-path", base_matrices,
        "--output-path", t2_healed_matrices,
        "--max-steps", "1",        # Only 1 backprop step for smoke test
        "--seq-len", "64",         
        "--prefix-len", "32",      
        "--tier1-size", "16",      
        "--dataset", "dummy",      
        "--device", "cpu"          
    ]
    run_step(healing_t2_cmd, "Phase 5: Tier 2 TransMLA Curriculum Healing")
    
    # Step 3: Phase 5 Curriculum Healing for Tier 3 
    # This distills TransMLA latents into SONIC Nexus tokens
    healing_t3_cmd = [
        sys.executable, "scripts/tier3/tier3_healing.py",
        "--model-id", model_id,
        "--matrices-path", t2_healed_matrices,
        "--output-path", t3_healed_matrices,
        "--max-steps", "1",        # Only 1 backprop step for smoke test
        "--seq-len", "64",
        "--prefix-len", "16",
        "--tier1-size", "8",
        "--tier2-size", "16",
        "--dataset", "dummy",
        "--device", "cpu"
    ]
    run_step(healing_t3_cmd, "Phase 5: Tier 3 SONIC Curriculum Healing")
    
    # Step 4: E2E Generation using the fully 'healed' Tier 3 compressed cache
    # Set Tier 1 & Tier 2 sizes extremely small (16 each) so it forcefully 
    # spills to Tier 3 Nexus Buffer during the generation prompt.
    generation_cmd = [
        sys.executable, "scripts/tier3/tier3_llama.py",
        "--model-id", model_id,
        "--matrices-path", t3_healed_matrices,
        "--tier1-size", "8",
        "--tier2-size", "16",
        "--max-new-tokens", "64"   # Generate enough tokens to guarantee tier 3 cache hits over time
    ]
    run_step(generation_cmd, "Final Evaluation: Tier 3 End-to-End Inference Generation")
    
    console.print("\n[bold green]🎉 E2E Tier 3 Phase 6 Pipeline completed! SONIC architecture integrates successfully! 🎉[/bold green]")

if __name__ == "__main__":
    main()
