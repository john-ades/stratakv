import os
import sys
import time
import typer
from rich.console import Console

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import the core extraction function
from src.models.llama.tier2_phase1_extraction import run_offline_calibration
from src.core.config import DEFAULT_MODEL_ID

app = typer.Typer(help="Tier 2 Phase 1: TransMLA Offline Calibration Extraction")
console = Console()

@app.command()
def extract(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id", "-m",
        help="HuggingFace Model ID to calibrate"
    ),
    dataset_name: str = typer.Option(
        "wikitext", 
        "--dataset", "-d",
        help="Dataset name for calibration"
    ),
    dataset_config: str = typer.Option(
        "wikitext-2-raw-v1",
        "--dataset-config", "-c",
        help="Dataset config name"
    ),
    seq_len: int = typer.Option(
        512, 
        "--seq-len", "-s",
        help="Sequence length per chunk during harvesting"
    ),
    target_rank: int = typer.Option(
        128, 
        "--target-rank", "-r",
        help="Target hidden dimension rank for the PCA projection"
    ),
    rope_dim: int = typer.Option(
        64, 
        "--rope-dim", "-rp",
        help="Dimension for retained RoPE features per head"
    ),
    num_samples: int = typer.Option(
        50, 
        "--num-samples", "-n",
        help="Number of sequence chunks to harvest"
    ),
    output_path: str = typer.Option(
        "outputs/llama_transmla_base.pt", 
        "--output-path", "-o",
        help="Path to save the extracted untrained TransMLA matrices"
    )
):
    """
    Runner for Phase 1 Offline Calibration.
    Harvests un-RoPE'd keys/values from a calibration dataset and 
    computes the initial PCA/RoRoPE matrices (U_l, alpha, R_KV).
    """
    console.print(f"[bold green]Starting Phase 1 Extraction for {model_id}[/bold green]")
    start_time = time.time()
    
    # Run the existing calibration
    # Note: run_offline_calibration currently hardcodes dataset=wikitext, config=wikitext-2-raw-v1, 
    # but we pass what we can that is supported by its signature.
    # The signature is: model_id, target_rank, rope_dim, num_samples, seq_len, save_path
    
    try:
        run_offline_calibration(
            model_id=model_id,
            target_rank=target_rank,
            rope_dim=rope_dim,
            num_samples=num_samples,
            seq_len=seq_len,
            save_path=output_path
        )
        
        elapsed = time.time() - start_time
        console.print(f"[bold green]Extraction complete in {elapsed:.2f} seconds![/bold green]")
        console.print(f"[bold blue]Saved base matrices to -> {output_path}[/bold blue]")
        
    except Exception as e:
        console.print(f"[bold red]Extraction failed: {str(e)}[/bold red]")
        raise e

if __name__ == "__main__":
    app()
