"""
Structured logging and artifact management for paper experiments.

Each training run creates a self-contained directory under results/paper/:

    results/paper/{experiment}/{model_name}_run{id}/
        config.json          # Frozen hyperparameters + hardware info
        train_log.jsonl      # One JSON line per log interval
        checkpoints/         # Model checkpoints at eval intervals
        eval/                # Eval results per checkpoint
        final_metrics.json   # Best checkpoint metrics + summary
"""

import json
import subprocess
import time
from pathlib import Path

import torch

RESULTS_ROOT = Path("results") / "paper"


def create_run_dir(experiment: str, model_name: str, run_id: int = 0) -> Path:
    """Create and return a run directory.

    Path: results/paper/{experiment}/{model_name}_run{run_id}/
    """
    run_dir = RESULTS_ROOT / experiment / f"{model_name}_run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    return run_dir


def capture_hardware_info() -> dict:
    """Auto-detect GPU, CUDA, and PyTorch versions."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
    return info


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_config(run_dir: Path, config: dict):
    """Save config.json with auto-detected hardware and git hash.

    The config dict should contain model_config, training, and any
    experiment-specific fields. Hardware and git info are added automatically.
    """
    full_config = {
        **config,
        "hardware": capture_hardware_info(),
        "git_hash": _get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(full_config, f, indent=2)
    return path


def log_step(run_dir: Path, step: int, metrics: dict):
    """Append one JSON line to train_log.jsonl."""
    entry = {"step": step, **metrics}
    path = run_dir / "train_log.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_checkpoint(run_dir: Path, model, optimizer, step: int):
    """Save model and optimizer state."""
    path = run_dir / "checkpoints" / f"step_{step}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(path: Path, model, optimizer=None):
    """Load a checkpoint. Returns the step number."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


def save_eval_results(run_dir: Path, step: int, benchmark: str, results: dict):
    """Save eval results for a specific checkpoint and benchmark."""
    path = run_dir / "eval" / f"step_{step}_{benchmark}.json"
    with open(path, "w") as f:
        json.dump({"step": step, "benchmark": benchmark, **results}, f, indent=2)
    return path


def save_final_metrics(run_dir: Path, metrics: dict):
    """Save summary metrics at end of run."""
    path = run_dir / "final_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_train_log(run_dir: Path) -> list[dict]:
    """Load all entries from train_log.jsonl."""
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
