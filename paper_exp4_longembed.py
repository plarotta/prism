"""
Experiment 4: LongEmbed benchmark evaluation.

Evaluates best checkpoints from Experiment 1 on LongEmbed (6 tasks)
at multiple sequence lengths. Zero-shot evaluation only.

Usage:
    uv run python paper_exp4_longembed.py --checkpoint results/paper/exp1_1c/prism/checkpoints/step_50000.pt
    uv run python paper_exp4_longembed.py --exp1-dir results/paper/exp1_1c --max-lens 2048,4096,8192
"""

import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from paper_log import capture_hardware_info
from data.longembed_eval import evaluate_longembed, LONGEMBED_TASKS

from prism import prism_small, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small
from linear_rnn import build_linear_rnn_small

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
RESULTS_DIR = Path("results") / "paper" / "exp4"

MODEL_BUILDERS = {
    "prism": ("PRISM-Simplified", lambda ml: _build_prism(ml)),
    "transformer": ("Transformer", lambda ml: TransformerForEmbedding(
        transformer_small(vocab_size=VOCAB_SIZE, max_len=ml)
    )),
    "mamba": ("Mamba-Bidir", lambda ml: build_mamba_bidir_small(VOCAB_SIZE, ml)),
    "linear_rnn": ("Linear-RNN", lambda ml: build_linear_rnn_small(VOCAB_SIZE, ml)),
}


def _build_prism(max_len):
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


def load_checkpoint_into_model(model, checkpoint_path, device="cpu"):
    """Load model weights from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Find the best checkpoint in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def eval_model_longembed(
    model_key: str,
    checkpoint_path: Path,
    max_lens: list[int],
    tokenizer,
    device: str,
    batch_size: int = 16,
) -> dict:
    """Evaluate one model at multiple sequence lengths on LongEmbed."""
    model_name = MODEL_BUILDERS[model_key][0]
    results = {"name": model_name, "lengths": {}}

    for max_len in max_lens:
        print(f"\n  {model_name} @ max_len={max_len}")
        try:
            build_fn = MODEL_BUILDERS[model_key][1]
            model = build_fn(max_len)
            load_checkpoint_into_model(model, checkpoint_path, device)
            model = model.to(device)
            model.eval()

            le_results = evaluate_longembed(
                model, tokenizer, max_len=max_len,
                batch_size=batch_size, device=device,
            )
            results["lengths"][str(max_len)] = le_results
            print(f"    avg nDCG@10: {le_results['avg_ndcg@10']:.4f}")
            for task, scores in le_results["per_task"].items():
                print(f"      {task}: {scores['nDCG@10']:.4f}")

            model.cpu()
            del model

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results["lengths"][str(max_len)] = {"oom": True}
                print(f"    OOM at max_len={max_len}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    return results


def plot_longembed_results(results: dict, max_lens: list[int]):
    """Generate LongEmbed result plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "prism": "#2563EB", "transformer": "#DC2626",
        "mamba": "#16A34A", "linear_rnn": "#9333EA",
    }
    markers = {"prism": "o", "transformer": "s", "mamba": "D", "linear_rnn": "^"}

    # Plot 1: avg nDCG@10 vs max_len for each model
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for model_key, data in results.items():
        lens, scores = [], []
        for ml in max_lens:
            ml_data = data["lengths"].get(str(ml), {})
            if not ml_data.get("oom") and "avg_ndcg@10" in ml_data:
                lens.append(ml)
                scores.append(ml_data["avg_ndcg@10"])
        if lens:
            ax.plot(lens, scores, marker=markers.get(model_key, "o"),
                    color=colors.get(model_key, "#888888"),
                    label=data["name"], linewidth=2, markersize=8)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Average nDCG@10")
    ax.set_title("LongEmbed: Zero-Shot Performance vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "longembed_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {out_path}")

    # Plot 2: per-task breakdown at longest length
    task_names = list(LONGEMBED_TASKS.keys())
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    n_models = len(results)
    width = 0.8 / max(n_models, 1)

    for i, (model_key, data) in enumerate(results.items()):
        # Use the longest available length
        best_ml = None
        for ml in reversed(max_lens):
            ml_data = data["lengths"].get(str(ml), {})
            if not ml_data.get("oom") and "per_task" in ml_data:
                best_ml = str(ml)
                break
        if best_ml is None:
            continue

        per_task = data["lengths"][best_ml]["per_task"]
        vals = [per_task.get(t, {}).get("nDCG@10", 0) for t in task_names]
        x = np.arange(len(task_names))
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=f"{data['name']} ({best_ml})",
               color=colors.get(model_key, "#888888"), alpha=0.85)

    ax.set_ylabel("nDCG@10")
    ax.set_title("LongEmbed: Per-Task Performance")
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_xticklabels([t.replace("LE", "") for t in task_names],
                       rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = RESULTS_DIR / "longembed_per_task.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 4: LongEmbed evaluation")
    parser.add_argument("--exp1-dir", type=str, default=None,
                        help="Base dir containing exp1 run dirs (auto-find checkpoints)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated model_key:path pairs (e.g., prism:path/to/ckpt.pt)")
    parser.add_argument("--max-lens", type=str, default="2048,4096,8192",
                        help="Comma-separated max sequence lengths to evaluate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_lens = [int(x) for x in args.max_lens.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 4: LongEmbed Evaluation")
    print(f"  Max lengths: {max_lens}")
    print(f"  Device: {device}")
    print(f"  Hardware: {capture_hardware_info()}")
    print("=" * 70)

    # Resolve checkpoints
    checkpoint_map = {}  # model_key -> path
    if args.checkpoints:
        for pair in args.checkpoints.split(","):
            key, path = pair.split(":")
            checkpoint_map[key] = Path(path)
    elif args.exp1_dir:
        exp1_base = Path(args.exp1_dir)
        for model_key in MODEL_BUILDERS:
            # Search for run dirs
            for d in sorted(exp1_base.glob(f"{model_key}*")):
                ckpt = find_best_checkpoint(d)
                if ckpt:
                    checkpoint_map[model_key] = ckpt
                    break
    else:
        parser.error("Specify --exp1-dir or --checkpoints")
        return

    all_results = {}
    for model_key, ckpt_path in checkpoint_map.items():
        if model_key not in MODEL_BUILDERS:
            print(f"Unknown model: {model_key}")
            continue
        print(f"\nLoading {model_key} from {ckpt_path}")
        result = eval_model_longembed(
            model_key, ckpt_path, max_lens, tokenizer, device, args.batch_size,
        )
        all_results[model_key] = result

    # Save results
    out_path = RESULTS_DIR / "longembed_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Generate plots
    if all_results:
        plot_longembed_results(all_results, max_lens)

    print("\n=== Experiment 4 complete ===")


if __name__ == "__main__":
    main()
