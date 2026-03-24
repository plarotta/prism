"""
Experiment 5: BEIR retrieval benchmark evaluation.

Evaluates best checkpoints from Experiment 1a/1b on BEIR (15 datasets).
Zero-shot evaluation at short sequence lengths (128-512 tokens).

Usage:
    uv run python paper_exp5_beir.py --checkpoints prism:path/ckpt.pt,transformer:path/ckpt.pt
    uv run python paper_exp5_beir.py --exp1-dir results/paper/exp1_1b
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
from data.beir_eval import evaluate_beir, BEIR_DATASETS

from prism import prism_small, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small
from linear_rnn import build_linear_rnn_small

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
RESULTS_DIR = Path("results") / "paper" / "exp5"

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


def eval_model_beir(
    model_key: str,
    checkpoint_path: Path,
    max_len: int,
    tokenizer,
    device: str,
    batch_size: int = 64,
    datasets: list[str] | None = None,
) -> dict:
    """Evaluate one model on BEIR."""
    model_name = MODEL_BUILDERS[model_key][0]
    print(f"\n  {model_name} @ max_len={max_len}")

    build_fn = MODEL_BUILDERS[model_key][1]
    model = build_fn(max_len)
    load_checkpoint_into_model(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    beir_results = evaluate_beir(
        model, tokenizer, max_len=max_len,
        batch_size=batch_size, device=device,
        datasets=datasets,
    )

    print(f"    avg nDCG@10: {beir_results['avg_ndcg@10']:.4f}")
    for ds_name, scores in beir_results.get("per_dataset", {}).items():
        print(f"      {ds_name}: {scores.get('nDCG@10', 0):.4f}")

    model.cpu()
    del model

    return {"name": model_name, "max_len": max_len, **beir_results}


# Published small-model baselines for comparison
PUBLISHED_BASELINES = {
    "bge-small-en-v1.5 (33M)": 51.68,
    "snowflake-arctic-s (33M)": 51.98,
    "e5-small-v2 (33M)": 49.04,
}


def plot_beir_results(results: dict, datasets_evaluated: list[str]):
    """Generate BEIR result plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "prism": "#2563EB", "transformer": "#DC2626",
        "mamba": "#16A34A", "linear_rnn": "#9333EA",
    }

    # Plot 1: Per-dataset grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    n_models = len(results)
    width = 0.8 / max(n_models, 1)

    for i, (model_key, data) in enumerate(results.items()):
        per_ds = data.get("per_dataset", {})
        vals = [per_ds.get(ds, {}).get("nDCG@10", 0) * 100 for ds in datasets_evaluated]
        x = np.arange(len(datasets_evaluated))
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=data["name"],
               color=colors.get(model_key, "#888888"), alpha=0.85)

    ax.set_ylabel("nDCG@10 (%)")
    ax.set_title("Experiment 5: BEIR Zero-Shot Retrieval")
    ax.set_xticks(np.arange(len(datasets_evaluated)))
    ax.set_xticklabels(datasets_evaluated, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = RESULTS_DIR / "beir_per_dataset.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {out_path}")

    # Plot 2: Average comparison with published baselines
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = []
    avgs = []
    bar_colors = []

    for model_key, data in results.items():
        names.append(data["name"])
        avgs.append(data.get("avg_ndcg@10", 0) * 100)
        bar_colors.append(colors.get(model_key, "#888888"))

    for baseline_name, baseline_score in PUBLISHED_BASELINES.items():
        names.append(baseline_name)
        avgs.append(baseline_score)
        bar_colors.append("#D1D5DB")

    bars = ax.barh(names, avgs, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, avgs):
        ax.annotate(f"{val:.1f}",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)

    ax.set_xlabel("Average nDCG@10 (%)")
    ax.set_title("BEIR Average: Our Models vs Published Baselines")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = RESULTS_DIR / "beir_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: BEIR evaluation")
    parser.add_argument("--exp1-dir", type=str, default=None,
                        help="Base dir containing exp1 run dirs")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated model_key:path pairs")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max sequence length for BEIR eval")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated BEIR dataset names (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    eval_datasets = args.datasets.split(",") if args.datasets else BEIR_DATASETS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 5: BEIR Evaluation")
    print(f"  Max length: {args.max_len}")
    print(f"  Datasets: {len(eval_datasets)} tasks")
    print(f"  Device: {device}")
    print(f"  Hardware: {capture_hardware_info()}")
    print("=" * 70)

    # Resolve checkpoints
    checkpoint_map = {}
    if args.checkpoints:
        for pair in args.checkpoints.split(","):
            key, path = pair.split(":")
            checkpoint_map[key] = Path(path)
    elif args.exp1_dir:
        exp1_base = Path(args.exp1_dir)
        for model_key in MODEL_BUILDERS:
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
        result = eval_model_beir(
            model_key, ckpt_path, args.max_len, tokenizer, device,
            args.batch_size, eval_datasets,
        )
        all_results[model_key] = result

    # Save results
    out_path = RESULTS_DIR / "beir_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Generate plots
    if all_results:
        plot_beir_results(all_results, eval_datasets)

    print("\n=== Experiment 5 complete ===")


if __name__ == "__main__":
    main()
