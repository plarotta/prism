"""
Experiment 7: Scale-up to ~80M parameters.

Tests whether findings from Experiment 1 (at 20M params) hold at M2-BERT
scale (~80M). Trains PRISM-Base and Transformer-Base with the same protocol
as Experiment 1c (2048 tokens, MS MARCO, 50K steps).

Model Configs:
  PRISM-Base:       d=768, 8 layers, 8 channels, d_c=96  (~84M params)
  Transformer-Base: d=768, 8 layers, 12 heads             (~87M params)

Usage:
    uv run python paper_exp7_scaleup.py --models prism
    uv run python paper_exp7_scaleup.py --all
    uv run python paper_exp7_scaleup.py --smoke-test
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from paper_log import create_run_dir, capture_hardware_info
from train_contrastive import train
from data.msmarco import MSMARCODataset
from data.loco_eval import evaluate_locov1

from prism import PRISMEncoder, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import TransformerEncoder, TransformerForEmbedding

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
RESULTS_DIR = Path("results") / "paper" / "exp7"


# ---------------------------------------------------------------------------
# Base-size model builders (~80M params)
# ---------------------------------------------------------------------------

def build_prism_base(max_len: int) -> PRISMForEmbedding:
    """PRISM-Base: d=768, 8 layers, 8 channels (~84M params)."""
    encoder = PRISMEncoder(
        vocab_size=VOCAB_SIZE, d=768, d_e=768,
        n_layers=8, n_channels=8, max_len=max_len,
        mlp_ratio=4.0, dropout=0.1,
    )
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


def build_transformer_base(max_len: int) -> TransformerForEmbedding:
    """Transformer-Base: d=768, 8 layers, 12 heads (~87M params)."""
    encoder = TransformerEncoder(
        vocab_size=VOCAB_SIZE, d=768, d_e=768,
        n_layers=8, n_heads=12, max_len=max_len,
        mlp_ratio=4.0, dropout=0.1,
    )
    return TransformerForEmbedding(encoder)


MODEL_BUILDERS = {
    "prism": ("PRISM-Base", build_prism_base),
    "transformer": ("Transformer-Base", build_transformer_base),
}


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer, eval_max_len, device):
    def eval_fn(model_wrapper, step):
        loco = evaluate_locov1(
            model_wrapper, tokenizer, max_len=eval_max_len,
            batch_size=16, device=device,  # smaller batch for larger models
        )
        return {
            "locov1_avg_ndcg@10": loco["avg_ndcg@10"],
            "locov1_per_task": loco["per_task"],
            "locov1_eval_time_s": loco["eval_time_s"],
        }
    return eval_fn


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def run_one(
    model_key: str,
    train_max_len: int = 2048,
    eval_max_len: int = 2048,
    n_steps: int = 50000,
    micro_batch: int = 8,
    grad_accum: int = 16,
    lr: float = 2e-4,
    eval_every: int = 5000,
    checkpoint_every: int = 10000,
    device: str | None = None,
    seed: int = 42,
):
    model_name, build_fn = MODEL_BUILDERS[model_key]

    print(f"\n{'='*70}")
    print(f"Experiment 7: Scale-Up — {model_name}")
    print(f"{'='*70}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_max_len = max(train_max_len, eval_max_len)
    model = build_fn(model_max_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = MSMARCODataset(tokenizer, max_len=train_max_len)
    dataset.load()

    eval_fn = make_eval_fn(tokenizer, eval_max_len, device)
    run_dir = create_run_dir("exp7_scaleup", model_key)

    config = {
        "experiment": "exp7_scaleup",
        "model_key": model_key,
        "model_name": model_name,
        "model_config": {
            "max_len": model_max_len,
            "vocab_size": VOCAB_SIZE,
            "total_params": total_params,
        },
        "train_max_len": train_max_len,
        "eval_max_len": eval_max_len,
    }

    result = train(
        model_wrapper=model,
        dataset=dataset,
        run_dir=run_dir,
        config=config,
        n_steps=n_steps,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        lr=lr,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        eval_fn=eval_fn,
        device=device,
        seed=seed,
    )

    print(f"\n  {model_name} complete: best nDCG@10={result['best_metric']:.4f} "
          f"@ step {result['best_step']}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 7: Scale-up to 80M params")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (prism,transformer)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-max-len", type=int, default=2048)
    parser.add_argument("--eval-max-len", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 7: Scale-Up to ~80M Parameters")
    print(f"  Hardware: {capture_hardware_info()}")
    print("=" * 70)

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        run_one(
            "prism", train_max_len=128, eval_max_len=128,
            n_steps=50, micro_batch=2, grad_accum=1,
            eval_every=25, checkpoint_every=50,
            device=args.device, seed=args.seed,
        )
        print("\n=== SMOKE TEST PASSED ===")
        return

    if args.all:
        model_keys = list(MODEL_BUILDERS.keys())
    elif args.models:
        model_keys = args.models.split(",")
    else:
        parser.error("Specify --models, --all, or --smoke-test")
        return

    all_results = {}
    for model_key in model_keys:
        if model_key not in MODEL_BUILDERS:
            print(f"Unknown model: {model_key}. Choose from: {list(MODEL_BUILDERS.keys())}")
            continue
        result = run_one(
            model_key,
            train_max_len=args.train_max_len,
            eval_max_len=args.eval_max_len,
            n_steps=args.n_steps,
            micro_batch=args.micro_batch,
            grad_accum=args.grad_accum,
            lr=args.lr,
            eval_every=args.eval_every,
            device=args.device,
            seed=args.seed,
        )
        all_results[model_key] = {
            "model_name": MODEL_BUILDERS[model_key][0],
            **result,
        }

    out_path = RESULTS_DIR / "scaleup_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print("\n=== Experiment 7 complete ===")


if __name__ == "__main__":
    main()
