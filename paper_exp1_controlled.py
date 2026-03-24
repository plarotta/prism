"""
Experiment 1: Controlled architecture comparison.

Trains all 4 models on MS MARCO with identical protocol,
evaluates zero-shot on LoCoV1 at multiple sequence lengths.

Usage:
    # Single sub-experiment, single model
    uv run python paper_exp1_controlled.py --sub-exp 1a --models prism

    # Full sub-experiment
    uv run python paper_exp1_controlled.py --sub-exp 1c

    # All sub-experiments
    uv run python paper_exp1_controlled.py --all

    # Smoke test (tiny run to validate pipeline)
    uv run python paper_exp1_controlled.py --smoke-test
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from paper_log import create_run_dir
from train_contrastive import train
from data.msmarco import MSMARCODataset
from data.loco_eval import evaluate_locov1

# Model builders
from prism import prism_small, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small
from linear_rnn import build_linear_rnn_small

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_prism(max_len: int) -> PRISMForEmbedding:
    """PRISM-Simplified: all-slow decay, no interference, mean pooling."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    # Replace interference with NoInterference
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    # Replace pooling with MeanPooling
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    # Set all decay rates to 0.99 (all-slow)
    for layer in encoder.layers:
        rec = layer.recurrence
        rec.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


def build_transformer(max_len: int) -> TransformerForEmbedding:
    """Parameter-matched Transformer baseline."""
    encoder = transformer_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    return TransformerForEmbedding(encoder)


def build_mamba(max_len: int):
    """Bidirectional Mamba baseline."""
    return build_mamba_bidir_small(vocab_size=VOCAB_SIZE, max_len=max_len)


def build_linear_rnn(max_len: int):
    """Single-channel linear RNN ablation baseline."""
    return build_linear_rnn_small(vocab_size=VOCAB_SIZE, max_len=max_len)


MODEL_BUILDERS = {
    "prism": ("PRISM-Simplified", build_prism),
    "transformer": ("Transformer", build_transformer),
    "mamba": ("Mamba-Bidir", build_mamba),
    "linear_rnn": ("Linear-RNN", build_linear_rnn),
}

# ---------------------------------------------------------------------------
# Sub-experiment configs
# ---------------------------------------------------------------------------

SUB_EXPERIMENTS = {
    "1a": {
        "desc": "Short-sequence comparison (128 tokens)",
        "train_max_len": 128,
        "eval_max_len": 128,
        "eval_locov1": False,
        "models": list(MODEL_BUILDERS.keys()),
    },
    "1b": {
        "desc": "Medium-sequence comparison (512 tokens)",
        "train_max_len": 512,
        "eval_max_len": 512,
        "eval_locov1": False,
        "models": list(MODEL_BUILDERS.keys()),
    },
    "1c": {
        "desc": "Long-sequence comparison (2048 tokens)",
        "train_max_len": 2048,
        "eval_max_len": 2048,
        "eval_locov1": True,
        "models": list(MODEL_BUILDERS.keys()),
    },
    "1d": {
        "desc": "LoCoV1 zero-shot (train@2048, eval@2048)",
        "train_max_len": 2048,
        "eval_max_len": 2048,
        "eval_locov1": True,
        "models": list(MODEL_BUILDERS.keys()),
    },
    "1e": {
        "desc": "LoCoV1 long-context (train@2048, eval@8192)",
        "train_max_len": 2048,
        "eval_max_len": 8192,
        "eval_locov1": True,
        # Transformer excluded — OOMs at 8K
        "models": ["prism", "mamba", "linear_rnn"],
    },
}


# ---------------------------------------------------------------------------
# Eval callback factory
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer, eval_max_len, device, do_locov1=True):
    """Create an eval callback for use during training."""

    def eval_fn(model_wrapper, step):
        results = {}
        if do_locov1:
            loco = evaluate_locov1(
                model_wrapper, tokenizer, max_len=eval_max_len,
                batch_size=32, device=device,
            )
            results["locov1_avg_ndcg@10"] = loco["avg_ndcg@10"]
            results["locov1_per_task"] = loco["per_task"]
            results["locov1_eval_time_s"] = loco["eval_time_s"]
        return results

    return eval_fn


# ---------------------------------------------------------------------------
# Run one model in one sub-experiment
# ---------------------------------------------------------------------------

def run_one(
    sub_exp_id: str,
    model_key: str,
    n_steps: int = 50000,
    micro_batch: int = 16,
    grad_accum: int = 8,
    lr: float = 3e-4,
    eval_every: int = 5000,
    checkpoint_every: int = 10000,
    device: str | None = None,
    seed: int = 42,
):
    """Train one model for one sub-experiment."""
    sub_exp = SUB_EXPERIMENTS[sub_exp_id]
    model_name, build_fn = MODEL_BUILDERS[model_key]

    print(f"\n{'='*70}")
    print(f"Experiment 1{sub_exp_id[1:]}: {sub_exp['desc']}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    # Use eval_max_len for model's max_len so it can handle eval sequences
    model_max_len = max(sub_exp["train_max_len"], sub_exp["eval_max_len"])
    model = build_fn(model_max_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = MSMARCODataset(tokenizer, max_len=sub_exp["train_max_len"])
    dataset.load()

    # Eval callback
    eval_fn = make_eval_fn(
        tokenizer, sub_exp["eval_max_len"], device,
        do_locov1=sub_exp.get("eval_locov1", False),
    )

    # Run dir
    run_dir = create_run_dir(f"exp1_{sub_exp_id}", model_key)

    # Config
    config = {
        "experiment": f"exp1_{sub_exp_id}",
        "sub_experiment": sub_exp_id,
        "model_key": model_key,
        "model_name": model_name,
        "model_config": {
            "max_len": model_max_len,
            "vocab_size": VOCAB_SIZE,
            "total_params": total_params,
        },
        "train_max_len": sub_exp["train_max_len"],
        "eval_max_len": sub_exp["eval_max_len"],
    }

    # Train
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
    parser = argparse.ArgumentParser(description="Experiment 1: Controlled architecture comparison")
    parser.add_argument("--sub-exp", choices=list(SUB_EXPERIMENTS.keys()),
                        help="Which sub-experiment to run")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (prism,transformer,mamba,linear_rnn)")
    parser.add_argument("--all", action="store_true",
                        help="Run all sub-experiments")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick pipeline validation (100 steps, small batch)")
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        print("Running PRISM at 128 tokens for 100 steps...")
        run_one(
            sub_exp_id="1a", model_key="prism",
            n_steps=100, micro_batch=4, grad_accum=1,
            eval_every=50, checkpoint_every=100,
            device=args.device, seed=args.seed,
        )
        print("\n=== SMOKE TEST PASSED ===")
        return

    if args.all:
        sub_exps = list(SUB_EXPERIMENTS.keys())
    elif args.sub_exp:
        sub_exps = [args.sub_exp]
    else:
        parser.error("Specify --sub-exp, --all, or --smoke-test")
        return

    for sub_exp_id in sub_exps:
        sub_exp = SUB_EXPERIMENTS[sub_exp_id]
        models = (
            args.models.split(",") if args.models
            else sub_exp["models"]
        )

        for model_key in models:
            if model_key not in MODEL_BUILDERS:
                print(f"Unknown model: {model_key}. "
                      f"Choose from: {list(MODEL_BUILDERS.keys())}")
                continue
            run_one(
                sub_exp_id=sub_exp_id,
                model_key=model_key,
                n_steps=args.n_steps,
                micro_batch=args.micro_batch,
                grad_accum=args.grad_accum,
                lr=args.lr,
                eval_every=args.eval_every,
                device=args.device,
                seed=args.seed,
            )

    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
