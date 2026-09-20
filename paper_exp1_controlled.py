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
import random
import numpy as np
import json
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

from paper_log import create_run_dir, finish_wandb
from train_contrastive import train
from experiment_protocol import evaluate_selected_locov1
from data.msmarco import MSMARCODataset, evaluate_msmarco_dev
from data.loco_eval import evaluate_locov1

# Model builders
from prism import prism_small, PRISMForEmbedding
from paper_components import MeanPooling, NoInterference, LearnedDecayRecurrence
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small, MAMBA_AVAILABLE
from linear_rnn import build_linear_rnn_small

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_prism(max_len: int, position_encoding="learned", learned_decay=False) -> PRISMForEmbedding:
    """PRISM-Simplified: all-slow decay, no interference, mean pooling."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len,
                          position_encoding=position_encoding)
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
        if learned_decay:
            new_rec = LearnedDecayRecurrence(rec.d_c, rec.n_channels, max_len, rec.bidirectional)
            with torch.no_grad():
                new_rec.lambda_logits.fill_(torch.logit(torch.tensor(0.99)).item())
            new_rec.gates_fwd.load_state_dict(rec.gates_fwd.state_dict())
            if rec.bidirectional:
                new_rec.gates_bwd.load_state_dict(rec.gates_bwd.state_dict())
            layer.recurrence = new_rec
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
    "prism_no_pos": ("PRISM-NoPosition", lambda ml: build_prism(ml, "none")),
    "prism_learned": ("PRISM-LearnedDecay", lambda ml: build_prism(ml, learned_decay=True)),
    "transformer_sinusoidal": ("Transformer-Sinusoidal", lambda ml: TransformerForEmbedding(
        transformer_small(vocab_size=VOCAB_SIZE, max_len=ml, position_encoding="sinusoidal"))),
}
CORE_MODELS = ["prism", "transformer", "mamba", "linear_rnn"]

# ---------------------------------------------------------------------------
# Sub-experiment configs
# ---------------------------------------------------------------------------

SUB_EXPERIMENTS = {
    "1a": {
        "desc": "Short-sequence comparison (128 tokens)",
        "train_max_len": 128,
        "eval_max_len": 128,
        "eval_locov1": False,
        "models": CORE_MODELS,
    },
    "1b": {
        "desc": "Medium-sequence comparison (512 tokens)",
        "train_max_len": 512,
        "eval_max_len": 512,
        "eval_locov1": False,
        "models": CORE_MODELS,
    },
    "1c": {
        "desc": "Long-sequence comparison (2048 tokens)",
        "train_max_len": 2048,
        "eval_max_len": 2048,
        "eval_locov1": True,
        "models": CORE_MODELS,
    },
    "1d": {
        "desc": "LoCoV1 zero-shot (train@2048, eval@2048)",
        "train_max_len": 2048,
        "eval_max_len": 2048,
        "eval_locov1": True,
        "models": CORE_MODELS,
    },
    "1e": {
        "desc": "LoCoV1 long-context (train@2048, eval@8192)",
        "train_max_len": 2048,
        "eval_max_len": 8192,
        "eval_locov1": True,
        # Re-measure the optimized attention baseline; old OOMs are not a gate.
        "models": CORE_MODELS,
    },
}


# ---------------------------------------------------------------------------
# Eval callback factory
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer, dataset, eval_max_len, device, do_locov1=False, batch_size=32):
    """Select checkpoints using source validation only.

    The legacy do_locov1 argument is accepted for callers but cannot enable
    target-benchmark evaluation during training. Transfer tests run once after
    checkpoint selection in run_one().
    """

    def eval_fn(model_wrapper, step):
        results = {}

        dev = evaluate_msmarco_dev(
            model_wrapper, dataset, max_len=eval_max_len,
            batch_size=batch_size, device=device,
        )
        results.update(dev)

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
    allow_mamba_fallback: bool = False,
    eval_early_stop_patience: int = 2,
    eval_early_stop_min_improvement: float = 0.01,
    batch_mode: str = "cached",
    hard_negatives: int = 0,
):
    """Train one model for one sub-experiment."""
    sub_exp = SUB_EXPERIMENTS[sub_exp_id]
    model_name, build_fn = MODEL_BUILDERS[model_key]

    # Guard: never silently train the SimpleDiagSSM fallback as the Mamba
    # baseline — it is not a faithful Mamba and would invalidate the comparison.
    if model_key == "mamba" and not MAMBA_AVAILABLE and not allow_mamba_fallback:
        raise RuntimeError(
            "mamba_ssm is not installed, so the Mamba baseline would fall back "
            "to SimpleDiagSSM (NOT a faithful Mamba). Install it on the GPU box "
            "(`uv add mamba-ssm causal-conv1d`), or pass --allow-mamba-fallback "
            "to deliberately run the approximate fallback."
        )

    print(f"\n{'='*70}")
    print(f"Experiment 1{sub_exp_id[1:]}: {sub_exp['desc']}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed before construction, not only after model weights are initialized.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build model
    # Use eval_max_len for model's max_len so it can handle eval sequences
    model_max_len = max(sub_exp["train_max_len"], sub_exp["eval_max_len"])
    model = build_fn(model_max_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = MSMARCODataset(tokenizer, max_len=sub_exp["train_max_len"],
                            n_hard_negatives=hard_negatives)
    dataset.load()

    # Eval callback
    eval_fn = make_eval_fn(
        tokenizer, dataset, sub_exp["train_max_len"], device, batch_size=micro_batch,
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
        "hard_negatives": hard_negatives,
        "protocol_version": 2,
        "transfer_selection": "source_validation_only",
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
        eval_early_stop_patience=eval_early_stop_patience,
        eval_early_stop_min_improvement=eval_early_stop_min_improvement,
        grad_cache=batch_mode == "cached",
    )

    if sub_exp.get("eval_locov1", False):
        evaluate_selected_locov1(model, tokenizer, run_dir, result,
                                 max_len=sub_exp["eval_max_len"],
                                 device=device, batch_size=micro_batch)

    print(f"\n  {model_name} complete: best source-dev MRR@10={result['best_metric']:.4f} "
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
    parser.add_argument("--batch-mode", choices=["cached", "accumulate"], default="cached")
    parser.add_argument("--hard-negatives", type=int, default=0,
                        help="BM25 negatives per query; requires a Tevatron cache")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-mamba-fallback", action="store_true",
                        help="Run the approximate SimpleDiagSSM if mamba_ssm "
                             "is missing (NOT a faithful Mamba baseline)")
    parser.add_argument("--eval-early-stop-patience", type=int, default=2,
                        help="Stop after N consecutive evals without sufficient "
                             "eval-metric improvement (0 to disable)")
    parser.add_argument("--eval-early-stop-min-improvement", type=float, default=0.01,
                        help="Min relative eval-metric gain per eval to count as "
                             "improvement (default 0.01 = 1%%)")
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

    failures = []
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
                failures.append((sub_exp_id, model_key))
                continue
            # Isolate failures: one model crashing must not silently abort the
            # rest of the sweep, and the traceback must be recorded.
            try:
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
                    batch_mode=args.batch_mode,
                    hard_negatives=args.hard_negatives,
                    allow_mamba_fallback=args.allow_mamba_fallback,
                    eval_early_stop_patience=args.eval_early_stop_patience,
                    eval_early_stop_min_improvement=args.eval_early_stop_min_improvement,
                )
            except Exception:
                failures.append((sub_exp_id, model_key))
                finish_wandb()  # close the crashed run's W&B session, if any
                tb = traceback.format_exc()
                err_dir = Path("results/paper") / f"exp1_{sub_exp_id}"
                err_dir.mkdir(parents=True, exist_ok=True)
                err_path = err_dir / f"{model_key}_ERROR.txt"
                err_path.write_text(tb)
                print(f"\n!!! {model_key} (exp1_{sub_exp_id}) FAILED — "
                      f"traceback saved to {err_path}\n{tb}")
                continue

    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {failures}")
    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
