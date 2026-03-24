"""
Experiment 3: Component ablation study.

Isolates which PRISM components matter by training 9 variants with
the Experiment 1c protocol (2048 tokens, MS MARCO, 50K steps).

Variants:
  A: Single channel (C=1, d_c=384)
  B: Geometric decay (lambda_c log-spaced)
  C: All-fast decay (lambda=0.5)
  D: Learned decay (lambda_c as parameters)
  E: No gating (raw linear recurrence)
  F: Unidirectional (forward only)
  G: + Interference (original cross-scale bilinear)
  H: + Covariance pooling (attentive covariance)
  I: Attentive pooling (learned query attention)

Usage:
    uv run python paper_exp3_ablations.py --variants A,B,C
    uv run python paper_exp3_ablations.py --all
    uv run python paper_exp3_ablations.py --smoke-test
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from paper_log import create_run_dir
from train_contrastive import train
from data.msmarco import MSMARCODataset
from data.loco_eval import evaluate_locov1

from prism import (
    prism_small, PRISMForEmbedding, PRISMEncoder,
    StratifiedRecurrence, _fast_fixed_decay_scan,
    CrossScaleInterference, AttentiveCovariancePooling, AttentivePooling,
)
from benchmark_ablations import MeanPooling, NoInterference, LearnedDecayRecurrence

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
MAX_LEN = 2048
RESULTS_DIR = Path("results") / "paper" / "exp3"


# ---------------------------------------------------------------------------
# Baseline: PRISM-Simplified (same as Exp 1)
# ---------------------------------------------------------------------------

def build_baseline(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """PRISM-Simplified: all-slow decay, no interference, mean pooling."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant A: Single channel (C=1, d_c=384)
# ---------------------------------------------------------------------------

def build_variant_a(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """Single channel: tests whether multi-channel structure is necessary."""
    encoder = prism_small(
        vocab_size=VOCAB_SIZE, max_len=max_len,
        n_channels=1,  # single channel of d_c=384
    )
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant B: Geometric decay (log-spaced lambdas)
# ---------------------------------------------------------------------------

def build_variant_b(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """Geometric decay: lambda_c log-spaced [0.0 ... 0.9998]."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    # Restore geometric decay (the original default from StratifiedRecurrence)
    for layer in encoder.layers:
        rec = layer.recurrence
        delta = math.log2(max_len) / max(rec.n_channels - 1, 1)
        for c in range(rec.n_channels):
            rec.lambdas[c] = 1.0 - 2.0 ** (-(c * delta))
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant C: All-fast decay (lambda=0.5)
# ---------------------------------------------------------------------------

def build_variant_c(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """All-fast decay: tests whether slow decay is necessary."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.5)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant D: Learned decay
# ---------------------------------------------------------------------------

def build_variant_d(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """Learned decay: lambda_c as learnable parameters."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    # Replace recurrence with learned-decay variant
    for layer in encoder.layers:
        old_rec = layer.recurrence
        new_rec = LearnedDecayRecurrence(
            old_rec.d_c, old_rec.n_channels, max_len=max_len,
            bidirectional=old_rec.bidirectional,
        )
        # Initialize at all-slow (0.99)
        target_logit = torch.log(torch.tensor(0.99 / 0.01))
        new_rec.lambda_logits.data.fill_(target_logit.item())
        new_rec.gates_fwd.load_state_dict(old_rec.gates_fwd.state_dict())
        if old_rec.bidirectional:
            new_rec.gates_bwd.load_state_dict(old_rec.gates_bwd.state_dict())
        layer.recurrence = new_rec
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant E: No gating (raw linear recurrence)
# ---------------------------------------------------------------------------

class UngatedRecurrence(StratifiedRecurrence):
    """Recurrence without input gating — raw linear scan."""

    def _run_direction(self, channels, gates):
        hiddens = []
        for c, z_c in enumerate(channels):
            lam = self.lambdas[c].item()
            h_c = _fast_fixed_decay_scan(lam, z_c)
            hiddens.append(h_c)
        return hiddens


def build_variant_e(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """No gating: tests whether input gating is necessary."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        old_rec = layer.recurrence
        new_rec = UngatedRecurrence(
            old_rec.d_c, old_rec.n_channels, max_len=max_len,
            bidirectional=old_rec.bidirectional,
        )
        new_rec.lambdas.fill_(0.99)
        layer.recurrence = new_rec
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant F: Unidirectional (forward only)
# ---------------------------------------------------------------------------

def build_variant_f(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """Unidirectional: forward only, no backward pass."""
    encoder = prism_small(
        vocab_size=VOCAB_SIZE, max_len=max_len,
        bidirectional=False,
    )
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant G: + Interference (original cross-scale bilinear)
# ---------------------------------------------------------------------------

def build_variant_g(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """+ Interference: restore original cross-scale bilinear interference."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    # Keep the default CrossScaleInterference (don't replace with NoInterference)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant H: + Covariance pooling
# ---------------------------------------------------------------------------

def build_variant_h(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """+ Covariance pooling: replace mean pool with attentive covariance."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    # Keep the default AttentiveCovariancePooling (don't replace with MeanPooling)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant I: Attentive pooling
# ---------------------------------------------------------------------------

def build_variant_i(max_len: int = MAX_LEN) -> PRISMForEmbedding:
    """Attentive pooling: learned query attention instead of mean pooling."""
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = AttentivePooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "baseline": ("PRISM-Simplified", build_baseline),
    "A": ("A: Single Channel", build_variant_a),
    "B": ("B: Geometric Decay", build_variant_b),
    "C": ("C: All-Fast Decay", build_variant_c),
    "D": ("D: Learned Decay", build_variant_d),
    "E": ("E: No Gating", build_variant_e),
    "F": ("F: Unidirectional", build_variant_f),
    "G": ("G: + Interference", build_variant_g),
    "H": ("H: + Cov Pooling", build_variant_h),
    "I": ("I: Attentive Pool", build_variant_i),
}


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer, device):
    def eval_fn(model_wrapper, step):
        loco = evaluate_locov1(
            model_wrapper, tokenizer, max_len=MAX_LEN,
            batch_size=32, device=device,
        )
        return {
            "locov1_avg_ndcg@10": loco["avg_ndcg@10"],
            "locov1_per_task": loco["per_task"],
            "locov1_eval_time_s": loco["eval_time_s"],
        }
    return eval_fn


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

def run_variant(
    variant_key: str,
    n_steps: int = 50000,
    micro_batch: int = 16,
    grad_accum: int = 8,
    lr: float = 3e-4,
    eval_every: int = 5000,
    checkpoint_every: int = 10000,
    device: str | None = None,
    seed: int = 42,
):
    variant_name, build_fn = VARIANTS[variant_key]

    print(f"\n{'='*70}")
    print(f"Experiment 3 Ablation: {variant_name}")
    print(f"{'='*70}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_fn(MAX_LEN)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = MSMARCODataset(tokenizer, max_len=MAX_LEN)
    dataset.load()

    eval_fn = make_eval_fn(tokenizer, device)
    run_dir = create_run_dir("exp3_ablation", variant_key)

    config = {
        "experiment": "exp3_ablation",
        "variant_key": variant_key,
        "variant_name": variant_name,
        "model_config": {
            "max_len": MAX_LEN,
            "vocab_size": VOCAB_SIZE,
            "total_params": total_params,
        },
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

    print(f"\n  {variant_name} complete: best nDCG@10={result['best_metric']:.4f} "
          f"@ step {result['best_step']}")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "baseline": "#2563EB",
    "A": "#DC2626",
    "B": "#16A34A",
    "C": "#9333EA",
    "D": "#F59E0B",
    "E": "#EC4899",
    "F": "#06B6D4",
    "G": "#EF4444",
    "H": "#8B5CF6",
    "I": "#14B8A6",
}


def plot_ablation_results(results: dict):
    """Generate bar chart of ablation results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variant_keys = list(results.keys())
    names = [results[k]["variant_name"] for k in variant_keys]
    metrics = [results[k].get("best_metric", 0) for k in variant_keys]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    baseline_val = results.get("baseline", {}).get("best_metric", 0)
    colors = [COLORS.get(k, "#888888") for k in variant_keys]

    bars = ax.barh(names, metrics, color=colors, alpha=0.85)
    if baseline_val > 0:
        ax.axvline(x=baseline_val, color="#2563EB", linewidth=2,
                   linestyle="--", alpha=0.7, label="Baseline")

    for bar, val in zip(bars, metrics):
        delta = val - baseline_val if baseline_val > 0 else 0
        label = f"{val:.4f} ({delta:+.4f})" if baseline_val > 0 else f"{val:.4f}"
        ax.annotate(label,
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)

    ax.set_xlabel("LoCoV1 nDCG@10")
    ax.set_title("Experiment 3: Component Ablation Study", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out_path = RESULTS_DIR / "ablation_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Ablation study")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant keys (A,B,C,...)")
    parser.add_argument("--all", action="store_true",
                        help="Run all variants including baseline")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick pipeline validation (100 steps)")
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
        print("Running baseline variant for 100 steps...")
        run_variant(
            "baseline", n_steps=100, micro_batch=4, grad_accum=1,
            eval_every=50, checkpoint_every=100,
            device=args.device, seed=args.seed,
        )
        print("\n=== SMOKE TEST PASSED ===")
        return

    if args.all:
        variant_keys = list(VARIANTS.keys())
    elif args.variants:
        variant_keys = args.variants.split(",")
    else:
        parser.error("Specify --variants, --all, or --smoke-test")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for key in variant_keys:
        if key not in VARIANTS:
            print(f"Unknown variant: {key}. Choose from: {list(VARIANTS.keys())}")
            continue
        result = run_variant(
            key, n_steps=args.n_steps,
            micro_batch=args.micro_batch, grad_accum=args.grad_accum,
            lr=args.lr, eval_every=args.eval_every,
            device=args.device, seed=args.seed,
        )
        all_results[key] = {
            "variant_name": VARIANTS[key][0],
            **result,
        }

    # Save combined results
    out_path = RESULTS_DIR / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    # Generate plot
    if len(all_results) >= 2:
        plot_ablation_results(all_results)

    print("\n=== Experiment 3 complete ===")


if __name__ == "__main__":
    main()
