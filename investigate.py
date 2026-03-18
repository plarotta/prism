"""
Post-mortem investigation:
1. Check if interference alpha values actually activated during training
2. Rerun with alpha initialized at 0.1 (not zero) to test if ReZero killed it
3. Long-sequence quality showdown: PRISM-MeanPool vs Transformer at 2000 steps, seq_len 512
"""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import prism_small, PRISMForEmbedding, CrossScaleInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from benchmark_quality import (
    SyntheticEmbeddingDataset, train_model, evaluate_retrieval,
    DEVICE, VOCAB_SIZE,
)
from benchmark_ablations import MeanPooling

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------
# Investigation 1: What are the alpha values after training?
# -----------------------------------------------------------------------

def investigate_alpha():
    print("=" * 70)
    print("INVESTIGATION 1: Are interference alphas actually activating?")
    print("=" * 70)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)
    model = prism_small(vocab_size=VOCAB_SIZE).to(DEVICE)
    wrapper = PRISMForEmbedding(model, temperature=0.05).to(DEVICE)

    # Print alpha before training
    print("\nAlpha values BEFORE training:")
    for i, layer in enumerate(model.layers):
        alpha_fwd = layer.interference_fwd.alpha.data
        print(f"  Layer {i} fwd: mean(|α|)={alpha_fwd.abs().mean():.6f}  max(|α|)={alpha_fwd.abs().max():.6f}")

    # Train 400 steps (same as the ablation run)
    losses, accs, _ = train_model(wrapper, dataset, 400, "PRISM-alpha-check",
                                  seq_len_q=64, seq_len_p=96)

    # Print alpha after training
    print("\nAlpha values AFTER 400 steps of training:")
    alpha_stats = []
    for i, layer in enumerate(model.layers):
        alpha_fwd = layer.interference_fwd.alpha.data
        alpha_bwd = layer.interference_bwd.alpha.data
        mean_abs = (alpha_fwd.abs().mean().item() + alpha_bwd.abs().mean().item()) / 2
        max_abs = max(alpha_fwd.abs().max().item(), alpha_bwd.abs().max().item())
        alpha_stats.append({"layer": i, "mean_abs": mean_abs, "max_abs": max_abs})
        print(f"  Layer {i}: mean(|α|)={mean_abs:.6f}  max(|α|)={max_abs:.6f}")
        # Print full matrix for first and last layer
        if i == 0 or i == len(model.layers) - 1:
            print(f"    fwd α:\n{alpha_fwd.cpu().numpy().round(4)}")

    overall_mean = np.mean([s["mean_abs"] for s in alpha_stats])
    print(f"\n  Overall mean(|α|) across all layers: {overall_mean:.6f}")
    if overall_mean < 0.01:
        print("  >>> VERDICT: Alpha barely moved from zero. ReZero init kept interference dormant.")
        print("  >>> The ablation removed a component that was never active.")
    else:
        print("  >>> Alpha values are non-trivial. Interference was active but not helpful.")

    return alpha_stats


# -----------------------------------------------------------------------
# Investigation 2: Does interference help with non-zero alpha init?
# -----------------------------------------------------------------------

def make_prism_warm_alpha(alpha_init=0.1):
    """PRISM with alpha initialized at a non-zero value instead of ReZero."""
    model = prism_small(vocab_size=VOCAB_SIZE)
    for layer in model.layers:
        nn.init.constant_(layer.interference_fwd.alpha, alpha_init)
        layer.interference_fwd.alpha.data.fill_diagonal_(0.0)
        nn.init.constant_(layer.interference_bwd.alpha, alpha_init)
        layer.interference_bwd.alpha.data.fill_diagonal_(0.0)
    return model


def investigate_warm_alpha():
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: Does warm-starting alpha change the picture?")
    print("=" * 70)

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)
    queries, corpus, qt, ct = dataset.generate_corpus_and_queries(500, 5000, 64, 96)

    results = {}
    all_losses = {}

    for name, builder in [
        ("Full PRISM (α=0 ReZero)", lambda: prism_small(vocab_size=VOCAB_SIZE)),
        ("Full PRISM (α=0.1 warm)", lambda: make_prism_warm_alpha(0.1)),
        ("Full PRISM (α=0.3 warm)", lambda: make_prism_warm_alpha(0.3)),
        ("No Interference", lambda: _make_no_interference()),
    ]:
        torch.manual_seed(42)
        random.seed(42)

        model = builder().to(DEVICE)
        wrapper = PRISMForEmbedding(model, temperature=0.05).to(DEVICE)
        losses, accs, t = train_model(wrapper, dataset, 400, name,
                                      seq_len_q=64, seq_len_p=96)
        metrics = evaluate_retrieval(wrapper, queries, corpus, qt, ct)

        results[name] = {"mrr": metrics["mrr"], "recall@1": metrics["recall@1"],
                         "final_loss": float(np.mean(losses[-50:]))}
        all_losses[name] = losses
        print(f"  {name}: MRR={metrics['mrr']:.4f}  R@1={metrics['recall@1']:.4f}")

    print("\n  Summary:")
    for name, r in results.items():
        print(f"    {name:<30} MRR={r['mrr']:.4f}  loss={r['final_loss']:.4f}")

    return results


def _make_no_interference():
    from benchmark_ablations import NoInterference
    model = prism_small(vocab_size=VOCAB_SIZE)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    return model


# -----------------------------------------------------------------------
# Investigation 3: Long-sequence showdown — PRISM-MeanPool vs Transformer
# -----------------------------------------------------------------------

def make_prism_meanpool():
    model = prism_small(vocab_size=VOCAB_SIZE)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


def investigate_long_sequence():
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: Long-sequence quality — PRISM-MeanPool vs Transformer")
    print("           2000 steps, seq_len_q=256, seq_len_p=512")
    print("=" * 70)

    N_STEPS = 2000
    SEQ_Q = 256
    SEQ_P = 512
    N_EVAL = 500
    CORPUS = 5000

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)
    queries, corpus, qt, ct = dataset.generate_corpus_and_queries(N_EVAL, CORPUS, SEQ_Q, SEQ_P)

    results = {}
    all_losses = {}

    for name, build_wrapper in [
        ("PRISM-MeanPool", lambda: (
            PRISMForEmbedding(make_prism_meanpool().to(DEVICE), temperature=0.05)
        )),
        ("Transformer", lambda: (
            TransformerForEmbedding(transformer_small(vocab_size=VOCAB_SIZE).to(DEVICE), temperature=0.05)
        )),
    ]:
        torch.manual_seed(42)
        random.seed(42)

        wrapper = build_wrapper().to(DEVICE)
        n_params = sum(p.numel() for p in wrapper.parameters())
        print(f"\n  {name}: {n_params:,} params")

        losses, accs, train_time = train_model(wrapper, dataset, N_STEPS, name,
                                               seq_len_q=SEQ_Q, seq_len_p=SEQ_P)
        metrics = evaluate_retrieval(wrapper, queries, corpus, qt, ct)

        results[name] = {
            "metrics": metrics,
            "train_time": train_time,
            "params": n_params,
            "final_loss": float(np.mean(losses[-100:])),
        }
        all_losses[name] = losses

        print(f"  {name}: MRR={metrics['mrr']:.4f}  R@1={metrics['recall@1']:.4f}  "
              f"R@5={metrics['recall@5']:.4f}  time={train_time:.1f}s")

    print(f"\n  Quality gap: {results['PRISM-MeanPool']['metrics']['mrr'] - results['Transformer']['metrics']['mrr']:+.4f} MRR")
    print(f"  Training time: PRISM={results['PRISM-MeanPool']['train_time']:.0f}s  "
          f"Transformer={results['Transformer']['train_time']:.0f}s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Long-Sequence Showdown (seq_q={SEQ_Q}, seq_p={SEQ_P}, {N_STEPS} steps)",
                 fontsize=14, fontweight="bold")

    window = 50
    colors = {"PRISM-MeanPool": "#2563EB", "Transformer": "#DC2626"}

    ax = axes[0]
    for name, losses in all_losses.items():
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smooth, color=colors[name], linewidth=2, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    metrics_names = ["mrr", "recall@1", "recall@5", "recall@10"]
    x = np.arange(len(metrics_names))
    width = 0.35
    for i, (name, r) in enumerate(results.items()):
        vals = [r["metrics"][k] for k in metrics_names]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name,
                      color=colors[name], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "long_sequence_showdown.png", bbox_inches="tight")
    print(f"\n  Saved: {RESULTS_DIR / 'long_sequence_showdown.png'}")
    plt.close("all")

    with open(RESULTS_DIR / "investigation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# -----------------------------------------------------------------------
# Patched train_model that accepts variable seq lens
# -----------------------------------------------------------------------

# We need to override train_model to accept seq_len params.
# Monkey-patch it locally.
_original_train_model = train_model

def train_model(model_wrapper, dataset, n_steps, model_name,
                seq_len_q=64, seq_len_p=96):
    """Train with configurable sequence lengths."""
    from benchmark_quality import LR, BATCH_SIZE
    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    losses = []
    accuracies = []
    model_wrapper.train()

    print(f"\nTraining {model_name}...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        q, p, _ = dataset.generate_batch(BATCH_SIZE, seq_len_q, seq_len_p)
        q, p = q.to(DEVICE), p.to(DEVICE)
        q_mask = torch.ones_like(q)
        p_mask = torch.ones_like(p)

        result = model_wrapper(q, q_mask, p, p_mask)
        loss = result["loss"]
        acc = result["accuracy"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if (step + 1) % 100 == 0:
            recent_loss = np.mean(losses[-100:])
            recent_acc = np.mean(accuracies[-100:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  acc={recent_acc:.3f}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")
    return losses, accuracies, total_time


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    alpha_stats = investigate_alpha()
    warm_results = investigate_warm_alpha()
    long_results = investigate_long_sequence()

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
    print("\nKey questions answered:")
    overall_alpha = np.mean([s["mean_abs"] for s in alpha_stats])
    print(f"  1. Did alpha activate? mean(|α|) = {overall_alpha:.6f}  "
          f"({'NO — dormant' if overall_alpha < 0.01 else 'YES — active'})")

    if warm_results:
        warm_mrr = warm_results.get("Full PRISM (α=0.1 warm)", {}).get("mrr", 0)
        cold_mrr = warm_results.get("Full PRISM (α=0 ReZero)", {}).get("mrr", 0)
        no_int_mrr = warm_results.get("No Interference", {}).get("mrr", 0)
        print(f"  2. Warm α helps? ReZero={cold_mrr:.4f}  α=0.1={warm_mrr:.4f}  "
              f"NoInterference={no_int_mrr:.4f}")

    if long_results:
        prism_mrr = long_results["PRISM-MeanPool"]["metrics"]["mrr"]
        trans_mrr = long_results["Transformer"]["metrics"]["mrr"]
        print(f"  3. Long-seq quality gap: PRISM-MeanPool={prism_mrr:.4f}  "
              f"Transformer={trans_mrr:.4f}  delta={prism_mrr - trans_mrr:+.4f}")
