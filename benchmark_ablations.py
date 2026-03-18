"""
PRISM Ablation Study

Isolates the contribution of each novel architectural component:
  A. Remove cross-scale interference → independent channels + MLP
  B. Replace fixed decay with learned decay rates
  C. Replace bilinear interference with additive mixing
  D. Replace covariance pooling with mean pooling

All variants trained on the same synthetic embedding task and compared.
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import (
    PRISMForEmbedding,
    StratifiedRecurrence, _fast_fixed_decay_scan, prism_small,
)
from benchmark_quality import (
    SyntheticEmbeddingDataset, train_model, evaluate_retrieval,
    DEVICE, VOCAB_SIZE, SEQ_LEN_Q, SEQ_LEN_P,
    N_EVAL_QUERIES, CORPUS_SIZE, N_TRAIN_STEPS,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Ablation A: No Interference (independent channels → concatenate)
# ---------------------------------------------------------------------------

class NoInterference(nn.Module):
    """Pass-through: no cross-scale interaction."""
    def __init__(self, d_c: int, n_channels: int):
        super().__init__()

    def forward(self, hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        return hiddens


def make_ablation_a(vocab_size=VOCAB_SIZE):
    """PRISM with interference replaced by identity."""
    model = prism_small(vocab_size=vocab_size)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    return model


# ---------------------------------------------------------------------------
# Ablation B: Learned Decay Rates (instead of fixed geometric)
# ---------------------------------------------------------------------------

class LearnedDecayRecurrence(StratifiedRecurrence):
    """Same as StratifiedRecurrence but decay rates are learned parameters."""

    def __init__(self, d_c, n_channels, max_len=8192, bidirectional=True):
        super().__init__(d_c, n_channels, max_len, bidirectional)
        # Override: make lambdas a learned parameter instead of buffer
        # Initialize at the same geometric values
        init_lambdas = self.lambdas.clone()
        self.lambdas = None  # remove buffer
        # Store as logit for unconstrained optimization, sigmoid maps to (0,1)
        self.lambda_logits = nn.Parameter(torch.log(init_lambdas / (1.0 - init_lambdas + 1e-8)))

    @property
    def _lambdas(self):
        return torch.sigmoid(self.lambda_logits)

    def _run_direction(self, channels, gates):
        hiddens = []
        lambdas = self._lambdas
        for c, (z_c, gate_c) in enumerate(zip(channels, gates)):
            g_t = torch.sigmoid(gate_c(z_c))
            gated_input = g_t * z_c
            lam = lambdas[c].item()
            h_c = _fast_fixed_decay_scan(lam, gated_input)
            hiddens.append(h_c)
        return hiddens


def make_ablation_b(vocab_size=VOCAB_SIZE):
    """PRISM with learned (not fixed) decay rates."""
    model = prism_small(vocab_size=vocab_size)
    for layer in model.layers:
        old_rec = layer.recurrence
        new_rec = LearnedDecayRecurrence(
            old_rec.d_c, old_rec.n_channels, bidirectional=old_rec.bidirectional
        )
        # Copy gate weights
        new_rec.gates_fwd.load_state_dict(old_rec.gates_fwd.state_dict())
        if old_rec.bidirectional:
            new_rec.gates_bwd.load_state_dict(old_rec.gates_bwd.state_dict())
        layer.recurrence = new_rec
    return model


# ---------------------------------------------------------------------------
# Ablation C: Additive Interference (instead of bilinear/multiplicative)
# ---------------------------------------------------------------------------

class AdditiveInterference(nn.Module):
    """Additive cross-channel mixing: φ(h^c, h^c') = U h^c + V h^c'
    instead of the multiplicative (U h^c) ⊙ (V h^c')."""

    def __init__(self, d_c: int, n_channels: int):
        super().__init__()
        self.d_c = d_c
        self.n_channels = n_channels
        C = n_channels
        self.U = nn.Parameter(torch.zeros(C, d_c, d_c))
        self.V = nn.Parameter(torch.zeros(C, d_c, d_c))
        self.alpha = nn.Parameter(torch.zeros(C, C))
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

    def forward(self, hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        C = self.n_channels
        H = torch.stack(hiddens, dim=0)  # (C, B, T, d_c)
        UH = torch.einsum("cbti, cij -> cbtj", H, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H, self.V)

        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)

        # Additive: interference[c] = Σ_{c'≠c} α[c,c'] * (UH[c] + VH[c'])
        #   = UH[c] * Σ_{c'} α[c,c'] + Σ_{c'} α[c,c'] * VH[c']
        alpha_row_sum = alpha_masked.sum(dim=1)  # (C,)
        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)  # (C, B, T, D)
        interference = UH * alpha_row_sum[:, None, None, None] + weighted_VH

        mixed = H + interference
        return [mixed[c] for c in range(C)]


def make_ablation_c(vocab_size=VOCAB_SIZE):
    """PRISM with additive (not bilinear) interference."""
    model = prism_small(vocab_size=vocab_size)
    for layer in model.layers:
        layer.interference_fwd = AdditiveInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = AdditiveInterference(layer.d_c, layer.n_channels)
    return model


# ---------------------------------------------------------------------------
# Ablation D: Mean Pooling (instead of attentive covariance pooling)
# ---------------------------------------------------------------------------

class MeanPooling(nn.Module):
    """Standard mean pooling over valid positions."""

    def __init__(self, d: int, d_e: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, f, query_state, mask=None):
        if mask is not None:
            f = f * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            pooled = f.sum(dim=1) / n_valid
        else:
            pooled = f.mean(dim=1)
        return self.norm(self.proj(pooled))


def make_ablation_d(vocab_size=VOCAB_SIZE):
    """PRISM with mean pooling (no covariance sketch, no attentive pooling)."""
    model = prism_small(vocab_size=vocab_size)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


# ---------------------------------------------------------------------------
# Run all ablations
# ---------------------------------------------------------------------------

ABLATIONS = {
    "Full PRISM": lambda: prism_small(vocab_size=VOCAB_SIZE),
    "A: No Interference": lambda: make_ablation_a(),
    "B: Learned Decay": lambda: make_ablation_b(),
    "C: Additive (not bilinear)": lambda: make_ablation_c(),
    "D: Mean Pooling": lambda: make_ablation_d(),
}


def run_ablations():
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Training steps: {N_TRAIN_STEPS}")
    print()

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)

    # Generate fixed eval set
    queries, corpus, q_topics, c_topics = dataset.generate_corpus_and_queries(
        N_EVAL_QUERIES, CORPUS_SIZE, SEQ_LEN_Q, SEQ_LEN_P
    )

    all_results = {}
    all_losses = {}
    all_accs = {}

    for name, builder in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"{'='*60}")

        torch.manual_seed(42)  # same init for fair comparison

        model = builder().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        wrapper = PRISMForEmbedding(model, temperature=0.05).to(DEVICE)
        losses, accs, train_time = train_model(wrapper, dataset, N_TRAIN_STEPS, name)
        metrics = evaluate_retrieval(wrapper, queries, corpus, q_topics, c_topics)

        all_results[name] = {
            "metrics": metrics,
            "train_time": train_time,
            "params": n_params,
            "final_loss": float(np.mean(losses[-50:])),
            "final_acc": float(np.mean(accs[-50:])),
        }
        all_losses[name] = losses
        all_accs[name] = accs

        print(f"  MRR: {metrics['mrr']:.4f}  R@1: {metrics['recall@1']:.4f}  R@5: {metrics['recall@5']:.4f}")

    # Save results
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<30} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'Loss':>8}")
    print("-" * 80)
    for name, r in all_results.items():
        m = r["metrics"]
        print(f"{name:<30} {m['mrr']:>8.4f} {m['recall@1']:>8.4f} {m['recall@5']:>8.4f} {m['recall@10']:>8.4f} {r['final_loss']:>8.4f}")

    # --- Plot ---
    COLORS = {
        "Full PRISM": "#2563EB",
        "A: No Interference": "#F59E0B",
        "B: Learned Decay": "#10B981",
        "C: Additive (not bilinear)": "#8B5CF6",
        "D: Mean Pooling": "#EF4444",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study: Contribution of Each PRISM Component", fontsize=14, fontweight="bold")

    # Training loss curves
    ax = axes[0]
    window = 20
    for name, losses in all_losses.items():
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smooth, color=COLORS[name], linewidth=2, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training accuracy curves
    ax = axes[1]
    for name, accs in all_accs.items():
        smooth = np.convolve(accs, np.ones(window)/window, mode='valid')
        ax.plot(smooth, color=COLORS[name], linewidth=2, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("In-Batch Accuracy")
    ax.set_title("Training Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Retrieval metrics bar chart
    ax = axes[2]
    metrics_to_plot = ["mrr", "recall@1", "recall@5"]
    x = np.arange(len(metrics_to_plot))
    n_variants = len(all_results)
    width = 0.8 / n_variants
    for i, (name, r) in enumerate(all_results.items()):
        vals = [r["metrics"][k] for k in metrics_to_plot]
        bars = ax.bar(x + i * width - 0.4 + width/2, vals, width,
                      label=name, color=COLORS[name], alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ablation_comparison.png", bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'ablation_comparison.png'}")
    plt.close("all")

    # --- Delta chart (drop from full PRISM) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    full_mrr = all_results["Full PRISM"]["metrics"]["mrr"]
    names = []
    deltas = []
    colors = []
    for name, r in all_results.items():
        if name == "Full PRISM":
            continue
        names.append(name)
        deltas.append(r["metrics"]["mrr"] - full_mrr)
        colors.append(COLORS[name])

    bars = ax.barh(names, deltas, color=colors, alpha=0.85)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("MRR Change vs Full PRISM")
    ax.set_title("Ablation Impact: MRR Drop When Removing Each Component", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, delta in zip(bars, deltas):
        ax.annotate(f'{delta:+.4f}',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5 if delta >= 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if delta >= 0 else 'right', va='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "ablation_impact.png", bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'ablation_impact.png'}")
    plt.close("all")

    return all_results


if __name__ == "__main__":
    run_ablations()
