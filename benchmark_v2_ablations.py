"""
PRISM V2 Ablation Study: Testing Targeted Fixes for Interference & Covariance Pooling

Based on the analysis in PRISM_v2fixes.txt, we test 7 variants:
  A: Interference — per-channel LayerNorm only
  B: Interference — 1/sqrt(d_c) scaling only
  C: Interference — alpha = 1/(C-1) init only
  D: Interference — learned gate only
  E: Covariance — LayerNorm on e_2 only
  F: Covariance — reduced rank (8) + projection to d
  G: All fixes combined (CrossScaleInterferenceV2 + AttentiveCovariancePoolingV2)

All variants tested at seq_len_q=256, seq_len_p=512, 2000 steps (matching Phase 2 setup).
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import (
    PRISMForEmbedding, CrossScaleInterference, AttentiveCovariancePooling,
    CrossScaleInterferenceV2, AttentiveCovariancePoolingV2, prism_small,
)
from benchmark_quality import (
    SyntheticEmbeddingDataset, evaluate_retrieval,
    DEVICE, VOCAB_SIZE, BATCH_SIZE, LR,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN_Q = 256
SEQ_LEN_P = 512
N_STEPS = 2000
N_EVAL = 500
CORPUS_SIZE = 5000


# ---------------------------------------------------------------------------
# Training (configurable seq lens, from investigate.py pattern)
# ---------------------------------------------------------------------------

def train_model(model_wrapper, dataset, n_steps, model_name,
                seq_len_q=SEQ_LEN_Q, seq_len_p=SEQ_LEN_P):
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

        if (step + 1) % 200 == 0:
            recent_loss = np.mean(losses[-200:])
            recent_acc = np.mean(accuracies[-200:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  acc={recent_acc:.3f}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")
    return losses, accuracies, total_time


# ---------------------------------------------------------------------------
# Variant A: Interference with per-channel LayerNorm only
# ---------------------------------------------------------------------------

class InterferenceNormOnly(CrossScaleInterference):
    def __init__(self, d_c, n_channels):
        super().__init__(d_c, n_channels)
        self.channel_norms = nn.ModuleList([nn.LayerNorm(d_c) for _ in range(n_channels)])

    def forward(self, hiddens):
        C = self.n_channels
        H_orig = torch.stack(hiddens, dim=0)
        H_normed = torch.stack(
            [self.channel_norms[c](hiddens[c]) for c in range(C)], dim=0
        )
        UH = torch.einsum("cbti, cij -> cbtj", H_normed, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H_normed, self.V)
        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)
        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)
        interference = UH * weighted_VH
        mixed = H_orig + interference
        return [mixed[c] for c in range(C)]


# ---------------------------------------------------------------------------
# Variant B: Interference with 1/sqrt(d_c) scaling only
# ---------------------------------------------------------------------------

class InterferenceScaleOnly(CrossScaleInterference):
    def __init__(self, d_c, n_channels):
        super().__init__(d_c, n_channels)
        self.scale = 1.0 / math.sqrt(d_c)

    def forward(self, hiddens):
        C = self.n_channels
        H = torch.stack(hiddens, dim=0)
        UH = torch.einsum("cbti, cij -> cbtj", H, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H, self.V)
        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)
        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)
        interference = UH * weighted_VH * self.scale
        mixed = H + interference
        return [mixed[c] for c in range(C)]


# ---------------------------------------------------------------------------
# Variant C: Interference with alpha = 1/(C-1) init only
# ---------------------------------------------------------------------------

class InterferenceAlphaInit(CrossScaleInterference):
    def __init__(self, d_c, n_channels):
        super().__init__(d_c, n_channels)
        C = n_channels
        nn.init.constant_(self.alpha, 1.0 / max(C - 1, 1))
        self.alpha.data.fill_diagonal_(0.0)


# ---------------------------------------------------------------------------
# Variant D: Interference with learned gate only
# ---------------------------------------------------------------------------

class InterferenceGateOnly(CrossScaleInterference):
    def __init__(self, d_c, n_channels):
        super().__init__(d_c, n_channels)
        self.gamma = nn.Parameter(torch.full((n_channels, 1, 1, 1), -3.0))

    def forward(self, hiddens):
        C = self.n_channels
        H = torch.stack(hiddens, dim=0)
        UH = torch.einsum("cbti, cij -> cbtj", H, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H, self.V)
        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)
        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)
        interference = UH * weighted_VH
        mixed = H + torch.sigmoid(self.gamma) * interference
        return [mixed[c] for c in range(C)]


# ---------------------------------------------------------------------------
# Variant E: Covariance with LayerNorm on e_2 only
# ---------------------------------------------------------------------------

class CovPoolingNormOnly(AttentiveCovariancePooling):
    def __init__(self, d, d_e, cov_rank=24):
        super().__init__(d, d_e, cov_rank)
        self.cov_norm = nn.LayerNorm(cov_rank * cov_rank)

    def forward(self, f, query_state, mask=None):
        B, T, D = f.shape
        query = self.W_q(query_state)
        scores = torch.einsum("bd, btd -> bt", query, f) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        e_1 = torch.einsum("bt, btd -> bd", attn, f)

        Pf = self.P(f)
        Qf = self.Q(f)
        if mask is not None:
            Pf = Pf * mask.unsqueeze(-1).float()
            Qf = Qf * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        else:
            n_valid = torch.tensor(T, dtype=f.dtype, device=f.device)
        cov = torch.einsum("btr, bts -> brs", Pf, Qf)
        if mask is not None:
            cov = cov / n_valid.unsqueeze(-1)
        else:
            cov = cov / T
        e_2 = self.cov_norm(cov.reshape(B, -1))

        combined = torch.cat([e_1, e_2], dim=-1)
        embedding = self.layer_norm(self.out_proj(combined))
        return embedding


# ---------------------------------------------------------------------------
# Variant F: Covariance with reduced rank + projection to d
# ---------------------------------------------------------------------------

class CovPoolingRankProj(nn.Module):
    def __init__(self, d, d_e, cov_rank=8):
        super().__init__()
        self.d = d
        self.cov_rank = cov_rank
        self.W_q = nn.Linear(d, d, bias=False)
        self.P = nn.Linear(d, cov_rank, bias=False)
        self.Q = nn.Linear(d, cov_rank, bias=False)
        self.cov_proj = nn.Linear(cov_rank * cov_rank, d)
        self.out_proj = nn.Linear(2 * d, d_e)
        self.layer_norm = nn.LayerNorm(d_e)

    def forward(self, f, query_state, mask=None):
        B, T, D = f.shape
        query = self.W_q(query_state)
        scores = torch.einsum("bd, btd -> bt", query, f) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        e_1 = torch.einsum("bt, btd -> bd", attn, f)

        Pf = self.P(f)
        Qf = self.Q(f)
        if mask is not None:
            Pf = Pf * mask.unsqueeze(-1).float()
            Qf = Qf * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        else:
            n_valid = torch.tensor(T, dtype=f.dtype, device=f.device)
        cov = torch.einsum("btr, bts -> brs", Pf, Qf)
        if mask is not None:
            cov = cov / n_valid.unsqueeze(-1)
        else:
            cov = cov / T
        e_2 = self.cov_proj(cov.reshape(B, -1))  # (B, d)

        combined = torch.cat([e_1, e_2], dim=-1)
        embedding = self.layer_norm(self.out_proj(combined))
        return embedding


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------

def _swap_interference(model, InterferenceClass):
    for layer in model.layers:
        layer.interference_fwd = InterferenceClass(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = InterferenceClass(layer.d_c, layer.n_channels)
    return model


def make_variant_a():
    return _swap_interference(prism_small(vocab_size=VOCAB_SIZE), InterferenceNormOnly)

def make_variant_b():
    return _swap_interference(prism_small(vocab_size=VOCAB_SIZE), InterferenceScaleOnly)

def make_variant_c():
    return _swap_interference(prism_small(vocab_size=VOCAB_SIZE), InterferenceAlphaInit)

def make_variant_d():
    return _swap_interference(prism_small(vocab_size=VOCAB_SIZE), InterferenceGateOnly)

def make_variant_e():
    model = prism_small(vocab_size=VOCAB_SIZE)
    model.pooling = CovPoolingNormOnly(model.d, model.d_e, cov_rank=24)
    return model

def make_variant_f():
    model = prism_small(vocab_size=VOCAB_SIZE)
    model.pooling = CovPoolingRankProj(model.d, model.d_e, cov_rank=8)
    return model

def make_variant_g():
    model = prism_small(vocab_size=VOCAB_SIZE)
    _swap_interference(model, CrossScaleInterferenceV2)
    model.pooling = AttentiveCovariancePoolingV2(model.d, model.d_e, cov_rank=8)
    return model


# Also include mean pool baseline (the current best simplified version)
def make_meanpool_baseline():
    from benchmark_ablations import MeanPooling, NoInterference
    model = prism_small(vocab_size=VOCAB_SIZE)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


ABLATIONS = {
    "Baseline (full original)": lambda: prism_small(vocab_size=VOCAB_SIZE),
    "Simplified (mean pool)": make_meanpool_baseline,
    "A: Interf. LayerNorm": make_variant_a,
    "B: Interf. 1/√d_c": make_variant_b,
    "C: Interf. α=1/(C-1)": make_variant_c,
    "D: Interf. Gate": make_variant_d,
    "E: Cov LayerNorm": make_variant_e,
    "F: Cov Rank8+Proj": make_variant_f,
    "G: All V2 Fixes": make_variant_g,
}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_v2_ablations():
    print("=" * 70)
    print("V2 ABLATION STUDY: Testing Targeted Fixes")
    print(f"  seq_q={SEQ_LEN_Q}, seq_p={SEQ_LEN_P}, steps={N_STEPS}")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)
    queries, corpus, q_topics, c_topics = dataset.generate_corpus_and_queries(
        N_EVAL, CORPUS_SIZE, SEQ_LEN_Q, SEQ_LEN_P
    )

    all_results = {}
    all_losses = {}

    for name, builder in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        random.seed(42)

        model = builder().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        wrapper = PRISMForEmbedding(model, temperature=0.05).to(DEVICE)
        losses, accs, train_time = train_model(wrapper, dataset, N_STEPS, name)
        metrics = evaluate_retrieval(wrapper, queries, corpus, q_topics, c_topics)

        all_results[name] = {
            "metrics": metrics,
            "train_time": train_time,
            "params": n_params,
            "final_loss": float(np.mean(losses[-100:])),
            "final_acc": float(np.mean(accs[-100:])),
        }
        all_losses[name] = losses

        print(f"  MRR: {metrics['mrr']:.4f}  R@1: {metrics['recall@1']:.4f}  "
              f"R@5: {metrics['recall@5']:.4f}  loss: {all_results[name]['final_loss']:.4f}")

    # Save
    with open(RESULTS_DIR / "v2_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 90)
    print("V2 ABLATION RESULTS")
    print("=" * 90)
    print(f"{'Variant':<30} {'Params':>10} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'Loss':>8} {'Time':>8}")
    print("-" * 90)
    for name, r in all_results.items():
        m = r["metrics"]
        print(f"{name:<30} {r['params']:>10,} {m['mrr']:>8.4f} {m['recall@1']:>8.4f} "
              f"{m['recall@5']:>8.4f} {r['final_loss']:>8.4f} {r['train_time']:>7.0f}s")

    # Delta table
    base_mrr = all_results["Baseline (full original)"]["metrics"]["mrr"]
    simple_mrr = all_results["Simplified (mean pool)"]["metrics"]["mrr"]
    print(f"\n  Baseline MRR: {base_mrr:.4f}")
    print(f"  Simplified (mean pool) MRR: {simple_mrr:.4f} (target to match or beat)")
    print(f"\n  {'Variant':<30} {'Delta vs Base':>14} {'Delta vs Simple':>16}")
    print("  " + "-" * 62)
    for name, r in all_results.items():
        if name in ("Baseline (full original)", "Simplified (mean pool)"):
            continue
        mrr = r["metrics"]["mrr"]
        print(f"  {name:<30} {mrr - base_mrr:>+14.4f} {mrr - simple_mrr:>+16.4f}")

    # Plot
    _plot_v2_results(all_results, all_losses)

    return all_results


def _plot_v2_results(all_results, all_losses):
    COLORS = [
        "#2563EB", "#14532D", "#F59E0B", "#10B981", "#8B5CF6",
        "#EF4444", "#EC4899", "#06B6D4", "#D97706",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"V2 Ablation Study (seq_q={SEQ_LEN_Q}, seq_p={SEQ_LEN_P}, {N_STEPS} steps)",
                 fontsize=14, fontweight="bold")

    # Loss curves
    ax = axes[0]
    window = 50
    for i, (name, losses) in enumerate(all_losses.items()):
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smooth, color=COLORS[i % len(COLORS)], linewidth=1.5,
                label=name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    # MRR bar chart
    ax = axes[1]
    names = list(all_results.keys())
    mrrs = [all_results[n]["metrics"]["mrr"] for n in names]
    bars = ax.barh(range(len(names)), mrrs,
                   color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                   alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("MRR")
    ax.set_title("Retrieval Quality")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, mrr in zip(bars, mrrs):
        ax.annotate(f"{mrr:.4f}", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)

    # Delta vs simplified baseline
    ax = axes[2]
    simple_mrr = all_results["Simplified (mean pool)"]["metrics"]["mrr"]
    delta_names = []
    deltas = []
    delta_colors = []
    for i, (name, r) in enumerate(all_results.items()):
        if name in ("Baseline (full original)", "Simplified (mean pool)"):
            continue
        delta_names.append(name)
        deltas.append(r["metrics"]["mrr"] - simple_mrr)
        delta_colors.append(COLORS[(i) % len(COLORS)])

    bars = ax.barh(range(len(delta_names)), deltas, color=delta_colors, alpha=0.85)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(delta_names)))
    ax.set_yticklabels(delta_names, fontsize=7)
    ax.set_xlabel("MRR Delta vs Simplified")
    ax.set_title("Impact vs Mean-Pool Baseline")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, d in zip(bars, deltas):
        ax.annotate(f"{d:+.4f}",
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(3 if d >= 0 else -3, 0), textcoords="offset points",
                   ha="left" if d >= 0 else "right", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "v2_ablation_comparison.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved: {RESULTS_DIR / 'v2_ablation_comparison.png'}")
    plt.close("all")


if __name__ == "__main__":
    run_v2_ablations()
