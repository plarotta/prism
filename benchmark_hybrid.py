"""
Hybrid PRISM Experiments: Baselines + Attentive Pooling + Decay Spacing Ablations

Implements:
  Experiment 0: LoCoV1 baselines (PRISM-MeanPool + Transformer) with LR sweep
                and checkpoint evaluation at intermediate training steps
  Experiment 1: AttentivePooling head (single learned query) at 2K and 8K
  Experiment 2: MultiHeadAttentivePooling (K=4,8) — conditional on Exp 1
  Experiment 3: Local attention + recurrence hybrid — conditional on Exp 1-2
  Experiment 4: Decay spacing ablation (geometric vs linear vs random vs all-slow vs all-fast)

Reuses the LoCoV1 training/eval infrastructure from benchmark_loco.py.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from prism import (
    PRISMEncoder, PRISMForEmbedding, prism_small,
    AttentivePooling, MultiHeadAttentivePooling,
)
from baseline_transformer import transformer_small, TransformerForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from benchmark_loco import (
    DEVICE, TOKENIZER_NAME, REAL_VOCAB_SIZE, LOCO_TASKS,
    PUBLISHED_BASELINES,
    load_loco_data, make_training_pairs,
    train_contrastive, evaluate_task_ndcg,
    _get_batch_config, _collate_ids,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results") / "hybrid"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_prism_attentive(vocab_size=REAL_VOCAB_SIZE, max_len=8192, **kwargs):
    """PRISM-Simplified with AttentivePooling (single learned query)."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = AttentivePooling(model.d, model.d_e)
    return model


def build_prism_multihead(vocab_size=REAL_VOCAB_SIZE, max_len=8192, n_heads=4, **kwargs):
    """PRISM-Simplified with MultiHeadAttentivePooling."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = MultiHeadAttentivePooling(model.d, model.d_e, n_heads=n_heads)
    return model


def build_prism_mean(vocab_size=REAL_VOCAB_SIZE, max_len=8192, **kwargs):
    """PRISM-Simplified with MeanPooling (existing baseline)."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


# ---------------------------------------------------------------------------
# Experiment 3: Local attention layer
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Single-head sliding-window local attention. O(n * window_size).

    Provides precise token-level matching within a local neighborhood,
    complementing the recurrence's global but lossy context.
    """

    def __init__(self, d: int, window_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.window_size = window_size
        self.norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d)
            mask: (B, T) bool — True for valid positions
        Returns:
            (B, T, d)
        """
        B, T, D = x.shape
        w = self.window_size
        residual = x
        x = self.norm(x)

        Q = self.q_proj(x)  # (B, T, d)
        K = self.k_proj(x)  # (B, T, d)
        V = self.v_proj(x)  # (B, T, d)

        # Pad K, V for sliding window: half_w left, half_w-1 right → T+w-1 elements
        # unfold(1, w, 1) on T+w-1 → exactly T windows
        half_w = w // 2
        pad_right = w - 1 - half_w
        K_pad = F.pad(K, (0, 0, half_w, pad_right))  # (B, T+w-1, d)
        V_pad = F.pad(V, (0, 0, half_w, pad_right))  # (B, T+w-1, d)

        # Unfold to get windows: (B, T, d, w) -> permute to (B, T, w, d)
        K_windows = K_pad.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T, w, d)
        V_windows = V_pad.unfold(1, w, 1).permute(0, 1, 3, 2)  # (B, T, w, d)

        # Attention scores: (B, T, w)
        scores = torch.einsum("btd, btwd -> btw", Q, K_windows) * self.scale

        # Mask: build window mask from padding mask
        if mask is not None:
            mask_pad = F.pad(mask.float(), (half_w, pad_right))  # (B, T+w-1)
            mask_windows = mask_pad.unfold(1, w, 1)  # (B, T, w)
            scores = scores.masked_fill(mask_windows == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, T, w)
        # Guard against NaN from all-masked windows (softmax of all -inf)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = torch.einsum("btw, btwd -> btd", attn, V_windows)  # (B, T, d)
        out = self.out_proj(out)
        return residual + out


def build_prism_local_attn(
    vocab_size=REAL_VOCAB_SIZE, max_len=8192, window_size=256, attn_after_layer=3, **kwargs
):
    """PRISM-Simplified + one local attention layer inserted into the stack.

    Default: local attention after layer 4 (0-indexed: 3) of 6.
    Uses attentive pooling head.
    """
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = AttentivePooling(model.d, model.d_e)

    # Insert local attention layer
    local_attn = SlidingWindowAttention(model.d, window_size=window_size)
    model.local_attn = local_attn
    model.local_attn_after = attn_after_layer

    # Monkey-patch forward to insert local attention
    original_forward = model.forward

    def hybrid_forward(input_ids, attention_mask=None):
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = (input_ids != model.pad_token_id).long()
        mask_bool = attention_mask.bool()

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = model.token_emb(input_ids) + model.pos_emb(positions)
        x = model.emb_dropout(model.emb_norm(x))
        x = x * mask_bool.unsqueeze(-1).float()

        fwd_global = None
        for i, layer in enumerate(model.layers):
            x, fwd_global = layer(x, mask_bool)
            x = x * mask_bool.unsqueeze(-1).float()
            # Insert local attention after the designated layer
            if i == model.local_attn_after:
                x = model.local_attn(x, mask_bool)
                x = x * mask_bool.unsqueeze(-1).float()

        x = model.final_norm(x)
        query = model.query_proj(fwd_global)
        embedding = model.pooling(x, query, mask_bool)
        return {"embedding": embedding, "token_embeddings": x}

    model.forward = hybrid_forward
    return model


# ---------------------------------------------------------------------------
# Experiment 4: Decay spacing variants
# ---------------------------------------------------------------------------

def _set_decay_rates(model: PRISMEncoder, decay_mode: str, max_len: int):
    """Override decay rates in all recurrence layers.

    Modes:
        geometric: λ_c = 1 - 2^(-c*Δ) where Δ = log2(max_len)/(C-1) [default]
        linear:    λ_c evenly spaced from 0 to 1-ε
        random:    λ_c sampled uniformly from (0, 1), frozen
        all_slow:  all channels λ=0.99
        all_fast:  all channels λ=0.1
    """
    for layer in model.layers:
        rec = layer.recurrence
        C = rec.n_channels

        if decay_mode == "geometric":
            delta = math.log2(max_len) / max(C - 1, 1)
            lambdas = [1.0 - 2.0 ** (-(c * delta)) for c in range(C)]
        elif decay_mode == "linear":
            lambdas = [c / C for c in range(C)]  # 0, 1/C, 2/C, ..., (C-1)/C
        elif decay_mode == "random":
            rng = random.Random(42)  # Fixed seed for reproducibility
            lambdas = sorted([rng.random() for _ in range(C)])
        elif decay_mode == "all_slow":
            lambdas = [0.99] * C
        elif decay_mode == "all_fast":
            lambdas = [0.1] * C
        else:
            raise ValueError(f"Unknown decay_mode: {decay_mode}")

        rec.lambdas.copy_(torch.tensor(lambdas, dtype=torch.float32))


def build_prism_decay_variant(
    decay_mode: str, vocab_size=REAL_VOCAB_SIZE, max_len=8192, use_attentive=True, **kwargs
):
    """PRISM-Simplified with a specific decay spacing strategy."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    if use_attentive:
        model.pooling = AttentivePooling(model.d, model.d_e)
    else:
        model.pooling = MeanPooling(model.d, model.d_e)
    _set_decay_rates(model, decay_mode, max_len)
    return model


# ---------------------------------------------------------------------------
# LR sweep (minimal)
# ---------------------------------------------------------------------------

LR_CANDIDATES = [1e-4, 3e-4, 5e-4, 1e-3]


def sweep_lr(
    build_fn,
    wrapper_cls,
    train_pairs: list,
    model_name: str,
    max_len: int = 2048,
    sweep_steps: int = 500,
    batch_size: int = 16,
) -> tuple[float, dict]:
    """Quick LR sweep for a single model. Returns (best_lr, {lr: final_loss})."""
    print(f"\n  Sweeping LR for {model_name}...")
    bcfg = _get_batch_config(model_name, max_len, batch_size)
    lr_losses = {}

    for lr in LR_CANDIDATES:
        torch.manual_seed(42)
        random.seed(42)

        encoder = build_fn().to(DEVICE)
        wrapper = wrapper_cls(encoder, temperature=0.05).to(DEVICE)

        losses, _ = train_contrastive(
            wrapper, train_pairs, sweep_steps, f"{model_name}/lr={lr:.0e}",
            max_len=max_len,
            micro_batch=bcfg["micro_batch"],
            grad_accum_steps=bcfg["grad_accum_steps"],
            lr=lr,
            log_interval=sweep_steps,  # only log at end
        )

        final_loss = float(np.mean(losses[-50:])) if len(losses) >= 50 else float(np.mean(losses))
        lr_losses[lr] = final_loss
        print(f"    lr={lr:.0e}  final_loss={final_loss:.4f}")

        del wrapper, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_lr = min(lr_losses, key=lr_losses.get)
    print(f"  -> Best for {model_name}: lr={best_lr:.0e} (loss={lr_losses[best_lr]:.4f})")
    return best_lr, lr_losses


# ---------------------------------------------------------------------------
# Training with checkpoint evaluation
# ---------------------------------------------------------------------------

def train_with_checkpoints(
    model_wrapper,
    train_pairs: list,
    tasks: dict,
    n_steps: int,
    model_name: str,
    max_len: int = 2048,
    micro_batch: int = 16,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    eval_steps: list[int] = None,
    eval_batch: int = 32,
    log_interval: int = 100,
) -> dict:
    """Train with InfoNCE, evaluating on LoCoV1 at intermediate checkpoints.

    Args:
        eval_steps: list of step numbers at which to run full LoCoV1 evaluation.
                    Defaults to [1000, 2000, 3000, 4000, 5000] (or [n_steps] if short).
    """
    if eval_steps is None:
        if n_steps >= 2000:
            eval_steps = list(range(1000, n_steps + 1, 1000))
            if n_steps not in eval_steps:
                eval_steps.append(n_steps)
        else:
            eval_steps = [n_steps]
    eval_steps_set = set(eval_steps)

    effective_batch = micro_batch * grad_accum_steps
    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )

    losses = []
    checkpoint_results = {}
    model_wrapper.train()

    print(f"\nTraining {model_name} ({n_steps} steps, max_len={max_len}, "
          f"micro_batch={micro_batch}, accum={grad_accum_steps}, "
          f"effective_batch={effective_batch})")
    print(f"  Checkpoint eval at steps: {eval_steps}")
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum_steps):
            batch_pairs = random.choices(train_pairs, k=micro_batch)
            q_ids_list = [p[0] for p in batch_pairs]
            d_ids_list = [p[1] for p in batch_pairs]

            q_ids, q_mask = _collate_ids(q_ids_list, max_len)
            d_ids, d_mask = _collate_ids(d_ids_list, max_len)
            q_ids, q_mask = q_ids.to(DEVICE), q_mask.to(DEVICE)
            d_ids, d_mask = d_ids.to(DEVICE), d_mask.to(DEVICE)

            result = model_wrapper(q_ids, q_mask, d_ids, d_mask)
            loss = result["loss"] / grad_accum_steps
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(step_loss)

        if (step + 1) % log_interval == 0:
            recent_loss = np.mean(losses[-log_interval:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  ({elapsed:.1f}s)")

        # Checkpoint evaluation
        if (step + 1) in eval_steps_set:
            ckpt_t0 = time.perf_counter()
            print(f"\n  [{model_name}] Checkpoint eval at step {step+1}...")
            model_wrapper.eval()

            task_results = {}
            for task_name in LOCO_TASKS:
                task_data = tasks[task_name]
                if not task_data["queries"] or not task_data["documents"]:
                    continue
                try:
                    metrics = evaluate_task_ndcg(
                        model_wrapper, task_data, max_len,
                        batch_size=eval_batch, device=DEVICE,
                    )
                    if metrics:
                        task_results[task_name] = metrics
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        try:
                            metrics = evaluate_task_ndcg(
                                model_wrapper, task_data, max_len,
                                batch_size=1, device=DEVICE,
                            )
                            if metrics:
                                task_results[task_name] = metrics
                        except RuntimeError:
                            pass

            valid_ndcg = [r["nDCG@10"] for r in task_results.values()
                          if isinstance(r, dict) and "nDCG@10" in r]
            avg_ndcg = float(np.mean(valid_ndcg)) if valid_ndcg else 0.0
            recent_loss = float(np.mean(losses[-min(100, len(losses)):]))
            ckpt_time = time.perf_counter() - ckpt_t0

            checkpoint_results[step + 1] = {
                "task_results": task_results,
                "avg_nDCG@10": avg_ndcg,
                "loss": recent_loss,
            }
            print(f"  [{model_name}] Step {step+1}: avg nDCG@10={avg_ndcg:.4f}, "
                  f"loss={recent_loss:.4f} (eval took {ckpt_time:.0f}s)")

            model_wrapper.train()

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")

    return {
        "losses": [float(l) for l in losses],
        "train_time": total_time,
        "checkpoints": checkpoint_results,
    }


# ---------------------------------------------------------------------------
# Experiment 0: LoCoV1 baselines with LR sweep + checkpoint eval
# ---------------------------------------------------------------------------

def run_experiment_0(
    tasks: dict,
    train_pairs: list,
    max_lens: list[int] = None,
    n_steps: int = 5000,
    batch_size: int = 16,
    sweep_steps: int = 500,
    eval_steps: list[int] = None,
):
    """Experiment 0: LoCoV1 baselines with LR sweep and checkpoint evaluation.

    Trains PRISM-MeanPool and Transformer from scratch on LoCoV1, evaluating
    at intermediate checkpoints to track quality during training. Uses a minimal
    LR sweep to find the best learning rate per model.

    Returns:
        results: dict keyed by "prism_mean_{max_len}" and "transformer_{max_len}"
        model_lrs: dict {model_name: best_lr}
    """
    if max_lens is None:
        max_lens = [2048]

    print("\n" + "#" * 70)
    print("# EXPERIMENT 0: LoCoV1 Baselines (LR sweep + checkpoint eval)")
    print("#" * 70)

    # --- LR sweep at the shortest max_len ---
    sweep_max_len = min(max_lens)
    model_configs = [
        ("PRISM-MeanPool", lambda ml: build_prism_mean(max_len=ml), PRISMForEmbedding),
        ("Transformer", lambda ml: transformer_small(vocab_size=REAL_VOCAB_SIZE, max_len=ml),
         TransformerForEmbedding),
    ]

    model_lrs = {}
    print(f"\n  LR Sweep ({sweep_steps} steps at max_len={sweep_max_len})")
    for name, build_fn, wrapper_cls in model_configs:
        best_lr, _ = sweep_lr(
            lambda bf=build_fn, sml=sweep_max_len: bf(sml),
            wrapper_cls, train_pairs, name,
            max_len=sweep_max_len, sweep_steps=sweep_steps, batch_size=batch_size,
        )
        model_lrs[name] = best_lr

    # --- Train with checkpoint eval at each max_len ---
    results = {}
    for max_len in max_lens:
        for name, build_fn, wrapper_cls in model_configs:
            print(f"\n{'='*60}")
            print(f"  {name} @ max_len={max_len}")
            print(f"{'='*60}")

            torch.manual_seed(42)
            random.seed(42)

            encoder = build_fn(max_len).to(DEVICE)
            wrapper = wrapper_cls(encoder, temperature=0.05).to(DEVICE)
            n_params = sum(p.numel() for p in wrapper.parameters())
            model_lr = model_lrs[name]
            bcfg = _get_batch_config(name, max_len, batch_size)

            print(f"  Parameters: {n_params:,}")
            print(f"  LR: {model_lr:.0e} (from sweep)")
            print(f"  Batch config: micro_batch={bcfg['micro_batch']}, "
                  f"grad_accum={bcfg['grad_accum_steps']}, "
                  f"effective_batch={bcfg['micro_batch'] * bcfg['grad_accum_steps']}")

            train_result = train_with_checkpoints(
                wrapper, train_pairs, tasks, n_steps, name,
                max_len=max_len,
                micro_batch=bcfg["micro_batch"],
                grad_accum_steps=bcfg["grad_accum_steps"],
                lr=model_lr,
                eval_steps=eval_steps,
                eval_batch=bcfg["eval_batch"],
            )

            # Final checkpoint is the authoritative result
            final_ckpt = train_result["checkpoints"].get(n_steps, {})

            key = f"{'prism_mean' if 'prism' in name.lower() else 'transformer'}_{max_len}"
            results[key] = {
                "task_results": final_ckpt.get("task_results", {}),
                "avg_nDCG@10": final_ckpt.get("avg_nDCG@10", 0.0),
                "train_time": train_result["train_time"],
                "final_loss": final_ckpt.get("loss", 0.0),
                "losses": train_result["losses"],
                "checkpoints": train_result["checkpoints"],
                "params": n_params,
                "max_len": max_len,
                "n_steps": n_steps,
                "lr": model_lr,
            }
            print(f"\n  {name} @ {max_len}: avg nDCG@10={results[key]['avg_nDCG@10']:.4f}")

            del wrapper, encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXPERIMENT 0 RESULTS")
    print("=" * 60)
    print(f"  {'Config':<30} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8} {'LR':>8}")
    print("  " + "-" * 70)
    for key, r in results.items():
        print(f"  {key:<30} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s {r['lr']:>8.0e}")

    # Print checkpoint trajectory
    for key, r in results.items():
        if "checkpoints" in r and r["checkpoints"]:
            steps_sorted = sorted(r["checkpoints"].keys())
            trajectory = ", ".join(
                f"{s}: {r['checkpoints'][s]['avg_nDCG@10']:.3f}"
                for s in steps_sorted
            )
            print(f"  {key} trajectory: {trajectory}")

    return results, model_lrs


def plot_experiment_0(results: dict):
    """Plot Experiment 0: checkpoint trajectories + final comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: nDCG@10 trajectory over training steps
    colors = {"prism_mean": "#2563EB", "transformer": "#DC2626"}
    for key, r in results.items():
        checkpoints = r.get("checkpoints", {})
        if not checkpoints:
            continue
        steps = sorted(checkpoints.keys())
        ndcgs = [checkpoints[s]["avg_nDCG@10"] for s in steps]
        model_type = "prism_mean" if "prism" in key else "transformer"
        max_len = r.get("max_len", "?")
        linestyle = "-" if max_len <= 2048 else "--"
        ax1.plot(steps, ndcgs, marker="o", color=colors.get(model_type, "#666"),
                 linestyle=linestyle, label=f"{key}", markersize=5)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Average nDCG@10")
    ax1.set_title("Experiment 0: Quality During Training")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Right: loss curves
    for key, r in results.items():
        losses = r.get("losses", [])
        if not losses:
            continue
        # Smooth losses for plotting
        window = min(50, len(losses) // 10 + 1)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        model_type = "prism_mean" if "prism" in key else "transformer"
        max_len = r.get("max_len", "?")
        linestyle = "-" if max_len <= 2048 else "--"
        ax2.plot(range(len(smoothed)), smoothed, color=colors.get(model_type, "#666"),
                 linestyle=linestyle, label=f"{key}", alpha=0.8)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Experiment 0: Training Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp0_baselines.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'exp0_baselines.png'}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Training and evaluation runner
# ---------------------------------------------------------------------------

def run_experiment(
    tasks: dict,
    train_pairs: list,
    name: str,
    build_fn,
    max_len: int,
    n_steps: int = 5000,
    batch_size: int = 16,
    lr: float = 3e-4,
    micro_batch_cap: int = None,
    eval_steps: list[int] = None,
) -> dict:
    """Train a model with checkpoint evaluation on all LoCoV1 tasks.

    Uses train_with_checkpoints() to evaluate at intermediate steps,
    matching the protocol from Experiment 0.

    Args:
        micro_batch_cap: If set, caps the micro-batch size (use for models with
            higher memory footprint than what _get_batch_config estimates, e.g.
            local attention variants with unfold overhead).
        eval_steps: Steps at which to run full LoCoV1 eval. Defaults to every
            1000 steps.
    """
    print(f"\n{'='*60}")
    print(f"  {name} (max_len={max_len})")
    print(f"{'='*60}")

    torch.manual_seed(42)
    random.seed(42)

    encoder = build_fn().to(DEVICE)
    wrapper = PRISMForEmbedding(encoder, temperature=0.05).to(DEVICE)
    n_params = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n_params:,}")

    bcfg = _get_batch_config(name, max_len, batch_size)
    if micro_batch_cap is not None and bcfg["micro_batch"] > micro_batch_cap:
        bcfg["micro_batch"] = micro_batch_cap
        bcfg["grad_accum_steps"] = max(1, batch_size // micro_batch_cap)
        bcfg["eval_batch"] = min(micro_batch_cap * 3, batch_size * 2)
    print(f"  Batch config: micro_batch={bcfg['micro_batch']}, "
          f"grad_accum={bcfg['grad_accum_steps']}, "
          f"effective_batch={bcfg['micro_batch'] * bcfg['grad_accum_steps']}")

    train_result = train_with_checkpoints(
        wrapper, train_pairs, tasks, n_steps, name,
        max_len=max_len,
        micro_batch=bcfg["micro_batch"],
        grad_accum_steps=bcfg["grad_accum_steps"],
        lr=lr,
        eval_steps=eval_steps,
        eval_batch=bcfg["eval_batch"],
    )

    # Final checkpoint is the authoritative result
    final_ckpt = train_result["checkpoints"].get(n_steps, {})

    result = {
        "task_results": final_ckpt.get("task_results", {}),
        "avg_nDCG@10": final_ckpt.get("avg_nDCG@10", 0.0),
        "train_time": train_result["train_time"],
        "final_loss": final_ckpt.get("loss", 0.0),
        "losses": train_result["losses"],
        "checkpoints": train_result["checkpoints"],
        "params": n_params,
        "max_len": max_len,
        "n_steps": n_steps,
        "lr": lr,
    }
    print(f"\n  {name} avg nDCG@10: {result['avg_nDCG@10']:.4f} "
          f"(loss={result['final_loss']:.4f}, {train_result['train_time']:.0f}s)")

    del wrapper, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Experiment 1: Attentive Pooling
# ---------------------------------------------------------------------------

def run_experiment_1(
    tasks, train_pairs, n_steps=5000, batch_size=16, lr=3e-4,
    exp0_results: dict = None,
):
    """Experiment 1: Compare attentive pooling vs mean pooling at 2K and 8K.

    If exp0_results is provided, uses those as mean pool baselines instead
    of re-running them (saves ~2 hours of GPU time).
    """
    print("\n" + "#" * 70)
    print("# EXPERIMENT 1: Attentive Pooling vs Mean Pooling")
    print("#" * 70)

    results = {}

    # Run 1a: Attentive pooling at 2048
    results["attentive_2048"] = run_experiment(
        tasks, train_pairs, "PRISM-Attentive",
        lambda: build_prism_attentive(max_len=2048),
        max_len=2048, n_steps=n_steps, batch_size=batch_size, lr=lr,
    )

    # Run 1b: Attentive pooling at 8192
    results["attentive_8192"] = run_experiment(
        tasks, train_pairs, "PRISM-Attentive",
        lambda: build_prism_attentive(max_len=8192),
        max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
    )

    # Mean pooling baselines: reuse from Experiment 0 if available
    if exp0_results and "prism_mean_2048" in exp0_results:
        results["mean_2048"] = exp0_results["prism_mean_2048"]
        print(f"\n  Using Exp 0 baseline for mean_2048: "
              f"nDCG@10={results['mean_2048']['avg_nDCG@10']:.4f}")
    else:
        results["mean_2048"] = run_experiment(
            tasks, train_pairs, "PRISM-MeanPool",
            lambda: build_prism_mean(max_len=2048),
            max_len=2048, n_steps=n_steps, batch_size=batch_size, lr=lr,
        )

    if exp0_results and "prism_mean_8192" in exp0_results:
        results["mean_8192"] = exp0_results["prism_mean_8192"]
        print(f"  Using Exp 0 baseline for mean_8192: "
              f"nDCG@10={results['mean_8192']['avg_nDCG@10']:.4f}")
    else:
        results["mean_8192"] = run_experiment(
            tasks, train_pairs, "PRISM-MeanPool",
            lambda: build_prism_mean(max_len=8192),
            max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
        )

    # Analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 60)
    print(f"  {'Config':<30} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8}")
    print("  " + "-" * 62)
    for key, r in results.items():
        pooling, ml = key.rsplit("_", 1)
        print(f"  {pooling + ' @ ' + ml:<30} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s")

    # Check success criteria
    att_8k = results["attentive_8192"]["avg_nDCG@10"]
    mean_8k = results["mean_8192"]["avg_nDCG@10"]
    mean_2k = results["mean_2048"]["avg_nDCG@10"]
    att_2k = results["attentive_2048"]["avg_nDCG@10"]

    print(f"\n  Primary: Attentive 8K ({att_8k:.4f}) vs Mean 8K ({mean_8k:.4f}): "
          f"{'PASS' if att_8k > mean_8k else 'FAIL'} (delta={att_8k - mean_8k:+.4f})")
    print(f"  Ideal:   Attentive 8K ({att_8k:.4f}) vs Mean 2K ({mean_2k:.4f}): "
          f"{'PASS' if att_8k >= mean_2k else 'FAIL'} (delta={att_8k - mean_2k:+.4f})")
    print(f"  Bonus:   Attentive 2K ({att_2k:.4f}) vs Mean 2K ({mean_2k:.4f}): "
          f"{'PASS' if att_2k > mean_2k else 'FAIL'} (delta={att_2k - mean_2k:+.4f})")

    improvement = att_8k > mean_8k
    return results, improvement


# ---------------------------------------------------------------------------
# Experiment 2: Multi-Head Attentive Pooling (conditional)
# ---------------------------------------------------------------------------

def run_experiment_2(tasks, train_pairs, n_steps=5000, batch_size=16, lr=3e-4):
    """Experiment 2: Multi-head attentive pooling at 8K."""
    print("\n" + "#" * 70)
    print("# EXPERIMENT 2: Multi-Head Attentive Pooling")
    print("#" * 70)

    results = {}

    # Run 2a: K=4 heads at 8192
    results["multihead_k4_8192"] = run_experiment(
        tasks, train_pairs, "PRISM-MultiHead-K4",
        lambda: build_prism_multihead(max_len=8192, n_heads=4),
        max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
    )

    # Run 2b: K=8 heads at 8192
    results["multihead_k8_8192"] = run_experiment(
        tasks, train_pairs, "PRISM-MultiHead-K8",
        lambda: build_prism_multihead(max_len=8192, n_heads=8),
        max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)
    print(f"  {'Config':<30} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8}")
    print("  " + "-" * 62)
    for key, r in results.items():
        print(f"  {key:<30} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Local Attention Hybrid (conditional)
# ---------------------------------------------------------------------------

def run_experiment_3(tasks, train_pairs, n_steps=5000, batch_size=16, lr=3e-4):
    """Experiment 3: Sliding-window local attention + recurrence hybrid at 8K."""
    print("\n" + "#" * 70)
    print("# EXPERIMENT 3: Local Attention + Recurrence Hybrid")
    print("#" * 70)

    # SlidingWindowAttention uses unfold which materializes (B, T, w, d) tensors.
    # At 8K this adds ~6 GB/sample (w=256) or ~12 GB/sample (w=512) on top of
    # base PRISM costs. Cap micro_batch to avoid OOM.
    results = {}

    # Run 3a: window=256 — ~7 GB/sample total, cap micro_batch=8 for safety
    results["local_attn_w256_8192"] = run_experiment(
        tasks, train_pairs, "PRISM-LocalAttn-W256",
        lambda: build_prism_local_attn(max_len=8192, window_size=256),
        max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
        micro_batch_cap=8,
    )

    # Run 3b: window=512 — ~13 GB/sample total, cap micro_batch=4 for safety
    results["local_attn_w512_8192"] = run_experiment(
        tasks, train_pairs, "PRISM-LocalAttn-W512",
        lambda: build_prism_local_attn(max_len=8192, window_size=512),
        max_len=8192, n_steps=n_steps, batch_size=batch_size, lr=lr,
        micro_batch_cap=4,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 60)
    print(f"  {'Config':<30} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8}")
    print("  " + "-" * 62)
    for key, r in results.items():
        print(f"  {key:<30} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Decay Spacing Ablation (independent)
# ---------------------------------------------------------------------------

DECAY_MODES = ["geometric", "linear", "random", "all_slow", "all_fast"]


def run_experiment_4(
    tasks, train_pairs, n_steps=5000, batch_size=16, lr=3e-4,
    use_attentive=True,
):
    """Experiment 4: Decay spacing ablation at 2048.

    Tests whether geometric decay spacing is a meaningful inductive bias.
    """
    print("\n" + "#" * 70)
    print("# EXPERIMENT 4: Decay Spacing Ablation")
    print("#" * 70)

    pooling_label = "attentive" if use_attentive else "mean"
    results = {}

    for decay_mode in DECAY_MODES:
        key = f"decay_{decay_mode}_2048"
        results[key] = run_experiment(
            tasks, train_pairs, f"PRISM-{decay_mode}-{pooling_label}",
            lambda dm=decay_mode: build_prism_decay_variant(
                dm, max_len=2048, use_attentive=use_attentive
            ),
            max_len=2048, n_steps=n_steps, batch_size=batch_size, lr=lr,
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 RESULTS: Decay Spacing Ablation")
    print("=" * 60)
    print(f"  {'Decay Mode':<20} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8}")
    print("  " + "-" * 52)
    for mode in DECAY_MODES:
        r = results[f"decay_{mode}_2048"]
        marker = " ← default" if mode == "geometric" else ""
        print(f"  {mode:<20} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s{marker}")

    # Analysis
    geo_score = results["decay_geometric_2048"]["avg_nDCG@10"]
    for mode in DECAY_MODES:
        if mode == "geometric":
            continue
        other = results[f"decay_{mode}_2048"]["avg_nDCG@10"]
        delta = geo_score - other
        print(f"\n  geometric vs {mode}: {delta:+.4f} "
              f"({'geometric better' if delta > 0 else mode + ' better or tied'})")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_experiment_1(results: dict):
    """Plot Experiment 1: pooling comparison across sequence lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of avg nDCG@10
    configs = ["mean_2048", "attentive_2048", "mean_8192", "attentive_8192"]
    labels = ["Mean\n2048", "Attentive\n2048", "Mean\n8192", "Attentive\n8192"]
    colors = ["#9CA3AF", "#2563EB", "#9CA3AF", "#2563EB"]
    hatches = ["", "", "//", "//"]
    scores = [results[c]["avg_nDCG@10"] for c in configs]

    bars = ax1.bar(range(len(configs)), scores, color=colors, edgecolor="black", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average nDCG@10")
    ax1.set_title("Experiment 1: Pooling Method Comparison")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: per-task comparison at 8K
    task_names = [t for t in LOCO_TASKS
                  if t in results["attentive_8192"]["task_results"]
                  and t in results["mean_8192"]["task_results"]]
    x = np.arange(len(task_names))
    width = 0.35

    mean_scores = [results["mean_8192"]["task_results"][t]["nDCG@10"] for t in task_names]
    att_scores = [results["attentive_8192"]["task_results"][t]["nDCG@10"] for t in task_names]

    ax2.bar(x - width / 2, mean_scores, width, label="Mean Pool", color="#9CA3AF")
    ax2.bar(x + width / 2, att_scores, width, label="Attentive Pool", color="#2563EB")
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("nDCG@10")
    ax2.set_title("Per-Task Comparison at max_len=8192")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_attentive_pooling.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'exp1_attentive_pooling.png'}")
    plt.close("all")


def plot_experiment_4(results: dict):
    """Plot Experiment 4: decay spacing ablation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = DECAY_MODES
    scores = [results[f"decay_{m}_2048"]["avg_nDCG@10"] for m in modes]
    colors = ["#2563EB", "#10B981", "#F59E0B", "#DC2626", "#8B5CF6"]

    bars = ax.bar(range(len(modes)), scores, color=colors, edgecolor="black", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([m.replace("_", "\n") for m in modes])
    ax.set_ylabel("Average nDCG@10")
    ax.set_title("Experiment 4: Decay Spacing Ablation (max_len=2048)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_decay_spacing.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'exp4_decay_spacing.png'}")
    plt.close("all")


def plot_all_results(all_results: dict):
    """Summary plot across all experiments."""
    fig, ax = plt.subplots(figsize=(16, 7))

    entries = []
    colors_map = {
        "mean": "#9CA3AF",
        "attentive": "#2563EB",
        "multihead": "#10B981",
        "local_attn": "#F59E0B",
        "decay": "#8B5CF6",
    }

    for key, r in sorted(all_results.items()):
        score = r["avg_nDCG@10"]
        color = "#666"
        for prefix, c in colors_map.items():
            if prefix in key:
                color = c
                break
        entries.append((key, score, color))

    if not entries:
        plt.close("all")
        return

    names, scores, colors = zip(*entries)
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="black", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Average nDCG@10")
    ax.set_title("Hybrid PRISM: All Experiment Results")
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Reference lines
    ax.axvline(0.689, color="blue", linestyle="--", alpha=0.5, label="Mean Pool 2K (0.689)")
    ax.axvline(0.578, color="red", linestyle="--", alpha=0.5, label="Mean Pool 8K (0.578)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "hybrid_all_results.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'hybrid_all_results.png'}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    n_steps: int = 5000,
    batch_size: int = 16,
    lr: float = 3e-4,
    max_lens: list[int] = None,
    sweep_steps: int = 500,
    skip_exp0: bool = False,
    skip_exp1: bool = False,
    skip_exp2: bool = False,
    skip_exp3: bool = False,
    skip_exp4: bool = False,
    exp4_use_attentive: bool = False,
):
    """Run all hybrid experiments per the plan.

    Experiment 0 establishes baselines (PRISM-MeanPool + Transformer) with
    LR sweep and checkpoint evaluation. Experiments 1-3 test pooling variants.
    Experiment 4 tests decay spacing.
    """
    if max_lens is None:
        max_lens = [2048, 8192]

    print("=" * 70)
    print("HYBRID PRISM EXPERIMENTS")
    print(f"Device: {DEVICE}")
    print(f"Steps: {n_steps}, Batch: {batch_size}, LR: {lr}")
    print(f"Max lengths: {max_lens}")
    print("=" * 70)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load data at max needed length
    tokenize_max_len = max(max_lens)
    tasks = load_loco_data(tokenizer, max_len=tokenize_max_len)
    train_pairs = make_training_pairs(tasks)

    all_results = {}
    exp0_results = {}
    model_lrs = {}

    # --- Experiment 0: Baselines with LR sweep + checkpoint eval ---
    if not skip_exp0:
        exp0_results, model_lrs = run_experiment_0(
            tasks, train_pairs, max_lens=max_lens,
            n_steps=n_steps, batch_size=batch_size,
            sweep_steps=sweep_steps,
        )
        all_results.update(exp0_results)
        plot_experiment_0(exp0_results)
    else:
        print("\n  Experiment 0 skipped — using default lr for all models")

    # --- Experiment 1: Attentive Pooling ---
    exp1_improved = False
    if not skip_exp1:
        exp1_lr = model_lrs.get("PRISM-MeanPool", lr)
        exp1_results, exp1_improved = run_experiment_1(
            tasks, train_pairs, n_steps=n_steps, batch_size=batch_size,
            lr=exp1_lr, exp0_results=exp0_results,
        )
        all_results.update(exp1_results)
    else:
        print("\n  Experiment 1 skipped by flag")

    # --- Experiment 2: Multi-Head Pooling (conditional) ---
    if not skip_exp2:
        if exp1_improved:
            print("\n  Experiment 1 showed improvement -> running Experiment 2")
            exp2_results = run_experiment_2(
                tasks, train_pairs, n_steps=n_steps, batch_size=batch_size,
                lr=model_lrs.get("PRISM-MeanPool", lr),
            )
            all_results.update(exp2_results)
        else:
            print("\n  Experiment 1 did NOT show improvement -> skipping Experiment 2")
            print("  (Bottleneck likely in backbone, not pooling)")
    else:
        print("\n  Experiment 2 skipped by flag")

    # --- Experiment 3: Local Attention Hybrid (conditional) ---
    if not skip_exp3:
        run_exp3 = not exp1_improved
        if exp1_improved:
            att_8k = all_results.get("attentive_8192", {}).get("avg_nDCG@10", 0)
            mean_2k = all_results.get("mean_2048", all_results.get("prism_mean_2048", {}))
            mean_2k_score = mean_2k.get("avg_nDCG@10", 0) if isinstance(mean_2k, dict) else 0
            if att_8k < mean_2k_score:
                run_exp3 = True
                print("\n  8K still trails 2K -> running Experiment 3 (local attention)")

        if run_exp3:
            exp3_results = run_experiment_3(
                tasks, train_pairs, n_steps=n_steps, batch_size=batch_size,
                lr=model_lrs.get("PRISM-MeanPool", lr),
            )
            all_results.update(exp3_results)
        else:
            print("\n  Experiment 3 not triggered (attentive pooling closed the gap)")
    else:
        print("\n  Experiment 3 skipped by flag")

    # --- Experiment 4: Decay Spacing (independent) ---
    if not skip_exp4:
        exp4_results = run_experiment_4(
            tasks, train_pairs, n_steps=n_steps, batch_size=batch_size,
            lr=model_lrs.get("PRISM-MeanPool", lr),
            use_attentive=exp4_use_attentive,
        )
        all_results.update(exp4_results)
    else:
        print("\n  Experiment 4 skipped by flag")

    # --- Plots ---
    print("\n  Generating plots...")
    if "attentive_2048" in all_results and ("mean_2048" in all_results or "prism_mean_2048" in all_results):
        # Build combined results for plot_experiment_1
        exp1_plot_data = {}
        for k in ["attentive_2048", "attentive_8192"]:
            if k in all_results:
                exp1_plot_data[k] = all_results[k]
        for ml in [2048, 8192]:
            key = f"mean_{ml}"
            if key not in all_results and f"prism_mean_{ml}" in all_results:
                exp1_plot_data[key] = all_results[f"prism_mean_{ml}"]
            elif key in all_results:
                exp1_plot_data[key] = all_results[key]
        if all(k in exp1_plot_data for k in ["attentive_2048", "mean_2048", "attentive_8192", "mean_8192"]):
            plot_experiment_1(exp1_plot_data)
    if not skip_exp4 and "decay_geometric_2048" in all_results:
        plot_experiment_4({k: v for k, v in all_results.items() if k.startswith("decay_")})
    plot_all_results(all_results)

    # --- Save all results ---
    save_data = {
        "config": {
            "device": DEVICE,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "lr": lr,
            "max_lens": max_lens,
            "model_lrs": {k: v for k, v in model_lrs.items()} if model_lrs else "no sweep",
        },
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("losses", "checkpoints")}
            for k, v in all_results.items()
        },
    }
    # Save checkpoints separately (they contain per-task detail)
    checkpoint_data = {}
    for k, v in all_results.items():
        if "checkpoints" in v and v["checkpoints"]:
            checkpoint_data[k] = {
                str(step): {
                    "avg_nDCG@10": ckpt["avg_nDCG@10"],
                    "loss": ckpt["loss"],
                }
                for step, ckpt in v["checkpoints"].items()
            }
    if checkpoint_data:
        save_data["checkpoints"] = checkpoint_data

    with open(RESULTS_DIR / "hybrid_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'hybrid_results.json'}")

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<35} {'Avg nDCG@10':>12} {'Loss':>8} {'Time':>8}")
    print("  " + "-" * 67)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {key:<35} {r['avg_nDCG@10']:>12.4f} "
              f"{r['final_loss']:>8.4f} {r['train_time']:>7.0f}s")

    # Reference baselines
    print("\n  Published baselines:")
    for name, scores in PUBLISHED_BASELINES.items():
        avg = float(np.mean(list(scores.values())))
        print(f"  {name:<35} {avg:>12.3f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid PRISM Experiments")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-lens", type=int, nargs="+", default=[2048, 8192],
                        help="Max sequence lengths for Exp 0 baselines")
    parser.add_argument("--sweep-steps", type=int, default=500,
                        help="Training steps per LR candidate in sweep")
    parser.add_argument("--skip-exp0", action="store_true",
                        help="Skip Experiment 0 (baselines + LR sweep)")
    parser.add_argument("--skip-exp1", action="store_true",
                        help="Skip Experiment 1 (attentive pooling)")
    parser.add_argument("--skip-exp2", action="store_true",
                        help="Skip Experiment 2 (multi-head pooling)")
    parser.add_argument("--skip-exp3", action="store_true",
                        help="Skip Experiment 3 (local attention hybrid)")
    parser.add_argument("--skip-exp4", action="store_true",
                        help="Skip Experiment 4 (decay spacing ablation)")
    parser.add_argument("--exp4-attentive-pool", action="store_true",
                        help="Use attentive pooling for Experiment 4 (default: mean)")
    args = parser.parse_args()

    main(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_lens=args.max_lens,
        sweep_steps=args.sweep_steps,
        skip_exp0=args.skip_exp0,
        skip_exp1=args.skip_exp1,
        skip_exp2=args.skip_exp2,
        skip_exp3=args.skip_exp3,
        skip_exp4=args.skip_exp4,
        exp4_use_attentive=args.exp4_attentive_pool,
    )
