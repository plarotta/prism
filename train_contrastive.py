"""
Unified contrastive training loop for paper experiments.

Model-agnostic: works with any model that implements:
  - .forward(query_ids, query_mask, pos_ids, pos_mask) -> {"loss": Tensor}
  - .encode(input_ids, attention_mask) -> Tensor  (for eval)

Handles: AdamW + cosine schedule with warmup, gradient accumulation,
gradient clipping, periodic eval via callback, checkpointing, and
structured JSONL logging.

Usage:
    from train_contrastive import train
    train(model, dataset, run_dir, config, n_steps=50000, eval_fn=eval_fn)
"""

import math
import random
import time

import numpy as np
import torch

from paper_log import (
    save_config,
    log_step,
    save_checkpoint,
    save_eval_results,
    save_final_metrics,
)


def _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    model_wrapper,
    dataset,
    run_dir,
    config: dict,
    n_steps: int = 50000,
    micro_batch: int = 16,
    grad_accum: int = 8,
    lr: float = 3e-4,
    warmup_fraction: float = 0.1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_every: int = 5000,
    checkpoint_every: int = 5000,
    log_every: int = 100,
    eval_fn=None,
    device: str | None = None,
    seed: int = 42,
):
    """Train a contrastive embedding model with structured logging.

    Args:
        model_wrapper: model with .forward() and .encode()
        dataset: object with .sample_batch(batch_size) -> dict of tensors
        run_dir: Path to results directory for this run
        config: dict of experiment config (saved to config.json)
        n_steps: total optimizer steps
        micro_batch: batch size per forward pass
        grad_accum: number of micro-batches per optimizer step
        lr: peak learning rate
        warmup_fraction: fraction of steps for linear warmup
        weight_decay: AdamW weight decay
        grad_clip: max gradient norm
        eval_every: run eval callback every N steps
        checkpoint_every: save checkpoint every N steps
        log_every: log metrics every N steps
        eval_fn: callable(model, step) -> dict, or None
        device: torch device string (auto-detected if None)
        seed: random seed

    Returns:
        dict with best_step, best_metric, train_losses
    """
    if device is None:
        device = _select_device()

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Save config
    config["training"] = {
        "n_steps": n_steps,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "effective_batch": micro_batch * grad_accum,
        "lr": lr,
        "warmup_fraction": warmup_fraction,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "seed": seed,
    }
    save_config(run_dir, config)

    # Setup
    model_wrapper = model_wrapper.to(device)
    total_params = sum(p.numel() for p in model_wrapper.parameters())
    trainable_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), lr=lr, weight_decay=weight_decay,
    )
    warmup_steps = int(n_steps * warmup_fraction)
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_steps)

    # Training state
    best_metric = -1.0
    best_step = 0
    all_losses = []
    t0 = time.perf_counter()

    print(f"  Training for {n_steps} steps "
          f"(micro_batch={micro_batch}, accum={grad_accum}, "
          f"effective_batch={micro_batch * grad_accum})")
    print(f"  LR: {lr} with {warmup_steps} warmup steps, cosine decay")
    print(f"  Eval every {eval_every} steps, checkpoint every {checkpoint_every} steps")

    model_wrapper.train()

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            batch = dataset.sample_batch(micro_batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model_wrapper(
                batch["query_ids"], batch["query_mask"],
                batch["pos_ids"], batch["pos_mask"],
            )
            loss = result["loss"] / grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_wrapper.parameters(), grad_clip,
        )
        optimizer.step()
        scheduler.step()

        all_losses.append(step_loss)

        # --- Logging ---
        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            recent_loss = np.mean(all_losses[-log_every:])
            gpu_mem = (
                torch.cuda.max_memory_allocated() // (1024**2)
                if torch.cuda.is_available() else 0
            )
            metrics = {
                "loss": round(float(recent_loss), 5),
                "lr": round(scheduler.get_last_lr()[0], 8),
                "grad_norm": round(float(grad_norm), 4),
                "gpu_mem_mb": gpu_mem,
                "elapsed_s": round(elapsed, 1),
            }
            log_step(run_dir, step, metrics)
            print(f"  [step {step}/{n_steps}]  loss={recent_loss:.4f}  "
                  f"lr={metrics['lr']:.2e}  grad_norm={float(grad_norm):.2f}  "
                  f"({elapsed:.0f}s)")

        # --- Checkpoint ---
        if step % checkpoint_every == 0:
            save_checkpoint(run_dir, model_wrapper, optimizer, step)

        # --- Eval ---
        if eval_fn is not None and step % eval_every == 0:
            model_wrapper.eval()
            print(f"  [step {step}] Running evaluation...")
            eval_results = eval_fn(model_wrapper, step)
            save_eval_results(run_dir, step, "eval", eval_results)

            metric_val = eval_results.get("locov1_avg_ndcg@10",
                         eval_results.get("avg_ndcg@10", 0))
            if metric_val > best_metric:
                best_metric = metric_val
                best_step = step
                # Save best checkpoint separately
                save_checkpoint(run_dir, model_wrapper, optimizer, step)

            print(f"  [step {step}] Eval: "
                  f"nDCG@10={metric_val:.4f}  "
                  f"(best={best_metric:.4f} @ step {best_step})")

            model_wrapper.train()

    # --- Final ---
    total_time = time.perf_counter() - t0
    print(f"  Training complete in {total_time:.0f}s")

    final = {
        "best_step": best_step,
        "best_metric": best_metric,
        "final_loss": round(float(np.mean(all_losses[-100:])), 5),
        "total_train_time_s": round(total_time, 1),
        "total_params": total_params,
        "n_steps_completed": n_steps,
    }
    save_final_metrics(run_dir, final)

    return final
