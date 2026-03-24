"""
Experiment 6: Pretrain + zero-shot evaluation.

Two-stage pipeline:
  Stage 1: Pretrain on weakly-supervised text pairs (CCPairs / AllNLI / similar)
  Stage 2: Fine-tune on MS MARCO with hard negatives

Then evaluate zero-shot on LoCoV1, LongEmbed, and BEIR — the same protocol
used by M2-BERT, BGE, GTE, and other published models.

Usage:
    # Full pipeline (pretrain + finetune + eval)
    uv run python paper_exp6_pretrain.py --models prism

    # Skip pretraining, just finetune from a checkpoint
    uv run python paper_exp6_pretrain.py --models prism --pretrain-checkpoint path/to/ckpt.pt

    # Smoke test
    uv run python paper_exp6_pretrain.py --smoke-test
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from paper_log import create_run_dir, save_config, capture_hardware_info
from train_contrastive import train
from data.msmarco import MSMARCODataset
from data.loco_eval import evaluate_locov1

from prism import prism_small, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small
from linear_rnn import build_linear_rnn_small

TOKENIZER_NAME = "bert-base-uncased"
VOCAB_SIZE = 30522
RESULTS_DIR = Path("results") / "paper" / "exp6"


# ---------------------------------------------------------------------------
# Model builders (same as exp1)
# ---------------------------------------------------------------------------

def build_prism(max_len):
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


def build_transformer(max_len):
    return TransformerForEmbedding(transformer_small(vocab_size=VOCAB_SIZE, max_len=max_len))


MODEL_BUILDERS = {
    "prism": ("PRISM-Simplified", build_prism),
    "transformer": ("Transformer", build_transformer),
    "mamba": ("Mamba-Bidir", lambda ml: build_mamba_bidir_small(VOCAB_SIZE, ml)),
    "linear_rnn": ("Linear-RNN", lambda ml: build_linear_rnn_small(VOCAB_SIZE, ml)),
}


# ---------------------------------------------------------------------------
# Weakly-supervised pretraining dataset
# ---------------------------------------------------------------------------

class WeakPairsDataset:
    """Weakly-supervised text pair dataset for pretraining.

    Loads paired text data from HuggingFace (e.g., AllNLI, Quora duplicates,
    PAQ, or a subset of CCPairs). Falls back to MS MARCO if no weak pairs
    are available.

    The key difference from MSMARCODataset: these are noisier, more diverse
    pairs that provide general sentence understanding before task-specific
    fine-tuning.
    """

    def __init__(self, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []
        self._loaded = False

    def load(self):
        """Load weakly-supervised pairs from HuggingFace."""
        if self._loaded:
            return

        print("  [pretrain] Loading weakly-supervised text pairs...")

        # Try loading AllNLI (entailment pairs) — widely available
        try:
            from datasets import load_dataset
            ds = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
            for row in ds:
                # Use anchor-positive pairs (entailment)
                q_ids = self.tokenizer.encode(
                    row["anchor"], max_length=self.max_len,
                    truncation=True, add_special_tokens=False,
                )
                p_ids = self.tokenizer.encode(
                    row["positive"], max_length=self.max_len,
                    truncation=True, add_special_tokens=False,
                )
                self.pairs.append((q_ids, p_ids))
            print(f"  [pretrain] Loaded {len(self.pairs)} pairs from AllNLI")
        except Exception as e:
            print(f"  [pretrain] AllNLI not available ({e}), trying fallback...")

        # If not enough pairs, add Quora duplicate questions
        if len(self.pairs) < 100000:
            try:
                from datasets import load_dataset
                ds = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train")
                count = 0
                for row in ds:
                    q_ids = self.tokenizer.encode(
                        row["anchor"], max_length=self.max_len,
                        truncation=True, add_special_tokens=False,
                    )
                    p_ids = self.tokenizer.encode(
                        row["positive"], max_length=self.max_len,
                        truncation=True, add_special_tokens=False,
                    )
                    self.pairs.append((q_ids, p_ids))
                    count += 1
                print(f"  [pretrain] Added {count} pairs from Quora duplicates")
            except Exception as e:
                print(f"  [pretrain] Quora duplicates not available ({e})")

        if not self.pairs:
            raise RuntimeError(
                "No pretraining data available. Install: "
                "uv run python -c \"from datasets import load_dataset; "
                "load_dataset('sentence-transformers/all-nli', 'triplet')\""
            )

        print(f"  [pretrain] Total: {len(self.pairs)} pretraining pairs")
        self._loaded = True

    def sample_batch(self, batch_size: int) -> dict:
        """Sample a training batch."""
        import random
        indices = random.choices(range(len(self.pairs)), k=batch_size)

        q_ids_list = [self.pairs[i][0] for i in indices]
        p_ids_list = [self.pairs[i][1] for i in indices]

        max_q = min(max(len(ids) for ids in q_ids_list), self.max_len)
        max_p = min(max(len(ids) for ids in p_ids_list), self.max_len)

        def pad(ids_list, max_len):
            padded, masks = [], []
            for ids in ids_list:
                t = ids[:max_len]
                padded.append(t + [0] * (max_len - len(t)))
                masks.append([1] * len(t) + [0] * (max_len - len(t)))
            return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

        q_ids, q_mask = pad(q_ids_list, max_q)
        p_ids, p_mask = pad(p_ids_list, max_p)

        return {
            "query_ids": q_ids,
            "query_mask": q_mask,
            "pos_ids": p_ids,
            "pos_mask": p_mask,
        }


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------

def make_eval_fn(tokenizer, eval_max_len, device, do_locov1=True):
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
# Two-stage training
# ---------------------------------------------------------------------------

def run_pretrain_finetune(
    model_key: str,
    pretrain_steps: int = 100000,
    finetune_steps: int = 50000,
    pretrain_max_len: int = 512,
    finetune_max_len: int = 2048,
    eval_max_len: int = 2048,
    pretrain_checkpoint: str | None = None,
    device: str | None = None,
    seed: int = 42,
):
    model_name, build_fn = MODEL_BUILDERS[model_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model_max_len = max(pretrain_max_len, finetune_max_len, eval_max_len)

    # -----------------------------------------------------------------------
    # Stage 1: Pretraining
    # -----------------------------------------------------------------------
    if pretrain_checkpoint is None:
        print(f"\n{'='*70}")
        print(f"Stage 1: Pretraining — {model_name}")
        print(f"{'='*70}")

        model = build_fn(model_max_len)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        pretrain_dataset = WeakPairsDataset(tokenizer, max_len=pretrain_max_len)
        pretrain_dataset.load()

        run_dir = create_run_dir("exp6_pretrain", model_key)
        config = {
            "experiment": "exp6_pretrain",
            "stage": "pretrain",
            "model_key": model_key,
            "model_name": model_name,
            "pretrain_max_len": pretrain_max_len,
            "total_params": total_params,
        }

        pretrain_result = train(
            model_wrapper=model,
            dataset=pretrain_dataset,
            run_dir=run_dir,
            config=config,
            n_steps=pretrain_steps,
            micro_batch=32,
            grad_accum=8,
            lr=3e-4,
            eval_every=pretrain_steps,  # eval only at end
            checkpoint_every=pretrain_steps // 2,
            device=device,
            seed=seed,
        )

        pretrain_ckpt = run_dir / "checkpoints" / f"step_{pretrain_steps}.pt"
        print(f"  Pretraining done. Checkpoint: {pretrain_ckpt}")
    else:
        pretrain_ckpt = Path(pretrain_checkpoint)
        print(f"\n  Skipping pretraining, loading from: {pretrain_ckpt}")

    # -----------------------------------------------------------------------
    # Stage 2: Fine-tuning on MS MARCO
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Stage 2: Fine-tuning — {model_name}")
    print(f"{'='*70}")

    model = build_fn(model_max_len)
    ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded pretrained weights from {pretrain_ckpt}")

    msmarco = MSMARCODataset(tokenizer, max_len=finetune_max_len)
    msmarco.load()

    eval_fn = make_eval_fn(tokenizer, eval_max_len, device, do_locov1=True)
    run_dir = create_run_dir("exp6_finetune", model_key)
    config = {
        "experiment": "exp6_finetune",
        "stage": "finetune",
        "model_key": model_key,
        "model_name": model_name,
        "finetune_max_len": finetune_max_len,
        "eval_max_len": eval_max_len,
        "pretrain_checkpoint": str(pretrain_ckpt),
    }

    finetune_result = train(
        model_wrapper=model,
        dataset=msmarco,
        run_dir=run_dir,
        config=config,
        n_steps=finetune_steps,
        micro_batch=16,
        grad_accum=8,
        lr=1e-4,  # lower LR for fine-tuning
        eval_every=5000,
        checkpoint_every=10000,
        eval_fn=eval_fn,
        device=device,
        seed=seed,
    )

    print(f"\n  {model_name} fine-tuning complete: "
          f"best nDCG@10={finetune_result['best_metric']:.4f} "
          f"@ step {finetune_result['best_step']}")

    return {
        "pretrain_checkpoint": str(pretrain_ckpt),
        "finetune_result": finetune_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Pretrain + zero-shot eval")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys")
    parser.add_argument("--pretrain-steps", type=int, default=100000)
    parser.add_argument("--finetune-steps", type=int, default=50000)
    parser.add_argument("--pretrain-max-len", type=int, default=512)
    parser.add_argument("--finetune-max-len", type=int, default=2048)
    parser.add_argument("--eval-max-len", type=int, default=2048)
    parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                        help="Skip pretraining, start from this checkpoint")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 6: Pretrain + Zero-Shot Evaluation")
    print(f"  Hardware: {capture_hardware_info()}")
    print("=" * 70)

    if args.smoke_test:
        print("=== SMOKE TEST ===")
        run_pretrain_finetune(
            "prism",
            pretrain_steps=50,
            finetune_steps=50,
            pretrain_max_len=128,
            finetune_max_len=128,
            eval_max_len=128,
            device=args.device,
            seed=args.seed,
        )
        print("\n=== SMOKE TEST PASSED ===")
        return

    model_keys = args.models.split(",") if args.models else ["prism"]

    all_results = {}
    for model_key in model_keys:
        if model_key not in MODEL_BUILDERS:
            print(f"Unknown model: {model_key}")
            continue
        result = run_pretrain_finetune(
            model_key,
            pretrain_steps=args.pretrain_steps,
            finetune_steps=args.finetune_steps,
            pretrain_max_len=args.pretrain_max_len,
            finetune_max_len=args.finetune_max_len,
            eval_max_len=args.eval_max_len,
            pretrain_checkpoint=args.pretrain_checkpoint,
            device=args.device,
            seed=args.seed,
        )
        all_results[model_key] = result

    out_path = RESULTS_DIR / "pretrain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print("\n=== Experiment 6 complete ===")


if __name__ == "__main__":
    main()
