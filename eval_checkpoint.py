"""
Offline checkpoint evaluator.

Re-evaluates an already-trained run's checkpoint without retraining. Built to
recover quality numbers for runs whose eval callback produced empty stubs
(e.g. Exp 1a/1b, where MS MARCO dev eval was not yet wired up).

Reads {run_dir}/config.json to rebuild the exact model, loads the requested
checkpoint, runs MS MARCO dev retrieval (and optionally LoCoV1), and writes
the results back to {run_dir}/eval/step_{step}_eval.json (overwriting the stub).

Usage:
    # Latest checkpoint in a run dir
    uv run python eval_checkpoint.py --run-dir results/paper/exp1_1a/prism_run0

    # A specific step, plus LoCoV1
    uv run python eval_checkpoint.py \
        --run-dir results/paper/exp1_1a/transformer_run0 --step 50000 --locov1

    # Every model under a sub-experiment
    uv run python eval_checkpoint.py --exp-dir results/paper/exp1_1a
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from paper_exp1_controlled import MODEL_BUILDERS, TOKENIZER_NAME, VOCAB_SIZE
from data.msmarco import MSMARCODataset, evaluate_msmarco_dev
from data.loco_eval import evaluate_locov1

# Cache datasets across multiple runs in one invocation (keyed by max_len).
_DATASET_CACHE: dict[int, MSMARCODataset] = {}


def _get_dataset(tokenizer, max_len: int) -> MSMARCODataset:
    if max_len not in _DATASET_CACHE:
        ds = MSMARCODataset(tokenizer, max_len=max_len)
        ds.load()
        _DATASET_CACHE[max_len] = ds
    return _DATASET_CACHE[max_len]


def _latest_step(run_dir: Path) -> int:
    ckpts = list((run_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir / 'checkpoints'}")
    return max(int(p.stem.split("_")[1]) for p in ckpts)


def _load_state(model, ckpt_path: Path):
    """Load a checkpoint, tolerating a torch.compile '_orig_mod.' prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return ckpt.get("step")


def eval_run(
    run_dir: Path,
    step: int | None,
    device: str,
    do_locov1: bool,
    tokenizer,
    batch_size: int = 64,
    max_queries: int | None = 2000,
) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    model_key = config["model_key"]
    model_max_len = config["model_config"]["max_len"]
    train_max_len = config.get("train_max_len", model_max_len)
    eval_max_len = config.get("eval_max_len", model_max_len)

    if model_key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_key {model_key!r} in {run_dir}")
    model_name, build_fn = MODEL_BUILDERS[model_key]

    if step is None:
        step = _latest_step(run_dir)

    print(f"\n{'='*70}\n{run_dir}\n  model={model_name}  step={step}  "
          f"train_len={train_max_len}  eval_len={eval_max_len}\n{'='*70}")

    model = build_fn(model_max_len).to(device)
    ckpt_path = run_dir / "checkpoints" / f"step_{step}.pt"
    _load_state(model, ckpt_path)
    model.eval()

    # MS MARCO dev tokenizes at the training length the cache was built with.
    dataset = _get_dataset(tokenizer, train_max_len)

    results: dict = {}
    dev = evaluate_msmarco_dev(
        model, dataset, max_len=eval_max_len, batch_size=batch_size,
        device=device, max_queries=max_queries,
    )
    results.update(dev)
    print(f"  MS MARCO dev: MRR@10={dev['msmarco_dev_mrr@10']:.4f}  "
          f"R@10={dev['msmarco_dev_recall@10']:.4f}  "
          f"R@100={dev['msmarco_dev_recall@100']:.4f}  "
          f"(n_q={dev['n_queries']}, pool={dev['n_pool']})")

    if do_locov1:
        loco = evaluate_locov1(
            model, tokenizer, max_len=eval_max_len,
            batch_size=32, device=device,
        )
        results["locov1_avg_ndcg@10"] = loco["avg_ndcg@10"]
        results["locov1_per_task"] = loco["per_task"]
        results["locov1_eval_time_s"] = loco["eval_time_s"]
        print(f"  LoCoV1: avg nDCG@10={loco['avg_ndcg@10']:.4f}")

    # Write back, overwriting the (possibly empty) eval stub for this step.
    out_path = run_dir / "eval" / f"step_{step}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"step": step, "benchmark": "eval", **results}, indent=2,
    ))
    print(f"  -> wrote {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline checkpoint evaluator")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str,
                   help="A single run dir (contains config.json + checkpoints/)")
    g.add_argument("--exp-dir", type=str,
                   help="A sub-experiment dir; evaluates every *_run* inside it")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: latest available)")
    parser.add_argument("--locov1", action="store_true",
                        help="Also run LoCoV1 zero-shot eval")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=2000,
                        help="Cap dev queries for speed (0 = all)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    max_queries = None if args.max_queries == 0 else args.max_queries

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = sorted(
            p for p in Path(args.exp_dir).iterdir()
            if p.is_dir() and (p / "config.json").exists()
        )
        if not run_dirs:
            parser.error(f"No run dirs with config.json under {args.exp_dir}")

    summary = {}
    for run_dir in run_dirs:
        try:
            res = eval_run(
                run_dir, args.step, device, args.locov1, tokenizer,
                batch_size=args.batch_size, max_queries=max_queries,
            )
            summary[run_dir.name] = res.get("msmarco_dev_mrr@10")
        except Exception as ex:
            print(f"  !!! {run_dir} FAILED: {ex}")
            summary[run_dir.name] = None

    if len(summary) > 1:
        print(f"\n{'='*70}\nSummary (MS MARCO dev MRR@10)\n{'='*70}")
        for name, mrr in summary.items():
            print(f"  {name:<28} {mrr if mrr is None else f'{mrr:.4f}'}")


if __name__ == "__main__":
    main()
