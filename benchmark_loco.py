"""
LoCoV1 Benchmark: PRISM-Simplified vs Transformer on Long-Context Retrieval

Evaluates on the LoCoV1 12-task long-context retrieval benchmark from
Hazy Research (Stanford). Compares against published M2-BERT results.

Dataset: hazyresearch/LoCoV1-Documents + hazyresearch/LoCoV1-Queries on HuggingFace
Metric: nDCG@10 per task, averaged across all 12 tasks
Reference: Saad-Falcon et al., arXiv:2402.07440

Phase 1: Train from scratch on LoCoV1 query-document pairs (no pretraining).
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer

from prism import prism_small, PRISMForEmbedding
from baseline_transformer import transformer_small, TransformerForEmbedding
from benchmark_ablations import MeanPooling, NoInterference

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
TOKENIZER_NAME = "bert-base-uncased"
REAL_VOCAB_SIZE = 30522
RESULTS_DIR = Path("results") / "loco"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The 12 LoCoV1 tasks
LOCO_TASKS = [
    "summ_screen_fd",
    "gov_report",
    "qmsum",
    "qasper_title",
    "qasper_abstract",
    "multifieldqa",
    "2wikimqa",
    "passage_retrieval",
    "courtlistener_Plain_Text",
    "courtlistener_HTML",
    "legal_case_reports",
    "stackoverflow",
]

# Published M2-BERT nDCG@10 baselines (from arXiv:2402.07440 Table 1)
# These are approximate values read from the paper for comparison
PUBLISHED_BASELINES = {
    "BM25": {
        "summ_screen_fd": 0.738, "gov_report": 0.432, "qmsum": 0.399,
        "qasper_title": 0.488, "qasper_abstract": 0.862, "multifieldqa": 0.469,
        "2wikimqa": 0.617, "passage_retrieval": 0.971, "courtlistener_Plain_Text": 0.124,
        "courtlistener_HTML": 0.102, "legal_case_reports": 0.125, "stackoverflow": 0.504,
    },
    "M2-BERT-80M-2K": {
        "summ_screen_fd": 0.640, "gov_report": 0.488, "qmsum": 0.314,
        "qasper_title": 0.440, "qasper_abstract": 0.846, "multifieldqa": 0.250,
        "2wikimqa": 0.500, "passage_retrieval": 0.717, "courtlistener_Plain_Text": 0.217,
        "courtlistener_HTML": 0.202, "legal_case_reports": 0.169, "stackoverflow": 0.595,
    },
    "M2-BERT-80M-8K": {
        "summ_screen_fd": 0.700, "gov_report": 0.550, "qmsum": 0.320,
        "qasper_title": 0.510, "qasper_abstract": 0.870, "multifieldqa": 0.330,
        "2wikimqa": 0.570, "passage_retrieval": 0.800, "courtlistener_Plain_Text": 0.301,
        "courtlistener_HTML": 0.276, "legal_case_reports": 0.200, "stackoverflow": 0.640,
    },
}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_simplified_prism(vocab_size=REAL_VOCAB_SIZE, max_len=8192, **kwargs):
    """Simplified PRISM: multi-channel recurrence + mean pooling, no interference."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


def build_transformer(vocab_size=REAL_VOCAB_SIZE, max_len=8192, **kwargs):
    """Parameter-matched Transformer baseline."""
    return transformer_small(vocab_size=vocab_size, max_len=max_len, **kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_loco_data(tokenizer, max_len=2048):
    """Load LoCoV1 queries and documents from HuggingFace.

    Returns:
        tasks: dict[task_name -> {
            "queries": {qid: {"text": str, "token_ids": list[int]}},
            "documents": {pid: {"text": str, "token_ids": list[int]}},
            "qrels": {qid: list[pid]},
        }]
    """
    print("Loading LoCoV1 datasets from HuggingFace...")
    docs_ds = load_dataset("hazyresearch/LoCoV1-Documents", split="test")
    queries_ds = load_dataset("hazyresearch/LoCoV1-Queries", split="test")

    print(f"  Documents: {len(docs_ds)} rows")
    print(f"  Queries: {len(queries_ds)} rows")

    # Organize by task
    tasks = {}
    for task_name in LOCO_TASKS:
        tasks[task_name] = {
            "queries": {},
            "documents": {},
            "qrels": {},
        }

    # Load documents
    print("  Tokenizing documents...")
    for row in docs_ds:
        task = row["dataset"]
        if task not in tasks:
            continue
        pid = row["pid"]
        text = row["passage"]
        token_ids = tokenizer.encode(text, max_length=max_len, truncation=True,
                                     add_special_tokens=False)
        tasks[task]["documents"][pid] = {
            "text": text,
            "token_ids": token_ids,
        }

    # Load queries
    print("  Tokenizing queries...")
    for row in queries_ds:
        task = row["dataset"]
        if task not in tasks:
            continue
        qid = row["qid"]
        query_text = row["query"]
        answer_pids = row["answer_pids"]
        token_ids = tokenizer.encode(query_text, max_length=max_len, truncation=True,
                                     add_special_tokens=False)
        tasks[task]["queries"][qid] = {
            "text": query_text,
            "token_ids": token_ids,
        }
        tasks[task]["qrels"][qid] = answer_pids

    # Print summary
    print("\n  Task summary:")
    print(f"  {'Task':<30} {'Queries':>8} {'Docs':>8} {'Avg Q Len':>10} {'Avg D Len':>10}")
    print("  " + "-" * 70)
    for task_name in LOCO_TASKS:
        t = tasks[task_name]
        n_q = len(t["queries"])
        n_d = len(t["documents"])
        if n_q > 0:
            avg_q = np.mean([len(q["token_ids"]) for q in t["queries"].values()])
        else:
            avg_q = 0
        if n_d > 0:
            avg_d = np.mean([len(d["token_ids"]) for d in t["documents"].values()])
        else:
            avg_d = 0
        print(f"  {task_name:<30} {n_q:>8} {n_d:>8} {avg_q:>10.0f} {avg_d:>10.0f}")

    return tasks


def profile_document_lengths(tasks, tokenizer_name="bert-base-uncased"):
    """Profile document length distributions and save a plot."""
    print("\nProfiling document length distributions...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(f"LoCoV1 Document Length Distributions ({tokenizer_name} tokens)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    length_stats = {}

    for i, task_name in enumerate(LOCO_TASKS):
        ax = axes[i]
        docs = tasks[task_name]["documents"]
        if not docs:
            ax.set_title(task_name, fontsize=9)
            ax.text(0.5, 0.5, "No documents", ha="center", va="center")
            continue

        lengths = [len(d["token_ids"]) for d in docs.values()]
        query_lengths = [len(q["token_ids"]) for q in tasks[task_name]["queries"].values()]

        stats = {
            "n_docs": len(lengths),
            "doc_mean": float(np.mean(lengths)),
            "doc_median": float(np.median(lengths)),
            "doc_p95": float(np.percentile(lengths, 95)),
            "doc_max": int(max(lengths)),
            "doc_min": int(min(lengths)),
            "n_queries": len(query_lengths),
            "query_mean": float(np.mean(query_lengths)) if query_lengths else 0,
        }
        length_stats[task_name] = stats

        ax.hist(lengths, bins=50, color="#2563EB", alpha=0.7, label="Documents")
        if query_lengths:
            ax.hist(query_lengths, bins=50, color="#DC2626", alpha=0.5, label="Queries")
        ax.axvline(2048, color="green", linestyle="--", alpha=0.7, label="2K")
        ax.axvline(8192, color="orange", linestyle="--", alpha=0.7, label="8K")
        ax.set_title(f"{task_name}\nmed={stats['doc_median']:.0f}, max={stats['doc_max']}",
                     fontsize=9)
        ax.set_xlabel("Token length")
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "loco_length_distributions.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'loco_length_distributions.png'}")
    plt.close("all")

    # Save stats
    with open(RESULTS_DIR / "loco_length_stats.json", "w") as f:
        json.dump(length_stats, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'loco_length_stats.json'}")

    return length_stats


# ---------------------------------------------------------------------------
# nDCG@10 computation
# ---------------------------------------------------------------------------

def compute_ndcg(ranked_pids: list[str], relevant_pids: set[str], k: int = 10) -> float:
    """Compute nDCG@k for a single query.

    Args:
        ranked_pids: list of document IDs in ranked order (most similar first)
        relevant_pids: set of relevant document IDs (binary relevance)
        k: cutoff
    """
    # DCG@k
    dcg = 0.0
    for i, pid in enumerate(ranked_pids[:k]):
        rel = 1.0 if pid in relevant_pids else 0.0
        dcg += rel / math.log2(i + 2)  # +2 because positions are 1-indexed

    # Ideal DCG@k (all relevant docs at the top)
    n_rel = min(len(relevant_pids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_task_ndcg(
    model_wrapper,
    task_data: dict,
    max_len: int,
    batch_size: int = 32,
    device: str = DEVICE,
) -> dict:
    """Evaluate nDCG@k on a single LoCoV1 task.

    Encodes all queries and documents independently, computes cosine similarity,
    ranks, and computes nDCG@{1,3,5,10}.
    """
    model_wrapper.eval()
    queries = task_data["queries"]
    documents = task_data["documents"]
    qrels = task_data["qrels"]

    if not queries or not documents:
        return None

    # Encode documents
    doc_pids = list(documents.keys())
    doc_token_ids = [documents[pid]["token_ids"] for pid in doc_pids]
    doc_embs = _encode_batch(model_wrapper, doc_token_ids, max_len, batch_size, device)

    # Encode queries
    query_qids = list(queries.keys())
    query_token_ids = [queries[qid]["token_ids"] for qid in query_qids]
    query_embs = _encode_batch(model_wrapper, query_token_ids, max_len, batch_size, device)

    # Compute cosine similarities: (n_queries, n_docs)
    sim_matrix = torch.matmul(query_embs, doc_embs.T)

    # Compute nDCG per query
    ndcg_scores = {k: [] for k in [1, 3, 5, 10]}
    for i, qid in enumerate(query_qids):
        relevant_pids = set(qrels.get(qid, []))
        if not relevant_pids:
            continue
        # Rank documents by similarity
        sorted_indices = sim_matrix[i].argsort(descending=True).tolist()
        ranked_pids = [doc_pids[idx] for idx in sorted_indices]

        for k in ndcg_scores:
            ndcg_scores[k].append(compute_ndcg(ranked_pids, relevant_pids, k))

    results = {}
    for k in ndcg_scores:
        if ndcg_scores[k]:
            results[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k]))
        else:
            results[f"nDCG@{k}"] = 0.0
    results["n_queries"] = len(query_qids)
    results["n_docs"] = len(doc_pids)
    return results


@torch.no_grad()
def _encode_batch(
    model_wrapper,
    token_ids_list: list[list[int]],
    max_len: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Encode a list of token ID sequences into normalized embeddings."""
    all_embs = []
    for i in range(0, len(token_ids_list), batch_size):
        batch_ids_list = token_ids_list[i:i + batch_size]
        ids, masks = _collate_ids(batch_ids_list, max_len)
        ids, masks = ids.to(device), masks.to(device)
        emb = model_wrapper.encode(ids, masks)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def _collate_ids(token_ids_list: list[list[int]], max_len: int, pad_id: int = 0):
    """Pad token ID lists to uniform length tensors."""
    ids_out, masks_out = [], []
    for ids in token_ids_list:
        t = ids[:max_len]
        padded = t + [pad_id] * (max_len - len(t))
        mask = [1] * len(t) + [0] * (max_len - len(t))
        ids_out.append(padded)
        masks_out.append(mask)
    return torch.tensor(ids_out, dtype=torch.long), torch.tensor(masks_out, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_training_pairs(tasks: dict) -> list[tuple[list[int], list[int]]]:
    """Create contrastive training pairs from LoCoV1 query-document relevance.

    Each (query, relevant_document) pair becomes a positive pair.
    In-batch negatives provide the negative signal during training.
    """
    pairs = []
    for _task_name, task_data in tasks.items():
        queries = task_data["queries"]
        documents = task_data["documents"]
        qrels = task_data["qrels"]

        for qid, relevant_pids in qrels.items():
            if qid not in queries:
                continue
            q_ids = queries[qid]["token_ids"]
            for pid in relevant_pids:
                if pid not in documents:
                    continue
                d_ids = documents[pid]["token_ids"]
                if len(q_ids) >= 4 and len(d_ids) >= 4:
                    pairs.append((q_ids, d_ids))

    random.shuffle(pairs)
    print(f"  Created {len(pairs)} training pairs from {len(tasks)} tasks")
    return pairs


def train_contrastive(
    model_wrapper,
    train_pairs: list[tuple[list[int], list[int]]],
    n_steps: int,
    model_name: str,
    max_len: int = 2048,
    micro_batch: int = 16,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    log_interval: int = 100,
):
    """Train with InfoNCE on query-document pairs.

    Supports gradient accumulation: each optimizer step accumulates gradients
    over `grad_accum_steps` micro-batches. The effective batch size for each
    micro-batch's InfoNCE loss is `micro_batch` (in-batch negatives come from
    the micro-batch). Gradient accumulation averages the loss across micro-batches.

    Args:
        micro_batch: number of pairs per forward pass (determines GPU memory)
        grad_accum_steps: number of micro-batches per optimizer step
    """
    effective_batch = micro_batch * grad_accum_steps

    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )

    losses = []
    model_wrapper.train()

    print(f"\nTraining {model_name} ({n_steps} steps, max_len={max_len}, "
          f"micro_batch={micro_batch}, accum={grad_accum_steps}, "
          f"effective_batch={effective_batch})...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad()
        step_loss = 0.0

        for accum_i in range(grad_accum_steps):
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

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")
    return losses, total_time


# ---------------------------------------------------------------------------
# Phase 0: Data inspection
# ---------------------------------------------------------------------------

def run_phase_0(tokenizer, max_len=8192):
    """Phase 0: Download, inspect, and profile LoCoV1."""
    print("\n" + "=" * 70)
    print("PHASE 0: LoCoV1 Data Inspection")
    print("=" * 70)

    tasks = load_loco_data(tokenizer, max_len=max_len)
    profile_document_lengths(tasks)

    # Summarize which tasks are feasible at 2K and 8K
    print("\n  Truncation impact analysis:")
    print(f"  {'Task':<30} {'Docs ≤ 2K':>10} {'Docs ≤ 8K':>10} {'Total':>8}")
    print("  " + "-" * 62)
    for task_name in LOCO_TASKS:
        docs = tasks[task_name]["documents"]
        if not docs:
            continue
        lengths = [len(d["token_ids"]) for d in docs.values()]
        n_le_2k = sum(1 for l in lengths if l <= 2048)
        n_le_8k = sum(1 for l in lengths if l <= 8192)
        total = len(lengths)
        print(f"  {task_name:<30} {n_le_2k:>8} ({100*n_le_2k/total:>4.0f}%)  "
              f"{n_le_8k:>5} ({100*n_le_8k/total:>4.0f}%)  {total:>6}")

    return tasks


# ---------------------------------------------------------------------------
# Phase 1: Train from scratch + evaluate
# ---------------------------------------------------------------------------

def _get_gpu_mem_gb() -> float:
    """Detect available GPU memory in GB."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        return total
    return 0.0


def _get_batch_config(model_name: str, max_len: int, base_batch: int) -> dict:
    """Determine micro-batch size, gradient accumulation, and eval batch size.

    PRISM scales linearly with sequence length — can use larger batches.
    Transformer has O(n^2) memory — needs smaller micro-batches at long sequences
    with gradient accumulation to compensate.

    Adapts to available GPU memory (auto-detected).

    Returns dict with keys: micro_batch, grad_accum_steps, eval_batch
    """
    gpu_mem = _get_gpu_mem_gb()
    is_transformer = "transformer" in model_name.lower()

    # Empirical training memory per sample (MB, fp32, fwd+bwd) from A100 benchmarks.
    # These are conservative estimates including activations + optimizer states.
    # Transformer: O(n^2) attention dominates at long sequences.
    # PRISM: O(n) recurrence, memory grows linearly.
    if is_transformer:
        # Approximate MB per sample for training (fwd+bwd):
        # 512: ~50, 2048: ~1200, 4096: ~5000, 8192: ~20000
        mem_per_sample = {512: 50, 1024: 170, 2048: 1200, 4096: 5000, 8192: 20000}
    else:
        # PRISM is much lighter:
        # 512: ~35, 2048: ~135, 4096: ~270, 8192: ~540
        mem_per_sample = {512: 35, 1024: 70, 2048: 135, 4096: 270, 8192: 540}

    # Interpolate/extrapolate for the target max_len
    lengths = sorted(mem_per_sample.keys())
    if max_len <= lengths[0]:
        per_sample_mb = mem_per_sample[lengths[0]]
    elif max_len >= lengths[-1]:
        per_sample_mb = mem_per_sample[lengths[-1]]
    else:
        # Linear interpolation between bracketing lengths
        for i in range(len(lengths) - 1):
            if lengths[i] <= max_len <= lengths[i + 1]:
                lo, hi = lengths[i], lengths[i + 1]
                frac = (max_len - lo) / (hi - lo)
                per_sample_mb = mem_per_sample[lo] + frac * (mem_per_sample[hi] - mem_per_sample[lo])
                break

    # Reserve ~4 GB for model params + optimizer states + overhead
    available_mb = max(1000, gpu_mem * 1024 - 4000) if gpu_mem > 0 else 8000

    # Max micro-batch that fits in memory
    max_micro = max(1, int(available_mb / per_sample_mb))

    # Desired effective batch (what we'd use if memory were unlimited)
    desired_effective = base_batch

    # micro_batch = min of what fits and what we want
    micro_batch = min(max_micro, desired_effective)

    # Gradient accumulation to reach the desired effective batch
    grad_accum_steps = max(1, desired_effective // micro_batch)

    # Eval is inference-only (~3x cheaper than training), so we can use bigger batches
    eval_batch = min(max_micro * 3, desired_effective * 2)

    return {
        "micro_batch": micro_batch,
        "grad_accum_steps": grad_accum_steps,
        "eval_batch": eval_batch,
        "_gpu_mem_gb": round(gpu_mem, 1),
        "_per_sample_mb": round(per_sample_mb),
        "_max_micro_fit": max_micro,
    }


def run_phase_1(
    tasks: dict,
    max_len: int = 2048,
    n_steps: int = 5000,
    batch_size: int = 16,
    lr: float = 3e-4,
):
    """Phase 1: Train PRISM-Simplified and Transformer from scratch on LoCoV1.

    No pretraining. Tests whether the architecture can learn the retrieval task.
    At long sequence lengths, the Transformer uses gradient accumulation with
    small micro-batches to avoid OOM.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 1: LoCoV1 Evaluation (max_len={max_len}, {n_steps} steps)")
    print("=" * 70)

    # Create training pairs
    train_pairs = make_training_pairs(tasks)

    all_results = {}

    for name, build_encoder, WrapperClass in [
        ("PRISM-Simplified", lambda: build_simplified_prism(max_len=max_len), PRISMForEmbedding),
        ("Transformer", lambda: build_transformer(max_len=max_len), TransformerForEmbedding),
    ]:
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        random.seed(42)

        encoder = build_encoder().to(DEVICE)
        wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)
        n_params = sum(p.numel() for p in wrapper.parameters())
        print(f"  Parameters: {n_params:,}")

        # Get per-model batch config
        bcfg = _get_batch_config(name, max_len, batch_size)
        print(f"  Batch config: micro_batch={bcfg['micro_batch']}, "
              f"grad_accum={bcfg['grad_accum_steps']}, "
              f"effective_batch={bcfg['micro_batch'] * bcfg['grad_accum_steps']}, "
              f"eval_batch={bcfg['eval_batch']}")

        # Train
        losses, train_time = train_contrastive(
            wrapper, train_pairs, n_steps, name,
            max_len=max_len,
            micro_batch=bcfg["micro_batch"],
            grad_accum_steps=bcfg["grad_accum_steps"],
            lr=lr,
        )

        # Evaluate per task
        print(f"\n  Evaluating {name} on all 12 LoCoV1 tasks...")
        task_results = {}
        for task_name in LOCO_TASKS:
            task_data = tasks[task_name]
            if not task_data["queries"] or not task_data["documents"]:
                print(f"    {task_name}: skipped (no data)")
                continue

            try:
                metrics = evaluate_task_ndcg(
                    wrapper, task_data, max_len,
                    batch_size=bcfg["eval_batch"], device=DEVICE,
                )
                if metrics:
                    task_results[task_name] = metrics
                    print(f"    {task_name}: nDCG@10={metrics['nDCG@10']:.4f}  "
                          f"({metrics['n_queries']} queries, {metrics['n_docs']} docs)")
                else:
                    print(f"    {task_name}: no results")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try again with batch=1
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    print(f"    {task_name}: OOM with batch={bcfg['eval_batch']}, retrying batch=1...")
                    try:
                        metrics = evaluate_task_ndcg(
                            wrapper, task_data, max_len,
                            batch_size=1, device=DEVICE,
                        )
                        if metrics:
                            task_results[task_name] = metrics
                            print(f"    {task_name}: nDCG@10={metrics['nDCG@10']:.4f}  "
                                  f"({metrics['n_queries']} queries, {metrics['n_docs']} docs)")
                        else:
                            print(f"    {task_name}: no results")
                    except RuntimeError as e2:
                        print(f"    {task_name}: FAILED even at batch=1 ({e2})")
                        task_results[task_name] = {"failed": True, "error": str(e2)}
                else:
                    print(f"    {task_name}: FAILED ({e})")
                    task_results[task_name] = {"failed": True, "error": str(e)}

        # Average nDCG@10
        valid_ndcg = [r["nDCG@10"] for r in task_results.values()
                      if isinstance(r, dict) and "nDCG@10" in r]
        avg_ndcg = float(np.mean(valid_ndcg)) if valid_ndcg else 0.0

        all_results[name] = {
            "task_results": task_results,
            "avg_nDCG@10": avg_ndcg,
            "train_time": train_time,
            "final_loss": float(np.mean(losses[-100:])) if losses else 0.0,
            "params": n_params,
            "max_len": max_len,
            "n_steps": n_steps,
            "micro_batch": bcfg["micro_batch"],
            "grad_accum_steps": bcfg["grad_accum_steps"],
        }
        print(f"\n  {name} average nDCG@10: {avg_ndcg:.4f}")

        # Free GPU memory before next model
        del wrapper, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_task_comparison(results: dict, max_len: int):
    """Bar chart comparing models per task with published baselines."""
    fig, ax = plt.subplots(figsize=(16, 8))

    task_names = [t for t in LOCO_TASKS
                  if any(t in r.get("task_results", {}) for r in results.values())]
    if not task_names:
        return

    x = np.arange(len(task_names))
    width = 0.15
    colors = {
        "PRISM-Simplified": "#2563EB",
        "Transformer": "#DC2626",
        "BM25": "#9CA3AF",
        "M2-BERT-80M-2K": "#F59E0B",
        "M2-BERT-80M-8K": "#10B981",
    }

    # Plot published baselines
    offset = -2
    for baseline_name, baseline_scores in PUBLISHED_BASELINES.items():
        scores = [baseline_scores.get(t, 0) for t in task_names]
        ax.bar(x + offset * width, scores, width, label=baseline_name,
               color=colors.get(baseline_name, "#666"), alpha=0.6)
        offset += 1

    # Plot our models
    for model_name, model_results in results.items():
        task_results = model_results.get("task_results", {})
        scores = []
        for t in task_names:
            r = task_results.get(t, {})
            scores.append(r.get("nDCG@10", 0) if isinstance(r, dict) else 0)
        ax.bar(x + offset * width, scores, width, label=model_name,
               color=colors.get(model_name, "#666"), alpha=0.9)
        offset += 1

    ax.set_xlabel("Task")
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"LoCoV1 Per-Task nDCG@10 (max_len={max_len})")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"loco_per_task_maxlen{max_len}.png",
                bbox_inches="tight", dpi=150)
    print(f"\n  Saved: {RESULTS_DIR / f'loco_per_task_maxlen{max_len}.png'}")
    plt.close("all")


def plot_average_comparison(all_len_results: dict):
    """Bar chart of average nDCG@10 across models and max_lens."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gather all model names and lengths
    model_names = set()
    max_lens = sorted(all_len_results.keys())
    for results in all_len_results.values():
        model_names.update(results.keys())
    model_names = sorted(model_names)

    # Add published baselines
    all_models = list(PUBLISHED_BASELINES.keys()) + model_names
    x = np.arange(len(all_models))
    width = 0.35

    for i, ml in enumerate(max_lens):
        scores = []
        for model in all_models:
            if model in PUBLISHED_BASELINES:
                vals = list(PUBLISHED_BASELINES[model].values())
                scores.append(float(np.mean(vals)) if vals else 0)
            elif model in all_len_results.get(ml, {}):
                scores.append(all_len_results[ml][model].get("avg_nDCG@10", 0))
            else:
                scores.append(0)

        bars = ax.bar(x + i * width, scores, width, label=f"max_len={ml}", alpha=0.8)
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{score:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model")
    ax.set_ylabel("Average nDCG@10")
    ax.set_title("LoCoV1 Average nDCG@10 Comparison")
    ax.set_xticks(x + width * (len(max_lens) - 1) / 2)
    ax.set_xticklabels(all_models, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "loco_average_comparison.png", bbox_inches="tight", dpi=150)
    print(f"\n  Saved: {RESULTS_DIR / 'loco_average_comparison.png'}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    max_lens: list[int] = None,
    n_steps: int = 5000,
    batch_size: int = 16,
    lr: float = 3e-4,
    skip_phase_0: bool = False,
):
    if max_lens is None:
        max_lens = [2048]

    print("=" * 70)
    print("LoCoV1 BENCHMARK: PRISM-Simplified vs Transformer")
    print(f"Device: {DEVICE}")
    print(f"Max lengths: {max_lens}")
    print(f"Training steps: {n_steps}")
    print("=" * 70)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Use the largest max_len for tokenization (tokens are truncated per-experiment)
    tokenize_max_len = max(max_lens)

    # Phase 0: Data inspection
    if not skip_phase_0:
        tasks = run_phase_0(tokenizer, max_len=tokenize_max_len)
    else:
        tasks = load_loco_data(tokenizer, max_len=tokenize_max_len)

    # Phase 1: Train and evaluate at each max_len
    all_len_results = {}
    for max_len in max_lens:
        results = run_phase_1(
            tasks, max_len=max_len, n_steps=n_steps,
            batch_size=batch_size, lr=lr,
        )
        all_len_results[max_len] = results

        # Per-task plot
        plot_per_task_comparison(results, max_len)

    # Average comparison plot
    if all_len_results:
        plot_average_comparison(all_len_results)

    # Save all results
    save_results = {
        "config": {
            "device": DEVICE,
            "tokenizer": TOKENIZER_NAME,
            "max_lens": max_lens,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "lr": lr,
        },
    }
    for max_len, results in all_len_results.items():
        save_results[f"max_len_{max_len}"] = results

    with open(RESULTS_DIR / "loco_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'loco_results.json'}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Average nDCG@10")
    print("=" * 70)

    # Published baselines
    for name, scores in PUBLISHED_BASELINES.items():
        avg = float(np.mean(list(scores.values())))
        print(f"  {name:<25} {avg:.4f}  (published)")

    # Our results
    for max_len, results in all_len_results.items():
        for model_name, model_results in results.items():
            avg = model_results.get("avg_nDCG@10", 0)
            print(f"  {model_name:<25} {avg:.4f}  (max_len={max_len}, {n_steps} steps)")

    return all_len_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoCoV1 Benchmark")
    parser.add_argument("--max-lens", type=int, nargs="+", default=[2048],
                        help="Max sequence lengths to evaluate")
    parser.add_argument("--n-steps", type=int, default=5000,
                        help="Training steps per model")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Base batch size (auto-adjusted for long sequences)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--skip-phase-0", action="store_true",
                        help="Skip data inspection phase")
    args = parser.parse_args()

    main(
        max_lens=args.max_lens,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_phase_0=args.skip_phase_0,
    )
