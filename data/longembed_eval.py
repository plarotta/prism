"""
LongEmbed zero-shot evaluator.

Evaluation-only module — no training. Loads LongEmbed from HuggingFace,
encodes queries and documents with a provided model, computes nDCG@k.

LongEmbed (Zhu et al., EMNLP 2024) has 6 tasks:
  - 2 synthetic: LEPasskeyRetrieval, LEConcatNeedleRetrieval
  - 4 real: LEQMSum, LEQualitySum (NarrativeQA), LE2WikiMultihopQA, LESummScreenFD

Usage:
    from data.longembed_eval import evaluate_longembed
    results = evaluate_longembed(model_wrapper, tokenizer, max_len=4096)
    print(results["avg_ndcg@10"])
"""

import math
import time

import numpy as np
import torch
from datasets import load_dataset

LONGEMBED_TASKS = {
    "LEPasskeyRetrieval": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LEPasskeyRetrieval",
    },
    "LEConcatNeedleRetrieval": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LEConcatNeedleRetrieval",
    },
    "LEQMSum": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LEQMSum",
    },
    "LEQualitySum": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LEQualitySum",
    },
    "LE2WikiMultihopQA": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LE2WikiMultihopQA",
    },
    "LESummScreenFD": {
        "corpus": "dwzhu/LongEmbed",
        "queries": "dwzhu/LongEmbed",
        "subset": "LESummScreenFD",
    },
}

_CACHED_DATA = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_longembed_task(task_name: str, tokenizer, max_len: int) -> dict:
    """Load and tokenize one LongEmbed task."""
    cache_key = (task_name, max_len)
    if cache_key in _CACHED_DATA:
        return _CACHED_DATA[cache_key]

    print(f"  [longembed_eval] Loading {task_name} (max_len={max_len})...")

    task_info = LONGEMBED_TASKS[task_name]
    subset = task_info["subset"]

    # LongEmbed stores corpus and queries as separate configs
    corpus_ds = load_dataset("dwzhu/LongEmbed", f"{subset}-corpus", split="train")
    queries_ds = load_dataset("dwzhu/LongEmbed", f"{subset}-queries", split="train")

    documents = {}
    for row in corpus_ds:
        doc_id = str(row["_id"])
        token_ids = tokenizer.encode(
            row["text"], max_length=max_len, truncation=True,
            add_special_tokens=False,
        )
        documents[doc_id] = token_ids

    queries = {}
    qrels = {}
    for row in queries_ds:
        qid = str(row["_id"])
        token_ids = tokenizer.encode(
            row["text"], max_length=max_len, truncation=True,
            add_special_tokens=False,
        )
        queries[qid] = token_ids
        # qrels: mapping from doc_id to relevance score
        if "relevant_docs" in row:
            qrels[qid] = row["relevant_docs"]
        elif "answer_pids" in row:
            qrels[qid] = row["answer_pids"]

    data = {"queries": queries, "documents": documents, "qrels": qrels}
    _CACHED_DATA[cache_key] = data

    print(f"  [longembed_eval]   {len(queries)} queries, {len(documents)} documents")
    return data


# ---------------------------------------------------------------------------
# Encoding + collation (same pattern as loco_eval)
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def _encode_batch(model_wrapper, token_ids_list, max_len, batch_size, device):
    """Encode a list of token ID sequences into L2-normalized embeddings."""
    all_embs = []
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i + batch_size]
        ids, masks = _collate_ids(batch, max_len)
        ids, masks = ids.to(device), masks.to(device)
        emb = model_wrapper.encode(ids, masks)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# nDCG computation
# ---------------------------------------------------------------------------

def _compute_ndcg(ranked_ids: list, relevant_ids: set, k: int = 10) -> float:
    """Compute nDCG@k for a single query (binary relevance)."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _evaluate_task(model_wrapper, task_data, max_len, batch_size, device):
    """Evaluate nDCG@{1,3,5,10} on a single LongEmbed task."""
    queries = task_data["queries"]
    documents = task_data["documents"]
    qrels = task_data["qrels"]

    if not queries or not documents:
        return None

    doc_ids = list(documents.keys())
    doc_token_ids = [documents[did] for did in doc_ids]
    doc_embs = _encode_batch(model_wrapper, doc_token_ids, max_len, batch_size, device)

    query_ids = list(queries.keys())
    query_token_ids = [queries[qid] for qid in query_ids]
    query_embs = _encode_batch(model_wrapper, query_token_ids, max_len, batch_size, device)

    sim_matrix = torch.matmul(query_embs, doc_embs.T)

    ndcg_scores = {k: [] for k in [1, 3, 5, 10]}
    for i, qid in enumerate(query_ids):
        relevant_raw = qrels.get(qid, [])
        # Handle both list of IDs and dict of {id: score}
        if isinstance(relevant_raw, dict):
            relevant = set(str(k) for k in relevant_raw.keys())
        elif isinstance(relevant_raw, list):
            relevant = set(str(r) for r in relevant_raw)
        else:
            continue

        if not relevant:
            continue

        sorted_indices = sim_matrix[i].argsort(descending=True).tolist()
        ranked = [doc_ids[idx] for idx in sorted_indices]
        for k in ndcg_scores:
            ndcg_scores[k].append(_compute_ndcg(ranked, relevant, k))

    results = {}
    for k in ndcg_scores:
        results[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
    results["n_queries"] = len(query_ids)
    results["n_docs"] = len(doc_ids)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_longembed(
    model_wrapper,
    tokenizer,
    max_len: int = 4096,
    batch_size: int = 16,
    device: str = "cuda",
    tasks_subset: list[str] | None = None,
) -> dict:
    """Zero-shot evaluation on LongEmbed.

    Args:
        model_wrapper: model with .encode(input_ids, attention_mask) -> embeddings
        tokenizer: HuggingFace tokenizer (for loading data)
        max_len: maximum sequence length for encoding
        batch_size: encoding batch size
        device: torch device string
        tasks_subset: optional subset of task names to evaluate

    Returns:
        {
            "avg_ndcg@10": float,
            "per_task": {task_name: {"nDCG@1": ..., "nDCG@10": ..., ...}},
            "n_queries_total": int,
            "n_docs_total": int,
            "eval_time_s": float,
        }
    """
    model_wrapper.eval()
    eval_tasks = tasks_subset or list(LONGEMBED_TASKS.keys())

    t0 = time.perf_counter()
    per_task = {}
    n_queries_total = 0
    n_docs_total = 0

    for task_name in eval_tasks:
        if task_name not in LONGEMBED_TASKS:
            print(f"  [longembed_eval] Unknown task: {task_name}, skipping")
            continue

        task_data = _load_longembed_task(task_name, tokenizer, max_len)
        result = _evaluate_task(
            model_wrapper, task_data, max_len, batch_size, device,
        )
        if result is not None:
            per_task[task_name] = result
            n_queries_total += result["n_queries"]
            n_docs_total += result["n_docs"]

    eval_time = time.perf_counter() - t0

    task_ndcg10 = [v["nDCG@10"] for v in per_task.values()]
    avg_ndcg10 = float(np.mean(task_ndcg10)) if task_ndcg10 else 0.0

    return {
        "avg_ndcg@10": avg_ndcg10,
        "per_task": per_task,
        "n_queries_total": n_queries_total,
        "n_docs_total": n_docs_total,
        "eval_time_s": round(eval_time, 1),
    }
