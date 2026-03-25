"""
LoCoV1 zero-shot evaluator.

Evaluation-only module — no training. Loads LoCoV1 from HuggingFace,
encodes queries and documents with a provided model, computes nDCG@k.

Usage:
    from data.loco_eval import evaluate_locov1
    results = evaluate_locov1(model_wrapper, max_len=2048)
    print(results["avg_ndcg@10"])
"""

import math
import time

import numpy as np
import torch
from datasets import load_dataset

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


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

_CACHED_DATA = {}


def _load_loco_data(tokenizer, max_len: int) -> dict:
    """Load and tokenize LoCoV1 data. Cached by max_len."""
    cache_key = max_len
    if cache_key in _CACHED_DATA:
        return _CACHED_DATA[cache_key]

    print(f"  [loco_eval] Loading LoCoV1 (max_len={max_len})...")
    docs_ds = load_dataset("hazyresearch/LoCoV1-Documents", split="test")
    queries_ds = load_dataset("hazyresearch/LoCoV1-Queries", split="test")

    tasks = {t: {"queries": {}, "documents": {}, "qrels": {}} for t in LOCO_TASKS}

    for row in docs_ds:
        task = row["dataset"]
        if task not in tasks:
            continue
        pid = row["pid"]
        token_ids = tokenizer.encode(
            row["passage"], max_length=max_len, truncation=True,
            add_special_tokens=False,
        )
        tasks[task]["documents"][pid] = token_ids

    for row in queries_ds:
        task = row["dataset"]
        if task not in tasks:
            continue
        qid = row["qid"]
        token_ids = tokenizer.encode(
            row["query"], max_length=max_len, truncation=True,
            add_special_tokens=False,
        )
        tasks[task]["queries"][qid] = token_ids
        tasks[task]["qrels"][qid] = row["answer_pids"]

    _CACHED_DATA[cache_key] = tasks
    n_q = sum(len(t["queries"]) for t in tasks.values())
    n_d = sum(len(t["documents"]) for t in tasks.values())
    print(f"  [loco_eval] Loaded {n_q} queries, {n_d} documents across {len(LOCO_TASKS)} tasks")
    return tasks


# ---------------------------------------------------------------------------
# Encoding + collation
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

def _compute_ndcg(ranked_pids: list, relevant_pids: set, k: int = 10) -> float:
    """Compute nDCG@k for a single query (binary relevance)."""
    dcg = 0.0
    for i, pid in enumerate(ranked_pids[:k]):
        if pid in relevant_pids:
            dcg += 1.0 / math.log2(i + 2)
    n_rel = min(len(relevant_pids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _evaluate_task(model_wrapper, task_data, max_len, batch_size, device):
    """Evaluate nDCG@{1,3,5,10} on a single LoCoV1 task."""
    queries = task_data["queries"]
    documents = task_data["documents"]
    qrels = task_data["qrels"]

    if not queries or not documents:
        return None

    doc_pids = list(documents.keys())
    doc_token_ids = [documents[pid] for pid in doc_pids]
    doc_embs = _encode_batch(model_wrapper, doc_token_ids, max_len, batch_size, device)

    query_qids = list(queries.keys())
    query_token_ids = [queries[qid] for qid in query_qids]
    query_embs = _encode_batch(model_wrapper, query_token_ids, max_len, batch_size, device)

    sim_matrix = torch.matmul(query_embs, doc_embs.T)

    ndcg_scores = {k: [] for k in [1, 3, 5, 10]}
    for i, qid in enumerate(query_qids):
        relevant = set(qrels.get(qid, []))
        if not relevant:
            continue
        sorted_indices = sim_matrix[i].argsort(descending=True).tolist()
        ranked = [doc_pids[idx] for idx in sorted_indices]
        for k in ndcg_scores:
            ndcg_scores[k].append(_compute_ndcg(ranked, relevant, k))

    results = {}
    for k in ndcg_scores:
        results[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
    results["n_queries"] = len(query_qids)
    results["n_docs"] = len(doc_pids)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_locov1(
    model_wrapper,
    tokenizer,
    max_len: int = 2048,
    batch_size: int = 32,
    device: str = "cuda",
    tasks_subset: list[str] | None = None,
) -> dict:
    """Zero-shot evaluation on LoCoV1.

    Args:
        model_wrapper: model with .encode(input_ids, attention_mask) -> embeddings
        tokenizer: HuggingFace tokenizer (for loading data)
        max_len: maximum sequence length for encoding
        batch_size: encoding batch size
        device: torch device string
        tasks_subset: optional subset of LOCO_TASKS to evaluate

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
    tasks = _load_loco_data(tokenizer, max_len)
    eval_tasks = tasks_subset or LOCO_TASKS

    t0 = time.perf_counter()
    per_task = {}
    n_queries_total = 0
    n_docs_total = 0

    for task_name in eval_tasks:
        if task_name not in tasks:
            continue
        result = _evaluate_task(
            model_wrapper, tasks[task_name], max_len, batch_size, device,
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
