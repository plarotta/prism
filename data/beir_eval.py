"""
BEIR zero-shot evaluator.

Wraps a PRISM/Transformer/Mamba model in an MTEB-compatible interface
and evaluates on BEIR retrieval datasets.

Requires: mteb (pip install mteb)

Usage:
    from data.beir_eval import evaluate_beir
    results = evaluate_beir(model_wrapper, tokenizer, max_len=512)
    print(results["avg_ndcg@10"])
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

# BEIR datasets available through HuggingFace / MTEB
BEIR_DATASETS = [
    "msmarco",
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]

_CACHED_DATA = {}


# ---------------------------------------------------------------------------
# MTEB-based evaluation (preferred)
# ---------------------------------------------------------------------------

class MTEBModelWrapper:
    """Wraps a PRISM-style model for MTEB evaluation.

    MTEB expects a model with .encode(sentences, batch_size, ...) that takes
    raw strings. This wrapper tokenizes and encodes through the model.
    """

    def __init__(self, model_wrapper, tokenizer, max_len: int, device: str):
        self.model = model_wrapper
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode sentences to embeddings (MTEB interface)."""
        self.model.eval()
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            emb = self.model.encode(ids, mask)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


def evaluate_beir_mteb(
    model_wrapper,
    tokenizer,
    max_len: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    datasets: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Evaluate on BEIR using the MTEB library.

    Args:
        model_wrapper: model with .encode(input_ids, attention_mask) -> embeddings
        tokenizer: HuggingFace tokenizer
        max_len: max sequence length
        batch_size: encoding batch size
        device: torch device
        datasets: subset of BEIR_DATASETS (default: all)
        output_dir: where MTEB saves results (optional)

    Returns:
        {
            "avg_ndcg@10": float,
            "per_dataset": {name: {"nDCG@10": float, ...}},
            "eval_time_s": float,
        }
    """
    try:
        import mteb
    except ImportError:
        raise ImportError(
            "MTEB is required for BEIR evaluation. Install with: uv add mteb"
        )

    t0 = time.perf_counter()
    wrapped = MTEBModelWrapper(model_wrapper, tokenizer, max_len, device)
    eval_datasets = datasets or BEIR_DATASETS

    # Build MTEB task list
    task_names = [f"BeIR/{ds}" for ds in eval_datasets]

    # Some BEIR datasets may not be available in MTEB — filter gracefully
    available_tasks = []
    for name in task_names:
        try:
            tasks = mteb.get_tasks(tasks=[name])
            if tasks:
                available_tasks.extend(tasks)
        except Exception:
            # Try without BeIR/ prefix
            try:
                tasks = mteb.get_tasks(tasks=[name.split("/")[-1]])
                if tasks:
                    available_tasks.extend(tasks)
            except Exception:
                print(f"  [beir_eval] Task {name} not found in MTEB, skipping")

    if not available_tasks:
        print("  [beir_eval] No MTEB tasks found. Falling back to manual evaluation.")
        return evaluate_beir_manual(
            model_wrapper, tokenizer, max_len, batch_size, device, eval_datasets,
        )

    evaluation = mteb.MTEB(tasks=available_tasks)
    results = evaluation.run(
        wrapped,
        output_folder=output_dir or "results/paper/exp5/mteb_output",
        batch_size=batch_size,
    )

    per_dataset = {}
    for result in results:
        task_name = result.task_name
        # Extract nDCG@10 from MTEB results
        scores = result.scores.get("test", result.scores.get("dev", [{}]))
        if scores:
            main_score = scores[0].get("ndcg_at_10", scores[0].get("main_score", 0))
            per_dataset[task_name] = {"nDCG@10": main_score}

    eval_time = time.perf_counter() - t0
    ndcg_values = [v["nDCG@10"] for v in per_dataset.values()]
    avg_ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0

    return {
        "avg_ndcg@10": avg_ndcg10,
        "per_dataset": per_dataset,
        "eval_time_s": round(eval_time, 1),
    }


# ---------------------------------------------------------------------------
# Manual evaluation fallback (no MTEB dependency)
# ---------------------------------------------------------------------------

def _collate_ids(token_ids_list, max_len, pad_id=0):
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
def _encode_texts(model_wrapper, tokenizer, texts, max_len, batch_size, device):
    """Encode raw text strings into embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        emb = model_wrapper.encode(ids, mask)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def _compute_ndcg(ranked_ids, relevant_ids, k=10):
    """Compute nDCG@k (binary relevance)."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_beir_manual(
    model_wrapper,
    tokenizer,
    max_len: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    datasets: list[str] | None = None,
) -> dict:
    """Manual BEIR evaluation without MTEB.

    Loads data from HuggingFace BeIR datasets, encodes queries and docs,
    computes nDCG@10.
    """
    model_wrapper.eval()
    eval_datasets = datasets or BEIR_DATASETS
    t0 = time.perf_counter()

    per_dataset = {}
    for ds_name in eval_datasets:
        print(f"  [beir_eval] Evaluating {ds_name}...")
        try:
            # Try loading from mteb/beir format
            corpus = load_dataset(f"mteb/{ds_name}", "corpus", split="test")
            queries = load_dataset(f"mteb/{ds_name}", "queries", split="test")

            # Build corpus
            doc_ids = [str(row["_id"]) for row in corpus]
            doc_texts = [row["text"] for row in corpus]

            # Limit corpus size for efficiency (top 100K docs)
            if len(doc_ids) > 100000:
                print(f"    Limiting corpus from {len(doc_ids)} to 100,000 docs")
                doc_ids = doc_ids[:100000]
                doc_texts = doc_texts[:100000]

            doc_embs = _encode_texts(
                model_wrapper, tokenizer, doc_texts, max_len, batch_size, device,
            )

            query_ids = [str(row["_id"]) for row in queries]
            query_texts = [row["text"] for row in queries]
            query_embs = _encode_texts(
                model_wrapper, tokenizer, query_texts, max_len, batch_size, device,
            )

            # Load qrels
            qrels_ds = load_dataset(f"mteb/{ds_name}", "default", split="test")
            qrels = {}
            for row in qrels_ds:
                qid = str(row["query-id"])
                did = str(row["corpus-id"])
                score = row.get("score", 1)
                if qid not in qrels:
                    qrels[qid] = set()
                if score > 0:
                    qrels[qid].add(did)

            # Compute similarities and nDCG
            sim_matrix = torch.matmul(query_embs, doc_embs.T)
            ndcg_scores = []
            for i, qid in enumerate(query_ids):
                relevant = qrels.get(qid, set())
                if not relevant:
                    continue
                sorted_indices = sim_matrix[i].argsort(descending=True).tolist()
                ranked = [doc_ids[idx] for idx in sorted_indices]
                ndcg_scores.append(_compute_ndcg(ranked, relevant, k=10))

            if ndcg_scores:
                per_dataset[ds_name] = {
                    "nDCG@10": float(np.mean(ndcg_scores)),
                    "n_queries": len(query_ids),
                    "n_docs": len(doc_ids),
                }
                print(f"    nDCG@10: {per_dataset[ds_name]['nDCG@10']:.4f}")

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    eval_time = time.perf_counter() - t0
    ndcg_values = [v["nDCG@10"] for v in per_dataset.values()]
    avg_ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0

    return {
        "avg_ndcg@10": avg_ndcg10,
        "per_dataset": per_dataset,
        "eval_time_s": round(eval_time, 1),
    }


# ---------------------------------------------------------------------------
# Main entry point (tries MTEB first, falls back to manual)
# ---------------------------------------------------------------------------

def evaluate_beir(
    model_wrapper,
    tokenizer,
    max_len: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    datasets: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Evaluate on BEIR retrieval datasets.

    Tries MTEB first for standardized evaluation. Falls back to manual
    evaluation if MTEB is not available.
    """
    try:
        import mteb
        return evaluate_beir_mteb(
            model_wrapper, tokenizer, max_len, batch_size, device,
            datasets, output_dir,
        )
    except ImportError:
        print("  [beir_eval] MTEB not available, using manual evaluation")
        return evaluate_beir_manual(
            model_wrapper, tokenizer, max_len, batch_size, device, datasets,
        )
