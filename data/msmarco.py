"""
MS MARCO passage retrieval dataset with hard negative support.

Loads from Tevatron/msmarco-passage (HuggingFace) which provides queries,
positive passages, and 30 BM25 hard negatives per query.

Falls back to mteb/msmarco for corpus + queries + qrels if Tevatron is
unavailable.

Pre-tokenizes on first load and caches to disk for fast subsequent access.

Usage:
    from data.msmarco import MSMARCODataset
    ds = MSMARCODataset(tokenizer, max_len=512)
    ds.load()
    batch = ds.sample_batch(16)

Preparation (pre-tokenize, ~10 min first time):
    uv run python -m data.msmarco --prepare --max-len 512
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

CACHE_DIR = Path("data") / ".cache" / "msmarco"


class MSMARCODataset:
    """MS MARCO passage retrieval dataset for contrastive training.

    Supports in-batch negatives and optional BM25 hard negatives.
    """

    def __init__(
        self,
        tokenizer,
        max_len: int = 512,
        n_hard_negatives: int = 7,
        cache_dir: Path = CACHE_DIR,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_hard_negatives = n_hard_negatives
        self.cache_dir = cache_dir / f"maxlen_{max_len}"

        # Populated by load()
        self.passages = None          # dict: pid (str) -> token_ids (list[int])
        self.train_queries = None     # dict: qid (str) -> token_ids (list[int])
        self.train_qrels = None       # dict: qid -> list[pid]  (positive passages)
        self.train_negatives = None   # dict: qid -> list[pid]  (hard negatives)
        self.dev_queries = None       # dict: qid -> token_ids
        self.dev_qrels = None         # dict: qid -> list[pid]
        self._train_qids = None       # list for sampling

    def load(self):
        """Load dataset, using cache if available."""
        if self._try_load_cache():
            return

        print("[msmarco] Loading from HuggingFace (first time — will cache)...")
        self._load_from_hf()
        self._save_cache()

    def _try_load_cache(self) -> bool:
        """Try to load pre-tokenized data from cache."""
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            return False

        print(f"[msmarco] Loading from cache: {self.cache_dir}")
        t0 = time.perf_counter()

        self.passages = _load_id_to_tokens(self.cache_dir / "passages.jsonl")
        self.train_queries = _load_id_to_tokens(self.cache_dir / "train_queries.jsonl")
        self.train_qrels = _load_id_to_list(self.cache_dir / "train_qrels.jsonl")
        self.dev_queries = _load_id_to_tokens(self.cache_dir / "dev_queries.jsonl")
        self.dev_qrels = _load_id_to_list(self.cache_dir / "dev_qrels.jsonl")

        neg_path = self.cache_dir / "train_negatives.jsonl"
        if neg_path.exists():
            self.train_negatives = _load_id_to_list(self.cache_dir / "train_negatives.jsonl")
        else:
            self.train_negatives = {}

        self._train_qids = [
            qid for qid in self.train_qrels
            if qid in self.train_queries and self.train_qrels[qid]
        ]
        elapsed = time.perf_counter() - t0
        print(f"[msmarco] Loaded {len(self.passages)} passages, "
              f"{len(self.train_queries)} train queries, "
              f"{len(self.dev_queries)} dev queries in {elapsed:.1f}s")
        return True

    def _load_from_hf(self):
        """Load MS MARCO from HuggingFace datasets."""
        from datasets import load_dataset

        self.passages = {}
        self.train_queries = {}
        self.train_qrels = {}
        self.train_negatives = {}
        self.dev_queries = {}
        self.dev_qrels = {}

        # Primary source: Tevatron/msmarco-passage
        # Has query + positive passages + 30 hard negatives per row
        try:
            self._load_tevatron()
        except Exception as e:
            print(f"  Tevatron source failed ({e}), falling back to mteb/msmarco...")
            self._load_mteb()

        self._train_qids = list(self.train_qrels.keys())
        print(f"  Final: {len(self.passages)} passages, "
              f"{len(self.train_queries)} train queries, "
              f"{len(self.dev_queries)} dev queries, "
              f"{len(self.train_negatives)} queries with hard negatives")

    def _load_tevatron(self):
        """Load from Tevatron/msmarco-passage — most convenient format."""
        from datasets import load_dataset

        print("  Loading from Tevatron/msmarco-passage...")
        ds = load_dataset("Tevatron/msmarco-passage", split="train")

        pid_counter = 0
        n_rows = len(ds)
        print(f"  Processing {n_rows} query-passage rows...")

        for i, row in enumerate(ds):
            qid = str(row["query_id"])
            query_text = row["query"]

            # Tokenize query
            q_tokens = self.tokenizer.encode(
                query_text, max_length=self.max_len, truncation=True,
                add_special_tokens=False,
            )
            if len(q_tokens) < 2:
                continue
            self.train_queries[qid] = q_tokens

            # Process positive passages
            pos_pids = []
            for pos in row.get("positive_passages", []):
                pid = str(pos.get("docid", f"pos_{pid_counter}"))
                text = pos.get("text", "")
                title = pos.get("title", "")
                full_text = f"{title} {text}".strip() if title else text
                p_tokens = self.tokenizer.encode(
                    full_text, max_length=self.max_len, truncation=True,
                    add_special_tokens=False,
                )
                if len(p_tokens) >= 4:
                    self.passages[pid] = p_tokens
                    pos_pids.append(pid)
                pid_counter += 1

            if pos_pids:
                self.train_qrels[qid] = pos_pids

            # Process negative passages
            neg_pids = []
            for neg in row.get("negative_passages", []):
                pid = str(neg.get("docid", f"neg_{pid_counter}"))
                text = neg.get("text", "")
                title = neg.get("title", "")
                full_text = f"{title} {text}".strip() if title else text
                if pid not in self.passages:
                    p_tokens = self.tokenizer.encode(
                        full_text, max_length=self.max_len, truncation=True,
                        add_special_tokens=False,
                    )
                    if len(p_tokens) >= 4:
                        self.passages[pid] = p_tokens
                neg_pids.append(pid)
                pid_counter += 1

            valid_negs = [p for p in neg_pids if p in self.passages]
            if valid_negs:
                self.train_negatives[qid] = valid_negs

            if (i + 1) % 100000 == 0:
                print(f"    {i+1}/{n_rows} rows processed, "
                      f"{len(self.passages)} passages so far...")

        # Tevatron doesn't have a separate dev split — create one from train
        # Take last 5000 queries as dev
        all_qids = list(self.train_qrels.keys())
        if len(all_qids) > 10000:
            dev_qids = set(all_qids[-5000:])
            for qid in dev_qids:
                if qid in self.train_queries:
                    self.dev_queries[qid] = self.train_queries.pop(qid)
                    self.dev_qrels[qid] = self.train_qrels.pop(qid)
                    self.train_negatives.pop(qid, None)

    def _load_mteb(self):
        """Fallback: load from mteb/msmarco (corpus + queries + qrels)."""
        from datasets import load_dataset

        # Load corpus
        print("  Loading corpus from mteb/msmarco...")
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus")
        print(f"  Tokenizing {len(corpus)} passages...")
        for i, row in enumerate(corpus):
            pid = str(row["_id"])
            text = row.get("text", "")
            title = row.get("title", "")
            full_text = f"{title} {text}".strip() if title else text
            token_ids = self.tokenizer.encode(
                full_text, max_length=self.max_len, truncation=True,
                add_special_tokens=False,
            )
            if len(token_ids) >= 4:
                self.passages[pid] = token_ids
            if (i + 1) % 1_000_000 == 0:
                print(f"    {i+1} passages tokenized...")

        # Load queries
        print("  Loading queries from mteb/msmarco...")
        queries_ds = load_dataset("mteb/msmarco", "queries", split="queries")
        all_queries = {}
        for row in queries_ds:
            qid = str(row["_id"])
            text = row.get("text", "")
            token_ids = self.tokenizer.encode(
                text, max_length=self.max_len, truncation=True,
                add_special_tokens=False,
            )
            if len(token_ids) >= 2:
                all_queries[qid] = token_ids

        # Load qrels (train split)
        print("  Loading qrels...")
        qrels_ds = load_dataset("mteb/msmarco", split="train")
        for row in qrels_ds:
            qid = str(row["query-id"])
            pid = str(row["corpus-id"])
            score = row.get("score", 1)
            if score > 0 and pid in self.passages and qid in all_queries:
                self.train_qrels.setdefault(qid, []).append(pid)
                self.train_queries[qid] = all_queries[qid]

        # Dev qrels
        try:
            dev_ds = load_dataset("mteb/msmarco", split="dev")
            for row in dev_ds:
                qid = str(row["query-id"])
                pid = str(row["corpus-id"])
                score = row.get("score", 1)
                if score > 0 and pid in self.passages and qid in all_queries:
                    self.dev_qrels.setdefault(qid, []).append(pid)
                    self.dev_queries[qid] = all_queries[qid]
        except Exception:
            # Split last 5000 from train as dev
            all_qids = list(self.train_qrels.keys())
            if len(all_qids) > 10000:
                for qid in all_qids[-5000:]:
                    self.dev_queries[qid] = self.train_queries.pop(qid)
                    self.dev_qrels[qid] = self.train_qrels.pop(qid)

    def _save_cache(self):
        """Save pre-tokenized data to cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[msmarco] Saving cache to {self.cache_dir}...")

        _save_id_to_tokens(self.cache_dir / "passages.jsonl", self.passages)
        _save_id_to_tokens(self.cache_dir / "train_queries.jsonl", self.train_queries)
        _save_id_to_list(self.cache_dir / "train_qrels.jsonl", self.train_qrels)
        _save_id_to_tokens(self.cache_dir / "dev_queries.jsonl", self.dev_queries)
        _save_id_to_list(self.cache_dir / "dev_qrels.jsonl", self.dev_qrels)
        if self.train_negatives:
            _save_id_to_list(self.cache_dir / "train_negatives.jsonl", self.train_negatives)

        meta = {
            "max_len": self.max_len,
            "n_passages": len(self.passages),
            "n_train_queries": len(self.train_queries),
            "n_dev_queries": len(self.dev_queries),
            "n_hard_negatives": len(self.train_negatives),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.cache_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print("[msmarco] Cache saved.")

    # ----- Batch sampling -----

    def sample_batch(self, batch_size: int) -> dict:
        """Sample a training batch with in-batch negatives.

        Returns dict of tensors:
            query_ids:  (B, max_q_len)
            query_mask: (B, max_q_len)
            pos_ids:    (B, max_p_len)
            pos_mask:   (B, max_p_len)
        """
        qids = random.choices(self._train_qids, k=batch_size)

        query_tokens = []
        pos_tokens = []

        for qid in qids:
            query_tokens.append(self.train_queries[qid])
            pos_pid = random.choice(self.train_qrels[qid])
            pos_tokens.append(self.passages[pos_pid])

        q_ids, q_mask = _collate(query_tokens, self.max_len)
        p_ids, p_mask = _collate(pos_tokens, self.max_len)

        return {
            "query_ids": q_ids,
            "query_mask": q_mask,
            "pos_ids": p_ids,
            "pos_mask": p_mask,
        }

    def get_dev_data(self) -> tuple[dict, dict]:
        """Return (dev_queries, dev_qrels) for MRR@10 evaluation."""
        return self.dev_queries, self.dev_qrels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collate(token_ids_list: list[list[int]], max_len: int, pad_id: int = 0):
    """Pad token ID lists to (B, max_len) tensors."""
    ids_out, masks_out = [], []
    for ids in token_ids_list:
        t = ids[:max_len]
        padded = t + [pad_id] * (max_len - len(t))
        mask = [1] * len(t) + [0] * (max_len - len(t))
        ids_out.append(padded)
        masks_out.append(mask)
    return torch.tensor(ids_out, dtype=torch.long), torch.tensor(masks_out, dtype=torch.long)


def _save_id_to_tokens(path: Path, data: dict):
    """Save {id: token_ids} as JSONL."""
    with open(path, "w") as f:
        for k, v in data.items():
            f.write(json.dumps({"id": k, "tokens": v}) + "\n")


def _load_id_to_tokens(path: Path) -> dict:
    """Load {id: token_ids} from JSONL."""
    result = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            result[row["id"]] = row["tokens"]
    return result


def _save_id_to_list(path: Path, data: dict):
    """Save {id: list[str]} as JSONL."""
    with open(path, "w") as f:
        for k, v in data.items():
            f.write(json.dumps({"id": k, "items": v}) + "\n")


def _load_id_to_list(path: Path) -> dict:
    """Load {id: list[str]} from JSONL."""
    result = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            result[row["id"]] = row["items"]
    return result


# ---------------------------------------------------------------------------
# CLI for preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MS MARCO data preparation")
    parser.add_argument("--prepare", action="store_true", help="Download and pre-tokenize")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    args = parser.parse_args()

    if args.prepare:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        ds = MSMARCODataset(tokenizer, max_len=args.max_len)
        ds.load()
        print(f"\nReady. Sample batch:")
        batch = ds.sample_batch(4)
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
    else:
        parser.print_help()
