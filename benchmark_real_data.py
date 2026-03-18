"""
Phase 3: Real Data Evaluation — Simplified PRISM vs Transformer

Tests the core claim: at sequence lengths beyond ~256 tokens, a fixed-rate
multi-channel recurrence learns better embeddings than a Transformer of equal
parameter count, with lower latency and memory.

Three evaluations:
  3A. STS-Benchmark (real language quality at natural lengths)
  3B. Length-controlled retrieval (quality vs sequence length — the money chart)
  3C. Inference scaling on real text (wall-clock confirmation)

Requires: uv add datasets transformers
"""

import json
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

from datasets import load_dataset
from transformers import AutoTokenizer

from prism import prism_small, PRISMEncoder, PRISMForEmbedding
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
REAL_VOCAB_SIZE = 30522  # bert-base-uncased vocab
BATCH_SIZE = 32
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_simplified_prism(vocab_size=REAL_VOCAB_SIZE, **kwargs):
    """Simplified PRISM: multi-channel recurrence + mean pooling, no interference."""
    model = prism_small(vocab_size=vocab_size, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


def build_transformer(vocab_size=REAL_VOCAB_SIZE, **kwargs):
    """Parameter-matched Transformer baseline."""
    return transformer_small(vocab_size=vocab_size, **kwargs)


# ---------------------------------------------------------------------------
# Data loading + tokenization
# ---------------------------------------------------------------------------

def load_nli_pairs(tokenizer, max_pairs=50000, max_len=128):
    """Load entailment pairs from SNLI + MNLI for contrastive training.

    Returns list of (query_ids, pos_ids) token ID tensors.
    """
    print("Loading NLI data...")
    pairs = []

    for dataset_name, config in [("snli", None), ("glue", "mnli")]:
        try:
            if config:
                ds = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: could not load {dataset_name}: {e}")
            continue

        # Column names differ between datasets
        if "premise" in ds.column_names:
            s1_col, s2_col = "premise", "hypothesis"
        elif "sentence1" in ds.column_names:
            s1_col, s2_col = "sentence1", "sentence2"
        else:
            print(f"  Warning: unknown columns in {dataset_name}: {ds.column_names}")
            continue

        for row in ds:
            if row["label"] == 0:  # entailment
                pairs.append((row[s1_col], row[s2_col]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    print(f"  Loaded {len(pairs)} entailment pairs")

    # Tokenize
    print("  Tokenizing...")
    tokenized = []
    for s1, s2 in pairs:
        t1 = tokenizer.encode(s1, max_length=max_len, truncation=True, add_special_tokens=False)
        t2 = tokenizer.encode(s2, max_length=max_len, truncation=True, add_special_tokens=False)
        if len(t1) >= 4 and len(t2) >= 4:  # skip trivially short
            tokenized.append((t1, t2))

    print(f"  Tokenized {len(tokenized)} pairs")
    return tokenized


def load_sts_benchmark(tokenizer, max_len=128):
    """Load STS-Benchmark test set for evaluation.

    Returns list of (s1_ids, s2_ids, score) where score is 0-5.
    """
    print("Loading STS-Benchmark...")
    ds = load_dataset("mteb/stsbenchmark-sts", split="test", trust_remote_code=True)

    examples = []
    for row in ds:
        s1 = tokenizer.encode(row["sentence1"], max_length=max_len, truncation=True,
                               add_special_tokens=False)
        s2 = tokenizer.encode(row["sentence2"], max_length=max_len, truncation=True,
                               add_special_tokens=False)
        score = row["score"]
        if len(s1) >= 2 and len(s2) >= 2:
            examples.append((s1, s2, score))

    print(f"  Loaded {len(examples)} STS-B test pairs")
    return examples


def load_long_documents(tokenizer, n_docs=5000, min_len=256):
    """Load long documents for retrieval evaluation.

    Tries Wikipedia first, falls back to CNN/DailyMail articles.
    Returns list of (doc_ids, article_id) where article_id groups related docs.
    """
    # Try Wikipedia (new HF path)
    for wiki_path, wiki_config in [
        ("wikimedia/wikipedia", "20231101.en"),
        ("wikipedia", "20220301.en"),
    ]:
        try:
            print(f"Loading Wikipedia paragraphs ({wiki_path}, {wiki_config})...")
            ds = load_dataset(wiki_path, wiki_config, split="train", streaming=True)
            documents = []
            article_id = 0
            for article in ds:
                text = article["text"]
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
                for para in paragraphs:
                    tokens = tokenizer.encode(para, max_length=2048, truncation=True,
                                               add_special_tokens=False)
                    if len(tokens) >= min_len:
                        documents.append((tokens, article_id))
                article_id += 1
                if len(documents) >= n_docs:
                    break
            if documents:
                print(f"  Loaded {len(documents)} paragraphs from {article_id} articles")
                return documents
        except Exception as e:
            print(f"  Wikipedia ({wiki_path}) failed: {e}")

    # Fallback: CNN/DailyMail (articles are 300-800 tokens naturally)
    print("Falling back to CNN/DailyMail...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    documents = []
    article_id = 0
    for article in ds:
        text = article["article"]
        tokens = tokenizer.encode(text, max_length=2048, truncation=True,
                                   add_special_tokens=False)
        if len(tokens) >= min_len:
            documents.append((tokens, article_id))
        # Also split long articles into chunks to get same-article pairs
        if len(tokens) >= min_len * 2:
            mid = len(tokens) // 2
            documents.append((tokens[:mid], article_id))
            documents.append((tokens[mid:], article_id))
        article_id += 1
        if len(documents) >= n_docs:
            break

    print(f"  Loaded {len(documents)} paragraphs (≥{min_len} tokens) from {article_id} articles")
    return documents


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------

def collate_pairs(pairs, max_len, pad_id=0):
    """Pad a list of (ids_a, ids_b) to uniform length tensors."""
    a_ids, b_ids = [], []
    a_masks, b_masks = [], []

    for ids_a, ids_b in pairs:
        a = ids_a[:max_len]
        b = ids_b[:max_len]
        a_pad = a + [pad_id] * (max_len - len(a))
        b_pad = b + [pad_id] * (max_len - len(b))
        a_ids.append(a_pad)
        b_ids.append(b_pad)
        a_masks.append([1] * len(a) + [0] * (max_len - len(a)))
        b_masks.append([1] * len(b) + [0] * (max_len - len(b)))

    return (
        torch.tensor(a_ids, dtype=torch.long),
        torch.tensor(a_masks, dtype=torch.long),
        torch.tensor(b_ids, dtype=torch.long),
        torch.tensor(b_masks, dtype=torch.long),
    )


def collate_docs(doc_ids_list, max_len, pad_id=0):
    """Pad a list of token ID lists to uniform length."""
    ids_out, masks_out = [], []
    for ids in doc_ids_list:
        t = ids[:max_len]
        padded = t + [pad_id] * (max_len - len(t))
        mask = [1] * len(t) + [0] * (max_len - len(t))
        ids_out.append(padded)
        masks_out.append(mask)
    return torch.tensor(ids_out, dtype=torch.long), torch.tensor(masks_out, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_contrastive(model_wrapper, train_pairs, n_steps, model_name,
                      max_len=128, lr=3e-4):
    """Train with InfoNCE on NLI pairs."""
    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    losses = []
    accuracies = []
    model_wrapper.train()

    print(f"\nTraining {model_name} ({n_steps} steps, max_len={max_len})...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Sample a batch
        batch_pairs = random.sample(train_pairs, min(BATCH_SIZE, len(train_pairs)))
        q_ids, q_mask, p_ids, p_mask = collate_pairs(batch_pairs, max_len)
        q_ids, q_mask = q_ids.to(DEVICE), q_mask.to(DEVICE)
        p_ids, p_mask = p_ids.to(DEVICE), p_mask.to(DEVICE)

        result = model_wrapper(q_ids, q_mask, p_ids, p_mask)
        loss = result["loss"]
        acc = result["accuracy"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if (step + 1) % 100 == 0:
            recent_loss = np.mean(losses[-100:])
            recent_acc = np.mean(accuracies[-100:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  acc={recent_acc:.3f}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")
    return losses, accuracies, total_time


# ---------------------------------------------------------------------------
# Evaluation: STS-B
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sts(model_wrapper, sts_examples, max_len=128):
    """Evaluate on STS-Benchmark: Spearman correlation of cosine similarities."""
    from scipy.stats import spearmanr

    model_wrapper.eval()
    pred_scores = []
    gold_scores = []

    for i in range(0, len(sts_examples), BATCH_SIZE):
        batch = sts_examples[i:i+BATCH_SIZE]
        pairs = [(ex[0], ex[1]) for ex in batch]
        scores = [ex[2] for ex in batch]

        q_ids, q_mask, p_ids, p_mask = collate_pairs(pairs, max_len)
        q_ids, q_mask = q_ids.to(DEVICE), q_mask.to(DEVICE)
        p_ids, p_mask = p_ids.to(DEVICE), p_mask.to(DEVICE)

        q_emb = model_wrapper.encode(q_ids, q_mask)
        p_emb = model_wrapper.encode(p_ids, p_mask)

        cos_sim = F.cosine_similarity(q_emb, p_emb, dim=-1)
        pred_scores.extend(cos_sim.cpu().tolist())
        gold_scores.extend(scores)

    spearman = spearmanr(pred_scores, gold_scores).correlation
    return {"spearman": spearman, "n_pairs": len(gold_scores)}


# ---------------------------------------------------------------------------
# Evaluation: Length-controlled retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval_at_length(model_wrapper, documents, seq_len,
                                  n_queries=200, corpus_size=2000):
    """Create a retrieval task from Wikipedia paragraphs at a specific length.

    Positive pairs: paragraphs from the same article.
    Evaluate MRR and Recall@k.
    """
    model_wrapper.eval()

    # Filter to documents long enough for this seq_len
    valid_docs = [(ids[:seq_len], aid) for ids, aid in documents if len(ids) >= seq_len]
    if len(valid_docs) < corpus_size:
        print(f"    Warning: only {len(valid_docs)} docs ≥ {seq_len} tokens (need {corpus_size})")
        if len(valid_docs) < n_queries * 2:
            return None
        corpus_size = len(valid_docs)
        n_queries = min(n_queries, corpus_size // 4)

    # Group by article
    from collections import defaultdict
    article_docs = defaultdict(list)
    for ids, aid in valid_docs:
        article_docs[aid].append(ids)

    # Build query-positive pairs: two paragraphs from the same article
    query_pos_pairs = []
    for aid, docs in article_docs.items():
        if len(docs) >= 2:
            for i in range(len(docs) - 1):
                query_pos_pairs.append((docs[i], docs[i + 1], aid))
                if len(query_pos_pairs) >= n_queries:
                    break
        if len(query_pos_pairs) >= n_queries:
            break

    if len(query_pos_pairs) < 20:
        print(f"    Warning: only {len(query_pos_pairs)} same-article pairs, skipping")
        return None

    n_queries = len(query_pos_pairs)

    # Build corpus: positives + random distractors
    corpus_ids = [p for _, p, _ in query_pos_pairs]
    used_aids = {aid for _, _, aid in query_pos_pairs}
    distractors = [ids for ids, aid in valid_docs if aid not in used_aids]
    random.shuffle(distractors)
    n_distractors = min(corpus_size - n_queries, len(distractors))
    corpus_ids.extend(distractors[:n_distractors])

    query_ids = [q for q, _, _ in query_pos_pairs]

    # Encode queries
    q_embs = []
    for i in range(0, len(query_ids), BATCH_SIZE):
        batch_ids, batch_mask = collate_docs(query_ids[i:i+BATCH_SIZE], seq_len)
        batch_ids, batch_mask = batch_ids.to(DEVICE), batch_mask.to(DEVICE)
        emb = model_wrapper.encode(batch_ids, batch_mask)
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, dim=0)

    # Encode corpus
    c_embs = []
    for i in range(0, len(corpus_ids), BATCH_SIZE):
        batch_ids, batch_mask = collate_docs(corpus_ids[i:i+BATCH_SIZE], seq_len)
        batch_ids, batch_mask = batch_ids.to(DEVICE), batch_mask.to(DEVICE)
        emb = model_wrapper.encode(batch_ids, batch_mask)
        c_embs.append(emb.cpu())
    c_embs = torch.cat(c_embs, dim=0)

    # Retrieval: query i's positive is corpus[i]
    sim = torch.matmul(q_embs, c_embs.T)
    ranks = []
    for i in range(n_queries):
        sorted_idx = sim[i].argsort(descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "n_queries": n_queries,
        "corpus_size": len(corpus_ids),
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_throughput(model_wrapper, seq_len, n_warmup=5, n_measure=20):
    """Measure encoding throughput at a given sequence length."""
    model_wrapper.eval()
    ids = torch.randint(1, REAL_VOCAB_SIZE, (BATCH_SIZE, seq_len), device=DEVICE)
    mask = torch.ones_like(ids)

    # Warmup
    for _ in range(n_warmup):
        model_wrapper.encode(ids, mask)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_measure):
        model_wrapper.encode(ids, mask)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    throughput = (BATCH_SIZE * n_measure) / elapsed
    latency_ms = (elapsed / n_measure) * 1000
    return {"throughput_seq_per_sec": throughput, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# 3A: STS-Benchmark evaluation
# ---------------------------------------------------------------------------

def run_phase_3a(tokenizer, train_pairs, sts_examples, n_steps=2000):
    print("\n" + "=" * 70)
    print("PHASE 3A: STS-Benchmark — Real Language Quality")
    print("=" * 70)

    results = {}

    for name, build_encoder, WrapperClass in [
        ("PRISM-Simplified", lambda: build_simplified_prism(), PRISMForEmbedding),
        ("Transformer", lambda: build_transformer(), TransformerForEmbedding),
    ]:
        torch.manual_seed(42)
        random.seed(42)

        encoder = build_encoder().to(DEVICE)
        wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)
        n_params = sum(p.numel() for p in wrapper.parameters())
        print(f"\n  {name}: {n_params:,} params")

        losses, accs, train_time = train_contrastive(
            wrapper, train_pairs, n_steps, name, max_len=128
        )
        sts_metrics = evaluate_sts(wrapper, sts_examples, max_len=128)

        results[name] = {
            "sts_spearman": sts_metrics["spearman"],
            "train_time": train_time,
            "params": n_params,
            "final_loss": float(np.mean(losses[-100:])),
        }
        print(f"  {name}: STS-B Spearman={sts_metrics['spearman']:.4f}  "
              f"loss={results[name]['final_loss']:.4f}  time={train_time:.1f}s")

    print(f"\n  STS-B gap: {results['PRISM-Simplified']['sts_spearman'] - results['Transformer']['sts_spearman']:+.4f}")
    return results


# ---------------------------------------------------------------------------
# 3B: Length-controlled retrieval (the money experiment)
# ---------------------------------------------------------------------------

def make_article_pairs(documents, min_len=128):
    """Create contrastive training pairs from same-article document chunks.

    Two chunks from the same article = positive pair. In-batch negatives provide
    negative signal. Returns list of (chunk_a_ids, chunk_b_ids).
    """
    from collections import defaultdict
    article_docs = defaultdict(list)
    for ids, aid in documents:
        article_docs[aid].append(ids)

    pairs = []
    for aid, docs in article_docs.items():
        if len(docs) >= 2:
            for i in range(len(docs) - 1):
                if len(docs[i]) >= min_len and len(docs[i + 1]) >= min_len:
                    pairs.append((docs[i], docs[i + 1]))
    random.shuffle(pairs)
    return pairs


def train_contrastive_long(model_wrapper, doc_pairs, n_steps, model_name,
                           max_len=512, lr=3e-4):
    """Train with InfoNCE on long document pairs."""
    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    losses = []
    accuracies = []
    model_wrapper.train()
    batch_size = min(BATCH_SIZE, 16)  # smaller batch for longer sequences

    print(f"\nTraining {model_name} ({n_steps} steps, max_len={max_len}, batch={batch_size})...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        batch_pairs = random.choices(doc_pairs, k=batch_size)
        q_ids, q_mask, p_ids, p_mask = collate_pairs(batch_pairs, max_len)
        q_ids, q_mask = q_ids.to(DEVICE), q_mask.to(DEVICE)
        p_ids, p_mask = p_ids.to(DEVICE), p_mask.to(DEVICE)

        result = model_wrapper(q_ids, q_mask, p_ids, p_mask)
        loss = result["loss"]
        acc = result["accuracy"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if (step + 1) % 100 == 0:
            recent_loss = np.mean(losses[-100:])
            recent_acc = np.mean(accuracies[-100:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  acc={recent_acc:.3f}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")
    return losses, accuracies, total_time


def run_phase_3b(tokenizer, documents, n_steps=2000):
    print("\n" + "=" * 70)
    print("PHASE 3B: Quality vs Sequence Length — The Money Chart")
    print("  Train at each length, then evaluate at that length.")
    print("=" * 70)

    # Create training pairs from same-article chunks
    doc_pairs = make_article_pairs(documents, min_len=128)
    print(f"  Created {len(doc_pairs)} same-article training pairs")

    if len(doc_pairs) < 100:
        print("  ERROR: too few training pairs, skipping 3B")
        return None

    SEQ_LENGTHS = [128, 256, 512]

    length_results = {}

    for seq_len in SEQ_LENGTHS:
        print(f"\n{'='*60}")
        print(f"  Training + Eval at seq_len={seq_len}")
        print(f"{'='*60}")

        # Filter pairs long enough for this seq_len
        valid_pairs = [(a[:seq_len], b[:seq_len]) for a, b in doc_pairs
                       if len(a) >= seq_len and len(b) >= seq_len]
        print(f"  {len(valid_pairs)} pairs with both chunks >= {seq_len} tokens")

        if len(valid_pairs) < 100:
            print(f"  Skipping: too few pairs at this length")
            continue

        length_results[seq_len] = {}

        for name, build_encoder, WrapperClass in [
            ("PRISM-Simplified", lambda: build_simplified_prism(), PRISMForEmbedding),
            ("Transformer", lambda: build_transformer(), TransformerForEmbedding),
        ]:
            torch.manual_seed(42)
            random.seed(42)

            encoder = build_encoder().to(DEVICE)
            wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)
            n_params = sum(p.numel() for p in wrapper.parameters())

            try:
                losses, accs, train_time = train_contrastive_long(
                    wrapper, valid_pairs, n_steps, f"{name}@{seq_len}",
                    max_len=seq_len
                )

                # Evaluate retrieval at this length
                ret = evaluate_retrieval_at_length(wrapper, documents, seq_len)
                tp = measure_throughput(wrapper, seq_len)

                result = {
                    "train_time": train_time,
                    "final_loss": float(np.mean(losses[-100:])),
                    "params": n_params,
                }
                if ret is not None:
                    result.update(ret)
                result.update(tp)
                length_results[seq_len][name] = result

                mrr_str = f"MRR={ret['mrr']:.4f}" if ret else "MRR=N/A"
                print(f"  {name}: {mrr_str}  loss={result['final_loss']:.4f}  "
                      f"{tp['throughput_seq_per_sec']:.1f} seq/s  time={train_time:.1f}s")

            except RuntimeError as e:
                print(f"  {name}: FAILED at seq_len={seq_len} ({e})")
                length_results[seq_len][name] = {"failed": True, "error": str(e)}

    # Plot
    _plot_money_chart_v2(length_results)

    return length_results


def _plot_money_chart_v2(length_results):
    """Plot quality and throughput vs sequence length (train-at-length version)."""
    colors = {"PRISM-Simplified": "#2563EB", "Transformer": "#DC2626"}

    # Reorganize: {model_name: {seq_len: metrics}}
    model_results = {}
    for seq_len, by_model in length_results.items():
        for name, metrics in by_model.items():
            if isinstance(metrics, dict) and not metrics.get("failed"):
                model_results.setdefault(name, {})[seq_len] = metrics

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Real Data: Train + Eval at Each Length (CNN/DailyMail)",
                 fontsize=14, fontweight="bold")

    # MRR vs length
    ax = axes[0]
    for name, by_len in model_results.items():
        lens = sorted([l for l in by_len if "mrr" in by_len[l]])
        mrrs = [by_len[l]["mrr"] for l in lens]
        if lens:
            ax.plot(lens, mrrs, "o-", color=colors.get(name, "#666"), linewidth=2,
                    markersize=8, label=name)
            for l, m in zip(lens, mrrs):
                ax.annotate(f"{m:.3f}", (l, m), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("MRR")
    ax.set_title("Retrieval Quality")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput vs length
    ax = axes[1]
    for name, by_len in model_results.items():
        lens = sorted([l for l in by_len if "throughput_seq_per_sec" in by_len[l]])
        tps = [by_len[l]["throughput_seq_per_sec"] for l in lens]
        if lens:
            ax.plot(lens, tps, "o-", color=colors.get(name, "#666"), linewidth=2,
                    markersize=8, label=name)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Sequences/sec")
    ax.set_title("Encoding Throughput")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training time vs length
    ax = axes[2]
    for name, by_len in model_results.items():
        lens = sorted([l for l in by_len if "train_time" in by_len[l]])
        times = [by_len[l]["train_time"] for l in lens]
        if lens:
            ax.plot(lens, times, "o-", color=colors.get(name, "#666"), linewidth=2,
                    markersize=8, label=name)
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Cost (2000 steps)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "real_data_money_chart.png", bbox_inches="tight", dpi=150)
    print(f"\n  Saved: {RESULTS_DIR / 'real_data_money_chart.png'}")
    plt.close("all")


# ---------------------------------------------------------------------------
# 3C: Inference scaling on real text
# ---------------------------------------------------------------------------

def run_phase_3c():
    print("\n" + "=" * 70)
    print("PHASE 3C: Inference Scaling on Real Vocab")
    print("=" * 70)

    SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    results = {}
    for name, build_encoder, WrapperClass in [
        ("PRISM-Simplified", lambda: build_simplified_prism(), PRISMForEmbedding),
        ("Transformer", lambda: build_transformer(), TransformerForEmbedding),
    ]:
        encoder = build_encoder().to(DEVICE)
        wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)
        results[name] = {}

        for seq_len in SEQ_LENGTHS:
            try:
                tp = measure_throughput(wrapper, seq_len)
                results[name][seq_len] = tp
                print(f"  {name} @ {seq_len}: {tp['throughput_seq_per_sec']:.1f} seq/s  "
                      f"({tp['latency_ms']:.1f} ms)")
            except RuntimeError as e:
                print(f"  {name} @ {seq_len}: FAILED ({e})")
                break

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: REAL DATA EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print("=" * 70)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load data
    train_pairs = load_nli_pairs(tokenizer, max_pairs=50000, max_len=128)
    sts_examples = load_sts_benchmark(tokenizer, max_len=128)

    # Phase 3A: STS-B quality
    results_3a = run_phase_3a(tokenizer, train_pairs, sts_examples, n_steps=2000)

    # Phase 3B: Length-controlled retrieval (train at each length)
    print("\n  Loading long documents for length-controlled training + eval...")
    documents = load_long_documents(tokenizer, n_docs=10000, min_len=128)
    results_3b = None
    if len(documents) >= 200:
        results_3b = run_phase_3b(tokenizer, documents, n_steps=2000)
    else:
        print("  Skipping Phase 3B: insufficient long documents")

    # Phase 3C: Scaling confirmation
    results_3c = run_phase_3c()

    # Save all results
    all_results = {
        "phase_3a": results_3a,
        "phase_3b": results_3b,
        "phase_3c": results_3c,
        "config": {
            "device": DEVICE,
            "tokenizer": TOKENIZER_NAME,
            "vocab_size": REAL_VOCAB_SIZE,
            "batch_size": BATCH_SIZE,
        }
    }
    with open(RESULTS_DIR / "real_data_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    if results_3a:
        print("\n3A — STS-Benchmark (Spearman correlation):")
        for name, r in results_3a.items():
            print(f"  {name}: {r['sts_spearman']:.4f}")

    if results_3b:
        print("\n3B — Retrieval quality vs length:")
        for name, results_by_len in results_3b.items():
            lens = sorted([l for l in results_by_len if "mrr" in results_by_len[l]])
            if lens:
                print(f"  {name}:")
                for l in lens:
                    print(f"    seq_len={l}: MRR={results_by_len[l]['mrr']:.4f}")

    if results_3c:
        print("\n3C — Throughput scaling:")
        for name, results_by_len in results_3c.items():
            lens = sorted(results_by_len.keys())
            if lens:
                print(f"  {name}:")
                for l in lens:
                    print(f"    seq_len={l}: {results_by_len[l]['throughput_seq_per_sec']:.1f} seq/s")
