"""
Hybrid PRISM v2 Experiments: Needle-in-a-Haystack + Scaling Curves

Tests whether memory-augmented PRISM can perform content-based retrieval
that all-slow PRISM (mean pooling) cannot. Self-contained benchmark.

Experiment 1: NIAH at fixed sequence lengths (2K and 8K)
Experiment 2: NIAH scaling curves across 512-16K
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
from datasets import load_dataset

from hybrid_prism import (
    HybridPRISMEncoder, HybridPRISMForEmbedding,
    hybrid_prism_small, hybrid_prism_medium,
)
from prism import PRISMLayer, prism_small, PRISMForEmbedding
from baseline_transformer import transformer_small, TransformerForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from benchmark_loco import (
    DEVICE, TOKENIZER_NAME, REAL_VOCAB_SIZE,
    _collate_ids, _get_batch_config,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results") / "hybrid_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# NIAH fact generation
FIRST_NAMES = [
    "James", "Sarah", "Robert", "Maria", "David", "Anna", "Michael", "Laura",
    "Thomas", "Helen", "Richard", "Emma", "Charles", "Julia", "George",
    "Diana", "Edward", "Claire", "William", "Sophie", "Daniel", "Olivia",
    "Andrew", "Rachel", "Joseph", "Hannah", "Samuel", "Grace", "Patrick",
    "Alice", "Stephen", "Eleanor", "Peter", "Margaret", "Henry", "Victoria",
    "Martin", "Catherine", "Philip", "Sophia", "Anthony", "Isabel", "Frank",
    "Louise", "Arthur", "Evelyn", "Simon", "Charlotte", "Luke", "Florence",
    "Nathan", "Beatrice", "Marcus", "Caroline", "Vincent", "Amelia", "Walter",
    "Iris", "Frederick", "Lillian", "Raymond", "Nora", "Oscar", "Hazel",
    "Leonard", "Ada", "Eugene", "Vera", "Clarence", "Mabel", "Howard",
    "Ruth", "Albert", "Dorothy", "Harold", "Edith", "Francis", "Ethel",
    "Theodore", "Gladys", "Bernard", "Mildred", "Douglas", "Agnes",
    "Kenneth", "Constance", "Clifford", "Blanche", "Gordon", "Hilda",
]

LAST_NAMES = [
    "Wilson", "Chen", "Murphy", "Patel", "Santos", "Novak", "Fischer",
    "Kowalski", "Bergman", "Torres", "Nakamura", "Singh", "Okafor",
    "Lindgren", "Morales", "Petrov", "Reeves", "Ashworth", "Brennan",
    "Caldwell", "Dalton", "Everett", "Fielding", "Garrett", "Hammond",
    "Jennings", "Kimball", "Lawson", "Mercer", "Norwood", "Osborne",
    "Prescott", "Quinlan", "Radford", "Shelton", "Thatcher", "Underwood",
    "Vaughn", "Whitmore", "Yates", "Zeller", "Abbott", "Barlow",
    "Clifton", "Dunbar", "Emerson", "Forster", "Griffith", "Hartley",
    "Ingram", "Jarvis", "Keating", "Langley", "Marsden", "Newell",
    "Ogden", "Penrose", "Ramsay", "Sinclair", "Talbot", "Upton",
    "Vickers", "Wainwright", "Yardley", "Archer", "Blackwell", "Crosby",
    "Delaney", "Eastwood", "Fairfax", "Graves", "Holden", "Irving",
    "Kingsley", "Lister", "Montague", "Neville", "Oldham", "Pemberton",
]

DIGIT_WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
]


# ---------------------------------------------------------------------------
# NIAH Data Generation
# ---------------------------------------------------------------------------

def load_filler_tokens(tokenizer, max_tokens=5_000_000):
    """Load wikitext-103 and tokenize into a flat array for document filler.

    Tokenizes in chunks to avoid choking on a single giant string.
    """
    print("Loading wikitext-103 for NIAH filler text...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

    all_ids = []
    batch = []
    batch_chars = 0
    CHUNK_SIZE = 100_000  # characters per tokenizer call

    for row in ds:
        text = row["text"]
        if not text.strip() or text.startswith(" ="):
            continue
        batch.append(text)
        batch_chars += len(text)

        if batch_chars >= CHUNK_SIZE:
            ids = tokenizer.encode(" ".join(batch), add_special_tokens=False)
            all_ids.extend(ids)
            batch = []
            batch_chars = 0
            if len(all_ids) >= max_tokens:
                break

    # Flush remaining
    if batch and len(all_ids) < max_tokens:
        ids = tokenizer.encode(" ".join(batch), add_special_tokens=False)
        all_ids.extend(ids)

    all_ids = all_ids[:max_tokens]
    print(f"  Filler pool: {len(all_ids):,} tokens")
    return all_ids


def generate_niah_facts(n, seed=42):
    """Generate n unique (needle_sentence, query_sentence) pairs.

    Uses common English names and spelled-out digit codes to avoid
    rare-token shortcuts. Each fact uses a unique (first, last) combo.
    """
    rng = random.Random(seed)
    used_names = set()
    facts = []

    for _ in range(n):
        # Ensure unique name combinations
        for _attempt in range(1000):
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            if (first, last) not in used_names:
                used_names.add((first, last))
                break

        code = " ".join(rng.choice(DIGIT_WORDS) for _ in range(4))
        needle = f"The password assigned to Dr. {first} {last} is {code}."
        query = f"What password was assigned to Dr. {first} {last}?"
        facts.append((needle, query))

    return facts


def generate_niah_dataset(
    tokenizer,
    filler_tokens,
    n_docs,
    max_len,
    seed=42,
):
    """Generate NIAH documents (filler + needle) and matching queries.

    Each document is max_len tokens: filler text from wikitext-103
    with a single needle sentence inserted at a random position
    (uniform over 10%-90% of document length).

    Returns:
        documents: list of n_docs token ID lists, each length max_len
        queries: list of n_docs token ID lists, each ~15 tokens
    """
    facts = generate_niah_facts(n_docs, seed=seed)
    rng = random.Random(seed + 1)  # separate rng for positions

    documents = []
    queries = []

    for i, (needle_text, query_text) in enumerate(facts):
        needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)

        # How much filler we need
        filler_needed = max_len - len(needle_ids)
        if filler_needed < 2:
            # Needle is too long for this max_len — truncate needle
            needle_ids = needle_ids[:max_len - 2]
            filler_needed = 2

        # Sample filler from the pool
        max_start = len(filler_tokens) - filler_needed - 1
        start = rng.randint(0, max(0, max_start))
        filler = filler_tokens[start:start + filler_needed]
        # Pad if filler pool is somehow too short
        while len(filler) < filler_needed:
            filler = filler + filler_tokens[:filler_needed - len(filler)]

        # Insert needle at random position (10%-90% of max_len)
        min_pos = max(1, int(0.1 * max_len))
        max_pos = min(filler_needed - 1, int(0.9 * max_len))
        if max_pos <= min_pos:
            max_pos = min_pos + 1
        insert_pos = rng.randint(min_pos, max_pos)

        doc_ids = filler[:insert_pos] + needle_ids + filler[insert_pos:]
        doc_ids = doc_ids[:max_len]  # ensure exact length
        assert len(doc_ids) == max_len, f"doc {i}: {len(doc_ids)} != {max_len}"

        documents.append(doc_ids)
        queries.append(query_ids)

    print(f"  Generated {n_docs} NIAH docs (max_len={max_len}, "
          f"needle ~{len(needle_ids)} tokens = {100*len(needle_ids)/max_len:.1f}% of doc)")
    return documents, queries


# ---------------------------------------------------------------------------
# Model Builders
# ---------------------------------------------------------------------------

def build_allslow_prism(vocab_size=REAL_VOCAB_SIZE, max_len=8192, n_layers=6, **kwargs):
    """All-slow PRISM: lambda=0.99, no interference, mean pooling."""
    model = prism_small(vocab_size=vocab_size, max_len=max_len, n_layers=n_layers, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
        layer.recurrence.lambdas.fill_(0.99)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model


def build_hybrid(vocab_size=REAL_VOCAB_SIZE, max_len=8192, config="medium", **kwargs):
    """HybridPRISM with memory bank."""
    if config == "small":
        return hybrid_prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    else:
        return hybrid_prism_medium(vocab_size=vocab_size, max_len=max_len, **kwargs)


def build_transformer(vocab_size=REAL_VOCAB_SIZE, max_len=8192, **kwargs):
    """Parameter-matched Transformer baseline."""
    return transformer_small(vocab_size=vocab_size, max_len=max_len, **kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_niah(
    model_wrapper,
    train_docs,
    train_queries,
    n_steps,
    model_name,
    max_len,
    micro_batch=16,
    lr=3e-4,
    log_interval=100,
    eval_fn=None,
    eval_interval=1000,
    memory_warmup_steps=0,
):
    """Train with InfoNCE on NIAH query-document pairs.

    Args:
        train_docs: list of token ID lists (documents with needles)
        train_queries: list of token ID lists (queries about needles)
        eval_fn: optional callable() -> dict for periodic evaluation
        eval_interval: evaluate every N steps
        memory_warmup_steps: freeze memory modules for first N steps (HybridPRISM only)
    """
    # Detect HybridPRISM and freeze memory during warmup
    from hybrid_prism import HybridPRISMEncoder
    encoder = getattr(model_wrapper, "encoder", None)
    has_memory = isinstance(encoder, HybridPRISMEncoder)

    if has_memory and memory_warmup_steps > 0:
        encoder.set_memory_enabled(False)
        print(f"  Memory disabled for first {memory_warmup_steps} steps")

    # Only optimize params that currently require grad
    optimizer = torch.optim.AdamW(
        [p for p in model_wrapper.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )

    n_pairs = len(train_docs)
    losses = []
    checkpoints = []
    model_wrapper.train()

    print(f"\nTraining {model_name} ({n_steps} steps, max_len={max_len}, "
          f"micro_batch={micro_batch}, lr={lr:.0e}, {n_pairs} pairs)...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Unfreeze memory after warmup
        if has_memory and memory_warmup_steps > 0 and step == memory_warmup_steps:
            encoder.set_memory_enabled(True)
            # Add memory params to optimizer now that they require grad
            optimizer.add_param_group({
                "params": list(encoder.memory_params()),
                "lr": lr,
                "weight_decay": 0.01,
            })
            print(f"  [{model_name}] Memory enabled at step {step}")

        optimizer.zero_grad()

        # Sample batch
        indices = random.choices(range(n_pairs), k=micro_batch)
        q_ids_list = [train_queries[i] for i in indices]
        d_ids_list = [train_docs[i] for i in indices]

        q_ids, q_mask = _collate_ids(q_ids_list, max_len)
        d_ids, d_mask = _collate_ids(d_ids_list, max_len)
        q_ids, q_mask = q_ids.to(DEVICE), q_mask.to(DEVICE)
        d_ids, d_mask = d_ids.to(DEVICE), d_mask.to(DEVICE)

        result = model_wrapper(q_ids, q_mask, d_ids, d_mask)
        loss = result["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % log_interval == 0:
            recent_loss = np.mean(losses[-log_interval:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  "
                  f"loss={recent_loss:.4f}  ({elapsed:.1f}s)")

        # Periodic eval
        if eval_fn is not None and (step + 1) % eval_interval == 0:
            model_wrapper.eval()
            metrics = eval_fn()
            checkpoints.append({"step": step + 1, **metrics})
            print(f"  [{model_name}] Eval @{step+1}: "
                  f"MRR={metrics['mrr']:.4f} R@1={metrics['recall_at_1']:.4f}")
            model_wrapper.train()

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")

    return losses, checkpoints, total_time


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_niah(
    model_wrapper,
    eval_docs,
    eval_queries,
    max_len,
    batch_size=32,
    device=DEVICE,
):
    """Evaluate NIAH retrieval: encode queries and docs, compute MRR + Recall@K.

    Each query_i should match document_i (identity mapping).

    Returns:
        dict with mrr, recall_at_1, recall_at_5, recall_at_10
    """
    model_wrapper.eval()
    n = len(eval_docs)

    # Encode documents
    doc_embs = _encode_list(model_wrapper, eval_docs, max_len, batch_size, device)

    # Encode queries
    query_embs = _encode_list(model_wrapper, eval_queries, max_len, batch_size, device)

    # Similarity matrix: (n_queries, n_docs)
    sim = torch.matmul(query_embs, doc_embs.T)

    # Compute metrics
    ranks = []
    for i in range(n):
        # Rank of the correct document for query i
        sorted_indices = sim[i].argsort(descending=True).tolist()
        rank = sorted_indices.index(i) + 1  # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    recall_at_1 = float(np.mean(ranks <= 1))
    recall_at_5 = float(np.mean(ranks <= 5))
    recall_at_10 = float(np.mean(ranks <= 10))

    return {
        "mrr": mrr,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


@torch.no_grad()
def _encode_list(model_wrapper, token_ids_list, max_len, batch_size, device):
    """Encode a list of token ID sequences into normalized embeddings."""
    all_embs = []
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i + batch_size]
        ids, masks = _collate_ids(batch, max_len)
        ids, masks = ids.to(device), masks.to(device)
        emb = model_wrapper.encode(ids, masks)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Experiment 1: NIAH at fixed sequence lengths
# ---------------------------------------------------------------------------

def run_experiment_niah(
    max_len=2048,
    n_train=5000,
    n_eval=500,
    n_steps=3000,
    micro_batch=16,
    lr=3e-4,
    seed=42,
    only_models=None,
    memory_warmup_steps=500,
):
    """Run NIAH experiment at a single sequence length.

    Trains and evaluates: all-slow PRISM (6L), HybridPRISM (Config B), Transformer (8L).
    Also runs 12-layer all-slow PRISM as a depth control.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: Needle-in-a-Haystack @ max_len={max_len}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Generate data
    filler = load_filler_tokens(tokenizer)
    print("\nGenerating training set...")
    train_docs, train_queries = generate_niah_dataset(
        tokenizer, filler, n_train, max_len, seed=seed
    )
    print("Generating eval set...")
    eval_docs, eval_queries = generate_niah_dataset(
        tokenizer, filler, n_eval, max_len, seed=seed + 10000
    )

    all_results = {}

    # Define models to test
    models = [
        ("AllSlow-PRISM-6L", lambda: build_allslow_prism(max_len=max_len, n_layers=6),
         PRISMForEmbedding),
        ("AllSlow-PRISM-12L", lambda: build_allslow_prism(max_len=max_len, n_layers=12),
         PRISMForEmbedding),
        ("HybridPRISM-12L", lambda: build_hybrid(max_len=max_len, config="medium"),
         HybridPRISMForEmbedding),
        ("Transformer-8L", lambda: build_transformer(max_len=max_len),
         TransformerForEmbedding),
    ]

    # Filter models if requested
    if only_models:
        models = [(n, b, w) for n, b, w in models if n in only_models]

    for model_name, build_fn, WrapperClass in models:
        print(f"\n--- {model_name} ---")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            encoder = build_fn().to(DEVICE)
            n_params = sum(p.numel() for p in encoder.parameters())
            print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

            wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)

            # Eval function for checkpointing
            def make_eval_fn(w, ed, eq, ml):
                def fn():
                    return evaluate_niah(w, ed, eq, ml, device=DEVICE)
                return fn

            eval_fn = make_eval_fn(wrapper, eval_docs, eval_queries, max_len)

            losses, checkpoints, train_time = train_niah(
                wrapper, train_docs, train_queries,
                n_steps=n_steps, model_name=model_name, max_len=max_len,
                micro_batch=micro_batch, lr=lr,
                eval_fn=eval_fn, eval_interval=1000,
                memory_warmup_steps=memory_warmup_steps,
            )

            # Final evaluation
            wrapper.eval()
            final_metrics = evaluate_niah(
                wrapper, eval_docs, eval_queries, max_len, device=DEVICE
            )

            result = {
                "model": model_name,
                "max_len": max_len,
                "n_params": n_params,
                "n_steps": n_steps,
                "lr": lr,
                "final_loss": float(np.mean(losses[-100:])),
                "train_time": train_time,
                "final_metrics": final_metrics,
                "checkpoints": checkpoints,
            }
            all_results[model_name] = result

            print(f"\n  {model_name} FINAL: MRR={final_metrics['mrr']:.4f}  "
                  f"R@1={final_metrics['recall_at_1']:.4f}  "
                  f"R@5={final_metrics['recall_at_5']:.4f}  "
                  f"loss={result['final_loss']:.4f}  time={train_time:.0f}s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {model_name}: OOM at max_len={max_len}")
                all_results[model_name] = {"model": model_name, "error": "OOM"}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise
        finally:
            # Cleanup
            for var in ["encoder", "wrapper"]:
                if var in locals():
                    del locals()[var]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    results_path = RESULTS_DIR / f"niah_maxlen{max_len}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # Plot
    _plot_niah_results(all_results, max_len)

    return all_results


# ---------------------------------------------------------------------------
# Experiment 2: NIAH Scaling Curves
# ---------------------------------------------------------------------------

def run_experiment_scaling(
    lengths=(512, 1024, 2048, 4096, 8192),
    n_train=3000,
    n_eval=500,
    n_steps=3000,
    micro_batch=16,
    lr=3e-4,
    seed=42,
    only_models=None,
):
    """Run NIAH at multiple sequence lengths and plot scaling curves.

    Trains a fresh model at each length.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: NIAH Scaling Curves — lengths={lengths}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    filler = load_filler_tokens(tokenizer)

    # Models to test at each length
    model_configs = [
        ("AllSlow-PRISM-6L", lambda ml: build_allslow_prism(max_len=ml, n_layers=6),
         PRISMForEmbedding),
        ("HybridPRISM-12L", lambda ml: build_hybrid(max_len=ml, config="medium"),
         HybridPRISMForEmbedding),
        ("Transformer-8L", lambda ml: build_transformer(max_len=ml),
         TransformerForEmbedding),
    ]

    # Filter models if requested
    if only_models:
        model_configs = [(n, b, w) for n, b, w in model_configs if n in only_models]

    scaling_results = {name: {} for name, _, _ in model_configs}

    for max_len in lengths:
        print(f"\n{'='*50}")
        print(f"  Length: {max_len}")
        print(f"{'='*50}")

        # Generate data for this length
        train_docs, train_queries = generate_niah_dataset(
            tokenizer, filler, n_train, max_len, seed=seed
        )
        eval_docs, eval_queries = generate_niah_dataset(
            tokenizer, filler, n_eval, max_len, seed=seed + 10000
        )

        for model_name, build_fn, WrapperClass in model_configs:
            print(f"\n  --- {model_name} @ {max_len} ---")
            torch.manual_seed(seed)
            random.seed(seed)

            try:
                encoder = build_fn(max_len).to(DEVICE)
                n_params = sum(p.numel() for p in encoder.parameters())
                wrapper = WrapperClass(encoder, temperature=0.05).to(DEVICE)

                losses, _, train_time = train_niah(
                    wrapper, train_docs, train_queries,
                    n_steps=n_steps, model_name=f"{model_name}@{max_len}",
                    max_len=max_len, micro_batch=micro_batch, lr=lr,
                    eval_fn=None,
                )

                wrapper.eval()
                metrics = evaluate_niah(
                    wrapper, eval_docs, eval_queries, max_len, device=DEVICE
                )

                scaling_results[model_name][max_len] = {
                    "metrics": metrics,
                    "final_loss": float(np.mean(losses[-100:])),
                    "train_time": train_time,
                    "n_params": n_params,
                }

                print(f"  {model_name}@{max_len}: MRR={metrics['mrr']:.4f}  "
                      f"R@1={metrics['recall_at_1']:.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {model_name}@{max_len}: OOM")
                    scaling_results[model_name][max_len] = {"error": "OOM"}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            finally:
                for var in ["encoder", "wrapper"]:
                    if var in locals():
                        del locals()[var]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save results
    results_path = RESULTS_DIR / "niah_scaling.json"
    with open(results_path, "w") as f:
        json.dump(scaling_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    _plot_scaling_curves(scaling_results, lengths)

    return scaling_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "AllSlow-PRISM-6L": "#2563EB",
    "AllSlow-PRISM-12L": "#60A5FA",
    "HybridPRISM-12L": "#DC2626",
    "Transformer-8L": "#16A34A",
}

MARKERS = {
    "AllSlow-PRISM-6L": "o",
    "AllSlow-PRISM-12L": "s",
    "HybridPRISM-12L": "D",
    "Transformer-8L": "^",
}


def _plot_niah_results(results, max_len):
    """Bar chart of NIAH results at a single sequence length."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = [m for m in results if "error" not in results[m]]
    mrrs = [results[m]["final_metrics"]["mrr"] for m in models]
    r1s = [results[m]["final_metrics"]["recall_at_1"] for m in models]
    colors = [COLORS.get(m, "#888888") for m in models]

    axes[0].bar(range(len(models)), mrrs, color=colors)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("MRR")
    axes[0].set_title(f"NIAH MRR @ max_len={max_len}")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(range(len(models)), r1s, color=colors)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Recall@1")
    axes[1].set_title(f"NIAH Recall@1 @ max_len={max_len}")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    path = RESULTS_DIR / f"niah_maxlen{max_len}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"Saved: {path}")


def _plot_scaling_curves(scaling_results, lengths):
    """Line plot of MRR and Recall@1 vs sequence length for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, len_results in scaling_results.items():
        valid_lens = sorted(
            l for l in len_results if "error" not in len_results[l]
        )
        if not valid_lens:
            continue

        mrrs = [len_results[l]["metrics"]["mrr"] for l in valid_lens]
        r1s = [len_results[l]["metrics"]["recall_at_1"] for l in valid_lens]
        color = COLORS.get(model_name, "#888888")
        marker = MARKERS.get(model_name, "o")

        axes[0].plot(valid_lens, mrrs, marker=marker, color=color,
                     label=model_name, linewidth=2, markersize=8)
        axes[1].plot(valid_lens, r1s, marker=marker, color=color,
                     label=model_name, linewidth=2, markersize=8)

    for ax, metric in zip(axes, ["MRR", "Recall@1"]):
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(metric)
        ax.set_title(f"NIAH {metric} vs Sequence Length")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(l) for l in lengths])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = RESULTS_DIR / "niah_scaling_curves.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid PRISM v2 Experiments")
    parser.add_argument("experiment", choices=["niah", "scaling", "all"],
                        help="Which experiment to run")
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Sequence length for single NIAH experiment")
    parser.add_argument("--n-steps", type=int, default=3000,
                        help="Training steps")
    parser.add_argument("--micro-batch", type=int, default=16,
                        help="Micro batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all)")
    parser.add_argument("--memory-warmup", type=int, default=500,
                        help="Steps to freeze memory modules (HybridPRISM only, default: 500)")
    args = parser.parse_args()

    only_models = args.models.split(",") if args.models else None

    if args.experiment in ("niah", "all"):
        run_experiment_niah(
            max_len=args.max_len,
            n_steps=args.n_steps,
            micro_batch=args.micro_batch,
            lr=args.lr,
            seed=args.seed,
            only_models=only_models,
            memory_warmup_steps=args.memory_warmup,
        )

    if args.experiment in ("scaling", "all"):
        run_experiment_scaling(
            n_steps=args.n_steps,
            micro_batch=args.micro_batch,
            lr=args.lr,
            seed=args.seed,
            only_models=only_models,
        )
