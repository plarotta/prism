"""
PRISM vs Transformer: Embedding Quality Comparison

Trains both models from scratch on a contrastive sentence embedding task
using synthetic paraphrase pairs, then evaluates retrieval quality.

This demonstrates that PRISM doesn't just scale better — it actually learns
competitive embeddings.
"""

import json
import random
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import prism_small, PRISMForEmbedding
from baseline_transformer import transformer_small, TransformerForEmbedding

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
VOCAB_SIZE = 1024       # small vocab for synthetic data
D_E = 384
BATCH_SIZE = 32
N_TRAIN_STEPS = 400
LR = 3e-4
SEQ_LEN_Q = 64
SEQ_LEN_P = 96
N_EVAL_QUERIES = 500
CORPUS_SIZE = 5000
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Synthetic dataset: structured paraphrase pairs
# ---------------------------------------------------------------------------

class SyntheticEmbeddingDataset:
    """Generates challenging query-positive pairs for contrastive learning.

    Difficulty is controlled by:
    - Many topics with overlapping distributions (hard negatives)
    - Short topic prefix (2 tokens) buried in noise
    - High noise rate (40% token replacement) in positives
    - Topic-specific token distributions that overlap heavily between similar topics

    The model must learn to embed sequences with the same underlying topic
    nearby, despite heavy surface variation.
    """

    def __init__(self, vocab_size: int, n_topics: int = 200, topic_len: int = 2):
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.topic_len = topic_len

        # Each topic has a fixed prefix pattern
        self.topic_prefixes = {}
        # Topic-specific token distributions (with overlap between nearby topics)
        self.topic_dists = {}
        for t in range(n_topics):
            self.topic_prefixes[t] = torch.randint(2, vocab_size, (topic_len,))
            dist = torch.ones(vocab_size)
            # Boost a topic-specific range, but make ranges overlap between nearby topics
            range_start = (t * 5) % (vocab_size - 30) + 2
            dist[range_start:range_start + 30] = 3.0
            # Add a secondary peak for extra complexity
            range2 = ((t * 7) + 100) % (vocab_size - 20) + 2
            dist[range2:range2 + 20] = 2.0
            self.topic_dists[t] = dist / dist.sum()

    def generate_pair(self, seq_len_q: int, seq_len_p: int):
        """Generate a (query, positive) pair from the same topic."""
        topic = random.randint(0, self.n_topics - 1)
        prefix = self.topic_prefixes[topic]
        base_dist = self.topic_dists[topic]

        q_body = torch.multinomial(base_dist, seq_len_q - self.topic_len, replacement=True)
        query = torch.cat([prefix, q_body])

        # Positive: same prefix + similar content with HIGH noise
        p_body_base = q_body[torch.randperm(len(q_body))]
        # Replace ~40% of tokens (much harder than 20%)
        n_replace = max(1, int(0.4 * len(p_body_base)))
        replace_idx = torch.randperm(len(p_body_base))[:n_replace]
        p_body_base[replace_idx] = torch.multinomial(base_dist, n_replace, replacement=True)

        # Pad or trim to seq_len_p
        if len(p_body_base) + self.topic_len < seq_len_p:
            extra = torch.multinomial(base_dist, seq_len_p - self.topic_len - len(p_body_base), replacement=True)
            p_body = torch.cat([p_body_base, extra])
        else:
            p_body = p_body_base[:seq_len_p - self.topic_len]

        positive = torch.cat([prefix, p_body])

        return query[:seq_len_q], positive[:seq_len_p], topic

    def generate_batch(self, batch_size: int, seq_len_q: int, seq_len_p: int):
        queries, positives, topics = [], [], []
        for _ in range(batch_size):
            q, p, t = self.generate_pair(seq_len_q, seq_len_p)
            queries.append(q)
            positives.append(p)
            topics.append(t)
        return (
            torch.stack(queries),
            torch.stack(positives),
            torch.tensor(topics),
        )

    def generate_corpus_and_queries(self, n_queries: int, corpus_size: int,
                                     seq_len_q: int, seq_len_p: int):
        """Generate a retrieval evaluation set."""
        queries = []
        positives = []
        query_topics = []
        corpus_topics = []

        # Generate query-positive pairs
        for _ in range(n_queries):
            q, p, t = self.generate_pair(seq_len_q, seq_len_p)
            queries.append(q)
            positives.append(p)
            query_topics.append(t)

        # Build corpus: positives + random distractors
        corpus = list(positives)
        corpus_topics = list(query_topics)
        n_distractors = corpus_size - n_queries
        for _ in range(n_distractors):
            topic = random.randint(0, self.n_topics - 1)
            _, p, _ = self.generate_pair(seq_len_q, seq_len_p)
            corpus.append(p)
            corpus_topics.append(topic)

        return (
            torch.stack(queries),
            torch.stack(corpus),
            torch.tensor(query_topics),
            torch.tensor(corpus_topics),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model_wrapper, dataset, n_steps, model_name):
    """Train a model with InfoNCE loss and return training curves."""
    optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    losses = []
    accuracies = []
    model_wrapper.train()

    print(f"\nTraining {model_name}...")
    t0 = time.perf_counter()

    for step in range(n_steps):
        q, p, _ = dataset.generate_batch(BATCH_SIZE, SEQ_LEN_Q, SEQ_LEN_P)
        q, p = q.to(DEVICE), p.to(DEVICE)
        q_mask = torch.ones_like(q)
        p_mask = torch.ones_like(p)

        result = model_wrapper(q, q_mask, p, p_mask)
        loss = result["loss"]
        acc = result["accuracy"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if (step + 1) % 50 == 0:
            recent_loss = np.mean(losses[-50:])
            recent_acc = np.mean(accuracies[-50:])
            elapsed = time.perf_counter() - t0
            print(f"  [{model_name}] Step {step+1}/{n_steps}  loss={recent_loss:.4f}  acc={recent_acc:.3f}  ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"  [{model_name}] Training complete in {total_time:.1f}s")

    return losses, accuracies, total_time


# ---------------------------------------------------------------------------
# Evaluation: retrieval metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(model_wrapper, queries, corpus, query_topics, corpus_topics):
    """Compute retrieval metrics: MRR, Recall@1, Recall@5, Recall@10."""
    model_wrapper.eval()

    # Encode queries
    q_embs = []
    for i in range(0, len(queries), BATCH_SIZE):
        batch = queries[i:i+BATCH_SIZE].to(DEVICE)
        mask = torch.ones_like(batch)
        emb = model_wrapper.encode(batch, mask)
        q_embs.append(emb.cpu())
    q_embs = torch.cat(q_embs, dim=0)  # (N_q, d_e)

    # Encode corpus
    c_embs = []
    for i in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[i:i+BATCH_SIZE].to(DEVICE)
        mask = torch.ones_like(batch)
        emb = model_wrapper.encode(batch, mask)
        c_embs.append(emb.cpu())
    c_embs = torch.cat(c_embs, dim=0)  # (N_c, d_e)

    # Compute similarity matrix
    sim = torch.matmul(q_embs, c_embs.T)  # (N_q, N_c)

    # Ground truth: query i's positive is corpus[i] (by construction)
    n_queries = len(queries)
    ranks = []
    for i in range(n_queries):
        sorted_indices = sim[i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    recall_1 = np.mean(ranks <= 1)
    recall_5 = np.mean(ranks <= 5)
    recall_10 = np.mean(ranks <= 10)

    # Topic-level accuracy: same-topic queries should cluster
    # For each query, check if the nearest corpus item has the same topic
    nearest = sim.argmax(dim=1)
    topic_match = (query_topics[torch.arange(n_queries)] == corpus_topics[nearest]).float().mean().item()

    return {
        "mrr": mrr,
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "topic_accuracy": topic_match,
        "mean_rank": np.mean(ranks),
        "median_rank": np.median(ranks),
    }


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_quality_comparison():
    print("=" * 60)
    print("QUALITY COMPARISON: PRISM vs Transformer")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Vocab: {VOCAB_SIZE}, Batch: {BATCH_SIZE}, Steps: {N_TRAIN_STEPS}")
    print(f"Query len: {SEQ_LEN_Q}, Passage len: {SEQ_LEN_P}")
    print()

    dataset = SyntheticEmbeddingDataset(VOCAB_SIZE)

    # Build models
    prism_enc = prism_small(vocab_size=VOCAB_SIZE).to(DEVICE)
    prism_wrapper = PRISMForEmbedding(prism_enc, temperature=0.05).to(DEVICE)

    trans_enc = transformer_small(vocab_size=VOCAB_SIZE).to(DEVICE)
    trans_wrapper = TransformerForEmbedding(trans_enc, temperature=0.05).to(DEVICE)

    prism_params = sum(p.numel() for p in prism_wrapper.parameters())
    trans_params = sum(p.numel() for p in trans_wrapper.parameters())
    print(f"PRISM params:       {prism_params:,}")
    print(f"Transformer params: {trans_params:,}")

    # Train both
    prism_losses, prism_accs, prism_time = train_model(prism_wrapper, dataset, N_TRAIN_STEPS, "PRISM")
    trans_losses, trans_accs, trans_time = train_model(trans_wrapper, dataset, N_TRAIN_STEPS, "Transformer")

    # Evaluate
    print("\nEvaluating retrieval quality...")
    queries, corpus, q_topics, c_topics = dataset.generate_corpus_and_queries(
        N_EVAL_QUERIES, CORPUS_SIZE, SEQ_LEN_Q, SEQ_LEN_P
    )

    prism_metrics = evaluate_retrieval(prism_wrapper, queries, corpus, q_topics, c_topics)
    trans_metrics = evaluate_retrieval(trans_wrapper, queries, corpus, q_topics, c_topics)

    print(f"\n{'Metric':<20} {'PRISM':>10} {'Transformer':>12}")
    print("-" * 45)
    for k in ["mrr", "recall@1", "recall@5", "recall@10", "topic_accuracy", "mean_rank"]:
        print(f"{k:<20} {prism_metrics[k]:>10.4f} {trans_metrics[k]:>12.4f}")
    print(f"{'train_time_s':<20} {prism_time:>10.1f} {trans_time:>12.1f}")

    results = {
        "prism": {"metrics": prism_metrics, "train_time": prism_time, "params": prism_params},
        "transformer": {"metrics": trans_metrics, "train_time": trans_time, "params": trans_params},
        "config": {
            "vocab_size": VOCAB_SIZE, "batch_size": BATCH_SIZE,
            "n_steps": N_TRAIN_STEPS, "lr": LR,
            "seq_len_q": SEQ_LEN_Q, "seq_len_p": SEQ_LEN_P,
            "n_eval_queries": N_EVAL_QUERIES, "corpus_size": CORPUS_SIZE,
        },
    }

    with open(RESULTS_DIR / "quality_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Plot training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Curves: PRISM vs Transformer", fontsize=14, fontweight="bold")

    window = 20

    # Loss
    ax = axes[0]
    prism_loss_smooth = np.convolve(prism_losses, np.ones(window)/window, mode='valid')
    trans_loss_smooth = np.convolve(trans_losses, np.ones(window)/window, mode='valid')
    ax.plot(prism_loss_smooth, color="#2563EB", linewidth=2, label="PRISM")
    ax.plot(trans_loss_smooth, color="#DC2626", linewidth=2, label="Transformer")
    ax.set_xlabel("Step")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Contrastive Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    prism_acc_smooth = np.convolve(prism_accs, np.ones(window)/window, mode='valid')
    trans_acc_smooth = np.convolve(trans_accs, np.ones(window)/window, mode='valid')
    ax.plot(prism_acc_smooth, color="#2563EB", linewidth=2, label="PRISM")
    ax.plot(trans_acc_smooth, color="#DC2626", linewidth=2, label="Transformer")
    ax.set_xlabel("Step")
    ax.set_ylabel("In-Batch Accuracy")
    ax.set_title("Contrastive Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'training_curves.png'}")

    # --- Retrieval metrics bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    metrics_to_plot = ["mrr", "recall@1", "recall@5", "recall@10", "topic_accuracy"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    prism_vals = [prism_metrics[k] for k in metrics_to_plot]
    trans_vals = [trans_metrics[k] for k in metrics_to_plot]

    bars1 = ax.bar(x - width/2, prism_vals, width, label="PRISM", color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + width/2, trans_vals, width, label="Transformer", color="#DC2626", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Quality: PRISM vs Transformer", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "quality_comparison.png", bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'quality_comparison.png'}")
    plt.close("all")

    return results


if __name__ == "__main__":
    run_quality_comparison()
