# PRISM Paper: Implementation & Development Plan

Implements the experiments from `PAPER_EXPERIMENT_PLAN.md`. This document
covers code structure, data pipelines, model implementations, evaluation
harnesses, logging, and artifact management.

---

## 1. File Structure

```
prism/
  # --- Existing (unchanged) ---
  prism.py                        # PRISMLayer, PRISMEncoder, PRISMForEmbedding
  baseline_transformer.py         # TransformerEncoder, TransformerForEmbedding
  benchmark_ablations.py          # MeanPooling, NoInterference

  # --- New: core models ---
  mamba_bidir.py                  # Bidirectional Mamba encoder baseline
  linear_rnn.py                   # Single-channel linear RNN ablation baseline

  # --- New: data ---
  data/
    msmarco.py                    # MS MARCO passage loader + hard negative mining
    loco_eval.py                  # LoCoV1 zero-shot evaluator (no training on eval data)
    beir_eval.py                  # BEIR 18-dataset evaluator
    longembed_eval.py             # LongEmbed 6-task evaluator

  # --- New: training + eval harness ---
  train_contrastive.py            # Unified contrastive training loop (all models)
  eval_harness.py                 # Unified eval dispatcher (LoCoV1, BEIR, LongEmbed)

  # --- New: experiment runners ---
  paper_exp1_controlled.py        # Exp 1: controlled architecture comparison
  paper_exp2_efficiency.py        # Exp 2: scaling curves
  paper_exp3_ablations.py         # Exp 3: component ablation study
  paper_exp4_longembed.py         # Exp 4: LongEmbed evaluation
  paper_exp5_beir.py              # Exp 5: BEIR evaluation
  paper_exp6_pretrain.py          # Exp 6: pretraining + zero-shot eval
  paper_exp7_scaleup.py           # Exp 7: 80M scale-up

  # --- New: logging + results ---
  paper_log.py                    # Structured logging utilities
  results/paper/                  # All paper experiment outputs
    exp1/                         # Per-experiment subdirectories
    exp2/
    ...
  PAPER_LOG.md                    # Fine-grained development log (updated as work progresses)
```

---

## 2. Logging & Data Saving

Every experiment must produce reproducible, inspectable artifacts. Two tiers:

### 2.1 Structured Run Logging (`paper_log.py`)

Each training run creates a run directory under `results/paper/`:

```
results/paper/exp1/prism_small_2048_run0/
  config.json          # Full hyperparameters + model config (frozen at start)
  train_log.jsonl      # One JSON line per log_interval steps
  checkpoints/         # Model checkpoints at eval intervals
    step_5000.pt
    step_10000.pt
    ...
  eval/                # Eval results per checkpoint
    step_5000_locov1.json
    step_10000_locov1.json
    ...
  final_metrics.json   # Best checkpoint metrics + summary
  stderr.log           # Captured stderr (warnings, CUDA errors, etc.)
```

**`config.json`** — frozen at run start, never modified:
```json
{
  "experiment": "exp1_controlled",
  "model_name": "prism_small",
  "model_config": {
    "d": 384, "n_layers": 6, "n_channels": 6,
    "decay_mode": "all_slow", "max_len": 8192,
    "vocab_size": 30522, "pooling": "mean"
  },
  "training": {
    "dataset": "msmarco_passage",
    "n_steps": 50000, "lr": 3e-4, "batch_size": 128,
    "micro_batch": 16, "grad_accum": 8,
    "max_len_train": 2048, "hard_negatives": 7,
    "warmup_fraction": 0.1, "weight_decay": 0.01,
    "grad_clip": 1.0, "temperature": 0.05, "seed": 42
  },
  "hardware": {
    "gpu": "auto-detected",
    "cuda_version": "auto-detected",
    "torch_version": "auto-detected"
  },
  "git_hash": "auto-captured",
  "timestamp": "2026-03-24T..."
}
```

**`train_log.jsonl`** — append-only, one line per log interval:
```json
{"step": 100, "loss": 2.45, "lr": 0.00012, "grad_norm": 0.83, "throughput_seqs_per_s": 42.1, "gpu_mem_mb": 8200, "elapsed_s": 120.5}
```

**`final_metrics.json`** — written at end of run:
```json
{
  "best_checkpoint": "step_40000",
  "best_locov1_avg": 0.72,
  "best_msmarco_mrr10": 0.31,
  "total_train_time_s": 14400,
  "total_params": 19200000,
  "locov1_per_task": {"summ_screen_fd": 0.65, ...},
  "converged": true
}
```

### 2.2 Human-Readable Log (`PAPER_LOG.md`)

Updated manually (or by the experiment runner) with high-level results and
decisions. Format matches existing `HYBRID_V2_LOG.md` style:

```markdown
## Exp 1a: PRISM vs Transformer at 128 tokens (2026-03-XX)

| Model | MS MARCO MRR@10 | LoCoV1 nDCG@10 | Train Time |
|-------|----------------|----------------|------------|
| PRISM-Small | ... | ... | ... |
| Transformer | ... | ... | ... |

**Takeaway:** ...
```

### 2.3 `paper_log.py` Implementation

```python
# Key functions:

def create_run_dir(experiment: str, model_name: str, run_id: int = 0) -> Path:
    """Create and return results/paper/{experiment}/{model_name}_run{id}/"""

def save_config(run_dir: Path, config: dict):
    """Save config.json with auto-detected hardware + git hash."""

def log_step(run_dir: Path, step: int, metrics: dict):
    """Append one line to train_log.jsonl."""

def save_checkpoint(run_dir: Path, model, optimizer, step: int):
    """Save model + optimizer state to checkpoints/step_{step}.pt"""

def load_checkpoint(run_dir: Path, step: int) -> dict:
    """Load a checkpoint."""

def save_eval_results(run_dir: Path, step: int, benchmark: str, results: dict):
    """Save eval/{step}_{benchmark}.json"""

def save_final_metrics(run_dir: Path, metrics: dict):
    """Save final_metrics.json"""

def capture_hardware_info() -> dict:
    """Return GPU name, CUDA version, torch version, etc."""
```

---

## 3. Data Pipelines

### 3.1 MS MARCO Passage Retrieval (`data/msmarco.py`) -- DONE

**Source:** `Tevatron/msmarco-passage` on HuggingFace (primary), `mteb/msmarco` (fallback).

**Contents (Tevatron):**
- ~400K training queries with positive passages + 30 BM25 hard negatives per query
- ~5.5M unique passages (inline with queries, not separate corpus)
- 5000-query dev split created from last entries

**Implementation:**

```python
class MSMARCODataset:
    """MS MARCO passage retrieval dataset with hard negative support."""

    def __init__(self, tokenizer, max_len=512, n_hard_negatives=7):
        ...

    def load(self):
        """Download and cache passages, queries, qrels, negatives."""
        # 1. Load passages (8.8M) — stream to avoid OOM
        # 2. Load queries (500K train, 6980 dev)
        # 3. Load qrels (query -> positive passage IDs)
        # 4. Load BM25 negatives (query -> list of hard neg passage IDs)
        #    Source: official MS MARCO BM25 negatives or
        #    sentence-transformers/msmarco-hard-negatives on HuggingFace

    def sample_batch(self, batch_size: int) -> dict:
        """Sample a training batch.

        Returns:
            {
                "query_ids": (B, max_q_len),
                "query_mask": (B, max_q_len),
                "pos_ids": (B, max_p_len),
                "pos_mask": (B, max_p_len),
                "neg_ids": (B, n_hard_neg, max_p_len),  # optional
                "neg_mask": (B, n_hard_neg, max_p_len),
            }
        """

    def get_dev_queries(self) -> list:
        """Return dev set for MRR@10 evaluation."""

    def get_passages_by_ids(self, pids: list) -> list:
        """Fetch passage token IDs by passage ID."""
```

**Hard Negative Strategy:**
- Start with BM25 negatives (available pre-computed)
- Cross-encoder re-scoring is a nice-to-have but not required for controlled
  comparison — all models see the same negatives, so it's a fair fight

**Memory management:**
- 8.8M passages won't fit in RAM as raw text. Options:
  1. Pre-tokenize and store as memory-mapped numpy arrays (recommended)
  2. Stream from disk with an LRU cache
- Pre-tokenization script: `uv run python data/msmarco.py prepare --max-len 2048`

### 3.2 LoCoV1 Zero-Shot Evaluator (`data/loco_eval.py`)

Refactored from existing `benchmark_loco.py` but with NO training component.
Evaluation only.

```python
def evaluate_locov1(
    model_wrapper,
    max_len: int = 2048,
    batch_size: int = 32,
    device: str = "cuda",
    tasks: list[str] | None = None,  # subset of 12 tasks, or all
) -> dict:
    """Zero-shot evaluation on LoCoV1.

    Loads data from HuggingFace, encodes queries and docs with
    the model, computes nDCG@{1,3,5,10} per task.

    Returns:
        {
            "avg_ndcg@10": float,
            "per_task": {task_name: {"nDCG@1": ..., "nDCG@10": ...}},
            "n_queries_total": int,
            "n_docs_total": int,
            "encode_time_s": float,
        }
    """
```

Port the existing `evaluate_task_ndcg`, `_encode_batch`, `_collate_ids`,
and `compute_ndcg` from `benchmark_loco.py`. No changes to evaluation
logic — just decouple it from the training loop.

### 3.3 BEIR Evaluator (`data/beir_eval.py`) -- DONE

**Implementation:** MTEB-based evaluator with manual fallback. 15 publicly available datasets.
Includes `MTEBModelWrapper` class for MTEB integration and manual `evaluate_beir()` fallback.

```python
def evaluate_beir(
    model_wrapper,
    max_len: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    datasets: list[str] | None = None,
) -> dict:
    """Zero-shot evaluation on BEIR retrieval datasets.

    Returns:
        {
            "avg_ndcg@10": float,
            "per_dataset": {name: {"nDCG@10": float, "n_queries": int, "n_docs": int}},
        }
    """
```

**Option A (recommended):** Use the `mteb` package directly:
```python
from mteb import MTEB
evaluation = MTEB(tasks=["BeIR/..."])
results = evaluation.run(model, output_folder="results/paper/...")
```

**Option B:** Implement custom evaluator (same pattern as LoCoV1 — encode
queries and docs, cosine similarity, rank, nDCG). More control but more code.

Recommend Option A — `mteb` handles data loading, evaluation metrics, and
produces standardized output. Add `mteb` to pyproject.toml dependencies.

### 3.4 LongEmbed Evaluator (`data/longembed_eval.py`) -- DONE

**Implementation:** Manual evaluator loading from `dwzhu/LongEmbed`. 6 tasks: NeedleInAHaystack,
PasskeyRetrieval, NarrativeQA, QMSum, 2WikiMultihopQA, SummScreenFD.

```python
def evaluate_longembed(
    model_wrapper,
    max_len: int = 4096,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """Zero-shot evaluation on LongEmbed tasks.

    Returns same structure as LoCoV1 evaluator.
    """
```

Can also use `mteb` if the LongEmbed tasks are registered there (they are
in MMTEB as of early 2025). Otherwise, implement manually — the tasks are
straightforward retrieval setups.

---

## 4. Model Implementations

### 4.1 PRISM-Simplified (existing)

Already implemented. Build function:

```python
# From benchmark_loco.py
def build_simplified_prism(vocab_size, max_len, **kwargs):
    model = prism_small(vocab_size=vocab_size, max_len=max_len, **kwargs)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(...)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(...)
    model.pooling = MeanPooling(model.d, model.d_e)
    return model
```

For the paper, consolidate into a clean factory:
```python
def build_prism_small(vocab_size=30522, max_len=8192, decay_mode="all_slow"):
    ...
def build_prism_base(vocab_size=30522, max_len=8192, decay_mode="all_slow"):
    ...  # d=768, 12 layers, 8 channels — ~80M params
```

### 4.2 Transformer (existing)

Already implemented in `baseline_transformer.py`. Needs a `_base` variant
for Experiment 7:

```python
def build_transformer_small(vocab_size=30522, max_len=8192):
    ...  # d=384, 6 layers, 6 heads — ~20M params
def build_transformer_base(vocab_size=30522, max_len=8192):
    ...  # d=768, 12 layers, 12 heads — ~85M params
```

### 4.3 Mamba-Bidir (`mamba_bidir.py`) — NEW

The key new baseline. A bidirectional Mamba encoder for embeddings.

**Architecture:**
```
Input tokens
    |
[Embedding + Position]
    |
[MambaBlock_fwd_1, MambaBlock_bwd_1] -> Gated Fusion -> LayerNorm
    |
[MambaBlock_fwd_2, MambaBlock_bwd_2] -> Gated Fusion -> LayerNorm
    |
... (n_layers)
    |
[Mean Pooling]
    |
[LayerNorm -> Projection -> L2 Normalize]
```

Each MambaBlock is a standard Mamba selective SSM block. The bidirectional
fusion mirrors PRISM's gated fusion:
```
f_t = beta_t * fwd_t + (1 - beta_t) * bwd_t
beta_t = sigmoid(W_beta [fwd_t || bwd_t])
```

**Dependencies:** `mamba_ssm` package (pip install mamba-ssm). Requires
CUDA (Mamba's selective scan kernel is CUDA-only). If mamba_ssm is not
available, fall back to a pure-PyTorch implementation of the selective scan
(slower but functional for correctness testing).

```python
class MambaBidirLayer(nn.Module):
    """Single bidirectional Mamba layer."""
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        self.mamba_fwd = Mamba(d_model=d, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.fusion_gate = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        fwd_out = self.mamba_fwd(x)
        bwd_out = self.mamba_bwd(x.flip(1)).flip(1)  # reverse, run, reverse back
        beta = torch.sigmoid(self.fusion_gate(torch.cat([fwd_out, bwd_out], dim=-1)))
        fused = beta * fwd_out + (1 - beta) * bwd_out
        return self.norm(x + self.dropout(fused))  # pre-norm residual

class MambaBidirEncoder(nn.Module):
    ...  # Stack of MambaBidirLayer + embedding + mean pool

class MambaBidirForEmbedding(nn.Module):
    ...  # Wrapper with encode() and forward(q, p) -> InfoNCE loss
```

**Parameter matching:** With d=384, expand=2, d_state=16, d_conv=4, and
the right number of layers, target ~20M total params. May need to adjust
n_layers (likely 8-10 layers since Mamba layers are thinner than PRISM layers).

### 4.4 Linear-RNN (`linear_rnn.py`) — NEW

Ablation baseline: single-channel gated linear recurrence (no multi-channel).

```python
class LinearRNNLayer(nn.Module):
    """Single-channel gated linear recurrence, bidirectional."""
    def __init__(self, d: int, decay: float = 0.99, dropout: float = 0.1):
        self.decay = decay
        self.gate_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        ...

    def forward(self, x, mask=None):
        # Forward scan
        g = torch.sigmoid(self.gate_proj(x))
        v = g * x
        h_fwd = parallel_scan(self.decay, v)
        # Backward scan
        h_bwd = parallel_scan(self.decay, v.flip(1)).flip(1)
        # Gated fusion
        ...
```

Reuse `_fast_fixed_decay_scan` from `prism.py`. This is essentially
PRISM with C=1 (one channel of dimension d=384 instead of 6 channels of
dimension 64).

---

## 5. Unified Training Loop (`train_contrastive.py`)

Consolidate the training logic from `benchmark_loco.py` into a reusable,
model-agnostic training loop. All experiments use this.

```python
def train(
    model_wrapper,                  # any model with .forward(q_ids, q_mask, p_ids, p_mask) -> {"loss": ...}
    dataset,                        # MSMARCODataset or similar with .sample_batch()
    run_dir: Path,                  # where to save everything
    config: dict,                   # full config (saved to config.json)
    n_steps: int = 50000,
    micro_batch: int = 16,
    grad_accum: int = 8,
    lr: float = 3e-4,
    warmup_fraction: float = 0.1,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_every: int = 5000,         # run eval benchmarks every N steps
    checkpoint_every: int = 5000,   # save model checkpoint every N steps
    log_every: int = 100,           # log metrics every N steps
    eval_fn = None,                 # callable(model, step) -> dict of metrics
    device: str = "cuda",
    seed: int = 42,
):
    """Train a contrastive embedding model with structured logging.

    Saves to run_dir/:
      config.json         at start
      train_log.jsonl      every log_every steps
      checkpoints/         every checkpoint_every steps
      eval/                every eval_every steps
      final_metrics.json   at end
    """

    # --- Setup ---
    save_config(run_dir, config)
    set_seed(seed)

    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(n_steps * warmup_fraction), n_steps
    )

    best_metric = -1.0
    best_step = 0

    # --- Training loop ---
    model_wrapper.train()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        step_grad_norm = 0.0

        for _ in range(grad_accum):
            batch = dataset.sample_batch(micro_batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model_wrapper(
                batch["query_ids"], batch["query_mask"],
                batch["pos_ids"], batch["pos_mask"],
            )
            loss = result["loss"] / grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_wrapper.parameters(), grad_clip
        )
        optimizer.step()
        scheduler.step()

        # --- Logging ---
        if step % log_every == 0:
            log_step(run_dir, step, {
                "loss": step_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "throughput_seqs_per_s": ...,
                "gpu_mem_mb": torch.cuda.max_memory_allocated() // (1024**2)
                              if torch.cuda.is_available() else 0,
                "elapsed_s": ...,
            })

        # --- Eval ---
        if eval_fn and step % eval_every == 0:
            model_wrapper.eval()
            eval_results = eval_fn(model_wrapper, step)
            save_eval_results(run_dir, step, "combined", eval_results)

            # Track best
            metric = eval_results.get("locov1_avg_ndcg@10", 0)
            if metric > best_metric:
                best_metric = metric
                best_step = step

            model_wrapper.train()

        # --- Checkpoint ---
        if step % checkpoint_every == 0:
            save_checkpoint(run_dir, model_wrapper, optimizer, step)

    # --- Final ---
    save_final_metrics(run_dir, {
        "best_step": best_step,
        "best_locov1_avg": best_metric,
        "total_train_time_s": ...,
        "total_params": sum(p.numel() for p in model_wrapper.parameters()),
    })
```

### 5.1 Eval Callback

```python
def make_eval_fn(max_len, device):
    """Create an eval callback that runs LoCoV1 + optionally MS MARCO dev."""

    # Pre-load LoCoV1 data once
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loco_data = load_loco_data(tokenizer, max_len=max_len)
    msmarco_dev = load_msmarco_dev(tokenizer, max_len=max_len)

    def eval_fn(model_wrapper, step):
        results = {}

        # LoCoV1
        loco_results = evaluate_locov1(model_wrapper, max_len, loco_data)
        results["locov1_avg_ndcg@10"] = loco_results["avg_ndcg@10"]
        results["locov1_per_task"] = loco_results["per_task"]

        # MS MARCO dev MRR@10
        mrr = evaluate_msmarco_dev(model_wrapper, msmarco_dev, max_len)
        results["msmarco_mrr@10"] = mrr

        return results

    return eval_fn
```

---

## 6. Experiment Runner Implementations

### 6.1 Exp 1: Controlled Comparison (`paper_exp1_controlled.py`)

```python
"""
Experiment 1: Controlled architecture comparison.

Trains all models on MS MARCO with identical protocol,
evaluates zero-shot on LoCoV1 at multiple sequence lengths.

Usage:
  uv run python paper_exp1_controlled.py --max-len 2048
  uv run python paper_exp1_controlled.py --max-len 2048 --models prism,transformer
  uv run python paper_exp1_controlled.py --all   # runs all sub-experiments 1a-1e
"""

MODELS = {
    "prism_small": build_prism_small,       # ~19M
    "transformer_small": build_transformer_small,  # ~20M
    "mamba_bidir_small": build_mamba_bidir_small,  # ~20M
    "linear_rnn_small": build_linear_rnn_small,    # ~19M
}

SUB_EXPERIMENTS = {
    "1a": {"train_max_len": 128,  "eval_max_len": 128},
    "1b": {"train_max_len": 512,  "eval_max_len": 512},
    "1c": {"train_max_len": 2048, "eval_max_len": 2048},
    "1d": {"train_max_len": 2048, "eval_max_len": 2048, "eval_locov1": True},
    "1e": {"train_max_len": 2048, "eval_max_len": 8192, "eval_locov1": True,
            "models": ["prism_small", "mamba_bidir_small", "linear_rnn_small"]},
            # Transformer excluded: OOMs at 8K eval
}

def run_sub_experiment(sub_exp_id, models=None):
    config = SUB_EXPERIMENTS[sub_exp_id]
    ...
    for model_name in (models or config.get("models", MODELS.keys())):
        run_dir = create_run_dir(f"exp1_{sub_exp_id}", model_name)
        model = MODELS[model_name](vocab_size=30522, max_len=config["eval_max_len"])
        dataset = MSMARCODataset(tokenizer, max_len=config["train_max_len"])
        eval_fn = make_eval_fn(config["eval_max_len"], DEVICE)

        train(
            model_wrapper=model,
            dataset=dataset,
            run_dir=run_dir,
            config={...},
            n_steps=50000,
            eval_every=5000,
            eval_fn=eval_fn,
        )
```

### 6.2 Exp 2: Efficiency (`paper_exp2_efficiency.py`)

```python
"""
Experiment 2: Scaling curves.

Measures inference latency, training throughput, and peak memory
for all architectures at sequence lengths from 64 to 16384.

Usage:
  uv run python paper_exp2_efficiency.py
"""

SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_SIZES = [1, 32]

def measure_inference_latency(model, seq_len, batch_size, n_runs=100, n_warmup=10):
    """Median latency over n_runs after n_warmup warmup passes."""
    ...
    return {"median_ms": ..., "p95_ms": ..., "p99_ms": ...}

def measure_training_throughput(model, seq_len, max_memory_gb=80):
    """Find max batch that fits in memory, report seqs/sec."""
    ...
    return {"max_batch": ..., "seqs_per_sec": ..., "peak_mem_mb": ...}

def measure_peak_memory(model, seq_len, batch_size):
    """Forward + backward peak memory."""
    ...
    return {"fwd_mb": ..., "fwd_bwd_mb": ...}

def run():
    results = {}
    for model_name, build_fn in MODELS.items():
        results[model_name] = {}
        for seq_len in SEQ_LENGTHS:
            model = build_fn(max_len=seq_len).to(DEVICE)
            try:
                results[model_name][seq_len] = {
                    "latency_b1": measure_inference_latency(model, seq_len, 1),
                    "latency_b32": measure_inference_latency(model, seq_len, 32),
                    "throughput": measure_training_throughput(model, seq_len),
                    "memory": measure_peak_memory(model, seq_len, 32),
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[model_name][seq_len] = {"oom": True}
                    torch.cuda.empty_cache()
            model.cpu()
            del model

    # Save results + generate plots
    save_json(results, "results/paper/exp2/scaling_results.json")
    plot_scaling_curves(results)  # latency, memory, throughput vs seq_len
```

**Plots to generate:**
1. `latency_vs_seqlen.png` — log-log, lines for each model, both batch=1 and batch=32
2. `memory_vs_seqlen.png` — log-log, show OOM points as X markers
3. `throughput_vs_seqlen.png` — sequences/sec vs seq_len
4. `crossover_table.png` — rendered table of crossover points

### 6.3 Exp 3: Ablations (`paper_exp3_ablations.py`)

```python
"""
Experiment 3: Component ablation study.

9 variants of PRISM, each removing or modifying one component.
All trained at max_len=2048 on MS MARCO, eval on LoCoV1 zero-shot.

Usage:
  uv run python paper_exp3_ablations.py --variant A
  uv run python paper_exp3_ablations.py --all
"""

VARIANTS = {
    "baseline": {"desc": "PRISM-Simplified (full)"},
    "A_single_channel":  {"n_channels": 1, "d_c": 384},
    "B_geometric_decay": {"decay_mode": "geometric"},
    "C_all_fast_decay":  {"decay_mode": "all_fast", "decay_val": 0.5},
    "D_learned_decay":   {"decay_mode": "learned"},
    "E_no_gating":       {"use_gating": False},
    "F_unidirectional":  {"bidirectional": False},
    "G_interference":    {"use_interference": True},
    "H_cov_pooling":     {"pooling": "attentive_covariance"},
    "I_attentive_pool":  {"pooling": "attentive"},
}
```

### 6.4 Experiments 4-5: Eval-Only

These run on checkpoints from Experiment 1. Minimal new code.

```python
# paper_exp4_longembed.py
"""Load best Exp 1 checkpoints, evaluate on LongEmbed at 2K, 4K, 8K."""

# paper_exp5_beir.py
"""Load best Exp 1 checkpoints, evaluate on BEIR 18 datasets."""
```

### 6.5 Exp 6: Pretraining (`paper_exp6_pretrain.py`)

```python
"""
Experiment 6: Pretrain on weakly-supervised pairs, fine-tune on MS MARCO,
evaluate zero-shot on LoCoV1 + BEIR + LongEmbed.

Phase 1: Contrastive pretraining on ~10M text pairs
  Sources: E5's public pair datasets (Reddit, StackExchange, Wikipedia, etc.)
  Loss: InfoNCE with in-batch negatives only (no hard negatives)
  Steps: 100K, batch=256, lr=1e-4

Phase 2: Fine-tune on MS MARCO with hard negatives
  Same as Experiment 1 protocol.

Phase 3: Zero-shot eval on all benchmarks.
"""
```

---

## 7. Dependencies -- DONE

Updated `pyproject.toml` with optional dependency groups:

```toml
[project.optional-dependencies]
paper = ["mteb>=1.0.0"]      # BEIR + LongEmbed evaluation
mamba = ["mamba-ssm>=2.0.0"] # Mamba baseline (requires CUDA)
```

Core deps unchanged. `mamba-ssm` and `mteb` are optional — gated imports
with clear error messages. `ir-datasets` not needed (using Tevatron/msmarco-passage
directly via HuggingFace `datasets`).

---

## 8. Implementation Sequence

### Phase A: Infrastructure (no GPU needed)

**Step A1: Logging framework** (`paper_log.py`) -- DONE
- [x] `create_run_dir`, `save_config`, `log_step`, `save_checkpoint`
- [x] `save_eval_results`, `save_final_metrics`, `capture_hardware_info`
- [x] Test: create a dummy run, write 100 log lines, verify JSON is valid

**Step A2: MS MARCO data pipeline** (`data/msmarco.py`) -- DONE
- [x] Download + cache MS MARCO passages, queries, qrels
- [x] Pre-tokenize passages with bert-base-uncased, save as JSONL cache
- [x] Hard negative loader (BM25 negatives from Tevatron/msmarco-passage)
- [x] `sample_batch()` returns properly shaped tensors
- [x] `get_dev_data()` returns dev queries + qrels
- [x] Test: smoke test passed (400K queries, 5.5M passages loaded)

**Step A3: LoCoV1 evaluator** (`data/loco_eval.py`) -- DONE
- [x] Port `evaluate_task_ndcg`, `_encode_batch`, `compute_ndcg` from
      `benchmark_loco.py` into standalone evaluation function
- [x] Remove all training code — eval only
- [x] Test: validated in smoke test

**Step A4: Unified training loop** (`train_contrastive.py`) -- DONE
- [x] Implement `train()` with logging, checkpointing, eval callbacks
- [x] Warmup + cosine schedule
- [x] Gradient accumulation
- [x] Test: smoke test passed (100 steps on MS MARCO, structured logs, eval)

**Step A5: Model factories** -- DONE
- [x] Build functions in each experiment file (not a separate models.py)
- [x] Verify parameter counts match expectations (~19-20M small, ~80M base)

### Phase B: New Baselines (needs CUDA for Mamba)

**Step B1: Mamba-Bidir** (`mamba_bidir.py`) -- DONE
- [x] Implement `MambaBidirLayer`, `MambaBidirEncoder`, `MambaBidirForEmbedding`
- [x] Match the interface: `.encode(ids, mask)` -> embeddings,
      `.forward(q_ids, q_mask, p_ids, p_mask)` -> `{"loss": ...}`
- [x] Pure PyTorch fallback when mamba_ssm not available
- [x] Parameter count ~20M

**Step B2: Linear-RNN** (`linear_rnn.py`) -- DONE
- [x] Single-channel bidirectional gated linear recurrence
- [x] Reuses parallel scan from prism.py
- [x] Parameter count ~20M

### Phase C: Experiments (GPU)

All experiment runner scripts are implemented and smoke-tested. Execution order:

```
C1: Exp 2 — Efficiency benchmarks (fast, no training, validates all models work)
C2: Exp 1a — Short-sequence comparison (128 tokens, fastest training)
C3: Exp 1b — Medium-sequence (512 tokens)
C4: Exp 1c — Long-sequence (2048 tokens) — the core result
C5: Exp 3 — Ablation study (can run in parallel with C4 if two GPUs)
C6: Exp 1d — LoCoV1 zero-shot eval on Exp 1c checkpoints
C7: Exp 1e — LoCoV1 at 8K (PRISM + Mamba only)
C8: Exp 4 — LongEmbed eval on Exp 1c checkpoints
C9: Exp 5 — BEIR eval on Exp 1a/1b checkpoints
C10: Exp 6 — Pretraining (only if C4 results are strong)
C11: Exp 7 — Scale-up (only if targeting top venue)
```

**Runner status:** All scripts written (`paper_exp1_controlled.py` through `paper_exp7_scaleup.py`).
Smoke test passed for the full pipeline (data → training → eval). Ready for GPU execution.

### Phase D: Analysis & Plots

**Step D1: Result aggregation**
- [ ] Script that walks `results/paper/exp*/` and produces summary tables
- [ ] LaTeX table generation for each experiment

**Step D2: Plots**
- [ ] Exp 1: bar chart of nDCG@10 per model per sequence length
- [ ] Exp 2: scaling curves (latency, memory, throughput)
- [ ] Exp 3: ablation bar chart
- [ ] Exp 4-5: comparison tables with published baselines

---

## 9. Milestones & Checkpoints

### Milestone 1: Infrastructure Ready -- ACHIEVED
- [x] MS MARCO loads and samples batches correctly
- [x] LoCoV1 evaluator runs standalone
- [x] Training loop produces structured logs + checkpoints
- [x] All 4 models (PRISM, Transformer, Mamba-Bidir, Linear-RNN) build
      and pass smoke tests
- **Deliverable:** Full smoke test passed (data download → training → eval)

### Milestone 2: Efficiency Benchmarks Complete
- [ ] Scaling curves for all 4 models at 9 sequence lengths
- [ ] Plots generated, crossover points identified
- [ ] Results match existing Phase 1 numbers for PRISM and Transformer
      (sanity check against `results/20260317/scaling_results.json`)
- **Deliverable:** `results/paper/exp2/` populated with plots + JSON

### Milestone 3: Core Quality Results (Exp 1c + 1d)
- [ ] All 4 models trained at 2048 tokens for 50K steps
- [ ] LoCoV1 zero-shot nDCG@10 computed for all models
- [ ] PRISM beats Transformer by 5+ points on LoCoV1
- **Deliverable:** The headline result table

### Milestone 4: Ablations Complete
- [ ] 9 variants trained and evaluated
- [ ] Negative results confirmed (interference harmful, cov pooling harmful)
- [ ] Ablation bar chart generated
- **Deliverable:** `results/paper/exp3/` with all variant results

### Milestone 5: Paper-Ready
- [ ] All P0 + P1 experiments complete
- [ ] All plots and tables generated
- [ ] Results written to PAPER_LOG.md
- **Deliverable:** Enough data to write the paper

---

## 10. GPU Budget

| Step | Description | Est. GPU-hours |
|------|------------|----------------|
| A4 | Smoke test training loop | 0.5h |
| B1-B2 | Mamba + LinearRNN smoke tests | 1h |
| C1 | Exp 2: efficiency benchmarks | 8h |
| C2 | Exp 1a: 4 models x 50K steps @ 128 tokens | 20h |
| C3 | Exp 1b: 4 models x 50K steps @ 512 tokens | 40h |
| C4 | Exp 1c: 4 models x 50K steps @ 2048 tokens | 80h |
| C5 | Exp 3: 9 ablations x 50K steps @ 2048 tokens | 150h |
| C6-C9 | Eval-only experiments | 15h |
| C10 | Exp 6: pretraining (100K steps) + fine-tune | 100h |
| **Total P0** | Steps A-C5 | **~300h** |
| **Total P0+P1** | All above | **~415h** |

**Cost-saving tactics:**
- Run Exp 1a first (128 tokens = fast) to validate the full pipeline
- LR sweep at 2K steps before committing to 50K
- If a model is clearly worse by step 10K, early-stop
- Ablation variants G and H (interference, cov pooling) are expected to
  fail — run them last, don't spend time debugging if they do
- Use bf16 if all models are stable (roughly halves memory, ~30% faster)

---

## 11. Risk Register

### R1: MS MARCO too large for single-GPU pre-tokenization
**Mitigation:** Stream passages, tokenize in chunks of 100K, write memmap
incrementally. Total tokenized size ~4GB for 8.8M passages at max_len=512.

### R2: Mamba-Bidir doesn't parameter-match at ~20M
**Mitigation:** Mamba layers are parameter-light (mostly in the SSM state
expansion). May need 10-12 layers to reach 20M. If still short, add a
small MLP per layer. Document exact param count in the paper.

### R3: 50K steps is too few for MS MARCO convergence
**Mitigation:** Monitor MS MARCO dev MRR@10 convergence. If still climbing
at 50K, extend to 100K for the final runs. The controlled comparison is
still fair as long as all models train for the same number of steps.

### R4: LoCoV1 results don't replicate under new protocol
**Mitigation:** The new protocol trains on MS MARCO (not LoCoV1), so absolute
numbers WILL change. What matters is the relative gap (PRISM > Transformer).
If the gap vanishes, the claim weakens — but this would itself be an important
finding to report honestly.

### R5: mamba-ssm installation fails on target GPU
**Mitigation:** Gate the import. If Mamba can't be installed, document the
issue and include Mamba results only for the hardware where it works.
Consider a pure-PyTorch fallback (selective scan implemented in Python).
