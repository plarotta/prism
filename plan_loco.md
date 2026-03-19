# LoCoV1 Benchmark Experiment Plan

## Goal

Evaluate PRISM-Simplified on the LoCoV1 long-context retrieval benchmark and compare directly against published M2-BERT results and Transformer baselines. This answers two questions:

1. Does PRISM-Simplified's quality advantage over Transformers hold on a real, established long-document retrieval benchmark?
2. How does PRISM-Simplified compare to M2-BERT (Monarch Mixer), the closest existing architecture to our approach?

---

## Background

LoCoV1 is a 12-task long-context retrieval benchmark from the Hazy Research group (Stanford). Tasks span law, medicine, science, finance, and government reports. Documents are long enough that chunking fails — models need genuine long-context understanding. Published baselines include M2-BERT (80M, state-space), E5-Mistral (7.1B), BGE-Large (335M), OpenAI Ada, Cohere, VoyageAI, BM25, and ColBERTv2.

Key reference: Saad-Falcon et al., "Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT" (arXiv:2402.07440).

- Dataset: `hazyresearch/LoCoV1-Documents` on HuggingFace
- Code: https://github.com/HazyResearch/m2-bert-retrieval

---

## Phase 0: Setup and Data (1–2 days)

### 0.1 — Download LoCoV1

- Pull queries and documents from HuggingFace (`hazyresearch/LoCoV1-Documents`)
- Inspect the dataset structure: understand how each of the 12 tasks is formatted (query-document pairs, relevance labels, splits)
- Profile document length distributions per task — identify which tasks have documents in the 2K–8K range (PRISM's sweet spot) vs 16K–32K (may need truncation or architecture changes)

### 0.2 — Read the M2-BERT Paper and Code

- Read the full paper, paying special attention to:
  - The fine-tuning procedure (orthogonal projection loss, single-sample batches)
  - Pretraining data mixture (short + long sequences from C4, Wikipedia, BookCorpus)
  - Evaluation protocol (metrics per task, how nDCG@10 is computed)
- Clone their repo and understand the evaluation harness so we can reuse it for PRISM

### 0.3 — Establish Evaluation Protocol

- Match their evaluation exactly: nDCG@10 per task, averaged across all 12 tasks
- Use the same query/document splits they used
- Confirm we can reproduce at least one of their published baselines (e.g., BM25) to validate our eval pipeline

---

## Phase 1: Direct Evaluation Without Pretraining (3–5 days)

This is the fast, cheap experiment. Train PRISM-Simplified from scratch on LoCoV1 training data (no pretraining on C4/Wikipedia) and see how far it gets. This won't be a fair comparison to M2-BERT (which was pretrained), but it tells us whether the architecture can learn the task at all and gives us a lower bound.

### 1.1 — Adapt PRISM-Simplified for LoCoV1

- Use the `bert-base-uncased` tokenizer (30,522 vocab) — same as our existing real-data experiments
- Decide on max sequence lengths to evaluate: 2048 and 8192 are the key targets
  - 2048: PRISM starts winning on speed here, and many LoCoV1 documents fit
  - 8192: strong scaling advantage, tests genuine long-context quality
- For documents longer than max_len: truncate (not ideal, but matches what Transformer baselines do)

### 1.2 — Fine-tuning Setup

- **Loss function:** Start with InfoNCE (our existing contrastive setup). If batch size is too constrained at long sequences, implement orthogonal projection loss (OPL) from the M2-BERT paper — it works with single-sample batches
- **Batch size considerations at each length:**
  - 2048 tokens: estimate ~8–16 pairs per batch on A40 (48GB)
  - 8192 tokens: likely 1–4 pairs per batch — this is where OPL becomes necessary
- **Training budget:** 5K–10K steps per length (more than our 2K-step CNN/DailyMail runs, since LoCoV1 has harder retrieval tasks)
- **Learning rate:** sweep {1e-4, 3e-4, 5e-4} on one task first, then fix for full benchmark

### 1.3 — Run Evaluation

- Train separate PRISM-Simplified models at 2048 and 8192 max_len
- Evaluate on all 12 LoCoV1 tasks
- Record nDCG@10 per task and average
- Record inference throughput (seq/s) at each length for the scaling comparison
- Compare against published M2-BERT numbers and Transformer baselines from their paper

### 1.4 — Expected Outcome

Without pretraining, PRISM will likely underperform M2-BERT significantly. That's fine — the point is to establish a baseline and validate the pipeline. If PRISM-Simplified without pretraining is even competitive with the Transformer baselines (BGE, Ada), that's already notable given the parameter count.

---

## Phase 2: Pretraining (1–2 weeks)

This is where the real comparison happens. M2-BERT was pretrained with MLM on a mixture of short and long sequences. We need to do the same for PRISM.

### 2.1 — Pretraining Data

- Use the same sources as M2-BERT: C4, Wikipedia, BookCorpus
- Implement the mixed-length strategy: sample both short sequences (10–128 tokens) and long sequences (up to max_len) during pretraining
- For long sequences: concatenate multiple short documents to fill the context window (same as M2-BERT)

### 2.2 — Pretraining Objective

- Masked Language Modeling (MLM) — standard 15% mask rate
- This requires modifying PRISM-Simplified to output per-token predictions (currently it only outputs pooled embeddings)
- Add a prediction head on top of the recurrence stack: LayerNorm → Linear(d, vocab_size)
- Train at two scales:
  - PRISM-Simplified-2K: max_len=2048
  - PRISM-Simplified-8K: max_len=8192 (warm-start from 2K checkpoint, following M2-BERT's approach)

### 2.3 — Pretraining Budget

- M2-BERT pretrained for 5K steps (per their ablation table)
- On a single A40, this will be slow for 8K sequences. Rough estimate:
  - 2K pretraining: ~1–2 days
  - 8K pretraining: ~3–5 days
- If wall-clock time is prohibitive, consider using a cloud GPU instance (A100 80GB) for the pretraining phase

### 2.4 — Fine-tune and Evaluate

- Take pretrained checkpoints, fine-tune on LoCoV1 using the same protocol as Phase 1
- This is the apples-to-apples comparison with M2-BERT

---

## Phase 3: Analysis and Differentiation (2–3 days)

### 3.1 — Per-Task Breakdown

- Compare PRISM vs M2-BERT vs Transformers on each of the 12 tasks individually
- Look for patterns: does PRISM do better on certain domains (legal, scientific) or document length ranges?
- The fixed geometric decay might give advantages on tasks with strong multi-scale structure (e.g., legal documents with section/paragraph/sentence hierarchy)

### 3.2 — Scaling Comparison

- Plot inference throughput vs sequence length for PRISM-Simplified alongside M2-BERT published numbers
- Plot memory usage at each length
- This is where PRISM's simplicity might matter: fewer parameters, simpler operations, potentially better hardware utilization than Monarch matrices

### 3.3 — Ablation: Does Geometric Decay Spacing Matter?

- Run a small ablation comparing:
  - Fixed geometric decay (current PRISM default)
  - Fixed linear decay spacing
  - Random fixed decay rates
  - Learned decay rates
- Do this on the LoCoV1 tasks where PRISM performs best
- This is the core scientific question: is the geometric spacing a meaningful inductive bias, or does any multi-channel recurrence work?

### 3.4 — Write Up Findings

- If PRISM is competitive with M2-BERT: the story is "fixed geometric decay recurrence matches structured state spaces for long-doc retrieval, with a simpler architecture"
- If PRISM is behind: analyze the gap. Is it pretraining? Architecture? What does Monarch Mixer's matrix structure give that plain recurrence doesn't?
- If PRISM wins on some tasks: identify what structural properties of those tasks align with the geometric decay inductive bias

---

## Hardware Requirements

| Phase | GPU | Estimated Time |
|-------|-----|----------------|
| 0: Setup | CPU only | 1–2 days |
| 1: No-pretrain eval | A40 48GB | 3–5 days |
| 2: Pretraining | A40 48GB (A100 preferred) | 1–2 weeks |
| 3: Analysis | A40 48GB | 2–3 days |

Total: ~3–4 weeks

---

## Risk Factors

- **Pretraining compute.** Single-GPU pretraining at 8K context is slow. If this becomes a bottleneck, consider pretraining only at 2K and evaluating at 2K. The 2K results alone are meaningful.
- **OPL implementation.** Orthogonal projection loss is less standard than InfoNCE. Budget time to implement and debug it. Alternatively, use gradient accumulation to maintain reasonable effective batch sizes with InfoNCE.
- **Tokenizer mismatch.** M2-BERT may use a different tokenizer. Document lengths in tokens will differ, which affects direct comparisons. Normalize by reporting actual token counts alongside results.
- **Fair comparison caveats.** M2-BERT is 80M parameters vs PRISM-Simplified at ~26M. If PRISM underperforms, scale up before concluding the architecture is worse. If PRISM matches at 26M what M2-BERT does at 80M, that's a strong result.

---

## Success Criteria

- **Minimum viable result:** PRISM-Simplified (no pretraining) beats naive Transformer baselines on LoCoV1 tasks with documents > 2K tokens
- **Strong result:** Pretrained PRISM-Simplified is competitive with M2-BERT (within 5 nDCG@10 points on average)
- **Home run:** PRISM-Simplified matches or beats M2-BERT with fewer parameters and simpler architecture, with the geometric decay ablation showing it's a meaningful inductive bias

---

## Phase 4: Hybrid Architecture Experiments (in progress)

Phase 1 results revealed a key problem: **PRISM at 8K (0.578 nDCG@10) underperforms PRISM at 2K (0.689)**. Mean pooling averages over all token positions equally, diluting the embedding when documents contain boilerplate or low-information sections. At 2K, truncation accidentally helps by keeping only the information-dense beginning. At 8K, the model sees the full document but can't distinguish signal from noise at the pooling stage.

This phase addresses the 8K < 2K gap through a series of targeted experiments implemented in `benchmark_hybrid.py`.

### Experiment 0: Improved Baselines (complete)

Re-ran the LoCoV1 baselines (PRISM-MeanPool + Transformer) with two key improvements over Phase 1:
- **LR sweep:** Tested [1e-4, 3e-4, 5e-4, 1e-3] at 500 steps each, selected best per model (PRISM: 3e-4, Transformer: 1e-4)
- **Extended training:** 7000 steps (up from 5000 in Phase 1)
- **Checkpoint evaluation:** Full 12-task eval every 1000 steps to track convergence curves

#### Results

| Model | max_len | Params | Avg nDCG@10 | Train Time | Notes |
|-------|---------|--------|-------------|------------|-------|
| PRISM-MeanPool | 2048 | 19.2M | **0.770** | 1754s | Best overall result |
| Transformer | 2048 | 19.8M | 0.672 | 3631s | Huge improvement from LR sweep |
| PRISM-MeanPool | 8192 | 21.6M | 0.675 | 6661s | 8K < 2K gap persists |
| Transformer | 8192 | 22.1M | — | discontinued | micro_batch=2, grad_accum=8; ~5s/step, discontinued at step 2200 after nDCG plateaued at 0.107 |

**Key observations from Experiment 0:**
- The Transformer jumped from 0.194 to 0.672 at 2K — the Phase 1 Transformer result was crippled by a suboptimal learning rate, not an architectural limitation. The LR sweep was critical.
- PRISM still wins at 2K (0.770 vs 0.672, +9.8 points) but the gap narrowed significantly from +49.4 to +9.8.
- The 8K < 2K gap for PRISM narrowed slightly (from 11.1 to 9.5 points) but persists, confirming mean pooling dilution as an ongoing problem.
- PRISM converges ~2x faster wall-clock than the Transformer at 2K, and the Transformer is impractical at 8K (5s/step with micro_batch=2).
- The Transformer at 8K was discontinued — at step 2200 it had only reached nDCG@10 = 0.107, with learning stalled. The O(n^2) cost + tiny micro-batches make it unviable.

#### Convergence Trajectories (PRISM-MeanPool @ 2048)

| Step | nDCG@10 | Loss |
|------|---------|------|
| 1000 | 0.218 | 0.689 |
| 2000 | 0.390 | 0.329 |
| 3000 | 0.513 | 0.153 |
| 4000 | 0.601 | 0.111 |
| 5000 | 0.704 | 0.063 |
| 6000 | 0.752 | 0.054 |
| 7000 | 0.770 | 0.032 |

### Experiment 1: Attentive Pooling (pending)

Replace mean pooling with a learned attentive pooling head — the core intervention to fix the 8K < 2K gap.

**Architecture:** Single learned query vector q, softmax attention over token positions, weighted sum + projection + LayerNorm. Adds ~384 parameters (negligible). The hypothesis is that the recurrence backbone already captures useful long-range information that mean pooling averages away.

| Run | Pooling | max_len | Purpose |
|-----|---------|---------|---------|
| 1a | Attentive | 2048 | Does attentive pooling help or hurt at 2K? |
| 1b | Attentive | 8192 | Does attentive pooling fix the 8K degradation? |

**Success criteria:** Run 1b > 0.675 (mean pool 8K). Ideal: Run 1b >= 0.770 (mean pool 2K).

### Experiment 2: Multi-Head Attentive Pooling (conditional on Exp 1)

Only runs if Experiment 1 shows improvement. K=4 and K=8 independent learned queries at 8192, each attending to different aspects of the document (e.g., entities vs topic sentences vs conclusions). Concatenated and projected down.

### Experiment 3: Local Attention Hybrid (conditional)

Insert a sliding-window attention layer (w=256 or w=512) after recurrence layer 4 of 6, giving the model one opportunity for precise local token-level matching within the recurrence stack. O(n*w) — preserves linear scaling. Only runs if the pooling fix alone doesn't fully close the 2K-vs-8K gap.

### Experiment 4: Decay Spacing Ablation (independent)

Tests whether geometric decay spacing is a meaningful inductive bias by comparing:
- **Geometric** (current default): log-spaced decay rates
- **Linear**: evenly spaced from 0 to 1-epsilon
- **Random fixed**: uniformly sampled, frozen (seed=42)
- **All-slow** (lambda=0.99): global context only
- **All-fast** (lambda=0.1): local context only

All at max_len=2048, 7000 steps. If geometric beats all variants, the log-spaced frequency decomposition is a genuine inductive bias — the core differentiator from M2-BERT.

### Running the Experiments

```bash
uv run python benchmark_hybrid.py                           # all experiments
uv run python benchmark_hybrid.py --skip-exp0               # skip baselines (already done)
uv run python benchmark_hybrid.py --skip-exp2 --skip-exp3   # just Exp 0, 1, 4
```