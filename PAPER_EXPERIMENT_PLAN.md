# PRISM Embedding Paper: Experiment Plan

## Paper Thesis

> Bidirectional multi-channel gated linear recurrence is a simple, effective
> architecture for long-document embeddings that matches Transformer quality
> at short sequences, exceeds it at long sequences (256+ tokens), and scales
> linearly in time and memory with sequence length.

The contribution is the first purpose-built bidirectional SSM encoder for
text embeddings, with rigorous evaluation showing it fills a gap between
short-context Transformer encoders and expensive long-context models.

---

## The Evaluation Protocol Problem (MUST FIX)

Current LoCoV1 results train on the LoCoV1 query-doc pairs and evaluate on
the same data. Published baselines (M2-BERT, BM25, etc.) use zero-shot or
held-out evaluation. The actual M2-BERT-8K score is ~0.869, not ~0.506.

**Fix:** All experiments below use proper train/eval splits. Two protocols:

1. **Controlled comparison** (primary): Train all architectures (PRISM,
   Transformer, Mamba encoder) with the SAME data and recipe, compare
   architectures head-to-head. This isolates the architecture contribution.

2. **Published baseline comparison** (secondary): Pretrain PRISM with a
   standard recipe, then evaluate zero-shot on benchmarks — the same
   protocol used by M2-BERT, BGE, GTE, etc. This places PRISM on the
   existing leaderboard.

---

## Experiment Overview

| # | Experiment | Purpose | Priority |
|---|-----------|---------|----------|
| 1 | Controlled architecture comparison on LoCoV1 | Core claim: PRISM beats Transformers at long sequences | P0 |
| 2 | Efficiency benchmarks (scaling curves) | Core claim: O(n) scaling | P0 |
| 3 | Component ablation study | Scientific rigor + valuable negative results | P0 |
| 4 | LongEmbed benchmark | Validate on second long-doc benchmark | P1 |
| 5 | BEIR retrieval (short-doc generalization) | Show PRISM is not only a long-doc trick | P1 |
| 6 | Pretrain + zero-shot LoCoV1 | Fair comparison with published models | P1 |
| 7 | Scale-up to 80M parameters | Show findings hold at M2-BERT scale | P2 |

---

## Experiment 1: Controlled Architecture Comparison (P0)

### Goal
Head-to-head comparison of architectures, all trained identically, across
sequence lengths. This is the centerpiece of the paper.

### Models (all ~20M params, d=384)

| Model | Description | Params |
|-------|------------|--------|
| **PRISM-Simplified** | Bidirectional multi-channel SSM (6ch, all-slow lambda=0.99, mean pool) | ~19M |
| **Transformer** | Bidirectional Transformer encoder (6 heads, 6 layers, attentive pool) | ~20M |
| **Mamba-Bidir** | Bidirectional Mamba encoder (selective SSM, fwd+bwd fusion, mean pool) | ~20M |
| **Linear-RNN** | Single-channel gated linear recurrence (no multi-channel, bidir, mean pool) | ~19M |

**Why these baselines:**
- Transformer: the standard. Already implemented.
- Mamba-Bidir: the closest SSM competitor. Mamba Retriever (causal, 130M)
  gets 0.891 on LoCoV0 — a bidirectional version at 20M is the fairest SSM
  comparison. Uses `mamba_ssm` package.
- Linear-RNN: ablation baseline — is multi-channel necessary, or does a
  single wide recurrence suffice?

### Training Protocol

**Data:** MS MARCO passage retrieval (8.8M passages, 500K training queries)
for contrastive fine-tuning. This is standard, publicly available, and
completely disjoint from LoCoV1/LongEmbed/BEIR eval sets.

**Loss:** InfoNCE with in-batch negatives + 7 hard negatives per query
(mined with BM25, then refined with a cross-encoder teacher).

**Training:**
- Tokenizer: bert-base-uncased (30,522 vocab)
- Optimizer: AdamW, lr=3e-4, weight decay=0.01
- Schedule: linear warmup 10%, cosine decay to 1e-5
- Gradient clipping: 1.0
- Batch size: 128 (effective, via gradient accumulation)
- Steps: 50,000
- Max sequence length: varies per sub-experiment (see below)

**Sub-experiments by sequence length:**

| Sub-exp | Train max_len | Eval max_len | Purpose |
|---------|--------------|-------------|---------|
| 1a | 128 | 128 | Short-sequence quality (Transformer's home turf) |
| 1b | 512 | 512 | Medium-length (crossover regime) |
| 1c | 2048 | 2048 | Long-sequence (PRISM's expected advantage) |
| 1d | 2048 | LoCoV1 (2048) | Zero-shot transfer to long-doc retrieval |
| 1e | 2048 | LoCoV1 (8192) | Long-context transfer (PRISM only — Transformer OOMs) |

**Evaluation metrics:**
- MS MARCO dev set: MRR@10, Recall@100
- LoCoV1 (12 tasks): nDCG@10 per task + average
- Report all per-task numbers in appendix

**Success criteria:**
- 1a: PRISM within 2 pts of Transformer (parity at short seq)
- 1b: PRISM matches or beats Transformer
- 1c: PRISM beats Transformer by 5+ nDCG@10 pts
- 1d: PRISM beats Transformer on LoCoV1 zero-shot
- 1e: PRISM runs where Transformer cannot

### Implementation Notes

Need to implement:
- [ ] MS MARCO data loader with hard negative mining
- [ ] Mamba-Bidir encoder (adapt mamba_ssm for bidirectional + mean pool)
- [ ] Linear-RNN baseline (single-channel variant of PRISM)
- [ ] Proper train/eval split infrastructure
- [ ] Cross-encoder teacher for hard negative scoring (can use a
      public cross-encoder like cross-encoder/ms-marco-MiniLM-L-6-v2)

---

## Experiment 2: Efficiency Benchmarks (P0)

### Goal
Demonstrate O(n) scaling with proper methodology. Current scaling numbers
are solid but need to be presented more rigorously.

### Measurements

**Inference latency** (batch=1 and batch=32):
- Sequence lengths: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Metric: median latency over 100 runs (after 10 warmup)
- All models from Experiment 1

**Training throughput** (forward + backward):
- Same sequence lengths
- Metric: sequences/second at max batch size that fits in memory
- Report peak GPU memory alongside

**Theoretical FLOPs:**
- Derive and report per-token FLOPs for each architecture
- Show both theoretical curves and measured wall-clock

### Presentation
- Latency vs sequence length (log-log plot showing O(n) vs O(n^2))
- Memory vs sequence length (show OOM points for Transformer)
- Throughput vs sequence length
- Table: crossover points for latency, memory, training throughput

### Hardware
Run on a single standardized GPU. Report exact GPU model, driver version,
CUDA version, PyTorch version. Ideally A100-80GB for reproducibility.

---

## Experiment 3: Ablation Study (P0)

### Goal
Isolate which components matter. The negative results (interference,
covariance pooling, geometric decay) are as valuable as the positive ones.

### Ablation Variants

All trained with the Experiment 1c protocol (2048 tokens, MS MARCO, 50K steps).

| Variant | Change from PRISM-Simplified | Tests |
|---------|------------------------------|-------|
| **A: Single channel** | C=1, d_c=384 | Is multi-channel necessary? |
| **B: Geometric decay** | lambda_c log-spaced [0.0 ... 0.9998] | Does decay spacing matter? |
| **C: All-fast decay** | all lambda=0.5 | Is slow decay necessary? |
| **D: Learned decay** | lambda_c as learned parameters | Fixed vs learned |
| **E: No gating** | Remove input gate g_t, raw linear recurrence | Is gating necessary? |
| **F: Unidirectional** | Forward only, no backward pass | Is bidirectionality necessary? |
| **G: + Interference** | Add cross-scale bilinear interference (original) | Confirm interference is harmful |
| **H: + Cov pooling** | Replace mean pool with attentive covariance | Confirm cov pooling is harmful |
| **I: Attentive pool** | Replace mean pool with learned query attention | Alternative pooling |

### Evaluation
- LoCoV1 zero-shot nDCG@10 (from Exp 1d)
- MS MARCO dev MRR@10
- Report convergence curves (loss vs step) for all variants

### Presentation
- Bar chart: nDCG@10 for each variant vs PRISM-Simplified baseline
- Table with full numbers
- Highlight the negative results (G, H) prominently — these are a
  contribution, not a weakness

---

## Experiment 4: LongEmbed Benchmark (P1)

### Goal
Validate long-document embedding quality on a second independent benchmark.

### Setup
LongEmbed (Zhu et al., EMNLP 2024) has 6 tasks:
- 2 synthetic: Passkey retrieval, Needle-in-a-Haystack
- 4 real: NarrativeQA, QMSum, 2WikiMultihopQA, SummScreenFD

Documents up to 32K tokens. Metric: nDCG@10.

### Models
Use the best checkpoints from Experiment 1c (trained on MS MARCO at 2048).
Evaluate zero-shot.

### Key comparisons
- PRISM vs Transformer at 2048 truncation
- PRISM at 2048 vs 4096 vs 8192 (can PRISM benefit from longer context?)
- Compare to published numbers: E5-base (41.0 avg at 512), Jina-v2 (~60 at 8K)

---

## Experiment 5: BEIR Retrieval (P1)

### Goal
Show PRISM generalizes to standard short-document retrieval, even if not SOTA.

### Setup
BEIR has 18 retrieval datasets (most documents under 512 tokens). This is
Transformer territory. The goal is NOT to beat Transformer here — it's to
show PRISM doesn't catastrophically fail at short sequences.

### Models
Use Experiment 1a or 1b checkpoints (trained at 128 or 512 tokens).

### Evaluation
- nDCG@10 on all 18 BEIR datasets
- Compare to published small-model baselines:
  - bge-small-en-v1.5 (33M): 51.68 avg
  - snowflake-arctic-embed-s (33M): 51.98 avg
  - e5-small-v2 (33M): 49.04 avg
- Note: PRISM at 19M params is smaller than all of these

### Acceptable outcome
Within 5 nDCG@10 points of bge-small at short sequences. The story is:
"PRISM is competitive at short sequences and dominant at long sequences."

---

## Experiment 6: Pretrain + Zero-Shot LoCoV1 (P1)

### Goal
Compare PRISM to published models under their evaluation protocol.

### Pretraining
- **Data:** Subset of E5's CCPairs or equivalent (~10M-50M weakly-supervised
  text pairs from Reddit, StackExchange, Wikipedia, CommonCrawl)
- **Objective:** Contrastive (InfoNCE with in-batch negatives)
- **Duration:** ~100K steps with batch=256 on single GPU
- **Max length:** 512 for pretraining (standard)

### Fine-tuning
- **Data:** MS MARCO (same as Experiment 1) with hard negatives
- **Duration:** 50K steps
- **Max lengths:** Train at 2048 (and optionally 8192)

### Evaluation
- Zero-shot on LoCoV1 (12 tasks, nDCG@10)
- Zero-shot on LongEmbed
- Zero-shot on BEIR

### Reference Published Baselines

| Model | Params | Pretrained | LoCoV1 2K | LoCoV1 8K |
|-------|--------|-----------|-----------|-----------|
| M2-BERT-80M | 80M | Yes (C4+Wiki) | 0.823 | 0.869 |
| BM25 | -- | -- | 0.799 | 0.799 |
| OpenAI Ada-002 | unknown | Yes | 0.632 | 0.632 |
| PRISM-19M | 19M | Yes (this exp) | ? | ? |

**Note:** PRISM at 19M has 4x fewer params than M2-BERT. The fair
comparison is efficiency-adjusted: PRISM quality per FLOP, quality per
parameter, and quality per second of inference.

---

## Experiment 7: Scale-Up to 80M Parameters (P2)

### Goal
Show results hold at larger scale. Necessary if reviewers ask "does it scale?"

### Model Configurations

| Config | d | layers | channels | d_c | params |
|--------|---|--------|----------|-----|--------|
| PRISM-Small | 384 | 6 | 6 | 64 | ~19M |
| PRISM-Base | 768 | 12 | 8 | 96 | ~80M |

| Config | d | layers | heads | params |
|--------|---|--------|-------|--------|
| Transformer-Small | 384 | 6 | 6 | ~20M |
| Transformer-Base | 768 | 12 | 12 | ~85M |

### Training
Same protocol as Experiment 6 (pretrain + fine-tune).

### Evaluation
- LoCoV1 zero-shot at 2K and 8K
- BEIR zero-shot
- Scaling efficiency: quality vs params plot

### Key question this answers
Does PRISM's advantage grow, shrink, or stay constant with scale?

---

## Benchmark Summary Table (for the paper)

The paper should present results in this format:

| Model | Params | Max Ctx | MS MARCO MRR@10 | LoCoV1 2K | LoCoV1 8K | BEIR Avg | LongEmbed Avg |
|-------|--------|---------|-----------------|-----------|-----------|----------|---------------|
| **PRISM-Small** | 19M | 16K+ | ? | ? | ? | ? | ? |
| Transformer-Small | 20M | ~2K* | ? | ? | OOM | ? | ? |
| Mamba-Bidir-Small | 20M | 16K+ | ? | ? | ? | ? | ? |
| Linear-RNN-Small | 19M | 16K+ | ? | ? | ? | ? | ? |
| *bge-small-en-v1.5* | 33M | 512 | -- | -- | -- | 51.7 | -- |
| *arctic-embed-s* | 33M | 512 | -- | -- | -- | 52.0 | -- |
| **PRISM-Base** | 80M | 16K+ | ? | ? | ? | ? | ? |
| Transformer-Base | 85M | ~4K* | ? | ? | OOM | ? | ? |
| *M2-BERT-80M* | 80M | 8K/32K | -- | 82.3 | 86.9 | -- | -- |

*Transformer max context limited by OOM at training batch sizes.
Published external baselines in italics.

---

## Negative Results to Highlight

These are genuine contributions — present them prominently:

1. **Cross-scale bilinear interference:** Theoretically motivated, inert at
   short sequences, catastrophically harmful at long sequences (-24.4 MRR).
   Not rescued by targeted numerical fixes (7 variants tested). Implication:
   cross-channel multiplicative interactions need either much larger scale or
   fundamentally different formulations.

2. **Attentive covariance pooling:** Second-order statistics are noisy and
   unhelpful for contrastive embeddings. Mean pooling strictly dominates.

3. **Geometric decay spacing:** The supposed core inductive bias is
   suboptimal. Uniform slow decay beats geometric by +7.2 nDCG@10. Input
   gating learns to differentiate channels without needing different base
   decay rates.

4. **Attentive pooling at long sequences:** Mean pooling dilution is real but
   minor (~1.8 pts at 8K). The bottleneck is backbone capacity, not pooling.

---

## Compute Budget Estimate

Assuming single RTX PRO 6000 (96GB):

| Experiment | GPU-hours (approx) |
|-----------|-------------------|
| Exp 1: Controlled comparison (5 models x 5 configs) | ~200h |
| Exp 2: Efficiency benchmarks | ~10h |
| Exp 3: Ablation study (9 variants) | ~150h |
| Exp 4: LongEmbed eval only | ~5h |
| Exp 5: BEIR eval only | ~10h |
| Exp 6: Pretraining + fine-tuning + eval | ~100h |
| Exp 7: Scale-up (80M, 2 models) | ~200h |
| **Total** | **~675h (~28 days)** |

P0 experiments alone: ~360h (~15 days).

---

## Implementation Order

1. **Fix data pipeline** — MS MARCO loader with hard negative mining
2. **Fix eval pipeline** — LoCoV1 zero-shot, BEIR, LongEmbed evaluators
3. **Implement Mamba-Bidir baseline** — bidirectional selective SSM encoder
4. **Run Experiment 1** (controlled comparison) — the core result
5. **Run Experiment 2** (scaling curves) — complements Exp 1
6. **Run Experiment 3** (ablations) — can partially parallelize with Exp 1
7. **Run Experiments 4-5** (LongEmbed, BEIR) — eval-only on Exp 1 checkpoints
8. **Run Experiment 6** (pretraining) — only if Exp 1 results are strong
9. **Run Experiment 7** (scale-up) — only if targeting a top venue

---

## What Makes This Publishable

1. **Gap in literature:** No published bidirectional SSM encoder for text
   embeddings. Mamba Retriever is causal. M2-BERT uses Monarch matrices.
   PRISM fills this gap.

2. **Clean architecture story:** The simplest thing works. Multi-channel
   recurrence + uniform slow decay + bidirectional fusion + mean pooling.
   No bells and whistles needed.

3. **Strong negative results:** Three theoretically motivated components
   (interference, covariance pooling, geometric decay) failed cleanly and
   informatively. The community benefits from knowing this.

4. **Efficiency narrative:** Linear scaling enables long-document embeddings
   on hardware where Transformers cannot even train.

5. **Rigorous evaluation:** Head-to-head controlled comparison + zero-shot
   evaluation on established benchmarks + multiple baselines at the same
   parameter count.
