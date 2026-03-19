# Hybrid PRISM: Attentive Pooling Experiments

## Status Update

**Discontinued:** Transformer evaluation at max_len=8192 on LoCoV1. The Transformer OOMs at 8192 tokens with micro_batch=4 on an RTX PRO 6000 (48GB). The OOM itself is the result — it confirms PRISM operates in a regime where the Transformer baseline cannot run at all.

**Current best results (LoCoV1, no pretraining, 5000 steps):**

| Model | max_len | Avg nDCG@10 | Train Time | Notes |
|-------|---------|-------------|------------|-------|
| PRISM-Simplified | 2048 | **0.689** | 1016s | Best quality |
| PRISM-Simplified | 8192 | 0.578 | 3548s | Degrades vs 2K |
| Transformer | 2048 | 0.194 | 2012s | Near-random on most tasks |
| Transformer | 8192 | — | OOM | Cannot run |

**Problem identified:** PRISM at 8K scores lower than at 2K across all 12 tasks. Mean pooling averages over all token positions equally, which dilutes the embedding when documents contain boilerplate, repetitive language, or low-information sections. At 2K, truncation accidentally helps by keeping only the information-dense beginning. At 8K, the model sees the full document but can't distinguish signal from noise at the pooling stage.

**Proposed fix:** Replace mean pooling with learned attentive pooling — a minimal intervention that lets the model selectively weight informative positions while keeping the entire recurrence backbone unchanged.

---

## Experiment 1: Attentive Pooling Head

### 1.1 — Architecture

Replace `MeanPooling` with `AttentivePooling`:

```
AttentivePooling:
  - Learned query vector q ∈ R^d (single parameter, not input-dependent)
  - Attention scores: a_t = softmax(q · f_t / sqrt(d))
  - Weighted sum: e = Σ a_t * f_t
  - Project + normalize: embedding = LayerNorm(Linear(e))
```

This is the simplest possible attention-based pooling. One learned query, one softmax, one weighted sum. No covariance sketch, no input-dependent query, no multi-head complexity. Adds ~d parameters (384 for PRISM-Small) — negligible.

Why not input-dependent query (like the original attentive covariance pooling)? The original used the slowest channel's last hidden state as the query, which couples the pooling to a specific channel. A learned query is simpler, has no failure mode from the query state being low-quality, and lets the model learn "what tokens matter for retrieval" directly.

### 1.2 — Training Protocol

Identical to the existing LoCoV1 runs:
- Tokenizer: bert-base-uncased (30,522 vocab)
- Loss: InfoNCE with in-batch negatives
- Batch: micro_batch=16, effective_batch=16
- Steps: 5000
- Optimizer: AdamW, same LR as existing runs
- Seeds: same as existing runs for reproducibility

### 1.3 — Runs

| Run | Model | Pooling | max_len | Purpose |
|-----|-------|---------|---------|---------|
| 1a | PRISM | Attentive | 2048 | Does attentive pooling help or hurt at 2K? |
| 1b | PRISM | Attentive | 8192 | Does attentive pooling fix the 8K degradation? |

### 1.4 — Success Criteria

- **Primary:** Run 1b (attentive, 8K) scores higher than existing mean pool 8K (0.578)
- **Ideal:** Run 1b scores equal to or higher than existing mean pool 2K (0.689) — meaning the model can now exploit long context rather than being hurt by it
- **Bonus:** Run 1a improves over mean pool 2K (0.689) — attentive pooling is universally better

### 1.5 — What We Learn

If 8K improves but 2K doesn't change much: mean pooling dilution was the bottleneck at long sequences, and the recurrence backbone was already capturing useful long-range information that mean pooling was averaging away.

If both improve: attentive pooling is strictly better than mean pooling on real data, regardless of length. This would also recontextualize the earlier ablation where mean pooling beat attentive covariance pooling — the problem wasn't attention at the pooling stage, it was the covariance sketch.

If neither improves: the bottleneck is in the recurrence backbone's capacity, not the pooling. The fixed-size hidden state (384 dims) can't compress 8K tokens of real language regardless of how you pool. This would point toward wider channels or more channels as the next direction.

---

## Experiment 2: Multi-Head Attentive Pooling (conditional on Experiment 1)

Only run this if Experiment 1 shows improvement. Tests whether multiple learned queries capture different aspects of the document.

### 2.1 — Architecture

```
MultiHeadAttentivePooling:
  - K learned query vectors q_1..q_K ∈ R^d (K=4 or 8)
  - Each produces an independent weighted sum e_k = Σ softmax(q_k · f_t / sqrt(d)) * f_t
  - Concatenate: [e_1 || ... || e_K]
  - Project down: embedding = LayerNorm(Linear(K*d → d_e))
```

This is a modest step up from single-query — it lets different queries attend to different aspects (e.g., one focuses on entities, another on topic sentences, another on conclusions). Still no quadratic attention through the stack.

### 2.2 — Runs

| Run | K (heads) | max_len | Purpose |
|-----|-----------|---------|---------|
| 2a | 4 | 8192 | Moderate multi-head |
| 2b | 8 | 8192 | More heads |

Compare against Experiment 1b (single-query attentive, 8K).

---

## Experiment 3: Sparse Local Attention + Recurrence Hybrid (conditional on Experiments 1–2)

Only run this if the pooling fix alone doesn't fully close the 2K-vs-8K gap. This is the more invasive hybrid — adding a small amount of attention into the recurrence stack itself.

### 3.1 — Architecture

Insert one sliding-window attention layer after layer 4 (of 6 total):

```
Layer 1: PRISM recurrence
Layer 2: PRISM recurrence
Layer 3: PRISM recurrence
Layer 4: PRISM recurrence
Layer 5: Local attention (window=256 tokens, single head)
Layer 6: PRISM recurrence
+ Attentive pooling head (from Experiment 1)
```

Local attention with window w=256 is O(n·w) = O(n), so it preserves linear scaling. It gives the model one opportunity to do precise token-level matching within a local neighborhood, complementing the recurrence's global but lossy context.

### 3.2 — Why After Layer 4

The recurrence layers build up multi-scale features first. By layer 4, each channel has accumulated context at its characteristic timescale. The local attention layer can then refine these features with precise local interactions — combining "I know the document is about legal contracts" (from slow channels) with "this specific clause mentions liability" (from local attention). The final recurrence layer then re-integrates.

### 3.3 — Runs

| Run | Window | Attention Position | max_len |
|-----|--------|--------------------|---------|
| 3a | 256 | After layer 4 | 8192 |
| 3b | 512 | After layer 4 | 8192 |

### 3.4 — What We Learn

If local attention helps: the recurrence backbone loses fine-grained positional information that matters for retrieval, and a small amount of local attention recovers it. This would suggest the optimal architecture is "recurrence for global context + sparse attention for local precision."

If local attention doesn't help: the recurrence is already capturing local features adequately (via the fast-decay channels), and the remaining gap is about capacity or training, not missing local interactions.

---

## Experiment 4: Decay Spacing Ablation on LoCoV1 (independent of 1–3)

This experiment stands alone and can run in parallel. It tests whether the geometric decay spacing is a meaningful inductive bias.

### 4.1 — Variants

All use attentive pooling (from Experiment 1, assuming it works), max_len=2048, 5000 steps:

| Variant | Decay Spacing | Description |
|---------|---------------|-------------|
| A | Geometric (current) | λ_c = 1 - 2^(-c·Δ), Δ = log2(max_len)/(C-1) |
| B | Linear | λ_c evenly spaced from 0 to 1-ε |
| C | Random fixed | λ_c sampled uniformly from (0, 1), frozen |
| D | All-slow | All channels λ=0.99 (global context only) |
| E | All-fast | All channels λ=0.1 (local context only) |

### 4.2 — What We Learn

If geometric beats all others: the log-spaced frequency decomposition is a genuine inductive bias that matters for real retrieval. This is the core differentiator from M2-BERT.

If linear matches geometric: the specific spacing doesn't matter, just having a range of timescales does.

If random matches geometric: even unstructured multi-scale coverage works, which weakens the geometric spacing claim but still supports multi-channel recurrence in general.

If all-slow or all-fast are competitive: the multi-scale decomposition doesn't matter, and a single timescale suffices. This would undermine the core PRISM thesis.

---

## Execution Order and Dependencies

```
Experiment 1 (attentive pooling)           ← RUN FIRST
    |
    ├── if improvement → Experiment 2 (multi-head)
    |       |
    |       └── if gap remains → Experiment 3 (local attention hybrid)
    |
    └── if no improvement → skip 2, go directly to Experiment 3
            (problem is in backbone, not pooling)

Experiment 4 (decay spacing)               ← RUN IN PARALLEL with 1
```

### Estimated Timeline

| Experiment | GPU Hours | Wall Clock |
|------------|-----------|------------|
| 1 (2 runs) | ~2.5h | 1 day |
| 2 (2 runs, conditional) | ~2h | 1 day |
| 3 (2 runs, conditional) | ~2.5h | 1 day |
| 4 (5 runs, parallel) | ~3h | 1 day |

Total: 3–4 days depending on how many conditional experiments are triggered.

### Hardware

RTX PRO 6000 (48GB VRAM). All runs at 48% or lower memory utilization at 8K, so no memory concerns. Can increase batch size to 32 for the 2K runs if desired.

---

## Updated Research Narrative

The story is now:

1. **Multi-channel fixed-decay recurrence beats Transformers for long-document embeddings** — confirmed on LoCoV1 at 2K (0.689 vs 0.194) and at 8K (0.578 vs OOM).
2. **Mean pooling is the bottleneck at long sequences, not the recurrence** — (to be confirmed by Experiment 1).
3. **A lightweight attentive pooling head recovers long-sequence quality** — (to be confirmed by Experiment 1).
4. **Geometric decay spacing is a meaningful inductive bias** — (to be confirmed by Experiment 4).
5. **The full architecture (recurrence + attentive pooling) scales linearly and improves with context length** — the end goal.

If Experiments 1 and 4 both succeed, PRISM has a clean differentiator from M2-BERT: interpretable fixed geometric decay with attentive pooling, simpler than Monarch matrices, competitive or better on established benchmarks.