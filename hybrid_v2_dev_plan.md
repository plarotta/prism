# HybridPRISM v2: Development & Implementation Plan

## 1. Architecture Overview

```
Input tokens
    |
[Embedding + Position]
    |
[PRISM Group 1: layers 1-N]       O(n*d^2)  linear-time recurrence
    |
[MemoryWrite 1 + MemoryRead 1]    O(n*k*d)  cross-attention to/from k slots
    |
[PRISM Group 2: layers N+1-2N]
    |
[MemoryWrite 2 + MemoryRead 2]
    |
[PRISM Group 3: layers 2N+1-3N]
    |
[Output head]                      pooling for embeddings, LM head for generation
```

Total complexity: O(L_prism * n * d^2) + O(L_mem * n * k * d), linear in n.

All PRISM blocks use the validated all-slow config (lambda=0.99, no interference, bidirectional gated fusion). The memory bank is the only new component.

---

## 2. Component Specifications

### 2.1 MemoryBank

A set of k learned vectors (k, d) that persist across write/read operations within a forward pass. Initialized as small random vectors (std=0.02) that specialize during training.

Not a `nn.Module` with its own forward — just a `nn.Parameter` owned by the encoder. Expanded to (B, k, d) at the start of each forward pass via `.expand()`.

### 2.2 MemoryWrite

Cross-attention where memory slots attend to token states. Each slot queries the full sequence and accumulates a weighted summary.

```
Input:  memory (B, k, d), token_states (B, n, d), mask (B, n)
Output: updated memory (B, k, d)

Q = W_q(memory)         # (B, k, d)
K = W_k(token_states)   # (B, n, d)
V = W_v(token_states)   # (B, n, d)

attn = softmax(Q @ K^T / sqrt(d_head), dim=-1)  # (B, heads, k, n)
retrieved = attn @ V                              # (B, k, d)

# Gated residual update (memory retains old content or overwrites)
gate = sigmoid(W_gate([memory; retrieved]))       # (B, k, d)
memory_out = gate * memory + (1 - gate) * retrieved
```

Multi-head (default 4 heads). The gated update is critical — without it, the second write would overwrite the first. With it, the model can accumulate across writes.

### 2.3 MemoryRead

Cross-attention where tokens attend to memory slots. Each token queries memory and retrieves relevant stored information.

```
Input:  token_states (B, n, d), memory (B, k, d), mask (B, n)
Output: updated token_states (B, n, d)

Q = W_q(token_states)   # (B, n, d)
K = W_k(memory)         # (B, k, d)
V = W_v(memory)         # (B, k, d)

attn = softmax(Q @ K^T / sqrt(d_head), dim=-1)  # (B, heads, n, k)
retrieved = attn @ V                              # (B, n, d)

# Pre-norm residual connection
output = LayerNorm(token_states + W_out(retrieved))
```

Multi-head (default 4 heads). Standard pre-norm residual — no gating needed since this is enriching token representations, not managing persistent state.

### 2.4 HybridPRISMEncoder

Orchestrates the full forward pass:

```python
class HybridPRISMEncoder(nn.Module):
    def __init__(self, vocab_size, d=384, d_e=384, n_layers=12,
                 n_channels=6, max_len=8192, k=32, n_mem_heads=4,
                 layers_per_group=4, mlp_ratio=2.0, dropout=0.1,
                 decay_mode="all_slow", pad_token_id=0):
        ...
```

Forward pass:
1. Token + positional embeddings -> LayerNorm -> dropout
2. For each group of PRISM layers:
   a. Run layers_per_group PRISMLayer modules sequentially
   b. If not the last group: MemoryWrite then MemoryRead
3. Final LayerNorm
4. Pooling head (mean pooling for embeddings)
5. Return embedding + token states

The PRISM layers are reused from `prism.py` but configured with:
- All decay rates set to 0.99 (all-slow)
- NoInterference replacing CrossScaleInterference
- Bidirectional gated fusion

### 2.5 HybridPRISMForEmbedding

Wrapper matching the existing PRISMForEmbedding interface:
- `encode(input_ids, attention_mask)` -> L2-normalized embeddings
- `forward(query_ids, query_mask, pos_ids, pos_mask, ...)` -> InfoNCE loss + accuracy

Same contrastive training loop as existing benchmarks.

---

## 3. Model Configurations

### Config A: Small (development & fast iteration)

| Parameter | Value |
|-----------|-------|
| d | 384 |
| d_e | 384 |
| n_layers | 6 |
| layers_per_group | 2 |
| n_groups | 3 |
| n_memory_points | 2 (between groups) |
| n_channels | 6 |
| k (memory slots) | 32 |
| n_mem_heads | 4 |
| decay | all-slow (0.99) |
| ~params | ~23M |

Use for: initial debugging, smoke tests, short-sequence experiments.

### Config B: Medium (main experiments)

| Parameter | Value |
|-----------|-------|
| d | 384 |
| d_e | 384 |
| n_layers | 12 |
| layers_per_group | 4 |
| n_groups | 3 |
| n_memory_points | 2 |
| n_channels | 6 |
| k (memory slots) | 32 |
| n_mem_heads | 4 |
| decay | all-slow (0.99) |
| ~params | ~38M |

Use for: Phase 1 and Phase 2 experiments. Larger than the ~20M PRISM-small, but the user accepted this tradeoff.

Note: for LoCoV1 regression, also run a parameter-matched all-slow PRISM at 12 layers (~34M) to isolate the memory bank's contribution vs simply having more layers.

---

## 4. File Structure

```
prism/
  hybrid_prism.py              # New: MemoryBank, MemoryWrite, MemoryRead,
                               #       HybridPRISMEncoder, HybridPRISMForEmbedding,
                               #       hybrid_prism_small(), hybrid_prism_medium()
  benchmark_hybrid_v2.py       # New: all Phase 1 & 2 experiments, self-contained
  hybrid_v2_dev_plan.md        # This file
  # Existing files unchanged:
  prism.py                     # PRISMLayer, PRISMEncoder, etc.
  baseline_transformer.py
  benchmark_loco.py            # Reuse: data loading, training, eval utilities
  benchmark_hybrid.py          # Phase 6 experiments (complete, not modified)
```

`benchmark_hybrid_v2.py` imports from:
- `hybrid_prism.py` for the new architecture
- `prism.py` for PRISMLayer, StratifiedRecurrence, etc.
- `benchmark_loco.py` for LoCoV1 data loading, training loop, eval
- `benchmark_ablations.py` for NoInterference, MeanPooling

---

## 5. Implementation Sequence

### Step 1: Core Architecture (hybrid_prism.py)

Build in this order, testing each component in isolation:

**1a. MemoryWrite module**
- Input: memory (B, k, d), token_states (B, n, d), mask (B, n)
- Output: updated memory (B, k, d)
- Test: verify shapes, gradient flow, mask handling
- Verify: attention weights sum to 1 over valid tokens, masked positions get zero weight

**1b. MemoryRead module**
- Input: token_states (B, n, d), memory (B, k, d)
- Output: updated token_states (B, n, d)
- Test: verify shapes, gradient flow, residual connection
- Verify: output matches input when memory is zeros (residual passthrough)

**1c. HybridPRISMEncoder**
- Compose PRISM layers (from prism.py) with memory write/read
- Configure all-slow decay by overwriting StratifiedRecurrence.lambdas buffer
- Replace interference with NoInterference
- Test: forward pass produces correct output shapes
- Test: backward pass runs without error
- Test: parameter count matches expectations

**1d. HybridPRISMForEmbedding**
- Wrap encoder with contrastive loss (match PRISMForEmbedding interface)
- Test: InfoNCE loss computes and backprops

**1e. Factory functions**
- `hybrid_prism_small()` -> Config A
- `hybrid_prism_medium()` -> Config B

**Smoke test**: Train Config A for 100 steps on a tiny synthetic contrastive task (random token sequences, batch=8, seq_len=128). Verify loss decreases. Run on CPU or MPS — no GPU needed.

### Step 2: LoCoV1 Regression (benchmark_hybrid_v2.py — Experiment 0)

Train HybridPRISM (Config B) on LoCoV1 at max_len=2048. Same protocol as Phase 6 Experiment 0:
- LR sweep: [1e-4, 3e-4, 5e-4, 1e-3]
- 7000 steps, checkpoint eval every 1000 steps
- bert-base-uncased tokenizer

**Baselines to compare against:**
- All-slow PRISM 6-layer (existing: 0.770 nDCG@10 at 7K steps)
- All-slow PRISM 12-layer (new: isolate depth vs memory contribution)
- Transformer 8-layer (existing: 0.672 nDCG@10)

**Success criterion:** HybridPRISM >= 0.77 nDCG@10 (no regression vs best PRISM).
**If it fails:** memory mechanism is interfering. Debug candidates: (1) memory gate initialization, (2) LR too high for memory params, (3) memory read disrupting residual stream. Fix before proceeding.

GPU: RTX PRO 6000. Expected time: ~4 runs x 30min = 2 hours for LR sweep.

### Step 3: Needle-in-a-Haystack Task (Experiment 1)

This is the key capability demonstration. Design below in Section 6.

### Step 4: Scaling Curves (Experiment 2)

Run needle-in-haystack at lengths [512, 1K, 2K, 4K, 8K, 16K].
Three models: all-slow PRISM (6-layer), HybridPRISM (Config B), Transformer (8-layer, until OOM).
Plot retrieval accuracy vs sequence length.

### Step 5: Phase 2 Experiments (Experiments 3-5)

Detailed in Section 7.

---

## 6. Phase 1 Experiment Design: Needle-in-a-Haystack

### Task Definition

**Goal:** Test whether HybridPRISM can retrieve a specific fact buried in a long document, where all-slow PRISM (mean pooling) cannot because the fact signal gets diluted.

**Formulation:** Contrastive retrieval. Given a short query referencing a specific fact, retrieve the correct long document (which contains that fact buried among filler text) from a corpus of distractor documents.

### Data Generation

**Filler text source:** `wikitext-103-v1` from HuggingFace (real Wikipedia text, tokenized with bert-base-uncased). Provides natural, coherent text as the "haystack."

**Needle design:** Synthetic key-value facts using templated sentences:
```
Templates:
  "The {attribute} of {entity} is {value}."
  "In the year {year}, {entity} achieved {event}."
  "{entity} is located in {location}."
  "The code assigned to {entity} is {code}."

Entities: randomly generated proper nouns (e.g., "Zarkon", "Meridia Corp", "Project Helios")
Values: randomly generated strings/numbers that are unique per fact
```

Using synthetic entities/values ensures the model can't rely on pretrained knowledge (there is none — models train from scratch). Each fact is unique and must be retrieved from the document.

**Document construction:**
1. Sample filler paragraphs from wikitext-103 to fill target length
2. Insert one needle sentence at a random position (uniform over [0.1*n, 0.9*n] to avoid trivial start/end positions)
3. Tokenize to target max_len

**Query construction:**
- Short question about the needle: "What is the {attribute} of {entity}?"
- Tokenized to ~20-30 tokens

**Corpus construction per training batch:**
- batch_size documents, each with a unique needle
- batch_size matching queries
- In-batch negatives: documents with different needles serve as negatives
- This is the same InfoNCE setup as LoCoV1

**Evaluation:**
- Fixed eval set of 500 query-document pairs
- Retrieval pool of 500 documents
- Metric: MRR and Recall@1 (did the model rank the correct document first?)

### Experimental Configurations

| Run | Model | max_len | Steps | Notes |
|-----|-------|---------|-------|-------|
| NIAH-A | All-slow PRISM 6L | 2048 | 5000 | baseline — should degrade at long seq |
| NIAH-B | HybridPRISM Config B | 2048 | 5000 | should maintain accuracy |
| NIAH-C | All-slow PRISM 6L | 8192 | 5000 | expect significant degradation |
| NIAH-D | HybridPRISM Config B | 8192 | 5000 | key test — should still work |
| NIAH-E | Transformer 8L | 2048 | 5000 | reference (full attention can do this) |
| NIAH-F | Transformer 8L | 8192 | 5000 | will OOM or be very slow |

### Scaling Curves (Experiment 2)

Sweep max_len in [512, 1024, 2048, 4096, 8192, 16384]:
- Train a fresh model at each length (5000 steps each)
- Plot Recall@1 vs max_len for each architecture
- Also plot training throughput (seq/s) and peak memory

**Expected result:** All-slow PRISM accuracy drops as length increases (needle signal diluted by mean pooling). HybridPRISM accuracy remains high (memory stores the needle). Transformer matches HybridPRISM at short lengths but OOMs at 8K+.

### Needle Position Ablation

After the main experiments, test whether HybridPRISM's accuracy depends on needle position:
- Fix max_len=4096
- Sweep needle position: [10%, 25%, 50%, 75%, 90%] of document
- Plot Recall@1 vs position for each model
- All-slow PRISM should be position-independent (mean pooling treats all positions equally)
- HybridPRISM should also be position-independent (if memory is working correctly)
- If HybridPRISM is position-dependent: reveals which PRISM group's memory write captures the needle

---

## 7. Phase 2 Experiment Design: Reasoning & Retrieval

### Experiment 3: Associative Recall

**Goal:** Test memory capacity — how many distinct facts can the memory bank store and retrieve?

**Task:** Plant N key-value facts in a single long document. Queries ask for specific values. Sweep N to find the capacity limit.

**Data generation:**
- Document: filler text with N needle facts inserted at random positions
- N queries per document, one per fact
- Contrastive setup: for each query, the positive is the document containing the matching fact. Negatives are documents with different facts.
- Sweep N in [1, 2, 4, 8, 16, 32] — k=32 slots, so N=32 tests the full capacity

**Key question:** Does Recall@1 degrade gracefully as N approaches k, or is there a sharp cliff?

**Dataset:** Synthetic (same wikitext filler + generated facts as NIAH). Self-contained generation in benchmark_hybrid_v2.py.

### Experiment 4: Long-Document QA Retrieval (QuALITY)

**Goal:** Demonstrate memory value on a real task with genuine long-document understanding requirements.

**Dataset:** QuALITY (Pang et al., 2022) — available on HuggingFace as `emozilla/quality`. Contains:
- Long articles (~3K-8K words, median ~5K)
- Multiple-choice questions requiring comprehension of the full article
- "Hard" subset where speed-readers got the answer wrong (requires careful reading)

**Reformulation as retrieval:**
- Corpus: all articles in the dataset
- Query: question text (without answer choices)
- Positive: the article the question was written about
- Negatives: other articles in the batch (in-batch negatives)
- Metric: MRR and Recall@1 over the eval set

This tests whether the model's embedding captures enough information from a long document for a specific question to find it. Memory should help: the question targets specific information that mean pooling dilutes in a 5K-word article.

**Configurations:**
- All-slow PRISM 6L at max_len=2048 (truncated articles)
- All-slow PRISM 6L at max_len=8192 (full articles)
- HybridPRISM Config B at max_len=2048
- HybridPRISM Config B at max_len=8192
- Transformer 8L at max_len=2048

**Alternative dataset if QuALITY is too small or noisy:** NarrativeQA (Kocisky et al., 2018) — QA over full books/movie scripts. Available on HuggingFace as `narrativeqa`. Much longer documents.

### Experiment 5: Memory Slot Analysis

**Goal:** Understand what the memory learns. Produces interpretability figures for the paper.

**Analyses (run on trained HybridPRISM from Experiments 1-4):**

1. **Write attention heatmaps:** For a sample document, visualize the (k, n) attention matrix from MemoryWrite. Shows which tokens write to which slots. Expect: slots specialize (e.g., one captures entities, one captures the needle fact, others capture topic/structure).

2. **Read attention heatmaps:** For a sample query, visualize the (n, k) attention matrix from MemoryRead. Shows which slots each query token reads from. Expect: query tokens attend to the slot that stored the relevant needle.

3. **Slot utilization:** Across the eval set, compute mean attention entropy per slot in MemoryWrite. High entropy = slot attends broadly (topic summary). Low entropy = slot attends to specific tokens (fact storage). Plot distribution.

4. **Slot ablation:** After training, zero out individual memory slots and measure accuracy drop. Identifies which slots are critical. Do this for the NIAH task.

5. **k sensitivity sweep:** Train HybridPRISM with k in [4, 8, 16, 32, 64] on the NIAH task at max_len=4096. Plot Recall@1 vs k. Find the minimum k that maintains high accuracy.

---

## 8. Phase 3: Language Modeling (TBD)

Placeholder — design depends on Phase 1-2 results.

### Key architectural changes needed for causal LM:
- Unidirectional recurrence (drop backward pass)
- Causal memory: write/read must respect autoregressive constraint
  - Option A: Chunked processing — process w tokens, write to memory, next chunk reads from previous chunks' memory
  - Option B: Causal masking in memory read — token at position t can only read memory written by positions <t
- Replace pooling with token-level LM head (linear -> vocab)
- Memory bank becomes a streaming buffer (not reset per sequence)

### Potential experiments:
- Small-scale LM on wikitext-103 (25M, 80M, 150M params)
- Perplexity comparison: HybridPRISM vs Transformer vs Mamba
- Long-context perplexity: sliding window eval at 8K, 16K, 32K

### Decision point:
Proceed to Phase 3 only if:
- Phase 1 confirms memory adds measurable capability (NIAH success)
- Phase 2 shows benefit on real tasks (QuALITY or associative recall improvement)
- The memory mechanism is stable and trainable (no optimization issues)

---

## 9. Hyperparameter Reference

### Training (default for all experiments unless noted)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | |
| LR | sweep [1e-4, 3e-4, 5e-4, 1e-3] | pick best per experiment |
| Weight decay | 0.01 | |
| LR schedule | linear warmup + cosine decay | |
| Warmup steps | 500 | |
| Batch size | 16 effective | via micro_batch + grad accum |
| Temperature | 0.05 | InfoNCE |
| Max grad norm | 1.0 | |
| Tokenizer | bert-base-uncased | 30,522 vocab |
| Precision | fp32 (bf16 if stable) | test bf16 early |

### Architecture

| Parameter | Config A (small) | Config B (medium) |
|-----------|-----------------|-------------------|
| d | 384 | 384 |
| d_e | 384 | 384 |
| n_layers | 6 | 12 |
| layers_per_group | 2 | 4 |
| n_channels | 6 | 6 |
| k (memory slots) | 32 | 32 |
| n_mem_heads | 4 | 4 |
| mlp_ratio | 2.0 | 2.0 |
| dropout | 0.1 | 0.1 |
| decay | all-slow (0.99) | all-slow (0.99) |

---

## 10. GPU Budget Strategy

**Principle:** Validate cheap, scale expensive. Never run a multi-hour GPU experiment without first confirming the code works on a quick sanity check.

| Stage | Hardware | Time Estimate | Cost Driver |
|-------|----------|---------------|-------------|
| Implementation + smoke tests | Local CPU/MPS | 1-2 days | Free |
| LoCoV1 regression (LR sweep, 4 runs) | RTX PRO 6000 | ~2 hours | ~$8-12 |
| NIAH at 2K (3 models x 5K steps) | RTX PRO 6000 | ~3 hours | ~$12-18 |
| NIAH scaling curves (3 models x 6 lengths) | RTX PRO 6000 | ~8 hours | ~$32-48 |
| Phase 2 experiments | RTX PRO 6000 | ~6-10 hours | ~$24-60 |
| Reruns / debugging | RTX PRO 6000 | ~4 hours buffer | ~$16-24 |

**Total estimated GPU time:** ~25-30 hours over Phase 1-2.

**Scale-up triggers:**
- Move to larger GPU only if: 16K sequences at Config B don't fit on 96GB
- Move to multi-GPU only if: Phase 3 LM experiments require it

**Cost-saving tactics:**
- Run LR sweeps at 2K steps first (enough to see trends), full 7K only on best LR
- Use checkpoint eval — if a run is clearly diverging by step 1000, kill it early
- Test new ideas at Config A (6 layers) before Config B (12 layers)

---

## 11. Risk Mitigation

### Risk 1: Memory mechanism is ignored during training

**Symptom:** HybridPRISM matches all-slow PRISM exactly — memory attention is uniform, slots store noise.

**Diagnosis:** Check memory write attention entropy. If near-maximum (uniform attention), the model isn't learning to use memory.

**Mitigations:**
- Auxiliary diversity loss: penalize slot attention distributions that are too similar to each other. Encourages slot specialization.
- Warm-start: initialize memory write Q/K with small pretrained attention patterns from a standard cross-attention layer.
- Learning rate separation: use higher LR for memory modules (e.g., 3x main LR) to accelerate memory learning.

### Risk 2: Memory destabilizes training

**Symptom:** Loss spikes or NaN after adding memory. Worse than all-slow PRISM baseline.

**Mitigations:**
- Initialize memory gate bias to +2.0 (sigmoid(2)=0.88), so memory defaults to retaining its initial state. Writes must earn the right to overwrite.
- Initialize MemoryRead output projection to near-zero (ReZero style), so the residual connection initially passes through unchanged.
- If persistent: decouple memory LR (lower, e.g., 0.1x main LR).

### Risk 3: NIAH task is too easy for all-slow PRISM

**Symptom:** All-slow PRISM also aces the needle task — memory provides no differentiation.

**Cause:** If the needle is distinctive enough (unique tokens), even mean pooling captures it.

**Fix:** Make needles use common vocabulary (not unique strings). Use natural-sounding facts with common words: "The project was approved on Tuesday" rather than "The code is X7Q9Z." This forces the model to distinguish meaning, not just token rarity.

### Risk 4: Memory overhead exceeds budget

**Symptom:** Memory write/read at n=8192, k=32 uses significantly more memory/time than expected.

**Reality check:** Memory attention is (B, heads, k, n) = (16, 4, 32, 8192) = 16M elements per write. Full self-attention would be (16, 4, 8192, 8192) = 17B elements. The memory mechanism is ~1000x smaller. This should not be a problem. If it is, reduce k or n_mem_heads.

### Risk 5: 12 layers alone explain the improvement (not memory)

**Symptom:** The 12-layer all-slow PRISM (no memory) matches HybridPRISM on all tasks.

**This is a real risk.** The memory modules add ~4M params, but going from 6 to 12 layers adds ~15M params. The depth increase alone could drive any improvement.

**Mitigation:** Always run the 12-layer all-slow PRISM baseline alongside HybridPRISM. The memory bank's contribution must be isolated from the depth increase. If depth alone explains everything, the memory mechanism is not earning its keep.

---

## 12. Implementation Milestones & Checkpoints

### Milestone 1: Architecture compiles and trains (end of Step 1)
- [ ] hybrid_prism.py passes shape tests on CPU
- [ ] Smoke test: 100 steps of contrastive training, loss decreases
- [ ] Parameter counts match estimates (Config A ~23M, Config B ~38M)

### Milestone 2: LoCoV1 regression passes (end of Step 2)
- [ ] HybridPRISM Config B >= 0.77 nDCG@10 on LoCoV1 at 2048
- [ ] No training instability (no loss spikes, NaN)
- [ ] Memory attention is non-uniform (slots are used, not ignored)

### Milestone 3: NIAH demonstrates capability gap (end of Step 3)
- [ ] HybridPRISM Recall@1 > 0.8 on NIAH at max_len=4096
- [ ] All-slow PRISM Recall@1 < 0.5 on NIAH at max_len=4096
- [ ] Gap widens with increasing sequence length

### Milestone 4: Scaling curves tell a clear story (end of Step 4)
- [ ] Plot shows HybridPRISM maintaining accuracy where PRISM degrades and Transformer OOMs
- [ ] This is the hero figure for any writeup

### Milestone 5: Real-task validation (end of Phase 2)
- [ ] HybridPRISM beats all-slow PRISM on at least one real task (QuALITY or associative recall)
- [ ] Memory slot analysis shows interpretable specialization
- [ ] Results are not explained by depth alone (12L PRISM baseline is controlled for)

---

## 13. Execution Order

```
Week 1:  Step 1 — implement hybrid_prism.py, smoke test
         Step 2 — LoCoV1 regression test

Week 2:  Step 3 — NIAH data generation + experiments at 2K and 8K
         Step 4 — NIAH scaling curves

Week 3:  Experiment 3 — Associative recall (multi-needle)
         Experiment 4 — QuALITY long-doc QA retrieval

Week 4:  Experiment 5 — Memory slot analysis + visualizations
         Consolidate results, identify Phase 3 go/no-go
```

Weeks are approximate — adjust based on debugging time. The critical path is Step 1 -> Step 2 -> Step 3. If LoCoV1 regression fails, everything stops until it's fixed.
