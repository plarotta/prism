# HybridPRISM v2: Development Log

Detailed progress log for the hybrid_v2_dev_plan. Updated as work progresses.
High-level results and takeaways get promoted to PROGRESS.md.

---

## Step 1: Core Architecture (hybrid_prism.py)

**Status:** COMPLETE

Implemented `hybrid_prism.py` with: MemoryWrite, MemoryRead, HybridPRISMEncoder, HybridPRISMForEmbedding, factory functions.

### Smoke test results (2026-03-20)

| | Config A (6L) | Config B (12L) |
|---|---|---|
| Params (vocab=30522) | 24.2M | 30.7M |
| Params (vocab=1000) | 9.8M | 16.3M |
| Forward shape | (B, 384) | (B, 384) |

- Gradient flow: memory_init grad is zero at step 0 (expected from ReZero init on MemoryRead.out_proj). Unblocks at step 1 (grad=0.0002). MemoryRead out_proj.weight grad=0.61 at step 0 — it learns to open the gate immediately.
- 10-step mini training: loss 2.32 -> 2.08, confirms model is learning.
- Write gate bias correctly initialized to 2.0 (sigmoid=0.88, memory retains by default).
- Plan estimated ~23M/~38M for Config A/B. Actual is 24.2M/30.7M. Difference is because plan overestimated the non-embedding param count. Close enough.

---

## Step 2: LoCoV1 Regression (Experiment 0)

**Status:** not started

---

## Step 3: Needle-in-a-Haystack (Experiment 1)

**Status:** implementation complete, ready for GPU

### Implementation (2026-03-20)

Created `benchmark_hybrid_v2.py` with:
- NIAH data generation: wikitext-103 filler + synthetic password facts using common English names
- 90x79 = 7110 unique name combos, enough for 5500 (5000 train + 500 eval)
- Needle: "The password assigned to Dr. {first} {last} is {code}." (~14 tokens)
- Filler tokenized in chunks to avoid memory issues
- Training loop (train_niah) with periodic checkpoint eval
- Eval: MRR, Recall@1/5/10 on held-out query-document pairs
- CLI: `uv run python benchmark_hybrid_v2.py niah --max-len 2048`
- Scaling curves: `uv run python benchmark_hybrid_v2.py scaling`

Models tested per experiment:
- AllSlow-PRISM-6L (baseline, no memory)
- AllSlow-PRISM-12L (depth control — isolates depth vs memory)
- HybridPRISM-12L Config B (memory + depth)
- Transformer-8L (reference)

End-to-end smoke test on CPU (max_len=128, 50 steps, batch=8): all components work.
Needs GPU for real experiments (3000 steps, max_len=2048+).

---

## Step 4: Scaling Curves (Experiment 2)

**Status:** not started

---

## Step 5: Phase 2 Experiments

### Experiment 3: Associative Recall

**Status:** not started

### Experiment 4: QuALITY Long-Doc QA Retrieval

**Status:** not started

### Experiment 5: Memory Slot Analysis

**Status:** not started

---

## Decisions & Changes

Log any deviations from the original plan here, with rationale.

---

## Issues & Debugging

Log blockers, bugs, and how they were resolved.
