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

**Status:** GPU run in progress (rerunning HybridPRISM + Transformer after fix)

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
- Added `--models` flag to selectively run specific models

Models tested per experiment:
- AllSlow-PRISM-6L (baseline, no memory)
- AllSlow-PRISM-12L (depth control — isolates depth vs memory)
- HybridPRISM-12L Config B (memory + depth)
- Transformer-8L (reference)

End-to-end smoke test on CPU (max_len=128, 50 steps, batch=8): all components work.

### GPU Results — NIAH @ max_len=2048, 3000 steps (2026-03-21)

**AllSlow baselines (complete):**

| Model | Params | Final Loss | MRR | R@1 | R@5 | Time |
|---|---|---|---|---|---|---|
| AllSlow-PRISM-6L | 19.2M | 0.849 | 0.014 | 0.000 | 0.014 | 2550s |
| AllSlow-PRISM-12L | 25.7M | 1.604 | 0.014 | 0.000 | 0.010 | 5060s |

Both at random-level retrieval despite decreasing contrastive loss. Confirms mean pooling completely dilutes the needle signal (~14 tokens in 2048). More depth (12L) doesn't help — actually harder to optimize (higher final loss). This cleanly isolates the variable: any HybridPRISM improvement is from the memory bank, not extra layers.

**HybridPRISM-12L (first run — FAILED, see Issue 1):**
Loss stuck at ln(16)=2.7726 from step ~200. Model collapse due to MemoryRead LayerNorm. Fixed by removing LayerNorm, rerunning.

**HybridPRISM-12L (rerun) + Transformer-8L: PENDING**

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

### Issue 1: HybridPRISM loss stuck at ln(batch_size) — model collapse (2026-03-21)

**Symptom:** HybridPRISM-12L loss flat at 2.7726 = ln(16) from step ~200 onwards. Model producing identical embeddings for all inputs. AllSlow-6L and AllSlow-12L trained normally.

**Root cause:** MemoryRead had a LayerNorm wrapping the residual: `output = LayerNorm(x + out_proj(retrieved))`. At init (ReZero, out_proj=0), this became `LayerNorm(x)` — an extra normalization between PRISM groups that disrupted gradient flow and caused the model to get stuck at a saddle point.

**Fix:** Removed LayerNorm from MemoryRead. Pure residual: `output = x + out_proj(retrieved)`. At init this is exactly identity. Verified loss decreases on local smoke test.

### Issue 2: HybridPRISM still collapsing after Issue 1 fix (2026-03-21)

**Symptom:** After removing MemoryRead LayerNorm, HybridPRISM-12L loss still converges to ln(16)=2.7726. Steps 100→400: 2.7834→2.7725. Compare AllSlow-12L at step 400: 2.7456 (escaping the saddle). HybridPRISM is attracted TO the degenerate solution rather than escaping it.

**Root cause:** MemoryWrite still had a LayerNorm (`return self.norm(updated)`). This normalized the shared memory_init (std=0.02) to unit variance. Combined with the high write gate retention (bias=+2.0, sigmoid≈0.88), memory stayed ~88% shared across batch elements at unit scale. As MemoryRead out_proj got its first non-zero gradients, it read this large, mostly-shared memory and added a nearly-constant vector to all token states → all embeddings converged after L2 normalization → collapse.

Without LayerNorm, memory stays at std≈0.02 — small enough that early MemoryRead contributions are negligible, letting the PRISM backbone escape the saddle normally.

**Fix:** Removed LayerNorm from MemoryWrite. Pure gated update: `return gate * memory + (1-gate) * retrieved`. Both memory modules now have no LayerNorm, consistent with the principle: no extra normalization between PRISM groups.

### Issue 3: HybridPRISM learning ~8x slower than AllSlow-12L after LayerNorm fixes (2026-03-21)

**Symptom:** After removing both LayerNorms, HybridPRISM-12L no longer collapses but learns very slowly. At step 3000: loss=2.6230 vs AllSlow-12L loss=1.6041. Also a loss spike to 2.8823 at step 1100.

**Root cause:** Gradient clipping dilution. `clip_grad_norm_` clips the global norm across ALL parameters. Memory modules (~2.7M params) get gradients but don't contribute to output (ReZero init). Their gradient norm inflates the denominator, effectively reducing the PRISM backbone's learning rate vs AllSlow-12L which has no memory overhead.

**Initial fix attempt:** Memory warmup (freeze memory for 500 steps). Did NOT help — loss trajectory during frozen period was identical to unfrozen run. Gradient clipping dilution was not the primary cause.

**Actual root cause:** `HybridPRISMEncoder._init_weights` only initialized embeddings. `PRISMEncoder._init_weights` also zeros all `nn.Linear` biases (except recurrence gates). This means every projection, MLP, and fusion layer in HybridPRISM started with random biases (PyTorch Kaiming default) instead of zeros — a completely different optimization starting point from AllSlow-12L.

**Fix:** Added bias zeroing to `HybridPRISMEncoder._init_weights`, matching `PRISMEncoder`. Excludes recurrence gates and MemoryWrite gate_proj (preserves bias=+2.0). Also kept memory warmup infrastructure (may still be useful later).
