# PRISM: Experiment Progress

## Current Status: Simplified Architecture Validated, Moving to Real Data

**The revised thesis:** A bidirectional multi-channel state-space encoder with fixed geometric
decay rates and mean pooling achieves competitive embedding quality with a Transformer of
equal parameter count, while scaling linearly in sequence length. At 2048+ tokens, it is
faster. At 8192 tokens, 6x faster. At 16384 tokens, the Transformer OOMs.

The original "cross-scale interference" and "covariance pooling" mechanisms are dropped.
They are somewhere between inert and actively harmful at this scale. The contribution is
the scaling proof, not the architectural novelty.

---

## Phase 1 Results: Scaling (A100 80GB)

PRISM vs Transformer inference latency (batch=8, ~26M params each):

| Seq Len | PRISM (ms) | Transformer (ms) | Speedup |
|---------|-----------|------------------|---------|
| 64      | 32.2      | 3.0              | 0.09x   |
| 256     | 36.5      | 3.8              | 0.10x   |
| 512     | 41.3      | 8.2              | 0.20x   |
| 1024    | 48.8      | 20.0             | 0.41x   |
| **2048**| **63.7**  | **68.9**         | **1.08x** (crossover) |
| **4096**| **93.2**  | **238.2**        | **2.56x** |
| **8192**| **170.5** | **1026.2**       | **6.02x** |
| **16384**| **362.3**| **OOM**          | **inf** |

Training (fwd+bwd): crossover at ~2048, 1.73x at 4096, Transformer OOMs at 8192+.

## Phase 1 Results: Quality (Synthetic Data)

Synthetic contrastive task (200 topics, 40% noise, 5K corpus):

| Metric         | PRISM  | Transformer |
|----------------|--------|-------------|
| MRR            | 0.950  | 0.991       |
| Recall@1       | 0.922  | 0.984       |
| Recall@5       | 0.984  | 1.000       |

4-point MRR gap at short sequences (64/96 tokens) — worst case for PRISM.

## Phase 2 Results: Ablation Conclusions

| Component                    | Verdict              |
|------------------------------|---------------------|
| Multi-channel recurrence     | **Works** — core engine |
| Bidirectional gated fusion   | **Works** — needed for embeddings |
| Mean pooling                 | **Works** — +3.4 MRR vs covariance |
| Linear scaling               | **Confirmed** — 6x at 8K |
| Cross-scale interference     | **Drop** — <0.01 MRR contribution |
| Fixed vs learned decay       | **No difference** — keep fixed (simpler) |
| Bilinear vs additive mixing  | **No difference** — both inert |
| Covariance pooling           | **Drop** — actively harmful (-3.4 MRR) |

**Simplified PRISM = multi-channel stratified recurrence + bidirectional fusion + mean pooling.**

---

## Phase 3: Real Data Evaluation (next)

### 3A — Real language, real tokenizer

Train simplified PRISM and parameter-matched Transformer on NLI contrastive pairs
(SNLI + MNLI entailment pairs, ~275K) using `bert-base-uncased` tokenizer (30,522 vocab).
Evaluate on STS-Benchmark (Spearman correlation). This proves the architecture works
on real language, not just synthetic token distributions.

### 3B — The money experiment: quality vs sequence length

The central claim is that PRISM's advantage grows with sequence length. To test this
on real text:

1. Train both models on NLI data (fixed training, same for both).
2. Evaluate retrieval on a long-document dataset at controlled lengths:
   128, 256, 512, 1024, 2048 tokens.
3. Plot quality and throughput as a function of sequence length.
4. Show the crossover: Transformer wins at short, PRISM wins at long.

Dataset candidates for long-document retrieval:
- Wikipedia paragraph matching (same-article = positive)
- IMDB reviews (natural 200-500 token range)
- CNN/DailyMail articles (natural 300-800 token range)

### 3C — Standard benchmarks (if 3A+3B are positive)

Run on MTEB subset: STS + Retrieval + Classification.
Compare with off-the-shelf baselines (all-MiniLM-L6-v2, GTE-base).

### Dependencies needed

```
uv add datasets transformers
```

### Success criteria

| Test | Pass condition |
|------|---------------|
| STS-B Spearman | PRISM within 3 points of Transformer |
| Retrieval at 128 tokens | Transformer ≥ PRISM (expected) |
| Retrieval at 512+ tokens | PRISM ≥ Transformer |
| Retrieval at 2048 tokens | PRISM > Transformer AND 2x+ throughput |

---

## File Inventory

```
prism/
├── prism.py                    # Core PRISM architecture
├── baseline_transformer.py     # Transformer baseline (parameter-matched)
├── benchmark_scaling.py        # Throughput / memory / latency vs seq length
├── benchmark_quality.py        # Synthetic contrastive training + retrieval eval
├── benchmark_ablations.py      # Component ablation study (4 variants)
├── benchmark_real_data.py      # Phase 3: real-data evaluation (NEW)
├── investigate.py              # Phase 2: alpha autopsy + warm init + long-seq
├── run_all.py                  # Master runner for Phase 1
├── PRISM_Technical_Report.txt  # Original architecture design
├── PRISM_Experiment_Plan.md    # Original 16-week plan
├── PROGRESS.md                 # This file
├── pyproject.toml              # uv project config
└── results/                    # All generated data + plots
```
