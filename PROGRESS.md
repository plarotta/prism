# PRISM: Experiment Progress

## Current Status: Moving to Real Data

Simplified PRISM (multi-channel recurrence + mean pooling) **beats the Transformer by
9.5 MRR points** at longer sequences (256/512 tokens, 2000 steps) while scaling linearly.
The original "novel" components (interference, covariance pooling) are dropped — they
don't help. The contribution is a clean, empirical result:

> A bidirectional multi-channel state-space encoder with fixed geometric decay rates
> learns better embeddings than a parameter-matched Transformer at sequences beyond
> ~256 tokens, with linear-time scaling, 6x inference speedup at 8K tokens, and
> 18x lower memory at 4K tokens.

Next step: real data (NLI training, STS-B eval, Wikipedia retrieval at controlled lengths).

---

## Phase 1 Results: A100 80GB

### Scaling

PRISM vs Transformer inference latency (batch=8, ~26M params each):

| Seq Len | PRISM (ms) | Transformer (ms) | Speedup | PRISM Mem | Trans Mem | Mem Ratio |
|---------|-----------|------------------|---------|-----------|-----------|-----------|
| 64      | 32.2      | 3.0              | 0.09x   | 11 MB     | 6 MB      | 0.5x      |
| 256     | 36.5      | 3.8              | 0.10x   | 45 MB     | 38 MB     | 0.8x      |
| 512     | 41.3      | 8.2              | 0.20x   | 89 MB     | 126 MB    | 1.4x      |
| 1024    | 48.8      | 20.0             | 0.41x   | 178 MB    | 453 MB    | 2.5x      |
| 2048    | 63.7      | 68.9             | 1.08x   | 357 MB    | 3,322 MB  | **9.3x**  |
| 4096    | 93.2      | 238.2            | 2.56x   | 713 MB    | 13,086 MB | **18.3x** |
| 8192    | 170.5     | 1,026.2          | 6.02x   | 1,426 MB  | 51,943 MB | **36.4x** |
| 16384   | 362.3     | OOM              | inf     | 2,852 MB  | OOM       | inf       |

Training (fwd+bwd): crossover at ~2048, 1.73x at 4096, Transformer OOMs at 8192+.

### Quality — Short Sequences (Phase 1)

Synthetic contrastive task, 400 steps, seq_len_q=64, seq_len_p=96:

| Metric    | PRISM (full) | Transformer |
|-----------|-------------|-------------|
| MRR       | 0.950       | 0.991       |
| Recall@1  | 0.922       | 0.984       |
| Recall@5  | 0.984       | 1.000       |

Transformer wins by ~4 MRR points at short sequences. Expected — attention has full
pairwise interaction at low cost for short inputs.

### Ablations — Novel Components Don't Help

| Variant                  | MRR    | Delta   | Verdict             |
|--------------------------|--------|---------|---------------------|
| Full PRISM               | 0.9505 | —       | baseline            |
| A: No Interference       | 0.9412 | -0.009  | within noise        |
| B: Learned Decay         | 0.9501 | -0.000  | identical           |
| C: Additive Interference | 0.9482 | -0.002  | within noise        |
| D: Mean Pooling          | 0.9844 | **+0.034** | **better without cov pooling** |

**Conclusion:** Drop interference (inert). Drop covariance pooling (actively harmful).
Keep fixed decay (simpler, same quality as learned). Mean pooling is strictly better.

---

## Phase 2 Results: Investigation (completed)

### The Reversal: PRISM Dominates at Longer Sequences

PRISM-MeanPool vs Transformer, 2000 steps, seq_q=256, seq_p=512:

| Metric    | PRISM-MeanPool | Transformer | Delta     |
|-----------|---------------|-------------|-----------|
| MRR       | **0.985**     | 0.890       | **+0.095** |
| Recall@1  | **0.976**     | 0.832       | **+0.144** |
| Recall@5  | **0.998**     | 0.966       | **+0.032** |
| Recall@10 | **1.000**     | 0.992       | **+0.008** |

**This is the central result.** At longer sequences, the Transformer's quality degrades
while PRISM's improves. The 4-point deficit at short sequences reverses to a **9.5-point
advantage** at 256/512 tokens.

Additional observations from the training curves:
- PRISM converges faster: loss drops below 0.2 by step ~300, Transformer takes until ~600.
- PRISM reaches lower final loss: 0.020 vs 0.034.
- Transformer trains 2.5x faster wall-clock (283s vs 714s for 2000 steps at this length),
  but PRISM uses that time more effectively — better final quality despite slower steps.

### Component Assessment (Final)

| Component                    | Verdict                       |
|------------------------------|-------------------------------|
| Multi-channel recurrence     | **Core engine** — works       |
| Fixed geometric decay rates  | **Keep** — simpler, same quality |
| Bidirectional gated fusion   | **Keep** — needed for embeddings |
| Mean pooling                 | **Keep** — +3.4 MRR vs covariance |
| Linear O(n) scaling          | **Confirmed** — 6x at 8K, 18x memory savings at 4K |
| Cross-scale interference     | **Drop** — <0.01 MRR contribution |
| Covariance pooling           | **Drop** — actively harmful   |

---

## Phase 3: Real Data Evaluation (next)

### 3A — Real language, real tokenizer

Train simplified PRISM and Transformer on NLI entailment pairs (SNLI + MNLI)
using `bert-base-uncased` tokenizer (30,522 vocab). Evaluate STS-Benchmark
(Spearman correlation). Proves the architecture works on real language.

### 3B — Quality vs sequence length (the money chart)

Evaluate retrieval quality at controlled lengths (128, 256, 512, 1024, 2048) on
Wikipedia paragraph matching. Plot MRR and throughput vs length. Show the crossover.

### 3C — Standard benchmarks (conditional)

If 3A+3B are positive: MTEB subset (STS + Retrieval + Classification).

### Success Criteria

| Test                      | Pass Condition                                    |
|---------------------------|---------------------------------------------------|
| STS-B Spearman            | PRISM within 3 points of Transformer              |
| Retrieval at 128 tokens   | Transformer >= PRISM (expected)                   |
| Retrieval at 512+ tokens  | PRISM >= Transformer                              |
| Retrieval at 2048 tokens  | PRISM > Transformer AND 2x+ throughput            |

### To Run

```bash
uv add datasets transformers scipy
uv run python benchmark_real_data.py
```

---

## File Inventory

```
prism/
├── prism.py                    # Core PRISM architecture
├── baseline_transformer.py     # Transformer baseline (parameter-matched)
├── benchmark_scaling.py        # Throughput / memory / latency vs seq length
├── benchmark_quality.py        # Synthetic contrastive training + retrieval eval
├── benchmark_ablations.py      # Component ablation study (4 variants)
├── benchmark_real_data.py      # Phase 3: real-data evaluation
├── investigate.py              # Phase 2: alpha autopsy + warm init + long-seq showdown
├── run_all.py                  # Master runner for Phase 1
├── PRISM_Technical_Report.txt  # Original architecture design
├── PRISM_Experiment_Plan.md    # Original 16-week plan
├── PROGRESS.md                 # This file
├── pyproject.toml              # uv project config
└── results/                    # All generated data + plots
```
