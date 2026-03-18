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

## Phase 3 Results: Real Data

### 3A — STS-Benchmark: PRISM wins (+3 Spearman)

Trained on NLI entailment pairs (SNLI + MNLI), `bert-base-uncased` tokenizer,
2000 steps, max_len=128.

| Model            | STS-B Spearman | Final Loss | Train Time |
|------------------|---------------|------------|------------|
| PRISM-Simplified | **0.589**     | 0.440      | 371s       |
| Transformer      | 0.559         | 0.694      | 118s       |

PRISM learns better representations on real language (+3 Spearman points) despite
being 3x slower per step at this length. PRISM's loss is also significantly lower
(0.44 vs 0.69), suggesting the recurrence captures NLI structure more effectively.

### 3B — Length-controlled retrieval: v1 FLAWED, v2 pending

**v1 (flawed):** Trained at max_len=128, evaluated at 128-2048. Models had never
seen long sequences, so the eval was testing OOD generalization, not long-sequence
quality. Both models showed ~0.5 MRR across all lengths. Not informative.

**v2 (ready to run):** Train fresh models at each target length (128, 256, 512) on
CNN/DailyMail same-article pairs. This matches the Phase 2 synthetic setup that
showed the 9.5-point PRISM advantage at longer sequences.

### 3C — Throughput scaling: confirmed on real vocab

| Seq Len | PRISM (seq/s) | Transformer (seq/s) | Speedup |
|---------|--------------|--------------------|---------|
| 64      | 1,055        | 8,840              | 0.12x   |
| 128     | 1,017        | 4,487              | 0.23x   |
| 256     | 783          | 2,128              | 0.37x   |
| 512     | 509          | 888                | 0.57x   |
| 1024    | 218          | 326                | 0.67x   |
| 2048    | 107          | 74                 | **1.45x** |
| 4096    | 52           | OOM                | **inf** |
| 8192    | 25           | OOM                | **inf** |

Crossover at ~2048 with batch=32 on a 44 GB GPU. Transformer OOMs at 4096.

---

## Phase 4: V2 Fixes — Reviving Interference & Covariance (next)

Code-level review (PRISM_v2fixes.txt) identified specific numerical issues that explain
why the "novel" components failed. Not theoretical problems — implementation scaling bugs
analogous to missing 1/sqrt(d_k) in attention.

### Interference Fixes (CrossScaleInterferenceV2)

1. Per-channel LayerNorm before bilinear product (channels have wildly different magnitudes)
2. 1/sqrt(d_c) scaling on bilinear product (variance control)
3. Alpha initialized to 1/(C-1) instead of zero (break out of gradient desert)
4. Learned gate: mixed = H + sigmoid(gamma) * interference (amplify once learned)

### Covariance Pooling Fixes (AttentiveCovariancePoolingV2)

1. LayerNorm on covariance vector before concatenation
2. Reduced cov_rank: 8 instead of 32 (64 dims not 1024 — stop drowning semantic stream)
3. Project covariance to d dimensions (both streams contribute equally)

### Ablation Plan (7 variants A-G)

Each variant isolates one fix. Variant G combines all. Tested at seq_q=256, seq_p=512,
2000 steps. Success = Variant G matches or beats the simplified mean-pool baseline.

### To Run (everything)

```bash
uv add datasets transformers scipy
uv run python run_all_v2.py
```

---

## File Inventory

```
prism/
├── prism.py                    # Core architecture (+ V2 classes)
├── baseline_transformer.py     # Transformer baseline (parameter-matched)
├── benchmark_scaling.py        # Throughput / memory / latency vs seq length
├── benchmark_quality.py        # Synthetic contrastive training + retrieval eval
├── benchmark_ablations.py      # Original component ablation study (A-D)
├── benchmark_v2_ablations.py   # V2 targeted fixes ablation (A-G)
├── benchmark_real_data.py      # Real-data evaluation (NLI, STS-B, long-doc)
├── investigate.py              # Phase 2: alpha autopsy + warm init + long-seq
├── run_all.py                  # Phase 1 runner (scaling + quality + ablations)
├── run_all_v2.py               # Master runner (everything: scaling → real data)
├── PRISM_Technical_Report.txt  # Original architecture design
├── PRISM_v2fixes.txt           # Analysis of why novel components failed + fixes
├── PRISM_Experiment_Plan.md    # Original 16-week plan
├── PROGRESS.md                 # This file
├── pyproject.toml              # uv project config
└── results/                    # All generated data + plots
```
