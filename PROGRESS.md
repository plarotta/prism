# PRISM: Experiment Progress & GPU Run Instructions

## Status: Ready for GPU Run

Everything is built, tested, and verified. One command runs it all:

```bash
uv run python run_all.py
```

---

## What's Done

### 1. Reference Implementation — Optimized (`prism.py`)

The original reference implementation had a critical bottleneck: the linear recurrence
scan used a Python `for t in range(T)` loop — catastrophically slow on GPU due to
per-step kernel launch overhead. Made PRISM **20x slower** than the Transformer.

**Fix: parallel scan via the doubling trick** (`_fast_fixed_decay_scan`). Exploits PRISM's
fixed scalar decay rates. The recurrence `h_t = λ·h_{t-1} + v_t` reduces to O(log T)
rounds of shift-multiply-add — fully vectorized, no Python loop. Result: PRISM went from
20x slower to **2.2x faster** at seq_len=4096 (MPS preliminary results).

Also optimized `CrossScaleInterference` memory from O(C²·B·T·D) → O(C·B·T·D) by
factoring the bilinear product: compute `weighted_VH = α @ VH` first, then
`interference = UH ⊙ weighted_VH`.

Also provides `_parallel_scan` for the general time-varying gate case.

### 2. Transformer Baseline (`baseline_transformer.py`)

Standard pre-norm Transformer encoder. Same embedding interface as PRISM
(input_ids → embedding vector). Attentive pooling. Parameter-matched configurations:
Small (~23M), Base (~110M). InfoNCE contrastive wrapper.

### 3. Scaling Benchmark (`benchmark_scaling.py`)

Measures throughput, latency, peak memory at seq lengths
[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384].
Forward-only (inference) and forward+backward (training).

**Preliminary MPS results (prism_small vs transformer_small, batch=8):**

| Seq Len | PRISM (ms) | Transformer (ms) | Speedup |
|---------|-----------|------------------|---------|
| 64      | 49.8      | 4.8              | 0.10x   |
| 256     | 54.3      | 13.6             | 0.25x   |
| 512     | 60.7      | 32.3             | 0.53x   |
| 1024    | 105.4     | 85.6             | 0.81x   |
| **2048**| **201.5** | **271.3**        | **1.35x** |
| **4096**| **432.7** | **957.9**        | **2.21x** |
| **8192**| **448.8** | **1871.0**       | **4.17x** |

Training: crossover ~1024 tokens. Transformer OOMs at 4096 training; PRISM runs fine.

Will be re-run on CUDA with accurate memory tracking and 16384 length.

### 4. Quality Comparison (`benchmark_quality.py`)

Synthetic contrastive embedding task (200 topics, 40% noise, 5K corpus).

| Metric         | PRISM  | Transformer |
|----------------|--------|-------------|
| MRR            | 0.902  | 0.922       |
| Recall@1       | 0.854  | 0.880       |
| Recall@5       | 0.966  | 0.980       |
| Recall@10      | 0.984  | 0.992       |

PRISM within **2 MRR points** on a short-sequence task where attention has maximum
advantage. Gap expected to narrow at longer sequences.

### 5. Ablation Study (`benchmark_ablations.py`)

Four variants, all verified to construct, train, and evaluate:

| Variant                   | What it tests |
|---------------------------|--------------|
| A: No Interference        | removes the core novel contribution (cross-scale bilinear) |
| B: Learned Decay          | tests whether fixed geometric spacing matters |
| C: Additive Interference  | tests whether multiplicative (bilinear) features matter |
| D: Mean Pooling           | tests whether covariance pooling matters |

All variants use the fast parallel scan. Not yet fully trained — awaiting GPU run.

### 6. Master Runner (`run_all.py`)

Single command runs all three benchmarks, generates all plots, writes combined
`summary.json`. Handles CUDA/MPS/CPU auto-detection. Each benchmark is wrapped in
try/except so a failure in one doesn't block the others.

---

## GPU Run Instructions

### Setup

```bash
# Clone/copy the repo to your GPU instance
# Then:
uv init   # if no pyproject.toml already
uv add torch numpy matplotlib
```

For CUDA with specific PyTorch build:
```bash
uv add torch --index-url https://download.pytorch.org/whl/cu124
```

### Run Everything

```bash
uv run python run_all.py
```

Estimated time on A100: ~30-45 minutes total.

Or run individually:
```bash
uv run python benchmark_scaling.py      # ~15 min (includes 16K length)
uv run python benchmark_quality.py      # ~3 min
uv run python benchmark_ablations.py    # ~8 min
```

### What You'll Get

```
results/
├── scaling_results.json     # raw timing + memory data
├── quality_results.json     # retrieval metrics
├── ablation_results.json    # per-variant metrics
├── summary.json             # combined key findings
├── scaling_comparison.png   # log-log latency + throughput (4 panels)
├── speedup.png              # speedup curve
├── memory_comparison.png    # peak GPU memory (CUDA only)
├── theoretical_flops.png    # theoretical FLOP comparison
├── training_curves.png      # loss + accuracy curves
├── quality_comparison.png   # retrieval metrics bar chart
├── ablation_comparison.png  # training curves + metrics (3 panels)
└── ablation_impact.png      # MRR delta waterfall chart
```

---

## Optional Follow-ups (not blocking)

### Scale to Base size (~110M params)

Both `prism.py` and `baseline_transformer.py` have `prism_base()` / `transformer_base()`
configs. Edit the benchmark imports to use them. The scaling advantage grows with model
dimension because attention's O(n²·d) grows with d.

### Fused CUDA scan kernel

`_fast_fixed_decay_scan` uses O(log T) PyTorch kernel launches. A fused Triton kernel
could do it in one launch, pushing the crossover from ~1024 down to ~256-512.
The `mamba_ssm` package has compatible infrastructure.

---

## File Inventory

```
prism/
├── prism.py                    # Core PRISM (optimized parallel scan + interference)
├── baseline_transformer.py     # Transformer baseline (parameter-matched)
├── benchmark_scaling.py        # Throughput / memory / latency benchmark
├── benchmark_quality.py        # Contrastive training + retrieval evaluation
├── benchmark_ablations.py      # Component ablation study (4 variants)
├── run_all.py                  # Master runner — single command for everything
├── PRISM_Technical_Report.txt  # Architecture design document
├── PRISM_Experiment_Plan.md    # Full 16-week experiment plan
├── PROGRESS.md                 # This file
└── pyproject.toml              # uv project config
```

## GPU Instance Recommendations

- **Minimum:** 1x A100 40GB — all Small experiments + 16K scaling
- **Ideal:** 1x A100 80GB — can also do Base size
- **Budget:** 1x A10 24GB — Small experiments, may OOM at 16K
