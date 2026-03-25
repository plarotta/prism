# PRISM: Experiment Progress

## Current Status: Paper Experiment Infrastructure (complete)

All 7 paper experiment runners are implemented and smoke-tested. The full pipeline is
validated end-to-end: MS MARCO data download (400K queries, 5.5M passages from
Tevatron/msmarco-passage), contrastive training loop, and LoCoV1 zero-shot evaluation.

> Infrastructure ready. Next step: run experiments on GPU (Exp 1 first, then 2-7).

See `PAPER_EXPERIMENT_PLAN.md` for the full experiment design and `paper_dev_plan.md`
for implementation details.

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

## Phase 5: LoCoV1 Long-Context Retrieval Benchmark

Evaluated PRISM-Simplified on the LoCoV1 12-task long-context retrieval benchmark
(Saad-Falcon et al., arXiv:2402.07440). This is the established benchmark from
Hazy Research (Stanford) used to evaluate M2-BERT, BM25, BGE, OpenAI Ada, etc.

**Setup:** Train from scratch on LoCoV1 query-document pairs (InfoNCE, in-batch negatives),
`bert-base-uncased` tokenizer, 5000 steps, no pretraining. RTX PRO 6000 (96 GB).

### 5A — LoCoV1 at max_len=2048 (complete)

| Task | PRISM nDCG@10 | Transformer nDCG@10 | Delta |
|------|--------------|---------------------|-------|
| summ_screen_fd | **0.648** | 0.090 | **+0.557** |
| gov_report | **0.819** | 0.248 | **+0.571** |
| qmsum | **0.648** | 0.409 | **+0.239** |
| qasper_title | **0.653** | 0.092 | **+0.562** |
| qasper_abstract | **0.822** | 0.135 | **+0.687** |
| multifieldqa | **0.890** | 0.547 | **+0.342** |
| 2wikimqa | **0.601** | 0.288 | **+0.313** |
| passage_retrieval | **0.729** | 0.196 | **+0.533** |
| courtlistener_Plain_Text | **0.803** | 0.111 | **+0.692** |
| courtlistener_HTML | **0.803** | 0.111 | **+0.692** |
| legal_case_reports | **0.234** | 0.017 | **+0.217** |
| stackoverflow | **0.613** | 0.088 | **+0.525** |
| **Average** | **0.689** | **0.194** | **+0.494** |

**PRISM wins every single task.** The gap is enormous — 3.5x the Transformer's score on average.
The Transformer barely learned the task (0.194 is close to random on many tasks), while PRISM
achieved competitive scores across all 12 domains. PRISM trained to loss=0.063 vs Transformer's
0.663 — the Transformer simply cannot fit long-document retrieval structure at this sequence length.

Training time: PRISM 1016s, Transformer 2012s. Both used micro_batch=16, no gradient accumulation.

### 5B — LoCoV1 at max_len=8192 (PRISM complete, Transformer OOM'd)

| Task | PRISM nDCG@10 |
|------|--------------|
| summ_screen_fd | 0.491 |
| gov_report | 0.781 |
| qmsum | 0.554 |
| qasper_title | 0.528 |
| qasper_abstract | 0.728 |
| multifieldqa | 0.680 |
| 2wikimqa | 0.526 |
| passage_retrieval | 0.616 |
| courtlistener_Plain_Text | 0.684 |
| courtlistener_HTML | 0.674 |
| legal_case_reports | 0.192 |
| stackoverflow | 0.484 |
| **Average** | **0.578** |

PRISM at 8192 scored 0.578 — lower than its 2048 result (0.689). This is likely because
the longer context makes the task harder (more information to compress), and 5000 steps
may not be enough to learn 8K-token document representations from scratch without pretraining.

The Transformer **OOM'd at micro_batch=4** on a 95 GB GPU. The O(n^2) attention at 8192
tokens requires ~40 GB per training pair — 4 pairs exceeded memory. Fix: reduce to
micro_batch=2 with grad_accum=8 (effective_batch=16). Re-run pending.

Training time: PRISM 3548s (~1 hour) at micro_batch=16 with no accumulation needed.

### Document Length Analysis

Most LoCoV1 documents are very long (median 4K-8K tokens). At max_len=2048, nearly all
documents are heavily truncated (only 0-11% fit fully, except StackOverflow at 65%).
At max_len=8192, all documents fit fully (100% across all tasks).

### Comparison with Published Baselines (approximate)

| Model | Avg nDCG@10 | Notes |
|-------|------------|-------|
| **PRISM-Simplified (2048)** | **0.689** | From scratch, no pretraining, 19M params |
| M2-BERT-80M-8K | ~0.506 | Pretrained, 80M params |
| M2-BERT-80M-2K | ~0.448 | Pretrained, 80M params |
| BM25 | ~0.486 | Lexical baseline |
| **PRISM-Simplified (8192)** | **0.578** | From scratch, no pretraining, 22M params |
| Transformer (2048) | 0.194 | From scratch, 20M params |

**PRISM-Simplified at 2048 tokens outperforms the published M2-BERT-80M-8K baseline** despite
having 4x fewer parameters and no pretraining. This is a strong result, though the comparison
has caveats: (1) we trained on LoCoV1 query-document pairs directly (M2-BERT was evaluated
zero-shot after pretraining + fine-tuning on other data), (2) the published M2-BERT numbers
are approximate values read from their paper.

---

## Phase 6: Hybrid Architecture Experiments (complete)

Addressed the 8K < 2K gap identified in Phase 5.

### Key Results

| Experiment | Finding |
|-----------|---------|
| Improved Baselines | Transformer's 0.194 was bad LR; tuned → 0.672. PRISM still wins +9.8 pts (0.770 vs 0.672) |
| Attentive Pooling | +1.8 pts at 8K, -1.7 pts at 2K. Pooling is not the bottleneck. |
| Decay Ablation | **All-slow (λ=0.99) beats geometric by +7.2 nDCG@10.** Multi-scale decomposition is suboptimal. |

### Conclusions

- The 8K < 2K gap is a backbone capacity problem, not pooling
- Geometric decay spacing is suboptimal — uniform slow decay is better
- The final architecture: all-slow decay, no interference, mean pooling

---

## Phase 7: Paper Experiment Infrastructure (complete)

Built the full experiment pipeline for a publication-ready evaluation. All infrastructure
is implemented, smoke-tested, and validated end-to-end.

### What Was Built

**Data pipelines:**
- `data/msmarco.py` — MS MARCO passage retrieval (Tevatron/msmarco-passage: 400K queries, 5.5M passages, 30 hard negatives/query). Pre-tokenizes and caches to disk.
- `data/loco_eval.py` — LoCoV1 12-task zero-shot evaluator (ported from benchmark_loco.py, eval-only)
- `data/longembed_eval.py` — LongEmbed 6-task zero-shot evaluator (dwzhu/LongEmbed)
- `data/beir_eval.py` — BEIR 15-dataset evaluator (MTEB-based with manual fallback)

**Training infrastructure:**
- `train_contrastive.py` — Model-agnostic contrastive training loop with structured logging, gradient accumulation, cosine schedule, checkpointing, and eval callbacks
- `paper_log.py` — Structured logging utilities (run directories, config saving, hardware capture)

**New model baselines:**
- `mamba_bidir.py` — Bidirectional Mamba encoder (~20M params, CUDA-only with PyTorch fallback)
- `linear_rnn.py` — Single-channel linear RNN ablation (~20M params)

**Experiment runners (7 total):**
- `paper_exp1_controlled.py` — Controlled 4-model comparison at 128/512/2048/8192 tokens
- `paper_exp2_efficiency.py` — Scaling curves (latency, memory, throughput) at 64-16384 tokens
- `paper_exp3_ablations.py` — 10-variant component ablation study (baseline + A through I)
- `paper_exp4_longembed.py` — LongEmbed evaluation on Exp 1 checkpoints
- `paper_exp5_beir.py` — BEIR evaluation on Exp 1 checkpoints
- `paper_exp6_pretrain.py` — Two-stage pretrain (AllNLI + Quora) → fine-tune (MS MARCO)
- `paper_exp7_scaleup.py` — Scale-up to ~80M params (d=768, 8 layers)

### Smoke Test: PASSED

Full pipeline validated:
1. MS MARCO data download and caching
2. Contrastive training (100 steps)
3. LoCoV1 zero-shot evaluation
4. Checkpoint saving and structured logging

### Next: Run Experiments on GPU

Execution order per `paper_dev_plan.md`:
1. Exp 2 (efficiency benchmarks — no training, validates all models)
2. Exp 1a-1e (controlled comparison — the core results)
3. Exp 3 (ablations)
4. Exp 4-5 (LongEmbed + BEIR eval on Exp 1 checkpoints)
5. Exp 6-7 (pretraining + scale-up, if core results are strong)

---

## File Inventory

```
prism/
  # --- Core architecture ---
  prism.py                        # PRISMLayer, PRISMEncoder, PRISMForEmbedding, pooling classes
  baseline_transformer.py         # TransformerEncoder, TransformerForEmbedding
  benchmark_ablations.py          # MeanPooling, NoInterference, LearnedDecayRecurrence, etc.

  # --- New model baselines (paper) ---
  mamba_bidir.py                  # Bidirectional Mamba encoder (~20M params)
  linear_rnn.py                   # Single-channel linear RNN ablation (~20M params)
  hybrid_prism.py                 # Hybrid PRISM experiments (Phase 6)

  # --- Data pipelines ---
  data/
    __init__.py
    msmarco.py                    # MS MARCO passage retrieval (Tevatron/msmarco-passage)
    loco_eval.py                  # LoCoV1 12-task zero-shot evaluator
    longembed_eval.py             # LongEmbed 6-task zero-shot evaluator
    beir_eval.py                  # BEIR 15-dataset evaluator (MTEB + manual fallback)

  # --- Paper experiment runners ---
  paper_log.py                    # Structured logging (run dirs, config, checkpoints)
  train_contrastive.py            # Unified contrastive training loop
  paper_exp1_controlled.py        # Exp 1: 4-model controlled comparison
  paper_exp2_efficiency.py        # Exp 2: scaling curves (latency, memory, throughput)
  paper_exp3_ablations.py         # Exp 3: 10-variant component ablation
  paper_exp4_longembed.py         # Exp 4: LongEmbed evaluation
  paper_exp5_beir.py              # Exp 5: BEIR evaluation
  paper_exp6_pretrain.py          # Exp 6: pretrain + fine-tune pipeline
  paper_exp7_scaleup.py           # Exp 7: scale-up to ~80M params

  # --- Earlier experiment runners (Phases 1-6) ---
  benchmark_scaling.py            # Throughput / memory / latency vs seq length
  benchmark_quality.py            # Synthetic contrastive training + retrieval eval
  benchmark_v2_ablations.py       # V2 targeted fixes ablation (A-G)
  benchmark_real_data.py          # Real-data evaluation (NLI, STS-B, long-doc)
  benchmark_loco.py               # LoCoV1 training + eval (Phase 5)
  benchmark_hybrid.py             # Phase 6: attentive pooling, decay ablations
  benchmark_hybrid_v2.py          # Phase 6 v2 experiments
  investigate.py                  # Phase 2: alpha autopsy + warm init + long-seq
  run_all.py                      # Phase 1 runner
  run_all_v2.py                   # Master runner (phases 1-4)
  dry_run.py                      # Quick validation script
  plot_hybrid_results.py          # Phase 6 result plotting

  # --- Documentation ---
  PROGRESS.md                     # This file
  RESEARCH_SUMMARY.md             # Full research narrative
  PAPER_EXPERIMENT_PLAN.md        # 7-experiment paper design
  paper_dev_plan.md               # Implementation plan for paper experiments
  HYBRID_V2_LOG.md                # Hybrid experiment log
  PRISM_Experiment_Plan.md        # Original 16-week plan
  hydbrid_arch_plan.md            # Hybrid architecture plan
  hybrid_planv2.md                # Hybrid plan v2
  hybrid_v2_dev_plan.md           # Hybrid v2 dev plan
  plan_loco.md                    # LoCoV1 experiment plan
  causal_prism_plan.md            # Causal LM exploration (shelved)
  prism_prior_work.md             # Prior work survey

  # --- Config ---
  pyproject.toml                  # uv project config (optional deps: [paper], [mamba])

  # --- Results ---
  results/                        # All generated data + plots
    loco/                         # LoCoV1 results (Phases 5-6)
    hybrid_v2/                    # Hybrid experiment results
    paper/                        # Paper experiment outputs (exp1-exp7 subdirs)
```
