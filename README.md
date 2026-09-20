# PRISM

**Projected Recurrent Information Stratification with Mixing** — a sub-quadratic,
bidirectional multi-channel state-space encoder for sequence embeddings
(retrieval / similarity), designed as a linear-time alternative to Transformer
encoders for long documents.

The research hypothesis is that a bidirectional multi-channel gated recurrence
with mean pooling can offer a useful quality/compute tradeoff for long-document
embeddings. **A clean quality advantage over an optimized Transformer is not yet
established.** Historical scores and scaling comparisons need to be rerun after
the correctness and evaluation fixes described in [REVIVAL_REPORT.md](REVIVAL_REPORT.md).

## Status

The revival implementation corrects padding leakage, restores learned-decay
gradients, enables optimized Transformer attention, and adds exact gradient
caching with hard negatives. Source validation selects checkpoints; LoCoV1 is
evaluated only after selection. Local CPU regression tests and diagnostic
results are complete. **The corrected GPU retrieval comparison is pending.**

Gradient caching reduced peak live autograd-saved tensor storage by **10.04×**
in the recorded CPU fixture, with identical loss and maximum gradient error
below `1e-6`. This is not a GPU allocator measurement or a retrieval-quality gain.

## Documentation

| File | What |
|------|------|
| `README.md` | This overview + index |
| `REVIVAL_REPORT.md` | Current findings, reproducible evidence, and the decisive rerun |
| `GPU_RUNBOOK.md` | Setup and how to run the experiment suite on a GPU box |
| `PROGRESS.md` | Historical results log; see the correction notice at the top |
| `PAPER_EXPERIMENT_PLAN.md` | The 7-experiment paper design |
| `RESEARCH_SUMMARY.md` | Full research narrative (the journey + findings) |
| `EXPERIMENT_LOG.md` | Fine-grained running experiment log |

Older Phase 1–6 plans, reports, and exploratory code are kept under `archive/`.

## Quick start

```bash
uv sync                                 # core dependencies
uv run --with pytest python -m pytest -q # offline correctness tests
uv run python benchmark_revival.py      # offline before/after audit
```

Full setup, data prep, run order, and memory notes are in `GPU_RUNBOOK.md`.
The staged GPU rerun is in `REVIVAL_REPORT.md`. The legacy PyTorch doubling scan
does `O(T log T)` work; the existing fixed-block Triton scan does `O(T)` work in
sequence length. GPU kernel speedups have not been measured in this revival.

## Code layout

```
# Architecture
prism.py                  # PRISM encoder, recurrence, pooling
baseline_transformer.py   # Transformer baseline
mamba_bidir.py            # Bidirectional Mamba baseline
linear_rnn.py             # Single-channel linear-RNN ablation baseline
paper_components.py       # MeanPooling / NoInterference / LearnedDecayRecurrence

# Training & eval infra
train_contrastive.py      # Model-agnostic InfoNCE training loop
paper_log.py              # Run dirs, config/checkpoint/metric logging
eval_checkpoint.py        # Offline re-evaluation of saved checkpoints
data/                     # MS MARCO loader + LoCoV1 / LongEmbed / BEIR evaluators

# Experiment runners
paper_exp1_controlled.py  # Controlled architecture comparison (core result)
paper_exp2_efficiency.py  # Scaling curves (latency / memory / throughput)
paper_exp3_ablations.py   # Component ablation study
paper_exp4_longembed.py   # LongEmbed evaluation
paper_exp5_beir.py        # BEIR evaluation
paper_exp6_pretrain.py    # Pretrain + fine-tune pipeline
paper_exp7_scaleup.py     # Scale-up to ~80M params

results/                  # Generated run outputs
archive/                  # Legacy Phase 1-6 code + docs
```
