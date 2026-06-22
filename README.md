# PRISM

**Projected Recurrent Information Stratification with Mixing** — a sub-quadratic,
bidirectional multi-channel state-space encoder for sequence embeddings
(retrieval / similarity), designed as a linear-time alternative to Transformer
encoders for long documents.

The surviving claim, after extensive ablation: *a bidirectional multi-channel
gated linear recurrence with fixed decay and mean pooling matches a
parameter-matched Transformer at short sequences, beats it at long sequences
(256+ tokens), and scales linearly in time and memory.*

## Status

Phases 1–6 (exploratory) are complete; the project is now executing the
**publication-quality evaluation** (Experiment 1–7) on GPU. See `PROGRESS.md`
for current results and `GPU_RUNBOOK.md` to run the suite.

## Documentation

| File | What |
|------|------|
| `README.md` | This overview + index |
| `GPU_RUNBOOK.md` | Setup and how to run the experiment suite on a GPU box |
| `PROGRESS.md` | High-level results log (the canonical status) |
| `PAPER_EXPERIMENT_PLAN.md` | The 7-experiment paper design |
| `RESEARCH_SUMMARY.md` | Full research narrative (the journey + findings) |
| `EXPERIMENT_LOG.md` | Fine-grained running experiment log |

Older Phase 1–6 plans, reports, and exploratory code are kept under `archive/`.

## Quick start

```bash
uv sync --extra paper --extra mamba      # deps (+ Mamba baseline)
uv run python paper_exp1_controlled.py --smoke-test
```

Full setup, data prep, run order, and memory notes are in `GPU_RUNBOOK.md`.

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
