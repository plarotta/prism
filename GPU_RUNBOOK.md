# PRISM Paper Experiment Suite — GPU Runbook

_Last updated: 2026-06-21_

How to set up and run the Experiment 1 controlled comparison (and friends) on a
CUDA GPU box. Written for the RTX 3090 / similar; larger cards relax the
sequence-length memory limits noted below.

---

## 0. Prerequisites

- NVIDIA GPU with a recent CUDA driver (the suite uses bf16 AMP when available)
- [`uv`](https://docs.astral.sh/uv/) installed
- Disk: ~10–20 GB free for the MS MARCO token cache + checkpoints
- Network access to HuggingFace (first run downloads MS MARCO / LoCoV1)

---

## 1. Setup

```bash
git clone <repo> && cd prism

# Core deps + paper extras (mteb for BEIR) + Mamba baseline
uv sync --extra paper --extra mamba
```

**Mamba note (important):** `mamba-ssm` needs a CUDA toolchain and
`causal-conv1d`. If the extra above fails to build the kernels:

```bash
uv pip install causal-conv1d>=1.4.0
uv pip install mamba-ssm>=2.0.0 --no-build-isolation
```

Verify Mamba is real (not the fallback) before running it as a baseline:

```bash
uv run python -c "import mamba_ssm; print('mamba_ssm OK')"
```

> The suite **refuses** to train the Mamba baseline on the `SimpleDiagSSM`
> fallback (it isn't a faithful Mamba and would invalidate the comparison).
> If `mamba_ssm` is missing it errors out; override only for debugging with
> `--allow-mamba-fallback`.

### Weights & Biases (results dashboard)

`wandb` ships in the `paper` extra. Every run pushes config, training curves,
eval metrics (MS MARCO MRR@10, LoCoV1/LongEmbed/BEIR nDCG@10), and final
summaries to your dashboard — no per-script flags needed. Log in once on the
box:

```bash
uv run wandb login        # or: export WANDB_API_KEY=...
```

Configure the target (optional):

| Env var | Default | Purpose |
|---------|---------|---------|
| `WANDB_PROJECT` | `prism` | dashboard project |
| `WANDB_ENTITY` | (your default) | team/user |
| `PRISM_WANDB=0` | unset | disable W&B for a run |
| `WANDB_MODE=offline` | unset | log locally, `wandb sync` later |

Runs are grouped by experiment (e.g. `exp1_1a`) and named by run dir
(`prism_run0`), so the 4-model comparisons line up on shared charts. If
`wandb` is absent or `PRISM_WANDB=0`, logging silently no-ops.

---

## 2. Prepare data (one-time, cached)

Pre-tokenize MS MARCO per training length. Each length is cached separately
under `data/.cache/msmarco/maxlen_<N>/`.

```bash
uv run python -m data.msmarco --prepare --max-len 128    # for 1a
uv run python -m data.msmarco --prepare --max-len 512    # for 1b
uv run python -m data.msmarco --prepare --max-len 2048   # for 1c–1e
```

LoCoV1 (used by 1c–1e eval) downloads automatically on first eval.

---

## 3. Smoke test (always run first)

Validates the full pipeline end-to-end in ~1 min on GPU:

```bash
uv run python paper_exp1_controlled.py --smoke-test
```

Expect it to finish with `=== SMOKE TEST PASSED ===` and a non-zero
`msmarco_dev_mrr@10` in the eval output.

---

## 4. Run the experiments

Recommended order (cheap/validating first, decisive next):

```bash
# Exp 2 — efficiency curves (no training; validates all 4 models load + run)
uv run python paper_exp2_efficiency.py

# Exp 1 — controlled comparison (the core result). One sub-exp at a time:
uv run python paper_exp1_controlled.py --sub-exp 1a     # short  (128)
uv run python paper_exp1_controlled.py --sub-exp 1b     # medium (512)
uv run python paper_exp1_controlled.py --sub-exp 1c     # long   (2048)
uv run python paper_exp1_controlled.py --sub-exp 1d     # LoCoV1 zero-shot @2048
uv run python paper_exp1_controlled.py --sub-exp 1e     # LoCoV1 @8192 (no Transformer)

# Then, if core results hold:
uv run python paper_exp3_ablations.py
uv run python paper_exp4_longembed.py
uv run python paper_exp5_beir.py
uv run python paper_exp6_pretrain.py
uv run python paper_exp7_scaleup.py
```

Useful flags on `paper_exp1_controlled.py`:

| Flag | Purpose |
|------|---------|
| `--models prism,mamba` | run a subset of models for a sub-exp |
| `--sub-exp 1c` / `--all` | one sub-exp / everything |
| `--n-steps`, `--micro-batch`, `--grad-accum`, `--lr` | training overrides |
| `--eval-every` | eval cadence (default 5000) |
| `--allow-mamba-fallback` | debug only — runs the non-faithful Mamba |

Default recipe: 50k steps, effective batch 128 (`micro_batch=16 × grad_accum=8`),
lr 3e-4 cosine, eval + checkpoint every 5k.

---

## 5. Re-scoring checkpoints offline

This run starts from scratch — no prior checkpoints are reused. But once runs
exist, `eval_checkpoint.py` re-scores any saved checkpoint without retraining
(useful if an eval was missed or you want LoCoV1 on a checkpoint trained without
it):

```bash
# Score every model under a sub-experiment at its latest checkpoint
uv run python eval_checkpoint.py --exp-dir results/paper/exp1_1a --max-queries 0

# A specific run + step, also against LoCoV1
uv run python eval_checkpoint.py \
    --run-dir results/paper/exp1_1c/prism_run0 --step 50000 --locov1
```

> Note: experiment artifacts (`*.pt`, `*.zip`) are gitignored — they are large
> and regenerated by each run, so don't commit them.

---

## 6. Where results land

```
results/paper/exp1_<id>/<model>_run0/
    config.json            # frozen config + hardware + git hash
    train_log.jsonl        # per-100-step loss / lr / grad_norm / mem
    final_metrics.json     # best_step, best_metric, timing, params
    checkpoints/step_*.pt
    eval/step_*_eval.json   # msmarco_dev_mrr@10 (+ locov1 nDCG@10 for 1c–1e)
```

If a model crashes, the sweep continues and the traceback is saved to
`results/paper/exp1_<id>/<model>_ERROR.txt` — check there first.

---

## 7. Memory notes (RTX 3090, 24 GB)

- 1a (128) / 1b (512): comfortable at the default `micro_batch=16`.
- 1c–1e (2048–8192): attention memory grows fast. If the **Transformer** OOMs,
  drop to `--micro-batch 4 --grad-accum 32` (keeps effective batch 128). PRISM /
  Mamba scale linearly and should be fine.
- 1e (8192) intentionally excludes the Transformer (OOMs); it runs PRISM, Mamba,
  Linear-RNN only.
- Bigger cards (A100/L40/6000): raise `--micro-batch` to use the headroom.
```
