# PRISM revival: corrected training, reproducible evidence, and the remaining question

PRISM now has a more reliable implementation and a substantially more memory-efficient
way to train with large contrastive batches. It does **not** yet have a demonstrated
retrieval-quality breakthrough. The old evidence overstated what had been established,
and the new structural probe is deliberately reported even though it did not produce a win.

Baseline revision: `3efe85c`. Local measurements: Python 3.12.3, PyTorch 2.10.0+cpu,
two CPU threads. No CUDA training, kernel benchmarking, or real-data retrieval run
was performed in this workspace.

## What was holding the project back

| Finding | Consequence | Implemented change |
|---|---|---|
| LayerNorm adds a learned bias after padding was zeroed | Padding can enter the backward recurrence and change valid-token representations | Mask after normalization; gather the last valid forward state for pooling |
| Learned decays were converted to Python scalars with `.item()` | The purported learned-decay ablation had no decay gradient | Differentiable tensor scan, finite logit initialization, derivative tests |
| Gradient accumulation computed a separate contrastive loss per microbatch | `16 × 8 = 128` examples per optimizer update still meant only 16 in-batch candidates | Exact gradient caching, including dropout RNG replay and uneven chunks |
| Stored BM25 negatives were never passed to the loss | Training omitted a planned source of difficult examples | Bounded negative loading, sampling, encoding and loss support |
| Repeated queries/documents could become in-batch false negatives | Contradictory negative labels | Stable group IDs and duplicate masking |
| LoCoV1 helped choose checkpoints and trigger early stopping | Transfer evaluation was being used for model selection | Source-dev selection; one held-out LoCoV1 evaluation after selection |
| Transformer requested discarded attention weights | Historical speed and OOM results did not use the optimized SDPA path | `need_weights=False`; narrower OOM detection; consistent memory batch size |
| Model seeds were set after initialization | Runs with the same nominal seed could start from different weights | Seed before model construction |
| Offline evaluators rebuilt absolute-position tables at evaluation length | Checkpoints could fail on shorter lengths, or loading could hide missing keys | Reconstruct checkpoint capacity, load strictly, reject unsupported extrapolation |

The Transformer change follows the documented `need_weights=False` optimization in
[PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
It enables SDPA dispatch; it does not guarantee a particular CUDA backend on every device.
The new cache implements the established method from
[Gao et al., Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983),
and is not claimed as a new algorithm.

## Verified results

The recorded audit is [verification.json](results/revival/verification.json).

| Measurement | Before | After |
|---|---:|---:|
| Minimum self-cosine after adding 100 padding tokens | 0.371 | 1.000 |
| Maximum normalized embedding drift from that padding | 0.418 | 0.000 |
| Learned-decay parameter gradients | Absent | Finite and nonzero |
| Finite initial learned-decay logits | No | Yes |

The padding fixture uses identical model weights and deliberately nonzero LayerNorm
offsets (standard deviation 0.1). It is a regression reproduction, not a measurement
of an existing trained checkpoint. Tests also check gradient invariance, covariance
pooling's last-valid-state path, and invariance to batch companions/masked token contents.

For a logical contrastive batch of 32, encoded in eight microbatches of four:

| Measurement | Full activation graph | Gradient cache |
|---|---:|---:|
| Contrastive loss | 5.952488899 | 5.952488899 |
| Peak live autograd-saved storage, excluding parameters/buffers | 25,533,000 bytes | 2,543,816 bytes |
| Maximum parameter-gradient difference | — | 9.54e-7 |

That is **10.04× less saved-tensor storage** in this fixture. It is not total RAM,
GPU peak allocation, training throughput, or a comparison against ordinary
gradient accumulation (which optimizes a different objective). Caching adds an
encoder replay pass and still stores the batch similarity matrix. Tests verify
loss/gradient parity with dropout, hard negatives, duplicate masking, and uneven chunks.

The regression suite contains 27 passing tests, including the existing fixed-scan
forward/gradient checks through sequence length 8192. CUDA/Triton and bf16 training
parity still require the GPU gate below.

## A harder quality probe, including the negative result

The old synthetic quality tasks could reward token overlap. `research_probe.py`
instead creates two key/value facts and a hard negative that swaps their values.
The positive and negative contain exactly the same tokens. A bag-of-words encoder
must tie, giving 50% pair accuracy. Examples use fresh random contexts; this is
not a split into previously unseen key/value identities.

Train on gaps 1–8 (at most 28 tokens); evaluate fresh examples at three separations.
All runs use a tiny two-layer encoder, 512 evaluation examples per gap, and three
seeds. These models are not parameter matched: PRISM with learned positions has
94,080 parameters, position-free PRISM 44,928, and the Transformer count is recorded
in the raw results. This is a mechanism diagnostic, not a fair efficiency contest.

The initial 400-step InfoNCE run left every model near chance. A follow-up with
1,200 steps and an additional pairwise loss (weight 2) also failed to establish
learning of the bindings:

| Model | Gap 8 / length 28 | Gap 32 / length 100 | Gap 128 / length 388 |
|---|---:|---:|---:|
| PRISM, learned positions | 51.1% ± 1.6% | 52.3% ± 2.2% | 48.1% ± 1.3% |
| PRISM, no position table | 50.8% ± 1.8% | 50.2% ± 0.4% | 45.8% ± 2.1% |
| Transformer, sinusoidal positions | 51.4% ± 1.0% | 50.8% ± 1.9% | 48.3% ± 1.9% |

Values are mean ± sample standard deviation across three seeds, not confidence
intervals. The follow-up changed both budget and loss, so it does not isolate
either change. Full results: [initial](results/revival/binding_probe.json) and
[follow-up](results/revival/binding_probe_pairwise.json).

**Interpretation:** the tested small models/training recipes have not demonstrated
binding retrieval, even at the training length. This cannot establish a PRISM-specific
capacity limit because the Transformer also fails. It does rule out presenting these
runs as evidence of a breakthrough or calling the position-free variant an improvement.
Before a large scale-up, establish a learned positive control on this task and then
test length transfer. The exact symbolic fact lookup is a trivial 100% correctness
oracle for the dataset; it is not a neural baseline.

## Architecture choices that can now be tested properly

The default remains simplified PRISM: all-slow decay, no interference, mean pooling.
Projection and channel gates are grouped into larger matrix operations while retaining
checkpoint parameter names. The existing fixed-decay custom backward now also serves
the CPU fallback, accumulating low-precision scans in fp32. No CUDA speedup is asserted.

Two opt-in PRISM variants are available in the controlled runner: `prism_learned`
actually learns decay rates; `prism_no_pos` removes the learned absolute-position table.
`transformer_sinusoidal` supplies a baseline whose unseen positions are deterministic.
These are hypotheses, not endorsed replacements. Position-free PRISM can execute past
its configured training length, but execution is not evidence of semantic extrapolation.

For a fixed decay, influence at distance k is proportional to lambda^k. At lambda=0.99,
the single-layer half-life is about 69 tokens, and influence after 8192 steps is about
1.75e-36. Layers and pooling complicate the effective receptive field, so this is not
a proof that PRISM cannot use long documents. It does show why linear scaling alone
does not establish long-range binding. In addition, padding short MS MARCO passages to
2048 tokens does not create 2048-token training dependencies. Actual unmasked length
distributions must accompany future claims about training length.

The legacy PyTorch doubling scan performs O(T log T) work. The existing fixed-block
Triton implementation performs O(T) work in T, but its GPU occupancy and wall-clock
behavior still need measurement. Optimized attention also changes the memory comparison;
the old 36× figure and old Transformer OOM cutoffs must not be carried forward.

## Reproduce the completed local work

```bash
uv run --with pytest python -m pytest -q
uv run python benchmark_revival.py --baseline-ref 3efe85c
uv run python research_probe.py --steps 400 --seeds 0,1,2
uv run python research_probe.py --steps 1200 --seeds 0,1,2 --pair-weight 2 \
  --out results/revival/binding_probe_pairwise.json
```

The JSON artifacts include runtime versions and full per-seed outcomes. Timing fields
are incidental CPU observations, not controlled speed comparisons. Existing checkpoint
shapes remain compatible for unchanged variants, but the corrected padding behavior
changes outputs. Re-evaluate old checkpoints and retrain before comparing quality.

## The next decisive GPU experiment

1. **Numerical/efficiency gate.** Run `test_fused_scan.py` on CUDA; compare fixed-scan
   gradients and a short bf16 training run against the reference. Then run Exp 2 with
   `--models prism,transformer --batch-sizes 1,16` at the same dtype/device settings.
   Record actual attention backend, training latency, and allocated memory. Failures
   other than OOM must remain errors. Do not promote the CPU memory ratio to a GPU claim.
2. **Source-only pilot.** Use Exp 1b (512 tokens), 2,000 steps, logical batch 128,
   one BM25 hard negative, and learning rates 1e-4, 3e-4, 1e-3 independently for
   PRISM and the Transformer. Add the position-free PRISM / sinusoidal Transformer
   pair as a length-transfer experiment. Exp 1b does not evaluate LoCoV1. Choose
   learning rates using only source validation, and record all attempts.
3. **Freeze and replicate.** Lock the recipe before opening transfer scores. Run
   three seeds of the selected configurations. Exp 1c selects checkpoints using
   source validation and then evaluates LoCoV1 once. Compare matched optimizer-step
   budgets and also report time-to-quality; do not tune one model on target scores.
4. **Transfer and decisive claim.** Evaluate the selected checkpoints on LongEmbed
   and BEIR using Exp 4/5, with strict checkpoint reconstruction. For learned absolute
   positions, stay inside checkpoint capacity and disclose which rows were trained.
   Use no-position/sinusoidal variants for genuine extrapolation. Report source-dev
   MRR as a small-corpus proxy, not official full-corpus MS MARCO performance.

Example pilot (run each model with its own LR sweep):

```bash
uv run python paper_exp1_controlled.py --sub-exp 1b \
  --models prism,transformer --n-steps 2000 --micro-batch 4 --grad-accum 32 \
  --batch-mode cached --hard-negatives 1 --lr 3e-4 --eval-every 500 \
  --eval-early-stop-patience 0 --device cuda --seed 0
```

Accept a quality claim only after the frozen, replicated held-out comparison supports
it. Accept a memory/throughput claim only at equal quality against the corrected
attention baseline. If neither survives, publish the corrected negative result rather
than scale to hundreds of millions of parameters. The completed work makes that decision
testable; it does not predetermine the answer.
