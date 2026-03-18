# PRISM Experiment Plan
## Implementation, Testing, and Validation Beyond Toy Data

---

## Phase 0: Infrastructure and Unit Testing (Weeks 1–2)

### 0.1 Environment setup

Set up a reproducible training environment with the following stack:

- **Hardware**: 8× A100 (80 GB) or 8× H100 node. Single-node training for Base; multi-node for Large. All experiments except Large can be developed on 4× A100.
- **Framework**: PyTorch 2.2+ with `torch.compile` for the interference and MLP kernels. FlashAttention for baseline comparisons.
- **Dependencies**: `mamba_ssm` (for the fused selective scan CUDA kernel — PRISM's recurrence is simpler and can reuse these), `transformers` (for tokeniser and baselines), `sentence-transformers` (for evaluation), `mteb` (for benchmark suite), `wandb` (for experiment tracking).
- **Containerisation**: Docker image with pinned versions. All experiments run from the same image for reproducibility.

### 0.2 Unit tests (must pass before any training)

Each architectural component gets an isolated test:

**Test 1 — Hadamard initialisation orthogonality.** Verify that `W_in^(c)^T W_in^(c')` is approximately the identity for c = c' and approximately zero for c ≠ c'. Tolerance: Frobenius norm of off-diagonal blocks < 0.01.

**Test 2 — Sequential vs. parallel scan equivalence.** Generate random gates and values of shape `(B=8, T=1024, D=64)`. Run both `_sequential_scan` and the parallel scan. Assert max absolute difference < 1e-5. Run for both float32 and bfloat16.

**Test 3 — Fixed decay rate coverage.** For `C = 8, n_max = 8192`, verify that the effective memory horizons `1/(1 - λ_c)` span from ~1 token to ~8192 tokens, with approximately log-uniform spacing. Print the horizons and visually confirm geometric progression.

**Test 4 — Interference ReZero.** Initialise a `CrossScaleInterference` module. Forward-pass random hidden states. Verify that the output equals the input to within 1e-6 (since α is initialised to zero, interference contributes nothing at init).

**Test 5 — Bidirectional fusion symmetry.** At initialisation (β ≈ 0.5), verify that swapping forward and backward inputs produces the same output (within tolerance). After one gradient step, verify this symmetry is broken.

**Test 6 — Gradient flow.** Compute the gradient of the output embedding with respect to the first input token. Verify it is non-zero (information flows from position 1 to the embedding). Repeat for the last token. Repeat for a token in the middle. All should have non-zero gradients.

**Test 7 — Masking correctness.** Create a batch where sequence 1 has length 100 and sequence 2 has length 50 (padded to 100). Verify that the embedding for sequence 2 is identical whether the padding is zeros, ones, or random (i.e., padding is correctly masked).

**Test 8 — Numerical stability.** Forward-pass a sequence of length 4096 through a 12-layer PRISM-Base in bfloat16. Verify no NaN or Inf in the output. Check that the embedding norm is within [0.1, 100].

**Test 9 — Covariance sketch invariance.** For a batch of identical sequences, verify the covariance sketch is identical across batch elements. For a shuffled sequence (same tokens, different order), verify the covariance sketch changes.

**Test 10 — Contrastive loss sanity.** With a batch of 4 query-positive pairs where queries equal their positives, the InfoNCE loss should be approximately `log(4) - 1/τ · 1` and accuracy should be 100%.

### 0.3 Integration smoke test

Run a full forward + backward pass on PRISM-Small with random data `(B=16, T=512)`. Verify:
- No OOM on a single A100 (40 GB)
- Backward pass completes without error
- All parameter gradients are non-zero
- 10 optimiser steps reduce the contrastive loss

---

## Phase 1: Pre-Training (Weeks 3–6)

### 1.1 Dataset

**Primary corpus**: English Wikipedia (2024 dump) + BookCorpus + CC-News subset, ~16 GB text. This matches the scale used by BERT-base and allows direct comparison.

**Tokeniser**: Use the `bert-base-uncased` tokeniser (WordPiece, 30522 vocab). This eliminates tokenisation as a confound when comparing with BERT-based baselines.

**Preprocessing**: Standard sentence segmentation, max sequence length 512 for Phase 1 (extended to 4096 in Phase 3).

### 1.2 Pre-training objective

**Masked Language Modelling (MLM)**: Mask 15% of tokens (80% [MASK], 10% random, 10% unchanged), predict the original token from the per-token representations of the last PRISM layer. This is the standard BERT objective and ensures the model learns rich contextual representations.

Use a linear classification head on top of the per-token outputs: `logits = W_head h_t + b_head`, cross-entropy loss.

### 1.3 Training configuration

| Hyperparameter | PRISM-Small | PRISM-Base |
|---|---|---|
| Batch size (effective) | 256 | 256 |
| Learning rate (peak) | 5e-4 | 3e-4 |
| Warmup steps | 10,000 | 10,000 |
| Total steps | 500,000 | 1,000,000 |
| LR schedule | Cosine decay to 1e-5 | Cosine decay to 1e-5 |
| Optimiser | AdamW (β₁=0.9, β₂=0.98, ε=1e-6) | AdamW |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Precision | bfloat16 mixed | bfloat16 mixed |
| Dropout | 0.1 | 0.1 |

**Checkpointing**: Save every 50,000 steps. Evaluate MLM perplexity on a held-out validation set (5% of corpus) at each checkpoint.

### 1.4 Pre-training baselines

Train the following models on the same data with the same tokeniser and comparable parameter counts:

- **BERT-base** (from scratch, 110M params) — the attention baseline
- **BiMamba** (bidirectional Mamba, ~110M params) — the linear recurrence baseline
- **RetNet-base** (multi-scale retention, ~110M params) — the multi-scale baseline

All baselines use MLM pre-training for fair comparison.

### 1.5 Pre-training success criteria

PRISM-Base should achieve MLM validation perplexity within 5% of BERT-base at the same step count. If it doesn't, the pre-training phase has failed and we need to debug before proceeding.

---

## Phase 2: Embedding Fine-Tuning (Weeks 6–8)

### 2.1 Fine-tuning data

Follow the E5-base-v2 training recipe:

- **Stage 2a — Weak supervision**: 500K+ query-passage pairs mined from web data (e.g., MS MARCO passages, Natural Questions, NQ for retrieval). Train with in-batch negatives only.
- **Stage 2b — Hard negatives**: Same data augmented with BM25-mined hard negatives (7 hard negatives per query). This is critical for retrieval quality.

### 2.2 Fine-tuning configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 128 (with in-batch negatives → 128² pairs) |
| Hard negatives per query | 7 (Stage 2b only) |
| Learning rate | 2e-5 |
| Warmup | 10% of steps |
| Total epochs | 3 (Stage 2a) + 1 (Stage 2b) |
| Temperature τ | 0.05 |
| Loss | InfoNCE with in-batch negatives |
| Max query length | 64 tokens |
| Max passage length | 256 tokens |

### 2.3 Fine-tuning baselines

Fine-tune all pre-trained models from Phase 1 using the identical recipe. Also include:

- **E5-base-v2** (off-the-shelf) — the strong production baseline
- **GTE-base** (off-the-shelf) — another strong production baseline
- **all-MiniLM-L6-v2** (off-the-shelf) — the efficiency baseline

---

## Phase 3: Evaluation (Weeks 8–10)

### 3.1 Primary benchmark: MTEB

Evaluate all models on the Massive Text Embedding Benchmark (MTEB), which covers 7 task categories across 56+ datasets:

| Category | Example datasets | What it tests |
|---|---|---|
| Classification | Banking77, TweetSentiment | Embedding quality for downstream classifiers |
| Clustering | ArXiv, Reddit | Whether similar documents cluster together |
| Pair classification | TwitterURLParaphrase | Pairwise semantic similarity |
| Reranking | AskUbuntuDupQuestions | Ordering passages by relevance |
| Retrieval | MSMARCO, NQ, HotpotQA, FEVER | The primary embedding use case |
| STS | STS-Benchmark, SICK-R | Semantic textual similarity |
| Summarisation | SummEval | Embedding quality for summary evaluation |

**Report**: Mean score per category + overall MTEB score. Compare PRISM vs. all baselines.

### 3.2 Long-context evaluation

MTEB doesn't stress long-context ability. We add:

**BEIR (zero-shot retrieval)**: 18 retrieval datasets with no fine-tuning on the target domain. This tests generalisation. Evaluate at both standard and extended passage lengths (up to 4096 tokens for PRISM-Base with extended positional embeddings).

**LongBench-Retrieval**: A subset of LongBench focusing on retrieval over long documents (4K–16K tokens). This is where PRISM's linear scaling should provide the clearest advantage.

**Custom long-range disambiguation test**: We construct a synthetic benchmark:
- Generate sentences of the form "The [ambiguous word] ... [disambiguating context] ..." where the distance between the ambiguous word and its disambiguating context varies from 10 to 2000 tokens.
- Create query-passage pairs where the query is the ambiguous word and the correct passage contains the appropriate sense.
- Measure retrieval accuracy as a function of disambiguation distance.
- Hypothesis: PRISM degrades less than BiMamba/RetNet as distance increases, thanks to cross-scale interference.

### 3.3 Efficiency benchmarks

Measure wall-clock performance at multiple sequence lengths on a single A100:

| Metric | Sequence lengths |
|---|---|
| Encoding throughput (sequences/sec) | 128, 256, 512, 1024, 2048, 4096, 8192 |
| Peak GPU memory (MB) | Same |
| Latency (ms/sequence, batch=1) | Same |

Report for all models. Plot throughput and memory as a function of sequence length. PRISM should show near-linear scaling vs. attention's quadratic curve.

### 3.4 Ablation studies

Critical ablations to isolate each contribution:

**Ablation A — Remove cross-scale interference.** Run channels independently; concatenate and MLP. This tests whether interference matters versus simple channel mixing.

**Ablation B — Replace fixed decay with learned decay.** Let λ_c be learned parameters (initialised at the geometric spacing). This tests whether fixing the rates helps.

**Ablation C — Replace bilinear interference with additive.** Use φ(h^(c), h^(c')) = U h^(c) + V h^(c') instead of the elementwise product. This tests whether multiplicative features matter.

**Ablation D — Replace covariance pooling with mean pooling.** Use standard mean pooling instead of the two-stream attentive + covariance approach. This tests whether second-order statistics matter.

**Ablation E — Vary channel count.** Test C ∈ {2, 4, 6, 8, 12, 16} with fixed total dimension. This maps the quality-efficiency tradeoff curve.

**Ablation F — Unidirectional vs. bidirectional.** Run PRISM in forward-only mode. This quantifies the bidirectional benefit.

Run all ablations on PRISM-Small (fastest iteration) and evaluate on a subset of MTEB (Classification + Retrieval + STS — the three most informative categories).

---

## Phase 4: Analysis and Diagnostics (Weeks 10–12)

### 4.1 Channel specialisation analysis

After training PRISM-Base, analyse what each channel has learned:

- **Decay horizon probing**: For each channel, measure the average attention weight (from the attentive pooling head) as a function of distance from the attended position. Channels with fast decay should have sharply peaked attention; slow channels should have flat attention. If this pattern doesn't emerge, the fixed-rate stratification isn't working as intended.
- **Probing classifiers**: Train linear probes on each channel's hidden states to predict syntactic features (POS tags, dependency labels), semantic features (NER, word sense), and sentiment. If channels are specialising, different channels should be most predictive for different features.
- **Interference activation analysis**: Track the magnitude of the interference terms |α_{c,c'} · φ(·)| during training. Which channel pairs interact most? Does the pattern match intuition (local × global should dominate)?

### 4.2 Attention approximation analysis

Compare PRISM's effective "attention pattern" (the implicit weighting of past positions in the final representation) with BERT's explicit attention:

- For each input position t, compute the gradient of the embedding with respect to all input positions. This gives an effective attention map.
- Compare this map with BERT's attention patterns on the same inputs.
- Measure the correlation between PRISM's implicit attention and BERT's explicit attention.
- Hypothesis: PRISM's implicit attention should approximate BERT's, especially for positions that are semantically important.

### 4.3 Failure case analysis

Identify systematic failure modes:

- Collect the top-100 queries where PRISM's retrieval rank is worst relative to BERT.
- Categorise by error type: missing long-range dependency, wrong sense disambiguation, negation failure, compositional failure, other.
- For each category, determine whether the failure is architectural (fundamental to the linear recurrence) or trainable (would be fixed with more data or better tuning).

---

## Phase 5: Scaling and Production Readiness (Weeks 12–16)

### 5.1 PRISM-Large training

If Base results are promising, train PRISM-Large (330M params, 24 layers) on a larger corpus (Wikipedia + BookCorpus + CC-Net filtered, ~100 GB). Training budget: ~1000 A100-hours.

### 5.2 Long-context extension

Extend PRISM-Base to 16K context:

- Replace absolute positional embeddings with RoPE (Rotary Positional Embeddings) — these generalise to longer sequences than the training length.
- Fine-tune on a mix of short (512) and long (4096–16384) documents for 50K steps.
- Evaluate on LongBench-Retrieval and the custom disambiguation test.

### 5.3 Distillation

If PRISM-Base matches BERT-base quality:

- Distill PRISM-Base into PRISM-Small using knowledge distillation (MSE on embeddings + KL on retrieval logits).
- Target: PRISM-Small should match all-MiniLM-L6-v2 quality at lower latency for long sequences.

### 5.4 ONNX export and inference optimisation

- Export PRISM-Small and PRISM-Base to ONNX.
- Benchmark inference latency on CPU (Intel Xeon) and GPU (T4, A10, A100).
- Compare with ONNX-exported BERT-base and all-MiniLM-L6-v2.

---

## Success Criteria (Go/No-Go)

The project succeeds if ANY of the following holds:

| Criterion | Threshold | Why it matters |
|---|---|---|
| MTEB overall score ≥ BERT-base | Within 1 point | Proves linear-time models can match attention for embeddings |
| MTEB retrieval ≥ BERT-base AND 3× throughput at n=2048 | Both conditions | The practical sweet spot: same quality, much faster |
| Custom disambiguation test: PRISM > BERT at distance > 512 | Statistically significant (p < 0.01) | Validates the cross-scale interference hypothesis |
| Any PRISM ablation B/C/D shows > 2-point MTEB drop | Reproducible | Validates that each novel component contributes |

The project fails if:

- PRISM-Base scores > 3 points below BERT-base on MTEB overall after full training.
- The interference ablation (Ablation A) shows < 0.5-point difference, meaning the novel component doesn't help.
- Training is unstable (divergence, NaN) and cannot be fixed with standard techniques within 2 weeks.

---

## Timeline Summary

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | Infrastructure + unit tests | All tests passing, baselines training |
| 3–6 | Pre-training | MLM-trained PRISM-Small/Base + baselines |
| 6–8 | Embedding fine-tuning | Contrastive-trained models |
| 8–10 | Evaluation | MTEB, BEIR, efficiency, ablations |
| 10–12 | Analysis | Channel probing, failure analysis, paper draft |
| 12–16 | Scaling (conditional) | Large model, long-context, distillation |

---

## Compute Budget Estimate

| Task | GPU-hours (A100) |
|---|---|
| PRISM-Small pre-training | ~200 |
| PRISM-Base pre-training | ~800 |
| 3 baselines pre-training | ~2400 |
| All fine-tuning | ~200 |
| Ablations (6 variants × Small) | ~600 |
| Evaluation + analysis | ~100 |
| **Total (through Phase 4)** | **~4300** |
| PRISM-Large (Phase 5, conditional) | ~1000 |

At approximately $2/A100-hour (cloud spot pricing), the total budget through Phase 4 is approximately $8,600, and $10,600 including Phase 5.
