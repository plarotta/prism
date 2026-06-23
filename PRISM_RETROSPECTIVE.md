# PRISM: What We Learned Building a Sub-Quadratic Embedding Encoder

*A negative-results writeup. We built a purpose-designed bidirectional
state-space encoder for text embeddings, found that its novel components didn't
work, found that the stripped-down remainder showed real promise, and then
decided to stop. Here's the honest arc and what we'd tell anyone trying the
same thing.*

---

## 1. The bet

Transformer encoders dominate text embeddings — retrieval, similarity,
classification — but their O(n²) attention makes long documents expensive.
Sub-quadratic sequence models (Mamba, RetNet, S4) mostly target autoregressive
generation; bidirectional *embedding* encoders built on them were
underexplored. So the bet was:

> A purpose-built bidirectional state-space encoder could match Transformers on
> short text and beat them on long documents, while scaling linearly in time and
> memory.

We designed **PRISM** (Projected Recurrent Information Stratification with
Mixing): ~20M params, d=384, 6 parallel "channels" each running a gated linear
recurrence at a different fixed decay rate (local → global), with three ideas we
thought were the contribution:

1. **Cross-scale bilinear interference** — multiplicative interaction between
   channels, meant to create conjunction features (local word identity × global
   context).
2. **Multi-scale decay stratification** — geometrically-spaced decay rates so
   different channels capture different temporal scales.
3. **Attentive covariance pooling** — pooling that adds second-order (feature
   co-occurrence) statistics on top of a weighted mean.

The scaling thesis held up immediately and never wavered: linear memory, a
latency crossover with the Transformer around 2K tokens, ~36× less memory at 8K,
and the Transformer OOMing at 16K where PRISM kept running. That part was real.

## 2. The pivot: the novel parts didn't work

Then we ablated the three "contributions," and they evaporated:

| Component | Result |
|---|---|
| Cross-scale interference | **Inert** — removing it changed MRR by −0.009 (noise) |
| Attentive covariance pooling | **Actively harmful** — plain mean pooling scored **+3.4 MRR** |
| Learned decay rates | **Identical** to fixed rates (−0.000) |

The three ideas that justified the architecture contributed nothing, or less
than nothing. We spent real effort trying to *rescue* interference and
covariance pooling — diagnosing them as implementation bugs (weak gradient
signal from triple-zero init; a covariance sketch that drowned the semantic
stream 3:1 on dimensionality). We wrote "V2" versions with LayerNorm, variance
scaling, and learned gates.

It didn't help. At longer sequences the original components weren't just inert —
they were **catastrophic** (the full original trailed the simplified model by
24 points), and the targeted fixes mostly made things *worse*, not better. The
honest conclusion shifted from "we have bugs" to **"these mechanisms don't
provide useful inductive bias at this scale."** Second-order pooling statistics
are noisy; cross-channel bilinear features need more capacity and signal than a
10M-param model gets from a contrastive objective.

**Lesson:** an architecture's theoretical selling points and its empirical load
-bearing parts are different things, and only ablation tells you which is which.
We'd have saved months by ablating aggressively in week one instead of building
the full design first.

## 3. What survived (and the long-sequence reversal)

Stripping PRISM down to its skeleton — a bidirectional multi-channel linear
recurrence with **fixed** decay and **mean** pooling — left something that
actually looked good.

At short sequences the Transformer won, as expected (attention is cheap and
fully-connected there): **0.950 vs 0.991 MRR**, a 4-point deficit. But at
256–512 tokens the ranking *reversed*:

| Model | MRR (seq 256/512) |
|---|---|
| PRISM-simplified | **0.985** |
| Transformer | 0.890 |

A 4-point deficit became a ~9.5-point lead. PRISM converged faster and to a
lower loss. The story writes itself: at the lengths where attention starts to
strain, a linear-recurrence encoder learns better representations *and* uses far
less memory.

**This is the part to be most careful about, because we were almost fooled by
our own numbers.** An early long-document benchmark (LoCoV1) showed PRISM
beating the Transformer by ~49 nDCG@10 points and even beating a published
80M-param baseline. Both were artifacts:

- The Transformer baseline was crippled by a bad learning rate (0.194). Tuned,
  it jumped to 0.672. PRISM still won — but by **~10 points, not 49**.
- We had trained *and* evaluated on the same benchmark, which is not the
  protocol the published baselines used. The honest, contamination-free
  comparison (train on MS MARCO, evaluate zero-shot) was the experiment we never
  finished.

So the real state of "what survived" is: **a genuinely promising signal that a
simplified bidirectional SSM encoder beats a param-matched Transformer at long
sequences — never rigorously nailed down under a clean protocol.**

## 4. The honest limits

Three things, in increasing order of how much they matter:

**Speed isn't the win you'd hope.** PRISM's recurrence, implemented as a
PyTorch parallel scan, is *kernel-launch-bound*: it fires many small CUDA kernels
per layer. At 128 tokens on an A100, a param-matched Transformer trained **~5×
faster per step** — because the Transformer is a few big matmuls that saturate
the GPU, while PRISM is lots of tiny ops dominated by launch overhead. PRISM's
real efficiency wins are **memory** and **asymptotic** long-sequence scaling, not
wall-clock at the lengths most people actually use. Closing that gap means
writing a fused scan kernel (à la Mamba) — a whole project of its own.

**Even the multi-scale premise was suboptimal.** A later ablation found that
making *all* channels use the same slow decay beat the geometrically-spaced
"stratification" by ~7 nDCG@10 points. The *S* in PRISM — the idea that
different channels should capture different temporal scales — didn't earn its
place either.

**The clean comparison is expensive.** A credible controlled study is 4
architectures × 5 sequence-length regimes × strong training (hard negatives,
pretraining) × ablations, each run many GPU-hours — and disproportionately slow
for exactly the launch-bound models the paper is about.

## 5. When to stop: a crowded landscape

While we were doing this, the niche filled in. Bidirectional SSM and efficient-
encoder work for embeddings arrived from several directions — Hydra and other
bidirectional SSMs, Mamba-2-based sentence embedders, and **ModernBERT**, an
efficient long-context *Transformer* encoder that directly contests the premise
that attention can't do long documents well.

Put it together:

- Our distinctive ideas (interference, stratification, covariance pooling)
  empirically **failed**.
- What remained — "a simple bidirectional linear-recurrence encoder is good at
  long-document embeddings" — is **increasingly non-novel**.
- Making it *competitive* on the axes reviewers care about (absolute quality
  with hard negatives + pretraining; wall-clock speed via a fused kernel) is a
  large additional **compute and engineering** investment.

That's three independent reasons pointing the same way. Continuing would be
sunk-cost reasoning. So we're stopping — and writing this instead.

## 6. Takeaways for anyone trying this

1. **Ablate before you build.** Your architecture's headline idea is a
   hypothesis, not a feature. Test it in isolation, early, on the smallest setup
   that can falsify it. We inverted this and paid for it.
2. **Distrust your own blowout numbers.** A +49-point result was a bad baseline
   LR plus a contaminated split. If a margin looks too good, the most likely
   explanation is a broken baseline or a leak — not a breakthrough. Fix the
   baseline before you celebrate.
3. **"Sub-quadratic" is about FLOPs and memory, not wall-clock.** A naive scan
   is launch-bound and can be *slower* than attention at short/medium lengths.
   If speed is a claim, budget for a fused kernel from the start.
4. **Memory is the durable win for linear-time encoders.** The cleanest,
   least-disputable advantage was running where the Transformer OOMs. If you
   pursue this family, lead with memory and long-context capability, not latency.
5. **Know the field's clock.** A niche can fill in faster than your experiments
   run. Re-survey related work periodically; "first to do X" has a short shelf
   life, and a contribution that rests on it is fragile.
6. **Stopping is a result.** A rigorous negative result — *these specific
   mechanisms don't help, here's where this architecture class actually wins and
   loses* — saves other people the same dead ends. That's worth writing down.

---

*Project status: shelved (June 2026). The code, evaluation harness, and a
fused-kernel task brief remain in the repo for anyone who wants to pick up the
one genuinely open thread — whether a fused scan kernel makes the simplified
encoder fast enough, in wall-clock, to matter.*
