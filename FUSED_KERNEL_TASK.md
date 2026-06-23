# Task: Fused kernel for PRISM's stratified linear recurrence

## TL;DR
PRISM's core sequence mixer is a **multi-channel linear recurrence with fixed
scalar decay**: `h_t = λ_c · h_{t-1} + x_t`. It is currently implemented as a
Hillis–Steele "doubling" prefix scan in pure PyTorch (`O(log T)` rounds of
pad+scale+add over the full tensor). This is correct but **kernel-launch-bound**:
each layer issues many small CUDA kernels, so at short/medium sequence lengths
the model is dominated by launch overhead, not compute. On our benchmark box a
parameter-matched Transformer trains **~5× faster per step** than PRISM at 128
tokens for this reason.

**Goal:** replace the scan with a single **fused kernel** (forward + backward)
that computes the same recurrence with far fewer kernel launches and lower
memory traffic, making PRISM's wall-clock throughput competitive — especially in
the launch-bound short-sequence regime — while exactly preserving numerics and
training behavior.

This is the same class of kernel that `mamba_ssm` (selective scan) and
flash-linear-attention provide; PRISM's case is **simpler** because the decay is
a per-channel **constant scalar** (data-independent, not learned).

---

## Background: what the recurrence computes

PRISM (`prism.py`) is a bidirectional, multi-channel state-space text encoder.
Per layer, the relevant pipeline is:

1. **Stratified projection** (`StratifiedProjection`, `prism.py:129`): project the
   `(B, T, d)` input into `C` channels, each `(B, T, d_c)` with `d_c = d / C`.
2. **Stratified recurrence** (`StratifiedRecurrence`, `prism.py:164`): per channel
   `c` and direction:
   - input gate: `g_t = sigmoid(W_c z_t + b_c)`, then `x_t = g_t ⊙ z_t`
     (time-varying gate applied to the **input only**, which keeps the
     recurrence linear).
   - linear scan: `h_t = λ_c · h_{t-1} + x_t`, with `h_{-1} = 0`.
   - `λ_c` is a **fixed scalar buffer** (not learned), geometrically spaced
     `λ_c = 1 − 2^{−cΔ}`. **In the paper config (`PRISM-Simplified`) all channels
     use the same `λ = 0.99` ("all-slow").**
   - **Bidirectional:** the same recurrence is also run on the time-reversed
     sequence and flipped back (`prism.py:248`).

Reference (ground-truth) implementation, `prism.py:110`:

```python
def _sequential_scan(gates, values):      # gates here = scalar λ broadcast
    h = zeros(B, D)
    out = []
    for t in range(T):
        h = gates[:, t] * h + values[:, t]
        out.append(h)
    return stack(out, dim=1)               # (B, T, D)
```

For PRISM the gate is the constant scalar `λ_c`, so the recurrence reduces to:
`h_t = Σ_{k≤t} λ_c^{(t−k)} · x_k` (a causal decay-weighted prefix sum).

Model config (`prism_small`, `prism.py:958`): `d=384`, `C=6` channels,
`d_c=64`, `n_layers=6`, `max_len` up to `8192`, bidirectional. Training batch
`micro_batch=16`; `T ∈ {128, 512, 2048, 8192, 16384}`. Compute dtype is **bf16**
(AMP) with fp32 master weights.

---

## The current bottleneck

`_fast_fixed_decay_scan(decay, values)` (`prism.py:56`):

```python
# h_t = decay * h_{t-1} + v_t  via Hillis-Steele doubling, O(log T) rounds
while stride < T:
    h_shifted = F.pad(h[:, :-stride], (0,0, stride,0))
    h = h + power * h_shifted
    power = power * power
    stride *= 2
```

Per scan this issues ~`3·log2(T)` kernels (pad / mul / add) and **materializes
the full `(C·B, T, d_c)` tensor every round**. It is called twice per layer
(fwd + bwd) × `n_layers`, plus the `C` per-channel gate and projection matmuls
are also separate small ops. Net effect: dozens–hundreds of small kernels per
forward pass → launch-bound, and high activation-memory traffic from the
repeated full-tensor materialization.

When all `λ_c` are equal (the paper config), `_run_direction` (`prism.py:204`)
already batches all channels into **one** scan over `(C·B, T, d_c)` (fast path,
`prism.py:227`); when decays differ it falls back to `C` separate scans.

---

## What to build

A **fused, autograd-enabled scan** with this semantics:

```
fused_decay_scan(x: (B, T, C, d_c) bf16/fp16/fp32,
                 decays: (C,) fp32,            # constant per-channel scalar in [0, 1)
                 reverse: bool = False)
    -> h: (B, T, C, d_c)        # h_t = decay_c * h_{t-1} + x_t  (per channel)
```

Requirements:

1. **Single (or very few) kernel launches** per call, for all `C` channels and
   the whole sequence — not `O(log T)` launches. Use a **chunked linear scan**:
   split `T` into chunks of length `L`; compute intra-chunk contributions
   (decay-masked, kept in shared memory/registers) and carry the chunk-final
   state across chunks with `decay^L`. This is the standard chunkwise form used
   by GLA / RetNet / lightning-attention kernels; for a **scalar constant decay**
   it is especially clean (the decay mask is `λ^{i−j}`, no per-step gates).
2. **Fused backward.** The gradient of a linear recurrence is itself a
   reverse-time linear recurrence; implement it in the kernel (or a second
   kernel). Because `decays` are **fixed buffers, you do NOT need a gradient
   w.r.t. `λ`** — only w.r.t. `x`. This simplifies the backward substantially.
3. **Bidirectional** via the `reverse` flag (or run twice on flipped input);
   reproduce the existing flip-run-flip behavior (`prism.py:248`).
4. **Numerical accumulation in fp32** even when inputs are bf16 (decay can be
   close to 1 over long `T`; prefix sums must accumulate in fp32).
5. **Handle `λ = 0`** (memoryless channel → output equals input) and arbitrary
   `T` (not a power of two; chunk remainder).
6. **Drop-in integration**: expose a function that `_run_direction` can call in
   place of `_fast_fixed_decay_scan`, behind a `use_fused_scan` flag on
   `StratifiedRecurrence`, with automatic fallback to the existing PyTorch path
   on CPU / when the kernel is unavailable.

**Optional stretch (secondary win):** fuse the per-channel input gating
(`g_t ⊙ z_t`) and/or batch the `C` gate/projection `nn.Linear`s into grouped
matmuls — these are also many-small-kernel sources. Keep this separate from the
core scan kernel; land the scan first.

---

## Suggested approach

- **Primary: Triton.** Write a chunked-scan Triton kernel (forward + backward).
  Triton is far less effort than raw CUDA, supports bf16 with fp32 accumulators,
  autotunes block sizes, and integrates cleanly via a `torch.autograd.Function`.
  Batch the grid over `(B, C)` (and the `d_c=64` feature dim within a block).
- **Check first — reuse, don't reinvent:** the
  [`flash-linear-attention` (`fla`)](https://github.com/fla-org/flash-linear-attention)
  library has chunked kernels for data-independent decay (e.g. simple-GLA /
  RetNet-style). PRISM's recurrence is the **scalar-decay, no key/value-outer-
  product** special case of these. If an `fla` primitive (e.g. a chunked
  cumulative-decay scan) matches the semantics, wrapping it may get a
  production-grade kernel for near-zero custom code. Evaluate this before writing
  from scratch.
- **Last resort: raw CUDA + pybind** (mamba_ssm style) for maximum performance —
  only if Triton/`fla` are insufficient.

---

## Correctness criteria (must pass)

- **Forward parity** with `_sequential_scan` (the reference, `prism.py:110`):
  fp32 max-abs error `< 1e-5`; bf16 relative error `< 1e-2`. Test over
  `T ∈ {1, 2, 7, 128, 512, 2048, 8192}` (include non-power-of-two), `C=6`,
  `d_c=64`, random `B`, and decays `∈ {0.0, 0.5, 0.9, 0.99}` plus the geometric
  set, both directions.
- **Gradient correctness:** `torch.autograd.gradcheck` in fp64 on small sizes;
  and `dx` parity vs autograd through `_sequential_scan` on realistic sizes.
- **Training parity:** a short `paper_exp1_controlled.py --sub-exp 1a
  --models prism --n-steps 1000` with the fused kernel must produce a loss curve
  matching the non-fused baseline within run-to-run noise (same seed).

## Performance criteria

- Benchmark fwd and fwd+bwd latency, throughput (seq/s), and peak memory vs the
  current `_fast_fixed_decay_scan`, at `T ∈ {128, 512, 2048, 8192}`, `B=16`,
  `C=6`, `d_c=64`, on a real (non-virtualized) CUDA GPU. Report kernel counts
  (e.g. via `torch.profiler` or `nsys`).
- **Targets:** large reduction in kernel-launch count (ideally O(1) per scan
  vs O(log T)); materially higher training throughput, with the biggest relative
  win at short `T` (the launch-bound regime); peak memory no worse than today.
- Then re-run `paper_exp2_efficiency.py` to confirm the end-to-end PRISM
  training-throughput curve improves and the linear-memory scaling is preserved.

---

## Gotchas / constraints

- **Match the reference exactly**, including how padding flows through the scan:
  the current code scans over the full padded `T` and masks later in pooling — do
  not introduce masking inside the scan unless you verify it matches existing
  behavior bit-for-bit on quality.
- Decays are **constant in time and not learned** → no `dλ`; do not add a
  spurious gradient path for them.
- Keep a **PyTorch fallback** (`_fast_fixed_decay_scan` / `_sequential_scan`) for
  CPU and for environments without the kernel; select via flag/device check.
- bf16 in, **fp32 state accumulation**, output cast back to input dtype.
- The bidirectional path currently does `flip → scan → flip`; replicate this
  (or implement a native reverse) and keep both directions numerically identical
  to the reference.

## Deliverables

1. Kernel module (e.g. `prism_scan_kernel.py`) with a `torch.autograd.Function`
   exposing `fused_decay_scan(x, decays, reverse=False)`.
2. Integration: `use_fused_scan` flag in `StratifiedRecurrence` (`prism.py:164`)
   wired through `_run_direction` (`prism.py:204`), with CPU/unavailable fallback.
3. Unit tests: forward parity + `gradcheck` + training-parity smoke test.
4. Benchmark script + a short results table (latency/throughput/memory/kernel
   count, fused vs current) at the sizes above.
5. Notes on numerical tolerances and any `fla`-vs-custom decision.

## Key files

- `prism.py:56` — `_fast_fixed_decay_scan` (current scan to replace)
- `prism.py:110` — `_sequential_scan` (correctness reference)
- `prism.py:164` — `StratifiedRecurrence` (integration point)
- `prism.py:204` — `_run_direction` (per-channel/batched scan dispatch)
- `prism.py:248` — bidirectional `forward`
- `prism.py:958` — `prism_small` config (d=384, C=6, d_c=64, n_layers=6)
- `paper_exp2_efficiency.py` — efficiency benchmark to re-run after integration
- `mamba_bidir.py` — for reference on how the Mamba baseline gets its fused kernel
