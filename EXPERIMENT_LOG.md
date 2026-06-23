# Experiment Log

## 2026-06-23 — Fused decay-scan kernel (implementation)

Implemented a fused linear-recurrence scan to replace the O(log T) Hillis–Steele
doubling scan (`prism._fast_fixed_decay_scan`), which is kernel-launch-bound at
short/medium T.

- New `prism_scan_kernel.py`: `fused_decay_scan(x:(B,T,C,d_c), decays:(C,), reverse)`
  as a `torch.autograd.Function`.
  - Triton chunked forward kernel: one program per (B,C), d_c in-block, chunk
    length L=64, intra-chunk decay-matrix matmul (`tl.dot`, fp32 `ieee`) + fp32
    cross-chunk carry. λ=0 handled via clamped `log` (memoryless → output==input).
  - Backward = forward scan with `reverse` flipped (dx only; no dλ). `reverse`
    implemented by flipping the time axis around a forward-only kernel.
  - PyTorch fp32 fallback (`_ref_forward`) for CPU / no-Triton; auto-selected.
- Integration: `use_fused_scan=True` flag on `StratifiedRecurrence`, wired into
  `_run_direction` (stack gated channels → (B,T,C,d_c) → one kernel call). Falls
  back to the existing PyTorch path on CPU / when Triton unavailable.

Local validation (this box is macOS, no CUDA → fallback + CPU tests only):
- `test_fused_scan.py`: forward parity (84 configs, fp32 + bf16, both directions),
  fp64 gradcheck, fp32 grad parity vs sequential autograd — ALL PASS.
- `prism_small` fwd/bwd runs clean with the integration in place.

Pending on GPU box (A100): Triton-path parity, `bench_fused_scan.py`
(latency/throughput/mem/kernel-count), training parity (`paper_exp1_controlled.py
--sub-exp 1a --models prism --n-steps 1000`, fused on vs off), and
`paper_exp2_efficiency.py --models prism`. High-level results → PROGRESS.md once run.
