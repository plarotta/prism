"""
Benchmark the fused decay scan vs the current PyTorch doubling scan.

Run (on a CUDA box):  uv run python bench_fused_scan.py

Reports forward and forward+backward latency, throughput (seq/s), peak memory,
and kernel-launch counts (via torch.profiler) for both implementations, at the
sizes from FUSED_KERNEL_TASK.md. Prints a markdown table for PROGRESS.md.
"""

import argparse

import torch

from prism import _fast_fixed_decay_scan
from prism_scan_kernel import fused_decay_scan, fused_scan_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    _sync()
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync()
        return start.elapsed_time(end) / iters
    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _peak_mem_mb(fn):
    if DEVICE != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    fn()
    _sync()
    return torch.cuda.max_memory_allocated() / 1e6


def _kernel_count(fn):
    """Number of CUDA kernel launches in one call, via torch.profiler."""
    if DEVICE != "cuda":
        return -1
    from torch.profiler import ProfilerActivity, profile

    fn()
    _sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        _sync()
    return sum(
        1 for e in prof.events() if e.device_type.name == "CUDA" and e.cuda_time_total >= 0
    )


# --- workloads ----------------------------------------------------------------

def make_current(B, T, C, D, decay):
    """The current model path: equal decays batched into (C*B, T, D)."""
    x = torch.randn(C * B, T, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    def fwd():
        return _fast_fixed_decay_scan(decay, x)

    def fwd_bwd():
        out = _fast_fixed_decay_scan(decay, x)
        out.sum().backward()
        x.grad = None

    return fwd, fwd_bwd


def make_fused(B, T, C, D, decay):
    x = torch.randn(B, T, C, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    decays = torch.full((C,), decay, dtype=torch.float32, device=DEVICE)

    def fwd():
        return fused_decay_scan(x, decays)

    def fwd_bwd():
        out = fused_decay_scan(x, decays)
        out.sum().backward()
        x.grad = None

    return fwd, fwd_bwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lengths", default="128,512,2048,8192")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--d-c", type=int, default=64)
    ap.add_argument("--decay", type=float, default=0.99)
    args = ap.parse_args()

    Ts = [int(t) for t in args.seq_lengths.split(",")]
    B, C, D = args.batch, args.channels, args.d_c

    print(f"device = {DEVICE}, triton available = {fused_scan_available()}")
    print(f"B={B} C={C} d_c={D} decay={args.decay} dtype=bf16\n")

    rows = []
    for T in Ts:
        cur_f, cur_fb = make_current(B, T, C, D, args.decay)
        fus_f, fus_fb = make_fused(B, T, C, D, args.decay)

        cur_fwd = _time_ms(cur_f)
        fus_fwd = _time_ms(fus_f)
        cur_fb_ms = _time_ms(cur_fb)
        fus_fb_ms = _time_ms(fus_fb)

        cur_k = _kernel_count(cur_f)
        fus_k = _kernel_count(fus_f)
        cur_mem = _peak_mem_mb(cur_fb)
        fus_mem = _peak_mem_mb(fus_fb)

        seqps_cur = B * 1000.0 / cur_fb_ms
        seqps_fus = B * 1000.0 / fus_fb_ms
        rows.append(
            (T, cur_fwd, fus_fwd, cur_fb_ms, fus_fb_ms, seqps_cur, seqps_fus,
             cur_k, fus_k, cur_mem, fus_mem)
        )

    # Markdown table
    print("| T | fwd cur (ms) | fwd fused (ms) | f+b cur (ms) | f+b fused (ms) | "
          "seq/s cur | seq/s fused | kern cur | kern fused | mem cur (MB) | mem fused (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {:.0f} | "
              "{} | {} | {:.1f} | {:.1f} |".format(*r))


if __name__ == "__main__":
    main()
