"""
PRISM vs Transformer: Computational Scaling Benchmark

Measures wall-clock throughput, peak memory, and latency as a function of
sequence length. Generates publication-quality plots proving PRISM's
sub-quadratic scaling advantage.
"""

import gc
import json
import time
import traceback
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism import prism_small
from baseline_transformer import transformer_small

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
DTYPE = torch.float32
VOCAB_SIZE = 32000
BATCH_SIZE = 8
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
N_WARMUP = 3
N_MEASURE = 10
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def sync_device():
    """Ensure all GPU ops are complete before timing."""
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_peak_memory():
    """Get peak GPU memory allocated (bytes)."""
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated()
    if DEVICE == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def reset_memory():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_forward(model, seq_len: int, batch_size: int = BATCH_SIZE) -> dict:
    """Measure forward pass throughput and latency."""
    model.eval()
    input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)
    mask = torch.ones(batch_size, seq_len, device=DEVICE, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(input_ids, mask)
            sync_device()

    # Measure
    reset_memory()
    mem_before = measure_peak_memory()

    times = []
    with torch.no_grad():
        for _ in range(N_MEASURE):
            sync_device()
            t0 = time.perf_counter()
            _ = model(input_ids, mask)
            sync_device()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    mem_after = measure_peak_memory()

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / avg_time  # sequences/sec

    return {
        "seq_len": seq_len,
        "avg_latency_ms": avg_time * 1000,
        "std_latency_ms": std_time * 1000,
        "throughput_seq_per_sec": throughput,
        "memory_MB": (mem_after - mem_before) / 1e6 if mem_after > mem_before else 0,
    }


def benchmark_forward_backward(model, seq_len: int, batch_size: int = BATCH_SIZE) -> dict:
    """Measure forward + backward pass (training throughput)."""
    model.train()
    input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)
    mask = torch.ones(batch_size, seq_len, device=DEVICE, dtype=torch.long)

    # Warmup
    for _ in range(N_WARMUP):
        out = model(input_ids, mask)
        loss = out["embedding"].sum()
        loss.backward()
        model.zero_grad()
        sync_device()

    # Measure
    reset_memory()
    mem_before = measure_peak_memory()

    times = []
    for _ in range(N_MEASURE):
        sync_device()
        t0 = time.perf_counter()
        out = model(input_ids, mask)
        loss = out["embedding"].sum()
        loss.backward()
        sync_device()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        model.zero_grad()

    mem_after = measure_peak_memory()

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / avg_time

    return {
        "seq_len": seq_len,
        "avg_latency_ms": avg_time * 1000,
        "std_latency_ms": std_time * 1000,
        "throughput_seq_per_sec": throughput,
        "memory_MB": (mem_after - mem_before) / 1e6 if mem_after > mem_before else 0,
    }


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks():
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence lengths: {SEQ_LENGTHS}")
    print(f"Warmup iterations: {N_WARMUP}, Measure iterations: {N_MEASURE}")
    print()

    # Build models
    prism_model = prism_small(vocab_size=VOCAB_SIZE).to(DEVICE)
    transformer_model = transformer_small(vocab_size=VOCAB_SIZE).to(DEVICE)

    print(f"PRISM params:       {count_params(prism_model):,}")
    print(f"Transformer params: {count_params(transformer_model):,}")
    print()

    results = {
        "config": {
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "seq_lengths": SEQ_LENGTHS,
            "prism_params": count_params(prism_model),
            "transformer_params": count_params(transformer_model),
        },
        "forward": {"prism": [], "transformer": []},
        "forward_backward": {"prism": [], "transformer": []},
    }

    # Forward-only benchmarks
    print("=" * 60)
    print("FORWARD PASS (inference)")
    print("=" * 60)
    for seq_len in SEQ_LENGTHS:
        print(f"\n--- seq_len = {seq_len} ---")
        reset_memory()

        try:
            r = benchmark_forward(prism_model, seq_len)
            results["forward"]["prism"].append(r)
            print(f"  PRISM:       {r['avg_latency_ms']:8.1f} ms  ({r['throughput_seq_per_sec']:7.1f} seq/s)")
        except Exception as e:
            print(f"  PRISM:       FAILED — {e}")
            traceback.print_exc()
            results["forward"]["prism"].append({"seq_len": seq_len, "failed": True})

        reset_memory()

        try:
            r = benchmark_forward(transformer_model, seq_len)
            results["forward"]["transformer"].append(r)
            print(f"  Transformer: {r['avg_latency_ms']:8.1f} ms  ({r['throughput_seq_per_sec']:7.1f} seq/s)")
        except Exception as e:
            print(f"  Transformer: FAILED — {e}")
            traceback.print_exc()
            results["forward"]["transformer"].append({"seq_len": seq_len, "failed": True})

    # Forward+backward benchmarks
    print("\n" + "=" * 60)
    print("FORWARD + BACKWARD (training)")
    print("=" * 60)
    for seq_len in SEQ_LENGTHS:
        print(f"\n--- seq_len = {seq_len} ---")
        reset_memory()

        try:
            r = benchmark_forward_backward(prism_model, seq_len)
            results["forward_backward"]["prism"].append(r)
            print(f"  PRISM:       {r['avg_latency_ms']:8.1f} ms  ({r['throughput_seq_per_sec']:7.1f} seq/s)")
        except Exception as e:
            print(f"  PRISM:       FAILED — {e}")
            results["forward_backward"]["prism"].append({"seq_len": seq_len, "failed": True})

        reset_memory()

        try:
            r = benchmark_forward_backward(transformer_model, seq_len)
            results["forward_backward"]["transformer"].append(r)
            print(f"  Transformer: {r['avg_latency_ms']:8.1f} ms  ({r['throughput_seq_per_sec']:7.1f} seq/s)")
        except Exception as e:
            print(f"  Transformer: FAILED — {e}")
            results["forward_backward"]["transformer"].append({"seq_len": seq_len, "failed": True})

    # Save raw results
    with open(RESULTS_DIR / "scaling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR / 'scaling_results.json'}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling_results(results: dict):
    """Generate publication-quality scaling plots."""

    # Color scheme
    PRISM_COLOR = "#2563EB"     # blue
    TRANS_COLOR = "#DC2626"     # red
    BG_COLOR = "#FAFAFA"
    GRID_COLOR = "#E5E7EB"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PRISM vs Transformer: Computational Scaling", fontsize=16, fontweight="bold", y=1.02)

    for row, mode in enumerate(["forward", "forward_backward"]):
        mode_label = "Inference" if mode == "forward" else "Training (fwd+bwd)"

        prism_data = [r for r in results[mode]["prism"] if not r.get("failed")]
        trans_data = [r for r in results[mode]["transformer"] if not r.get("failed")]

        if not prism_data or not trans_data:
            continue

        prism_lens = [r["seq_len"] for r in prism_data]
        trans_lens = [r["seq_len"] for r in trans_data]
        prism_latency = [r["avg_latency_ms"] for r in prism_data]
        trans_latency = [r["avg_latency_ms"] for r in trans_data]
        prism_throughput = [r["throughput_seq_per_sec"] for r in prism_data]
        trans_throughput = [r["throughput_seq_per_sec"] for r in trans_data]

        # --- Latency plot (log-log) ---
        ax = axes[row, 0]
        ax.set_facecolor(BG_COLOR)
        ax.loglog(prism_lens, prism_latency, "o-", color=PRISM_COLOR, linewidth=2.5,
                  markersize=7, label="PRISM (linear)", zorder=5)
        ax.loglog(trans_lens, trans_latency, "s-", color=TRANS_COLOR, linewidth=2.5,
                  markersize=7, label="Transformer (quadratic)", zorder=5)

        # Add reference slopes
        ref_lens = np.array([prism_lens[0], prism_lens[-1]])
        linear_ref = prism_latency[0] * (ref_lens / ref_lens[0])
        quadratic_ref = trans_latency[0] * (ref_lens / ref_lens[0]) ** 2
        ax.loglog(ref_lens, linear_ref, "--", color=PRISM_COLOR, alpha=0.3, linewidth=1.5, label="O(n) reference")
        ax.loglog(ref_lens, quadratic_ref, "--", color=TRANS_COLOR, alpha=0.3, linewidth=1.5, label="O(n²) reference")

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{mode_label} — Latency vs Sequence Length")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.set_xticks(prism_lens)
        ax.set_xticklabels([str(s) for s in prism_lens])

        # --- Throughput plot ---
        ax = axes[row, 1]
        ax.set_facecolor(BG_COLOR)
        ax.semilogx(prism_lens, prism_throughput, "o-", color=PRISM_COLOR, linewidth=2.5,
                     markersize=7, label="PRISM", zorder=5)
        ax.semilogx(trans_lens, trans_throughput, "s-", color=TRANS_COLOR, linewidth=2.5,
                     markersize=7, label="Transformer", zorder=5)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Throughput (sequences/sec)")
        ax.set_title(f"{mode_label} — Throughput vs Sequence Length")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.set_xticks(prism_lens)
        ax.set_xticklabels([str(s) for s in prism_lens])

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "scaling_comparison.png", bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'scaling_comparison.png'}")

    # --- Speedup plot ---
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_facecolor(BG_COLOR)

    for mode, style, label in [("forward", "o-", "Inference"), ("forward_backward", "s-", "Training")]:
        prism_data = {r["seq_len"]: r for r in results[mode]["prism"] if not r.get("failed")}
        trans_data = {r["seq_len"]: r for r in results[mode]["transformer"] if not r.get("failed")}
        common_lens = sorted(set(prism_data.keys()) & set(trans_data.keys()))
        speedups = [trans_data[s]["avg_latency_ms"] / prism_data[s]["avg_latency_ms"] for s in common_lens]
        ax.semilogx(common_lens, speedups, style, linewidth=2.5, markersize=7, label=label)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Speedup (Transformer time / PRISM time)", fontsize=12)
    ax.set_title("PRISM Speedup over Transformer", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    if common_lens:
        ax.set_xticks(common_lens)
        ax.set_xticklabels([str(s) for s in common_lens])

    fig2.savefig(RESULTS_DIR / "speedup.png", bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'speedup.png'}")

    plt.close("all")

    # --- Memory plot (only meaningful on CUDA) ---
    has_mem = False
    for mode in ["forward", "forward_backward"]:
        for arch in ["prism", "transformer"]:
            for r in results[mode][arch]:
                if not r.get("failed") and r.get("memory_MB", 0) > 0:
                    has_mem = True

    if has_mem:
        fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_facecolor(BG_COLOR)
        for arch, color, marker, label in [
            ("prism", PRISM_COLOR, "o", "PRISM"),
            ("transformer", TRANS_COLOR, "s", "Transformer"),
        ]:
            data = [r for r in results["forward_backward"][arch] if not r.get("failed") and r.get("memory_MB", 0) > 0]
            if data:
                lens = [r["seq_len"] for r in data]
                mem = [r["memory_MB"] for r in data]
                ax.loglog(lens, mem, f"{marker}-", color=color, linewidth=2.5, markersize=7, label=label)

        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
        ax.set_title("Peak Memory: PRISM O(n·d) vs Transformer O(n²+n·d)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        fig3.savefig(RESULTS_DIR / "memory_comparison.png", bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR / 'memory_comparison.png'}")
        plt.close("all")


# ---------------------------------------------------------------------------
# Theoretical FLOP analysis
# ---------------------------------------------------------------------------

def plot_theoretical_flops():
    """Plot theoretical FLOP counts for PRISM vs Transformer."""
    d = 384
    C = 6
    d_c = d // C
    L = 6

    seq_lengths = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    # Transformer: dominant cost is self-attention O(n² · d) + MLP O(n · d²)
    # Per layer: 4·n·d² (QKV + output proj) + 2·n²·d (attention matmul) + 2·n·d·4d (MLP)
    trans_flops = L * (4 * seq_lengths * d**2 + 2 * seq_lengths**2 * d + 8 * seq_lengths * d**2)

    # PRISM: dominant cost is recurrence O(n·d) + interference O(n·C²·d_c²) + MLP O(n·d²)
    # Per layer: n·d (recurrence) + n·C²·d_c² (interference) + 2*n*d*mlp_dim (MLP)
    mlp_dim = int(d * 2.0)
    prism_flops = L * (
        seq_lengths * d +                          # recurrence
        seq_lengths * C**2 * d_c**2 +               # interference
        2 * seq_lengths * d +                        # input projections
        2 * seq_lengths * d * mlp_dim                # MLP
    )
    # Bidirectional doubles recurrence + interference cost
    prism_flops = L * (
        2 * seq_lengths * d +                        # recurrence (both dirs)
        2 * seq_lengths * C**2 * d_c**2 +            # interference (both dirs)
        2 * seq_lengths * d +                        # input projections
        2 * seq_lengths * d * mlp_dim                # MLP
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_facecolor("#FAFAFA")
    ax.loglog(seq_lengths, prism_flops, "o-", color="#2563EB", linewidth=2.5, markersize=6, label="PRISM (theoretical)")
    ax.loglog(seq_lengths, trans_flops, "s-", color="#DC2626", linewidth=2.5, markersize=6, label="Transformer (theoretical)")

    # Reference slopes
    ax.loglog(seq_lengths, prism_flops[0] * (seq_lengths / seq_lengths[0]),
              "--", color="#2563EB", alpha=0.3, label="O(n)")
    ax.loglog(seq_lengths, trans_flops[0] * (seq_lengths / seq_lengths[0])**2,
              "--", color="#DC2626", alpha=0.3, label="O(n²)")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("FLOPs (per batch element)", fontsize=12)
    ax.set_title("Theoretical FLOPs: PRISM O(n·d²) vs Transformer O(n²·d + n·d²)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.savefig(RESULTS_DIR / "theoretical_flops.png", bbox_inches="tight")
    print(f"Saved: {RESULTS_DIR / 'theoretical_flops.png'}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_theoretical_flops()
    results = run_all_benchmarks()
    plot_scaling_results(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Speedup (Transformer / PRISM)")
    print("=" * 80)
    print(f"{'Seq Len':>8} | {'Inference':>12} | {'Training':>12}")
    print("-" * 40)
    for mode in ["forward", "forward_backward"]:
        prism_map = {r["seq_len"]: r for r in results[mode]["prism"] if not r.get("failed")}
        trans_map = {r["seq_len"]: r for r in results[mode]["transformer"] if not r.get("failed")}

    for s in SEQ_LENGTHS:
        fwd_p = {r["seq_len"]: r for r in results["forward"]["prism"] if not r.get("failed")}
        fwd_t = {r["seq_len"]: r for r in results["forward"]["transformer"] if not r.get("failed")}
        fb_p = {r["seq_len"]: r for r in results["forward_backward"]["prism"] if not r.get("failed")}
        fb_t = {r["seq_len"]: r for r in results["forward_backward"]["transformer"] if not r.get("failed")}

        fwd_speedup = fwd_t[s]["avg_latency_ms"] / fwd_p[s]["avg_latency_ms"] if s in fwd_p and s in fwd_t else float("nan")
        fb_speedup = fb_t[s]["avg_latency_ms"] / fb_p[s]["avg_latency_ms"] if s in fb_p and s in fb_t else float("nan")
        print(f"{s:>8} | {fwd_speedup:>11.2f}x | {fb_speedup:>11.2f}x")
