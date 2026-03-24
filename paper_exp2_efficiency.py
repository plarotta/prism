"""
Experiment 2: Scaling curve benchmarks.

Measures inference latency, training throughput, and peak GPU memory
for all 4 architectures across sequence lengths 64-16384.

Usage:
    uv run python paper_exp2_efficiency.py
    uv run python paper_exp2_efficiency.py --models prism,transformer
    uv run python paper_exp2_efficiency.py --batch-sizes 1,32
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_log import create_run_dir, save_config, capture_hardware_info

# Model builders (same as exp1)
from prism import prism_small, PRISMForEmbedding
from benchmark_ablations import MeanPooling, NoInterference
from baseline_transformer import transformer_small, TransformerForEmbedding
from mamba_bidir import build_mamba_bidir_small
from linear_rnn import build_linear_rnn_small

VOCAB_SIZE = 30522
SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
RESULTS_DIR = Path("results") / "paper" / "exp2"


def build_prism(max_len):
    encoder = prism_small(vocab_size=VOCAB_SIZE, max_len=max_len)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
        if layer.bidirectional:
            layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)
    encoder.pooling = MeanPooling(encoder.d, encoder.d_e)
    for layer in encoder.layers:
        layer.recurrence.lambdas.fill_(0.99)
    return PRISMForEmbedding(encoder)


def build_transformer(max_len):
    return TransformerForEmbedding(transformer_small(vocab_size=VOCAB_SIZE, max_len=max_len))


MODEL_BUILDERS = {
    "prism": ("PRISM-Simplified", build_prism),
    "transformer": ("Transformer", build_transformer),
    "mamba": ("Mamba-Bidir", lambda ml: build_mamba_bidir_small(VOCAB_SIZE, ml)),
    "linear_rnn": ("Linear-RNN", lambda ml: build_linear_rnn_small(VOCAB_SIZE, ml)),
}


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def _make_dummy_batch(batch_size, seq_len, device):
    ids = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return ids, mask


@torch.no_grad()
def measure_inference_latency(
    model, seq_len, batch_size, device, n_runs=100, n_warmup=10,
):
    """Median inference latency in ms."""
    model.eval()
    ids, mask = _make_dummy_batch(batch_size, seq_len, device)

    # Warmup
    for _ in range(n_warmup):
        model.encode(ids, mask)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encode(ids, mask)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "median_ms": round(float(np.median(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "mean_ms": round(float(np.mean(times)), 2),
    }


def measure_training_step(model, seq_len, batch_size, device, n_runs=20, n_warmup=5):
    """Measure forward+backward time and throughput."""
    model.train()
    ids, mask = _make_dummy_batch(batch_size, seq_len, device)

    # Warmup
    for _ in range(n_warmup):
        result = model(ids, mask, ids, mask)
        result["loss"].backward()
        model.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        model.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model(ids, mask, ids, mask)
        result["loss"].backward()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_s = float(np.median(times))
    seqs_per_s = (batch_size * 2) / median_s  # *2 because query + pos

    return {
        "median_ms": round(median_s * 1000, 2),
        "seqs_per_s": round(seqs_per_s, 1),
    }


def measure_peak_memory(model, seq_len, batch_size, device):
    """Peak GPU memory for forward+backward."""
    if device != "cuda":
        return {"fwd_bwd_mb": 0}

    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    ids, mask = _make_dummy_batch(batch_size, seq_len, device)
    result = model(ids, mask, ids, mask)
    result["loss"].backward()

    peak_mb = torch.cuda.max_memory_allocated() // (1024**2)
    model.zero_grad()
    return {"fwd_bwd_mb": int(peak_mb)}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmarks(
    model_keys: list[str],
    seq_lengths: list[int],
    batch_sizes: list[int],
    device: str,
):
    """Run all benchmarks and return results dict."""
    results = {}

    for model_key in model_keys:
        model_name, build_fn = MODEL_BUILDERS[model_key]
        results[model_key] = {"name": model_name, "lengths": {}}

        for seq_len in seq_lengths:
            print(f"\n  {model_name} @ seq_len={seq_len}")
            length_results = {}

            try:
                model = build_fn(max(seq_len, 256)).to(device)  # min max_len=256 for pos emb

                # Inference latency at each batch size
                for bs in batch_sizes:
                    try:
                        lat = measure_inference_latency(model, seq_len, bs, device)
                        length_results[f"latency_b{bs}"] = lat
                        print(f"    Inference (batch={bs}): {lat['median_ms']:.1f}ms")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            length_results[f"latency_b{bs}"] = {"oom": True}
                            print(f"    Inference (batch={bs}): OOM")
                            torch.cuda.empty_cache()
                        else:
                            raise

                # Training throughput (batch=16)
                try:
                    train_bs = min(16, batch_sizes[-1])
                    thr = measure_training_step(model, seq_len, train_bs, device)
                    length_results["training"] = thr
                    print(f"    Training (batch={train_bs}): "
                          f"{thr['median_ms']:.1f}ms, {thr['seqs_per_s']:.0f} seq/s")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        length_results["training"] = {"oom": True}
                        print(f"    Training: OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise

                # Peak memory
                try:
                    mem = measure_peak_memory(model, seq_len, 16, device)
                    length_results["memory"] = mem
                    print(f"    Peak memory: {mem['fwd_bwd_mb']} MB")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        length_results["memory"] = {"oom": True}
                        print(f"    Peak memory: OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise

                model.cpu()
                del model

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    length_results["build"] = {"oom": True}
                    print(f"    Model build: OOM")
                else:
                    raise

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results[model_key]["lengths"][str(seq_len)] = length_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "prism": "#2563EB",
    "transformer": "#DC2626",
    "mamba": "#16A34A",
    "linear_rnn": "#9333EA",
}

MARKERS = {
    "prism": "o",
    "transformer": "s",
    "mamba": "D",
    "linear_rnn": "^",
}


def plot_scaling_curves(results, seq_lengths, batch_size=32):
    """Generate latency, memory, and throughput plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Latency ---
    ax = axes[0]
    for model_key, data in results.items():
        lengths, latencies = [], []
        for sl in seq_lengths:
            sl_data = data["lengths"].get(str(sl), {})
            lat = sl_data.get(f"latency_b{batch_size}", {})
            if not lat.get("oom"):
                med = lat.get("median_ms")
                if med is not None:
                    lengths.append(sl)
                    latencies.append(med)
        if lengths:
            ax.plot(lengths, latencies, marker=MARKERS[model_key],
                    color=COLORS[model_key], label=data["name"], linewidth=2)
            # Mark OOM points
            for sl in seq_lengths:
                sl_data = data["lengths"].get(str(sl), {})
                lat = sl_data.get(f"latency_b{batch_size}", {})
                if lat.get("oom") and lengths:
                    ax.plot(sl, latencies[-1] * 2, "x", color=COLORS[model_key],
                            markersize=12, markeredgewidth=3)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Inference Latency (batch={batch_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Memory ---
    ax = axes[1]
    for model_key, data in results.items():
        lengths, mems = [], []
        for sl in seq_lengths:
            sl_data = data["lengths"].get(str(sl), {})
            mem = sl_data.get("memory", {})
            if not mem.get("oom"):
                val = mem.get("fwd_bwd_mb")
                if val is not None:
                    lengths.append(sl)
                    mems.append(val)
        if lengths:
            ax.plot(lengths, mems, marker=MARKERS[model_key],
                    color=COLORS[model_key], label=data["name"], linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Training Peak Memory (batch=16)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Throughput ---
    ax = axes[2]
    for model_key, data in results.items():
        lengths, thrpts = [], []
        for sl in seq_lengths:
            sl_data = data["lengths"].get(str(sl), {})
            thr = sl_data.get("training", {})
            if not thr.get("oom"):
                val = thr.get("seqs_per_s")
                if val is not None:
                    lengths.append(sl)
                    thrpts.append(val)
        if lengths:
            ax.plot(lengths, thrpts, marker=MARKERS[model_key],
                    color=COLORS[model_key], label=data["name"], linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Sequences / second")
    ax.set_title("Training Throughput (batch=16)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "scaling_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Efficiency benchmarks")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys")
    parser.add_argument("--seq-lengths", type=str, default=None,
                        help="Comma-separated seq lengths (default: 64-16384)")
    parser.add_argument("--batch-sizes", type=str, default="1,32",
                        help="Comma-separated batch sizes for latency")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = args.models.split(",") if args.models else list(MODEL_BUILDERS.keys())
    seq_lengths = (
        [int(x) for x in args.seq_lengths.split(",")]
        if args.seq_lengths else SEQ_LENGTHS
    )
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 2: Efficiency Benchmarks")
    print(f"  Models: {model_keys}")
    print(f"  Seq lengths: {seq_lengths}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Device: {device}")
    print(f"  Hardware: {capture_hardware_info()}")
    print("=" * 70)

    results = run_benchmarks(model_keys, seq_lengths, batch_sizes, device)

    # Save results
    out_path = RESULTS_DIR / "scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Generate plots
    for bs in batch_sizes:
        plot_scaling_curves(results, seq_lengths, batch_size=bs)

    print("\n=== Experiment 2 complete ===")


if __name__ == "__main__":
    main()
