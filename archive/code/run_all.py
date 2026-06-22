"""
PRISM: Complete Experimental Suite

Runs all benchmarks and generates a comprehensive results summary:
1. Theoretical FLOP analysis
2. Empirical scaling benchmarks (throughput + memory vs sequence length)
3. Quality comparison (contrastive training + retrieval evaluation)
4. Ablation study (isolating each component's contribution)
5. Summary report with all plots

Usage:
    uv run python run_all.py
"""

import json
import time
import traceback
from pathlib import Path

import torch

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def section(title):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


def main():
    t_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    scaling_results = None
    quality_results = None
    ablation_results = None

    # ---- 1. Scaling Benchmarks ----
    try:
        section("1. COMPUTATIONAL SCALING BENCHMARKS")
        from benchmark_scaling import run_all_benchmarks, plot_scaling_results, plot_theoretical_flops
        plot_theoretical_flops()
        scaling_results = run_all_benchmarks()
        plot_scaling_results(scaling_results)
    except Exception:
        print("SCALING BENCHMARK FAILED:")
        traceback.print_exc()

    # ---- 2. Quality Comparison ----
    try:
        section("2. EMBEDDING QUALITY COMPARISON")
        from benchmark_quality import run_quality_comparison
        quality_results = run_quality_comparison()
    except Exception:
        print("QUALITY BENCHMARK FAILED:")
        traceback.print_exc()

    # ---- 3. Ablation Study ----
    try:
        section("3. ABLATION STUDY")
        from benchmark_ablations import run_ablations
        ablation_results = run_ablations()
    except Exception:
        print("ABLATION STUDY FAILED:")
        traceback.print_exc()

    # ---- Summary Report ----
    section("FINAL SUMMARY")
    total_time = time.perf_counter() - t_start

    print(f"Total experiment time: {total_time / 60:.1f} minutes")
    print(f"Device: {device}")
    print(f"\nAll results saved in: {RESULTS_DIR.absolute()}/")

    summary = {"device": device, "total_time_min": total_time / 60}

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Scaling
    if scaling_results:
        fwd_prism = {r["seq_len"]: r for r in scaling_results["forward"]["prism"] if not r.get("failed")}
        fwd_trans = {r["seq_len"]: r for r in scaling_results["forward"]["transformer"] if not r.get("failed")}
        fb_prism = {r["seq_len"]: r for r in scaling_results["forward_backward"]["prism"] if not r.get("failed")}
        fb_trans = {r["seq_len"]: r for r in scaling_results["forward_backward"]["transformer"] if not r.get("failed")}

        common_fwd = sorted(set(fwd_prism) & set(fwd_trans))
        common_fb = sorted(set(fb_prism) & set(fb_trans))

        print("\n1. SCALING (inference speedup = Transformer_time / PRISM_time):")
        for s in common_fwd:
            sp = fwd_trans[s]["avg_latency_ms"] / fwd_prism[s]["avg_latency_ms"]
            print(f"   seq_len={s:>6}: {sp:.2f}x")

        if common_fb:
            max_fb = common_fb[-1]
            sp = fb_trans[max_fb]["avg_latency_ms"] / fb_prism[max_fb]["avg_latency_ms"]
            print(f"\n   Training speedup at seq_len={max_fb}: {sp:.2f}x")

        # Check where Transformer OOMed
        trans_oom = [r["seq_len"] for r in scaling_results["forward_backward"]["transformer"] if r.get("failed")]
        prism_oom = [r["seq_len"] for r in scaling_results["forward_backward"]["prism"] if r.get("failed")]
        if trans_oom and not prism_oom:
            print(f"   Transformer OOM at training seq_len={trans_oom[0]}; PRISM still runs.")

        summary["scaling"] = {
            "inference_speedups": {s: fwd_trans[s]["avg_latency_ms"] / fwd_prism[s]["avg_latency_ms"] for s in common_fwd},
            "training_speedups": {s: fb_trans[s]["avg_latency_ms"] / fb_prism[s]["avg_latency_ms"] for s in common_fb},
            "transformer_oom_at": trans_oom,
        }

    # Quality
    if quality_results:
        pm = quality_results["prism"]["metrics"]
        tm = quality_results["transformer"]["metrics"]
        print(f"\n2. QUALITY:")
        print(f"   PRISM MRR={pm['mrr']:.4f}  R@1={pm['recall@1']:.4f}  R@5={pm['recall@5']:.4f}")
        print(f"   Trans MRR={tm['mrr']:.4f}  R@1={tm['recall@1']:.4f}  R@5={tm['recall@5']:.4f}")
        print(f"   Gap: {pm['mrr'] - tm['mrr']:+.4f} MRR (positive = PRISM better)")
        summary["quality"] = {"prism_mrr": pm["mrr"], "transformer_mrr": tm["mrr"]}

    # Ablations
    if ablation_results:
        full_mrr = ablation_results["Full PRISM"]["metrics"]["mrr"]
        print(f"\n3. ABLATIONS (Full PRISM MRR = {full_mrr:.4f}):")
        summary["ablations"] = {}
        for name, r in ablation_results.items():
            if name == "Full PRISM":
                continue
            delta = r["metrics"]["mrr"] - full_mrr
            print(f"   {name}: {delta:+.4f} MRR")
            summary["ablations"][name] = delta

    # Save combined summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'summary.json'}")

    print("\n" + "=" * 70)
    print(f"DONE in {total_time / 60:.1f} minutes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
