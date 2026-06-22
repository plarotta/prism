"""
PRISM: Complete V2 Experimental Suite

Runs ALL benchmarks in sequence:
1. Computational scaling (throughput, memory, latency vs seq length)
2. Quality comparison (PRISM vs Transformer, synthetic contrastive)
3. Original ablation study (interference, decay, pooling)
4. V2 ablation study (targeted fixes from PRISM_v2fixes.txt)
5. Real data evaluation (NLI + STS-B + length-controlled retrieval)

Usage:
    uv run python run_all_v2.py
"""

import json
import random
import time
import traceback
from pathlib import Path

import torch
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def section(title):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


def main():
    t_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 70)
    print("PRISM V2 COMPLETE EXPERIMENTAL SUITE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    scaling_results = None
    quality_results = None
    ablation_results = None
    v2_ablation_results = None
    real_data_results = None

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

    # ---- 3. Original Ablation Study ----
    try:
        section("3. ORIGINAL ABLATION STUDY")
        from benchmark_ablations import run_ablations
        ablation_results = run_ablations()
    except Exception:
        print("ORIGINAL ABLATION STUDY FAILED:")
        traceback.print_exc()

    # ---- 4. V2 Ablation Study ----
    try:
        section("4. V2 ABLATION STUDY (TARGETED FIXES)")
        from benchmark_v2_ablations import run_v2_ablations
        v2_ablation_results = run_v2_ablations()
    except Exception:
        print("V2 ABLATION STUDY FAILED:")
        traceback.print_exc()

    # ---- 5. Real Data Evaluation ----
    try:
        section("5. REAL DATA EVALUATION")
        from benchmark_real_data import (
            build_simplified_prism, build_transformer,
            load_nli_pairs, load_sts_benchmark, load_long_documents,
            run_phase_3a, run_phase_3b, run_phase_3c,
            TOKENIZER_NAME,
        )
        from transformers import AutoTokenizer

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        train_pairs = load_nli_pairs(tokenizer, max_pairs=50000, max_len=128)
        sts_examples = load_sts_benchmark(tokenizer, max_len=128)

        real_data_results = {}

        # 3A: STS-B
        real_data_results["phase_3a"] = run_phase_3a(
            tokenizer, train_pairs, sts_examples, n_steps=2000
        )

        # 3B: Length-controlled retrieval
        documents = load_long_documents(tokenizer, n_docs=10000, min_len=128)
        if len(documents) >= 200:
            real_data_results["phase_3b"] = run_phase_3b(tokenizer, documents, n_steps=2000)

        # 3C: Throughput scaling
        real_data_results["phase_3c"] = run_phase_3c()

        with open(RESULTS_DIR / "real_data_results.json", "w") as f:
            json.dump(real_data_results, f, indent=2, default=str)

    except ImportError as e:
        print(f"REAL DATA EVALUATION SKIPPED (missing dependency: {e})")
        print("Install with: uv add datasets transformers scipy")
    except Exception:
        print("REAL DATA EVALUATION FAILED:")
        traceback.print_exc()

    # ---- Summary ----
    section("FINAL SUMMARY")
    total_time = time.perf_counter() - t_start

    print(f"Total experiment time: {total_time / 60:.1f} minutes")
    print(f"Device: {device}")

    summary = {"device": device, "total_time_min": total_time / 60}

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # 1. Scaling
    if scaling_results:
        fwd_prism = {r["seq_len"]: r for r in scaling_results["forward"]["prism"] if not r.get("failed")}
        fwd_trans = {r["seq_len"]: r for r in scaling_results["forward"]["transformer"] if not r.get("failed")}
        common = sorted(set(fwd_prism) & set(fwd_trans))

        print("\n1. SCALING (inference speedup):")
        for s in common:
            sp = fwd_trans[s]["avg_latency_ms"] / fwd_prism[s]["avg_latency_ms"]
            print(f"   seq_len={s:>6}: {sp:.2f}x")

        trans_oom = [r["seq_len"] for r in scaling_results["forward_backward"]["transformer"] if r.get("failed")]
        if trans_oom:
            print(f"   Transformer OOM at training seq_len={trans_oom[0]}")

        summary["scaling"] = {
            "inference_speedups": {s: fwd_trans[s]["avg_latency_ms"] / fwd_prism[s]["avg_latency_ms"] for s in common},
            "transformer_oom_at": trans_oom,
        }

    # 2. Quality
    if quality_results:
        pm = quality_results["prism"]["metrics"]
        tm = quality_results["transformer"]["metrics"]
        print(f"\n2. QUALITY (synthetic, short seq):")
        print(f"   PRISM MRR={pm['mrr']:.4f}  Transformer MRR={tm['mrr']:.4f}  gap={pm['mrr'] - tm['mrr']:+.4f}")
        summary["quality"] = {"prism_mrr": pm["mrr"], "transformer_mrr": tm["mrr"]}

    # 3. Original ablations
    if ablation_results:
        full_mrr = ablation_results["Full PRISM"]["metrics"]["mrr"]
        print(f"\n3. ORIGINAL ABLATIONS (Full PRISM MRR={full_mrr:.4f}):")
        summary["ablations"] = {}
        for name, r in ablation_results.items():
            if name == "Full PRISM":
                continue
            delta = r["metrics"]["mrr"] - full_mrr
            print(f"   {name}: {delta:+.4f}")
            summary["ablations"][name] = delta

    # 4. V2 ablations
    if v2_ablation_results:
        base_mrr = v2_ablation_results["Baseline (full original)"]["metrics"]["mrr"]
        simple_mrr = v2_ablation_results["Simplified (mean pool)"]["metrics"]["mrr"]
        print(f"\n4. V2 ABLATIONS (Baseline={base_mrr:.4f}, Simplified={simple_mrr:.4f}):")
        summary["v2_ablations"] = {}
        for name, r in v2_ablation_results.items():
            if name in ("Baseline (full original)", "Simplified (mean pool)"):
                continue
            mrr = r["metrics"]["mrr"]
            print(f"   {name}: MRR={mrr:.4f} (vs base: {mrr-base_mrr:+.4f}, vs simple: {mrr-simple_mrr:+.4f})")
            summary["v2_ablations"][name] = {"mrr": mrr, "vs_base": mrr - base_mrr, "vs_simple": mrr - simple_mrr}

        g_mrr = v2_ablation_results.get("G: All V2 Fixes", {}).get("metrics", {}).get("mrr")
        if g_mrr:
            if g_mrr >= simple_mrr - 0.005:
                print(f"\n   >>> V2 FIXES SUCCESSFUL: G ({g_mrr:.4f}) matches/beats simplified ({simple_mrr:.4f})")
            else:
                print(f"\n   >>> V2 fixes fell short: G ({g_mrr:.4f}) < simplified ({simple_mrr:.4f})")

    # 5. Real data
    if real_data_results:
        print(f"\n5. REAL DATA:")
        if "phase_3a" in real_data_results:
            for name, r in real_data_results["phase_3a"].items():
                print(f"   STS-B {name}: Spearman={r['sts_spearman']:.4f}")
        if "phase_3b" in real_data_results and real_data_results["phase_3b"]:
            print("   Length-controlled retrieval:")
            for seq_len, by_model in sorted(real_data_results["phase_3b"].items(), key=lambda x: int(x[0]) if isinstance(x[0], (int, str)) and str(x[0]).isdigit() else 0):
                if not isinstance(by_model, dict):
                    continue
                for name, r in by_model.items():
                    if isinstance(r, dict) and "mrr" in r:
                        print(f"     seq={seq_len} {name}: MRR={r['mrr']:.4f}")
        summary["real_data"] = real_data_results

    with open(RESULTS_DIR / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'summary_v2.json'}")

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {total_time / 60:.1f} minutes.")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
