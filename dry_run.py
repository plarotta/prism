"""
Dry run: exercises the entire V2 pipeline with minimal sample sizes.
Verifies imports, model construction, forward/backward, eval, and plotting.
Should complete in < 60 seconds on CPU/MPS.

Usage:
    uv run python dry_run.py
"""

import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Tiny config for dry run
B = 4          # batch size
T_Q = 16       # query length
T_P = 32       # passage length
STEPS = 5      # training steps
VOCAB = 256    # vocabulary size
N_EVAL = 10    # eval queries
CORPUS = 50    # corpus size

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

passed = []
failed = []


def check(name, fn):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS ({elapsed:.1f}s)")
        passed.append(name)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  FAIL ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        failed.append(name)


# ---------------------------------------------------------------------------
# 1. Core imports
# ---------------------------------------------------------------------------

def check_imports():
    from prism import (
        prism_small, PRISMEncoder, PRISMForEmbedding,
        CrossScaleInterference, CrossScaleInterferenceV2,
        AttentiveCovariancePooling, AttentiveCovariancePoolingV2,
        _fast_fixed_decay_scan, _parallel_scan,
        StratifiedProjection, StratifiedRecurrence, DirectionalFusion,
    )
    from baseline_transformer import transformer_small, TransformerForEmbedding
    from benchmark_quality import SyntheticEmbeddingDataset, evaluate_retrieval
    from benchmark_ablations import (
        NoInterference, LearnedDecayRecurrence, AdditiveInterference, MeanPooling,
    )
    from benchmark_v2_ablations import (
        InterferenceNormOnly, InterferenceScaleOnly, InterferenceAlphaInit,
        InterferenceGateOnly, CovPoolingNormOnly, CovPoolingRankProj,
        ABLATIONS,
    )
    print(f"  All imports OK. {len(ABLATIONS)} V2 ablation variants found.")

check("Core imports", check_imports)


# ---------------------------------------------------------------------------
# 2. Model construction (all variants)
# ---------------------------------------------------------------------------

def check_model_construction():
    from prism import prism_small
    from baseline_transformer import transformer_small
    from benchmark_v2_ablations import ABLATIONS

    models = {}

    # PRISM small
    m = prism_small(vocab_size=VOCAB)
    models["prism_small"] = sum(p.numel() for p in m.parameters())

    # Transformer small
    m = transformer_small(vocab_size=VOCAB)
    models["transformer_small"] = sum(p.numel() for p in m.parameters())

    # All V2 ablation variants
    for name, builder in ABLATIONS.items():
        # Monkey-patch VOCAB_SIZE in the builder's closure
        import benchmark_v2_ablations as bv2
        old_vocab = bv2.VOCAB_SIZE
        bv2.VOCAB_SIZE = VOCAB
        try:
            m = builder()
            models[name] = sum(p.numel() for p in m.parameters())
        finally:
            bv2.VOCAB_SIZE = old_vocab

    for name, n in models.items():
        print(f"    {name}: {n:,} params")

check("Model construction", check_model_construction)


# ---------------------------------------------------------------------------
# 3. Forward + backward pass (PRISM + Transformer)
# ---------------------------------------------------------------------------

def check_forward_backward():
    from prism import prism_small, PRISMForEmbedding
    from baseline_transformer import transformer_small, TransformerForEmbedding

    for name, build_enc, Wrapper in [
        ("PRISM", lambda: prism_small(vocab_size=VOCAB), PRISMForEmbedding),
        ("Transformer", lambda: transformer_small(vocab_size=VOCAB), TransformerForEmbedding),
    ]:
        enc = build_enc().to(device)
        wrapper = Wrapper(enc, temperature=0.05).to(device)

        q = torch.randint(1, VOCAB, (B, T_Q), device=device)
        p = torch.randint(1, VOCAB, (B, T_P), device=device)
        q_mask = torch.ones_like(q)
        p_mask = torch.ones_like(p)

        result = wrapper(q, q_mask, p, p_mask)
        loss = result["loss"]
        loss.backward()

        grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                      for p in wrapper.parameters() if p.requires_grad)
        print(f"    {name}: loss={loss.item():.4f}  grads_nonzero={grad_ok}")

check("Forward + backward", check_forward_backward)


# ---------------------------------------------------------------------------
# 4. V2 interference forward pass
# ---------------------------------------------------------------------------

def check_v2_interference():
    from prism import CrossScaleInterferenceV2

    C, d_c = 6, 64
    module = CrossScaleInterferenceV2(d_c, C).to(device)
    hiddens = [torch.randn(B, T_Q, d_c, device=device) for _ in range(C)]
    out = module(hiddens)
    assert len(out) == C
    assert out[0].shape == (B, T_Q, d_c)

    # Check gate values
    gate_vals = torch.sigmoid(module.gamma).squeeze()
    print(f"    Output shape OK: {C} x ({B}, {T_Q}, {d_c})")
    print(f"    Gate values at init: {gate_vals.detach().cpu().numpy().round(4)}")
    print(f"    Alpha mean (off-diag): {module.alpha.data[~torch.eye(C, dtype=bool)].mean():.4f}")

check("V2 interference", check_v2_interference)


# ---------------------------------------------------------------------------
# 5. V2 covariance pooling forward pass
# ---------------------------------------------------------------------------

def check_v2_pooling():
    from prism import AttentiveCovariancePoolingV2

    d, d_e, rank = 384, 384, 8
    module = AttentiveCovariancePoolingV2(d, d_e, cov_rank=rank).to(device)
    f = torch.randn(B, T_P, d, device=device)
    query = torch.randn(B, d, device=device)
    mask = torch.ones(B, T_P, dtype=torch.bool, device=device)

    emb = module(f, query, mask)
    assert emb.shape == (B, d_e), f"Expected ({B}, {d_e}), got {emb.shape}"
    print(f"    Output shape OK: ({B}, {d_e})")
    print(f"    Cov rank={rank}, cov vector dim={rank*rank}, projected to {d}")

check("V2 covariance pooling", check_v2_pooling)


# ---------------------------------------------------------------------------
# 6. Synthetic training loop (tiny)
# ---------------------------------------------------------------------------

def check_training_loop():
    from prism import prism_small, PRISMForEmbedding
    from benchmark_quality import SyntheticEmbeddingDataset

    dataset = SyntheticEmbeddingDataset(VOCAB)
    model = prism_small(vocab_size=VOCAB).to(device)
    wrapper = PRISMForEmbedding(model, temperature=0.05).to(device)
    opt = torch.optim.AdamW(wrapper.parameters(), lr=3e-4)

    wrapper.train()
    losses = []
    for step in range(STEPS):
        q, p, _ = dataset.generate_batch(B, T_Q, T_P)
        q, p = q.to(device), p.to(device)
        result = wrapper(q, torch.ones_like(q), p, torch.ones_like(p))
        result["loss"].backward()
        opt.step()
        opt.zero_grad()
        losses.append(result["loss"].item())

    print(f"    {STEPS} steps OK. Loss: {losses[0]:.3f} → {losses[-1]:.3f}")

check("Training loop", check_training_loop)


# ---------------------------------------------------------------------------
# 7. Eval pipeline (tiny)
# ---------------------------------------------------------------------------

def check_eval_pipeline():
    from prism import prism_small, PRISMForEmbedding
    from benchmark_quality import SyntheticEmbeddingDataset, evaluate_retrieval

    dataset = SyntheticEmbeddingDataset(VOCAB)
    model = prism_small(vocab_size=VOCAB).to(device)
    wrapper = PRISMForEmbedding(model, temperature=0.05).to(device)

    queries, corpus, qt, ct = dataset.generate_corpus_and_queries(N_EVAL, CORPUS, T_Q, T_P)
    metrics = evaluate_retrieval(wrapper, queries, corpus, qt, ct)
    print(f"    MRR={metrics['mrr']:.4f}  R@1={metrics['recall@1']:.4f}")

check("Eval pipeline", check_eval_pipeline)


# ---------------------------------------------------------------------------
# 8. V2 ablation variant builds + forward pass
# ---------------------------------------------------------------------------

def check_v2_ablation_variants():
    from benchmark_v2_ablations import ABLATIONS
    from prism import PRISMForEmbedding
    import benchmark_v2_ablations as bv2

    old_vocab = bv2.VOCAB_SIZE
    bv2.VOCAB_SIZE = VOCAB
    try:
        for name, builder in ABLATIONS.items():
            model = builder().to(device)
            wrapper = PRISMForEmbedding(model, temperature=0.05).to(device)

            q = torch.randint(1, VOCAB, (B, T_Q), device=device)
            p = torch.randint(1, VOCAB, (B, T_P), device=device)
            result = wrapper(q, torch.ones_like(q), p, torch.ones_like(p))
            result["loss"].backward()

            print(f"    {name}: loss={result['loss'].item():.3f} OK")
    finally:
        bv2.VOCAB_SIZE = old_vocab

check("V2 ablation variants", check_v2_ablation_variants)


# ---------------------------------------------------------------------------
# 9. Scaling benchmark (single length)
# ---------------------------------------------------------------------------

def check_scaling():
    from prism import prism_small
    from baseline_transformer import transformer_small

    for name, builder in [("PRISM", prism_small), ("Transformer", transformer_small)]:
        model = builder(vocab_size=VOCAB).to(device)
        ids = torch.randint(1, VOCAB, (B, 64), device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(ids, mask)
        emb = out["embedding"]
        print(f"    {name}: embedding shape {tuple(emb.shape)}")

check("Scaling (smoke)", check_scaling)


# ---------------------------------------------------------------------------
# 10. Real data imports + tokenizer
# ---------------------------------------------------------------------------

def check_real_data_imports():
    from benchmark_real_data import (
        build_simplified_prism, build_transformer,
        load_nli_pairs, load_sts_benchmark, load_long_documents,
        run_phase_3a, run_phase_3b, run_phase_3c,
        collate_pairs, collate_docs,
        train_contrastive, evaluate_sts,
        make_article_pairs, train_contrastive_long,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_ids = tokenizer.encode("This is a dry run test.", add_special_tokens=False)
    print(f"    Tokenizer OK: 'This is a dry run test.' → {len(test_ids)} tokens")

    # Test model build with real vocab
    from prism import PRISMForEmbedding
    model = build_simplified_prism().to(device)
    wrapper = PRISMForEmbedding(model, temperature=0.05).to(device)
    ids = torch.tensor([test_ids[:T_Q] + [0] * (T_Q - len(test_ids[:T_Q]))],
                       device=device)
    mask = torch.ones_like(ids)
    mask[0, len(test_ids[:T_Q]):] = 0
    emb = wrapper.encode(ids, mask)
    print(f"    Real-vocab model encode OK: shape {tuple(emb.shape)}")

check("Real data imports + tokenizer", check_real_data_imports)


# ---------------------------------------------------------------------------
# 11. Real data collation + training (tiny)
# ---------------------------------------------------------------------------

def check_real_data_training():
    from benchmark_real_data import (
        build_simplified_prism, collate_pairs, train_contrastive,
        make_article_pairs, train_contrastive_long,
        REAL_VOCAB_SIZE,
    )
    from prism import PRISMForEmbedding
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Fake NLI pairs
    pairs = []
    for _ in range(20):
        s1 = tokenizer.encode("The cat sat on the mat", add_special_tokens=False)
        s2 = tokenizer.encode("A feline rested on a rug", add_special_tokens=False)
        pairs.append((s1, s2))

    model = build_simplified_prism().to(device)
    wrapper = PRISMForEmbedding(model, temperature=0.05).to(device)
    losses, accs, t = train_contrastive(wrapper, pairs, STEPS, "dry-run", max_len=32, lr=3e-4)
    print(f"    NLI training OK: {STEPS} steps, loss {losses[0]:.3f} → {losses[-1]:.3f}")

    # Fake long-doc pairs
    doc_pairs = [(list(range(50, 150)), list(range(60, 160))) for _ in range(20)]
    model2 = build_simplified_prism().to(device)
    wrapper2 = PRISMForEmbedding(model2, temperature=0.05).to(device)
    losses2, _, _ = train_contrastive_long(wrapper2, doc_pairs, STEPS, "dry-run-long", max_len=64)
    print(f"    Long-doc training OK: {STEPS} steps, loss {losses2[0]:.3f} → {losses2[-1]:.3f}")

check("Real data training (tiny)", check_real_data_training)


# ---------------------------------------------------------------------------
# 12. Plotting (verify matplotlib works)
# ---------------------------------------------------------------------------

def check_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Dry run plot")
    fig.savefig(RESULTS_DIR / "dry_run_test.png")
    plt.close("all")
    (RESULTS_DIR / "dry_run_test.png").unlink()
    print("    Matplotlib save/close OK")

check("Plotting", check_plotting)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"DRY RUN COMPLETE: {len(passed)} passed, {len(failed)} failed")
print("=" * 60)

if failed:
    print("\nFAILED:")
    for name in failed:
        print(f"  ✗ {name}")

print("\nPASSED:")
for name in passed:
    print(f"  ✓ {name}")

if failed:
    print(f"\nFix the {len(failed)} failure(s) before running the full suite.")
    sys.exit(1)
else:
    print("\nAll clear. Run the full suite with:")
    print("  uv run python run_all_v2.py")
    sys.exit(0)
