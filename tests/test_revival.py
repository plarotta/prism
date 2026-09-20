"""Regression checks for bugs that could change research conclusions."""
import copy
import json
import random
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from baseline_transformer import TransformerEncoder, TransformerEncoderLayer, TransformerForEmbedding
from contrastive_batch import cached_contrastive_backward, contrastive_loss
from paper_components import LearnedDecayRecurrence, MeanPooling, NoInterference
from prism import PRISMEncoder, PRISMForEmbedding, StratifiedProjection, StratifiedRecurrence

torch.set_num_threads(2)


def tiny_prism(dropout=0.0, pooling="mean", position_encoding="learned", max_len=256):
    model = PRISMEncoder(vocab_size=128, d=32, d_e=24, n_layers=2, n_channels=4,
                         max_len=max_len, dropout=dropout, cov_rank=2,
                         position_encoding=position_encoding)
    for layer in model.layers:
        layer.interference_fwd = NoInterference(8, 4)
        layer.interference_bwd = NoInterference(8, 4)
        layer.recurrence.lambdas.fill_(0.99)
    if pooling == "mean":
        model.pooling = MeanPooling(32, 24)
    return PRISMForEmbedding(model)


@pytest.mark.parametrize("pooling", ["mean", "covariance"])
def test_padding_does_not_change_embedding_or_parameter_gradients(pooling):
    torch.manual_seed(7)
    m = tiny_prism(pooling=pooling).eval()
    with torch.no_grad():
        for layer in m.encoder.layers:
            layer.norm.bias.normal_(0, 0.1)
    ids = torch.randint(1, 128, (2, 13))
    padded = F.pad(ids, (0, 100))
    a = m.encode(ids)
    b = m.encode(padded)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    a[:, 0].sum().backward()
    expected = {name: p.grad.clone() for name, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    b[:, 0].sum().backward()
    for name, p in m.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], atol=2e-5, rtol=2e-4)


def test_batch_composition_and_masked_token_content_do_not_change_embedding():
    torch.manual_seed(4)
    m = tiny_prism().eval()
    with torch.no_grad():
        for layer in m.encoder.layers:
            layer.norm.bias.normal_()
        ids = torch.randint(1, 128, (1, 9))
        batch = torch.randint(1, 128, (2, 100))
        batch[0, :9] = ids[0]
        mask = torch.ones_like(batch)
        mask[0, 9:] = 0
        torch.testing.assert_close(m.encode(ids)[0], m.encode(batch, mask)[0], atol=2e-6, rtol=2e-5)


def test_learned_decay_has_finite_nonzero_gradients_and_updates():
    torch.manual_seed(3)
    rec = LearnedDecayRecurrence(4, 3, max_len=128).double()
    assert torch.isfinite(rec.lambda_logits).all()
    channels = [torch.randn(2, 9, 4, dtype=torch.double) for _ in range(3)]
    before = rec._lambdas.detach().clone()
    optimizer = torch.optim.SGD(rec.parameters(), lr=0.1)
    fwd, bwd = rec(channels)
    sum(h.square().mean() for h in fwd + bwd).backward()
    assert torch.isfinite(rec.lambda_logits.grad).all()
    assert (rec.lambda_logits.grad.abs() > 1e-8).all()
    optimizer.step()
    assert ((rec._lambdas.detach() - before).abs() > 0).all()


def test_learned_decay_derivative_matches_finite_difference():
    from prism_scan_kernel import _ref_forward
    x = torch.randn(1, 7, 3, 2, dtype=torch.double, requires_grad=True)
    logits = torch.tensor([-2., 1., 4.], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: _ref_forward(a, b.sigmoid()), (x, logits))


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("same_decay", [False, True])
def test_grouped_recurrence_matches_independent_sequential_oracle(bidirectional, same_decay):
    torch.manual_seed(0)
    rec = StratifiedRecurrence(4, 3, max_len=257, bidirectional=bidirectional).double()
    if same_decay:
        rec.lambdas.fill_(0.99)
    channels = [torch.randn(2, 19, 4, dtype=torch.double, requires_grad=True) for _ in range(3)]
    fwd, bwd = rec(channels)
    for reverse, outputs, gates in [(False, fwd, rec.gates_fwd)] + (
            [(True, bwd, rec.gates_bwd)] if bidirectional else []):
        for c, (x, gate, actual) in enumerate(zip(channels, gates, outputs)):
            inp = x.flip(1) if reverse else x
            values = inp * torch.sigmoid(gate(inp))
            state = torch.zeros_like(values[:, 0])
            expected = []
            for t in range(values.shape[1]):
                state = rec.lambdas[c] * state + values[:, t]
                expected.append(state)
            expected = torch.stack(expected, 1)
            if reverse:
                expected = expected.flip(1)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            got_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
            ref_grad = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
            torch.testing.assert_close(got_grad, ref_grad, atol=1e-10, rtol=1e-10)


def test_grouped_projection_preserves_checkpoint_keys_and_gradients():
    projection = StratifiedProjection(32, 4).double()
    x = torch.randn(2, 11, 32, dtype=torch.double, requires_grad=True)
    got = torch.cat(projection(x), -1)
    reference = torch.cat([p(x) for p in projection.projections], -1)
    torch.testing.assert_close(got, reference)
    actual = torch.autograd.grad(got.square().sum(), tuple(projection.parameters()))
    expected = torch.autograd.grad(reference.square().sum(), tuple(projection.parameters()))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_transformer_optimized_attention_matches_explicit_attention():
    torch.manual_seed(1)
    layer = TransformerEncoderLayer(32, 4, dropout=0).double()
    x = torch.randn(2, 13, 32, dtype=torch.double)
    mask = torch.zeros(2, 13, dtype=torch.bool)
    mask[0, 8:] = True
    normed = layer.norm1(x)
    attention, _ = layer.attn(normed, normed, normed, key_padding_mask=mask, need_weights=True)
    ref = x + attention
    ref = ref + layer.mlp(layer.norm2(ref))
    torch.testing.assert_close(layer(x, mask), ref, atol=1e-10, rtol=1e-10)


def make_batches(hard=True):
    batches = []
    for size, length in [(3, 9), (2, 13)]:
        b = {}
        for role in ["query", "pos"] + (["neg"] if hard else []):
            shape = (size, 2, length) if role == "neg" else (size, length)
            b[f"{role}_ids"] = torch.randint(1, 128, shape)
            b[f"{role}_mask"] = torch.ones(shape, dtype=torch.long)
        batches.append(b)
    # Duplicate documents and queries across microbatches exercise masking.
    batches[0]["positive_groups"] = torch.tensor([1, 2, 3])
    batches[1]["positive_groups"] = torch.tensor([2, 4])
    batches[0]["query_groups"] = torch.tensor([5, 6, 7])
    batches[1]["query_groups"] = torch.tensor([5, 8])
    return batches


def graph_oracle(model, batches):
    # Same encoder chunk order as caching, but keep ALL activation graphs.
    embs = {role: [] for role in ["query", "pos", "neg"]}
    for b in batches:
        size = b["query_ids"].shape[0]
        for role in embs:
            if f"{role}_ids" not in b:
                continue
            ids, mask = b[f"{role}_ids"], b[f"{role}_mask"]
            ids, mask = ids.reshape(-1, ids.shape[-1]), mask.reshape(-1, mask.shape[-1])
            for start in range(0, len(ids), size):
                embs[role].append(model.encode(ids[start:start+size], mask[start:start+size]))
    q, p = torch.cat(embs["query"]), torch.cat(embs["pos"])
    n = torch.cat(embs["neg"]).reshape(len(q), 2, -1) if embs["neg"] else None
    return contrastive_loss(q, p, model.temperature, n,
                            torch.cat([b["positive_groups"] for b in batches]),
                            torch.cat([b["query_groups"] for b in batches]))


@pytest.mark.parametrize("kind", ["prism", "transformer"])
@pytest.mark.parametrize("hard", [False, True])
def test_cached_batch_matches_full_graph_with_dropout_hard_negatives_and_uneven_chunks(kind, hard):
    torch.manual_seed(11)
    model = tiny_prism(dropout=0.2) if kind == "prism" else TransformerForEmbedding(
        TransformerEncoder(128, d=32, d_e=24, n_layers=1, n_heads=4, max_len=32, dropout=0.2))
    cached = copy.deepcopy(model)
    batches = make_batches(hard)
    torch.manual_seed(1337)
    expected = graph_oracle(model, batches)
    expected.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(1337)
    actual = cached_contrastive_backward(cached, batches)
    torch.testing.assert_close(actual, expected.detach(), atol=1e-6, rtol=1e-6)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for (name, p), (_, q) in zip(model.named_parameters(), cached.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(q.grad, p.grad, atol=3e-5, rtol=1e-4, msg=name)


def test_false_negatives_are_excluded():
    q = torch.tensor([[1., 0.], [1., 0.]])
    assert contrastive_loss(q, q, positive_groups=torch.tensor([7, 7])).item() == 0
    assert contrastive_loss(q, q).item() > 0.69


def test_position_free_variant_encodes_beyond_training_position_table():
    model = tiny_prism(position_encoding="none").eval()
    assert model.encoder.pos_emb is None
    with torch.no_grad():
        emb = model.encode(torch.randint(1, 128, (1, 513)))
    assert emb.shape == (1, 24) and torch.isfinite(emb).all()


def test_hard_negative_cache_round_trip_and_sampling(tmp_path):
    from data.msmarco import MSMARCODataset
    dataset = MSMARCODataset(None, max_len=32, n_hard_negatives=2, cache_dir=tmp_path)
    dataset.passages = {str(i): [i + 1] * (i + 2) for i in range(8)}
    dataset.train_queries = {"a": [1, 2], "b": [3, 4]}
    dataset.train_qrels = {"a": ["0"], "b": ["1"]}
    dataset.train_negatives = {"a": ["0", "2", "3"], "b": ["4", "5", "6"]}
    dataset.dev_queries = {"c": [4, 5]}
    dataset.dev_qrels = {"c": ["7"]}
    dataset._save_cache()
    loaded = MSMARCODataset(None, max_len=32, n_hard_negatives=2, cache_dir=tmp_path)
    loaded.load()
    assert "6" not in loaded.passages
    random.seed(0)
    batch = loaded.sample_batch(2)
    assert batch["neg_ids"].shape[:2] == (2, 2)
    for p, n in zip(batch["pos_ids"], batch["neg_ids"]):
        assert all(neg[0] != p[0] for neg in n)


def test_training_selects_on_source_validation_and_saves_last_step(tmp_path, monkeypatch):
    from train_contrastive import train
    monkeypatch.setenv("PRISM_WANDB", "0")
    (tmp_path / "eval").mkdir()
    (tmp_path / "checkpoints").mkdir()
    model = tiny_prism()
    batches = make_batches()
    class Dataset:
        def sample_batch(self, size):
            return batches[1]
    def evaluate(model, step):
        return {"msmarco_dev_mrr@10": 1.0 / step, "locov1_avg_ndcg@10": float(step)}
    result = train(model, Dataset(), tmp_path, {}, n_steps=3, micro_batch=2, grad_accum=2,
                   eval_every=1, checkpoint_every=10, log_every=1, grad_cache=True,
                   eval_fn=evaluate, device="cpu", eval_early_stop_patience=0)
    assert result["best_step"] == 1
    assert (tmp_path / "checkpoints" / "step_3.pt").exists()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["training"]["contrastive_batch"] == 4


def test_binding_probe_removes_bag_of_words_shortcut():
    from research_probe import sample_binding_batch
    q, p, n = sample_binding_batch(64, 8, torch.Generator().manual_seed(4))
    torch.testing.assert_close(p.sort(1).values, n.sort(1).values)
    assert ((p != n).sum(1) == 2).all()
    assert ((q[:, 0] >= 33) & (q[:, 0] < 65)).all()


def test_sinusoidal_transformer_supports_longer_positions():
    model = TransformerForEmbedding(TransformerEncoder(128, d=32, d_e=24, n_layers=1,
                                     n_heads=4, max_len=8, dropout=0,
                                     position_encoding="sinusoidal")).eval()
    with torch.no_grad():
        result = model.encode(torch.ones(2, 33, dtype=torch.long))
    assert torch.isfinite(result).all()


def test_checkpoint_loading_preserves_position_table_and_fails_on_mismatch(tmp_path, monkeypatch):
    from experiment_protocol import load_encoder_for_eval
    import paper_exp1_controlled as controlled
    monkeypatch.setitem(controlled.MODEL_BUILDERS, "tiny", ("Tiny", lambda ml: tiny_prism(max_len=ml)))
    model = tiny_prism().eval()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = load_encoder_for_eval("tiny", path, 128)
    ids = torch.randint(1, 128, (2, 13))
    with torch.no_grad():
        torch.testing.assert_close(model.encode(ids), restored.encode(ids))
    with pytest.raises(ValueError, match="position-table"):
        load_encoder_for_eval("tiny", path, 257)
    state = model.state_dict()
    del state["encoder.token_emb.weight"]
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder_for_eval("tiny", path, 128)


def test_only_out_of_memory_is_classified_as_oom():
    from paper_exp2_efficiency import _is_oom
    assert _is_oom(RuntimeError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("CUDA error: illegal memory access"))
    assert not _is_oom(RuntimeError("CUBLAS_STATUS_INVALID_VALUE"))


def test_fixed_decay_scan_rejects_learned_parameters():
    from prism_scan_kernel import fused_decay_scan
    with pytest.raises(ValueError, match="fixed decays"):
        fused_decay_scan(torch.randn(1, 4, 2, 3), torch.tensor([0.5, 0.9], requires_grad=True))
