"""Full-batch InfoNCE with encoder activation memory bounded by a microbatch.

Cache embeddings without a graph, differentiate the shared contrastive loss,
then replay each encoder chunk with its exact RNG state and embedding gradient.
This is gradient caching (Gao et al., 2021), not ordinary gradient accumulation.
The score matrix still uses O(batch_size ** 2) memory. Encoders must not mutate
training buffers on forward (e.g. BatchNorm running statistics).
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def contrastive_loss(queries, positives, temperature=0.05, negatives=None,
                     positive_groups=None, query_groups=None):
    """In-batch positives plus optional per-query hard negatives.

Repeated queries or positive documents are excluded as false negatives. The
diagonal remains the designated positive. Groups are stable integer IDs.
"""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = queries.float() @ positives.float().T / temperature
    batch_size = queries.shape[0]
    if positive_groups is not None or query_groups is not None:
        duplicate = torch.zeros_like(scores, dtype=torch.bool)
        for groups in (positive_groups, query_groups):
            if groups is not None:
                duplicate |= groups[:, None] == groups[None, :]
        duplicate.fill_diagonal_(False)
        scores = scores.masked_fill(duplicate, -torch.inf)
    if negatives is not None:
        hard = torch.einsum("bd,bnd->bn", queries.float(), negatives.float())
        scores = torch.cat((scores, hard / temperature), dim=1)
    labels = torch.arange(batch_size, device=scores.device)
    return F.cross_entropy(scores, labels)


def _capture_rng(device):
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda_state


@contextmanager
def _replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


def cached_contrastive_backward(model, batches, *, use_amp=False):
    """Accumulate exact full-batch parameter gradients; return detached loss.

    Call optimizer.zero_grad() before this and optimizer.step() afterward.
    CPU batches are retained, but each chunk is moved to the model's device
    only while encoding/replaying it. Dropout is replayed exactly. Supports
    CPU and CUDA; bf16 autocast is enabled only for CUDA when requested.
    """
    if not batches:
        raise ValueError("at least one microbatch is required")
    device = next(model.parameters()).device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("gradient caching supports CPU and CUDA RNG replay")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("gradient caching requires encoders without BatchNorm state")
    use_amp = use_amp and device.type == "cuda"
    records = []
    embeddings = {"query": [], "pos": [], "neg": []}
    negative_count = batches[0].get("neg_ids")
    negative_count = negative_count.shape[1] if negative_count is not None else 0

    for batch in batches:
        this_count = batch["neg_ids"].shape[1] if "neg_ids" in batch else 0
        if this_count != negative_count:
            raise ValueError("all microbatches must use the same hard-negative count")
        chunk_size = batch["query_ids"].shape[0]
        for role in ("query", "pos", "neg"):
            if role == "neg" and not negative_count:
                continue
            ids, mask = batch[f"{role}_ids"], batch[f"{role}_mask"]
            ids, mask = ids.reshape(-1, ids.shape[-1]), mask.reshape(-1, mask.shape[-1])
            for start in range(0, ids.shape[0], chunk_size):
                chunk_ids, chunk_mask = ids[start:start + chunk_size], mask[start:start + chunk_size]
                state = _capture_rng(device)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    emb = model.encode(chunk_ids.to(device), chunk_mask.to(device))
                leaf = emb.detach().float().requires_grad_()
                embeddings[role].append(leaf)
                records.append((chunk_ids, chunk_mask, state, leaf))

    q = torch.cat(embeddings["query"])
    p = torch.cat(embeddings["pos"])
    n = torch.cat(embeddings["neg"]).reshape(q.shape[0], negative_count, -1) \
        if negative_count else None
    groups = {}
    for name in ("positive_groups", "query_groups"):
        present = [name in batch for batch in batches]
        if any(present) and not all(present):
            raise ValueError(f"{name} must be present in every microbatch")
        groups[name] = torch.cat([batch[name] for batch in batches]).to(device) if all(present) else None
    loss = contrastive_loss(q, p, model.temperature, n, **groups)
    loss.backward()

    for ids, mask, state, leaf in records:
        with _replay_rng(state, device), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            replay = model.encode(ids.to(device), mask.to(device))
        replay.backward(leaf.grad.to(replay.dtype))
    return loss.detach()
