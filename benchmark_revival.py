"""Reproduce the revival audit and measure gradient-cache saved-tensor memory.

The baseline is loaded from a pinned LOCAL git revision, without a network call.
CPU memory figures count distinct live autograd-saved storage excluding model
parameters/buffers; they are not process RSS or CUDA allocator peak memory.
"""
import argparse
import copy
import gc
import json
import linecache
import platform
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

import prism
from contrastive_batch import cached_contrastive_backward, contrastive_loss
from paper_components import LearnedDecayRecurrence, MeanPooling, NoInterference


def module_from_git(revision, path, module_name):
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    module = types.ModuleType(module_name)
    filename = f"{revision}/{path}"
    module.__file__ = filename
    sys.modules[module_name] = module
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def build(module):
    encoder = module.PRISMEncoder(128, d=32, d_e=32, n_layers=2, n_channels=4,
                                  max_len=256, dropout=0, cov_rank=2)
    encoder.pooling = MeanPooling(32, 32)
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(8, 4)
        layer.interference_bwd = NoInterference(8, 4)
        layer.recurrence.lambdas.fill_(0.99)
    return module.PRISMForEmbedding(encoder)


def padding_audit(legacy):
    torch.manual_seed(7)
    before = build(legacy).eval()
    # Nonzero learned normalization offsets expose padding injection. This is
    # a controlled regression fixture, not a claim about a trained checkpoint.
    with torch.no_grad():
        for layer in before.encoder.layers:
            layer.norm.bias.normal_(0, 0.1)
    after = build(prism).eval()
    after.load_state_dict(before.state_dict())
    ids = torch.randint(1, 128, (2, 13))
    padded = F.pad(ids, (0, 100))
    result = {"fixture": "identical weights; LayerNorm offsets sampled at std=0.1; 13 valid + 100 pad tokens"}
    with torch.no_grad():
        for name, model in [("before", before), ("after", after)]:
            a, b = model.encode(ids), model.encode(padded)
            result[name] = {"min_cosine": F.cosine_similarity(a, b).min().item(),
                            "max_abs_embedding_drift": (a - b).abs().max().item()}
    return result


def decay_audit(legacy, revision):
    previous = sys.modules["prism"]
    try:
        sys.modules["prism"] = legacy
        components = module_from_git(revision, "paper_components.py", "prism_legacy_components")
    finally:
        sys.modules["prism"] = previous
    result = {}
    for name, cls in [("before", components.LearnedDecayRecurrence), ("after", LearnedDecayRecurrence)]:
        torch.manual_seed(17)
        rec = cls(4, 3, max_len=128)
        channels = [torch.randn(2, 9, 4, requires_grad=True) for _ in range(3)]
        fwd, bwd = rec(channels)
        sum(x.square().mean() for x in fwd + bwd).backward()
        grad = rec.lambda_logits.grad
        result[name] = {"all_logits_finite": bool(torch.isfinite(rec.lambda_logits).all()),
                        "decay_gradient": grad.tolist() if grad is not None else None}
    return result


class SavedTensorMeter:
    def __init__(self, model):
        self.excluded = {p.untyped_storage().data_ptr() for p in list(model.parameters()) + list(model.buffers())}
        self.refs = {}
        self.live = 0
        self.peak = 0

    def pack(self, tensor):
        meter = self
        pointer = tensor.untyped_storage().data_ptr()
        size = tensor.untyped_storage().nbytes()
        tracked = pointer not in self.excluded
        if tracked:
            if pointer not in self.refs:
                self.live += size
                self.peak = max(self.peak, self.live)
                self.refs[pointer] = 0
            self.refs[pointer] += 1
        class Saved:
            def __init__(self):
                self.value = tensor
            def __del__(self):
                if tracked:
                    meter.refs[pointer] -= 1
                    if meter.refs[pointer] == 0:
                        meter.live -= size
                        del meter.refs[pointer]
        return Saved()

    @staticmethod
    def unpack(saved):
        return saved.value


def cache_audit():
    torch.manual_seed(29)
    full = build(prism).train()
    cached = copy.deepcopy(full)
    batches = []
    for _ in range(8):
        b = {}
        for role, length in [("query", 32), ("pos", 128)]:
            b[f"{role}_ids"] = torch.randint(1, 128, (4, length))
            b[f"{role}_mask"] = torch.ones(4, length, dtype=torch.long)
        batches.append(b)
    peak, losses = {}, {}
    for name, model in [("full_graph", full), ("cached", cached)]:
        meter = SavedTensorMeter(model)
        with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
            if name == "cached":
                loss = cached_contrastive_backward(model, batches)
            else:
                qe, pe = [], []
                for b in batches:
                    qe.append(model.encode(b["query_ids"], b["query_mask"]))
                    pe.append(model.encode(b["pos_ids"], b["pos_mask"]))
                loss = contrastive_loss(torch.cat(qe), torch.cat(pe), model.temperature)
                loss.backward()
            losses[name] = loss.item()
        gc.collect()
        peak[name] = meter.peak
    max_grad_error = max((p.grad - q.grad).abs().max().item()
                         for p, q in zip(full.parameters(), cached.parameters()) if p.grad is not None)
    return {"logical_batch": 32, "micro_batch": 4, "microbatches": 8, "losses": losses,
            "max_abs_gradient_error": max_grad_error,
            "peak_live_autograd_saved_bytes_excluding_parameters": peak,
            "saved_tensor_reduction_factor": peak["full_graph"] / peak["cached"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="3efe85c")
    parser.add_argument("--out", type=Path, default=Path("results/revival/verification.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    legacy = module_from_git(args.baseline_ref, "prism.py", "prism_legacy")
    results = {"baseline_ref": args.baseline_ref, "torch": torch.__version__,
               "python": platform.python_version(), "device": "cpu", "threads": 2,
               "padding": padding_audit(legacy), "learned_decay": decay_audit(legacy, args.baseline_ref),
               "gradient_cache": cache_audit(),
               "gpu_results": "not run; no CUDA device in this workspace"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
