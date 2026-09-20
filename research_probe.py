"""Order-sensitive retrieval probe with identical positive/negative token bags.

Each document contains two key/value facts. The hard negative swaps the values,
preserving every token count. A bag-of-words encoder must tie (50% pair accuracy).
Train on gaps 1..8; report fresh examples at gaps 8, 32, and 128 without tuning
on them. This is a synthetic mechanism diagnostic, NOT a natural-language result.

Example: python research_probe.py --steps 400 --seeds 0,1,2 --device cpu
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from baseline_transformer import TransformerEncoder, TransformerForEmbedding
from contrastive_batch import contrastive_loss
from experiment_protocol import seed_everything
from paper_components import MeanPooling, NoInterference
from prism import PRISMEncoder, PRISMForEmbedding


def sample_binding_batch(batch_size, gap, generator):
    keys = torch.randint(33, 65, (batch_size, 2), generator=generator)
    values = torch.randint(65, 97, (batch_size, 2), generator=generator)
    # Distinct keys and values make the swapped document unambiguously wrong.
    keys[:, 1] = 33 + (keys[:, 0] - 33 + torch.randint(1, 32, (batch_size,), generator=generator)) % 32
    values[:, 1] = 65 + (values[:, 0] - 65 + torch.randint(1, 32, (batch_size,), generator=generator)) % 32
    ids = torch.randint(1, 33, (batch_size, 4 + 3 * gap), generator=generator)
    positions = [0, gap + 1, 2 * (gap + 1), 3 * (gap + 1)]
    ids[:, positions] = torch.stack((keys[:, 0], values[:, 0], keys[:, 1], values[:, 1]), 1)
    negatives = ids.clone()
    negatives[:, [positions[1], positions[3]]] = values.flip(1)
    target = torch.randint(0, 2, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    queries = torch.stack((keys[rows, target], values[rows, target]), 1)
    return queries, ids, negatives


def build_probe_model(name):
    if name == "transformer_sinusoidal":
        encoder = TransformerEncoder(128, d=48, d_e=48, n_layers=2, n_heads=4,
                                     max_len=1024, mlp_ratio=2, dropout=0.1,
                                     position_encoding="sinusoidal")
        return TransformerForEmbedding(encoder)
    encoder = PRISMEncoder(128, d=48, d_e=48, n_layers=2, n_channels=4,
                           max_len=1024, dropout=0.1, cov_rank=2,
                           position_encoding="none" if name == "prism_no_pos" else "learned")
    for layer in encoder.layers:
        layer.interference_fwd = NoInterference(12, 4)
        layer.interference_bwd = NoInterference(12, 4)
        layer.recurrence.lambdas.fill_(0.99)
    encoder.pooling = MeanPooling(48, 48)
    return PRISMForEmbedding(encoder)


@torch.no_grad()
def evaluate(model, device, n_examples=512):
    model.eval()
    results = {}
    for gap in [8, 32, 128]:
        generator = torch.Generator().manual_seed(9000 + gap)
        wins, ties, total_margin = 0, 0, 0.0
        for start in range(0, n_examples, 32):
            q, p, n = sample_binding_batch(min(32, n_examples-start), gap, generator)
            q, p, n = q.to(device), p.to(device), n.to(device)
            qe, pe, ne = [model.encode(x, torch.ones_like(x)) for x in (q, p, n)]
            margins = (qe * (pe - ne)).sum(-1)
            wins += int((margins > 1e-6).sum())
            ties += int((margins.abs() <= 1e-6).sum())
            total_margin += margins.sum().item()
        results[str(gap)] = {"seq_len": 4 + 3 * gap, "pair_accuracy": (wins + 0.5 * ties) / n_examples,
                             "tie_fraction": ties / n_examples, "mean_cosine_margin": total_margin / n_examples}
    return results


def run(name, seed, steps, device, pair_weight=0.0):
    seed_everything(seed)
    model = build_probe_model(name).to(device)
    generator = torch.Generator().manual_seed(1000 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    losses = []
    for step in range(steps):
        model.train()
        gap = int(torch.randint(1, 9, (), generator=generator))
        q, p, n = [x.to(device) for x in sample_binding_batch(32, gap, generator)]
        optimizer.zero_grad(set_to_none=True)
        qe, pe, ne = [model.encode(x, torch.ones_like(x)) for x in (q, p, n)]
        groups = q[:, 0] * 128 + q[:, 1]
        loss = contrastive_loss(qe, pe, model.temperature, ne[:, None], query_groups=groups)
        if pair_weight:
            margin = (qe * (pe - ne)).sum(-1) / model.temperature
            loss = loss + pair_weight * F.softplus(-margin).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"{name} seed={seed} step={step+1} loss={sum(losses[-50:])/len(losses[-50:]):.4f}", flush=True)
    return {"model": name, "seed": seed, "steps": steps,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": time.perf_counter()-started,
            "final_50_step_loss": sum(losses[-50:])/len(losses[-50:]),
            "evaluation": evaluate(model, device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--models", default="prism,prism_no_pos,transformer_sinusoidal")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pair-weight", type=float, default=0.0,
                        help="Auxiliary pairwise loss on the token-identical hard negative")
    parser.add_argument("--out", type=Path, default=Path("results/revival/binding_probe.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    models = args.models.split(",")
    if any(name not in {"prism", "prism_no_pos", "transformer_sinusoidal"} for name in models):
        parser.error("unknown model name")
    torch.set_num_threads(args.threads)
    results = {"description": "Synthetic binding diagnostic; identical positive/negative token bags",
               "torch": torch.__version__, "device": args.device, "threads": args.threads,
               "training_gap_range": [1, 8], "learning_rate": 0.001, "batch_size": 32,
               "pair_weight": args.pair_weight,
               "bag_of_words_pair_accuracy": 0.5, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in map(int, args.seeds.split(",")):
        for name in models:
            result = run(name, seed, args.steps, args.device, args.pair_weight)
            results["runs"].append(result)
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
