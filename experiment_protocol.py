"""Shared source-validation / held-out-transfer protocol for paper runs."""
import random
import json
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    """Call before model construction as well as at the start of training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_validation_callback(dataset, max_len, device, batch_size=32):
    def evaluate(model, step):
        from data.msmarco import evaluate_msmarco_dev
        return evaluate_msmarco_dev(model, dataset, max_len=max_len,
                                    batch_size=batch_size, device=device)
    return evaluate


def evaluate_selected_locov1(model, tokenizer, run_dir, training_result, *, max_len,
                             device, batch_size=16):
    """Evaluate the already selected checkpoint once; never select on LoCoV1."""
    from data.loco_eval import evaluate_locov1
    from paper_log import load_checkpoint, save_eval_results, save_final_metrics
    step = training_result["best_step"] or training_result["n_steps_completed"]
    load_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model)
    model.to(device).eval()
    results = evaluate_locov1(model, tokenizer, max_len=max_len,
                              batch_size=batch_size, device=device)
    results["checkpoint_selection"] = training_result["selection_metric"] \
        if training_result["best_step"] else "final_step"
    save_eval_results(run_dir, step, "locov1_heldout", results)
    training_result["heldout_locov1"] = results
    save_final_metrics(run_dir, training_result)
    return results


def load_encoder_for_eval(model_key, checkpoint_path, max_eval_len, device="cpu"):
    """Reconstruct a controlled-run encoder without inventing position weights.

    A checkpoint with an absolute-position table may be evaluated at shorter
    lengths using its original table. Exceeding its capacity is an error; use
    the position-free/sinusoidal variants for a length-extrapolation experiment.
    """
    from paper_exp1_controlled import MODEL_BUILDERS
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("model_state_dict", checkpoint)
    state = {key.removeprefix("_orig_mod."): value for key, value in raw.items()}
    position_weights = state.get("encoder.pos_emb.weight")
    model_max_len = position_weights.shape[0] if position_weights is not None else max_eval_len
    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get("model_key", model_key) != model_key:
            raise ValueError("checkpoint model_key does not match requested model")
        model_max_len = config.get("model_config", {}).get("max_len", model_max_len)
    if position_weights is not None and max_eval_len > position_weights.shape[0]:
        raise ValueError("Evaluation exceeds checkpoint position-table capacity; "
                         "use a length-capable variant or a checkpoint trained for this length")
    model = MODEL_BUILDERS[model_key][1](model_max_len)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
