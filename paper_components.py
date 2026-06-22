"""
Architecture components used by the paper experiment suite.

These three nn.Modules define the PRISM-Simplified configuration that survived
the Phase 1-6 ablations (see RESEARCH_SUMMARY.md):

    - NoInterference        — drop cross-scale interference (found inert)
    - MeanPooling           — replace covariance pooling (found harmful)
    - LearnedDecayRecurrence — learned (vs fixed) decay rates, for ablations

Extracted from the legacy benchmark_ablations.py so the paper runners have no
dependency on the archived Phase 1-6 code.
"""

import torch
import torch.nn as nn

from prism import StratifiedRecurrence, _fast_fixed_decay_scan


class NoInterference(nn.Module):
    """Pass-through: no cross-scale interaction."""

    def __init__(self, d_c: int, n_channels: int):
        super().__init__()

    def forward(self, hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        return hiddens


class MeanPooling(nn.Module):
    """Standard mean pooling over valid positions."""

    def __init__(self, d: int, d_e: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, f, query_state, mask=None):
        if mask is not None:
            f = f * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            pooled = f.sum(dim=1) / n_valid
        else:
            pooled = f.mean(dim=1)
        return self.norm(self.proj(pooled))


class LearnedDecayRecurrence(StratifiedRecurrence):
    """Same as StratifiedRecurrence but decay rates are learned parameters."""

    def __init__(self, d_c, n_channels, max_len=8192, bidirectional=True):
        super().__init__(d_c, n_channels, max_len, bidirectional)
        # Override: make lambdas a learned parameter instead of a buffer,
        # initialized at the same geometric values.
        init_lambdas = self.lambdas.clone()
        self.lambdas = None  # remove buffer
        # Store as logit for unconstrained optimization; sigmoid maps to (0,1).
        self.lambda_logits = nn.Parameter(
            torch.log(init_lambdas / (1.0 - init_lambdas + 1e-8))
        )

    @property
    def _lambdas(self):
        return torch.sigmoid(self.lambda_logits)

    def _run_direction(self, channels, gates):
        hiddens = []
        lambdas = self._lambdas
        for c, (z_c, gate_c) in enumerate(zip(channels, gates)):
            g_t = torch.sigmoid(gate_c(z_c))
            gated_input = g_t * z_c
            lam = lambdas[c].item()
            h_c = _fast_fixed_decay_scan(lam, gated_input)
            hiddens.append(h_c)
        return hiddens
