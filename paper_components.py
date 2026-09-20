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

from prism import StratifiedRecurrence
from prism_scan_kernel import _ref_forward


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
        init_lambdas = self.lambdas.clone().clamp(1e-4, 1 - 1e-4)
        self.lambdas = None  # remove buffer
        # Store as logit for unconstrained optimization; sigmoid maps to (0,1).
        self.lambda_logits = nn.Parameter(torch.logit(init_lambdas))

    @property
    def _lambdas(self):
        return torch.sigmoid(self.lambda_logits)

    def _run_direction(self, channels, gates):
        lambdas = self._lambdas
        gated = []
        for z_c, gate_c in zip(channels, gates):
            g_t = torch.sigmoid(gate_c(z_c))
            gated.append(g_t * z_c)
        # Tensor-valued decays preserve d(loss)/d(lambda). The fixed-decay
        # custom backward deliberately cannot be used for this ablation.
        h = _ref_forward(torch.stack(gated, dim=2), lambdas)
        return list(h.unbind(2))
