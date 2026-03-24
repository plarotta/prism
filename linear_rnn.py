"""
Linear-RNN: single-channel bidirectional gated linear recurrence.

Ablation baseline for PRISM — tests whether the multi-channel structure
is necessary, or if a single wide recurrence suffices.

Architecture:
    Embedding -> [LinearRNNLayer x N] -> Mean Pooling -> LayerNorm -> L2 Norm

Each layer: bidirectional gated linear scan (decay=0.99) + FFN + residual.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prism import _fast_fixed_decay_scan


class LinearRNNLayer(nn.Module):
    """Single-channel bidirectional gated linear recurrence + FFN."""

    def __init__(self, d: int, decay: float = 0.99, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.decay = decay

        # Pre-norm
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # Input gate (forward)
        self.gate_fwd = nn.Linear(d, d)
        nn.init.zeros_(self.gate_fwd.weight)
        nn.init.constant_(self.gate_fwd.bias, 1.0)

        # Input gate (backward)
        self.gate_bwd = nn.Linear(d, d)
        nn.init.zeros_(self.gate_bwd.weight)
        nn.init.constant_(self.gate_bwd.bias, 1.0)

        # Bidirectional fusion gate
        self.fusion_gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, T, d)
            mask: (B, T) optional
        Returns:
            (B, T, d)
        """
        # --- Recurrence sublayer ---
        residual = x
        h = self.norm1(x)

        # Forward scan
        g_fwd = torch.sigmoid(self.gate_fwd(h))
        v_fwd = g_fwd * h
        if mask is not None:
            v_fwd = v_fwd * mask.unsqueeze(-1)
        h_fwd = _fast_fixed_decay_scan(self.decay, v_fwd)

        # Backward scan
        g_bwd = torch.sigmoid(self.gate_bwd(h))
        v_bwd = g_bwd * h
        if mask is not None:
            v_bwd = v_bwd * mask.unsqueeze(-1)
        h_bwd = _fast_fixed_decay_scan(self.decay, v_bwd.flip(1)).flip(1)

        # Gated fusion
        beta = torch.sigmoid(self.fusion_gate(torch.cat([h_fwd, h_bwd], dim=-1)))
        fused = beta * h_fwd + (1 - beta) * h_bwd

        x = residual + self.dropout(fused)

        # --- FFN sublayer ---
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class LinearRNNEncoder(nn.Module):
    """Stack of LinearRNNLayers with embedding and mean pooling."""

    def __init__(
        self,
        vocab_size: int = 30522,
        d: int = 384,
        d_e: int = 384,
        n_layers: int = 8,
        max_len: int = 8192,
        decay: float = 0.99,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            LinearRNNLayer(d, decay=decay, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

        # Mean pooling + projection
        self.pool_proj = nn.Linear(d, d_e)
        self.pool_norm = nn.LayerNorm(d_e)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in (
                layer.gate_fwd for layer in self.layers
            ) and module not in (
                layer.gate_bwd for layer in self.layers
            ) and module not in (
                layer.fusion_gate for layer in self.layers
            ):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> dict:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.final_norm(x)

        # Mean pooling (masked)
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        embedding = self.pool_norm(self.pool_proj(pooled))
        return {"embedding": embedding, "token_states": x}


class LinearRNNForEmbedding(nn.Module):
    """Contrastive wrapper matching PRISMForEmbedding interface."""

    def __init__(self, encoder: LinearRNNEncoder, temperature: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        out = self.encoder(input_ids, attention_mask)
        return F.normalize(out["embedding"], p=2, dim=-1)

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        pos_ids: Tensor,
        pos_mask: Tensor,
        neg_ids: Optional[Tensor] = None,
        neg_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        q_emb = self.encode(query_ids, query_mask)
        p_emb = self.encode(pos_ids, pos_mask)
        B = q_emb.shape[0]

        sim = torch.matmul(q_emb, p_emb.T) / self.temperature
        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        acc = (sim.argmax(dim=-1) == labels).float().mean()

        return {"loss": loss, "accuracy": acc}


def build_linear_rnn_small(
    vocab_size: int = 30522,
    max_len: int = 8192,
    **kwargs,
) -> LinearRNNForEmbedding:
    """Build a ~19M parameter Linear-RNN model."""
    encoder = LinearRNNEncoder(
        vocab_size=vocab_size,
        d=384,
        d_e=384,
        n_layers=6,
        max_len=max_len,
        decay=0.99,
        dropout=0.1,
        **kwargs,
    )
    return LinearRNNForEmbedding(encoder)
