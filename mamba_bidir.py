"""
Bidirectional Mamba encoder for embedding tasks.

Architecture:
    Embedding -> [MambaBidirLayer x N] -> Mean Pooling -> LayerNorm -> L2 Norm

Each layer runs a forward and backward Mamba selective SSM, then fuses
with a learned gated residual — the same fusion pattern as PRISM.

Requires: mamba_ssm (pip install mamba-ssm). Falls back to a pure-PyTorch
diagonal SSM if mamba_ssm is not available (functional but slower).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try to import Mamba; provide fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SimpleDiagSSM(nn.Module):
    """Fallback: simple diagonal SSM when mamba_ssm is not installed.

    Not a faithful Mamba reimplementation — just a gated linear recurrence
    with input-dependent gating, enough to test the bidirectional wrapper.
    For real experiments, install mamba_ssm.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, 2 * d_inner)  # z and x
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        # Fixed decay (simplified — real Mamba uses input-dependent dt)
        self.register_buffer("decay", torch.ones(d_inner) * 0.99)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv = self.conv(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Simple recurrence (not selective — just fixed decay scan)
        dt = F.softplus(self.dt_proj(x_conv))
        # h_t = decay * h_{t-1} + dt * x
        vals = dt * x_conv
        hiddens = []
        h = torch.zeros(B, self.d_inner, device=x.device)
        for t in range(T):
            h = self.decay * h + vals[:, t]
            hiddens.append(h)
        y = torch.stack(hiddens, dim=1)

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBidirLayer(nn.Module):
    """Single bidirectional Mamba layer with gated fusion."""

    def __init__(
        self,
        d: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # Pre-norm
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # Forward and backward Mamba blocks
        MambaClass = Mamba if MAMBA_AVAILABLE else SimpleDiagSSM
        self.mamba_fwd = MambaClass(
            d_model=d, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.mamba_bwd = MambaClass(
            d_model=d, d_state=d_state, d_conv=d_conv, expand=expand,
        )

        # Gated fusion (same pattern as PRISM's bidirectional fusion)
        self.fusion_gate = nn.Linear(2 * d, d)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)  # beta=0.5 at init

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
            mask: (B, T) — not directly used by Mamba (it's causal),
                  but we zero out padded positions before/after.
        Returns:
            (B, T, d)
        """
        # --- Mamba sublayer ---
        residual = x
        h = self.norm1(x)

        if mask is not None:
            h = h * mask.unsqueeze(-1).float()

        # Forward pass
        h_fwd = self.mamba_fwd(h)

        # Backward pass: reverse, run forward, reverse back
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)

        # Gated fusion
        beta = torch.sigmoid(self.fusion_gate(torch.cat([h_fwd, h_bwd], dim=-1)))
        fused = beta * h_fwd + (1 - beta) * h_bwd

        if mask is not None:
            fused = fused * mask.unsqueeze(-1).float()

        x = residual + self.dropout(fused)

        # --- FFN sublayer ---
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class MambaBidirEncoder(nn.Module):
    """Stack of bidirectional Mamba layers with embedding and mean pooling."""

    def __init__(
        self,
        vocab_size: int = 30522,
        d: int = 384,
        d_e: int = 384,
        n_layers: int = 6,
        max_len: int = 8192,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
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
            MambaBidirLayer(
                d=d, d_state=d_state, d_conv=d_conv,
                expand=expand, dropout=dropout,
            )
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
            if isinstance(module, nn.Linear):
                if module.bias is not None and module not in (
                    layer.fusion_gate for layer in self.layers
                ):
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


class MambaBidirForEmbedding(nn.Module):
    """Contrastive wrapper matching PRISMForEmbedding interface."""

    def __init__(self, encoder: MambaBidirEncoder, temperature: float = 0.05):
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


def build_mamba_bidir_small(
    vocab_size: int = 30522,
    max_len: int = 8192,
    **kwargs,
) -> MambaBidirForEmbedding:
    """Build a ~20M parameter bidirectional Mamba model.

    With d=384, expand=2, d_state=16, d_conv=4:
      - Each MambaBidirLayer has ~1.8M params (two Mamba blocks + fusion + FFN)
      - 8 layers ≈ 14.4M + embeddings ≈ 17M + overhead ≈ ~20M total
    Adjust n_layers to match target param count.
    """
    if not MAMBA_AVAILABLE:
        print("WARNING: mamba_ssm not installed. Using SimpleDiagSSM fallback. "
              "Install mamba-ssm for real Mamba experiments.")
    # expand=2 with 2 layers gives ~7.9M non-embedding params,
    # close to PRISM (~6.7M) and Transformer (~7.3M).
    # Fewer but wider layers — standard Mamba design.
    encoder = MambaBidirEncoder(
        vocab_size=vocab_size,
        d=384,
        d_e=384,
        n_layers=2,
        max_len=max_len,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        **kwargs,
    )
    return MambaBidirForEmbedding(encoder)
