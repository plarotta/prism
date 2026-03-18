"""
Transformer Baseline Encoder for fair comparison with PRISM.

Standard bidirectional Transformer encoder with self-attention,
matching PRISM's interface (vocab → embedding vector).
Parameter counts are matched to PRISM configurations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm Transformer encoder layer."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        mlp_dim = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Full Transformer encoder: token embedding → L layers → pooling → embedding.

    Matches PRISM's interface for fair comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 768,
        d_e: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_len: int = 8192,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d)

        # Attentive pooling (single learned query, same as CLS-style but differentiable)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_proj = nn.Linear(d, d_e)
        self.pool_norm = nn.LayerNorm(d_e)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        B, T = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        # MultiheadAttention expects True = ignore
        key_padding_mask = ~attention_mask.bool()

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Attentive pooling
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d)
        scores = torch.matmul(query, x.transpose(-1, -2)) / math.sqrt(self.d)  # (B, 1, T)
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B, 1, T)
        pooled = torch.matmul(attn, x).squeeze(1)  # (B, d)

        embedding = self.pool_norm(self.pool_proj(pooled))

        return {
            "embedding": embedding,
            "token_embeddings": x,
        }


class TransformerForEmbedding(nn.Module):
    """Wrapper with InfoNCE loss, matching PRISMForEmbedding interface."""

    def __init__(self, encoder: TransformerEncoder, temperature: float = 0.05):
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

        if neg_ids is not None:
            B_n, N, T_n = neg_ids.shape
            n_emb = self.encode(neg_ids.reshape(B_n * N, T_n), neg_mask.reshape(B_n * N, T_n))
            n_emb = n_emb.reshape(B, N, -1)
            hard_sim = torch.einsum("bd, bnd -> bn", q_emb, n_emb) / self.temperature
            sim = torch.cat([sim, hard_sim], dim=-1)

        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        with torch.no_grad():
            accuracy = (sim.argmax(dim=-1) == labels).float().mean()

        return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Configuration presets (parameter-matched to PRISM)
# ---------------------------------------------------------------------------

def transformer_small(vocab_size: int = 32000, **kwargs) -> TransformerEncoder:
    """~23M params — matched to prism_small."""
    defaults = dict(d=384, d_e=384, n_layers=6, n_heads=6,
                    max_len=8192, mlp_ratio=2.0, dropout=0.1)
    defaults.update(kwargs)
    return TransformerEncoder(vocab_size=vocab_size, **defaults)


def transformer_base(vocab_size: int = 32000, **kwargs) -> TransformerEncoder:
    """~110M params — matched to prism_base / BERT-base."""
    defaults = dict(d=768, d_e=768, n_layers=12, n_heads=12,
                    max_len=8192, mlp_ratio=4.0, dropout=0.1)
    defaults.update(kwargs)
    return TransformerEncoder(vocab_size=vocab_size, **defaults)


if __name__ == "__main__":
    print("Transformer Baseline — Smoke Test")
    model = transformer_small(vocab_size=32000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B, T = 4, 256
    ids = torch.randint(1, 32000, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[:, -20:] = 0

    with torch.no_grad():
        out = model(ids, mask)
    print(f"Embedding shape: {tuple(out['embedding'].shape)}")
    print("OK")
