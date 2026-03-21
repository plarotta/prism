"""
HybridPRISM: Memory-Augmented Sub-Quadratic Sequence Encoder

Interleaves PRISM recurrence blocks with a fixed-size memory bank
accessed via cross-attention. Total complexity: O(n*d^2 + n*k*d),
linear in sequence length n.

Architecture:
    [PRISM Group 1] -> [MemWrite + MemRead] -> [PRISM Group 2] -> ... -> [Pooling]

All PRISM blocks use the validated all-slow config (lambda=0.99, no interference,
bidirectional gated fusion). The memory bank is the only new component.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prism import PRISMLayer
from benchmark_ablations import MeanPooling, NoInterference


# ---------------------------------------------------------------------------
# Memory Components
# ---------------------------------------------------------------------------

class MemoryWrite(nn.Module):
    """Cross-attention: memory slots attend to token states.

    Each slot queries the full token sequence and accumulates a weighted summary.
    A gated residual update lets memory retain old content across multiple writes.
    No LayerNorm -- avoids normalizing shared memory to unit variance,
    which would cause embedding collapse (see Issue 1/2 in HYBRID_V2_LOG.md).

    Complexity: O(n * k * d) -- linear in n for fixed k.
    """

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        # Gated residual update
        self.gate_proj = nn.Linear(2 * d, d)
        # Bias = +2.0 so sigmoid(2)=0.88: memory defaults to retaining old content
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        memory: Tensor,
        token_states: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            memory:       (B, k, d) -- current memory state
            token_states: (B, n, d) -- PRISM layer outputs
            mask:         (B, n)    -- True for valid tokens

        Returns:
            updated memory: (B, k, d)
        """
        B, k, d = memory.shape
        n = token_states.shape[1]
        H, dh = self.n_heads, self.d_head

        Q = self.q_proj(memory).view(B, k, H, dh).transpose(1, 2)
        K = self.k_proj(token_states).view(B, n, H, dh).transpose(1, 2)
        V = self.v_proj(token_states).view(B, n, H, dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        retrieved = torch.matmul(attn, V)
        retrieved = retrieved.transpose(1, 2).reshape(B, k, d)

        # Gated update: memory can retain old content or overwrite
        gate = torch.sigmoid(self.gate_proj(torch.cat([memory, retrieved], dim=-1)))
        updated = gate * memory + (1.0 - gate) * retrieved

        return updated


class MemoryRead(nn.Module):
    """Cross-attention: tokens attend to memory slots.

    Each token queries the memory bank and retrieves relevant stored information.
    Uses a pure residual with ReZero-initialized output projection so the
    residual stream passes through unchanged at initialization.
    No LayerNorm -- avoids disrupting gradient flow between PRISM groups.

    Complexity: O(n * k * d) -- linear in n for fixed k.
    """

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # ReZero: output starts at zero, residual passes through at init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_states: Tensor,
        memory: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            token_states: (B, n, d) -- current token representations
            memory:       (B, k, d) -- memory bank
            mask:         (B, n)    -- True for valid tokens (for zeroing padding)

        Returns:
            updated token_states: (B, n, d)
        """
        B, n, d = token_states.shape
        k = memory.shape[1]
        H, dh = self.n_heads, self.d_head

        Q = self.q_proj(token_states).view(B, n, H, dh).transpose(1, 2)
        K = self.k_proj(memory).view(B, k, H, dh).transpose(1, 2)
        V = self.v_proj(memory).view(B, k, H, dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        retrieved = torch.matmul(attn, V)
        retrieved = retrieved.transpose(1, 2).reshape(B, n, d)

        # Pure residual -- no LayerNorm to avoid disrupting gradient flow
        output = token_states + self.out_proj(retrieved)

        return output


# ---------------------------------------------------------------------------
# Hybrid Encoder
# ---------------------------------------------------------------------------

class HybridPRISMEncoder(nn.Module):
    """PRISM encoder with memory bank.

    Interleaves groups of PRISM layers with memory write/read operations.
    PRISM layers use all-slow (lambda=0.99) decay and no interference.

    Architecture:
        Embed -> [Group 1] -> Write/Read -> [Group 2] -> Write/Read -> [Group 3] -> Pool
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 384,
        d_e: int = 384,
        n_layers: int = 12,
        n_channels: int = 6,
        max_len: int = 8192,
        k: int = 32,
        n_mem_heads: int = 4,
        layers_per_group: int = 4,
        bidirectional: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        assert n_layers % layers_per_group == 0, (
            f"n_layers={n_layers} must be divisible by layers_per_group={layers_per_group}"
        )

        self.d = d
        self.d_e = d_e
        self.k = k
        self.pad_token_id = pad_token_id
        self.n_channels = n_channels
        n_groups = n_layers // layers_per_group

        # Token + positional embedding
        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d)

        # PRISM layer groups
        self.groups = nn.ModuleList()
        for _ in range(n_groups):
            group = nn.ModuleList([
                PRISMLayer(d, n_channels, max_len, bidirectional, mlp_ratio, dropout)
                for _ in range(layers_per_group)
            ])
            self.groups.append(group)

        # Memory bank: k learned slot vectors
        self.memory_init = nn.Parameter(torch.randn(k, d) * 0.02)

        # Memory write/read between groups (n_groups - 1 interaction points)
        self.mem_writes = nn.ModuleList([
            MemoryWrite(d, n_mem_heads, dropout) for _ in range(n_groups - 1)
        ])
        self.mem_reads = nn.ModuleList([
            MemoryRead(d, n_mem_heads, dropout) for _ in range(n_groups - 1)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d)

        # Pooling: mean pooling (proven best for embeddings)
        self.pooling = MeanPooling(d, d_e)

        # Configure all-slow decay and no interference on PRISM layers
        self._configure_all_slow()
        self._configure_no_interference()

        self._init_weights()

    def _configure_all_slow(self):
        """Set all decay rates to 0.99 across all channels and layers."""
        for group in self.groups:
            for layer in group:
                layer.recurrence.lambdas.fill_(0.99)

    def _configure_no_interference(self):
        """Replace CrossScaleInterference with NoInterference (identity)."""
        for group in self.groups:
            for layer in group:
                layer.interference_fwd = NoInterference(layer.d_c, layer.n_channels)
                if layer.bidirectional:
                    layer.interference_bwd = NoInterference(layer.d_c, layer.n_channels)

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # Zero all Linear biases except recurrence gates and memory write
        # gates (matches PRISMEncoder init; preserves gate_proj bias=+2.0)
        skip = set()
        for group in self.groups:
            for layer in group:
                skip.update(layer.recurrence.gates_fwd)
                if layer.bidirectional:
                    skip.update(layer.recurrence.gates_bwd)
        for mw in self.mem_writes:
            skip.add(mw.gate_proj)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                if module not in skip:
                    nn.init.zeros_(module.bias)

    def memory_params(self):
        """Yield all memory-related parameters (for separate LR groups / freezing)."""
        yield self.memory_init
        yield from self.mem_writes.parameters()
        yield from self.mem_reads.parameters()

    def set_memory_frozen(self, frozen: bool):
        """Freeze or unfreeze memory modules (for warmup)."""
        self.memory_init.requires_grad_(not frozen)
        for m in list(self.mem_writes) + list(self.mem_reads):
            for p in m.parameters():
                p.requires_grad_(not frozen)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            input_ids:      (B, T) -- token IDs
            attention_mask: (B, T) -- 1 for valid, 0 for padding

        Returns:
            dict with "embedding" (B, d_e) and "token_embeddings" (B, T, d)
        """
        B, T = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        mask_bool = attention_mask.bool()

        # Token + position embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))
        x = x * mask_bool.unsqueeze(-1).float()

        # Initialize memory bank for this batch
        memory = self.memory_init.unsqueeze(0).expand(B, -1, -1)

        # Process groups with memory interactions between them
        for i, group in enumerate(self.groups):
            for layer in group:
                x, _ = layer(x, mask_bool)
                x = x * mask_bool.unsqueeze(-1).float()

            # Memory write/read between groups (not after the last group)
            if i < len(self.groups) - 1:
                memory = self.mem_writes[i](memory, x, mask_bool)
                x = self.mem_reads[i](x, memory, mask_bool)
                x = x * mask_bool.unsqueeze(-1).float()

        x = self.final_norm(x)

        embedding = self.pooling(x, query_state=None, mask=mask_bool)

        return {
            "embedding": embedding,
            "token_embeddings": x,
        }


# ---------------------------------------------------------------------------
# Contrastive Training Wrapper
# ---------------------------------------------------------------------------

class HybridPRISMForEmbedding(nn.Module):
    """Wrapper adding InfoNCE contrastive loss for training.

    Same interface as PRISMForEmbedding and TransformerForEmbedding.
    """

    def __init__(self, encoder: HybridPRISMEncoder, temperature: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.encoder(input_ids, attention_mask)
        emb = out["embedding"]
        return F.normalize(emb, p=2, dim=-1)

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        pos_ids: Tensor,
        pos_mask: Tensor,
        neg_ids: Optional[Tensor] = None,
        neg_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute InfoNCE loss with in-batch negatives."""
        q_emb = self.encode(query_ids, query_mask)
        p_emb = self.encode(pos_ids, pos_mask)
        B = q_emb.shape[0]

        sim = torch.matmul(q_emb, p_emb.T) / self.temperature

        if neg_ids is not None:
            B_n, N, T_n = neg_ids.shape
            n_emb = self.encode(
                neg_ids.reshape(B_n * N, T_n),
                neg_mask.reshape(B_n * N, T_n),
            ).reshape(B_n, N, -1)
            hard_sim = torch.einsum("bd, bnd -> bn", q_emb, n_emb) / self.temperature
            sim = torch.cat([sim, hard_sim], dim=-1)

        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            preds = sim.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def hybrid_prism_small(vocab_size: int = 32000, **kwargs) -> HybridPRISMEncoder:
    """Config A: 6 layers (3 groups of 2), k=32 memory slots. ~23M params."""
    defaults = dict(
        d=384, d_e=384, n_layers=6, n_channels=6,
        max_len=8192, k=32, n_mem_heads=4, layers_per_group=2,
        mlp_ratio=2.0, dropout=0.1,
    )
    defaults.update(kwargs)
    return HybridPRISMEncoder(vocab_size=vocab_size, **defaults)


def hybrid_prism_medium(vocab_size: int = 32000, **kwargs) -> HybridPRISMEncoder:
    """Config B: 12 layers (3 groups of 4), k=32 memory slots. ~38M params."""
    defaults = dict(
        d=384, d_e=384, n_layers=12, n_channels=6,
        max_len=8192, k=32, n_mem_heads=4, layers_per_group=4,
        mlp_ratio=2.0, dropout=0.1,
    )
    defaults.update(kwargs)
    return HybridPRISMEncoder(vocab_size=vocab_size, **defaults)
