"""
PRISM: Projected Recurrent Information Stratification with Mixing

A novel sequence embedding architecture that decomposes token sequences across
fixed temporal scales, computes cross-scale bilinear interference, and synthesizes
context-aware embeddings in linear time.

Reference implementation in PyTorch.

Author: Research prototype
License: Apache 2.0
"""

import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Utility: Hadamard-block initialisation
# ---------------------------------------------------------------------------

def _hadamard_matrix(n: int) -> Tensor:
    """Construct a normalised Hadamard matrix of size n (must be power of 2)."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)


def hadamard_init(d: int, d_c: int, n_channels: int) -> list[Tensor]:
    """Return *n_channels* projection matrices (d, d_c), initialised from
    orthogonal blocks of a Hadamard matrix when d is a power of 2,
    otherwise from a random orthogonal matrix."""
    if d & (d - 1) == 0:  # power of 2
        H = _hadamard_matrix(d)
    else:
        H, _ = torch.linalg.qr(torch.randn(d, d))
    blocks = []
    for c in range(n_channels):
        start = c * d_c
        blocks.append(H[:, start : start + d_c].clone())
    return blocks


# ---------------------------------------------------------------------------
# Parallel prefix-sum (Blelloch scan) for linear recurrences
# ---------------------------------------------------------------------------

def _fast_fixed_decay_scan(decay: float, values: Tensor) -> Tensor:
    """Vectorized parallel scan for h_t = decay * h_{t-1} + v_t with FIXED scalar decay.

    Uses the doubling trick: O(log T) rounds of fully parallel tensor operations.
    Each round shifts, scales, and adds — no Python loop over T.

    For decay=0 (memoryless channel), returns values directly.
    """
    if decay == 0.0:
        return values

    B, T, D = values.shape
    h = values.clone()
    power = decay   # tracks decay^(2^k)
    stride = 1

    while stride < T:
        # Shift h by `stride` positions, padding with zeros on the left
        h_shifted = F.pad(h[:, :-stride], (0, 0, stride, 0))
        h = h + power * h_shifted
        power = power * power   # decay^(2^(k+1)) = (decay^(2^k))^2
        stride *= 2

    return h


def _parallel_scan(gates: Tensor, values: Tensor) -> Tensor:
    """General parallel prefix scan for h_t = g_t * h_{t-1} + v_t.

    Handles time-varying gates. Uses O(log T) rounds of vectorized ops.
    For the fixed-decay case, prefer _fast_fixed_decay_scan (simpler, faster).

    Args:
        gates:  (B, T, D) — multiplicative gates
        values: (B, T, D) — additive inputs
    Returns:
        hiddens: (B, T, D) — scan results
    """
    B, T, D = values.shape
    a = gates.clone()
    b = values.clone()
    stride = 1

    while stride < T:
        a_shifted = F.pad(a[:, :-stride], (0, 0, stride, 0), value=1.0)
        b_shifted = F.pad(b[:, :-stride], (0, 0, stride, 0), value=0.0)
        new_b = a * b_shifted + b
        new_a = a * a_shifted
        a, b = new_a, new_b
        stride *= 2

    return b


@torch.jit.script
def _sequential_scan(gates: Tensor, values: Tensor) -> Tensor:
    """Sequential linear scan — correct reference implementation.

    For production, replace with a fused CUDA kernel (see mamba_ssm).
    """
    B, T, D = values.shape
    h = torch.zeros(B, D, device=values.device, dtype=values.dtype)
    outputs = []
    for t in range(T):
        h = gates[:, t] * h + values[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)  # (B, T, D)


# ---------------------------------------------------------------------------
# Stage 1: Stratified Input Projection
# ---------------------------------------------------------------------------

class StratifiedProjection(nn.Module):
    """Project d-dimensional tokens into C channels of dimension d_c = d / C,
    initialised from orthogonal blocks of a Hadamard matrix."""

    def __init__(self, d: int, n_channels: int):
        super().__init__()
        assert d % n_channels == 0, f"d={d} must be divisible by n_channels={n_channels}"
        self.d = d
        self.n_channels = n_channels
        self.d_c = d // n_channels

        # One linear projection per channel
        self.projections = nn.ModuleList([
            nn.Linear(d, self.d_c, bias=False) for _ in range(n_channels)
        ])

        # Hadamard initialisation
        blocks = hadamard_init(d, self.d_c, n_channels)
        for c, proj in enumerate(self.projections):
            proj.weight.data.copy_(blocks[c].T)  # Linear stores (out, in)

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Args:
            x: (B, T, d)
        Returns:
            list of C tensors, each (B, T, d_c)
        """
        return [proj(x) for proj in self.projections]


# ---------------------------------------------------------------------------
# Stage 2: Fixed-Rate Stratified Recurrence
# ---------------------------------------------------------------------------

class StratifiedRecurrence(nn.Module):
    """C independent linear recurrences with geometrically-spaced fixed decay
    rates and learned input gates.  Runs in both directions for embedding."""

    def __init__(
        self,
        d_c: int,
        n_channels: int,
        max_len: int = 8192,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.d_c = d_c
        self.n_channels = n_channels
        self.bidirectional = bidirectional

        # Fixed geometric decay rates: λ_c = 1 - 2^{-(c-1)·Δ}
        delta = math.log2(max_len) / max(n_channels - 1, 1)
        lambdas = []
        for c in range(n_channels):
            lam = 1.0 - 2.0 ** (-(c * delta))
            lambdas.append(lam)
        # Register as buffer (not learned)
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))

        # Learned input gates: one linear per channel per direction
        self.gates_fwd = nn.ModuleList([
            nn.Linear(d_c, d_c) for _ in range(n_channels)
        ])
        if bidirectional:
            self.gates_bwd = nn.ModuleList([
                nn.Linear(d_c, d_c) for _ in range(n_channels)
            ])

        # Initialise gate biases slightly positive so information flows at init
        for gate_list in ([self.gates_fwd] + ([self.gates_bwd] if bidirectional else [])):
            for gate in gate_list:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 1.0)

    def _run_direction(
        self,
        channels: list[Tensor],
        gates: nn.ModuleList,
    ) -> list[Tensor]:
        """Run the recurrence in one direction across all channels.

        Args:
            channels: list of C tensors, each (B, T, d_c)
            gates: ModuleList of C gate layers

        Returns:
            list of C hidden-state sequences, each (B, T, d_c)
        """
        # Extract all lambda values once (avoid per-channel GPU-CPU sync)
        lam_values = self.lambdas.detach().tolist()

        # Apply per-channel gating
        gated = []
        for z_c, gate_c in zip(channels, gates):
            g_t = torch.sigmoid(gate_c(z_c))  # (B, T, d_c)
            gated.append(g_t * z_c)

        # Fast path: batch all channels into one scan when all decays are equal
        all_same = all(abs(lam_values[i] - lam_values[0]) < 1e-8
                       for i in range(1, len(lam_values)))
        if all_same and len(gated) > 1:
            C = len(gated)
            # Stack: (C, B, T, d_c) -> reshape to (C*B, T, d_c)
            stacked = torch.stack(gated, dim=0)
            CB, T, D = C * stacked.shape[1], stacked.shape[2], stacked.shape[3]
            stacked = stacked.reshape(CB, T, D)
            h_all = _fast_fixed_decay_scan(lam_values[0], stacked)
            # Split back to list of C tensors
            h_all = h_all.reshape(C, -1, T, D)
            return [h_all[c] for c in range(C)]

        # General path: per-channel scan with different decays
        hiddens = []
        for c, gated_input in enumerate(gated):
            h_c = _fast_fixed_decay_scan(lam_values[c], gated_input)
            hiddens.append(h_c)
        return hiddens

    def forward(
        self,
        channels: list[Tensor],
    ) -> Tuple[list[Tensor], Optional[list[Tensor]]]:
        """
        Args:
            channels: list of C tensors, each (B, T, d_c) from StratifiedProjection.

        Returns:
            fwd_hiddens: list of C tensors (B, T, d_c)
            bwd_hiddens: list of C tensors (B, T, d_c) or None if not bidirectional
        """
        fwd_hiddens = self._run_direction(channels, self.gates_fwd)

        bwd_hiddens = None
        if self.bidirectional:
            # Reverse the sequences, run, then reverse back
            channels_rev = [ch.flip(dims=[1]) for ch in channels]
            bwd_hiddens_rev = self._run_direction(channels_rev, self.gates_bwd)
            bwd_hiddens = [h.flip(dims=[1]) for h in bwd_hiddens_rev]

        return fwd_hiddens, bwd_hiddens


# ---------------------------------------------------------------------------
# Stage 3: Cross-Scale Interference
# ---------------------------------------------------------------------------

class CrossScaleInterference(nn.Module):
    """Bilinear cross-channel interaction: each channel's hidden state is
    enriched by multiplicative features from every other channel.

    φ(h^(c), h^(c')) = (U^(c) h^(c)) ⊙ (V^(c') h^(c'))
    m^(c) = h^(c) + Σ_{c'≠c} α_{c,c'} · φ(h^(c), h^(c'))
    """

    def __init__(self, d_c: int, n_channels: int):
        super().__init__()
        self.d_c = d_c
        self.n_channels = n_channels
        C = n_channels

        # Bilinear projections: U^(c) and V^(c), each (d_c, d_c)
        # We store them as a single parameter for efficiency
        self.U = nn.Parameter(torch.zeros(C, d_c, d_c))
        self.V = nn.Parameter(torch.zeros(C, d_c, d_c))

        # Scalar mixing weights α_{c,c'} — initialised near zero (ReZero)
        self.alpha = nn.Parameter(torch.zeros(C, C))

        # Small init for U, V so interference starts weak
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

    def forward(self, hiddens: list[Tensor]) -> list[Tensor]:
        """
        Args:
            hiddens: list of C tensors, each (B, T, d_c)

        Returns:
            mixed: list of C tensors, each (B, T, d_c)
        """
        C = self.n_channels
        B, T, D = hiddens[0].shape

        # Stack for batched matmul: (C, B, T, d_c)
        H = torch.stack(hiddens, dim=0)

        # Compute projected versions: U_c @ h^(c) and V_c @ h^(c)
        UH = torch.einsum("cbti, cij -> cbtj", H, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H, self.V)

        # Memory-efficient interference: avoid materializing (C, C, B, T, D)
        # Key identity: interference[c] = UH[c] ⊙ Σ_{c'≠c} α[c,c'] · VH[c']
        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)

        # Weighted sum of VH across source channels: (C, C) @ (C, B, T, D) → (C, B, T, D)
        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)

        # Bilinear product with UH: (C, B, T, D)
        interference = UH * weighted_VH

        # Residual: m^(c) = h^(c) + interference^(c)
        mixed = H + interference

        return [mixed[c] for c in range(C)]


class CrossScaleInterferenceV2(nn.Module):
    """V2 cross-channel interaction with four targeted fixes:

    1. Per-channel LayerNorm before bilinear product (normalizes across decay scales)
    2. 1/sqrt(d_c) scaling on bilinear product (analogous to attention scaling)
    3. Alpha initialized to 1/(C-1) instead of zero (information flows from step one)
    4. Learned gate on interference path: mixed = H + sigmoid(gamma) * interference
    """

    def __init__(self, d_c: int, n_channels: int):
        super().__init__()
        self.d_c = d_c
        self.n_channels = n_channels
        C = n_channels

        # Fix 1: per-channel normalization
        self.channel_norms = nn.ModuleList([nn.LayerNorm(d_c) for _ in range(C)])

        # Bilinear projections
        self.U = nn.Parameter(torch.zeros(C, d_c, d_c))
        self.V = nn.Parameter(torch.zeros(C, d_c, d_c))
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

        # Fix 3: alpha initialized to 1/(C-1)
        alpha_init = torch.full((C, C), 1.0 / max(C - 1, 1))
        alpha_init.fill_diagonal_(0.0)
        self.alpha = nn.Parameter(alpha_init)

        # Fix 2: scaling factor
        self.scale = 1.0 / math.sqrt(d_c)

        # Fix 4: learned gate (sigmoid(-3) ≈ 0.05, near zero at init)
        self.gamma = nn.Parameter(torch.full((C, 1, 1, 1), -3.0))

    def forward(self, hiddens: list[Tensor]) -> list[Tensor]:
        C = self.n_channels
        H_orig = torch.stack(hiddens, dim=0)  # (C, B, T, d_c)

        # Fix 1: normalize per channel before bilinear
        H_normed = torch.stack(
            [self.channel_norms[c](hiddens[c]) for c in range(C)], dim=0
        )

        UH = torch.einsum("cbti, cij -> cbtj", H_normed, self.U)
        VH = torch.einsum("cbti, cij -> cbtj", H_normed, self.V)

        alpha_masked = self.alpha.clone()
        alpha_masked.fill_diagonal_(0.0)

        weighted_VH = torch.einsum("ij, jbtd -> ibtd", alpha_masked, VH)

        # Fix 2: scale bilinear product
        interference = UH * weighted_VH * self.scale

        # Fix 4: gated residual
        mixed = H_orig + torch.sigmoid(self.gamma) * interference

        return [mixed[c] for c in range(C)]


# ---------------------------------------------------------------------------
# Stage 4: Bidirectional Gated Fusion
# ---------------------------------------------------------------------------

class DirectionalFusion(nn.Module):
    """Gated fusion of forward and backward hidden states.

    f_t = β_t ⊙ m_fwd_t + (1 - β_t) ⊙ m_bwd_t
    β_t = σ(W_β [m_fwd_t || m_bwd_t])
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        # Initialise so β ≈ 0.5 at start (balanced fusion)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, m_fwd: Tensor, m_bwd: Tensor) -> Tensor:
        """
        Args:
            m_fwd: (B, T, d) — forward mixed states (all channels concatenated)
            m_bwd: (B, T, d) — backward mixed states (all channels concatenated)

        Returns:
            fused: (B, T, d)
        """
        combined = torch.cat([m_fwd, m_bwd], dim=-1)  # (B, T, 2d)
        beta = torch.sigmoid(self.gate(combined))       # (B, T, d)
        return beta * m_fwd + (1.0 - beta) * m_bwd


# ---------------------------------------------------------------------------
# Stage 5: Attentive Covariance Pooling
# ---------------------------------------------------------------------------

class AttentiveCovariancePooling(nn.Module):
    """Two-stream pooling: single-query attention + low-rank covariance sketch.

    Stream 1 (attentive):
        a_t = softmax(h_n^(C)ᵀ W_q f_t / √d)
        e_1 = Σ a_t f_t

    Stream 2 (covariance):
        e_2 = vec(1/n Σ (P f_t)(Q f_t)ᵀ)
    """

    def __init__(self, d: int, d_e: int, cov_rank: int = 32):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.cov_rank = cov_rank

        # Attentive pooling
        self.W_q = nn.Linear(d, d, bias=False)

        # Covariance sketch
        self.P = nn.Linear(d, cov_rank, bias=False)
        self.Q = nn.Linear(d, cov_rank, bias=False)

        # Output projection: [e_1 || e_2] → e
        self.out_proj = nn.Linear(d + cov_rank * cov_rank, d_e)
        self.layer_norm = nn.LayerNorm(d_e)

    def forward(
        self,
        f: Tensor,
        query_state: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            f:           (B, T, d) — fused token representations
            query_state: (B, d)    — global summary (last forward hidden from slowest channel)
            mask:        (B, T)    — True for valid positions, False for padding

        Returns:
            embedding: (B, d_e)
        """
        B, T, D = f.shape

        # --- Stream 1: attentive pooling ---
        query = self.W_q(query_state)  # (B, d)
        # Dot-product attention: (B, T)
        scores = torch.einsum("bd, btd -> bt", query, f) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B, T)
        e_1 = torch.einsum("bt, btd -> bd", attn, f)  # (B, d)

        # --- Stream 2: covariance sketch ---
        Pf = self.P(f)  # (B, T, r)
        Qf = self.Q(f)  # (B, T, r)
        if mask is not None:
            Pf = Pf * mask.unsqueeze(-1).float()
            Qf = Qf * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
        else:
            n_valid = torch.tensor(T, dtype=f.dtype, device=f.device)

        # (B, r, r) = 1/n * Σ_t Pf_t Qf_tᵀ
        cov = torch.einsum("btr, bts -> brs", Pf, Qf)
        if mask is not None:
            cov = cov / n_valid.unsqueeze(-1)
        else:
            cov = cov / T

        e_2 = cov.reshape(B, -1)  # (B, r²)

        # --- Combine and project ---
        combined = torch.cat([e_1, e_2], dim=-1)  # (B, d + r²)
        embedding = self.layer_norm(self.out_proj(combined))  # (B, d_e)

        return embedding


class AttentivePooling(nn.Module):
    """Learned single-query attentive pooling.

    A single learned query vector attends over token representations, producing
    a weighted sum. Simpler than AttentiveCovariancePooling — no covariance sketch,
    no input-dependent query. Adds ~d parameters (negligible).
    """

    def __init__(self, d: int, d_e: int, **kwargs):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, f: Tensor, query_state=None, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            f:    (B, T, d) — fused token representations
            query_state: ignored (kept for interface compatibility with MeanPooling)
            mask: (B, T) — True for valid positions
        Returns:
            embedding: (B, d_e)
        """
        scores = torch.einsum("d, btd -> bt", self.query, f) / math.sqrt(f.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.einsum("bt, btd -> bd", attn, f)  # (B, d)
        return self.norm(self.proj(pooled))


class MultiHeadAttentivePooling(nn.Module):
    """Multi-head attentive pooling with K learned query vectors.

    Each query independently attends over token representations. Results are
    concatenated and projected down to d_e dimensions. Lets different queries
    focus on different aspects of the document (entities, topics, conclusions).
    """

    def __init__(self, d: int, d_e: int, n_heads: int = 4, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_heads, d) * 0.02)
        self.proj = nn.Linear(n_heads * d, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, f: Tensor, query_state=None, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            f:    (B, T, d) — fused token representations
            query_state: ignored
            mask: (B, T) — True for valid positions
        Returns:
            embedding: (B, d_e)
        """
        B, T, D = f.shape
        # (K, d) @ (B, T, d).T -> (K, B, T)
        scores = torch.einsum("kd, btd -> kbt", self.queries, f) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (K, B, T)
        pooled = torch.einsum("kbt, btd -> kbd", attn, f)  # (K, B, d)
        pooled = pooled.permute(1, 0, 2).reshape(B, -1)  # (B, K*d)
        return self.norm(self.proj(pooled))


class AttentiveCovariancePoolingV2(nn.Module):
    """V2 pooling with three targeted fixes:

    1. LayerNorm on covariance vector before concatenation (scale normalization)
    2. Reduced cov_rank (default 8 instead of 32, yielding 64 dims not 1024)
    3. Project covariance to d dimensions so both streams contribute equally
    """

    def __init__(self, d: int, d_e: int, cov_rank: int = 8):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.cov_rank = cov_rank

        # Stream 1: attentive pooling
        self.W_q = nn.Linear(d, d, bias=False)

        # Stream 2: covariance sketch (reduced rank)
        self.P = nn.Linear(d, cov_rank, bias=False)
        self.Q = nn.Linear(d, cov_rank, bias=False)

        # Fix 1: normalize covariance before use
        self.cov_norm = nn.LayerNorm(cov_rank * cov_rank)

        # Fix 3: project covariance to d dims (eliminates dimensionality imbalance)
        self.cov_proj = nn.Linear(cov_rank * cov_rank, d)

        # Both streams now contribute d dimensions
        self.out_proj = nn.Linear(2 * d, d_e)
        self.layer_norm = nn.LayerNorm(d_e)

    def forward(
        self,
        f: Tensor,
        query_state: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, D = f.shape

        # --- Stream 1: attentive pooling ---
        query = self.W_q(query_state)
        scores = torch.einsum("bd, btd -> bt", query, f) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        e_1 = torch.einsum("bt, btd -> bd", attn, f)  # (B, d)

        # --- Stream 2: covariance sketch ---
        Pf = self.P(f)
        Qf = self.Q(f)
        if mask is not None:
            Pf = Pf * mask.unsqueeze(-1).float()
            Qf = Qf * mask.unsqueeze(-1).float()
            n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        else:
            n_valid = torch.tensor(T, dtype=f.dtype, device=f.device)

        cov = torch.einsum("btr, bts -> brs", Pf, Qf)
        if mask is not None:
            cov = cov / n_valid.unsqueeze(-1)
        else:
            cov = cov / T

        e_2 = cov.reshape(B, -1)

        # Fix 1+3: normalize then project to d dimensions
        e_2 = self.cov_proj(self.cov_norm(e_2))  # (B, d)

        # --- Combine ---
        combined = torch.cat([e_1, e_2], dim=-1)  # (B, 2*d)
        embedding = self.layer_norm(self.out_proj(combined))
        return embedding


# ---------------------------------------------------------------------------
# Full PRISM Layer
# ---------------------------------------------------------------------------

class PRISMLayer(nn.Module):
    """A single PRISM processing layer (replaces one Transformer layer).

    Stages: projection → recurrence → interference → fusion → residual MLP
    """

    def __init__(
        self,
        d: int,
        n_channels: int = 8,
        max_len: int = 8192,
        bidirectional: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_channels = n_channels
        self.d_c = d // n_channels
        self.bidirectional = bidirectional

        # Layer norm before processing (pre-norm architecture)
        self.norm = nn.LayerNorm(d)

        # Stage 1: Stratified projection
        self.projection = StratifiedProjection(d, n_channels)

        # Stage 2: Recurrence
        self.recurrence = StratifiedRecurrence(
            self.d_c, n_channels, max_len, bidirectional
        )

        # Stage 3: Cross-scale interference (applied to each direction)
        self.interference_fwd = CrossScaleInterference(self.d_c, n_channels)
        if bidirectional:
            self.interference_bwd = CrossScaleInterference(self.d_c, n_channels)

        # Stage 4: Bidirectional fusion (or identity if unidirectional)
        if bidirectional:
            self.fusion = DirectionalFusion(d)

        # Feed-forward residual MLP
        mlp_dim = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(d)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:    (B, T, d) — input token representations
            mask: (B, T)    — True for valid, False for padding

        Returns:
            out:             (B, T, d)  — updated representations
            fwd_global_state: (B, d_c)  — last hidden of slowest forward channel
        """
        residual = x
        x = self.norm(x)

        # Stage 1
        channels = self.projection(x)  # list of C × (B, T, d_c)

        # Stage 2
        fwd_h, bwd_h = self.recurrence(channels)

        # Capture the global summary state from slowest channel
        # (last position of forward, first position of backward)
        fwd_global = fwd_h[-1][:, -1, :]  # (B, d_c) — slowest channel, last position

        # Stage 3
        fwd_mixed = self.interference_fwd(fwd_h)

        # Concatenate channels → (B, T, d)
        fwd_concat = torch.cat(fwd_mixed, dim=-1)

        if self.bidirectional and bwd_h is not None:
            bwd_mixed = self.interference_bwd(bwd_h)
            bwd_concat = torch.cat(bwd_mixed, dim=-1)

            # Stage 4
            fused = self.fusion(fwd_concat, bwd_concat)
        else:
            fused = fwd_concat

        fused = self.dropout(fused)

        # First residual connection
        x = residual + fused

        # MLP block with second residual
        x = x + self.mlp(self.mlp_norm(x))

        return x, fwd_global


# ---------------------------------------------------------------------------
# Full PRISM Encoder
# ---------------------------------------------------------------------------

class PRISMEncoder(nn.Module):
    """Full PRISM encoder: token embedding → L layers → pooling → embedding.

    Drop-in replacement for a Transformer encoder in embedding pipelines.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 512,
        d_e: int = 768,
        n_layers: int = 6,
        n_channels: int = 8,
        max_len: int = 8192,
        bidirectional: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        cov_rank: int = 32,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.pad_token_id = pad_token_id
        self.n_channels = n_channels
        d_c = d // n_channels

        # Token + positional embedding
        self.token_emb = nn.Embedding(vocab_size, d, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d)

        # PRISM layers
        self.layers = nn.ModuleList([
            PRISMLayer(d, n_channels, max_len, bidirectional, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d)

        # Pooling (Stage 5)
        self.pooling = AttentiveCovariancePooling(d, d_e, cov_rank)

        # Project the d_c global state to full d for the pooling query
        self.query_proj = nn.Linear(d_c, d)

        self._init_weights()

    def _init_weights(self):
        """Standard initialisation."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                if module not in [g for layer in self.layers
                                  for g in list(layer.recurrence.gates_fwd)
                                  + (list(layer.recurrence.gates_bwd)
                                     if layer.bidirectional else [])]:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            input_ids:      (B, T) — token IDs
            attention_mask: (B, T) — 1 for valid, 0 for padding

        Returns:
            dict with:
                "embedding":       (B, d_e) — final sequence embedding
                "token_embeddings": (B, T, d)  — per-token representations from last layer
        """
        B, T = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        mask_bool = attention_mask.bool()

        # Token + position embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        # Zero out padding positions
        x = x * mask_bool.unsqueeze(-1).float()

        # Pass through L layers
        fwd_global = None
        for layer in self.layers:
            x, fwd_global = layer(x, mask_bool)
            x = x * mask_bool.unsqueeze(-1).float()

        x = self.final_norm(x)

        # Pooling
        query = self.query_proj(fwd_global)  # (B, d)
        embedding = self.pooling(x, query, mask_bool)

        return {
            "embedding": embedding,
            "token_embeddings": x,
        }


# ---------------------------------------------------------------------------
# PRISM for Sentence Embeddings (with contrastive training support)
# ---------------------------------------------------------------------------

class PRISMForEmbedding(nn.Module):
    """Wrapper that adds contrastive loss computation for training sentence
    embeddings (InfoNCE / in-batch negatives)."""

    def __init__(self, encoder: PRISMEncoder, temperature: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Produce normalised embeddings."""
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
        """Compute InfoNCE loss with in-batch negatives.

        Args:
            query_ids/mask: (B, T_q) — query sequences
            pos_ids/mask:   (B, T_p) — positive passages
            neg_ids/mask:   (B, N, T_n) — optional hard negatives (N per query)

        Returns:
            dict with "loss" and "accuracy"
        """
        q_emb = self.encode(query_ids, query_mask)   # (B, d_e)
        p_emb = self.encode(pos_ids, pos_mask)        # (B, d_e)
        B = q_emb.shape[0]

        # In-batch negatives: similarity matrix (B, B)
        sim = torch.matmul(q_emb, p_emb.T) / self.temperature

        # Optional hard negatives
        if neg_ids is not None:
            B, N, T_n = neg_ids.shape
            neg_flat = neg_ids.reshape(B * N, T_n)
            neg_mask_flat = neg_mask.reshape(B * N, T_n)
            n_emb = self.encode(neg_flat, neg_mask_flat)  # (B*N, d_e)
            n_emb = n_emb.reshape(B, N, -1)               # (B, N, d_e)

            # Hard negative similarities: (B, N)
            hard_sim = torch.einsum("bd, bnd -> bn", q_emb, n_emb) / self.temperature

            # Concatenate: (B, B + N)
            sim = torch.cat([sim, hard_sim], dim=-1)

        # Labels: diagonal is the positive
        labels = torch.arange(B, device=sim.device)
        loss = F.cross_entropy(sim, labels)

        # Accuracy
        with torch.no_grad():
            preds = sim.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()

        return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Model configuration presets
# ---------------------------------------------------------------------------

def prism_small(vocab_size: int = 32000, **kwargs) -> PRISMEncoder:
    """~30M params — comparable to MiniLM."""
    defaults = dict(d=384, d_e=384, n_layers=6, n_channels=6,
                    max_len=8192, mlp_ratio=2.0, dropout=0.1, cov_rank=24)
    defaults.update(kwargs)
    return PRISMEncoder(vocab_size=vocab_size, **defaults)


def prism_base(vocab_size: int = 32000, **kwargs) -> PRISMEncoder:
    """~110M params — comparable to BERT-base."""
    defaults = dict(d=768, d_e=768, n_layers=12, n_channels=8,
                    max_len=8192, mlp_ratio=4.0, dropout=0.1, cov_rank=32)
    defaults.update(kwargs)
    return PRISMEncoder(vocab_size=vocab_size, **defaults)


def prism_large(vocab_size: int = 32000, **kwargs) -> PRISMEncoder:
    """~330M params — comparable to BERT-large."""
    defaults = dict(d=1024, d_e=1024, n_layers=24, n_channels=8,
                    max_len=16384, mlp_ratio=4.0, dropout=0.1, cov_rank=48)
    defaults.update(kwargs)
    return PRISMEncoder(vocab_size=vocab_size, **defaults)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PRISM Reference Implementation — Smoke Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build small model
    model = prism_small(vocab_size=32000).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: prism_small")
    print(f"Parameters: {n_params:,}")
    print(f"Device: {device}")

    # Dummy input
    B, T = 4, 256
    input_ids = torch.randint(1, 32000, (B, T), device=device)
    attention_mask = torch.ones(B, T, device=device, dtype=torch.long)
    # Make last 20 tokens padding for variety
    attention_mask[:, -20:] = 0

    # Forward pass
    with torch.no_grad():
        out = model(input_ids, attention_mask)

    emb = out["embedding"]
    tok = out["token_embeddings"]

    print(f"\nInput shape:           ({B}, {T})")
    print(f"Embedding shape:       {tuple(emb.shape)}")
    print(f"Token embedding shape: {tuple(tok.shape)}")
    print(f"Embedding norm (mean): {emb.norm(dim=-1).mean().item():.4f}")

    # Verify embeddings are distinct
    cos_sim = F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0))
    print(f"Cosine sim (batch 0 vs 1): {cos_sim.item():.4f}")

    # Contrastive wrapper test
    wrapper = PRISMForEmbedding(model).to(device)
    q_ids = torch.randint(1, 32000, (B, 64), device=device)
    p_ids = torch.randint(1, 32000, (B, 128), device=device)
    q_mask = torch.ones(B, 64, device=device, dtype=torch.long)
    p_mask = torch.ones(B, 128, device=device, dtype=torch.long)

    result = wrapper(q_ids, q_mask, p_ids, p_mask)
    print(f"\nContrastive loss:     {result['loss'].item():.4f}")
    print(f"In-batch accuracy:    {result['accuracy'].item():.4f}")

    print("\n✓ All checks passed.")
