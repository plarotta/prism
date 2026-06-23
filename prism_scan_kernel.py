"""
Fused linear-recurrence scan for PRISM.

PRISM's sequence mixer is a multi-channel linear recurrence with a *fixed*
per-channel scalar decay:

    h_t = lambda_c * h_{t-1} + x_t        (h_{-1} = 0, per channel c)

The reference PyTorch implementation (`prism._fast_fixed_decay_scan`) computes
this with an O(log T) Hillis-Steele doubling scan, issuing many small CUDA
kernels per call. This module provides a single fused Triton kernel that computes
the same recurrence in one launch (a chunked linear scan), with fp32 accumulation
and a drop-in PyTorch fallback for CPU / Triton-unavailable environments.

Public API:
    fused_decay_scan(x, decays, reverse=False) -> Tensor
        x:      (B, T, C, d_c)  bf16/fp16/fp32
        decays: (C,)            fp32, constant per-channel lambda in [0, 1)
        returns (B, T, C, d_c)  same dtype as x

    fused_scan_available() -> bool

Key facts that keep this simple:
  * The backward of a linear recurrence is the same recurrence run in reverse
    time on the output gradient (dx_k = sum_{t>=k} lambda^{t-k} g_t). So the
    backward pass reuses the forward scan with `reverse` flipped, and there is
    NO gradient w.r.t. `decays` (they are fixed buffers).
  * `reverse=True` is implemented by flipping the time axis around a forward-only
    kernel, exactly matching PRISM's existing flip->scan->flip bidirectional path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def fused_scan_available() -> bool:
    """True when the Triton kernel can be used (Triton importable)."""
    return TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Triton chunked forward scan
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _scan_fwd_kernel(
        X,           # *in*  (B, T, C, D)
        OUT,         # *out* (B, T, C, D)
        DECAYS,      # *in*  (C,) fp32
        T,
        C,
        D,
        NUM_CHUNKS,
        stride_b,
        stride_t,
        stride_c,
        stride_d,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # One program per (batch, channel) pair; the d_c feature dim lives in-block.
        pid = tl.program_id(0)
        b = pid // C
        c = pid % C

        lam = tl.load(DECAYS + c).to(tl.float32)
        # log(lambda); clamp so lambda == 0 (memoryless) yields a ~ -inf log whose
        # exp() underflows cleanly to 0 for any positive exponent (and 1 at exp 0).
        loglam = tl.log(tl.maximum(lam, 1e-30))

        row = tl.arange(0, BLOCK_T)
        col = tl.arange(0, BLOCK_D)
        d_mask = col < D

        # Intra-chunk decay matrix M[i, j] = lambda^(i-j) for j <= i, else 0.
        diff = row[:, None] - row[None, :]
        M = tl.where(diff >= 0, tl.exp(diff.to(tl.float32) * loglam), 0.0)
        # Cross-chunk carry weight for in-chunk position i: lambda^(i+1).
        carry_decay = tl.exp((row + 1).to(tl.float32) * loglam)  # (BLOCK_T,)

        base = X + b * stride_b + c * stride_c
        obase = OUT + b * stride_b + c * stride_c

        carry = tl.zeros((BLOCK_D,), dtype=tl.float32)  # h of last position so far

        for k in range(NUM_CHUNKS):
            start = k * BLOCK_T
            t_idx = start + row
            t_mask = t_idx < T
            load_mask = t_mask[:, None] & d_mask[None, :]

            ptrs = base + t_idx[:, None] * stride_t + col[None, :] * stride_d
            x_chunk = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)

            # h = (intra-chunk decay-weighted prefix sum) + (carry from prior chunks)
            intra = tl.dot(M, x_chunk, input_precision="ieee")  # (BLOCK_T, BLOCK_D)
            h = intra + carry_decay[:, None] * carry[None, :]

            optrs = obase + t_idx[:, None] * stride_t + col[None, :] * stride_d
            tl.store(optrs, h.to(OUT.dtype.element_ty), mask=load_mask)

            # Carry forward the hidden state at the last valid row of this chunk.
            chunk_len = tl.minimum(BLOCK_T, T - start)
            sel = row[:, None] == (chunk_len - 1)
            carry = tl.sum(tl.where(sel, h, 0.0), axis=0)


def _triton_forward(x: Tensor, decays: Tensor) -> Tensor:
    """Forward-only scan via the Triton kernel. x: (B, T, C, D) contiguous-ish."""
    x = x.contiguous()
    B, T, C, D = x.shape
    out = torch.empty_like(x)
    decays = decays.to(device=x.device, dtype=torch.float32).contiguous()

    BLOCK_T = 64
    BLOCK_D = triton.next_power_of_2(D)
    num_chunks = triton.cdiv(T, BLOCK_T)
    grid = (B * C,)

    _scan_fwd_kernel[grid](
        x,
        out,
        decays,
        T,
        C,
        D,
        num_chunks,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# PyTorch fallback (CPU / no-Triton) — also the numerical reference helper
# ---------------------------------------------------------------------------

def _ref_forward(x: Tensor, decays: Tensor) -> Tensor:
    """Forward-only scan in pure PyTorch with fp32 accumulation.

    Uses the same O(log T) doubling recurrence as the legacy scan but batched
    over channels with per-channel decay, accumulating in fp32 regardless of the
    input dtype. Returns the input dtype.
    """
    orig_dtype = x.dtype
    # Accumulate in fp32 for low-precision inputs; keep fp32/fp64 as-is (so fp64
    # gradcheck stays exact).
    acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    h = x.to(acc_dtype)
    T = h.shape[1]
    # (1, 1, C, 1) so it broadcasts over (B, T, C, D).
    power = decays.to(device=x.device, dtype=acc_dtype).view(1, 1, -1, 1).clone()

    stride = 1
    while stride < T:
        # Shift along the time axis (dim 1), zero-padding the front.
        h_shifted = F.pad(h[:, :-stride], (0, 0, 0, 0, stride, 0))
        h = h + power * h_shifted
        power = power * power
        stride *= 2

    return h.to(orig_dtype)


# ---------------------------------------------------------------------------
# Direction dispatch + autograd
# ---------------------------------------------------------------------------

def _scan(x: Tensor, decays: Tensor, reverse: bool) -> Tensor:
    """Run the (optionally reversed) forward scan, choosing Triton or fallback."""
    if reverse:
        x = x.flip(1)

    if x.is_cuda and TRITON_AVAILABLE:
        out = _triton_forward(x, decays)
    else:
        out = _ref_forward(x, decays)

    if reverse:
        out = out.flip(1)
    return out


class _FusedDecayScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, decays: Tensor, reverse: bool) -> Tensor:
        ctx.decays = decays
        ctx.reverse = reverse
        return _scan(x, decays, reverse)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        # Gradient of a linear recurrence is the same recurrence in reverse time;
        # decays are fixed buffers, so no gradient flows to them.
        grad_x = _scan(grad_out.contiguous(), ctx.decays, not ctx.reverse)
        return grad_x, None, None


def fused_decay_scan(x: Tensor, decays: Tensor, reverse: bool = False) -> Tensor:
    """Fused per-channel linear recurrence  h_t = decay_c * h_{t-1} + x_t.

    Args:
        x:       (B, T, C, d_c) input, bf16/fp16/fp32.
        decays:  (C,) fp32 constant per-channel decay in [0, 1).
        reverse: if True, scan from t=T-1 down to t=0 (time-reversed).

    Returns:
        (B, T, C, d_c) hidden states, same dtype as x.
    """
    if x.dim() != 4:
        raise ValueError(f"x must be (B, T, C, d_c); got shape {tuple(x.shape)}")
    if decays.dim() != 1 or decays.shape[0] != x.shape[2]:
        raise ValueError(
            f"decays must be (C,) with C={x.shape[2]}; got shape {tuple(decays.shape)}"
        )
    return _FusedDecayScan.apply(x, decays, reverse)
