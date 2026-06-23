"""
Correctness tests for the fused decay scan (`prism_scan_kernel.fused_decay_scan`).

Run:  uv run python test_fused_scan.py

On a CPU-only box this exercises the PyTorch fallback (still numerically exact)
and the autograd wiring. On a CUDA box it additionally exercises the Triton
kernel. The reference oracle is `prism._sequential_scan` (the ground-truth
sequential loop).
"""

import itertools
import math

import torch

from prism import _sequential_scan
from prism_scan_kernel import fused_decay_scan, fused_scan_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def reference_scan(x, decays, reverse=False):
    """Ground-truth per-channel scan via prism._sequential_scan.

    x: (B, T, C, d_c), decays: (C,). Loops channels, builds a constant gate
    tensor = lambda_c, runs the reference sequential scan. Returns (B, T, C, d_c).
    """
    B, T, C, D = x.shape
    outs = []
    for c in range(C):
        xc = x[:, :, c, :]                      # (B, T, D)
        if reverse:
            xc = xc.flip(1)
        gate = torch.full_like(xc, float(decays[c]))
        hc = _sequential_scan(gate, xc)         # (B, T, D)
        if reverse:
            hc = hc.flip(1)
        outs.append(hc)
    return torch.stack(outs, dim=2)             # (B, T, C, d_c)


def geometric_decays(C, max_len=8192):
    delta = math.log2(max_len) / max(C - 1, 1)
    return [1.0 - 2.0 ** (-(c * delta)) for c in range(C)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_parity():
    """fp32 max-abs < 1e-5, bf16 rel < 1e-2 vs the sequential reference."""
    C, D = 6, 64
    Ts = [1, 2, 7, 128, 512, 2048, 8192]
    decay_sets = {
        "zero": [0.0] * C,
        "half": [0.5] * C,
        "p9": [0.9] * C,
        "p99": [0.99] * C,
        "geometric": geometric_decays(C),
        "mixed": [0.0, 0.5, 0.9, 0.99, 0.999, 0.5][:C],
    }
    torch.manual_seed(0)
    n_checks = 0
    for T, (name, decays), reverse in itertools.product(
        Ts, decay_sets.items(), [False, True]
    ):
        B = 3
        dec = torch.tensor(decays, dtype=torch.float32, device=DEVICE)

        # fp32 parity. Both sides are fp32 but use different summation orders
        # (sequential vs chunked/doubling), so compare with a magnitude-aware
        # tolerance: atol 1e-5 plus a small rtol on the output scale.
        x = torch.randn(B, T, C, D, device=DEVICE, dtype=torch.float32)
        ref = reference_scan(x, dec, reverse=reverse)
        got = fused_decay_scan(x, dec, reverse=reverse)
        err = (got - ref).abs().max().item()
        tol = 1e-5 + 1e-4 * ref.abs().max().item()
        assert err < tol, f"fp32 {name} T={T} rev={reverse}: max-abs {err:.2e} > {tol:.2e}"

        # bf16 parity: compare the bf16 scan against the fp32 reference run on the
        # *same* bf16-quantized input, isolating scan error from input rounding.
        # Use a global-scale relative metric (max-abs error over max-abs signal);
        # per-element ratios are meaningless near zeros (and the geometric set
        # includes a lambda=0 channel).
        xb = x.to(torch.bfloat16)
        ref_b = reference_scan(xb.float(), dec, reverse=reverse)
        gotb = fused_decay_scan(xb, dec, reverse=reverse).float()
        rel = (gotb - ref_b).abs().max().item() / ref_b.abs().max().clamp_min(1e-6).item()
        assert rel < 1e-2, f"bf16 {name} T={T} rev={reverse}: rel {rel:.2e}"
        n_checks += 1

    print(f"[ok] forward parity: {n_checks} configs (fp32 + bf16, both directions)")


def test_gradcheck():
    """fp64 gradcheck on small sizes, both directions.

    Pinned to CPU: this validates the device-independent autograd formula
    (backward = reverse-time scan) in full fp64. The Triton kernel computes in
    fp32, so its backward is covered instead by test_grad_parity (fp32).
    """
    torch.manual_seed(0)
    for reverse in [False, True]:
        B, T, C, D = 2, 7, 2, 3
        dec = torch.tensor([0.5, 0.9], dtype=torch.float64)
        x = torch.randn(B, T, C, D, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda inp: fused_decay_scan(inp, dec, reverse=reverse), (x,), atol=1e-6
        )
    print("[ok] gradcheck (fp64, both directions)")


def test_grad_parity():
    """dx parity vs autograd through the sequential reference, realistic sizes."""
    torch.manual_seed(0)
    for reverse in [False, True]:
        B, T, C, D = 4, 256, 6, 64
        dec = torch.tensor(geometric_decays(C), dtype=torch.float32, device=DEVICE)

        x1 = torch.randn(B, T, C, D, device=DEVICE, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        g = torch.randn(B, T, C, D, device=DEVICE)

        out_ref = reference_scan(x1, dec, reverse=reverse)
        out_ref.backward(g)

        out_fused = fused_decay_scan(x2, dec, reverse=reverse)
        out_fused.backward(g)

        err = (x1.grad - x2.grad).abs().max().item()
        assert err < 1e-4, f"grad parity rev={reverse}: max-abs {err:.2e}"
    print("[ok] gradient parity vs sequential autograd")


if __name__ == "__main__":
    print(f"device = {DEVICE}, triton available = {fused_scan_available()}")
    if DEVICE == "cpu":
        print("(CPU run: exercising the PyTorch fallback path, not the Triton kernel)")
    test_forward_parity()
    test_gradcheck()
    test_grad_parity()
    print("\nALL TESTS PASSED")
