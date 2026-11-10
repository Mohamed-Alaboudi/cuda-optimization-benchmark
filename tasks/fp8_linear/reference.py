"""fp8_linear (FP8 E4M3 linear, inference): PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (fp64 dequantize and matmul, then one rounding to bf16).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: NVIDIA TransformerEngine Linear (Apache-2.0), inference math only.
"""

import numpy as np
import torch

E4M3_MAX = 448.0


def _quantize(x):
    """Per-tensor max-abs E4M3 quantization. scale is rounded to fp32, the type the kernel receives."""
    scale = float(np.float32(E4M3_MAX / max(x.abs().max().item(), 1e-9)))
    return (x * scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    M, K, N = case["M"], case["K"], case["N"]
    f = dict(generator=g, device=device, dtype=torch.float32)
    x = torch.randn(M, K, **f)
    w = torch.randn(N, K, **f) * (1.0 / K ** 0.5)
    bias = (torch.randn(N, **f) * 0.05).to(torch.bfloat16)
    x_fp8, scale_x = _quantize(x)
    w_fp8, scale_w = _quantize(w)
    return {"M": M, "K": K, "N": N, "x_fp8": x_fp8, "scale_x": scale_x, "w_fp8": w_fp8,
            "scale_w": scale_w, "bias": bias}


def _forward(inp, dtype):
    x = inp["x_fp8"].to(dtype) / inp["scale_x"]
    w = inp["w_fp8"].to(dtype) / inp["scale_w"]
    return (x @ w.t() + inp["bias"].to(dtype)).to(torch.bfloat16)


def reference(inp):
    return {"output": _forward(inp, torch.float32)}


def reference_exact(inp):
    return {"output": _forward(inp, torch.float64)}
