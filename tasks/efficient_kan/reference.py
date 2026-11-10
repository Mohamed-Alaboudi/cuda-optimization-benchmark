"""efficient_kan (KANLinear forward): PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (fp64).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: efficient-kan KANLinear (MIT).
"""

import math

import torch
import torch.nn.functional as F


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    B, I, O = case["B"], case["I"], case["O"]
    G, K = case.get("grid_size", 8), case.get("spline_order", 3)
    f = dict(generator=g, device=device, dtype=torch.float32)
    xr = float(case.get("x_range", 1.0))  # x ~ U[-x_range, x_range)
    x = (torch.rand(B, I, **f) * 2.0 - 1.0) * xr
    step = 2.0 / G
    knots = torch.arange(-K, G + K + 1, device=device, dtype=torch.float32)
    grid = (knots * step - 1.0).expand(I, -1).contiguous()
    bound = 1.0 / math.sqrt(I)  # kaiming_uniform_(a=sqrt(5))
    base_weight = (torch.rand(O, I, **f) * 2.0 - 1.0) * bound
    spline_weight = torch.randn(O, I, G + K, **f) * 0.05
    spline_scaler = torch.randn(O, I, **f) * 0.1 + 1.0
    return {"x": x, "grid": grid, "base_weight": base_weight, "spline_weight": spline_weight,
            "spline_scaler": spline_scaler, "grid_size": G, "spline_order": K, "B": B, "I": I, "O": O}


def _b_splines(x, grid, spline_order):
    xe = x.unsqueeze(-1)
    bases = ((xe >= grid[:, :-1]) & (xe < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = ((xe - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])) * bases[:, :, :-1] + \
                ((grid[:, k + 1:] - xe) / (grid[:, k + 1:] - grid[:, 1:-k])) * bases[:, :, 1:]
    return bases.contiguous()


def _forward(x, grid, base_weight, spline_weight, spline_scaler, spline_order):
    bases = _b_splines(x, grid, spline_order)
    base_out = F.linear(F.silu(x), base_weight)
    w = (spline_weight * spline_scaler.unsqueeze(-1)).reshape(base_weight.shape[0], -1)
    return base_out + F.linear(bases.reshape(x.shape[0], -1), w)


def reference(inp):
    return {"out": _forward(inp["x"], inp["grid"], inp["base_weight"], inp["spline_weight"],
                            inp["spline_scaler"], inp["spline_order"])}


def reference_exact(inp):
    d = {k: inp[k].double() for k in ("x", "grid", "base_weight", "spline_weight", "spline_scaler")}
    return {"out": _forward(d["x"], d["grid"], d["base_weight"], d["spline_weight"], d["spline_scaler"],
                            inp["spline_order"])}
