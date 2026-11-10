"""leaky (LIF layer): PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (the same fp32 code: the operation order is part of the spec).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: snnTorch Leaky (MIT), reset by subtraction.
"""

import torch


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    T, B, H = case["T"], case["B"], case["H"]
    f = dict(generator=g, device=device, dtype=torch.float32)
    input_current = torch.randn(T, B, H, **f) * 0.35 + 0.12
    mem0_std = float(case.get("mem0_std", 0.0))
    mem0 = torch.randn(B, H, **f) * mem0_std if mem0_std else torch.zeros(B, H, device=device)
    lo, hi = case.get("beta_range", [0.65, 0.98])
    beta = torch.rand(H, **f) * (hi - lo) + lo
    threshold = torch.rand(H, **f) * 0.6 + 0.8
    return {"input_current": input_current, "mem0": mem0, "beta": beta, "threshold": threshold,
            "T": T, "B": B, "H": H}


def reference(inp):
    x = inp["input_current"]
    T, B, H = x.shape
    beta = inp["beta"].clamp(0.0, 1.0).view(1, H)
    th = inp["threshold"].view(1, H)
    mem = inp["mem0"].clone()
    spikes = torch.empty_like(x)
    for t in range(T):
        reset = (mem >= th).to(x.dtype)
        mem = beta * mem + x[t] - reset * th
        spikes[t] = (mem >= th).to(x.dtype)
    return {"spikes": spikes, "final_mem": mem}


reference_exact = reference
