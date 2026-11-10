"""pow_abs_sqrt_mean: PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (fp64).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: KernelBook pow -> abs -> sqrt -> mean.
"""

import torch


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (case["batch_size"], 4, case["height"], case["width"])
    scale = float(case.get("scale", 1.0))  # values are U[0, scale)
    inp = torch.rand(shape, generator=g, device=device, dtype=torch.float32) * scale
    target = torch.rand(shape, generator=g, device=device, dtype=torch.float32) * scale
    return {"inp": inp, "target": target,
            "batch_size": case["batch_size"], "height": case["height"], "width": case["width"]}


def _forward(inp, target):
    diff = torch.pow(torch.abs(inp - target), 2) + (0.001 * 0.001)
    return torch.mean(torch.sqrt(diff))


def reference(inp):
    return {"output": _forward(inp["inp"], inp["target"])}


def reference_exact(inp):
    return {"output": _forward(inp["inp"].double(), inp["target"].double())}
