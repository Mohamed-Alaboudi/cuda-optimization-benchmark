"""reward_to_go (discounted reward-to-go with done flags): PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (fp64, a plain backward loop).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: TorchRL reward2go (MIT), the same definition.
"""

import torch


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    B, T = case["B"], case["T"]
    reward = torch.rand(B, T, generator=g, device=device, dtype=torch.float32)
    done = torch.rand(B, T, generator=g, device=device) < float(case["done_p"])
    done[:, -1] = True
    gamma = float(torch.tensor(case["gamma"], dtype=torch.float32))  # the fp32 value the kernel gets
    return {"reward": reward, "done": done, "gamma": gamma, "B": B, "T": T}


def reference(inp):
    """fp32 eager baseline: a log-depth affine scan over time, out[t] = a[t] * out[t+1] + reward[t]."""
    a = torch.where(inp["done"], 0.0, inp["gamma"]).to(torch.float32)
    b = inp["reward"].clone()
    T, d = b.shape[1], 1
    while d < T:
        b = torch.cat([b[:, :-d] + a[:, :-d] * b[:, d:], b[:, -d:]], dim=1)
        a = torch.cat([a[:, :-d] * a[:, d:], a[:, -d:] * 0.0], dim=1)
        d *= 2
    return {"output": b}


def reference_exact(inp):
    r = inp["reward"].double()
    keep = (~inp["done"]).double()
    out = torch.empty_like(r)
    nxt = torch.zeros_like(r[:, 0])
    for t in range(r.shape[1] - 1, -1, -1):
        nxt = r[:, t] + inp["gamma"] * keep[:, t] * nxt
        out[:, t] = nxt
    return {"output": out}
