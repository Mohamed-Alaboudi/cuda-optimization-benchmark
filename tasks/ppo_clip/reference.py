"""ppo_clip (PPO clipped objective and metrics): PyTorch reference.

reference(inp)       PyTorch eager, fp32: the speed baseline.
reference_exact(inp) the correctness reference (fp64).
make_inputs(case, seed, device) builds one input from a case in public_cases.json or the hidden case list.
Op source: TorchRL ClipPPOLoss.forward (MIT), tensor core only.
"""

import math

import torch


def make_inputs(case, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (case["B"], case["T"], case["H"])
    f = dict(generator=g, device=device, dtype=torch.float32)
    old = torch.randn(shape, **f) * 0.5
    # log weight = current - old ~ N(0, delta_std), cut to [-20, 20] so exp() stays finite in fp32
    delta = (torch.randn(shape, **f) * float(case.get("delta_std", 0.12))).clamp(-20.0, 20.0)
    cur = old + delta
    adv = torch.randn(shape, **f)
    return {"B": case["B"], "T": case["T"], "H": case["H"], "current_log_prob": cur,
            "old_log_prob": old, "advantage": adv, "clip_epsilon": float(case.get("clip_epsilon", 0.2))}


def _forward(cur, old, adv, eps):
    lw = cur - old
    ratio = torch.exp(lw)
    lwc = lw.clamp(math.log1p(-eps), math.log1p(eps))
    gain = torch.minimum(ratio * adv, torch.exp(lwc) * adv)
    flat = lw.reshape(-1)
    ess = torch.exp(2.0 * flat.logsumexp(0) - (2.0 * flat).logsumexp(0)) / flat.numel()
    return torch.stack([-gain.mean(), (lwc != lw).to(lw.dtype).mean(), ((ratio - 1.0) - lw).mean(),
                        ess, ratio.max(), ratio.mean()])


def reference(inp):
    return {"output": _forward(inp["current_log_prob"], inp["old_log_prob"], inp["advantage"], inp["clip_epsilon"])}


def reference_exact(inp):
    eps = float(torch.tensor(inp["clip_epsilon"], dtype=torch.float32))
    return {"output": _forward(inp["current_log_prob"].double(), inp["old_log_prob"].double(),
                               inp["advantage"].double(), eps)}
