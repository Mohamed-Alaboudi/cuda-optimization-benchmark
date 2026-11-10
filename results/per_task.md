# Results by task

Built by `report/make_report.py` from the same logs. Labels are the grader's (`graders/verdict.py`). Closest call = the failing try that got furthest (stage, inputs passed, fewest wrong elements); for a model with no failure, its slowest passing try.

## efficient_kan

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try3: slowest pass, 4.84x PyTorch eager, 55% of reference speed | [pass try1](trials/gpt-6.1-sol/efficient_kan/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try1: slowest pass, 3.46x PyTorch eager, 39% of reference speed | [pass try1](trials/claude-opus-5-5/efficient_kan/try1.cu) |
| kimi-k3 | 3/3 | none | try1: slowest pass, 3.48x PyTorch eager, 39% of reference speed | [pass try1](trials/kimi-k3/efficient_kan/try1.cu) |

## fp8_linear

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try2: slowest pass, 9.74x PyTorch eager, 81% of reference speed | [pass try1](trials/gpt-6.1-sol/fp8_linear/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try3: slowest pass, 2.86x PyTorch eager, 24% of reference speed | [pass try1](trials/claude-opus-5-5/fp8_linear/try1.cu) |
| kimi-k3 | 3/3 | none | try3: slowest pass, 5.20x PyTorch eager, 43% of reference speed | [pass try1](trials/kimi-k3/fp8_linear/try1.cu) |

## leaky

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try3: slowest pass, 14.33x PyTorch eager, 91% of reference speed | [pass try1](trials/gpt-6.1-sol/leaky/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try1: slowest pass, 12.03x PyTorch eager, 77% of reference speed | [pass try1](trials/claude-opus-5-5/leaky/try1.cu) |
| kimi-k3 | 3/3 | none | try1: slowest pass, 13.03x PyTorch eager, 83% of reference speed | [pass try1](trials/kimi-k3/leaky/try1.cu) |

## pow_abs_sqrt_mean

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try2: slowest pass, 10.76x PyTorch eager, 81% of reference speed | [pass try1](trials/gpt-6.1-sol/pow_abs_sqrt_mean/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try1: slowest pass, 3.15x PyTorch eager, 24% of reference speed | [pass try1](trials/claude-opus-5-5/pow_abs_sqrt_mean/try1.cu) |
| kimi-k3 | 3/3 | none | try1: slowest pass, 10.19x PyTorch eager, 77% of reference speed | [pass try1](trials/kimi-k3/pow_abs_sqrt_mean/try1.cu) |

## ppo_clip

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try1: slowest pass, 36.02x PyTorch eager, 69% of reference speed | [pass try1](trials/gpt-6.1-sol/ppo_clip/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try1: slowest pass, 7.29x PyTorch eager, 14% of reference speed | [pass try1](trials/claude-opus-5-5/ppo_clip/try1.cu) |
| kimi-k3 | 3/3 | none | try1: slowest pass, 3.13x PyTorch eager, 6% of reference speed | [pass try1](trials/kimi-k3/ppo_clip/try1.cu) |

## reward_to_go

| Model | Passed | Failures by label | Closest call | Kernel files |
|---|---|---|---|---|
| gpt-6.1-sol | 3/3 | none | try2: slowest pass, 196.14x PyTorch eager, 98% of reference speed | [pass try1](trials/gpt-6.1-sol/reward_to_go/try1.cu) |
| claude-opus-5-5 | 3/3 | none | try2: slowest pass, 73.19x PyTorch eager, 36% of reference speed | [pass try1](trials/claude-opus-5-5/reward_to_go/try1.cu) |
| kimi-k3 | 3/3 | none | try2: slowest pass, 4.08x PyTorch eager, 2% of reference speed | [pass try1](trials/kimi-k3/reward_to_go/try1.cu) |
