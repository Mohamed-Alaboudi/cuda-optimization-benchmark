# PPO clipped objective and diagnostic metrics

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Implement the PPO clipped policy objective and its diagnostic metrics.

You are given three fp32 tensors of identical shape `(B, T, H)`: 1. current log probabilities (`current_log_prob`) 2. old log probabilities (`old_log_prob`) 3. advantages (`advantage`). You are also given a scalar clip epsilon (`clip_epsilon`).

Compute the **log importance weight** `log_weight = current_log_prob - old_log_prob` and clip it to the range `[log(1 - eps), log(1 + eps)]`. Using the exponential of the clipped and unclipped weights (the **ratio**), calculate the **clipped surrogate gain**: the element-wise minimum of the ratio-weighted advantages. Use that to calculate the **objective loss**: the negative mean of the clipped surrogate gain.

Additionally, calculate the following diagnostic metrics:
- the fraction of samples that were clipped (clipped log weight != log weight),
- the approximate **KL divergence**: the mean of `(ratio - 1) - log_weight`,
- the **effective sample size (ESS)**: `exp(2 * logsumexp(lw) - logsumexp(2 * lw)) / N` over the flattened log weights lw, with N the element count,
- the **max** and **mean** of the ratio.

Return a length-6 fp32 vector in this exact order: `[loss_objective, clip_fraction, approx_kl, ess, max_ratio, mean_ratio]`.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const int32_t B, const int32_t T, const int32_t H, const float* current_log_prob, const float* old_log_prob, const float* advantage, const float clip_epsilon, float* output);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu`
- You may use the CUDA runtime, CUB and Thrust. You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 1e-05 + 0.0001 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (the same math in fp64). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | B | T | H | clip_epsilon | delta_std | Marks |
|---|---|---|---|---|---|---|
| `B1_T1_H1` | 1 | 1 | 1 |  |  |  |
| `B7_T13_H3` | 7 | 13 | 3 |  |  |  |
| `B1000_T333_H2` | 1,000 | 333 | 2 |  |  | sanitizer |
| `B512_T256_H1_wide` | 512 | 256 | 1 | 0.1 | 1.0 |  |
| `B4096_T256_H1` | 4,096 | 256 | 1 |  |  | timed |

Values: `old_log_prob ~ 0.5 * N(0, 1)`; `current_log_prob = old_log_prob + delta`, `delta ~ delta_std * N(0, 1)` (delta_std 0.12 unless the table says otherwise) cut to [-20, 20]; `advantage ~ N(0, 1)`; `clip_epsilon` 0.2 unless the table says otherwise.

Input fix: the log weights are cut to [-20, 20]. Beyond about 88, `exp()` overflows fp32 and the PyTorch reference itself returns inf for the max ratio, so no input goes there.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
