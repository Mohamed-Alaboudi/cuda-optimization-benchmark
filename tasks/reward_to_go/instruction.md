# Batched discounted reward-to-go with done flags

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Implement batched reward-to-go.

Inputs are a contiguous float32 `reward` and bool `done` tensor with shape `(B, T)`, plus a scalar float32 discount factor `gamma`.

Output is a float32 tensor `output` with shape `(B, T)`.

For each batch row `b`, compute in reverse time order using float32 arithmetic:
`output[b, t] = reward[b, t] + gamma * next`
where `next = 0.0` if `t == T - 1` or `done[b, t]` is true; otherwise `next = output[b, t + 1]`.

When `done[b, t]` is true, `output[b, t]` equals `reward[b, t]`, and later timesteps must not contribute across that boundary. The reward at a terminal position is included there and may contribute to earlier positions in the same packed episode. Multiple terminal flags may appear in one row.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu`
- You may use the CUDA runtime, CUB and Thrust. You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 1e-05 + 1e-05 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (the same math in fp64). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | B | T | gamma | done_p | Marks |
|---|---|---|---|---|---|
| `B3_T1` | 3 | 1 | 0.5 | 0.1 |  |
| `B7_T999` | 7 | 999 | 0.9 | 0.1 | sanitizer |
| `B256_T1024` | 256 | 1,024 | 0.99 | 0.01 |  |
| `B1_T8000` | 1 | 8,000 | 0.99 | 0.001 |  |
| `B64_T1024` | 64 | 1,024 | 0.99 | 0.1 | timed |

Values: `reward ~ U[0, 1)`; each `done` flag is true with probability `done_p` (per input), and `done[:, T-1]` is always true; `gamma` per input, in [0.5, 0.99].

Input fix: gamma 0 is not used. The original PyTorch reference for this task (TorchRL `reward2go`) does not handle gamma 0 correctly (the domain expert's finding; not re-checked here). The `reference.py` given to you is a fresh PyTorch implementation of the definition above.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
