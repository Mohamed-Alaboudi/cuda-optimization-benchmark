# Leaky integrate-and-fire (LIF) layer, forward

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Implement the *forward pass* of a leaky integrate-and-fire (LIF) *spiking neuron layer* that processes a multi-step input sequence.

You are given contiguous fp32 tensors as inputs:
- `input_current`: shape `(T, B, H)`, input currents over T timesteps for B batches and H hidden units.
- `mem0`: shape `(B, H)`, the initial membrane potential.
- `beta`: shape `(H,)`, membrane decay factors that MUST be clamped to the range `[0, 1]`.
- `threshold`: shape `(H,)`, firing thresholds.

For each timestep `t`, use fp32 arithmetic and these steps, in this order:
1. Clamp `beta` to `[0, 1]`.
2. For the current membrane state, compute `reset = 1.0` where `mem >= threshold` and `0.0` otherwise.
3. Update `mem = beta * mem + input_current[t] - reset * threshold`.
4. Store `spikes[t] = 1.0` where the updated membrane is *at least* `threshold`, otherwise `0.0`.

Return `spikes` and `final_mem`, where `spikes` is fp32 with shape `(T, B, H)` and contains **only** `0.0` or `1.0`, and `final_mem` is fp32 with shape `(B, H)`.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const float* input_current, const float* mem0, const float* beta, const float* threshold, int32_t T, int32_t B, int32_t H, float* spikes, float* final_mem);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu`
- You may use the CUDA runtime, CUB and Thrust. You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 1e-06 + 1e-05 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (fp32, the steps above in order). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | T | B | H | beta_range | mem0_std | Marks |
|---|---|---|---|---|---|---|
| `T1_B1_H1021` | 1 | 1 | 1,021 |  |  |  |
| `T33_B65_H129_mem0` | 33 | 65 | 129 |  | 0.1 | sanitizer |
| `T160_B85_H200` | 160 | 85 | 200 |  |  |  |
| `T32_B64_H128_beta_out_of_range` | 32 | 64 | 128 | [-0.2, 1.2] |  |  |
| `T256_B512_H1024` | 256 | 512 | 1,024 |  |  | timed |

Values: `input_current ~ 0.35 * N(0, 1) + 0.12`; `mem0` is 0, or `N(0, mem0_std)` where the table gives `mem0_std`; `beta ~ U[0.65, 0.98)` unless the table gives another range (one input has `beta` in [-0.2, 1.2) to test the clamp); `threshold ~ U[0.8, 1.4)`.

A spike is 0.0 or 1.0, so a spike that differs from the reference is always outside the tolerance.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
