# KAN linear layer (KANLinear), forward

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Implement the KAN linear layer forward pass.

Takes a fp32 input x of shape `(B, I)` and produces an output of shape `(B, O)`. There are two terms that get summed at the end. G (grid_size) = 8 and K (spline_order) = 3 for every input.

**1. Base term:** apply SiLU to x, then project with base_weight `(O, I)` using a linear layer: `F.linear(F.silu(x), base_weight)`.

**2. Spline term:** evaluate B-spline bases of order K over the per-feature knot grid of shape `(I, G + 2K + 1)`. The recurrence follows Cox-de Boor starting from indicator bases (`grid[j] <= x < grid[j+1]`), runs K iterations, and yields bases of shape `(B, I, C)` where `C = G + K`. Scale the weights: `scaled_spline_weight = spline_weight * spline_scaler.unsqueeze(-1)` where spline_weight is `(O, I, C)` and spline_scaler is `(O, I)`. Then project with `F.linear(bases.reshape(B, I*C), scaled_spline_weight.reshape(O, I*C))`.

Final output is `base_output + spline_output`, shape `(B, O)`. `reference.py` holds the exact definition, including the grid.

All tensors (`x, grid, base_weight, spline_weight, spline_scaler, out`) are fp32 and contiguous.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const float* x, const float* grid, const float* base_weight, const float* spline_weight, const float* spline_scaler, int32_t grid_size, int32_t spline_order, int32_t B, int32_t I, int32_t O, float* out);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu -lcublas`
- You may use the CUDA runtime, CUB and Thrust and cuBLAS (`-lcublas` is in the build line). You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 1e-05 + 1e-05 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (the same math in fp64). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | B | I | O | x_range | Marks |
|---|---|---|---|---|---|
| `B1_I256_O512` | 1 | 256 | 512 |  |  |
| `B1000_I256_O512` | 1,000 | 256 | 512 |  |  |
| `B64_I100_O300` | 64 | 100 | 300 |  | sanitizer |
| `B512_I256_O512_wide_x` | 512 | 256 | 512 | 2.0 |  |
| `B4096_I256_O512` | 4,096 | 256 | 512 |  | timed |

Values: `x ~ U[-x_range, x_range)`, x_range 1 unless the table says otherwise (with x_range 2, some x fall outside the grid and get zero spline bases); grid uniform on [-1, 1] with 3 extra knots each side; `base_weight ~ U[-1/sqrt(I), 1/sqrt(I))`; `spline_weight ~ 0.05 * N(0, 1)`; `spline_scaler ~ 1 + 0.1 * N(0, 1)`.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
