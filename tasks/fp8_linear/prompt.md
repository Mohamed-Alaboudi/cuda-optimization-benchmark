# FP8 (E4M3) linear forward, inference

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Implement the forward pass of an fp8_e4m3 linear layer for inference.

- Inputs
    - `M`, `K`, `N`: int32_t
    - `x_fp8`: row-major fp8_e4m3 buffer, shape `(M, K)`
    - `scale_x`: fp32, scale for `x_fp8`
    - `w_fp8`: row-major fp8_e4m3 buffer, shape `(N, K)`
    - `scale_w`: fp32, scale for `w_fp8`
    - `bias`: bfloat16 buffer, shape `(N,)`
- Outputs
    - `output`: bfloat16 buffer, shape `(M, N)`
- Computation
    - Dequantize fp8_e4m3 to fp32: `X = float(x_fp8) / scale_x`, `W = float(w_fp8) / scale_w`.
    - Accumulate the matrix product and bias in fp32: `X @ W^T + float(bias)`.
    - Downcast to bfloat16 and store in `output`.
- The scales were computed as `scale = 448.0 / max(|fp32 tensor|)` before quantization; you receive them as fp32 values.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const int32_t M, const int32_t K, const int32_t N, const __nv_fp8_e4m3* x_fp8, float scale_x, const __nv_fp8_e4m3* w_fp8, float scale_w, const __nv_bfloat16* bias, __nv_bfloat16* output);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu -lcublasLt`
- You may use the CUDA runtime, CUB and Thrust and cuBLASLt (`-lcublasLt` is in the build line). You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 0.01 + 0.01 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (the same math in fp64, then one rounding to bfloat16). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | M | K | N | Marks |
|---|---|---|---|---|
| `M1_K1024_N4096` | 1 | 1,024 | 4,096 |  |
| `M16_K64_N128` | 16 | 64 | 128 | sanitizer |
| `M333_K1000_N3000` | 333 | 1,000 | 3,000 |  |
| `M2048_K1024_N4096` | 2,048 | 1,024 | 4,096 | timed |

Values: before quantization `x ~ N(0, 1)` and `w ~ N(0, 1) / sqrt(K)`; `bias ~ 0.05 * N(0, 1)` rounded to bfloat16. K is a multiple of 8 in every input.

The reference rounds to bfloat16 once, at the end. Two correct fp32 sums can round to neighbouring bfloat16 values; the tolerance allows that.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
