# Mean of sqrt(squared difference + 1e-6)

You are working alone in a shell in `/app` on a machine with one NVIDIA RTX PRO 6000 Blackwell GPU (sm_120), CUDA 12.8 (nvcc 12.8.1, compute-sanitizer) and PyTorch 2.9.1. You have **20 minutes**. Your answer is the file `/app/solution.cu` as it stands when time runs out.

## The operation
Compute one fp32 scalar: the mean over all elements of the square root of the squared difference between `inp` and `target` plus a small constant.

Inputs:
- `inp`: fp32 tensor, shape `(batch_size, 4, height, width)`.
- `target`: fp32 tensor, shape `(batch_size, 4, height, width)`.

Both tensors are contiguous. The second dimension is always 4.

For each element position, in fp32 and in this order:
1. d is `inp` minus `target`.
2. a is the absolute value of d.
3. p is a raised to the power 2.
4. s is p plus 0.000001 (the constant 0.001 times 0.001).
5. v is the square root of s.

Then `output` is the mean of v over every element: the sum of all v divided by the element count batch_size times 4 times height times width. No dimension is kept. The result is a single value.

Constraints:
- No value is clamped and no input value range is assumed. Any finite fp32 input is valid.
- s is at least 0.000001, so the square root never takes a negative argument.
- Each call writes `output` in full. Nothing carries over from an earlier call.

Output:
- `output`: fp32, a single element (0-d scalar), the mean described above.

## What you must write
`/app/solution.cu` must define this function, with exactly this signature (plain C++ linkage, not `extern "C"`):

```cpp
void reference(const float* inp, const float* target, int64_t batch_size, int64_t height, int64_t width, float* output);
```

- Every pointer is a device pointer to contiguous row-major data.
- Launch your work on the default stream. The grader synchronizes after each call.
- The grader compiles your file with a small wrapper `shim.cu` that includes `solution.cu` and exposes the function to Python:
  `nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu`
- You may use the CUDA runtime, CUB and Thrust. You may not call PyTorch from `solution.cu`.

## How it is graded
1. **Inputs.** Every input below runs in its own process, with a 300 s limit. The grader calls your function 3 times at the same pointers: with the input's values, with new values from the same distribution, then with the first values again. Identical bits on the third call are reported as a column, not failed.
2. **Correctness.** Every element of every output must satisfy `abs(out - ref) <= 0.001 + 0.001 * abs(ref)`, where `ref` comes from `reference_exact()` in `reference.py` (the same math in fp64). NaN must match NaN. Outputs are filled with 0xFF bytes before each call, so every element must be written on every call.
3. **Integrity.** Guard bytes sit before and after every output; writing outside an output fails. Inputs must not change. Allocation inside the call is allowed; it is counted and reported, not failed.
4. **Memory checks.** compute-sanitizer memcheck, racecheck, synccheck and initcheck run on the input marked "sanitizer". Any error fails.
5. **Pass** = correct on every input, in every call, plus the integrity and memory checks.
6. **Speed (reported, does not decide pass or fail).** On the input marked "timed", your kernel, the domain expert's reference kernel and the PyTorch eager `reference()` are timed in the same run on the same GPU. Only a kernel that passes is timed: 3 rounds, interleaving the three. The goal is **at least 1.2x as fast as PyTorch eager**. The speedup and the percentage of the reference kernel's speed are reported next to each answer.

## Inputs
| Input | batch_size | height | width | scale | Marks |
|---|---|---|---|---|---|
| `b4_h4_w4` | 4 | 4 | 4 |  |  |
| `b16_h4_w8` | 16 | 4 | 8 |  |  |
| `b1_h1_w1` | 1 | 1 | 1 |  |  |
| `b3_h129_w257` | 3 | 129 | 257 |  | sanitizer |
| `b8_h64_w64_scale1000` | 8 | 64 | 64 | 1000.0 |  |
| `b16_h512_w512` | 16 | 512 | 512 |  | timed |

Values: `inp` and `target` are uniform in [0, scale), scale 1 unless the table says otherwise.

## Files in /app
- `reference.py`: the PyTorch definition. `reference(inp)` is the eager fp32 version (the speed baseline); `reference_exact(inp)` is the one the grader compares with; `make_inputs(case, seed, device)` builds an input.
- `public_cases.json`: 2 small public inputs, the signature, outputs and tolerance.
- `check.py`: `python3 check.py` builds your `solution.cu` the same way, runs the public inputs, compares with `reference_exact()` and prints a rough speedup. Passing it does not mean passing the grader.
