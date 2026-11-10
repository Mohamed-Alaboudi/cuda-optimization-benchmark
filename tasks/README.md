# Tasks: 6 CUDA kernel tasks

The tasks, the operation text, the prompt frames, the input lists, `reference.py` (PyTorch), `check.py` and the Harbor files are the domain expert's. File formats: `FORMAT.md`.

| Id | Operation | Op source (licence) | Reference kernel (domain expert's) | Tolerance |
|---|---|---|---|---|
| pow_abs_sqrt_mean | mean of sqrt((inp - target)^2 + 1e-6) | KernelBook (op only) | one fused pass, last block finishes the mean | atol 1e-3, rtol 1e-3 |
| leaky | LIF spiking layer, T steps, reset by subtraction | snnTorch `Leaky` (MIT) | one thread per neuron; `__fmul_rn` / `__fadd_rn` keep the stated order | atol 1e-6, rtol 1e-5 |
| efficient_kan | KANLinear forward (SiLU base + B-spline term) | efficient-kan (MIT) | builds bases, then one cuBLAS fp32 GEMM (pedantic math); caches buffers | atol 1e-5, rtol 1e-5 |
| fp8_linear | E4M3 linear, fp32 accumulate, bf16 out | TransformerEngine `Linear` (Apache-2.0) | dequantize, cuBLASLt GEMM with TF32 compute and bias epilogue, cast to bf16 | atol 1e-2, rtol 1e-2 |
| ppo_clip | PPO clipped loss + 5 metrics | TorchRL `ClipPPOLoss` (MIT) | one reduction pass plus a finalize kernel; allocates its scratch in every call | atol 1e-5, rtol 1e-4 |
| reward_to_go | discounted reward-to-go with done flags | TorchRL `reward2go` (MIT) | one block per row, whole row in shared memory, fp64 scan | atol 1e-5, rtol 1e-5 |

## What changed from the expert's original task files
- **Inputs.** Each task now has 4 to 6 stated inputs with seeds (`tests/task.json`): a tiny edge case, odd sizes, one sanitizer input and the original performance size as the timed input. Values follow the expert's original distributions.
- **Input fixes where PyTorch is wrong:** `reward_to_go` uses gamma in [0.5, 0.99], never 0, and gets a fresh PyTorch reference instead of the TorchRL file. `ppo_clip` cuts log weights to [-20, 20], so `exp()` stays finite. Both are stated in the prompts.
- **Correctness reference.** `reference_exact()` is the same math in fp64 (`leaky` stays fp32: the step order is the spec). On the CPU, the fp32 eager `reference()` sits inside the stated tolerance on every case run (worst 0.49 of the tolerance, `fp8_linear`, bf16 rounding; measured, CPU, torch 2.15).
- **Shape changes.** `efficient_kan` varies B (1 to 4,096) and has one I = 100 input; the original prompt said the shapes are static. `reward_to_go` stops at T = 8,000 because the expert kernel keeps the whole row in shared memory without the opt-in.
- **Libraries.** The prompts allow cuBLAS for `efficient_kan` and cuBLASLt for `fp8_linear`, because the expert's kernels use them.

## Network
Each `task.toml` says the model has model-API access only. In the real run the pod had open network access, but the models did not use it (the run notes say so; not re-checked per transcript).