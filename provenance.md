# Provenance

## Built by the author
The 6 tasks and their reference kernels (`reference/<task>/solution.cu`) are the author's work as a CUDA engineer. The kernels are in `reference/<task>/solution.cu`. The op sources and licences:

| Task | Expert kernel, git dates | Op source and licence | Changed for this eval |
|---|---|---|---|
| `pow_abs_sqrt_mean` | 2026-10-14 to 2026-10-21 | KernelBook op | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |
| `leaky` | 2026-06-10 | snnTorch `Leaky` (MIT) | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |
| `efficient_kan` | 2026-07-17 | efficient-kan `KANLinear` (MIT) | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |
| `fp8_linear` | 2026-07-17 | NVIDIA TransformerEngine `Linear` (Apache-2.0) | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |
| `ppo_clip` | 2026-07-17 | TorchRL `ClipPPOLoss` (MIT) | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |
| `reward_to_go` | 2026-07-17 | TorchRL reward-to-go (MIT) | Packaged as a standalone task with hidden tests; kernel tuned for the RTX PRO 6000 |

The PyTorch references (`tasks/<task>/reference.py`) are trimmed from the public libraries in the table. All inputs are synthetic, made by each task's input generator.

## Built Oct 30 to Nov 8, 2026
- the 6 prompts for this eval, which state every check the grader runs
- the grader (correctness, integrity checks, compute-sanitizer, speed vs PyTorch eager and vs the expert kernel)
- the agent-loop runner (compile, test, time; 20 minutes per try, 3 tries per task) and the RunPod scripts
- all results and this README (full transcripts are not in this copy; the kernels the models wrote are in `results/trials/`)

