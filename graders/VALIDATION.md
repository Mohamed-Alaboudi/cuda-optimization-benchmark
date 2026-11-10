# VALIDATION

Status: filled from the real runs on the pod (NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 580.178.04, nvcc 12.8.93, compute-sanitizer 2025.1.0, torch 2.9.1+cu128, Python 3.12.3, sm_120). Times are ET, Sun Nov 8, taken from the result files.

## Measured on the Mac, Sun Nov 8 about 18:50 ET (no GPU)
- `sh graders/compile_check.sh` in `nvidia/cuda:12.9.2-devel-ubuntu24.04` (colima): 73 ok, 0 failed. the domain expert's 6 kernels and the 30 planted kernels each compile for sm_120 through the grader's shim with each task's link flags and export `grader_entry`; the call counter compiles.
- `pytest graders/tests` (torch on the CPU): 41 passed (40 first, plus the sanitizer summary test added after P4). They cover signature parsing and the shim, every label from a case record, flip, compile_error, sanitizer errors and timeouts, the sanitizer summary lines (racecheck "RACECHECK SUMMARY", first "ERROR SUMMARY" line), the speed and expert columns, regrade tables, finding kernels in a folder, extraction, the comparison (tolerance edges, NaN, bf16, 0xFF fill, wrong shape), each task's own files on its small case, and the whole `proof` flow with a stand-in for the GPU process.

## GPU proof (`results/gpu_proof.json`, final run 23:39 ET on the reference kernels that ship)
| Control | Expected | Result |
|---|---|---|
| the domain expert's 6 kernels, every input, 3 of 3 passes, 4 sanitizer tools clean | pass | 6 of 6 pass |
| 30 planted kernels (5 per task) | each fails with its label (`README.md`) | 30 of 30 fail with the expected label (wrong value, stale state, out-of-bounds write, race, uninitialized read) |
| timing spread of the domain expert's kernels over 3 rounds | small (reported) | under 0.2% for efficient_kan, leaky, pow_abs_sqrt_mean, reward_to_go; 5.7% fp8_linear; 24% ppo_clip |
| allocations per call (column) | ppo_clip allocates every call; efficient_kan and fp8_linear on the first call | efficient_kan 2, fp8_linear 5, ppo_clip 2, leaky 0, pow_abs_sqrt_mean 0, reward_to_go 0 |

The first P4 run marked every kernel `grader_error`. The cause was a grader bug: racecheck in compute-sanitizer 2025.1 prints "RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)", not "ERROR SUMMARY". The grader also took the second initcheck "ERROR SUMMARY" line (44 not printed) instead of the first (64). Both are fixed, with a test, before the run above.

Speed of the reference kernels in P4 (x vs PyTorch eager, timed input): efficient_kan 3.60, fp8_linear 2.90, leaky 13.15, pow_abs_sqrt_mean 12.46, ppo_clip 1.28, reward_to_go 66.80.

## Grading of the model tries (P6, `results/grade_logs/`, written 20:18 ET)
| Control | Expected | Result |
|---|---|---|
| 54 try kernels (3 models x 6 tasks x 3 tries) | graded, each with a verdict | 54 of 54 pass |
| within-run timing spread over 3 rounds | small (reported) | under 0.6% for 32 of 54; 0.6% to 1.4% for 13; 3.3% to 7.2% for 7; about 50% for 2 |

## Fresh-push check (`results/grade_logs_fixcheck/`, written 20:25 ET)
| Control | Expected | Result |
|---|---|---|
| fresh copy without `graders/build/`: `grade-dir` builds `interpose.so` itself | kernels called and graded | 12 of 12 graded pass (the domain expert's 6 kernels, 6 model tries) |
| the 6 model tries match their P6 verdicts | same status | 6 of 6 same status; speed within 0.1% to 1.4% of P6 |

Before this fix, a missing `interpose.so` meant the kernel was never called, so every output read as unwritten.

## Timing noise
| Measure | Result |
|---|---|
| the domain expert's kernel vs itself ("% of expert speed" when the expert is graded as a try) | 99.9% to 100.2% on 5 tasks; ppo_clip 89.4% (P4) and 58.3% (fresh-push check) |
| within-run spread over 3 rounds | under 0.6% for most kernels (see the P6 row); ppo_clip is the noisy task |
| run-to-run, same kernel | 1% to 4% (measured earlier); P6 vs fresh-push check measured here: 0.1% to 1.4% on the 6 tries |

So a speed gap under about 5% is timing noise, and ppo_clip speed numbers need several runs (pick.md uses the median of 4).

## Known limits (for the README)
- The grading process uses torch for the inputs and the references; the kernel library shares torch's CUDA runtime. On the pod both are CUDA 12.8 and every kernel loaded.
- The call counter sees calls through the CUDA runtime only, and it is a column: reference kernels allocate inside the call, so allocation cannot be a fail.
- Planted race and uninitialized-read kernels keep the values right; only compute-sanitizer can catch them, on the small case. On the pod it caught all 12.
- Not a sandbox: a kernel runs as the grading user. Model kernels never see `tests/` or the expert kernel.
