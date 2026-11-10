# graders/

One code grader for the 6 tasks. It scores the models' kernels and the domain expert's kernels the same way, in the same run, on the same GPU. There is no LLM judge.

## Pass rule
Pass = every stated input correct (vs `reference_exact` under the task's stated tolerance) on every call, inputs and guard bytes intact, and compute-sanitizer clean on the small input. Speed vs PyTorch eager, % of the expert's speed, identical repeat bits and allocations per call are reported columns; they never decide pass or fail.

## Commands (from `submission/`, on the GPU host; needs torch, nvcc, compute-sanitizer; no API keys)
| Command | What it does |
|---|---|
| `python graders/grade.py build` | builds the call counter (`build/interpose.so`) |
| `python graders/grade.py proof` | the GPU proof: the domain expert's 6 kernels (3 passes, sanitizer, timing) and 30 planted kernels; writes `results/gpu_proof.json` and `results/gpu_proof_logs/` |
| `python graders/grade.py grade-dir --kernels DIR --out LOGDIR` | every `.cu` under DIR (the task id is a path part, e.g. `DIR/<model>/<task>/try1.cu`); one log per kernel, `LOGDIR/results.json`, `LOGDIR/table.md`; resumable |
| `python graders/grade.py regrade --logs LOGDIR` | verdicts and tables again from the logs only; no GPU, no torch |
| `sh graders/compile_check.sh` | compile check without a GPU (CUDA devel container) |
`ARCH` (default `sm_120`), `PASSES` (default 1 for model kernels; experts use 3), `CASE_TIMEOUT`, `SANITIZER_TIMEOUT` are environment variables.

## One kernel, step by step
1. **Compile** once per source hash: `nvcc -O3 -arch=$ARCH -shared -Xcompiler -fPIC -cudart shared` plus the task's `link` flags (`-lcublas` for efficient_kan, `-lcublasLt` for fp8_linear), on a grader-written shim: `#include "solution.cu"` and an `extern "C"` entry that calls the C++ `void reference(...)` with the task's exact signature. A wrong signature is a `compile_error`.
2. **Each input case in its own process** (`child.py`: torch + ctypes, the call counter preloaded, a fresh HOME and TMPDIR):
   - inputs from `make_inputs(case, seed, "cuda")`, copied into buffers with 4 KB of guard bytes before and after; outputs filled with 0xFF bytes (NaN for float types) and guarded too;
   - call 1 with the seed's inputs, call 2 with new values (seed + 7919) at the same pointers (catches cached results and outputs not written), call 3 with the first inputs again (identical bits is a column);
   - after each call: CUDA errors, guard bytes, inputs unchanged, the call counter (allocations, host copies, device syncs: a column, because the domain expert's kernels may allocate inside the call);
   - only after the kernel calls: `reference_exact` for calls 1 and 2 and the comparison of every element. The expected output never exists while the kernel runs.
3. **compute-sanitizer** memcheck, racecheck, synccheck and initcheck on the task's small case (inputs made on the CPU, so the only kernels in the run are the graded ones and torch fills).
4. **Timing** (passing kernels only) on the timed case: 3 rounds; each round times the kernel, the domain expert's kernel and PyTorch eager `reference()` one after another, with CUDA events around K back-to-back calls (K so a batch takes at least 1 ms), the median of 10 batches. Columns: median of the 3 rounds, spread (max / min - 1).

## Statuses and labels
Statuses: `pass`, `fail`, `compile_error`, `timeout`, `flip` (passes disagree), `grader_error` (rerun, never counted).
Labels (first failing check): cuda error, out-of-bounds write, input modified, crash, wrong shape, unwritten output, wrong value, stale state (call 2 wrong), then sanitizer: memory error, race, sync error, uninitialized read.

## Planted kernels (`planted.py`)
Made from each task's expert kernel, so they compile against the same signature. 5 per task, 30 in all:
| Planted | What it does | Expected label |
|---|---|---|
| wrong_value | expert, then the first output element set to a far value | wrong value |
| cached_result | computes on the first call only, then returns | stale state |
| oob_write | expert, then writes the byte just before the output | out-of-bounds write |
| race | expert, then a shared-memory read of another warp's slot with no `__syncthreads()` (values unchanged) | race |
| uninit_read | expert, then reads a fresh `cudaMalloc` buffer that was never written (values unchanged) | uninitialized read |

## Files
`grade.py` (CLI, compile, processes, sanitizer, folders, proof), `child.py` (one graded process), `verdict.py` (verdicts and tables, plain Python), `planted.py`, `interpose.c` (call counter), `extract.py` (code block from model text), `compile_check.sh`, `tests/` (CPU tests). `build/` is made at run time and is not shipped.
