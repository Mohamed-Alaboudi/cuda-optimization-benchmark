# Task file formats (shared by tasks/, graders/ and run_eval.py)

The domain expert owns the tasks, the inputs, the tolerances and the reference kernels.

## Task ids
pow_abs_sqrt_mean, leaky, efficient_kan, fp8_linear, ppo_clip, reward_to_go

## Files per task (`tasks/<id>/`)
| File | Holds |
|---|---|
| `prompt.md` | the prompt the model gets (`instruction.md` is a link to it, for the Harbor layout) |
| `reference.py` | PyTorch: `make_inputs(case, seed, device)`, `reference(inp)` (eager fp32, the speed baseline), `reference_exact(inp)` (the correctness reference) |
| `task.toml` | Harbor settings: 20 min per try, 1 GPU, verifier limits |
| `environment/` | what the model sees in `/app`: `reference.py` (link to `../reference.py`), `check.py`, `public_cases.json`, `Dockerfile` |
| `tests/task.json` | hidden: signature, outputs, tolerance, link flags and the full case list with seeds |
| `tests/test.sh` | hidden test entry: runs `graders/grade.py` on `/app/solution.cu` |
| `solution/solution.cu` | link to `../../../reference/<id>/solution.cu`, the domain expert's kernel |
| 
Links are relative; anything that packs a task must follow them (for example `tar -h`, or `zipfile` from `os.walk`, which reads the target).

## `reference.py`
- `make_inputs(case, seed, device) -> dict`: keys are the C argument names of the signature (tensors and Python scalars). Uses `torch.Generator(device).manual_seed(seed)` only.
- `reference(inp) -> dict`: keys are the output argument names; PyTorch eager in fp32 (the speed baseline).
- `reference_exact(inp) -> dict`: the same keys; fp64 where the op allows; `leaky` is fp32 (the step order is the spec); `fp8_linear` is fp64 then one rounding to bfloat16.

## `tests/task.json` (and `environment/public_cases.json`, same schema, 2 small cases)
```json
{"task": "reward_to_go",
 "signature": "void reference(const float* reward, const bool* done, float gamma, int32_t B, int32_t T, float* output);",
 "outputs": {"output": {"dtype": "float32", "shape": ["B", "T"]}},
 "tolerance": {"atol": 1e-05, "rtol": 1e-05},
 "link": [],
 "cases": [{"name": "B64_T1024", "seed": 605, "B": 64, "T": 1024, "gamma": 0.99, "done_p": 0.1, "small": false, "timed": true}]}
```
- `signature`: the reference C++ signature, verbatim. Answers keep plain C++ `void reference(...)`; the grader adds an `extern "C"` shim that includes `solution.cu`.
- `outputs.shape`: names are input argument names or numbers; `[]` is a 0-d scalar.
- Pass rule per element: `abs(out - ref) <= atol + rtol * abs(ref)`, NaN matches NaN.
- `small`: the sanitizer input. `timed`: the speed input. Other fields are inputs to `make_inputs`.
- `link`: extra nvcc flags (`efficient_kan`: `-lcublas`; `fp8_linear`: `-lcublasLt`).

## Build (the grader and `check.py`)
`nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC -cudart shared -o kernel.so shim.cu <link>`, where `shim.cu` starts with `#include "solution.cu"`.
