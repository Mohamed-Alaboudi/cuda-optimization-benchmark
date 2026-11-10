
# CUDA Optimization Benchmark: expert kernels vs frontier agents on 6 real PyTorch ops

## 1. Short description
An agentic eval of kernel optimization, presented through CUDA. Each of the 6 tasks asks for a CUDA kernel that replaces a real PyTorch op: exact on every input, clean under compute-sanitizer, and faster than PyTorch eager. Three frontier models work alone in an agent loop with a compiler, a test script and a timer, 20 minutes per try, 3 tries per task. Their kernels are graded next to my own kernels on the same GPU, and the gap to my kernels is the score that matters.

## 2. Domain and expertise
Built by Mohamed Alaboudi, an independent CUDA engineer. The 6 ops come from my own kernel work; the reference kernels are written and tuned for the GPU this eval runs on. The question the eval answers: how close do frontier models get to an expert's kernels?

## 3. What it tests and why it matters
**What it tests.** Can a frontier model, working alone with a compiler, a test script and a timer, write a CUDA kernel that:
- matches a PyTorch op on every stated input, within a stated tolerance,
- keeps its memory contract (no writes outside its outputs, no changed inputs, no stale results),
- is clean under compute-sanitizer, and
- runs faster than PyTorch eager?

**Who it is for.** Companies that want AI models that can optimize kernels (CUDA in this case): inference and training teams deciding whether a model can stand in for the engineer who hand-tunes their hot ops, and what a slow or wrong kernel costs them.

**The bar.** Every task has an expert kernel that I wrote before this eval. The grader runs it in the same run, on the same GPU, as the model kernels. So every result has two yardsticks: PyTorch eager, and the expert.

## 4. Setup and run instructions
Run from this folder. The trials run on any machine with Python 3.10 to 3.13; the model calls go out from it. Only the tools (compile, test, time) and the grading run on the GPU pod.
```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # fill in the keys, then: set -a; . ./.env; set +a
python run_eval.py pod create --confirm                # 1 x RTX PRO 6000 on RunPod
python run_eval.py trials --executor ssh --host HOST --port PORT   # 3 models x 6 tasks x 3 tries
python run_eval.py grade build
ARCH=sm_120 python run_eval.py grade proof             # expert kernels pass, planted kernels fail
ARCH=sm_120 python run_eval.py grade grade-dir --kernels results/trials --out results/grade_logs
python run_eval.py grade regrade --logs results/grade_logs   # verdicts again from the logs, no GPU
python run_eval.py pod terminate
```
- **Keys** (names only, in `.env.example`): `OPENAI_API_KEY`, `KIMI_API_KEY`, `RUNPOD_API_KEY`. No key goes on the pod.
- **Claude row:** model `claude-opus-5-5`, effort high (`models.json`), called through Anthropic's API. To rerun, set `ANTHROPIC_API_KEY` and use provider `anthropic_api` (`models.json`, `runner/agent_loop.py`).
- **GPU host:** NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), driver 580.178.04, CUDA 12.8 (nvcc 12.8.93), PyTorch 2.9.1+cu128, image `runpod/pytorch:1.5.1-cu1281-torch291-ubuntu2404`. GPU-side packages: `requirements-gpu.txt`.
- **Regrade without a GPU:** the shipped `results/grade_logs/` hold every grader measurement, so `regrade` rebuilds every verdict and table.
- **Runtime and cost of a full run:** about 2 hours of pod time ($3.30 at $1.69/h: trials about 55 min, grading about 30 min), plus $3.28 of OpenAI calls.
Verified once from a clean copy on a fresh pod (Nov 8, 2026): the grader builds its own tools, and 12 of 12 verdicts match the main run.

## Task overview
These ops look simple, and that is the trap: each is memory-bound or dependency-bound, so the gap between a correct kernel and a fast one is layout, vectorization, and knowing which hardware shortcuts the tolerance allows, not the math.

| Task | What the kernel computes | Op source (licence) | Why it is hard |
|---|---|---|---|
| `pow_abs_sqrt_mean` | mean of sqrt((input - target)^2 + 1e-6) over a large tensor | KernelBook op | Fused reduction chains like this sit inside normalization and loss blocks. Hard because it is memory-bound: the win is vectorized loads plus the approximate-sqrt instruction the tolerance barely allows; a 'correct' sqrtf leaves 4x on the table |
| `leaky` | leaky integrate-and-fire spiking layer over T steps, reset by subtraction | snnTorch `Leaky` (MIT) | Spiking networks pay this cost every timestep. Hard because the T loop is sequential per neuron: the win is layout and vectorized loads, not more parallelism |
| `efficient_kan` | KAN linear layer forward: SiLU base path plus B-spline term | efficient-kan `KANLinear` (MIT) | KAN layers are a proposed MLP replacement. Hard because the spline term is many small ops: fusing them without changing the rounding is where the speed is |
| `fp8_linear` | FP8 (E4M3) linear: fp32 accumulate, bias, BF16 output | NVIDIA TransformerEngine `Linear` (Apache-2.0) | fp8 matmul is the main lever for cutting LLM inference cost. Hard because the matmul itself is library work: the speed is in killing launch overhead with a cached CUDA graph |
| `ppo_clip` | PPO clipped policy loss plus 5 metrics | TorchRL `ClipPPOLoss` (MIT) | RL trainers recompute this loss every batch. Hard because six metrics must come out of one pass: keeping accumulators in fp32 while fusing separates 8x from 52x |
| `reward_to_go` | discounted reward-to-go with done flags | TorchRL reward-to-go (MIT) | Every RL pipeline computes discounted returns. Hard because the obvious one-thread-per-row scan is correct but uncoalesced: 200x needs a parallel scan layout |

Each task folder (`tasks/<task>/`) follows the Harbor layout: `instruction.md` and `prompt.md` (what the model sees), `environment/` (the PyTorch reference, a public check script on small inputs, the build command), `tests/` (hidden inputs and the grader entry, copied in only for grading) and `solution/` (the expert kernel). Each task has 4 to 6 seeded inputs: a tiny edge case, odd sizes, one sanitizer input, and the timed performance size. Two input ranges are limited: `reward_to_go` uses gamma in [0.5, 0.99] (PyTorch's reference is wrong at gamma 0) and T up to 8,000; `ppo_clip` clips log weights to [-20, 20] so `exp()` stays finite. These limits also fit the expert kernels. All are stated in the prompts. My notes on each task: `tasks/<task>/why.md`.

## Grading
One code grader scores the model kernels and the expert kernels the same way. There is no LLM judge.
- **Pass** = every stated input correct (against an fp64 reference, except `leaky`, whose reference is fp32 step by step, within the task's stated tolerance) on every call, inputs and guard bytes intact, and compute-sanitizer clean (memcheck, racecheck, synccheck, initcheck) on the small input.
- **Each input runs in its own process.** Outputs start filled with NaN and are fenced by 4 KB guard bytes. The kernel is called 3 times: the seed's inputs, new values at the same pointers (catches cached results and unwritten outputs), then the first inputs again. The expected output is computed only after the kernel calls, so it never exists while the kernel runs.
- **Reported, never part of pass:** speed vs PyTorch eager (the prompt states a goal of 1.2x), % of the expert's speed, identical bits on repeat, and allocations per call. Timing: 3 rounds; each round times the model kernel, the expert kernel and PyTorch one after another with CUDA events; median and spread.
- **Agent loop:** each model works alone in a fresh folder, as its own unix user, with 3 tools: `bash`, `write_file`, `done`. 20 minutes per try, 3 tries per task. The kernel left at the end of a try is graded. Every response and tool call is in the transcript, with the prompt and system prompt each model got (the Claude row also got the fixed line above).
- **How the grader was validated:** on the RTX PRO 6000, the 6 reference kernels pass every check, and 30 planted wrong kernels (5 per task: wrong value, cached result, out-of-bounds write, race, uninitialized read) all fail with the right label. The planted kernels are generated from each task's reference kernel.

## 5. Models tested and results
| Model (exact id) | Settings | Tries per task | pass^3 | Speed vs PyTorch | Median % of reference (all 18 tries) | % of reference (min to max) |
|---|---|---|---|---|---|---|
| gpt-6.1-sol | reasoning effort high | 3 | 1.00 | 13.8x | 92% | 55% to 101% |
| claude-opus-5-5 | effort high | 3 | 1.00 | 9.7x | 40% | 14% to 99% |
| Kimi K3 | thinking, effort high | 3 | 1.00 | 9.3x | 49% | 2% to 91% |
| Reference kernel | | | 6/6 (proof run) | | 100% | |

Temperature: provider default (not sent). Run date: Nov 8, 2026. "% of reference speed" divides each kernel's speed vs PyTorch (its own grading run) by the reference kernel's speed (`results/reference_speeds.json`). Both were timed on the same GPU model and driver; on a different GPU or driver the reference speeds move and the shares move with them. Averages per task over the 3 tries: `results/avg_vs_reference.md`; per-task table: `results/table.md`; per-task notes: `results/per_task.md`; the kernels the models wrote: `results/trials/`.

## 6. Failure analysis
All 54 tries produced correct, sanitizer-clean kernels, so every failure below is speed, measured against my reference kernels on the same GPU. Percentages are mean-of-3 vs the reference; the per-try kernels are in `results/trials/`.

- `pow_abs_sqrt_mean`: Opus averaged 40%. Two of its three tries landed near 25% because the code chose the IEEE-rounded `sqrtf`; the tolerance allows the hardware approximate sqrt my kernel uses (`results/trials/claude-opus-5-5/pow_abs_sqrt_mean/try1.cu`).
- `leaky`: the tightest task, models at 83-95%. The slower versions walk the T timesteps per lane without vectorized loads (`results/trials/kimi-k3/leaky/try1.cu`).
- `efficient_kan`: sol 70%, Opus and Kimi about 40%. The slow versions rely on generic kernels plus library calls; my kernel fuses the spline-basis application into custom kernels (`results/trials/kimi-k3/efficient_kan/try1.cu`).
- `fp8_linear`: sol 93%, Opus 62%, Kimi 66%. None of the models cached a CUDA graph for the fixed-shape GEMM, which is where my kernel gets its edge.
- `ppo_clip`: Opus 17%, Kimi 11%, sol 70%. The slow tries carry double-precision math through the inner reduction, expensive on this GPU (`results/trials/claude-opus-5-5/ppo_clip/try1.cu`).
- `reward_to_go`: the widest spread. Kimi's tries ranged from 2% to 54% on the same task; the slow one assigns one thread per sequence, so global loads are uncoalesced (`results/trials/kimi-k3/reward_to_go/try2.cu`). Meanwhile gpt-6.1-sol reached 99% of my kernel here, the closest any model came on any task.

### Main failure mode and hot take
The models write correct CUDA on every single try: 54 of 54 kernels passed every input, every integrity check, and compute-sanitizer. What they cannot do reliably is write fast CUDA. The same model on the same task swung from 2% to 54% of my kernel's speed between tries, and on 4 of the 6 tasks at least one model averaged under 50% of my kernel. Correctness is solved; optimization judgment is not.

## 7. Prior work and sources
**Built by me:** the 6 tasks and the reference kernels, from my own CUDA work. The kernels are in `reference/<task>/solution.cu`. I've worked on optimizing a few CUDA kernels for various open source projects; for this eval I packaged 6 of these ops as self-contained tasks and tuned my kernels for the RTX PRO 6000 (Blackwell) the eval runs on.

**Also built by me:** the eval prompts, the grader, the agent-loop runner and RunPod scripts, the results and this README. Details: `provenance.md`.

**Public sources:** KernelBook; snnTorch (MIT); efficient-kan (MIT); NVIDIA TransformerEngine (Apache-2.0); TorchRL (MIT); the CUDA C++ Programming Guide. All inputs are synthetic.

**Benchmarks we build on or overlap with:** KernelBench and KernelBook also ask models for CUDA kernels of PyTorch ops. What is new here: models are compared against an expert CUDA engineer's kernels, not only against PyTorch eager; they work in an agent loop with real tools instead of single-shot generation; and every kernel must pass planted-bug integrity checks and compute-sanitizer, so cached results and memory bugs count as failures.

