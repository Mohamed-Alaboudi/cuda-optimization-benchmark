# GPU proof and grading on the pod (P4, P6).

The pod is started only after the owner approves. No API keys go to the pod.

## Once per pod (from `submission/`)
```
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv     # 12.0 means ARCH=sm_120
nvcc --version && compute-sanitizer --version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python3 graders/grade.py build
```

## P4: the GPU proof (before any model try)
```
ARCH=sm_120 python3 graders/grade.py proof          # writes results/gpu_proof.json and results/gpu_proof_logs/
python3 -c "import json; print(json.load(open('results/gpu_proof.json'))['summary'])"
cat results/gpu_proof_logs/table.md
```
Go when `references_pass` is 6 of 6 and `planted_as_expected` is 30 of 30. A reference that fails: tell the owner (the domain expert decides; the kernel is theirs). A planted kernel not caught: the grader author fixes the check or the README says that check is stated, not enforced.
Time: about 10 to 25 min, I think (not measured; compute-sanitizer with torch is the slow part).

## P6: grade every model try
```
ARCH=sm_120 python3 graders/grade.py grade-dir --kernels results/trials --out results/grade_logs
```
Resumable: rerun the same command after a stop; logged kernels are skipped, `grader_error` ones are redone.

## Anywhere, no GPU
`python3 graders/grade.py regrade --logs results/grade_logs` rebuilds `results.json`, `verdicts.json` and `table.md` from the logs.
