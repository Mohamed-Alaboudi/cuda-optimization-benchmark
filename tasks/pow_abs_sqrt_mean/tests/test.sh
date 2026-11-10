#!/bin/bash
# Hidden test entry: grades /app/solution.cu with the shared grader on the hidden cases in tests/task.json.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /logs/verifier
cd /submission
python3 graders/grade.py build
python3 graders/grade.py grade-dir --task pow_abs_sqrt_mean --kernels /app --out /logs/verifier/pow_abs_sqrt_mean
python3 -c "import json,sys; v=json.load(open('/logs/verifier/pow_abs_sqrt_mean/verdicts.json')); print(1 if v.get('pow_abs_sqrt_mean',{}).get('status')=='pass' else 0)" > /logs/verifier/reward.txt
