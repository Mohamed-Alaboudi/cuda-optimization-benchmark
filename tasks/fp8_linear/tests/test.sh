#!/bin/bash
# Hidden test entry: grades /app/solution.cu with the shared grader on the hidden cases in tests/task.json.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /logs/verifier
cd /submission
python3 graders/grade.py build
python3 graders/grade.py grade-dir --task fp8_linear --kernels /app --out /logs/verifier/fp8_linear
python3 -c "import json,sys; v=json.load(open('/logs/verifier/fp8_linear/verdicts.json')); print(1 if v.get('fp8_linear',{}).get('status')=='pass' else 0)" > /logs/verifier/reward.txt
