#!/bin/bash
# Hidden test entry: grades /app/solution.cu with the shared grader on the hidden cases in tests/task.json.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /logs/verifier
cd /submission
python3 graders/grade.py build
python3 graders/grade.py grade-dir --task ppo_clip --kernels /app --out /logs/verifier/ppo_clip
python3 -c "import json,sys; v=json.load(open('/logs/verifier/ppo_clip/verdicts.json')); print(1 if v.get('ppo_clip',{}).get('status')=='pass' else 0)" > /logs/verifier/reward.txt
