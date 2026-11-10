#!/bin/bash
# Hidden test entry: grades /app/solution.cu with the shared grader on the hidden cases in tests/task.json.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /logs/verifier
cd /submission
python3 graders/grade.py build
python3 graders/grade.py grade-dir --task reward_to_go --kernels /app --out /logs/verifier/reward_to_go
python3 -c "import json,sys; v=json.load(open('/logs/verifier/reward_to_go/verdicts.json')); print(1 if v.get('reward_to_go',{}).get('status')=='pass' else 0)" > /logs/verifier/reward.txt
