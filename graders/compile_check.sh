#!/bin/sh
# Compile check without a GPU: the shim around every expert kernel and every planted kernel,
# with each task's Build block, plus the call counter. Run from submission/ in the CUDA devel image:
#   docker run --rm -v "$PWD":/w -w /w nvidia/cuda:<version>-devel-ubuntu24.04 sh graders/compile_check.sh
# Needs python3 in the image (apt-get install -y python3) for the shim and planted generation.
ARCH=${ARCH:-sm_120}
OUT=${OUT:-/tmp/compile_check}
mkdir -p "$OUT"
python3 - "$OUT" <<'EOF'
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, ".")
from graders import grade
from graders.planted import write_planted
out = Path(sys.argv[1])
grade.BUILD = out
ok = bad = 0
def note(good, what, log=""):
    global ok, bad
    ok, bad = ok + good, bad + (not good)
    print(("ok    " if good else "FAIL  ") + what + ("" if good else "\n" + log[-1500:]), flush=True)
ids = grade.task_ids()
items = [(t, grade.TASKS / t / "solution" / "solution.cu") for t in ids]
items += write_planted(out / "planted", ids)
for t, p in items:
    if not p.exists():
        note(False, f"{t}: missing {p}")
        continue
    rec = grade.compile_kernel(grade.load_task(t), p, "sm_120")
    note(rec["ok"], f"{t}: {p.relative_to(out) if out in p.parents else p.relative_to(grade.SUB)}", rec["log_tail"])
    if rec["ok"]:
        syms = subprocess.run(["nm", "-D", rec["lib"]], capture_output=True, text=True).stdout
        note(" T grader_entry" in syms, f"{t}: grader_entry exported")
r = subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-I/usr/local/cuda/include", "-o", str(out / "interpose.so"), "graders/interpose.c", "-ldl"], capture_output=True, text=True)
note(r.returncode == 0, "graders/interpose.c", r.stderr)
print(f"compile check: {ok} ok, {bad} failed")
EOF
