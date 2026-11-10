#!/usr/bin/env python3
"""Dry run of the runner on any machine: stub models, CPU stand-in tasks, no keys, no GPU, no pod.
Copies run_eval.py and runner/ to a temp folder with 2 stand-in tasks, runs the trials twice, and checks:
done status, the wall cap, the per-command timeout, writes outside the folder refused, the cost cap,
resume without repeats, one result and one transcript per try.
Usage: python3 runner/dry_run.py [--executor local|ssh --host H --port P]"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--executor", default="local")
ap.add_argument("--host"), ap.add_argument("--port")
a = ap.parse_args()
tmp = Path(tempfile.mkdtemp(prefix="runner-dry-"))
shutil.copy(SUB / "run_eval.py", tmp)
shutil.copytree(SUB / "runner", tmp / "runner", ignore=shutil.ignore_patterns("__pycache__"))
for t in ("standin_a", "standin_b"):
    (tmp / "tasks" / t).mkdir(parents=True)
    (tmp / "tasks" / t / "prompt.md").write_text(f"Stand-in task {t}: write solution.cu.\n")
    (tmp / "tasks" / t / "environment").mkdir()
    (tmp / "tasks" / t / "environment" / "reference.py").write_text("print('reference')\n")
    (tmp / "tasks" / t / "environment" / "Dockerfile").write_text("FROM x\n")
(tmp / "tasks" / "standin_a" / "expert_solution.cu").write_text("// must never reach the try folder\n")
extra = ["--executor", a.executor] + (["--host", a.host, "--port", a.port] if a.executor == "ssh" else ["--local-dir", str(tmp / "tries")])


def trials(models, wall):
    cmd = [sys.executable, "run_eval.py", "trials", "--models-file", "runner/models.stub.json", "--models", models,
           "--tasks", "standin_a,standin_b", "--tries", "2", "--wall-min", str(wall), "--workers", "4", "--per-model", "2",
           "--results", "results/trials", "--transcripts", "transcripts"] + extra
    p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env={**os.environ, "MICRO_LOGS": str(tmp / "logs")})
    return p.returncode, p.stdout + p.stderr


checks = []
rc, out = trials("stub-ok", 2)
print(out[-1500:])
res = {p.relative_to(tmp / "results" / "trials").as_posix(): json.loads(p.read_text()) for p in (tmp / "results" / "trials").rglob("*.json")}
checks.append(("4 tries of stub-ok end with status done", len(res) == 4 and all(r["status"] == "done" for r in res.values())))
checks.append(("each try has solution .cu and transcript", all((tmp / "results" / "trials" / k).with_suffix(".cu").exists()
                                                             and (tmp / r["transcript"]).exists() for k, r in res.items())))
tr = json.loads((tmp / "transcripts" / "stub-ok" / "standin_a" / "try1.json").read_text())
tools = [e for e in tr["events"] if e["kind"] == "tool"]
checks.append(("write outside the folder refused", any(e["name"] == "write_file" and "error" in e["result"] for e in tools)))
checks.append(("per-command timeout enforced (sleep 3 with timeout_s 1)", any(e.get("exit_code") == 124 for e in tools)))
checks.append(("expert file not in the try folder", "expert_solution" not in tr.get("final_listing", "")))
checks.append(("transcript has prompt, system and model responses", bool(tr["prompt"] and tr["system"] and
                                                                         any(e["kind"] == "model" for e in tr["events"]))))
rc2, out2 = trials("stub-ok", 2)
checks.append(("rerun skips all 4 finished tries", '"skip": 4' in out2))
rc3, out3 = trials("stub-loop", 0.25)
res3 = [json.loads(p.read_text()) for p in (tmp / "results" / "trials" / "stub-loop").rglob("*.json")]
checks.append(("looping stub stops at the wall cap or cost cap", len(res3) == 4 and all(r["status"] in ("wall_cap", "cost_cap") for r in res3)))
checks.append(("wall cap holds (each try under cap + 30 s)", all(r["seconds"] < 0.25 * 60 + 30 for r in res3)))
print(json.dumps([r["status"] for r in res3]), [r["seconds"] for r in res3])
ok = all(c for _, c in checks)
for name, c in checks:
    print(("PASS " if c else "FAIL ") + name)
print("DRY RUN:", "PASS" if ok else "FAIL", "| folder", tmp)
sys.exit(0 if ok else 1)
