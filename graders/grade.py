"""Grader for the 6 CUDA tasks. One NVIDIA GPU. No LLM judge.

  python graders/grade.py build                                  # the call counter (interpose.so)
  python graders/grade.py grade-dir --kernels DIR --out LOGDIR   # every .cu under DIR (task id from the path)
  python graders/grade.py experts --out LOGDIR                   # the domain expert's kernels (tasks/<id>/solution/solution.cu)
  python graders/grade.py planted --out LOGDIR                   # the planted wrong kernels (graders/planted.py)
  python graders/grade.py proof --out results/gpu_proof.json     # experts + planted, the GPU proof (P4)
  python graders/grade.py regrade --logs LOGDIR                  # verdicts and tables from logs only (no GPU, no torch)

Per kernel: compile once (Build block plus a grader-written extern "C" shim around the
C++ `void reference(...)`), then every input case in its own Python process (torch + ctypes):
outputs filled with 0xFF bytes and fenced by guard bytes, call 1 (seed), call 2 (seed + 7919,
new values at the same pointers), call 3 (seed again: identical bits is a column), then the
comparison with reference_exact() under the task's tolerance, computed only after the kernel
has run. Then compute-sanitizer (memcheck, racecheck, synccheck, initcheck) on the small case,
then timing on the timed case: 3 rounds, each interleaving the kernel, the domain expert's kernel
and PyTorch eager reference(). Speed is a column; it never decides pass or fail."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True     # no __pycache__ with local paths in the shipped folder
import tempfile
import time
from pathlib import Path

GRADERS = Path(__file__).resolve().parent
SUB = GRADERS.parent
sys.path.insert(0, str(SUB))
from graders.verdict import verdict, write_tables, sanitizer_errors      # noqa: E402

BUILD = Path(os.environ.get("GRADER_BUILD", GRADERS / "build"))
TASKS = SUB / "tasks"
BUILD_FLAGS = ["-O3", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared"]
SANITIZERS = ["memcheck", "racecheck", "synccheck", "initcheck"]
CASE_TIMEOUT = int(os.environ.get("CASE_TIMEOUT", 300))
SANITIZER_TIMEOUT = int(os.environ.get("SANITIZER_TIMEOUT", 900))
COMPILE_TIMEOUT = 300
SET_B = 7919


def nvcc():
    return os.environ.get("NVCC", "nvcc")


def run(cmd, timeout, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, **kw)


def tail(s, n=3000):
    return (s or "")[-n:]


# ---- tasks and the shim ---------------------------------------------------------------------

def task_ids():
    return sorted(p.parent.parent.name for p in TASKS.glob("*/tests/task.json"))


def load_task(tid):
    t = json.loads((TASKS / tid / "tests" / "task.json").read_text())
    t["dir"] = str(TASKS / tid)
    t["params"] = parse_signature(t["signature"])
    return t


def parse_signature(sig):
    """[(c type, name)] of `void reference(...)`, in order."""
    m = re.search(r"\breference\s*\((.*)\)", sig, re.S)
    out = []
    for p in m.group(1).split(","):
        p = " ".join(p.split())
        name = re.findall(r"[A-Za-z_]\w*", p)[-1]
        ctype = p[: p.rindex(name)].strip()
        out.append((ctype, name))
    return out


def shim_source(task):
    """extern "C" entry that unpacks an array of argument addresses into the C++ call."""
    args = ", ".join(f"*({c.replace('const ', '', 1) if '*' not in c else c}*)a[{i}]" for i, (c, _) in enumerate(task["params"]))
    return ('#include "solution.cu"\n'
            f'extern "C" void grader_entry(void** a) {{ reference({args}); }}\n')


def compile_kernel(task, src_path, arch):
    src = Path(src_path).read_bytes()
    flags = BUILD_FLAGS + task.get("link", [])
    h = hashlib.sha256(src + arch.encode() + " ".join(flags).encode() + task["signature"].encode()).hexdigest()[:20]
    d = BUILD / "kernels" / h
    rec_path = d / "compile.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    d.mkdir(parents=True, exist_ok=True)
    (d / "solution.cu").write_bytes(src)
    (d / "shim.cu").write_text(shim_source(task))
    t0 = time.time()
    try:
        # -Bsymbolic: the shim calls its own `reference`, even when the expert's library is loaded in the same process (timing)
        r = run([nvcc(), f"-arch={arch}", *flags, "-Xlinker", "-Bsymbolic", "-o", "kernel.so", "shim.cu", *task.get("link", [])],
                COMPILE_TIMEOUT, cwd=d)
        rec = {"ok": r.returncode == 0, "sha": h, "lib": str(d / "kernel.so"), "seconds": round(time.time() - t0, 2),
               "log_tail": tail(r.stderr)}
    except subprocess.TimeoutExpired:
        rec = {"ok": False, "sha": h, "lib": str(d / "kernel.so"), "seconds": COMPILE_TIMEOUT, "log_tail": "nvcc timed out"}
    rec_path.write_text(json.dumps(rec, indent=1))
    return rec


# ---- one process per case ---------------------------------------------------------------------

def child(spec, wrap=None, timeout=CASE_TIMEOUT, preload=True):
    """Run graders/child.py on one spec. Returns its record (proc = ok | crash | timeout)."""
    work = Path(tempfile.mkdtemp(prefix="g_"))
    (work / "spec.json").write_text(json.dumps(spec))
    env = dict(os.environ, HOME=str(work), TMPDIR=str(work))
    ip = BUILD / "interpose.so"
    if preload and ip.exists():
        env["LD_PRELOAD"] = str(ip)
    cmd = [sys.executable, str(GRADERS / "child.py"), str(work / "spec.json"), str(work / "result.json")]
    if wrap:
        cmd = wrap + cmd
    t0 = time.time()
    try:
        r = run(cmd, timeout, cwd=work, env=env)
        out = tail(r.stdout + r.stderr, 4000)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        return {"proc": "timeout", "seconds": timeout}
    res = {}
    rp = work / "result.json"
    if rp.exists():
        try:
            res = json.loads(rp.read_text())
        except json.JSONDecodeError:
            res = {}
    shutil.rmtree(work, ignore_errors=True)
    res.update(proc="ok" if res.get("finished") else "crash", rc=rc, tail=out, seconds=round(time.time() - t0, 2))
    return res


def sanitizer_runs(task, lib):
    case = next(c for c in task["cases"] if c.get("small"))
    res = []
    for tool in SANITIZERS:
        wrap = [os.environ.get("COMPUTE_SANITIZER", "compute-sanitizer"), "--tool", tool, "--print-limit", "20"]
        r = child({"task": task["task"], "lib": lib, "case": case, "mode": "sanitize"}, wrap, SANITIZER_TIMEOUT, preload=False)
        errors = sanitizer_errors(r.get("tail", ""))
        status = "ok" if errors is not None and r["proc"] != "timeout" else ("timeout" if r["proc"] == "timeout" else "no_summary")
        res.append({"case": case["name"], "tool": tool, "status": status, "errors": errors, "kernel_ran": r.get("finished", False),
                    "seconds": r["seconds"], "tail": r.get("tail", "")[-1500:]})
    return res


def grade_one(tid, src, arch, passes=1, timing=True, sanitizer=True, all_checks=False):
    task = load_task(tid)
    log = {"task": tid, "kernel": str(src), "arch": arch, "host": platform.node(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "passes": []}
    t0 = time.time()
    comp = compile_kernel(task, src, arch)
    log["compile"] = comp
    if comp["ok"]:
        for k in range(1, passes + 1):
            cases = [dict(child({"task": tid, "lib": comp["lib"], "case": c, "mode": "check"}), name=c["name"]) for c in task["cases"]]
            log["passes"].append({"pass": k, "cases": cases})
            if not all_checks and verdict(log)["status"] != "pass":
                break
        if sanitizer and (all_checks or verdict(log)["status"] == "pass"):
            log["sanitizer"] = sanitizer_runs(task, comp["lib"])
        if timing and verdict(log)["status"] == "pass":
            expert = compile_kernel(task, TASKS / tid / "solution" / "solution.cu", arch)
            case = next(c for c in task["cases"] if c.get("timed"))
            log["timing"] = child({"task": tid, "lib": comp["lib"], "expert_lib": expert["lib"] if expert["ok"] else None,
                                   "case": case, "mode": "time"}, preload=False)
    log["seconds"] = round(time.time() - t0, 2)
    log["verdict"] = verdict(log)
    return log


# ---- folders -------------------------------------------------------------------------------------

def find_kernels(root, ids):
    """Every .cu under root whose path names exactly one task id."""
    out = []
    for p in sorted(Path(root).rglob("*.cu")):
        hit = [t for t in ids if t in p.relative_to(root).parts or p.stem == t]
        if len(hit) == 1:
            out.append((hit[0], p))
    return out


def grade_list(items, out_dir, arch, **kw):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, p, key in items:
        lp = out_dir / (key.replace("/", "__") + ".json")
        if lp.exists() and json.loads(lp.read_text())["verdict"]["status"] != "grader_error":
            print(f"skip (logged) {key}", flush=True)
            continue
        log = grade_one(tid, p, arch, **kw)
        log["key"] = key
        lp.write_text(json.dumps(log, indent=1, default=str))
        v = log["verdict"]
        print(f"{key:60s} {v['status']:14s} {v['label'] or '':24s} x_torch={v['speedup_vs_torch']} "
              f"pct_expert={v['pct_of_expert']} ({log['seconds']} s)", flush=True)
    write_tables(out_dir)


def env_info():
    def cmd(c):
        try:
            return run(c, 60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {"nvidia_smi": cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
            "nvcc": cmd([nvcc(), "--version"]), "compute_sanitizer": cmd(["compute-sanitizer", "--version"]),
            "python": platform.python_version(), "torch": cmd([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]),
            "date": time.strftime("%Y-%m-%d %H:%M %Z")}


def proof(ids, out, logs, arch):
    """P4: every expert kernel (3 passes, sanitizer, timing) and every planted kernel."""
    from graders.planted import write_planted, EXPECT
    env = env_info()
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "env.json").write_text(json.dumps(env, indent=1))
    grade_list([(t, TASKS / t / "solution" / "solution.cu", f"expert/{t}") for t in ids], logs, arch, passes=3)
    files = write_planted(BUILD / "planted", ids)
    grade_list([(t, p, f"planted/{t}/{p.stem}") for t, p in files], logs, arch, passes=1, timing=False, all_checks=True)
    rows = {r["key"]: r for r in write_tables(logs)}
    res = {"gpu": (env.get("nvidia_smi") or "").split(",")[0] or None, "env": env, "arch": arch, "references": {}, "planted": {}}
    for t in ids:
        r = rows.get(f"expert/{t}", {})
        res["references"][t] = {k: r.get(k) for k in ("status", "label", "failed_case", "failed_check", "speedup_vs_torch",
                                                     "pct_of_expert", "kernel_ms", "spread", "identical_repeat", "allocs_per_call", "seconds")}
        res["references"][t]["verdict"] = r.get("status")
    for t, p in files:
        r = rows.get(f"planted/{t}/{p.stem}", {})
        want = EXPECT[p.stem]
        res["planted"][f"{t}/{p.stem}"] = {"expect": f"fail / {want}", "status": r.get("status"), "label": r.get("label"),
                                            "ok": r.get("status") == "fail" and r.get("label") == want}
    res["summary"] = {"references_pass": sum(v["verdict"] == "pass" for v in res["references"].values()), "references": len(ids),
                      "planted_as_expected": sum(v["ok"] for v in res["planted"].values()), "planted": len(res["planted"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["summary"]))


def build_counter():
    """Build the call counter (build/interpose.so). Every grading command runs this when it is missing."""
    BUILD.mkdir(parents=True, exist_ok=True)
    inc = Path(shutil.which(nvcc()) or "/usr/local/cuda/bin/nvcc").resolve().parent.parent / "include"
    r = run([os.environ.get("CC", "gcc"), "-O2", "-shared", "-fPIC", f"-I{inc}", "-o", str(BUILD / "interpose.so"),
             str(GRADERS / "interpose.c"), "-ldl"], COMPILE_TIMEOUT)
    if r.returncode:
        sys.exit("building the call counter failed:\n" + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    arch = os.environ.get("ARCH", "sm_120")
    sub.add_parser("build")
    g = sub.add_parser("grade-dir")
    g.add_argument("--kernels", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--task", help="grade every .cu under DIR as this task (for one-task folders such as /app)")
    pr = sub.add_parser("proof")
    pr.add_argument("--out", default=str(SUB / "results" / "gpu_proof.json"))
    pr.add_argument("--logs", default=str(SUB / "results" / "gpu_proof_logs"))
    for name in ("experts", "planted"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--only", nargs="*")
    rg = sub.add_parser("regrade")
    rg.add_argument("--logs", required=True)
    for s in sub.choices.values():
        s.add_argument("--arch", default=arch)
        s.add_argument("--passes", type=int, default=int(os.environ.get("PASSES", 1)))
    a = ap.parse_args()
    if a.cmd == "build":
        build_counter()
        return
    if a.cmd == "regrade":
        write_tables(Path(a.logs), recompute=True)
        return
    if not (BUILD / "interpose.so").exists():
        build_counter()
    ids = task_ids()
    if a.cmd == "proof":
        proof(ids, Path(a.out), Path(a.logs), a.arch)
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "env.json").write_text(json.dumps(env_info(), indent=1))
    if a.cmd == "grade-dir":
        if a.task:
            files = sorted(Path(a.kernels).rglob("*.cu"))
            items = [(a.task, p, a.task if len(files) == 1 else f"{a.task}/{p.relative_to(a.kernels).with_suffix('')}") for p in files]
        else:
            items = [(t, p, f"{p.relative_to(a.kernels).with_suffix('')}") for t, p in find_kernels(a.kernels, ids)]
        grade_list(items, a.out, a.arch, passes=a.passes)
    elif a.cmd == "experts":
        items = [(t, TASKS / t / "solution" / "solution.cu", f"expert/{t}") for t in ids if not a.only or t in a.only]
        grade_list(items, a.out, a.arch, passes=3)
    else:
        from graders.planted import write_planted
        files = write_planted(BUILD / "planted")
        items = [(t, p, f"planted/{t}/{p.stem}") for t, p in files if not a.only or f"{t}/{p.stem}" in a.only or t in a.only]
        grade_list(items, a.out, a.arch, passes=1, timing=False, all_checks=True)


if __name__ == "__main__":
    main()
