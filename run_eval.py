#!/usr/bin/env python3
"""run_eval.py: run the CUDA agent eval.

Commands (run from this folder):
  trials   each model tries each task alone in an agent loop (bash, write_file, done; 20 min per try)
  pod      RunPod GPU pod: create, status, balance, ssh, stop, terminate, watchdog (see runner/pod.py)
  grade    pass-through to graders/grade.py (same arguments)

Model calls go out from the machine that runs this script. Keys come from environment variables
(.env.example). On the pod only the tools run, each try in its own folder as its own unix user.
Finished tries are skipped on a rerun; a new --results folder starts a fresh run.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runner"))
import agent_loop as AL  # noqa: E402

TASK_IDS = ["pow_abs_sqrt_mean", "leaky", "efficient_kan", "fp8_linear", "ppo_clip", "reward_to_go"]


def task_files(tid):
    """What the model gets in its folder: prompt.md plus tasks/<id>/environment/* (reference.py, check.py,
    public_cases.json; not the Dockerfile). Hidden tests (tests/) and the expert kernel (solution/) never go in."""
    d = ROOT / "tasks" / tid
    files = [d / "prompt.md"] + sorted(p for p in (d / "environment").iterdir() if p.is_file() and p.name != "Dockerfile")
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"task {tid}: missing {missing}")
    return files


def load_models(path, names):
    models = json.loads(Path(path).read_text())["models"]
    if names:
        want = names.split(",")
        models = [m for m in models if m["name"] in want]
        if len(models) != len(want):
            raise SystemExit(f"unknown model in {want}")
    return models


def cmd_trials(a):
    if a.executor == "ssh" and not (a.host and a.port):
        raise SystemExit("--executor ssh needs --host and --port (python run_eval.py pod ssh prints them)")
    models = load_models(a.models_file, a.models)
    tasks = a.tasks.split(",") if a.tasks else TASK_IDS
    system = (ROOT / "runner" / "agent_system.md").read_text()
    results, transcripts = Path(a.results).resolve(), Path(a.transcripts).resolve()
    jobs = [(m, t, k) for k in range(1, a.tries + 1) for t in tasks for m in models]
    sems = {m["name"]: threading.Semaphore(m.get("max_concurrency", a.per_model)) for m in models}
    lock = threading.Lock()
    counts = {}

    def one(m, tid, k):
        res = results / m["name"] / tid / f"try{k}.json"
        if res.exists():
            return "skip"
        tpath = transcripts / m["name"] / tid / f"try{k}.json"
        if tpath.exists():  # an earlier run died mid-try: keep its transcript, start the try again
            tpath.rename(tpath.with_name(f"aborted-try{k}-{int(time.time())}.json"))
        with sems[m["name"]]:
            ex = AL.LocalExecutor(a.local_dir) if a.executor == "local" else AL.SSHExecutor(a.host, a.port)
            files = task_files(tid)
            prompt = (ROOT / "tasks" / tid / "prompt.md").read_text()
            print(f"{m['name']} {tid} try{k}: start", flush=True)
            tr, sol = AL.run_trial(m, tid, files, prompt, system, ex, a.wall_min * 60, m.get("trial_cap_usd"), tpath,
                                   log=lambda s: print(s, flush=True))
        res.parent.mkdir(parents=True, exist_ok=True)
        if sol is not None:
            (res.with_suffix(".cu")).write_bytes(sol)
        summary = {k2: tr.get(k2) for k2 in ("trial", "model", "model_id", "provider", "params", "task", "status", "turns",
                                               "seconds", "usd", "started_et", "ended_et", "wall_cap_s", "trial_cap_usd", "error")}
        summary.update(try_number=k, executor=a.executor, solution=f"try{k}.cu" if sol is not None else None,
                       solution_sha256=hashlib.sha256(sol).hexdigest() if sol is not None else None,
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        res.write_text(json.dumps(summary, indent=1))
        return tr["status"]

    print(f"{len(jobs)} tries: {len(models)} models x {len(tasks)} tasks x {a.tries}; executor {a.executor}; "
          f"workers {a.workers}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(one, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            m, tid, k = futs[f]
            try:
                st = f.result()
            except Exception as e:
                st = "runner_error"
                print(f"RUNNER ERROR {m['name']} {tid} try{k}: {type(e).__name__}: {e}", flush=True)
            with lock:
                counts[st] = counts.get(st, 0) + 1
            print(f"{m['name']} {tid} try{k}: {st}", flush=True)
    print(json.dumps(counts))


def main():
    if sys.argv[1:2] == ["pod"]:
        import pod
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return pod.main()
    if sys.argv[1:2] == ["grade"]:
        return subprocess.call([sys.executable, str(ROOT / "graders" / "grade.py")] + sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pod", help="RunPod pod commands: python run_eval.py pod --help")
    sub.add_parser("grade", help="pass-through to graders/grade.py")
    p = sub.add_parser("trials", help="agent-loop tries (model calls from this machine)")
    p.add_argument("--models-file", default=str(ROOT / "models.json"))
    p.add_argument("--models", help="comma list of model names (default all in models.json)")
    p.add_argument("--tasks", help="comma list of task ids (default all 6)")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--wall-min", type=float, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--per-model", type=int, default=4, help="concurrent tries per model (models.json max_concurrency wins)")
    p.add_argument("--executor", choices=["ssh", "local"], default="ssh")
    p.add_argument("--host"), p.add_argument("--port", type=int)
    p.add_argument("--local-dir", default="/tmp/micro-trials", help="local executor: base folder for try folders")
    p.add_argument("--results", default=str(ROOT / "results" / "trials"))
    p.add_argument("--transcripts", default=str(ROOT / "transcripts"))
    p.set_defaults(fn=cmd_trials)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
