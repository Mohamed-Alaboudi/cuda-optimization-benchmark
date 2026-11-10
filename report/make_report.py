#!/usr/bin/env python3
"""Build results/table.md and results/per_task.md from the logs only. Nothing is typed by hand.

  python3 report/make_report.py --logs results/grade_logs
      [--trials results/trials] [--spend ../micro/logs/spend.jsonl] [--models models.json] [--out results]

Inputs:
  --trials  results/trials/<model>/<task>/try<k>.json  (run_eval.py trials; usd, status)
  --logs    one folder from `graders/grade.py grade-dir --kernels results/trials --out <logs>`
            (<model>__<task>__try<k>.json, key "<model>/<task>/try<k>"); verdicts are recomputed
            from each log with graders/verdict.py, the grader's own rules
  --spend   micro/logs/spend.jsonl (one line per charge), shown as a cross-check of the trial costs
  --models  models.json (model ids and settings)
Plain Python, no GPU, no torch. Paths in the output are relative to the output folder.
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB))
from graders.verdict import case_result, verdict  # noqa: E402

def _load_ref_speeds():
    p = Path(__file__).resolve().parents[1] / "results" / "reference_speeds.json"
    if p.exists():
        return {k: float(v) for k, v in json.loads(p.read_text()).get("per_task", {}).items()}
    return {}

REF_SPEEDS = _load_ref_speeds()


def load_json(p):
    return json.loads(Path(p).read_text())


def pass_hat_k(n, c, k):
    """Chance that k tries drawn without replacement from n all pass (c passed): C(c,k) / C(n,k)."""
    return math.comb(c, k) / math.comb(n, k) if n >= k else None


def fmt(x, nd=2, suffix=""):
    return "" if x is None else f"{x:.{nd}f}{suffix}"


def closeness(log, v):
    """Higher = closer to passing. (stage, cases passed in pass 1, -fraction of bad elements in the failed case)."""
    if v["status"] == "pass":
        return (3, 0, 0.0)
    if not log or not (log.get("compile") or {}).get("ok"):
        return (0, 0, -1.0)
    cases = (log.get("passes") or [{}])[0].get("cases", [])
    ok = sum(1 for c in cases if case_result(c)[0])
    frac = 0.0
    for c in cases:
        if c.get("name") == v.get("failed_case"):
            for call in ("call1", "call2"):
                for s in (c.get("compare") or {}).get(call, {}).values():
                    if s.get("bad"):
                        frac = max(frac, s["bad"] / max(s.get("n", 1), 1))
    stage = 2 if str(v.get("failed_check") or "").startswith("sanitizer") else 1
    return (stage, ok, -frac)


def describe_fail(log, v):
    if v["status"] == "pass":
        return "passed"
    if v["status"] in ("no_solution", "not_graded"):
        return v["status"].replace("_", " ")
    if v["status"] == "compile_error":
        return "compile error" + (f" ({v['label']})" if v.get("label") else "")
    cases = (log.get("passes") or [{}])[0].get("cases", []) if log else []
    ok = sum(1 for c in cases if case_result(c)[0])
    extra = ""
    for c in cases:
        if c.get("name") == v.get("failed_case"):
            for call in ("call1", "call2"):
                for name, s in (c.get("compare") or {}).get(call, {}).items():
                    if s.get("bad"):
                        extra = f"; {s['bad']:,} of {s.get('n', 0):,} elements of `{name}` wrong, worst {s.get('worst', 0):.3g}x the tolerance"
                        break
                if extra:
                    break
    return f"{v['status']}: {v.get('label') or ''} at `{v.get('failed_case')}` ({v.get('failed_check')}); {ok} of {len(cases)} inputs passed{extra}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--trials", default=str(SUB / "results" / "trials"))
    ap.add_argument("--spend", default=str(SUB.parent / "micro" / "logs" / "spend.jsonl"))
    ap.add_argument("--models", default=str(SUB / "models.json"))
    ap.add_argument("--out", default=str(SUB / "results"))
    ap.add_argument("--k", type=int, default=3, help="k in pass^k")
    a = ap.parse_args()
    trials_dir, logs_dir, out = Path(a.trials), Path(a.logs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, out)

    models = {m["name"]: m for m in load_json(a.models)["models"]}
    logs = {}
    for p in sorted(logs_dir.glob("*.json")):
        d = load_json(p)
        if isinstance(d, dict) and "task" in d and d.get("key"):
            logs[d["key"]] = d
    rows, warnings = [], []
    for tj in sorted(trials_dir.glob("*/*/try*.json")):
        model, task, name = tj.parts[-3], tj.parts[-2], tj.stem
        t = load_json(tj)
        key = f"{model}/{task}/{name}"
        log = logs.get(key)
        if log is not None:
            v = verdict(log)
            if v.get("speedup_vs_torch") and task in REF_SPEEDS:
                v["pct_of_expert"] = v["speedup_vs_torch"] / REF_SPEEDS[task] * 100
        elif not t.get("solution"):
            v = {"status": "no_solution", "label": t.get("status")}
        else:
            v = {"status": "not_graded", "label": None}
            warnings.append(f"{key}: has a solution but no grade log in {logs_dir}")
        rows.append({"model": model, "task": task, "try": name, "key": key, "trial": t, "log": log, "v": v,
                     "usd": t.get("usd") or 0.0})
    if not rows:
        sys.exit(f"no trial files under {trials_dir}")
    mnames = sorted({r["model"] for r in rows}, key=lambda m: list(models).index(m) if m in models else 99)
    tasks = sorted({r["task"] for r in rows})

    spend = defaultdict(float)
    if Path(a.spend).exists():
        for line in Path(a.spend).read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                spend[s.get("source", "?")] += float(s.get("dollars") or 0)

    L = ["# Results by model", "",
         f"Built by `report/make_report.py` from {len(rows)} tries in `{rel(trials_dir)}` and the grade logs in `{rel(logs_dir)}`. "
         "Pass = correct on every input and call, integrity checks kept, sanitizer clean. Speed columns use passing tries only "
         "(timed input of each task, medians).", "",
         "| Model (version, settings) | Trials per task | pass^%d | Median speedup vs PyTorch eager | Median %% of reference speed |" % a.k,
         "|---|---|---|---|---|"]
    for m in mnames:
        mr = [r for r in rows if r["model"] == m]
        per_task = defaultdict(list)
        for r in mr:
            per_task[r["task"]].append(r["v"]["status"] == "pass")
        tries = sorted({len(x) for x in per_task.values()})
        phk = [pass_hat_k(len(x), sum(x), a.k) for x in per_task.values()]
        phk = [x for x in phk if x is not None]
        sp = [r["v"]["speedup_vs_torch"] for r in mr if r["v"].get("speedup_vs_torch")]
        pe = [r["v"]["pct_of_expert"] for r in mr if r["v"].get("pct_of_expert")]
        cfg = models.get(m, {})
        ids = sorted({str(r["trial"].get("model_id")) for r in mr if r["trial"].get("model_id")})
        settings = json.dumps(cfg.get("params", {}), separators=(",", ":"))
        L.append(f"| {m}: id {', '.join(ids) or cfg.get('model', '?')}; `{settings}` | {'/'.join(map(str, tries))} | "
                 f"{fmt(statistics.mean(phk)) if phk else ''} | {fmt(statistics.median(sp), 2, 'x') if sp else ''} | {fmt(statistics.median(pe), 0, '%') if pe else ''} |")
    L += ["", f"pass^{a.k} = mean over tasks of C(passes, {a.k}) / C(tries, {a.k}).", ""]
    if spend:
        L += ["Spend log cross-check (`spend.jsonl`, all charges by source, USD): " +
              ", ".join(f"{k} {v:.2f}" for k, v in sorted(spend.items())), ""]
    if warnings:
        L += ["Warnings:"] + [f"- {w}" for w in warnings] + [""]
    (out / "table.md").write_text("\n".join(L))

    P = ["# Results by task", "",
         f"Built by `report/make_report.py` from the same logs. Labels are the grader's (`graders/verdict.py`). "
         "Closest call = the failing try that got furthest (stage, inputs passed, fewest wrong elements); "
         "for a model with no failure, its slowest passing try.", ""]
    for t in tasks:
        P += [f"## {t}", "", "| Model | Passed | Failures by label | Closest call | Kernel files |", "|---|---|---|---|---|"]
        for m in mnames:
            tr = [r for r in rows if r["model"] == m and r["task"] == t]
            if not tr:
                continue
            fails = [r for r in tr if r["v"]["status"] != "pass"]
            labels = Counter((r["v"].get("label") or r["v"]["status"]) for r in fails)
            if fails:
                best = max(fails, key=lambda r: closeness(r["log"], r["v"]))
                close = f"{best['try']}: {describe_fail(best['log'], best['v'])}"
            else:
                best = min(tr, key=lambda r: r["v"].get("speedup_vs_torch") or float("inf"))
                close = (f"{best['try']}: slowest pass, {fmt(best['v'].get('speedup_vs_torch'), 2, 'x')} PyTorch eager, "
                         f"{fmt(best['v'].get('pct_of_expert'), 0, '%')} of reference speed")
            links = []
            f1 = fails[0] if fails else None
            p1 = next((r for r in tr if r["v"]["status"] == "pass"), None)
            for lab, r in (("failure", f1), ("pass", p1)):
                if r:
                    links.append(f"[{lab} {r['try']}](trials/{r['model']}/{r['task']}/{r['try']}.cu)")
            P.append(f"| {m} | {len(tr) - len(fails)}/{len(tr)} | "
                     f"{', '.join(f'{k} x{n}' for k, n in labels.most_common()) or 'none'} | {close} | {' '.join(links)} |")
        P.append("")
    (out / "per_task.md").write_text("\n".join(P))
    print(f"wrote {out / 'table.md'} and {out / 'per_task.md'} ({len(rows)} tries, {len(logs)} grade logs, {len(warnings)} warnings)")


if __name__ == "__main__":
    main()
