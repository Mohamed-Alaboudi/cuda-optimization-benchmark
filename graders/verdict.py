"""Verdicts and tables from grading logs. Plain Python: no torch, no numpy, no GPU, so
`grade.py regrade` runs anywhere. Pass = every input correct on every call, inputs and
guards intact, sanitizer clean. Speed and the call counter are columns, never the verdict."""
import json
import re
from pathlib import Path

def sanitizer_errors(text):
    """Error count from compute-sanitizer output, or None when there is no summary line.
    memcheck, synccheck, initcheck print "ERROR SUMMARY: N errors" (and, past the print
    limit, a second line "ERROR SUMMARY: M errors were not printed"); racecheck prints
    "RACECHECK SUMMARY: H hazards displayed (N errors, W warnings)"."""
    m = re.search(r"ERROR SUMMARY: (\d+) errors?\s*$", text, re.M)
    if m:
        return int(m.group(1))
    m = re.search(r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?", text)
    if m:
        return int(m.group(1))
    return None


def sanitizer_status(s):
    """(status, errors) for one sanitizer record, from its saved output when present."""
    if s.get("status") == "timeout":
        return "timeout", None
    if "tail" in s:
        n = sanitizer_errors(s["tail"])
        return ("ok", n) if n is not None else ("no_summary", None)
    return s["status"], s.get("errors")


SANITIZER_LABEL = {"memcheck": "memory error", "initcheck": "uninitialized read", "racecheck": "race", "synccheck": "sync error"}


def case_result(rec):
    """(ok, check, label) for one case record. ok None = grader fault."""
    if rec.get("proc") == "timeout":
        return False, "timeout", "timeout"
    calls = rec.get("calls", [])
    for c in calls:
        if c.get("cuda_error"):
            return False, f"call{c['call']}:cuda_error", "cuda error"
        if not all(c.get("guards_ok", {}).values()):
            return False, f"call{c['call']}:guards", "out-of-bounds write"
        if not all(c.get("inputs_unchanged", {}).values()):
            return False, f"call{c['call']}:inputs", "input modified"
    if rec.get("proc") != "ok":
        if not calls:
            return None, "no_call", None          # the process died before calling the kernel
        return False, "crash", "crash"
    for name, s in rec["compare"]["call1"].items():
        if s["bad"]:
            if not s.get("shape_ok", True):
                return False, f"call1:{name}", "wrong shape"
            if s.get("bad_poison") == s["bad"]:
                return False, f"call1:{name}", "unwritten output"
            return False, f"call1:{name}", "wrong value"
    for name, s in rec["compare"]["call2"].items():
        if s["bad"]:
            return False, f"call2:{name}", "stale state"
    return True, None, None


def verdict(log):
    v = {"status": None, "label": None, "failed_case": None, "failed_check": None, "speedup_vs_torch": None,
         "pct_of_expert": None, "kernel_ms": None, "identical_repeat": None, "allocs_per_call": None, "passes": []}
    comp = log.get("compile") or {}
    if not comp.get("ok"):
        v["status"], v["label"] = "compile_error", ("signature mismatch" if "reference" in comp.get("log_tail", "") and "shim.cu" in comp.get("log_tail", "") else None)
        return v
    results = []
    for p in log.get("passes", []):
        first = None
        for rec in p["cases"]:
            ok, check, label = case_result(rec)
            if ok is None:
                first = ("grader_error", rec["name"], check, None)
                break
            if not ok and first is None:
                first = ("timeout" if check == "timeout" else "fail", rec["name"], check, label)
        results.append(first or ("pass", None, None, None))
    v["passes"] = [r[0] for r in results]
    if not results:
        v["status"] = "grader_error"
        return v
    bad = next((r for r in results if r[0] != "pass"), None)
    if bad is None:
        san = log.get("sanitizer")
        for s in san or []:
            status, errors = sanitizer_status(s)
            if status != "ok":
                bad = ("grader_error", s["case"], f"sanitizer:{s['tool']}", None)
                break
            if errors:
                bad = ("fail", s["case"], f"sanitizer:{s['tool']}", SANITIZER_LABEL[s["tool"]])
                break
    if bad is None:
        v["status"] = "pass"
    elif bad[0] == "grader_error" or all(r[0] != "pass" for r in results) or bad[2].startswith("sanitizer"):
        v["status"], v["failed_case"], v["failed_check"], v["label"] = bad
    else:
        v["status"], v["failed_case"], v["failed_check"], v["label"] = ("flip",) + bad[1:]
    recs = [r for p in log.get("passes", []) for r in p["cases"] if r.get("proc") == "ok"]
    if recs:
        v["identical_repeat"] = all(r.get("identical_repeat") for r in recs)
        counts = [c["count"]["alloc"] for r in recs for c in r["calls"] if c.get("count")]
        v["allocs_per_call"] = max(counts) if counts else None
    t = log.get("timing") or {}
    m = t.get("median_ms") or {}
    if v["status"] == "pass" and m.get("kernel"):
        v["kernel_ms"] = m["kernel"]
        if m.get("torch"):
            v["speedup_vs_torch"] = round(m["torch"] / m["kernel"], 3)
        if m.get("expert"):
            v["pct_of_expert"] = round(100 * m["expert"] / m["kernel"], 1)
    return v


def write_tables(log_dir, recompute=False):
    """results.json (one row per kernel) and table.md, from the logs in log_dir."""
    log_dir = Path(log_dir)
    rows = []
    for p in sorted(log_dir.glob("*.json")):
        if p.name in ("results.json", "verdicts.json", "env.json"):
            continue
        log = json.loads(p.read_text())
        if "task" not in log:
            continue
        v = verdict(log) if recompute or "verdict" not in log else log["verdict"]
        t = log.get("timing") or {}
        rows.append(dict(key=log.get("key", p.stem), task=log["task"], **v,
                         spread=(t.get("spread") or {}).get("kernel"), seconds=log.get("seconds")))
    (log_dir / "results.json").write_text(json.dumps(rows, indent=1))
    (log_dir / "verdicts.json").write_text(json.dumps({r["key"]: r for r in rows}, indent=1))
    lines = ["| Kernel | Task | Status | Label | Failed case / check | x PyTorch eager | % of expert speed | Identical repeat | Allocs per call |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['key']} | {r['task']} | {r['status']} | {r['label'] or ''} | {r['failed_case'] or ''} {r['failed_check'] or ''} | "
                     f"{r['speedup_vs_torch'] or ''} | {r['pct_of_expert'] or ''} | {r['identical_repeat']} | {r['allocs_per_call']} |")
    (log_dir / "table.md").write_text("\n".join(lines) + "\n")
    return rows
