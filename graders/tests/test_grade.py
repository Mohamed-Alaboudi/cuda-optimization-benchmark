"""CPU tests for the grader (no GPU). From submission/:
    PYTHONDONTWRITEBYTECODE=1 uv run --python 3.12 --with torch --with numpy --with pytest pytest -p no:cacheprovider graders/tests -q
They prove: signature parsing and the shim, extraction, the verdict and labels for every
failure kind, the comparison under a tolerance (incl. bf16, NaN, poison), every task's files
work with the grader's input and output handling (small cases, torch on the CPU), and the
planted kernels are generated. They cannot prove the GPU side (guards, counter, sanitizer):
that is the GPU proof (P4)."""
import json
from pathlib import Path

import pytest
import torch

from graders import grade, planted
from graders.extract import extract
from graders.verdict import verdict, case_result

IDS = grade.task_ids()


def test_six_tasks():
    assert IDS == sorted(["pow_abs_sqrt_mean", "leaky", "efficient_kan", "fp8_linear", "ppo_clip", "reward_to_go"])


@pytest.mark.parametrize("tid", IDS)
def test_signature_and_shim(tid):
    t = grade.load_task(tid)
    names = [n for _, n in t["params"]]
    assert set(t["outputs"]) <= set(names)
    assert all("*" in c for c, n in t["params"] if n in t["outputs"])
    s = grade.shim_source(t)
    assert s.count("a[") == len(names) and 'extern "C" void grader_entry' in s
    cases = t["cases"]
    assert sum(bool(c.get("small")) for c in cases) >= 1 and sum(bool(c.get("timed")) for c in cases) == 1


def test_shim_types():
    t = {"params": grade.parse_signature("void reference(const int32_t M, const __nv_fp8_e4m3* x, float s, bool* d, float* out);")}
    assert t["params"] == [("const int32_t", "M"), ("const __nv_fp8_e4m3*", "x"), ("float", "s"), ("bool*", "d"), ("float*", "out")]
    assert "reference(*(int32_t*)a[0], *(const __nv_fp8_e4m3**)a[1], *(float*)a[2], *(bool**)a[3], *(float**)a[4])" in grade.shim_source(t)


@pytest.mark.parametrize("tid", IDS)
def test_task_files_with_grader(tid):
    """make_inputs, output shapes, both references, and a correct answer scoring 0 bad elements (small case, CPU)."""
    from graders import child
    t = grade.load_task(tid)
    ref = child.ref_module(t)
    case = next(c for c in t["cases"] if c.get("small"))
    inp = ref.make_inputs(case, case["seed"], "cpu")
    for ctype, name in t["params"]:
        assert name in inp or name in t["outputs"], name
    want = child.as_dict(t, ref.reference_exact(inp))
    fast = child.as_dict(t, ref.reference(inp))
    for name, o in t["outputs"].items():
        shp = child.out_shape(t, name, case, inp)
        assert list(want[name].reshape(shp).shape) == shp
        got = fast[name].to(child.DTYPES[o["dtype"]]).reshape(shp)
        tol = t["tolerance"].get(name, t["tolerance"]) if isinstance(t["tolerance"].get(name), dict) else t["tolerance"]
        s = child.compare(got, want[name], tol)
        assert s["bad"] == 0, (tid, name, s)
    inp_b = ref.make_inputs(case, case["seed"] + grade.SET_B, "cpu")
    assert any(not torch.equal(inp[k], inp_b[k]) for k in inp if isinstance(inp[k], torch.Tensor) and inp[k].dtype.is_floating_point and inp[k].element_size() >= 2)


def test_compare():
    from graders import child
    tol = {"atol": 1e-6, "rtol": 1e-4}
    ref = torch.tensor([1.0, -2.0, 0.0, float("inf"), float("nan")], dtype=torch.float64)
    ok = ref.float()
    assert child.compare(ok, ref, tol)["bad"] == 0
    off = ok.clone(); off[0] = 1.0 + 1e-6 + 2e-4
    assert child.compare(off, ref, tol)["bad"] == 1
    lost = ok.clone(); lost[4] = 0.0
    assert child.compare(lost, ref, tol)["bad"] == 1
    poison = torch.full((5,), 0xFF, dtype=torch.uint8).view(torch.uint8).repeat_interleave(4).view(torch.float32)
    s = child.compare(poison, ref, tol)
    assert s["bad"] == 4 and s["bad_poison"] == 4          # the NaN slot matches NaN
    bf = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert child.compare(bf, torch.tensor([1.0, 2.0], dtype=torch.float64), {"atol": 0.01, "rtol": 0.01})["bad"] == 0
    assert child.compare(bf, torch.tensor([1.0, 2.1], dtype=torch.float64), {"atol": 0.01, "rtol": 0.01})["bad"] == 1
    assert child.compare(torch.zeros(3), torch.zeros(4, dtype=torch.float64), tol)["shape_ok"] is False


# ---- verdict and labels ----------------------------------------------------------------------

def call(n, **kw):
    c = {"call": n, "count": {"alloc": 0, "host_copy": 0, "device_sync": 0, "launch": 1},
         "guards_ok": {"x": True, "out": True}, "inputs_unchanged": {"x": True}}
    c.update(kw)
    return c


def rec(name="c1", call1=0, call2=0, poison=0, **kw):
    r = {"name": name, "proc": "ok", "finished": True, "calls": [call(1), call(2), call(3)], "identical_repeat": True,
         "compare": {"call1": {"out": {"n": 10, "bad": call1, "bad_poison": poison, "shape_ok": True}},
                     "call2": {"out": {"n": 10, "bad": call2, "shape_ok": True}}}}
    r.update(kw)
    return r


def log(*passes, sanitizer=None, compile_ok=True, timing=None):
    l = {"task": "t", "compile": {"ok": compile_ok, "log_tail": ""}, "passes": [{"pass": i + 1, "cases": list(p)} for i, p in enumerate(passes)]}
    if sanitizer is not None:
        l["sanitizer"] = sanitizer
    if timing:
        l["timing"] = timing
    return l


SAN_OK = [{"case": "c1", "tool": t, "status": "ok", "errors": 0} for t in grade.SANITIZERS]


@pytest.mark.parametrize("r,status,label", [
    (rec(), "pass", None),
    (rec(call1=3), "fail", "wrong value"),
    (rec(call1=3, poison=3), "fail", "unwritten output"),
    (rec(call2=10), "fail", "stale state"),
    (rec(calls=[call(1, guards_ok={"out": False})]), "fail", "out-of-bounds write"),
    (rec(calls=[call(1, inputs_unchanged={"x": False})]), "fail", "input modified"),
    (rec(proc="crash", calls=[call(1, cuda_error="illegal memory access")]), "fail", "cuda error"),
    (rec(proc="crash", calls=[call(1)]), "fail", "crash"),
    ({"name": "c1", "proc": "timeout"}, "timeout", "timeout"),
    ({"name": "c1", "proc": "crash", "calls": []}, "grader_error", None),
])
def test_case_labels(r, status, label):
    v = verdict(log([r], sanitizer=SAN_OK))
    assert (v["status"], v["label"]) == (status, label)


def test_sanitizer_flip_compile_and_columns():
    for tool, label in [("racecheck", "race"), ("initcheck", "uninitialized read"), ("memcheck", "memory error")]:
        san = [dict(s, errors=2 if s["tool"] == tool else 0) for s in SAN_OK]
        assert (verdict(log([rec()], sanitizer=san))["label"]) == label
    san = [dict(s, status="timeout", errors=None) if s["tool"] == "racecheck" else s for s in SAN_OK]
    assert verdict(log([rec()], sanitizer=san))["status"] == "grader_error"
    assert verdict(log([rec()], [rec(call1=1)], [rec()], sanitizer=SAN_OK))["status"] == "flip"
    assert verdict(log([rec()], compile_ok=False))["status"] == "compile_error"
    v = verdict(log([rec(identical_repeat=False)], sanitizer=SAN_OK,
                    timing={"median_ms": {"kernel": 2.0, "torch": 3.0, "expert": 1.0}}))
    assert v["status"] == "pass" and v["speedup_vs_torch"] == 1.5 and v["pct_of_expert"] == 50.0 and v["identical_repeat"] is False


def test_regrade_tables(tmp_path):
    from graders.verdict import write_tables
    (tmp_path / "a.json").write_text(json.dumps(dict(log([rec()], sanitizer=SAN_OK), key="m/t/try1")))
    (tmp_path / "b.json").write_text(json.dumps(dict(log([rec(call2=1)]), key="m/t/try2")))
    rows = write_tables(tmp_path, recompute=True)
    assert [r["status"] for r in rows] == ["pass", "fail"]
    assert "stale state" in (tmp_path / "table.md").read_text()
    assert json.loads((tmp_path / "verdicts.json").read_text())["m/t/try2"]["status"] == "fail"


def test_find_kernels(tmp_path):
    for p in ["gpt/leaky/try1/solution.cu", "gpt/leaky/try2.cu", "kimi/ppo_clip/try1/solution.cu", "notes/x.cu"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("//")
    got = [(t, str(p.relative_to(tmp_path))) for t, p in grade.find_kernels(tmp_path, IDS)]
    assert got == [("leaky", "gpt/leaky/try1/solution.cu"), ("leaky", "gpt/leaky/try2.cu"), ("ppo_clip", "kimi/ppo_clip/try1/solution.cu")]


# ---- planted kernels -------------------------------------------------------------------------

@pytest.mark.parametrize("tid", IDS)
def test_planted_generated(tid):
    if not (grade.TASKS / tid / "solution" / "solution.cu").exists():
        pytest.skip("expert kernel not copied in yet")
    srcs = planted.planted_sources(tid)
    assert set(srcs) == set(planted.EXPECT)
    for kind, s in srcs.items():
        assert "#define reference expert_reference" in s and "expert_reference(" in s.split("#undef reference")[1] or kind == "cached_result"


# ---- extraction ----------------------------------------------------------------------------------

GOOD = "void reference(const float* x, float* out) {}"


@pytest.mark.parametrize("text,status", [
    (f"```cuda\n{GOOD}\n```\n", "ok"), ("prose only", "parse_error"), (f"```cuda\n{GOOD}\n", "truncated"),
    (f"```cuda\n//a\n```\n```cuda\n{GOOD}\n```\n", "ok"), (f"```\n{GOOD}\n```\n", "ok")])
def test_extract(text, status):
    code, st = extract(text, "reference")
    assert st == status and (code is None or "reference" in code)


def test_proof_flow(tmp_path, monkeypatch):
    """grade.py proof end to end with a stand-in for the GPU process: experts pass, each planted kind gets its label."""
    from graders import planted as pl
    if not all((grade.TASKS / t / "solution" / "solution.cu").exists() for t in IDS):
        pytest.skip("expert kernels not copied in yet")
    monkeypatch.setattr(grade, "BUILD", tmp_path / "build")
    monkeypatch.setattr(grade, "compile_kernel", lambda task, src, arch: {"ok": True, "lib": str(src), "sha": "x", "log_tail": ""})
    monkeypatch.setattr(grade, "env_info", lambda: {"nvidia_smi": "FAKE GPU, 1, 2, 12.0"})

    def fake_child(spec, wrap=None, timeout=0, preload=True):
        kind = Path(spec["lib"]).stem
        if spec["mode"] == "time":
            return {"proc": "ok", "finished": True, "median_ms": {"kernel": 1.0, "expert": 1.0, "torch": 2.0}, "spread": {"kernel": 0.01}}
        if spec["mode"] == "sanitize":
            tool = wrap[2]
            n = 3 if (kind, tool) in (("race", "racecheck"), ("uninit_read", "initcheck")) else 0
            return {"proc": "ok", "finished": True, "seconds": 1, "tail": f"========= ERROR SUMMARY: {n} errors"}
        r = rec(name=spec["case"]["name"])
        if kind == "wrong_value":
            r = rec(call1=1)
        elif kind == "cached_result":
            r = rec(call2=10, )
        elif kind == "oob_write":
            r = rec(calls=[call(1, guards_ok={"output": False})])
        return r
    from pathlib import Path
    monkeypatch.setattr(grade, "child", fake_child)
    grade.proof(IDS, tmp_path / "gpu_proof.json", tmp_path / "logs", "sm_120")
    p = json.loads((tmp_path / "gpu_proof.json").read_text())
    assert p["gpu"] == "FAKE GPU"
    assert all(p["references"][t]["verdict"] == "pass" for t in IDS)
    assert p["summary"] == {"references_pass": 6, "references": 6, "planted_as_expected": 30, "planted": 30}
    assert p["references"][IDS[0]]["speedup_vs_torch"] == 2.0 and p["references"][IDS[0]]["pct_of_expert"] == 100.0


def test_sanitizer_summary_lines():
    from graders.verdict import sanitizer_errors
    assert sanitizer_errors("========= ERROR SUMMARY: 0 errors\n") == 0
    assert sanitizer_errors("========= ERROR SUMMARY: 64 errors\n========= ERROR SUMMARY: 44 errors were not printed. Use --print-limit\n") == 64
    assert sanitizer_errors("========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n") == 0
    assert sanitizer_errors("========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n") == 1
    assert sanitizer_errors("========= COMPUTE-SANITIZER\n") is None
