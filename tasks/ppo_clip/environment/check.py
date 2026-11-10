"""Public check for your solution.cu (small inputs only; the grader uses more and larger inputs).

Usage (in /app):  python3 check.py [path/to/solution.cu]
It builds solution.cu with the build command from instruction.md, calls your `reference(...)`
through a small extern "C" wrapper on the cases in public_cases.json, compares every output with
reference_exact() under the stated tolerance, and prints a rough speed ratio against the PyTorch
eager reference(). Passing here does not mean passing the grader.
"""
import ctypes
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch

sys.dont_write_bytecode = True  # no __pycache__ next to reference.py
HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.environ.get("ARCH", "sm_120")
CTYPES = {"int32_t": ctypes.c_int32, "int64_t": ctypes.c_int64, "float": ctypes.c_float}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_signature(sig):
    """'void reference(const float* x, int32_t n, float* out);' -> [(type, name, is_pointer)]."""
    body = sig[sig.index("(") + 1: sig.rindex(")")]
    args = []
    for part in body.split(","):
        part = part.strip()
        name = re.findall(r"[A-Za-z_]\w*$", part)[0]
        typ = part[: -len(name)].strip()
        args.append((typ, name, "*" in typ))
    return args


def build(solution, spec, work):
    """The grader's build: shim.cu includes solution.cu and adds an extern "C" entry."""
    args = parse_signature(spec["signature"])
    params = ", ".join(f"{t} {n}" for t, n, _ in args)
    names = ", ".join(n for _, n, _ in args)
    shutil.copyfile(solution, os.path.join(work, "solution.cu"))
    with open(os.path.join(work, "shim.cu"), "w") as f:
        f.write('#include "solution.cu"\n'
                f'extern "C" void reference_c({params}) {{ reference({names}); }}\n')
    cmd = ["nvcc", "-O3", f"-arch={ARCH}", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared",
           "-o", "kernel.so", "shim.cu"] + spec["link"]
    print("build:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if r.returncode:
        sys.exit("compile error:\n" + r.stderr[-4000:])
    return ctypes.CDLL(os.path.join(work, "kernel.so")).reference_c, args


def call(fn, args, inp, outs):
    vals = []
    for typ, name, ptr in args:
        v = outs[name] if name in outs else inp[name]
        if ptr:
            vals.append(ctypes.c_void_p(v.data_ptr()))
        else:
            vals.append(CTYPES[typ.replace("const", "").strip()](v))
    fn(*vals)


def ms(f, reps=20):
    f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        a.record(); f(); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def main():
    solution = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "solution.cu"))
    spec = json.load(open(os.path.join(HERE, "public_cases.json")))
    s = importlib.util.spec_from_file_location("reference", os.path.join(HERE, "reference.py"))
    ref = importlib.util.module_from_spec(s)
    s.loader.exec_module(ref)
    fn, args = build(solution, spec, tempfile.mkdtemp())
    atol, rtol = spec["tolerance"]["atol"], spec["tolerance"]["rtol"]
    ok_all = True
    for case in spec["cases"]:
        inp = ref.make_inputs(case, case["seed"], "cuda")
        shape = lambda o: [inp[d] if isinstance(d, str) else d for d in spec["outputs"][o]["shape"]]
        outs = {o: torch.full(shape(o), float("nan"), device="cuda", dtype=DTYPES[m["dtype"]])
                for o, m in spec["outputs"].items()}
        call(fn, args, inp, outs)
        torch.cuda.synchronize()
        exact = ref.reference_exact(inp)
        for o in outs:
            got, want = outs[o].double(), exact[o].double()
            bad = ~((got - want).abs() <= atol + rtol * want.abs())
            bad &= ~(torch.isnan(got) & torch.isnan(want))
            n = int(bad.sum())
            ok_all &= n == 0
            print(f"{case['name']}: {o}: {'OK' if n == 0 else f'{n} of {got.numel()} elements outside tolerance'}")
        t_you = ms(lambda: call(fn, args, inp, outs))
        t_ref = ms(lambda: ref.reference(inp))
        print(f"{case['name']}: your time {t_you:.4f} ms, PyTorch eager {t_ref:.4f} ms, speedup {t_ref / t_you:.2f}x (small input, rough)")
    print("ALL PUBLIC CASES OK" if ok_all else "SOME PUBLIC CASES FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
