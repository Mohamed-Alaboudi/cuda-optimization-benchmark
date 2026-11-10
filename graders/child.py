"""One graded process: one kernel, one input case. Run by grade.py, never by hand.
  python graders/child.py spec.json result.json
spec: {"task", "lib", "case", "mode": check | sanitize | time, "expert_lib" (time mode)}"""
import ctypes
import importlib.util
import json
import statistics
import sys

sys.dont_write_bytecode = True     # no __pycache__ with local paths in the shipped folder
import time
from pathlib import Path

import torch

GRADERS = Path(__file__).resolve().parent
SUB = GRADERS.parent
sys.path.insert(0, str(SUB))
from graders.grade import load_task, SET_B          # noqa: E402

GUARD = 4096
GUARD_BYTE = 0xA5
POISON = 0xFF
CTYPES = {"float": ctypes.c_float, "double": ctypes.c_double, "int": ctypes.c_int, "int32_t": ctypes.c_int32,
          "int64_t": ctypes.c_int64, "long long": ctypes.c_longlong, "long": ctypes.c_long, "bool": ctypes.c_bool,
          "size_t": ctypes.c_size_t, "unsigned int": ctypes.c_uint, "uint32_t": ctypes.c_uint32}
DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16,
          "bool": torch.bool, "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None), "uint8": torch.uint8, "int32": torch.int32, "int64": torch.int64}
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def ref_module(task):
    p = Path(task["dir"]) / "environment" / "reference.py"
    if not p.exists():
        p = Path(task["dir"]) / "reference.py"
    spec = importlib.util.spec_from_file_location("task_reference", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def as_dict(task, res):
    names = list(task["outputs"])
    if isinstance(res, dict):
        return res
    if isinstance(res, torch.Tensor):
        return {names[0]: res}
    return dict(zip(names, res))


class Guarded:
    """A device buffer with guard bytes before and after; .t is the tensor the kernel sees."""

    def __init__(self, nbytes, dtype, shape):
        self.n = nbytes
        self.raw = torch.full((GUARD + nbytes + GUARD,), GUARD_BYTE, dtype=torch.uint8, device="cuda")
        self.t = self.raw[GUARD:GUARD + nbytes].view(dtype).view(shape)

    def guards_ok(self):
        return bool((self.raw[:GUARD] == GUARD_BYTE).all() and (self.raw[GUARD + self.n:] == GUARD_BYTE).all())

    def poison(self):
        self.raw[GUARD:GUARD + self.n].fill_(POISON)


def nbytes_of(shape, dtype):
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


def out_shape(task, name, case, inp):
    env = dict(case, **{k: v for k, v in inp.items() if not isinstance(v, torch.Tensor)})
    return [int(eval(str(s), {}, env)) for s in task["outputs"][name]["shape"]]


class Kernel:
    def __init__(self, lib):
        self.fn = ctypes.CDLL(lib).grader_entry
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.fn.restype = None

    def bind(self, task, bufs, scalars):
        """Argument array for grader_entry: the address of each argument's value."""
        keep, addrs = [], []
        for ctype, name in task["params"]:
            if "*" in ctype:
                v = ctypes.c_void_p(bufs[name].t.data_ptr())
            else:
                base = " ".join(w for w in ctype.split() if w != "const")
                v = CTYPES[base](scalars[name])
            keep.append(v)
            addrs.append(ctypes.addressof(v))
        arr = (ctypes.c_void_p * len(addrs))(*addrs)
        return arr, keep

    def __call__(self, arr):
        self.fn(arr)


def counter():
    try:
        lib = ctypes.CDLL(None)
        arm, reset, read = lib.grader_counter_arm, lib.grader_counter_reset, lib.grader_counter_read
    except AttributeError:
        return None
    def wrap(fn):
        reset(); arm(1)
        try:
            fn()
        finally:
            arm(0)
        c = (ctypes.c_long * 8)()
        read(c)
        return {"alloc": c[0], "host_copy": c[1], "device_sync": c[2], "launch": c[5]}
    return wrap


def setup(task, case, mode):
    ref = ref_module(task)
    dev = "cpu" if mode == "sanitize" else "cuda"    # sanitize: inputs made on the CPU, so no torch kernel runs before the call
    sets = [ref.make_inputs(case, int(case["seed"]) + k, dev) for k in (0, SET_B)]
    bufs, outs = {}, []
    for ctype, name in task["params"]:
        if "*" not in ctype:
            continue
        if name in task["outputs"]:
            o = task["outputs"][name]
            dt = DTYPES[o["dtype"]]
            shp = out_shape(task, name, case, sets[0])
            bufs[name] = Guarded(nbytes_of(shp, dt), dt, shp)
            outs.append(name)
        else:
            x = sets[0][name]
            bufs[name] = Guarded(x.numel() * x.element_size(), x.dtype, list(x.shape))
    return ref, sets, bufs, outs


def load_set(task, bufs, outs, inp):
    for name, b in bufs.items():
        if name in outs:
            b.poison()
        else:
            b.t.copy_(inp[name])
    return {n: v for n, v in inp.items() if not isinstance(v, torch.Tensor)}


def compare(out, ref, tol):
    """Summary of one output against reference_exact under the stated tolerance."""
    raw = out.contiguous().reshape(-1).view(torch.uint8).view(out.numel(), -1)
    poison = (raw == POISON).all(dim=1)
    o = out.reshape(-1).double()
    r = ref.reshape(-1).to(o.device).double()
    s = {"n": o.numel(), "shape_ok": o.numel() == r.numel()}
    if not s["shape_ok"]:
        s["bad"] = o.numel()
        return s
    atol, rtol = float(tol.get("atol", 0)), float(tol.get("rtol", 0))
    bound = atol + rtol * r.abs()
    err = (o - r).abs()
    fin = torch.isfinite(r)
    ok = torch.where(fin, err <= bound, (torch.isnan(r) & torch.isnan(o)) | (o == r))
    bad = ~ok
    s["bad"] = int(bad.sum())
    ratio = torch.where(fin & torch.isfinite(o), err / bound.clamp_min(1e-300), torch.where(ok, 0.0, float("inf")))
    s["worst"] = float(ratio.max()) if ratio.numel() else 0.0
    if s["bad"]:
        idx = torch.nonzero(bad.reshape(-1))[:, 0]
        s["first_bad"] = int(idx[0])
        s["bad_poison"] = int(poison.reshape(-1)[idx].sum())
        i = int(idx[0])
        s["example"] = {"index": i, "got": float(o[i]), "want": float(r[i])}
    return s


def check(task, case, lib):
    ref, sets, bufs, outs = setup(task, case, "check")
    k = Kernel(lib)
    cnt = counter()
    res = {"mode": "check", "counter": cnt is not None, "calls": []}
    got = {}
    for call, which in ((1, 0), (2, 1), (3, 0)):
        scalars = load_set(task, bufs, outs, sets[which])
        torch.cuda.synchronize()
        arr, keep = k.bind(task, bufs, scalars)
        t0 = time.time()
        if cnt:
            c = cnt(lambda: k(arr))
        else:                       # no call counter preloaded: call the kernel without counting
            k(arr)
            c = None
        host_ms = (time.time() - t0) * 1e3
        rec = {"call": call, "count": c, "host_ms": round(host_ms, 4)}
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            rec["cuda_error"] = str(e)[:300]
            res["calls"].append(rec)
            json.dump(res, open(sys.argv[2], "w"))      # the context is gone; record and stop
            return res
        rec["guards_ok"] = {n: b.guards_ok() for n, b in bufs.items()}
        rec["inputs_unchanged"] = {n: bool(torch.equal(b.t, sets[which][n].to("cuda"))) for n, b in bufs.items() if n not in outs}
        got[call] = {n: bufs[n].t.clone() for n in outs}
        res["calls"].append(rec)
    res["identical_repeat"] = all(torch.equal(got[1][n].reshape(-1).view(torch.uint8), got[3][n].reshape(-1).view(torch.uint8)) for n in outs)
    res["compare"] = {}
    for call, which in ((1, 0), (2, 1)):
        want = as_dict(task, ref.reference_exact(sets[which]))
        tol = task["tolerance"]
        res["compare"][f"call{call}"] = {n: compare(got[call][n], want[n], tol.get(n, tol) if isinstance(tol.get(n), dict) else tol)
                                        for n in outs}
    res["finished"] = True
    return res


def sanitize(task, case, lib):
    ref, sets, bufs, outs = setup(task, case, "sanitize")
    k = Kernel(lib)
    scalars = load_set(task, bufs, outs, sets[0])
    arr, keep = k.bind(task, bufs, scalars)
    k(arr)
    torch.cuda.synchronize()
    return {"mode": "sanitize", "finished": True}


def events(fn, min_ms=1.0, batches=10):
    """Median ms per call: CUDA events around K back-to-back calls, K so one batch takes at least min_ms."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); fn(); e1.record(); e1.synchronize()
    one = e0.elapsed_time(e1)
    kk = max(1, min(1000, int(min_ms / one) + 1)) if one > 0 else 1000
    per = []
    for _ in range(batches):
        e0.record()
        for _ in range(kk):
            fn()
        e1.record(); e1.synchronize()
        per.append(e0.elapsed_time(e1) / kk)
    return statistics.median(per), kk


def timing(task, case, lib, expert_lib):
    ref, sets, bufs, outs = setup(task, case, "time")
    scalars = load_set(task, bufs, outs, sets[0])
    sides = {"kernel": Kernel(lib)}
    if expert_lib:
        sides["expert"] = Kernel(expert_lib)
    calls = {}
    for name, k in sides.items():
        arr, keep = k.bind(task, bufs, scalars)
        calls[name] = (lambda k=k, arr=arr, keep=keep: k(arr))
    inp = sets[0]
    calls["torch"] = lambda: ref.reference(inp)
    rounds = []
    for _ in range(3):                    # 3 rounds, each interleaving the 3 sides
        rounds.append({n: events(f)[0] for n, f in calls.items()})
    med = {n: statistics.median(r[n] for r in rounds) for n in calls}
    spread = {n: (max(r[n] for r in rounds) / min(r[n] for r in rounds) - 1) if min(r[n] for r in rounds) > 0 else None for n in calls}
    return {"mode": "time", "case": case["name"], "rounds": rounds, "median_ms": med, "spread": spread,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "finished": True}


def main():
    spec = json.load(open(sys.argv[1]))
    task = load_task(spec["task"])
    if spec["mode"] == "check":
        res = check(task, spec["case"], spec["lib"])
    elif spec["mode"] == "sanitize":
        res = sanitize(task, spec["case"], spec["lib"])
    else:
        res = timing(task, spec["case"], spec["lib"], spec.get("expert_lib"))
    json.dump(res, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
