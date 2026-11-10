"""Planted wrong kernels, made from each task's expert kernel (tasks/<id>/solution/solution.cu)
so they compile against the same signature. Each must fail with its named label when graded. `grade.py planted` writes them to the build folder and grades them."""
import json
from pathlib import Path

GRADERS = Path(__file__).resolve().parent
TASKS = GRADERS.parent / "tasks"

# first output element set to a value far from any reference (the raw bits, per dtype)
WRONG_BITS = {"float32": ("unsigned int", "0x4640E400u"), "bfloat16": ("unsigned short", "0x4640"),
              "float16": ("unsigned short", "0x7200"), "float64": ("unsigned long long", "0x40C81C8000000000ull")}

EXPECT = {"wrong_value": "wrong value", "cached_result": "stale state", "oob_write": "out-of-bounds write",
          "race": "race", "uninit_read": "uninitialized read"}

TAIL = {
    "wrong_value": """__global__ void planted_wrong(void* out) {{ *({bits_t}*)out = {bits}; }}
{sig} {{
  expert_reference({names});
  planted_wrong<<<1, 1>>>((void*){out});
}}""",
    "cached_result": """static bool planted_done = false;
{sig} {{
  if (planted_done) return;   // assumes the inputs never change between calls
  planted_done = true;
  expert_reference({names});
}}""",
    "oob_write": """__global__ void planted_oob(unsigned char* out) {{ out[-1] = 0; }}
{sig} {{
  expert_reference({names});
  planted_oob<<<1, 1>>>((unsigned char*){out});
}}""",
    "race": """__global__ void planted_race(float* sink) {{
  __shared__ float s[64];
  s[threadIdx.x] = (float)threadIdx.x;
  float v = s[(threadIdx.x + 33) % 64];   // read another warp's slot with no __syncthreads()
  if (v == -1.0f) sink[0] = v;
}}
{sig} {{
  expert_reference({names});
  planted_race<<<1, 64>>>((float*){out});
}}""",
    "uninit_read": """__global__ void planted_uninit(const float* scratch, float* sink) {{
  if (scratch[threadIdx.x] == 12345.0f) sink[0] = 0.0f;   // scratch was never written
}}
{sig} {{
  float* scratch = nullptr;
  cudaMalloc(&scratch, 256 * sizeof(float));
  expert_reference({names});
  planted_uninit<<<1, 64>>>(scratch, (float*){out});
  cudaFree(scratch);
}}""",
}


def planted_sources(tid):
    task = json.loads((TASKS / tid / "tests" / "task.json").read_text())
    from graders.grade import parse_signature
    params = parse_signature(task["signature"])
    names = ", ".join(n for _, n in params)
    out = next(iter(task["outputs"]))
    bits_t, bits = WRONG_BITS[task["outputs"][out]["dtype"]]
    sig = task["signature"].strip().rstrip(";")
    expert = (TASKS / tid / "solution" / "solution.cu").read_text()
    head = f"// PLANTED from the expert kernel of {tid}.\n#define reference expert_reference\n"
    res = {}
    for kind, t in TAIL.items():
        body = t.format(sig=sig, names=names, out=out, bits_t=bits_t, bits=bits)
        res[kind] = head + expert + f"\n#undef reference\n// ---- planted part: expected label \"{EXPECT[kind]}\"\n" + body + "\n"
    return res


def write_planted(dest, ids=None):
    from graders.grade import task_ids
    files = []
    for tid in ids or task_ids():
        d = Path(dest) / tid
        d.mkdir(parents=True, exist_ok=True)
        for kind, s in planted_sources(tid).items():
            p = d / f"{kind}.cu"
            p.write_text(s)
            files.append((tid, p))
    return files
