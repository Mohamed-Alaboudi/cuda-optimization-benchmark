#!/usr/bin/env python3
"""Agent loop: one model tries one CUDA task alone, with tools, under a wall-clock cap.

Runs on the Mac. Model calls go out from here; only the bash and write_file tools run on the
pod over SSH (or in a local folder for a dry run). No API key or token ever leaves this machine.

One trial = fresh folder (and on the pod its own unix user) + the task's public files + a loop:
  model -> tool calls -> results -> model ... until the model calls `done`, the wall cap, the cost cap,
  or an error. Every request, response, tool call and result goes into one transcript JSON.
"""
import datetime as dt, json, os, re, shlex, shutil, subprocess, tempfile, threading, time
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
SUB = HERE.parent
PROJECT = SUB.parent
LOGS = Path(os.environ.get("MICRO_LOGS", PROJECT / "micro" / "logs"))
SPEND = LOGS / "spend.jsonl"   # one line per charge: time_et, source, dollars
HEARTBEAT = LOGS / "runner.heartbeat"  # pod.py watchdog reads it
_ledger_lock = threading.Lock()
OUT_LIMIT = 20000          # characters of tool output shown to the model (head + tail)
MAX_CMD_TIMEOUT = 600      # seconds, per bash command
DEFAULT_CMD_TIMEOUT = 300
IDLE_TIMEOUT = 600         # seconds without a streamed byte before a retry


def now_et():
    return dt.datetime.now(ET).isoformat(timespec="seconds")


def clip(text, limit=OUT_LIMIT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... {len(text) - limit} characters cut ...]\n" + text[-half:]


TOOLS = [
    {"name": "bash", "description": "Run a bash command in your task folder (the current directory). "
     "stdout and stderr are returned together, cut to 20,000 characters. Default timeout 300 s, at most 600 s.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"},
         "timeout_s": {"type": "integer", "description": "seconds, 1 to 600"}}, "required": ["command"]}},
    {"name": "write_file", "description": "Write a text file inside your task folder (path relative to it). Overwrites.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "done", "description": "Finish. Call this once solution.cu in your task folder holds your final kernel.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


# ---------------- executors ----------------

class LocalExecutor:
    """Dry run on the Mac: a fresh folder per trial, no separate user, no GPU."""

    def __init__(self, base):
        self.base = Path(base)

    def setup(self, slug, files):
        self.dir = self.base / slug
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        for src in files:
            shutil.copy(src, self.dir / Path(src).name)
        return str(self.dir)

    def run(self, command, timeout):
        t0 = time.time()
        try:
            p = subprocess.run(["bash", "-c", command], cwd=self.dir, capture_output=True, timeout=timeout,
                               env={"PATH": os.environ["PATH"], "HOME": str(self.dir)}, start_new_session=True)
            return p.returncode, (p.stdout + p.stderr).decode(errors="replace"), time.time() - t0
        except subprocess.TimeoutExpired as e:
            out = ((e.stdout or b"") + (e.stderr or b"")).decode(errors="replace")
            return 124, out + f"\n[timeout after {timeout} s]", time.time() - t0

    def write(self, rel, content):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def fetch(self, rel):
        p = self.dir / rel
        return p.read_bytes() if p.exists() else None

    def listing(self):
        return self.run("ls -la", 30)[1]

    def cleanup(self):
        pass


class SSHExecutor:
    """Pod: /trials/<slug> owned by its own unix user; commands run as that user with `timeout`."""

    def __init__(self, host, port, key=os.path.expanduser("~/.ssh/id_ed25519_personal")):
        self.ssh = ["ssh", "-p", str(port), "-i", key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "ControlMaster=auto",
                    "-o", f"ControlPath=/tmp/micro-cm-%r@%h-%p", "-o", "ControlPersist=30m", f"root@{host}"]

    def _root(self, script, timeout=120, data=None):
        p = subprocess.run(self.ssh + ["bash -s"], input=(script.encode() if data is None else data),
                           capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).decode(errors="replace")

    def setup(self, slug, files):
        self.slug, self.user, self.dir = slug, "t_" + re.sub(r"[^a-z0-9]", "", slug.lower())[-28:], f"/trials/{slug}"
        u, d = self.user, self.dir
        for attempt in range(4):  # concurrent useradd calls can hit the /etc/passwd lock, and ssh can drop
            rc, out = self._root(f"set -e; pkill -u {u} || true; userdel {u} 2>/dev/null || true; rm -rf {d}; mkdir -p /trials; "
                                 f"chmod 711 /trials; useradd -M -d {d} -s /bin/bash {u}; mkdir -p {d}; chown {u}:{u} {d}; chmod 700 {d}")
            if rc == 0:
                break
            time.sleep(3 + 5 * attempt)
        if rc:
            raise RuntimeError("pod setup failed: " + out[-500:])
        for src in files:
            self.write(Path(src).name, Path(src).read_text())
        return d

    def run(self, command, timeout):
        t0 = time.time()
        env = ("export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
               "LD_LIBRARY_PATH=/usr/local/cuda/lib64 CUDA_HOME=/usr/local/cuda; ")  # ssh sessions do not get the image ENV
        inner = f"{env}cd {self.dir} && HOME={self.dir} timeout -k 5 {int(timeout)} bash -c {shlex.quote(command)}"
        script = f"runuser -u {self.user} -- bash -c {shlex.quote(inner)} 2>&1"
        try:
            rc, out = self._root(script, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            rc, out = 124, f"[timeout after {timeout} s]"
        if rc == 124:
            out += f"\n[timeout after {timeout} s]"
        return rc, out, time.time() - t0

    def write(self, rel, content):
        dest = f"{self.dir}/{rel}"
        script = (f"runuser -u {self.user} -- bash -c {shlex.quote(f'mkdir -p $(dirname {shlex.quote(dest)}) && cat > {shlex.quote(dest)}')}")
        p = subprocess.run(self.ssh + [script], input=content.encode(), capture_output=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stderr.decode(errors="replace")[-500:])

    def fetch(self, rel):
        p = subprocess.run(self.ssh + [f"cat {shlex.quote(self.dir + '/' + rel)}"], capture_output=True, timeout=120)
        return p.stdout if p.returncode == 0 else None

    def listing(self):
        return self._root(f"ls -la {self.dir}", 60)[1]

    def cleanup(self):
        self._root(f"pkill -u {self.user} || true", 60)


APP = re.compile(r"/app(?=/|\b)")


def map_app(text, folder):
    """The prompts name the task folder /app (as in a container). Here each try has its own folder, so /app maps to it."""
    return APP.sub(folder, text)


def safe_rel(path):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not str(path).strip():
        raise ValueError("path must be relative to the task folder, without '..'")
    return str(p)


# ---------------- streaming helpers ----------------

def sse_events(resp, deadline):
    """Yield parsed JSON events from a server-sent-events response; stop at the wall deadline."""
    for line in resp.iter_lines():
        if time.time() > deadline:
            raise TimeoutError("wall cap reached during the model call")
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        yield json.loads(data)


def post_stream(url, headers, body, deadline, parse, tries=6, log=print):
    import requests
    for attempt in range(1, tries + 1):
        if time.time() > deadline:
            raise TimeoutError("wall cap reached before the model call")
        try:
            with requests.post(url, headers=headers, json=body, stream=True, timeout=(30, IDLE_TIMEOUT)) as r:
                if r.status_code == 200:
                    out = parse(sse_events(r, deadline))
                    if out is not None:
                        return out, attempt
                    err = "stream ended early"
                elif r.status_code in (408, 409, 429, 529) or r.status_code >= 500:
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
        except requests.RequestException as e:
            err = f"network: {type(e).__name__}"
        wait = min(120, 5 * 2 ** (attempt - 1))
        log(f"retry {attempt}/{tries} in {wait}s: {err}")
        time.sleep(wait)
    raise RuntimeError(f"gave up after {tries} tries")


# ---------------- providers ----------------
# Each provider keeps its own native history. step() returns
# {"text", "calls": [{"id", "name", "args"}], "usage": {...}, "raw": <assembled response>}.

class OpenAIResponses:
    """gpt-6.1-sol: tool calling needs the Responses API; history chained with previous_response_id."""

    def __init__(self, cfg, system, prompt):
        self.cfg, self.prev = cfg, None
        self.pending = [{"role": "developer", "content": system}, {"role": "user", "content": prompt}]

    def step(self, deadline, log):
        body = {"model": self.cfg["model"], "input": self.pending, "store": True, "stream": True,
                "tools": [{"type": "function", **t} for t in TOOLS], **self.cfg.get("params", {})}
        if self.prev:
            body["previous_response_id"] = self.prev
        key = os.environ[self.cfg["key_env"]]

        def parse(events):
            for ev in events:
                if ev.get("type") in ("response.completed", "response.incomplete"):
                    return ev["response"]
                if ev.get("type") == "response.failed":
                    raise RuntimeError("response.failed: " + json.dumps(ev.get("response", {}).get("error"))[:500])
            return None

        resp, tries = post_stream("https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}"},
                                  body, deadline, parse, log=log)
        self.prev, self.pending = resp["id"], []
        text, calls = [], []
        for item in resp.get("output", []):
            if item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
            elif item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "name": item["name"], "args": item.get("arguments") or "{}"})
        u = resp.get("usage") or {}
        usage = {"input": u.get("input_tokens", 0), "cached_input": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                 "output": u.get("output_tokens", 0), "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        return {"text": "\n".join(text), "calls": calls, "usage": usage, "raw": resp, "tries": tries,
                "stop": resp.get("status")}

    def add_results(self, results):
        self.pending += [{"type": "function_call_output", "call_id": cid, "output": out} for cid, out in results]

    def nudge(self, text):
        self.pending.append({"role": "user", "content": text})


class AnthropicMessages:
    """Claude through the Anthropic Messages API with a plain API key (ANTHROPIC_API_KEY)."""

    def __init__(self, cfg, system, prompt):
        self.cfg, self.system = cfg, system
        self.messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def _headers(self):
        return {"x-api-key": os.environ[self.cfg.get("key_env") or "ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    def _system(self):
        return [{"type": "text", "text": self.system}]

    def step(self, deadline, log):
        for m in self.messages:  # one cache breakpoint, on the newest user turn
            for c in m["content"] if isinstance(m["content"], list) else []:
                c.pop("cache_control", None)
        self.messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        body = {"model": self.cfg["model"], "stream": True, "messages": self.messages,
                "system": self._system(),
                "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS],
                **self.cfg.get("params", {})}

        def parse(events):
            blocks, usage, stop, model = {}, {}, None, None
            for ev in events:
                t = ev.get("type")
                if t == "message_start":
                    usage.update(ev["message"].get("usage") or {})
                    model = ev["message"].get("model")
                elif t == "content_block_start":
                    blocks[ev["index"]] = dict(ev["content_block"], _json="")
                elif t == "content_block_delta":
                    b, d = blocks[ev["index"]], ev["delta"]
                    if d["type"] == "text_delta":
                        b["text"] = b.get("text", "") + d["text"]
                    elif d["type"] == "thinking_delta":
                        b["thinking"] = b.get("thinking", "") + d["thinking"]
                    elif d["type"] == "signature_delta":
                        b["signature"] = b.get("signature", "") + d["signature"]
                    elif d["type"] == "input_json_delta":
                        b["_json"] += d["partial_json"]
                elif t == "message_delta":
                    stop = (ev.get("delta") or {}).get("stop_reason")
                    usage.update(ev.get("usage") or {})
                elif t == "error":
                    raise RuntimeError("stream error: " + json.dumps(ev.get("error"))[:500])
                elif t == "message_stop":
                    content = []
                    for i in sorted(blocks):
                        b = blocks[i]
                        js = b.pop("_json")
                        if b["type"] == "tool_use":
                            b["input"] = json.loads(js) if js else {}
                        content.append(b)
                    return {"model": model, "content": content, "stop_reason": stop, "usage": usage}
            return None

        resp, tries = post_stream("https://api.anthropic.com/v1/messages", self._headers(), body, deadline, parse, log=log)
        self.messages.append({"role": "assistant", "content": resp["content"]})
        text = "\n".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "args": b["input"]} for b in resp["content"] if b["type"] == "tool_use"]
        u = resp["usage"]
        usage = {"input": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                 "cached_input": u.get("cache_read_input_tokens", 0), "output": u.get("output_tokens", 0),
                 "reasoning": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0)}
        return {"text": text, "calls": calls, "usage": usage, "raw": resp, "tries": tries, "stop": resp["stop_reason"]}

    def add_results(self, results):
        self.messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": out}
                                                          for cid, out in results]})

    def nudge(self, text):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


class ChatCompletions:
    """Kimi k3 (OpenAI-compatible chat completions with tools); reasoning_content is sent back each turn."""

    def __init__(self, cfg, system, prompt):
        self.cfg = cfg
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    def step(self, deadline, log):
        body = {"model": self.cfg["model"], "messages": self.messages, "stream": True,
                "tools": [{"type": "function", "function": t} for t in TOOLS], **self.cfg.get("params", {})}
        key = os.environ[self.cfg["key_env"]]

        def parse(events):
            text, reasoning, calls, finish, usage, model = [], [], {}, None, {}, None
            for ev in events:
                model = ev.get("model") or model
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    text.append(d.get("content") or "")
                    reasoning.append(d.get("reasoning_content") or "")
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                        c["id"] = tc.get("id") or c["id"]
                        f = tc.get("function") or {}
                        c["name"] += f.get("name") or ""
                        c["args"] += f.get("arguments") or ""
                    if ch.get("usage"):
                        usage = ch["usage"]
                    finish = ch.get("finish_reason") or finish
            if finish is None:
                return None
            return {"model": model, "text": "".join(text), "reasoning": "".join(reasoning),
                    "calls": [calls[i] for i in sorted(calls)], "finish": finish, "usage": usage}

        resp, tries = post_stream(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                                  {"Authorization": f"Bearer {key}"}, body, deadline, parse, log=log)
        msg = {"role": "assistant", "content": resp["text"]}
        if resp["reasoning"]:
            msg["reasoning_content"] = resp["reasoning"]
        if resp["calls"]:
            msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                 for c in resp["calls"]]
        self.messages.append(msg)
        u = resp["usage"]
        usage = {"input": u.get("prompt_tokens", 0), "cached_input": u.get("cached_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                 "output": u.get("completion_tokens", 0), "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}
        return {"text": resp["text"], "calls": resp["calls"], "usage": usage, "raw": resp, "tries": tries, "stop": resp["finish"]}

    def add_results(self, results):
        self.messages += [{"role": "tool", "tool_call_id": cid, "content": out} for cid, out in results]

    def nudge(self, text):
        self.messages.append({"role": "user", "content": text})


class StubModel:
    """Dry run: a scripted model. script names: 'ok' (write, compile stand-in, done), 'loop' (never stops)."""

    def __init__(self, cfg, system, prompt):
        self.cfg, self.i, self.script = cfg, 0, cfg.get("script", "ok")

    def step(self, deadline, log):
        time.sleep(0.2)
        self.i += 1
        if self.script == "loop":
            calls = [{"id": f"c{self.i}", "name": "bash", "args": {"command": "sleep 1; echo working", "timeout_s": 5}}]
        else:
            plan = {1: [{"id": "c1", "name": "write_file", "args": {"path": "solution.cu", "content": "// stub kernel\nextern \"C\" void f() {}\n"}},
                        {"id": "c1b", "name": "write_file", "args": {"path": "../escape.txt", "content": "x"}}],
                    2: [{"id": "c2", "name": "bash", "args": {"command": "ls; wc -c solution.cu; whoami; sleep 3", "timeout_s": 1}}],
                    3: [{"id": "c3", "name": "bash", "args": {"command": "cat solution.cu && echo compiled-ok"}}],
                    4: [{"id": "c4", "name": "done", "args": {"summary": "stub finished"}}]}
            calls = plan.get(self.i, plan[4])
        return {"text": f"stub step {self.i}", "calls": calls, "usage": {"input": 1000, "cached_input": 0, "output": 2000, "reasoning": 1000},
                "raw": {"stub": self.i}, "tries": 1, "stop": "tool_calls"}

    def add_results(self, results):
        pass

    def nudge(self, text):
        pass


PROVIDERS = {"openai_responses": OpenAIResponses, "anthropic_api": AnthropicMessages,
             "chat_completions": ChatCompletions, "stub": StubModel}


def dollars(usage, price):
    out = usage["output"] + (usage["reasoning"] if price.get("reasoning_separate") else 0)
    return ((usage["input"] - usage["cached_input"]) * price["input"] + usage["cached_input"] * price.get("cached_input", price["input"])
            + out * price["output"]) / 1e6


def spend_log(row):
    with _ledger_lock:
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        with open(SPEND, "a") as fh:
            fh.write(json.dumps(row) + "\n")


def heartbeat():
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(now_et() + "\n")


# ---------------- one trial ----------------

def run_trial(model_cfg, task_id, task_files, prompt, system, executor, wall_s, trial_cap_usd, transcript_path, log=print):
    slug = f"{model_cfg['name']}-{task_id}-{transcript_path.stem}"
    t0, deadline = time.time(), time.time() + wall_s
    tr = {"trial": slug, "model": model_cfg["name"], "model_id": model_cfg["model"], "provider": model_cfg["provider"],
          "params": model_cfg.get("params", {}), "task": task_id, "started_et": now_et(), "wall_cap_s": wall_s,
          "trial_cap_usd": trial_cap_usd, "system": system, "prompt": prompt, "events": [], "usd": 0.0}

    def save():
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(json.dumps(tr, indent=1, default=str))

    tr["folder"] = executor.setup(slug, task_files)
    system = system + (f"\nIn this run your task folder is {tr['folder']}. Where prompt.md says /app, it means this folder; "
                       "paths that start with /app are mapped to it.\n")
    tr["system"] = system
    model = PROVIDERS[model_cfg["provider"]](model_cfg, system, prompt)
    status, turn = None, 0
    try:
        while status is None:
            if time.time() > deadline:
                status = "wall_cap"
                break
            if trial_cap_usd is not None and tr["usd"] >= trial_cap_usd:
                status = "cost_cap"
                break
            turn += 1
            heartbeat()
            c0 = time.time()
            try:
                step = model.step(deadline, lambda m: log(f"  {slug}: {m}"))
            except TimeoutError:
                status = "wall_cap"
                break
            usd = dollars(step["usage"], model_cfg["price"])
            tr["usd"] += usd
            spend_log({"time_et": now_et(), "source": f"model:{model_cfg['name']}", "dollars": round(usd, 6),
                       "trial": slug, "turn": turn,
                       **{k: step["usage"][k] for k in ("input", "cached_input", "output", "reasoning")}})
            tr["events"].append({"kind": "model", "turn": turn, "et": now_et(), "seconds": round(time.time() - c0, 1),
                                 "http_tries": step["tries"], "stop": step["stop"], "usage": step["usage"], "usd": round(usd, 6),
                                 "text": step["text"], "calls": step["calls"], "response": step["raw"]})
            if not step["calls"]:
                model.nudge("Use the tools to work on the task. Call done when solution.cu holds your final kernel.")
                tr["events"].append({"kind": "nudge", "turn": turn})
                if sum(e["kind"] == "nudge" for e in tr["events"]) >= 3:
                    status = "no_tool_calls"
                save()
                continue
            results = []
            for c in step["calls"]:
                try:
                    args = c["args"] if isinstance(c["args"], dict) else json.loads(c["args"] or "{}")
                except json.JSONDecodeError as e:
                    args, out = None, f"error: arguments are not valid JSON ({e})"
                ev = {"kind": "tool", "turn": turn, "name": c["name"], "args": args, "et": now_et()}
                if args is None:
                    pass
                elif c["name"] == "done":
                    status, out = "done", "ok"
                elif c["name"] == "write_file":
                    try:
                        path = map_app(args["path"], tr["folder"])
                        if path.startswith(tr["folder"] + "/"):
                            path = path[len(tr["folder"]) + 1:]
                        executor.write(safe_rel(path), args["content"])
                        out = f"wrote {args['path']} ({len(args['content'])} characters)"
                    except Exception as e:
                        out = f"error: {e}"
                elif c["name"] == "bash":
                    remaining = deadline - time.time()
                    tmo = max(1, min(int(args.get("timeout_s") or DEFAULT_CMD_TIMEOUT), MAX_CMD_TIMEOUT, int(remaining)))
                    rc, text, secs = executor.run(map_app(args["command"], tr["folder"]), tmo)
                    ev.update(exit_code=rc, seconds=round(secs, 1), timeout_s=tmo, output=text)
                    out = clip(f"exit code {rc}\n{text}")
                else:
                    out = f"error: unknown tool {c['name']}"
                ev["result"] = out
                tr["events"].append(ev)
                results.append((c["id"], out))
                heartbeat()
            model.add_results(results)
            save()
    except Exception as e:
        status = "error"
        tr["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        sol = executor.fetch("solution.cu")
        try:
            tr["final_listing"] = executor.listing()
            executor.cleanup()
        except Exception as e:
            tr["final_listing"] = f"listing failed: {e}"
    tr.update(status=status, turns=turn, seconds=round(time.time() - t0, 1), ended_et=now_et(),
              has_solution=sol is not None, usd=round(tr["usd"], 6))
    save()
    return tr, sol
