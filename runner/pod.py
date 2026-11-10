#!/usr/bin/env python3
"""pod.py: one RunPod GPU pod for the eval. Key: RUNPOD_API_KEY (environment). The SSH key used is the one
already on the RunPod account (id_ed25519_personal). No model key ever goes on the pod.

  balance                 account balance and spend per hour
  create --confirm        create the pod (RTX PRO 6000 first, RTX 5090 as fallback), wait for SSH
  status                  pod state, GPU, cost per hour, SSH host and port
  ssh                     print `host port` for the agent loop and the grader
  touch                   mark activity (the agent loop does this on every step)
  push | pull PATH        copy submission/ (tar over ssh) to /root/submission on the pod; pull a results folder back
  seal                    before the model tries: delete /root/submission (expert kernels, tests) and lock /root
  stop | terminate        stop (disk kept, small storage charge) or delete the pod
  watchdog                every 60 s: terminate if no activity for 15 min or balance under $3
State: micro/logs/pod.json; at terminate: micro/logs/pod_final.json and a line in micro/logs/spend.jsonl.
"""
import argparse, datetime as dt, json, os, shlex, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo
import urllib.error, urllib.request

ET = ZoneInfo("America/New_York")
PROJECT = Path(__file__).resolve().parents[2]
LOGS = PROJECT / "micro" / "logs"
STATE, FINAL, SPEND, HEARTBEAT = LOGS / "pod.json", LOGS / "pod_final.json", LOGS / "spend.jsonl", LOGS / "runner.heartbeat"
REST, GQL = "https://rest.runpod.io/v1", "https://api.runpod.io/graphql"
GPUS = ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA GeForce RTX 5090"]
IMAGE = "runpod/pytorch:1.5.1-cu1281-torch291-ubuntu2404"  # CUDA 12.8.1 devel (nvcc) + PyTorch 2.9.1 built for cu128: same CUDA on both sides
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_personal")
IDLE_MIN, MIN_BALANCE = 15, 3.0


def now_et():
    return dt.datetime.now(ET).isoformat(timespec="seconds")


def hdr():
    return {"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json",
            "User-Agent": "cuda-bench-pod/1"}


def http(method, url, body=None, timeout=60):
    """Standard library only, so system python3 can run this file. Returns (status, parsed JSON or text)."""
    req = urllib.request.Request(url, method=method, headers=hdr(), data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read().decode(errors="replace")
    try:
        return code, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return code, raw


def balance():
    _, j = http("POST", GQL, {"query": "query { myself { clientBalance currentSpendPerHr } }"})
    me = j["data"]["myself"]
    return {"balance_usd": me["clientBalance"], "spend_per_hr": me["currentSpendPerHr"], "time_et": now_et()}


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(s):
    LOGS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def get_pod(pid):
    code, j = http("GET", f"{REST}/pods/{pid}")
    return j if code == 200 and isinstance(j, dict) else None


def ssh_addr(pod):
    ip, pm = pod.get("publicIp"), pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    return (ip, int(port)) if ip and port else (None, None)


def ssh_ok(host, port):
    p = subprocess.run(["ssh", "-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10", f"root@{host}", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -1"],
                       capture_output=True, text=True, timeout=60)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def cmd_balance(a):
    print(json.dumps(balance()))


def cmd_create(a):
    if not a.confirm:
        raise SystemExit("refusing: pass --confirm (only after the parent says 'pod on')")
    s = state()
    if s and not s.get("terminated_et"):
        raise SystemExit(f"a pod is already recorded in {STATE}: {s['id']}; terminate it first")
    b = balance()
    if b["balance_usd"] < 5:
        raise SystemExit(f"balance ${b['balance_usd']} is under $5; not creating")
    last = None
    for cloud in a.cloud.split(","):
        body = {"name": "cuda-bench", "imageName": IMAGE, "gpuTypeIds": GPUS, "gpuTypePriority": "custom", "gpuCount": 1,
                "cloudType": cloud, "containerDiskInGb": 80, "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
                "allowedCudaVersions": ["12.8", "12.9", "13.0"], "minVCPUPerGPU": 8, "minRAMPerGPU": 32}
        code, j = http("POST", f"{REST}/pods", body, timeout=120)
        if code in (200, 201) and isinstance(j, dict):
            pod = j
            break
        last = f"{cloud}: HTTP {code} {str(j)[:300]}"
        print("create failed, " + last, flush=True)
    else:
        raise SystemExit("no pod created: " + str(last))
    s = {"id": pod["id"], "created_et": now_et(), "cloud": cloud, "gpu": (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpu"),
         "cost_per_hr": pod.get("costPerHr"), "image": IMAGE, "balance_at_create": b["balance_usd"]}
    save_state(s)
    print(json.dumps(s), flush=True)
    t0 = time.time()
    while time.time() - t0 < a.wait_min * 60:
        pod = get_pod(s["id"]) or {}
        host, port = ssh_addr(pod)
        if host:
            ok, out = ssh_ok(host, port)
            if ok:
                s.update(host=host, port=port, ready_et=now_et(), gpu_check=out, cost_per_hr=pod.get("costPerHr", s["cost_per_hr"]))
                save_state(s)
                print(f"ready: {host} {port}\n{out}")
                return
        time.sleep(15)
    print(f"not ready after {a.wait_min} min; check `status`, then terminate if stuck", file=sys.stderr)
    sys.exit(1)


def cmd_status(a):
    s = state()
    if not s:
        print(json.dumps({"pod": None, **balance()}))
        return
    pod = get_pod(s["id"])
    host, port = ssh_addr(pod or {})
    print(json.dumps({"state": s, "desiredStatus": (pod or {}).get("desiredStatus"), "costPerHr": (pod or {}).get("costPerHr"),
                      "ssh": [host, port], **balance()}, indent=1))


def cmd_ssh(a):
    s = state()
    pod = get_pod(s["id"]) if s else None
    host, port = ssh_addr(pod or {})
    if not host:
        raise SystemExit("no running pod with SSH")
    print(host, port)


def cmd_touch(a):
    LOGS.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(now_et() + "\n")


def _ssh_base():
    s = state() or sys.exit("no pod recorded")
    host, port = s.get("host"), s.get("port")
    if not host:
        host, port = ssh_addr(get_pod(s["id"]) or {})
    return ["-p", str(port), "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR"], host


def cmd_push(a):
    """Copy submission/ to /root/submission on the pod (without results/trials and transcripts)."""
    opts, host = _ssh_base()
    sub = Path(__file__).resolve().parents[1]
    ssh = " ".join(["ssh"] + opts + [f"root@{host}"])
    cmd = (f"COPYFILE_DISABLE=1 tar --no-xattrs -C {shlex.quote(str(sub))} --exclude ./results/trials --exclude ./transcripts --exclude __pycache__ --exclude ./.env "
           f"-czf - . | {ssh} 'rm -rf /root/submission && mkdir -p /root/submission && tar --warning=no-unknown-keyword -xzf - -C /root/submission && echo pushed'")
    sys.exit(subprocess.call(cmd, shell=True))


def cmd_pull(a):
    """Copy a folder under submission/ back from the pod, e.g. results/grade_logs."""
    opts, host = _ssh_base()
    sub = Path(__file__).resolve().parents[1]
    rel = a.path.strip("/")
    (sub / rel).mkdir(parents=True, exist_ok=True)
    ssh = " ".join(["ssh"] + opts + [f"root@{host}"])
    sys.exit(subprocess.call(f"{ssh} 'tar -czf - -C /root/submission/{rel} .' | tar -xzf - -C {shlex.quote(str(sub / rel))}", shell=True))


def cmd_seal(a):
    """Before the model tries: delete /root/submission (expert kernels, grader, tests) and lock /root.
    Push it again for grading (P6)."""
    opts, host = _ssh_base()
    sys.exit(subprocess.call(["ssh"] + opts + [f"root@{host}", "rm -rf /root/submission && chmod 700 /root && "
                              "ls -ld /root && test ! -e /root/submission && echo sealed"]))


def cmd_stop(a):
    s = state() or sys.exit("no pod recorded")
    code, j = http("POST", f"{REST}/pods/{s['id']}/stop")
    print(code, str(j)[:300])


def terminate(reason):
    s = state()
    if not s or s.get("terminated_et"):
        print("no live pod recorded")
        return
    code, _ = http("DELETE", f"{REST}/pods/{s['id']}")
    ok = code in (200, 204) or get_pod(s["id"]) is None
    time.sleep(5)
    b = balance()
    s.update(terminated_et=now_et(), terminate_reason=reason, terminate_http=code, terminated=ok,
             balance_after=b["balance_usd"], spent_usd=round(s["balance_at_create"] - b["balance_usd"], 4))
    save_state(s)
    FINAL.write_text(json.dumps(s, indent=1))
    with open(SPEND, "a") as fh:
        fh.write(json.dumps({"time_et": now_et(), "source": "runpod", "dollars": s["spent_usd"], "pod": s["id"],
                             "note": "balance at create minus balance after terminate"}) + "\n")
    print(json.dumps(s, indent=1))


def cmd_terminate(a):
    terminate(a.reason)


def cmd_watchdog(a):
    while True:
        s = state()
        if not s or s.get("terminated_et"):
            print(f"{now_et()} no live pod; watchdog ends", flush=True)
            return
        try:
            b = balance()
        except (urllib.error.URLError, OSError) as e:
            print(f"{now_et()} balance check failed ({e}); retry in 60 s", flush=True)
            time.sleep(60)
            continue
        hb = HEARTBEAT.stat().st_mtime if HEARTBEAT.exists() else time.mktime(time.strptime(s["created_et"][:19], "%Y-%m-%dT%H:%M:%S"))
        idle = (time.time() - max(hb, os.path.getmtime(STATE))) / 60
        print(f"{now_et()} balance ${b['balance_usd']:.2f}, idle {idle:.1f} min", flush=True)
        if b["balance_usd"] < MIN_BALANCE:
            return terminate(f"watchdog: balance ${b['balance_usd']} under ${MIN_BALANCE}")
        if idle > a.idle_min:
            return terminate(f"watchdog: no activity for {idle:.0f} min")
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(fn=cmd_balance)
    p = sub.add_parser("create")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--cloud", default="COMMUNITY,SECURE", help="tried in order")
    p.add_argument("--wait-min", type=float, default=20)
    p.set_defaults(fn=cmd_create)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("ssh").set_defaults(fn=cmd_ssh)
    sub.add_parser("touch").set_defaults(fn=cmd_touch)
    sub.add_parser("push").set_defaults(fn=cmd_push)
    p = sub.add_parser("pull")
    p.add_argument("path")
    p.set_defaults(fn=cmd_pull)
    sub.add_parser("seal").set_defaults(fn=cmd_seal)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    p = sub.add_parser("terminate")
    p.add_argument("--reason", default="manual")
    p.set_defaults(fn=cmd_terminate)
    p = sub.add_parser("watchdog")
    p.add_argument("--idle-min", type=float, default=IDLE_MIN)
    p.set_defaults(fn=cmd_watchdog)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
