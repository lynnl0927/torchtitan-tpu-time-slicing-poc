"""Disaggregated, time-sliced TPU workload — one gen or train half of an RL job.

Each RL job is TWO of these processes in one RayCluster: a generator pod on the
generator node and a trainer pod on the trainer node. Each holds its node's 8
chips only while inside acquire/yield, exposes /checkpoint & /restore (HAL C/R
on its own PID) for the orchestrator to call back, and enforces the RL loop's
gen->train data dependency by passing a token to its peer after every slice:

    gen step k  --token-->  train step k  --token-->  gen step k+1  ...

Two jobs' gen pods contend on gen-pool and their train pods on train-pool, so
the jobs settle into anti-phase (A trains while B generates) purely through the
orchestrator's per-pool locks — that is the utilization win of oversubscribing
a disaggregated setup.

Env:
  ORCH_URL    base URL of the orchestrator (e.g. http://ts-orchestrator:9000)
  WID         workload id (e.g. job-a-gen)
  POOL        pool this workload contends on (gen-pool | train-pool)
  ROLE        gen | train (gen kicks off step 1 without waiting for a token)
  PEER_WID    workload id of the other half of this job (e.g. job-a-train)
  STEPS       number of gen->train cycles (default 3)
  WL_PORT     port for this workload's own C/R + token server (default 9100)
  SLICE_SECS  simulated compute time per slice (default 2)
"""
import glob
import json
import os
import queue
import re
import select
import socket
import struct
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORCH_URL = os.environ["ORCH_URL"]
WID = os.environ["WID"]
POOL = os.environ["POOL"]
ROLE = os.environ["ROLE"]
PEER_WID = os.environ["PEER_WID"]
STEPS = int(os.environ.get("STEPS", "3"))
WL_PORT = int(os.environ.get("WL_PORT", "9100"))
SLICE_SECS = float(os.environ.get("SLICE_SECS", "2"))

_dev_lock = threading.Lock()   # serialize all device ops within this process
_state = {}                    # a, b, baseline
_torch = {}                    # dev handle
_token = queue.Queue()         # peer -> us: "your turn" signals
_self_url = None


def log(m):
    print(f"[{WID} {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- libtpu v2 C/R: gVisor tpu_control pipe protocol -----------------------
# With LIBTPU_CHECKPOINTING_ENABLED=true, each TPU process spawns a control
# thread named libtpu{RRRRSSSS} (hex request-write / response-read pipe FDs).
# Checkpoint/Restore are 4-byte-BE length-prefixed protobuf messages through
# /proc/self/fd/<N>. (v2 removed the old TpuHalService gRPC UDS.)
_HAL_THREAD_RE = re.compile(r"^libtpu([0-9a-fA-F]{4})([0-9a-fA-F]{4})$")
_HAL_ACTIONS = {"Checkpoint": 2, "Restore": 3}


def _varint(val):
    out = b""
    while val >= 0x80:
        out += bytes([val & 0x7F | 0x80])
        val >>= 7
    return out + bytes([val])


def _read_varint(data, i):
    val, shift = 0, 0
    while True:
        b = data[i]
        val |= (b & 0x7F) << shift
        i += 1
        if b < 0x80:
            return val, i
        shift += 7


def own_pipes():
    """[(req_write_fd, rsp_read_fd)] for this process's libtpu control threads."""
    pipes = []
    for comm in glob.glob("/proc/self/task/*/comm"):
        try:
            m = _HAL_THREAD_RE.match(open(comm).read().strip())
        except OSError:
            continue
        if m:
            pipes.append((int(m.group(1), 16), int(m.group(2), 16)))
    return sorted(pipes)


def _pipe_control(pipe, action, timeout):
    req_fd, rsp_fd = pipe
    body = _varint((1 << 3) | 0) + _varint(_HAL_ACTIONS[action])
    body += _varint((2 << 3) | 0) + _varint(int(timeout))
    req = os.open(f"/proc/self/fd/{req_fd}", os.O_WRONLY)
    rsp = os.open(f"/proc/self/fd/{rsp_fd}", os.O_RDONLY)
    try:
        os.write(req, struct.pack(">I", len(body)) + body)
        deadline = time.time() + timeout

        def read_exact(n):
            buf = b""
            while len(buf) < n:
                r, _, _ = select.select([rsp], [], [],
                                        max(0.0, deadline - time.time()))
                if not r:
                    raise TimeoutError(f"no {action} response in {timeout}s")
                chunk = os.read(rsp, n - len(buf))
                if not chunk:
                    raise IOError("response pipe closed")
                buf += chunk
            return buf

        (size,) = struct.unpack(">I", read_exact(4))
        data = read_exact(size)
    finally:
        os.close(req)
        os.close(rsp)

    success, err, i = False, "", 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(data, i)
            if field == 1:
                success = bool(val)
        elif wire == 2:
            ln, i = _read_varint(data, i)
            if field == 3:
                err = data[i:i + ln].decode(errors="replace")
            i += ln
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return success, err


def hal(method, timeout=120):
    pipes = own_pipes()
    if not pipes:
        return False, "no-control-pipes", 0.0
    t0 = time.time()
    for pipe in pipes:
        try:
            ok, err = _pipe_control(pipe, method, timeout)
        except Exception as e:
            ok, err = False, str(e)
        if not ok:
            return False, err[:120], (time.time() - t0) * 1000
    return True, "", (time.time() - t0) * 1000


def init_device():
    torch = _torch["torch"]
    dev = _torch["dev"]
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    _state["a"], _state["b"] = a, b
    cs = (a @ b).sum().to("cpu").item()
    _state["baseline"] = cs
    log(f"init baseline={cs:.6f} pipes={own_pipes()}")
    return cs


def compute():
    cs = (_state["a"] @ _state["b"]).sum().to("cpu").item()
    match = abs(cs - _state["baseline"]) < 1e-2
    return cs, match


# ---- HTTP server: C/R called BY the orchestrator, /go called by the peer ----
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"status": "ok", "wid": WID, "pid": os.getpid()})

    def do_POST(self):
        path = self.path.strip("/")
        if path == "go":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            _token.put(body.get("step", 0))
            self._send(200, {"ok": True})
            return
        method = path.capitalize()  # /checkpoint -> Checkpoint
        if method not in ("Checkpoint", "Restore"):
            self._send(404, {"error": "unknown"})
            return
        with _dev_lock:
            ok, err, ms = hal(method)
        log(f"{method} ok={ok} {ms:.0f}ms {err}")
        self._send(200 if ok else 500, {"ok": ok, "ms": round(ms), "err": err})


def orch_post(path, timeout=900):
    req = urllib.request.Request(
        f"{ORCH_URL}/{path}",
        data=json.dumps({"workload_id": WID, "pool": POOL,
                         "url": _self_url}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def lookup_peer(timeout=300):
    """Poll the orchestrator registry until the peer workload registers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                f"{ORCH_URL}/lookup?wid={PEER_WID}", timeout=10)
            return json.loads(resp.read().decode())["url"]
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"peer {PEER_WID} never registered")


def send_token(step):
    url = lookup_peer()
    req = urllib.request.Request(
        f"{url}/go", data=json.dumps({"step": step}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(req, timeout=30)


def main():
    global _self_url
    import torch
    import torch_tpu  # noqa: F401
    _torch["torch"] = torch
    _torch["dev"] = torch.device("tpu")

    ip = socket.gethostbyname(socket.gethostname())
    _self_url = f"http://{ip}:{WL_PORT}"

    srv = ThreadingHTTPServer(("0.0.0.0", WL_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"C/R+token server on {_self_url} pool={POOL} role={ROLE}")

    orch_post("register")
    log("registered with orchestrator")

    for step in range(1, STEPS + 1):
        # RL data dependency: gen k+1 waits for train k; train k waits for gen k.
        # The gen pod starts step 1 unprompted.
        if not (ROLE == "gen" and step == 1):
            log(f"step={step} waiting for token from {PEER_WID}")
            _token.get()
        a = orch_post("acquire")
        with _dev_lock:
            if "baseline" not in _state:
                init_device()
            cs, match = compute()
        log(f"step={step} acquired(wait={a['wait_ms']}ms "
            f"restore={a['restore_ms']}ms) checksum={cs:.6f} "
            f"matches_baseline={match}")
        time.sleep(SLICE_SECS)  # simulated gen/train slice
        y = orch_post("yield")
        log(f"step={step} yielded(checkpoint={y['checkpoint_ms']}ms)")
        # The trainer's final token has no consumer (gen exits after its last
        # step), so skip it — otherwise the POST races the peer's shutdown.
        if not (ROLE == "train" and step == STEPS):
            send_token(step)

    log("DONE all steps")


if __name__ == "__main__":
    main()
