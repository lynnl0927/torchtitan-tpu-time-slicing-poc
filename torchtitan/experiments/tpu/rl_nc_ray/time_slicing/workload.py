"""Time-sliced TPU workload — one 'RL job' per Ray cluster.

Holds 8 TPU chips, exposes /checkpoint & /restore (HAL C/R on its own PID) for
the orchestrator to call back, and self-drives an acquire->compute->yield loop.
Two of these (one per RayCluster, colocated on the same node) oversubscribe the
same chips: the orchestrator's pool lock guarantees only one computes at a time,
and checkpoint/restore preserves each job's device state across the other's turn.

Env:
  ORCH_URL   base URL of the orchestrator (e.g. http://ts-orchestrator:9000)
  WID        workload id (e.g. job-a)
  STEPS      number of acquire/compute/yield cycles (default 3)
  WL_PORT    port for this workload's own C/R server (default 9100)
"""
import glob
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GRPCURL = "/tmp/grpcurl"
ORCH_URL = os.environ["ORCH_URL"]
WID = os.environ.get("WID", "job")
STEPS = int(os.environ.get("STEPS", "3"))
WL_PORT = int(os.environ.get("WL_PORT", "9100"))

_dev_lock = threading.Lock()   # serialize all device ops within this process
_state = {}                    # a, b, baseline
_torch = {}                    # dev handle


def log(m):
    print(f"[{WID}] {m}", flush=True)


def own_socket():
    s = f"/run/tpu_hal_{os.getpid()}.sock"
    if os.path.exists(s):
        return s
    socks = sorted(glob.glob("/run/tpu_hal_*.sock"))
    return socks[-1] if socks else None


def hal(method, timeout=120):
    s = own_socket()
    if not s:
        return False, "no-socket", 0.0
    t0 = time.time()
    r = subprocess.run([GRPCURL, "-plaintext", "-unix", s,
                        f"tpu.TpuHalService/{method}"],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stderr.strip()[:120], (time.time() - t0) * 1000


def init_device():
    torch = _torch["torch"]
    dev = _torch["dev"]
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    _state["a"], _state["b"] = a, b
    cs = (a @ b).sum().to("cpu").item()
    _state["baseline"] = cs
    log(f"init baseline={cs:.6f} sock={own_socket()}")
    return cs


def compute():
    torch = _torch["torch"]
    cs = (_state["a"] @ _state["b"]).sum().to("cpu").item()
    match = abs(cs - _state["baseline"]) < 1e-2
    return cs, match


# ---- C/R HTTP server (called BY the orchestrator) ----------------------------
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
        method = self.path.strip("/").capitalize()  # /checkpoint -> Checkpoint
        if method not in ("Checkpoint", "Restore"):
            self._send(404, {"error": "unknown"})
            return
        with _dev_lock:
            ok, err, ms = hal(method)
        log(f"{method} ok={ok} {ms:.0f}ms {err}")
        self._send(200 if ok else 500, {"ok": ok, "ms": round(ms), "err": err})


def orch(path, timeout=900):
    req = urllib.request.Request(
        f"{ORCH_URL}/{path}",
        data=json.dumps({"workload_id": WID, "url": _self_url}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


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
    log(f"C/R server on {_self_url}")

    orch("register")
    log("registered with orchestrator")

    for step in range(STEPS):
        a = orch("acquire")
        with _dev_lock:
            if step == 0:
                init_device()
            cs, match = compute()
        log(f"step={step+1} acquired(wait={a['wait_ms']}ms restore={a['restore_ms']}ms) "
            f"checksum={cs:.6f} matches_baseline={match}")
        # simulate a slice of work
        time.sleep(2)
        y = orch("yield")
        log(f"step={step+1} yielded(checkpoint={y['checkpoint_ms']}ms)")

    log("DONE all steps")


if __name__ == "__main__":
    main()
