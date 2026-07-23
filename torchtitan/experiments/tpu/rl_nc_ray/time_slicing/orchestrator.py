"""Minimal TPU time-slicing orchestrator (stdlib only).

Serializes ownership of a single TPU "pool" across N workloads. On yield it
calls the holder's /checkpoint (device state snapshot); on the next holder's
acquire it calls that workload's /restore. This is the standalone lock manager
from rl-disagg, reduced to the essentials and dependency-free so it runs on any
python3 image.

Endpoints (JSON POST unless noted):
  POST /register {workload_id, url}     - url is the workload's own C/R server
  POST /acquire  {workload_id}          - blocks until pool free; restores if checkpointed
  POST /yield    {workload_id}          - checkpoints holder, releases pool
  GET  /health
  GET  /metrics                          - per-step timing log
"""
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ORCH_PORT", "9000"))

_pool_lock = threading.Lock()          # single TPU pool
_holder = None                         # workload_id currently holding the pool
_state_lock = threading.Lock()
workloads = {}                         # workload_id -> {url, checkpointed, step}
metrics = []


def _call(url, method, timeout=900):
    """Call a workload's /checkpoint or /restore; return elapsed ms."""
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{url}/{method}", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    body = json.loads(resp.read().decode())
    return (time.perf_counter() - t0) * 1000, body


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

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "holder": _holder,
                             "workloads": list(workloads)})
        elif self.path == "/metrics":
            self._send(200, {"metrics": metrics})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _holder
        req = self._body()
        wid = req.get("workload_id")
        if self.path == "/register":
            with _state_lock:
                workloads[wid] = {"url": req["url"], "checkpointed": False, "step": 0}
            print(f"[orch] register {wid} url={req['url']}", flush=True)
            self._send(200, {"status": "ok"})
        elif self.path == "/acquire":
            t_wait = time.perf_counter()
            _pool_lock.acquire()
            wait_ms = (time.perf_counter() - t_wait) * 1000
            _holder = wid
            wl = workloads[wid]
            restore_ms = 0.0
            if wl["checkpointed"]:
                restore_ms, _ = _call(wl["url"], "restore")
                wl["checkpointed"] = False
            wl["step"] += 1
            rec = {"type": "acquire", "wid": wid, "step": wl["step"],
                   "wait_ms": round(wait_ms), "restore_ms": round(restore_ms)}
            metrics.append(rec)
            print(f"[orch] {rec}", flush=True)
            self._send(200, rec)
        elif self.path == "/yield":
            wl = workloads[wid]
            ckpt_ms, _ = _call(wl["url"], "checkpoint")
            wl["checkpointed"] = True
            _holder = None
            if _pool_lock.locked():
                _pool_lock.release()
            rec = {"type": "yield", "wid": wid, "checkpoint_ms": round(ckpt_ms)}
            metrics.append(rec)
            print(f"[orch] {rec}", flush=True)
            self._send(200, rec)
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"[orch] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
