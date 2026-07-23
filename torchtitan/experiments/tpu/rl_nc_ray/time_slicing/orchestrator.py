"""Minimal TPU time-slicing orchestrator (stdlib only).

Serializes ownership of named TPU "pools" across workloads. A pool is one
physical slice (e.g. gen-pool = the generator node's 8 chips, train-pool =
the trainer node's 8 chips). On yield the orchestrator calls the holder's
/checkpoint (device state snapshot); on the next holder's acquire it calls
that workload's /restore. This is the standalone lock manager from rl-disagg,
reduced to the essentials and dependency-free so it runs on any python3 image.

For disaggregated RL time-slicing there are two pools and four workloads:
job-a-gen + job-b-gen contend on gen-pool, job-a-train + job-b-train contend
on train-pool. The anti-phase interleave (A trains while B generates) emerges
from the per-pool locks; no global scheduling logic is needed.

Endpoints (JSON POST unless noted):
  POST /register {workload_id, pool, url}   - url is the workload's C/R server
  POST /acquire  {workload_id}              - blocks until pool free; restores if checkpointed
  POST /yield    {workload_id}              - checkpoints holder, releases pool
  GET  /lookup?wid=<workload_id>            - peer URL discovery
  GET  /health
  GET  /metrics                             - per-step timing log
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ORCH_PORT", "9000"))
T0 = time.time()

_state_lock = threading.Lock()
_pool_locks = {}                       # pool -> Lock (one per physical slice)
_holders = {}                          # pool -> workload_id holding it
workloads = {}                         # workload_id -> {url, pool, checkpointed, step}
metrics = []


def _pool_lock(pool):
    with _state_lock:
        if pool not in _pool_locks:
            _pool_locks[pool] = threading.Lock()
            _holders[pool] = None
        return _pool_locks[pool]


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
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, {"status": "ok", "holders": dict(_holders),
                             "workloads": list(workloads)})
        elif parsed.path == "/metrics":
            self._send(200, {"metrics": metrics})
        elif parsed.path == "/lookup":
            wid = urllib.parse.parse_qs(parsed.query).get("wid", [""])[0]
            wl = workloads.get(wid)
            if wl:
                self._send(200, {"url": wl["url"]})
            else:
                self._send(404, {"error": f"{wid} not registered"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        req = self._body()
        wid = req.get("workload_id")
        if self.path == "/register":
            with _state_lock:
                workloads[wid] = {"url": req["url"], "pool": req["pool"],
                                  "checkpointed": False, "step": 0}
            print(f"[orch] register {wid} pool={req['pool']} url={req['url']}",
                  flush=True)
            self._send(200, {"status": "ok"})
        elif self.path == "/acquire":
            wl = workloads[wid]
            pool = wl["pool"]
            t_wait = time.perf_counter()
            _pool_lock(pool).acquire()
            wait_ms = (time.perf_counter() - t_wait) * 1000
            _holders[pool] = wid
            restore_ms = 0.0
            if wl["checkpointed"]:
                restore_ms, _ = _call(wl["url"], "restore")
                wl["checkpointed"] = False
            wl["step"] += 1
            rec = {"t": round(time.time() - T0, 1), "type": "acquire",
                   "wid": wid, "pool": pool, "step": wl["step"],
                   "wait_ms": round(wait_ms), "restore_ms": round(restore_ms)}
            metrics.append(rec)
            print(f"[orch] {rec}", flush=True)
            self._send(200, rec)
        elif self.path == "/yield":
            wl = workloads[wid]
            pool = wl["pool"]
            ckpt_ms, _ = _call(wl["url"], "checkpoint")
            wl["checkpointed"] = True
            _holders[pool] = None
            lock = _pool_lock(pool)
            if lock.locked():
                lock.release()
            rec = {"t": round(time.time() - T0, 1), "type": "yield",
                   "wid": wid, "pool": pool, "checkpoint_ms": round(ckpt_ms)}
            metrics.append(rec)
            print(f"[orch] {rec}", flush=True)
            self._send(200, rec)
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"[orch] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
