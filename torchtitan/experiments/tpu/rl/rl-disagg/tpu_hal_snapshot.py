"""TPU checkpoint/restore for libtpu v2 (gVisor tpu_control pipe protocol).

v2 libtpu removed the TpuHalService gRPC-over-UDS API (/run/tpu_hal_<pid>.sock).
With LIBTPU_CHECKPOINTING_ENABLED=true each TPU process instead spawns a control
thread named "libtpu{RRRRSSSS}" (hex request-write / response-read pipe FDs,
visible in /proc/<pid>/task/*/comm). We send 4-byte-BE length-prefixed
tpu_control.proto messages through /proc/<pid>/fd/<N>.

Public API (unchanged from the UDS version): checkpoint_tpu(), restore_tpu().
Discovery is per-PID instead of per-socket: one control thread per TPU process
(8 ranks per pod -> 8 PIDs). The container's PID namespace only contains this
pod's processes, so no cgroup filtering is needed.

Contract: a checkpointed (state=detached) process must issue NO TPU ops until
restored, or it aborts with "enqueueProgram ... Current state: TearedDown".
The servers' CPU-blocked command loop satisfies this.
"""
import glob
import os
import re
import select
import struct

_THREAD_RE = re.compile(r"^libtpu([0-9a-fA-F]{4})([0-9a-fA-F]{4})$")
_ACTIONS = {"Checkpoint": 2, "Restore": 3}
_STATES = {0: "unspecified", 1: "running", 2: "locked", 3: "detached",
           4: "restoring", 5: "faulted"}


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


def get_all_pipes():
    """Return [(pid, tid, req_write_fd, rsp_read_fd)] for every TPU process
    in this container (identified by its libtpu control thread)."""
    pipes = []
    for comm in glob.glob("/proc/[0-9]*/task/[0-9]*/comm"):
        try:
            name = open(comm).read().strip()
        except OSError:
            continue
        m = _THREAD_RE.match(name)
        if m:
            parts = comm.split("/")
            pipes.append((int(parts[2]), int(parts[4]),
                          int(m.group(1), 16), int(m.group(2), 16)))
    pipes.sort()
    print(f"[TPU Snapshot] Found {len(pipes)} libtpu control pipes: "
          f"{[(p, t) for p, t, _, _ in pipes]}", flush=True)
    return pipes


def _control(pipe, method, timeout=120):
    """Send one tpu_control action and wait for the response."""
    pid, tid, req_fd, rsp_fd = pipe
    body = _varint((1 << 3) | 0) + _varint(_ACTIONS[method])
    body += _varint((2 << 3) | 0) + _varint(int(timeout))
    req = os.open(f"/proc/{pid}/fd/{req_fd}", os.O_WRONLY)
    rsp = os.open(f"/proc/{pid}/fd/{rsp_fd}", os.O_RDONLY)
    try:
        os.write(req, struct.pack(">I", len(body)) + body)
        import time as _t
        deadline = _t.time() + timeout

        def read_exact(n):
            buf = b""
            while len(buf) < n:
                r, _, _ = select.select([rsp], [], [],
                                        max(0.0, deadline - _t.time()))
                if not r:
                    raise TimeoutError(f"pid={pid} no {method} response "
                                       f"within {timeout}s")
                chunk = os.read(rsp, n - len(buf))
                if not chunk:
                    raise IOError(f"pid={pid} response pipe closed")
                buf += chunk
            return buf

        (size,) = struct.unpack(">I", read_exact(4))
        data = read_exact(size)
    finally:
        os.close(req)
        os.close(rsp)

    success, state, err, i = False, 0, "", 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(data, i)
            if field == 1:
                success = bool(val)
            elif field == 2:
                state = val
        elif wire == 2:
            ln, i = _read_varint(data, i)
            if field == 3:
                err = data[i:i + ln].decode(errors="replace")
            i += ln
        else:
            raise ValueError(f"unsupported wire type {wire}")
    if not success:
        raise RuntimeError(f"{method} pid={pid} failed: state="
                           f"{_STATES.get(state, state)} {err}")
    return _STATES.get(state, state)


def _run_control(pipe, method, retries=0, backoff=2.0, timeout=120):
    import time
    pid = pipe[0]
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        print(f"[TPU Snapshot] -> {method} pid={pid} "
              f"(attempt {attempt + 1}/{retries + 1})...", flush=True)
        try:
            state = _control(pipe, method, timeout)
            print(f"[TPU Snapshot] <- {method} pid={pid} OK state={state} "
                  f"in {time.time() - t0:.2f}s", flush=True)
            return
        except Exception as e:
            last = str(e)
            print(f"[TPU Snapshot] <- {method} pid={pid} FAILED in "
                  f"{time.time() - t0:.2f}s: {last}", flush=True)
            if attempt < retries:
                time.sleep(backoff)
    raise RuntimeError(f"{method} pid={pid} failed after "
                       f"{retries + 1} attempt(s): {last}")


def _fan_out(method, retries):
    """Run Checkpoint/Restore on every TPU process in this container.

    Concurrent by default; CR_SEQUENTIAL=1 for one PID at a time. Raises if
    any process ultimately fails so /checkpoint & /restore report honestly.
    Returns the same summary shape as the old UDS implementation ("sockets"
    key kept for caller compatibility).
    """
    import concurrent.futures
    import time

    sequential = os.environ.get("CR_SEQUENTIAL", "").lower() in ("1", "true", "yes")
    mode = "SEQUENTIAL" if sequential else "concurrent"
    print(f"[TPU Snapshot] Initiating {method} sequence "
          f"(retries={retries}, mode={mode})...", flush=True)
    pipes = get_all_pipes()
    if not pipes:
        print(f"[TPU Snapshot] No control pipes found to {method}.", flush=True)
        return {"sockets": 0, "ok": 0, "failed": {}}

    t0_total = time.time()
    failures = {}
    if sequential:
        for idx, pipe in enumerate(pipes):
            print(f"[TPU Snapshot] [seq {idx + 1}/{len(pipes)}] {method} "
                  f"pid={pipe[0]}", flush=True)
            try:
                _run_control(pipe, method, retries)
            except Exception as e:
                failures[pipe[0]] = str(e)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pipes)) as ex:
            futs = {ex.submit(_run_control, p, method, retries): p for p in pipes}
            for fut in concurrent.futures.as_completed(futs):
                pipe = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failures[pipe[0]] = str(e)

    elapsed = time.time() - t0_total
    ok = len(pipes) - len(failures)
    print(f"[TPU Snapshot] {method} sequence ({mode}): {ok}/{len(pipes)} ok "
          f"in {elapsed:.2f}s", flush=True)
    if failures:
        raise RuntimeError(f"{method} failed on {len(failures)}/{len(pipes)} "
                           f"processes after retries: {failures}")
    return {"sockets": len(pipes), "ok": ok, "failed": failures}


def _clear_lockfile():
    for path in ("/tmp/libtpu_lockfile",):
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[TPU Snapshot] Removed stale {path}", flush=True)
        except Exception as e:
            print(f"[TPU Snapshot] Could not remove {path}: {e}", flush=True)


def checkpoint_tpu():
    result = _fan_out("Checkpoint", retries=1)
    _clear_lockfile()
    return result


def restore_tpu():
    return _fan_out("Restore", retries=3)
