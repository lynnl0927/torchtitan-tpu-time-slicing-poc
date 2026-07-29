"""Persistent TPU time-slicing agent.

Inits the TPU once, then loops on a file-based command channel so a single
long-lived process (and thus a stable set of libtpu control pipes) can be
driven through compute / checkpoint / restore steps by an external
orchestrator.

C/R uses libtpu v2's gVisor tpu_control pipe protocol (requires
LIBTPU_CHECKPOINTING_ENABLED=true): the control thread named
libtpu{RRRRSSSS} encodes request/response pipe FDs, reached through
/proc/self/fd/<N> with 4-byte-BE length-prefixed protobuf messages.

Commands (write one word to CMD, read RESULT):
  init      - materialize inputs + first matmul, record baseline checksum
  compute   - rerun matmul on the SAME inputs, report checksum
  checkpoint- Checkpoint own process (yield device)
  restore   - Restore own process (reacquire device)
  quit
"""
import glob
import os
import re
import select
import struct
import sys
import time

CMD = "/tmp/agent_cmd"
RESULT = "/tmp/agent_result"
TAG = os.environ.get("AGENT_TAG", "agent")

_HAL_THREAD_RE = re.compile(r"^libtpu([0-9a-fA-F]{4})([0-9a-fA-F]{4})$")
_HAL_ACTIONS = {"Checkpoint": 2, "Restore": 3}


def log(m):
    print(f"[{TAG}] {m}", flush=True)


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


def hal(method, timeout=60):
    pipes = own_pipes()
    if not pipes:
        return False, "no-control-pipes"
    t0 = time.time()
    for pipe in pipes:
        try:
            ok, err = _pipe_control(pipe, method, timeout)
        except Exception as e:
            ok, err = False, str(e)
        if not ok:
            return False, f"pipes={pipes} {time.time() - t0:.2f}s {err[:150]}"
    return True, f"pipes={pipes} {time.time() - t0:.2f}s"


def write_result(s):
    with open(RESULT + ".tmp", "w") as f:
        f.write(s)
    os.replace(RESULT + ".tmp", RESULT)


def main():
    import torch
    import torch_tpu  # noqa: F401

    dev = torch.device("tpu")
    state = {}
    log(f"pid={os.getpid()} ready, waiting for commands")
    if os.path.exists(CMD):
        os.remove(CMD)

    while True:
        if not os.path.exists(CMD):
            time.sleep(0.3)
            continue
        with open(CMD) as f:
            cmd = f.read().strip()
        os.remove(CMD)
        log(f"cmd={cmd}")

        try:
            if cmd == "init":
                a = torch.randn(1024, 1024, device=dev)
                b = torch.randn(1024, 1024, device=dev)
                state["a"], state["b"] = a, b
                cs = (a @ b).sum().to("cpu").item()
                state["baseline"] = cs
                write_result(f"OK init checksum={cs:.6f} pipes={own_pipes()}")
            elif cmd == "compute":
                a, b = state["a"], state["b"]
                cs = (a @ b).sum().to("cpu").item()
                match = abs(cs - state["baseline"]) < 1e-2
                write_result(f"OK compute checksum={cs:.6f} matches_baseline={match}")
            elif cmd == "checkpoint":
                ok, info = hal("Checkpoint")
                write_result(f"{'OK' if ok else 'FAIL'} checkpoint {info}")
            elif cmd == "restore":
                ok, info = hal("Restore")
                write_result(f"{'OK' if ok else 'FAIL'} restore {info}")
            elif cmd == "quit":
                write_result("OK quit")
                return
            else:
                write_result(f"ERR unknown cmd {cmd}")
        except Exception as e:
            write_result(f"ERR {cmd}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
