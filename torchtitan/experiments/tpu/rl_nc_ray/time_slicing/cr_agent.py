"""Persistent TPU time-slicing agent.

Inits the TPU once, then loops on a file-based command channel so a single
long-lived process (and thus a stable /run/tpu_hal_<pid>.sock) can be driven
through compute / checkpoint / restore steps by an external orchestrator.

Commands (write one word to CMD, read RESULT):
  init      - materialize inputs + first matmul, record baseline checksum
  compute   - rerun matmul on the SAME inputs, report checksum
  checkpoint- HAL Checkpoint own socket (yield device)
  restore   - HAL Restore own socket (reacquire device)
  quit
"""
import glob
import os
import subprocess
import sys
import time

CMD = "/tmp/agent_cmd"
RESULT = "/tmp/agent_result"
GRPCURL = "/tmp/grpcurl"
TAG = os.environ.get("AGENT_TAG", "agent")


def log(m):
    print(f"[{TAG}] {m}", flush=True)


def own_socket():
    s = f"/run/tpu_hal_{os.getpid()}.sock"
    if os.path.exists(s):
        return s
    socks = sorted(glob.glob("/run/tpu_hal_*.sock"))
    return socks[-1] if socks else None


def hal(method, timeout=60):
    s = own_socket()
    if not s:
        return False, "no-socket"
    cmd = [GRPCURL, "-plaintext", "-unix", s, f"tpu.TpuHalService/{method}"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    dt = time.time() - t0
    return r.returncode == 0, f"{s} {dt:.2f}s rc={r.returncode} {r.stderr.strip()[:150]}"


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
                write_result(f"OK init checksum={cs:.6f} sock={own_socket()}")
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
