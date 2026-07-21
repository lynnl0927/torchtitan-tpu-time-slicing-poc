import os
import subprocess

def get_all_sockets():
    import glob
    import os

    def get_cgroup(pid):
        try:
            with open(f"/proc/{pid}/cgroup", "r") as f:
                return f.read().strip()
        except Exception:
            return None

    my_cgroup = get_cgroup("self")

    valid_sockets = []
    for sock in glob.glob("/run/tpu_hal_*.sock"):
        try:
            pid_str = sock.split("tpu_hal_")[1].split(".sock")[0]
            pid = int(pid_str)
            sock_cgroup = get_cgroup(pid)
            if sock_cgroup and my_cgroup and sock_cgroup == my_cgroup:
                valid_sockets.append(sock)
        except Exception as e:
            print(f"[TPU Snapshot] Error checking cgroup for {sock}: {e}", flush=True)

    print(f"[TPU Snapshot] Found {len(valid_sockets)} libtpu sockets in current cgroup: {valid_sockets}", flush=True)
    return valid_sockets

def _run_grpcurl(sock, method, retries=0, backoff=2.0, timeout=45):
    # Use grpcurl in a subprocess to avoid in-process gRPC symbol clashes
    # between Python's grpcio wheel and libtpu.so's static gRPC.
    #
    # Retries: TpuHalService/Restore is observed to be flaky (some or all
    # sockets fail with a transient "Unavailable"/EOF while the runtime is
    # mid-reacquire). Retrying the failed socket often succeeds. We RAISE if
    # all attempts fail so the caller can surface a real error instead of a
    # false "ok" (the previous behavior silently swallowed failures, which led
    # to SIGSEGV/SIGTRAP when compute resumed on an un-restored device).
    import time

    cmd = ["grpcurl", "-plaintext", "-unix", f"{sock}", f"tpu.TpuHalService/{method}"]
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        print(f"[TPU Snapshot] -> {method} on {sock} (attempt {attempt + 1}/{retries + 1})...", flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
            print(f"[TPU Snapshot] <- {method} on {sock} TIMEOUT ({timeout}s) attempt {attempt + 1}", flush=True)
            if attempt < retries:
                time.sleep(backoff)
            continue
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"[TPU Snapshot] <- {method} on {sock} OK in {elapsed:.2f}s (attempt {attempt + 1})", flush=True)
            return
        last = f"rc={result.returncode} stderr={result.stderr.strip()[:300]}"
        print(f"[TPU Snapshot] <- {method} on {sock} FAILED in {elapsed:.2f}s (attempt {attempt + 1}): {last}", flush=True)
        if attempt < retries:
            time.sleep(backoff)
    raise RuntimeError(f"{method} on {sock} failed after {retries + 1} attempt(s): {last}")

def _fan_out(method, retries):
    """Run a TpuHalService method on every live socket.

    By default the method is issued to all sockets CONCURRENTLY (ThreadPoolExecutor).
    Set CR_SEQUENTIAL=1 to issue it to one socket at a time, in socket order — this
    lets us test whether concurrent Restore is what races on the shared device/vfio
    state, and to attribute a failure to a specific socket index.

    Collects per-socket outcomes and RAISES if any socket ultimately fails, so
    /checkpoint and /restore report honest success/failure instead of a blanket 'ok'.
    """
    import time
    import concurrent.futures

    sequential = os.environ.get("CR_SEQUENTIAL", "").lower() in ("1", "true", "yes")
    mode = "SEQUENTIAL" if sequential else "concurrent"
    print(f"[TPU Snapshot] Initiating {method} sequence (retries={retries}, mode={mode})...", flush=True)
    sockets = get_all_sockets()
    if not sockets:
        print(f"[TPU Snapshot] No local sockets found to {method}.", flush=True)
        return {"sockets": 0, "ok": 0, "failed": {}}

    t0_total = time.time()
    failures = {}
    if sequential:
        for idx, sock in enumerate(sockets):
            print(f"[TPU Snapshot] [seq {idx + 1}/{len(sockets)}] {method} on {sock}", flush=True)
            try:
                _run_grpcurl(sock, method, retries)
            except Exception as e:
                failures[sock] = str(e)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sockets)) as executor:
            futures = {executor.submit(_run_grpcurl, sock, method, retries): sock for sock in sockets}
            for future in concurrent.futures.as_completed(futures):
                sock = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failures[sock] = str(e)

    elapsed = time.time() - t0_total
    ok = len(sockets) - len(failures)
    print(f"[TPU Snapshot] {method} sequence ({mode}): {ok}/{len(sockets)} ok in {elapsed:.2f}s", flush=True)
    if failures:
        raise RuntimeError(f"{method} failed on {len(failures)}/{len(sockets)} sockets after retries: {failures}")
    return {"sockets": len(sockets), "ok": ok, "failed": failures}

def _clear_lockfile():
    # Doc caveat: some libtpu builds leave /tmp/libtpu_lockfile after a
    # checkpoint that must be cleared before the next process reacquires the
    # device. No-op if absent (this torch libtpu build does not create it).
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
