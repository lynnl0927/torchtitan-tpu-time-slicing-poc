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

def _run_grpcurl(sock, method):
    # Use grpcurl in a subprocess to avoid in-process gRPC symbol clashes
    # between Python's grpcio wheel and libtpu.so's static gRPC.
    import time
    
    cmd = [
        "grpcurl", 
        "-plaintext", 
        "-unix",
        f"{sock}", 
        f"tpu.TpuHalService/{method}"
    ]
    # grpcurl is installed in the Docker image

    t0 = time.time()
    print(f"[TPU Snapshot] -> Starting {method} on {sock}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        print(f"[TPU Snapshot] <- {method} on {sock} succeeded in {elapsed:.2f}s", flush=True)
    else:
        print(f"[TPU Snapshot] <- {method} on {sock} FAILED in {elapsed:.2f}s! stdout: {result.stdout.strip()} stderr: {result.stderr.strip()}", flush=True)
        result.check_returncode()

def checkpoint_tpu():
    print("[TPU Snapshot] Initiating checkpoint_tpu() sequence...", flush=True)
    sockets = get_all_sockets()
    if not sockets:
        print("[TPU Snapshot] No local sockets found to checkpoint.", flush=True)
        return
    import time
    import concurrent.futures
    t0_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sockets)) as executor:
        futures = {executor.submit(_run_grpcurl, sock, "Checkpoint"): sock for sock in sockets}
        for future in concurrent.futures.as_completed(futures):
            sock = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[TPU Snapshot] Exception during checkpoint for {sock}: {e}", flush=True)
                
    print(f"[TPU Snapshot] Checkpoint sequence completed for {len(sockets)} sockets in {time.time() - t0_total:.2f}s", flush=True)

def restore_tpu():
    print("[TPU Snapshot] Initiating restore_tpu() sequence...", flush=True)
    sockets = get_all_sockets()
    if not sockets:
        print("[TPU Snapshot] No local sockets found to restore.", flush=True)
        return
    import time
    import concurrent.futures
    t0_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sockets)) as executor:
        futures = {executor.submit(_run_grpcurl, sock, "Restore"): sock for sock in sockets}
        for future in concurrent.futures.as_completed(futures):
            sock = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[TPU Snapshot] Exception during restore for {sock}: {e}", flush=True)
                
    print(f"[TPU Snapshot] Restore sequence completed for {len(sockets)} sockets in {time.time() - t0_total:.2f}s", flush=True)
