import os
import subprocess

def get_all_sockets():
    import glob
    sockets = glob.glob("/run/tpu_hal_*.sock")
    return sockets

def _run_grpcurl(sock, method):
    # Use grpcurl in a subprocess to avoid in-process gRPC symbol clashes
    # between Python's grpcio wheel and libtpu.so's static gRPC.
    cmd = [
        "grpcurl", 
        "-plaintext", 
        f"unix://{sock}", 
        f"tpu.TpuHalService/{method}"
    ]
    try:
        # Check if grpcurl is available
        subprocess.run(["which", "grpcurl"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Download grpcurl to /tmp if missing
        grpcurl_path = "/tmp/grpcurl"
        if not os.path.exists(grpcurl_path):
            print("[TPU Snapshot] Downloading grpcurl...", flush=True)
            subprocess.run(
                "curl -sL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz | tar -xz -C /tmp grpcurl",
                shell=True, check=True
            )
        cmd[0] = grpcurl_path

    print(f"[TPU Snapshot] Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

def checkpoint_tpu():
    sockets = get_all_sockets()
    if not sockets:
        print("[TPU Snapshot] No local sockets found to checkpoint.", flush=True)
        return
    for sock in sockets:
        print(f"[TPU Snapshot] Checkpointing {sock}...", flush=True)
        try:
            _run_grpcurl(sock, "Checkpoint")
            print(f"[TPU Snapshot] Checkpoint complete for {sock}", flush=True)
        except Exception as e:
            print(f"[TPU Snapshot] Failed to checkpoint {sock}: {e}", flush=True)

def restore_tpu():
    sockets = get_all_sockets()
    if not sockets:
        print("[TPU Snapshot] No local sockets found to restore.", flush=True)
        return
    for sock in sockets:
        print(f"[TPU Snapshot] Restoring {sock}...", flush=True)
        try:
            _run_grpcurl(sock, "Restore")
            print(f"[TPU Snapshot] Restore complete for {sock}", flush=True)
        except Exception as e:
            print(f"[TPU Snapshot] Failed to restore {sock}: {e}", flush=True)
