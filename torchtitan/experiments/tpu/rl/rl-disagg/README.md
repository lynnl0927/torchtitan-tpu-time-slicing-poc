# Disaggregated TPU GRPO Reinforcement Learning (`rl-disagg`)

This directory contains a fully disaggregated, strictly synchronous, on-policy Group Relative Policy Optimization (GRPO) reinforcement learning pipeline for [TorchTitan](https://github.com/google-pytorch/torchtitan) on Google Cloud TPU v5e meshes.

By decoupling autoregressive sampling (generation) and gradient backpropagation (training) across separate TPU virtual machines, this architecture eliminates memory contention between inference KV-caches and training optimizer states while maintaining 100% on-policy weight synchronization.

---

## 1. Architecture & Overview

* **Reference Baseline**: Extends the monolithic TorchTitan TPU RL implementation at [`torchtitan/experiments/tpu/rl`](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl).
* **Trainer Mesh (`VM 1`)**: Executes forward evaluation, policy gradient loss computation, and AdamW optimizer steps across an 8-chip FSDP mesh ([server_trainer.py](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/server_trainer.py)).
* **Sampler Mesh (`VM 2`)**: Executes distributed autoregressive sampling across an independent 8-chip FSDP mesh ([server_sampler.py](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/server_sampler.py)).
* **Orchestrator Client**: Coordinates synchronous execution and cross-VM FSDP weight updates via HTTP REST endpoints ([rl_driver.py](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/rl_driver.py)).

> [!NOTE]
> **Why Standalone TPU Spot VMs?**
> During initial testing, GKE node pool creation failed due to on-demand quota limitations even when requesting Spot instances. To bypass this, two standalone TPU v5e spot VMs (`test-v5e-spot`) were provisioned directly via Google Compute Engine (GCE) / `gcloud`.

---

## 2. Connecting to TPU VMs

### Querying VM Hostnames
To retrieve the internal FQDN / guest attributes of your provisioned TPU spot VM:
```bash
gcloud compute tpus tpu-vm get-guest-attributes \
  --zone=us-east1-c \
  --query-path deviceInfo/hostname \
  test-v5e-spot
```
Example Output:
```text
NAMESPACE   KEY       WORKER_ID  VALUE
deviceInfo  hostname  0          t1v-n-cca1e90f-w-0
```

### SSH Access
You can connect to the VMs directly via internal SSH FQDNs or using `gcloud`:

#### Option A: Direct SSH (from Cloudtop / Workstation)
```bash
ssh linglinll_google_com@nic0.t1v-n-cca1e90f-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com
```

#### Option B: `gcloud` SSH with Proxy Command
```bash
gcloud compute tpus tpu-vm ssh test-v5e-spot \
  --zone=us-east1-c \
  -- -o Hostname=nic0.t1v-n-cca1e90f-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com \
  -o ProxyCommand='corp-ssh-helper %h %p'
```

---

## 3. Environment Setup & Dependency Installation

Perform the following setup steps on **both** the Trainer VM and Sampler VM.

### Step 1: Copy Codebase to VMs
From your local workstation, securely copy your clean TorchTitan repository to both TPU VMs:
```bash
# 1. Copy to Trainer VM:
scp -r ~/clean/torchtitan linglinll_google_com@nic0.t1v-n-0f7bbb5c-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com:~

# 2. Copy to Sampler VM:
scp -r ~/clean/torchtitan linglinll_google_com@nic0.t1v-n-cca1e90f-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com:~
```

### Step 2: System Packages & Python 3.12 Virtual Environment
SSH into each VM and initialize a clean Python 3.12 virtual environment:
```bash
# Refresh package lists and install Python 3.12 venv support
sudo apt update
sudo apt install -y python3.12 python3.12-venv

# Create and activate virtual environment
python3.12 -m venv ~/titan_env
source ~/titan_env/bin/activate

# Upgrade pip and verify version
pip install --upgrade pip
python --version  # Must print Python 3.12.x
```

### Step 3: Install JAX, PyTorch, and Tokamax
> [!IMPORTANT]
> **Strict Installation Sequence**
> The package installation order below is critical to prevent dependency conflicts with `tokamax` and `libtpu` (see [Tokamax Issue #240](https://github.com/openxla/tokamax/issues/240)). Do not reorder or combine these `pip install` commands.

```bash
pip install fastapi uvicorn requests

# Install JAX and Tokamax sequence
pip install "jax[tpu]==0.9.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/openxla/tokamax.git@8cba6a6a1e52e9efbb7ff8facb66f18f0bfcbe4c

# Install CPU PyTorch and LibTPU runtime
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install libtpu==0.0.40 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Re-verify HTTP server dependencies
pip install fastapi uvicorn requests
```

### Step 4: Download and Install `torch_tpu`
Because `torch_tpu` requires authenticated access to Google Cloud Artifact Registry, download the wheel on your **local workstation** (authenticated with `@google.com`) and copy it to both VMs:

#### On Local Workstation:
```bash
# Download torch_tpu wheel
pip download --pre --no-deps \
    --python-version 3.12 \
    --only-binary=:all: \
    --platform manylinux_2_31_x86_64 \
    --index-url "https://oauth2accesstoken:$(gcloud auth application-default print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
    torch_tpu

# SCP to Trainer VM and Sampler VM
scp ~/torch_tpu-*-cp312-*.whl linglinll_google_com@nic0.t1v-n-0f7bbb5c-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com:~
scp ~/torch_tpu-*-cp312-*.whl linglinll_google_com@nic0.t1v-n-cca1e90f-w-0.us-east1-c.c.linglinll-gke-dev.internal.gcpnode.com:~
```

#### On Both TPU VMs:
```bash
source ~/titan_env/bin/activate
pip install --no-deps ~/torch_tpu-*-cp312-*.whl
```

### Step 5: Install TorchTitan & Verify Readiness
```bash
source ~/titan_env/bin/activate
cd ~/torchtitan

# Install project dependencies
pip install -r requirements.txt
pip install portpicker absl-py numpy gcsfs frozendict triton fairscale tamm

# Restore typeguard to 2.13.3 for Tokamax compatibility
pip install 'typeguard==2.13.3'

# Verify end-to-end installation readiness
python3 -c "import torch, torch_tpu, fastapi, uvicorn, requests; print('VM is 100 percent ready')"
```

---

## 4. Running the Disaggregated RL Pipeline

To execute a 16-chip disaggregated GRPO training loop on **Qwen 3 (0.6B)** across your two VMs, open three separate terminal sessions:

### Terminal 1 — On Trainer VM (`t1v-n-0f7bbb5c-w-0`)
Start the 8-chip FSDP Trainer server on port `8000`:
```bash
source ~/titan_env/bin/activate
export PYTHONPATH=~/torchtitan:$PYTHONPATH

python3 -m torch.distributed.run --nproc_per_node=8 ~/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/server_trainer.py \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b_glp
```
*(Wait until Uvicorn logs: `INFO: Uvicorn running on http://0.0.0.0:8000`).*

### Terminal 2 — On Sampler VM (`t1v-n-cca1e90f-w-0`)
Start the 8-chip FSDP Sampler server on port `8001`:
```bash
source ~/titan_env/bin/activate
export PYTHONPATH=~/torchtitan:$PYTHONPATH

python3 -m torch.distributed.run --nproc_per_node=8 ~/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/server_sampler.py \
    --module=torchtitan.experiments.tpu.rl \
    --config=grpo_qwen3_0_6b_glp
```
*(Wait until Uvicorn logs: `INFO: Uvicorn running on http://0.0.0.0:8001`).*

### Terminal 3 — On Sampler VM (`t1v-n-cca1e90f-w-0`)
Once both servers are actively listening, launch the orchestrator client. 

> [!TIP]
> **Internal VPC Routing**
> When communicating between virtual machines inside Google Cloud VPCs, always use internal private IPv4 addresses (retrieved via `hostname -I` on the Trainer VM) or short internal VPC hostnames (`t1v-n-0f7bbb5c-w-0`). Do not use external `.gcpnode.com` FQDNs for cross-VM HTTP REST requests.

```bash
source ~/titan_env/bin/activate
export PYTHONPATH=~/torchtitan:$PYTHONPATH

# Replace with the internal private IP of your Trainer VM (e.g., 10.142.0.84)
export TRAINER_IP="10.142.0.84"
export SAMPLER_IP="127.0.0.1"

python3 ~/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/rl_driver.py
```

---

## 5. What to Expect (Execution Lifecycle)

When [rl_driver.py](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/rl_driver.py) executes, it coordinates a 5-step GRPO reinforcement learning loop:

```text
Connecting to Trainer (10.142.0.84:8000) and Sampler (127.0.0.1:8001)

--- Step 0 ---
Generated 4 completions in 64.12s       <-- (Step 0: XLA graph compilation takes ~60s)
Trained step with loss -0.2500 in 61.45s <-- (Step 0: Training XLA graph compilation)
Synced weights in 1.26s                 <-- (Initial checkpoint export & FSDP reload)

--- Step 1 ---
Generated 4 completions in 0.53s        <-- (Compiled execution takes ~0.5s!)
Trained step with loss -0.2500 in 0.52s  <-- (Compiled backward pass takes ~0.5s!)
Synced weights in 0.09s                 <-- (Cross-VM weight sync in < 100ms!)
...
```

1. **Step 0 (XLA Graph Compilation)**: The first generation and training passes will take roughly **60 to 90 seconds** per phase as PyTorch/XLA compiles the distributed SPMD graphs across all 16 TPU chips.
2. **Steps 1 to 4 (Sub-Second Execution)**: Once compiled, autoregressive generation and GRPO gradient updates execute in **~0.5 seconds** per step.
3. **Synchronous On-Policy Merging**: At the conclusion of each step, [rl_driver.py](file:///usr/local/google/home/linglinll/clean/torchtitan/torchtitan/experiments/tpu/rl/rl-disagg/rl_driver.py) triggers `/export_weights` on the Trainer and `/update_weights` on the Sampler. The Sampler downloads the updated state dict over HTTP and reloads it into FSDP in **~50–100 milliseconds**, guaranteeing 100% on-policy policy evolution without weight staleness.
