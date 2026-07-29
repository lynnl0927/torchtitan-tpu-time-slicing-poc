# Disaggregated RL Time-Slicing with Two Ray Clusters

Oversubscribe TPU hardware across two **disaggregated** RL jobs, `rl/rl-disagg`
style but at the **pod level** with two independent KubeRay clusters.

Each RL job is disaggregated: a **generator** worker on one TPU node and a
**trainer** worker on another. Within one job those phases alternate — while
the job trains, its generator node idles, and vice versa, capping each node at
~50% utilization. Time-slicing fills those bubbles with a second job running in
**anti-phase**: while job A trains, job B generates on the node A just vacated,
then they swap. A standalone orchestrator serializes ownership per node ("pool"):
on `yield` it snapshots the holder's device state (TPU HAL **Checkpoint**), on
the next `acquire` it restores the incoming job (TPU HAL **Restore**), so each
job resumes bit-identically after the other used its chips.

```
                    gen node (8 chips)          train node (8 chips)
                 ┌──────────┬──────────┐     ┌───────────┬───────────┐
   Cluster A     │  gen-a   │          │     │  train-a  │           │
   (RL job A)    │ tpu:8    │          │     │  tpu:8    │           │
                 ├──────────┼──────────┤     ├───────────┼───────────┤
   Cluster B     │          │  gen-b   │     │           │  train-b  │
   (RL job B)    │          │ no req + │     │           │  no req + │
                 │          │ affinity │     │           │  affinity │
                 └──────────┴──────────┘     └───────────┴───────────┘
                       gen-pool lock              train-pool lock
                        └───────────────┬───────────────┘
                                ts-orchestrator
                     /register /acquire /yield /lookup

   time ─────────►
   gen node:    [A gen 1][B gen 1][A gen 2][B gen 2][A gen 3][B gen 3]
   train node:           [A trn 1][B trn 1][A trn 2][B trn 2][A trn 3]...
                          ▲ anti-phase: A trains WHILE B generates
```

Within each job, the RL data dependency (`gen k → train k → gen k+1`) is
enforced by token passing between the job's two pods. Across jobs, nothing is
scheduled explicitly — the anti-phase interleave **emerges** from the per-pool
locks.

## Why it's built this way

- **Two clusters, not two actors.** A Ray named actor can't span clusters, so
  the lock manager lives *outside* both clusters as a plain Service.
- **Job B bypasses the k8s device plugin.** Only one pod can hold
  `google.com/tpu:8` on a node. Job B's workers request **no** TPU, use
  `podAffinity` to land on Job A's gen/train nodes respectively, and reach the
  chips via `privileged` + `/dev/vfio` + the plugin's vbar socket.
- **HAL Checkpoint/Restore, not vLLM sleep.** vLLM `/sleep` frees *memory* and
  is CUDA-only; it does not release the *device*. TPU HAL C/R snapshots and
  relinquishes the device itself, which is what lets a *different pod* acquire
  the chips.

## Prerequisites (validated configuration)

| Requirement | Value used |
|---|---|
| TPU generation | **v5e only** (`v5litepod-8`, topology `2x4`). v6e untested with the v2 libtpu (the old build failed there at TPU init). |
| GKE cluster | `tpu-cluster` in `us-central1` (project `linglinll-gke-dev`), two `tpu-v5e-8chip` nodes |
| Custom libtpu | v2 checkpointing build at `gs://linglinll-gke-dev-libtpu/_libtpuv2.so` (~709 MB). Stock libtpu has no C/R. Must be enabled with `LIBTPU_CHECKPOINTING_ENABLED=true`. |
| Worker image | `us-central1-docker.pkg.dev/linglinll-gke-dev/test/torchtitan-ray:vllm-0612` (torch_tpu + Ray) |

All commands below assume:

```bash
CTX=gke_linglinll-gke-dev_us-central1_tpu-cluster   # kubectl --context $CTX ...
```

## Files in this folder

| File | Purpose |
|---|---|
| `ray_cluster_a.yaml` | RayCluster **A** — worker groups `gen-a` + `train-a`, each `google.com/tpu:8` (the scheduler spreads them across the two v5e nodes); libtpu staged via initContainer |
| `ray_cluster_b.yaml` | RayCluster **B** — worker groups `gen-b` + `train-b`, no TPU request, `podAffinity` to `gen-a`/`train-a`, privileged VFIO |
| `orchestrator.yaml` | Deployment + Service for the lock manager (`python:3.11-slim`, script via ConfigMap) |
| `orchestrator.py` | The lock manager (stdlib only): per-pool locks, `/register`, `/acquire`, `/yield`, `/lookup`, `/health`, `/metrics` |
| `workload.py` | One gen or train half of an RL job: holds its node's chips, exposes `/checkpoint` & `/restore`, token-passes with its peer, loops acquire→compute→yield |
| `cr_agent.py` | Manual file-channel agent for the raw two-pod smoke test (optional, step 0) |

## One-time cluster setup

```bash
# 1. Enable the GKE-managed KubeRay operator (installs CRDs; operator is managed).
gcloud container clusters update tpu-cluster --region us-central1 \
  --project linglinll-gke-dev --update-addons=RayOperator=ENABLED

# 2. Ray heads need a real CPU node — the default-pool has only ~940m allocatable.
gcloud container node-pools create ray-head-pool --cluster tpu-cluster \
  --region us-central1 --project linglinll-gke-dev \
  --machine-type e2-standard-8 --num-nodes 2

# 3. Verify CRDs exist and both v5e-8chip nodes are Ready.
kubectl --context $CTX get crd | grep ray.io
kubectl --context $CTX get nodes -l cloud.google.com/gke-nodepool=tpu-v5e-8chip
```

## Deploy

Apply **A first** — B's `podAffinity` needs A's workers to exist so it can
colocate onto their nodes. Both v5e nodes must be otherwise empty (A's two
groups each request 8 chips, one lands per node).

```bash
kubectl --context $CTX apply -f ray_cluster_a.yaml
# wait for gen-a and train-a to be Running on the TWO different v5e nodes:
kubectl --context $CTX get pods -l ray.io/cluster=ts-cluster-a -o wide -w

kubectl --context $CTX apply -f ray_cluster_b.yaml
# confirm pairwise colocation: gen-a+gen-b on one node, train-a+train-b on the other:
kubectl --context $CTX get pods -l ray.io/node-type=worker \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName'

# Orchestrator (script mounted from a ConfigMap):
kubectl --context $CTX create configmap ts-orch-script \
  --from-file=orchestrator.py=orchestrator.py
kubectl --context $CTX apply -f orchestrator.yaml
kubectl --context $CTX rollout status deploy/ts-orchestrator
```

The initContainer in each worker downloads the custom libtpu to an emptyDir at
`/shared/_libtpuv2.so`; `TPU_LIBRARY_PATH` is already pointed at it and
`LIBTPU_CHECKPOINTING_ENABLED=true` is set (v2 gates C/R behind it).

## Stage the workload into the workers

C/R needs no external tooling — `workload.py` drives libtpu v2's control
pipes directly.

```bash
for W in $(kubectl --context $CTX get pod -l ray.io/node-type=worker -o name | cut -d/ -f2); do
  kubectl --context $CTX cp workload.py default/$W:/tmp/workload.py -c ray-worker
done
```

## Run the time-sliced disaggregated jobs

Launch four workloads — one per pod. Each job's gen pod kicks off step 1; its
train pod waits for the gen token. The `& exit 0` makes the exec return
immediately while the process detaches — this avoids the intermittent SIGTERM
(143) seen with `nohup ... & disown; sleep`.

```bash
ORCH=http://ts-orchestrator.default.svc.cluster.local:9000
launch() {  # launch <pod> <wid> <pool> <role> <peer-wid>
  kubectl --context $CTX exec "$1" -c ray-worker -- bash -c \
    "cd /tmp; rm -f wl.log; setsid env TPU_LIBRARY_PATH=/shared/_libtpuv2.so \
       LIBTPU_CHECKPOINTING_ENABLED=true \
       ORCH_URL=$ORCH WID=$2 POOL=$3 ROLE=$4 PEER_WID=$5 STEPS=3 WL_PORT=9100 \
       python3 workload.py </dev/null >/tmp/wl.log 2>&1 & exit 0"
}
GA=$(kubectl --context $CTX get pod -l ray.io/group=gen-a   -o name | cut -d/ -f2)
TA=$(kubectl --context $CTX get pod -l ray.io/group=train-a -o name | cut -d/ -f2)
GB=$(kubectl --context $CTX get pod -l ray.io/group=gen-b   -o name | cut -d/ -f2)
TB=$(kubectl --context $CTX get pod -l ray.io/group=train-b -o name | cut -d/ -f2)

launch $GA job-a-gen   gen-pool   gen   job-a-train
launch $TA job-a-train train-pool train job-a-gen
launch $GB job-b-gen   gen-pool   gen   job-b-train
launch $TB job-b-train train-pool train job-b-gen

# Watch the interleave (t = seconds since orchestrator start):
kubectl --context $CTX logs deploy/ts-orchestrator -f

# Per-workload logs (checksum must match baseline every step):
kubectl --context $CTX exec $GA -c ray-worker -- grep -E "step|DONE" /tmp/wl.log
```

### Expected result

The per-pool locks settle the two jobs into anti-phase — **A trains while B
generates**, concurrently on the two nodes — with near-zero waits in steady
state, and every restore is bit-identical:

```
{'t': 42.4, 'acquire', 'job-a-train', 'train-pool', 'step': 2, 'wait_ms': 1, 'restore_ms': 2270}
{'t': 42.4, 'acquire', 'job-b-gen',   'gen-pool',   'step': 2, 'wait_ms': 0, 'restore_ms': 2272}
{'t': 48.8, 'acquire', 'job-a-gen',   'gen-pool',   'step': 3, 'wait_ms': 32, 'restore_ms': 2250}
{'t': 48.8, 'acquire', 'job-b-train', 'train-pool', 'step': 2, 'wait_ms': 0, 'restore_ms': 2278}

[job-a-gen]   step=3 acquired(...) checksum=632.524414   matches_baseline=True
[job-b-train] step=3 acquired(...) checksum=9066.619141  matches_baseline=True
```

Checkpoint ≈ 2.1–2.5 s, restore ≈ 2.3 s per switch on v5e.

## Optional: raw two-pod smoke test (no Ray)

To isolate the HAL C/R primitive before involving Ray, use `cr_agent.py` in two
privileged pods pinned to one v5e node (Job A with `google.com/tpu:8`, Job B with
none). Drive `init → checkpoint → <peer> init → checkpoint → restore → compute`
via a file channel (`/tmp/agent_cmd`, `/tmp/agent_result`); `compute` after a
restore must reproduce the pre-checkpoint checksum.

## How the pieces talk

1. `workload.py` starts a threaded HTTP server exposing `/checkpoint`, `/restore`
   (each drives libtpu v2's gVisor `tpu_control` pipes: the control thread
   named `libtpu{RRRRSSSS}` encodes request/response pipe FDs, reached via
   `/proc/self/fd/<N>` with length-prefixed protobuf Checkpoint/Restore
   messages — see `../../rl/tpu-checkpoint` on the `checkpoint` branch)
   and `/go` (the peer's turn token).
2. It `POST /register`s its own URL + pool with the orchestrator, then loops:
   wait for the peer token (gen skips this on step 1) → `POST /acquire`
   (orchestrator restores it if it was checkpointed) → matmul + verify checksum
   → `POST /yield` (orchestrator calls its `/checkpoint`) → token the peer.
3. The orchestrator holds one lock per pool; only the holder computes on that
   node. Peer URLs are discovered through the orchestrator's `/lookup` registry.

To turn this into a real RL loop, replace `workload.py`'s matmul with the
`rl_nc_ray` vLLM generator (gen role) / trainer (train role), and replace the
token with the actual rollout/weight handoff — the `/acquire`, `/yield`, and
`/checkpoint`+`/restore` calls stay where they are.

## Teardown

```bash
kubectl --context $CTX delete raycluster ts-cluster-a ts-cluster-b
kubectl --context $CTX delete -f orchestrator.yaml
kubectl --context $CTX delete configmap ts-orch-script
gcloud container node-pools delete ray-head-pool --cluster tpu-cluster \
  --region us-central1 --project linglinll-gke-dev   # optional
```

## Known limitations

- **v5e only.** The old libtpu build failed on v6e at TPU init (`connect to
  [::]:8353`); the v2 build has not been re-tested on v6e.
- **Custom libtpu required.** Stock libtpu has no C/R. The v2 build additionally
  requires `LIBTPU_CHECKPOINTING_ENABLED=true` or the control pipes never exist.
- **1 host per slice.** Validated on single-host `v5litepod-8` (`numOfHosts: 1`).
  Multi-host slices need C/R fanned out across all hosts' processes.
- **Proxy workload.** `workload.py` runs a matmul-checksum proxy, not the actual
  trainer + vLLM generator — that integration is the remaining work.
