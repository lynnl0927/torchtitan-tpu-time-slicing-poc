# TPU Time-Slicing with Two Ray Clusters

Reproduce `rl/rl-disagg`-style TPU **oversubscription** (two RL jobs alternating on
the *same* physical chips) at the **pod level**, using two independent KubeRay
clusters instead of one JAX process pair.

Two RL jobs are each a separate `RayCluster`. Their TPU worker pods are scheduled
onto the **same** TPU node and take turns owning the 8 chips. A standalone
orchestrator serializes ownership: on `yield` it snapshots the holder's device
state (TPU HAL **Checkpoint**), on the next `acquire` it restores the incoming
job (TPU HAL **Restore**). Each job resumes bit-identically, as if it had never
been evicted.

```
          ┌──────────────┐   acquire/yield    ┌──────────────┐
          │  RayCluster  │◄──────────────────►│  RayCluster  │
          │      A       │                    │      B       │
          │  (Job A)     │   both workers on  │  (Job B)     │
          └──────┬───────┘   the SAME node    └──────┬───────┘
                 │                                    │
   worker-a  google.com/tpu:8            worker-b  NO tpu request
   (device plugin)                       (privileged VFIO + podAffinity)
                 └───────────── 8 v5e chips ─────────┘
                                   ▲
                       pool lock + checkpoint/restore
                    ┌──────────────┴──────────────┐
                    │      ts-orchestrator          │
                    │  /register /acquire /yield    │
                    └───────────────────────────────┘
```

## Why it's built this way

- **Two clusters, not two actors.** A Ray named actor can't span clusters, so the
  lock manager lives *outside* both clusters as a plain Service.
- **Job B bypasses the k8s device plugin.** Only one pod can hold `google.com/tpu:8`
  on a node. Job B requests **no** TPU, uses `podAffinity` to land on Job A's node,
  and reaches the chips via `privileged` + `/dev/vfio` + the plugin's vbar socket.
- **HAL Checkpoint/Restore, not vLLM sleep.** vLLM `/sleep` frees *memory* and is
  CUDA-only; it does not release the *device*. TPU HAL C/R snapshots/relinquishes
  the device itself, which is what lets a *different pod* acquire the chips.

## Prerequisites (validated configuration)

| Requirement | Value used |
|---|---|
| TPU generation | **v5e only** (`v5litepod-8`, topology `2x4`). v6e fails — the custom libtpu is v5e-era. |
| GKE cluster | `tpu-cluster` in `us-central1` (project `linglinll-gke-dev`), two `tpu-v5e-8chip` nodes |
| Custom libtpu | UDS-enabled build at `gs://linglinll-gke-dev-libtpu/_libtpu.so` (~667 MB). Stock libtpu lacks the HAL Unix socket. |
| Worker image | `us-central1-docker.pkg.dev/linglinll-gke-dev/test/torchtitan-ray:vllm-0612` (torch_tpu + Ray) |
| grpcurl | v1.9.1 linux-amd64, staged into worker pods (drives the HAL gRPC) |

All commands below assume:

```bash
CTX=gke_linglinll-gke-dev_us-central1_tpu-cluster   # kubectl --context $CTX ...
```

## Files in this folder

| File | Purpose |
|---|---|
| `ray_cluster_a.yaml` | RayCluster **A** — Job A pattern (`google.com/tpu:8`), stages libtpu via initContainer |
| `ray_cluster_b.yaml` | RayCluster **B** — Job B pattern (no TPU request, podAffinity to A, privileged VFIO) |
| `orchestrator.yaml` | Deployment + Service for the lock manager (`python:3.11-slim`, script via ConfigMap) |
| `orchestrator.py` | The lock manager (stdlib only): `/register`, `/acquire`, `/yield`, `/health`, `/metrics` |
| `workload.py` | Self-driving "RL job": holds the chips, exposes `/checkpoint` & `/restore`, loops acquire→compute→yield |
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

Apply **A first** — B's `podAffinity` needs A's worker pod to exist so it can
colocate onto the same node.

```bash
kubectl --context $CTX apply -f ray_cluster_a.yaml
# wait for worker-a to be Running on a tpu-v5e-8chip node:
kubectl --context $CTX get pods -l ray.io/cluster=ts-cluster-a -o wide -w

kubectl --context $CTX apply -f ray_cluster_b.yaml
# confirm both TPU workers are on the SAME node:
kubectl --context $CTX get pods -l ray.io/node-type=worker \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName'

# Orchestrator (script mounted from a ConfigMap):
kubectl --context $CTX create configmap ts-orch-script \
  --from-file=orchestrator.py=orchestrator.py
kubectl --context $CTX apply -f orchestrator.yaml
kubectl --context $CTX rollout status deploy/ts-orchestrator
```

The initContainer in each worker downloads the custom libtpu to an emptyDir at
`/shared/_libtpu.so`; `TPU_LIBRARY_PATH` is already pointed at it.

## Stage the C/R tooling into the workers

```bash
WA=$(kubectl --context $CTX get pod -l ray.io/cluster=ts-cluster-a,ray.io/node-type=worker -o name | cut -d/ -f2)
WB=$(kubectl --context $CTX get pod -l ray.io/cluster=ts-cluster-b,ray.io/node-type=worker -o name | cut -d/ -f2)

# grpcurl (download v1.9.1 linux-amd64 locally first, then copy in):
for W in $WA $WB; do
  kubectl --context $CTX cp grpcurl    default/$W:/tmp/grpcurl    -c ray-worker
  kubectl --context $CTX cp workload.py default/$W:/tmp/workload.py -c ray-worker
  kubectl --context $CTX exec $W -c ray-worker -- chmod +x /tmp/grpcurl
done
```

## Run the automated time-slice

Launch one self-driving workload per cluster. The `& exit 0` makes the exec return
immediately while the process detaches — this avoids the intermittent SIGTERM (143)
seen with `nohup ... & disown; sleep`.

```bash
ORCH=http://ts-orchestrator.default.svc.cluster.local:9000
launch() {  # launch <pod> <workload-id>
  kubectl --context $CTX exec "$1" -c ray-worker -- bash -c \
    "cd /tmp; rm -f wl.log; setsid env TPU_LIBRARY_PATH=/shared/_libtpu.so \
       ORCH_URL=$ORCH WID=$2 STEPS=3 WL_PORT=9100 \
       python3 workload.py </dev/null >/tmp/wl.log 2>&1 & exit 0"
}
launch $WA job-a
launch $WB job-b

# Watch the interleave from the orchestrator's metrics:
kubectl --context $CTX exec $WA -c ray-worker -- python3 -c \
  'import urllib.request,json; \
   [print(m) for m in json.loads(urllib.request.urlopen("http://ts-orchestrator.default.svc.cluster.local:9000/metrics").read())["metrics"]]'

# Per-job logs (checksum must match baseline every step):
kubectl --context $CTX exec $WA -c ray-worker -- grep -E "init|step|DONE" /tmp/wl.log
kubectl --context $CTX exec $WB -c ray-worker -- grep -E "init|step|DONE" /tmp/wl.log
```

### Expected result

The orchestrator serializes the two jobs into strict alternation, and each job's
checksum stays **bit-identical to its own baseline** even though the other job
fully used the chips in between:

```
{'type': 'acquire', 'wid': 'job-a', 'step': 1, 'wait_ms': 0,    'restore_ms': 0}
{'type': 'yield',   'wid': 'job-a', 'checkpoint_ms': 2427}
{'type': 'acquire', 'wid': 'job-b', 'step': 1, 'wait_ms': 5116, 'restore_ms': 0}
{'type': 'yield',   'wid': 'job-b', 'checkpoint_ms': 2463}
{'type': 'acquire', 'wid': 'job-a', 'step': 2, 'wait_ms': 7505, 'restore_ms': 2270}
...
[job-a] step=2 acquired(...) checksum=38246.769531 matches_baseline=True
[job-b] step=2 acquired(...) checksum=38608.851562 matches_baseline=True
```

Checkpoint ≈ 2.1–2.5 s, restore ≈ 2.3 s per switch on v5e.

## Optional: raw two-pod smoke test (no Ray)

To isolate the HAL C/R primitive before involving Ray, use `cr_agent.py` in two
privileged pods pinned to one v5e node (Job A with `google.com/tpu:8`, Job B with
none). Drive `init → checkpoint → <peer> init → checkpoint → restore → compute`
via a file channel (`/tmp/agent_cmd`, `/tmp/agent_result`); `compute` after a
restore must reproduce the pre-checkpoint checksum.

## How the pieces talk

1. `workload.py` starts a threaded HTTP server exposing `/checkpoint` & `/restore`
   (each runs `grpcurl -unix /run/tpu_hal_<pid>.sock tpu.TpuHalService/{Checkpoint,Restore}`).
2. It `POST /register`s its own URL with the orchestrator, then loops:
   `POST /acquire` (orchestrator restores it if it was checkpointed) → matmul +
   verify checksum → `POST /yield` (orchestrator calls its `/checkpoint`).
3. The orchestrator holds a single `pool` lock; only the holder computes. On yield
   it snapshots + releases; the waiting job's `acquire` unblocks and restores.

To turn this into a real RL loop, replace `workload.py`'s matmul with the
`rl_nc_ray` trainer / vLLM generator and keep the `/acquire`, `/yield`, and
`/checkpoint`+`/restore` calls where they are.

## Teardown

```bash
kubectl --context $CTX delete raycluster ts-cluster-a ts-cluster-b
kubectl --context $CTX delete -f orchestrator.yaml
kubectl --context $CTX delete configmap ts-orch-script
gcloud container node-pools delete ray-head-pool --cluster tpu-cluster \
  --region us-central1 --project linglinll-gke-dev   # optional
```

## Known limitations

- **v5e only.** v6e fails at TPU init with the current custom libtpu (`connect to
  [::]:8353`). Needs a v6e-compatible UDS-enabled libtpu build.
- **Custom libtpu required.** Stock libtpu has no HAL Unix socket, so no C/R.
- **1 host per slice.** Validated on single-host `v5litepod-8` (`numOfHosts: 1`).
  Multi-host slices need C/R fanned out across all hosts' sockets.
- **Proxy workload.** `workload.py` runs a matmul-checksum proxy, not the actual
  trainer + vLLM generator — that integration is the remaining work.
