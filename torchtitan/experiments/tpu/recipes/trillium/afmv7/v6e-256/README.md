# Recipe for training AFMv7 model (full fine-tuning, non-LoRA) on 256 TPU trillium cluster (v6e-256)

These are instructions for deploying AFMv7 full fine-tuning (all 3B
parameters trained) to a 256 chip v6e cluster using `xpk`.

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
For `xpk` installation, cluster set up, docker image build, and (optionally) attaching GCS bucket to save profiler traces please consult the main [README.md](../../../../README.md) for multi-host set up instructions.

After setting up your cluster, the following variables should be set:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=YOUR_ZONE
export CLUSTER_NAME=YOUR_CLUSTER_NAME
export REPOSITORY=YOUR_REPOSITORY # Repository containing docker image
```

## Notes as of 4/23/26

- **Different configuration from v6e-8 / v6e-128**: at fsdp=256 `simple_fsdp` **hangs indefinitely** (4 attempts, both eager and compile). Both recipes below use **plain FSDP (fsdp2)** instead — omit the `--tpu_config.use_simple_fsdp` flag. On smaller slices simple_fsdp is the best non-LoRA config (+24% TPS over plain FSDP for compile), but at fsdp=256 it's broken.
- **Do NOT use `train_minimal` entry** — it hangs at fsdp=256. Use `-m torchtitan.experiments.tpu.train`.
- TorchTPU version used is **`torch_tpu==0.1.1.dev20260422092830`** (nightly build from 2026-04-22).
- **Master weights are float32** (hard requirement for full fine-tuning). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- Local batch size 4 is the ceiling.
- vmem tuning: both recipes below use `--xla_tpu_scoped_vmem_limit_kib=65536`.


## FSDP with torch.compile (plain FSDP — replacement for simple_fsdp at this scale)

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-256" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
--command "
    torchrun \
    --nnodes=64 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.train \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b.toml \
    --compile.enable \
    --tpu_config.sa_block_kv_compute=1024 \
    --tpu_config.loss_b_block_size=2048"
```

*Note: `--tpu_config.force_lora_parameter_ddp` is **not** passed on v6e-256
(unlike the smaller-slice recipes). At fsdp=256 that flag triggers a
torch.compile/dynamo `linalg_vecdot` FakeTensor error on the
compile path. The toml default is already `true`, so omitting the
explicit flag on the CLI doesn't change semantics for non-LoRA
runs anyway.*

**4/23/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **10,045**
- Average MFU: **30.70%**
- Total TPS (256 chips): 2,571,520


## FSDP eager mode — **not currently supported on v6e-256**

As of 4/23/26, full FT (non-LoRA) eager on v6e-256 does not have a
working recipe with the current image:

- `--tpu_config.use_simple_fsdp`: **hangs** (same simple_fsdp-at-fsdp=256 issue as the LoRA side).
- **Plain FSDP (fsdp2)** (no simple_fsdp): crashes with a
  `torch._dynamo.exc.Unsupported: Attempted to call function marked
  as skipped` graph break at model forward. The `eager_mode=DEFER_AND_FUSE`
  path invokes dynamo internally, and at fsdp=256 scale it trips on
  `torch._dynamo.decorators.disable`. Not reproduced on v6e-8 /
  v6e-128.

Use the compile recipe above until a fix lands.
