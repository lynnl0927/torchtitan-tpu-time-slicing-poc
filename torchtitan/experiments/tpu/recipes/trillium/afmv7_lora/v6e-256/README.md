# Recipe for training AFMv7 model with LoRA on 256 TPU trillium cluster (v6e-256)

These are instructions for deploying AFMv7 LoRA training job to a 256 chip
v6e cluster using `xpk`.

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

- **Different configuration from v6e-8 / v6e-128**: at fsdp=256 the `DDP with torch.compile` recipe that's best on smaller slices **fails** (`--tpu_config.enable_manual_ddp --compile.enable` crashes in 2+ attempts). The compile recipe below uses **FSDP with plain FSDP (fsdp2)** instead, via the `tpu.train` entry point.
- **Do NOT pass `--tpu_config.use_simple_fsdp`** at this scale — it hangs indefinitely on fsdp=256 (4 attempts, both eager and compile). Use plain FSDP (no flag).
- TorchTPU version used is **`torch_tpu==0.1.1.dev20260422092830`** (nightly build from 2026-04-22).
- LoRA adapter dtype is **float32** (`--tpu_config.lora_dtype=float32`, with `--tpu_config.lora_rank=16`). fp32 storage keeps AdamW state precise; matmul compute still runs in bf16 via `mixed_precision_param`, with native fp32 accumulation on the MXU.
- FSDP compile fits bs=3; FSDP eager fits bs=4.
- vmem tuning: both recipes below use `--xla_tpu_scoped_vmem_limit_kib=65536`.


## FSDP with torch.compile and AMP (plain FSDP — replacement for DDP+compile at this scale)

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
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --compile.enable \
    --training.local_batch_size=3 \
    --tpu_config.lora_rank=16 \
    --tpu_config.lora_dtype=float32 \
    --tpu_config.enable_amp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

*Note: `--tpu_config.force_lora_parameter_ddp` is **not** passed here
(unlike the v6e-8/v6e-128 DDP+compile recipes). On v6e-256 at
fsdp=256, adding that flag to the FSDP+compile path triggers a
torch.compile/dynamo `linalg_vecdot` FakeTensor error. The flag works
fine at smaller scales but breaks at this scale — leaving it off is
the working path.*

**4/23/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **6,458**
- Average MFU: **20.32%**
- Total TPS (256 chips): 1,653,248


## FSDP eager mode with AMP

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
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --training.local_batch_size=4 \
    --tpu_config.lora_rank=16 \
    --tpu_config.lora_dtype=float32 \
    --tpu_config.enable_amp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**4/23/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **~6,186** (steps 5-20 range 6,165-6,208, 3 independent runs)
- Average MFU: **~19.5%** (range 19.40-19.51%)
- Total TPS (256 chips): ~1,583,616
