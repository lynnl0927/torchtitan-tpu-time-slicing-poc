# Recipe for training AFMv7 model with LoRA on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying AFMv7 LoRA training job to a 8 chip
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

## Notes as of 5/21/26

- TorchTPU version used is at torch_tpu commit hash `d294bbc9` (from 2026-06-215).
- LoRA adapter dtype is **float32** (`--tpu_config.lora_dtype=float32`, with `--tpu_config.lora_rank=16`). fp32 storage keeps AdamW state precise; matmul compute still runs in bf16 via `mixed_precision_param`, with native fp32 accumulation on the MXU.
- SimpleFSDP+compile fits bs=1; FSDP eager fits bs=4.
- vmem tuning: both recipes below use `--xla_tpu_scoped_vmem_limit_kib=65536` (was 131072 in 4/16/26 baseline).


## SimpleFSDP with torch.compile

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --module=torchtitan.experiments.tpu.afmv7 \
    --config=afmv7_3b_lora \
    --compile.enable \
    --tpu_config.use_simple_fsdp \
    --training.local_batch_size=1 \
    --lora.lora_rank=16 \
    --lora.lora_dtype=float32 \
    --lora.force_lora_parameter_ddp \
    --tpu_config.no-enable_amp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --loss_kernel.loss_h_block_size=256 \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**6/15/26: With this configuration you should observe the following metrics**

- Average TPS/chip: 10,858
- Average MFU: 26.80%


## FSDP eager mode with AMP

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-8" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --module=torchtitan.experiments.tpu.afmv7 \
    --config=afmv7_3b_lora \
    --training.local_batch_size=4 \
    --tpu_config.use_simple_fsdp \
    --lora.lora_rank=16 \
    --lora.lora_dtype=float32 \
    --tpu_config.enable_amp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**6/15/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): **8,037**
- Average MFU: **19.84%**
