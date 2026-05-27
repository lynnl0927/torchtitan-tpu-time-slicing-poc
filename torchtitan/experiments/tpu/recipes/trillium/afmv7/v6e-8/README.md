# Recipe for training AFMv7 model (full fine-tuning, non-LoRA) on 8 TPU trillium cluster (v6e-8)

These are instructions for deploying AFMv7 full fine-tuning (all 3B
parameters trained) to an 8 chip v6e cluster using `xpk`.

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

- TorchTPU version used is at torch_tpu commit hash `9094bfa8` (from 2026-05-21).
- **Master weights are float32** (`training.dtype=float32`, toml default). This is a hard requirement for full fine-tuning. Matmul compute still runs in bf16 via `mixed_precision_param=bfloat16`, with native fp32 accumulation on the MXU.
- `--tpu_config.use_simple_fsdp` is the winner on v6e-8 — **+24% TPS** over plain FSDP for compile, +4% for eager.
- Automatic mixed precision (AMP) is enabled by default in `afmv7_3b` configuration (`enable_amp = true`) and is load-bearing for full fine-tuning.
- vmem tuning: both recipes below use `--xla_tpu_scoped_vmem_limit_kib=65536` (was 131072 in prior baselines).


## FSDP with torch.compile

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
    --config=afmv7_3b \
    --compile.enable \
    --training.local_batch_size=4 \
    --tpu_config.use_simple_fsdp \
    --lora.force_lora_parameter_ddp \
    --splash_attention_kernel.sa_block_kv_compute=1024 \
    --loss_kernel.loss_b_block_size=2048 \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**5/21/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **11,130**
- Average MFU: **34.02%**
- Total TPS (8 chips): 89,040


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
    --config=afmv7_3b \
    --training.local_batch_size=4 \
    --tpu_config.use_simple_fsdp \
    --tpu_config.eager_mode=DEFER_AND_FUSE \
    --splash_attention_kernel.sa_block_kv_compute=1024 \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**5/21/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): **6,870**
- Average MFU: **21.00%**
- Total TPS (8 chips): 54,960
