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

## Notes as of 4/16/26

- TorchTPU whl has to be manually built from source code, follow instructions at: https://github.com/google-pytorch/torch_tpu. TorchTPU hash that was used for this repro: feaf47c
- TorchTitan codebase is actively evolving, hash used for this repro: 23891bb
- By default LoRA adapter type is set to bfloat16. Setting to float32 causes a crash that team is investigating
- For compile, we drop the local batch size to 3 to prevent OOM.
- Automatic mixed precision (AMP) is disabled for 'DDP with compile' configuration (The configuration enables `--tpu_config.enable_manual_ddp` causing `enable_amp` flag to have no effect. We make this explicit with `--tpu_config.no-enable_amp`)


## DDP with torch.compile

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
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --compile.enable \
    --training.local_batch_size=2 \
    --tpu_config.lora_dtype=bfloat16 \
    --tpu_config.enable_manual_ddp \
    --tpu_config.no-enable_amp \
    --parallelism.data_parallel_replicate_degree=-1 \
    --parallelism.data_parallel_shard_degree=1"
```

**4/16/26: With this configuration you should observe the following metrics**

- Average TPS/chip: 9776
- Average MFU: 24.13%


## FSDP with torch.compile and AMP

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
--env TORCH_LOGS="dynamo,recompiles" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --compile.enable \
    --training.local_batch_size=3 \
    --tpu_config.lora_dtype=bfloat16 \
    --tpu_config.enable_amp \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```
**4/16/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): 9268
- Average MFU: 22.88%


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
--env TORCH_LOGS="dynamo,recompiles" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --training.local_batch_size=4 \
    --tpu_config.lora_dtype=bfloat16 \
    --tpu_config.enable_amp \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=-1"
```

**4/16/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): 7207
- Average MFU: 17.79%
