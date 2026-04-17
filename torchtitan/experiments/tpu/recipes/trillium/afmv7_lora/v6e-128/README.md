# Recipe for training AFMv7 model with LoRA on 128 TPU trillium cluster (v6e-128)

These are instructions for deploying AFMv7 LoRA training job to a 128 chip
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
## DDP w/ torch.compile

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-128" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env TORCH_LOGS="dynamo,recompiles" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=32 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora_ddp_compile.toml \
    --job.dump_folder=/data/torchtitan-$WORKLOAD_NAME \
    --training.steps=20 \
    --training.local_batch_size=1 \
    --tpu_config.enable_manual_ddp \
    --tpu_config.no-enable_amp"
```

**4/16/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): 9525
- Average TFlops: 215.84
- Average MFU: 23.51%

Notes: 

 - The configuration is available at `torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora_ddp_compile.toml`
 - Automatic mixed precision is disabled for this configuration (The configuration enables `--tpu_config.enable_manual_ddp` causing `enable_amp` flag to have no effect. We make this explicit with `--tpu_config.no-enable_amp`)
 - For compile, we drop the local batch size to 1 to prevent OOM.
 - Expect a total compilation time (Dynamo + XLA) of approx. ~5mins

Disable torch compile by adding the `--compile.no-enable` flag to the torchrun command. This achieves ~7.5k TPS (with the default batch size of 4 and torch_tpu eager mode `DEFER_AND_FUSE` exposed via `--tpu_config.eager_mode=DEFER_AND_FUSE`)


## FSDP w/ torch.compile

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-128" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env TORCH_LOGS="dynamo,recompiles" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=32 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora_ddp_compile.toml \
    --job.dump_folder=/data/torchtitan-$WORKLOAD_NAME \
    --training.local_batch_size=3 \
    --tpu_config.no-enable_manual_ddp"
```
**4/16/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): 9102
- Average TFlops: 206.25
- Average MFU: 22.47%

Notes:

 - The configuration is available at `torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora_ddp_compile.toml`. We add `--tpu_config.no-enable_manual_ddp` to disable DDP in favor of FSDP.
 - For compile, we drop the local batch size to 3 to prevent OOM.
 - Expect a total compilation time (Dynamo + XLA) of ~5mins

## FSDP eager mode

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
--workload=$WORKLOAD_NAME \
--cluster=$CLUSTER_NAME \
--zone=$ZONE \
--project=$PROJECT_ID \
--tpu-type="v6e-128" \
--num-slices=1 \
--docker-image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest \
--env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
--env TORCH_LOGS="dynamo,recompiles" \
--env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=131072" \
--command "
    torchrun \
    --nnodes=32 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --job.dump_folder=/data/torchtitan-$WORKLOAD_NAME"
```

**4/16/26: With this configuration you should observe the following metrics**

- Average TPS (excl. 10 warmup steps): 7207
- Average TFlops: 163.32
- Average MFU: 17.79%

Notes:

 - The configuration is available at `torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml`
 - This configuration sets the `torch_tpu` eager execution mode to `DEFER_AND_FUSE` as opposed to the present default `DEFER_NEVER`. This is exposed via the `--tpu_config.eager_mode` flag.