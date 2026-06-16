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

## Notes as of 5/21/26

- TorchTPU version used is at torch_tpu commit hash `d294bbc9` (from 2026-05-21).
- **Master weights are float32** (hard requirement for full fine-tuning). Compute runs in bf16 via `mixed_precision_param=bfloat16`.
- Local batch size 4 is the ceiling.
- vmem tuning: both recipes below use `--xla_tpu_scoped_vmem_limit_kib=65536`.

## FSDP with torch.compile

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
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --module=torchtitan.experiments.tpu.afmv7 \
    --config=afmv7_3b \
    --compile.enable \
    --tpu-config.use-simple-fsdp \
    --splash_attention_kernel.sa_block_kv_compute=1024 \
    --loss_kernel.loss_b_block_size=2048"
```

*Note: `--tpu_config.force_lora_parameter_ddp` is **not** passed on v6e-256
(unlike the smaller-slice recipes). At fsdp=256 that flag triggers a
torch.compile/dynamo `linalg_vecdot` FakeTensor error on the
compile path. The toml default is already `true`, so omitting the
explicit flag on the CLI doesn't change semantics for non-LoRA
runs anyway.*

**6/15/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **11,472**
- Average MFU: **35.06%**
- Total TPS (256 chips): 2,936,832

## FSDP eager mode

```bash
export WORKLOAD_NAME=YOUR_WORKLOAD_NAME

xpk workload create \
  --workload=$WORKLOAD_NAME \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$PROJECT_ID \
  --tpu-type="v6e-256" \
  --num-slices=1 \
  --docker-image="us-west1-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/torchtitan-container:latest" \
  --env WORKERS_0_HOSTNAME="$WORKLOAD_NAME-slice-job-0-0.$WORKLOAD_NAME" \
  --env LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536" \
  --command "
    torchrun \
    --nnodes=64 \
    --nproc_per_node=4 \
    --rdzv_backend=static \
    --rdzv_endpoint=\$WORKERS_0_HOSTNAME:29501 \
    --node_rank=\$TPU_WORKER_ID \
    -m torchtitan.experiments.tpu.afmv7.train_minimal \
    --module=torchtitan.experiments.tpu.afmv7 \
    --config=afmv7_3b \
    --training.steps=30 \
    --training.local-batch-size=4 \
    --tpu-config.use-simple-fsdp \
    --tpu-config.eager-mode=DEFER_AND_FUSE \
    --splash-attention-kernel.sa-block-kv-compute=1024 \
    --parallelism.data-parallel-replicate-degree=1 \
    --parallelism.data-parallel-shard-degree=-1"
```

**6/15/26: With this configuration you should observe the following metrics**

- Average TPS/chip: **7,747**
- Average MFU: **28.71%**
- Total TPS (256 chips): 1,983,232
