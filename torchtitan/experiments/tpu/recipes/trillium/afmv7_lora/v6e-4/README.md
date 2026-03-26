# Recipe for training AFMv7 model with LoRA on TPU trillium (v6e-4)

This assumes that training happens on v6e-4 CloudTPU VM.

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
Setup instructions are available in [README.md](../../../../README.md)

DISCLAIMER: AFMv7 integration is WIP at the moment, this example demonstrates
functionality. Performance and quality are not optimized yet.

```bash
torchrun \
    --nproc_per_node=4 \
    -m torchtitan.experiments.tpu.train \
    --job.config_file ./torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
    --model.name="afmv7_tpu" \
    --training.local_batch_size=1 \
    --parallelism.data_parallel_replicate_degree=1 \
    --parallelism.data_parallel_shard_degree=4
```