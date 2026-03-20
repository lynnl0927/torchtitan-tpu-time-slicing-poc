#!/bin/bash
torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train \
  --job.config_file ./torchtitan/experiments/tpu/afmv7/train_configs/debug_model_lora.toml \
  --model.name="afmv7_tpu" \
  --training.seq_len=8192 \
  --training.local_batch_size=1 \
  --parallelism.data_parallel_replicate_degree=1 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.dtype=bfloat16 \
  --training.mixed_precision_param=bfloat16 \
  --training.mixed_precision_reduce=float32 \
  --tpu_config.use_splash_attention_kernel
