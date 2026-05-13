from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.tools.profiler import Profiler
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.experiments.tpu.afmv7.infra.parallelize import parallelize_afmv7 as tpu_parallelize_afmv7
from torchtitan.experiments.tpu.afmv7 import afmv7_args
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.experiments.tpu.afmv7.model.model import AFMTextV7Wrapper

# Model registration spec for afmv7_tpu models
def _afmv7_model_spec(flavor: str) -> ModelSpec:
  model_config = afmv7_args[flavor]
  return ModelSpec(
      name="afmv7_tpu",
      flavor=flavor,
      model=model_config,
      parallelize_fn=tpu_parallelize_afmv7,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afmv7_debugmodel() -> TPUTrainerConfig:
  """AFMTextV7 debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=_afmv7_model_spec("debugmodel"),
      optimizer=OptimizersContainer.Config(lr=8e-4, name="AdamW", eps=1e-8),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=2,
          seq_len=32,
          steps=10,
          dtype="bfloat16",
          mixed_precision_param="bfloat16",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=1),
      parallelism=ParallelismConfig(
          data_parallel_replicate_degree=1,
          data_parallel_shard_degree=-1,
          fsdp_reshard_after_forward="default",
          tensor_parallel_degree=1,
          pipeline_parallel_degree=1,
          context_parallel_degree=1,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=10,
      ),
      activation_checkpoint=ActivationCheckpointConfig(
          mode="full",
      ),
      compile=CompileConfig(
          enable=False,
          backend="tpu",
          components=["model", "loss"],
      ),
      validator=Validator.Config(
          freq=5,
          steps=10,
      ),
  )


def afmv7_debugmodel_lora() -> TPUTrainerConfig:
  """AFMTextV7 debug LoRA model configuration for TPU."""
  config = afmv7_debugmodel()
  config.model_spec = _afmv7_model_spec("debugmodel-lora")
  config.training.seq_len = 128
  config.tpu_config.use_graph_split = True
  config.tpu_config.enable_amp = False
  config.lora.use_lora = True
  config.lora.lora_dtype = "float32"
  config.splash_attention_kernel.use_splash_attention_kernel = True
  config.loss_kernel.use_loss_kernel = True
  return config


def afmv7_medium_lora() -> TPUTrainerConfig:
  """AFMTextV7 medium LoRA model configuration for TPU."""
  config = afmv7_debugmodel()
  config.model_spec = _afmv7_model_spec("medium-lora")
  config.training.seq_len = 128
  config.tpu_config.use_graph_split = True
  config.tpu_config.enable_amp = False
  config.lora.use_lora = True
  config.lora.lora_dtype = "float32"
  config.splash_attention_kernel.use_splash_attention_kernel = True
  config.loss_kernel.use_loss_kernel = True
  config.compile.enable = True
  return config


def afmv7_3b() -> TPUTrainerConfig:
  """AFMTextV7 3B model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=_afmv7_model_spec("3B"),
      optimizer=OptimizersContainer.Config(
          lr=3e-4, name="AdamW", eps=1e-8, implementation="foreach"
      ),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=10,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          local_batch_size=4,
          seq_len=8192,
          steps=20,
          dtype="float32",
          mixed_precision_param="bfloat16",
      ),
      dataloader=HuggingFaceTextDataLoader.Config(
          dataset="c4_test",
      ),
      metrics=MetricsProcessor.Config(log_freq=5),
      profiler=Profiler.Config(
          enable_profiling=False,
          profile_freq=19,
          profiler_active=3,
      ),
      parallelism=ParallelismConfig(
          data_parallel_replicate_degree=1,
          data_parallel_shard_degree=-1,
          fsdp_reshard_after_forward="never",
          tensor_parallel_degree=1,
          pipeline_parallel_degree=1,
          context_parallel_degree=1,
      ),
      tpu_config=TPUTrainerConfig().tpu_config.__class__(
          eager_mode="DEFER_AND_FUSE",
          enable_amp=True,
          use_graph_split=True,
          compile_mode="layer",
          use_simple_fsdp=False,
      ),
      lora=TPUTrainerConfig().lora.__class__(
          force_lora_parameter_ddp=True,
          use_lora=False,
      ),
      splash_attention_kernel=TPUTrainerConfig().splash_attention_kernel.__class__(
          use_splash_attention_kernel=True,
          sa_block_q=2048,
          sa_block_kv=2048,
          sa_block_dkv=2048,
          sa_block_kv_compute=2048,
          sa_block_q_dkv=2048,
          sa_block_kv_dkv=2048,
          sa_block_kv_dkv_compute=2048,
          sa_block_q_dq=2048,
          sa_block_kv_dq=2048,
      ),
      loss_kernel=TPUTrainerConfig().loss_kernel.__class__(
          use_loss_kernel=True,
      ),
      checkpoint=CheckpointManager.Config(
          enable=False,
          interval=500,
          last_save_model_only=True,
          export_dtype="float32",
          async_mode="disabled",
      ),
      activation_checkpoint=ActivationCheckpointConfig(
          mode="full",
      ),
      compile=CompileConfig(
          enable=False,
          backend="tpu",
          components=["model", "loss"],
      ),
      validator=Validator.Config(
          freq=500,
          steps=10,
      ),
  )


def afmv7_3b_lora() -> TPUTrainerConfig:
  """AFMTextV7 3B LoRA model configuration for TPU."""
  config = afmv7_3b()
  config.model_spec = _afmv7_model_spec("3B-lora")
  config.training.dtype = "bfloat16"
  config.lora.use_lora = True
  config.lora.force_lora_parameter_ddp = False
  config.lora.lora_rank = 16
  config.lora.lora_dtype = "float32"
  config.splash_attention_kernel.sa_block_kv_compute = 1024
  config.loss_kernel.loss_b_block_size = 2048
  return config


def afmv7_3b_lora_ddp_compile() -> TPUTrainerConfig:
  """AFMTextV7 3B LoRA DDP Compile model configuration for TPU."""
  config = afmv7_3b_lora()
  config.training.dtype = "float32"
  config.parallelism.data_parallel_replicate_degree = -1
  config.parallelism.data_parallel_shard_degree = 1
  config.lora.force_lora_parameter_ddp = True
  config.loss_kernel.loss_h_block_size = 256
  config.afmv7.enable_manual_ddp = True
  config.compile.enable = True
  return config
