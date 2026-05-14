from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.experiments.tpu.afm_pt_moe.infra.parallelize import parallelize_afm_pt_moe as tpu_parallelize_afm_pt_moe
from torchtitan.experiments.tpu.afm_pt_moe import afm_pt_moe_args
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.experiments.tpu.afm_pt_moe.model.model import AFMPTMoeWrapper


def _afm_pt_moe_model_spec(flavor: str) -> ModelSpec:
  model_config = afm_pt_moe_args[flavor]
  return ModelSpec(
      name="afm_pt_moe_tpu",
      flavor=flavor,
      model=model_config,
      parallelize_fn=tpu_parallelize_afm_pt_moe,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afm_pt_moe_debugmodel() -> TPUTrainerConfig:
  """AFMPTMoe debug model configuration for TPU."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      model_spec=_afm_pt_moe_model_spec("debugmodel"),
      optimizer=OptimizersContainer.Config(lr=8e-4, name="AdamW", eps=1e-8),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          local_batch_size=1,
          seq_len=32,
          steps=10,
          dtype="float32",
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


def afm_pt_moe_3b() -> TPUTrainerConfig:
  """AFMPTMoe 3B model configuration for TPU."""
  config = afm_pt_moe_debugmodel()
  config.model_spec = _afm_pt_moe_model_spec("3b")
  config.optimizer.lr = 3e-4
  config.optimizer.implementation = "foreach"
  config.lr_scheduler.min_lr_factor = 0.1
  config.training.local_batch_size = 1
  config.training.seq_len = 512
  config.training.steps = 5
  config.dataloader.dataset = "random"
  config.training.dtype = "float32"
  config.tpu_config.eager_mode = "DEFER_AND_FUSE"
  config.tpu_config.enable_amp = True
  config.afm_pt_moe.use_chunked_loss = True
  config.checkpoint.interval = 500
  config.activation_checkpoint.mode = "none"
  return config


def afm_pt_moe_24b() -> TPUTrainerConfig:
  """AFMPTMoe 24B model configuration for TPU."""
  config = afm_pt_moe_3b()
  config.model_spec = _afm_pt_moe_model_spec("24b")
  return config


def afm_pt_moe_24b_lora() -> TPUTrainerConfig:
  """AFMPTMoe 24B LoRA model configuration for TPU."""
  config = afm_pt_moe_24b()
  config.model_spec = _afm_pt_moe_model_spec("24b-lora")
  config.lora.use_lora = True
  config.lora.lora_rank = 16
  config.lora.lora_alpha = 16.0
  config.lora.lora_dtype = "float32"
  return config
