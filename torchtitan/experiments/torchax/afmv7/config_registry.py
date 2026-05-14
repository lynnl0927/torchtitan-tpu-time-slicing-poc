"""Config registry for the torchax-lane AFMv7 model."""

from torchtitan.experiments.tpu.afmv7 import afmv7_args
from torchtitan.experiments.tpu.afmv7 import config_registry as tpu_afmv7
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/afmv7/sharding.py``."""
  return ModelSpec(
      name="afmv7",
      flavor=flavor,
      model=afmv7_args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afmv7_debugmodel() -> TorchaxJobConfig:
  """AFMv7 debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_afmv7.afmv7_debugmodel())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.training.seq_len = 128
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg


def afmv7_3b() -> TorchaxJobConfig:
  """AFMv7 3B full fine-tune on torchax — v6e-8 / v6e-32 production recipe."""
  cfg = TorchaxJobConfig.derive_from(tpu_afmv7.afmv7_3b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def afmv7_3b_lora() -> TorchaxJobConfig:
  """AFMv7 3B LoRA on torchax — replicate LoRA adapters, scan over base."""
  cfg = TorchaxJobConfig.derive_from(tpu_afmv7.afmv7_3b_lora())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg
