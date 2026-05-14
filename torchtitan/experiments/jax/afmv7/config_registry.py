"""Config registry for the JAX-lane AFMv7 model."""

from torchtitan.experiments.jax import afmv7 as jax_afmv7
from torchtitan.experiments.jax.jax_job_config import JaxJobConfig
from torchtitan.experiments.tpu.afmv7 import config_registry as tpu_afmv7
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """JAX-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the JAX trainer
  performs sharding via the JAX named mesh + ``afmv7.sharding_map_*``."""
  return ModelSpec(
      name="afmv7",
      flavor=flavor,
      model=jax_afmv7.args[flavor],
      parallelize_fn=lambda *args, **kwargs: None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afmv7_debugmodel() -> JaxJobConfig:
  """AFMv7 debug model on JAX (smoke test)."""
  cfg = JaxJobConfig.derive_from(tpu_afmv7.afmv7_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  # JAX trainer treats local_batch_size as the global batch (single
  # controller) and shards along the fsdp axis (= num_devices). Default 4 so
  # it divides on v6e-4 / v6e-8 / v6e-16 without extra flags.
  cfg.training.local_batch_size = 4
  cfg.training.seq_len = 128
  cfg.jax_config.use_scan = False
  return cfg


def afmv7_3b() -> JaxJobConfig:
  """AFMv7 3B full fine-tune on JAX — v6e-16 production recipe."""
  cfg = JaxJobConfig.derive_from(tpu_afmv7.afmv7_3b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.jax_config.use_scan = True
  return cfg


def afmv7_3b_lora() -> JaxJobConfig:
  """AFMv7 3B LoRA on JAX."""
  cfg = JaxJobConfig.derive_from(tpu_afmv7.afmv7_3b_lora())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.jax_config.use_scan = True
  return cfg
