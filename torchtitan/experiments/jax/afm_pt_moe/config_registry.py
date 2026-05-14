"""Config registry for the JAX-lane AFM PT MoE model."""

from torchtitan.experiments.jax import afm_pt_moe as jax_afm_pt_moe
from torchtitan.experiments.jax.jax_job_config import JaxJobConfig
from torchtitan.experiments.tpu.afm_pt_moe import config_registry as tpu_afm_pt_moe
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """JAX-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the JAX trainer
  performs sharding via the JAX named mesh + ``afm_pt_moe.sharding_map_*``."""
  return ModelSpec(
      name="afm_pt_moe",
      flavor=flavor,
      model=jax_afm_pt_moe.args[flavor],
      parallelize_fn=lambda *args, **kwargs: None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afm_pt_moe_debugmodel() -> JaxJobConfig:
  """AFM PT MoE debug model on JAX (smoke test)."""
  cfg = JaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_debugmodel())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  # JAX trainer treats local_batch_size as the global batch (single
  # controller) and shards along the fsdp axis (= num_devices). Default 4 so
  # it divides on v6e-4 / v6e-8 / v6e-16.
  cfg.training.local_batch_size = 4
  cfg.training.seq_len = 128
  cfg.jax_config.use_scan = False
  return cfg


def afm_pt_moe_3b() -> JaxJobConfig:
  """AFM PT MoE 3B on JAX — v6e-16 production recipe."""
  cfg = JaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_3b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.optimizer.implementation = "foreach"
  cfg.training.steps = 20
  cfg.jax_config.use_scan = True
  return cfg


def afm_pt_moe_24b() -> JaxJobConfig:
  """AFM PT MoE 24B on JAX — currently HBM-tight at v6e-32 compile."""
  cfg = JaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_24b())
  assert cfg.model_spec is not None
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.optimizer.implementation = "foreach"
  cfg.training.steps = 20
  cfg.jax_config.use_scan = True
  return cfg
