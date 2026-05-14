"""Config registry for the torchax-lane AFM PT MoE model."""

from torchtitan.experiments.tpu.afm_pt_moe import afm_pt_moe_args
from torchtitan.experiments.tpu.afm_pt_moe import config_registry as tpu_afm_pt_moe
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.experiments.torchax.torchax_job_config import TorchaxJobConfig
from torchtitan.protocols.model_spec import ModelSpec


def _model_spec(flavor: str) -> ModelSpec:
  """Torchax-lane ``ModelSpec``. ``parallelize_fn`` is ``None`` — the torchax
  trainer performs sharding via the JAX named mesh + the per-flavor sharding
  maps under ``experiments/torchax/afm_pt_moe/sharding.py``."""
  return ModelSpec(
      name="afm_pt_moe",
      flavor=flavor,
      model=afm_pt_moe_args[flavor],
      parallelize_fn=None,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=None,
  )


def afm_pt_moe_debugmodel() -> TorchaxJobConfig:
  """AFM PT MoE debug model on torchax (smoke test)."""
  cfg = TorchaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_debugmodel())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.training.seq_len = 128
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = False
  return cfg


def afm_pt_moe_3b() -> TorchaxJobConfig:
  """AFM PT MoE 3B on torchax — v6e-8 / v6e-32 production recipe."""
  cfg = TorchaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_3b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg


def afm_pt_moe_24b() -> TorchaxJobConfig:
  """AFM PT MoE 24B on torchax — currently OOM-blocked at v6e-32 compile.

  See wiki ``experiments/afm_pt_moe_autoresearch_optimization/torchax/`` for
  the open HBM blocker (126 GiB transient at compile)."""
  cfg = TorchaxJobConfig.derive_from(tpu_afm_pt_moe.afm_pt_moe_24b())
  cfg.model_spec = _model_spec(cfg.model_spec.flavor)
  cfg.tokenizer = AFMTokenizerWrapper.Config()
  cfg.torchax_config.use_torchax = True
  cfg.torchax_config.use_scan = True
  return cfg
