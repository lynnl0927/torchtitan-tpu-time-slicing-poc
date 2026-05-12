"""Model args for AFMPTMoe."""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.protocols.model import BaseModel


@dataclass(kw_only=True, slots=True)
class AFMPTMoeModelArgs(BaseModel.Config):
  """Arguments for AFMPTMoe model configuration."""

  vocab_size: int = 153600
  num_tracks: int = 8
  num_layers_per_track: int = 48
  num_layers_per_track_per_sync_point: int = 4
  hidden_dim: int = 2048
  attention_hidden_dim: int = 512
  dense_feed_forward_hidden_dim: int = 5888
  sparse_feed_forward_hidden_dim: int = 2944
  num_heads: int = 4
  num_kv_heads: Optional[int] = None
  rope_theta: float = 500000.0
  num_experts: int = 40
  num_experts_per_token: int = 2

  # Optional TAMM AFMParallelTrackMoEConfig fields. Default `None` means
  # "omit the kwarg so TAMM's own default applies"; set explicitly on a
  # flavor that needs to pin a specific architectural choice (e.g. mixed
  # local/global attention pattern, RMSNorm variant, sdpa backend).
  attention_layer_pattern: Optional[list[str]] = None
  feed_forward_layer_pattern: Optional[list[str]] = None
  local_attention_window_size: Optional[int] = None
  norm_eps: Optional[float] = None
  scale_qk_norm: Optional[bool] = None
  pre_norm: Optional[str] = None
  pre_residual_norm: Optional[str] = None
  tracks_combine_norm: Optional[str] = None
  tracks_combine_op: Optional[str] = None
  tracks_dispatch_norm: Optional[str] = None
  sdpa_implementation: Optional[str] = None
  experts_router_logits_cap: Optional[float] = None

  # LoRA fine-tuning (disabled by default → full parameter training).
  # When use_lora=True, all base model parameters are frozen and only the
  # LoRA adapter parameters are trained. Mirrors the afmv7 wrapper's LoRA
  # surface; see model.py for which TAMM modules get adapters injected.
  use_lora: bool = False
  lora_rank: int = 16
  lora_alpha: float = 16.0
  lora_dtype: str = "float32"

  def build(self) -> "AFMPTMoeWrapper":
    from torchtitan.experiments.tpu.afm_pt_moe.model.model import AFMPTMoeWrapper

    return AFMPTMoeWrapper(self)

  def update_from_config(
      self, *, trainer_config: Configurable.Config, **kwargs
  ) -> None:
    # Let `tpu_config.*` flags in the toml override the flavor's LoRA
    # defaults — same pattern as afmv7's wrapper. Imported lazily to keep
    # this module free of TPUJobConfig dependency at import time.
    from torchtitan.experiments.tpu.tpu_job_config import TPUJobConfig

    job_config = trainer_config
    if isinstance(job_config, TPUJobConfig):
      self.use_lora = job_config.lora.use_lora
      self.lora_rank = job_config.lora.lora_rank
      self.lora_alpha = job_config.lora.lora_alpha
      self.lora_dtype = job_config.lora.lora_dtype

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    # Simplified FLOPs estimate to satisfy >0 assertion in metrics.py
    active_fraction = self.num_experts_per_token / max(1, self.num_experts)
    num_flops_per_token = int(6 * nparams * active_fraction) or 1000
    return nparams, num_flops_per_token
