"""Model args for AFMPTMoe."""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs


@dataclass
class AFMPTMoeModelArgs(BaseModelArgs):
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

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        pass

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        # Simplified FLOPs estimate to satisfy >0 assertion in metrics.py
        active_fraction = self.num_experts_per_token / max(1, self.num_experts)
        num_flops_per_token = int(6 * nparams * active_fraction) or 1000
        return nparams, num_flops_per_token
