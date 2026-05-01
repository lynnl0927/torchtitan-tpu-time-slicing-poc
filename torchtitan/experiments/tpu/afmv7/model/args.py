"""Model args for AFMTextV7."""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torchtitan.config import JobConfig
import torchtitan.experiments.tpu.tpu_job_config
from torchtitan.protocols.model import BaseModelArgs


TPUJobConfig = torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig


@dataclass
class AFMTextV7ModelArgs(BaseModelArgs):
    """Arguments for AFMTextV7 model configuration.

    Default values match the full AFMTextV7 production model.
    """
    # Model architecture (matching TAMM AFMTextV7.Config fields).
    vocab_size: int = 153600
    hidden_dim: int = 2048
    num_layers: int = 56
    num_kv_reuse_layers: int = 21
    num_heads: int = 16
    num_kv_heads: Optional[int] = 2
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 500000.0

    # LoRA fine-tuning (disabled by default → full parameter training).
    # When use_lora=True, all base model parameters are frozen and only
    # the LoRA adapter parameters are trained.
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dtype: str = "bfloat16"

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
      # AFMTextV7 does not take seq_len as a construction parameter;
      # the model handles variable-length inputs natively.
      if isinstance(job_config, TPUJobConfig):
        self.use_lora = job_config.lora.use_lora
        self.lora_rank = job_config.lora.lora_rank
        self.lora_alpha = job_config.lora.lora_alpha
        self.lora_dtype = job_config.lora.lora_dtype

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """Return (nparams, flops_per_token) for MFU calculation.

        Follows the same formula as get_dense_model_nparams_and_flops:
          flops_per_token = 6 * (N - N_emb) + 6 * num_layers * num_heads * head_dims * seq_len

        where head_dims = 2 * (hidden_dim // num_heads) (qk + v dims combined).
        N_emb is subtracted once because the embedding weight is tied with the
        output projection and counted only once by parameters().
        """
        nparams = sum(p.numel() for p in model.parameters())

        # Embedding and output_transform share the same weight (tied).
        # parameters() deduplicates, so nparams counts it once.
        nparams_embedding = self.vocab_size * self.hidden_dim

        head_dim = self.hidden_dim // self.num_heads
        # head_dims follows the convention from get_dense_model_nparams_and_flops:
        # sum of qk and v head dims (each = head_dim for AFMTextV7).
        head_dims = head_dim * 2

        if self.use_lora:
            # With LoRA, frozen base params skip weight gradient computation.
            # Frozen layers cost 4× (forward + input grad only); only the
            # trainable LoRA adapter params incur the full 6× (weight grad too).
            nparams_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            nparams_frozen = nparams - nparams_trainable
            num_flops_per_token = (
                4 * (nparams_frozen - nparams_embedding)
                + 6 * nparams_trainable
                + 6 * self.num_layers * self.num_heads * head_dims * seq_len
            )
        else:
            num_flops_per_token = (
                6 * (nparams - nparams_embedding)
                + 6 * self.num_layers * self.num_heads * head_dims * seq_len
            )
        return nparams, num_flops_per_token
