"""Model args for AFMTextV7 (JAX/Flax NNX version).

Mirrors torchtitan.experiments.tpu.afmv7.model.args.AFMTextV7ModelArgs so that
the JAX model can be configured with the same arguments as the PyTorch/TAMM
reference implementation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AFMTextV7ModelArgs:
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
    # When use_lora=True, a LoRA adapter is added to every adapted layer and
    # the training loop freezes the base weights.
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dtype: str = "float32"

    # Seq length (set from job_config.training.seq_len at construction time).
    max_seq_len: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def ffn_hidden_dim(self) -> int:
        return round(self.hidden_dim * self.hidden_dim_scale_factor)

    @property
    def num_regular_layers(self) -> int:
        return self.num_layers - self.num_kv_reuse_layers
