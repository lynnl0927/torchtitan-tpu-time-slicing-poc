"""Model args for AFMTextV7 (JAX/Flax NNX version).

Mirrors torchtitan.experiments.tpu.afmv7.model.args.AFMTextV7ModelArgs so that
the JAX model can be configured with the same arguments as the PyTorch/TAMM
reference implementation.
"""

import dataclasses
from torchtitan.experiments.tpu.afmv7.model import args as tpu_args


@dataclasses.dataclass
class AFMTextV7ModelArgs(tpu_args.AFMTextV7ModelArgs):
    """Arguments for AFMTextV7 model configuration."""

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
