"""Model args for AFM PT MoE (JAX/Flax NNX version).

Mirrors torchtitan.experiments.tpu.afm_pt_moe.model.args.AFMPTMoeModelArgs so
the JAX model can be configured with the same arguments as the PyTorch/TAMM
reference. Factory functions at the bottom mirror the
torchtitan.experiments.tpu.afm_pt_moe.afm_pt_moe_args registry — i.e. the
exact 3b / 24b / debugmodel architectures used in the torch_tpu lane.
"""

import dataclasses
from torchtitan.experiments.tpu import afm_pt_moe
from torchtitan.experiments.tpu.afm_pt_moe.model import args as tpu_args


@dataclasses.dataclass
class AFMPTMoeModelArgs(tpu_args.AFMPTMoeModelArgs):
    """Arguments for AFM PT MoE model configuration."""

    # Seq length set from job_config.training.seq_len at construction time.
    max_seq_len: int = 8192

    # Convenience properties --------------------------------------------------
    @property
    def head_dim(self) -> int:
        # AFM PT MoE puts the contraction dim into attention_hidden_dim, not
        # hidden_dim; head_dim = attention_hidden_dim // num_heads.
        return self.attention_hidden_dim // self.num_heads

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads


# -----------------------------------------------------------------------------
# Flavor factories — exact mirror of
# torchtitan.experiments.tpu.afm_pt_moe.afm_pt_moe_args.
# -----------------------------------------------------------------------------

def afm_pt_moe_24b() -> AFMPTMoeModelArgs:
    """~24.7 B total params at num_experts=4, top-2 routing (~15.6 B active)."""
    return AFMPTMoeModelArgs(**dataclasses.asdict(afm_pt_moe.afm_pt_moe_args["24b"]))


def afm_pt_moe_3b() -> AFMPTMoeModelArgs:
    """Same as 24b but num_layers_per_track 48 → 4. ~3.1 B total params,
    comparable to afmv7-3B for apples-to-apples comparisons."""
    return AFMPTMoeModelArgs(**dataclasses.asdict(afm_pt_moe.afm_pt_moe_args["3b"]))


def afm_pt_moe_debug() -> AFMPTMoeModelArgs:
    """Tiny model for unit tests and CPU smoke runs (no TPU required)."""
    return AFMPTMoeModelArgs(**dataclasses.asdict(afm_pt_moe.afm_pt_moe_args["debugmodel"]))
