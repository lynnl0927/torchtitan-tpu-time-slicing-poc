"""AFM PT MoE model for the JAX experiment.

Mirrors torchtitan.experiments.jax.afmv7's structure: pure-JAX/Flax NNX
implementation (own args.py + model.py + sharding.py + tests/) — NOT a
torchax bridge over the PyTorch/TAMM model.

Why a separate JAX-native implementation:
  - The torch_tpu lane deadlocks at v6e-32 multi-host on AFM PT MoE
    (see wiki/analyses/2026-05-02-afm-pt-moe-v6e32-modeling-perspective-and-real-fix-point.md).
  - The torchax bridge (experiments/torchax/afm_pt_moe) reuses the TAMM
    PyTorch model via torchax — works for v6e-{4,8} but inherits TAMM's
    eager-dispatch quirks and is hard to scan/jit cleanly across tracks.
  - A native JAX/Flax NNX model lets us scan over (tracks × layers), apply
    SPMD sharding directly, and profile with xprof without the torchax
    indirection layer.
"""

from .args import (
    AFMPTMoeModelArgs,
    afm_pt_moe_3b,
    afm_pt_moe_24b,
    afm_pt_moe_debug,
)
from .model import Transformer
from .sharding import (
    sharding_map_original,
    sharding_map_scan,
    sharding_map_scan_lora,
    sharding_map_scan_lora_ddp,
    sharding_map_scan_moe,
)


# Registry mirroring torchtitan.experiments.tpu.afm_pt_moe.afm_pt_moe_args.
# Keys MUST match the PyTorch-side keys exactly so a single train_minimal
# `--model.flavor=...` works for both lanes.  Values are AFMPTMoeModelArgs
# *instances*, not factory functions, to match the convention in
# torchtitan/experiments/jax/afmv7/__init__.py (the trainer does
# ``args = jax_model_mod.args[model_flavor]`` and expects an instance).
args = {
    "3b": afm_pt_moe_3b(),
    "24b": afm_pt_moe_24b(),
    "debugmodel": afm_pt_moe_debug(),
}


__all__ = [
    "AFMPTMoeModelArgs",
    "Transformer",
    "afm_pt_moe_3b",
    "afm_pt_moe_24b",
    "afm_pt_moe_debug",
    "args",
    "sharding_map_original",
    "sharding_map_scan",
    "sharding_map_scan_lora",
    "sharding_map_scan_lora_ddp",
    "sharding_map_scan_moe",
]
