"""AFMTextV7 model for the JAX experiment."""

from .args import AFMTextV7ModelArgs
from .model import (
    Transformer,
    afmv7_3b,
    afmv7_3b_lora,
    afmv7_debug,
    afmv7_debug_lora,
)
from .sharding import sharding_map_original, sharding_map_scan

# Registry mirroring torchtitan.experiments.tpu.afmv7.afmv7_args.
args = {
    "3B": afmv7_3b,
    "3B-lora": afmv7_3b_lora,
    "debugmodel": afmv7_debug,
    "debugmodel-lora": afmv7_debug_lora,
}

__all__ = [
    "AFMTextV7ModelArgs",
    "Transformer",
    "args",
    "sharding_map_original",
    "sharding_map_scan",
]
