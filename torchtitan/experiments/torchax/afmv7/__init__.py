"""AFMV7 model with torchax."""

import torchtitan.experiments.tpu.afmv7
from torchtitan.protocols.train_spec import get_train_spec, register_train_spec
import tamm.models.afm_text

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_lora
from .sharding import sharding_map_scan_moe

try:
  register_train_spec('afmv7', get_train_spec('afmv7_tpu'))
except ValueError:
  pass

import torch
# HACK: torch_xla2 does not support `aten.normal_` for in-place modifications
# during meta-device tracing (which LoRA uses). We monkeypatch `torch.nn.init.normal_`
# to use `torch.randn_like` and `copy_()`, which XLA perfectly lowers to JAX.
_original_normal_ = torch.nn.init.normal_


def _patched_normal_(tensor, mean=0.0, std=1.0):
  with torch.no_grad():
    random_tensor = torch.randn_like(tensor) * std + mean
    return tensor.copy_(random_tensor)


torch.nn.init.normal_ = _patched_normal_

args = torchtitan.experiments.tpu.afmv7.afmv7_args

from torchtitan.experiments.tpu.afmv7.model.model import AFMTextV7Wrapper as model

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_lora',
    'sharding_map_scan_moe',
    'args',
    'model',
]
