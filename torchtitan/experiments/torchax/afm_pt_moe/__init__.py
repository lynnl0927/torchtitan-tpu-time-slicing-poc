"""AFM PT MoE model with torchax."""

import torch
import torchtitan.experiments.tpu.afm_pt_moe
from torchtitan.protocols.train_spec import get_train_spec, register_train_spec
import tamm._adapters_v1.layer_adapters.lora as tamm_lora

from .sharding import (
    sharding_map_original,
    sharding_map_scan,
    sharding_map_scan_lora,
    sharding_map_scan_lora_ddp,
    sharding_map_scan_moe,
)

try:
  register_train_spec('afm_pt_moe', get_train_spec('afm_pt_moe_tpu'))
except ValueError:
  pass


# Same monkeypatch as afmv7: torchax does not support `aten.normal_` for
# in-place modifications during meta-device tracing.
_original_normal_ = torch.nn.init.normal_


def _patched_normal_(tensor, mean=0.0, std=1.0):
  with torch.no_grad():
    random_tensor = torch.randn_like(tensor) * std + mean
    return tensor.copy_(random_tensor)


torch.nn.init.normal_ = _patched_normal_


# Same patch as afmv7: TAMM's LoRA `_transform_outputs_impl` uses `torch.addmm`
# which is opaque to torchax sharding propagation. Replace with pure matmul
# chain so JAX can propagate sharding through it.
def _patched_transform_outputs_impl(self, x, outputs):
  out_dtype = outputs.dtype
  a = self.a_transpose.to(out_dtype) if self.a_transpose.dtype != out_dtype else self.a_transpose
  b = self.b_transpose.to(out_dtype) if self.b_transpose.dtype != out_dtype else self.b_transpose
  return outputs + ((x @ a) * self.scale) @ b


tamm_lora.LoRA._transform_outputs_impl = _patched_transform_outputs_impl

args = torchtitan.experiments.tpu.afm_pt_moe.afm_pt_moe_args

from torchtitan.experiments.tpu.afm_pt_moe.model.model import AFMPTMoeWrapper as model

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_lora',
    'sharding_map_scan_lora_ddp',
    'sharding_map_scan_moe',
    'args',
    'model',
]
