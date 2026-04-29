"""AFMV7 model with torchax."""

import torch
import torchtitan.experiments.tpu.afmv7
from torchtitan.protocols.train_spec import get_train_spec, register_train_spec
import tamm._adapters_v1.layer_adapters.lora as tamm_lora
import tamm.models.afm_text

from .sharding import sharding_map_original
from .sharding import sharding_map_scan
from .sharding import sharding_map_scan_lora
from .sharding import sharding_map_scan_lora_ddp
from .sharding import sharding_map_scan_moe

try:
  register_train_spec('afmv7', get_train_spec('afmv7_tpu'))
except ValueError:
  pass


# HACK: torchax does not support `aten.normal_` for in-place modifications
# during meta-device tracing (which LoRA uses). We monkeypatch `torch.nn.init.normal_`
# to use `torch.randn_like` and `copy_()`, which XLA perfectly lowers to JAX.
_original_normal_ = torch.nn.init.normal_


def _patched_normal_(tensor, mean=0.0, std=1.0):
  with torch.no_grad():
    random_tensor = torch.randn_like(tensor) * std + mean
    return tensor.copy_(random_tensor)


torch.nn.init.normal_ = _patched_normal_

# HACK: TAMM's LoRA implementation uses `torch.addmm` which acts as a black box
# to TorchAX / JAX sharding propagation under tracing. We monkey-patch it to pure
# batched matrix multiplications using standard `@` decorators.



def _patched_transform_outputs_impl(
    self, x: torch.Tensor, outputs: torch.Tensor
) -> torch.Tensor:
  # Cast fp32 LoRA params down to activation dtype so the matmul chain runs
  # in bf16 with native fp32 accumulation on the MXU. Without these casts,
  # `x @ self.a_transpose` auto-promotes the large activation to fp32 (since
  # a_transpose is fp32), doubling the memory traffic and halving the
  # matmul throughput. The small R×H param downcasts are cheap; accumulation
  # stays fp32 inside the MXU even with bf16 inputs.
  out_dtype = outputs.dtype
  a = self.a_transpose.to(out_dtype) if self.a_transpose.dtype != out_dtype else self.a_transpose
  b = self.b_transpose.to(out_dtype) if self.b_transpose.dtype != out_dtype else self.b_transpose
  return outputs + ((x @ a) * self.scale) @ b


# pylint: disable=protected-access
tamm_lora.LoRA._transform_outputs_impl = _patched_transform_outputs_impl

args = torchtitan.experiments.tpu.afmv7.afmv7_args

from torchtitan.experiments.tpu.afmv7.model.model import AFMTextV7Wrapper as model

__all__ = [
    'sharding_map_original',
    'sharding_map_scan',
    'sharding_map_scan_lora',
    'sharding_map_scan_lora_ddp',
    'sharding_map_scan_moe',
    'args',
    'model',
]
