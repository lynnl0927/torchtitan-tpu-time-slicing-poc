"""Parallelization utilities for AFMTextV7 on TPU."""

import torch
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffload, MixedPrecisionPolicy, fully_shard
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.models.llama3.infra.parallelize import disable_fsdp_gradient_division
from torchtitan.tools.logging import logger


def parallelize_afmv7(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    job_config: (
        torchtitan.config.JobConfig
        | torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
    ),
) -> nn.Module:
  """Apply parallelism and activation checkpointing to AFMTextV7.

  Supports:
  - Activation checkpointing (AC) via TAMM's native checkpoint_activations()
  - FSDP2 data parallelism (shard across fsdp mesh)
  - DDP-style replication (dp_replicate mesh, handled by FSDP2 replicate dim)
  - LoRA fine-tuning: auto-detected from model structure; non-adapter params
  frozen

  Tensor Parallelism is not supported for AFMTextV7 since the TAMM model's
  internal layer structure is not wired for DTensor-based TP.

  Args:
    model: The AFMTextV7 model to parallelize.
    parallel_dims: Parallelism dimensions.
    job_config: Job configuration.

  Returns:
    The parallelized model.

  Raises:
    RuntimeError: If DDP has not supported > 1D parallelism, or if Tensor
      Parallelism is enabled.
  """
  if parallel_dims.tp_enabled:
    raise RuntimeError("Tensor Parallelism is not supported for AFMTextV7 yet")

  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_splash_attention_kernel
  ):
    workarounds.use_splash_attention_patch(model)
  if (
      isinstance(job_config, tpu_job_config.TPUJobConfig)
      and job_config.tpu_config.use_loss_kernel
  ):
    workarounds.use_output_projection_patch(model)

  # Auto-detect LoRA: freeze base model params and unfreeze adapter params.
  _maybe_freeze_for_lora(model)

  model_compile_enabled = (
      job_config.compile.enable and "model" in job_config.compile.components
  )

  # Apply activation checkpointing via TAMM's native API.
  if job_config.activation_checkpoint.mode != "none":
    apply_ac(model)

  # Apply torch.compile to each TransformerLayer in both segments.
  # Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.
  if model_compile_enabled:
    apply_compile(model, job_config.compile)

  if parallel_dims.fsdp_enabled:
    names = (
        ["dp_replicate", "fsdp"]
        if parallel_dims.dp_replicate_enabled
        else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(names)
    apply_fsdp(
        model,
        dp_mesh=dp_mesh,
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_reduce
        ],
        cpu_offload=job_config.training.enable_cpu_offload,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=parallel_dims.pp_enabled,
    )

    if parallel_dims.dp_replicate_enabled:
      logger.info("Applied HSDP to the model")
    else:
      logger.info("Applied FSDP to the model")

    if job_config.training.enable_cpu_offload:
      logger.info("Applied CPU Offloading to the model")

  elif parallel_dims.dp_replicate_enabled:
    dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
    # TODO b/494360665: remove this once torch.autocast is suppoted
    # if parallel_dims.world_size != dp_replicate_mesh.size():
    #   raise RuntimeError("DDP has not supported > 1D parallelism")
    # apply_ddp(
    #     model,
    #     dp_replicate_mesh,
    #     enable_compile=model_compile_enabled,
    # )
    apply_fsdp(
        model,
        dp_mesh=dp_replicate_mesh,
        param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            job_config.training.mixed_precision_reduce
        ],
        cpu_offload=job_config.training.enable_cpu_offload,
        reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=parallel_dims.pp_enabled,
    )

  return model


def _maybe_freeze_for_lora(model: nn.Module) -> None:
  """If the model contains TAMM AdaptedLayer modules (LoRA), freeze all base

  model parameters and unfreeze only the adapter parameters.

  This must run before FSDP wrapping so that FSDP2 can propagate
  requires_grad=False to the sharded DTensors and skip gradient sync for
  frozen parameters.
  """
  try:
    from tamm.adapters import AdaptedLayer
  except ImportError:
    return

  adapted_layers = [
      m for m in model.model.modules() if isinstance(m, AdaptedLayer)
  ]
  if not adapted_layers:
    return

  # Freeze everything first.
  frozen = 0
  for p in model.model.parameters():
    if p.requires_grad:
      p.requires_grad_(False)
      frozen += p.numel()

  # Unfreeze only adapter parameters.
  trainable = 0
  for m in adapted_layers:
    for p in m.adapters.parameters():
      p.requires_grad_(True)
      trainable += p.numel()

  logger.info(
      f"LoRA param freeze: {frozen:,} base params frozen, "
      f"{trainable:,} adapter params trainable "
      f"({100 * trainable / (frozen + trainable):.2f}%)"
  )


def apply_ac(model: nn.Module) -> None:
  """Apply activation checkpointing via TAMM's native API."""
  logger.info("Applying activation checkpointing to AFMTextV7 segments.")
  for segment in model.model.layers.children():
    for layer in segment.children():
      layer.checkpoint_activations(use_reentrant=False)


def apply_compile(
    model: nn.Module, compile_config: "torchtitan.config.job_config.Compile"
) -> None:
  """Apply torch.compile to each TransformerLayer in both segments.

  Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.

  Args:
    model: The AFMTextV7 model.
    compile_config: Compile configuration.
  """
  logger.info("Applying torch.compile to AFMTextV7 TransformerLayers.")
  inner = model.model
  for seg in (inner.layers.segment_0, inner.layers.segment_1):
    for layer_id, layer in seg.named_children():
      compiled = torch.compile(
          layer, backend=compile_config.backend, fullgraph=True
      )
      seg.register_module(layer_id, compiled)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    pp_enabled: bool,
) -> None:
  """Wrap AFMTextV7 with FSDP2 (fully_shard).

  Shards at the individual TransformerLayer granularity inside each segment.
  Top-level non-layer children (embedding, norm, positional_encoding, etc.)
  are NOT sharded individually to avoid issues with TAMM's tied weights
  between the embedding and output_transform layers; those are handled by the
  top-level fully_shard call on the inner model.

  Args:
    model: Model to wrap.
    dp_mesh: Device mesh.
    param_dtype: Parameter dtype.
    reduce_dtype: Reduce dtype.
    cpu_offload: Whether to enable CPU offloading.
    reshard_after_forward_policy: Reshard after forward policy.
    pp_enabled: Whether pipeline parallelism is enabled.

  Raises:
    ValueError: If reshard_after_forward_policy is invalid.
  """
  mp_policy = MixedPrecisionPolicy(
      param_dtype=param_dtype, reduce_dtype=reduce_dtype
  )
  fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
  if cpu_offload:
    fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)

  match reshard_after_forward_policy:
    case "always":
      reshard = True
    case "never":
      reshard = False
    case "default":
      reshard = not pp_enabled
    case _:
      raise ValueError(
          "Invalid reshard_after_forward_policy:"
          f" {reshard_after_forward_policy}"
      )

  inner = model.model  # the TAMM AFMTextV7 module

  # Shard each individual TransformerLayer inside both segments for
  # fine-grained memory and compute overlap.
  for layer in inner.layers.segment_0.children():
    fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
  for layer in inner.layers.segment_1.children():
    fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)

  # Shard the full TAMM model (handles embedding, norm, positional_encoding,
  # and the TiedWeightLinear output_transform which shares weights with the
  # embedding — these must be sharded together to avoid mesh conflicts).
  fully_shard(inner, **fsdp_config, reshard_after_forward=reshard)

  # Shard the outer wrapper.
  fully_shard(model, **fsdp_config)

  # Disable FSDP's automatic gradient division for all FSDP modules
  disable_fsdp_gradient_division(model)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    # pyrefly: ignore [invalid-param-spec]
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
