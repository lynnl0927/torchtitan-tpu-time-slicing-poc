"""Parallelization utilities for AFMTextV7 on TPU."""

import torch
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy as FSDPMixedPrecisionPolicy
import torchtitan.config
import torchtitan.distributed
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel as simple_fsdp_data_parallel
from torchtitan.experiments.simple_fsdp.simple_fsdp import MixedPrecisionPolicy as SimpleFSDPMixedPrecisionPolicy
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.afmv7.model.model import OutputMode
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

  use_splash_attention_kernel = False
  use_loss_kernel = False
  enable_amp = True
  if isinstance(job_config, tpu_job_config.TPUJobConfig):
    use_splash_attention_kernel = (
        job_config.tpu_config.use_splash_attention_kernel
    )
    use_loss_kernel = job_config.tpu_config.use_loss_kernel
    enable_amp = job_config.tpu_config.enable_amp

  if use_splash_attention_kernel:
    if isinstance(job_config, tpu_job_config.TPUJobConfig):
      workarounds.use_splash_attention_patch(
          model,
          block_q=job_config.tpu_config.sa_block_q,
          block_kv=job_config.tpu_config.sa_block_kv,
          block_dkv=job_config.tpu_config.sa_block_dkv,
          block_kv_compute=job_config.tpu_config.sa_block_kv_compute,
          block_q_dkv=job_config.tpu_config.sa_block_q_dkv,
          block_kv_dkv=job_config.tpu_config.sa_block_kv_dkv,
          block_kv_dkv_compute=job_config.tpu_config.sa_block_kv_dkv_compute,
          block_q_dq=job_config.tpu_config.sa_block_q_dq,
          block_kv_dq=job_config.tpu_config.sa_block_kv_dq,
          use_fused_bwd_kernel=job_config.tpu_config.sa_use_fused_bwd_kernel,
          q_layout=job_config.tpu_config.sa_q_layout,
          k_layout=job_config.tpu_config.sa_k_layout,
          v_layout=job_config.tpu_config.sa_v_layout,
      )
    else:
      workarounds.use_splash_attention_patch(model)

  if use_loss_kernel:
    # Enable Pallas loss: forward returns (hidden, weight.t()) so
    # pallas_cross_entropy_loss can do fused linear+CE without materializing
    # the full (B, S, vocab_size) logit tensor.
    model._output_mode = OutputMode.HIDDEN_AND_WEIGHT
    logger.info(
        "Pallas loss kernel enabled for AFMv7: skipping output projection "
        "in forward, deferring to pallas_cross_entropy_loss."
    )
    workarounds.use_output_projection_patch(model)

  # Auto-detect LoRA: freeze base model params and unfreeze adapter params.
  lora_params = _maybe_freeze_for_lora(model)

  # Determine if we should force LoRA params to be DDP (ignored by FSDP)
  ignored_lora_params = (
      lora_params if job_config.tpu_config.force_lora_parameter_ddp else None
  )

  model_compile_enabled = (
      job_config.compile.enable and "model" in job_config.compile.components
  )

  # Apply activation checkpointing via TAMM's native API.
  if job_config.activation_checkpoint.mode != "none":
    apply_ac(model)

  # Apply torch.compile to each TransformerLayer in both segments.
  # Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.
  if model_compile_enabled:
    apply_compile(model, job_config)

  model_output_mode = getattr(model, "_output_mode", OutputMode.LOGITS)

  use_simple_fsdp = False
  if isinstance(job_config, tpu_job_config.TPUJobConfig):
    use_simple_fsdp = job_config.tpu_config.use_simple_fsdp

  if parallel_dims.fsdp_enabled:
    if parallel_dims.dp_replicate_enabled:
      dp_mesh_dim_names = ["dp_replicate", "fsdp"]
      dp_mode = "hybrid_shard"
    else:
      dp_mesh_dim_names = ["fsdp"]
      dp_mode = "fully_shard"
    dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)

    if use_simple_fsdp:
      mp_policy = SimpleFSDPMixedPrecisionPolicy(
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              job_config.training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              job_config.training.mixed_precision_reduce
          ],
      ) if enable_amp else None
      model = simple_fsdp_data_parallel(
          model,
          dp_mesh,
          mode=dp_mode,
          mp_policy=mp_policy,
      )
      logger.info(
          "Applied Simple FSDP (dp mode=%s) to the model", dp_mode
      )
    else:
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
          keep_output_weight_gathered=(
              model_output_mode == OutputMode.HIDDEN_AND_WEIGHT),
          enable_amp=enable_amp,
          lora_params=ignored_lora_params,
      )

      if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
      else:
        logger.info("Applied FSDP to the model")

      if job_config.training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

  elif parallel_dims.dp_replicate_enabled:
    dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
    if use_simple_fsdp:
      raise RuntimeError("Simple FSDP does not support dp_replicate without fsdp enabled")

    if parallel_dims.world_size != dp_replicate_mesh.size():
      raise RuntimeError("DDP has not supported > 1D parallelism")

    if job_config.tpu_config.enable_manual_ddp:
      logger.info("Skipping FSDP/DDP wrapping for manual All-Reduce DDP")
    elif enable_amp:
      # TODO b/494360665: remove this once torch.autocast is supported
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
          keep_output_weight_gathered=(
              model_output_mode == OutputMode.HIDDEN_AND_WEIGHT
          ),
          enable_amp=enable_amp,
      )
      logger.info(
          "Applied FSDP to the model for replication mode because of AMP"
      )
    else:
      apply_ddp(
          model,
          dp_replicate_mesh,
          enable_compile=model_compile_enabled,
      )

  model.lora_params = lora_params
  model.force_lora_parameter_ddp = (
      job_config.tpu_config.force_lora_parameter_ddp
  )

  if job_config.tpu_config.force_lora_parameter_ddp and lora_params:
    if parallel_dims.fsdp_enabled:
      lora_dp_mesh = parallel_dims.get_mesh(
          ["dp_replicate", "fsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["fsdp"]
      )
    elif parallel_dims.dp_replicate_enabled:
      lora_dp_mesh = parallel_dims.get_mesh("dp_replicate")
    else:
      lora_dp_mesh = None

    if lora_dp_mesh:
      from tamm.adapters import AdaptedLayer
      adapted_layers = [
          m for m in model.model.modules() if isinstance(m, AdaptedLayer)
      ]
      for m in adapted_layers:
        replicate(
            m.adapters,
            device_mesh=lora_dp_mesh,
            bucket_cap_mb=100,
        )
      logger.info("Applied DDP/replicate to LoRA adapters")

  return model


def _maybe_freeze_for_lora(model: nn.Module) -> set[nn.Parameter]:
  """If the model contains TAMM AdaptedLayer modules (LoRA), freeze all base

  model parameters and unfreeze only the adapter parameters.

  This must run before FSDP wrapping so that FSDP2 can propagate
  requires_grad=False to the sharded DTensors and skip gradient sync for
  frozen parameters.

  Returns:
      The set of LoRA parameters that are trainable.
  """
  lora_params = set()
  try:
    from tamm.adapters import AdaptedLayer
  except ImportError:
    return lora_params

  adapted_layers = [
      m for m in model.model.modules() if isinstance(m, AdaptedLayer)
  ]
  if not adapted_layers:
    return lora_params

  # Freeze everything first.
  for p in model.model.parameters():
    if p.requires_grad:
      p.requires_grad_(False)

  # Unfreeze only adapter parameters.
  for m in adapted_layers:
    for p in m.adapters.parameters():
      p.requires_grad_(True)
      lora_params.add(p)
  
  return lora_params


def apply_ac(model: nn.Module) -> None:
  """Apply activation checkpointing via PyTorch Distributed's wrapper."""
  logger.info("Applying activation checkpointing via ptd_checkpoint_wrapper.")
  from torchtitan.experiments.tpu.utils import ptd_checkpoint_wrapper_with_early_stop as ptd_checkpoint_wrapper
  for segment in model.model.layers.children():
    for layer_id, layer in segment.named_children():
      wrapped_layer = ptd_checkpoint_wrapper(layer, preserve_rng_state=False)
      segment.register_module(layer_id, wrapped_layer)


def apply_compile(
    model: nn.Module, job_config: "torchtitan.config.JobConfig"
) -> None:
  """Apply torch.compile to each TransformerLayer, Segment or Whole Model.

  Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.

  Args:
    model: The AFMTextV7 model.
    job_config: Job configuration.
  """
  compile_mode = job_config.tpu_config.compile_mode

  inner = model.model
  if compile_mode == "whole":
    logger.info("Applying torch.compile to Whole Model.")
    model.model = torch.compile(
        inner, backend=job_config.compile.backend, fullgraph=True
    )
  elif compile_mode == "block":
    logger.info("Applying torch.compile to AFMTextV7 Segments (Block-level).")
    for seg_id, seg in inner.layers.named_children():
      logger.info(f"Compiling segment {seg_id}")
      compiled = torch.compile(
          seg, backend=job_config.compile.backend, fullgraph=True
      )
      inner.layers.register_module(seg_id, compiled)
  else:
    logger.info(
        "Applying torch.compile to AFMTextV7 TransformerLayers (Layer-level)."
    )
    for seg in (inner.layers.segment_0, inner.layers.segment_1):
      for layer_id, layer in seg.named_children():
        compiled = torch.compile(
            layer, backend=job_config.compile.backend, fullgraph=True
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
    keep_output_weight_gathered: bool = False,
    enable_amp: bool = True,
    lora_params: set[nn.Parameter] | None = None,
) -> None:
  """Wrap AFMTextV7 with FSDP2 (fully_shard).

  Shards at the individual TransformerLayer granularity inside each segment.
  Top-level non-layer children (embedding, norm, positional_encoding, etc.)
  are NOT sharded individually to avoid issues with TAMM's tied weights
  between the embedding and output_transform layers; those are handled by the
  top-level fully_shard call on the inner model.

  Args:
    model: Model to wrap.
    dtype: Base model dtype (what params are stored/initialized in).
    param_dtype: Dtype to cast params to during all-gather for compute.
    reduce_dtype: Dtype to use for gradient reduction (reduce-scatter).
    cpu_offload: Whether to enable CPU offloading.
    reshard_after_forward_policy: Reshard after forward policy.
    pp_enabled: Whether pipeline parallelism is enabled.
    keep_output_weight_gathered: Whether to keep the output weight gathered.
    enable_amp: Whether to enable AMP.
    lora_params: Parameters to be ignored by FSDP (e.g., LoRA adapters).

  Raises:
    ValueError: If reshard_after_forward_policy is invalid.
  """
  fsdp_config: dict = {"mesh": dp_mesh}
  if lora_params:
    fsdp_config["ignored_params"] = lora_params
  if enable_amp:
    fsdp_config["mp_policy"] = FSDPMixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
  if cpu_offload:
    fsdp_config["offload_policy"] = CPUOffloadPolicy()

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
  use_segments = hasattr(inner.layers, "segment_0")

  # Shard each individual TransformerLayer inside both segments for
  # fine-grained memory and compute overlap.
  if use_segments:
    logger.info("Sharding segments individually (FSDP).")
    for layer in inner.layers.segment_0.children():
      fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
    for layer in inner.layers.segment_1.children():
      fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)
  else:
    logger.info("No segments found, sharding layers (FSDP).")
    for layer in inner.layers.children():
      fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)

  # Shard the full TAMM model (handles embedding, norm, positional_encoding,
  # and the TiedWeightLinear output_transform which shares weights with the
  # embedding — these must be sharded together to avoid mesh conflicts).
  # When use_loss_kernel=True, TAMM returns (hidden, output_weight) and the
  # loss kernel uses output_weight after the model forward. Setting
  # reshard_after_forward=False keeps the output_transform weight gathered
  # through the loss computation; otherwise FSDP frees it and the Pallas
  # kernel receives a tensor with null data_ptr.
  inner_reshard = False if keep_output_weight_gathered else reshard
  fully_shard(inner, **fsdp_config, reshard_after_forward=inner_reshard)

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
