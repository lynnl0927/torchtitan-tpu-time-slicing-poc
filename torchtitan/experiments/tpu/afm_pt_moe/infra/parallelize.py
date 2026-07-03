"""Parallelization utilities for AFMPTMoe on TPU."""

import torch
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
import torchtitan.config
import torchtitan.distributed
import torchtitan.trainer
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.afm_pt_moe.model.model import OutputMode
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.tools.logging import logger


def parallelize_afm_pt_moe(
    model: nn.Module,
    parallel_dims: torchtitan.distributed.ParallelDims,
    train_config: (
        torchtitan.trainer.Trainer.Config
        | torchtitan.experiments.tpu.tpu_job_config.TPUTrainerConfig
    ),
) -> nn.Module:
  """Apply parallelism and activation checkpointing to AFMPTMoe.

  Supports:
  - Activation checkpointing (AC) via TAMM's native checkpoint_activations()
  - FSDP2 data parallelism (shard across fsdp mesh)
  - DDP-style replication (dp_replicate mesh, handled by FSDP2 replicate dim)

  Tensor Parallelism is not supported for AFMPTMoe since the TAMM model's
  internal layer structure is not wired for DTensor-based TP.

  Args:
    model: The AFMPTMoe model to parallelize.
    parallel_dims: Parallelism dimensions.
    train_config: Job configuration.

  Returns:
    The parallelized model.

  Raises:
    RuntimeError: If DDP has not supported > 1D parallelism, or if Tensor
      Parallelism is enabled.
  """
  if parallel_dims.tp_enabled:
    raise RuntimeError("Tensor Parallelism is not supported for AFMPTMoe yet")

  if (
      isinstance(train_config, tpu_job_config.TPUTrainerConfig)
      and train_config.splash_attention_kernel.use_splash_attention_kernel
  ):
    workarounds.use_splash_attention_patch(model)
  if (
      isinstance(train_config, tpu_job_config.TPUTrainerConfig)
      and train_config.loss_kernel.use_loss_kernel
  ):
    # Switch the wrapper into HIDDEN_AND_WEIGHT mode: forward returns
    # (hidden, output_transform.weight.t()) so loss.pallas_cross_entropy_loss
    # can run fused linear+softmax+CE without materializing the full
    # (B, S, vocab_size) logit tensor. Mirrors afmv7's parallelize.py
    # mechanism (workarounds.use_output_projection_patch is afmv7-specific —
    # it monkey-patches model.output, but AFM PT MoE's LM head lives at
    # model.output_transform; the wrapper-level OutputMode handles this
    # uniformly without monkey-patching).
    model._output_mode = OutputMode.HIDDEN_AND_WEIGHT
    logger.info(
        "Pallas loss kernel enabled for AFM PT MoE: forward returns "
        "(hidden, output_transform.weight.t()); FSDP will keep the inner "
        "TAMM module gathered after forward so the weight tensor stays "
        "valid for the loss kernel."
    )

  model_output_mode = getattr(model, "_output_mode", OutputMode.LOGITS)

  model_compile_enabled = (
      train_config.compile.enable and "model" in train_config.compile.components
  )

  # Apply activation checkpointing via TAMM's native API.
  if train_config.activation_checkpoint.mode != "none":
    apply_ac(model)

  # Apply torch.compile to each TransformerLayer in both segments.
  # Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.
  if model_compile_enabled:
    apply_compile(model, train_config.compile)

  keep_output_weight_gathered = (
      model_output_mode == OutputMode.HIDDEN_AND_WEIGHT
  )

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
            train_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            train_config.training.mixed_precision_reduce
        ],
        cpu_offload=train_config.training.enable_cpu_offload,
        reshard_after_forward_policy=train_config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=parallel_dims.pp_enabled,
        keep_output_weight_gathered=keep_output_weight_gathered,
    )

    if parallel_dims.dp_replicate_enabled:
      logger.info("Applied HSDP to the model")
    else:
      logger.info("Applied FSDP to the model")

    if train_config.training.enable_cpu_offload:
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
            train_config.training.mixed_precision_param
        ],
        reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
            train_config.training.mixed_precision_reduce
        ],
        cpu_offload=train_config.training.enable_cpu_offload,
        reshard_after_forward_policy=train_config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=parallel_dims.pp_enabled,
        keep_output_weight_gathered=keep_output_weight_gathered,
    )

  return model


def apply_ac(model: nn.Module) -> None:
  """Apply activation checkpointing via TAMM's native API.

  Args:
    model: The AFMPTMoe model.
  """
  logger.info("Applying activation checkpointing to AFMPTMoe segments.")
  for segment in model.model.layers.children():
    for layer in segment.children():
      if hasattr(layer, "checkpoint_activations"):
        layer.checkpoint_activations(use_reentrant=False)


def apply_compile(
    model: nn.Module, compile_config: "torchtitan.config.CompileConfig"
) -> None:
  """Apply torch.compile to each TransformerLayer in segments.

  Note: torch.compile may not work correctly on CPU — use only on TPU/GPU.

  Args:
    model: The AFMPTMoe model.
    compile_config: Compile configuration.
  """
  logger.info("Applying torch.compile to AFMPTMoe TransformerLayers.")
  inner = model.model
  for seg in inner.layers.children():
    for layer_id, layer in seg.named_children():
      if type(layer).__name__ == "TransformerLayer":
        compiled = torch.compile(
            layer, backend=compile_config.backend, fullgraph=True
        )
        seg.register_module(layer_id, compiled)  # pyrefly: ignore[bad-argument-type]


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    pp_enabled: bool,
    keep_output_weight_gathered: bool = False,
) -> None:
  """Wrap AFMPTMoe with FSDP2 (fully_shard).

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
    keep_output_weight_gathered: When True, the inner TAMM module is sharded
      with reshard_after_forward=False so output_transform.weight stays
      gathered through the loss computation. Required when the wrapper's
      _output_mode is HIDDEN_AND_WEIGHT (Pallas linear-softmax-CE path);
      otherwise FSDP frees the weight after forward and the kernel receives a
      tensor with null data_ptr.

  Raises:
    ValueError: If reshard_after_forward_policy is invalid.
  """
  mp_policy = MixedPrecisionPolicy(
      param_dtype=param_dtype, reduce_dtype=reduce_dtype
  )
  fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
  if cpu_offload:
    # FSDP2 fully_shard takes `offload_policy` (CPUOffloadPolicy), not the old
    # FSDP1 `cpu_offload` (CPUOffload) kwarg. Mismatch was a latent bug —
    # cpu_offload was unreachable until afm_pt_moe_24b.toml enabled it.
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

  inner = model.model  # the TAMM module

  # Shard each individual TransformerLayer inside segments for
  # fine-grained memory and compute overlap.
  for seg in inner.layers.children():
    for layer in seg.children():
      if type(layer).__name__ == "TransformerLayer":
        fully_shard(layer, **fsdp_config, reshard_after_forward=reshard)

  # Shard the full TAMM model (handles embedding, norm, positional_encoding,
  # and the TiedWeightLinear output_transform which shares weights with the
  # embedding — these must be sharded together to avoid mesh conflicts).
  # When keep_output_weight_gathered=True (Pallas loss path), force
  # reshard_after_forward=False so output_transform.weight stays gathered
  # through the loss computation; the layer-level shards above still reshard
  # under the default policy.
  inner_reshard = False if keep_output_weight_gathered else reshard
  fully_shard(inner, **fsdp_config, reshard_after_forward=inner_reshard)

  # Disable FSDP's automatic gradient division for all FSDP modules
  disable_fsdp_gradient_division(model)
