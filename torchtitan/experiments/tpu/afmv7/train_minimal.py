r"""AFMv7 minimal training script using torchtitan infra.

Example with v6e-4 vm
  export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=65536' && \
  torchrun --nproc_per_node=4 \
  -m torchtitan.experiments.tpu.afmv7.train_minimal \
  --module=torchtitan.experiments.tpu.afmv7 \
  --config=afmv7_3b_lora
"""

import functools
import os
import time
import typing
from typing import Tuple

from jax import profiler as jax_profiler
import torch
import torch.nn.functional as F
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import model_annotator
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.tools.profiler as torchtitan_profiler
import torchtitan.experiments.tpu.afmv7  # trigger afmv7_tpu model registration
from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.tools import utils
import torchtitan.tools.logging


TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
ParallelDims = torchtitan.distributed.ParallelDims

logger = torchtitan.tools.logging.logger


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

LINEAR_SOFTMAX_LOSS_CHUNK_SIZE = 8192


def partitioned_linear_softmax_cross_entropy_loss(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    output_linear,
    chunk_size: int = LINEAR_SOFTMAX_LOSS_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Chunked linear → softmax → CE loss to avoid materialising the full logits.

  Instead of a single (B, S, vocab_size) tensor, processes the sequence in
  chunks so peak memory is O(chunk_size × vocab_size) rather than O(S ×
  vocab_size).

  Args:
    hidden: Shape (B, S, H) — last hidden states from the model body.
    targets: Shape (B, S) — token ids shifted by 1 (next-token targets).
    output_linear: Either a weight tensor of shape (H, vocab_size) as returned
      by the model forward when use_loss_kernel=True (kept gathered by FSDP), or
      a callable (nn.Module) such as model.model.output_transform that accepts a
      chunk tensor and returns logits.
    chunk_size: Size of the chunks to process.

  Returns:
    (loss, hidden_grad): scalar loss and gradient w.r.t. hidden.
  """
  if isinstance(output_linear, torch.Tensor):
    # FSDP2 keeps the weight as a DTensor (Replicate placement)
    # even after forward
    # when reshard_after_forward=False. Extract the local tensor so F.linear
    # receives a plain torch.Tensor and avoids the
    # DTensor/plain-tensor mismatch.
    w = (
        output_linear.to_local()
        if hasattr(output_linear, "to_local")
        else output_linear
    )
    apply_linear = lambda chunk, w=w: F.linear(chunk, w.t())
  else:
    apply_linear = output_linear

  B, S, _ = hidden.shape
  hidden_grad = torch.zeros_like(hidden)
  total_loss = hidden.new_zeros(())

  for b in range(B):
    for start in range(0, S, chunk_size):
      end = min(start + chunk_size, S)
      chunk = hidden[b : b + 1, start:end, :].detach().requires_grad_(True)
      logits = apply_linear(chunk)  # (1, end-start, vocab_size)
      chunk_loss = F.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          targets[b : b + 1, start:end].reshape(-1),
          reduction="sum",
      )
      chunk_loss.backward()
      hidden_grad[b : b + 1, start:end, :] = chunk.grad
      total_loss += chunk_loss.detach()

  return total_loss, hidden_grad


def train_step(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_config: TPUTrainerConfig,
    use_pallas: bool = False,
    use_chunked_loss: bool = False,
    graph_split: bool = False,
    manual_ddp: bool = False,
    should_log: bool = False,
) -> torch.Tensor:
  """Single training step.

  Three loss paths (mutually exclusive, checked in priority order):

  Pallas path (use_pallas=True):
    loss_fn = pallas_cross_entropy_loss (from build_cross_entropy_loss).
    Forward returns (hidden, weight.t()); fused linear+CE kernel runs without
    materialising the full logit tensor. Standard loss.backward().

  Chunked path (use_chunked_loss=True):
    loss_fn = partitioned_linear_softmax_cross_entropy_loss.
    Forward returns (hidden, None); output_transform weight is all-gathered
    once as a plain tensor to avoid the CPUSafeHistcMode/DTensor aten.bmm
    mismatch. Gradients w.r.t. hidden accumulated manually.

  Logits path (default, both flags False):
    Forward returns the full logit tensor (B, S, vocab_size).
    Standard F.cross_entropy + loss.backward().

  Args:
    model: Parallelised AFMv7 wrapper.
    tokens: Input token ids, shape (B, S).
    targets: Target token ids, shape (B, S).
    optimizer: Optimizer for the trainable (adapter) parameters.
    loss_fn: pallas_cross_entropy_loss or
      partitioned_linear_softmax_cross_entropy_loss; unused for logits path.
    train_config: The configuration for the training job.
    use_pallas: Enable the Pallas fused-kernel path.
    use_chunked_loss: Enable the chunked CE path.
    graph_split: Insert synchronize() between forward+loss and backward.
    manual_ddp: Enable manual DDP all-reduce in training loop.
    should_log: Whether to log metrics in this step.

  Returns:
    Scalar loss tensor.
  """
  with jax_profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  if graph_split:
    import torch_tpu._internal.sync  # pylint: disable=g-import-not-at-top

  if use_chunked_loss:
    # Inner model reshards all params after forward; only all-gather the
    # output_transform weight (~628 MB) as a plain tensor before the chunk loop.
    hidden, _ = model(tokens)  # hidden: (B, S, H)
    w = model.model.output_transform.weight
    w_plain = (
        w.full_tensor().detach() if hasattr(w, "full_tensor") else w.detach()
    )
    with jax_profiler.TraceAnnotation("chunked_loss"):
      loss, hidden_grad = loss_fn(hidden, targets, w_plain.t())
    if graph_split:
      torch_tpu._internal.sync.synchronize(hidden_grad, wait=False)
    hidden.backward(hidden_grad)
  else:
    if use_pallas:
      hidden, output_weight_t = model(tokens)
      # hidden: (B, S, H)
      # output_weight_t: (H, vocab_size) — gathered tensor kept valid by FSDP
      # (reshard_after_forward=False on inner module).
      with jax_profiler.TraceAnnotation("pallas_loss"):
        loss = loss_fn((hidden, output_weight_t), targets)  # pytype: disable=missing-parameter
    else:
      # Standard logits path: forward returns (B, S, vocab_size) logits.
      logits = model(tokens)
      with jax_profiler.TraceAnnotation("loss"):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
    if graph_split:
      torch_tpu._internal.sync.synchronize(loss, wait=False)
    loss.backward()

  if manual_ddp:
    with jax_profiler.TraceAnnotation("all_reduce"):
      for p in model.parameters():
        if p.requires_grad and p.grad is not None:
          torch.distributed.all_reduce(
              p.grad, op=torch.distributed.ReduceOp.AVG
          )
  if train_config.training.max_norm > 0.0:
    with jax_profiler.TraceAnnotation("grad_norm"):
      grad_norm = clip_grad_norm_(
          model.parameters(), train_config.training.max_norm
      )
  else:
    grad_norm = torch.tensor(0.0)

  with jax_profiler.TraceAnnotation("optimizer.step"):
    optimizer.step()

  return loss, grad_norm


def build_random_dataloader(train_config, vocab_size, device):
  """Yields the same random tokens every step by generating the tensor once

  outside the loop. This matches the behavior of the train_adhoc.py script.

  Args:
    train_config: The configuration for the training job.
    vocab_size: The size of the vocabulary.
    device: The device to put the data on.
  """
  local_batch_size = train_config.training.local_batch_size
  seq_len = train_config.training.seq_len
  x = torch.randint(
      0, vocab_size, (local_batch_size, seq_len + 1), device=device
  )
  while True:
    yield {"input": x[:, :-1]}, x[:, 1:]


def start_trainer(train_config: TPUTrainerConfig) -> None:
  """Starts the training process.

  Args:
    train_config: The configuration for the training job.
  """
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()
    train_config.maybe_log()

  device = tpu_utils.get_device()

  if world_size > 1:
    dist_utils.init_distributed(
        train_config.comm,
        enable_cpu_backend=train_config.training.enable_cpu_offload,
        base_folder=train_config.dump_folder,
    )
    # Force all ranks to flush pending device-init XLA ops
    torch.distributed.barrier()

  dp_replicate = train_config.parallelism.data_parallel_replicate_degree
  dp_shard = train_config.parallelism.data_parallel_shard_degree

  if dp_replicate == -1:
    dp_shard = 1  # For full DDP, we don't want sharding
    cp = train_config.parallelism.context_parallel_degree
    tp = train_config.parallelism.tensor_parallel_degree
    pp = train_config.parallelism.pipeline_parallel_degree
    dp_replicate = world_size // (dp_shard * cp * tp * pp)

  parallel_dims = ParallelDims(
      dp_shard=dp_shard,
      dp_replicate=dp_replicate,
      cp=train_config.parallelism.context_parallel_degree,
      tp=train_config.parallelism.tensor_parallel_degree,
      pp=train_config.parallelism.pipeline_parallel_degree,
      ep=train_config.parallelism.expert_parallel_degree,
      world_size=world_size,
  )
  logger.info("parallel_dims: %s", parallel_dims)
  train_config.maybe_log()

  # Default to 1 data worker and rank 0 for single-device training.
  batch_degree, batch_rank = 1, 0
  if world_size > 1:
    if parallel_dims.dp_enabled:
      # If data parallelism is enabled, shard data across replicas.
      batch_mesh = parallel_dims.get_mesh("batch")
      batch_degree, batch_rank = (
          batch_mesh.size(),
          batch_mesh.get_local_rank(),
      )
    else:
      # If only model parallelism is used, all devices are in the same replica.
      # They must see the same data, so degree is 1 and rank is 0.
      batch_degree, batch_rank = 1, 0

  # TODO b/498659628: Re-enable set_determinism once the hang on TPU is fixed.
  seed = train_config.debug.seed or 42
  if utils.get_device_type() == "tpu":
    torch.manual_seed(seed)
    logger.info(
        "Set manual seed to %d on all ranks (workaround for set_determinism"
        " hang)",
        seed,
    )
  else:
    if world_size > 1:
      logger.info("world_mesh: %s", parallel_dims.world_mesh)
      dist_utils.set_determinism(
          parallel_dims,
          device,
          train_config.debug,
          distinct_seed_mesh_dims=["pp"],
      )

  use_loss_kernel = (
      isinstance(train_config, TPUTrainerConfig)
      and train_config.loss_kernel.use_loss_kernel
  )
  use_chunked_loss = (
      isinstance(train_config, TPUTrainerConfig)
      and train_config.afmv7.use_chunked_loss
  )
  if use_loss_kernel and use_chunked_loss:
    raise ValueError(
        "use_loss_kernel and use_chunked_loss are mutually exclusive. "
        "Pass --loss_kernel.no-use_loss_kernel to disable the Pallas kernel "
        "when using --afmv7.use_chunked_loss."
    )
  is_manual_ddp = (
      parallel_dims.dp_replicate_enabled
      and not parallel_dims.fsdp_enabled
      and isinstance(train_config, TPUTrainerConfig)
      and train_config.afmv7.enable_manual_ddp
  )
  if is_manual_ddp:
    logger.info("Enabling manual DDP all-reduce in training loop")

  use_graph_split = (
      isinstance(train_config, TPUTrainerConfig)
      and train_config.tpu_config.use_graph_split
  )
  log_freq = train_config.metrics.log_freq

  assert train_config.model_spec is not None
  model_spec = train_config.model_spec
  model_args = model_spec.model
  model_args.update_from_config(trainer_config=train_config)

  tokenizer = typing.cast(
      torchtitan.components.tokenizer.HuggingFaceTokenizer,
      train_config.tokenizer.build(tokenizer_path=train_config.hf_assets_path),
  )

  if train_config.dataloader.dataset == "random":
    logger.info("Using random data loader instead of real dataset.")
    dataloader = build_random_dataloader(
        train_config, model_args.vocab_size, device  # pytype: disable=attribute-error
    )
  else:
    dataloader = train_config.dataloader.build(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=tokenizer,
        seq_len=train_config.training.seq_len,
        local_batch_size=train_config.training.local_batch_size,
    )

  logger.info(
      "Building %s %s with %s",
      train_config.model_spec.name,
      train_config.model_spec.flavor,
      model_args,
  )

  metrics_processor = train_config.metrics.build(
      parallel_dims=parallel_dims,
      dump_folder=train_config.dump_folder,
      pp_schedule=train_config.parallelism.pipeline_parallel_schedule,
      config_dict=train_config.to_dict(),
      has_quantization=False,
  )

  with (
      torch.device("meta"),
      utils.set_default_dtype(TORCH_DTYPE_MAP[train_config.training.dtype]),
  ):
    model = model_args.build()

  model_param_count, _ = model_args.get_nparams_and_flops(
      model, train_config.training.seq_len
  )
  logger.info(
      "Model %s %s size: %d total parameters",
      train_config.model_spec.name,
      train_config.model_spec.flavor,
      model_param_count,
  )

  from torchtitan.experiments.tpu.afmv7.model.model import OutputMode  # pylint: disable=g-import-not-at-top

  if use_loss_kernel:
    model._output_mode = OutputMode.HIDDEN_AND_WEIGHT
  elif use_chunked_loss:
    model._output_mode = OutputMode.HIDDEN
  # else: default OutputMode.LOGITS — forward returns full logit tensor.

  if train_config.training.enable_cpu_offload:
    logger.info("Materializing model on CPU for CPU offloading")
    model = model.to_empty(device="cpu")

  if world_size > 1:
    model_spec.parallelize_fn(model, parallel_dims, train_config)

  if use_loss_kernel:
    loss_fn = build_cross_entropy_loss(train_config)
    logger.info(
        "Loss: pallas_cross_entropy_loss (fused linear+CE Pallas kernel)"
    )
  elif use_chunked_loss:
    loss_fn = partitioned_linear_softmax_cross_entropy_loss
    logger.info(
        "Loss: partitioned_linear_softmax_cross_entropy_loss (chunked). "
    )
  else:
    loss_fn = None
    logger.info("Loss: F.cross_entropy on full logits.")

  logger.info("Moving model to device %s", device)
  if not train_config.training.enable_cpu_offload:
    model = model.to_empty(device=device)
  with torch.no_grad():
    model.init_weights()

  # Enable profiling annotations only if compilation is disabled.
  if not train_config.compile.enable and train_config.profiler.enable_profiling:
    model_annotator.wrap_model(model)

  # Build optimizer over trainable (adapter) params only.
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  logger.info(
      "Trainable parameters: %d / %d",
      sum(p.numel() for p in trainable_params),
      model_param_count,
  )
  if train_config.optimizer.implementation == "fused":
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    train_config.optimizer.implementation = "foreach"

  foreach = train_config.optimizer.implementation == "foreach"
  fused = train_config.optimizer.implementation == "fused"

  optimizer = torch.optim.AdamW(
      trainable_params,
      lr=train_config.optimizer.lr,
      eps=train_config.optimizer.eps,
      foreach=foreach,
      fused=fused,
  )

  if (
      train_config.compile.enable
      and "optimizer" in train_config.compile.components
  ):
    logger.info("Applying torch.compile to optimizer.step")
    optimizer.step = torch.compile(
        optimizer.step,
        backend=train_config.compile.backend,
        fullgraph=True,
        dynamic=False,
    )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()

  steps = train_config.training.steps
  local_batch_size = train_config.training.local_batch_size
  seq_len = train_config.training.seq_len

  # calculate model size and flops per token
  (
      model_param_count,
      num_flops_per_token,
  ) = model_args.get_nparams_and_flops(model, train_config.training.seq_len)
  metrics_processor.num_flops_per_token = num_flops_per_token

  logger.info(
      "Model %s %s size: %s total parameters",
      train_config.model_spec.name,
      train_config.model_spec.flavor,
      model_param_count,
  )

  device_name = tpu_utils.get_device_module().get_device_name()
  peak_flops = utils.get_peak_flops(device_name)
  logger.info(
      "Peak FLOPS for %s used for computing MFU: %d",
      device_name,
      peak_flops,
  )

  total_tokens = 0
  total_time = 0.0
  start_time = None
  accumulated_tokens = 0
  accumulated_time = 0.0
  accumulated_steps = 0

  warmup_steps = train_config.lr_scheduler.warmup_steps
  profiler_workaround.init(train_config)

  ntokens_seen = 0
  with torchtitan_profiler.Profiler(
      train_config.profiler,
      global_step=0,
      base_folder=train_config.dump_folder,
  ) as profiler:
    data_iterator = iter(dataloader)
    for step in range(steps):
      if step == warmup_steps:
        if torch.distributed.is_initialized():
          torch.distributed.barrier()
        start_time = time.perf_counter()
      step_start = time.perf_counter()
      step_tokens = local_batch_size * seq_len
      ntokens_seen += step_tokens
      metrics_processor.ntokens_since_last_log += step_tokens

      t0 = time.perf_counter()
      batch = next(data_iterator)
      metrics_processor.data_loading_times.append(time.perf_counter() - t0)

      tokens = batch[0]["input"].to(device)
      targets = batch[1].to(device)

      should_log = (step + 1) % log_freq == 0 or step == (steps - 1)

      with jax_profiler.TraceAnnotation("train", step_num=step):
        loss, grad_norm = train_step(
            model,
            tokens,
            targets,
            optimizer,
            loss_fn,
            train_config,
            use_pallas=use_loss_kernel,
            use_chunked_loss=use_chunked_loss,
            graph_split=use_graph_split,
            manual_ddp=is_manual_ddp,
            should_log=should_log,
        )

      if should_log:
        with jax_profiler.TraceAnnotation("D2H", step_num=step):
          loss_cpu = loss.cpu().item()
      else:
        if use_graph_split:
          import torch_tpu._internal.sync  # pylint: disable=g-import-not-at-top
          torch_tpu._internal.sync.synchronize(loss, wait=False)
        loss_cpu = float("nan")

      if profiler:
        profiler.step()

      step_end = time.perf_counter()
      step_time = step_end - step_start

      accumulated_tokens += step_tokens
      accumulated_time += step_time
      accumulated_steps += 1

      if step >= warmup_steps:
        total_tokens += step_tokens

      if should_log:
        avg_step_time = accumulated_time / accumulated_steps
        accumulated_tokens = 0
        accumulated_time = 0.0
        accumulated_steps = 0

        extra_metrics = {
            "lr": optimizer.param_groups[0]["lr"],
            "n_tokens_seen": ntokens_seen,
            "avg_step_time": avg_step_time,
        }
        metrics_processor.log(
            step + 1,
            loss_cpu / step_tokens,
            loss_cpu / step_tokens,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    if torch.distributed.is_initialized():
      torch.distributed.barrier()
    if start_time is not None:
      total_time = time.perf_counter() - start_time

  avg_tps = total_tokens / total_time if total_time > 0 else 0.0
  avg_tflops = num_flops_per_token * avg_tps / 1e12
  avg_mfu = 100 * num_flops_per_token * avg_tps / peak_flops
  if rank == 0:
    logger.info(
        "Average TPS (excl. %d warmup steps): %.0f  "
        "avg TFlops: %.2f  avg MFU: %.2f%%",
        warmup_steps,
        avg_tps,
        avg_tflops,
        avg_mfu,
    )

  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
  logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
