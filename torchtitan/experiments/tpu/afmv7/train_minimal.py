r"""AFMv7 minimal training script using torchtitan infra.

Example with v6e-4 vm (yeild ~7.1k TPS / chip):
  export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=131072' && \
  torchrun --nproc_per_node=4 \
  -m torchtitan.experiments.tpu.afmv7.train_minimal \
  --job.config_file=torchtitan/experiments/tpu/afmv7/train_configs/afmv7_3b_lora.toml \
  --model.name=afmv7_tpu \
  --model.flavor=3B-lora
"""

import os
import time
import typing
from typing import Tuple

from jax import profiler as jax_profiler
import torch
import torch.nn.functional as F
from torchtitan.components import metrics
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import model_annotator
from torchtitan.experiments.tpu import utils as tpu_utils
import torchtitan.experiments.tpu.afmv7  # trigger afmv7_tpu model registration
from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
import torchtitan.experiments.tpu.tpu_job_config
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
import torchtitan.tools.logging


TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
JobConfig = torchtitan.config.JobConfig
ParallelDims = torchtitan.distributed.ParallelDims
TPUJobConfig = torchtitan.experiments.tpu.tpu_job_config.TPUJobConfig
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
    optimizer: torch.optim.Optimizer,
    loss_fn,
    use_pallas: bool = False,
    use_chunked_loss: bool = False,
    graph_split: bool = False,
    manual_ddp: bool = False,
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
    optimizer: Optimizer for the trainable (adapter) parameters.
    loss_fn: pallas_cross_entropy_loss or
      partitioned_linear_softmax_cross_entropy_loss; unused for logits path.
    use_pallas: Enable the Pallas fused-kernel path.
    use_chunked_loss: Enable the chunked CE path.
    graph_split: Insert synchronize() between forward+loss and backward.

  Returns:
    Scalar loss tensor.
  """
  with jax_profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  # Shift labels by 1: position i predicts token i+1.
  targets = torch.roll(tokens, -1, dims=1)

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
  with jax_profiler.TraceAnnotation("optimizer.step"):
    optimizer.step()

  return loss


def start_trainer(job_config: JobConfig) -> None:
  """Starts the training process.

  Args:
    job_config: The configuration for the training job.
  """
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()
    job_config.maybe_log()

  device = tpu_utils.get_device()

  if world_size > 1:
    dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )
    # Force all ranks to flush pending device-init XLA ops
    torch.distributed.barrier()

  dp_replicate = job_config.parallelism.data_parallel_replicate_degree
  dp_shard = job_config.parallelism.data_parallel_shard_degree

  if dp_replicate == -1:
    dp_shard = 1  # For full DDP, we don't want sharding
    cp = job_config.parallelism.context_parallel_degree
    tp = job_config.parallelism.tensor_parallel_degree
    pp = job_config.parallelism.pipeline_parallel_degree
    dp_replicate = world_size // (dp_shard * cp * tp * pp)

  parallel_dims = ParallelDims(
      dp_shard=dp_shard,
      dp_replicate=dp_replicate,
      cp=job_config.parallelism.context_parallel_degree,
      tp=job_config.parallelism.tensor_parallel_degree,
      pp=job_config.parallelism.pipeline_parallel_degree,
      ep=job_config.parallelism.expert_parallel_degree,
      etp=job_config.parallelism.expert_tensor_parallel_degree,
      world_size=world_size,
  )
  logger.info("parallel_dims: %s", parallel_dims)
  job_config.maybe_log()

  # TODO b/498659628: Re-enable set_determinism once the hang on TPU is fixed.
  seed = job_config.debug.seed or 42
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
          job_config.debug,
          distinct_seed_mesh_dims=["pp"],
      )

  use_loss_kernel = (
      isinstance(job_config, TPUJobConfig)
      and job_config.tpu_config.use_loss_kernel
  )
  use_chunked_loss = (
      isinstance(job_config, TPUJobConfig)
      and job_config.tpu_config.use_chunked_loss
  )
  if use_loss_kernel and use_chunked_loss:
    raise ValueError(
        "use_loss_kernel and use_chunked_loss are mutually exclusive. "
        "Pass --tpu_config.no-use_loss_kernel to disable the Pallas kernel "
        "when using --tpu_config.use_chunked_loss."
    )
  is_manual_ddp = (
      parallel_dims.dp_replicate_enabled
      and not parallel_dims.fsdp_enabled
      and isinstance(job_config, TPUJobConfig)
      and job_config.tpu_config.enable_manual_ddp
  )
  if is_manual_ddp:
    logger.info("Enabling manual DDP all-reduce in training loop")

  use_graph_split = (
      isinstance(job_config, TPUJobConfig)
      and job_config.tpu_config.use_graph_split
  )
  log_freq = job_config.metrics.log_freq

  train_spec = train_spec_module.get_train_spec(job_config.model.name)

  model_args = train_spec.model_args[job_config.model.flavor]
  model_args.update_from_config(job_config)

  logger.info(
      "Building %s %s with %s",
      job_config.model.name,
      job_config.model.flavor,
      model_args,
  )

  build_metrics_processor_fn = (
      metrics.build_metrics_processor
      if train_spec.build_metrics_processor_fn is None
      else train_spec.build_metrics_processor_fn
  )
  metrics_processor = build_metrics_processor_fn(
      job_config, parallel_dims, model_args
  )

  with (
      torch.device("meta"),
      utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
  ):
    model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))

  model_param_count, _ = model_args.get_nparams_and_flops(
      model, job_config.training.seq_len
  )
  logger.info(
      "Model %s %s size: %d total parameters",
      job_config.model.name,
      job_config.model.flavor,
      model_param_count,
  )

  from torchtitan.experiments.tpu.afmv7.model.model import OutputMode  # pylint: disable=g-import-not-at-top
  if use_loss_kernel:
    model._output_mode = OutputMode.HIDDEN_AND_WEIGHT
  elif use_chunked_loss:
    model._output_mode = OutputMode.HIDDEN
  # else: default OutputMode.LOGITS — forward returns full logit tensor.

  if job_config.training.enable_cpu_offload:
    logger.info("Materializing model on CPU for CPU offloading")
    model = model.to_empty(device="cpu")

  if world_size > 1:
    train_spec.parallelize_fn(model, parallel_dims, job_config)

  if use_loss_kernel:
    loss_fn = build_cross_entropy_loss(job_config)
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
  if not job_config.training.enable_cpu_offload:
    model = model.to_empty(device=device)
  with torch.no_grad():
    model.init_weights()

  # Enable profiling annotations only if compilation is disabled.
  if not job_config.compile.enable and job_config.profiling.enable_profiling:
    model_annotator.wrap_model(model)

  # Build optimizer over trainable (adapter) params only.
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  logger.info(
      "Trainable parameters: %d / %d",
      sum(p.numel() for p in trainable_params),
      model_param_count,
  )
  if job_config.optimizer.implementation == "fused":
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    job_config.optimizer.implementation = "foreach"

  foreach = job_config.optimizer.implementation == "foreach"
  fused = job_config.optimizer.implementation == "fused"

  optimizer = torch.optim.AdamW(
      trainable_params,
      lr=job_config.optimizer.lr,
      eps=job_config.optimizer.eps,
      foreach=foreach,
      fused=fused,
  )

  if job_config.compile.enable and "optimizer" in job_config.compile.components:
    logger.info("Applying torch.compile to optimizer.step")
    optimizer.step = torch.compile(
        optimizer.step,
        backend=job_config.compile.backend,
        fullgraph=True,
        dynamic=False,
    )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()

  steps = job_config.training.steps
  local_batch_size = job_config.training.local_batch_size
  seq_len = job_config.training.seq_len
  vocab_size = model_args.vocab_size  # pytype: disable=attribute-error

  # Fixed dummy input — same tokens every step (matches scratchpad behaviour).
  tokens = torch.randint(
      0, vocab_size, (local_batch_size, seq_len), device=device
  )

  # calculate model size and flops per token
  (
      model_param_count,
      num_flops_per_token,
  ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)
  metrics_processor.num_flops_per_token = num_flops_per_token

  logger.info(
      "Model %s %s size: %s total parameters",
      job_config.model.name,
      job_config.model.flavor,
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
  accumulated_tokens = 0
  accumulated_time = 0.0
  accumulated_steps = 0

  warmup_steps = job_config.lr_scheduler.warmup_steps

  from torchtitan.experiments.tpu import jax_profiling

  maybe_enable_profiling = jax_profiling.maybe_enable_profiling

  ntokens_seen = 0
  with maybe_enable_profiling(
      job_config.profiling,
      global_step=0,
      base_folder=job_config.job.dump_folder,
  ) as profiler:
    for step in range(steps):
      step_start = time.perf_counter()
      step_tokens = local_batch_size * seq_len
      ntokens_seen += step_tokens
      metrics_processor.ntokens_since_last_log += step_tokens
      metrics_processor.data_loading_times.append(0.0)
      with jax_profiler.TraceAnnotation("train", step_num=step):
        loss = train_step(
            model,
            tokens,
            optimizer,
            loss_fn,
            use_pallas=use_loss_kernel,
            use_chunked_loss=use_chunked_loss,
            graph_split=use_graph_split,
            manual_ddp=is_manual_ddp,
        )

      if (step + 1) % log_freq == 0 or step == (steps - 1):
        with jax_profiler.TraceAnnotation("D2H", step_num=step):
          loss_cpu = loss.cpu().item()
        should_log = True
      else:
        if use_graph_split:
          import torch_tpu._internal.sync  # pylint: disable=g-import-not-at-top
          torch_tpu._internal.sync.synchronize(loss, wait=False)
        loss_cpu = float("nan")
        should_log = False

      if profiler:
        profiler.step()

      step_end = time.perf_counter()
      step_time = step_end - step_start

      accumulated_tokens += step_tokens
      accumulated_time += step_time
      accumulated_steps += 1

      if step >= warmup_steps:
        total_tokens += step_tokens
        total_time += step_time

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
            float("nan"),
            extra_metrics=extra_metrics,
        )

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
