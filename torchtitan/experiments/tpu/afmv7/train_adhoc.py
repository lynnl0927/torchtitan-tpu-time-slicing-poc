"""AFMv7 adhoc training script without dependency on TorchTitan."""

import gc
import time
from typing import Final

from jax import profiler
import torch
from torch import distributed as dist
from torch.distributed import fsdp
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import utils as tpu_utils
import tamm
import tamm.adapters
import tamm.layers


# KEY HYPERPARAMETERS
LOCAL_BATCH_SIZE = 1
# OOM starting 8192
CONTEXT_LENGTH = 4096
STEPS = 10
LINEAR_SOFTMAX_LOSS_CHUNK_SIZE = 8192

PROFILE_DIR_BASE = "./traces/"
PROFILE_START_STEP = 7
PROFILE_END_STEP = 9


def train_step(model, inp, optimizer):
  with profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  with profiler.TraceAnnotation("fwd"):
    output = model(inp, mode="SKIP_OUTPUT_LAYER")
    hidden = output.last_hidden_state  # (B, S, H)

  # with profiler.TraceAnnotation("output_transform"):
  #     # Apply the output linear layer to the hidden states to get logits.
  #     pred = model.output_transform(hidden)[:, :-1, :]  # (B, S-1, vocab_size)
  #     target = inp[:, 1:]  # (B, S-1)
  #     loss = torch.nn.functional.cross_entropy(
  #         pred.reshape(-1, pred.shape[-1]), target.reshape(-1)
  #     )

  # with profiler.TraceAnnotation("bwd"):
  #     loss.backward()

  with profiler.TraceAnnotation("chunked_loss"):
    # Targets are the next tokens: inp shifted left by 1.
    targets = inp[:, 1:]  # (B, S-1)
    hidden_for_loss = hidden[:, :-1, :]  # (B, S-1, H)
    loss, hidden_grad = partitioned_linear_softmax_cross_entropy_loss(
        hidden_for_loss, targets, model.output_transform
    )

  with profiler.TraceAnnotation("bwd"):
    # Propagate gradients back through the transformer body.
    hidden[:, :-1, :].backward(hidden_grad)

  with profiler.TraceAnnotation("optimizer.step()"):
    optimizer.step()

  return loss


def partitioned_linear_softmax_cross_entropy_loss(hidden, targets, linear):
  """Partitions the final linear -> softmax -> cross entropy loss to save memory.

  Instead of materializing the full (B, S, vocab_size) logit tensor, processes
  the sequence in chunks so peak memory is O(chunk_size * vocab_size) rather
  than O(S * vocab_size).

  Args:
      hidden: Shape (B, S, H) — last hidden states, detached from the graph.
      targets: Shape (B, S) — token ids shifted by 1 (next-token targets).
      linear: The final linear layer mapping hidden_dim -> vocab_size.

  Returns:
      (loss, hidden_grad): scalar loss value and gradient w.r.t. hidden.
  """
  B, S, _ = hidden.shape
  hidden_grad = torch.zeros_like(hidden)
  total_loss = hidden.new_zeros(())

  for b in range(B):
    for start in range(0, S, LINEAR_SOFTMAX_LOSS_CHUNK_SIZE):
      end = min(start + LINEAR_SOFTMAX_LOSS_CHUNK_SIZE, S)
      chunk = hidden[b : b + 1, start:end, :].detach().requires_grad_(True)
      logits = linear(chunk)  # (1, end-start, vocab)
      chunk_loss = torch.nn.functional.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          targets[b : b + 1, start:end].reshape(-1),
      )
      # Weight so that the losses from all chunks average correctly.
      weight = (end - start) / S
      (chunk_loss * weight).backward()
      hidden_grad[b : b + 1, start:end, :] = chunk.grad
      total_loss += chunk_loss.detach() * weight

  return total_loss, hidden_grad


def start_trainer(job_config: tpu_job_config.TPUJobConfig) -> None:
  print("***")
  print("Hyperparameters:")
  print(f"  {LOCAL_BATCH_SIZE=}")
  print(f"  {CONTEXT_LENGTH=}")
  print(f"  {STEPS=}")
  print(f"  {LINEAR_SOFTMAX_LOSS_CHUNK_SIZE=}")
  print("***\n")

  # Enables "tpu" to be used as device.
  device = tpu_utils.get_device()

  # Setup distributed.
  dist.init_process_group(backend="tpu_dist")

  RANK: Final = dist.get_rank()
  WORLD_SIZE: Final = dist.get_world_size()
  PROFILE_DIR = f"{PROFILE_DIR_BASE}-{RANK}_of_{WORLD_SIZE}"

  # Set up input data.
  inp = torch.randint(
      0, 153600, (LOCAL_BATCH_SIZE, CONTEXT_LENGTH), device="tpu"
  )

  # Set the default type, however, the lora config has a default dtype flag that
  # must also be set.
  torch.set_default_dtype(torch.bfloat16)

  # default values for Config args from afm_text_v7.py:
  # vocab_size: int = 153600,
  # hidden_dim: int = 2048,
  # num_layers: int = 56,
  # num_kv_reuse_layers: int = 21,
  # num_heads: int = 16,
  # num_kv_heads: _Optional[int] = 2,
  # hidden_dim_scale_factor: float = 3.25,
  config = tamm.models.afm_text.AFMTextV7.Config(
      pretrained=False,
      adapters={
          "lora": tamm.adapters.LoRAModelAdapter(
              rank=16,
              alpha=16,  # common default: alpha == rank → scale of 1.0
              dtype=torch.bfloat16,  # This is the dtype arg for lora that must also be set.
              adapt_attention_queries=True,
              adapt_attention_keys=True,
              adapt_attention_values=True,
              adapt_attention_outputs=True,
              adapt_feed_forward_hidden_states=True,
              adapt_feed_forward_outputs=True,
          )
      },
  )

  with torch.device("meta"):
    model = config.create_model()

  # Freeze base model.
  for name, param in model.named_parameters():
    if "adapters" not in name:
      param.requires_grad = False

  # Apply AC and FSDP on layers.
  for segment in [model.layers.segment_0, model.layers.segment_1]:
    for module in segment.children():
      if isinstance(module, tamm.layers.TransformerLayer):
        module.checkpoint_activations(use_reentrant=False)

  # Apply FSDP on layers per https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#model-initialization
  for segment in [model.layers.segment_0, model.layers.segment_1]:
    for module in segment.children():
      if isinstance(module, tamm.layers.TransformerLayer):
        fsdp.fully_shard(module)
  fsdp.fully_shard(model)

  model.to_empty(device="tpu")

  # TAMM weight initialization will calculate fan_in without factoring in
  # model parallelism, resulting in activation explosion.
  with torch.no_grad():
    for param in model.parameters():
      param.uniform_(-0.01, 0.01)
    for buffer in model.buffers():
      buffer.fill_(0)

  # Print a few stats about the model.
  if RANK == 0:
    num_params = sum(p.numel() for p in model.parameters())
    num_params_requires_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("***")
    print(f"{num_params=}")
    print(f"{num_params_requires_grad=}")
    print(
        f"{tamm._adapters_v1.utils.get_num_adapter_params(model)=}"
    )
    print("dtype of layers (torch.bfloat16 skipped):")
    for name, param in model.named_parameters():
      if param.dtype != torch.bfloat16:
        print(f"  {name}: {param.dtype}")
    print("***\n")

  # Set up the training components.
  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad], lr=1e-4
  )

  dist.barrier()

  # Setup metrics and traces.
  prev_split_time = time.perf_counter()

  for step in range(STEPS):
    if step == PROFILE_START_STEP:
      print(">>> starting trace for: ", PROFILE_DIR)
      profiler.start_trace(
          PROFILE_DIR, create_perfetto_link=False, profiler_options=None
      )

    if step == PROFILE_END_STEP:
      profiler.stop_trace()

    loss = train_step(model, inp, optimizer)

    # Get the loss. This is a barrier so we have to be careful with it.
    loss_cpu = loss.cpu()

    # Print stats.
    current_time = time.perf_counter()
    step_time = current_time - prev_split_time
    prev_split_time = current_time
    if RANK == 0:
      print(
          f"{step=}/{STEPS=}, "
          f"Loss: {loss_cpu:.4f}, "
          f"TPS: {LOCAL_BATCH_SIZE * CONTEXT_LENGTH / step_time:.2f}, "
          f"Step time: {step_time:.2f} seconds"
          f"{torch.tpu._get_cache_misses()=}"
      )
    gc.collect()

  dist.destroy_process_group()


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
