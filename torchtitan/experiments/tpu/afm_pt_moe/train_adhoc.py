"""AFM PT MoE adhoc training script without dependency on TorchTitan.
"""

import gc
import time
from typing import Final

from absl import flags
from jax import profiler
import torch
from torch import distributed as dist
from torch.distributed import fsdp
import torchtitan.config
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import tpu_job_config
from torchtitan.experiments.tpu import utils as tpu_utils
import tamm
import tamm.layers


TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP


_SIZE = flags.DEFINE_enum(
    "size",
    "24b",
    ["3b", "24b"],
    "AFM PT MoE size variant. 3b = num_layers_per_track=4 (~3.1 B params, "
    "comparable to afmv7-3B); 24b = num_layers_per_track=48 (production "
    "~24.7 B). Both keep num_tracks=8, num_experts=4, attention_hidden_dim="
    "1536, sparse_feed_forward_hidden_dim=2048 — only depth changes.",
)


# KEY HYPERPARAMETERS — local_batch_size, seq_len, and steps are now read
# from the toml's [training] section (same as train_minimal_simple.py); the
# old hardcoded values have been moved into the toml so a single config
# applies to both entry points. seq=8192 is the program target; was OOMing
# the JIT staging arena in earlier TPUTrainer-driven runs, so the toml ships
# with seq_len=512 as the smallest workable shape.
LINEAR_SOFTMAX_LOSS_CHUNK_SIZE = 8192

# Model architecture (matches the production AFM PT MoE 24B snippet; the
# `--size=3b` flag overrides num_layers_per_track to 4).
VOCAB_SIZE = 262000
NUM_EXPERTS = 4

PROFILE_DIR_BASE = "./traces/"
PROFILE_START_STEP = 7
PROFILE_END_STEP = 9


def train_step(model, inp, optimizer):
  with profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  with profiler.TraceAnnotation("fwd"):
    output = model(inp, mode="SKIP_OUTPUT_LAYER")
    hidden = output.last_hidden_state  # (B, S, H)

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
  size = _SIZE.value
  # Both sizes share architecture except for depth. num_tracks defaults to 8
  # (TAMM default) so total layer-equivalents = 8 * num_layers_per_track.
  num_layers_per_track = {"3b": 4, "24b": 48}[size]

  local_batch_size = job_config.training.local_batch_size
  context_length = job_config.training.seq_len
  steps = job_config.training.steps

  print("***")
  print(f"AFM PT MoE adhoc training (size={size})")
  print("Hyperparameters:")
  print(f"  {local_batch_size=}")
  print(f"  {context_length=}")
  print(f"  {steps=}")
  print(f"  {LINEAR_SOFTMAX_LOSS_CHUNK_SIZE=}")
  print(f"  {VOCAB_SIZE=}")
  print(f"  {NUM_EXPERTS=}")
  print(f"  {num_layers_per_track=}")
  print("***\n")

  # Enables "tpu" to be used as device.
  device = tpu_utils.get_device()

  # Setup distributed.
  dist.init_process_group(backend="tpu_dist")

  RANK: Final = dist.get_rank()
  WORLD_SIZE: Final = dist.get_world_size()

  # Workaround for the TPU set_determinism hang (b/498659628). Without a
  # manual seed, RNG ops on the TPU device — including the `torch.randint`
  # input setup below and the per-param `uniform_()` initialization later —
  # trigger cross-rank RNG-state accesses that hang the next collective.
  # Mirrors train_minimal.py.
  torch.manual_seed(42)

  PROFILE_DIR = f"{PROFILE_DIR_BASE}-{RANK}_of_{WORLD_SIZE}"

  # Set up input data.
  inp = torch.randint(
      0, VOCAB_SIZE, (local_batch_size, context_length), device="tpu"
  )

  default_dtype = TORCH_DTYPE_MAP[job_config.training.dtype]
  torch.set_default_dtype(default_dtype)
  print(f">>> set default dtype to {default_dtype}")

  # Production AFM PT MoE config. Fields not specified use TAMM defaults
  # (hidden_dim=2048, num_tracks=8, num_layers_per_track_per_sync_point=4,
  # num_kv_heads=None ⇒ MHA, num_experts_per_token=2,
  # dense_feed_forward_hidden_dim=5888 unused).
  config = tamm.models.afm_text.afm_pt_moe.AFMParallelTrackMoEConfig(
      pretrained=False,
      attention_hidden_dim=1536,
      attention_layer_pattern=[
          "local_rope",
          "local_rope",
          "local_rope",
          "global_nope",
      ],
      experts_router_logits_cap=None,
      feed_forward_layer_pattern=["sparse"],
      local_attention_window_size=511,
      norm_eps=1e-06,
      num_experts=NUM_EXPERTS,
      num_heads=12,
      num_layers_per_track=num_layers_per_track,
      pre_norm="pre_scale_rms_norm",
      pre_residual_norm="rms_norm",
      rope_theta=10000,
      scale_qk_norm=False,
      sparse_feed_forward_hidden_dim=2048,
      tracks_combine_norm="pre_scale_rms_norm",
      tracks_combine_op="sum",
      tracks_dispatch_norm="rms_norm",
      vocab_size=VOCAB_SIZE,
      sdpa_implementation="native_torch",
  )

  print(">>> creating model on meta device")
  with torch.device("meta"):
    model = config.create_model()
  print(">>> done creating model on meta device")

  # Activation checkpointing intentionally OMITTED — TAMM dropless MoE breaks
  # both AC reentrancy modes (see module docstring).

  # Apply FSDP on each TransformerLayer. AFM PT MoE's layer iterator yields
  # transformer layers across all parallel tracks (matches the user's
  # standalone scripts/tamm_train_tpu.py pattern). Per-layer sharding avoids
  # the XLA 32-bit indexing overflow that whole-model allgather hits when
  # >2.1 B elements are gathered at once.
  for layer in model.layers.iter_transformer_layers():
    if hasattr(layer, "attention"):
      fsdp.fully_shard(layer.attention)
    if hasattr(layer, "feed_forward"):
      fsdp.fully_shard(layer.feed_forward)
  fsdp.fully_shard(model)

  print(">>> moving model to tpu device")
  model.to_empty(device="tpu")
  print(">>> done moving model to tpu device")

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
    print(f"dtype of layers (default {default_dtype} skipped):")
    for name, param in model.named_parameters():
      if param.dtype != default_dtype:
        print(f"  {name}: {param.dtype}")
    print("***\n")

  # Set up the training components.
  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad], lr=1e-4
  )

  print(">>> barrier before starting training")
  dist.barrier()
  print(">>> barrier removed")
  print(">>> starting training")

  # Setup metrics and traces.
  prev_split_time = time.perf_counter()

  for step in range(steps):
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
          f"{step=}/{steps=}, "
          f"Loss: {loss_cpu:.4f}, "
          f"TPS: {local_batch_size * context_length / step_time:.2f}, "
          f"Step time: {step_time:.2f} seconds"
          f"{torch.tpu._get_cache_misses()=}"
      )
    gc.collect()

  dist.destroy_process_group()


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
