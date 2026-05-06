"""AFM PT MoE-specific workarounds and patches.


Currently provides:

* ``use_segment_matmul_patch()`` — replaces TAMM's stock
  ``segment_matmul`` (which calls ``group_sizes.tolist()`` and deadlocks at
  multi-host) with a tokamax-backed Pallas ragged_dot kernel that takes
  ``group_sizes`` as a runtime tensor input.

Each ``use_*_patch`` function is idempotent and gated externally by a flag in
``TPUConfig`` (see ``torchtitan/experiments/tpu/tpu_job_config.py``).
"""

import torchtitan.tools.logging


logger = torchtitan.tools.logging.logger


_SEGMENT_MATMUL_PATCH_APPLIED = False


def use_segment_matmul_patch() -> None:
  """Replace TAMM's ``segment_matmul`` with a tokamax ragged_dot Pallas kernel.

  Why: ``tamm._ops.segment_matmul.torch.segment_matmul`` calls
  ``group_sizes.tolist()`` to obtain Python ints for ``torch.split``. At v6e-32
  multi-host (8 hosts × 4 chips, torch_tpu/PjRt) this D2H sync drains the
  deferred-op queue and tries to execute downstream collectives whose per-rank
  shapes diverge → deadlock. py-spy shows every rank stuck on the same
  ``group_sizes.tolist()`` frame.

  How the patch fixes this: the replacement keeps ``group_sizes`` as a TPU
  tensor end-to-end. It loops over the leading ``num_tracks`` dim and, for each
  track, calls ``segment_matmul_pallas_2d`` (which routes to
  ``tokamax.ragged_dot(implementation="mosaic")`` via ``pallas.jax_op``). The
  output shape is ``(tracks, total_tokens, n)`` — independent of routing
  values — so downstream collectives have matching shapes across ranks.

  DTensor handling: TAMM may pass DTensor for the FSDP-sharded weight (and
  possibly for inputs in some configurations). The patch unwraps via
  ``.to_local()`` before the Pallas call (the bridge is not DTensor-aware), and
  re-wraps the output via ``DTensor.from_local`` only if the input was a
  DTensor.

  Idempotent: calling twice is a no-op (the second call returns immediately).
  """
  global _SEGMENT_MATMUL_PATCH_APPLIED
  if _SEGMENT_MATMUL_PATCH_APPLIED:
    return

  import torch  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.kernels import segment_matmul as _kernel  # pylint: disable=g-import-not-at-top
  from tamm._ops.segment_matmul import torch as _tamm_seg  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from tamm._ops.segment_matmul import interface as _tamm_seg_interface  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _DTensor = torch.distributed.tensor.DTensor

  def _patched_segment_matmul(segmented_input, group_sizes, weight):
    """Drop-in for tamm.segment_matmul. ``group_sizes`` stays as a tensor."""
    if segmented_input.dim() != 3:
      raise ValueError(
          "segment_matmul currently only accepts 3-dimensional inputs"
          f" (got {segmented_input.dim()}-D)."
      )

    # DTensor unwrap. Same reason as splash_attention's unwrap in
    # ``experiments/tpu/workarounds.py:_splash_sdpa_tamm`` — if the kernel
    # bridge sees a DTensor, JAX treats inputs and any closure constants as
    # living on a multi-device mesh, and MLIR lowering triggers cross-host
    # gathers that hang at multi-host scale.
    si_is_dt = isinstance(segmented_input, _DTensor)
    si_mesh = si_placements = None
    if si_is_dt:
      si_mesh, si_placements = (
          segmented_input.device_mesh,
          segmented_input.placements,
      )
      segmented_input = segmented_input.to_local()
    if isinstance(weight, _DTensor):
      weight = weight.to_local()
    if isinstance(group_sizes, _DTensor):
      group_sizes = group_sizes.to_local()

    tracks = segmented_input.shape[0]
    track_results = []
    for track in range(tracks):
      out = _kernel.segment_matmul_pallas_2d(
          segmented_input[track],   # (total_tokens, hidden_dim)
          weight[track],            # (num_experts, hidden_dim, output_dim)
          group_sizes[track],       # (num_experts,) — int32 TENSOR
      )
      track_results.append(out)
    result = torch.stack(track_results, dim=0)
    if si_is_dt:
      result = _DTensor.from_local(result, si_mesh, si_placements)
    return result

  _tamm_seg.segment_matmul = _patched_segment_matmul
  for alias in ("segment_matmul", "segment_matmul_torch"):
    if hasattr(_tamm_seg_interface, alias):
      setattr(_tamm_seg_interface, alias, _patched_segment_matmul)
  _SEGMENT_MATMUL_PATCH_APPLIED = True
  logger.info(
      "Patched tamm.segment_matmul with tokamax-backed Pallas ragged_dot"
      " (tensor group_sizes; multi-host-safe replacement for the"
      " .tolist()-based stock implementation)."
  )
