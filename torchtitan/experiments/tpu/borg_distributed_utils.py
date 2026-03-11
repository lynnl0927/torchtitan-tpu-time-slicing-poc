"""Utilities for configuring torch_tpu for TAP/Borg distributed tests."""

import os
import portpicker
from torch_tpu._internal.distributed import tpu_topology
from absl import logging


def maybe_init_distributed(num_devices: int) -> bool:
  """Sets up the environment for multi-process TPU tests on Borg.

  Args:
    num_devices: The number of devices to use.

  Returns:
    True if the environment variables were successfully initialized for
    distributed TPU training on TAP/Borg, False otherwise.

  Raises:
    RuntimeError: If a multi-host TPU environment is detected, as only
    single-host
      is currently supported.
    RuntimeError: If the number of detected devices does not match the number of
      requested devices.
  """

  if num_devices <= 1:
    return False

  # Validate that this is a single-host environment.
  # TPU_WORKER_HOSTNAMES contains a comma-separated list of hosts.
  hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
  if hostnames and "," in hostnames:
    raise RuntimeError(
        "borg_distributed_utils detected a multi-host TPU environment"
        f" ({hostnames}), but only single-host is currently supported."
    )

  logging.info(
      f"Initializing distributed environment for {num_devices} TPU devices on"
      " Borg."
  )

  # Coordinate SliceBuilder addresses (inherited by all workers)
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    sb_ports = [portpicker.pick_unused_port() for _ in range(num_devices)]
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in sb_ports]
    )

  # Set Global Topology
  topo, found_devices = tpu_topology.get_tpu_topology()
  if found_devices != num_devices:
    raise RuntimeError(
        f"borg_distributed_utils detected {found_devices} TPU devices, but"
        f" {num_devices} were requested."
    )
  if topo:
    os.environ["TORCH_TPU_TOPOLOGY"] = topo

  return True
