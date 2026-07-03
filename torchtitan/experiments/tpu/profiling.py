"""Utilities for running distributed functions with xprof profiling."""

import getpass
import os
from typing import Any, Callable, Tuple

import torch
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import distributed as torchtitan_distributed

from google3.perftools.accelerators.xprof import xprof_service_pb2
from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
from google3.perftools.accelerators.xprof.api.python import xprof_session
from google3.pyglib import gfile
from google3.third_party.tensorflow.tsl.profiler.protobuf import xplane_pb2


class _FuncWrapperWithXprofUrl:
  """Wraps a function to start a xprof for this worker."""

  def __init__(
      self,
      func: Callable[..., Any],
      profiler_kwargs: dict[str, Any],
  ):
    """Initializes the _FuncWrapper.

    Args:
      func: The callable function to wrap. It should accept device, rank,
        world_size, and other arguments.
      profiler_kwargs: Keyword arguments to pass to
        `xprof_session.XprofSession.start_session`.
    """
    self.func = func
    self.profiler_kwargs = profiler_kwargs

  def __call__(
      self, device: torch.device, rank: int, world_size: int, *args: Any
  ) -> Any:
    result_queue = args[-1]
    args = args[:-1]
    profiler = xprof_session.XprofSession()
    profiler.start_session(**self.profiler_kwargs)

    print(f"Starting xprof session for rank {rank}")
    out = self.func(device, rank, world_size, *args)
    url = profiler.end_session_and_get_url(tag=f"rank_{rank}")
    print(f"Xprof URL for rank {rank}: {url}")
    result_queue.put((rank, (out, url)))


def run_distributed_with_profiling(
    num_devices: int,
    accelerator_device_type: device_type.AcceleratorDeviceType,
    func: Callable[..., Any],
    combine_profiles: bool = True,
    *worker_args: Tuple[Any, ...],
) -> None:
  """Runs a function in a distributed manner with profiling enabled."""

  result_queue = mp.Queue()  # pyrefly: ignore[missing-attribute]
  worker_args = worker_args + (result_queue,)

  torchtitan_distributed.run_distributed(
      num_devices,
      accelerator_device_type,
      _FuncWrapperWithXprofUrl(
          func,
          {
              "enable_python_tracer": True,
              "host_trace_level": 2,
              "host_cpu_profile": True,
          },
      ),
      *worker_args,
  )

  results_with_rank = []
  for _ in range(num_devices):
    results_with_rank.append(result_queue.get())

  # Sort results by rank
  results_with_rank.sort(key=lambda x: x[0])
  results = [result for rank, result in results_with_rank]

  if not results:
    # No results means the function returned None, no need to combine profiles.
    return

  if combine_profiles:
    # Download individual xprof sessions.
    try:
      xspaces = []
      client = xprof_analysis_client.XprofAnalysisClient(getpass.getuser())

      for rank, worker_output in enumerate(results):
        _, url = worker_output
        session_id = url.split("=")[-1]
        _, data = client.get_profile_data("xplane.pb", session_id)
        if data:
          space = xplane_pb2.XSpace()
          space.ParseFromString(data)
          xspaces.append(space)
          print(f"Successfully downloaded {session_id} for rank {rank}")
        else:
          print(f"Failed to download {session_id} for rank {rank}")

    except Exception as e:
      print(f"Downloading individual xprof sessions failed with error: {e}")
      return

    # Combine and upload xprof sessions.
    try:
      session_id = client.upload(xspaces)
      if session_id:
        print(f"Xprof URL: http://xprof/?session_id={session_id}")
      else:
        print("Failed to combine and upload xprof sessions.")
        return
    except Exception as e:
      print(f"Failed to combine and upload xprof sessions with error: {e}")
      return

  else:
    # Print all xprof URLs for each worker.
    for rank, worker_output in enumerate(results):
      _, url = worker_output
      print(f"Xprof URL for rank {rank}: {url}")


def upload_xprof_xplane_pb_files_from_dir(
    directory_path: str, merge: str = "latest"
) -> str | None:
  """Finds all .xplane.pb files in a directory and its subdirectories,

  and uploads them to Xprof.

  Args:
      directory_path (str): The path to the directory to search.
      merge (str): How to merge multiple files. "latest" or "all".

  Returns:
      The session id if upload is successful, None otherwise.
  """
  pb_files = []
  print(f"Searching for .xplane.pb files in: {directory_path}")
  for dirname, _, filenames in gfile.Walk(directory_path):
    for filename in filenames:
      if filename.endswith(".xplane.pb"):
        full_path = os.path.join(dirname, filename)
        pb_files.append(full_path)

  if not pb_files:
    print("No .xplane.pb files found in the directory.")
    return None

  print(f"Found {len(pb_files)} .xplane.pb files:")

  if merge == "latest":
    pb_files = sorted(
        pb_files, key=lambda x: os.path.getmtime(x), reverse=True
    )[:1]
    print(f"Pick the latest one: {pb_files[0]}")
  elif merge == "all":
    for f in pb_files:
      print(f"  - {f}")
    print("... and merge them together")
  else:
    raise ValueError(
        f"Parameter `merge` should be `all` or `latest` but get `{merge}`"
    )

  xprof_responses = []
  for file_path in pb_files:
    try:
      with gfile.Open(file_path, "rb") as f:
        pb_data = f.read()

      response = xprof_service_pb2.XprofResponse()
      try:
        # The .xplane.pb file is expected to contain a serialized XSpace proto.
        response.space.ParseFromString(pb_data)
        print(f"Successfully loaded and parsed: {file_path}")

        # Add hostname to the response if missing
        if not response.hostname:
          response.hostname = response.hostname = file_path.strip("/").replace(
              "/", "_"
          )
          print(f"Added hostname to response: {response.hostname}")

        xprof_responses.append(response)

      except Exception as e:
        print(f"Failed to parse {file_path} as XSpace: {e}. Skipping.")

    except gfile.Error as e:
      print(f"Error accessing file {file_path}: {e}. Skipping.")
    except Exception as e:
      print(f"An error occurred with file {file_path}: {e}. Skipping.")

  if not xprof_responses:
    print("No .xplane.pb files could be loaded as XprofResponses.")
    return None

  try:
    client = xprof_analysis_client.XprofAnalysisClient()
    print(f"Uploading {len(xprof_responses)} profiles to Xprof...")

    session_id = client.upload(xprof_response=xprof_responses)

    if session_id:
      print("Successfully uploaded profiles to Xprof!")
      print(f"View at: http://xprof/?session_id={session_id}")
      return session_id
    else:
      print("Failed to upload to Xprof.")
      print("Check logs for more details.")
      return None

  except Exception as e:
    print(f"An error occurred during upload: {e}")

  return None
