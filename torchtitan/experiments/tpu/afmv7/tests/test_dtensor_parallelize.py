"""Distributed tests for the AFMv7 DTensor FSDP parallelization."""

from absl import logging
from absl.testing import absltest
import torch
import torchtitan.distributed
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu import tpu_job_config
import torchtitan.experiments.tpu.afmv7 as afmv7_package
from torchtitan.experiments.tpu.afmv7.infra import parallelize as afmv7_parallelize
from torchtitan.experiments.tpu.afmv7.model import model as afmv7_model



TEST_BATCH_SIZE = 8
TEST_SEQ_LEN = 128
TEST_TRAINING_STEPS = 3


def _verify_fsdp2_afmv7_training_loop_worker(device: torch.device, rank: int,
                                             world_size: int):
  """Worker function to verify FSDP2 numerical equivalence on AFMv7 model."""

  model_args = afmv7_package.afmv7_args["debugmodel-lora"]
  model_args.use_lora = False

  # Apply FSDP wrapper
  def apply_fsdp_wrapper(model):
    train_config = tpu_job_config.TPUTrainerConfig()
    train_config.tpu_config.use_simple_fsdp = False
    train_config.training.mixed_precision_param = "float32"
    train_config.training.mixed_precision_reduce = "float32"
    train_config.parallelism.data_parallel_shard_degree = world_size
    parallel_dims = torchtitan.distributed.ParallelDims(
        dp_shard=world_size,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    afmv7_parallelize.parallelize_afmv7(model, parallel_dims, train_config)

  runner = base_distributed_device_test.DistributedUnitTestRunner(
      device=device,
      rank=rank,
      world_size=world_size,
      model_class=afmv7_model.AFMTextV7Wrapper,
      model_args=model_args,
      parallelism_func=apply_fsdp_wrapper,
      input_distribution=base_distributed_device_test.InputDistribution.SPLIT_BATCH,  # FSDP = Split Batch
      use_meta_init=True,
  )

  # Verify loop (checks gradients on multiple random data batches)
  runner.run_backward_parity(
        num_steps=TEST_TRAINING_STEPS,
        batch_size=TEST_BATCH_SIZE,
        seq_len=TEST_SEQ_LEN,
        atol_loss=1,
        rtol_loss=1,
        atol_grad=9e-1,
        rtol_grad=9e-1,
    )


class AFMv7DTensorParallelizeTest(
        base_distributed_device_test.BaseDistributedDeviceTest):

  def test_apply_fsdp_full_training_loop_equivalence_distributed(self):
    """Verifies numerical equivalence of a full training loop with FSDP2."""
    logging.info(
        "Launching FSDP2 training equivalence test with %d devices on AFMv7"
        " model.",
        self.num_devices,
    )
    distributed.run_distributed(
            num_devices=self.num_devices,
            accelerator_device_type=self.accelerator_device_type,
            func=_verify_fsdp2_afmv7_training_loop_worker,
        )  # type: ignore
    logging.info("Distributed FSDP2 training equivalence test finished.")


if __name__ == "__main__":
    absltest.main()
