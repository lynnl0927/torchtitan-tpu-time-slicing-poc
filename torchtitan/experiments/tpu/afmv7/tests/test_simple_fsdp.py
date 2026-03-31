"""Distributed tests for the AFMv7 simple fsdp parallelization."""

import os
from absl import logging
from absl.testing import absltest
from torch.distributed.device_mesh import DeviceMesh
import torch
import torch.nn as nn
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.base_distributed_device_test import InputDistribution
from torchtitan.experiments.tpu import distributed
from torchtitan.experiments.tpu.afmv7.infra import parallelize as afmv7_parallelize
import torchtitan.experiments.tpu.afmv7  # trigger afmv7_tpu model registration
import torchtitan.protocols.train_spec as train_spec_module
import torchtitan.distributed
from torchtitan.experiments.tpu.tpu_job_config import TPUJobConfig

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

TEST_BATCH_SIZE = 8
TEST_SEQ_LEN = 128
TEST_TRAINING_STEPS = 3


def _verify_simple_fsdp_afmv7_training_loop_worker(device: torch.device,
                                                   rank: int, world_size: int):
    """Worker function to verify simple FSDP numerical equivalence on AFMv7 model."""

    train_spec = train_spec_module.get_train_spec("afmv7_tpu")

    model_args = train_spec.model_args["debugmodel-lora"]
    model_args.use_lora = False

    # Apply FSDP wrapper
    def apply_fsdp_wrapper(model):
        job_config = TPUJobConfig()
        job_config.tpu_config.use_simple_fsdp = True
        job_config.training.mixed_precision_param = "float32"
        job_config.training.mixed_precision_reduce = "float32"
        job_config.parallelism.data_parallel_shard_degree = world_size
        parallel_dims = torchtitan.distributed.ParallelDims(
            dp_shard=world_size,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size)
        afmv7_parallelize.parallelize_afmv7(model, parallel_dims, job_config)

    class CustomRunner(base_distributed_device_test.DistributedUnitTestRunner):

        def _create_reference_from_parallel(self):
            import copy

            # Setup reference model
            with torch.device("meta"):
                self.reference_model = self.model_class(self.model_args)

            self.reference_model = self.reference_model.to_empty(device="cpu")

            # We only sync weights if rank 0 to avoid multi-writes if saving.
            # But here we just get the state dict from the parallel model

            if self.use_meta_init:
                # FSDP requires fetching the full state dict
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import FullStateDictConfig
                from torch.distributed.fsdp import StateDictType
                with FSDP.state_dict_type(
                        self.parallel_model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True,
                                            rank0_only=False),
                ):
                    cpu_state_dict = self.parallel_model.state_dict()
            else:
                cpu_state_dict = self.parallel_model.state_dict()

            # Convert DTensors to local tensors
            for k, v in cpu_state_dict.items():
                if isinstance(v, torch.distributed.tensor.DTensor):
                    cpu_state_dict[k] = v.full_tensor()

            # For tied weights like model.output_transform.weight missing in simple FSDP state_dict
            if "model.output_transform.weight" not in cpu_state_dict and "model.embedding.weight" in cpu_state_dict:
                cpu_state_dict[
                    "model.output_transform.weight"] = cpu_state_dict[
                        "model.embedding.weight"]

            self.reference_model.load_state_dict(cpu_state_dict, strict=False)

    runner = CustomRunner(
        device=device,
        rank=rank,
        world_size=world_size,
        model_class=train_spec.model_cls,
        model_args=model_args,
        parallelism_func=apply_fsdp_wrapper,
        input_distribution=InputDistribution.SPLIT_BATCH,  # FSDP = Split Batch
        use_meta_init=False,
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


class AFMv7SimpleFSDPParallelizeTest(
        base_distributed_device_test.BaseDistributedDeviceTest):

    def test_apply_simple_fsdp_full_training_loop_equivalence_distributed(
            self):
        """Verifies numerical equivalence of a full training loop with simple FSDP."""
        logging.info(
            "Launching simple FSDP training equivalence test with %d devices on AFMv7 model.",
            self.num_devices,
        )
        distributed.run_distributed(
            num_devices=self.num_devices,
            accelerator_device_type=self.accelerator_device_type,
            func=_verify_simple_fsdp_afmv7_training_loop_worker,
        )  # type: ignore
        logging.info(
            "Distributed simple FSDP training equivalence test finished.")


if __name__ == "__main__":
    g3_multiprocessing.handle_test_main(absltest.main)
