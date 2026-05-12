"""Tests for the minimal distributed training setup for AFMv7."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.afm_pt_moe import train_minimal as afm_pt_moe_train_minimal




class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests train_minimal execution in distributed settings for AFMv7."""

  @parameterized.named_parameters([
      dict(
          testcase_name="afm_pt_moe_ddp",
          data_parallel_shard_degree=1,
          data_parallel_replicate_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
      dict(
          testcase_name="afm_pt_moe_fsdp",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
      dict(
          testcase_name="afm_pt_moe_fsdp_segment_matmul_kernel",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          extra_args=["--afm_pt_moe.use_segment_matmul_kernel"],
      ),
  ])
  def test_afm_pt_moe_train_minimal(
      self,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      extra_args=None,
      skip_devices=None,
  ):
    """Runs execution test specifically for AFM PT MoE's minimal trainer (experiments/tpu/afm_pt_moe/train_minimal.py)."""

    config_args = [
        "--module=torchtitan.experiments.tpu.afm_pt_moe",
        "--config=afm_pt_moe_debugmodel",
        "--hf_assets_path=tests/assets/tokenizer",
        "--dataloader.dataset_path=tests/assets/c4_test",
        "--training.seq_len=128",
        "--dataloader.dataset=c4_test",
        "--training.steps=3",
        "--training.local_batch_size=4",
    ]
    if extra_args:
      config_args.extend(extra_args)

    self._test_train_distributed(
        config_args=config_args,
        tensor_parallel_degree=tensor_parallel_degree,
        data_parallel_shard_degree=data_parallel_shard_degree,
        data_parallel_replicate_degree=data_parallel_replicate_degree,
        skip_devices=skip_devices,
        start_trainer=afm_pt_moe_train_minimal.start_trainer,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
