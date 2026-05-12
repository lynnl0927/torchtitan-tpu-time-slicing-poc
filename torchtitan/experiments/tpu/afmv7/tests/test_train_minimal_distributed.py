"""Tests for the minimal distributed training setup for AFMv7."""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.afmv7 import train_minimal as afmv7_train_minimal




class TrainMinimalDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests train_minimal execution in distributed settings for AFMv7."""

  @parameterized.named_parameters([
      dict(
          testcase_name="afmv7_ddp",
          _model_name="afmv7_tpu",
          _config_file="torchtitan/experiments/tpu/afmv7/train_configs/debug_model.toml",
          data_parallel_shard_degree=1,
          data_parallel_replicate_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
      dict(
          testcase_name="afmv7_fsdp",
          _model_name="afmv7_tpu",
          _config_file="torchtitan/experiments/tpu/afmv7/train_configs/debug_model.toml",
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          skip_devices=None,
      ),
  ])
  def test_afmv7_train_minimal(
      self,
      _model_name,
      _config_file,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      data_parallel_replicate_degree=None,
      extra_args=None,
      skip_devices=None,
  ):
    """Runs execution test specifically for AFMv7's minimal trainer (experiments/tpu/afmv7/train_minimal.py)."""

    config_args = [
        "--module=torchtitan.experiments.tpu.afmv7",
        "--config=afmv7_debugmodel",
        "--hf_assets_path=tests/assets/tokenizer",
        "--dataloader.dataset_path=tests/assets/c4_test",
        "--training.seq_len=128",
        "--dataloader.dataset=c4_test",
        "--training.steps=5",
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
        start_trainer=afmv7_train_minimal.start_trainer,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
