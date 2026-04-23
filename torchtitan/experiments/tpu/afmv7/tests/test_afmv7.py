"""Tests for AFMv7 model correctness on CPU and accelerator devices.

This file contains tests to ensure parity between CPU and accelerator device
implementations of AFMv7 model layers and full model testing.
"""

from absl.testing import absltest
import torch
from torch import nn
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu.afmv7.model.args import AFMTextV7ModelArgs
from torchtitan.experiments.tpu.afmv7.model.model import AFMTextV7Wrapper
from tamm.layers.activation import SwiGLU
from tamm.layers.attention import KVReuseTransformerAttention, TransformerAttention
from tamm.layers.feed_forward import TransformerFeedForward
from tamm.layers.norm import RMSNorm
from tamm.layers.transformer.layer import TransformerLayer


class AFMv7Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for AFMv7 model components on CPU and accelerator devices.

  This class contains tests to ensure parity between CPU and accelerator device
  implementations of AFMv7 model layers and full model testing.
  """

  def _get_model_args(
      self,
      num_layers: int = 2,
      vocab_size: int = 32,
      hidden_dim: int = 64,
      num_heads: int = 4,
      num_kv_heads: int = 2,
      num_kv_reuse_layers: int = 1,
  ) -> AFMTextV7ModelArgs:
    """Returns model arguments for AFMv7 model."""
    return AFMTextV7ModelArgs(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        num_kv_reuse_layers=num_kv_reuse_layers,
    )

  def test_afmv7_embedding_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Embedding layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    embedding_cpu, embedding_device = self._setup_module_cpu_device(
        nn.Embedding,
        init_fn=lambda m: nn.init.normal_(m.weight),
        num_embeddings=args.vocab_size,
        embedding_dim=args.hidden_dim
    )

    tokens_cpu = torch.randint(
        0, args.vocab_size, (batch, seq_len), device="cpu",
        requires_grad=False
    )
    tokens_device = tokens_cpu.to(self.accelerator_device)

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = embedding_cpu(tokens_cpu)
      out_device = embedding_device(tokens_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=1e-8,
          rtol=1e-8,
          check_name="Embedding Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        out_cpu.requires_grad_(True)
        out_device.requires_grad_(True)
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            embedding_device.weight.grad.cpu(),
            embedding_cpu.weight.grad,
            atol=1e-9,
            rtol=1e-9,
            check_name="Embedding Weight Grad"
        )

  def test_afmv7_feedforward_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the FeedForward layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    feedforward_cpu, feedforward_device = self._setup_module_cpu_device(
        lambda: TransformerFeedForward.create_basic_builder(
            input_dim=args.hidden_dim,
            hidden_dim=round(args.hidden_dim * args.hidden_dim_scale_factor),
            norm=RMSNorm.Builder([args.hidden_dim]),
            activation=SwiGLU.Builder(),
        ).build(),
        init_fn=lambda m: None,
    )

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.hidden_dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = feedforward_cpu(x_cpu)
      out_device = feedforward_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=2e-2,
          rtol=2e-2,
          check_name="FFN Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-2,
            rtol=5e-2,
            check_name="FFN Input Grad",
        )
        for (name, p_cpu), p_dev in zip(
            feedforward_cpu.named_parameters(), feedforward_device.parameters()
        ):
          test_utils.check_equivalence(
              p_dev.grad.cpu(),
              p_cpu.grad,
              atol=2.5e-1,
              rtol=2.5e-1,
              check_name=f"FFN Param Grad: {name}",
          )

  def test_afmv7_attention_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the Attention layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_cpu, attention_device = self._setup_module_cpu_device(
        lambda: TransformerAttention.create_basic_builder(
            target_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            norm=RMSNorm.Builder([args.hidden_dim]),
            apply_rope=False,
            apply_qk_norm=True,
        ).build(),
        init_fn=lambda m: None,
    )

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.hidden_dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = attention_cpu(x_cpu)
      out_device = attention_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=2e-2,
          rtol=2e-2,
          check_name="Attention Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-2,
            rtol=5e-2,
            check_name="Attention Input Grad",
        )
        for (name, p_cpu), p_dev in zip(
            attention_cpu.named_parameters(), attention_device.parameters()
        ):
          test_utils.check_equivalence(
              p_dev.grad.cpu(),
              p_cpu.grad,
              atol=2.5e-1,
              rtol=2.5e-1,
              check_name=f"Attention Param Grad: {name}",
          )

  def test_afmv7_transformer_block_cpu_device_parity(self):
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer."""
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # CPU and DEVICE setup
    attention_builder = TransformerAttention.create_basic_builder(
        target_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        norm=RMSNorm.Builder([args.hidden_dim]),
        apply_rope=False,
        apply_qk_norm=True,
    )

    feedforward_builder = TransformerFeedForward.create_basic_builder(
        input_dim=args.hidden_dim,
        hidden_dim=round(args.hidden_dim * args.hidden_dim_scale_factor),
        norm=RMSNorm.Builder([args.hidden_dim]),
        activation=SwiGLU.Builder(),
    )

    transformer_block_cpu, transformer_block_device = (
        self._setup_module_cpu_device(
            lambda: TransformerLayer(
                attention=attention_builder.build(),
                feed_forward=feedforward_builder.build(),
            ),
            init_fn=lambda m: None,
        )
    )

    x_cpu, x_device = self._create_input_tensor_cpu_device(
        (batch, seq_len, args.hidden_dim), requires_grad=True
    )

    with self.subTest(name="ForwardPass"):
      # Forward pass
      out_cpu = transformer_block_cpu(x_cpu)
      out_device = transformer_block_device(x_device)
      test_utils.check_equivalence(
          out_device.cpu(),
          out_cpu,
          atol=2e-2,
          rtol=2e-2,
          check_name="TransformerBlock Forward",
      )

      with self.subTest(name="BackwardPass"):
        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        test_utils.check_equivalence(
            x_device.grad.cpu(),
            x_cpu.grad,
            atol=5e-2,
            rtol=5e-2,
            check_name="TransformerBlock Input Grad",
        )
        for (name, p_cpu), p_dev in zip(
            transformer_block_cpu.named_parameters(),
            transformer_block_device.parameters(),
        ):
          test_utils.check_equivalence(
              p_dev.grad.cpu(),
              p_cpu.grad,
              atol=2.5e-1,
              rtol=2.5e-1,
              check_name=f"TransformerBlock Param Grad: {name}",
          )

  def test_afmv7_full_model_training_steps_cpu_device_parity(self):
    """Tests CPU vs. DEVICE parity of full AFMv7 model after 3 training steps.

    Uses baseclass function wrapper for full model training steps parity test.
    """
    args = self._get_model_args()
    batch, seq_len = 2, 8

    # We need to use the wrapper to construct the model as it handles
    # the inner model creation via TAMM config.
    model_cpu = AFMTextV7Wrapper(args).cpu()

    # The full model test helper expects the model to have an init_weights method
    # or it will use default init. AFMTextV7Wrapper has init_weights.

    self._run_full_model_training_steps_parity_test(
        model_cpu=model_cpu,
        batch=batch,
        seq_len=seq_len,
        loss_atol=5e-3,
        loss_rtol=1e-2,
        grad_atol=0.1,
        grad_rtol=1e-2,
        param_atol=5e-2,
        param_rtol=1e-2,
    )

if __name__ == "__main__":
  absltest.main()
