"""Tests for kernels on AFMv7 model (Splash Attention and Loss)."""

from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
from torch.nn import attention

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss as loss_kernel
import torchtitan.experiments.tpu.afmv7 as afmv7_package
from torchtitan.experiments.tpu.afmv7.model import model as afmv7_model


class KernelNumericsTest(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for custom kernels on the AFMv7 model."""

  def _get_dummy_model(self):
    model_args = afmv7_package.afmv7_args["debugmodel"]
    # Ensure LoRA is disabled for standard testing
    model_args.use_lora = False
    model = afmv7_model.AFMTextV7Wrapper(model_args).to(self.accelerator_device)
    # Materialize weights
    model.init_weights()
    return model

  def test_splash_attention_parity(self):
    """Verifies Splash Attention forward and backward parity on AFMv7 model."""
    device = self.accelerator_device
    model_wrapper = self._get_dummy_model()
    model = model_wrapper.model # Get inner nn.Module

    b, s = 2, 128
    tokens = torch.randint(0, 2048, (b, s), device=device)

    # 1. Reference run (Native Torch Attention)
    # We need to zero out grads as we might reuse the model
    model.zero_grad()

    # We use a dummy hook or just run forward.
    # To ensure it uses MATH backend for reference:
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = model(tokens)
      logits_ref = out_ref.predictions

    grad_out = torch.randn_like(logits_ref)
    logits_ref.backward(grad_out)

    # Save reference gradients
    ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    # 2. Patched run (Splash Attention)
    model.zero_grad()
    workarounds.use_splash_attention_patch(model)

    out_test = model(tokens)
    logits_test = out_test.predictions

    # Verify forward parity
    torch.testing.assert_close(
        logits_test.cpu(), logits_ref.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("Splash Attention Forward parity passed.")

    logits_test.backward(grad_out)

    # Verify backward parity
    for name, p in model.named_parameters():
      if name in ref_grads:
        self.assertIsNotNone(p.grad, f"Gradient for {name} is None in patched run")
        torch.testing.assert_close(
            p.grad.cpu(), ref_grads[name].cpu(), rtol=5e-2, atol=5e-2
        )
    logging.info("Splash Attention Backward parity passed.")

  def test_tokamax_splash_attention_parity(self):
    """Verifies Tokamax Splash Attention forward and backward parity on AFMv7 model."""
    device = self.accelerator_device
    model_wrapper = self._get_dummy_model()
    model = model_wrapper.model # Get inner nn.Module

    b, s = 2, 128
    tokens = torch.randint(0, 2048, (b, s), device=device)

    # 1. Reference run (Native Torch Attention)
    model.zero_grad()

    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = model(tokens)
      logits_ref = out_ref.predictions

    grad_out = torch.randn_like(logits_ref)
    logits_ref.backward(grad_out)

    # Save reference gradients
    ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    # 2. Patched run (Tokamax Splash Attention)
    model.zero_grad()
    class MockConfig:
      pass
    workarounds.use_tokamax_splash_attention_patch(model, MockConfig())

    out_test = model(tokens)
    logits_test = out_test.predictions

    # Verify forward parity
    torch.testing.assert_close(
        logits_test.cpu(), logits_ref.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("Tokamax Splash Attention Forward parity passed.")

    logits_test.backward(grad_out)

    # Verify backward parity
    for name, p in model.named_parameters():
      if name in ref_grads:
        self.assertIsNotNone(p.grad, f"Gradient for {name} is None in patched run")
        torch.testing.assert_close(
            p.grad.cpu(), ref_grads[name].cpu(), rtol=5e-2, atol=5e-2
        )
    logging.info("Tokamax Splash Attention Backward parity passed.")


  def test_loss_numerics_parity(self):
    """Verifies Linear Softmax Cross Entropy Loss forward and backward parity on AFMv7 model."""
    device = self.accelerator_device
    model_wrapper = self._get_dummy_model()

    b, s = 2, 128
    tokens = torch.randint(0, 2048, (b, s), device=device)
    # Labels for loss (usually shifted tokens, but random is fine for parity)
    labels = torch.randint(0, 2048, (b * s,), device=device)

    # 1. Reference run (Linear + CrossEntropy separate)
    # We use HIDDEN_AND_WEIGHT mode to get hidden states and weights
    from torchtitan.experiments.tpu.afmv7.model.model import OutputMode
    model_wrapper._output_mode = OutputMode.HIDDEN_AND_WEIGHT

    model_wrapper.zero_grad()
    hidden, weight = model_wrapper(tokens)
    # hidden is (B, S, H), weight is (H, V)

    h_dim = hidden.shape[-1]
    hidden_flat = hidden.view(-1, h_dim)

    # Reference loss: hidden @ weight -> logits -> CE
    logits_ref = hidden_flat @ weight
    loss_ref = F.cross_entropy(logits_ref, labels, reduction="mean")

    loss_ref.backward()

    ref_grads = {name: p.grad.clone() for name, p in model_wrapper.model.named_parameters() if p.grad is not None}

    # 2. Custom run (Pallas Loss)
    model_wrapper.zero_grad()
    # hidden, weight are the same since weights didn't change and we don't
    # run optimizer.
    # We can just reuse hidden and weight from before or run forward again.

    hidden_test, weight_test = model_wrapper(tokens)
    hidden_flat_test = hidden_test.view(-1, h_dim)

    loss_test = loss_kernel.linear_softmax_cross_entropy_loss(
        hidden_flat_test,
        labels,
        weight_test,
        reduction="mean",
        implementation="mosaic_tpu", # or xla
        b_block_size=256,
    )

    # Verify forward parity
    torch.testing.assert_close(
        loss_test.cpu(), loss_ref.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("Loss Forward parity passed.")

    loss_test.backward()

    # Verify backward parity for model parameters
    for name, p in model_wrapper.model.named_parameters():
      if name in ref_grads:
        self.assertIsNotNone(
            p.grad, f"Gradient for {name} is None in custom run"
        )
        torch.testing.assert_close(
            p.grad.cpu(), ref_grads[name].cpu(), rtol=5e-2, atol=5e-2
        )


if __name__ == "__main__":
  absltest.main()
