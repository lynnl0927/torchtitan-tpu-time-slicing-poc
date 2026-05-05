"""Tests for kernels on AFM PT MoE (Splash Attention).

* ``test_splash_attention_parity`` — model-level forward+backward parity on
  the AFM PT MoE debugmodel, splash patch applied.
* ``test_splash_sdpa_full_causal`` — kernel-level: ``splash_sdpa`` with
  ``local_window_size=None`` matches eager F.SDPA causal (regression check
  that the old path is unchanged).
* ``test_splash_sdpa_local_window`` — kernel-level: ``splash_sdpa`` with
  ``local_window_size=W`` matches eager F.SDPA driven by an explicit
  windowed-causal ``attn_mask`` (verifies the new ``LocalMask`` wiring).
"""

from absl import logging
from absl.testing import absltest
import torch
import torch.nn.functional as F
from torch.nn import attention

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
import torchtitan.experiments.tpu.afm_pt_moe  # trigger train_spec registration
import torchtitan.protocols.train_spec as train_spec_module


# Build a (s, s) causal sliding-window additive bias whose semantics match
# splash_attention_mask.LocalMask(window_size=(window, 0), offset=0):
# q at position i attends to k at positions in [max(0, i-window), i]
# (i.e., self plus up to ``window`` prior tokens — total window+1 keys).
def _causal_local_window_bias(seq_len: int, window: int, dtype, device):
  i = torch.arange(seq_len, device=device)[:, None]
  j = torch.arange(seq_len, device=device)[None, :]
  delta = i - j  # >0 below diagonal (past), <0 above (future)
  allowed = (delta >= 0) & (delta <= window)
  bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
  bias.masked_fill_(~allowed, float("-inf"))
  return bias


class SplashAttentionTest(base_device_test.BaseAcceleratorDeviceTest):
  """Splash Attention tests on AFM PT MoE model + kernel-level checks."""

  def _get_dummy_model(self):
    train_spec = train_spec_module.get_train_spec("afm_pt_moe_tpu")
    model_args = train_spec.model_args["debugmodel"]
    model_args.use_lora = False
    model = train_spec.model_cls(model_args).to(self.accelerator_device)
    model.init_weights()
    return model

  def test_splash_attention_parity(self):
    """Forward + backward parity on AFM PT MoE debugmodel.

    Note on tolerances: this test runs the *whole model* in fp32 (default
    dtype, no mp_policy) — yet splash on TPU drives the MXU at bf16
    regardless of input dtype (the v6e MXU's native input is bf16). So
    the reference (MATH backend, fp32 matmul) and the test (splash,
    bf16-MXU matmul) differ at bf16-precision level even though both
    inputs and weights are fp32. We therefore use ratio-based tolerances
    on grads (mean_abs / ref_max < 0.05) rather than absolute element-wise
    tolerances — element-wise atol=5e-2 is correct for non-MoE models
    (afmv7) but too tight here because PT MoE's router softmax routing is
    sensitive to the small bf16-MXU matmul drift.
    """
    device = self.accelerator_device
    # Be explicit that this is a fp32 parity test, even though it's the
    # default. If a future change adds an autocast or sets bf16 default
    # somewhere in this file, we want to know.
    torch.set_default_dtype(torch.float32)
    wrapper = self._get_dummy_model()
    model = wrapper.model  # bare TAMM module

    b, s = 2, 128
    tokens = torch.randint(0, 2048, (b, s), device=device)

    model.zero_grad()
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = model(tokens)
      logits_ref = out_ref.predictions
    grad_out = torch.randn_like(logits_ref)
    logits_ref.backward(grad_out)
    ref_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    model.zero_grad()
    workarounds.use_splash_attention_patch(model)

    out_test = model(tokens)
    logits_test = out_test.predictions

    # Forward parity: ratio-based tolerance. mean(|test - ref|) /
    # max(|ref|) < 1% — captures distributional parity without being
    # skewed by routing-induced per-token outliers.
    diff = (logits_test.float() - logits_ref.float()).abs()
    ref_max = logits_ref.float().abs().max().item() + 1e-6
    rel_mean = diff.mean().item() / ref_max
    logging.info(
        "Splash forward: mean_abs=%.4f ref_max=%.2f rel_mean=%.4f%%",
        diff.mean().item(), ref_max, 100 * rel_mean,
    )
    self.assertLess(
        rel_mean, 0.01,
        f"Splash forward rel mean diff {rel_mean:.4%} > 1%",
    )
    logging.info("Splash Attention forward parity passed (AFM PT MoE).")
    logits_test.backward(grad_out)

    # Backward parity: same ratio-based check per parameter. We allow up
    # to 5% mean-abs / ref-max drift per parameter — broad enough to
    # tolerate MoE router amplification of bf16-MXU matmul drift, tight
    # enough to flag genuine regressions (e.g. mask wired wrong, dtype
    # cast missing, etc.).
    failures = []
    grad_summaries = []
    for name, p in model.named_parameters():
      if name in ref_grads:
        self.assertIsNotNone(
            p.grad, f"Gradient for {name} is None in patched run"
        )
        gdiff = (p.grad.float() - ref_grads[name].float()).abs()
        gmean = gdiff.mean().item()
        gscale = ref_grads[name].abs().max().item() + 1e-6
        rel = gmean / gscale
        grad_summaries.append((name, gmean, gscale, rel))
        if rel > 0.05:
          failures.append(
              f"  {name}: rel={100*rel:.2f}% (mean_abs={gmean:.4f} "
              f"ref_max={gscale:.4f})"
          )
    # Log the worst 5 to make tightening the bound debuggable.
    grad_summaries.sort(key=lambda x: -x[3])
    for name, gmean, gscale, rel in grad_summaries[:5]:
      logging.info(
          "Splash bwd worst: %s rel=%.3f%% mean_abs=%.4f ref_max=%.4f",
          name, 100 * rel, gmean, gscale,
      )
    self.assertEqual(
        failures, [],
        "Splash backward parity exceeded 5% rel drift for:\n"
        + "\n".join(failures),
    )
    logging.info("Splash Attention backward parity passed (AFM PT MoE).")

  def test_splash_sdpa_full_causal(self):
    """splash_sdpa(local_window_size=None) ≈ F.SDPA causal.

    Regression: the old non-windowed code path (CausalMask) must remain
    unchanged after the local_window_size addition. This tests the kernel
    directly so any drift caused by the new LocalMask branch surfaces here.
    """
    device = self.accelerator_device
    b, h, s, d = 1, 4, 128, 128
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    ref_q, ref_k, ref_v = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = None
    k.grad = None
    v.grad = None

    out_test = splash_sdpa(q, k, v, is_causal=True, local_window_size=None)
    torch.testing.assert_close(
        out_test.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info("splash_sdpa full-causal forward parity passed.")

    out_test.backward(g)
    torch.testing.assert_close(q.grad.cpu(), ref_q.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(k.grad.cpu(), ref_k.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v.grad.cpu(), ref_v.cpu(), rtol=5e-2, atol=5e-2)
    logging.info("splash_sdpa full-causal backward parity passed.")

  def test_splash_sdpa_local_window(self):
    """splash_sdpa(local_window_size=W) ≈ F.SDPA with windowed-causal attn_mask.

    Verifies the new LocalMask wiring produces the same output and gradients
    as eager attention with a hand-built sliding-window bias. Window chosen
    well below seq_len so the masked region is non-trivial.
    """
    device = self.accelerator_device
    b, h, s, d = 1, 4, 128, 128
    window = 16
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    bias = _causal_local_window_bias(s, window, dtype=q.dtype, device=device)
    with attention.sdpa_kernel([attention.SDPBackend.MATH]):
      out_ref = F.scaled_dot_product_attention(
          q, k, v, attn_mask=bias, is_causal=False
      )
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    ref_q, ref_k, ref_v = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = None
    k.grad = None
    v.grad = None

    out_test = splash_sdpa(
        q, k, v, is_causal=True, local_window_size=window
    )
    torch.testing.assert_close(
        out_test.cpu(), out_ref.cpu(), rtol=5e-2, atol=5e-2
    )
    logging.info(
        "splash_sdpa local-window forward parity passed (W=%d, S=%d).",
        window, s,
    )

    out_test.backward(g)
    torch.testing.assert_close(q.grad.cpu(), ref_q.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(k.grad.cpu(), ref_k.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v.grad.cpu(), ref_v.cpu(), rtol=5e-2, atol=5e-2)
    logging.info("splash_sdpa local-window backward parity passed.")


if __name__ == "__main__":
  absltest.main()
