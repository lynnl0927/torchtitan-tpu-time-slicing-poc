"""AFMTextV7 model wrapper for TorchTitan."""

import enum

import torch
import torch.nn as nn

from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger

from .args import AFMTextV7ModelArgs


_LORA_DTYPE_PATCH_APPLIED = False


def _patch_tamm_lora_for_mixed_precision() -> None:
    """Make LoRA tolerate dtype mismatches between weights and LoRA dtype.

    When base weights are in a different dtype than LoRA weights, the matmul in
    `_transform_outputs_impl` will fail due to the dtype mismatch.
    This happens when using different dtypes for base weights and LoRA weights,
    and when using FSDP Mixed Precision that's only applied to base weights.

    Fix: cast x to LoRA weight dtype for the matmul, then cast the result
    back to the surrounding activation dtype. Same-dtype `.to()` is a no-op.
    """
    global _LORA_DTYPE_PATCH_APPLIED
    if _LORA_DTYPE_PATCH_APPLIED:
        return

    from tamm._adapters_v1.layer_adapters import lora as _lora_mod

    def _transform_outputs_impl(self, x, outputs):
        x = x.flatten(end_dim=-2)
        x = x.to(self.a_transpose.dtype)
        x = torch.matmul(x, self.a_transpose)
        batch_shape = outputs.shape[:-1]
        outputs_f = outputs.flatten(end_dim=-2)
        result = torch.addmm(
            outputs_f.to(x.dtype), x, self.b_transpose, alpha=self.scale
        )
        return result.to(outputs.dtype).reshape(*batch_shape, -1)

    _lora_mod.LoRA._transform_outputs_impl = _transform_outputs_impl
    _LORA_DTYPE_PATCH_APPLIED = True
    logger.info(
        "Patched tamm.LoRA._transform_outputs_impl for mixed-precision safety"
    )


class OutputMode(enum.Enum):
    """Controls what AFMTextV7Wrapper.forward returns."""
    LOGITS = "logits"             # normal path: output.predictions
    HIDDEN = "hidden"             # chunked CE: (hidden, None)
    HIDDEN_AND_WEIGHT = "hidden_and_weight"  # Pallas: (hidden, weight.t())


class AFMTextV7Wrapper(ModelProtocol):
    """Wraps TAMM's AFMTextV7 as a TorchTitan ModelProtocol.

    The wrapper handles:
    - Meta-device construction (TAMM models support torch.device("meta"))
    - Optional LoRA adapter injection (use_lora=True in model args)
    - Weight initialization via reset_parameters() after FSDP materializtion
    - Forward pass that returns logits compatible with cross-entropy loss
    """

    def __init__(self, model_args: AFMTextV7ModelArgs) -> None:
        super().__init__(model_args)
        self._model_args = model_args
        # Controls the forward return value.  Set by train_minimal/parallelize:
        #   "logits"           — normal path, returns output.predictions
        #   "hidden"           — chunked CE path, returns (hidden, None);
        #                        caller gathers output_transform weight.
        #   "hidden_and_weight"— Pallas path, returns (hidden, weight.t());
        #                        FSDP keeps the inner model gathered after
        #                        forward.
        self._output_mode: OutputMode = OutputMode.LOGITS

        import tamm.models.afm_text

        adapters = None
        if model_args.use_lora:
            import tamm.adapters
            _patch_tamm_lora_for_mixed_precision()
            lora_dtype = getattr(torch, model_args.lora_dtype)
            adapters = {
                "lora": tamm.adapters.LoRAModelAdapter(
                    rank=model_args.lora_rank,
                    alpha=float(model_args.lora_alpha),
                    dtype=lora_dtype,
                    adapt_attention_queries=True,
                    adapt_attention_keys=True,
                    adapt_attention_values=True,
                    adapt_attention_outputs=True,
                    adapt_feed_forward_hidden_states=True,
                    adapt_feed_forward_outputs=True,
                )
            }
            logger.info(
                f"LoRA enabled: rank={model_args.lora_rank}, "
                f"alpha={model_args.lora_alpha}, dtype={model_args.lora_dtype}"
            )

        cfg = tamm.models.afm_text.AFMTextV7.Config(
            vocab_size=model_args.vocab_size,
            hidden_dim=model_args.hidden_dim,
            num_layers=model_args.num_layers,
            num_kv_reuse_layers=model_args.num_kv_reuse_layers,
            num_heads=model_args.num_heads,
            num_kv_heads=model_args.num_kv_heads,
            hidden_dim_scale_factor=model_args.hidden_dim_scale_factor,
            rope_theta=model_args.rope_theta,
            pretrained=False,
            adapters=adapters,
        )
        # Store config for use in init_weights().
        self._tamm_cfg = cfg
        # Create the inner model. When called inside torch.device("meta"),
        # all parameters are placed on the meta device.
        self.model: nn.Module = cfg.create_model()

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor | tuple:
        """Run the model forward pass.

        Return value depends on _output_mode:
          "logits"            — returns output.predictions (B, S, vocab_size)
          "hidden"            — returns (hidden, None); caller gathers
                                output_transform weight before the chunk loop.
          "hidden_and_weight" — returns (hidden, weight.t()); FSDP keeps the
                                inner model gathered so weight.t() stays valid
                                for pallas_cross_entropy_loss.
        """
        if self._output_mode == OutputMode.HIDDEN_AND_WEIGHT:
            output = self.model(tokens, mode="SKIP_OUTPUT_LAYER")
            hidden = output.last_hidden_state  # (B, S, H)
            weight = self.model.output_transform.weight  # (vocab_size, H)
            return hidden, weight.t()  # (H, vocab_size)
        if self._output_mode == OutputMode.HIDDEN:
            output = self.model(tokens, mode="SKIP_OUTPUT_LAYER")
            return output.last_hidden_state, None
        output = self.model(tokens)
        return output.predictions

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize weights after model.to_empty(device) materializes storage.

        Uses simple uniform init rather than TAMM's reset_parameters() because
        TAMM computes fan_in without accounting for FSDP parameter sharding,
        which causes activation explosion and NaN loss when training with FSDP.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                local_tensor = param.to_local() if hasattr(param, "to_local") else param
                if "lora" in name:
                    if "lora_B" in name:
                        local_tensor.zero_()
                    else:
                        local_tensor.uniform_(-0.01, 0.01)
                else:
                    local_tensor.uniform_(-0.01, 0.01)
            for buffer in self.model.buffers():
                buffer.fill_(0)
