"""AFMPTMoe model wrapper for TorchTitan."""

import enum

import torch
import torch.nn as nn

from torchtitan.protocols.model import ModelProtocol
from torchtitan.tools.logging import logger

from .args import AFMPTMoeModelArgs


class OutputMode(enum.Enum):
    """Controls what AFMPTMoeWrapper.forward returns. Mirrors AFMTextV7Wrapper.

    LOGITS:             normal path — runs full model and returns
                        output.predictions (B, S, vocab_size).
    HIDDEN:             chunked-CE path — returns (hidden, None); the caller
                        gathers output_transform.weight before the chunk loop.
    HIDDEN_AND_WEIGHT:  Pallas linear-softmax-CE path — returns
                        (hidden, weight.t()) so loss.pallas_cross_entropy_loss
                        can do fused linear+CE without materializing the full
                        (B, S, vocab_size) logit tensor. Requires FSDP to keep
                        output_transform.weight gathered through the loss
                        computation (see parallelize.py
                        `keep_output_weight_gathered`).
    """
    LOGITS = "logits"
    HIDDEN = "hidden"
    HIDDEN_AND_WEIGHT = "hidden_and_weight"


# Optional fields on AFMParallelTrackMoEConfig — passed through only when the
# user explicitly set them on AFMPTMoeModelArgs (i.e. value is not None).
_OPTIONAL_TAMM_FIELDS = (
    "attention_layer_pattern",
    "feed_forward_layer_pattern",
    "local_attention_window_size",
    "norm_eps",
    "scale_qk_norm",
    "pre_norm",
    "pre_residual_norm",
    "tracks_combine_norm",
    "tracks_combine_op",
    "tracks_dispatch_norm",
    "sdpa_implementation",
    "experts_router_logits_cap",
)


_LORA_DTYPE_PATCH_APPLIED = False


def _patch_tamm_lora_for_mixed_precision() -> None:
    """Make LoRA tolerate dtype mismatches between weights and LoRA dtype.

    Same patch the afmv7 wrapper applies — the AFM PT MoE model uses the same
    `tamm._adapters_v1.layer_adapters.lora.LoRA` module, so the same fix is
    needed when LoRA params are fp32 but base weights are bf16 under FSDP
    mixed precision.
    """
    global _LORA_DTYPE_PATCH_APPLIED
    if _LORA_DTYPE_PATCH_APPLIED:
        return

    from tamm._adapters_v1.layer_adapters import lora as _lora_mod

    def _transform_outputs_impl(self, x, outputs):
        out_dtype = outputs.dtype
        if self.a_transpose.dtype == out_dtype:
            inner = torch.matmul(x, self.a_transpose)
            batch_shape = outputs.shape[:-1]
            return torch.addmm(
                outputs.flatten(end_dim=-2),
                inner.flatten(end_dim=-2),
                self.b_transpose,
                alpha=self.scale,
            ).reshape(*batch_shape, -1)
        a = self.a_transpose.to(out_dtype)
        b = self.b_transpose.to(out_dtype)
        inner = torch.matmul(x, a)
        batch_shape = outputs.shape[:-1]
        return torch.addmm(
            outputs.flatten(end_dim=-2),
            inner.flatten(end_dim=-2),
            b,
            alpha=self.scale,
        ).reshape(*batch_shape, -1)

    _lora_mod.LoRA._transform_outputs_impl = _transform_outputs_impl
    _LORA_DTYPE_PATCH_APPLIED = True
    logger.info(
        "Patched tamm.LoRA._transform_outputs_impl for mixed-precision safety"
    )


class AFMPTMoeWrapper(ModelProtocol):
    """Wraps TAMM's AFMPTMoe as a TorchTitan ModelProtocol."""

    def __init__(self, model_args: AFMPTMoeModelArgs) -> None:
        super().__init__(model_args)
        self._model_args = model_args
        # Default to LOGITS mode; parallelize.py flips this to HIDDEN_AND_WEIGHT
        # when tpu_config.use_loss_kernel is True. Mirrors AFMTextV7Wrapper.
        self._output_mode: OutputMode = OutputMode.LOGITS

        import tamm.models.afm_text.afm_pt_moe as afm_pt_moe

        extra_kwargs = {
            k: getattr(model_args, k)
            for k in _OPTIONAL_TAMM_FIELDS
            if getattr(model_args, k) is not None
        }

        adapters = None
        if model_args.use_lora:
            import tamm.adapters
            _patch_tamm_lora_for_mixed_precision()
            lora_dtype = getattr(torch, model_args.lora_dtype)
            # Attention-only LoRA. Sparse-FFN (expert) LoRA on AFM PT MoE
            # interacts with the routing layer in ways that aren't validated
            # in this codebase yet; standard practice for MoE fine-tuning
            # (e.g. Mixtral) is to adapt only attention. Flip on
            # adapt_feed_forward_* below if you have a reason to adapt
            # experts.
            adapters = {
                "lora": tamm.adapters.LoRAModelAdapter(
                    rank=model_args.lora_rank,
                    alpha=float(model_args.lora_alpha),
                    dtype=lora_dtype,
                    adapt_attention_queries=True,
                    adapt_attention_keys=True,
                    adapt_attention_values=True,
                    adapt_attention_outputs=True,
                )
            }
            logger.info(
                f"LoRA enabled: rank={model_args.lora_rank}, "
                f"alpha={model_args.lora_alpha}, dtype={model_args.lora_dtype} "
                f"(attention-only on MoE)"
            )

        # pytype: disable=wrong-keyword-args
        cfg = afm_pt_moe.AFMParallelTrackMoEConfig(
            vocab_size=model_args.vocab_size,
            num_tracks=model_args.num_tracks,
            num_layers_per_track=model_args.num_layers_per_track,
            num_layers_per_track_per_sync_point=model_args.num_layers_per_track_per_sync_point,
            hidden_dim=model_args.hidden_dim,
            attention_hidden_dim=model_args.attention_hidden_dim,
            dense_feed_forward_hidden_dim=model_args.dense_feed_forward_hidden_dim,
            sparse_feed_forward_hidden_dim=model_args.sparse_feed_forward_hidden_dim,
            num_heads=model_args.num_heads,
            num_kv_heads=model_args.num_kv_heads,
            rope_theta=model_args.rope_theta,
            num_experts=model_args.num_experts,
            num_experts_per_token=model_args.num_experts_per_token,
            adapters=adapters,
            **extra_kwargs,
        )
        # pytype: enable=wrong-keyword-args
        self._tamm_cfg = cfg
        self.model: nn.Module = cfg.create_basic_builder().build()

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor | tuple:
        """Run the model forward pass.

        Return value depends on _output_mode:
          LOGITS            — returns output.predictions (B, S, vocab_size)
          HIDDEN            — returns (hidden, None); caller gathers
                              output_transform weight before the chunk loop.
          HIDDEN_AND_WEIGHT — returns (hidden, weight.t()); FSDP must keep the
                              inner model gathered (reshard_after_forward=False)
                              so weight.t() stays valid for
                              pallas_cross_entropy_loss.
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
        """Initialize weights after model.to_empty(device) materializes storage."""
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
