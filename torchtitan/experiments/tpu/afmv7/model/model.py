"""AFMTextV7 model wrapper for TorchTitan."""

import torch
import torch.nn as nn

from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger

from .args import AFMTextV7ModelArgs


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

        import tamm.models.afm_text

        adapters = None
        if model_args.use_lora:
            import tamm.adapters
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

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the model and return logits of shape (batch, seq_len, vocab_size)."""
        output = self.model(tokens)
        return output.predictions

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize weights after model.to_empty(device) materializes storage.

        Calls reset_parameters() on every submodule that has it. This covers
        all standard PyTorch layers (nn.Linear, nn.Embedding, nn.LayerNorm, etc.)
        and works correctly with FSDP2 DTensor parameters via in-place ops.
        """
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
