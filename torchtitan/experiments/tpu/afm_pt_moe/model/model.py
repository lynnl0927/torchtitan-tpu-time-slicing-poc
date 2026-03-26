"""AFMPTMoe model wrapper for TorchTitan."""

import torch
import torch.nn as nn

from torchtitan.protocols.model import ModelProtocol

from .args import AFMPTMoeModelArgs


class AFMPTMoeWrapper(ModelProtocol):
    """Wraps TAMM's AFMPTMoe as a TorchTitan ModelProtocol."""

    def __init__(self, model_args: AFMPTMoeModelArgs) -> None:
        super().__init__(model_args)
        self._model_args = model_args

        import tamm.models.afm_text.afm_pt_moe as afm_pt_moe

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
        )
        # pytype: enable=wrong-keyword-args
        self._tamm_cfg = cfg
        self.model: nn.Module = cfg.create_basic_builder().build()

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the model and return logits."""
        output = self.model(tokens)
        return output.predictions

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize weights after model.to_empty(device) materializes storage."""
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
