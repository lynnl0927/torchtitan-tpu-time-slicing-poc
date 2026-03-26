"""AFMParallelTrackMoE model registration for TorchTitan TPU experiments."""

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.validate import build_validator
from torchtitan.experiments.tpu.afmv7.tokenizer import build_afm_tokenizer
from torchtitan.experiments.tpu.loss import build_cross_entropy_loss
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import register_train_spec
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_afm_pt_moe
from .model.args import AFMPTMoeModelArgs
from .model.model import AFMPTMoeWrapper

__all__ = [
    "parallelize_afm_pt_moe",
    "AFMPTMoeModelArgs",
    "AFMPTMoeWrapper",
    "afm_pt_moe_args",
]

afm_pt_moe_args = {
    "debugmodel": AFMPTMoeModelArgs(
        vocab_size=2048,
        num_tracks=2,
        num_layers_per_track=4,
        num_layers_per_track_per_sync_point=2,
        hidden_dim=64,
        attention_hidden_dim=64,
        dense_feed_forward_hidden_dim=256,
        sparse_feed_forward_hidden_dim=128,
        num_heads=4,
        num_experts=4,
    ),
}

# pytype: disable=wrong-arg-types
register_train_spec(
    name="afm_pt_moe_tpu",
    train_spec=TrainSpec(
        model_cls=AFMPTMoeWrapper,
        model_args=afm_pt_moe_args,
        parallelize_fn=parallelize_afm_pt_moe,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_afm_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    ),
)
# pytype: enable=wrong-arg-types
