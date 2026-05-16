"""AFMParallelTrackMoE model registration for TorchTitan TPU experiments."""

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.experiments.tpu.afmv7.tokenizer import AFMTokenizerWrapper
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
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
        hidden_dim=128,
        attention_hidden_dim=64,
        dense_feed_forward_hidden_dim=256,
        sparse_feed_forward_hidden_dim=128,
        num_heads=2,
        num_experts=2,
    ),
    "24b": AFMPTMoeModelArgs(
        vocab_size=262000,
        attention_hidden_dim=1536,
        sparse_feed_forward_hidden_dim=2048,
        num_heads=12,
        rope_theta=10000.0,
        num_experts=4,
        attention_layer_pattern=[
            "local_rope",
            "local_rope",
            "local_rope",
            "global_nope",
        ],
        feed_forward_layer_pattern=["sparse"],
        local_attention_window_size=511,
        norm_eps=1e-06,
        scale_qk_norm=False,
        pre_norm="pre_scale_rms_norm",
        pre_residual_norm="rms_norm",
        tracks_combine_norm="pre_scale_rms_norm",
        tracks_combine_op="sum",
        tracks_dispatch_norm="rms_norm",
        sdpa_implementation="native_torch",
        experts_router_logits_cap=None,
    ),
    # Same architecture as "24b" but with num_layers_per_track shrunk from 48
    # to 4 (the only change).
    "3b": AFMPTMoeModelArgs(
        vocab_size=262000,
        attention_hidden_dim=1536,
        sparse_feed_forward_hidden_dim=2048,
        num_heads=12,
        rope_theta=10000.0,
        num_experts=4,
        num_layers_per_track=4,
        attention_layer_pattern=[
            "local_rope",
            "local_rope",
            "local_rope",
            "global_nope",
        ],
        feed_forward_layer_pattern=["sparse"],
        local_attention_window_size=511,
        norm_eps=1e-06,
        scale_qk_norm=False,
        pre_norm="pre_scale_rms_norm",
        pre_residual_norm="rms_norm",
        tracks_combine_norm="pre_scale_rms_norm",
        tracks_combine_op="sum",
        tracks_dispatch_norm="rms_norm",
        sdpa_implementation="native_torch",
        experts_router_logits_cap=None,
    ),
    "24b-lora": AFMPTMoeModelArgs(
        vocab_size=262000,
        attention_hidden_dim=1536,
        sparse_feed_forward_hidden_dim=2048,
        num_heads=12,
        rope_theta=10000.0,
        num_experts=4,
        attention_layer_pattern=[
            "local_rope",
            "local_rope",
            "local_rope",
            "global_nope",
        ],
        feed_forward_layer_pattern=["sparse"],
        local_attention_window_size=511,
        norm_eps=1e-06,
        scale_qk_norm=False,
        pre_norm="pre_scale_rms_norm",
        pre_residual_norm="rms_norm",
        tracks_combine_norm="pre_scale_rms_norm",
        tracks_combine_op="sum",
        tracks_dispatch_norm="rms_norm",
        sdpa_implementation="native_torch",
        experts_router_logits_cap=None,
        use_lora=True,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dtype="float32",
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
        loss_config=CrossEntropyLoss.Config(),
        optimizer_config=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler_config=LRSchedulersContainer.Config(warmup_steps=2),
        dataloader_config=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        tokenizer_config=AFMTokenizerWrapper.Config(),
        validator_config=Validator.Config(enable=True),
        state_dict_adapter=None,
    ),
)
# pytype: enable=wrong-arg-types
