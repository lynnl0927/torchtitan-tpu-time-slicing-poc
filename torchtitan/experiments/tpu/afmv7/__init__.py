"""AFMTextV7 model registration for TorchTitan TPU experiments."""

from .infra.parallelize import parallelize_afmv7
from .model.args import AFMTextV7ModelArgs
from .model.model import AFMTextV7Wrapper
from .tokenizer import AFMTokenizerWrapper
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

__all__ = [
    "parallelize_afmv7",
    "AFMTextV7ModelArgs",
    "AFMTextV7Wrapper",
    "afmv7_args",
]

afmv7_args = {
    # Full production model (~3B params, default TAMM AFMTextV7 config).
    "3B": AFMTextV7ModelArgs(
        vocab_size=153600,
        hidden_dim=2048,
        num_layers=56,
        num_kv_reuse_layers=21,
        num_heads=16,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
    ),
    # 3B variant: num_heads=8 instead of 16 (head_dim=2048//8=256 vs baseline 128).
    # Per-token attention FLOPs unchanged (num_heads*head_dim is invariant);
    # K/V projection params double; GQA grouping becomes 4 Q-heads per KV-head.
    "3B-h8": AFMTextV7ModelArgs(
        vocab_size=153600,
        hidden_dim=2048,
        num_layers=56,
        num_kv_reuse_layers=21,
        num_heads=8,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
    ),
    # 3B model with 50k vocabulary size.
    "3B-50k": AFMTextV7ModelArgs(
        vocab_size=49152,
        hidden_dim=3072,
        num_layers=26,
        num_kv_reuse_layers=0,
        num_heads=24,
        num_kv_heads=8,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
    ),
    # Tiny model for local debugging without a TPU.
    # vocab_size matches the test tokenizer at tests/assets/tokenizer.
    "debugmodel": AFMTextV7ModelArgs(
        vocab_size=2048,
        hidden_dim=64,
        num_layers=4,
        num_kv_reuse_layers=1,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
    ),
    # Tiny debug model with LoRA fine-tuning enabled.
    "debugmodel-lora": AFMTextV7ModelArgs(
        vocab_size=2048,
        hidden_dim=64,
        num_layers=4,
        num_kv_reuse_layers=1,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
        use_lora=True,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dtype="float32",
    ),
    # Medium model for faster iteration on Sparse Core tests.
    "medium-lora": AFMTextV7ModelArgs(
        vocab_size=2048,
        hidden_dim=1024,
        num_layers=12,
        num_kv_reuse_layers=3,
        num_heads=8,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
        use_lora=True,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dtype="float32",
    ),
    # Production 3B model with LoRA fine-tuning (matching the sample script).
    "3B-lora": AFMTextV7ModelArgs(
        vocab_size=153600,
        hidden_dim=2048,
        num_layers=56,
        num_kv_reuse_layers=21,
        num_heads=16,
        num_kv_heads=2,
        hidden_dim_scale_factor=3.25,
        rope_theta=500000.0,
        use_lora=True,
        lora_rank=16,
        lora_alpha=16.0,
        lora_dtype="float32",
    ),
}

# pytype: disable=wrong-arg-types
register_train_spec(
    name="afmv7_tpu",
    train_spec=TrainSpec(
        model_cls=AFMTextV7Wrapper,
        model_args=afmv7_args,
        parallelize_fn=parallelize_afmv7,
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


