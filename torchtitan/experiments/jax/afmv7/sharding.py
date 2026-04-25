"""Sharding maps for the JAX AFMTextV7 model.

The matcher in ``torchtitan.experiments.jax.distributed.apply_sharding_to_state``
accepts either literal path strings (with integer segments replaced by ``*``)
or regex patterns. We use regex patterns here to accommodate both the LoRA
and non-LoRA layouts: the LoRA build adds an intermediate ``wrapped/`` node
under each adapted linear, while the non-LoRA build exposes the base
``nnx.Linear`` under ``linear/`` (or ``fused_linear/`` for the fused QKV).

Representative parameter paths (Flax NNX, integer indices omitted):
    embedding/embedding                                          [vocab, hidden]
    layers/segment_0/layers/<i>/attention/norm/weight            [hidden]
    layers/segment_0/layers/<i>/attention/qkv_transform/fused_linear/kernel          (no LoRA)
    layers/segment_0/layers/<i>/attention/qkv_transform/wrapped/fused_linear/kernel  (LoRA)
    layers/segment_0/layers/<i>/attention/qkv_transform/adapters/lora_<0,1,2>/{a,b}_transpose
    layers/segment_0/layers/<i>/attention/qk_norm/query_norm/weight
    layers/segment_0/layers/<i>/attention/qk_norm/key_norm/weight
    layers/segment_0/layers/<i>/attention/output_transform/linear/kernel             (no LoRA)
    layers/segment_0/layers/<i>/attention/output_transform/wrapped/kernel            (LoRA)
    layers/segment_0/layers/<i>/attention/output_transform/adapters/lora/{a,b}_transpose
    layers/segment_0/layers/<i>/feed_forward/norm/weight
    layers/segment_0/layers/<i>/feed_forward/hidden_transform/linear_<0,1>/kernel            (no LoRA)
    layers/segment_0/layers/<i>/feed_forward/hidden_transform/linear_<0,1>/wrapped/kernel    (LoRA)
    layers/segment_0/layers/<i>/feed_forward/hidden_transform/linear_<0,1>/adapters/lora/{a,b}_transpose
    layers/segment_0/layers/<i>/feed_forward/output_transform/{linear|wrapped}/kernel
    layers/segment_0/layers/<i>/feed_forward/output_transform/adapters/lora/{a,b}_transpose
    layers/segment_1/layers/<i>/attention/q_transform/{linear|wrapped}/kernel
    layers/segment_1/layers/<i>/attention/q_norm/query_norm/weight
    layers/segment_1/layers/<i>/attention/output_transform/{linear|wrapped}/kernel
    layers/segment_1/layers/<i>/feed_forward/...
    output_norm/weight
    rope

Kernel tensors in Flax follow ``[in, out]`` (transposed relative to PyTorch),
so the sharding axes below are the PyTorch layout's axes swapped: ``in`` axis
gets ``fsdp`` for column-parallel linears, ``out`` axis gets ``fsdp`` for
row-parallel linears.
"""

# ---------------------------------------------------------------------------
# Original (non-scan) sharding map.
# ---------------------------------------------------------------------------

sharding_map_original = {
    # RoPE table and embedding.
    r"rope": (),
    r"embedding/embedding": ("tp", "fsdp"),

    # Fused QKV projection (column parallel): kernel [in=hidden, out=Q+K+V].
    r".*attention/qkv_transform/(?:wrapped/)?fused_linear/kernel": ("fsdp", "tp"),
    # Q-only projection (segment_1, column parallel).
    r".*attention/q_transform/(?:linear|wrapped)/kernel": ("fsdp", "tp"),
    # Attention output projection (row parallel): kernel [in=hidden, out=hidden].
    r".*attention/output_transform/(?:linear|wrapped)/kernel": ("tp", "fsdp"),

    # QK / Q norms.
    r".*attention/qk_norm/query_norm/weight": ("fsdp",),
    r".*attention/qk_norm/key_norm/weight": ("fsdp",),
    r".*attention/q_norm/query_norm/weight": ("fsdp",),

    # Pre-norms inside attention / feed-forward blocks.
    r".*attention/norm/weight": ("fsdp",),
    r".*feed_forward/norm/weight": ("fsdp",),

    # Feed-forward hidden transform (gate + up, column parallel): [in=hidden, out=ffn].
    r".*feed_forward/hidden_transform/linear_[01]/(?:wrapped/)?kernel": ("fsdp", "tp"),
    # Feed-forward output transform (down, row parallel): [in=ffn, out=hidden].
    r".*feed_forward/output_transform/(?:linear|wrapped)/kernel": ("tp", "fsdp"),

    # Final output norm.
    r"^output_norm/weight": ("fsdp",),

    # --- LoRA adapters ---
    # Column-parallel sources (qkv_transform, q_transform, hidden_transform
    # linear_0 / linear_1): a_transpose [in, rank] replicated; b_transpose
    # [rank, out] has its output on tp.
    r".*(qkv_transform|q_transform|linear_[01])/adapters/[^/]+/a_transpose": (None, None),
    r".*(qkv_transform|q_transform|linear_[01])/adapters/[^/]+/b_transpose": (None, "tp"),
    # Row-parallel sinks (attention output_transform, feed-forward
    # output_transform): a_transpose [in, rank] has its input sharded on tp;
    # b_transpose [rank, out] replicated.
    r".*(output_transform)/adapters/[^/]+/a_transpose": ("tp", None),
    r".*(output_transform)/adapters/[^/]+/b_transpose": (None, None),
}


# ---------------------------------------------------------------------------
# Scan variant: stacked weights live under:
#   layers/segment_0/scanned/<flat_attr>        (n-1 regular layers, stacked)
#   layers/segment_0/last_layer/...             (non-scanned last reg layer)
#   layers/segment_1/<flat_attr>                (all kv-reuse layers, stacked)
# `<flat_attr>` = the path of each ref-block leaf with '/' → '_', e.g.
#   attention_norm_weight
#   attention_qkv_transform_fused_linear_kernel
#   feed_forward_hidden_transform_linear_0_kernel
#   feed_forward_output_transform_linear_kernel
# The leading axis is the stacked-layer dim — always unsharded (`None`).
# ---------------------------------------------------------------------------

sharding_map_scan = {
    # Tables and norms outside layer stacks.
    r"rope": (),
    r"embedding/embedding": ("tp", "fsdp"),
    r"^output_norm/weight": ("fsdp",),

    # --- Non-scanned `last_layer` of segment_0: reuse the same patterns as
    # the original sharding map (matched by the ``.*`` prefix regex). ---
    r".*last_layer.*attention/qkv_transform/(?:wrapped/)?fused_linear/kernel":
        ("fsdp", "tp"),
    r".*last_layer.*attention/output_transform/(?:linear|wrapped)/kernel":
        ("tp", "fsdp"),
    r".*last_layer.*attention/qk_norm/query_norm/weight": ("fsdp",),
    r".*last_layer.*attention/qk_norm/key_norm/weight": ("fsdp",),
    r".*last_layer.*attention/norm/weight": ("fsdp",),
    r".*last_layer.*feed_forward/norm/weight": ("fsdp",),
    r".*last_layer.*feed_forward/hidden_transform/linear_[01]/(?:wrapped/)?kernel":
        ("fsdp", "tp"),
    r".*last_layer.*feed_forward/output_transform/(?:linear|wrapped)/kernel":
        ("tp", "fsdp"),
    r".*last_layer.*(qkv_transform|q_transform|linear_[01])/adapters/[^/]+/a_transpose":
        (None, None),
    r".*last_layer.*(qkv_transform|q_transform|linear_[01])/adapters/[^/]+/b_transpose":
        (None, "tp"),
    r".*last_layer.*(output_transform)/adapters/[^/]+/a_transpose": ("tp", None),
    r".*last_layer.*(output_transform)/adapters/[^/]+/b_transpose": (None, None),

    # --- Scanned stacks: leading axis is stacked-layer dim (None). The
    # remainder follows the PyTorch-style weight orientation [out, in]-ish
    # mapped to Flax kernel [in, out] — matches sharding_map_original. ---
    # segment_0 scanned regular blocks:
    r".*scanned/attention_norm_weight$": (None, "fsdp"),
    r".*scanned/attention_qkv_transform_fused_linear_kernel$": (None, "fsdp", "tp"),
    r".*scanned/attention_qkv_transform_wrapped_fused_linear_kernel$": (None, "fsdp", "tp"),
    r".*scanned/attention_qk_norm_query_norm_weight$": (None, "fsdp"),
    r".*scanned/attention_qk_norm_key_norm_weight$": (None, "fsdp"),
    r".*scanned/attention_output_transform_linear_kernel$": (None, "tp", "fsdp"),
    r".*scanned/attention_output_transform_wrapped_kernel$": (None, "tp", "fsdp"),
    r".*scanned/feed_forward_norm_weight$": (None, "fsdp"),
    r".*scanned/feed_forward_hidden_transform_linear_[01]_kernel$": (None, "fsdp", "tp"),
    r".*scanned/feed_forward_hidden_transform_linear_[01]_wrapped_kernel$": (None, "fsdp", "tp"),
    r".*scanned/feed_forward_output_transform_linear_kernel$": (None, "tp", "fsdp"),
    r".*scanned/feed_forward_output_transform_wrapped_kernel$": (None, "tp", "fsdp"),
    # scanned LoRA adapters
    r".*scanned/.*(qkv_transform|q_transform|linear_[01]).*adapters.*a_transpose$":
        (None, None, None),
    r".*scanned/.*(qkv_transform|q_transform|linear_[01]).*adapters.*b_transpose$":
        (None, None, "tp"),
    r".*scanned/.*(output_transform).*adapters.*a_transpose$":
        (None, "tp", None),
    r".*scanned/.*(output_transform).*adapters.*b_transpose$":
        (None, None, None),

    # segment_1 (ScannedKVReuseBlocks — attributes live directly on the
    # segment_1 module, no extra 'scanned' path segment):
    r"^layers/segment_1/attention_norm_weight$": (None, "fsdp"),
    r"^layers/segment_1/attention_q_transform_kernel$": (None, "fsdp", "tp"),
    r"^layers/segment_1/attention_q_transform_wrapped_kernel$": (None, "fsdp", "tp"),
    r"^layers/segment_1/attention_q_norm_query_norm_weight$": (None, "fsdp"),
    r"^layers/segment_1/attention_output_transform_linear_kernel$": (None, "tp", "fsdp"),
    r"^layers/segment_1/attention_output_transform_wrapped_kernel$": (None, "tp", "fsdp"),
    r"^layers/segment_1/feed_forward_norm_weight$": (None, "fsdp"),
    r"^layers/segment_1/feed_forward_hidden_transform_linear_[01]_kernel$": (None, "fsdp", "tp"),
    r"^layers/segment_1/feed_forward_hidden_transform_linear_[01]_wrapped_kernel$": (None, "fsdp", "tp"),
    r"^layers/segment_1/feed_forward_output_transform_linear_kernel$": (None, "tp", "fsdp"),
    r"^layers/segment_1/feed_forward_output_transform_wrapped_kernel$": (None, "tp", "fsdp"),
    r"^layers/segment_1/.*(q_transform|linear_[01]).*adapters.*a_transpose$":
        (None, None, None),
    r"^layers/segment_1/.*(q_transform|linear_[01]).*adapters.*b_transpose$":
        (None, None, "tp"),
    r"^layers/segment_1/.*output_transform.*adapters.*a_transpose$":
        (None, "tp", None),
    r"^layers/segment_1/.*output_transform.*adapters.*b_transpose$":
        (None, None, None),
}
