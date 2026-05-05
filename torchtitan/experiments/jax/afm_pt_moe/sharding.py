"""Sharding maps for the JAX AFM PT MoE model.

The matcher in ``torchtitan.experiments.jax.distributed.apply_sharding_to_state``
builds path strings with ``/`` separators and the trailing
``GetAttrKey('value')`` from nnx.Variable already stripped. So the regex
patterns below match the **slash-separated** Flax NNX path layout — *not*
the dot-separated PyTorch/TAMM-style names that the torchax bridge uses.

Representative parameter paths (Flax NNX, integer segment indices kept):
    embedding/embedding                                          [vocab, hidden]
    layers/segment_<S>/input_transform/norm/weight              [vec, hidden]
    layers/segment_<S>/output_transform/norm/weight             [vec, hidden]
    layers/segment_<S>/layer_<L>/attention/norm/weight          [vec, hidden]
    layers/segment_<S>/layer_<L>/attention/qkv_transform/fused_linear/weight
                                                                [vec, hidden, q+k+v]
    layers/segment_<S>/layer_<L>/attention/output_transform/weight
                                                                [vec, attn_hidden, hidden]
    layers/segment_<S>/layer_<L>/attention/residual_connection/pre_residual_norm/weight
                                                                [vec, hidden]
    layers/segment_<S>/layer_<L>/feed_forward/norm/weight       [vec, hidden]
    layers/segment_<S>/layer_<L>/feed_forward/router/input_transform/weight
                                                                [vec, hidden, num_experts]
    layers/segment_<S>/layer_<L>/feed_forward/experts/hidden_transform/linear_<0,1>/weight
                                                                [vec, num_experts, hidden, ffn_hidden]
    layers/segment_<S>/layer_<L>/feed_forward/experts/output_transform/weight
                                                                [vec, num_experts, ffn_hidden, hidden]
    layers/segment_<S>/layer_<L>/feed_forward/residual_connection/pre_residual_norm/weight
                                                                [vec, hidden]
    output_norm/weight                                          [hidden]

Sharding strategy: 1D-effective mesh ('fsdp', 'tp'), with tp=1 in the typical
v6e-N FSDP run ⇒ all sharding load lands on the ``fsdp`` axis.
  - 2D ``(vec, hidden)`` norms: shard hidden along fsdp     → (None, 'fsdp')
  - 2D ``(vocab, hidden)`` embedding: ('tp', 'fsdp')        — vocab on tp, hidden on fsdp
  - 3D ``(vec, in, out)`` Q/K/V/router (col-parallel):      → (None, 'tp', 'fsdp')
  - 3D ``(vec, in, out)`` O (row-parallel on in):           → (None, 'fsdp', 'tp')
  - 4D MoE expert ``(vec, num_experts, in, out)``:
      gate/up linear_0/_1 (col-parallel on out):            → (None, None, 'tp', 'fsdp')
      output_transform (row-parallel on in):                → (None, None, 'fsdp', 'tp')
  - 1D ``output_norm/weight``: ('fsdp',)
"""

# Patterns are matched in INSERTION ORDER (Python 3.7+ dict). More specific
# patterns must come before more general ones.
sharding_map_original = {
    # Embedding + tied output head.
    # Flax nnx.Embed names the param ``embedding/embedding``; we also accept
    # the PyTorch/TAMM ``embedding/weight`` form (it would only show up if a
    # state was loaded directly from PyTorch).
    r"embedding/embedding": ("tp", "fsdp"),
    r"embedding/weight": ("tp", "fsdp"),
    r"^model/output_transform/weight$": ("tp", "fsdp"),  # tied; usually not in state

    # 1D output_norm (post-segments scalar RMSNorm).
    r"output_norm/weight": ("fsdp",),

    # Convention (matches jax/afmv7/sharding.py):
    #   - Column-parallel  (split outputs):  IN sharded by FSDP, OUT sharded by TP.
    #   - Row-parallel     (split inputs):   IN sharded by TP,   OUT sharded by FSDP.
    # Earlier revisions of this file had the in/out axes swapped, which left
    # the qkv_out axis (4608) effectively unsharded under tp=4 + fsdp=8 and
    # blew up the QKV intermediate to bf16[t, qkv, B, S] = 288 MB at seq=4096.

    # 4D MoE expert linear_0/linear_1 (vec, num_experts, hidden_in, ffn_out):
    # col-parallel on out (last dim).
    r".*feed_forward/experts/hidden_transform/linear_(0|1)/weight$":
        (None, None, "fsdp", "tp"),
    # 4D MoE expert output_transform (vec, num_experts, ffn_in, hidden_out):
    # row-parallel on in.
    r".*feed_forward/experts/output_transform/weight$":
        (None, None, "tp", "fsdp"),

    # 3D MoE router input_transform (vec, hidden_in, num_experts):
    # col-parallel on out (num_experts axis is small but consistent).
    r".*feed_forward/router/input_transform/weight$":
        (None, "fsdp", "tp"),

    # 3D dense FFN gate/up (vec, hidden, ffn): col-parallel on out.
    r".*feed_forward/hidden_transform/linear_(0|1)/weight$":
        (None, "fsdp", "tp"),
    # 3D dense FFN down (vec, ffn, hidden): row-parallel on in.
    r".*feed_forward/output_transform/weight$":
        (None, "tp", "fsdp"),

    # 3D fused QKV (vec, hidden_in, q+k+v=12*128*3=4608): col-parallel on out.
    # Heads = 12 ÷ tp(=4) = 3 per chip cleanly.
    r".*attention/qkv_transform/fused_linear/weight$":
        (None, "fsdp", "tp"),
    # Separate Q/K/V (in case any flavor uses unfused).
    r".*attention/q_transform/weight$": (None, "fsdp", "tp"),
    r".*attention/k_transform/weight$": (None, "fsdp", "tp"),
    r".*attention/v_transform/weight$": (None, "fsdp", "tp"),
    # 3D O (vec, attn_hidden_in, hidden_out): row-parallel on in.
    r".*attention/output_transform/weight$": (None, "tp", "fsdp"),

    # 2D vectorized norms — many call sites:
    #   layer-level pre-norms (attention/norm/weight, feed_forward/norm/weight)
    #   per-residual norms    (.*/residual_connection/pre_residual_norm/weight)
    #   per-segment dispatch / combine norms
    #     (input_transform/norm/weight, output_transform/norm/weight)
    #   QK norms              (qk_norm/(query|key)_norm/weight)
    r".*/norm/weight$": (None, "fsdp"),
    r".*/pre_residual_norm/weight$": (None, "fsdp"),
    r".*qk_norm/(query|key)_norm/weight$": (None, "fsdp"),
}


sharding_map_scan = sharding_map_original
sharding_map_scan_lora = sharding_map_original
sharding_map_scan_lora_ddp = sharding_map_original
sharding_map_scan_moe = sharding_map_original
