"""Sharding map for AFM PT MoE model in torchax.

  embedding.weight                                         (vocab=262000, hidden=2048)   2D
  layers.segment_0.input_transform.norm.weight             (vec_dim=8, hidden=2048)      2D — segment dispatch norm
  layers.segment_0.output_transform.norm.weight            (vec_dim=8, hidden=2048)      2D — segment combine norm
  layers.segment_X.layer_Y.attention.norm.weight           (vec_dim, hidden)             2D
  layers.segment_X.layer_Y.attention.qkv_transform.fused_linear.weight
                                                           (vec_dim, hidden, 3*head_dim*num_heads_total) 3D
  layers.segment_X.layer_Y.attention.output_transform.weight (vec_dim, attn_hidden, hidden) 3D
  layers.segment_X.layer_Y.attention.residual_connection.pre_residual_norm.weight (vec_dim, hidden) 2D
  layers.segment_X.layer_Y.feed_forward.norm.weight        (vec_dim, hidden)             2D
  layers.segment_X.layer_Y.feed_forward.router.input_transform.weight
                                                           (vec_dim, hidden, num_experts) 3D
  layers.segment_X.layer_Y.feed_forward.experts.hidden_transform.linear_0.weight
                                                           (vec_dim, num_experts, hidden, ffn_hidden) 4D
  layers.segment_X.layer_Y.feed_forward.experts.hidden_transform.linear_1.weight
                                                           (vec_dim, num_experts, hidden, ffn_hidden) 4D
  layers.segment_X.layer_Y.feed_forward.experts.output_transform.weight
                                                           (vec_dim, num_experts, ffn_hidden, hidden) 4D
  layers.segment_X.layer_Y.feed_forward.residual_connection.pre_residual_norm.weight (vec_dim, hidden) 2D
  output_norm.weight                                       (hidden,)                     1D
  output_transform.weight (TIED to embedding.weight, no separate state_dict entry)

Sharding strategy: 1D mesh ('fsdp', 'tp') with tp=1 ⇒ all sharding lands on fsdp.
  - 2D vec_dim×hidden: shard hidden along fsdp                 → (None, 'fsdp')
  - 2D embedding (vocab, hidden): shard along (fsdp, tp)       → ('fsdp', 'tp')
  - 3D Q/K/V/router (vec, in, out): col-parallel on out        → (None, 'tp', 'fsdp')
  - 3D O / dense_ffn_down (vec, in, out): row-parallel on in   → (None, 'fsdp', 'tp')
  - 4D MoE expert linear_0/linear_1 (vec, num_experts, in, out): col-parallel on out → (None, None, 'tp', 'fsdp')
  - 4D MoE expert output_transform (vec, num_experts, in, out): row-parallel on in   → (None, None, 'fsdp', 'tp')
  - 1D output_norm: shard along fsdp                           → ('fsdp',)
  - Catch-all 1D: shard along fsdp
  - Catch-all 2D: replicated (safer than wrong split)
"""

# Patterns are matched in INSERTION ORDER (Python 3.7+ dict). More specific
# patterns must come before more general ones.
sharding_map_original = {
    # Embedding + tied output head
    r'.*\bembedding\.weight$': ('tp', 'fsdp'),
    r'^model\.output_transform\.weight$': ('tp', 'fsdp'),  # tied; usually not in state_dict
    # 1D output_norm
    r'.*\boutput_norm\.weight$': ('fsdp',),
    # 4D MoE expert linear_0/linear_1: column-parallel on out
    r'.*feed_forward\.experts\.hidden_transform\.linear_(0|1)\.weight$': (None, None, 'tp', 'fsdp'),
    # 4D MoE expert output_transform: row-parallel on in
    r'.*feed_forward\.experts\.output_transform\.weight$': (None, None, 'fsdp', 'tp'),
    # 3D MoE router input_transform (vec, hidden, num_experts): row-parallel on hidden
    r'.*feed_forward\.router\.input_transform\.weight$': (None, 'fsdp', 'tp'),
    # 3D dense FFN gate/up (vec, hidden, ffn_hidden): col-parallel on out
    r'.*feed_forward\.hidden_transform\.linear_(0|1)\.weight$': (None, 'tp', 'fsdp'),
    # 3D dense FFN down (vec, ffn_hidden, hidden): row-parallel on in
    r'.*feed_forward\.output_transform\.weight$': (None, 'fsdp', 'tp'),
    # 3D fused QKV: col-parallel on out
    r'.*attention\.qkv_transform\.fused_linear\.weight$': (None, 'tp', 'fsdp'),
    # 3D Q/K/V (separate, in case the config uses unfused): col-parallel on out
    r'.*attention\.q_transform\.weight$': (None, 'tp', 'fsdp'),
    r'.*attention\.k_transform\.weight$': (None, 'tp', 'fsdp'),
    r'.*attention\.v_transform\.weight$': (None, 'tp', 'fsdp'),
    # 3D O projection: row-parallel on in
    r'.*attention\.output_transform\.weight$': (None, 'fsdp', 'tp'),
    # 2D vectorized norms (vec_dim, hidden) — many variants:
    r'.*\.norm\.weight$': (None, 'fsdp'),
    r'.*\.pre_residual_norm\.weight$': (None, 'fsdp'),
    r'.*qk_norm\.(query|key)_norm\.weight$': (None, 'fsdp'),
}

sharding_map_scan = sharding_map_original
sharding_map_scan_lora = sharding_map_original
sharding_map_scan_lora_ddp = sharding_map_original
sharding_map_scan_moe = sharding_map_original
