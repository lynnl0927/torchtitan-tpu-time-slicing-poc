"""Sharding map for afmv7 model."""

sharding_map_original = {
    'freqs_cis': (),
    r'.*embedding\.weight': ('fsdp', 'tp'),
    # Attention
    r'.*attention\.qkv_transform\.(?:wrapped\.)?fused_linear\.weight': (
        'tp',
        'fsdp',
    ),
    r'.*attention\.q_transform\.(?:wrapped\.)?weight': ('tp', 'fsdp'),
    r'.*attention\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    # Feed Forward
    r'.*feed_forward\.hidden_transform\.linear_0\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.hidden_transform\.linear_1\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    # Norms
    r'.*norm\.weight': ('fsdp',),
    # Output Transform
    r'^model\.output_transform\.weight$': ('tp', 'fsdp'),
    # LoRA Column Parallel (qkv, q, linear_0, linear_1)
    r'.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None),
    r'.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, 'tp'),
    # LoRA Row Parallel (output_transform)
    r'.*(output_transform).*adapters.*a_transpose.*': ('tp', None),
    r'.*(output_transform).*adapters.*b_transpose.*': (None, None),
}

sharding_map_scan = {
    'freqs_cis': (),
    r'.*embedding\.weight': ('tp', 'fsdp'),
    # Scanned layers
    r'.*params.*attention___qkv_transform___(?:wrapped___)?fused_linear___weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*attention___q_transform___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*attention___output_transform___(?:wrapped___)?weight': (
        None,
        'fsdp',
        'tp',
    ),
    r'.*params.*feed_forward___hidden_transform___linear_0___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*feed_forward___hidden_transform___linear_1___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*feed_forward___output_transform___(?:wrapped___)?weight': (
        None,
        'fsdp',
        'tp',
    ),
    r'.*params.*norm.*weight': (None, 'fsdp'),
    # Original layers (e.g. the last layer in segment_0)
    r'.*attention\.qkv_transform\.(?:wrapped\.)?fused_linear\.weight': (
        'tp',
        'fsdp',
    ),
    r'.*attention\.q_transform\.(?:wrapped\.)?weight': ('tp', 'fsdp'),
    r'.*attention\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    r'.*feed_forward\.hidden_transform\.linear_0\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.hidden_transform\.linear_1\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    r'.*norm\.weight': ('fsdp',),
    r'^model\.output_transform\.weight$': ('tp', 'fsdp'),
    # LoRA Scanned Column Parallel
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None, None),
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, None, 'tp'),
    # LoRA Scanned Row Parallel
    r'.*params.*(output_transform).*adapters.*a_transpose.*': (None, 'tp', None),
    r'.*params.*(output_transform).*adapters.*b_transpose.*': (None, None, None),
    # LoRA Original Non-scanned Column Parallel
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None),
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, 'tp'),
    # LoRA Original Non-scanned Row Parallel
    r'^(?!.*params).*(output_transform).*adapters.*a_transpose.*': ('tp', None),
    r'^(?!.*params).*(output_transform).*adapters.*b_transpose.*': (None, None),
}

sharding_map_scan_lora = {
    'freqs_cis': (),
    r'.*embedding\.weight': ('tp', 'fsdp'),
    # Scanned layers
    r'.*params.*attention___qkv_transform___(?:wrapped___)?fused_linear___weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*attention___q_transform___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*attention___output_transform___(?:wrapped___)?weight': (
        None,
        'fsdp',
        'tp',
    ),
    r'.*params.*feed_forward___hidden_transform___linear_0___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*feed_forward___hidden_transform___linear_1___(?:wrapped___)?weight': (
        None,
        'tp',
        'fsdp',
    ),
    r'.*params.*feed_forward___output_transform___(?:wrapped___)?weight': (
        None,
        'fsdp',
        'tp',
    ),
    r'.*params.*norm.*weight': (None, 'fsdp'),
    # Original layers (e.g. the last layer in segment_0)
    r'.*attention\.qkv_transform\.(?:wrapped\.)?fused_linear\.weight': (
        'tp',
        'fsdp',
    ),
    r'.*attention\.q_transform\.(?:wrapped\.)?weight': ('tp', 'fsdp'),
    r'.*attention\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    r'.*feed_forward\.hidden_transform\.linear_0\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.hidden_transform\.linear_1\.(?:wrapped\.)?weight': (
        'tp',
        'fsdp',
    ),
    r'.*feed_forward\.output_transform\.(?:wrapped\.)?weight': ('fsdp', 'tp'),
    r'.*norm\.weight': ('fsdp',),
    r'^model\.output_transform\.weight$': ('tp', 'fsdp'),
    # LoRA Scanned Column Parallel
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None, None),
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, None, 'tp'),
    # LoRA Scanned Row Parallel
    r'.*params.*(output_transform).*adapters.*a_transpose.*': (None, 'tp', None),
    r'.*params.*(output_transform).*adapters.*b_transpose.*': (None, None, None),
    # LoRA Original Non-scanned Column Parallel
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None),
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, 'tp'),
    # LoRA Original Non-scanned Row Parallel
    r'^(?!.*params).*(output_transform).*adapters.*a_transpose.*': ('tp', None),
    r'^(?!.*params).*(output_transform).*adapters.*b_transpose.*': (None, None),
}

sharding_map_scan_moe = {}


# Hybrid DDP(core) + DDP(LoRA) map. All weights replicated across chips; only
# the input batch is sharded on the fsdp axis. Peak weight memory is lower
# than FSDP (no transient gathered copy per forward: shard + gather = 7.5 GB
# per chip vs. 6 GB for straight replication on a 3B bf16 model), and all
# per-layer all-gather collectives vanish. Use when the model fits replicated
# on each chip — which it does for AFMv7 3B at bf16 on v6e.
sharding_map_scan_lora_ddp = {
    'freqs_cis': (),
    r'.*embedding\.weight': (None, None),
    # Scanned base-model weights: replicated on all mesh axes including the
    # scan leading dim. (None, None, None) = replicated everywhere.
    r'.*params.*attention___qkv_transform___(?:wrapped___)?fused_linear___weight': (None, None, None),
    r'.*params.*attention___q_transform___(?:wrapped___)?weight': (None, None, None),
    r'.*params.*attention___output_transform___(?:wrapped___)?weight': (None, None, None),
    r'.*params.*feed_forward___hidden_transform___linear_0___(?:wrapped___)?weight': (None, None, None),
    r'.*params.*feed_forward___hidden_transform___linear_1___(?:wrapped___)?weight': (None, None, None),
    r'.*params.*feed_forward___output_transform___(?:wrapped___)?weight': (None, None, None),
    r'.*params.*norm.*weight': (None, None),
    # Non-scanned original layers (one per segment edge): replicated.
    r'.*attention\.qkv_transform\.(?:wrapped\.)?fused_linear\.weight': (None, None),
    r'.*attention\.q_transform\.(?:wrapped\.)?weight': (None, None),
    r'.*attention\.output_transform\.(?:wrapped\.)?weight': (None, None),
    r'.*feed_forward\.hidden_transform\.linear_0\.(?:wrapped\.)?weight': (None, None),
    r'.*feed_forward\.hidden_transform\.linear_1\.(?:wrapped\.)?weight': (None, None),
    r'.*feed_forward\.output_transform\.(?:wrapped\.)?weight': (None, None),
    r'.*norm\.weight': (None,),
    r'^model\.output_transform\.weight$': (None, None),
    # LoRA adapters: replicated (same as sharding_map_scan_lora — this is the
    # "DDP for LoRA" behaviour that was already in place; preserved here).
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None, None),
    r'.*params.*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, None, None),
    r'.*params.*(output_transform).*adapters.*a_transpose.*': (None, None, None),
    r'.*params.*(output_transform).*adapters.*b_transpose.*': (None, None, None),
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*a_transpose.*': (None, None),
    r'^(?!.*params).*(qkv_transform|q_transform|linear_0|linear_1).*adapters.*b_transpose.*': (None, None),
    r'^(?!.*params).*(output_transform).*adapters.*a_transpose.*': (None, None),
    r'^(?!.*params).*(output_transform).*adapters.*b_transpose.*': (None, None),
}
