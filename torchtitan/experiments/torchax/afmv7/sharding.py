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
    # LoRA specific sharding
    r'.*adapters.*a_transpose.*': ('fsdp', None),
    r'.*adapters.*b_transpose.*': (None, 'fsdp'),
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
    # LoRA specific sharding
    r'.*params.*adapters.*a_transpose.*': (None, 'fsdp', None),
    r'.*params.*adapters.*b_transpose.*': (None, None, 'fsdp'),
    r'.*adapters.*a_transpose.*': ('fsdp', None),
    r'.*adapters.*b_transpose.*': (None, 'fsdp'),
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
    # LoRA specific sharding
    r'.*params.*adapters.*a_transpose.*': (None, 'fsdp', None),
    r'.*params.*adapters.*b_transpose.*': (None, None, 'fsdp'),
    r'.*adapters.*a_transpose.*': ('fsdp', None),
    r'.*adapters.*b_transpose.*': (None, 'fsdp'),
}

sharding_map_scan_moe = {}
