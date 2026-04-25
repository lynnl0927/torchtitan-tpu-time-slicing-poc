"""Equivalence test: JAX AFMTextV7 vs. torchax/TAMM AFMTextV7.

Strategy:
  1. Build a tiny TAMM PyTorch AFMTextV7. TAMM's ``reset_parameters`` gives
     each weight a deterministic random init — treat that as our "checkpoint".
  2. Build an equivalent Flax NNX model.
  3. Translate the PyTorch state_dict to the JAX parameter layout using a
     single mapping function (handles ``weight`` → ``kernel`` renaming, the
     ``[out, in]`` → ``[in, out]`` transpose, the ``layer_i`` → ``layers.i``
     container difference, the ``layers.segment_0`` vs. flat
     ``layers.layer_i`` split when ``num_kv_reuse_layers == 0``, and the
     non-LoRA ``<linear>/weight`` → ``<linear>/linear/kernel`` insertion).
  4. Forward both models on the same input and compare logits.

This "checkpoint-style" setup is much shorter than hand-wired per-layer
assignments because it walks the state_dict once and delegates all path
translation to one helper.
"""

import re

import numpy as np
import jax
import jax.numpy as jnp
import torch
from absl.testing import absltest
from flax import nnx

from torchtitan.experiments.jax import afmv7

JaxArgs = afmv7.AFMTextV7ModelArgs
JaxTransformer = afmv7.Transformer


# ---------------------------------------------------------------------------
# PyTorch / TAMM model builder
# ---------------------------------------------------------------------------

def _patch_tamm_lora_for_mixed_precision() -> None:
    """Make LoRA tolerate dtype mismatches between base weights and LoRA dtype.

    Same workaround used by torchtitan.experiments.tpu.afmv7.model.model.
    """
    from tamm._adapters_v1.layer_adapters import lora as _lora_mod

    def _impl(self, x, outputs):
        x = x.flatten(end_dim=-2)
        x = x.to(self.a_transpose.dtype)
        x = torch.matmul(x, self.a_transpose)
        batch_shape = outputs.shape[:-1]
        outputs_f = outputs.flatten(end_dim=-2)
        result = torch.addmm(
            outputs_f.to(x.dtype), x, self.b_transpose, alpha=self.scale
        )
        return result.to(outputs.dtype).reshape(*batch_shape, -1)

    _lora_mod.LoRA._transform_outputs_impl = _impl


def _build_pt_model(args: JaxArgs) -> torch.nn.Module:
    """Build a TAMM PyTorch AFMTextV7 matching ``args``; return an eval'd model
    with its parameters freshly initialized (TAMM's default init)."""
    import tamm.models.afm_text

    adapters = None
    if args.use_lora:
        import tamm.adapters
        _patch_tamm_lora_for_mixed_precision()
        adapters = {
            "lora": tamm.adapters.LoRAModelAdapter(
                rank=args.lora_rank,
                alpha=float(args.lora_alpha),
                dtype=getattr(torch, args.lora_dtype),
                adapt_attention_queries=True,
                adapt_attention_keys=True,
                adapt_attention_values=True,
                adapt_attention_outputs=True,
                adapt_feed_forward_hidden_states=True,
                adapt_feed_forward_outputs=True,
            )
        }

    cfg = tamm.models.afm_text.AFMTextV7.Config(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_kv_reuse_layers=args.num_kv_reuse_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        hidden_dim_scale_factor=args.hidden_dim_scale_factor,
        rope_theta=args.rope_theta,
        pretrained=False,
        adapters=adapters,
    )
    torch.manual_seed(0)
    model = cfg.create_model()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# State-dict translation: PyTorch → JAX
# ---------------------------------------------------------------------------

# Attribute names that wrap a plain ``nn.Linear`` in TAMM. Without LoRA these
# appear as ``<name>.weight``; in the JAX model they live at
# ``<name>/linear/kernel``. With LoRA both backends add a ``wrapped/`` node.
_WRAPPED_LINEAR_ATTRS = ("output_transform", "q_transform")


def _pt_key_to_jax_path(pt_key: str, args: JaxArgs):
    """Translate a PyTorch state_dict key to a JAX parameter path.

    Returns ``(jax_path_tuple, transpose)``. ``transpose`` is True for any
    ``nn.Linear`` weight (stored [out, in] in PyTorch and [in, out] in Flax).

    Raises KeyError if the PT key is not recognised, so forgotten mappings
    surface loudly rather than silently skipping parameters.
    """
    parts = pt_key.split(".")

    # embedding.weight → embedding.embedding
    if pt_key == "embedding.weight":
        return ("embedding", "embedding"), False

    if pt_key == "output_transform.weight":
        # TAMM exposes this as a tied view of embedding.weight; our JAX
        # model ties via ``x @ embedding.weight.T`` so there is no separate
        # parameter to copy.
        return None, False

    if pt_key == "output_norm.weight":
        return ("output_norm", "weight"), False

    # layers.<...>
    if parts[0] != "layers":
        raise KeyError(f"Unhandled PT key: {pt_key}")

    # Normalise the container shape. TAMM uses either:
    #   layers.segment_{0,1}.layer_{i}....   (when num_kv_reuse_layers > 0)
    #   layers.layer_{i}....                 (when num_kv_reuse_layers == 0,
    #                                         flat UniformTransformerLayerSequence)
    # Our JAX model always uses layers.segment_{0,1}.layers.{i}....
    if parts[1].startswith("segment_"):
        segment = parts[1]
        # parts[2] is 'layer_{i}'.
        m = re.match(r"layer_(\d+)", parts[2])
        if m is None:
            raise KeyError(f"Unhandled PT key: {pt_key}")
        layer_idx = int(m.group(1))
        rest = parts[3:]
    elif parts[1].startswith("layer_"):
        # Flat kv_reuse=0 case: map into segment_0.
        assert args.num_kv_reuse_layers == 0, pt_key
        segment = "segment_0"
        layer_idx = int(parts[1].split("_", 1)[1])
        rest = parts[2:]
    else:
        raise KeyError(f"Unhandled PT key: {pt_key}")

    # nnx.List indexes children by int, not str.
    jax_layer_prefix = ("layers", segment, "layers", layer_idx)
    # rest looks like:
    #   ['attention', 'norm', 'weight']
    #   ['attention', 'qkv_transform', 'fused_linear', 'weight']            # no LoRA
    #   ['attention', 'qkv_transform', 'wrapped', 'fused_linear', 'weight'] # LoRA
    #   ['attention', 'qkv_transform', 'adapters', 'lora', 'lora_0', 'a_transpose']
    #   ['attention', 'output_transform', 'weight']                          # no LoRA
    #   ['attention', 'output_transform', 'wrapped', 'weight']               # LoRA
    #   ['attention', 'output_transform', 'adapters', 'lora', 'a_transpose']
    #   ['feed_forward', 'hidden_transform', 'linear_0', 'weight']           # no LoRA
    #   ['feed_forward', 'hidden_transform', 'linear_0', 'wrapped', 'weight']# LoRA
    #   ['feed_forward', 'hidden_transform', 'linear_0', 'adapters', 'lora', 'a_transpose']

    # --- LoRA adapter paths.
    # TAMM: adapters.lora.a_transpose (single) or adapters.lora.lora_<i>.a_transpose (fused).
    # JAX: adapters.lora.a_transpose (single) or adapters.lora_<i>.a_transpose (fused).
    # So for the fused form we strip the outer "lora." level; for the single
    # form we keep it.
    if "adapters" in rest:
        i = rest.index("adapters")
        lora_inner = rest[i + 2:]   # what follows ".adapters.lora"
        prefix = rest[:i]
        if len(lora_inner) >= 2 and lora_inner[0].startswith("lora_"):
            jax_tail = ("adapters",) + tuple(lora_inner)   # fused
        else:
            jax_tail = ("adapters", "lora") + tuple(lora_inner)  # single
        jax_path = jax_layer_prefix + tuple(prefix) + jax_tail
        return jax_path, False

    # --- Base (non-adapter) weights.
    # Norm weights: ...norm.weight (stay)
    if rest[-1] == "weight" and any(
        rest[j].endswith("norm") or rest[j] == "norm"
        for j in range(len(rest) - 1)
    ):
        # e.g. ['attention', 'norm', 'weight'] or
        #      ['attention', 'qk_norm', 'query_norm', 'weight']
        return jax_layer_prefix + tuple(rest), False

    # Linear weights: replace trailing 'weight' with 'kernel' and transpose.
    if rest[-1] != "weight":
        raise KeyError(f"Unhandled PT key: {pt_key}")

    # Insert 'linear' node for non-LoRA wrapped-style attributes
    # (output_transform, q_transform): TAMM stores them as
    # ``<attr>.weight`` but our JAX model uses ``<attr>/linear/kernel``.
    if (not args.use_lora) and len(rest) >= 2 and rest[-2] in _WRAPPED_LINEAR_ATTRS:
        jax_path = jax_layer_prefix + tuple(rest[:-1]) + ("linear", "kernel")
        return jax_path, True

    # Everything else: straight rename.
    jax_path = jax_layer_prefix + tuple(rest[:-1]) + ("kernel",)
    return jax_path, True


def _load_pt_checkpoint_into_jax(
    pt_state: dict, jax_model: JaxTransformer, args: JaxArgs
) -> None:
    """Overwrite JAX parameters from a PyTorch state_dict."""
    state = nnx.state(jax_model, nnx.Param)

    for pt_key, pt_tensor in pt_state.items():
        jax_path, transpose = _pt_key_to_jax_path(pt_key, args)
        if jax_path is None:
            continue  # tied view; nothing to copy.
        arr = pt_tensor.detach().to(torch.float32).cpu().numpy()
        if transpose:
            arr = np.ascontiguousarray(arr.T)

        # Descend into the nested nnx.State dict.
        node = state
        for key in jax_path[:-1]:
            node = node[key]
        leaf = node[jax_path[-1]]
        expected_shape = np.asarray(leaf[...]).shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {pt_key} → {jax_path}: "
                f"PT {arr.shape} vs. JAX {expected_shape}"
            )
        # nnx.Variable is in-place writable via the [...] setter.
        leaf[...] = jnp.asarray(arr, dtype=leaf.dtype)

    nnx.update(jax_model, state)


# ---------------------------------------------------------------------------
# Forward & comparison
# ---------------------------------------------------------------------------

def _run_pair(args: JaxArgs, tokens_np: np.ndarray):
    mesh = jax.make_mesh(
        (1, 1), ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    pt_model = _build_pt_model(args)
    with mesh:
        # Use fp32 master params so the transfer from fp32 torch weights is exact.
        jax_model = JaxTransformer(args, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
        _load_pt_checkpoint_into_jax(pt_model.state_dict(), jax_model, args)
        tokens_jax = jnp.asarray(tokens_np, dtype=jnp.int32)
        jax_out = np.asarray(jax.device_get(jax_model(tokens_jax))).astype(np.float32)

    with torch.no_grad():
        pt_tokens = torch.from_numpy(tokens_np).to(torch.long)
        pt_out = pt_model(pt_tokens).predictions.to(torch.float32).numpy()

    return pt_out, jax_out


class AFMTextV7EquivalenceTest(absltest.TestCase):

    def _assert_close(self, pt_out, jax_out, atol=1e-1, rtol=1e-1):
        self.assertEqual(pt_out.shape, jax_out.shape)
        diff = np.abs(pt_out - jax_out)
        np.testing.assert_allclose(
            pt_out, jax_out, atol=atol, rtol=rtol,
            err_msg=f"max abs diff={diff.max():.4f}"
        )

    def test_no_lora(self):
        args = JaxArgs(
            vocab_size=2048, hidden_dim=64, num_layers=4, num_kv_reuse_layers=1,
            num_heads=4, num_kv_heads=2, hidden_dim_scale_factor=3.25,
            rope_theta=500000.0, max_seq_len=32,
        )
        tokens = np.arange(1, 17, dtype=np.int32).reshape(1, 16) % args.vocab_size
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out)

    def test_no_kv_reuse(self):
        args = JaxArgs(
            vocab_size=512, hidden_dim=32, num_layers=3, num_kv_reuse_layers=0,
            num_heads=4, num_kv_heads=2, hidden_dim_scale_factor=3.25,
            rope_theta=500000.0, max_seq_len=16,
        )
        tokens = (np.arange(8, dtype=np.int32)[None, :] % args.vocab_size)
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out)

    def test_with_lora(self):
        args = JaxArgs(
            vocab_size=2048, hidden_dim=64, num_layers=4, num_kv_reuse_layers=1,
            num_heads=4, num_kv_heads=2, hidden_dim_scale_factor=3.25,
            rope_theta=500000.0, max_seq_len=32,
            use_lora=True, lora_rank=4, lora_alpha=4.0, lora_dtype="float32",
        )
        tokens = np.arange(1, 17, dtype=np.int32).reshape(1, 16) % args.vocab_size
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out, atol=1.5e-1, rtol=1.5e-1)


if __name__ == "__main__":
    absltest.main()
