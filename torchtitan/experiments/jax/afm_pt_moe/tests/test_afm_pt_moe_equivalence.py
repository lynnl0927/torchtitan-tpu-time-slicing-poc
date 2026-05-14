"""Equivalence test: JAX AFM PT MoE vs. TAMM/PyTorch AFMParallelTrackMoE.

Strategy (mirrors ``jax/afmv7/tests/test_afmv7_equivalence.py``):
  1. Build a tiny TAMM PyTorch ``AFMParallelTrackMoE``. Its random init is
     our "checkpoint".
  2. Build an equivalent Flax NNX model
     (``torchtitan.experiments.jax.afm_pt_moe.Transformer``).
  3. Translate the PyTorch state_dict to the JAX parameter layout. Both
     backends store ``VectorizedLinear`` weights as
     ``[num_tracks, in, out]`` and use identical container names
     (``layers.segment_X.layer_Y.attention.qkv_transform.fused_linear.weight``
     etc.), so the only translation rules are
       - ``embedding.weight``       → JAX ``embedding/embedding``  (no transpose)
       - ``output_norm.weight``     → JAX ``output_norm/weight``
       - ``output_transform.weight``→ tied LM head; copied into PT's
         ``output_transform`` from PT's ``embedding`` before forward so the
         PT model matches the JAX tied head.
       - everything else            → split by ``.``, descend the JAX state.
  4. Forward both models on the same input; compare logits.

To get a well-defined equivalence we configure the TAMM model so its
behaviour matches what the JAX model implements (the JAX model is a
restricted subset — see ``jax/afm_pt_moe/model.py`` docstring):
  * attention pattern         = all ``local_rope``
  * feed_forward pattern      = all ``sparse``  (MoE)
  * ``scale_qk_norm=False``   — no per-head QK norm
  * ``tracks_dispatch_norm`` / ``tracks_combine_norm`` = ``rms_norm``
  * ``tracks_combine_op='sum'``
  * ``pre_residual_norm='rms_norm'`` — TAMM adds a per-residual norm,
    matching JAX's ``residual_connection.pre_residual_norm``.
"""

import numpy as np
import jax
import jax.numpy as jnp
import torch
from absl.testing import absltest
from flax import nnx

from torchtitan.experiments.jax import afm_pt_moe

JaxArgs = afm_pt_moe.AFMPTMoeModelArgs
JaxTransformer = afm_pt_moe.Transformer


# ---------------------------------------------------------------------------
# PyTorch / TAMM model builder
# ---------------------------------------------------------------------------

def _build_pt_model(args: JaxArgs) -> torch.nn.Module:
    """Build a TAMM PyTorch AFM PT MoE matching ``args``. The model is
    instantiated with default random init (used as the "checkpoint" to load
    into the JAX model)."""
    import tamm.models.afm_text.afm_pt_moe as afm_pt_moe_mod

    cfg = afm_pt_moe_mod.AFMParallelTrackMoEConfig(
        vocab_size=args.vocab_size,
        num_tracks=args.num_tracks,
        num_layers_per_track=args.num_layers_per_track,
        num_layers_per_track_per_sync_point=args.num_layers_per_track_per_sync_point,
        hidden_dim=args.hidden_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        dense_feed_forward_hidden_dim=args.dense_feed_forward_hidden_dim,
        sparse_feed_forward_hidden_dim=args.sparse_feed_forward_hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        rope_theta=args.rope_theta,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        # Constraints needed to align with the JAX subset (see file docstring).
        attention_layer_pattern=("local_rope",),
        feed_forward_layer_pattern=("sparse",),
        scale_qk_norm=False,
        tracks_dispatch_norm="rms_norm",
        tracks_combine_norm="rms_norm",
        tracks_combine_op="sum",
        pre_residual_norm="rms_norm",
        local_attention_window_size=args.local_attention_window_size or 4096,
        experts_router_logits_cap=None,
        pretrained=False,
    )
    torch.manual_seed(0)
    model = cfg.create_basic_builder().build()
    model.eval()

    # The JAX model uses a tied LM head (``logits = h @ embedding.T``); TAMM
    # has a separate ``output_transform``. Tie them by copying the embedding
    # weight into ``output_transform.weight`` so both models project through
    # the same matrix.
    with torch.no_grad():
        model.output_transform.weight.data.copy_(model.embedding.weight.data)

    # TAMM AFM PT MoE hardcodes ``apply_qk_norm=True`` and applies an unscaled
    # ``RMSNorm`` to Q/K even when ``scale_qk_norm=False`` (the config flag
    # only suppresses the learnable scale, not the normalization itself —
    # see ``tamm/models/afm_text/afm_pt_moe.py::_create_local_rope_attention``).
    # The JAX model has no QK norm, so disable it on the PT side by setting
    # the inner norms to ``None``; the surrounding ``QKNorm`` then degenerates
    # to ``add_heads → remove_heads`` and passes Q/K through unchanged.
    for module in model.modules():
        if type(module).__name__ == "QKNorm":
            module.query_norm = None
            module.key_norm = None
    return model


# ---------------------------------------------------------------------------
# State-dict translation: PyTorch → JAX
# ---------------------------------------------------------------------------

def _pt_key_to_jax_path(pt_key: str):
    """Translate a PyTorch state_dict key to a JAX parameter path.

    Returns ``jax_path_tuple`` or ``None`` if the param has no JAX counterpart
    (e.g. ``output_transform.weight`` which is tied via ``embedding`` on the
    JAX side). Raises ``KeyError`` if the key is not recognised so forgotten
    mappings surface loudly instead of silently skipping a parameter.

    No transpose is needed for any AFM PT MoE parameter: both backends store
    ``VectorizedLinear`` weights as ``[num_tracks, in, out]`` (see
    ``torchtitan.experiments.jax.afm_pt_moe.model.VectorizedLinear``
    docstring) and use the same layout for the 4D expert weights.
    """
    if pt_key == "embedding.weight":
        return ("embedding", "embedding")

    if pt_key == "output_transform.weight":
        # Tied via embedding on the JAX side; PT was forced-tied in
        # ``_build_pt_model`` so the input to this matmul is identical, but
        # there is no JAX state to write.
        return None

    if pt_key == "output_norm.weight":
        return ("output_norm", "weight")

    parts = pt_key.split(".")
    if parts[0] != "layers":
        raise KeyError(f"Unhandled PT key: {pt_key}")

    # ``layers.segment_<S>....`` — both backends use the same container
    # names, so the only difference is ``.`` vs ``/`` between path
    # components. Pass attribute strings through unchanged.
    return tuple(parts)


def _load_pt_checkpoint_into_jax(
    pt_state: dict, jax_model: JaxTransformer
) -> None:
    """Overwrite JAX parameters from a PyTorch state_dict."""
    state = nnx.state(jax_model, nnx.Param)

    for pt_key, pt_tensor in pt_state.items():
        jax_path = _pt_key_to_jax_path(pt_key)
        if jax_path is None:
            continue
        arr = pt_tensor.detach().to(torch.float32).cpu().numpy()

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
        # fp32 master params so the transfer from fp32 torch weights is exact.
        # Note: the JAX model still casts hidden states to bf16 inside
        # ``Transformer._forward_hidden`` (see model.py:915), so the forward
        # is mixed-precision regardless. Tolerances are set accordingly.
        jax_model = JaxTransformer(args, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
        _load_pt_checkpoint_into_jax(pt_model.state_dict(), jax_model)
        tokens_jax = jnp.asarray(tokens_np, dtype=jnp.int32)
        jax_out = np.asarray(jax.device_get(jax_model(tokens_jax))).astype(np.float32)

    with torch.no_grad():
        pt_tokens = torch.from_numpy(tokens_np).to(torch.long)
        pt_out = pt_model(pt_tokens).predictions.to(torch.float32).numpy()

    return pt_out, jax_out


class AFMPTMoeEquivalenceTest(absltest.TestCase):

    def _assert_close(self, pt_out, jax_out, atol=2e-1, rtol=2e-1):
        self.assertEqual(pt_out.shape, jax_out.shape)
        diff = np.abs(pt_out - jax_out)
        np.testing.assert_allclose(
            pt_out, jax_out, atol=atol, rtol=rtol,
            err_msg=f"max abs diff={diff.max():.4f}"
        )

    def test_tiny_moe(self):
        """End-to-end equivalence on the smallest viable AFM PT MoE config
        (2 tracks × 2 layers × 2 experts, top-2 routing). Exercises every
        component on both sides: embedding → segment dispatch norm →
        per-track attention (vectorized fused-QKV, RoPE, vmapped causal SDPA,
        per-track output projection, pre-residual norm) → sparse MoE FFN
        (norm → router → dispatch/combine → expert gate/up/down) →
        segment combine norm → output_norm → tied LM head.

        ``test_multi_segment`` and ``test_route_to_all_experts`` are narrower
        variants that change *one* axis of this baseline (segment count and
        ``num_experts``/``k`` respectively) to isolate per-axis regressions."""
        args = JaxArgs(
            vocab_size=64,
            num_tracks=2,
            num_layers_per_track=2,
            num_layers_per_track_per_sync_point=2,
            hidden_dim=16,
            attention_hidden_dim=16,
            dense_feed_forward_hidden_dim=16,
            sparse_feed_forward_hidden_dim=8,
            num_heads=2,
            num_kv_heads=2,
            num_experts=2,
            num_experts_per_token=2,
            rope_theta=500000.0,
            max_seq_len=16,
            local_attention_window_size=16,
        )
        tokens = (np.arange(1, 9, dtype=np.int32)[None, :] % args.vocab_size)
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out)

    def test_route_to_all_experts(self):
        """``num_experts_per_token == num_experts`` — every token goes to every
        expert. The router weights are a proper softmax over all experts in
        both backends in this regime, so the two implementations agree
        end-to-end. (When ``k < num_experts`` they diverge: TAMM computes
        ``softmax`` over **all** experts then restricts to the chosen top-k
        indices, while ``jax.afm_pt_moe._topk_router`` does ``top_k`` first
        and then ``softmax`` over only the top-k logits — see model.py:317.
        That divergence is not exercised here on purpose.)
        """
        args = JaxArgs(
            vocab_size=64,
            num_tracks=2,
            num_layers_per_track=2,
            num_layers_per_track_per_sync_point=2,
            hidden_dim=16,
            attention_hidden_dim=16,
            dense_feed_forward_hidden_dim=16,
            sparse_feed_forward_hidden_dim=8,
            num_heads=2,
            num_kv_heads=2,
            num_experts=4,
            num_experts_per_token=4,
            rope_theta=500000.0,
            max_seq_len=16,
            local_attention_window_size=16,
        )
        tokens = (np.arange(1, 9, dtype=np.int32)[None, :] % args.vocab_size)
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out)

    def test_multi_segment(self):
        """4 layers per track with sync every 2 → 2 segments."""
        args = JaxArgs(
            vocab_size=128,
            num_tracks=2,
            num_layers_per_track=4,
            num_layers_per_track_per_sync_point=2,
            hidden_dim=16,
            attention_hidden_dim=16,
            dense_feed_forward_hidden_dim=16,
            sparse_feed_forward_hidden_dim=8,
            num_heads=2,
            num_kv_heads=2,
            num_experts=2,
            num_experts_per_token=2,
            rope_theta=500000.0,
            max_seq_len=16,
            local_attention_window_size=16,
        )
        tokens = (np.arange(1, 9, dtype=np.int32)[None, :] % args.vocab_size)
        pt_out, jax_out = _run_pair(args, tokens)
        self._assert_close(pt_out, jax_out)


if __name__ == "__main__":
    absltest.main()
