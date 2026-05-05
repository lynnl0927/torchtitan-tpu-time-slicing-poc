"""AFM PT MoE in pure JAX using Flax NNX.

Mirrors tamm.models.afm_text.afm_pt_moe.AFMParallelTrackMoEConfig closely
enough that the parameter layout matches what the torchax bridge expects
(experiments/torchax/afm_pt_moe/sharding.py). Specifically:

  * Multi-track parallel architecture: every per-layer Linear is "vectorized"
    along a leading ``num_tracks`` axis, producing 3D weights for dense linears
    and 4D weights for MoE expert linears.
  * Mixed local/global attention: ``attention_layer_pattern`` cycles through
    a small ordered list (e.g. ``["local_rope", "local_rope", "local_rope",
    "global_nope"]``) — local layers use sliding-window attention with RoPE,
    global layers use causal attention with NO position encoding.
  * Sparse MoE FFN: top-K router (3D weights) + expert dispatch + per-expert
    matmul via a masked ``segment_matmul`` impl (mirrors the working
    ``_segment_matmul_masked`` from torchax/afm_pt_moe/sitecustomize.py).
    Combines back via assignment_weights·v dot.
  * Per-segment dispatch & combine norms split a single (B, S, hidden) hidden
    state into ``num_tracks`` parallel hidden states and merge them again.
  * Pre-norm layers + per-residual norm before each residual add.
  * Tied embedding / output head (lm_head.weight = embedding.weight.T).

Per-conversation user choices that drive this implementation:
  * **No scan, just unrolled loops** (option 1c) — segments and per-track-layers
    are nested Python ``for`` loops at construction time. Trades compile cost
    for trace/debug simplicity. Revisit later if compile latency becomes the
    binding constraint.
  * **Splash with LocalMask** (option 2a) — local-attention layers use
    ``splash_attention_mask.LocalMask`` so the Pallas kernel sees the windowed
    sparsity pattern directly; falls back to dense banded SDPA when no
    ``attn_fn`` is supplied (e.g. CPU smoke tests).

LoRA is intentionally omitted in this first pass — neither 3b nor 24b enable
it; if you need it, mirror the torchax/afm_pt_moe LoRA-on-attention-only
pattern from torchtitan.experiments.tpu.afm_pt_moe.model.model.AFMPTMoeWrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import ad_checkpoint
from flax import nnx

from .args import AFMPTMoeModelArgs


# =============================================================================
# Dtype helpers (shared with afmv7)
# =============================================================================

_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


# =============================================================================
# RoPE (interleaved pair rotation, matches tamm.layers.rope) — used only on
# the local-attention layers (global_nope skips RoPE entirely).
# =============================================================================

def precompute_rope_coefficients(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> jax.Array:
    """Return ``[max_seq_len, head_dim // 2, 2, 2]`` rotation matrices.

    Same algorithm as jax/afmv7's `precompute_rope_coefficients`.
    """
    half = head_dim // 2
    exponents = jnp.arange(half, dtype=jnp.float32) * (-2.0 / head_dim)
    freqs = jnp.power(jnp.float32(theta), exponents)
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    row0 = jnp.stack([cos, -sin], axis=-1)
    row1 = jnp.stack([sin, cos], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def apply_rope(x: jax.Array, rope: jax.Array) -> jax.Array:
    """Apply RoPE to ``[..., n_heads, head_dim]`` using adjacent pair rotation.

    Works for any leading shape (BATCH, [TRACK,] SEQ, HEADS, DIM).
    """
    input_dtype = x.dtype
    *leading, n_heads, head_dim = x.shape
    seq_len = x.shape[-3]
    x_pairs = x.reshape(*leading, n_heads, head_dim // 2, 2)
    # rope: [seq, h/2, 2, 2]; broadcast to [..., seq, 1, h/2, 2, 2].
    # We rely on the second-to-last leading dim being SEQ.
    extra = (1,) * (len(leading) - 1)
    rope_b = rope[(*([None] * (len(leading) - 1)),
                  slice(None), None, slice(None), slice(None), slice(None))].astype(jnp.float32)
    x_pairs_f32 = x_pairs.astype(jnp.float32)
    # einsum index pattern depends on rank; use explicit dims via moveaxis-free
    # contraction: rope shape ['seq', 'pair', 'i', 'j'], x_pairs ['seq',
    # 'heads', 'pair', 'j'].
    # Easier: write the contraction on the trailing 4 dims with broadcasting.
    # rotated[..., s, h, p, i] = sum_j rope[s, p, i, j] * x_pairs[..., s, h, p, j]
    rotated = jnp.einsum("...shpj,spij->...shpi", x_pairs_f32, rope[: seq_len].astype(jnp.float32))
    return rotated.reshape(*leading, n_heads, head_dim).astype(input_dtype)


# =============================================================================
# RMSNorm — two flavors: scalar (1D weight) and vectorized-per-track
# (2D weight ``[num_tracks, hidden]``).
# =============================================================================

class RMSNormScalar(nnx.Module):
    """Standard RMSNorm with 1D weight ``[dim]``. fp32 compute."""

    def __init__(self, dim: int, *, eps: float, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        del rngs
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(mean_sq + self.eps)
        return (x_f32 * inv_norm * self.weight[...].astype(jnp.float32)).astype(input_dtype)


class RMSNormVec(nnx.Module):
    """Per-track RMSNorm with 2D weight ``[num_tracks, dim]``.

    Input  ``[..., num_tracks, dim]``  (track axis is second-to-last).
    Output same shape; each track applies its own ``weight[t]``.
    """

    def __init__(self, num_tracks: int, dim: int, *, eps: float,
                 param_dtype: jnp.dtype, rngs: nnx.Rngs):
        del rngs
        self.weight = nnx.Param(jnp.ones((num_tracks, dim), dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(mean_sq + self.eps)
        # weight is [num_tracks, dim]. Broadcast to [..., num_tracks, dim].
        w = self.weight[...].astype(jnp.float32)
        return (x_f32 * inv_norm * w).astype(input_dtype)


# =============================================================================
# Vectorized Linear (per-track):
#   Input weight ``[num_tracks, in, out]``, input ``[..., num_tracks, in]``.
#   Compute  ``out[..., t, j] = sum_i input[..., t, i] * weight[t, i, j]``.
# =============================================================================

class VectorizedLinear(nnx.Module):
    """3D Linear with leading per-track axis.

    Parameter layout matches TAMM's VectorizedLinear:
      ``weight``: ``[num_tracks, in_features, out_features]``.

    Wrapped under ``fused_linear/weight`` to match the param-name for FusedQKV
    (set ``name_prefix="fused_linear"`` to mirror exactly). For everywhere
    else, the path is just ``weight``.
    """

    def __init__(self, num_tracks: int, in_features: int, out_features: int,
                 *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        # Init: kaiming-uniform-like via gaussian with std = 1/sqrt(in).
        std = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (num_tracks, in_features, out_features),
                              dtype=param_dtype) * std
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [..., t, i]; weight: [t, i, j]; out: [..., t, j].
        return jnp.einsum("...ti,tij->...tj", x, self.weight[...])


class VectorizedFusedQKVLinear(nnx.Module):
    """Wraps a single VectorizedLinear under ``fused_linear`` and exposes
    Q/K/V split. Param path: ``fused_linear/weight`` ``[num_tracks, hidden,
    q_out + k_out + v_out]``.
    """

    def __init__(self, num_tracks: int, in_features: int,
                 *, q_out: int, k_out: int, v_out: int,
                 param_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.output_dims = (q_out, k_out, v_out)
        total_out = q_out + k_out + v_out
        self.fused_linear = VectorizedLinear(
            num_tracks, in_features, total_out,
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        fused_out = self.fused_linear(x)
        q_out, k_out, v_out = self.output_dims
        q = fused_out[..., :q_out]
        k = fused_out[..., q_out: q_out + k_out]
        v = fused_out[..., q_out + k_out:]
        return q, k, v


class VectorizedMultiOutputLinear(nnx.Module):
    """Parallel per-track linears with independent outputs (e.g. SwiGLU
    gate/up). Param paths: ``linear_0/weight``, ``linear_1/weight``, ...
    """

    def __init__(self, num_tracks: int, in_features: int, output_dims: list[int],
                 *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.output_dims = tuple(output_dims)
        for i, out_dim in enumerate(output_dims):
            setattr(self, f"linear_{i}",
                    VectorizedLinear(num_tracks, in_features, out_dim,
                                     param_dtype=param_dtype, rngs=rngs))

    def __call__(self, x: jax.Array) -> tuple:
        return tuple(getattr(self, f"linear_{i}")(x)
                     for i in range(len(self.output_dims)))


# =============================================================================
# Expert Linear (4D weight: ``[num_tracks, num_experts, in, out]``).
# =============================================================================

class ExpertVectorizedLinear(nnx.Module):
    """Holds the expert weight tensor; the actual matmul is done by
    ``masked_segment_matmul`` outside this class (which needs the assignment
    pattern, not just the weight)."""

    def __init__(self, num_tracks: int, num_experts: int, in_features: int,
                 out_features: int, *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        std = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(),
                              (num_tracks, num_experts, in_features, out_features),
                              dtype=param_dtype) * std
        )


class ExpertMultiOutputLinear(nnx.Module):
    """Parallel expert linears (gate/up for MoE FFN)."""

    def __init__(self, num_tracks: int, num_experts: int, in_features: int,
                 output_dims: list[int], *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.output_dims = tuple(output_dims)
        for i, out_dim in enumerate(output_dims):
            setattr(self, f"linear_{i}",
                    ExpertVectorizedLinear(num_tracks, num_experts, in_features, out_dim,
                                           param_dtype=param_dtype, rngs=rngs))


# =============================================================================
# Masked segment_matmul — JIT-friendly per-expert matmul + cumsum mask.
#
# Mirrors `_segment_matmul_masked` from torchax/afm_pt_moe/sitecustomize.py.
# Inputs:
#   segmented_input: [num_tracks, total_tokens, in_features]   (token-major)
#   group_sizes:     [num_tracks, num_experts] int32 — number of tokens routed
#                    to expert e on track t (sum over e == total_tokens for
#                    each track).
#   weight:          [num_tracks, num_experts, in_features, out_features]
# Output:
#   output:          [num_tracks, total_tokens, out_features]
#
# Implementation: derive a per-token expert id from cumsum(group_sizes), then
# loop over experts, masking out the rows that don't belong to that expert.
# Quadratic in num_experts but works for the small num_experts=4 case.
# =============================================================================

def masked_segment_matmul(
    segmented_input: jax.Array,
    group_sizes: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    tracks, total_tokens, in_features = segmented_input.shape
    _, num_experts = group_sizes.shape
    out_features = weight.shape[-1]

    cum = jnp.cumsum(group_sizes.astype(jnp.int32), axis=-1)  # [t, e]
    # Token-position grid: [1, T, 1].
    positions = jnp.arange(total_tokens, dtype=jnp.int32)[None, :, None]
    # Expert id for each token: number of cum-thresholds <= position, clamped.
    ge = positions >= cum[:, None, :]  # [t, T, e]
    # Use ``min=``/``max=`` (jax 0.9.x dropped the legacy ``a_min``/``a_max``
    # kwargs from jnp.clip).
    token_expert_id = jnp.clip(ge.sum(axis=-1).astype(jnp.int32),
                               min=0, max=num_experts - 1)  # [t, T]

    output = jnp.zeros(
        (tracks, total_tokens, out_features), dtype=segmented_input.dtype
    )
    for expert_idx in range(num_experts):
        # Cast mask to compute dtype to avoid an fp32 broadcast.
        mask = (token_expert_id == expert_idx)[..., None].astype(segmented_input.dtype)
        # Per-expert matmul: [t, T, in] @ [t, in, out] = [t, T, out].
        expert_out = jnp.einsum("tTi,tij->tTj", segmented_input, weight[:, expert_idx])
        output = output + mask * expert_out
    return output


# =============================================================================
# Top-K router + dispatch / combine — mirrors the torchax-side
# DroplessMixtureOfExpertsDispatcher monkey-patches in
# torchax/afm_pt_moe/sitecustomize.py.
# =============================================================================

def _topk_router(
    hidden: jax.Array,                  # [t, B, S, H]
    router_weight: jax.Array,           # [t, H, num_experts]
    num_experts_per_token: int,
    logits_cap: Optional[float],
) -> tuple[jax.Array, jax.Array]:
    """Returns (expert_assignments, assignment_weights).

    expert_assignments: int32 ``[t, B, S, k]`` — which experts each token goes to.
    assignment_weights: float ``[t, B, S, k]`` — softmax weight per assigned expert.
    """
    # logits: [t, B, S, num_experts]
    logits = jnp.einsum("tbsh,the->tbse", hidden, router_weight)
    if logits_cap is not None:
        logits = logits_cap * jnp.tanh(logits / logits_cap)
    # Top-K experts per token.
    top_logits, top_indices = jax.lax.top_k(logits, num_experts_per_token)
    # Softmax over the top-K logits — per-token assignment weights.
    weights = jax.nn.softmax(top_logits.astype(jnp.float32), axis=-1).astype(hidden.dtype)
    return top_indices.astype(jnp.int32), weights


def _moe_dispatch_combine(
    hidden: jax.Array,                       # [t, B, S, H]
    expert_assignments: jax.Array,           # [t, B, S, k]
    assignment_weights: jax.Array,           # [t, B, S, k]
    num_experts: int,
    expert_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Run the expert pipeline per batch via ``jax.vmap`` over axis B.

    Naive flattening of (B, S, k) into a single 1D token axis loses the SPMD
    batch-sharding annotation when XLA's GSPMD partitioner traces the
    program — it sees a single non-sharded ``B*S*k``-wide axis and
    fully-replicates the resulting tensor. On a v6e-32 mesh that turns a
    per-chip ~128 MB allocation into a 4 GB one (×6 such buffers in flight
    = 24 GB blown on MoE dispatch alone). Iterating per-batch via
    ``jax.vmap(in_axes=1, out_axes=1)`` keeps batch as an explicit axis so
    sharding propagates cleanly.

    expert_fn signature: ``(segmented_input [t, T_per_b, H], group_sizes
    [t, e]) -> expert_output [t, T_per_b, H]``, where T_per_b = S * k.
    """
    t, B, S, H = hidden.shape
    k = expert_assignments.shape[-1]

    def _per_batch(h_b, ea_b, aw_b):
        # h_b:  [t, S, H]; ea_b: [t, S, k]; aw_b: [t, S, k].
        # Replicate each token k times → [t, S, k, H].
        x_expanded = jnp.broadcast_to(h_b[..., None, :], (t, S, k, H))
        flat = x_expanded.reshape(t, S * k, H)
        ea = ea_b.reshape(t, S * k)

        sort_idx = jnp.argsort(ea, axis=-1)
        inv_sort = jnp.argsort(sort_idx, axis=-1)
        sorted_x = jnp.take_along_axis(flat, sort_idx[..., None], axis=1)

        one_hot = jax.nn.one_hot(ea, num_experts, dtype=jnp.int32)
        group_sizes = one_hot.sum(axis=1)

        expert_out = expert_fn(sorted_x, group_sizes)

        unsorted = jnp.take_along_axis(expert_out, inv_sort[..., None], axis=1)
        unsorted = unsorted.reshape(t, S, k, H)

        weights = aw_b.astype(unsorted.dtype)
        return jnp.einsum("tskh,tsk->tsh", unsorted, weights)

    # vmap over batch axis B (axis 1 in [t, B, S, *]).
    return jax.vmap(_per_batch, in_axes=1, out_axes=1)(
        hidden, expert_assignments, assignment_weights
    )


# =============================================================================
# Attention block (vectorized over tracks).
#
# Two flavors:
#   * "local_rope": apply RoPE, then attention masked to a sliding window
#     of width ``local_attention_window_size`` (also causal).
#   * "global_nope": NO RoPE, full causal attention.
#
# The choice is set at construction time by ``attention_type``.
# =============================================================================

class VecAttention(nnx.Module):
    """Per-track multi-head attention. 3D Q/K/V/O weights along track axis.

    Param paths matching TAMM:
        norm.weight                              [num_tracks, hidden]
        qkv_transform.fused_linear.weight        [num_tracks, hidden, q+k+v]
        output_transform.weight                  [num_tracks, attn_hidden, hidden]
        residual_connection.pre_residual_norm.weight  [num_tracks, hidden]
    """

    def __init__(
        self,
        args: AFMPTMoeModelArgs,
        attention_type: str,
        *,
        attn_fn: Optional[Callable],
        attn_fn_local: Optional[Callable] = None,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        if attention_type not in ("local_rope", "global_nope"):
            raise ValueError(f"unknown attention_type {attention_type!r}; expected "
                             "'local_rope' or 'global_nope'")
        self.attention_type = attention_type
        self.num_tracks = args.num_tracks
        self.n_heads = args.num_heads
        self.n_kv_heads = args.effective_num_kv_heads
        self.head_dim = args.head_dim
        self.attn_hidden = args.attention_hidden_dim
        self.local_window = args.local_attention_window_size
        self._attn_fn_global = attn_fn        # CausalMask splash, may be None on CPU
        self._attn_fn_local = attn_fn_local   # LocalMask splash, may be None on CPU
        self.eps = args.norm_eps if args.norm_eps is not None else 1e-5

        self.norm = RMSNormVec(args.num_tracks, args.hidden_dim,
                               eps=self.eps, param_dtype=param_dtype, rngs=rngs)
        self.qkv_transform = VectorizedFusedQKVLinear(
            args.num_tracks, args.hidden_dim,
            q_out=self.n_heads * self.head_dim,
            k_out=self.n_kv_heads * self.head_dim,
            v_out=self.n_kv_heads * self.head_dim,
            param_dtype=param_dtype, rngs=rngs,
        )
        # Output projection: [num_tracks, n_heads*head_dim, hidden_dim].
        self.output_transform = VectorizedLinear(
            args.num_tracks, self.n_heads * self.head_dim, args.hidden_dim,
            param_dtype=param_dtype, rngs=rngs,
        )
        # The post-attention "pre-residual" norm.
        self.residual_connection = _ResidualNormHolder(
            RMSNormVec(args.num_tracks, args.hidden_dim, eps=self.eps,
                       param_dtype=param_dtype, rngs=rngs))

    def __call__(self, x: jax.Array, rope: Optional[jax.Array]) -> jax.Array:
        """``x``: ``[num_tracks, B, S, hidden]``. Returns same shape."""
        # 1. pre-norm.
        h = self.norm(x.swapaxes(0, -2))  # tmp: move track to second-to-last for norm
        # Actually, RMSNormVec expects [..., t, dim]; x is [t, B, S, H]. Move track
        # to second-to-last by transposing.
        # ↑ The above swap is incorrect — do it properly:
        h = jnp.einsum("...->...", x)  # noop; placeholder so code below uses x_orig
        # Recompute via a track-last view: [B, S, t, H].
        x_tslast = jnp.moveaxis(x, 0, -2)
        h_tslast = self.norm(x_tslast)  # [B, S, t, H]
        # Back to [t, B, S, H].
        h = jnp.moveaxis(h_tslast, -2, 0)

        # 2. QKV projection. Need x with [..., t, in] order; current is
        # [t, B, S, H], so move t to be second-to-last for the matmul.
        h_for_qkv = jnp.moveaxis(h, 0, -2)        # [B, S, t, H]
        q, k, v = self.qkv_transform(h_for_qkv)   # each [B, S, t, *]
        # Move track back to leading: [t, B, S, *].
        q = jnp.moveaxis(q, -2, 0)
        k = jnp.moveaxis(k, -2, 0)
        v = jnp.moveaxis(v, -2, 0)

        # 3. RoPE on local layers only.
        if self.attention_type == "local_rope":
            assert rope is not None, "local_rope layers need a rope tensor"
            # Reshape to per-head and apply rotation.
            t, B, S, _ = q.shape
            q = q.reshape(t, B, S, self.n_heads, self.head_dim)
            k = k.reshape(t, B, S, self.n_kv_heads, self.head_dim)
            q = apply_rope(q, rope[:S])
            k = apply_rope(k, rope[:S])
            q = q.reshape(t, B, S, self.n_heads * self.head_dim)
            k = k.reshape(t, B, S, self.n_kv_heads * self.head_dim)
        # global_nope: no RoPE.

        # 4. Attention. Process tracks via jax.vmap rather than packing
        # ``t * B`` into a single leading axis — the reshape was breaking SPMD
        # sharding propagation (XLA loses the batch-sharding annotation when
        # two axes are merged into one), causing 32x memory blowup at compile
        # time. With vmap, sharding on B propagates per-track call cleanly.
        # Inputs to vmap are [t, B, S, *] with vmap over axis 0 (t).
        attention_type = self.attention_type
        local_window = self.local_window
        # Pick splash kernel by attention type: global_nope → CausalMask
        # variant; local_rope → LocalMask variant. Both are ``None`` on CPU /
        # when --jax_config.use_splash_attention_kernel=False, in which case
        # _scaled_dot_product_attention falls back to dense banded SDPA.
        if self.attention_type == "global_nope":
            attn_fn = self._attn_fn_global
        else:
            attn_fn = self._attn_fn_local

        def _per_track_attn(q_t, k_t, v_t):
            return _scaled_dot_product_attention(
                q_t, k_t, v_t,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                attention_type=attention_type,
                local_window=local_window,
                attn_fn=attn_fn,
            )

        # vmap over leading t dim → output shape [t, B, S, n_heads*head_dim].
        attn_out = jax.vmap(_per_track_attn)(q, k, v)

        # 5. Output projection.
        attn_out_tslast = jnp.moveaxis(attn_out, 0, -2)  # [B, S, t, n_heads*hd]
        out_tslast = self.output_transform(attn_out_tslast)  # [B, S, t, hidden]
        out = jnp.moveaxis(out_tslast, -2, 0)            # [t, B, S, hidden]

        # 6. Pre-residual norm + residual add.
        out_norm = self.residual_connection.pre_residual_norm(
            jnp.moveaxis(out, 0, -2)
        )
        out = jnp.moveaxis(out_norm, -2, 0)
        return x + out


class _ResidualNormHolder(nnx.Module):
    """Tiny container so the param path lands at
    ``residual_connection/pre_residual_norm/weight`` rather than directly under
    the parent. Mirrors TAMM's structure."""

    def __init__(self, pre_residual_norm: RMSNormVec):
        self.pre_residual_norm = pre_residual_norm


def _scaled_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array,
    *, n_heads: int, n_kv_heads: int, head_dim: int,
    attention_type: str,
    local_window: Optional[int],
    attn_fn: Optional[Callable],
) -> jax.Array:
    """Causal SDPA with optional local windowing. Shapes:
        q: [B, S, n_heads * head_dim]
        k: [B, S, n_kv_heads * head_dim]
        v: [B, S, n_kv_heads * head_dim]
    Returns [B, S, n_heads * head_dim].
    """
    B, S, _ = q.shape

    q = q.reshape(B, S, n_heads, head_dim)
    k = k.reshape(B, S, n_kv_heads, head_dim)
    v = v.reshape(B, S, n_kv_heads, head_dim)

    if attn_fn is not None:
        # Splash kernel path. The caller chose the right mask variant
        # (CausalMask for global_nope, LocalMask for local_rope) and bound
        # it into ``attn_fn``. Same call shape either way.
        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)
        out = attn_fn(q_t, k_t, v_t, None)
        out = out.transpose(0, 2, 1, 3)
        return out.reshape(B, S, n_heads * head_dim)

    # Dense fallback. Tile K/V to match Q head count for GQA.
    n_rep = n_heads // n_kv_heads
    if n_rep != 1:
        k = jnp.repeat(k, n_rep, axis=2)
        v = jnp.repeat(v, n_rep, axis=2)
    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bshd,bthd->bsht", q, k) * scale

    # Build mask: causal, plus optional local windowing.
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    if attention_type == "local_rope" and local_window is not None:
        # Sliding-window mask: token t can attend to [t - window + 1, t].
        idx = jnp.arange(S)
        window = (idx[:, None] - idx[None, :]) < local_window
        causal = causal & window
    scores = jnp.where(causal[None, :, None, :], scores, jnp.finfo(scores.dtype).min)
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.einsum("bsht,bthd->bshd", attn, v)
    return out.reshape(B, S, n_heads * head_dim)


# =============================================================================
# Sparse MoE FeedForward block.
#
# Param paths matching TAMM:
#   norm.weight                                              [num_tracks, hidden]
#   router.input_transform.weight                            [num_tracks, hidden, num_experts]
#   experts.hidden_transform.linear_0.weight                 [t, e, hidden, ffn_hidden]
#   experts.hidden_transform.linear_1.weight                 [t, e, hidden, ffn_hidden]
#   experts.output_transform.weight                          [t, e, ffn_hidden, hidden]
#   residual_connection.pre_residual_norm.weight             [num_tracks, hidden]
# =============================================================================

class _Router(nnx.Module):
    def __init__(self, num_tracks: int, hidden_dim: int, num_experts: int,
                 *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.input_transform = VectorizedLinear(
            num_tracks, hidden_dim, num_experts,
            param_dtype=param_dtype, rngs=rngs,
        )


class _Experts(nnx.Module):
    def __init__(self, num_tracks: int, num_experts: int, hidden_dim: int,
                 ffn_hidden_dim: int, *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        # Two parallel branches (gate and up) — SwiGLU style.
        self.hidden_transform = ExpertMultiOutputLinear(
            num_tracks, num_experts, hidden_dim, [ffn_hidden_dim, ffn_hidden_dim],
            param_dtype=param_dtype, rngs=rngs,
        )
        self.output_transform = ExpertVectorizedLinear(
            num_tracks, num_experts, ffn_hidden_dim, hidden_dim,
            param_dtype=param_dtype, rngs=rngs,
        )


class SparseMoEFeedForward(nnx.Module):
    """Top-K-routed sparse MoE FFN over per-track hidden states."""

    def __init__(
        self,
        args: AFMPTMoeModelArgs,
        *,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.num_tracks = args.num_tracks
        self.num_experts = args.num_experts
        self.num_experts_per_token = args.num_experts_per_token
        self.logits_cap = args.experts_router_logits_cap
        self.eps = args.norm_eps if args.norm_eps is not None else 1e-5

        self.norm = RMSNormVec(args.num_tracks, args.hidden_dim, eps=self.eps,
                               param_dtype=param_dtype, rngs=rngs)
        self.router = _Router(args.num_tracks, args.hidden_dim, args.num_experts,
                              param_dtype=param_dtype, rngs=rngs)
        self.experts = _Experts(
            args.num_tracks, args.num_experts, args.hidden_dim,
            args.sparse_feed_forward_hidden_dim,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.residual_connection = _ResidualNormHolder(
            RMSNormVec(args.num_tracks, args.hidden_dim, eps=self.eps,
                       param_dtype=param_dtype, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        """``x``: ``[num_tracks, B, S, hidden]``. Returns same shape."""
        # 1. pre-norm.
        h_tslast = self.norm(jnp.moveaxis(x, 0, -2))     # [B, S, t, H]
        h = jnp.moveaxis(h_tslast, -2, 0)                # [t, B, S, H]

        # 2. Router → (assignments, weights).
        router_w = self.router.input_transform.weight[...]
        expert_assignments, assignment_weights = _topk_router(
            h, router_w, self.num_experts_per_token, self.logits_cap,
        )

        # 3. expert_fn — runs gate/up/down on the sorted tokens.
        gate_w = self.experts.hidden_transform.linear_0.weight[...]
        up_w = self.experts.hidden_transform.linear_1.weight[...]
        down_w = self.experts.output_transform.weight[...]

        def expert_fn(sorted_x, group_sizes):
            gate = masked_segment_matmul(sorted_x, group_sizes, gate_w)
            up = masked_segment_matmul(sorted_x, group_sizes, up_w)
            inner = jax.nn.silu(gate) * up
            return masked_segment_matmul(inner, group_sizes, down_w)

        # 4. Dispatch + combine.
        ffn_out = _moe_dispatch_combine(
            h, expert_assignments, assignment_weights,
            num_experts=self.num_experts, expert_fn=expert_fn,
        )

        # 5. Pre-residual norm + residual add.
        ffn_out = jnp.moveaxis(
            self.residual_connection.pre_residual_norm(jnp.moveaxis(ffn_out, 0, -2)),
            -2, 0,
        )
        return x + ffn_out


# =============================================================================
# Single TransformerLayer = Attention + FFN (FFN type per
# ``feed_forward_layer_pattern`` — for AFM PT MoE 24b/3b this is always
# ``["sparse"]``, so MoE).
# =============================================================================

class VecTransformerLayer(nnx.Module):
    def __init__(
        self,
        args: AFMPTMoeModelArgs,
        *,
        attention_type: str,
        attn_fn: Optional[Callable],
        attn_fn_local: Optional[Callable] = None,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.attention = VecAttention(
            args, attention_type=attention_type, attn_fn=attn_fn,
            attn_fn_local=attn_fn_local,
            param_dtype=param_dtype, rngs=rngs,
        )
        # Only "sparse" MoE supported for now (matches 24b/3b configs).
        self.feed_forward = SparseMoEFeedForward(
            args, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array, rope: Optional[jax.Array]) -> jax.Array:
        x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
        x = self.attention(x, rope)
        x = self.feed_forward(x)
        return x


# =============================================================================
# Segment = (per-track dispatch norm) + N TransformerLayers + (per-track
# combine norm). The dispatch norm produces ``num_tracks`` per-track copies of
# the input; the combine norm + sum reduces them back.
# =============================================================================

class _SegmentDispatch(nnx.Module):
    """Produce ``[num_tracks, B, S, H]`` from ``[B, S, H]`` via per-track norm.

    Param path: ``input_transform/norm/weight`` ``[num_tracks, hidden]``.
    """

    def __init__(self, num_tracks: int, hidden_dim: int, eps: float,
                 *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        self.norm = RMSNormVec(num_tracks, hidden_dim, eps=eps,
                               param_dtype=param_dtype, rngs=rngs)
        self.num_tracks = num_tracks

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, S, H]. Broadcast to per-track and norm.
        x_per_track = jnp.broadcast_to(x[..., None, :],
                                       x.shape[:-1] + (self.num_tracks, x.shape[-1]))
        x_per_track = self.norm(x_per_track)              # [B, S, t, H]
        return jnp.moveaxis(x_per_track, -2, 0)            # [t, B, S, H]


class _SegmentCombine(nnx.Module):
    """Combine ``[num_tracks, B, S, H]`` → ``[B, S, H]`` via per-track norm
    then sum over the track axis.

    Param path: ``output_transform/norm/weight`` ``[num_tracks, hidden]``.
    """

    def __init__(self, num_tracks: int, hidden_dim: int, eps: float,
                 op: str, *, param_dtype: jnp.dtype, rngs: nnx.Rngs):
        if op != "sum":
            raise NotImplementedError(f"tracks_combine_op={op!r} not implemented")
        self.norm = RMSNormVec(num_tracks, hidden_dim, eps=eps,
                               param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x_tslast = jnp.moveaxis(x, 0, -2)              # [B, S, t, H]
        x_tslast = self.norm(x_tslast)
        return jnp.sum(x_tslast, axis=-2)               # [B, S, H]


class Segment(nnx.Module):
    """A single sync-point segment: dispatch + N per-track layers + combine.

    Param paths:
        input_transform/norm/weight                       [t, H]
        output_transform/norm/weight                      [t, H]
        layer_<i>/...                                     per-layer subtree
    """

    def __init__(
        self,
        args: AFMPTMoeModelArgs,
        n_layers_in_segment: int,
        attention_pattern: list[str],
        *,
        attn_fn: Optional[Callable],
        attn_fn_local: Optional[Callable] = None,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        eps = args.norm_eps if args.norm_eps is not None else 1e-5
        self.input_transform = _SegmentDispatch(
            args.num_tracks, args.hidden_dim, eps=eps,
            param_dtype=param_dtype, rngs=rngs,
        )
        # n_layers_in_segment stored as a name-prefix list of attribute names.
        self.n_layers = n_layers_in_segment
        for i in range(n_layers_in_segment):
            attn_type = attention_pattern[i % len(attention_pattern)]
            layer = VecTransformerLayer(
                args, attention_type=attn_type, attn_fn=attn_fn,
                attn_fn_local=attn_fn_local,
                param_dtype=param_dtype, rngs=rngs,
            )
            setattr(self, f"layer_{i}", layer)
        self.output_transform = _SegmentCombine(
            args.num_tracks, args.hidden_dim, eps=eps,
            op=(args.tracks_combine_op or "sum"),
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array, rope: Optional[jax.Array]) -> jax.Array:
        # Dispatch: [B, S, H] -> [t, B, S, H].
        h = self.input_transform(x)
        # Run per-track layers (unrolled).
        for i in range(self.n_layers):
            layer = getattr(self, f"layer_{i}")
            h = layer(h, rope)
        # Combine: [t, B, S, H] -> [B, S, H].
        return self.output_transform(h)


# =============================================================================
# Top-level transformer.
# =============================================================================

class Transformer(nnx.Module):
    """AFM PT MoE — embedding + N segments + output_norm + tied LM head.

    Constructor signature matches jax/afmv7's ``Transformer`` so that
    jax/trainer.py's ``setup_model`` can swap models with no special-casing.
    ``use_scan`` is accepted for compatibility but currently always treated as
    False (per user choice 1c — unrolled loops). ``checkpoint_policy`` wraps
    each segment's per-track-layer call in ``jax.checkpoint`` (gradient
    checkpointing).
    """

    def __init__(
        self,
        args: AFMPTMoeModelArgs,
        use_scan: bool = False,
        attn_fn: Optional[Callable] = None,
        checkpoint_policy=None,
        *,
        attn_fn_local: Optional[Callable] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if use_scan:
            # Trainer may pass --jax_config.use_scan=True; we accept the flag
            # for parity with afmv7 but the underlying impl is unrolled (the
            # M2-M6 milestone settled on option 1c per the conversation).
            # Surface a single-line note rather than crash.
            import warnings
            warnings.warn(
                "afm_pt_moe.Transformer ignores use_scan=True (no-scan path "
                "only in this implementation; revisit M9+ if compile time "
                "becomes the binding constraint).",
                stacklevel=2,
            )
        self.args = args
        self.checkpoint_policy = checkpoint_policy

        # Embedding (tied with LM head).
        self.embedding = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.hidden_dim,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Number of segments: num_layers_per_track / num_layers_per_track_per_sync_point.
        per_sync = args.num_layers_per_track_per_sync_point
        if args.num_layers_per_track % per_sync != 0:
            raise ValueError(
                f"num_layers_per_track={args.num_layers_per_track} must be divisible by "
                f"num_layers_per_track_per_sync_point={per_sync}"
            )
        self.n_segments = args.num_layers_per_track // per_sync

        attention_pattern = args.attention_layer_pattern or ["local_rope"]

        # Build segments.
        self.layers = _SegmentList()
        for s in range(self.n_segments):
            seg = Segment(
                args, per_sync, attention_pattern,
                attn_fn=attn_fn, attn_fn_local=attn_fn_local,
                param_dtype=param_dtype, rngs=rngs,
            )
            setattr(self.layers, f"segment_{s}", seg)

        eps = args.norm_eps if args.norm_eps is not None else 1e-5
        self.output_norm = RMSNormScalar(
            args.hidden_dim, eps=eps, param_dtype=param_dtype, rngs=rngs,
        )

        # RoPE coefficients are constants — store the spec, recompute inline in
        # `_forward_hidden`. Storing `precompute_rope_coefficients(...)` as
        # `self.rope` would land it in `nnx.state(model)` and confuse sharding
        # (and FSDP all-gather), since NNX captures any jax.Array attribute
        # whether or not it's marked as a Param.
        self._rope_head_dim = args.head_dim
        self._rope_max_seq_len = args.max_seq_len
        self._rope_theta = args.rope_theta

    def _forward_hidden(self, tokens: jax.Array) -> jax.Array:
        """Returns final hidden state ``[B, S, H]`` in bf16."""
        x = self.embedding(tokens).astype(jnp.bfloat16)  # [B, S, H]
        rope = precompute_rope_coefficients(
            self._rope_head_dim, x.shape[1], self._rope_theta,
        )
        policy = self.checkpoint_policy
        for s in range(self.n_segments):
            seg = getattr(self.layers, f"segment_{s}")
            seg_fn = seg
            if policy is not None:
                # Wrap each segment call in jax.checkpoint so the per-track
                # layer activations get rematerialized on backward instead of
                # held in HBM. ``decoder_layer_input`` markers inside the
                # layer's __call__ are picked up by save_and_offload_only_these_names
                # / dots_saveable / nothing_saveable policies just like in
                # jax/afmv7.
                seg_fn = jax.checkpoint(seg_fn, policy=policy)
            x = seg_fn(x, rope)
        return self.output_norm(x)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Forward pass producing logits ``[B, S, vocab]`` (tied head)."""
        h = self._forward_hidden(tokens)
        emb = self.embedding.embedding[...]  # [vocab, hidden]
        return jnp.einsum("bsh,vh->bsv", h.astype(emb.dtype), emb).astype(jnp.bfloat16)

    def compute_loss(
        self,
        tokens: jax.Array,
        labels: jax.Array,
        *,
        chunk_size: int = 512,
        remat_chunks: bool = False,
    ) -> jax.Array:
        """Sum-of-CE loss with chunked output projection. fp32 scalar.

        Mirrors jax/afmv7/model.py::Transformer.compute_loss.
        """
        h = self._forward_hidden(tokens)
        B, S, H = h.shape
        N = B * S
        emb = self.embedding.embedding[...]
        emb_compute = emb.astype(jnp.bfloat16)
        flat = h.reshape(N, H)
        y = labels.reshape(N)

        if N % chunk_size != 0:
            chunk_size = N
        n_chunks = N // chunk_size
        h_chunks = flat.reshape(n_chunks, chunk_size, H)
        y_chunks = y.reshape(n_chunks, chunk_size)

        def chunk_loss(carry, chunk):
            h_c, y_c = chunk
            logits = jnp.einsum("nh,vh->nv", h_c, emb_compute)
            log_z = jax.nn.logsumexp(logits, axis=-1).astype(jnp.float32)
            target = jnp.take_along_axis(
                logits, y_c[:, None], axis=-1
            ).squeeze(-1).astype(jnp.float32)
            loss = (log_z - target).sum()
            return carry + loss, None

        if remat_chunks:
            chunk_loss = jax.checkpoint(chunk_loss)
        total, _ = jax.lax.scan(chunk_loss, jnp.float32(0.0), (h_chunks, y_chunks))
        return total


class _SegmentList(nnx.Module):
    """Holds segments as named attributes ``segment_0``, ``segment_1``, ...
    so the sharding-map regex ``layers.segment_X.*`` matches the produced
    parameter paths exactly. Mirrors how the TAMM model surfaces segments."""

    pass
