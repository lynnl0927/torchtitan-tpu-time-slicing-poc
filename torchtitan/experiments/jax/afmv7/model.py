"""AFMTextV7 in pure JAX using Flax NNX.

Architecture mirrors tamm.models.afm_text.AFMTextV7 so that weights from the
PyTorch/TAMM implementation in torchtitan.experiments.tpu.afmv7 can be loaded
transparently and produce equivalent outputs. Key features:

  * RMSNorm with fp32 compute
  * Interleaved RoPE (adjacent pairs rotated — matches tamm.layers.rope)
  * Grouped Query Attention (GQA) with QK-norm per head, optional splash kernel
  * Two segments of transformer layers:
      - segment_0 (num_layers - num_kv_reuse_layers regular layers).  The last
        layer exports its K/V projections as side outputs.
      - segment_1 (num_kv_reuse_layers KV-reuse layers) consumes those K/V
        states and only recomputes Q for its attention.
  * SwiGLU FFN via separate gate/up projections and a down projection
  * Tied embedding and output head (output = hidden @ embedding.weight.T)
  * Optional LoRA adapters on every adapted Linear (matches TAMM LoRAModelAdapter
    with all adapt_* flags enabled — Q/K/V, attention output, FFN hidden &
    output).
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import ad_checkpoint
from flax import nnx

from .args import AFMTextV7ModelArgs


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _lora_dtype(dtype_str: str) -> jnp.dtype:
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# RoPE utilities (interleaved — matches tamm.layers.rope)
# ---------------------------------------------------------------------------

def precompute_rope_coefficients(
    head_dim: int,
    max_seq_len: int,
    theta: float,
) -> jax.Array:
    """Precompute RoPE rotation matrices for every position.

    Mirrors tamm.layers.rope.compute_rope_coefficients but returns a single
    tensor indexed by position. Exponents follow TAMM's reformulation
    (``x * (2/dim - 1) / (dim/2 - 1) = -2x/dim``) which is mathematically
    identical to the standard LLaMA recipe.

    Returns:
        Tensor of shape ``[max_seq_len, head_dim // 2, 2, 2]`` holding a
        2×2 rotation matrix ``[[cos, -sin], [sin, cos]]`` per (position,
        frequency).
    """
    half = head_dim // 2
    exponents = jnp.arange(half, dtype=jnp.float32) * (-2.0 / head_dim)
    freqs = jnp.power(jnp.float32(theta), exponents)  # [head_dim/2]
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]  # [seq, head_dim/2]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    # Assemble 2×2 rotation matrix [[cos, -sin], [sin, cos]]:
    row0 = jnp.stack([cos, -sin], axis=-1)  # [seq, h/2, 2]
    row1 = jnp.stack([sin, cos], axis=-1)
    return jnp.stack([row0, row1], axis=-2)  # [seq, h/2, 2, 2]


def apply_rope(x: jax.Array, rope: jax.Array) -> jax.Array:
    """Apply RoPE to a tensor using adjacent-pair rotation.

    Args:
        x: ``[batch, seq_len, n_heads, head_dim]``.
        rope: ``[seq_len, head_dim // 2, 2, 2]`` rotation matrices.
    Returns:
        Rotated tensor with the same shape and dtype as ``x``.
    """
    input_dtype = x.dtype
    batch, seq_len, n_heads, head_dim = x.shape
    # Reshape to adjacent pairs: [batch, seq, heads, head_dim/2, 2].
    x_pairs = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
    # Apply 2×2 rotation per pair. Broadcast rope over (batch, heads).
    #   rope: [seq, h/2, 2, 2]  —>  [1, seq, 1, h/2, 2, 2]
    #   out[..., i] = sum_j rope[..., i, j] * x_pairs[..., j]
    rope_b = rope[None, :, None, :, :, :].astype(jnp.float32)
    x_pairs_f32 = x_pairs.astype(jnp.float32)
    rotated = jnp.einsum("bshpij,bshpj->bshpi", rope_b, x_pairs_f32)
    return rotated.reshape(batch, seq_len, n_heads, head_dim).astype(input_dtype)


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------

class RMSNorm(nnx.Module):
    """RMSNorm matching tamm.layers.RMSNorm (fp32 compute, weight in storage dtype)."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        del rngs  # unused — weight is always initialized to ones
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean_sq = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        inv_norm = jax.lax.rsqrt(mean_sq + self.eps)
        return (x_f32 * inv_norm * self.weight[...].astype(jnp.float32)).astype(
            input_dtype
        )


class LoRAAdapter(nnx.Module):
    """Single LoRA adapter matching tamm LoRA layer.

    Stored as (a_transpose, b_transpose) to match the TAMM parameter names
    and the sharding patterns used in torchax/afmv7.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        *,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        # a_transpose init: standard LoRA uses kaiming-uniform on A; we follow
        # a simple normal init (matching the training-time patch behaviour in
        # torchax/afmv7/__init__.py which monkeypatches ``torch.nn.init.normal_``
        # to ``randn``). b_transpose init is zero so the adapter starts off as
        # a no-op.
        std = 1.0 / math.sqrt(rank)
        self.a_transpose = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, rank), dtype=lora_dtype)
            * std
        )
        self.b_transpose = nnx.Param(
            jnp.zeros((rank, out_features), dtype=lora_dtype)
        )
        self.scale = alpha / rank

    def __call__(self, x: jax.Array, outputs: jax.Array) -> jax.Array:
        """Returns ``outputs + scale * (x @ a_transpose) @ b_transpose``.

        Mirrors the simplified LoRA formulation used in torchax/afmv7, which
        replaces TAMM's ``torch.addmm`` with pure matmul so the compiler can
        propagate shardings (see torchax/afmv7/__init__.py).
        """
        # Cast x to the LoRA dtype for the low-rank matmul, then cast back to
        # the outputs' dtype. Matches tamm LoRA._transform_outputs_impl
        # semantics under mixed precision.
        x_lora = x.astype(self.a_transpose.dtype)
        delta = (x_lora @ self.a_transpose[...]) @ self.b_transpose[...]
        return outputs + (delta.astype(outputs.dtype) * self.scale)


class LoRAAdapterGroup(nnx.Module):
    """Holds one or more LoRA adapters under a single ``adapters.lora``
    namespace (matching TAMM's adapter naming)."""

    def __init__(self, loras: dict):
        for name, lora in loras.items():
            setattr(self, name, lora)
        self._lora_names = tuple(loras.keys())


class AdaptedLinear(nnx.Module):
    """Linear layer with an optional single-LoRA adapter.

    Parameter paths (non-LoRA): ``kernel``.
    Parameter paths (LoRA):    ``wrapped/kernel``, ``adapters/lora/a_transpose``,
                                ``adapters/lora/b_transpose``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.use_lora = use_lora
        linear = nnx.Linear(
            in_features,
            out_features,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if use_lora:
            self.wrapped = linear
            adapter = LoRAAdapter(
                in_features,
                out_features,
                rank=lora_rank,
                alpha=lora_alpha,
                lora_dtype=lora_dtype,
                rngs=rngs,
            )
            self.adapters = LoRAAdapterGroup({"lora": adapter})
        else:
            # Store directly — no "wrapped" nesting when LoRA is off.
            self.linear = linear

    def _base_linear(self) -> nnx.Linear:
        return self.wrapped if self.use_lora else self.linear

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self._base_linear()(x)
        if self.use_lora:
            y = self.adapters.lora(x, y)  # pyrefly: ignore[missing-attribute]
        return y


class AdaptedMultiOutputLinear(nnx.Module):
    """MultiOutputLinear equivalent (parallel linears with independent outputs).

    The TAMM LoRAModelAdapter produces one adapter per output branch
    (``adapters.lora.lora_0``, ``lora_1``, ...). We mirror that layout when
    ``use_lora=True``.
    """

    def __init__(
        self,
        in_features: int,
        output_dims: list[int],
        *,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.use_lora = use_lora
        self.output_dims = tuple(output_dims)
        for i, out_dim in enumerate(output_dims):
            base = nnx.Linear(
                in_features,
                out_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            if use_lora:
                branch = _LinearWithLoRA(
                    base=base,
                    in_features=in_features,
                    out_features=out_dim,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dtype=lora_dtype,
                    rngs=rngs,
                )
                setattr(self, f"linear_{i}", branch)
            else:
                setattr(self, f"linear_{i}", base)

    def __call__(self, x: jax.Array) -> tuple:
        outs = []
        for i in range(len(self.output_dims)):
            branch = getattr(self, f"linear_{i}")
            outs.append(branch(x))
        return tuple(outs)


class _LinearWithLoRA(nnx.Module):
    """Helper: a single Linear branch with a ``wrapped`` + ``adapters.lora`` layout."""

    def __init__(
        self,
        *,
        base: nnx.Linear,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.wrapped = base
        adapter = LoRAAdapter(
            in_features,
            out_features,
            rank=lora_rank,
            alpha=lora_alpha,
            lora_dtype=lora_dtype,
            rngs=rngs,
        )
        self.adapters = LoRAAdapterGroup({"lora": adapter})

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.wrapped(x)
        y = self.adapters.lora(x, y)  # pyrefly: ignore[missing-attribute]
        return y


class FusedQKVLinear(nnx.Module):
    """Fused QKV projection (equivalent to tamm FusedMultiOutputLinear).

    Base-parameter path: ``fused_linear/kernel`` (with optional ``wrapped``
    parent under LoRA). LoRA adapters live under ``adapters.lora.lora_{0,1,2}``
    for Q, K, V respectively.
    """

    def __init__(
        self,
        in_features: int,
        *,
        q_out: int,
        k_out: int,
        v_out: int,
        use_lora: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.use_lora = use_lora
        self.output_dims = (q_out, k_out, v_out)
        total_out = q_out + k_out + v_out

        fused = nnx.Linear(
            in_features,
            total_out,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if use_lora:
            self.wrapped = _FusedLinearBranch(fused_linear=fused)
            loras = {}
            for i, out_dim in enumerate(self.output_dims):
                loras[f"lora_{i}"] = LoRAAdapter(
                    in_features,
                    out_dim,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    lora_dtype=lora_dtype,
                    rngs=rngs,
                )
            self.adapters = LoRAAdapterGroup(loras)
        else:
            self.fused_linear = fused

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.use_lora:
            fused_out = self.wrapped.fused_linear(x)
        else:
            fused_out = self.fused_linear(x)
        # Split along last axis.
        q_out, k_out, v_out = self.output_dims
        q = fused_out[..., :q_out]
        k = fused_out[..., q_out : q_out + k_out]
        v = fused_out[..., q_out + k_out :]
        if self.use_lora:
            # Apply one adapter per branch, mirroring tamm LoRA for
            # FusedMultiOutputLinear (three separate LoRA modules).
            q = self.adapters.lora_0(x, q)  # pyrefly: ignore[missing-attribute]
            k = self.adapters.lora_1(x, k)  # pyrefly: ignore[missing-attribute]
            v = self.adapters.lora_2(x, v)  # pyrefly: ignore[missing-attribute]
        return q, k, v


class _FusedLinearBranch(nnx.Module):
    """Wraps a fused linear inside a ``wrapped`` attribute for LoRA layout."""

    def __init__(self, *, fused_linear: nnx.Linear):
        self.fused_linear = fused_linear


# ---------------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------------

def _scaled_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    n_heads: int,
    n_kv_heads: int,
    *,
    attn_fn: Optional[Callable] = None,
) -> jax.Array:
    """Causal scaled-dot-product attention with GQA.

    Inputs are pre-projected and already have RoPE / QK-norm applied. Shapes:
        q: [B, S, n_heads * head_dim]
        k: [B, S, n_kv_heads * head_dim]
        v: [B, S, n_kv_heads * head_dim]
    Output: [B, S, n_heads * head_dim].
    """
    batch, seq_len, _ = q.shape
    head_dim = q.shape[-1] // n_heads

    q = q.reshape(batch, seq_len, n_heads, head_dim)
    k = k.reshape(batch, seq_len, n_kv_heads, head_dim)
    v = v.reshape(batch, seq_len, n_kv_heads, head_dim)

    if attn_fn is not None:
        # Splash attention: supports GQA natively.
        q_t = q.transpose(0, 2, 1, 3)  # [B, H_q, S, D]
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)
        out = attn_fn(q_t, k_t, v_t, None)
        out = out.transpose(0, 2, 1, 3)  # [B, S, H_q, D]
    else:
        # Fallback: tile K/V to match Q head count, then standard SDPA.
        n_rep = n_heads // n_kv_heads
        if n_rep != 1:
            k = jnp.repeat(k, n_rep, axis=2)
            v = jnp.repeat(v, n_rep, axis=2)
        scale = 1.0 / math.sqrt(head_dim)
        scores = jnp.einsum("bshd,bthd->bsht", q, k) * scale
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(
            mask[None, :, None, :], scores, jnp.finfo(scores.dtype).min
        )
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        out = jnp.einsum("bsht,bthd->bshd", attn, v)

    return out.reshape(batch, seq_len, n_heads * head_dim)


class _QKNorm(nnx.Module):
    """Per-head RMSNorm applied to Q and/or K with shape [dim_per_head]."""

    def __init__(
        self,
        dim_per_head: int,
        *,
        eps: float,
        with_key: bool,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.query_norm = RMSNorm(
            dim_per_head, eps=eps, param_dtype=param_dtype, rngs=rngs
        )
        self.key_norm = (
            RMSNorm(dim_per_head, eps=eps, param_dtype=param_dtype, rngs=rngs)
            if with_key
            else None
        )

    def apply_q(self, q: jax.Array, n_heads: int, head_dim: int) -> jax.Array:
        b, s, _ = q.shape
        q_h = q.reshape(b, s, n_heads, head_dim)
        q_h = self.query_norm(q_h)
        return q_h.reshape(b, s, n_heads * head_dim)

    def apply_k(self, k: jax.Array, n_kv_heads: int, head_dim: int) -> jax.Array:
        if self.key_norm is None:
            return k
        b, s, _ = k.shape
        k_h = k.reshape(b, s, n_kv_heads, head_dim)
        k_h = self.key_norm(k_h)
        return k_h.reshape(b, s, n_kv_heads * head_dim)


class RegularAttention(nnx.Module):
    """Full self-attention block for segment_0 layers."""

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        *,
        attn_fn: Optional[Callable],
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.n_heads = args.num_heads
        self.n_kv_heads = args.effective_num_kv_heads
        self.head_dim = args.head_dim
        self._attn_fn = attn_fn

        self.norm = RMSNorm(
            args.hidden_dim, param_dtype=param_dtype, rngs=rngs
        )
        self.qkv_transform = FusedQKVLinear(
            in_features=args.hidden_dim,
            q_out=args.num_heads * args.head_dim,
            k_out=self.n_kv_heads * args.head_dim,
            v_out=self.n_kv_heads * args.head_dim,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.qk_norm = _QKNorm(
            args.head_dim,
            eps=1e-5,
            with_key=True,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output_transform = AdaptedLinear(
            args.num_heads * args.head_dim,
            args.hidden_dim,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rope: jax.Array,
        *,
        return_kv: bool = False,
    ):
        """Returns the attention residual output, optionally with (k, v)
        tensors (pre-SDPA inputs) for KV reuse.
        """
        h = self.norm(x)
        q, k, v = self.qkv_transform(h)
        # RoPE + QK-norm on head-shaped tensors.
        q = self._rope(q, rope, self.n_heads)
        k = self._rope(k, rope, self.n_kv_heads)
        q = self.qk_norm.apply_q(q, self.n_heads, self.head_dim)
        k = self.qk_norm.apply_k(k, self.n_kv_heads, self.head_dim)

        out = _scaled_dot_product_attention(
            q, k, v,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            attn_fn=self._attn_fn,
        )
        out = self.output_transform(out)
        if return_kv:
            return x + out, (k, v)
        return x + out

    def _rope(self, t: jax.Array, rope: jax.Array, n_heads: int) -> jax.Array:
        b, s, _ = t.shape
        t_h = t.reshape(b, s, n_heads, self.head_dim)
        t_h = apply_rope(t_h, rope[:s])
        return t_h.reshape(b, s, n_heads * self.head_dim)


class KVReuseAttention(nnx.Module):
    """Attention block that reuses K/V from segment_0's last regular layer."""

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        *,
        attn_fn: Optional[Callable],
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.n_heads = args.num_heads
        self.n_kv_heads = args.effective_num_kv_heads
        self.head_dim = args.head_dim
        self._attn_fn = attn_fn

        self.norm = RMSNorm(
            args.hidden_dim, param_dtype=param_dtype, rngs=rngs
        )
        self.q_transform = AdaptedLinear(
            args.hidden_dim,
            args.hidden_dim,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # No key norm in the KV-reuse variant (mirrors TAMM behaviour: the
        # segment_0 last layer already applied key_norm).
        self.q_norm = _QKNorm(
            args.head_dim,
            eps=1e-5,
            with_key=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output_transform = AdaptedLinear(
            args.num_heads * args.head_dim,
            args.hidden_dim,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rope: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        h = self.norm(x)
        q = self.q_transform(h)
        b, s, _ = q.shape
        q_h = q.reshape(b, s, self.n_heads, self.head_dim)
        q_h = apply_rope(q_h, rope[:s])
        q = q_h.reshape(b, s, self.n_heads * self.head_dim)
        q = self.q_norm.apply_q(q, self.n_heads, self.head_dim)

        out = _scaled_dot_product_attention(
            q, k, v,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            attn_fn=self._attn_fn,
        )
        out = self.output_transform(out)
        return x + out


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class FeedForward(nnx.Module):
    """Pre-norm SwiGLU FFN matching tamm TransformerFeedForward."""

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        *,
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        ffn_hidden = args.ffn_hidden_dim
        self.norm = RMSNorm(
            args.hidden_dim, param_dtype=param_dtype, rngs=rngs
        )
        # SwiGLU gate & up projections share the same input.
        self.hidden_transform = AdaptedMultiOutputLinear(
            in_features=args.hidden_dim,
            output_dims=[ffn_hidden, ffn_hidden],
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.output_transform = AdaptedLinear(
            ffn_hidden,
            args.hidden_dim,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dtype=lora_dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.norm(x)
        gate, up = self.hidden_transform(h)
        # SwiGLU = up * silu(gate) — see tamm.layers.functional.swiglu.
        h = up * jax.nn.silu(gate)
        h = self.output_transform(h)
        return x + h


# ---------------------------------------------------------------------------
# Transformer layers
# ---------------------------------------------------------------------------

class RegularTransformerLayer(nnx.Module):
    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        *,
        attn_fn: Optional[Callable],
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.attention = RegularAttention(
            args,
            attn_fn=attn_fn,
            param_dtype=param_dtype,
            lora_dtype=lora_dtype,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            args,
            param_dtype=param_dtype,
            lora_dtype=lora_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, rope: jax.Array) -> jax.Array:
        x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
        x = self.attention(x, rope)
        x = self.feed_forward(x)
        return x

    def call_with_kv(
        self, x: jax.Array, rope: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Like ``__call__`` but also returns the attention layer's K/V for
        use by downstream KV-reuse layers. Kept as a separate method so
        ``jax.checkpoint`` does not see a Python bool flag (which it cannot
        trace)."""
        x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
        x, kv = self.attention(x, rope, return_kv=True)
        x = self.feed_forward(x)
        return x, kv


class KVReuseTransformerLayer(nnx.Module):
    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        *,
        attn_fn: Optional[Callable],
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.attention = KVReuseAttention(
            args,
            attn_fn=attn_fn,
            param_dtype=param_dtype,
            lora_dtype=lora_dtype,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            args,
            param_dtype=param_dtype,
            lora_dtype=lora_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rope: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        x = ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
        x = self.attention(x, rope, k, v)
        x = self.feed_forward(x)
        return x


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Segment(nnx.Module):
    """Holds a list of transformer layers under integer-keyed attributes.

    Uses nnx.List for integer indexing so the sharding map matcher can
    recognise paths like ``layers/segment_0/0/attention/...``.
    """

    def __init__(self, layers: list):
        self.layers = nnx.List(layers)


class Layers(nnx.Module):
    """Container holding segment_0 and segment_1."""

    def __init__(self, segment_0, segment_1):
        self.segment_0 = segment_0
        self.segment_1 = segment_1


class _ScannedRegSegment(nnx.Module):
    """segment_0 container for the scan path: a ``ScannedRegularBlocks``
    running the first (n-1) layers + one non-scanned ``last_layer`` that
    additionally exports K/V for segment_1 reuse.

    ``scanned`` may be ``None`` when num_regular_layers == 1 (the "last"
    layer is then the only layer).
    """

    def __init__(self, scanned, last_layer):
        self.scanned = scanned  # may be None
        self.last_layer = last_layer


# ---------------------------------------------------------------------------
# Scan-based segment classes
# ---------------------------------------------------------------------------

def _get_block_param_paths(state) -> list[str]:
    """Path strings for every leaf in an nnx.State (see jax/llama3 for the
    same helper). Strips the trailing ``.value`` GetAttrKey."""
    paths: list[str] = []

    def _collect(path, leaf):
        parts = []
        for k in path:
            if isinstance(k, jax.tree_util.DictKey):
                parts.append(str(k.key))
            elif isinstance(k, jax.tree_util.GetAttrKey):
                if k.name != "value":
                    parts.append(k.name)
            elif isinstance(k, jax.tree_util.SequenceKey):
                parts.append(str(k.idx))
            else:
                parts.append(str(k))
        paths.append("/".join(p for p in parts if p))
        return leaf

    jax.tree_util.tree_map_with_path(_collect, state)
    return paths


class ScannedRegularBlocks(nnx.Module):
    """Runs ``n_layers`` RegularTransformerLayer blocks with ``jax.lax.scan``
    over stacked parameters. Mirrors jax/llama3's ``ScannedTransformerBlocks``.

    The scan emits the trailing hidden state as carry; per-step ys are
    ignored because the segment_0 last-layer K/V is handled by a separate
    non-scanned ``_last_regular_layer`` on the parent Transformer.
    """

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        n_layers: int,
        attn_fn: Optional[Callable],
        checkpoint_policy,
        *,
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.n_layers = n_layers
        self.checkpoint_policy = checkpoint_policy

        ref = RegularTransformerLayer(
            args, attn_fn=attn_fn, param_dtype=param_dtype,
            lora_dtype=lora_dtype, rngs=rngs,
        )
        ref_graphdef, ref_state = nnx.split(ref)
        self._graphdef = ref_graphdef
        ref_leaves, treedef = jax.tree_util.tree_flatten(ref_state)
        self._treedef = treedef
        param_paths = _get_block_param_paths(ref_state)
        attr_names = [p.replace("/", "_") for p in param_paths]
        self._param_attr_names = attr_names

        all_blocks = [ref] + [
            RegularTransformerLayer(
                args, attn_fn=attn_fn, param_dtype=param_dtype,
                lora_dtype=lora_dtype, rngs=nnx.Rngs(1000 + i),
            )
            for i in range(1, n_layers)
        ]
        all_values = []
        for block in all_blocks:
            _, state = nnx.split(block)
            leaves, _ = jax.tree_util.tree_flatten(state)
            all_values.append(leaves)
        stacked = [
            jnp.stack([all_values[b][p] for b in range(n_layers)], axis=0)
            for p in range(len(ref_leaves))
        ]
        for name, arr in zip(attr_names, stacked):
            setattr(self, name, nnx.Param(arr))

    def __call__(self, x: jax.Array, rope: jax.Array) -> jax.Array:
        graphdef = self._graphdef
        treedef = self._treedef
        policy = self.checkpoint_policy
        stacked_arrays = [getattr(self, n)[...] for n in self._param_attr_names]

        def scan_fn(carry, layer_arrays):
            layer_state = treedef.unflatten(layer_arrays)
            block = nnx.merge(graphdef, layer_state)
            return block(carry, rope), None

        if policy is not None:
            scan_fn = jax.checkpoint(scan_fn, policy=policy)

        x, _ = jax.lax.scan(scan_fn, x, stacked_arrays)
        return x


class ScannedKVReuseBlocks(nnx.Module):
    """Runs ``n_layers`` KVReuseTransformerLayer blocks with ``jax.lax.scan``.

    k/v from segment_0 are broadcast into every iteration (scan treats them
    as constants because they are captured in the closure).
    """

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        n_layers: int,
        attn_fn: Optional[Callable],
        checkpoint_policy,
        *,
        param_dtype: jnp.dtype,
        lora_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.n_layers = n_layers
        self.checkpoint_policy = checkpoint_policy

        ref = KVReuseTransformerLayer(
            args, attn_fn=attn_fn, param_dtype=param_dtype,
            lora_dtype=lora_dtype, rngs=rngs,
        )
        ref_graphdef, ref_state = nnx.split(ref)
        self._graphdef = ref_graphdef
        ref_leaves, treedef = jax.tree_util.tree_flatten(ref_state)
        self._treedef = treedef
        param_paths = _get_block_param_paths(ref_state)
        attr_names = [p.replace("/", "_") for p in param_paths]
        self._param_attr_names = attr_names

        all_blocks = [ref] + [
            KVReuseTransformerLayer(
                args, attn_fn=attn_fn, param_dtype=param_dtype,
                lora_dtype=lora_dtype, rngs=nnx.Rngs(2000 + i),
            )
            for i in range(1, n_layers)
        ]
        all_values = []
        for block in all_blocks:
            _, state = nnx.split(block)
            leaves, _ = jax.tree_util.tree_flatten(state)
            all_values.append(leaves)
        stacked = [
            jnp.stack([all_values[b][p] for b in range(n_layers)], axis=0)
            for p in range(len(ref_leaves))
        ]
        for name, arr in zip(attr_names, stacked):
            setattr(self, name, nnx.Param(arr))

    def __call__(
        self, x: jax.Array, rope: jax.Array, k: jax.Array, v: jax.Array,
    ) -> jax.Array:
        graphdef = self._graphdef
        treedef = self._treedef
        policy = self.checkpoint_policy
        stacked_arrays = [getattr(self, n)[...] for n in self._param_attr_names]

        def scan_fn(carry, layer_arrays):
            layer_state = treedef.unflatten(layer_arrays)
            block = nnx.merge(graphdef, layer_state)
            return block(carry, rope, k, v), None

        if policy is not None:
            scan_fn = jax.checkpoint(scan_fn, policy=policy)

        x, _ = jax.lax.scan(scan_fn, x, stacked_arrays)
        return x


class Transformer(nnx.Module):
    """Full AFMTextV7 transformer."""

    def __init__(
        self,
        args: AFMTextV7ModelArgs,
        use_scan: bool = False,
        attn_fn: Optional[Callable] = None,
        checkpoint_policy=None,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # use_scan=True stacks uniform layer weights along a leading layer
        # axis and drives them through ``jax.lax.scan`` — one XLA op per
        # segment instead of N per-layer copies. Segment_0's LAST regular
        # layer stays unscanned because it additionally exports K/V for
        # segment_1 reuse; that method (``call_with_kv``) would break the
        # scan_fn signature otherwise.
        # ``checkpoint_policy`` is honoured via jax.checkpoint wrapping
        # around each layer call (non-scan path) or around the scan_fn
        # (scan path).
        self.args = args
        self.use_scan = use_scan
        self.checkpoint_policy = checkpoint_policy
        lora_dtype = _lora_dtype(args.lora_dtype) if args.use_lora else jnp.bfloat16

        self.embedding = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.hidden_dim,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if use_scan:
            # segment_0: scan over (num_regular - 1) identical layers + one
            # non-scanned last layer that calls `call_with_kv`.
            n_scanned = max(0, args.num_regular_layers - 1)
            scanned_segment_0 = ScannedRegularBlocks(
                args, n_layers=n_scanned,
                attn_fn=attn_fn, checkpoint_policy=checkpoint_policy,
                param_dtype=param_dtype, lora_dtype=lora_dtype,
                rngs=nnx.Rngs(100),
            ) if n_scanned > 0 else None
            last_regular = RegularTransformerLayer(
                args, attn_fn=attn_fn,
                param_dtype=param_dtype, lora_dtype=lora_dtype,
                rngs=nnx.Rngs(args.num_regular_layers - 1),
            )
            # segment_1: scan over all kv-reuse layers.
            scanned_segment_1 = ScannedKVReuseBlocks(
                args, n_layers=args.num_kv_reuse_layers,
                attn_fn=attn_fn, checkpoint_policy=checkpoint_policy,
                param_dtype=param_dtype, lora_dtype=lora_dtype,
                rngs=nnx.Rngs(200),
            ) if args.num_kv_reuse_layers > 0 else None
            self.layers = Layers(
                segment_0=_ScannedRegSegment(
                    scanned=scanned_segment_0, last_layer=last_regular,
                ),
                segment_1=scanned_segment_1,
            )
        else:
            segment_0_layers = [
                RegularTransformerLayer(
                    args, attn_fn=attn_fn, param_dtype=param_dtype,
                    lora_dtype=lora_dtype, rngs=nnx.Rngs(i),
                )
                for i in range(args.num_regular_layers)
            ]
            segment_1_layers = [
                KVReuseTransformerLayer(
                    args, attn_fn=attn_fn, param_dtype=param_dtype,
                    lora_dtype=lora_dtype,
                    rngs=nnx.Rngs(args.num_regular_layers + i),
                )
                for i in range(args.num_kv_reuse_layers)
            ]
            self.layers = Layers(
                segment_0=Segment(segment_0_layers),
                segment_1=Segment(segment_1_layers),
            )
        self.output_norm = RMSNorm(
            args.hidden_dim, param_dtype=param_dtype, rngs=rngs
        )
        # Precompute RoPE table — replicated across all devices.
        self.rope = precompute_rope_coefficients(
            args.head_dim, args.max_seq_len, args.rope_theta
        )

    def _forward_hidden(self, tokens: jax.Array) -> jax.Array:
        """Run tokens through embedding, both segments, and the output
        norm. Returns the final hidden state ``[B, S, H]`` in bf16 —
        callers tie with the embedding weight for logits or compute_loss.
        """
        # Tokens arrive batch-sharded P('fsdp'); the embedding table is
        # sharded ('tp', 'fsdp'). Explicit with_sharding_constraint on
        # tokens is disallowed on multi-host Auto-typed meshes.
        x = self.embedding(tokens)  # [B, S, H]
        x = x.astype(jnp.bfloat16)
        rope = self.rope[: x.shape[1]]

        policy = self.checkpoint_policy

        if self.use_scan:
            if self.layers.segment_0.scanned is not None:
                x = self.layers.segment_0.scanned(x, rope)
            last = self.layers.segment_0.last_layer
            last_fn = last.call_with_kv
            if policy is not None:
                last_fn = jax.checkpoint(last_fn, policy=policy)
            x, (k_reuse, v_reuse) = last_fn(x, rope)

            if self.layers.segment_1 is not None:
                x = self.layers.segment_1(x, rope, k_reuse, v_reuse)
        else:
            regular_layers = list(self.layers.segment_0.layers)
            last_idx = len(regular_layers) - 1
            kv = None
            for i, layer in enumerate(regular_layers):
                if i == last_idx:
                    fn = layer.call_with_kv
                    if policy is not None:
                        fn = jax.checkpoint(fn, policy=policy)
                    x, kv = fn(x, rope)
                else:
                    fn = layer
                    if policy is not None:
                        fn = jax.checkpoint(fn, policy=policy)
                    x = fn(x, rope)
            assert kv is not None, "segment_0 must have at least one layer"
            k_reuse, v_reuse = kv

            segment_1 = self.layers.segment_1
            if segment_1 is not None:
                for layer in segment_1.layers:
                    fn = layer
                    if policy is not None:
                        fn = jax.checkpoint(fn, policy=policy)
                    x = fn(x, rope, k_reuse, v_reuse)

        return self.output_norm(x)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Forward pass producing logits ``[B, S, vocab]``.

        Kept for the equivalence test and non-training callers. Training
        should use :meth:`compute_loss` instead, which chunk-projects
        hidden → vocab and never materialises the full logits tensor.
        """
        x = self._forward_hidden(tokens)
        emb = self.embedding.embedding[...]  # [vocab, hidden]
        return jnp.einsum("bsh,vh->bsv", x.astype(emb.dtype), emb).astype(
            jnp.bfloat16
        )

    def compute_loss(
        self,
        tokens: jax.Array,
        labels: jax.Array,
        *,
        chunk_size: int = 512,
        remat_chunks: bool = False,
    ) -> jax.Array:
        """Cross-entropy loss with chunked output projection.

        Without this, ``logits = hidden @ embed.T`` would materialise a
        ``[B, S, V]`` tensor (~80 GiB at B=8 S=8192 V=153600 bf16). We
        chunk the flattened (B*S) axis into ``chunk_size`` tokens, compute
        logits + log_softmax + target-logit pick per chunk, and sum the
        CE into a scalar via ``jax.lax.scan``.

        Returns a fp32 scalar = sum of per-token CE (not mean — matches
        the trainer's existing convention, which normalises by ntokens).
        """
        hidden = self._forward_hidden(tokens)  # [B, S, H]
        B, S, H = hidden.shape
        N = B * S
        emb = self.embedding.embedding[...]  # [V, H]
        emb_compute = emb.astype(jnp.bfloat16)

        flat = hidden.reshape(N, H)
        y = labels.reshape(N)

        # Pick chunk count that evenly divides N; fall back to a single
        # chunk if chunk_size does not divide N.
        if N % chunk_size != 0:
            chunk_size = N
        n_chunks = N // chunk_size
        h_chunks = flat.reshape(n_chunks, chunk_size, H)
        y_chunks = y.reshape(n_chunks, chunk_size)

        def chunk_loss(carry, chunk):
            h_c, y_c = chunk
            # [chunk, V] logits stay in bf16 — promoting to fp32 here
            # doubles this tensor to ~40 GiB at B=8 S=8192 V=153600 which
            # doesn't fit on v6e-8. logsumexp is max-stabilised so bf16 is
            # adequate; we cast its scalar output to fp32 before the
            # subtraction so the CE accumulates in fp32.
            logits = jnp.einsum("nh,vh->nv", h_c, emb_compute)
            log_z = jax.nn.logsumexp(logits, axis=-1).astype(jnp.float32)
            target = jnp.take_along_axis(
                logits, y_c[:, None], axis=-1
            ).squeeze(-1).astype(jnp.float32)
            loss = (log_z - target).sum()
            return carry + loss, None

        # Optional: wrap the scan body in jax.checkpoint so backward
        # recomputes per-chunk logits instead of saving [n_chunks,
        # chunk_size, V] all at once. Needed when the scan's default tape
        # would materialise every chunk's logits — e.g. at B=16 S=8192
        # V=153600 chunk=512 that's bf16[256, 512, 153600] ≈ 40 GiB. At
        # B=8 this fits without remat and remat costs ~10% throughput, so
        # remat is opt-in.
        if remat_chunks:
            chunk_loss = jax.checkpoint(chunk_loss)
        total, _ = jax.lax.scan(chunk_loss, jnp.float32(0.0), (h_chunks, y_chunks))
        return total


# ---------------------------------------------------------------------------
# Pre-defined configs (mirrors tpu/afmv7/__init__.py::afmv7_args)
# ---------------------------------------------------------------------------

afmv7_3b = AFMTextV7ModelArgs(
    vocab_size=153600,
    hidden_dim=2048,
    num_layers=56,
    num_kv_reuse_layers=21,
    num_heads=16,
    num_kv_heads=2,
    hidden_dim_scale_factor=3.25,
    rope_theta=500000.0,
)

afmv7_debug = AFMTextV7ModelArgs(
    vocab_size=2048,
    hidden_dim=64,
    num_layers=4,
    num_kv_reuse_layers=1,
    num_heads=4,
    num_kv_heads=2,
    hidden_dim_scale_factor=3.25,
    rope_theta=500000.0,
)

afmv7_debug_lora = AFMTextV7ModelArgs(
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
)

afmv7_3b_lora = AFMTextV7ModelArgs(
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
)
