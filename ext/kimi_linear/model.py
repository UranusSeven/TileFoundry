"""Kimi-Linear-48B-A3B-Instruct's full decode shell: tokens in, logits out.

The shipped authored model (``tests/models/kimi_linear_48b_a3b/model.py``)
describes this model's three *kinds* of submodule -- KDA, MLA, and the MoE --
and nothing around them: no embedding, no 27-layer walk, no closing norm, no
head. This file is the shell. Its Module tree is the published stack in layer
order -- layer 0 is KDA + dense MLP (``first_k_dense_replace: 1``), layers
3/7/11/15/19/23/26 (0-based) are MLA + MoE, the remaining nineteen are
KDA + MoE -- plus the step around it: ``embed``, the walk, ``final_rms_norm``,
``lm_head``.

Provenance
----------
The ``@func`` bodies for the three mixers and the MoE are the shipped source,
copied verbatim in what they compute. Three things change, and nothing else:

1. **Weights become ``ConstTensor``.** The shipped model declares every
   parameter a plain ``Tensor`` because its harness (``reference.py``) passes
   weights explicitly to ``evaluate``. A shell loads: ``ConstTensor`` lands in
   ``Module.weights``, ``load`` binds it, and ``prepare`` validates it
   (migrate.md step one). Activations and caches stay ``Tensor``.
2. **Per-weight converters.** The checkpoint is ``nn.Linear``-shaped -- every
   projection is stored ``(out, in)`` against the ``(1, in, out)`` the matmuls
   here want. ``A_log`` and ``dt_bias`` are stored f32 against their declared
   bf16. Every norm gamma is flat: ``KimiRMSNorm`` is ``weight * normed`` with
   no ``1 +`` offset (unlike Qwen3.5), so no norm carries a converter.
3. **The stack and the step.** Two layer kinds per mixer -- the dense-MLP
   layer 0 and the MoE layers -- sharing one walk, which composes Modules and
   so cannot be a ``@func``.

Decode, one token per step. ``S`` is the literal 1 and ``ctx_len`` is the only
range, exactly as the shipped model states it. The KV cache is explicit tensors
in and out:

- ``mla_attention`` reads ``ctx_len`` prior positions and returns this token's
  own key and value. The caller **appends** (``append_cache`` below).
- ``kda_attention`` reads a fixed-size recurrent state and three fixed-size
  convolution windows and returns all four updated. The caller **replaces**.

Kimi's MLA is NoPE (``mla_use_nope: true``): the 64 rotary-half dimensions are
not rotated, which ``prepare_inputs_for_generation`` expresses by handing the
mixer ``cos = 1, sin = 0`` -- the identity by the arithmetic of
``apply_rotary_pos_emb``, as the shipped model's docstring records.

Weight layout in the checkpoint (``model.safetensors.index.json``): experts are
stored per-expert as ``block_sparse_moe.experts.{i}.w1/w2/w3.weight`` (w1 gate,
w3 up, w2 down -- ``KimiBlockSparseMLP``), so ``w_gate``/``w_up``/``w_down``
here are one-to-many alias groups that ``prepare`` stacks; every other weight
is one tensor. ``hf_alias`` at the bottom is the {canonical: raw} table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from tilefoundry import DType, func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings for @func bodies
from tilefoundry.evaluator import to_torch_dtype
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.runtime import Absolute
from tilefoundry.target import CudaTarget


# ── the checkpoint's own configuration class ─────────────────────────────────
#
# Vendored, byte for byte, from the shipped model (which vendored it from
# `configuration_kimi.py` in moonshotai/Kimi-Linear-48B-A3B-Instruct at revision
# e1df551a447157d4658b573f9a695d57658590e9). `transformers` has no `kimi_linear`:
# the model type is absent from `CONFIG_MAPPING`, so the class that reads this
# checkpoint is this one. It is the *config* and nothing else: it imports only
# `PretrainedConfig`, defines no layers, and executes nothing at import.


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        model_max_length: Optional[int] = None,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.model_max_length = model_max_length

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )


# ── this checkpoint ──────────────────────────────────────────────────────────


def published(path: Path | None = None) -> KimiLinearConfig:
    """The checkpoint's own configuration, read by the class it names.

    The file sits beside this module, so a copy of this directory carries its own
    dimensions and needs nothing importable around it.
    """
    path = Path(__file__).parent / "config.json" if path is None else path
    return KimiLinearConfig(**json.loads(path.read_text(encoding="utf-8")))


def build(cfg: KimiLinearConfig):
    """This model at *cfg*: the mixers, the two layer kinds each mixer heads,
    and the root that walks them.

    Public because the model is asked about at more than one structural
    configuration -- ``published()``, or a reduced one a test builds to prepare
    and load cheaply. Callers name the cfg they mean and get a tree that
    shares no IR node with any other call's.
    """
    # The prior-cache length the MLA submodule reads: the only range this model
    # carries. Zero is a first step, and the exclusive upper bound comes from
    # the published configuration.
    C = DimVar("ctx_len", 0, cfg.model_max_length)

    # One token per step.
    S = 1

    _HID = cfg.hidden_size               # 2304
    _H = cfg.num_attention_heads         # 32, MLA and KDA alike
    _NOPE = cfg.qk_nope_head_dim         # 128
    _ROPE = cfg.qk_rope_head_dim         # 64
    _QK = _NOPE + _ROPE                     # 192, the score dim and so the scaling one
    _V = cfg.v_head_dim                  # 128
    _KVB = _NOPE + _V                       # 256, kv_b_proj's per-head output

    # KDA's dimensions are published nested, and its head_dim is not the top-level
    # one: `head_dim: 72` is hidden_size // num_attention_heads and is read by
    # neither path -- KDA uses 128 and MLA uses 192 (q/k) and 128 (v).
    _KDA = cfg.linear_attn_config
    _KH = _KDA["num_heads"]                 # 32
    _KD = _KDA["head_dim"]                  # 128
    _KP = _KH * _KD                         # 4096
    _W = _KDA["short_conv_kernel_size"]     # 4
    _WS = _W - 1                            # 3 stored positions

    _E = cfg.num_experts
    _TOPK = cfg.num_experts_per_token
    _MI = cfg.moe_intermediate_size
    _SI = _MI * cfg.num_shared_experts

    _I = cfg.intermediate_size           # 9216, the dense layer-0 MLP

    _EPS = cfg.rms_norm_eps

    # The published dtype as the DSL spells it. The checkpoint stores its weights
    # at this precision, so it is what a kernel reading them consumes.
    _DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
        str(cfg.dtype).removeprefix("torch.")
    ]

    @module(entry="kda_attention")
    class KimiKda:
        """A KDA layer's mixer: Kimi Delta Attention, decode step.

        A gated delta rule whose forget gate is **per channel**: ``g`` is a
        128-wide vector per head and the state decays column by column, which is
        what makes it KDA rather than the scalar-per-head gated delta net that
        ships as ``Qwen3NextGatedDeltaNet``. Body transcribed from vLLM's
        ``KimiDeltaAttention.forward`` and the ``fused_recurrent_kda`` kernel
        body; see the shipped model for the full provenance.
        """

        @func
        def short_conv(
            x: Tensor[(1, S, _KP), _DT],
            conv_w: Tensor[(_W, _KP), _DT],
            conv_state: Tensor[(1, _WS, _KP), _DT],
        ):
            # Causal depthwise convolution of kernel 4 with a silu, evaluated for one
            # token: the window is the three stored positions followed by this one, so
            # the convolution is a weighted sum over the window's time axis rather
            # than a sliding op. Returns the activation and the window to store next,
            # which is this window with its oldest position dropped.
            # `conv_w` stays a plain Tensor: a Module's weights are keyed by name,
            # and the entry declares this weight three times (q/k/v), so the helper
            # cannot share one ConstTensor name with it. Every call site passes the
            # weight explicitly, which a plain Tensor requires anyway.
            window = tf.concat([conv_state, x], axis=1)
            acc = tf.reduce(
                window
                * tf.reshape(
                    conv_w,
                    new_shape=(1, _W, _KP),
                ),
                axes=(1,),
                keepdim=True,
                kind="sum",
            )
            out = tf.silu(acc)
            state_next = window[:, 1 : _W, :]
            return out, state_next

        @func
        def l2_normalize(
            x: Tensor[(1, S, _KH, _KD), _DT],
        ) -> Tensor[(1, S, _KH, _KD), _DT]:
            # x / sqrt(sum(x*x) + 1e-6), per head. The epsilon sits inside the square
            # root, matching the kernel; it is not an rms_norm, which would divide by
            # the *mean* of the squares and carry a weight.
            sq = tf.reduce(tf.square(x), axes=(-1,), keepdim=True, kind="sum")
            return x * tf.rsqrt(sq + tf.full_like(sq, value=1e-6))

        @func
        def kda_gate(
            hidden_norm: Tensor[(1, S, _HID), _DT],
            w_f_a: ConstTensor[(1, _HID, _KD), _DT],
            w_f_b: ConstTensor[(1, _KD, _KP), _DT],
            dt_bias: ConstTensor[(_KP,), _DT],
            a_log: ConstTensor[(_KH,), _DT],
        ) -> Tensor[(1, S, _KH, _KD), _DT]:
            # The per-channel forget gate: a low-rank projection through
            # kda_head_dim, biased, softplus'd, and scaled by -exp(A_log) per head.
            # softplus here is beta=1, which is what the kernel computes; the kernel's
            # threshold=20 switch to the linear branch is a numerical guard on the
            # same function, not a different one.
            low = tf.matmul(hidden_norm, w_f_a)
            g_raw = tf.reshape(
                tf.matmul(low, w_f_b) + dt_bias,
                new_shape=(1, S, _KH, _KD),
            )
            decay_rate = -tf.exp(tf.reshape(a_log, new_shape=(1, 1, _KH, 1)))
            return decay_rate * tf.softplus(g_raw)

        @func
        def kda_attention(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_in: ConstTensor[(_HID,), _DT],
            w_q: ConstTensor[(1, _HID, _KP), _DT],
            w_k: ConstTensor[(1, _HID, _KP), _DT],
            w_v: ConstTensor[(1, _HID, _KP), _DT],
            conv_w_q: ConstTensor[(_W, _KP), _DT],
            conv_w_k: ConstTensor[(_W, _KP), _DT],
            conv_w_v: ConstTensor[(_W, _KP), _DT],
            conv_state_q: Tensor[(1, _WS, _KP), _DT],
            conv_state_k: Tensor[(1, _WS, _KP), _DT],
            conv_state_v: Tensor[(1, _WS, _KP), _DT],
            w_f_a: ConstTensor[(1, _HID, _KD), _DT],
            w_f_b: ConstTensor[(1, _KD, _KP), _DT],
            dt_bias: ConstTensor[(_KP,), _DT],
            a_log: ConstTensor[(_KH,), _DT],
            w_b: ConstTensor[(1, _HID, _KH), _DT],
            w_g_a: ConstTensor[(1, _HID, _KD), _DT],
            w_g_b: ConstTensor[(1, _KD, _KP), _DT],
            gamma_o: ConstTensor[(_KD,), _DT],
            w_o: ConstTensor[(1, _KP, _HID), _DT],
            state: Tensor[(1, _KH, _KD, _KD), _DT],
            scale: Tensor[(1, 1, 1), _DT],
        ):
            # One decode step. Returns the output, the updated recurrent state, and
            # the three updated convolution windows -- all fixed size, none carrying
            # ctx_len. The caller replaces rather than appends.
            # `KimiRMSNorm` ends `self.weight * hidden_states.to(input_dtype)`: the
            # normalised value is rounded to the input dtype before the learned scale
            # multiplies it. `tf.rms_norm` is the generic op and keeps f32 through
            # that multiply.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_in

            q_c, conv_q_next = short_conv(tf.matmul(hn, w_q), conv_w_q, conv_state_q)
            k_c, conv_k_next = short_conv(tf.matmul(hn, w_k), conv_w_k, conv_state_k)
            v_c, conv_v_next = short_conv(tf.matmul(hn, w_v), conv_w_v, conv_state_v)

            q_h = tf.reshape(q_c, new_shape=(1, S, _KH, _KD))
            k_h = tf.reshape(k_c, new_shape=(1, S, _KH, _KD))
            v_h = tf.reshape(v_c, new_shape=(1, S, _KH, _KD))

            # l2 normalisation happens inside the kernel, before the scale.
            q_n = l2_normalize(q_h)
            k_n = l2_normalize(k_h)
            q_s = tf.reshape(q_n, new_shape=(1, _KH, 1, _KD)) * scale

            g = kda_gate(hn, w_f_a, w_f_b, dt_bias, a_log)
            beta = tf.reshape(
                tf.sigmoid(tf.matmul(hn, w_b)), new_shape=(1, _KH, 1)
            )

            # The delta rule, one token. The state is [heads, v_dim, k_dim]; the decay
            # multiplies it column-wise along k_dim, which is the per-channel gate.
            decay = tf.reshape(tf.exp(g), new_shape=(1, _KH, 1, _KD))
            k_r = tf.reshape(k_n, new_shape=(1, _KH, 1, _KD))
            h_decayed = state * decay
            kv_mem = tf.reduce(h_decayed * k_r, axes=(-1,), keepdim=False, kind="sum")
            delta = (tf.reshape(v_h, new_shape=(1, _KH, _KD)) - kv_mem) * beta
            state_next = (
                h_decayed + tf.reshape(delta, new_shape=(1, _KH, _KD, 1)) * k_r
            )
            attn = tf.reduce(state_next * q_s, axes=(-1,), keepdim=False, kind="sum")

            # Gated output norm: rms_norm(attn) * sigmoid(g2), the "sigmoid" activation
            # of the kernel's fused gated RMSNorm -- not a swish, which would be
            # g2 * sigmoid(g2).
            g2 = tf.reshape(
                tf.matmul(tf.matmul(hn, w_g_a), w_g_b), new_shape=(1, _KH, _KD)
            )
            gated = tf.rms_norm(attn, gamma_o, eps=_EPS) * tf.sigmoid(g2)
            out = tf.matmul(tf.reshape(gated, new_shape=(1, S, _KP)), w_o)
            return out, state_next, conv_q_next, conv_k_next, conv_v_next

        # ---- raw checkpoint -> declared weight ---------------------------
        #
        # Every projection is an `nn.Linear`, stored (out, in) against the
        # (1, in, out) the matmuls above want. `gamma_in` and `gamma_o` are
        # flat (`KimiRMSNorm` / the gated RMSNorm carry no `1 +` offset) and
        # need no converter; `gamma_in` is the layer's `input_layernorm`, which
        # the alias table addresses Absolutely.

        @kda_attention.converter("w_q")
        def _(
            q_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(q_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_attention.converter("w_k")
        def _(
            k_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(k_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_attention.converter("w_v")
        def _(
            v_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(v_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_attention.converter("conv_w_q")
        def _(
            q_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            # A depthwise `nn.Conv1d` stores (channels, 1, kernel); short_conv
            # wants (kernel, channels).
            return tf.transpose(
                tf.reshape(q_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_attention.converter("conv_w_k")
        def _(
            k_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            return tf.transpose(
                tf.reshape(k_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_attention.converter("conv_w_v")
        def _(
            v_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            return tf.transpose(
                tf.reshape(v_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_attention.converter("w_f_a")
        def _(
            f_a_proj_weight: ConstTensor[(_KD, _HID), _DT],
        ) -> Tensor[(1, _HID, _KD), _DT]:
            return tf.reshape(
                tf.transpose(f_a_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KD)
            )

        @kda_attention.converter("w_f_b")
        def _(
            f_b_proj_weight: ConstTensor[(_KP, _KD), _DT],
        ) -> Tensor[(1, _KD, _KP), _DT]:
            return tf.reshape(
                tf.transpose(f_b_proj_weight, perm=(1, 0)), new_shape=(1, _KD, _KP)
            )

        @kda_attention.converter("dt_bias")
        def _(
            dt_bias_raw: ConstTensor[(_KP,), "f32"],
        ) -> Tensor[(_KP,), _DT]:
            # Stored f32; declared at the kernel's own dtype.
            return tf.cast(dt_bias_raw, dtype=_DT)

        @kda_attention.converter("a_log")
        def _(
            a_log_raw: ConstTensor[(1, 1, _KH, 1), "f32"],
        ) -> Tensor[(_KH,), _DT]:
            # Stored f32 as (1, 1, heads, 1); declared bf16 (heads,).
            return tf.cast(tf.reshape(a_log_raw, new_shape=(_KH,)), dtype=_DT)

        @kda_attention.converter("w_b")
        def _(
            b_proj_weight: ConstTensor[(_KH, _HID), _DT],
        ) -> Tensor[(1, _HID, _KH), _DT]:
            return tf.reshape(
                tf.transpose(b_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KH)
            )

        @kda_attention.converter("w_g_a")
        def _(
            g_a_proj_weight: ConstTensor[(_KD, _HID), _DT],
        ) -> Tensor[(1, _HID, _KD), _DT]:
            return tf.reshape(
                tf.transpose(g_a_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KD)
            )

        @kda_attention.converter("w_g_b")
        def _(
            g_b_proj_weight: ConstTensor[(_KP, _KD), _DT],
        ) -> Tensor[(1, _KD, _KP), _DT]:
            return tf.reshape(
                tf.transpose(g_b_proj_weight, perm=(1, 0)), new_shape=(1, _KD, _KP)
            )

        @kda_attention.converter("w_o")
        def _(
            o_proj_weight: ConstTensor[(_HID, _KP), _DT],
        ) -> Tensor[(1, _KP, _HID), _DT]:
            return tf.reshape(
                tf.transpose(o_proj_weight, perm=(1, 0)), new_shape=(1, _KP, _HID)
            )

    @module(entry="mla_attention")
    class KimiMla:
        """An MLA layer's mixer: multi-head latent attention, decode step.

        Mirrors ``DeepseekV3Attention.forward`` at Kimi's ranks, which is the
        same parameter set ``KimiMLAAttention`` builds: ``q_lora_rank`` null so
        a plain ``q_proj``, ``kv_a_proj_with_mqa`` (512 + 64 out),
        ``kv_a_layernorm``, ``kv_b_proj`` (32 * (128 + 128) out), ``o_proj``.
        """

        @func
        def mla_attention(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_in: ConstTensor[(_HID,), _DT],
            w_q: ConstTensor[(1, _HID, (_H * _QK)), _DT],
            w_kv_a: ConstTensor[(1, _HID, (cfg.kv_lora_rank + _ROPE)), _DT],
            gamma_kv_a: ConstTensor[(cfg.kv_lora_rank,), _DT],
            w_kv_b: ConstTensor[(1, cfg.kv_lora_rank, (_H * _KVB)), _DT],
            cos_cache: Tensor[(S, cfg.qk_rope_head_dim), _DT],
            sin_cache: Tensor[(S, cfg.qk_rope_head_dim), _DT],
            pos_ids: Tensor[(S,), "i32"],
            k_cache: Tensor[(1, C, _H, _QK), _DT],
            v_cache: Tensor[(1, C, _H, _V), _DT],
            scale: Tensor[(1, 1, 1, 1), _DT],
            w_o: ConstTensor[(1, (_H * _V), _HID), _DT],
        ):
            # Fused input RMSNorm + MLA, no residual (the layer owns that). Returns
            # the attention output with this token's key and value for the caller to
            # append.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_in

            # The query is a plain projection here: q_lora_rank is null, so there is
            # no q_a/q_b pair to fold.
            q = tf.reshape(tf.matmul(hn, w_q), new_shape=(1, S, _H, _QK))
            q_pass = q[:, :, :, :_NOPE]
            q_rot = q[:, :, :, _NOPE:_QK]

            # One projection yields the latent and the rope part together, and the
            # rope part is shared across heads -- that is the "MQA" in
            # kv_a_proj_with_mqa.
            compressed = tf.matmul(hn, w_kv_a)
            latent = compressed[:, :, : cfg.kv_lora_rank]
            k_rot_shared = compressed[:, :, cfg.kv_lora_rank : (cfg.kv_lora_rank + _ROPE)]

            kv_n32 = tf.cast(latent, dtype="f32")
            kv_n_var = tf.reduce(kv_n32 * kv_n32, axes=(-1,), keepdim=True, kind="mean")
            kv_n = tf.cast(kv_n32 * tf.rsqrt(kv_n_var + _EPS), dtype="bf16") * gamma_kv_a
            kv = tf.reshape(
                tf.matmul(kv_n, w_kv_b),
                new_shape=(1, S, _H, _KVB),
            )
            k_nope = kv[:, :, :, :_NOPE]
            v_new = kv[:, :, :, _NOPE:_KVB]

            # Rotate the shared 64-wide part once, then broadcast it over the heads;
            # repeat_interleave on a length-1 axis is that broadcast.
            k_rot_1 = tf.reshape(k_rot_shared, new_shape=(1, S, 1, _ROPE))
            _kq, k_rot = tf.rope(k_rot_1, k_rot_1, cos_cache, sin_cache, pos_ids)
            k_rot_h = tf.repeat_interleave(k_rot, repeats=_H, axis=2)
            q_rot_r, _kr = tf.rope(q_rot, q_rot, cos_cache, sin_cache, pos_ids)

            q_full = tf.concat([q_pass, q_rot_r], axis=-1)
            k_new = tf.concat([k_nope, k_rot_h], axis=-1)

            # Online softmax over two differently shaped score groups: the cache and
            # the token itself. No mask -- one query at the end of the context may
            # attend every position there is. Every key/value head serves exactly one
            # query head here (num_key_value_heads == num_attention_heads), so there
            # is no GQA expansion.
            q_s = q_full * scale
            k_ctx = tf.reshape(
                tf.transpose(k_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _QK)
            )
            v_ctx = tf.reshape(
                tf.transpose(v_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _V)
            )
            q_e = tf.reshape(q_s, new_shape=(1, S, _H, 1, _QK))
            score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
            score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

            peak = tf.max(
                tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
            )
            peak_e = tf.reshape(peak, new_shape=(1, S, _H, 1, 1))
            p_ctx = tf.exp(score_ctx - peak_e)
            p_new = tf.exp(score_new - peak)
            total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
            weighted = (
                tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
                + p_new * v_new
            )
            attn = weighted / total
            out = tf.matmul(tf.reshape(attn, new_shape=(1, S, (_H * _V))), w_o)
            return out, k_new, v_new

        # ---- raw checkpoint -> declared weight ---------------------------

        @mla_attention.converter("w_q")
        def _(
            q_proj_weight: ConstTensor[((_H * _QK), _HID), _DT],
        ) -> Tensor[(1, _HID, (_H * _QK)), _DT]:
            return tf.reshape(
                tf.transpose(q_proj_weight, perm=(1, 0)), new_shape=(1, _HID, (_H * _QK))
            )

        @mla_attention.converter("w_kv_a")
        def _(
            kv_a_proj_weight: ConstTensor[((cfg.kv_lora_rank + _ROPE), _HID), _DT],
        ) -> Tensor[(1, _HID, (cfg.kv_lora_rank + _ROPE)), _DT]:
            return tf.reshape(
                tf.transpose(kv_a_proj_weight, perm=(1, 0)),
                new_shape=(1, _HID, (cfg.kv_lora_rank + _ROPE)),
            )

        @mla_attention.converter("w_kv_b")
        def _(
            kv_b_proj_weight: ConstTensor[((_H * _KVB), cfg.kv_lora_rank), _DT],
        ) -> Tensor[(1, cfg.kv_lora_rank, (_H * _KVB)), _DT]:
            return tf.reshape(
                tf.transpose(kv_b_proj_weight, perm=(1, 0)),
                new_shape=(1, cfg.kv_lora_rank, (_H * _KVB)),
            )

        @mla_attention.converter("w_o")
        def _(
            o_proj_weight: ConstTensor[(_HID, (_H * _V)), _DT],
        ) -> Tensor[(1, (_H * _V), _HID), _DT]:
            return tf.reshape(
                tf.transpose(o_proj_weight, perm=(1, 0)), new_shape=(1, (_H * _V), _HID)
            )

    @module(entry="moe")
    class KimiMoe:
        """The MoE every non-dense layer holds: sigmoid router, 256 experts,
        top-8, one shared expert.

        Selection reads ``sigmoid(logits) + e_score_correction_bias``; the
        routing weights are selected from the *unbiased* sigmoid scores. So the
        bias moves *which* experts run without appearing in *how much* they
        count. ``num_expert_group = topk_group = 1`` makes the published
        grouped-top-k the identity -- one group holds every expert -- so there
        is no group stage here.
        """

        @func
        def router(
            tokens: Tensor[(S, _HID), _DT],
            w_router: ConstTensor[(_HID, _E), _DT],
            bias: ConstTensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
        ):
            # f32 throughout: selection has to agree with the oracle's, and a top-k
            # over bf16 scores can tie differently.
            logits = tf.cast(tf.matmul(tokens, w_router), dtype="f32")
            scores = tf.sigmoid(logits)
            biased = scores + tf.cast(bias, dtype="f32")
            top_biased, indices = tf.topk(biased, k=cfg.num_experts_per_token, axis=-1)
            # The weights come from the unbiased scores; subtracting the selected
            # experts' bias recovers them exactly.
            selected_bias = tf.reshape(
                tf.index_select(
                    bias,
                    tf.reshape(indices, new_shape=(S * _TOPK,)),
                    dim=0,
                ),
                new_shape=(S, _TOPK),
            )
            unbiased = top_biased - tf.cast(selected_bias, dtype="f32")
            denom = tf.reduce(unbiased, axes=(-1,), keepdim=True, kind="sum")
            # normalise, *then* scale: moe_renormalize is true and the scaling factor
            # is applied to the normalised weights, not folded into the denominator.
            weights = unbiased / denom * tf.cast(routed_scale, dtype="f32")
            return tf.cast(weights, dtype=_DT), indices

        @func
        def shared_expert(
            tokens: Tensor[(S, _HID), _DT],
            sh_gate: ConstTensor[(1, _HID, _SI), _DT],
            sh_up: ConstTensor[(1, _HID, _SI), _DT],
            sh_down: ConstTensor[(1, _SI, _HID), _DT],
        ) -> Tensor[(S, _HID), _DT]:
            # One dense SwiGLU expert every token pays for, unscaled: the routed
            # scaling factor applies to the routed branch only. Parameters are
            # named as `moe` names them: one Module's weights are keyed by name,
            # so a shared expert weight called `w_gate` here would collide with
            # the expert stack `moe` declares under that name.
            x = tf.reshape(tokens, new_shape=(1, S, _HID))
            gate = tf.matmul(x, sh_gate)
            up = tf.matmul(x, sh_up)
            h = tf.silu(gate) * up
            return tf.reshape(tf.matmul(h, sh_down), new_shape=(S, _HID))

        @func
        def moe(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_post: ConstTensor[(_HID,), _DT],
            w_router: ConstTensor[(_HID, _E), _DT],
            bias: ConstTensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
            w_gate: ConstTensor[(_E, cfg.moe_intermediate_size, _HID), _DT],
            w_up: ConstTensor[(_E, cfg.moe_intermediate_size, _HID), _DT],
            w_down: ConstTensor[(_E, _HID, cfg.moe_intermediate_size), _DT],
            sh_gate: ConstTensor[(1, _HID, _SI), _DT],
            sh_up: ConstTensor[(1, _HID, _SI), _DT],
            sh_down: ConstTensor[(1, _SI, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            # Fused post-attention RMSNorm + MoE, no residual (the layer owns that).
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_post
            tokens = tf.reshape(hn, new_shape=(S, _HID))
            weights, indices = router(tokens, w_router, bias, routed_scale)

            # Expert selection is runtime data: the indices select the
            # expert weights and a batched matmul over [tokens, top_k]. No static
            # 256-way expansion and no Python control flow.
            flat_indices = tf.reshape(indices, new_shape=(S * _TOPK,))
            g_sel = tf.reshape(
                tf.index_select(w_gate, flat_indices, dim=0),
                new_shape=(S, _TOPK, _MI, _HID),
            )
            u_sel = tf.reshape(
                tf.index_select(w_up, flat_indices, dim=0),
                new_shape=(S, _TOPK, _MI, _HID),
            )
            d_sel = tf.reshape(
                tf.index_select(w_down, flat_indices, dim=0),
                new_shape=(S, _TOPK, _HID, _MI),
            )
            tok4 = tf.reshape(tokens, new_shape=(S, 1, _HID, 1))
            gate = tf.reshape(tf.matmul(g_sel, tok4), new_shape=(S, _TOPK, _MI))
            up = tf.reshape(tf.matmul(u_sel, tok4), new_shape=(S, _TOPK, _MI))
            h = tf.silu(gate) * up
            h4 = tf.reshape(h, new_shape=(S, _TOPK, _MI, 1))
            down = tf.reshape(tf.matmul(d_sel, h4), new_shape=(S, _TOPK, _HID))
            routed = tf.reduce(
                down * tf.reshape(weights, new_shape=(S, _TOPK, 1)),
                axes=(1,),
                keepdim=False,
                kind="sum",
            )
            shared = shared_expert(tokens, sh_gate, sh_up, sh_down)
            return tf.reshape(routed + shared, new_shape=(1, S, _HID))

        # ---- raw checkpoint -> declared weight ---------------------------
        #
        # `gamma_post` is the layer's `post_attention_layernorm` (Absolutely
        # addressed, no converter: `KimiRMSNorm` is flat). `bias` is stored as
        # declared. `w_gate` / `w_up` / `w_down` are stored per expert as
        # `experts.{i}.w1/w3/w2.weight`, already (out, in) per expert -- the
        # alias table names the one-to-many group and `prepare` stacks it, so
        # no converter. The shared expert is three `nn.Linear`s.

        @moe.converter("w_router")
        def _(
            router_weight: ConstTensor[(_E, _HID), _DT],
        ) -> Tensor[(_HID, _E), _DT]:
            # `KimiMoEGate.weight` is (experts, hidden); the matmul wants it the
            # other way.
            return tf.transpose(router_weight, perm=(1, 0))

        @moe.converter("sh_gate")
        def _(
            shared_gate_proj_weight: ConstTensor[(_SI, _HID), _DT],
        ) -> Tensor[(1, _HID, _SI), _DT]:
            return tf.reshape(
                tf.transpose(shared_gate_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _SI)
            )

        @moe.converter("sh_up")
        def _(
            shared_up_proj_weight: ConstTensor[(_SI, _HID), _DT],
        ) -> Tensor[(1, _HID, _SI), _DT]:
            return tf.reshape(
                tf.transpose(shared_up_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _SI)
            )

        @moe.converter("sh_down")
        def _(
            shared_down_proj_weight: ConstTensor[(_HID, _SI), _DT],
        ) -> Tensor[(1, _SI, _HID), _DT]:
            return tf.reshape(
                tf.transpose(shared_down_proj_weight, perm=(1, 0)), new_shape=(1, _SI, _HID)
            )

    @module(entry="mlp")
    class KimiDenseMlp:
        """Layer 0's feed-forward: a dense SwiGLU (``KimiMLP``), which is what
        ``first_k_dense_replace: 1`` puts in place of the MoE. Fused
        post-attention RMSNorm, no residual (the layer owns that)."""

        @func
        def mlp(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_post: ConstTensor[(_HID,), _DT],
            w_gate: ConstTensor[(1, _HID, _I), _DT],
            w_up: ConstTensor[(1, _HID, _I), _DT],
            w_down: ConstTensor[(1, _I, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_post
            gate = tf.matmul(hn, w_gate)
            up = tf.matmul(hn, w_up)
            return tf.matmul(tf.silu(gate) * up, w_down)

        # ---- raw checkpoint -> declared weight ---------------------------

        @mlp.converter("w_gate")
        def _(
            mlp_gate_proj_weight: ConstTensor[(_I, _HID), _DT],
        ) -> Tensor[(1, _HID, _I), _DT]:
            return tf.reshape(
                tf.transpose(mlp_gate_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _I)
            )

        @mlp.converter("w_up")
        def _(
            mlp_up_proj_weight: ConstTensor[(_I, _HID), _DT],
        ) -> Tensor[(1, _HID, _I), _DT]:
            return tf.reshape(
                tf.transpose(mlp_up_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _I)
            )

        @mlp.converter("w_down")
        def _(
            mlp_down_proj_weight: ConstTensor[(_HID, _I), _DT],
        ) -> Tensor[(1, _I, _HID), _DT]:
            return tf.reshape(
                tf.transpose(mlp_down_proj_weight, perm=(1, 0)), new_shape=(1, _I, _HID)
            )

    def _moe_layer_forward(self, hidden, mixer_args, routed_scale):
        """One decode step: mixer + residual, then MoE + residual.

        Mirrors ``KimiDecoderLayer.forward``. The two pre-norms are not here
        because each block fuses its own -- the mixer fuses ``input_layernorm``
        and the MoE block fuses ``post_attention_layernorm``.

        *mixer_args* is what the mixer is handed after the hidden state, its
        cache already spliced in. What comes back is the layer output and
        whatever state the mixer produced, passed through untouched for the
        caller to advance.
        """
        mixed, *state = self.mixer(hidden, *mixer_args)
        attended = self.residual_add(hidden, mixed)
        ffn_out = self.moe(attended, routed_scale)
        return self.residual_add(attended, ffn_out), tuple(state)

    def _dense_layer_forward(self, hidden, mixer_args, routed_scale):
        """Layer 0's step: mixer + residual, then the dense MLP + residual.

        Same shape as the MoE layers' step; *routed_scale* is taken and ignored
        so the walk hands every layer the same arguments.
        """
        mixed, *state = self.mixer(hidden, *mixer_args)
        attended = self.residual_add(hidden, mixed)
        ffn_out = self.mlp(attended)
        return self.residual_add(attended, ffn_out), tuple(state)

    @module(entry="residual_add")
    class KimiKdaDenseLayer:
        mixer = KimiKda.renamed("mixer")
        mlp = KimiDenseMlp.renamed("mlp")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _dense_layer_forward

    @module(entry="residual_add")
    class KimiKdaMoeLayer:
        mixer = KimiKda.renamed("mixer")
        moe = KimiMoe.renamed("moe")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _moe_layer_forward

    @module(entry="residual_add")
    class KimiMlaMoeLayer:
        mixer = KimiMla.renamed("mixer")
        moe = KimiMoe.renamed("moe")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _moe_layer_forward

    # ── the stack ────────────────────────────────────────────────────────────

    def _layer_kinds(cfg: KimiLinearConfig) -> tuple:
        """Each layer's (mixer, ffn) kind, in published order.

        The mixers come from `linear_attn_config`'s 1-based lists; the ffn from
        `first_k_dense_replace` / `moe_layer_freq` -- the same facts
        ``KimiDecoderLayer.__init__`` reads.
        """
        kinds = []
        for index in range(cfg.num_hidden_layers):
            mixer = "kda" if cfg.is_kda_layer(index) else "mla"
            moe = (
                cfg.num_experts is not None
                and index >= cfg.first_k_dense_replace
                and index % cfg.moe_layer_freq == 0
            )
            kinds.append((mixer, "moe" if moe else "dense"))
        return tuple(kinds)

    LAYER_KINDS = _layer_kinds(cfg)

    #: Which layer class each (mixer, ffn) kind names.
    LAYER_TYPE = {
        ("kda", "dense"): KimiKdaDenseLayer,
        ("kda", "moe"): KimiKdaMoeLayer,
        ("mla", "moe"): KimiMlaMoeLayer,
    }

    #: _DT as torch spells it -- the state below is at the kernels' own dtype.
    _TORCH_DT = to_torch_dtype(DType.from_name(_DT))

    #: The parameters a mixer declares for its own state, whichever kind it is.
    #: The root splices a layer's cache in at the first of them.
    _CACHE_PARAMS = frozenset(
        {"k_cache", "v_cache", "conv_state_q", "conv_state_k", "conv_state_v", "state"}
    )

    def _with_cache(mixer, mixer_args, cache):
        """*mixer_args* with *cache* spliced in where *mixer* declares its state.

        The position is counted over the parameters a step is handed, since a
        loading fills the weights by name, and read from the Module a loading
        stands over so that one rule answers for both.
        """
        node = getattr(mixer, "module", mixer)
        names = [
            param.name for param in node.entry_function().params if not param.is_const
        ][1:]
        # `next`, not `min`: `from tilefoundry.dsl.tf import *` binds `min` to the op.
        at = next(index for index, name in enumerate(names) if name in _CACHE_PARAMS)
        return (*mixer_args[:at], *cache, *mixer_args[at:])

    def advance_state(kind, state, fresh):
        """A layer of *kind*'s next state, from what its mixer returned.

        KDA **replaces**: the recurrent matrix and the three convolution windows
        are fixed size, and the mixer returns them whole -- reordered here into
        the cache's (conv_q, conv_k, conv_v, state) layout, which is the order
        the entry declares them in. MLA **appends**: key and value grow along
        the position axis by the one entry the step returned.
        """
        import torch  # noqa: PLC0415

        if kind == "kda":
            state_next, conv_q_next, conv_k_next, conv_v_next = fresh
            return (conv_q_next, conv_k_next, conv_v_next, state_next)
        return tuple(torch.cat([old, new], dim=1) for old, new in zip(state, fresh))

    @module(
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 132), Topology("thread", 512)),
    )
    class KimiLinear48BA3B:
        """The layer stack in published order, and the step around it --
        embedding, the walk, the closing norm, the head. Each layer is an
        independent copy, so an analysis of one annotates only it."""

        # The published layer cycle determines each layer Module.
        layers = tuple(
            LAYER_TYPE[kind].renamed(f"layer{index}")
            for index, kind in enumerate(LAYER_KINDS)
        )

        @func
        def embed(
            table: ConstTensor[(cfg.vocab_size, _HID), _DT],
            token_ids: Tensor[(1,), "i64"],
        ) -> Tensor[(1, S, _HID), _DT]:
            # HF `KimiLinearModel.embed_tokens`: the decoded token's own row.
            return tf.reshape(
                tf.index_select(table, token_ids, dim=0), new_shape=(1, S, _HID)
            )

        @func
        def final_rms_norm(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_final: ConstTensor[(_HID,), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            # HF `KimiLinearModel.norm`, applied once after the last layer.
            # `KimiRMSNorm` rounds the normalised value to the input dtype before
            # the learned scale multiplies it, so the norm is written out rather
            # than `tf.rms_norm`, which keeps f32 through that multiply.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            return tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_final

        @func
        def lm_head(
            hidden: Tensor[(1, S, _HID), _DT],
            w_head: ConstTensor[(_HID, cfg.vocab_size), _DT],
        ) -> Tensor[(1, cfg.vocab_size), _DT]:
            # HF `KimiLinearForCausalLM.lm_head`, over the one token being decoded.
            return tf.matmul(tf.reshape(hidden, new_shape=(1, _HID)), w_head)

        @lm_head.converter("w_head")
        def _(
            head_weight_raw: ConstTensor[(cfg.vocab_size, _HID), _DT],
        ) -> Tensor[(_HID, cfg.vocab_size), _DT]:
            # HF stores the head as (vocab, hidden); the matmul above wants it the
            # other way. `tie_word_embeddings` is false and the checkpoint ships a
            # real `lm_head.weight`, so nothing aliases the embedding table.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        def decode_hidden(self, hidden, layer_args, caches, routed_scale):
            """One decode step through every layer, then the final norm.

            *layer_args* is one layer's mixer arguments per layer, carrying no
            state; *caches* is each layer's own state, spliced into its mixer
            call; *routed_scale* is the MoE's scaling factor, the same tensor
            for every layer. What comes back is the normed hidden state and
            each layer's fresh state.
            """
            if len(layer_args) != len(self.modules) or len(caches) != len(self.modules):
                raise ValueError(
                    f"decoder has {len(self.modules)} layers but was given "
                    f"{len(layer_args)} argument tuples and {len(caches)} caches"
                )
            states = []
            for layer, mixer_args, cache in zip(self.modules, layer_args, caches):
                hidden, state = layer(
                    hidden, _with_cache(layer.mixer, mixer_args, cache), routed_scale
                )
                states.append(state)
            return self.final_rms_norm(hidden), tuple(states)

        def forward(self, token_ids, layer_args, caches, routed_scale):
            """One decode step of the whole model: token ids in, logits out.

            The fresh per-layer state comes out beside the logits; growing
            *caches* with it is the caller's step, through `append_cache`.
            """
            hidden = self.embed(token_ids)
            normed, states = self.decode_hidden(hidden, layer_args, caches, routed_scale)
            return self.lm_head(normed), states

        def init_caches(self, device=None):
            """The per-layer state container, one entry per layer.

            A KDA layer's four halves are genuinely zero at the start: Hugging
            Face left-pads the convolution windows when the context is shorter
            than them, and `initial_state=None` is the zero recurrent matrix.
            An MLA layer gets a container of no positions, which `ctx_len`
            admits: the first step of a sequence attends the one position it
            brings itself.
            """
            import torch  # noqa: PLC0415

            device = torch.accelerator.current_accelerator() if device is None else device
            entries = []
            for mixer, _ffn in LAYER_KINDS:
                if mixer == "kda":
                    entries.append((
                        torch.zeros(1, _WS, _KP, dtype=_TORCH_DT, device=device),
                        torch.zeros(1, _WS, _KP, dtype=_TORCH_DT, device=device),
                        torch.zeros(1, _WS, _KP, dtype=_TORCH_DT, device=device),
                        torch.zeros(1, _KH, _KD, _KD, dtype=_TORCH_DT, device=device),
                    ))
                else:
                    entries.append((
                        torch.zeros(1, 0, _H, _QK, dtype=_TORCH_DT, device=device),
                        torch.zeros(1, 0, _H, _V, dtype=_TORCH_DT, device=device),
                    ))
            return tuple(entries)

        def append_cache(self, caches, fresh):
            """Every layer's state advanced by the step it just took: a kernel
            hands back its own token's entry, and joining it on (or replacing
            with it) is the caller's."""
            return tuple(
                advance_state(mixer, cache, new)
                for (mixer, _ffn), cache, new in zip(LAYER_KINDS, caches, fresh)
            )

        def prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
            """The token and each layer's non-state activations for one decode step.

            The MLA mixers get the identity rotary -- cos = 1, sin = 0 -- which
            is what `mla_use_nope: true` means at runtime: the 64 rotary-half
            dimensions still enter the score and the scaling denominator; they
            are simply not rotated.
            """
            import torch  # noqa: PLC0415

            device = torch.accelerator.current_accelerator() if device is None else device
            token_ids = input_ids[step].reshape(1).to(device=device, dtype=torch.int64)
            layer_args = []
            for mixer, _ffn in LAYER_KINDS:
                if mixer == "mla":
                    cos_cache = torch.ones(S, _ROPE, dtype=_TORCH_DT, device=device)
                    sin_cache = torch.zeros(S, _ROPE, dtype=_TORCH_DT, device=device)
                    pos_ids = torch.zeros(S, dtype=torch.int32, device=device)
                    scale = torch.full(
                        (1, 1, 1, 1), _QK ** -0.5, dtype=_TORCH_DT, device=device
                    )
                    layer_args.append((cos_cache, sin_cache, pos_ids, scale))
                else:
                    scale = torch.full(
                        (1, 1, 1), _KD ** -0.5, dtype=_TORCH_DT, device=device
                    )
                    layer_args.append((scale,))
            routed_scale = torch.full(
                (1, 1), cfg.routed_scaling_factor, dtype=_TORCH_DT, device=device
            )
            return token_ids, tuple(layer_args), caches, routed_scale

    return {
        "KimiKda": KimiKda,
        "KimiMla": KimiMla,
        "KimiMoe": KimiMoe,
        "KimiDenseMlp": KimiDenseMlp,
        "KimiKdaDenseLayer": KimiKdaDenseLayer,
        "KimiKdaMoeLayer": KimiKdaMoeLayer,
        "KimiMlaMoeLayer": KimiMlaMoeLayer,
        "KimiLinear48BA3B": KimiLinear48BA3B,
        "LAYER_KINDS": LAYER_KINDS,
        "LAYER_TYPE": LAYER_TYPE,
        "advance_state": advance_state,
    }


# ---------------------------------------------------------------------------
# The published configuration, as module-level names a CLI selector can address.
# ---------------------------------------------------------------------------

_REAL = build(published())

KimiKda = _REAL["KimiKda"]
KimiMla = _REAL["KimiMla"]
KimiMoe = _REAL["KimiMoe"]
KimiDenseMlp = _REAL["KimiDenseMlp"]
KimiKdaDenseLayer = _REAL["KimiKdaDenseLayer"]
KimiKdaMoeLayer = _REAL["KimiKdaMoeLayer"]
KimiMlaMoeLayer = _REAL["KimiMlaMoeLayer"]
KimiLinear48BA3B = _REAL["KimiLinear48BA3B"]
LAYER_KINDS = _REAL["LAYER_KINDS"]
LAYER_TYPE = _REAL["LAYER_TYPE"]
advance_state = _REAL["advance_state"]

config = published()


# ---------------------------------------------------------------------------
# {canonical: raw} for the published checkpoint.
# ---------------------------------------------------------------------------
#
# The resource prefix is made to *track* the checkpoint's own paths: segment
# aliases rename each layer Module onto `model.layers.{i}`, `mixer` onto
# `self_attn`, and `moe` onto `block_sparse_moe` (the dense layer-0 MLP is
# already named `mlp`, as the checkpoint names it). Every leaf below is then a
# bare relative name that serves every instance of its scope uniformly, except
# the two layer-owned norms a child consumes (`input_layernorm` is the layer's,
# not the mixer's), which aliasing can only reach downward from -- those are
# `Absolute`, keyed by the resolved path the lookup is made at. The expert
# stacks are the one-to-many case: a tuple value, `load_group` reads it and
# `prepare` stacks it in declared order.


def hf_alias(cfg: KimiLinearConfig | None = None, *, prefix="model", head="lm_head.weight"):
    """The alias table `prepare` reads the published checkpoint through.

    Every expert group is 256 raw tensors in index order; every other entry is
    one-to-one. *head* is a parameter because a tied model would alias it to
    the embedding table instead -- this one is untied and ships its own.
    """
    cfg = config if cfg is None else cfg
    kinds = tuple(
        (
            "kda" if cfg.is_kda_layer(index) else "mla",
            "moe"
            if (
                cfg.num_experts is not None
                and index >= cfg.first_k_dense_replace
                and index % cfg.moe_layer_freq == 0
            )
            else "dense",
        )
        for index in range(cfg.num_hidden_layers)
    )
    alias: dict[str, object] = {
        # root leaves
        "table": f"{prefix}.embed_tokens.weight",
        "gamma_final": f"{prefix}.norm.weight",
        "head_weight_raw": head,
        # segment renames
        "mixer": "self_attn",
        "moe": "block_sparse_moe",
        # mixer leaves shared by both kinds
        "q_proj_weight": "q_proj.weight",
        "o_proj_weight": "o_proj.weight",
        # KDA leaves
        "k_proj_weight": "k_proj.weight",
        "v_proj_weight": "v_proj.weight",
        "q_conv1d_weight": "q_conv1d.weight",
        "k_conv1d_weight": "k_conv1d.weight",
        "v_conv1d_weight": "v_conv1d.weight",
        "f_a_proj_weight": "f_a_proj.weight",
        "f_b_proj_weight": "f_b_proj.weight",
        "dt_bias_raw": "dt_bias",
        "a_log_raw": "A_log",
        "b_proj_weight": "b_proj.weight",
        "g_a_proj_weight": "g_a_proj.weight",
        "g_b_proj_weight": "g_b_proj.weight",
        "gamma_o": "o_norm.weight",
        # MLA leaves
        "kv_a_proj_weight": "kv_a_proj_with_mqa.weight",
        "gamma_kv_a": "kv_a_layernorm.weight",
        "kv_b_proj_weight": "kv_b_proj.weight",
        # MoE leaves
        "router_weight": "gate.weight",
        "bias": "gate.e_score_correction_bias",
        "w_gate": tuple(f"experts.{j}.w1.weight" for j in range(cfg.num_experts)),
        "w_up": tuple(f"experts.{j}.w3.weight" for j in range(cfg.num_experts)),
        "w_down": tuple(f"experts.{j}.w2.weight" for j in range(cfg.num_experts)),
        "shared_gate_proj_weight": "shared_experts.gate_proj.weight",
        "shared_up_proj_weight": "shared_experts.up_proj.weight",
        "shared_down_proj_weight": "shared_experts.down_proj.weight",
        # dense MLP leaves (layer 0)
        "mlp_gate_proj_weight": "gate_proj.weight",
        "mlp_up_proj_weight": "up_proj.weight",
        "mlp_down_proj_weight": "down_proj.weight",
    }
    for index, (_mixer_kind, ffn_kind) in enumerate(kinds):
        alias[f"layer{index}"] = f"{prefix}.layers.{index}"
        layer = f"{prefix}.layers.{index}"
        alias[f"{layer}.self_attn.gamma_in"] = Absolute(f"{layer}.input_layernorm.weight")
        child = "block_sparse_moe" if ffn_kind == "moe" else "mlp"
        alias[f"{layer}.{child}.gamma_post"] = Absolute(
            f"{layer}.post_attention_layernorm.weight"
        )
    return alias


__all__ = [
    "KimiDenseMlp",
    "KimiKda",
    "KimiKdaDenseLayer",
    "KimiKdaMoeLayer",
    "KimiLinear48BA3B",
    "KimiLinearConfig",
    "KimiMla",
    "KimiMlaMoeLayer",
    "KimiMoe",
    "LAYER_KINDS",
    "LAYER_TYPE",
    "advance_state",
    "build",
    "config",
    "hf_alias",
    "published",
]
