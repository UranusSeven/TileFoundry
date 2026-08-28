"""Independent S-symbolic prefill HIR for Kimi-Linear-48B-A3B-Instruct.

This is the sole authored model: a whole prompt enters, 27 prefill layers run,
and the last position produces logits. KDA recurrence is represented by the
HIR-only ``tf.kda_prefill`` external op. No decode function or cache handoff is
part of this tree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import ops  # noqa: F401 -- registers external tf ops
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




def _layer_kinds(cfg: KimiLinearConfig) -> tuple:
    """Each published layer's (mixer, FFN) kind in stack order."""
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


def build(cfg: KimiLinearConfig):
    """This model's prefill shell at *cfg*, mirroring ``model.build``.

    The prompt length is the one range this tree carries; everything else is
    the published dimension. The layer-kind table and every function are built directly from the config.
    """
    # The prompt length: one token at minimum, the published horizon at most.
    S = DimVar("seq_len", 1, cfg.model_max_length)

    _HID = cfg.hidden_size               # 2304
    _H = cfg.num_attention_heads         # 32
    _NOPE = cfg.qk_nope_head_dim         # 128
    _ROPE = cfg.qk_rope_head_dim         # 64, carried unrotated (NoPE)
    _QK = _NOPE + _ROPE                     # 192, the score dim and so the scaling one
    _V = cfg.v_head_dim                  # 128
    _KVB = _NOPE + _V                       # 256, kv_b_proj's per-head output

    _KDA = cfg.linear_attn_config
    _KH = _KDA["num_heads"]                 # 32
    _KD = _KDA["head_dim"]                  # 128
    _KP = _KH * _KD                         # 4096
    _W = _KDA["short_conv_kernel_size"]      # 4 convolution taps
    _WS = _W - 1                              # 3 left-padding positions

    _E = cfg.num_experts
    _TOPK = cfg.num_experts_per_token
    _MI = cfg.moe_intermediate_size
    _SI = _MI * cfg.num_shared_experts

    _I = cfg.intermediate_size           # 9216, the dense layer-0 MLP

    _EPS = cfg.rms_norm_eps

    _DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
        str(cfg.dtype).removeprefix("torch.")
    ]

    LAYER_KINDS = _layer_kinds(cfg)

    @module(entry="kda_prefill")
    class KimiKdaPrefill:
        """KDA projections and gates over S, with one external recurrence."""

        @func
        def kda_prefill(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_in: ConstTensor[(_HID,), _DT],
            w_q: ConstTensor[(1, _HID, _KP), _DT],
            w_k: ConstTensor[(1, _HID, _KP), _DT],
            w_v: ConstTensor[(1, _HID, _KP), _DT],
            conv_w_q: ConstTensor[(_W, _KP), _DT],
            conv_w_k: ConstTensor[(_W, _KP), _DT],
            conv_w_v: ConstTensor[(_W, _KP), _DT],
            w_f_a: ConstTensor[(1, _HID, _KD), _DT],
            w_f_b: ConstTensor[(1, _KD, _KP), _DT],
            dt_bias: ConstTensor[(_KP,), _DT],
            a_log: ConstTensor[(_KH,), _DT],
            w_b: ConstTensor[(1, _HID, _KH), _DT],
            w_g_a: ConstTensor[(1, _HID, _KD), _DT],
            w_g_b: ConstTensor[(1, _KD, _KP), _DT],
            gamma_o: ConstTensor[(_KD,), _DT],
            w_o: ConstTensor[(1, _KP, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype=_DT) * gamma_in
            q = tf.reshape(tf.causal_depthwise_conv1d(tf.matmul(hn, w_q), conv_w_q), new_shape=(1, S, _KH, _KD))
            k = tf.reshape(tf.causal_depthwise_conv1d(tf.matmul(hn, w_k), conv_w_k), new_shape=(1, S, _KH, _KD))
            v = tf.reshape(tf.causal_depthwise_conv1d(tf.matmul(hn, w_v), conv_w_v), new_shape=(1, S, _KH, _KD))
            g_raw = tf.reshape(tf.matmul(tf.matmul(hn, w_f_a), w_f_b) + dt_bias, new_shape=(1, S, _KH, _KD))
            decay_rate = -tf.exp(tf.reshape(a_log, new_shape=(1, 1, _KH, 1)))
            g = decay_rate * tf.softplus(g_raw)
            beta = tf.reshape(tf.sigmoid(tf.matmul(hn, w_b)), new_shape=(1, S, _KH))
            recurrent = tf.kda_prefill(q, k, v, g, beta, scale=_KD ** -0.5)
            g2 = tf.reshape(tf.matmul(tf.matmul(hn, w_g_a), w_g_b), new_shape=(1, S, _KH, _KD))
            gated = tf.rms_norm(recurrent, gamma_o, eps=_EPS) * tf.sigmoid(g2)
            return tf.matmul(tf.reshape(gated, new_shape=(1, S, _KP)), w_o)

        @kda_prefill.converter("w_q")
        def _(
            q_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(q_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_prefill.converter("w_k")
        def _(
            k_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(k_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_prefill.converter("w_v")
        def _(
            v_proj_weight: ConstTensor[(_KP, _HID), _DT],
        ) -> Tensor[(1, _HID, _KP), _DT]:
            return tf.reshape(
                tf.transpose(v_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KP)
            )

        @kda_prefill.converter("conv_w_q")
        def _(
            q_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            # A depthwise `nn.Conv1d` stores (channels, 1, kernel); short_conv
            # wants (kernel, channels).
            return tf.transpose(
                tf.reshape(q_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_prefill.converter("conv_w_k")
        def _(
            k_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            return tf.transpose(
                tf.reshape(k_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_prefill.converter("conv_w_v")
        def _(
            v_conv1d_weight: ConstTensor[(_KP, 1, _W), _DT],
        ) -> Tensor[(_W, _KP), _DT]:
            return tf.transpose(
                tf.reshape(v_conv1d_weight, new_shape=(_KP, _W)), perm=(1, 0)
            )

        @kda_prefill.converter("w_f_a")
        def _(
            f_a_proj_weight: ConstTensor[(_KD, _HID), _DT],
        ) -> Tensor[(1, _HID, _KD), _DT]:
            return tf.reshape(
                tf.transpose(f_a_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KD)
            )

        @kda_prefill.converter("w_f_b")
        def _(
            f_b_proj_weight: ConstTensor[(_KP, _KD), _DT],
        ) -> Tensor[(1, _KD, _KP), _DT]:
            return tf.reshape(
                tf.transpose(f_b_proj_weight, perm=(1, 0)), new_shape=(1, _KD, _KP)
            )

        @kda_prefill.converter("dt_bias")
        def _(
            dt_bias_raw: ConstTensor[(_KP,), "f32"],
        ) -> Tensor[(_KP,), _DT]:
            # Stored f32; declared at the kernel's own dtype.
            return tf.cast(dt_bias_raw, dtype=_DT)

        @kda_prefill.converter("a_log")
        def _(
            a_log_raw: ConstTensor[(1, 1, _KH, 1), "f32"],
        ) -> Tensor[(_KH,), _DT]:
            # Stored f32 as (1, 1, heads, 1); declared bf16 (heads,).
            return tf.cast(tf.reshape(a_log_raw, new_shape=(_KH,)), dtype=_DT)

        @kda_prefill.converter("w_b")
        def _(
            b_proj_weight: ConstTensor[(_KH, _HID), _DT],
        ) -> Tensor[(1, _HID, _KH), _DT]:
            return tf.reshape(
                tf.transpose(b_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KH)
            )

        @kda_prefill.converter("w_g_a")
        def _(
            g_a_proj_weight: ConstTensor[(_KD, _HID), _DT],
        ) -> Tensor[(1, _HID, _KD), _DT]:
            return tf.reshape(
                tf.transpose(g_a_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _KD)
            )

        @kda_prefill.converter("w_g_b")
        def _(
            g_b_proj_weight: ConstTensor[(_KP, _KD), _DT],
        ) -> Tensor[(1, _KD, _KP), _DT]:
            return tf.reshape(
                tf.transpose(g_b_proj_weight, perm=(1, 0)), new_shape=(1, _KD, _KP)
            )

        @kda_prefill.converter("w_o")
        def _(
            o_proj_weight: ConstTensor[(_HID, _KP), _DT],
        ) -> Tensor[(1, _KP, _HID), _DT]:
            return tf.reshape(
                tf.transpose(o_proj_weight, perm=(1, 0)), new_shape=(1, _KP, _HID)
            )


    @module(entry="mla_prefill")
    class KimiMlaPrefill:
        """An MLA layer's mixer over a whole prompt.

        The projection path is the decode step's, evaluated for S positions
        in one pass; the attention is the textbook masked-softmax form the
        HF eager reference (``eager_attention_forward`` reached from
        ``KimiMLAAttention.forward``) computes. The cache this layer owns is
        built, not read: prefill starts from no context.
        """

        @func
        def mla_prefill(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_in: ConstTensor[(_HID,), _DT],
            w_q: ConstTensor[(1, _HID, (_H * _QK)), _DT],
            w_kv_a: ConstTensor[(1, _HID, (cfg.kv_lora_rank + _ROPE)), _DT],
            gamma_kv_a: ConstTensor[(cfg.kv_lora_rank,), _DT],
            w_kv_b: ConstTensor[(1, cfg.kv_lora_rank, (_H * _KVB)), _DT],
            scale: Tensor[(1, 1, 1, 1), _DT],
            w_o: ConstTensor[(1, (_H * _V), _HID), _DT],
        ):
            # Fused input RMSNorm (round-before-gamma, as KimiRMSNorm) + MLA
            # over the prompt, no residual (the layer owns that). Returns the
            # per-position output and the prompt's keys and values, which the
            # caller adopts as the decode cache.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_in

            # The query is a plain projection: q_lora_rank is null, so there is
            # no q_a/q_b pair to fold.
            q = tf.reshape(tf.matmul(hn, w_q), new_shape=(1, S, _H, _QK))

            # One projection yields the latent and the rope-width part together,
            # and that part is shared across heads -- the "MQA" in
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
            v_all = kv[:, :, :, _NOPE:_KVB]

            # NoPE: the shared 64-wide part is broadcast over the heads
            # *unrotated*. The checkpoint asserts mla_use_nope and the oracle's
            # forward applies no rotary at all, so there are no cos/sin inputs
            # here -- decode's cos = 1, sin = 0 identity is this same fact.
            k_rot_h = tf.repeat_interleave(
                tf.reshape(k_rot_shared, new_shape=(1, S, 1, _ROPE)), repeats=_H, axis=2
            )
            k_all = tf.concat([k_nope, k_rot_h], axis=-1)

            # Heads-first for the batched score matmul. Every key head serves
            # exactly one query head (num_key_value_heads == num_attention_heads),
            # so there is no GQA expansion.
            q_h = tf.transpose(q, perm=(0, 2, 1, 3))
            k_h = tf.transpose(k_all, perm=(0, 2, 1, 3))
            v_h = tf.transpose(v_all, perm=(0, 2, 1, 3))

            # HF eager scales the scores after the matmul, by the scalar; the
            # order is kept so the two agree at the same rounding points.
            scores = tf.matmul(q_h, tf.transpose(k_h, perm=(0, 1, 3, 2))) * scale

            # The causal mask, stated as data: a position may attend itself and
            # everything before it. -1e30 stands in for the additive
            # torch.finfo(bf16).min mask HF builds; both vanish in the softmax.
            pos = tf.arange(Tensor[(S,), "i64"])
            keep = tf.cmp_ge(
                tf.reshape(pos, new_shape=(S, 1)), tf.reshape(pos, new_shape=(1, S))
            )
            masked = tf.where(keep, scores, tf.full_like(scores, value=-1e30))

            # tf.softmax normalises in f32 and rounds back to bf16 -- precisely
            # softmax(..., dtype=torch.float32).to(query.dtype).
            probs = tf.softmax(masked, axis=-1)
            attn = tf.matmul(probs, v_h)
            out = tf.matmul(
                tf.reshape(
                    tf.transpose(attn, perm=(0, 2, 1, 3)), new_shape=(1, S, (_H * _V))
                ),
                w_o,
            )
            return out, k_all, v_all

        # ---- raw checkpoint -> declared weight ---------------------------
        #
        # Identical to the decode mixer's: the projections are the same
        # tensors, stored (out, in) against the (1, in, out) the matmuls want.
        # `gamma_in` is the layer's `input_layernorm` (Absolutely addressed,
        # no converter: `KimiRMSNorm` is flat).

        @mla_prefill.converter("w_q")
        def _(
            q_proj_weight: ConstTensor[((_H * _QK), _HID), _DT],
        ) -> Tensor[(1, _HID, (_H * _QK)), _DT]:
            return tf.reshape(
                tf.transpose(q_proj_weight, perm=(1, 0)), new_shape=(1, _HID, (_H * _QK))
            )

        @mla_prefill.converter("w_kv_a")
        def _(
            kv_a_proj_weight: ConstTensor[((cfg.kv_lora_rank + _ROPE), _HID), _DT],
        ) -> Tensor[(1, _HID, (cfg.kv_lora_rank + _ROPE)), _DT]:
            return tf.reshape(
                tf.transpose(kv_a_proj_weight, perm=(1, 0)),
                new_shape=(1, _HID, (cfg.kv_lora_rank + _ROPE)),
            )

        @mla_prefill.converter("w_kv_b")
        def _(
            kv_b_proj_weight: ConstTensor[((_H * _KVB), cfg.kv_lora_rank), _DT],
        ) -> Tensor[(1, cfg.kv_lora_rank, (_H * _KVB)), _DT]:
            return tf.reshape(
                tf.transpose(kv_b_proj_weight, perm=(1, 0)),
                new_shape=(1, cfg.kv_lora_rank, (_H * _KVB)),
            )

        @mla_prefill.converter("w_o")
        def _(
            o_proj_weight: ConstTensor[(_HID, (_H * _V)), _DT],
        ) -> Tensor[(1, (_H * _V), _HID), _DT]:
            return tf.reshape(
                tf.transpose(o_proj_weight, perm=(1, 0)), new_shape=(1, (_H * _V), _HID)
            )

    @module(entry="moe_prefill")
    class KimiMoePrefill:
        """The MoE over a whole prompt: the decode block with the token axis open.

        The block is position-wise, so the body is the decode body's with S
        the prompt length; selection, weighting, and the fused post-norm are
        unchanged token by token.
        """

        @func
        def router_prefill(
            tokens: Tensor[(S, _HID), _DT],
            w_router: ConstTensor[(_HID, _E), _DT],
            bias: ConstTensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
        ):
            # f32 throughout: selection has to agree with the oracle's, and a
            # top-k over bf16 scores can tie differently.
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
            # normalise, *then* scale: moe_renormalize is true and the scaling
            # factor is applied to the normalised weights, not folded into the
            # denominator.
            weights = unbiased / denom * tf.cast(routed_scale, dtype="f32")
            return tf.cast(weights, dtype=_DT), indices

        @func
        def shared_expert_prefill(
            tokens: Tensor[(S, _HID), _DT],
            sh_gate: ConstTensor[(1, _HID, _SI), _DT],
            sh_up: ConstTensor[(1, _HID, _SI), _DT],
            sh_down: ConstTensor[(1, _SI, _HID), _DT],
        ) -> Tensor[(S, _HID), _DT]:
            # One dense SwiGLU expert every token pays for, unscaled: the routed
            # scaling factor applies to the routed branch only.
            x = tf.reshape(tokens, new_shape=(1, S, _HID))
            gate = tf.matmul(x, sh_gate)
            up = tf.matmul(x, sh_up)
            h = tf.silu(gate) * up
            return tf.reshape(tf.matmul(h, sh_down), new_shape=(S, _HID))

        @func
        def moe_prefill(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_post: ConstTensor[(_HID,), _DT],
            w_router: ConstTensor[(_HID, _E), _DT],
            bias: ConstTensor[(_E,), _DT],
            routed_scale: Tensor[(1, 1), _DT],
            w_gate_up: ConstTensor[(_E, (2 * _MI), _HID), _DT],
            w_down: ConstTensor[(_E, _HID, cfg.moe_intermediate_size), _DT],
            sh_gate: ConstTensor[(1, _HID, _SI), _DT],
            sh_up: ConstTensor[(1, _HID, _SI), _DT],
            sh_down: ConstTensor[(1, _SI, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            # Fused post-attention RMSNorm + MoE, no residual (the layer owns
            # that).
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            hn = tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_post
            tokens = tf.reshape(hn, new_shape=(S, _HID))
            weights, indices = router_prefill(tokens, w_router, bias, routed_scale)

            # The checkpoint ABI is one packed gate/up tensor. HIR exposes its
            # two semantic halves as static views without a prefill-time concat.
            w_gate = w_gate_up[:, :_MI, :]
            w_up = w_gate_up[:, _MI:(2 * _MI), :]
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
            shared = shared_expert_prefill(tokens, sh_gate, sh_up, sh_down)
            return tf.reshape(routed + shared, new_shape=(1, S, _HID))

        # ---- raw checkpoint -> declared weight ---------------------------
        #
        # Identical to the decode block's: same tensors, same stores.
        # `gamma_post` is the layer's `post_attention_layernorm` (Absolutely
        # addressed, no converter). The expert stacks are one-to-many alias
        # groups that `prepare` stacks; the shared expert is three nn.Linears.

        @moe_prefill.converter("w_router")
        def _(
            router_weight: ConstTensor[(_E, _HID), _DT],
        ) -> Tensor[(_HID, _E), _DT]:
            # `KimiMoEGate.weight` is (experts, hidden); the matmul wants it the
            # other way.
            return tf.transpose(router_weight, perm=(1, 0))

        @moe_prefill.converter("sh_gate")
        def _(
            shared_gate_proj_weight: ConstTensor[(_SI, _HID), _DT],
        ) -> Tensor[(1, _HID, _SI), _DT]:
            return tf.reshape(
                tf.transpose(shared_gate_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _SI)
            )

        @moe_prefill.converter("sh_up")
        def _(
            shared_up_proj_weight: ConstTensor[(_SI, _HID), _DT],
        ) -> Tensor[(1, _HID, _SI), _DT]:
            return tf.reshape(
                tf.transpose(shared_up_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _SI)
            )

        @moe_prefill.converter("sh_down")
        def _(
            shared_down_proj_weight: ConstTensor[(_HID, _SI), _DT],
        ) -> Tensor[(1, _SI, _HID), _DT]:
            return tf.reshape(
                tf.transpose(shared_down_proj_weight, perm=(1, 0)), new_shape=(1, _SI, _HID)
            )

    @module(entry="mlp_prefill")
    class KimiDenseMlpPrefill:
        """Layer 0's feed-forward over a whole prompt: a dense SwiGLU.

        Fused post-attention RMSNorm, no residual (the layer owns that).
        Position-wise, so the decode body with S open.
        """

        @func
        def mlp_prefill(
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

        @mlp_prefill.converter("w_gate")
        def _(
            mlp_gate_proj_weight: ConstTensor[(_I, _HID), _DT],
        ) -> Tensor[(1, _HID, _I), _DT]:
            return tf.reshape(
                tf.transpose(mlp_gate_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _I)
            )

        @mlp_prefill.converter("w_up")
        def _(
            mlp_up_proj_weight: ConstTensor[(_I, _HID), _DT],
        ) -> Tensor[(1, _HID, _I), _DT]:
            return tf.reshape(
                tf.transpose(mlp_up_proj_weight, perm=(1, 0)), new_shape=(1, _HID, _I)
            )

        @mlp_prefill.converter("w_down")
        def _(
            mlp_down_proj_weight: ConstTensor[(_HID, _I), _DT],
        ) -> Tensor[(1, _I, _HID), _DT]:
            return tf.reshape(
                tf.transpose(mlp_down_proj_weight, perm=(1, 0)), new_shape=(1, _I, _HID)
            )

    # ── the layers ───────────────────────────────────────────────────────────
    #
    # Same two layer shapes as decode (mixer + residual, then FFN + residual),
    # with the mixer step replaced by the prefill form of its kind. The KDA
    # mixer is the decode module itself -- its step *is* the recurrence the
    # prefill loops.

    def _kda_layer_prefill(self, hidden, routed_scale):
        mixed = self.mixer(hidden)
        attended = self.residual_add(hidden, mixed)
        ffn_out = self.moe(attended, routed_scale) if hasattr(self, "moe") else self.mlp(attended)
        return self.residual_add(attended, ffn_out)

    def _mla_layer_prefill(self, hidden, routed_scale, scale):
        mixed, _keys, _values = self.mixer(hidden, scale)
        attended = self.residual_add(hidden, mixed)
        return self.residual_add(attended, self.moe(attended, routed_scale))

    @module(entry="residual_add")
    class KimiKdaDenseLayerPrefill:
        mixer = KimiKdaPrefill.renamed("mixer")
        mlp = KimiDenseMlpPrefill.renamed("mlp")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _kda_layer_prefill

    @module(entry="residual_add")
    class KimiKdaMoeLayerPrefill:
        mixer = KimiKdaPrefill.renamed("mixer")
        moe = KimiMoePrefill.renamed("moe")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _kda_layer_prefill

    @module(entry="residual_add")
    class KimiMlaMoeLayerPrefill:
        mixer = KimiMlaPrefill.renamed("mixer")
        moe = KimiMoePrefill.renamed("moe")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        forward = _mla_layer_prefill

    #: Which layer class each (mixer, ffn) kind names.
    LAYER_TYPE = {
        ("kda", "dense"): KimiKdaDenseLayerPrefill,
        ("kda", "moe"): KimiKdaMoeLayerPrefill,
        ("mla", "moe"): KimiMlaMoeLayerPrefill,
    }

    #: _DT as torch spells it -- the state below is at the kernels' own dtype.
    _TORCH_DT = to_torch_dtype(DType.from_name(_DT))

    @module(
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 132), Topology("thread", 512)),
    )
    class KimiLinear48BA3BPrefill:
        """The layer stack in published order over a whole prompt.

        And the step around it -- embedding, the walk, the closing norm, the
        head over the last position. Each layer is an independent copy, so an
        analysis of one annotates only it.
        """

        layers = tuple(
            LAYER_TYPE[kind].renamed(f"layer{index}")
            for index, kind in enumerate(LAYER_KINDS)
        )

        @func
        def embed_prefill(
            table: ConstTensor[(cfg.vocab_size, _HID), _DT],
            token_ids: Tensor[(S,), "i64"],
        ) -> Tensor[(1, S, _HID), _DT]:
            # HF `KimiLinearModel.embed_tokens`: each prompt token's own row.
            return tf.reshape(
                tf.index_select(table, token_ids, dim=0), new_shape=(1, S, _HID)
            )

        @func
        def final_rms_norm_prefill(
            hidden: Tensor[(1, S, _HID), _DT],
            gamma_final: ConstTensor[(_HID,), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            # HF `KimiLinearModel.norm`, applied once after the last layer, to
            # every position. `KimiRMSNorm` rounds the normalised value to the
            # input dtype before the learned scale multiplies it, so the norm
            # is written out rather than `tf.rms_norm`.
            hn32 = tf.cast(hidden, dtype="f32")
            hn_var = tf.reduce(hn32 * hn32, axes=(-1,), keepdim=True, kind="mean")
            return tf.cast(hn32 * tf.rsqrt(hn_var + _EPS), dtype="bf16") * gamma_final

        @func
        def lm_head_prefill(
            hidden: Tensor[(1, S, _HID), _DT],
            last_index: Tensor[(1,), "i64"],
            w_head: ConstTensor[(_HID, cfg.vocab_size), _DT],
        ) -> Tensor[(1, cfg.vocab_size), _DT]:
            # HF `KimiLinearForCausalLM.lm_head`, over the prompt's last
            # position only -- the next-token distribution. The gather is an
            # index_select on an index the orchestration passes: a
            # symbolic-bound slice (hidden[:, S-1:S]) type-infers but the
            # evaluator has no visit routine for a DimVar bound.
            last = tf.reshape(
                tf.index_select(hidden, last_index, dim=1), new_shape=(1, _HID)
            )
            return tf.matmul(last, w_head)

        @lm_head_prefill.converter("w_head")
        def _(
            head_weight_raw: ConstTensor[(cfg.vocab_size, _HID), _DT],
        ) -> Tensor[(_HID, cfg.vocab_size), _DT]:
            # HF stores the head as (vocab, hidden); the matmul above wants it
            # the other way. `tie_word_embeddings` is false and the checkpoint
            # ships a real `lm_head.weight`.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        def forward(self, token_ids, routed_scale, mla_scale):
            """Run the whole prompt and return last-position logits only."""
            import torch  # noqa: PLC0415

            hidden = self.embed_prefill(token_ids)
            for layer, (mixer_kind, _ffn_kind) in zip(self.modules, LAYER_KINDS):
                hidden = (
                    layer(hidden, routed_scale)
                    if mixer_kind == "kda"
                    else layer(hidden, routed_scale, mla_scale)
                )
            hidden = self.final_rms_norm_prefill(hidden)
            last_index = torch.tensor(
                [token_ids.shape[0] - 1], dtype=torch.int64, device=token_ids.device
            )
            return self.lm_head_prefill(hidden, last_index)

    return {
        "KimiKdaPrefill": KimiKdaPrefill,
        "KimiMlaPrefill": KimiMlaPrefill,
        "KimiMoePrefill": KimiMoePrefill,
        "KimiDenseMlpPrefill": KimiDenseMlpPrefill,
        "KimiKdaDenseLayerPrefill": KimiKdaDenseLayerPrefill,
        "KimiKdaMoeLayerPrefill": KimiKdaMoeLayerPrefill,
        "KimiMlaMoeLayerPrefill": KimiMlaMoeLayerPrefill,
        "KimiLinear48BA3BPrefill": KimiLinear48BA3BPrefill,
        "LAYER_KINDS": LAYER_KINDS,
        "LAYER_TYPE": LAYER_TYPE,
    }


# ---------------------------------------------------------------------------
# The published tree, importable the way the decode shell's is. Building it
# also builds the decode tree it borrows the KDA step and layer kinds from.
# ---------------------------------------------------------------------------

config = published()
_REAL = build(config)

KimiKdaPrefill = _REAL["KimiKdaPrefill"]
KimiMlaPrefill = _REAL["KimiMlaPrefill"]
KimiMoePrefill = _REAL["KimiMoePrefill"]
KimiDenseMlpPrefill = _REAL["KimiDenseMlpPrefill"]
KimiKdaDenseLayerPrefill = _REAL["KimiKdaDenseLayerPrefill"]
KimiKdaMoeLayerPrefill = _REAL["KimiKdaMoeLayerPrefill"]
KimiMlaMoeLayerPrefill = _REAL["KimiMlaMoeLayerPrefill"]
KimiLinear48BA3BPrefill = _REAL["KimiLinear48BA3BPrefill"]
LAYER_KINDS = _REAL["LAYER_KINDS"]
LAYER_TYPE = _REAL["LAYER_TYPE"]


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
        "w_gate_up": tuple(
            name
            for j in range(cfg.num_experts)
            for name in (f"experts.{j}.w1.weight", f"experts.{j}.w3.weight")
        ),
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
    "KimiDenseMlpPrefill", "KimiKdaPrefill", "KimiLinear48BA3BPrefill",
    "KimiLinearConfig", "KimiMlaPrefill", "KimiMoePrefill", "LAYER_KINDS",
    "LAYER_TYPE", "build", "config", "hf_alias", "published",
]
