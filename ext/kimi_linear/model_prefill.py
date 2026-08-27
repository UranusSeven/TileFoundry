"""Kimi-Linear-48B-A3B-Instruct's prefill shell: a prompt in, logits and caches out.

``model.py`` is the decode shell and is S=1-only: its ``S`` is the literal 1
and its only range is ``ctx_len``. This file is the prefill half of the same
model, authored beside it -- decode is not modified, imported only. Every
position-wise block (the MoE, the dense layer-0 MLP, the norms, the
embedding, the head) is the decode body re-stated over
``S = DimVar("seq_len", 1, cfg.model_max_length)`` instead of the literal.
The two mixers differ in kind, and each difference is a decision:

**MLA is one masked pass in MHA form.** The same projections as decode run
over all S positions at once; the scores are a batched matmul per head; the
causal mask is an ``arange``/``cmp_ge``/``where`` upper triangle, and
``tf.softmax`` -- which normalises in f32 and rounds back, exactly what HF's
eager ``softmax(..., dtype=torch.float32).to(query.dtype)`` does -- closes
the attention. The ground truth is ``KimiMLAAttention.forward`` at S > 1.
There is **no rope**: the checkpoint is NoPE (``mla_use_nope: true``), and
the oracle's forward applies no rotation at all -- decode's identity rotary
(cos = 1, sin = 0) is the same semantics spelled out, so this shell simply
does not take cos/sin/pos_ids. The token's own key and value are returned as
the whole prompt's ``k_all``/``v_all`` in the decode cache layout
``[1, S, heads, dim]`` -- that *is* the cache the first decode step reads.

**KDA is the decode recurrence, looped.** A chunked delta rule is not
authored in HIR: the prefill semantics of a linear-attention layer *are* the
per-step recurrence applied S times from the zero state, and the production
twin (fla's ``chunk_kda``) is validated against exactly this. ``kda_prefill``
is therefore an orchestration method that walks the prompt position by
position through the decode ``kda_attention`` and joins the outputs; the
final recurrent state and the three final convolution windows come out in
the decode shell's cache layout.

**The head reads only the last position.** ``lm_head_prefill`` returns
``[1, vocab]``: the next-token distribution. The last row is gathered with
``index_select`` on an index the orchestration passes, because the
evaluator has no visit routine for a symbolic-bound slice (``x[:, S-1:S]``
type-infers but does not evaluate).

Cache handoff: ``forward`` returns ``(logits, caches)`` where ``caches`` is
one entry per layer in the decode shell's own layout -- KDA:
``(conv_q, conv_k, conv_v, state)``, MLA: ``(k_cache, v_cache)`` -- so a
decode ``Session`` continues from a prefill by taking ``caches`` as its
starting state. Weight names, module names, and converters are identical to
the decode shell's, so one prepared resource loads both trees and
``model.hf_alias()`` serves this one unchanged.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as decode_shell  # noqa: E402 -- the untouched decode shell

from tilefoundry import DType, func, module  # noqa: E402
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: E402, F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: E402, F401, F403 -- bare op bindings for @func bodies
from tilefoundry.evaluator import to_torch_dtype  # noqa: E402
from tilefoundry.ir.types.dim import DimVar  # noqa: E402
from tilefoundry.ir.types.shard import Topology  # noqa: E402
from tilefoundry.target import CudaTarget  # noqa: E402

KimiLinearConfig = decode_shell.KimiLinearConfig
published = decode_shell.published


def build(cfg: KimiLinearConfig):
    """This model's prefill shell at *cfg*, mirroring ``model.build``.

    The prompt length is the one range this tree carries; everything else is
    the published dimension. The decode tree built here is used only for its
    KDA step module -- the recurrence the KDA prefill loops -- and for the
    layer-kind table, so the two shells can never disagree about which layer
    is which kind.
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
    _WS = _KDA["short_conv_kernel_size"] - 1  # 3 stored conv positions

    _E = cfg.num_experts
    _TOPK = cfg.num_experts_per_token
    _MI = cfg.moe_intermediate_size
    _SI = _MI * cfg.num_shared_experts

    _I = cfg.intermediate_size           # 9216, the dense layer-0 MLP

    _EPS = cfg.rms_norm_eps

    _DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
        str(cfg.dtype).removeprefix("torch.")
    ]

    decode = decode_shell.build(cfg)
    KimiKdaStep = decode["KimiKda"]
    LAYER_KINDS = decode["LAYER_KINDS"]

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
            w_gate: ConstTensor[(_E, cfg.moe_intermediate_size, _HID), _DT],
            w_up: ConstTensor[(_E, cfg.moe_intermediate_size, _HID), _DT],
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

            # Expert selection is runtime data: the indices select the expert
            # weights and a batched matmul over [tokens, top_k]. No static
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

    def _kda_prefill(self, hidden, mixer_args, cache):
        """A KDA mixer's prefill: the decode recurrence applied S times.

        Not a fused kernel and not a chunked delta rule in HIR -- the semantic
        reference a chunked implementation is checked against. The state and
        the three convolution windows thread position to position exactly as
        a decode session's would, starting from the zero state, so the
        returned cache is bit-identical in kind to what S decode steps leave.
        """
        import torch  # noqa: PLC0415

        (scale,) = mixer_args
        conv_q, conv_k, conv_v, state = cache
        steps = []
        for position in range(hidden.shape[1]):
            out, state, conv_q, conv_k, conv_v = self.mixer(
                hidden[:, position : position + 1, :],
                conv_q,
                conv_k,
                conv_v,
                state,
                scale,
            )
            steps.append(out)
        return torch.cat(steps, dim=1), (conv_q, conv_k, conv_v, state)

    def _mla_prefill(self, hidden, mixer_args, cache):
        """An MLA mixer's prefill: one masked pass over the prompt.

        *cache* is the empty container ``init_caches`` issues -- prefill
        starts from no context, so it is read for nothing. The returned
        ``(k_all, v_all)`` becomes the decode cache.
        """
        (scale,) = mixer_args
        out, k_all, v_all = self.mixer(hidden, scale)
        return out, (k_all, v_all)

    def _moe_layer_prefill(self, hidden, mixer_args, cache, routed_scale):
        """One prefill layer: mixer over the prompt + residual, then MoE + residual.

        Mirrors the decode layers' forward with the cache threading made
        explicit.
        """
        mixed, fresh = self.mixer_prefill(hidden, mixer_args, cache)
        attended = self.residual_add(hidden, mixed)
        ffn_out = self.moe(attended, routed_scale)
        return self.residual_add(attended, ffn_out), fresh

    def _dense_layer_prefill(self, hidden, mixer_args, cache, routed_scale):
        """Layer 0's prefill: mixer over the prompt + residual, then the dense MLP + residual.

        *routed_scale* is taken and ignored so the walk hands every layer the
        same arguments.
        """
        mixed, fresh = self.mixer_prefill(hidden, mixer_args, cache)
        attended = self.residual_add(hidden, mixed)
        ffn_out = self.mlp(attended)
        return self.residual_add(attended, ffn_out), fresh

    @module(entry="residual_add")
    class KimiKdaDenseLayerPrefill:
        mixer = KimiKdaStep.renamed("mixer")
        mlp = KimiDenseMlpPrefill.renamed("mlp")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        mixer_prefill = _kda_prefill
        forward = _dense_layer_prefill

    @module(entry="residual_add")
    class KimiKdaMoeLayerPrefill:
        mixer = KimiKdaStep.renamed("mixer")
        moe = KimiMoePrefill.renamed("moe")

        @func
        def residual_add(
            a: Tensor[(1, S, _HID), _DT],
            b: Tensor[(1, S, _HID), _DT],
        ) -> Tensor[(1, S, _HID), _DT]:
            return a + b

        mixer_prefill = _kda_prefill
        forward = _moe_layer_prefill

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

        mixer_prefill = _mla_prefill
        forward = _moe_layer_prefill

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

        def init_caches(self, device=None):
            """The empty per-layer state a prefill starts from.

            A KDA layer's four halves are genuinely zero at the start:
            Hugging Face left-pads the convolution windows when the context is
            shorter than them, and `initial_state=None` is the zero recurrent
            matrix. An MLA layer's container of no positions is read for
            nothing -- the mixer builds the cache -- and is carried only so
            `caches` is one entry per layer here and after.
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

        def prefill_hidden(self, hidden, layer_args, caches, routed_scale):
            """The prompt through every layer, then the final norm.

            *layer_args* is one layer's mixer arguments per layer, carrying no
            state; *caches* is the empty per-layer state `init_caches` issued.
            What comes back is the normed hidden state and each layer's fresh
            cache, in the decode shell's layout.
            """
            if len(layer_args) != len(self.modules) or len(caches) != len(self.modules):
                raise ValueError(
                    f"prefill has {len(self.modules)} layers but was given "
                    f"{len(layer_args)} argument tuples and {len(caches)} caches"
                )
            states = []
            for layer, mixer_args, cache in zip(self.modules, layer_args, caches):
                hidden, state = layer(hidden, mixer_args, cache, routed_scale)
                states.append(state)
            return self.final_rms_norm_prefill(hidden), tuple(states)

        def forward(self, token_ids, layer_args, caches, routed_scale):
            """One prefill of the whole model: token ids in, logits and caches out.

            The logits are the last position's; the caches are the decode
            shell's own layout, so the first decode step continues from them.
            """
            import torch  # noqa: PLC0415

            hidden = self.embed_prefill(token_ids)
            normed, states = self.prefill_hidden(hidden, layer_args, caches, routed_scale)
            last_index = torch.tensor(
                [token_ids.shape[0] - 1], dtype=torch.int64, device=token_ids.device
            )
            return self.lm_head_prefill(normed, last_index), states

        def prefill_inputs(self, input_ids, device=None):
            """The token and each layer's non-state arguments for one prefill.

            `input_ids` is the prompt, any sequenceable container of ints or a
            one-row tensor. The MLA scale is 192**-0.5 and the KDA scale
            128**-0.5, as the decode step's are; NoPE means no rotary tables
            exist to prepare.
            """
            import torch  # noqa: PLC0415

            device = torch.accelerator.current_accelerator() if device is None else device
            token_ids = torch.as_tensor(input_ids, dtype=torch.int64, device=device)
            token_ids = token_ids.reshape(-1)
            layer_args = []
            for mixer, _ffn in LAYER_KINDS:
                if mixer == "mla":
                    scale = torch.full(
                        (1, 1, 1, 1), _QK ** -0.5, dtype=_TORCH_DT, device=device
                    )
                else:
                    scale = torch.full(
                        (1, 1, 1), _KD ** -0.5, dtype=_TORCH_DT, device=device
                    )
                layer_args.append((scale,))
            routed_scale = torch.full(
                (1, 1), cfg.routed_scaling_factor, dtype=_TORCH_DT, device=device
            )
            return token_ids, tuple(layer_args), self.init_caches(device), routed_scale

    return {
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

_REAL = build(published())

KimiMlaPrefill = _REAL["KimiMlaPrefill"]
KimiMoePrefill = _REAL["KimiMoePrefill"]
KimiDenseMlpPrefill = _REAL["KimiDenseMlpPrefill"]
KimiKdaDenseLayerPrefill = _REAL["KimiKdaDenseLayerPrefill"]
KimiKdaMoeLayerPrefill = _REAL["KimiKdaMoeLayerPrefill"]
KimiMlaMoeLayerPrefill = _REAL["KimiMlaMoeLayerPrefill"]
KimiLinear48BA3BPrefill = _REAL["KimiLinear48BA3BPrefill"]
LAYER_KINDS = _REAL["LAYER_KINDS"]
LAYER_TYPE = _REAL["LAYER_TYPE"]
