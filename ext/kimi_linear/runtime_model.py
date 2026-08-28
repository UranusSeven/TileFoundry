"""`model.py` as a fast runtime: tilelang twins plus the decode driver.

Two layers, answering different questions (the qwen3.5 example's structure,
which this follows file for file).

**The twins** (`KimiKda` ... `KimiLinear48BA3B`) are runtime counterparts of
`model.py`'s Modules, one `@runtime_func` per authored `@func`, same names,
same order, same shapes. They keep the authored contract exactly: state
arrives as a parameter and leaves as a result, nothing is mutated in place.

**The driver** (`Session`) owns what a step needs that is not weights: the
constant scale/rope tensors, the KDA states (replaced each step, as the
authored contract's `advance_state` does), and the MLA caches -- which it
holds as fixed-capacity buffers, writing each step's k/v into slot `pos`
itself. Appending into a preallocated buffer IS the authored append (the
shell's docstring: "joining it on ... is the caller's"); it is what lets the
attention kernel be compiled once per capacity bucket
(`kernels/attn.mla_attention_cap`) instead of once per token.

Every function is independently switchable for bisection:

    TF_IMPL=torch                 everything in plain torch
    TF_IMPL=moe:torch,kda_attention:torch   those two in torch, rest tilelang

The KDA torch bodies are `kernels/torch_ref.py`'s; the rest are private
transcriptions below (the MLA/MoE lines validated their kernels against the
authored evaluator and the HF oracle directly, so no torch_ref exists for
them). The torch transcriptions compute in f32 where the authored IR rounds
to bf16 at every op -- they are a bisection tool, checked at 2e-2, not a
precision reference.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as sem  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import weights as wt  # noqa: E402

from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

# ---------------------------------------------------------------------------
# Which implementation each function uses.
# ---------------------------------------------------------------------------

_IMPL_SPEC = os.environ.get("TF_IMPL", "").strip()
_ALL_TORCH = _IMPL_SPEC == "torch"
_PER_FN = {}
if _IMPL_SPEC and not _ALL_TORCH:
    for entry in _IMPL_SPEC.split(","):
        name, _, which = entry.partition(":")
        _PER_FN[name.strip()] = (which or "torch").strip()


def _impl(name, fast, slow):
    """The callable for *name*: the tilelang one unless asked for otherwise."""
    if _ALL_TORCH or _PER_FN.get(name) == "torch":
        return slow
    return fast


from kernels import attn as _attn  # noqa: E402
from kernels import basic as _basic  # noqa: E402
from kernels import kda as _kda  # noqa: E402
from kernels import moe as _moe  # noqa: E402
from kernels import torch_ref as _tr  # noqa: E402

# Published dimensions (ext/kimi_linear/config.json), spelled out.
_HID = 2304
_NH = 32           # attention heads, MLA and KDA alike
_DK = 128          # KDA head_dim
_KP = _NH * _DK    # 4096
_NOPE = 128
_ROPE = 64
_QK = _NOPE + _ROPE
_V = 128
_LAT = 512         # kv_lora_rank
_KVB = _NOPE + _V
_TOPK = 8
_EPS = 1e-5

_BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Torch transcriptions (the slow halves of `_impl`; KDA's live in torch_ref).
# ---------------------------------------------------------------------------


def _t_rms(hidden, gamma):
    """`KimiRMSNorm`: normalise in f32, round to bf16, *then* multiply gamma."""
    h32 = hidden.float()
    hn = (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + _EPS)).to(_BF16)
    return hn * gamma


def _t_kda_gate(hidden_norm, w_f_a, w_f_b, dt_bias, a_log):
    low = torch.matmul(hidden_norm, w_f_a)
    g_raw = (torch.matmul(low, w_f_b) + dt_bias).view(1, 1, _NH, _DK)
    decay = -a_log.float().exp().view(1, 1, _NH, 1)
    return (decay * F.softplus(g_raw.float())).to(_BF16)


def _t_mla_attention(hidden, gamma_in, w_q, w_kv_a, gamma_kv_a, w_kv_b,
                     cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o):
    hn = _t_rms(hidden, gamma_in)
    q = torch.matmul(hn, w_q).view(1, 1, _NH, _QK).float()
    compressed = torch.matmul(hn, w_kv_a)
    latent, k_rot_shared = compressed[..., :_LAT], compressed[..., _LAT:]
    kvn = _t_rms(latent, gamma_kv_a)
    kv = torch.matmul(kvn, w_kv_b).view(1, 1, _NH, _KVB).float()
    k_nope, v_new = kv[..., :_NOPE], kv[..., _NOPE:]

    # The rope tables may be one row (authored NoPE call) or a capacity
    # table (the driver's); index the position's row either way.
    p = int(pos_ids.reshape(-1)[0].item())
    cos = cos_cache.reshape(-1, _ROPE)[p].float()
    sin = sin_cache.reshape(-1, _ROPE)[p].float()

    def rope(x):
        x1, x2 = x[..., : _ROPE // 2], x[..., _ROPE // 2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    q_full = torch.cat([q[..., :_NOPE], rope(q[..., _NOPE:])], dim=-1)
    q_full = q_full * scale.float().reshape(())
    k_rot = rope(k_rot_shared.view(1, 1, 1, _ROPE).float())
    k_new = torch.cat([k_nope, k_rot.expand(1, 1, _NH, _ROPE)], dim=-1)

    # The live cache is the first `p` slots: at a decode step the position
    # is the prior length (see kernels/attn.py's capacity entry point).
    q4 = q_full[0, 0]                                         # (H, QK)
    scores = torch.einsum("hq,phq->hp", q4, k_cache[0, :p].float())
    self_score = (q4 * k_new[0, 0]).sum(-1, keepdim=True)     # (H, 1)
    probs = torch.softmax(torch.cat([scores, self_score], dim=-1), dim=-1)
    attn = (
        torch.einsum("hp,phv->hv", probs[:, :-1], v_cache[0, :p].float())
        + probs[:, -1:] * v_new[0, 0]
    )
    out = torch.matmul(
        attn.to(_BF16).view(1, 1, _NH * _V), w_o.view(_NH * _V, _HID)
    )
    return out, k_new.to(_BF16), v_new.to(_BF16)


def _t_router(tokens, w_router, bias, routed_scale):
    logits = torch.matmul(tokens, w_router).float()
    scores = torch.sigmoid(logits)
    biased = scores + bias.float()
    top_biased, indices = torch.topk(biased, _TOPK, dim=-1)
    selected_bias = bias.float().reshape(-1)[indices.reshape(-1)].view_as(top_biased)
    unbiased = top_biased - selected_bias
    weights = unbiased / unbiased.sum(-1, keepdim=True) * routed_scale.float()
    return weights.to(_BF16), indices


def _t_shared_expert(tokens, sh_gate, sh_up, sh_down):
    x = tokens.reshape(1, -1, _HID)
    gate = torch.matmul(x, sh_gate)
    up = torch.matmul(x, sh_up)
    return torch.matmul(F.silu(gate) * up, sh_down).reshape(1, _HID)


def _t_moe(hidden, gamma_post, w_router, bias, routed_scale,
           w_gate, w_up, w_down, sh_gate, sh_up, sh_down):
    hn = _t_rms(hidden, gamma_post)
    tokens = hn.reshape(1, _HID)
    weights, indices = _t_router(tokens, w_router, bias, routed_scale)
    flat = indices.reshape(-1)
    tok = tokens.reshape(1, _HID, 1).expand(_TOPK, _HID, 1)
    gate = torch.matmul(w_gate[flat], tok)
    up = torch.matmul(w_up[flat], tok)
    h = F.silu(gate) * up
    down = torch.matmul(w_down[flat], h).reshape(_TOPK, _HID)
    routed = (down.float() * weights.float().reshape(_TOPK, 1)).sum(0)
    shared = _t_shared_expert(tokens, sh_gate, sh_up, sh_down)
    return (routed + shared.float()).to(_BF16).reshape(1, 1, _HID)


def _t_mlp(hidden, gamma_post, w_gate, w_up, w_down):
    hn = _t_rms(hidden, gamma_post)
    return torch.matmul(
        F.silu(torch.matmul(hn, w_gate)) * torch.matmul(hn, w_up), w_down
    )


def _t_embed(table, token_ids):
    return table[token_ids.reshape(-1)].reshape(1, 1, _HID)


def _t_lm_head(hidden, w_head):
    return torch.matmul(hidden.reshape(1, _HID), w_head)


# ---------------------------------------------------------------------------
# The twins.
# ---------------------------------------------------------------------------


@runtime_module(sem.KimiKda)
class KimiKda:
    @runtime_func
    def short_conv(self, x, conv_w, conv_state):
        return _impl("short_conv", _kda.short_conv, _tr.short_conv)(
            x, conv_w, conv_state
        )

    @runtime_func
    def l2_normalize(self, x):
        return _impl("l2_normalize", _kda.l2_normalize, _tr.l2_normalize)(x)

    @runtime_func
    def kda_gate(self, hidden_norm, w_f_a, w_f_b, dt_bias, a_log):
        # No standalone kernel: the gate is fused into `kda_step`'s launches.
        # Standalone (a per-function check) it is the torch body.
        return _t_kda_gate(hidden_norm, w_f_a, w_f_b, dt_bias, a_log)

    @runtime_func
    def kda_attention(
        self, hidden, gamma_in, w_q, w_k, w_v, conv_w_q, conv_w_k, conv_w_v,
        conv_state_q, conv_state_k, conv_state_v, w_f_a, w_f_b, dt_bias, a_log,
        w_b, w_g_a, w_g_b, gamma_o, w_o, state, scale,
    ):
        return _impl("kda_attention", _kda.kda_step, _tr.kda_step)(
            hidden, gamma_in, w_q, w_k, w_v, conv_w_q, conv_w_k, conv_w_v,
            conv_state_q, conv_state_k, conv_state_v, w_f_a, w_f_b, dt_bias,
            a_log, w_b, w_g_a, w_g_b, gamma_o, w_o, state, scale,
        )


@runtime_module(sem.KimiMla)
class KimiMla:
    @runtime_func
    def mla_attention(
        self, hidden, gamma_in, w_q, w_kv_a, gamma_kv_a, w_kv_b,
        cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
    ):
        # The fast path is the fixed-capacity entry point: `k_cache`/`v_cache`
        # are bucket-capacity prefixes of the driver's persistent buffers and
        # `pos_ids` carries the position, which is the live length.
        return _impl("mla_attention", _attn.mla_attention_cap, _t_mla_attention)(
            hidden, gamma_in, w_q, w_kv_a, gamma_kv_a, w_kv_b,
            cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
        )


@runtime_module(sem.KimiMoe)
class KimiMoe:
    @runtime_func
    def router(self, tokens, w_router, bias, routed_scale):
        def fast(tokens, w_router, bias, routed_scale):
            weights, indices = _moe.routing(tokens, w_router, bias, routed_scale)
            # The kernel's weights are f32 holding bf16 values; the authored
            # return is the bf16 cast of them, which is exact.
            return weights.to(_BF16), indices

        return _impl("router", fast, _t_router)(tokens, w_router, bias, routed_scale)

    @runtime_func
    def shared_expert(self, tokens, sh_gate, sh_up, sh_down):
        def fast(tokens, sh_gate, sh_up, sh_down):
            return _moe.shared_expert(
                tokens,
                sh_gate.view(_HID, -1), sh_up.view(_HID, -1), sh_down.view(-1, _HID),
            ).to(_BF16)

        return _impl("shared_expert", fast, _t_shared_expert)(
            tokens, sh_gate, sh_up, sh_down
        )

    @runtime_func
    def moe(
        self, hidden, gamma_post, w_router, bias, routed_scale,
        w_gate, w_up, w_down, sh_gate, sh_up, sh_down,
    ):
        return _impl("moe", _moe.moe_block, _t_moe)(
            hidden, gamma_post, w_router, bias, routed_scale,
            w_gate, w_up, w_down, sh_gate, sh_up, sh_down,
        )


@runtime_module(sem.KimiDenseMlp)
class KimiDenseMlp:
    @runtime_func
    def mlp(self, hidden, gamma_post, w_gate, w_up, w_down):
        return _impl("mlp", _moe.dense_mlp, _t_mlp)(
            hidden, gamma_post, w_gate, w_up, w_down
        )


def _residual_add(self, a, b):
    """The layer's residual. One function, three layer kinds, one body."""
    return _impl("residual_add", _basic.residual_add, lambda a, b: a + b)(a, b)


def _layer_twin(sem_layer, mixer_twin, ffn_name, ffn_twin):
    namespace = {
        "mixer": mixer_twin,
        ffn_name: ffn_twin,
        "residual_add": runtime_func(_residual_add),
    }
    cls_name = f"{sem_layer.name}Twin"
    return runtime_module(sem_layer)(type(cls_name, (), namespace))


KimiKdaDenseLayer = _layer_twin(
    sem.KimiKdaDenseLayer, KimiKda, "mlp", KimiDenseMlp
)
KimiKdaMoeLayer = _layer_twin(sem.KimiKdaMoeLayer, KimiKda, "moe", KimiMoe)
KimiMlaMoeLayer = _layer_twin(sem.KimiMlaMoeLayer, KimiMla, "moe", KimiMoe)

_LAYER_TWIN = {
    ("kda", "dense"): KimiKdaDenseLayer,
    ("kda", "moe"): KimiKdaMoeLayer,
    ("mla", "moe"): KimiMlaMoeLayer,
}


def _kind_of(layer_module):
    """A layer Module's (mixer, ffn) kind, read from its own structure.

    Structure rather than the kinds list, so a truncated tree (whose kinds
    list is not the published one) works too.
    """
    mixer = next(m for m in layer_module.modules if m.name == "mixer")
    mixer_kind = (
        "kda" if any(f.name == "kda_attention" for f in mixer.functions) else "mla"
    )
    ffn_kind = "moe" if any(m.name == "moe" for m in layer_module.modules) else "dense"
    return mixer_kind, ffn_kind


def _embed(self, table, token_ids):
    return _impl("embed", _basic.embed, _t_embed)(table, token_ids)


def _final_rms_norm(self, hidden, gamma_final):
    return _impl("final_rms_norm", _basic.rms_norm, _t_rms)(hidden, gamma_final)


def _lm_head(self, hidden, w_head):
    return _impl("lm_head", _basic.lm_head, _t_lm_head)(hidden, w_head)


def _root_twin(sem_root):
    """The root twin of *sem_root*, one child class per layer.

    Built with `type()` because `@runtime_module` reads the children out of
    `vars(cls)` and requires one class attribute per authored child, named
    exactly `layer0` .. `layer{N-1}`. The layer twin classes are shared
    across instances of a kind, as in the qwen3.5 example.
    """
    namespace = {
        child.name: _LAYER_TWIN[_kind_of(child)] for child in sem_root.modules
    }
    namespace.update(
        embed=runtime_func(_embed),
        final_rms_norm=runtime_func(_final_rms_norm),
        lm_head=runtime_func(_lm_head),
    )
    return runtime_module(sem_root)(type("KimiLinear48BA3B", (), namespace))


KimiLinear48BA3B = _root_twin(sem.KimiLinear48BA3B)


def precompile(capacity: int, nh: int = _NH) -> int:
    """Compile every attention bucket a run up to *capacity* crosses.

    The capacity attention kernel compiles once per bucket (tilelang caches
    on disk thereafter); left to chance, a long decode pays a compilation at
    every bucket crossing *inside* the timed loop. This is the same category
    as weight loading -- paid once, up front -- so the driver pays it there.
    Returns the number of buckets compiled (or found in the disk cache).
    """
    n, count = 128, 0
    while n <= capacity:
        _kern, _nsplit, slots = _attn._attn_kernel_cap(n, _attn._KSA, _attn._KSB, nh)
        _attn._out_kernel(slots, nh)
        count += 1
        n += 128 if n < 2048 else 1024
    return count


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


class Session:
    """One loaded model, driven a token at a time.

    Owns the things a step needs that are not weights: the constant scale and
    identity-rope tensors, the per-layer KDA states (replaced each step, per
    the authored contract), and the per-layer MLA caches as fixed-capacity
    buffers with a device-side fill level (the position), which is what the
    bucket-compiled attention kernel reads.
    """

    #: Heads in one rank's MLA caches -- all of them at world size 1.
    #: `runtime_tp2.SessionTP2` overrides.
    _cache_heads = _NH

    def _twin_cls(self):
        """The twin class to instantiate (runtime_tp2 substitutes its own)."""
        if self.cfg is sem.config:
            return KimiLinear48BA3B
        return _root_twin(self.sem)

    def _post_load(self) -> None:
        """After the weights bind: a TP session slices them here."""

    def _shard_kda_entry(self, entry):
        """A KDA cache entry as this rank holds it (whole, at world size 1)."""
        return entry

    def __init__(
        self,
        cfg=None,
        *,
        ckpt=wt.CKPT,
        device="cuda",
        capacity: int = 1024,
        verbose=False,
        prepare_moe_prefill=False,
    ) -> None:
        self.cfg = sem.config if cfg is None else cfg
        self.device = device
        if cfg is None:
            self.sem = sem.KimiLinear48BA3B
            self.kinds = sem.LAYER_KINDS
        else:
            built = sem.build(cfg)
            self.sem = built["KimiLinear48BA3B"]
            self.kinds = built["LAYER_KINDS"]
        self.twin = self._twin_cls()()

        t0 = time.perf_counter()
        resource, tally = wt.decoder_resource(
            self.sem, ckpt=ckpt, cfg=cfg, device=device, verbose=verbose
        )
        self.twin.load(resource)
        self._post_load()
        try:
            from .kernels.moe_prefill_vllm import prepare_fused_weights
        except ImportError:
            from kernels.moe_prefill_vllm import prepare_fused_weights
        if prepare_moe_prefill:
            prepare_fused_weights(self.twin, self.kinds)
        self.load_seconds = time.perf_counter() - t0
        self.loaded_bytes = tally["bytes"]

        # The constant per-step tensors. The scales are the authored
        # `prepare_inputs_for_generation`'s; the rope tables are the identity
        # (NoPE), one row per position because the capacity kernel indexes
        # them by position.
        self.scale_kda = torch.full(
            (1, 1, 1), _DK ** -0.5, dtype=_BF16, device=device
        )
        self.scale_mla = torch.full(
            (1, 1, 1, 1), _QK ** -0.5, dtype=_BF16, device=device
        )
        self.routed_scale = torch.full(
            (1, 1), self.cfg.routed_scaling_factor, dtype=_BF16, device=device
        )
        self.capacity = _attn.bucket(capacity + 2)
        self.cos, self.sin = wt.nope_caches(self.capacity, device=device)

        # Fixed-capacity MLA caches. Slots past the fill level are never
        # read (the kernel masks them); zeroed anyway so a bug reads a
        # number, not garbage.
        self.kbuf = {}
        self.vbuf = {}
        for index, (mixer, _ffn) in enumerate(self.kinds):
            if mixer == "mla":
                self.kbuf[index] = torch.zeros(
                    1, self.capacity, self._cache_heads, _QK, dtype=_BF16, device=device
                )
                self.vbuf[index] = torch.zeros(
                    1, self.capacity, self._cache_heads, _V, dtype=_BF16, device=device
                )
        self.reset()

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Back to position zero: zero KDA states, empty MLA caches."""
        self.kda = {}
        for i, entry in enumerate(self.twin.init_caches(device=self.device)):
            if self.kinds[i][0] == "kda":
                self.kda[i] = self._shard_kda_entry(entry)
        for buf in (*self.kbuf.values(), *self.vbuf.values()):
            buf.zero_()
        self.pos = 0

    def _layer_args(self, pos: int):
        """One tuple per layer of what its mixer takes besides hidden and state.

        KDA takes its scale; MLA takes the identity rope tables (bucket
        prefix), the position -- which doubles as the cache fill level -- and
        its scale. Which slot the state is spliced into is the authored
        side's business (`_with_cache`), not this function's.
        """
        pos_ids = torch.tensor([pos], dtype=torch.int32, device=self.device)
        cap = _attn.bucket(pos + 1)
        return tuple(
            (self.scale_kda,)
            if mixer == "kda"
            else (self.cos[:cap], self.sin[:cap], pos_ids, self.scale_mla)
            for mixer, _ffn in self.kinds
        )

    def _caches(self, pos: int):
        """Each layer's cache as the authored contract hands it to the mixer.

        KDA: the four state tensors. MLA: bucket-capacity prefix views of the
        persistent buffers -- prefix views of a contiguous buffer are
        contiguous.
        """
        cap = _attn.bucket(pos + 1)
        return tuple(
            self.kda[i] if mixer == "kda"
            else (self.kbuf[i][:, :cap], self.vbuf[i][:, :cap])
            for i, (mixer, _ffn) in enumerate(self.kinds)
        )

    # -- the step ----------------------------------------------------------

    def step(self, token_id: int):
        """One decode step through the authored orchestration. Returns logits.

        `forward` here is `model.py`'s own `KimiLinear48BA3B.forward`, reused
        verbatim. Advancing the caches is the caller's step: KDA replaces
        (the authored `advance_state`), MLA writes this token's k/v into slot
        `pos` of the persistent buffers -- the append, preallocated.
        """
        ids = torch.tensor([token_id], dtype=torch.int64, device=self.device)
        logits, fresh = self.twin.forward(
            ids, self._layer_args(self.pos), self._caches(self.pos),
            self.routed_scale,
        )
        for i, (mixer, _ffn) in enumerate(self.kinds):
            if mixer == "kda":
                self.kda[i] = sem.advance_state(mixer, None, fresh[i])
            else:
                k_new, v_new = fresh[i]
                self.kbuf[i][:, self.pos : self.pos + 1].copy_(k_new)
                self.vbuf[i][:, self.pos : self.pos + 1].copy_(v_new)
        self.pos += 1
        return logits

    # -- the authored path over the same weights, for `--check` ------------

    def authored(self):
        """A `LoadedModule` view of the authored tree over the twin's weights.

        Building it re-reads nothing: the constants are the twin's own bound
        tensors, so the authored evaluator and the tilelang twins run the
        same numbers, and a disagreement is the implementation's, not the
        loading's.
        """
        from tilefoundry.ir.core.module import LoadedModule  # noqa: PLC0415

        def view(node):
            return LoadedModule(
                module=node.module,
                constants=node._bound,
                modules=tuple(view(child) for child in node.modules),
            )

        return view(self.twin)

    def authored_step(self, loaded, input_ids, step: int, caches):
        """One decode step through the authored evaluator, exact-length caches."""
        token_ids, layer_args, caches, routed_scale = (
            loaded.prepare_inputs_for_generation(
                input_ids, step, caches, device=self.device
            )
        )
        logits, fresh = loaded.forward(token_ids, layer_args, caches, routed_scale)
        return logits, loaded.append_cache(caches, fresh)


__all__ = [
    "KimiDenseMlp",
    "KimiKda",
    "KimiKdaDenseLayer",
    "KimiKdaMoeLayer",
    "KimiLinear48BA3B",
    "KimiMla",
    "KimiMlaMoeLayer",
    "KimiMoe",
    "Session",
    "precompile",
]
