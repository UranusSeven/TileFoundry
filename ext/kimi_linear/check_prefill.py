"""M3-A validation of the prefill shell (model_prefill.py).

Three independent checks, all on random weights at a fixed seed:

A. Prefill-vs-decode consistency at the real dimensions, S=8: the MLA and
   MoE prefill funcs against the decode funcs stepped eight times (MLA cache
   appended per step), and the dense MLP likewise. The prefill KDA path is
   the decode recurrence looped, so its per-step semantics need no new
   ground truth; its *wiring* is checked at the shell level in B.

B. A reduced four-layer config (kda+dense, kda+moe, mla+moe, kda+moe) run
   whole: prefill over eight tokens versus eight authored decode steps from
   the zero state -- last-position logits and every layer's final cache --
   then one decode step *continued from the prefill caches* against the
   ninth decode step, which is the handoff a serving loop depends on.

C. The MLA prefill against the checkpoint's own `KimiMLAAttention` (kimi_ref,
   eager attention, causal mask) at the real dimensions with random weights:
   out, and the k/v the oracle computes inside, all at bf16 tolerance.

Run: CUDA_VISIBLE_DEVICES=2 python3 ext/kimi_linear/check_prefill.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as decode_shell  # noqa: E402
import model_prefill as prefill_shell  # noqa: E402
import torch  # noqa: E402

from tilefoundry.evaluator import evaluate, to_torch_dtype  # noqa: E402
from tilefoundry.runtime.resource import DictResource  # noqa: E402

DEVICE = "cuda"
S = 8
TOL = 2e-2

FAILURES = []


def check(name, actual, expected, tol=TOL):
    """Compare two tensors, recording the outcome by name."""
    if actual.shape != expected.shape:
        FAILURES.append(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        print(f"FAIL {name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        return
    diff = (actual.float() - expected.float()).abs().max().item()
    ok = diff <= tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: max|d| = {diff:.3e}")
    if not ok:
        FAILURES.append(f"{name}: max|d| {diff:.3e} > {tol}")


def draw_weights(module, seed, sigma=0.05):
    """Every weight a module declares, drawn once at its declared type.

    Norm gammas are drawn around one (a gamma around zero crushes the signal
    the comparison looks at); everything else is small gaussian, the scale
    reference.py draws KDA's weights at.
    """
    generator = torch.Generator().manual_seed(seed)
    drawn = {}
    for name, decl in module.weights.items():
        shape = tuple(int(d) for d in decl.shape)
        dtype = to_torch_dtype(decl.dtype)
        if name.startswith("gamma"):
            value = 1.0 + sigma * torch.randn(*shape, generator=generator)
        else:
            value = sigma * torch.randn(*shape, generator=generator)
        drawn[name] = value.to(dtype).to(DEVICE)
    return drawn


def mla_f64(hidden, weights, scale):
    """The MLA prefill formula in f64 from the same weights.

    No bf16 rounding anywhere -- the truth both bf16 paths approximate.
    """
    w = lambda n: weights[n].double()  # noqa: E731
    hn32 = hidden.double()
    hn = hn32 * torch.rsqrt(hn32.pow(2).mean(-1, keepdim=True) + 1e-5) * w("gamma_in")
    q = (hn @ w("w_q")).reshape(1, S, 32, 192)
    compressed = hn @ w("w_kv_a")
    latent = compressed[..., :512]
    k_rot = compressed[..., 512:]
    kv_n = latent * torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + 1e-5)
    kv_n = kv_n * w("gamma_kv_a")
    kv = (kv_n @ w("w_kv_b")).reshape(1, S, 32, 256)
    # k = [nope 128 | shared rope-width 64 broadcast over heads]
    k_all = torch.cat([kv[..., :128], k_rot[:, :, None, :].expand(1, S, 32, 64)], dim=-1)
    v_all = kv[..., 128:]
    q_h = q.permute(0, 2, 1, 3)
    k_h = k_all.permute(0, 2, 1, 3)
    v_h = v_all.permute(0, 2, 1, 3)
    scores = (q_h @ k_h.transpose(-1, -2)) * scale.double()
    mask = torch.triu(torch.ones(S, S, device=hidden.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = probs @ v_h
    return attn.permute(0, 2, 1, 3).reshape(1, S, 4096) @ w("w_o")


# ── A: prefill vs decode at the real dimensions ──────────────────────────────


def check_mla_consistency():
    """mla_prefill over S positions vs the decode step.

    The decode side appends its cache position by position; both sides share
    one draw of the mixer's weights.
    """
    print("── A1: MLA prefill vs 8 decode steps (real dims) ──")
    weights = draw_weights(prefill_shell.KimiMlaPrefill, seed=11)
    w = lambda n: weights[n]  # noqa: E731
    torch.manual_seed(12)
    hidden = (0.5 * torch.randn(1, S, 2304)).bfloat16().to(DEVICE)
    scale = torch.full((1, 1, 1, 1), 192**-0.5, dtype=torch.bfloat16, device=DEVICE)

    out_p, k_all, v_all = evaluate(
        prefill_shell.KimiMlaPrefill.lookup("mla_prefill"),
        hidden, w("gamma_in"), w("w_q"), w("w_kv_a"), w("gamma_kv_a"), w("w_kv_b"),
        scale, w("w_o"),
        device=DEVICE,
    )

    mla_step = decode_shell.KimiMla.lookup("mla_attention")
    cos = torch.ones(1, 64, dtype=torch.bfloat16, device=DEVICE)
    sin = torch.zeros(1, 64, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    k_cache = torch.zeros(1, 0, 32, 192, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.zeros(1, 0, 32, 128, dtype=torch.bfloat16, device=DEVICE)
    steps = []
    for i in range(S):
        out_i, k_new, v_new = evaluate(
            mla_step,
            hidden[:, i : i + 1, :], w("gamma_in"), w("w_q"), w("w_kv_a"),
            w("gamma_kv_a"), w("w_kv_b"), cos, sin, pos, k_cache, v_cache,
            scale, w("w_o"),
            device=DEVICE,
        )
        k_cache = torch.cat([k_cache, k_new], dim=1)
        v_cache = torch.cat([v_cache, v_new], dim=1)
        steps.append(out_i)
    out_d = torch.cat(steps, dim=1)
    # The two sides evaluate scores differently at bf16: prefill's batched
    # matmul accumulates f32, while decode's GEMV-form reduce rounds each
    # q*k product to bf16 before summing. At sigma=0.05 weights that gap is
    # ~1e-1, wider than the 2e-2 gate -- so the out comparison is made
    # against an f64 recomputation of the same formula, and the gate is that
    # prefill sits no farther from truth than decode does.
    truth = mla_f64(hidden, weights, scale)
    d_p = (out_p.float() - truth.float()).abs().max().item()
    d_d = (out_d.float() - truth.float()).abs().max().item()
    gate = max(TOL, 1.25 * d_d)
    ok = d_p <= gate
    print(
        f"{'PASS' if ok else 'FAIL'} mla out vs f64 truth: "
        f"prefill {d_p:.3e}, decode {d_d:.3e}, gate {gate:.3e}"
    )
    if not ok:
        FAILURES.append(f"mla out: prefill {d_p:.3e} beyond gate {gate:.3e}")
    check("mla k_all == appended cache", k_all, k_cache)
    check("mla v_all == appended cache", v_all, v_cache)


def check_ffn_consistency():
    """The position-wise blocks: prefill over S vs decode per position.

    Both sides share one weight draw per block.
    """
    print("── A2: MoE + dense MLP prefill vs per-position decode (real dims) ──")
    torch.manual_seed(13)
    hidden = (0.5 * torch.randn(1, S, 2304)).bfloat16().to(DEVICE)
    routed_scale = torch.full((1, 1), 2.446, dtype=torch.bfloat16, device=DEVICE)

    weights = draw_weights(prefill_shell.KimiMoePrefill, seed=14)
    w = lambda n: weights[n]  # noqa: E731
    args = lambda h: (  # noqa: E731
        h, w("gamma_post"), w("w_router"), w("bias"), routed_scale,
        w("w_gate"), w("w_up"), w("w_down"), w("sh_gate"), w("sh_up"), w("sh_down"),
    )
    out_p = evaluate(
        prefill_shell.KimiMoePrefill.lookup("moe_prefill"), *args(hidden), device=DEVICE
    )
    moe_step = decode_shell.KimiMoe.lookup("moe")
    out_d = torch.cat(
        [evaluate(moe_step, *args(hidden[:, i : i + 1, :]), device=DEVICE) for i in range(S)],
        dim=1,
    )
    check("moe out [1,8,2304]", out_p, out_d)

    weights = draw_weights(prefill_shell.KimiDenseMlpPrefill, seed=15)
    w = lambda n: weights[n]  # noqa: E731
    args = lambda h: (h, w("gamma_post"), w("w_gate"), w("w_up"), w("w_down"))  # noqa: E731
    out_p = evaluate(
        prefill_shell.KimiDenseMlpPrefill.lookup("mlp_prefill"), *args(hidden), device=DEVICE
    )
    mlp_step = decode_shell.KimiDenseMlp.lookup("mlp")
    out_d = torch.cat(
        [evaluate(mlp_step, *args(hidden[:, i : i + 1, :]), device=DEVICE) for i in range(S)],
        dim=1,
    )
    check("dense mlp out [1,8,2304]", out_p, out_d)


# ── B: whole reduced shells, prefill vs decode, then the handoff ─────────────

_REDUCED_OVERRIDES = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "kv_lora_rank": 64,
    "vocab_size": 1000,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_intermediate_size": 64,
    "model_max_length": 128,
    "linear_attn_config": {
        "kda_layers": [1, 2, 4],
        "full_attn_layers": [3],
        "num_heads": 4,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
    },
}


def reduced_config():
    raw = json.loads((open(os.path.join(_HERE, "config.json")).read()))
    raw.update(_REDUCED_OVERRIDES)
    return decode_shell.KimiLinearConfig(**raw)


def draw_tree_weights(root, seed):
    """A flat {dotted path: tensor} resource of every declared weight."""
    flat = {}

    def walk(module, prefix):
        for name, value in draw_weights(module, seed=seed).items():
            flat[prefix + name] = value
        for child in module.modules:
            walk(child, prefix + child.name + ".")

    walk(root, "")
    return flat


def check_end_to_end():
    print("── B: reduced 4-layer shells, S=8, then a decode step from the handoff ──")
    cfg = reduced_config()
    kinds = decode_shell.build(cfg)["LAYER_KINDS"]
    assert kinds == (("kda", "dense"), ("kda", "moe"), ("mla", "moe"), ("kda", "moe")), kinds
    dec_root = decode_shell.build(cfg)["KimiLinear48BA3B"]
    pre_root = prefill_shell.build(cfg)["KimiLinear48BA3BPrefill"]

    resource = DictResource(draw_tree_weights(dec_root, seed=16))
    dec = dec_root.load(resource)
    pre = pre_root.load(resource)

    ids = torch.tensor([3, 17, 42, 99, 250, 511, 640, 818])

    # decode: eight steps from the zero state
    caches_d = dec.init_caches(DEVICE)
    logits_d = None
    for step in range(S):
        token_ids, layer_args, caches_d, routed = dec.prepare_inputs_for_generation(
            ids, step, caches_d, DEVICE
        )
        logits_d, fresh = dec(token_ids, layer_args, caches_d, routed)
        caches_d = dec.append_cache(caches_d, fresh)

    # prefill: one call
    token_ids, layer_args, caches0, routed = pre.prefill_inputs(ids, DEVICE)
    logits_p, caches_p = pre(token_ids, layer_args, caches0, routed)

    check("B last-position logits", logits_p, logits_d)
    for index, (kind, cache_d, cache_p) in enumerate(zip(kinds, caches_d, caches_p)):
        for half, (d, p) in enumerate(zip(cache_d, cache_p)):
            check(f"B layer{index} ({kind[0]}) cache[{half}]", p, d)

    # the handoff: decode one more token from the prefill caches, and compare
    # against the ninth step of the decode-only run
    ids9 = torch.cat([ids, torch.tensor([777])])
    for step in range(S, S + 1):
        token_ids, layer_args, caches_d, routed = dec.prepare_inputs_for_generation(
            ids9, step, caches_d, DEVICE
        )
        logits_d, fresh = dec(token_ids, layer_args, caches_d, routed)
        caches_d = dec.append_cache(caches_d, fresh)

    token_ids, layer_args, _caches_unused, routed = dec.prepare_inputs_for_generation(
        ids9, S, caches_p, DEVICE
    )
    logits_h, fresh_h = dec(token_ids, layer_args, caches_p, routed)
    check("B handoff decode-step logits", logits_h, logits_d)


# ── C: MLA prefill against the checkpoint's own attention ────────────────────


def check_mla_oracle():
    """mla_prefill against kimi_ref's KimiMLAAttention at S=8, causal.

    Eager attention, real dims, one shared random weight draw. The oracle's
    key/value states are captured out of its eager attention call.
    """
    print("── C: MLA prefill vs kimi_ref KimiMLAAttention (real dims, causal) ──")
    oracle_parent = "/root/develop/yingshan/oracle"
    if oracle_parent not in sys.path:
        sys.path.insert(0, oracle_parent)
    from kimi_ref.configuration_kimi import KimiLinearConfig as HFConfig  # noqa: PLC0415
    from kimi_ref.modeling_kimi import KimiMLAAttention  # noqa: PLC0415

    cfg = HFConfig(**json.loads(open(os.path.join(_HERE, "config.json")).read()))
    cfg._attn_implementation = "eager"
    torch.manual_seed(17)
    oracle = KimiMLAAttention(cfg, layer_idx=3).to(DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        for param in oracle.parameters():
            param.data = (0.05 * torch.randn_like(param, dtype=torch.float32)).bfloat16()

    torch.manual_seed(18)
    hidden = (0.5 * torch.randn(1, S, cfg.hidden_size)).bfloat16().to(DEVICE)
    gamma = (1.0 + 0.05 * torch.randn(cfg.hidden_size)).bfloat16().to(DEVICE)

    # The authored func fuses the input RMSNorm; hand the oracle the normed
    # hidden, computed with KimiRMSNorm's exact rounding chain.
    hn32 = hidden.float()
    hn = (hn32 * torch.rsqrt(hn32.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps))
    hn = hn.bfloat16() * gamma

    mask = torch.triu(
        torch.full((S, S), torch.finfo(torch.bfloat16).min, device=DEVICE), diagonal=1
    ).reshape(1, 1, S, S).bfloat16()

    captured = {}
    import kimi_ref.modeling_kimi as modeling_kimi  # noqa: PLC0415

    real_eager = modeling_kimi.eager_attention_forward

    def capturing(module, query, key, value, attention_mask, scaling, **kwargs):
        captured["k"] = key.detach()
        captured["v"] = value.detach()
        return real_eager(module, query, key, value, attention_mask, scaling, **kwargs)

    modeling_kimi.eager_attention_forward = capturing
    try:
        with torch.no_grad():
            out_o = oracle(hn, attention_mask=mask)
    finally:
        modeling_kimi.eager_attention_forward = real_eager

    # The same weights in the authored layout (the converters' transposes).
    def t(linear):  # (out, in) -> (1, in, out)
        return linear.weight.t().reshape(1, *reversed(linear.weight.shape)).contiguous()

    scale = torch.full(
        (1, 1, 1, 1), (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5,
        dtype=torch.bfloat16, device=DEVICE,
    )
    out_p, k_all, v_all = evaluate(
        prefill_shell.KimiMlaPrefill.lookup("mla_prefill"),
        hidden, gamma, t(oracle.q_proj), t(oracle.kv_a_proj_with_mqa),
        oracle.kv_a_layernorm.weight.contiguous(), t(oracle.kv_b_proj),
        scale, t(oracle.o_proj),
        device=DEVICE,
    )

    check("C mla_prefill out vs oracle", out_p, out_o)
    # The oracle keeps keys heads-first [1, H, S, 192]; the authored cache is
    # [1, S, H, 192].
    check("C k_all vs oracle key_states", k_all, captured["k"].transpose(1, 2))
    check("C v_all vs oracle value_states", v_all, captured["v"].transpose(1, 2))


def main():
    check_mla_consistency()
    check_ffn_consistency()
    check_end_to_end()
    check_mla_oracle()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
