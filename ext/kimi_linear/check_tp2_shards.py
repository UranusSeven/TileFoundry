"""The TP2 sharding math, proven on one GPU before any NCCL is involved.

Every TP2 claim this milestone makes reduces to: *the two rank-halves of a
sharded mixer/FFN, summed, equal the full-width computation.* That is a
statement about the kernels and the slicing rules, not about communication,
so it is checked here by running the halved shapes back to back on one GPU:

* **KDA**: heads 32 -> 16 per rank. q/k/v projections, the f/g low-rank up
  projections, dt_bias, a_log, w_b, gamma... sliced exactly as the runtime
  slices them; the recurrent state and conv windows follow their heads.
  Per-head math is rank-independent, so a rank's updated state must equal
  the full state restricted to its heads.
* **MLA**: heads 32 -> 16 per rank. w_q / w_kv_b column halves, w_o row
  half, per-head cache slices; w_kv_a / the latent norm replicated.
* **MoE**: expert-internal TP -- every rank keeps all 256 experts, the
  intermediate dim halves (1024 -> 512): gate/up column halves, down row
  halves, same for the shared expert; router replicated.
* **dense MLP** (layer 0): the same halving at 9216 -> 4608.

Outputs are compared in f32 (`keep_f32`): a TP2 rank all-reduces the
un-rounded partial, so the check sums the two partials and wants the full
run's accumulator back, up to atomic-add order (~1e-6).

Reproduce (container dev-yingshan-7cf9dbcf45-xtm8p):

    cd /root/develop/yingshan/TileFoundry && \
      CUDA_VISIBLE_DEVICES=0 python3 ext/kimi_linear/check_tp2_shards.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
from kernels import attn as _attn  # noqa: E402
from kernels import kda as _kda  # noqa: E402
from kernels import moe as _moe  # noqa: E402

BF16 = torch.bfloat16
HID, NH, DK = 2304, 32, 128
KP = NH * DK
QK, V, LAT, KVB = 192, 128, 512, 256
E, I, IS, TOPK = 256, 1024, 9216, 8

OK = True


def report(name, got, want, tol, exact=False):
    global OK
    d = (got.float() - want.float()).abs().max().item()
    good = d == 0.0 if exact else d <= tol
    OK &= good
    print(f"  {name:34s} max|d|={d:.3e}  tol={tol:.0e}  {'PASS' if good else 'FAIL'}")


def half(t, dim, r):
    """Rank *r*'s half of *t* along *dim* (contiguous, kernel-ready)."""
    n = t.shape[dim] // 2
    return t.narrow(dim, r * n, n).contiguous()


def check_kda(dev):
    print("KDA: heads 32 -> 2 x 16")
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, scale=0.02):
        return torch.randn(*shape, generator=g).mul(scale).to(BF16).to(dev)

    w = dict(
        gamma_in=rnd(HID, scale=1.0), w_q=rnd(1, HID, KP), w_k=rnd(1, HID, KP),
        w_v=rnd(1, HID, KP), conv_w_q=rnd(4, KP, scale=0.3),
        conv_w_k=rnd(4, KP, scale=0.3), conv_w_v=rnd(4, KP, scale=0.3),
        w_f_a=rnd(1, HID, DK), w_f_b=rnd(1, DK, KP), dt_bias=rnd(KP, scale=0.1),
        a_log=torch.rand(NH, generator=g).mul(2).to(BF16).to(dev),
        w_b=rnd(1, HID, NH), w_g_a=rnd(1, HID, DK), w_g_b=rnd(1, DK, KP),
        gamma_o=rnd(DK, scale=1.0), w_o=rnd(1, KP, HID),
    )
    hidden = rnd(1, 1, HID)
    state = rnd(1, NH, DK, DK, scale=0.05)
    convs = [rnd(1, 3, KP) for _ in range(3)]
    scale = torch.full((1, 1, 1), DK ** -0.5, dtype=BF16, device=dev)

    full = _kda.kda_step(hidden, state=state, scale=scale,
                         conv_state_q=convs[0], conv_state_k=convs[1],
                         conv_state_v=convs[2], keep_f32=True, **w)

    outs, states, cn = [], [], [[], [], []]
    for r in range(2):
        rw = dict(
            gamma_in=w["gamma_in"],
            w_q=half(w["w_q"], 2, r), w_k=half(w["w_k"], 2, r),
            w_v=half(w["w_v"], 2, r),
            conv_w_q=half(w["conv_w_q"], 1, r), conv_w_k=half(w["conv_w_k"], 1, r),
            conv_w_v=half(w["conv_w_v"], 1, r),
            w_f_a=w["w_f_a"], w_f_b=half(w["w_f_b"], 2, r),
            dt_bias=half(w["dt_bias"], 0, r), a_log=half(w["a_log"], 0, r),
            w_b=half(w["w_b"], 2, r),
            w_g_a=w["w_g_a"], w_g_b=half(w["w_g_b"], 2, r),
            gamma_o=w["gamma_o"], w_o=half(w["w_o"], 1, r),
        )
        o, s, cq, ck, cv = _kda.kda_step(
            hidden, state=half(state, 1, r), scale=scale,
            conv_state_q=half(convs[0], 2, r), conv_state_k=half(convs[1], 2, r),
            conv_state_v=half(convs[2], 2, r), keep_f32=True, **rw)
        outs.append(o)
        states.append(s)
        for i, c in enumerate((cq, ck, cv)):
            cn[i].append(c)

    report("out: half0 + half1 == full", outs[0] + outs[1], full[0], 1e-4)
    report("state: cat(halves) == full", torch.cat(states, 1), full[1], 1e-5)
    for i, name in enumerate(("conv_q", "conv_k", "conv_v")):
        report(f"{name}: cat(halves) == full", torch.cat(cn[i], 2), full[2 + i], 0,
               exact=True)


def check_mla(dev):
    print("MLA: heads 32 -> 2 x 16, ctx 24, rope exercised")
    g = torch.Generator(device="cpu").manual_seed(1)

    def rnd(*shape, scale=0.02):
        return torch.randn(*shape, generator=g).mul(scale).to(BF16).to(dev)

    C = 24
    w = dict(
        gamma_in=rnd(HID, scale=1.0), w_q=rnd(1, HID, NH * QK),
        w_kv_a=rnd(1, HID, LAT + 64), gamma_kv_a=rnd(LAT, scale=1.0),
        w_kv_b=rnd(1, LAT, NH * KVB), w_o=rnd(1, NH * V, HID),
    )
    hidden = rnd(1, 1, HID)
    k_cache, v_cache = rnd(1, C, NH, QK), rnd(1, C, NH, V)
    cos = torch.rand(_attn.bucket(C + 1), 64, generator=g).to(BF16).to(dev)
    sin = torch.rand(_attn.bucket(C + 1), 64, generator=g).to(BF16).to(dev)
    pos = torch.tensor([C], dtype=torch.int32, device=dev)
    pos0 = torch.zeros(1, dtype=torch.int32, device=dev)
    scale = torch.full((1, 1, 1, 1), QK ** -0.5, dtype=BF16, device=dev)

    # The exact-length kernel takes the position's own rope row and pos_ids=0
    # (it reads Cos[pos]); the capacity kernel takes the table and the real
    # position.
    full = _attn.mla_attention(hidden, **w, cos_cache=cos[C:C + 1], sin_cache=sin[C:C + 1],
                               pos_ids=pos0, k_cache=k_cache, v_cache=v_cache,
                               scale=scale)

    outs, kn, vn = [], [], []
    for r in range(2):
        rw = dict(
            gamma_in=w["gamma_in"], w_q=half(w["w_q"], 2, r),
            w_kv_a=w["w_kv_a"], gamma_kv_a=w["gamma_kv_a"],
            w_kv_b=half(w["w_kv_b"], 2, r), w_o=half(w["w_o"], 1, r),
        )
        o, k, v = _attn.mla_attention(
            hidden, **rw, cos_cache=cos[C:C + 1], sin_cache=sin[C:C + 1], pos_ids=pos0,
            k_cache=half(k_cache, 2, r), v_cache=half(v_cache, 2, r),
            scale=scale)
        outs.append(o)
        kn.append(k)
        vn.append(v)

    report("out: half0 + half1 == full", outs[0] + outs[1], full[0], 1e-2)
    report("k_new: cat(halves) == full", torch.cat(kn, 2), full[1], 1e-5)
    report("v_new: cat(halves) == full", torch.cat(vn, 2), full[2], 1e-5)

    # The driver's capacity kernel at 16 heads: same numbers through the
    # bucket path.
    cap = _attn.bucket(C + 1)
    kb = torch.zeros(1, cap, NH // 2, QK, dtype=BF16, device=dev)
    vb = torch.zeros(1, cap, NH // 2, V, dtype=BF16, device=dev)
    kb[:, :C] = half(k_cache, 2, 0)
    vb[:, :C] = half(v_cache, 2, 0)
    o, k, v = _attn.mla_attention_cap(
        hidden, gamma_in=w["gamma_in"], w_q=half(w["w_q"], 2, 0),
        w_kv_a=w["w_kv_a"], gamma_kv_a=w["gamma_kv_a"],
        w_kv_b=half(w["w_kv_b"], 2, 0), w_o=half(w["w_o"], 1, 0),
        cos_cache=cos[:cap].contiguous(), sin_cache=sin[:cap].contiguous(),
        pos_ids=pos, k_buf=kb, v_buf=vb, scale=scale, keep_f32=True)
    # outs[0] is the exact kernel's bf16 return; land the f32 partial the same
    # way before comparing (dbg: bf16-vs-bf16 the two are bit-identical).
    report("cap kernel at nh=16 == exact kernel", o.to(BF16), outs[0], 1e-5)
    report("cap k_new == exact k_new", k, kn[0], 0, exact=True)


def check_moe(dev):
    print("MoE: expert-internal TP, I 1024 -> 2 x 512; dense MLP 9216 -> 2 x 4608")
    g = torch.Generator(device="cpu").manual_seed(2)

    def rnd(*shape, scale=0.02):
        return torch.randn(*shape, generator=g).mul(scale).to(BF16).to(dev)

    hidden = rnd(1, 1, HID)
    gamma_post = rnd(HID, scale=1.0)
    w_router, bias = rnd(HID, E), rnd(E, scale=0.1)
    routed_scale = torch.full((1, 1), 2.446, dtype=BF16, device=dev)
    w_gate, w_up = rnd(E, I, HID), rnd(E, I, HID)
    w_down = rnd(E, HID, I)
    sh_gate, sh_up = rnd(1, HID, I), rnd(1, HID, I)
    sh_down = rnd(1, I, HID)

    full = _moe.moe_block(hidden, gamma_post, w_router, bias, routed_scale,
                          w_gate, w_up, w_down, sh_gate, sh_up, sh_down,
                          keep_f32=True)
    outs = []
    for r in range(2):
        outs.append(_moe.moe_block(
            hidden, gamma_post, w_router, bias, routed_scale,
            half(w_gate, 1, r), half(w_up, 1, r), half(w_down, 2, r),
            half(sh_gate, 2, r), half(sh_up, 2, r), half(sh_down, 1, r),
            keep_f32=True))
    report("moe: half0 + half1 == full", outs[0] + outs[1], full, 1e-4)

    dg, du = rnd(1, HID, IS), rnd(1, HID, IS)
    dd = rnd(1, IS, HID)
    full_d = _moe.dense_mlp(hidden, gamma_post, dg, du, dd, keep_f32=True)
    outs = [
        _moe.dense_mlp(hidden, gamma_post, half(dg, 2, r), half(du, 2, r),
                       half(dd, 1, r), keep_f32=True)
        for r in range(2)
    ]
    report("dense_mlp: half0 + half1 == full", outs[0] + outs[1], full_d, 1e-4)


def main() -> int:
    dev = torch.device("cuda")
    check_kda(dev)
    check_mla(dev)
    check_moe(dev)
    print()
    print("VERDICT:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
