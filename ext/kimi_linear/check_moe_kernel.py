"""MoE + basic tilelang kernels vs the HF oracle and the authored IR.

Legs, all on cuda at the published dims (hidden 2304, 256 experts, top-8,
shared 1, vocab 163840), bf16:

  1. basic ops (`kernels/basic.py`) vs the authored semantics in plain torch:
     rms_norm / residual_add / embed must be BIT-EXACT (they reproduce the
     authored rounding placement exactly); lm_head at one-ulp gates.
  2. unit legs for `kernels/moe.py` at E=64, random weights, vs plain torch
     (routing weights/indices, routed, shared, fused) in bf16 and f32.
  3. router selection vs torch.topk over sigmoid+bias, 220 random draws --
     the index is the load-bearing output, a moved logit must not move it.
  4. the published 256-expert block, 4 activation draws, three ways:
     A = HF oracle (reference.build_hf_moe / moe_oracle, DeepseekV3MoE at
         Kimi's numbers, nonzero correction bias),
     B = authored IR (tests model `KimiLinear48BA3B.moe.lookup("moe")` via
         tilefoundry.evaluator.evaluate),
     K = kernels (`moe.moe_block`).
  5. latency: wall time per call inside a CUDA graph, 100 calls per replay,
     weight copies rotated so L2 is cold, the way a decode step actually
     reads them.

Reproduce (container dev-yingshan-7cf9dbcf45-xtm8p):

  cd /root/develop/yingshan/TileFoundry && \
    CUDA_VISIBLE_DEVICES=4 python3 ext/kimi_linear/check_moe_kernel.py
"""
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ext"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from kimi_linear.kernels import basic as kb  # noqa: E402
from kimi_linear.kernels import moe as km  # noqa: E402
from tests.models.kimi_linear_48b_a3b import reference  # noqa: E402
from tests.models.kimi_linear_48b_a3b.model import KimiLinear48BA3B  # noqa: E402
from tilefoundry.evaluator import evaluate  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16
H = 2304
EPS = 1e-5
TOPK = 8
GATE = 2e-2  #: max-abs gate against the bf16 oracle, as the smoke uses


def check(name, got, want, gate, *, exact=False):
    got, want = got.float(), want.float()
    d = (got - want).abs().max().item()
    ok = (d == 0.0) if exact else (d <= gate)
    print(f"  {name:44s} max|d|={d:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def ulp_gate(want: torch.Tensor, n: float = 1.5) -> float:
    """`n` bf16 ulps at the scale of the largest entry of *want*.

    For outputs whose magnitude is ~60 (dense MLP) or ~5 (logits), the one
    bf16 rounding of the reference itself is larger than any plausible kernel
    error, so an absolute gate misfires and an exact gate is unmeetable: the
    kernel and a torch f32 reference sum in different orders, and a value
    sitting within an f32 epsilon of a bf16 tie flips. What a real bug looks
    like is O(output-magnitude) error, which a 1.5-ulp gate still refuses.
    """
    m = want.float().abs().max().item()
    return n * 2.0 ** (math.floor(math.log2(m)) - 7)


def torch_kimi_rms_norm(x, gamma):
    """The authored rounding: f32 normalise, round to bf16, then * gamma."""
    x32 = x.float()
    hn = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(DTYPE)
    return hn * gamma


def torch_router(tokens, w_router, bias, scale, k=TOPK):
    """sigmoid + selection-only bias + renorm + routed scale, f32."""
    scores = torch.sigmoid(tokens.float() @ w_router.float())
    biased = scores + bias.float()
    top_biased, idx = torch.topk(biased, k, dim=-1)
    unbiased = top_biased - bias.float()[idx]
    w = unbiased / unbiased.sum(-1, keepdim=True) * scale.float()
    return w.to(DTYPE), idx


def torch_routed(tokens, w, idx, wg, wu, wd):
    acc = torch.zeros(1, H, device=DEVICE)
    tok = tokens.float().view(-1)
    for s in range(idx.shape[1]):
        e = int(idx[0, s])
        g = wg[e].float() @ tok
        u = wu[e].float() @ tok
        acc += (wd[e].float() @ (F.silu(g) * u)) * w[0, s].float()
    return acc


def torch_shared(tokens, sg, su, sd):
    h = F.silu(tokens.float() @ sg.float()) * (tokens.float() @ su.float())
    return h @ sd.float()


def graph_bench(call, reps=100, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            call()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters / reps * 1e6


def main():
    ok = True
    torch.manual_seed(0)

    # ── 1. basic ops ──────────────────────────────────────────────────────
    print("=" * 74)
    print("(1) basic ops vs the authored semantics in plain torch")
    x = (torch.randn(1, 1, H, device=DEVICE) * 0.1).to(DTYPE)
    gamma = (torch.randn(H, device=DEVICE) * 0.1 + 1.0).to(DTYPE)
    ok &= check("rms_norm", kb.rms_norm(x, gamma),
                torch_kimi_rms_norm(x, gamma), 0, exact=True)
    y = (torch.randn(1, 1, H, device=DEVICE) * 0.1).to(DTYPE)
    ok &= check("residual_add", kb.residual_add(x, y),
                (x.float() + y.float()).to(DTYPE), 0, exact=True)
    table = (torch.randn(reference.CONFIG.vocab_size, H, device=DEVICE) * 0.02
             ).to(DTYPE)
    ids = torch.tensor([12345], device=DEVICE, dtype=torch.int64)
    ok &= check("embed", kb.embed(table, ids),
                table[12345].view(1, 1, H), 0, exact=True)
    w_head = (torch.randn(H, reference.CONFIG.vocab_size, device=DEVICE) * 0.02
              ).to(DTYPE)
    logits_ref = (torch_kimi_rms_norm(x, gamma).float().view(1, H)
                  @ w_head.float())
    logits = kb.lm_head(torch_kimi_rms_norm(x, gamma), w_head)
    ok &= check("lm_head", logits, logits_ref.to(DTYPE),
                ulp_gate(logits_ref))
    del table, w_head, logits, logits_ref
    torch.cuda.empty_cache()

    # ── 2. unit legs at E=64, both weight dtypes ──────────────────────────
    print("=" * 74)
    print("(2) moe entry points vs plain torch, E=64, both weight dtypes")
    E, I, IS = 64, 1024, 1024
    tok = torch_kimi_rms_norm(x, gamma).view(1, H)
    for dt in (torch.bfloat16, torch.float32):
        def mk(*shape, s=0.05):
            return (torch.randn(*shape, device=DEVICE) * s).to(dt)

        wr, bias = mk(H, E), mk(E)
        scale = torch.full((1, 1), 2.446, device=DEVICE, dtype=dt)
        wg, wu, wd = mk(E, I, H), mk(E, I, H), mk(E, H, I)
        sg, su, sd = mk(H, IS), mk(H, IS), mk(IS, H)
        wgt, idx = km.routing(tok, wr, bias, scale)
        rw, ri = torch_router(tok, wr, bias, scale)
        same = bool((idx == ri).all())
        tr = torch_routed(tok, wgt.to(DTYPE), idx, wg, wu, wd)
        ts = torch_shared(tok, sg, su, sd)
        print(f"  --- weights {str(dt).split('.')[1]:8s}  indices "
              f"{'EXACT' if same else 'MISMATCH'}")
        ok &= same
        ok &= check(f"routing weights [{dt}]", wgt, rw, 1e-2)
        ok &= check(f"routed_experts [{dt}]",
                    km.routed_experts(tok, wgt, idx, wg, wu, wd), tr, 1e-2)
        ok &= check(f"shared_expert [{dt}]",
                    km.shared_expert(tok, sg, su, sd), ts, 1e-2)
        got = km.experts(tok, wgt, idx, wg, wu, wd, sg, su, sd)
        ok &= check(f"experts (fused) [{dt}]", got, (tr + ts), 1e-2)
        del wr, wg, wu, wd, sg, su, sd
        torch.cuda.empty_cache()

    # dense MLP (layer 0 shape, IS=9216)
    I0 = 9216
    dg = (torch.randn(1, H, I0, device=DEVICE) * 0.05).to(DTYPE)
    du = (torch.randn(1, H, I0, device=DEVICE) * 0.05).to(DTYPE)
    dd = (torch.randn(1, I0, H, device=DEVICE) * 0.05).to(DTYPE)
    h0 = F.silu(tok.float() @ dg.view(H, I0).float()) * \
        (tok.float() @ du.view(H, I0).float())
    want0 = h0 @ dd.view(I0, H).float()  # f32: the kernel is finer than bf16
    got0 = km.shared_expert(kb.rms_norm(x, gamma), dg.view(H, I0),
                            du.view(H, I0), dd.view(I0, H))
    ok &= check("dense_mlp (layer 0, f32 path)", got0, want0, 1e-2)
    ok &= check("dense_mlp (wrapper, bf16 out)",
                km.dense_mlp(x, gamma, dg, du, dd), want0, ulp_gate(want0))
    del dg, du, dd, h0, want0
    torch.cuda.empty_cache()

    # ── 3. router selection vs torch.topk, 220 draws ─────────────────────
    print("=" * 74)
    print("(3) top-8-of-256 selection on sigmoid+bias vs torch.topk")
    E = 256
    bad = 0
    for trial in range(220):
        t = (torch.randn(1, H, device=DEVICE) * 0.5).to(DTYPE)
        w = (torch.randn(H, E, device=DEVICE) * 0.05).to(DTYPE)
        b = torch.randn(E, device=DEVICE) * 0.5
        sc = torch.full((1, 1), 2.446, device=DEVICE, dtype=DTYPE)
        _, i = km.routing(t.view(1, H), w, b, sc)
        _, ri = torch.topk(
            torch.sigmoid(t.float() @ w.float()) + b, TOPK, dim=-1)
        if not bool((i == ri).all()):
            bad += 1
            if bad <= 3:
                bb = (torch.sigmoid(t.float() @ w.float()) + b)[0]
                sv = bb.sort(descending=True).values
                print(f"  draw {trial}: mine {i[0].tolist()} vs torch "
                      f"{ri[0].tolist()}  gap(8,9)={float(sv[7] - sv[8]):.2e}")
    print(f"  220 random draws: {220 - bad}/220 exactly equal")
    ok &= bad == 0

    # ── 4. the published block: oracle A, authored B, kernels K ───────────
    print("=" * 74)
    print("(4) 256 experts, 4 draws: K vs authored IR (B) vs HF oracle (A)")
    hf_moe = reference.build_hf_moe()
    fn = KimiLinear48BA3B.moe.lookup("moe")
    for act_seed in reference.MOE_DRAWS:
        step = reference.moe_inputs(act_seed=act_seed, hf_moe=hf_moe)
        want_a = reference.moe_oracle(step).view(1, 1, H)
        out_b = evaluate(fn, *step.args, device=DEVICE)
        out_k = km.moe_block(*step.args)
        d_bk = (out_k.float() - out_b.float()).abs().max().item()
        d_ka = (out_k.float() - want_a.float()).abs().max().item()
        d_ba = (out_b.float() - want_a.float()).abs().max().item()
        ok &= d_ka <= GATE and d_ba <= GATE
        print(f"  draw {act_seed}:  K-B {d_bk:.3e}   K-A {d_ka:.3e}   "
              f"B-A {d_ba:.3e}   [{'PASS' if d_ka <= GATE else 'FAIL'}]")
    del hf_moe
    torch.cuda.empty_cache()

    # ── 5. latency, cold L2, CUDA graph, 100-call average ────────────────
    print("=" * 74)
    print("(5) wall time per call in a CUDA graph, 256 experts bf16, "
          "rotated weight copies (cold L2)")
    E, I, IS = 256, 1024, 1024
    NCOPY, nrep = 6, 100
    mkb = lambda *s: (torch.randn(*s, device=DEVICE) * 0.05).to(DTYPE)
    WG = [mkb(E, I, H) for _ in range(NCOPY)]
    WU = [mkb(E, I, H) for _ in range(NCOPY)]
    WD = [mkb(E, H, I) for _ in range(NCOPY)]
    SG = [mkb(H, IS) for _ in range(NCOPY)]
    SU = [mkb(H, IS) for _ in range(NCOPY)]
    SD = [mkb(IS, H) for _ in range(NCOPY)]
    WR = [mkb(H, E) for _ in range(NCOPY)]
    IDX = [torch.randperm(E, device=DEVICE)[:TOPK].unsqueeze(0).contiguous()
           for _ in range(nrep)]
    wgt = torch.full((1, TOPK), 0.125, device=DEVICE)
    bias = torch.randn(E, device=DEVICE) * 0.5
    scale = torch.full((1, 1), 2.446, device=DEVICE, dtype=DTYPE)
    tok = torch_kimi_rms_norm(x, gamma)
    n = [0]

    def rot(fn):
        def call():
            c = n[0] % NCOPY
            fn(c, IDX[n[0] % nrep])
            n[0] += 1
        return call

    DG = [mkb(H, 9216) for _ in range(2)]
    DU = [mkb(H, 9216) for _ in range(2)]
    DD = [mkb(9216, H) for _ in range(2)]
    WH = [mkb(H, reference.CONFIG.vocab_size) for _ in range(2)]

    legs = (
        ("routing", rot(lambda c, i: km.routing(tok, WR[c], bias, scale))),
        ("experts (2 kernels)",
         rot(lambda c, i: km.experts(tok, wgt, i, WG[c], WU[c], WD[c],
                                     SG[c], SU[c], SD[c]))),
        ("moe_block (norm+route+experts)",
         rot(lambda c, i: km.moe_block(x, gamma, WR[c], bias, scale,
                                       WG[c], WU[c], WD[c],
                                       SG[c], SU[c], SD[c]))),
        ("dense_mlp (layer 0, IS=9216)",
         rot(lambda c, i: km.dense_mlp(x, gamma, DG[c % 2], DU[c % 2],
                                       DD[c % 2]))),
        ("lm_head (vocab 163840)",
         rot(lambda c, i: kb.lm_head(tok, WH[c % 2]))),
    )
    for name, call in legs:
        graph_bench(call, reps=20, iters=3)  # clock ramp
        print(f"  {name:32s} {graph_bench(call, reps=nrep, iters=20):8.2f} us")

    print("=" * 74)
    print("ALL PASS" if ok else "FAILURES -- see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
