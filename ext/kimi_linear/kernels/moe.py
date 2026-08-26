"""Kimi-Linear's MoE block in tilelang: sigmoid router with a selection-only
correction bias, 256 routed experts at top-8, one shared expert -- plus layer
0's dense SwiGLU MLP, which is the shared expert's shape at 9x the width.

Adapted from `examples/qwen3_5_35b_a3b-tilelang/kernels/moe.py`; that file's
measurements carry over (same H200, same decode shape), and its docstring
explains the design: the GEMV-as-GEMM with the vector in row 0 of a 16-row
tile, the free hi/lo residual row that turns a bf16 activation into a 17-bit
one, and why the block is two kernels (one produces every SwiGLU hidden,
routed and shared; one consumes them) rather than four.

What is different from Qwen's block
-----------------------------------
* **The router is sigmoid + bias, not softmax.** Selection reads
  `sigmoid(logits) + e_score_correction_bias`; the routing *weights* are the
  unbiased sigmoid scores of the selected experts, renormalised over the top 8
  and multiplied by `routed_scaling_factor` (2.446). The bias moves *which*
  experts run without appearing in *how much* they count, so `_router_select`
  selects on the biased score and recovers the weight as the sigmoid of the
  selected logit -- exact, and no second pass over the score row. The weights
  leave the kernel rounded to bf16, which is what the authored `router` hands
  the experts.
* **The shared expert is unscaled.** Qwen's has a learned scalar gate; Kimi's
  is a plain SwiGLU, so there is no gate block in either fused kernel and the
  shared contribution is added unweighted.
* **The post-attention RMSNorm rounds before gamma.** `post_norm` is
  `basic.rms_norm`: `bf16(bf16(x*rsqrt) * gamma)`, not the template's
  f32-throughout norm.
* **H = 2304.** The router's K-split picks 12 blocks of 192 (16 does not
  divide 2304 into 64-wide stages); everything else tiles the same.

Weight layouts (the authored declarations, after the shell's converters):
  w_router (H, E); w_gate/w_up (E, I, H) and w_down (E, H, I), expert-major;
  sh_gate/sh_up (H, IS) and sh_down (IS, H), dense `(in, out)`.
"""
from __future__ import annotations

import functools

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

try:
    from . import basic
except ImportError:  # `python kernels/moe.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kernels import basic

#: `config.num_experts_per_token`. Not derivable from any tensor `routing` is
#: handed -- the weights it returns are (S, K), which is the answer.
TOP_K = 8

BM = 16  #: rows of the MMA tile; row 0 carries the vector, 1..15 are dead

#: Routed gate/up: 64 out-channels per block, so 1024/64 x 8 slots = 128
#: blocks (plus 16 for the shared expert: 144). The template measured this
#: tile cold on the same GPU: BN=64's 64 rows x 512 B per k-step buys more
#: DRAM concurrency per block than BN=32, and more blocks (BN=128) do not
#: make it back.
BN_H = 64

#: Down projection: 2304/64 x 8 slots = 288 blocks (plus 36 shared). The
#: reduction is only 1024 long, so block count is what buys concurrency.
BN_D, BK_D = 64, 128

THREADS = 128

#: A fused kernel branches on the block index into a `transpose_B` path (the
#: routed experts' (out, in) weights) and a plain path (the shared expert's
#: (in, out) ones), and warp specialisation races across that branch -- the
#: template measured a non-deterministic 30%-wrong shared hidden and a nan
#: scalar gate. Off is also faster (15.7 us against 20.9).
_NO_WS = {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _wdt(t: torch.Tensor) -> str:
    if t.dtype is torch.bfloat16:
        return "bfloat16"
    if t.dtype is torch.float32:
        return "float32"
    raise TypeError(f"weights must be bfloat16 or float32, got {t.dtype}")


def _hcfg(wdt: str, mode: str) -> tuple[int, int]:
    """(BK, stages) for the hidden kernel.

    The fused mode carries four weight tiles (gate and up, both orientations)
    where the others carry two, and an f32 weight doubles that again. At
    H=2304 the staged token is 72 KB, so "both" gets BK=128 where the template
    (H=2048) could afford 256.
    """
    if wdt == "float32":
        return (64, 2) if mode == "both" else (128, 2)
    return (128, 2) if mode == "both" else (256, 2)


def _stages(wdt: str) -> int:
    return 2


# ---------------------------------------------------------------- staging ----
# Macros rather than inlined text: the same lines appear at every call site,
# and `T.macro` gets the same source rewrite as `T.prim_func`, which is what
# makes `hi[i, j] = ...` legal inside one.


@T.macro
def _stage2(hi, lo, src, r0, c0, D0, D1, split: bool):
    """`src[r0:r0+D0, c0:c0+D1]` into shared, as bf16 (hi + lo if split)."""
    if split:
        for i, j in T.Parallel(D0, D1):
            v = src[r0 + i, c0 + j]
            h = T.cast(v, "bfloat16")
            hi[i, j] = h
            lo[i, j] = T.cast(v - T.cast(h, "float32"), "bfloat16")
    else:
        T.copy(src[r0:r0 + D0, c0:c0 + D1], hi)


@T.macro
def _stage3(hi, lo, src, e, r0, c0, D0, D1, split: bool):
    """`src[e, r0:r0+D0, c0:c0+D1]` into shared, as bf16. `e` is the gather."""
    if split:
        for i, j in T.Parallel(D0, D1):
            v = src[e, r0 + i, c0 + j]
            h = T.cast(v, "bfloat16")
            hi[i, j] = h
            lo[i, j] = T.cast(v - T.cast(h, "float32"), "bfloat16")
    else:
        T.copy(src[e, r0:r0 + D0, c0:c0 + D1], hi)


@T.macro
def _split2(xs, v, j):
    """`v` into rows 0 and 1 of the staged vector: bf16 hi, then the residual.

    Row 1 is free MMA work -- the MMA computes all 16 rows anyway.
    """
    r0 = T.cast(v, "bfloat16")
    xs[0, j] = r0
    xs[1, j] = T.cast(v - T.cast(r0, "float32"), "bfloat16")


# ---------------------------------------------------------------- router ----


def _router_split(H: int, wdt: str) -> int:
    """Blocks the router's GEMV splits into: the largest split whose chunk

    stages whole BK tiles (H=2304 admits 12 x 192, not the template's 16).
    """
    bk = 64 if wdt == "bfloat16" else 32
    for ns in (16, 12, 9, 8, 6, 4, 3, 2, 18, 24, 36, 1):
        if H % ns == 0 and (H // ns) % bk == 0:
            return ns
    return 1


@functools.lru_cache(maxsize=None)
def _router_logits(wdt: str, H: int, E: int, NS: int, xdt: str):
    """`Lp[c] = x[c-th chunk] @ W[c-th chunk]`, f32, one block per chunk.

    Three staged rows, not two: this GEMV's consumer is a *selection*, and the
    gap between the 8th and 9th biased score of 256 can be far inside bf16
    rounding, so the logits are computed to ~26 bits. The rows are free: the
    MMA computes all 16 anyway. Partials rather than atomics so the sum is
    over `c` in the same order for every column -- two identical columns of
    `W` then give bitwise identical logits, which is what makes a tie a tie.
    """
    CH = H // NS
    BK = min(CH, 64 if wdt == "bfloat16" else 32)
    ST = min(CH // BK, 4)
    THR = max(32, min(256, 32 * (E // 8)))

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, H), xdt),
            W: T.Tensor((H, E), wdt),
            Lp: T.Tensor((NS, E), "float32"),
        ):
            with T.Kernel(NS, threads=THR) as c:
                xs = T.alloc_shared((BM, CH), "bfloat16")
                wh = T.alloc_shared((BK, E), "bfloat16")
                wl = T.alloc_shared((BK, E), "bfloat16") if wdt == "float32" else wh
                acc = T.alloc_fragment((BM, E), "float32")
                out = T.alloc_shared((BM, E), "float32")
                for j in T.Parallel(CH):
                    # The `< H` guard is a bounds check and also the reason
                    # `H` resolves at all: tilelang evaluates the annotations
                    # against the kernel's closure cells, and a name the body
                    # never references gets no cell (basic.py records the
                    # same trap). The `xdt` conditional is the same favour.
                    v = X[0, c * CH + j] if c * CH + j < H else 0.0
                    vf = v if xdt == "float32" else T.cast(v, "float32")
                    r0 = T.cast(vf, "bfloat16")
                    d1 = vf - T.cast(r0, "float32")
                    r1 = T.cast(d1, "bfloat16")
                    xs[0, j] = r0
                    xs[1, j] = r1
                    xs[2, j] = T.cast(d1 - T.cast(r1, "float32"), "bfloat16")
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(CH // BK, num_stages=ST):
                    _stage2(wh, wl, W, c * CH + ko * BK, 0, BK, E,
                            wdt == "float32")
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], wh, acc)
                    if wdt == "float32":
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], wl, acc)
                T.copy(acc, out)
                for j in T.Parallel(E):
                    Lp[c, j] = out[0, j] + out[1, j] + out[2, j]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _router_select(E: int, K: int, NS: int, bdt: str, sdt: str):
    """Select on sigmoid(logits) + bias -> weights from the UNBIASED scores.

    The selection is K rounds of one `reduce_max` over an int64 key,
    `bits(exp(b - max)) << 32 | (E-1-j)`: `exp(b - max)` is non-negative so
    its bit pattern sorts as an integer, and packing the complement of the
    index underneath means a single max returns the largest value *and*, among
    equals, the lowest index -- `torch.topk`'s selection rule in one
    reduction. The max subtraction keeps `exp` in range and changes nothing
    about the order.

    The weight of a selected expert is `sigmoid(logit)` -- exactly the
    authored `top_biased - bias[index]` -- renormalised over the top K, scaled
    by `routed_scaling_factor`, and rounded to bf16, which is the dtype the
    authored `router` returns. One warp; K+1 whole-warp reductions over E and
    nothing else.
    """
    THR = 32

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Lp: T.Tensor((NS, E), "float32"),
            Bias: T.Tensor((E,), bdt),
            Scale: T.Tensor((1,), sdt),
            Wt: T.Tensor((1, K), "float32"),
            Ind: T.Tensor((1, K), "int64"),
        ):
            with T.Kernel(1, threads=THR) as _:
                part = T.alloc_fragment((NS, E), "float32")
                lg = T.alloc_fragment((E,), "float32")
                lgs = T.alloc_shared((E,), "float32")
                biased = T.alloc_fragment((E,), "float32")
                mx = T.alloc_fragment((1,), "float32")
                bmax = T.alloc_shared((1,), "float32")
                key = T.alloc_shared((E,), "int64")
                kf = T.alloc_fragment((E,), "int64")
                top = T.alloc_fragment((1,), "int64")
                seli = T.alloc_shared((K,), "int64")
                tot = T.alloc_shared((1,), "float32")

                # Every partial at once, then one reduction: a serial loop
                # over `c` would chain NS dependent global loads.
                for c, j in T.Parallel(NS, E):
                    part[c, j] = Lp[c, j]
                T.reduce_sum(part, lg, dim=0)
                T.copy(lg, lgs)
                T.sync_threads()
                for j in T.Parallel(E):
                    # `bdt` spelled out in the body so the annotation resolves
                    # (the closure-cell trap, again).
                    bj = Bias[j] if bdt == "float32" else T.cast(Bias[j], "float32")
                    biased[j] = 1.0 / (1.0 + T.exp(-lgs[j])) + bj
                T.reduce_max(biased, mx, dim=0, clear=True)
                if T.get_thread_binding() == 0:
                    bmax[0] = mx[0]
                T.sync_threads()
                for j in T.Parallel(E):
                    key[j] = (T.cast(T.reinterpret(
                        "int32", T.exp(biased[j] - bmax[0])), "int64") << 32) \
                        | T.cast(E - 1 - j, "int64")
                T.sync_threads()
                for r in T.serial(K):
                    for j in T.Parallel(E):
                        kf[j] = key[j]
                    T.reduce_max(kf, top, dim=0, clear=True)
                    if T.get_thread_binding() == 0:
                        i = E - 1 - T.cast(top[0] & T.cast(0xFFFFFFFF, "int64"),
                                           "int32")
                        seli[r] = T.cast(i, "int64")
                        key[i] = T.cast(-1, "int64")  # below every real key
                    T.sync_threads()
                if T.get_thread_binding() == 0:
                    tot[0] = 0.0
                    for r in T.serial(K):
                        i = T.cast(seli[r], "int32")
                        tot[0] += 1.0 / (1.0 + T.exp(-lgs[i]))
                    for r in T.serial(K):
                        i = T.cast(seli[r], "int32")
                        sc = Scale[0] if sdt == "float32" else T.cast(Scale[0], "float32")
                        w = (1.0 / (1.0 + T.exp(-lgs[i]))) / tot[0] * sc
                        # The authored router hands the experts bf16 weights.
                        Wt[0, r] = T.cast(T.cast(w, "bfloat16"), "float32")
                        Ind[0, r] = seli[r]

        return main

    return build()


# ------------------------------------------------------------ the experts ----


def _h_grid(mode: str, K: int, I: int, IS: int) -> tuple[int, int, int]:
    """(routed blocks, shared blocks, total) for the hidden kernel."""
    nr = K * (I // BN_H) if mode != "shared" else 0
    ns = IS // BN_H if mode != "routed" else 0
    return nr, ns, nr + ns


@functools.lru_cache(maxsize=None)
def _h_kernel(mode: str, wdt: str, H: int, E: int, K: int, I: int, IS: int,
              xdt: str):
    """`h[s] = silu(w_gate[e_s] @ x) * (w_up[e_s] @ x)` for the K routed slots,

    and the same for the shared expert.

    One kernel for both because they read the same token and their outputs are
    all consumed by `_down_kernel`: two launches would cost 1.6 us more, and
    the shared expert's 16 blocks cannot fill the machine on their own but
    slot into the routed kernel's grid for free. The block index selects the
    job. The branch exists because the orientations differ: an expert's
    `w_gate[e]` is (out, in), so the MMA takes `transpose_B`; the shared
    expert's `sh_gate` is (in, out) and its tile is the transpose of the
    routed one.

    Blocks in the routed range also zero `Out`, which `_down_kernel`
    atomically accumulates into -- 2304 stores spread over 128 blocks that
    were going to run anyway, against a 1.2 us `Tensor.zero_()` graph node.
    """
    BK, ST = _hcfg(wdt, mode)
    NR, NS, NB = _h_grid(mode, K, I, IS)
    NTR = I // BN_H
    KOR = H // BK
    CH = (H + NB - 1) // NB  # zeroing: this block's slice of Out

    # One signature serves all three modes: the tensors a mode never reads are
    # declared 1-element and the caller passes a dummy. The shapes live in a
    # dict because a name that appears *only* in an annotation gets no closure
    # cell and raises `NameError` -- `SH` is read by `T.Kernel(SH["nb"], ...)`
    # below, which is what makes every shape in it resolve.
    SH = dict(
        nb=NB,
        ex=(E, I, H) if mode != "shared" else (1, 1, 1),
        k=K if mode != "shared" else 1,
        hr=(K, I) if mode != "shared" else (1, 1),
        sg=(H, IS) if mode != "routed" else (1, 1),
        hs=IS if mode != "routed" else 1,
    )

    @tilelang.jit(pass_configs=_NO_WS if mode == "both" else None)
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, H), xdt),
            Idx: T.Tensor((1, SH["k"]), "int64"),
            WG: T.Tensor(SH["ex"], wdt),
            WU: T.Tensor(SH["ex"], wdt),
            WSG: T.Tensor(SH["sg"], wdt),
            WSU: T.Tensor(SH["sg"], wdt),
            HR: T.Tensor(SH["hr"], "float32"),
            HS: T.Tensor((SH["hs"],), "float32"),
            Out: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(SH["nb"], threads=THREADS) as b:
                xs = T.alloc_shared((BM, H), "bfloat16")
                gh = T.alloc_shared((BN_H, BK), "bfloat16")
                gl = T.alloc_shared((BN_H, BK), "bfloat16") \
                    if wdt == "float32" else gh
                uh = T.alloc_shared((BN_H, BK), "bfloat16")
                ul = T.alloc_shared((BN_H, BK), "bfloat16") \
                    if wdt == "float32" else uh
                # (BK, BN) against the routed (BN, BK): the same bytes, not
                # the same buffer.
                sgh = T.alloc_shared((BK, BN_H), "bfloat16")
                sgl = T.alloc_shared((BK, BN_H), "bfloat16") \
                    if wdt == "float32" else sgh
                suh = T.alloc_shared((BK, BN_H), "bfloat16")
                sul = T.alloc_shared((BK, BN_H), "bfloat16") \
                    if wdt == "float32" else suh
                ag = T.alloc_fragment((BM, BN_H), "float32")
                au = T.alloc_fragment((BM, BN_H), "float32")
                og = T.alloc_shared((BM, BN_H), "float32")
                ou = T.alloc_shared((BM, BN_H), "float32")

                for j in T.Parallel(H):
                    v = X[0, j]
                    _split2(xs, v if xdt == "float32" else T.cast(v, "float32"), j)
                T.clear(ag)
                T.clear(au)
                if mode != "shared":
                    for j in T.Parallel(CH):
                        if b * CH + j < H:
                            Out[b * CH + j] = 0.0
                T.sync_threads()

                if mode != "shared":
                    if b < NR:
                        s = b // NTR
                        ti = b % NTR
                        e = Idx[0, s]
                        for ko in T.Pipelined(KOR, num_stages=ST):
                            _stage3(gh, gl, WG, e, ti * BN_H, ko * BK,
                                    BN_H, BK, wdt == "float32")
                            _stage3(uh, ul, WU, e, ti * BN_H, ko * BK,
                                    BN_H, BK, wdt == "float32")
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], gh, ag,
                                   transpose_B=True)
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], uh, au,
                                   transpose_B=True)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], gl, ag,
                                       transpose_B=True)
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], ul, au,
                                       transpose_B=True)
                        T.copy(ag, og)
                        T.copy(au, ou)
                        for j in T.Parallel(BN_H):
                            g = og[0, j] + og[1, j]
                            u = ou[0, j] + ou[1, j]
                            HR[s, ti * BN_H + j] = g / (1.0 + T.exp(-g)) * u

                if mode != "routed":
                    if NR <= b:
                        ti = b - NR
                        for ko in T.Pipelined(KOR, num_stages=ST):
                            _stage2(sgh, sgl, WSG, ko * BK, ti * BN_H,
                                    BK, BN_H, wdt == "float32")
                            _stage2(suh, sul, WSU, ko * BK, ti * BN_H,
                                    BK, BN_H, wdt == "float32")
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], sgh, ag)
                            T.gemm(xs[:, ko * BK:(ko + 1) * BK], suh, au)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], sgl, ag)
                                T.gemm(xs[:, ko * BK:(ko + 1) * BK], sul, au)
                        T.copy(ag, og)
                        T.copy(au, ou)
                        for j in T.Parallel(BN_H):
                            g = og[0, j] + og[1, j]
                            u = ou[0, j] + ou[1, j]
                            HS[ti * BN_H + j] = g / (1.0 + T.exp(-g)) * u

        return main

    return build()


def _down_grid(mode: str, H: int, K: int) -> tuple[int, int, int]:
    nt = H // BN_D
    nr = nt * K if mode != "shared" else 0
    ns = nt if mode != "routed" else 0
    return nr, ns, nr + ns


@functools.lru_cache(maxsize=None)
def _down_kernel(mode: str, wdt: str, H: int, E: int, K: int, I: int, IS: int):
    """`out = sum_s weights[s] * (w_down[e_s] @ h[s]) + h_s @ sh_down`.

    One block per (output tile, slot) and `atomic_add` into a zeroed output,
    not one block per output tile looping the slots: the template measured
    3.1x between those shapes at this reduction length, all of it block count.
    The routing weight is folded into the staged vector rather than the
    epilogue, so one accumulator serves the whole (slot, k) space.

    The shared expert's vector is staged a BK tile at a time inside the
    pipeline rather than resident: layer 0's dense MLP is this kernel at
    IS=9216, where a resident vector would want 288 KB of shared memory.
    """
    ST = _stages(wdt)
    NR, NS, NB = _down_grid(mode, H, K)
    NT = H // BN_D
    KOD = I // BK_D
    KOS = IS // BK_D

    # See `_h_kernel`: one signature, mode-dependent shapes, in a dict so the
    # annotations resolve against a name the body reads.
    SH = dict(
        nb=NB,
        ex=(E, H, I) if mode != "shared" else (1, 1, 1),
        k=K if mode != "shared" else 1,
        hr=(K, I) if mode != "shared" else (1, 1),
        sd=(IS, H) if mode != "routed" else (1, 1),
        hs=IS if mode != "routed" else 1,
        xw=I,
        h=H,
    )

    @tilelang.jit(pass_configs=_NO_WS if mode == "both" else None)
    def build():
        @T.prim_func
        def main(
            HR: T.Tensor(SH["hr"], "float32"),
            Wt: T.Tensor((1, SH["k"]), "float32"),
            Idx: T.Tensor((1, SH["k"]), "int64"),
            WD: T.Tensor(SH["ex"], wdt),
            WSD: T.Tensor(SH["sd"], wdt),
            HS: T.Tensor((SH["hs"],), "float32"),
            Out: T.Tensor((SH["h"],), "float32"),
        ):
            with T.Kernel(SH["nb"], threads=THREADS) as b:
                xs = T.alloc_shared((BM, SH["xw"]), "bfloat16")
                xcs = T.alloc_shared((BM, BK_D), "bfloat16")
                dh = T.alloc_shared((BN_D, BK_D), "bfloat16")
                dl = T.alloc_shared((BN_D, BK_D), "bfloat16") \
                    if wdt == "float32" else dh
                sh = T.alloc_shared((BK_D, BN_D), "bfloat16")
                sl = T.alloc_shared((BK_D, BN_D), "bfloat16") \
                    if wdt == "float32" else sh
                acc = T.alloc_fragment((BM, BN_D), "float32")
                out = T.alloc_shared((BM, BN_D), "float32")
                T.clear(acc)

                if mode != "shared":
                    if b < NR:
                        # Tile-major so the 8 blocks sharing an output tile
                        # are not issued back to back; their atomics collide.
                        s = b // NT
                        ti = b % NT
                        e = Idx[0, s]
                        w = Wt[0, s]
                        for j in T.Parallel(I):
                            _split2(xs, HR[s, j] * w, j)
                        T.sync_threads()
                        for ko in T.Pipelined(KOD, num_stages=ST):
                            _stage3(dh, dl, WD, e, ti * BN_D, ko * BK_D,
                                    BN_D, BK_D, wdt == "float32")
                            T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], dh, acc,
                                   transpose_B=True)
                            if wdt == "float32":
                                T.gemm(xs[:, ko * BK_D:(ko + 1) * BK_D], dl,
                                       acc, transpose_B=True)
                        T.copy(acc, out)
                        for j in T.Parallel(BN_D):
                            T.atomic_add(Out[ti * BN_D + j],
                                         out[0, j] + out[1, j])

                if mode != "routed":
                    if NR <= b:
                        ti = b - NR
                        for ko in T.Pipelined(KOS, num_stages=ST):
                            for j in T.Parallel(BK_D):
                                _split2(xcs, HS[ko * BK_D + j], j)
                            _stage2(sh, sl, WSD, ko * BK_D, ti * BN_D,
                                    BK_D, BN_D, wdt == "float32")
                            T.gemm(xcs, sh, acc)
                            if wdt == "float32":
                                T.gemm(xcs, sl, acc)
                        T.copy(acc, out)
                        for j in T.Parallel(BN_D):
                            v = out[0, j] + out[1, j]
                            if mode == "shared":
                                Out[ti * BN_D + j] = v  # sole writer
                            else:
                                T.atomic_add(Out[ti * BN_D + j], v)

        return main

    return build()


# ------------------------------------------------------------ entry points ----

_DUMMY: dict = {}


def _dummy(shape, dtype, device):
    """A 1-element stand-in for a tensor this mode's kernel never reads.

    Cached at module scope so its address is stable across a graph capture.
    """
    key = (shape, dtype, device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(shape, dtype=dtype, device=device)
    return _DUMMY[key]


def post_norm(hidden: torch.Tensor, gamma_post: torch.Tensor) -> torch.Tensor:
    """The fused post-attention RMSNorm: bf16 in, `(1, H)` bf16 out.

    `bf16(bf16(x * rsqrt(mean + eps)) * gamma)` -- the authored rounding, see
    `basic.rms_norm`.
    """
    H = hidden.shape[-1]
    out = torch.empty((1, H), device=hidden.device, dtype=torch.bfloat16)
    basic._rms_norm()(hidden.view(H), gamma_post.view(H), out.view(H))
    return out


def routing(tokens: torch.Tensor, w_router: torch.Tensor, bias: torch.Tensor,
            routed_scale: torch.Tensor):
    """(weights (1, TOP_K) f32 holding bf16 values, indices (1, TOP_K) int64).

    *tokens* is the `(1, H)` bf16 output of `post_norm`; the GEMV reads it
    exactly (bf16 staged is the identity cast).
    """
    H, E = w_router.shape
    dev = tokens.device
    wdt = _wdt(w_router)
    ns = _router_split(H, wdt)
    x = tokens.view(1, H)
    xdt = _wdt(x)
    weights = torch.empty((1, TOP_K), device=dev, dtype=torch.float32)
    indices = torch.empty((1, TOP_K), device=dev, dtype=torch.int64)
    logits = torch.empty((ns, E), device=dev, dtype=torch.float32)
    _router_logits(wdt, H, E, ns, xdt)(x, w_router, logits)
    _router_select(E, TOP_K, ns, _wdt(bias), _wdt(routed_scale))(
        logits,
        bias.view(E),
        routed_scale.reshape(1),
        weights,
        indices,
    )
    return weights, indices


def routed_experts(tokens, weights, indices, w_gate, w_up, w_down) -> torch.Tensor:
    """The 8 selected experts, mixed by `weights`. (1, H) f32."""
    E, I, H = w_gate.shape
    K = weights.shape[1]
    wdt = _wdt(w_gate)
    dev = tokens.device
    x = tokens.view(1, H)
    xdt = _wdt(x)
    out = torch.empty((1, H), device=dev, dtype=torch.float32)
    hr = torch.empty((K, I), device=dev, dtype=torch.float32)
    d3 = _dummy((1, 1), torch.__dict__[wdt], dev)
    d1 = _dummy((1,), torch.float32, dev)
    _h_kernel("routed", wdt, H, E, K, I, 1, xdt)(
        x, indices, w_gate, w_up, d3, d3, hr, d1, out.view(H))
    _down_kernel("routed", wdt, H, E, K, I, 1)(
        hr, weights, indices, w_down, d3, d1, out.view(H))
    return out


def shared_expert(tokens, sh_gate, sh_up, sh_down) -> torch.Tensor:
    """The unscaled dense expert every token goes through. (1, H) f32."""
    H, IS = sh_gate.shape
    wdt = _wdt(sh_gate)
    dev = tokens.device
    x = tokens.view(1, H)
    out = torch.empty((1, H), device=dev, dtype=torch.float32)
    hs = torch.empty((IS,), device=dev, dtype=torch.float32)
    d3 = _dummy((1, 1, 1), torch.__dict__[wdt], dev)
    d2 = _dummy((1, 1), torch.__dict__[wdt], dev)
    di = _dummy((1, 1), torch.int64, dev)
    df = _dummy((1, 1), torch.float32, dev)
    _h_kernel("shared", wdt, H, 1, 1, 1, IS, _wdt(x))(
        x, di, d3, d3, sh_gate, sh_up, df, hs, out.view(H))
    _down_kernel("shared", wdt, H, 1, 1, 1, IS)(
        df, df, di, d3, sh_down, hs, out.view(H))
    return out


def experts(tokens, weights, indices, w_gate, w_up, w_down,
            sh_gate, sh_up, sh_down) -> torch.Tensor:
    """Routed + shared, (1, 1, H) f32, in two kernels.

    The shared expert's gate/up joins the routed gate/up kernel (same token,
    16 more blocks) and its down joins the routed down kernel (36 more blocks,
    accumulating into the same output), so the whole block is two launches
    instead of four.
    """
    E, I, H = w_gate.shape
    K = weights.shape[1]
    IS = sh_gate.shape[-1]
    wdt = _wdt(w_gate)
    dev = tokens.device
    x = tokens.view(1, H)
    out = torch.empty((1, 1, H), device=dev, dtype=torch.float32)
    hr = torch.empty((K, I), device=dev, dtype=torch.float32)
    hs = torch.empty((IS,), device=dev, dtype=torch.float32)
    _h_kernel("both", wdt, H, E, K, I, IS, _wdt(x))(
        x, indices, w_gate, w_up, sh_gate, sh_up, hr, hs, out.view(H))
    _down_kernel("both", wdt, H, E, K, I, IS)(
        hr, weights, indices, w_down, sh_down, hs, out.view(H))
    return out


def moe_block(hidden, gamma_post, w_router, bias, routed_scale,
              w_gate, w_up, w_down, sh_gate, sh_up, sh_down,
              keep_f32: bool = False) -> torch.Tensor:
    """The authored `moe.moe`, end to end: post-norm -> routing -> experts.

    Takes and returns the authored shapes -- `hidden` `[1, 1, H]` bf16, the
    shared weights `[1, H, IS]` / `[1, IS, H]` -- and returns `[1, 1, H]`
    bf16. The residual belongs to the layer, not here.
    """
    H = w_router.shape[0]
    tok = post_norm(hidden, gamma_post)
    w, i = routing(tok, w_router, bias, routed_scale)
    out = experts(
        tok, w, i, w_gate, w_up, w_down,
        sh_gate.view(H, -1), sh_up.view(H, -1), sh_down.view(-1, H),
    )
    # keep_f32: a TP2 rank all-reduces the un-rounded partial and lands bf16
    # once, on the sum.
    return out if keep_f32 else out.to(torch.bfloat16)


def dense_mlp(hidden, gamma_post, w_gate, w_up, w_down,
              keep_f32: bool = False) -> torch.Tensor:
    """Layer 0's dense SwiGLU (`KimiDenseMlp.mlp`): fused post-norm, then the

    shared-expert kernels at IS=9216. Weights are the authored `[1, H, I]` /
    `[1, I, H]`; returns `[1, 1, H]` bf16.
    """
    H = hidden.shape[-1]
    tok = post_norm(hidden, gamma_post)
    out = shared_expert(
        tok, w_gate.view(H, -1), w_up.view(H, -1), w_down.view(-1, H)
    )
    out = out.view(1, 1, H)
    return out if keep_f32 else out.to(torch.bfloat16)


__all__ = [
    "TOP_K",
    "dense_mlp",
    "experts",
    "moe_block",
    "post_norm",
    "routed_experts",
    "routing",
    "shared_expert",
]
