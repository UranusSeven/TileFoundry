"""`KimiDeltaAttention`: the KDA mixer at one token per step, in tilelang.

Three kernels, mirroring `examples/qwen3_5_35b_a3b-tilelang/kernels/gdn.py`
-----------------------------------------------------------------------------
A decode step moves ~79 MB of weights (56.6 MB q/k/v + 2.3 MB gates + 18.9 MB
w_o) and 2 MB of recurrent state, so -- exactly as in the GDN example -- the
step is memory-bound and the design questions are *how many bytes* and *how
many SMs are pulling them*.

    1  `_in_proj`   rms_norm(hidden) + q/k/v projections + the low-rank gate
                    projections (f_a, g_a, b), plus the convolution-window
                    shift and zeroing kernel 3's atomic accumulator   123 blocks
    2  `_conv_delta`  causal conv + l2 norm + the per-channel-gated delta rule
                    + the g2 output gate                                128 blocks
    3  `_out_proj`  gated output rms norm + w_o                         144 blocks

The seams are the forced ones: the conv needs the whole projection column that
kernel 1's blocks jointly produce (no grid-wide barrier), and the gated output
norm is an RMS over a whole head's 128 dims while kernel 2 splits those dims
across 4 blocks -- kernel 3's K-split is 512 = 4 whole heads, so it normalises
where it reads.

What is different from GDN
--------------------------
* **The forget gate is per channel.** `decay[k] = exp(-exp(A_log[h]) *
  softplus(g[h,k] + dt_bias[h,k]))` is a 128-wide vector per head, and the
  authored state layout is `[1, H, V, K]` (v-major; fla stores the transpose).
  A V-row group is self-contained -- `kv_mem[v]`, `delta[v]`, `updated[v,:]`
  and `read[v]` all need only `S[v,:]`, the full K axis of that one row -- so
  blocks are `(head, 32-row group)` = 32*4 = 128, no communication, the exact
  transpose of the GDN decomposition.
* **The gate projections are low-rank.** f_a/g_a are 2304->128 and b is
  2304->32: too narrow for a `T.gemm` tile to fill the machine, so kernel 1
  gives each a `NBA`-way K-split of hand-rolled fragment dot products riding in
  its idle SMs, and kernel 2 sums the partials. f_b/g_b (128->4096) are then
  per-head 128x128 matmuls recomputed inside kernel 2 -- 4x redundant per head
  and free next to the state traffic, the same trade GDN measured.
* **Everything the MMA reads is genuinely bf16.** `hidden`, `gamma_in` and all
  weights are bf16 on both sides of the comparison (the checkpoint stores bf16
  and the authored IR rounds `hidden_norm` to bf16 before scaling), so the
  in-kernel norm reproduces the authored rounding bit for bit --
  `bf16(bf16(x*rsqrt)*gamma)` -- and no hi/lo split row is needed. The
  projections are then bf16 x bf16 with f32 accumulate, the same arithmetic
  the oracle's cuBLAS matmuls perform.
"""
from __future__ import annotations

import functools
import os

import tilelang
import tilelang.language as T
import torch

try:  # `python -m kernels.kda` and `import kernels.kda` both have to work
    from . import torch_ref
except ImportError:  # pragma: no cover -- `python kernels/kda.py`
    import torch_ref

# ---------------------------------------------------------------------------
# `linear_attn_config` + top-level config, spelled out. Every one of these
# appears in a `T.Tensor` annotation, and tilelang resolves annotation names
# against the kernel's globals and closure cells -- module level is the one
# scope that always resolves (see the GDN example's note on the closure-cell
# trap), so dimensions live here rather than as factory arguments.
# ---------------------------------------------------------------------------
H = 2304  #: hidden_size
NH = 32  #: linear_attn_config num_heads
DK = 128  #: linear_attn_config head_dim (both the key and the value dim)
KP = NH * DK  #: 4096, one projection's width
QKV = 3 * KP  #: 12288, the stacked q/k/v projection width
CW = 4  #: short_conv_kernel_size
WS = CW - 1  #: 3 stored window positions

EPS = 1e-5  #: rms_norm_eps, the input norm and the gated output norm both
L2_EPS = 1e-6  #: l2_normalize: rsqrt of the *sum* of squares plus this
QSCALE = DK**-0.5  #: applied to q *after* the l2 norm

BM = 16  #: rows of the MMA tile; row 0 carries the vector, 1..15 are dead

NQ = KP // 128  #: 32 output tiles per q/k/v projection
NBA = 9  #: K-splits of the low-rank gate projections
KSL = H // NBA  #: 256 rows of K per gate split
NCH = 64  #: rows per serial chunk inside a gate split

NT = H // 128  #: 18 output tiles of w_o
OKS = 8  #: K-splits of w_o -> 144 blocks
NK = KP // OKS  #: 512 = exactly 4 whole heads per kernel-3 block
HPB = NK // DK  #: 4

NJ = 4  #: V-row groups per delta-rule block
BJ = DK // NJ  #: 32 V rows per block
GCH = 64  #: K-chunk of the in-kernel-2 gate matmuls


def _use_torch() -> bool:
    return os.environ.get("TF_IMPL", "tilelang").strip().lower() == "torch"


# ---------------------------------------------------------------------------
# 1. rms_norm(hidden) + q/k/v + the low-rank gate projections
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _in_proj():
    """`entry`, the shifted convolution windows, gate partials, zeroed `Out`.

    `entry` is this token's raw q/k/v projections. The q/k/v blocks are uniform
    because the caller stacks the three weights
    into one (H, 3*KP) tensor once at load time: block `b` owns output tile
    `[b*128, (b+1)*128)` of the stacked width, which *is* the right projection's
    slice. That is what lets one `T.gemm` site serve all three -- two `T.gemm`
    calls in two `T.Pipelined` loops in one kernel do not compile (documented in
    the GDN example), and a per-block branch over three separate weight tensors
    would be three such sites.

    Every block recomputes the input rms norm: 4.6 KB of redundant L2 traffic
    against 590 KB of weight tile, and it saves a launch. The norm reproduces
    the authored rounding exactly: normalise in f32, round to bf16, multiply by
    the bf16 gamma in f32, round to bf16 again.

    The gate blocks (b >= 96) are hand-rolled fragment dot products, not
    `T.gemm`: N=128/32 outputs do not fill an MMA tile usefully, and this keeps
    the one-`T.gemm`-site rule. Each writes `KSL=256`-row partial sums that
    kernel 2 reduces.

    `Out` is zeroed here because kernel 3 accumulates into it with
    `T.atomic_add` and a kernel boundary is the only grid-wide barrier; 96 qkv
    blocks x 24 scalars cover it.
    """
    BN, BK, threads = 128, 128, 128
    KO = H // BK  #: 18
    OPB = H // (3 * NQ)  #: 24 out scalars each qkv block zeroes
    # xs (16, 2304) bf16 = 73.7 KB resident + stages * 32 KB weight tile + 8 KB
    # epilogue has to stay under the 227 KB an H200 SM hands out.
    stages = 4

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Hid: T.Tensor((H,), "bfloat16"),
            Gin: T.Tensor((H,), "bfloat16"),
            Wqkv: T.Tensor((H, QKV), "bfloat16"),
            Wfa: T.Tensor((H, DK), "bfloat16"),
            Wga: T.Tensor((H, DK), "bfloat16"),
            Wb: T.Tensor((H, NH), "bfloat16"),
            CSq: T.Tensor((1, WS, KP), "bfloat16"),
            CSk: T.Tensor((1, WS, KP), "bfloat16"),
            CSv: T.Tensor((1, WS, KP), "bfloat16"),
            Entry: T.Tensor((QKV,), "bfloat16"),
            CNq: T.Tensor((1, WS, KP), "bfloat16"),
            CNk: T.Tensor((1, WS, KP), "bfloat16"),
            CNv: T.Tensor((1, WS, KP), "bfloat16"),
            PF: T.Tensor((NBA, DK), "float32"),
            PG: T.Tensor((NBA, DK), "float32"),
            PBt: T.Tensor((NBA, NH), "float32"),
            Out: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(3 * NQ + 3 * NBA, threads=threads) as b:
                xs = T.alloc_shared((BM, H), "bfloat16")
                whi = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                sq = T.alloc_fragment((H,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                sc = T.alloc_shared((1,), "float32")
                accf = T.alloc_fragment((DK,), "float32")
                accs = T.alloc_fragment((NH,), "float32")
                tmpf = T.alloc_fragment((NCH, DK), "float32")
                tmps = T.alloc_fragment((NCH, NH), "float32")
                redf = T.alloc_fragment((DK,), "float32")
                reds = T.alloc_fragment((NH,), "float32")

                for i in T.Parallel(H):
                    xi = T.cast(Hid[i], "float32")
                    sq[i] = xi * xi
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    sc[0] = T.rsqrt(tot[0] / T.cast(H, "float32") + EPS)
                T.sync_threads()

                if b < 3 * NQ:
                    for i in T.Parallel(H):
                        # The authored `hidden_norm`: round to bf16 *before* the
                        # learned scale multiplies, then round the product too
                        # (torch computes bf16*bf16 in f32 and rounds once).
                        x1 = T.cast(T.cast(Hid[i], "float32") * sc[0], "bfloat16")
                        xs[0, i] = T.cast(
                            T.cast(x1, "float32") * T.cast(Gin[i], "float32"),
                            "bfloat16",
                        )
                    for j in T.Parallel(OPB):
                        Out[b * OPB + j] = 0.0
                    T.clear(acc)
                    T.sync_threads()
                    for ko in T.Pipelined(KO, num_stages=stages):
                        T.copy(Wqkv[ko * BK:(ko + 1) * BK, b * BN:(b + 1) * BN], whi)
                        T.gemm(xs[:, ko * BK:(ko + 1) * BK], whi, acc)
                    T.copy(acc, out)
                    # The entry is rounded to bf16 once here; the convolution in
                    # kernel 2 and the window this block appends it to both read
                    # that same rounded value, exactly as the oracle's bf16
                    # matmul output feeds both.
                    for j in T.Parallel(BN):
                        Entry[b * BN + j] = T.cast(out[0, j], "bfloat16")
                    role = b // NQ
                    cl0 = (b % NQ) * BN
                    if role == 0:
                        for j in T.Parallel(BN):
                            CNq[0, 0, cl0 + j] = CSq[0, 1, cl0 + j]
                            CNq[0, 1, cl0 + j] = CSq[0, 2, cl0 + j]
                            CNq[0, 2, cl0 + j] = T.cast(out[0, j], "bfloat16")
                    elif role == 1:
                        for j in T.Parallel(BN):
                            CNk[0, 0, cl0 + j] = CSk[0, 1, cl0 + j]
                            CNk[0, 1, cl0 + j] = CSk[0, 2, cl0 + j]
                            CNk[0, 2, cl0 + j] = T.cast(out[0, j], "bfloat16")
                    else:
                        for j in T.Parallel(BN):
                            CNv[0, 0, cl0 + j] = CSv[0, 1, cl0 + j]
                            CNv[0, 1, cl0 + j] = CSv[0, 2, cl0 + j]
                            CNv[0, 2, cl0 + j] = T.cast(out[0, j], "bfloat16")
                else:
                    # f_a / g_a / b: 288 outputs over K=2304, too narrow for an
                    # MMA tile. `NBA` K-splits ride along in the blocks the qkv
                    # projection leaves idle; kernel 2 sums the partials.
                    e = b - 3 * NQ
                    split = e % NBA
                    role2 = e // NBA
                    for j in T.Parallel(DK):
                        accf[j] = 0.0
                    for j in T.Parallel(NH):
                        accs[j] = 0.0
                    for c in T.serial(KSL // NCH):
                        if role2 == 0:
                            for m, j in T.Parallel(NCH, DK):
                                krow = split * KSL + c * NCH + m
                                x1 = T.cast(
                                    T.cast(Hid[krow], "float32") * sc[0], "bfloat16"
                                )
                                xn = T.cast(
                                    T.cast(x1, "float32") * T.cast(Gin[krow], "float32"),
                                    "bfloat16",
                                )
                                tmpf[m, j] = (
                                    T.cast(xn, "float32")
                                    * T.cast(Wfa[krow, j], "float32")
                                )
                            T.reduce_sum(tmpf, redf, dim=0)
                            for j in T.Parallel(DK):
                                accf[j] = accf[j] + redf[j]
                        elif role2 == 1:
                            for m, j in T.Parallel(NCH, DK):
                                krow = split * KSL + c * NCH + m
                                x1 = T.cast(
                                    T.cast(Hid[krow], "float32") * sc[0], "bfloat16"
                                )
                                xn = T.cast(
                                    T.cast(x1, "float32") * T.cast(Gin[krow], "float32"),
                                    "bfloat16",
                                )
                                tmpf[m, j] = (
                                    T.cast(xn, "float32")
                                    * T.cast(Wga[krow, j], "float32")
                                )
                            T.reduce_sum(tmpf, redf, dim=0)
                            for j in T.Parallel(DK):
                                accf[j] = accf[j] + redf[j]
                        else:
                            for m, j in T.Parallel(NCH, NH):
                                krow = split * KSL + c * NCH + m
                                x1 = T.cast(
                                    T.cast(Hid[krow], "float32") * sc[0], "bfloat16"
                                )
                                xn = T.cast(
                                    T.cast(x1, "float32") * T.cast(Gin[krow], "float32"),
                                    "bfloat16",
                                )
                                tmps[m, j] = (
                                    T.cast(xn, "float32")
                                    * T.cast(Wb[krow, j], "float32")
                                )
                            T.reduce_sum(tmps, reds, dim=0)
                            for j in T.Parallel(NH):
                                accs[j] = accs[j] + reds[j]
                    if role2 == 0:
                        for j in T.Parallel(DK):
                            PF[split, j] = accf[j]
                    elif role2 == 1:
                        for j in T.Parallel(DK):
                            PG[split, j] = accf[j]
                    else:
                        for j in T.Parallel(NH):
                            PBt[split, j] = accs[j]

        return main

    return build()


# ---------------------------------------------------------------------------
# 2. causal conv + l2 norm + the per-channel-gated delta rule + the g2 gate
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _conv_delta():
    """One block is `(head, 32-row group of the V axis)`; 32*4 = 128 blocks.

    Each block recomputes its head's q/k convolutions and l2 norms, the low-rank
    partial-sum reductions, and the f_b gate matmul -- 4x redundant per head and
    free next to the state traffic, exactly the trade the GDN example measured.
    The `vg == 0` block of each head additionally computes g2 (the 128->4096
    g_b projection, its head's slice) so kernel 3 does not have to: 18 of its
    blocks share each head's K-slice and would re-read that slice 18x.

    The delta rule consumes and produces the authored `[1, H, V, K]` state
    layout. Per V row, with the whole K axis resident:

        decayed[v, k] = S[v, k] * exp(g[k])
        kv_mem[v]     = sum_k decayed[v, k] * k[k]
        delta[v]      = (v[v] - kv_mem[v]) * beta
        S'[v, k]      = decayed[v, k] + delta[v] * k[k]
        read[v]       = sum_k S'[v, k] * q[k]      (retrieval reads the update)

    The conv output and silu are rounded to bf16 before the l2 norm, matching
    the dtype the fla kernel's q/k/v carry; all arithmetic is f32.
    """
    threads = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Entry: T.Tensor((QKV,), "bfloat16"),
            CSq: T.Tensor((1, WS, KP), "bfloat16"),
            CSk: T.Tensor((1, WS, KP), "bfloat16"),
            CSv: T.Tensor((1, WS, KP), "bfloat16"),
            CWq: T.Tensor((CW, KP), "bfloat16"),
            CWk: T.Tensor((CW, KP), "bfloat16"),
            CWv: T.Tensor((CW, KP), "bfloat16"),
            PF: T.Tensor((NBA, DK), "float32"),
            PG: T.Tensor((NBA, DK), "float32"),
            PBt: T.Tensor((NBA, NH), "float32"),
            Wfb: T.Tensor((DK, KP), "bfloat16"),
            Wgb: T.Tensor((DK, KP), "bfloat16"),
            Dtb: T.Tensor((KP,), "bfloat16"),
            Alog: T.Tensor((NH,), "bfloat16"),
            State: T.Tensor((1, NH, DK, DK), "bfloat16"),
            Read: T.Tensor((1, NH, DK), "float32"),
            Up: T.Tensor((1, NH, DK, DK), "bfloat16"),
            G2: T.Tensor((NH, DK), "float32"),
        ):
            with T.Kernel(NH * NJ, threads=threads) as b:
                h = b // NJ
                v0 = (b % NJ) * BJ

                low = T.alloc_shared((DK,), "float32")
                ga = T.alloc_shared((DK,), "float32")
                qf = T.alloc_shared((DK,), "float32")
                kf = T.alloc_shared((DK,), "float32")
                dsh = T.alloc_shared((DK,), "float32")
                vf = T.alloc_shared((BJ,), "float32")
                dl = T.alloc_shared((BJ,), "float32")
                bta = T.alloc_shared((1,), "float32")
                sc2 = T.alloc_shared((2,), "float32")
                gr = T.alloc_fragment((DK,), "float32")
                gtmp = T.alloc_fragment((GCH, DK), "float32")
                gred = T.alloc_fragment((DK,), "float32")
                dec = T.alloc_fragment((BJ, DK), "float32")
                tmp = T.alloc_fragment((BJ, DK), "float32")
                red = T.alloc_fragment((BJ,), "float32")
                nsq = T.alloc_fragment((DK,), "float32")
                ntot = T.alloc_fragment((1,), "float32")

                # low = hn @ w_f_a, summed from kernel 1's NBA partial rows.
                for j in T.Parallel(DK):
                    low[j] = 0.0
                for e in T.serial(NBA):
                    for j in T.Parallel(DK):
                        low[j] = low[j] + PF[e, j]
                # beta: 9 f32 to add, so one thread; every other thread would
                # compute the same.
                if T.get_thread_binding() == 0:
                    sb = T.alloc_var("float32")
                    sb = 0.0
                    for e in T.serial(NBA):
                        sb += PBt[e, h]
                    bta[0] = 1.0 / (1.0 + T.exp(-sb))
                T.sync_threads()

                # q and k: depthwise causal conv (the window's 3 stored slots
                # plus this token's entry), silu, round to bf16, l2 norm with
                # the eps *inside* the root over the *sum* of squares.
                for j in T.Parallel(DK):
                    c = h * DK + j
                    a = (
                        T.cast(CSq[0, 0, c], "float32") * T.cast(CWq[0, c], "float32")
                        + T.cast(CSq[0, 1, c], "float32") * T.cast(CWq[1, c], "float32")
                        + T.cast(CSq[0, 2, c], "float32") * T.cast(CWq[2, c], "float32")
                        + T.cast(Entry[c], "float32") * T.cast(CWq[3, c], "float32")
                    )
                    a = a / (1.0 + T.exp(-a))
                    qf[j] = T.cast(T.cast(a, "bfloat16"), "float32")
                    nsq[j] = qf[j] * qf[j]
                T.reduce_sum(nsq, ntot, dim=0)
                if T.get_thread_binding() == 0:
                    sc2[0] = T.rsqrt(ntot[0] + L2_EPS)
                T.sync_threads()
                for j in T.Parallel(DK):
                    qf[j] = qf[j] * sc2[0] * QSCALE  # the scale rides on q
                for j in T.Parallel(DK):
                    c = h * DK + j
                    a = (
                        T.cast(CSk[0, 0, c], "float32") * T.cast(CWk[0, c], "float32")
                        + T.cast(CSk[0, 1, c], "float32") * T.cast(CWk[1, c], "float32")
                        + T.cast(CSk[0, 2, c], "float32") * T.cast(CWk[2, c], "float32")
                        + T.cast(Entry[KP + c], "float32") * T.cast(CWk[3, c], "float32")
                    )
                    a = a / (1.0 + T.exp(-a))
                    kf[j] = T.cast(T.cast(a, "bfloat16"), "float32")
                    nsq[j] = kf[j] * kf[j]
                T.reduce_sum(nsq, ntot, dim=0)
                if T.get_thread_binding() == 0:
                    sc2[1] = T.rsqrt(ntot[0] + L2_EPS)
                T.sync_threads()
                for j in T.Parallel(DK):
                    kf[j] = kf[j] * sc2[1]
                T.sync_threads()

                # The per-channel forget gate for this head: g = -exp(A_log) *
                # softplus(low @ w_f_b + dt_bias), decay = exp(g). softplus in
                # the non-overflowing form.
                for j in T.Parallel(DK):
                    gr[j] = 0.0
                for c in T.serial(DK // GCH):
                    for i, j in T.Parallel(GCH, DK):
                        gtmp[i, j] = low[c * GCH + i] * T.cast(
                            Wfb[c * GCH + i, h * DK + j], "float32"
                        )
                    T.reduce_sum(gtmp, gred, dim=0)
                    for j in T.Parallel(DK):
                        gr[j] = gr[j] + gred[j]
                for j in T.Parallel(DK):
                    x = gr[j] + T.cast(Dtb[h * DK + j], "float32")
                    sp = T.log(1.0 + T.exp(-T.abs(x))) + T.max(x, 0.0)
                    dsh[j] = T.exp(-T.exp(T.cast(Alog[h], "float32")) * sp)
                T.sync_threads()

                # v, this block's 32 channels only.
                for i in T.Parallel(BJ):
                    c = h * DK + v0 + i
                    a = (
                        T.cast(CSv[0, 0, c], "float32") * T.cast(CWv[0, c], "float32")
                        + T.cast(CSv[0, 1, c], "float32") * T.cast(CWv[1, c], "float32")
                        + T.cast(CSv[0, 2, c], "float32") * T.cast(CWv[2, c], "float32")
                        + T.cast(Entry[2 * KP + c], "float32")
                        * T.cast(CWv[3, c], "float32")
                    )
                    a = a / (1.0 + T.exp(-a))
                    vf[i] = T.cast(T.cast(a, "bfloat16"), "float32")
                T.sync_threads()

                # ---- the rank-one update, per V row.
                for i, j in T.Parallel(BJ, DK):
                    dec[i, j] = T.cast(State[0, h, v0 + i, j], "float32") * dsh[j]
                for i, j in T.Parallel(BJ, DK):
                    tmp[i, j] = dec[i, j] * kf[j]
                T.reduce_sum(tmp, red, dim=1)  # kv_mem[v], over the K axis
                for i in T.Parallel(BJ):
                    dl[i] = (vf[i] - red[i]) * bta[0]
                T.sync_threads()
                for i, j in T.Parallel(BJ, DK):
                    # `updated` is formed once and both consumed and stored, so
                    # `read` is literally sum_k updated[v,k] * q[k].
                    up = dec[i, j] + kf[j] * dl[i]
                    Up[0, h, v0 + i, j] = T.cast(up, "bfloat16")
                    tmp[i, j] = up * qf[j]
                T.reduce_sum(tmp, red, dim=1)
                for i in T.Parallel(BJ):
                    Read[0, h, v0 + i] = red[i]

                # g2 = sigmoid((hn @ w_g_a) @ w_g_b), this head's slice, from
                # the vg == 0 block only; kernel 3 multiplies it in.
                if b % NJ == 0:
                    for j in T.Parallel(DK):
                        ga[j] = 0.0
                    for e in T.serial(NBA):
                        for j in T.Parallel(DK):
                            ga[j] = ga[j] + PG[e, j]
                    T.sync_threads()
                    for j in T.Parallel(DK):
                        gr[j] = 0.0
                    for c in T.serial(DK // GCH):
                        for i, j in T.Parallel(GCH, DK):
                            gtmp[i, j] = ga[c * GCH + i] * T.cast(
                                Wgb[c * GCH + i, h * DK + j], "float32"
                            )
                        T.reduce_sum(gtmp, gred, dim=0)
                        for j in T.Parallel(DK):
                            gr[j] = gr[j] + gred[j]
                    for j in T.Parallel(DK):
                        G2[h, j] = 1.0 / (1.0 + T.exp(-gr[j]))

        return main

    return build()


# ---------------------------------------------------------------------------
# 3. gated output norm + w_o
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _out_proj():
    """`out = (rms_norm(read, gamma_o) * sigmoid(g2)) @ w_o`.

    w_o is K=4096, N=2304: 18 tiles of 128 would put 18 SMs on 18.9 MB, so K is
    split 8 ways into `T.atomic_add` -- 144 blocks. The K-split is what makes
    the gated norm fusible here at all: `NK = 512` is exactly 4 whole heads, so
    a block holds every value dim of every head it touches and takes each head's
    RMS by itself. `G2` arrives with the sigmoid already applied (kernel 2).
    """
    BN, BK, threads = 128, 128, 128
    KO = NK // BK  #: 4
    stages = 5  #: 16 KB vector + 5 * 32 KB tile + 8 KB epilogue = 184 KB

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Read: T.Tensor((1, NH, DK), "float32"),
            G2: T.Tensor((NH, DK), "float32"),
            Go: T.Tensor((DK,), "bfloat16"),
            Wo: T.Tensor((KP, H), "bfloat16"),
            Out: T.Tensor((H,), "float32"),
        ):
            with T.Kernel(NT * OKS, threads=threads) as b:
                nt = b % NT
                ks = b // NT
                h0 = ks * HPB

                xs = T.alloc_shared((BM, NK), "bfloat16")
                whi = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                rsq = T.alloc_fragment((HPB, DK), "float32")
                rsm = T.alloc_fragment((HPB,), "float32")
                rsc = T.alloc_shared((HPB,), "float32")

                for hh, j in T.Parallel(HPB, DK):
                    rsq[hh, j] = Read[0, h0 + hh, j] * Read[0, h0 + hh, j]
                T.reduce_sum(rsq, rsm, dim=1)
                for hh in T.Parallel(HPB):
                    rsc[hh] = T.rsqrt(rsm[hh] / T.cast(DK, "float32") + EPS)
                T.sync_threads()

                for i in T.Parallel(NK):
                    hh = i // DK
                    j = i % DK
                    xs[0, i] = T.cast(
                        Read[0, h0 + hh, j]
                        * rsc[hh]
                        * T.cast(Go[j], "float32")
                        * G2[h0 + hh, j],
                        "bfloat16",
                    )
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(KO, num_stages=stages):
                    T.copy(
                        Wo[ks * NK + ko * BK:ks * NK + (ko + 1) * BK, nt * BN:(nt + 1) * BN],
                        whi,
                    )
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], whi, acc)
                T.copy(acc, out)
                for j in T.Parallel(BN):
                    T.atomic_add(Out[nt * BN + j], out[0, j])

        return main

    return build()


# ---------------------------------------------------------------------------
# The two standalone `@func` boundaries, so `TF_IMPL` bisects to one of them.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _short_conv():
    """`silu(sum_j window[j] * conv_w[j])` over all KP channels, plus the window.

    The window to store next drops the oldest slot and appends this token. One
    thread per channel; a channel's four taps sit at `conv_w[j, c]`, a
    KP-strided column, so this is four coalesced row reads and the kernel is
    launch-bound.
    """
    threads = 256

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, 1, KP), "bfloat16"),
            CWt: T.Tensor((CW, KP), "bfloat16"),
            CS: T.Tensor((1, WS, KP), "bfloat16"),
            Y: T.Tensor((1, 1, KP), "bfloat16"),
            CN: T.Tensor((1, WS, KP), "bfloat16"),
        ):
            with T.Kernel(KP // threads, threads=threads) as b:
                for t in T.Parallel(threads):
                    c = b * threads + t
                    a = (
                        T.cast(CS[0, 0, c], "float32") * T.cast(CWt[0, c], "float32")
                        + T.cast(CS[0, 1, c], "float32") * T.cast(CWt[1, c], "float32")
                        + T.cast(CS[0, 2, c], "float32") * T.cast(CWt[2, c], "float32")
                        + T.cast(X[0, 0, c], "float32") * T.cast(CWt[3, c], "float32")
                    )
                    Y[0, 0, c] = T.cast(a / (1.0 + T.exp(-a)), "bfloat16")
                    CN[0, 0, c] = CS[0, 1, c]
                    CN[0, 1, c] = CS[0, 2, c]
                    CN[0, 2, c] = X[0, 0, c]

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _l2_normalize():
    """`x * rsqrt(sum(x^2) + 1e-6)` per head. One block per head, 128 threads."""

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((1, 1, NH, DK), "bfloat16"),
            Y: T.Tensor((1, 1, NH, DK), "bfloat16"),
        ):
            with T.Kernel(NH, threads=DK) as h:
                sq = T.alloc_fragment((DK,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                sc = T.alloc_shared((1,), "float32")
                for i in T.Parallel(DK):
                    xi = T.cast(X[0, 0, h, i], "float32")
                    sq[i] = xi * xi
                T.reduce_sum(sq, tot, dim=0)
                if T.get_thread_binding() == 0:
                    sc[0] = T.rsqrt(tot[0] + L2_EPS)
                T.sync_threads()
                for i in T.Parallel(DK):
                    Y[0, 0, h, i] = T.cast(T.cast(X[0, 0, h, i], "float32") * sc[0], "bfloat16")

        return main

    return build()


# ---------------------------------------------------------------------------
# Entry points. Each allocates its outputs and returns them; none synchronises,
# branches on a device value, or has a shape that depends on one, so all three
# capture into a CUDA graph.
# ---------------------------------------------------------------------------


def short_conv(x: torch.Tensor, conv_w: torch.Tensor, conv_state: torch.Tensor):
    """One token's depthwise causal conv with silu, plus the window to store.

    Returns `(out (1, 1, KP), conv_next (1, WS, KP))`, bf16 in and out.
    """
    if _use_torch():
        return torch_ref.short_conv(x, conv_w, conv_state)
    y = torch.empty_like(x)
    cn = torch.empty_like(conv_state)
    _short_conv()(x.contiguous(), conv_w.contiguous(), conv_state.contiguous(), y, cn)
    return y, cn


def l2_normalize(x: torch.Tensor):
    """`(1, 1, NH, DK)` bf16 -- per-head l2 normalisation, eps inside the root."""
    if _use_torch():
        return torch_ref.l2_normalize(x)
    y = torch.empty_like(x)
    _l2_normalize()(x.contiguous(), y)
    return y


_QKV_CACHE: dict = {}
_SCALE_SEEN: dict = {}


def _stack_qkv(w_q, w_k, w_v):
    """The three (1, H, KP) weights as one (H, 3*KP) matrix, cached.

    Kernel 1 needs them as one tensor (see its docstring); stacking is a 57 MB
    copy, so it happens once per weight-set identity, not once per token.
    Weights are persistent in every real caller; keyed on the three data_ptrs.
    """
    key = (w_q.data_ptr(), w_k.data_ptr(), w_v.data_ptr())
    hit = _QKV_CACHE.get(key)
    if hit is None:
        hit = (
            torch.cat([w_q.view(H, KP), w_k.view(H, KP), w_v.view(H, KP)], dim=1)
            .contiguous()
        )
        _QKV_CACHE[key] = hit
    return hit


def kda_step(
    hidden: torch.Tensor,
    gamma_in: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    conv_w_q: torch.Tensor,
    conv_w_k: torch.Tensor,
    conv_w_v: torch.Tensor,
    conv_state_q: torch.Tensor,
    conv_state_k: torch.Tensor,
    conv_state_v: torch.Tensor,
    w_f_a: torch.Tensor,
    w_f_b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_g_a: torch.Tensor,
    w_g_b: torch.Tensor,
    gamma_o: torch.Tensor,
    w_o: torch.Tensor,
    state: torch.Tensor,
    scale: torch.Tensor,
):
    """One KDA decode step: `(out, state_next, conv_q_next, conv_k_next, conv_v_next)`.

    Same argument list and dtypes as the authored
    `tests/models/kimi_linear_48b_a3b/model.py:KimiKda.kda_attention` (weights
    `[1, in, out]` bf16, conv weights `(CW, KP)`, windows `(1, WS, KP)`, state
    `(1, NH, DK, DK)` v-major). `out` and `state_next` are bf16; all internal
    arithmetic is f32. The caller replaces states with the returned ones.

    Three launches. `scale` is read once per tensor identity and asserted to be
    DK**-0.5 -- the kernel bakes the constant, which is the only value the
    authored model ever passes.
    """
    if _use_torch():
        return torch_ref.kda_step(
            hidden, gamma_in, w_q, w_k, w_v, conv_w_q, conv_w_k, conv_w_v,
            conv_state_q, conv_state_k, conv_state_v, w_f_a, w_f_b, dt_bias,
            a_log, w_b, w_g_a, w_g_b, gamma_o, w_o, state, scale,
        )
    dev = hidden.device
    skey = scale.data_ptr()
    if skey not in _SCALE_SEEN:
        s = float(scale.reshape(()).float())
        assert abs(s - QSCALE) < 1e-3, f"scale {s} != DK**-0.5 {QSCALE}"
        _SCALE_SEEN[skey] = True

    # The convolution windows the caller hands back may be sliced views; the
    # packed ABI wants declared strides, and 24 KB per copy is free (the GDN
    # example documents the same trap).
    cs_q = conv_state_q.contiguous()
    cs_k = conv_state_k.contiguous()
    cs_v = conv_state_v.contiguous()

    entry = torch.empty((QKV,), dtype=torch.bfloat16, device=dev)
    cn_q = torch.empty((1, WS, KP), dtype=torch.bfloat16, device=dev)
    cn_k = torch.empty((1, WS, KP), dtype=torch.bfloat16, device=dev)
    cn_v = torch.empty((1, WS, KP), dtype=torch.bfloat16, device=dev)
    pf = torch.empty((NBA, DK), dtype=torch.float32, device=dev)
    pg = torch.empty((NBA, DK), dtype=torch.float32, device=dev)
    pbt = torch.empty((NBA, NH), dtype=torch.float32, device=dev)
    read = torch.empty((1, NH, DK), dtype=torch.float32, device=dev)
    g2 = torch.empty((NH, DK), dtype=torch.float32, device=dev)
    updated = torch.empty((1, NH, DK, DK), dtype=torch.bfloat16, device=dev)
    out32 = torch.empty((H,), dtype=torch.float32, device=dev)

    wqkv = _stack_qkv(w_q, w_k, w_v)
    _in_proj()(
        hidden.reshape(H), gamma_in, wqkv,
        w_f_a.view(H, DK), w_g_a.view(H, DK), w_b.view(H, NH),
        cs_q, cs_k, cs_v,
        entry, cn_q, cn_k, cn_v, pf, pg, pbt, out32,
    )
    _conv_delta()(
        entry, cs_q, cs_k, cs_v,
        conv_w_q.contiguous(), conv_w_k.contiguous(), conv_w_v.contiguous(),
        pf, pg, pbt,
        w_f_b.view(DK, KP), w_g_b.view(DK, KP), dt_bias, a_log,
        state.contiguous(), read, updated, g2,
    )
    _out_proj()(read, g2, gamma_o, w_o.view(KP, H), out32)
    return out32.view(1, 1, H).to(torch.bfloat16), updated, cn_q, cn_k, cn_v


__all__ = ["kda_step", "l2_normalize", "short_conv"]
