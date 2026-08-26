
"""Kimi-Linear-48B-A3B MLA decode step, as five tilelang kernels.

One token per step, shapes fixed by the checkpoint's config: hidden 2304,
32 heads, qk head dim 192 (128 nope + 64 rope), v head dim 128, kv lora
rank 512. Every builder also takes `nh`, the head count it serves: 32 for
TP1, 16 for one rank of TP2. The capitalized locals shadow the module
constants -- tilelang resolves annotation and body names against the
defining frame, so the one source serves both (same trick as
``kernels/kda.py``). Semantics follow ``tests/models/kimi_linear_48b_a3b/model.py``'s
``mla_attention`` exactly -- same roundings, same rope, same cache contract
(the caller appends; this returns only the new token's k and v).

Per-step weight traffic is ~58 MB (w_q 28.3, w_kv_b 8.4, w_o 18.9, w_kv_a
2.7) plus 2 * ctx_len * 32 * 320 * 2 B of cache reads, so this is a
memory-traffic problem with an attention kernel in the middle. It is spent
in five launches:

    1  _input_proj(N=6144)   rms_norm(hidden) @ w_q       -> q partials
    2  _input_proj(N=576)    rms_norm(hidden) @ w_kv_a    -> compressed partials
    3  _kv_b_kernel          latent rms_norm + @ w_kv_b   -> kv partials
    4  _attn_kernel          rope, softmax over cache + self, k_new/v_new out
    5  _out_kernel           log-sum-exp merge + @ w_o    -> out (atomic adds)

Two entry points wrap the same five launches. `mla_attention` is the authored
contract: the cache's length is its shape and the caller appends -- which
compiles kernel 4 once per context length, so it serves checks, not runs.
`mla_attention_cap` is the driver's: the cache is a fixed-capacity buffer
whose live length arrives on the device (`_attn_kernel_cap`), compiled once
per capacity bucket.

Every GEMV is the 16-row-MMA idiom from the qwen3.5 example's
``kernels/basic.py``: the vector sits in row 0 of a 16-row tile, rows 1..15
are left uninitialised (an MMA computes each output row from its own input
row, so garbage lands only where nothing reads it), and row 0 of the
accumulator is reached through shared memory in the epilogue. The hi/lo
split of that example is *not* needed here: the authored IR rounds each
activaation to bf16 before its matmul anyway (DeepseekV3RMSNorm ends
``weight * hidden.to(input_dtype)``; HF casts attention probabilities back
to bf16 for the PV matmul), so a single bf16 row reproduces the reference's
rounding rather than approximating it.

K-splits: a GEMV at BN=128 over these N gives at most 48 (w_q) or 18 (w_o)
column tiles, too few blocks for 132 SMs at ~20 GB/s of per-block reads, so
K is split and partial rows land in f32 buffers. Consumers sum the partials
in their prologues -- a few thousand f32 reads against tens of MB of
weights -- except w_o's splits, which accumulate into the output with
atomics (the attention kernel's block 0 zeroes it).

Launch 2 exists separately from launch 1 on purpose: the latent RMSNorm
needs *all* 512 latent values, which only exist once every w_kv_a partial
is summed, so w_kv_b cannot start in the same launch. Launch 4 recomputes
the rope of the shared 64-wide k part per block (64 elements against 80 KB
of cache per block) instead of paying a launch for it.

NoPE note: Kimi ships ``mla_use_nope: true``, expressed by the caller as
``cos = 1, sin = 0`` -- the identity by arithmetic. The rotary is always
applied here; there is one code path.
"""
# NOTE: no `from __future__ import annotations` here -- annotations must
# evaluate eagerly at `def main` time so the nh shadowing above each
# builder is what T.Tensor shapes see. Under PEP 563 they would be
# strings, resolved later against module globals (TP1 constants) --
# tilelang's get_type_hints only adds closure names the *body* uses,
# so annotation-only names silently fell back to the 32-head shapes.

import functools

import tilelang
import tilelang.language as T
import torch

#: Rows of the MMA tile. 16 is the smallest the SM90 path accepts.
BM = 16

# Published dimensions, spelled out (config.json of the checkpoint).
_HID = 2304
_NH = 32
_NOPE = 128
_ROPE = 64
_QK = _NOPE + _ROPE      # 192, the score dim and so the scaling one
_V = 128
_LAT = 512               # kv_lora_rank
_KVB = _NOPE + _V        # 256, kv_b_proj's per-head output
_NQ = _NH * _QK          # 6144
_NA = _LAT + _ROPE       # 576, kv_a_proj_with_mqa's output
_NBV = _NH * _KVB        # 8192
_OI = _NH * _V           # 4096, o_proj's contraction
_EPS = 1e-5              # rms_norm_eps
_HALF = _ROPE // 2

#: The identity of the log-sum-exp semiring: a partial with l = 0 and this m
#: contributes nothing to the merge. Not -inf: exp(-inf - -inf) is NaN, and a
#: split that saw no position at all has exactly this m.
_MNEG = -1.0e30

#: Context positions per attention tile, and the cap on how many splits one
#: head's context is cut into. P=32 was measured best in the qwen3.5 example
#: (the tile's global->shared staging is the fixed cost and scales with P).
_ATT_P = 32
_ATT_SPLITS = 8


# --------------------------------------------------------------------------
# Expression helpers. Module-level on purpose: tilelang's builder rewrites the
# *decorated* function's AST, so a plain Python function called from a kernel
# body is ordinary Python building PrimExprs -- no SSA rebinding rules.
# --------------------------------------------------------------------------
def _psum(buf, idx, splits: int):
    """A split-K kernel's `splits` partial rows, summed at flat column `idx`."""
    acc = buf[0, idx]
    for t in range(1, splits):
        acc = acc + buf[t, idx]
    return acc


def _partner(d):
    """The index `rotate_half` pairs `d` with, within the 64-wide rope part."""
    return T.if_then_else(d < _HALF, d + _HALF, d - _HALF)


def _rope64(val, partner, cos, sin, d):
    """HF `rotate_half` RoPE at one index `d` of the 64-wide rope part.

        x1, x2 = x[:32], x[32:];  rotate_half = cat(-x2, x1)
        y[d] = x[d]*cos[d] + rotate_half(x)[d]*sin[d]

    `cos` is 64 wide and already `cat(freqs, freqs)`, so indexing it at `d`
    (as HF does) is the same as indexing at `d % 32`.
    """
    return T.if_then_else(d < _HALF, val * cos - partner * sin, val * cos + partner * sin)


# --------------------------------------------------------------------------
# 1/2. rms_norm(hidden) @ W  ->  (KS, N) f32 partials   (w_q and w_kv_a)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _input_proj(N: int, KS: int, BN: int, BK: int, ST: int):
    """Fused input RMSNorm + one GEMV, split `KS` ways over K.

    The norm is recomputed in every block: 4.6 KB of bf16 reads per block
    against >= 147 KB of weights, and it saves a launch. Roundings match the
    authored IR (and DeepseekV3RMSNorm) exactly: normalise in f32, round to
    bf16, multiply by gamma, round to bf16 again.
    """
    KB = _HID // KS
    KO = KB // BK
    NBN = N // BN
    NB = NBN * KS
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((_HID,), "bfloat16"),
            Gin: T.Tensor((_HID,), "bfloat16"),
            W: T.Tensor((_HID, N), "bfloat16"),
            Y: T.Tensor((KS, N), "float32"),
        ):
            with T.Kernel(NB, threads=TH) as b:
                nb = b % (N // BN)  # N spelled out: the annotation needs the cell
                # `% KS` doubles as the closure reference tilelang needs for
                # the annotation above, and is a no-op: b // (N // BN) < KS.
                ks = (b // (N // BN)) % KS
                tid = T.get_thread_binding()
                kb = _HID // KS

                sq = T.alloc_fragment((_HID,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                rsc = T.alloc_shared((1,), "float32")
                xs = T.alloc_shared((BM, KB), "bfloat16")
                ws = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                osh = T.alloc_shared((BM, BN), "float32")

                for i in T.Parallel(_HID):
                    v = T.cast(X[i], "float32")
                    sq[i] = v * v
                T.reduce_sum(sq, tot, dim=0)
                if tid == 0:
                    rsc[0] = T.rsqrt(tot[0] / T.cast(_HID, "float32") + _EPS)
                T.sync_threads()
                # Only this block's K range is staged. Rows 1..15 of `xs` stay
                # uninitialised, by design (module docstring).
                for j in T.Parallel(KB):
                    i0 = ks * kb + j
                    hi = T.cast(T.cast(X[i0], "float32") * rsc[0], "bfloat16")
                    xs[0, j] = T.cast(T.cast(hi, "float32") * T.cast(Gin[i0], "float32"), "bfloat16")
                T.clear(acc)
                T.sync_threads()

                for ko in T.Pipelined(KO, num_stages=ST):
                    r0 = ks * kb + ko * BK
                    T.copy(W[r0 : r0 + BK, nb * BN : nb * BN + BN], ws)
                    T.gemm(xs[:, ko * BK : ko * BK + BK], ws, acc)
                T.copy(acc, osh)
                for j in T.Parallel(BN):
                    Y[ks, nb * BN + j] = osh[0, j]

        return main

    return build()


# --------------------------------------------------------------------------
# 3. latent rms_norm + @ w_kv_b  ->  (KS, 8192) f32 partials
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _kv_b_kernel(KSA: int, KS: int, nh: int = _NH):
    """`kv = rms_norm(latent) @ w_kv_b`, latent summed from `KSA` partials.

    The latent norm is recomputed per block (512 values against 131 KB of
    weights each). Same roundings as the IR: f32 normalise, bf16, gamma,
    bf16. The rope tail of the compressed vector (columns 512..576) is not
    touched here; the attention kernel reads it from the same partials.
    """
    _NBV = nh * _KVB
    BN, BK, ST = 128, 64, 3
    KB = _LAT // KS
    KO = KB // BK
    NBN = _NBV // BN
    NB = NBN * KS
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            CA: T.Tensor((KSA, _NA), "float32"),
            Gkv: T.Tensor((_LAT,), "bfloat16"),
            W: T.Tensor((_LAT, _NBV), "bfloat16"),
            Y: T.Tensor((KS, _NBV), "float32"),
        ):
            with T.Kernel(NB, threads=TH) as b:
                nb = b % NBN
                ks = (b // NBN) % KS  # no-op; keeps KS in the closure
                tid = T.get_thread_binding()
                kb = _LAT // KS

                lat = T.alloc_shared((_LAT,), "float32")
                sq = T.alloc_fragment((_LAT,), "float32")
                tot = T.alloc_fragment((1,), "float32")
                rsc = T.alloc_shared((1,), "float32")
                xs = T.alloc_shared((BM, KB), "bfloat16")
                ws = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                osh = T.alloc_shared((BM, BN), "float32")

                for i in T.Parallel(_LAT):
                    v = _psum(CA, i, KSA)
                    lat[i] = v
                    sq[i] = v * v
                T.reduce_sum(sq, tot, dim=0)
                if tid == 0:
                    rsc[0] = T.rsqrt(tot[0] / T.cast(_LAT, "float32") + _EPS)
                T.sync_threads()
                for j in T.Parallel(KB):
                    i0 = ks * kb + j
                    hi = T.cast(lat[i0] * rsc[0], "bfloat16")
                    xs[0, j] = T.cast(T.cast(hi, "float32") * T.cast(Gkv[i0], "float32"), "bfloat16")
                T.clear(acc)
                T.sync_threads()

                for ko in T.Pipelined(KO, num_stages=ST):
                    r0 = ks * kb + ko * BK
                    T.copy(W[r0 : r0 + BK, nb * BN : nb * BN + BN], ws)
                    T.gemm(xs[:, ko * BK : ko * BK + BK], ws, acc)
                T.copy(acc, osh)
                for j in T.Parallel(BN):
                    Y[ks, nb * BN + j] = osh[0, j]

        return main

    return build()


# --------------------------------------------------------------------------
# 4. rope + softmax over the cache and this token, per (head, split)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _attn_kernel(C: int, KSA: int, KSB: int, nh: int = _NH):
    """Online softmax over `C` cached positions plus this token, per head.

    One block per (head, context split). MLA here has no GQA sharing -- every
    head has its own expanded k/v -- so the MMA's 16 rows hold one q row and
    15 dead ones; the arithmetic is free against 20 KB of cache per tile.

    This token's own position is its own log-sum-exp partial in slot
    `NSPLIT`: exp(s - s) = 1, so the partial is (m = score, l = 1, o = v_new)
    with no exponential. Split `s == 0` additionally writes k_new / v_new for
    the caller to append; block 0 zeroes the o_proj accumulator.

    The shared 64-wide rope part of k is rotated once per block (recomputed
    from the compressed partials) and the rotated result is what lands in
    k_new for every head -- the "MQA" in kv_a_proj_with_mqa. q's rope part is
    rotated per head and the scale is folded into q once, here, rather than
    into every score.
    """
    _NH = nh
    _NQ = nh * _QK
    _NBV = nh * _KVB
    P = _ATT_P
    NT = max((C + P - 1) // P, 1)  # context tiles the kernel may look at
    TPB = (NT + min(_ATT_SPLITS, NT) - 1) // min(_ATT_SPLITS, NT)
    NSPLIT = (NT + TPB - 1) // TPB  # recomputed from TPB: no empty blocks
    SLOTS = NSPLIT + 1
    CDIM = max(C, 1)  # a 0-length tensor is not a legal kernel argument
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            QP: T.Tensor((2, _NQ), "float32"),
            KVP: T.Tensor((KSB, _NBV), "float32"),
            CA: T.Tensor((KSA, _NA), "float32"),
            Cos: T.Tensor((1, _ROPE), "bfloat16"),
            Sin: T.Tensor((1, _ROPE), "bfloat16"),
            Pos: T.Tensor((1,), "int32"),
            Kc: T.Tensor((CDIM, _NH, _QK), "bfloat16"),
            Vc: T.Tensor((CDIM, _NH, _V), "bfloat16"),
            Scale: T.Tensor((1,), "bfloat16"),
            Op: T.Tensor((SLOTS, _NH, _V), "float32"),
            Mp: T.Tensor((SLOTS, _NH), "float32"),
            Lp: T.Tensor((SLOTS, _NH), "float32"),
            Kn: T.Tensor((_NH, _QK), "bfloat16"),
            Vn: T.Tensor((_NH, _V), "bfloat16"),
            Oacc: T.Tensor((_HID,), "float32"),
        ):
            with T.Kernel(_NH * NSPLIT, threads=TH) as b:
                h = b % _NH
                s = b // _NH
                tid = T.get_thread_binding()
                pos = Pos[0]
                nctx = C  # a constant: every guard below folds

                qraw = T.alloc_shared((_QK,), "float32")
                krr = T.alloc_shared((_ROPE,), "float32")
                knew = T.alloc_shared((_QK,), "float32")
                vnew = T.alloc_shared((_V,), "float32")
                css = T.alloc_shared((_ROPE,), "float32")
                sss = T.alloc_shared((_ROPE,), "float32")
                qsh = T.alloc_shared((_QK,), "float32")
                qs = T.alloc_shared((BM, _QK), "bfloat16")
                kf = T.alloc_shared((P, _QK), "bfloat16")
                vf = T.alloc_shared((P, _V), "bfloat16")
                psh = T.alloc_shared((BM, P), "bfloat16")
                accs = T.alloc_fragment((BM, P), "float32")
                ssh = T.alloc_shared((BM, P), "float32")
                acco = T.alloc_fragment((BM, _V), "float32")
                osh = T.alloc_shared((BM, _V), "float32")
                scf = T.alloc_fragment((P,), "float32")
                red = T.alloc_fragment((1,), "float32")
                msh = T.alloc_shared((1,), "float32")
                lsh = T.alloc_shared((1,), "float32")
                nsh = T.alloc_shared((1,), "float32")
                csh = T.alloc_shared((1,), "float32")

                # ---- stage q, the shared k rope part, this token's k/v ----
                for d in T.Parallel(_ROPE):
                    css[d] = T.cast(Cos[pos, d], "float32")
                    sss[d] = T.cast(Sin[pos, d], "float32")
                for d in T.Parallel(_QK):
                    qraw[d] = _psum(QP, h * _QK + d, 2)
                for d in T.Parallel(_ROPE):
                    krr[d] = _psum(CA, _LAT + d, KSA)
                for d in T.Parallel(_NOPE):
                    knew[d] = _psum(KVP, h * _KVB + d, KSB)
                for d in T.Parallel(_V):
                    vnew[d] = _psum(KVP, h * _KVB + _NOPE + d, KSB)
                T.sync_threads()

                sc = T.cast(Scale[0], "float32")
                for d in T.Parallel(_NOPE):
                    qsh[d] = qraw[d] * sc
                for d in T.Parallel(_ROPE):
                    qsh[_NOPE + d] = (
                        _rope64(qraw[_NOPE + d], qraw[_NOPE + _partner(d)],
                                css[d], sss[d], d) * sc
                    )
                    knew[_NOPE + d] = _rope64(krr[d], krr[_partner(d)],
                                              css[d], sss[d], d)
                for d in T.Parallel(_QK):
                    qs[0, d] = T.cast(qsh[d], "bfloat16")
                if tid == 0:
                    msh[0] = _MNEG
                    lsh[0] = 0.0
                T.clear(acco)
                T.sync_threads()

                if s == 0:
                    # This token's k/v, for the caller to append.
                    for d in T.Parallel(_QK):
                        Kn[h, d] = T.cast(knew[d], "bfloat16")
                    for d in T.Parallel(_V):
                        Vn[h, d] = T.cast(vnew[d], "bfloat16")
                    T.sync_threads()
                    # Its score against itself, from the same bf16-rounded
                    # operands the cache scores use. Serial on one thread: a
                    # fragment reduce inside this runtime conditional fails
                    # tilelang's layout inference ("no available layout
                    # found"; hoisting it to unconditional scope compiles but
                    # puts the reduce in every block and measured +2.3 us on
                    # the launch), and 192 MACs once per head is nothing.
                    if tid == 0:
                        sacc = T.alloc_var("float32", init=0.0)
                        for d in T.serial(_QK):
                            sacc = sacc + T.cast(T.cast(qsh[d], "bfloat16"), "float32") * T.cast(
                                T.cast(knew[d], "bfloat16"), "float32")
                        Mp[SLOTS - 1, h] = sacc
                        Lp[SLOTS - 1, h] = 1.0
                    for d in T.Parallel(_V):
                        Op[SLOTS - 1, h, d] = vnew[d]
                if b == 0:
                    # kernel 5 accumulates o_proj's K-splits with atomics, so
                    # somebody has to zero the accumulator. 2304 stores in one
                    # block, against a launch.
                    for i in T.Parallel(_HID):
                        Oacc[i] = 0.0

                # ---- the cached positions --------------------------------
                for it in T.serial(TPB):
                    base = (s * TPB + it) * P
                    if base < nctx:
                        if base + P <= nctx:
                            # Whole tile in range: one strided copy each.
                            T.copy(Kc[base : base + P, h, :], kf)
                            T.copy(Vc[base : base + P, h, :], vf)
                        else:
                            # Only the tail tile takes the scalar path. Slots
                            # past nctx are zeroed so a stale slot cannot reach
                            # the PV product.
                            for pp, d in T.Parallel(P, _QK):
                                t = base + pp
                                tc = T.min(t, CDIM - 1)
                                kf[pp, d] = T.if_then_else(
                                    t < nctx, Kc[tc, h, d], T.cast(0.0, "bfloat16"))
                            for pp, d in T.Parallel(P, _V):
                                t = base + pp
                                tc = T.min(t, CDIM - 1)
                                vf[pp, d] = T.if_then_else(
                                    t < nctx, Vc[tc, h, d], T.cast(0.0, "bfloat16"))
                        T.sync_threads()

                        T.clear(accs)
                        T.gemm(qs, kf, accs, transpose_B=True)
                        T.copy(accs, ssh)
                        for pp in T.Parallel(P):
                            t = base + pp
                            scf[pp] = T.if_then_else(
                                t < nctx, ssh[0, pp], T.float32(_MNEG))
                        T.reduce_max(scf, red, dim=0)
                        T.copy(red, nsh)
                        T.sync_threads()
                        if tid == 0:
                            nm = T.max(nsh[0], msh[0])
                            csh[0] = T.exp(msh[0] - nm)
                            msh[0] = nm
                        T.sync_threads()
                        for pp in T.Parallel(P):
                            e = T.exp(scf[pp] - msh[0])
                            psh[0, pp] = T.cast(e, "bfloat16")
                            scf[pp] = e
                        T.reduce_sum(scf, red, dim=0)
                        T.copy(red, nsh)
                        T.sync_threads()
                        if tid == 0:
                            lsh[0] = lsh[0] * csh[0] + nsh[0]
                        for j, d in T.Parallel(BM, _V):
                            acco[j, d] = acco[j, d] * csh[0]
                        T.sync_threads()
                        T.gemm(psh, vf, acco)

                T.copy(acco, osh)
                for d in T.Parallel(_V):
                    Op[s, h, d] = osh[0, d]
                if tid == 0:
                    Mp[s, h] = msh[0]
                    Lp[s, h] = lsh[0]

        return main

    return build(), NSPLIT, SLOTS


# --------------------------------------------------------------------------
# 4b. the same kernel, with the live cache length a device value
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _attn_kernel_cap(CAP: int, KSA: int, KSB: int, nh: int = _NH):
    """`_attn_kernel` over a fixed-capacity cache, live length in `Pos`.

    The contract kernel compiles `C` into the binary, so a decode run through
    it would pay one tilelang compilation per token. Here the cache is a
    fixed `CAP`-row buffer (a bucket, see `_bucket`) and the live length rides
    in `Pos`: at a decode step the position IS the prior length, and NoPE
    makes the rope tables constant, so one tensor carries both. Loads past
    the live length are predicated to zero and a dead slot's score is forced
    to `_MNEG` -- the masking the contract kernel's tail tile does, at every
    tile -- so dead positions contribute exactly nothing to the softmax.

    Two costs, both bounded by the bucket granularity. The whole-tile fast
    path (`T.copy`) is gone: its guard would be a runtime branch around a
    GEMM, which tilelang's layout inference rejects (the KDA line measured
    the same for a reduce). And every block walks all of CAP, live or not,
    since the grid is static. `Cos`/`Sin` are `(CAP, _ROPE)` so `pos` is
    always in bounds; the caller guarantees `pos < CAP` by switching buckets.
    """
    _NH = nh
    _NQ = nh * _QK
    _NBV = nh * _KVB
    P = _ATT_P
    NT = max(CAP // P, 1)
    TPB = (NT + min(_ATT_SPLITS, NT) - 1) // min(_ATT_SPLITS, NT)
    NSPLIT = (NT + TPB - 1) // TPB
    SLOTS = NSPLIT + 1
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            QP: T.Tensor((2, _NQ), "float32"),
            KVP: T.Tensor((KSB, _NBV), "float32"),
            CA: T.Tensor((KSA, _NA), "float32"),
            Cos: T.Tensor((CAP, _ROPE), "bfloat16"),
            Sin: T.Tensor((CAP, _ROPE), "bfloat16"),
            Pos: T.Tensor((1,), "int32"),
            Kc: T.Tensor((CAP, _NH, _QK), "bfloat16"),
            Vc: T.Tensor((CAP, _NH, _V), "bfloat16"),
            Scale: T.Tensor((1,), "bfloat16"),
            Op: T.Tensor((SLOTS, _NH, _V), "float32"),
            Mp: T.Tensor((SLOTS, _NH), "float32"),
            Lp: T.Tensor((SLOTS, _NH), "float32"),
            Kn: T.Tensor((_NH, _QK), "bfloat16"),
            Vn: T.Tensor((_NH, _V), "bfloat16"),
            Oacc: T.Tensor((_HID,), "float32"),
        ):
            with T.Kernel(_NH * NSPLIT, threads=TH) as b:
                h = b % _NH
                s = b // _NH
                tid = T.get_thread_binding()
                pos = Pos[0]
                nctx = pos  # decode: the position IS the prior length

                qraw = T.alloc_shared((_QK,), "float32")
                krr = T.alloc_shared((_ROPE,), "float32")
                knew = T.alloc_shared((_QK,), "float32")
                vnew = T.alloc_shared((_V,), "float32")
                css = T.alloc_shared((_ROPE,), "float32")
                sss = T.alloc_shared((_ROPE,), "float32")
                qsh = T.alloc_shared((_QK,), "float32")
                qs = T.alloc_shared((BM, _QK), "bfloat16")
                kf = T.alloc_shared((P, _QK), "bfloat16")
                vf = T.alloc_shared((P, _V), "bfloat16")
                psh = T.alloc_shared((BM, P), "bfloat16")
                accs = T.alloc_fragment((BM, P), "float32")
                ssh = T.alloc_shared((BM, P), "float32")
                acco = T.alloc_fragment((BM, _V), "float32")
                osh = T.alloc_shared((BM, _V), "float32")
                scf = T.alloc_fragment((P,), "float32")
                red = T.alloc_fragment((1,), "float32")
                msh = T.alloc_shared((1,), "float32")
                lsh = T.alloc_shared((1,), "float32")
                nsh = T.alloc_shared((1,), "float32")
                csh = T.alloc_shared((1,), "float32")

                # ---- stage q, the shared k rope part, this token's k/v ----
                for d in T.Parallel(_ROPE):
                    css[d] = T.cast(Cos[pos, d], "float32")
                    sss[d] = T.cast(Sin[pos, d], "float32")
                for d in T.Parallel(_QK):
                    qraw[d] = _psum(QP, h * _QK + d, 2)
                for d in T.Parallel(_ROPE):
                    krr[d] = _psum(CA, _LAT + d, KSA)
                for d in T.Parallel(_NOPE):
                    knew[d] = _psum(KVP, h * _KVB + d, KSB)
                for d in T.Parallel(_V):
                    vnew[d] = _psum(KVP, h * _KVB + _NOPE + d, KSB)
                T.sync_threads()

                sc = T.cast(Scale[0], "float32")
                for d in T.Parallel(_NOPE):
                    qsh[d] = qraw[d] * sc
                for d in T.Parallel(_ROPE):
                    qsh[_NOPE + d] = (
                        _rope64(qraw[_NOPE + d], qraw[_NOPE + _partner(d)],
                                css[d], sss[d], d) * sc
                    )
                    knew[_NOPE + d] = _rope64(krr[d], krr[_partner(d)],
                                              css[d], sss[d], d)
                for d in T.Parallel(_QK):
                    qs[0, d] = T.cast(qsh[d], "bfloat16")
                if tid == 0:
                    msh[0] = _MNEG
                    lsh[0] = 0.0
                T.clear(acco)
                T.sync_threads()

                if s == 0:
                    for d in T.Parallel(_QK):
                        Kn[h, d] = T.cast(knew[d], "bfloat16")
                    for d in T.Parallel(_V):
                        Vn[h, d] = T.cast(vnew[d], "bfloat16")
                    T.sync_threads()
                    if tid == 0:
                        sacc = T.alloc_var("float32", init=0.0)
                        for d in T.serial(_QK):
                            sacc = sacc + T.cast(T.cast(qsh[d], "bfloat16"), "float32") * T.cast(
                                T.cast(knew[d], "bfloat16"), "float32")
                        Mp[SLOTS - 1, h] = sacc
                        Lp[SLOTS - 1, h] = 1.0
                    for d in T.Parallel(_V):
                        Op[SLOTS - 1, h, d] = vnew[d]
                if b == 0:
                    for i in T.Parallel(_HID):
                        Oacc[i] = 0.0

                # ---- the cached positions, live length from the device ----
                for it in T.serial(TPB):
                    base = (s * TPB + it) * P
                    # Predicated scalar loads everywhere: a slot past the live
                    # length lands as zero and its score is forced to _MNEG
                    # below, so it contributes exp(-1e30 - m) * 0 = 0.
                    for pp, d in T.Parallel(P, _QK):
                        t = base + pp
                        tc = T.min(t, CAP - 1)
                        kf[pp, d] = T.if_then_else(
                            t < nctx, Kc[tc, h, d], T.cast(0.0, "bfloat16"))
                    for pp, d in T.Parallel(P, _V):
                        t = base + pp
                        tc = T.min(t, CAP - 1)
                        vf[pp, d] = T.if_then_else(
                            t < nctx, Vc[tc, h, d], T.cast(0.0, "bfloat16"))
                    T.sync_threads()

                    T.clear(accs)
                    T.gemm(qs, kf, accs, transpose_B=True)
                    T.copy(accs, ssh)
                    for pp in T.Parallel(P):
                        t = base + pp
                        scf[pp] = T.if_then_else(
                            t < nctx, ssh[0, pp], T.float32(_MNEG))
                    T.reduce_max(scf, red, dim=0)
                    T.copy(red, nsh)
                    T.sync_threads()
                    if tid == 0:
                        nm = T.max(nsh[0], msh[0])
                        csh[0] = T.exp(msh[0] - nm)
                        msh[0] = nm
                    T.sync_threads()
                    for pp in T.Parallel(P):
                        e = T.exp(scf[pp] - msh[0])
                        psh[0, pp] = T.cast(e, "bfloat16")
                        scf[pp] = e
                    T.reduce_sum(scf, red, dim=0)
                    T.copy(red, nsh)
                    T.sync_threads()
                    if tid == 0:
                        lsh[0] = lsh[0] * csh[0] + nsh[0]
                    for j, d in T.Parallel(BM, _V):
                        acco[j, d] = acco[j, d] * csh[0]
                    T.sync_threads()
                    T.gemm(psh, vf, acco)

                T.copy(acco, osh)
                for d in T.Parallel(_V):
                    Op[s, h, d] = osh[0, d]
                if tid == 0:
                    Mp[s, h] = msh[0]
                    Lp[s, h] = lsh[0]

        return main

    return build(), NSPLIT, SLOTS


# --------------------------------------------------------------------------
# 5. log-sum-exp merge + o_proj
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _out_kernel(SLOTS: int, nh: int = _NH):
    """`out = merge(partials) @ w_o`, K-split with atomic accumulation.

    o_proj contracts 4096, which at BN=128 is 18 column tiles -- a seventh of
    the machine -- so K is split 8 ways and the 144 blocks add into the
    output with `T.atomic_add` (kernel 4's block 0 zeroed it). Summation
    order is then not reproducible bit-for-bit; at 8 f32 addends that is
    ~1e-7 and buys the parallelism.

    Each K-split owns 512 of the 4096, exactly four query heads, so a block
    merges only the heads it contracts.
    """
    _NH = nh
    _OI = nh * _V
    KS, BN, BK, ST = 8, 128, 64, 3
    KB = _OI // KS       # 512
    KO = KB // BK
    HPB = KB // _V       # query heads per K-split
    NBO = _HID // BN     # 18
    NB = NBO * KS
    TH = 128

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Op: T.Tensor((SLOTS, _NH, _V), "float32"),
            Mp: T.Tensor((SLOTS, _NH), "float32"),
            Lp: T.Tensor((SLOTS, _NH), "float32"),
            Wo: T.Tensor((_OI, _HID), "bfloat16"),
            Out: T.Tensor((_HID,), "float32"),
        ):
            with T.Kernel(NB, threads=TH) as b:
                nb = b % NBO
                ks = (b // NBO) % KS  # no-op; keeps KS in the closure
                h0 = ks * HPB
                tid = T.get_thread_binding()

                wgt = T.alloc_shared((SLOTS, HPB), "float32")
                lsh = T.alloc_shared((HPB,), "float32")
                num = T.alloc_fragment((HPB, _V), "float32")
                xs = T.alloc_shared((BM, KB), "bfloat16")
                ws = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                osh = T.alloc_shared((BM, BN), "float32")

                # Log-sum-exp merge of the splits against their joint max. A
                # split that saw no position has l = 0 and m = -1e30, so it
                # contributes exp(-1e30 - M) * 0 = 0 and never a NaN.
                if tid < HPB:
                    mx = T.alloc_var("float32", init=_MNEG)
                    for t in T.serial(SLOTS):
                        mx = T.max(mx, Mp[t, h0 + tid])
                    den = T.alloc_var("float32", init=0.0)
                    for t in T.serial(SLOTS):
                        w = T.exp(Mp[t, h0 + tid] - mx)
                        wgt[t, tid] = w
                        den = den + Lp[t, h0 + tid] * w
                    lsh[tid] = den
                T.clear(num)
                T.sync_threads()
                # Serial over the slots *inside* the parallel loop: each
                # thread then has SLOTS independent loads in flight.
                for j, d in T.Parallel(HPB, _V):
                    for t in T.serial(SLOTS):
                        num[j, d] += Op[t, h0 + j, d] * wgt[t, j]

                for j, d in T.Parallel(HPB, _V):
                    xs[0, j * _V + d] = T.cast(num[j, d] / lsh[j], "bfloat16")
                T.clear(acc)
                T.sync_threads()

                for ko in T.Pipelined(KO, num_stages=ST):
                    r0 = ks * KB + ko * BK
                    T.copy(Wo[r0 : r0 + BK, nb * BN : nb * BN + BN], ws)
                    T.gemm(xs[:, ko * BK : ko * BK + BK], ws, acc)
                T.copy(acc, osh)
                for j in T.Parallel(BN):
                    T.atomic_add(Out[nb * BN + j], osh[0, j])

        return main

    return build()


# --------------------------------------------------------------------------
# the fused entry point
# --------------------------------------------------------------------------

#: K-splits of the w_kv_a projection (its partial-row count). Measured on
#: H200: 6 splits (54 blocks) 6.4 us against 8.5 us at 12 splits.
_KSA = 6

#: K-splits of the w_kv_b projection.
_KSB = 2

#: Intermediate buffers reused between calls, keyed by (device, ctx_len).
#: Only true workspaces are cached -- never the returned tensors, which the
#: caller owns (and appends from).
_WS: dict = {}


def _workspaces(dev, C: int, slots: int, nh: int):
    key = (str(dev), C, nh)
    ws = _WS.get(key)
    if ws is None:
        f32 = torch.float32
        ws = dict(
            qp=torch.empty(2, nh * _QK, dtype=f32, device=dev),
            ca=torch.empty(_KSA, _NA, dtype=f32, device=dev),
            kvp=torch.empty(_KSB, nh * _KVB, dtype=f32, device=dev),
            op=torch.empty(slots, nh, _V, dtype=f32, device=dev),
            mp=torch.empty(slots, nh, dtype=f32, device=dev),
            lp=torch.empty(slots, nh, dtype=f32, device=dev),
            out=torch.empty(_HID, dtype=f32, device=dev),
        )
        _WS[key] = ws
    return ws


def mla_attention(hidden, gamma_in, w_q, w_kv_a, gamma_kv_a, w_kv_b,
                  cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o):
    """The reference contract: cache length is a shape; caller appends k/v.

    Arguments are exactly ``mla_attention``'s in the authored IR (same order,
    same dtypes, bf16 throughout). Returns
    ``(out (1,1,2304), k_new (1,1,32,192), v_new (1,1,32,128))``, bf16.
    """
    assert hidden.dtype is torch.bfloat16, f"expected bf16, got {hidden.dtype}"
    dev = hidden.device
    C = int(k_cache.shape[1])  # a host-side shape, not a device value
    nh = w_q.shape[-1] // _QK  # 32, or 16 for one TP2 rank

    x = hidden.reshape(_HID)
    gin = gamma_in.reshape(_HID)

    kern, _, slots = _attn_kernel(C, _KSA, _KSB, nh)
    ws = _workspaces(dev, C, slots, nh)

    # 1. q = rms_norm(hidden) @ w_q, 2 K-splits.
    _input_proj(nh * _QK, 2, 128, 64, 3)(x, gin, w_q.reshape(_HID, nh * _QK), ws["qp"])

    # 2. compressed = rms_norm(hidden) @ w_kv_a.
    _input_proj(_NA, _KSA, 64, 64, 3)(x, gin, w_kv_a.reshape(_HID, _NA), ws["ca"])

    # 3. kv = rms_norm(latent) @ w_kv_b.
    _kv_b_kernel(_KSA, _KSB, nh)(ws["ca"], gamma_kv_a.reshape(_LAT),
                                 w_kv_b.reshape(_LAT, nh * _KVB), ws["kvp"])

    # 4. attention, returning this token's k/v for the caller to append.
    k_new = torch.empty(nh, _QK, dtype=torch.bfloat16, device=dev)
    v_new = torch.empty(nh, _V, dtype=torch.bfloat16, device=dev)
    if C == 0:
        kc = torch.zeros(1, nh, _QK, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros(1, nh, _V, dtype=torch.bfloat16, device=dev)
    else:
        kc = k_cache.view(C, nh, _QK)
        vc = v_cache.view(C, nh, _V)
    kern(ws["qp"], ws["kvp"], ws["ca"],
         cos_cache.reshape(1, _ROPE), sin_cache.reshape(1, _ROPE),
         pos_ids.reshape(1), kc, vc, scale.reshape(1),
         ws["op"], ws["mp"], ws["lp"], k_new, v_new, ws["out"])

    # 5. log-sum-exp merge + o_proj.
    _out_kernel(slots, nh)(ws["op"], ws["mp"], ws["lp"],
                           w_o.reshape(nh * _V, _HID), ws["out"])

    return (ws["out"].to(torch.bfloat16).view(1, 1, _HID),
            k_new.view(1, 1, nh, _QK),
            v_new.view(1, 1, nh, _V))


#: Bucket sizes the capacity kernel is compiled at. Small contexts waste
#: little at 128; past 2K the cache reads dominate and a coarser bucket costs
#: nothing measurable. The distinct buckets a run crosses are the number of
#: compilations it pays, once (tilelang caches them on disk).
def bucket(n: int) -> int:
    """The capacity an attention kernel is compiled for: *n* rounded up."""
    if n <= 2048:
        return max(128, ((n + 127) // 128) * 128)
    return ((n + 1023) // 1024) * 1024


def mla_attention_cap(hidden, gamma_in, w_q, w_kv_a, gamma_kv_a, w_kv_b,
                      cos_cache, sin_cache, pos_ids, k_buf, v_buf, scale, w_o,
                      keep_f32: bool = False):
    """`mla_attention` over fixed-capacity buffers, for the decode driver.

    Same arguments and returns as `mla_attention`, except `k_buf` / `v_buf`
    are `(1, CAP, ...)` -- a bucket-capacity prefix of the caller's persistent
    cache, contiguous by construction -- and `pos_ids` carries the position,
    which at a decode step is also the live cache length (see
    `_attn_kernel_cap`). `cos_cache` / `sin_cache` are `(CAP, _ROPE)`. The
    caller writes the returned k/v into slot `pos` itself.

    The head count comes from `w_q`: 32 for the full model, 16 for one TP2
    rank. With `keep_f32` the output is returned un-rounded -- the o_proj
    accumulator itself -- for a TP rank to all-reduce in f32 before the one
    bf16 landing; it aliases the workspace, so the caller must consume it
    before the next call.
    """
    assert hidden.dtype is torch.bfloat16, f"expected bf16, got {hidden.dtype}"
    dev = hidden.device
    CAP = int(k_buf.shape[1])
    assert CAP % _ATT_P == 0, f"CAP {CAP} must be a multiple of {_ATT_P}"
    nh = w_q.shape[-1] // _QK  # 32, or 16 for one TP2 rank

    x = hidden.reshape(_HID)
    gin = gamma_in.reshape(_HID)

    kern, _, slots = _attn_kernel_cap(CAP, _KSA, _KSB, nh)
    ws = _workspaces(dev, CAP, slots, nh)

    _input_proj(nh * _QK, 2, 128, 64, 3)(x, gin, w_q.reshape(_HID, nh * _QK), ws["qp"])
    _input_proj(_NA, _KSA, 64, 64, 3)(x, gin, w_kv_a.reshape(_HID, _NA), ws["ca"])
    _kv_b_kernel(_KSA, _KSB, nh)(ws["ca"], gamma_kv_a.reshape(_LAT),
                                 w_kv_b.reshape(_LAT, nh * _KVB), ws["kvp"])

    k_new = torch.empty(nh, _QK, dtype=torch.bfloat16, device=dev)
    v_new = torch.empty(nh, _V, dtype=torch.bfloat16, device=dev)
    kern(ws["qp"], ws["kvp"], ws["ca"],
         cos_cache.reshape(CAP, _ROPE), sin_cache.reshape(CAP, _ROPE),
         pos_ids.reshape(1), k_buf.view(CAP, nh, _QK), v_buf.view(CAP, nh, _V),
         scale.reshape(1),
         ws["op"], ws["mp"], ws["lp"], k_new, v_new, ws["out"])

    _out_kernel(slots, nh)(ws["op"], ws["mp"], ws["lp"],
                           w_o.reshape(nh * _V, _HID), ws["out"])

    out = ws["out"].view(1, 1, _HID)
    return (out if keep_f32 else out.to(torch.bfloat16),
            k_new.view(1, 1, nh, _QK),
            v_new.view(1, 1, nh, _V))


__all__ = ["bucket", "mla_attention", "mla_attention_cap"]
