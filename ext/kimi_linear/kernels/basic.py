"""The shared basic ops of the Kimi-Linear decode shell: embed, RMSNorm,
residual add, and the lm_head GEMV.

-- Section owned by the MoE/basic kernel work. `kda.py` and `attn.py` are
   self-contained by design; if another section lands in this file, keep the
   banner separation. --

Every kernel is decode-shaped: sequence length 1, so every tensor is a vector
of `H` (or `VOCAB`) and every matmul is a GEMV. Roundings match the authored
IR in `ext/kimi_linear/model.py` exactly -- in particular the RMSNorm rounds
the normalised value to bf16 *before* the learned scale multiplies
(`KimiRMSNorm` semantics), which a plain `x * rsqrt * gamma` in f32 does not
do; `kda.py` measured the same placement bit-exact against the checkpoint.

The GEMV is the template's (`examples/qwen3_5_35b_a3b-tilelang/kernels/
basic.py`): the vector rides row 0 of a 16-row MMA tile with rows 1..15 left
uninitialised (an MMA computes each output row from its own input row, so the
garbage never reaches the output), because `T.gemm` is the only path that gets
the Hopper pipeline. Here the input is genuinely bf16 -- the authored lm_head
consumes the bf16 hidden state -- so no hi/lo residual row is needed.
"""
from __future__ import annotations

import functools

import tilelang
import tilelang.language as T
import torch

H = 2304  #: hidden_size
VOCAB = 163840  #: vocab_size
EPS = 1e-5  #: rms_norm_eps

BM = 16  #: rows of the MMA tile; row 0 carries the vector, 1..15 are dead

#: How much shared memory one SM may be given, minus a margin for the
#: pipeline's own bookkeeping (the template measured the limit at ~227 KB).
_SMEM_BUDGET = 220 * 1024


@functools.lru_cache(maxsize=None)
def _rms_norm():
    """`y = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * gamma)`, one block, f32 sum.

    The double rounding is deliberate: the authored shells (every fused
    post-norm, and the root's `final_rms_norm`) normalise in f32, round to
    bf16, and only then multiply by the bf16 gamma.
    """
    threads = 256

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((H,), "bfloat16"),
            G: T.Tensor((H,), "bfloat16"),
            Y: T.Tensor((H,), "bfloat16"),
        ):
            with T.Kernel(1, threads=threads) as _:
                xf = T.alloc_fragment((H,), "float32")
                sq = T.alloc_fragment((H,), "float32")
                total = T.alloc_fragment((1,), "float32")
                scale = T.alloc_shared((1,), "float32")
                for i in T.Parallel(H):
                    xf[i] = T.cast(X[i], "float32")
                for i in T.Parallel(H):
                    sq[i] = xf[i] * xf[i]
                T.reduce_sum(sq, total, dim=0)
                if T.get_thread_binding() == 0:
                    scale[0] = T.rsqrt(total[0] / T.cast(H, "float32") + EPS)
                T.sync_threads()
                for i in T.Parallel(H):
                    hn = T.cast(xf[i] * scale[0], "bfloat16")
                    Y[i] = T.cast(
                        T.cast(hn, "float32") * T.cast(G[i], "float32"), "bfloat16"
                    )

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _residual_add():
    """`c = bf16(f32(a) + f32(b))`: one exact sum, one rounding -- the same

    value a native bf16 add produces.
    """
    threads = 256
    per = (H + threads - 1) // threads

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            A: T.Tensor((H,), "bfloat16"),
            B: T.Tensor((H,), "bfloat16"),
            C: T.Tensor((H,), "bfloat16"),
        ):
            with T.Kernel(1, threads=threads) as _:
                for p in T.serial(per):
                    idx = p * threads + T.get_thread_binding()
                    if idx < H:
                        C[idx] = T.cast(
                            T.cast(A[idx], "float32") + T.cast(B[idx], "float32"),
                            "bfloat16",
                        )

        return main

    return build()


@functools.lru_cache(maxsize=None)
def _embed():
    """One row of the embedding table, by a token id held on the device.

    The id is a device tensor rather than a Python int so a decode step never
    has to bring the sampled token back to the host.
    """
    threads = 256
    per = (H + threads - 1) // threads

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            Table: T.Tensor((VOCAB, H), "bfloat16"),
            Ids: T.Tensor((1,), "int64"),
            Y: T.Tensor((H,), "bfloat16"),
        ):
            with T.Kernel(1, threads=threads) as _:
                for p in T.serial(per):
                    idx = p * threads + T.get_thread_binding()
                    # The `< VOCAB` half is a bounds check and also the reason
                    # `VOCAB` resolves at all: tilelang evaluates the
                    # annotations against the kernel's closure cells, and a
                    # name the body never references gets no cell (the
                    # template's basic.py records the same trap).
                    if idx < H and Ids[0] < VOCAB:
                        Y[idx] = Table[Ids[0], idx]

        return main

    return build()


def _gemv_config(K: int, N: int) -> tuple[int, int, int, int]:
    """(BN, BK, threads, stages) for one GEMV shape: enough blocks to fill 132

    SMs, and a shared-memory footprint that fits with the vector resident.
    """
    threads = 128
    BK = 128
    vector_bytes = BM * K * 2
    for BN in (128, 64):
        if N % BN:
            continue
        for stages in (4, 3, 2):
            if vector_bytes + stages * BK * BN * 2 + BM * BN * 4 <= _SMEM_BUDGET:
                return BN, BK, threads, stages
    return (64 if N % 128 else 128), 64, threads, 2


@functools.lru_cache(maxsize=None)
def _lm_head():
    """`y[VOCAB] = x[H] @ W[H, VOCAB]`, bf16 x and W, f32 out.

    163840 output channels = 1280 blocks of 128, the one GEMV in the shell that
    fills the machine on its own (756 MB of weight read per call).
    """
    K, N = H, VOCAB
    BN, BK, threads, stages = _gemv_config(K, N)
    KO = K // BK

    @tilelang.jit
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((K,), "bfloat16"),
            W: T.Tensor((K, N), "bfloat16"),
            Y: T.Tensor((N,), "float32"),
        ):
            with T.Kernel(N // BN, threads=threads) as bn:
                xs = T.alloc_shared((BM, K), "bfloat16")
                ws = T.alloc_shared((BK, BN), "bfloat16")
                acc = T.alloc_fragment((BM, BN), "float32")
                out = T.alloc_shared((BM, BN), "float32")
                # Row 0 only; rows 1..15 stay uninitialised -- see the module
                # docstring. K is a multiple of BK at the published config.
                for j in T.Parallel(K):
                    xs[0, j] = X[j]
                T.clear(acc)
                T.sync_threads()
                for ko in T.Pipelined(KO, num_stages=stages):
                    T.copy(W[ko * BK:(ko + 1) * BK, bn * BN:(bn + 1) * BN], ws)
                    T.gemm(xs[:, ko * BK:(ko + 1) * BK], ws, acc)
                T.copy(acc, out)
                for j in T.Parallel(BN):
                    Y[bn * BN + j] = out[0, j]

        return main

    return build()


# ------------------------------------------------------------- entry points --


def rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """The authored `KimiRMSNorm` on the last axis of a `[..., H]` bf16 tensor."""
    flat = x.reshape(-1)
    out = torch.empty_like(flat)
    _rms_norm()(flat, gamma.reshape(-1), out)
    return out.view_as(x)


def residual_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a + b` at the authored rounding, last-axis `H` bf16 tensors."""
    fa, fb = a.reshape(-1), b.reshape(-1)
    out = torch.empty_like(fa)
    _residual_add()(fa, fb, out)
    return out.view_as(a)


def embed(table: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """`table[token_ids]` as `[1, 1, H]` bf16; *token_ids* is a device `(1,)`."""
    out = torch.empty(H, device=table.device, dtype=torch.bfloat16)
    _embed()(table, token_ids.reshape(1), out)
    return out.view(1, 1, H)


def lm_head(hidden: torch.Tensor, w_head: torch.Tensor) -> torch.Tensor:
    """`hidden @ w_head` as `[1, VOCAB]` bf16; *w_head* is the authored `[H, V]`."""
    K, N = w_head.shape
    out = torch.empty(N, device=hidden.device, dtype=torch.float32)
    _lm_head()(hidden.reshape(K), w_head, out)
    return out.view(1, N).to(torch.bfloat16)


__all__ = ["embed", "lm_head", "residual_add", "rms_norm"]
