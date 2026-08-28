"""TileLang grouped GEMM implementation for the S>1 Kimi MoE prefill."""
from __future__ import annotations
import functools
import torch
import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey

TOP_K = 8
BM = 128
BN_I = 128
BN_H = 128
BK = 128
THREADS = 256
_NO_WS = {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

@functools.lru_cache(maxsize=None)
def _kernel(H, I, E, total_blocks, xdt, wdt):
    @tilelang.jit(pass_configs=_NO_WS)
    def build():
        # TileLang resolves postponed annotations from function globals.
        globals().update(H=H, I=I, E=E, total_blocks=total_blocks, xdt=xdt, wdt=wdt)
        @T.prim_func
        def main(X: T.Tensor((total_blocks * BM, H), "bfloat16"),
                 WG: T.Tensor((E, I, H), "bfloat16"), WU: T.Tensor((E, I, H), "bfloat16"),
                 WD: T.Tensor((E, H, I), "bfloat16"),
                 group_sizes: T.Tensor((E,), T.int32),
                 group_offsets: T.Tensor((E,), T.int32),
                 padded_offsets: T.Tensor((E,), T.int32),
                 group_for_block: T.Tensor((total_blocks,), T.int32),
                 weights: T.Tensor((total_blocks * BM,), "float32"),
                 HBuf: T.Tensor((total_blocks * BM, I), "float32"),
                 Out: T.Tensor((total_blocks * BM, H), "float32")):
            with T.Kernel(total_blocks, T.ceildiv(I, BN_I), threads=THREADS) as (bx, by):
                e = group_for_block[bx]
                p0 = bx * BM
                start = p0 - padded_offsets[e] + group_offsets[e]
                size = group_sizes[e]
                rows = T.max(0, T.min(BM, size - (p0 - padded_offsets[e])))
                xs = T.alloc_shared((BM, BK), "bfloat16")
                gh = T.alloc_shared((BK, BN_I), "bfloat16")
                uh = T.alloc_shared((BK, BN_I), "bfloat16")
                ga = T.alloc_fragment((BM, BN_I), "float32")
                ua = T.alloc_fragment((BM, BN_I), "float32")
                ho = T.alloc_shared((BM, BN_I), "float32")
                T.clear(ga); T.clear(ua)
                for k in T.Pipelined(T.ceildiv(H, BK), num_stages=2):
                    T.copy(X[p0, k * BK], xs)
                    T.copy(WG[e, by * BN_I, k * BK], gh)
                    T.copy(WU[e, by * BN_I, k * BK], uh)
                    T.gemm(xs, gh, ga, transpose_B=True)
                    T.gemm(xs, uh, ua, transpose_B=True)
                T.copy(ga, ho)
                for i, j in T.Parallel(BM, BN_I):
                    if i < rows and by * BN_I + j < I:
                        g = ho[i, j]
                        u = ua[i, j]
                        HBuf[p0 + i, by * BN_I + j] = g / (1.0 + T.exp(-g)) * u

            with T.Kernel(total_blocks, T.ceildiv(H, BN_H), threads=THREADS) as (bx, by):
                e = group_for_block[bx]
                p0 = bx * BM
                start = p0 - padded_offsets[e] + group_offsets[e]
                size = group_sizes[e]
                rows = T.max(0, T.min(BM, size - (p0 - padded_offsets[e])))
                hs = T.alloc_shared((BM, BK), "bfloat16")
                dw = T.alloc_shared((BK, BN_H), "bfloat16")
                acc = T.alloc_fragment((BM, BN_H), "float32")
                out = T.alloc_shared((BM, BN_H), "float32")
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(I, BK), num_stages=2):
                    T.copy(HBuf[p0, k * BK], hs)
                    T.copy(WD[e, by * BN_H, k * BK], dw)
                    T.gemm(hs, dw, acc, transpose_B=True)
                T.copy(acc, out)
                for i, j in T.Parallel(BM, BN_H):
                    if i < rows and by * BN_H + j < H:
                        Out[p0 + i, by * BN_H + j] = out[i, j] * weights[p0 + i]
        return main
    return build()

def grouped_routed(tokens, weights, indices, w_gate, w_up, w_down):
    """Run all expert assignments in two grouped GEMM launches."""
    S, H = tokens.shape
    E, I, _ = w_gate.shape
    flat_e = indices.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    sorted_tokens = tokens.repeat_interleave(TOP_K, 0)[order]
    sorted_weights = weights.reshape(-1).float()[order]
    counts = torch.bincount(sorted_e, minlength=E).to(torch.int32)
    offsets = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    padded_counts = ((counts + BM - 1) // BM) * BM
    padded_offsets = (torch.cumsum(padded_counts, 0) - padded_counts).to(torch.int32)
    total_blocks = int(((counts + BM - 1) // BM).sum().item())
    if total_blocks == 0:
        return torch.zeros((S, H), device=tokens.device, dtype=torch.float32)
    group_for_block = torch.repeat_interleave(torch.arange(E, device=tokens.device, dtype=torch.int32),
                                              (counts + BM - 1) // BM)
    total_padded = total_blocks * BM
    xpad = torch.zeros((total_padded, H), device=tokens.device, dtype=tokens.dtype)
    wpad = torch.zeros((total_padded,), device=tokens.device, dtype=torch.float32)
    rank = torch.arange(sorted_e.numel(), device=tokens.device, dtype=torch.int32)
    within = rank - offsets[sorted_e]
    padded_pos = padded_offsets[sorted_e] + within
    xpad[padded_pos.long()] = sorted_tokens
    wpad[padded_pos.long()] = sorted_weights
    hbuf = torch.empty((total_padded, I), device=tokens.device, dtype=torch.float32)
    outbuf = torch.empty((total_padded, H), device=tokens.device, dtype=torch.float32)
    fn = _kernel(H, I, E, total_blocks, "bfloat16", "bfloat16")
    fn(xpad, w_gate, w_up, w_down, counts, offsets, padded_offsets,
       group_for_block, wpad, hbuf, outbuf)
    routed = torch.zeros((S * TOP_K, H), device=tokens.device, dtype=torch.float32)
    routed.index_add_(0, order, outbuf[padded_pos.long()])
    return routed.view(S, TOP_K, H).sum(1)
