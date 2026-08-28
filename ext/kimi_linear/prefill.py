"""Paged prefill for Kimi-Linear-48B-A3B: the prompt in one pass, at S > 1.

M3-B. Three pieces:

* `BlockManager` -- the page pool the MLA layers' prefill KV lives in. Pages
  are vllm_flash_attn-shaped ``[num_pages, page_size, H, D]``; a sequence's
  block table is its physical page ids in logical order. Allocated at
  prefill, freed when the driver hands over to decode.
* The prefill twins. MLA: `flash_attn_varlen_func` (FA3 -- FA2's paged
  kernel requires headdim_v == headdim_qk, and here they are 128 and 192)
  over the pages, the production counterpart of the `tf.paged_mla_prefill`
  HIR op in `ops.py`. KDA: fla's `chunk_kda` over the whole prompt, with the
  causal depthwise convolutions and the forget gate computed batched in
  torch. MoE / dense MLP: plain batched torch -- the decode kernels are
  S=1-shaped, and prefill here is correctness first (the gather-per-token
  routed-expert loop reads every selected expert's weights per token; a
  grouped-by-expert variant is the performance follow-up).
* `prefill(session, prompt_ids)` -- the driver: embed, walk the layers,
  closing norm and head on the last position, then install the state the
  decode driver continues from. The handoff is contract option (a): KDA
  recurrent state and conv windows carry over directly (fla's final state is
  k-major `[1, H, K, V]`; the authored layout is v-major `[1, H, V, K]` --
  transposed here), and the MLA pages are repacked once into the session's
  contiguous decode caches.

Numerics follow the authored IR's rounding placements, as the decode torch
transcriptions in `runtime_model.py` do: the rms norms round to bf16 before
gamma multiplies, the KDA gate's softplus input rounds through bf16, and the
attention / MoE reductions accumulate in f32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from vllm.vllm_flash_attn import flash_attn_varlen_func
try:
    from .kernels.moe_prefill import grouped_routed
except ImportError:
    from kernels.moe_prefill import grouped_routed

# The dims, spelled as runtime_model spells them (the published config's).
_HID = 2304
_NH = 32
_DK = 128
_KP = _NH * _DK
_NOPE = 128
_ROPE = 64
_QK = _NOPE + _ROPE
_V = 128
_LAT = 512
_KVB = _NOPE + _V
_TOPK = 8
_EPS = 1e-5
_W = 4                    # short conv kernel size
_WS = _W - 1              # stored conv positions
_BF16 = torch.bfloat16

#: The page size the MLA pages are laid out at (vllm_flash_attn's own).
PAGE_SIZE = 16


def _rms(hidden, gamma):
    """`KimiRMSNorm`: normalise in f32, round to bf16, *then* multiply gamma."""
    h32 = hidden.float()
    hn = (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + _EPS)).to(_BF16)
    return hn * gamma


class BlockManager:
    """The page pool the MLA layers' prefill KV lives in.

    One pool of physical pages, one block table per live sequence; a page id
    names the same slot in every MLA layer (the sequence occupies it in all
    of them at once), so one free list serves the whole stack. The driver
    holds exactly one sequence, but alloc/free are written as a real manager
    so the accounting is honest when that changes.
    """

    def __init__(self, capacity: int, device, nh: int = _NH) -> None:
        self.page_size = PAGE_SIZE
        self.num_pages = (capacity + PAGE_SIZE - 1) // PAGE_SIZE
        self.device = device
        self.nh = nh
        self._pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._free = list(range(self.num_pages))
        self.block_table: torch.Tensor | None = None  # [1, live pages], i32

    def pages(self, layer: int):
        """The layer's (k_pages, v_pages), allocated on first touch."""
        if layer not in self._pages:
            self._pages[layer] = (
                torch.zeros(
                    self.num_pages, PAGE_SIZE, self.nh, _QK,
                    dtype=_BF16, device=self.device,
                ),
                torch.zeros(
                    self.num_pages, PAGE_SIZE, self.nh, _V,
                    dtype=_BF16, device=self.device,
                ),
            )
        return self._pages[layer]

    def alloc(self, length: int) -> torch.Tensor:
        """Pages for *length* positions; returns the block table."""
        if self.block_table is not None:
            raise RuntimeError("a sequence is live; free it before allocating")
        n = (length + PAGE_SIZE - 1) // PAGE_SIZE
        if n > len(self._free):
            raise RuntimeError(
                f"page pool exhausted: need {n}, have {len(self._free)}"
            )
        taken, self._free = self._free[:n], self._free[n:]
        self.block_table = torch.tensor(
            taken, dtype=torch.int32, device=self.device
        ).reshape(1, n)
        return self.block_table

    def free(self) -> None:
        if self.block_table is not None:
            self._free.extend(self.block_table.reshape(-1).tolist())
            self.block_table = None

    def slot_ids(self, length: int) -> torch.Tensor:
        """The flat page-pool slot each position occupies, by the block table."""
        idx = torch.arange(length, device=self.device)
        pages = self.block_table.reshape(-1)
        return (
            pages[idx // PAGE_SIZE].long() * PAGE_SIZE + idx % PAGE_SIZE
        )


def mla_prefill(hidden, w, mgr: BlockManager, layer: int, length: int):
    """One MLA layer over the whole prompt: project, scatter to pages, attend.

    *w* is the mixer's bound weights. Returns the layer's attention output,
    ``[1, S, _HID]``; the K/V live in the manager's pages afterwards.
    """
    hn = _rms(hidden, w["gamma_in"])
    q = torch.matmul(hn, w["w_q"]).view(length, mgr.nh, _QK)
    compressed = torch.matmul(hn, w["w_kv_a"])
    latent, k_rot = compressed[..., :_LAT], compressed[..., _LAT:]
    kvn = _rms(latent, w["gamma_kv_a"])
    kv = torch.matmul(kvn, w["w_kv_b"]).view(1, length, mgr.nh, _KVB)
    # NoPE: the rope half is shared across heads and unrotated (cos = 1,
    # sin = 0), so the cached key is the nope part plus the broadcast share.
    k_new = torch.cat(
        [
            kv[..., :_NOPE],
            k_rot.view(1, length, 1, _ROPE).expand(1, length, mgr.nh, _ROPE),
        ],
        dim=-1,
    )
    v_new = kv[..., _NOPE:]
    k_pages, v_pages = mgr.pages(layer)
    slots = mgr.slot_ids(length)
    k_pages.view(-1, mgr.nh, _QK)[slots] = k_new[0].contiguous()
    v_pages.view(-1, mgr.nh, _V)[slots] = v_new[0].contiguous()

    out = flash_paged(q, k_pages, v_pages, mgr.block_table, length)
    return torch.matmul(out.view(1, length, mgr.nh * _V), w["w_o"])


def flash_paged(q, k_pages, v_pages, block_table, length: int):
    """The paged causal attention call itself, batch one, no prefix.

    *q* is ``[S, H, Dqk]``; the pages and block table are the pinned layout
    (``[pages, page_size, H, D]``, ``[1, live_pages]`` i32). FA3: FA2's paged
    kernel requires headdim_v == headdim_qk (here 128 vs 192). With
    ``seqused_k == S`` and causal=True, query row i attends keys 0..i --
    vllm's own prefill call shape (flash_attn.py's forward).
    """
    device = q.device
    cu_q = torch.tensor([0, length], dtype=torch.int32, device=device)
    seqused = torch.tensor([length], dtype=torch.int32, device=device)
    return flash_attn_varlen_func(
        q=q,
        k=k_pages,
        v=v_pages,
        cu_seqlens_q=cu_q,
        max_seqlen_q=length,
        seqused_k=seqused,
        max_seqlen_k=length,
        softmax_scale=_QK ** -0.5,
        causal=True,
        block_table=block_table,
        fa_version=3,
    )


def _conv_prefill(x_raw, conv_w):
    """The 4-tap causal depthwise conv + silu over the whole prompt.

    *x_raw* is ``[1, S, _KP]`` pre-conv activations; the left side pads with
    the zero initial conv window. Returns ``[1, S, _KP]``. (The conv weight
    is ``[W, KP]`` tap-major -- it must be transposed before broadcasting
    against the unfolded window's channel-major layout.)
    """
    xp = torch.cat(
        [torch.zeros(1, _WS, _KP, dtype=x_raw.dtype, device=x_raw.device), x_raw],
        dim=1,
    )
    win = xp.unfold(1, _W, 1)  # [1, S, KP, W], windows [t-3 .. t]
    acc = (win.float() * conv_w.float().permute(1, 0).unsqueeze(0).unsqueeze(0)).sum(-1)
    return F.silu(acc).to(x_raw.dtype)


def _conv_state(x_raw, length: int):
    """The decode window after the prompt.

    The last three raw rows, zero
    left-padded when the prompt is shorter than the window (HF left-pads).
    """
    if length >= _WS:
        return x_raw[:, length - _WS :].contiguous()
    pad = torch.zeros(
        1, _WS - length, _KP, dtype=x_raw.dtype, device=x_raw.device
    )
    return torch.cat([pad, x_raw], dim=1)


def kda_prefill(hidden, w, length: int):
    """One KDA layer over the whole prompt via fla's `chunk_kda`.

    Returns the layer output ``[1, S, _HID]``, the recurrent state in the
    authored v-major layout ``[1, H, V, K]`` (fla's final state is k-major),
    and the three conv windows for the decode handoff.
    """
    hn = _rms(hidden, w["gamma_in"])
    q_raw = torch.matmul(hn, w["w_q"])
    k_raw = torch.matmul(hn, w["w_k"])
    v_raw = torch.matmul(hn, w["w_v"])
    q_c = _conv_prefill(q_raw, w["conv_w_q"])
    k_c = _conv_prefill(k_raw, w["conv_w_k"])
    v_c = _conv_prefill(v_raw, w["conv_w_v"])

    # The per-channel forget gate: the softplus input rounds through bf16
    # (the authored placement), the activation runs in f32.
    g_raw = torch.matmul(torch.matmul(hn, w["w_f_a"]), w["w_f_b"]) + w["dt_bias"]
    g = -w["a_log"].float().exp().reshape(1, 1, _NH, 1) * F.softplus(
        g_raw.view(1, length, _NH, _DK).float()
    )
    beta = torch.sigmoid(torch.matmul(hn, w["w_b"]).float()).reshape(
        1, length, _NH
    )

    o, final_state = chunk_kda(
        q=q_c.view(1, length, _NH, _DK),
        k=k_c.view(1, length, _NH, _DK),
        v=v_c.view(1, length, _NH, _DK),
        g=g,
        beta=beta,
        scale=_DK ** -0.5,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # Gated output norm: rms_norm(o) * gamma * sigmoid(g2), in f32, landed
    # once into bf16 for the output projection.
    g2 = torch.matmul(torch.matmul(hn, w["w_g_a"]), w["w_g_b"]).view(
        1, length, _NH, _DK
    ).float()
    of = o.float()
    on = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + _EPS)
    on = on * w["gamma_o"].float() * torch.sigmoid(g2)
    out = torch.matmul(on.reshape(1, length, _KP).to(_BF16), w["w_o"])

    state = final_state.transpose(-1, -2).to(_BF16).contiguous()
    return (
        out,
        state,
        _conv_state(q_raw, length),
        _conv_state(k_raw, length),
        _conv_state(v_raw, length),
    )


def _router(tokens, w_router, bias, routed_scale):
    """The authored router, batched.

    f32 selection, weights normalised then
    scaled, rounded to bf16 as the authored return does.
    """
    logits = torch.matmul(tokens, w_router).float()
    scores = torch.sigmoid(logits)
    biased = scores + bias.float()
    top_biased, indices = torch.topk(biased, _TOPK, dim=-1)
    selected_bias = bias.float().reshape(-1)[indices.reshape(-1)].view_as(top_biased)
    unbiased = top_biased - selected_bias
    weights = (
        unbiased / unbiased.sum(-1, keepdim=True) * routed_scale.float().reshape(())
    )
    return weights.to(_BF16), indices


def moe_prefill(hidden, w, routed_scale, chunk: int = 32):
    """Prefill MoE using expert-sorted grouped GEMM."""
    length = hidden.shape[1]
    hn = _rms(hidden, w["gamma_post"])
    tokens = hn.reshape(length, _HID)
    weights, indices = _router(tokens, w["w_router"], w["bias"], routed_scale)
    routed = grouped_routed(tokens, weights, indices, w["w_gate"], w["w_up"], w["w_down"])
    x = tokens.reshape(1, length, _HID)
    shared = torch.matmul(
        F.silu(torch.matmul(x, w["sh_gate"])) * torch.matmul(x, w["sh_up"]),
        w["sh_down"],
    )
    return (routed + shared.reshape(length, _HID).float()).to(_BF16).reshape(1, length, _HID)


def mlp_prefill(hidden, w):
    """Layer 0's dense SwiGLU over the whole prompt."""
    hn = _rms(hidden, w["gamma_post"])
    return torch.matmul(
        F.silu(torch.matmul(hn, w["w_gate"])) * torch.matmul(hn, w["w_up"]),
        w["w_down"],
    )


@torch.no_grad()
def prefill(session, prompt_ids, moe_chunk: int = 32):
    """The prompt through the whole stack in one pass.

    Returns last-position
    logits with the session's state installed for decode.

    KDA layers hand over (conv_q, conv_k, conv_v, state) -- the session's
    cache layout -- and the MLA pages are repacked once into the contiguous
    decode buffers, so the decode path is exactly the M1 one from position
    `len(prompt_ids)` on.
    """
    twin = session.twin
    device = session.device
    length = len(prompt_ids)
    if length + 1 > session.capacity:
        raise RuntimeError(
            f"prompt of {length} exceeds the session capacity {session.capacity}"
        )
    ids = torch.tensor(prompt_ids, dtype=torch.int64, device=device)
    hidden = twin._bound["table"][ids].reshape(1, length, _HID)

    mgr = BlockManager(session.capacity, device)
    mgr.alloc(length)
    kda_states: dict[int, tuple] = {}
    try:
        for index, (mixer_kind, ffn_kind) in enumerate(session.kinds):
            layer = twin.modules[index]
            bw = layer.mixer._bound
            if mixer_kind == "kda":
                mixed, state, cq, ck, cv = kda_prefill(hidden, bw, length)
                kda_states[index] = (cq, ck, cv, state)
            else:
                mixed = mla_prefill(hidden, bw, mgr, index, length)
            hidden = hidden + mixed
            ffn = layer.moe if ffn_kind == "moe" else layer.mlp
            if ffn_kind == "moe":
                hidden = hidden + moe_prefill(
                    hidden, ffn._bound, session.routed_scale, moe_chunk
                )
            else:
                hidden = hidden + mlp_prefill(hidden, ffn._bound)

        normed = _rms(hidden, twin._bound["gamma_final"])
        logits = torch.matmul(
            normed[:, -1].reshape(1, _HID), twin._bound["w_head"]
        )

        # The handoff. reset() first so every buffer is in a known state.
        session.reset()
        slots = mgr.slot_ids(length)
        for index, (mixer_kind, _ffn) in enumerate(session.kinds):
            if mixer_kind == "kda":
                session.kda[index] = kda_states[index]
            else:
                k_pages, v_pages = mgr.pages(index)
                session.kbuf[index][:, :length] = (
                    k_pages.view(-1, mgr.nh, _QK)[slots].unsqueeze(0)
                )
                session.vbuf[index][:, :length] = (
                    v_pages.view(-1, mgr.nh, _V)[slots].unsqueeze(0)
                )
        session.pos = length
    finally:
        mgr.free()
    return logits


__all__ = ["PAGE_SIZE", "BlockManager", "flash_paged", "kda_prefill",
           "mla_prefill", "mlp_prefill", "moe_prefill", "prefill"]
