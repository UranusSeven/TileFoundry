"""Self-contained prefill runtime for Kimi-Linear-48B-A3B.

``PrefillRunner`` owns the published prefill weights and executes exactly one
request at a time: embedding, 27 batched layers, final norm, and the
last-position LM head. MLA uses request-local FA3 pages, KDA uses FLA
``chunk_kda`` without returning recurrent state, and MoE uses vLLM fused
experts with the checkpoint's packed ``gate_up`` ABI. No decode state, cache
handoff, or generation resource is part of this module.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import fused_kda_gate
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
from vllm.vllm_flash_attn import flash_attn_varlen_func

try:
    from .kernels.moe_prefill import grouped_routed
    from .kernels.moe_prefill_vllm import (
        fused_routed,
        implementation,
    )
except ImportError:
    from kernels.moe_prefill import grouped_routed
    from kernels.moe_prefill_vllm import (
        fused_routed,
        implementation,
    )

# Published checkpoint dimensions used by this transitional runtime.
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
_W = 4  # short conv kernel size
_WS = _W - 1  # stored conv positions
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
                    self.num_pages,
                    PAGE_SIZE,
                    self.nh,
                    _QK,
                    dtype=_BF16,
                    device=self.device,
                ),
                torch.zeros(
                    self.num_pages,
                    PAGE_SIZE,
                    self.nh,
                    _V,
                    dtype=_BF16,
                    device=self.device,
                ),
            )
        return self._pages[layer]

    def alloc(self, length: int) -> torch.Tensor:
        """Pages for *length* positions; returns the block table."""
        if self.block_table is not None:
            raise RuntimeError("a sequence is live; free it before allocating")
        n = (length + PAGE_SIZE - 1) // PAGE_SIZE
        if n > len(self._free):
            raise RuntimeError(f"page pool exhausted: need {n}, have {len(self._free)}")
        taken, self._free = self._free[:n], self._free[n:]
        self.block_table = torch.tensor(taken, dtype=torch.int32, device=self.device).reshape(1, n)
        return self.block_table

    def free(self) -> None:
        if self.block_table is not None:
            self._free.extend(self.block_table.reshape(-1).tolist())
            self.block_table = None

    def slot_ids(self, length: int) -> torch.Tensor:
        """The flat page-pool slot each position occupies, by the block table."""
        idx = torch.arange(length, device=self.device)
        pages = self.block_table.reshape(-1)
        return pages[idx // PAGE_SIZE].long() * PAGE_SIZE + idx % PAGE_SIZE


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
        softmax_scale=_QK**-0.5,
        causal=True,
        block_table=block_table,
        fa_version=3,
    )


def _conv_prefill(x_raw, conv_w, conv_metadata=None):
    """Run the four-tap causal depthwise convolution and SiLU."""
    del conv_metadata
    if conv_w.shape[0] == _W:
        xp = torch.cat(
            [
                torch.zeros(
                    1, _WS, x_raw.shape[-1], dtype=x_raw.dtype, device=x_raw.device
                ),
                x_raw,
            ],
            dim=1,
        )
        window = xp.unfold(1, _W, 1)
        acc = (
            window.float()
            * conv_w.float().permute(1, 0).unsqueeze(0).unsqueeze(0)
        ).sum(-1)
        return F.silu(acc).to(x_raw.dtype)
    convolved = F.conv1d(
        x_raw.transpose(1, 2),
        conv_w.unsqueeze(1),
        padding=_WS,
        groups=x_raw.shape[-1],
    )[..., : x_raw.shape[1]]
    return F.silu(convolved).transpose(1, 2)


def kda_prefill(hidden, w, length: int, conv_metadata=None):
    """One KDA layer over the whole prompt via fla's `chunk_kda`.

    Returns only the layer output ``[1, S, _HID]``. The recurrence starts at
    zero and its final state is intentionally discarded.
    """
    hn = _rms(hidden, w["gamma_in"])
    if "w_qkv" in w:
        kp = w["w_qkv"].shape[-1] // 3
        qkv = torch.matmul(hn, w["w_qkv"])
        q_raw, k_raw, v_raw = qkv.split(kp, dim=-1)
    else:
        kp = w["w_q"].shape[-1]
        q_raw = torch.matmul(hn, w["w_q"])
        k_raw = torch.matmul(hn, w["w_k"])
        v_raw = torch.matmul(hn, w["w_v"])
    nh = kp // _DK
    q_c = _conv_prefill(q_raw, w["conv_w_q"], conv_metadata)
    k_c = _conv_prefill(k_raw, w["conv_w_k"], conv_metadata)
    v_c = _conv_prefill(v_raw, w["conv_w_v"], conv_metadata)

    # The packed projection preserves the authored bf16 landing before the
    # gate B projections and activation.
    if "w_fg_beta" in w:
        fg_beta = torch.matmul(hn, w["w_fg_beta"])
        f_a, g_a, beta_raw = fg_beta.split((_DK, _DK, nh), dim=-1)
        fg = torch.bmm(torch.stack((f_a[0], g_a[0])), w["w_fg_b"])
    else:
        f_a = torch.matmul(hn, w["w_f_a"])
        g_a = torch.matmul(hn, w["w_g_a"])
        beta_raw = torch.matmul(hn, w["w_b"])
        fg = torch.stack(
            (
                torch.matmul(f_a, w["w_f_b"])[0],
                torch.matmul(g_a, w["w_g_b"])[0],
            )
        )
    g = fused_kda_gate(
        fg[0].view(1, length, nh, _DK),
        w["a_log"],
        w["dt_bias"],
        output_dtype=torch.float32,
    )
    beta = torch.sigmoid(beta_raw.float()).reshape(1, length, nh)

    o, _ = chunk_kda(
        q=q_c.view(1, length, nh, _DK),
        k=k_c.view(1, length, nh, _DK),
        v=v_c.view(1, length, nh, _DK),
        g=g,
        beta=beta,
        scale=_DK**-0.5,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    # Gated output norm: rms_norm(o) * gamma * sigmoid(g2), in f32, landed
    # once into bf16 for the output projection.
    g2 = fg[1].view(1, length, nh, _DK)
    on = rms_norm_gated(
        o,
        g2,
        w["gamma_o"],
        None,
        activation="sigmoid",
        eps=_EPS,
    )
    out = torch.matmul(on.reshape(1, length, kp), w["w_o"])
    return out


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
    weights = unbiased / unbiased.sum(-1, keepdim=True) * routed_scale.float().reshape(())
    return weights.to(_BF16), indices


def moe_prefill(hidden, w, routed_scale):
    """Prefill MoE with GPU-only vLLM dispatch or the TileLang fallback."""
    length = hidden.shape[1]
    hn = _rms(hidden, w["gamma_post"])
    tokens = hn.reshape(length, _HID)
    weights, indices = _router(tokens, w["w_router"], w["bias"], routed_scale)
    if implementation() == "vllm":
        routed = fused_routed(tokens, weights, indices, w["w_gate_up"], w["w_down"])
    else:
        intermediate = w["w_gate_up"].shape[1] // 2
        routed = grouped_routed(
            tokens,
            weights,
            indices,
            w["w_gate_up"][:, :intermediate],
            w["w_gate_up"][:, intermediate:],
            w["w_down"],
        )
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


class PrefillRunner:
    """Weight-owning, single-device prefill runner."""

    def __init__(self, ckpt=None, *, device="cuda", verbose=False):
        import model as sem  # noqa: PLC0415
        import weights as wt  # noqa: PLC0415

        self.device = torch.device(device)
        self.config = sem.config
        self.kinds = sem.LAYER_KINDS
        resource, totals = wt.prefill_resource(
            ckpt=wt.CKPT if ckpt is None else ckpt,
            device=str(self.device),
            verbose=verbose,
        )
        started = time.perf_counter()
        self.weights = sem.KimiLinear48BA3BPrefill.load(resource)
        self.load_seconds = time.perf_counter() - started
        self.loaded_bytes = totals["bytes"]
        self.routed_scale = torch.tensor(
            self.config.routed_scaling_factor, dtype=_BF16, device=self.device
        )

    @torch.no_grad()
    def __call__(self, input_ids):
        """Return ``[1, vocab]`` BF16 logits for the prompt's last position."""
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device=self.device)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("input_ids must be a non-empty one-dimensional sequence")
        length = ids.numel()
        if length > self.config.model_max_length:
            raise ValueError(
                f"prompt length {length} exceeds model_max_length {self.config.model_max_length}"
            )

        hidden = self.weights.constants["table"][ids].reshape(1, length, _HID)
        pages = BlockManager(length, self.device)
        pages.alloc(length)
        try:
            for index, ((mixer_kind, ffn_kind), layer) in enumerate(
                zip(self.kinds, self.weights.modules)
            ):
                mixer = layer.mixer.constants
                if mixer_kind == "kda":
                    mixed = kda_prefill(hidden, mixer, length)
                else:
                    mixed = mla_prefill(hidden, mixer, pages, index, length)
                hidden = hidden + mixed
                ffn = layer.moe.constants if ffn_kind == "moe" else layer.mlp.constants
                if ffn_kind == "moe":
                    hidden = hidden + moe_prefill(hidden, ffn, self.routed_scale)
                else:
                    hidden = hidden + mlp_prefill(hidden, ffn)

            normed = _rms(hidden, self.weights.constants["gamma_final"])
            return torch.matmul(normed[:, -1], self.weights.constants["w_head"])
        finally:
            pages.free()


__all__ = [
    "PAGE_SIZE",
    "BlockManager",
    "PrefillRunner",
    "flash_paged",
    "kda_prefill",
    "mla_prefill",
    "mlp_prefill",
    "moe_prefill",
]
