"""Two-rank NCCL prefill runtime with checkpoint-time tensor sharding."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

try:
    from .prefill import BlockManager, _rms, kda_prefill, mla_prefill, mlp_prefill, moe_prefill
except ImportError:
    from prefill import BlockManager, _rms, kda_prefill, mla_prefill, mlp_prefill, moe_prefill

_BF16 = torch.bfloat16
_HID = 2304


def _load_unchecked(node, resource):
    children = tuple(_load_unchecked(child, resource.subtree(child.name)) for child in node.modules)
    loaded = SimpleNamespace(
        constants={name: resource.load(name) for name in node.weights},
        modules=children,
    )
    for child, bound in zip(node.modules, children):
        setattr(loaded, child.name, bound)
    return loaded


def _converge(value):
    dist.all_reduce(value)
    return value


def _pack_kda_weights(weights, kinds):
    """Replace KDA matrices with equal-size packed runtime layouts."""
    for (mixer_kind, _), layer in zip(kinds, weights.modules):
        if mixer_kind != "kda":
            continue
        w = layer.mixer.constants
        w["w_qkv"] = torch.cat((w.pop("w_q"), w.pop("w_k"), w.pop("w_v")), dim=-1)
        w["w_fg_beta"] = torch.cat(
            (w.pop("w_f_a"), w.pop("w_g_a"), w.pop("w_b")), dim=-1
        )
        w["w_fg_b"] = torch.stack((w.pop("w_f_b")[0], w.pop("w_g_b")[0]))
        for name in ("conv_w_q", "conv_w_k", "conv_w_v"):
            w[name] = w[name].transpose(0, 1).contiguous()


class PrefillRunnerTP2:
    """Prefill-only TP=2 runner; each process owns one rank-local shard."""

    def __init__(self, ckpt=None, *, device=None, verbose=False):
        import model as sem  # noqa: PLC0415
        import weights as wt  # noqa: PLC0415

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if dist.get_world_size() != 2:
            raise RuntimeError("PrefillRunnerTP2 requires exactly two torchrun ranks")
        self.rank = dist.get_rank()
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}" if device is None else device)
        self.config = sem.config
        self.kinds = sem.LAYER_KINDS
        torch.cuda.reset_peak_memory_stats(self.device)
        before = torch.cuda.memory_allocated(self.device)
        resource, totals = wt.prefill_resource(
            ckpt=wt.CKPT if ckpt is None else ckpt,
            device=str(self.device),
            verbose=verbose,
            tp_rank=self.rank,
        )
        started = time.perf_counter()
        self.weights = _load_unchecked(sem.KimiLinear48BA3BPrefill, resource)
        _pack_kda_weights(self.weights, self.kinds)
        self.load_seconds = time.perf_counter() - started
        self.loaded_bytes = totals["bytes"]
        self.resident_bytes = torch.cuda.memory_allocated(self.device) - before
        self.load_peak_bytes = torch.cuda.max_memory_allocated(self.device) - before
        self.routed_scale = torch.tensor(
            self.config.routed_scaling_factor, dtype=_BF16, device=self.device
        )

    @torch.no_grad()
    def __call__(self, input_ids):
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device=self.device)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("input_ids must be a non-empty one-dimensional sequence")
        length = ids.numel()
        hidden = self.weights.constants["table"][ids].reshape(1, length, _HID)
        pages = BlockManager(length, self.device, nh=16)
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
                hidden = hidden + _converge(mixed)
                ffn = layer.moe.constants if ffn_kind == "moe" else layer.mlp.constants
                partial = (
                    moe_prefill(hidden, ffn, self.routed_scale)
                    if ffn_kind == "moe"
                    else mlp_prefill(hidden, ffn)
                )
                hidden = hidden + _converge(partial)
            normed = _rms(hidden, self.weights.constants["gamma_final"])
            logits = torch.matmul(normed[:, -1], self.weights.constants["w_head"])
            dist.broadcast(logits, src=0)
            return logits
        finally:
            pages.free()


__all__ = ["PrefillRunnerTP2"]
