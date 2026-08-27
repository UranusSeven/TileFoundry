"""The external HIR op `tf.all_reduce`: converge a `Partial` value over its mesh axis.

Registered from the ext tree (external ops name their dialect/category
explicitly; builtins derive them from the module path). Semantics per
[shard §6](docs/spec/shard.md#6-shardattr): a `Partial(reduction)` value on a
mesh axis is the un-reduced per-shard state, and the full value is the
`reduction` over the shards -- this op is the boundary that performs it, the
layout-level statement of "NCCL allreduce goes here".

Four pieces, as every op:

- **typeinfer** implements the Partial -> converged transition: each mesh
  axis carrying `Partial(r)` (with `r` matching the op's reduction) becomes
  `Broadcast`; the shape, dtype and storage pass through. A `Split` axis is
  refused (converging a *placement* is an all-gather, a different op). A
  tensor with no `ShardLayout` at all passes through unchanged -- that is the
  single-device / per-rank-program case, where the collective is one rank
  deep and therefore the identity.
- **eval** is the identity on the logical value: the evaluator holds whole
  tensors, and an allreduce over one process's value is that value.
- **access relation** is the identity: every element is read once and
  written once (the network movement is not an element access).

The runtime counterpart -- what a `@runtime_func` twin actually calls -- is
`torch.distributed.all_reduce`; see `runtime_tp2.py`.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    Split,
    shard_layout_of,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op(dialect="tf", category="sharding", name="all_reduce")
class AllReduce(Op):
    """Converge `x`: `Partial(reduction)` on a mesh axis becomes `Broadcast`."""

    x = ParamDef(kind="input", pattern=Tensor)
    reduction = ParamDef(kind="attribute", annotation=str, default="sum")


register_access_relation(AllReduce)(identity_relations(1))


@register_typeinfer(AllReduce)
def _(call: "Call", ctx: "TypeInferContext"):
    ty = ctx.type_of(call.args[0])
    layout = shard_layout_of(ty.layout)
    if layout is None:
        # No placement: a single device holds the whole value, so the
        # collective is the identity.
        return ty
    attrs = list(layout.attrs)
    converged = False
    for axis, attr in enumerate(attrs):
        if isinstance(attr, Partial):
            if attr.reduction != call.target.reduction:
                ctx.error(
                    call,
                    f"all_reduce(reduction={call.target.reduction!r}) but the "
                    f"value is Partial({attr.reduction!r}) on mesh axis {axis}",
                )
            attrs[axis] = Broadcast()
            converged = True
        elif isinstance(attr, Split):
            ctx.error(
                call,
                f"all_reduce: mesh axis {axis} is Split; converging a "
                "placement is an all_gather, not an all_reduce",
            )
    if not converged:
        ctx.error(
            call,
            "all_reduce: the value carries no Partial state -- it is already "
            "converged, and the collective would double it",
        )
    return replace(ty, layout=replace(layout, attrs=tuple(attrs)))


@register_eval(AllReduce)
def _eval_all_reduce(ctx):
    """Identity on the logical value: the evaluator holds whole tensors."""
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


__all__ = ["AllReduce"]


# ---------------------------------------------------------------------------
# The external HIR op `tf.paged_mla_prefill`: causal MHA over a paged KV cache.
# ---------------------------------------------------------------------------
#
# The prefill counterpart of the decode attention: one call attends an S-row
# query block to S keys/values held in fixed-size pages, addressed indirectly
# through a block table. Page layout is vllm_flash_attn's:
# `k_pages [num_pages, page_size, H, head_dim_qk]`,
# `v_pages [num_pages, page_size, H, head_dim_v]` (head_dim_v may differ from
# head_dim_qk -- 128 vs 192 here), `block_table [num_live_pages]` the physical
# page ids in logical order. The op is HIR-only: nothing in TileFoundry lowers
# it; the runtime counterpart is the FA3 `flash_attn_varlen_func` call in
# `prefill.py`, and `eval` below is its stated semantics (gather the pages in
# block-table order, causal softmax attention in f32, land in the input
# dtype), which `check_paged_prefill.py` scores against that kernel.

import torch  # noqa: E402

from tilefoundry.ir.types import DType, TensorType  # noqa: E402
from tilefoundry.visitor_registry.access_relation import (  # noqa: E402
    AccessRelations,
    BoundaryRelation,
    identity_access,
    iterating,
    reached_at,
)


@register_op(dialect="tf", category="nn", name="paged_mla_prefill")
class PagedMlaPrefill(Op):
    """Causal prefill attention over a paged KV cache.

    `out[s, h, :]` attends `q[s, h, :]` to the first
    `s + 1` key/value rows of the paged cache of the paged cache, `scale * q . k` per score.

    `scale <= 0` means the default `head_dim_qk ** -0.5`.
    """

    q = ParamDef(kind="input", pattern=Tensor)
    k_pages = ParamDef(kind="input", pattern=Tensor)
    v_pages = ParamDef(kind="input", pattern=Tensor)
    block_table = ParamDef(kind="input", pattern=Tensor)
    scale = ParamDef(kind="attribute", annotation=float, default=-1.0)


@register_typeinfer(PagedMlaPrefill)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    q_ty, k_ty, v_ty, bt_ty = (ctx.type_of(call.args[i]) for i in range(4))
    if len(q_ty.shape) != 3:
        ctx.error(call, f"q must be [S, H, Dqk], got shape {q_ty.shape}")
    if len(k_ty.shape) != 4 or len(v_ty.shape) != 4:
        ctx.error(
            call,
            f"k_pages/v_pages must be [pages, page_size, H, D], got "
            f"{k_ty.shape} and {v_ty.shape}",
        )
    if len(bt_ty.shape) != 1:
        ctx.error(call, f"block_table must be 1-D, got shape {bt_ty.shape}")
    if bt_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"block_table must be i32 or i64, got {bt_ty.dtype}")
    s, h, qk = q_ty.shape
    if k_ty.shape[2] != h or v_ty.shape[2] != h:
        ctx.error(
            call,
            f"head counts differ: q has {h}, k_pages {k_ty.shape[2]}, "
            f"v_pages {v_ty.shape[2]}",
        )
    if k_ty.shape[3] != qk:
        ctx.error(
            call, f"key head_dim {k_ty.shape[3]} != query head_dim {qk}"
        )
    if k_ty.shape[1] != v_ty.shape[1] or k_ty.shape[0] != v_ty.shape[0]:
        ctx.error(call, "k_pages and v_pages must share the page grid")
    return TensorType(
        shape=(s, h, v_ty.shape[3]),
        dtype=q_ty.dtype,
        layout=None,
        storage=q_ty.storage,
    )


@register_access_relation(PagedMlaPrefill)
def _paged_mla_prefill_access(call: "Call", ctx) -> AccessRelations:
    """The access relation at the GLOBAL level.

    The output space is walked; pages are reached through the
    block table, so the page axis (and the within-page position, a causal
    range) is `free` -- the deciding value lives in `block_table`, exactly as
    `index_select`'s index axis is free.
    """
    q_ty, k_ty, v_ty, bt_ty = (ctx.type_of(call.args[i]) for i in range(4))
    out_ty = TensorType(
        shape=(q_ty.shape[0], q_ty.shape[1], v_ty.shape[3]),
        dtype=q_ty.dtype,
        layout=None,
    )
    return iterating(
        out_ty.shape,
        AccessRelations(
            inputs=(
                BoundaryRelation(
                    reached_at(3, q_ty, q_ty, {0: "d0", 1: "d1"}, free=(2,))
                ),
                BoundaryRelation(
                    reached_at(3, k_ty, k_ty, {2: "d1"}, free=(0, 1, 3))
                ),
                BoundaryRelation(
                    reached_at(3, v_ty, v_ty, {2: "d1", 3: "d2"}, free=(0, 1))
                ),
                BoundaryRelation(reached_at(3, bt_ty, bt_ty, {}, free=(0,))),
            ),
            outputs=(BoundaryRelation(identity_access(3)),),
        ),
    )


@register_eval(PagedMlaPrefill)
def _eval_paged_mla_prefill(ctx):
    """The op's stated semantics: gather, then causal attention in f32."""
    q, k_pages, v_pages, block_table = (ctx.args[i].data for i in range(4))
    s, h, qk = q.shape
    v_dim = v_pages.shape[-1]
    k = k_pages.index_select(0, block_table.long()).reshape(-1, h, qk)[:s]
    v = v_pages.index_select(0, block_table.long()).reshape(-1, h, v_dim)[:s]
    scale = ctx.op.scale if ctx.op.scale > 0 else qk ** -0.5
    qf = q.float().permute(1, 0, 2)                       # [H, S, QK]
    kf = k.float().permute(1, 0, 2)                       # [H, S, QK]
    vf = v.float().permute(1, 0, 2)                       # [H, S, V]
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    mask = torch.full(
        (s, s), float("-inf"), device=q.device
    ).triu(1)
    probs = torch.softmax(scores + mask, dim=-1)
    out = torch.matmul(probs, vf).permute(1, 0, 2)        # [S, H, V]
    return TensorValue(data=out.to(q.dtype), type=ctx.result_type)


__all__ = ["AllReduce", "PagedMlaPrefill"]
