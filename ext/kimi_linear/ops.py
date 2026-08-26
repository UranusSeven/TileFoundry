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
