"""Bounded expert dispatch and reverse accumulation with explicit routing tensors."""

from __future__ import annotations

import isl

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.sharding.collective import device_mesh_shape
from tilefoundry.ir.types import DType, FloatDType, TensorType, TupleType
from tilefoundry.ir.types.shard import canonical_shard_layout
from tilefoundry.ir.types.shard.scope_match import covered_by_scope
from tilefoundry.ir.types.shard.shard_layout import Broadcast, ShardLayout, Split, split_target_axes
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    iterating,
    reached_at,
    register_access_relation,
)


@register_op(name="alltoall_dispatch")
class AllToAllDispatch(Op):
    """Expand routed tokens into bounded expert-major buffers and return routing metadata."""

    x = ParamDef(kind="input", pattern=Tensor)
    expert_indices = ParamDef(kind="input", pattern=Tensor)
    expert_weights = ParamDef(kind="input", pattern=Tensor)
    num_experts = ParamDef(kind="attribute", annotation=int)
    capacity = ParamDef(kind="attribute", annotation=int)
    mesh_axis = ParamDef(kind="attribute", annotation=int)


@register_op(name="alltoall_combine")
class AllToAllCombine(Op):
    """Return weighted expert contributions to their source tokens and sum them."""

    x = ParamDef(kind="input", pattern=Tensor)
    source_indices = ParamDef(kind="input", pattern=Tensor)
    counts = ParamDef(kind="input", pattern=Tensor)
    like = ParamDef(kind="input", pattern=Tensor)
    mesh_axis = ParamDef(kind="attribute", annotation=int)


def _attrs(call, ctx, type_, mesh):
    layout = type_.layout
    if not isinstance(layout, ShardLayout) or layout.mesh != mesh:
        ctx.error(call, "all routing operands must carry the same device mesh")
    if len(layout.attrs) != len(mesh.layout.shape) or not all(
        isinstance(attr, (Broadcast, Split)) for attr in layout.attrs
    ):
        ctx.error(call, "routing operands require Broadcast or Split ownership on every mesh axis")
    targets = split_target_axes(layout, type_.shape)
    return tuple(
        Split(target) if target is not None else attr for target, attr in zip(targets, layout.attrs)
    )


def _source(call, ctx, source, *, check_scope=True):
    if len(source.shape) < 2 or not isinstance(source.layout, ShardLayout):
        ctx.error(call, "source must be a sharded [..., tokens, hidden] tensor")
    mesh = source.layout.mesh
    try:
        shape = device_mesh_shape(mesh)
    except ValueError as error:
        ctx.error(call, str(error))
    if check_scope and (ctx.current_mesh is None or not covered_by_scope(mesh, ctx.current_mesh)):
        ctx.error(call, "all-to-all mesh must be bound by the current mesh scope")
    axis = call.target.mesh_axis
    if type(axis) is not int or not 0 <= axis < len(shape):
        ctx.error(call, "mesh_axis must index the input mesh")
    attrs = _attrs(call, ctx, source, mesh)
    token_axis = len(source.shape) - 2
    if attrs[axis] != Split(token_axis):
        ctx.error(call, "source tokens must be Split on mesh_axis")
    if any(
        i != axis and isinstance(attr, Split) and attr.axis >= token_axis
        for i, attr in enumerate(attrs)
    ):
        ctx.error(call, "other mesh axes may split only leading batch dimensions")
    tokens = source.shape[-2]
    if isinstance(tokens, int) and tokens % shape[axis]:
        ctx.error(call, "source token extent must be divisible by the selected mesh extent")
    return mesh, attrs


def _placed(shape, dtype, source, mesh, attrs):
    return TensorType(
        shape=shape,
        dtype=dtype,
        storage=source.storage,
        layout=canonical_shard_layout(shape, mesh, attrs),
    )


def dispatch_types(call, ctx, *, check_scope=True):
    """Derive fixed capacity shapes while leaving live route counts as tensor values."""
    x, indices, weights = (ctx.type_of(arg) for arg in call.args)
    mesh, attrs = _source(call, ctx, x, check_scope=check_scope)
    experts, capacity = call.target.num_experts, call.target.capacity
    if (
        type(experts) is not int
        or experts <= 0
        or experts % mesh.layout.shape[call.target.mesh_axis]
    ):
        ctx.error(call, "num_experts must be positive and divisible by the selected mesh extent")
    if type(capacity) is not int or capacity < 0:
        ctx.error(call, "capacity must be a nonnegative integer per expert")
    if len(indices.shape) != len(x.shape) or indices.shape[:-1] != x.shape[:-1]:
        ctx.error(call, "expert_indices must have shape [..., tokens, topk] matching x")
    if indices.dtype not in (DType.i32, DType.i64):
        ctx.error(call, "expert_indices must have dtype i32 or i64")
    if weights.shape != indices.shape or not isinstance(weights.dtype, FloatDType):
        ctx.error(call, "expert_weights must be floating point with the expert_indices shape")
    for type_ in (indices, weights):
        if _attrs(call, ctx, type_, mesh) != attrs:
            ctx.error(call, "tokens, expert indices and weights must have matching ownership")
    prefix = (*x.shape[:-2], experts, capacity)
    return (
        _placed((*prefix, x.shape[-1]), x.dtype, x, mesh, attrs),
        _placed((*prefix, 1), weights.dtype, x, mesh, attrs),
        _placed(prefix, DType.i64, x, mesh, attrs),
        _placed(prefix[:-1], DType.i64, x, mesh, attrs),
    )


@register_typeinfer(AllToAllDispatch)
def _infer_dispatch(call, ctx):
    return TupleType(fields=dispatch_types(call, ctx))


def combine_type(call, ctx, *, check_scope=True):
    """Recover source ownership with the expert output's hidden extent and dtype."""
    x, indices, counts, source = (ctx.type_of(arg) for arg in call.args)
    mesh, attrs = _source(call, ctx, source, check_scope=check_scope)
    if len(x.shape) != len(source.shape) + 1 or x.shape[:-3] != source.shape[:-2]:
        ctx.error(call, "expert output must have shape [..., experts, capacity, hidden]")
    experts = x.shape[-3]
    if (
        type(experts) is not int
        or experts <= 0
        or experts % mesh.layout.shape[call.target.mesh_axis]
    ):
        ctx.error(call, "expert extent must be positive and divisible by the selected mesh extent")
    if indices.shape != x.shape[:-1] or counts.shape != x.shape[:-2]:
        ctx.error(call, "source_indices and counts must match expert output capacity and experts")
    if indices.dtype not in (DType.i32, DType.i64) or counts.dtype not in (DType.i32, DType.i64):
        ctx.error(call, "source_indices and counts must be i32 or i64")
    for type_ in (x, indices, counts):
        if _attrs(call, ctx, type_, mesh) != attrs:
            ctx.error(
                call, "expert output and routing metadata must have matching expert ownership"
            )
    return _placed((*source.shape[:-1], x.shape[-1]), x.dtype, source, mesh, attrs)


register_typeinfer(AllToAllCombine)(combine_type)


@register_eval(AllToAllDispatch)
def _eval_dispatch(ctx):
    if not ctx.distributed:
        raise EvalError("device collectives require distributed evaluation")
    from tilefoundry.evaluator.alltoall import dispatch  # noqa: PLC0415

    return dispatch(ctx)


@register_eval(AllToAllCombine)
def _eval_combine(ctx):
    if not ctx.distributed:
        raise EvalError("device collectives require distributed evaluation")
    from tilefoundry.evaluator.alltoall import combine  # noqa: PLC0415

    return combine(ctx)


@register_access_relation(AllToAllDispatch)
def _dispatch_access(call, ctx):
    """Bound index-dependent reads without assuming a balanced or fixed routing pattern."""
    x, indices, weights = (ctx.type_of(arg) for arg in call.args)
    outputs = dispatch_types(call, ctx, check_scope=False)
    batch = len(x.shape) - 2
    return iterating(
        (*x.shape[:batch], 1),
        AccessRelations(
            inputs=tuple(_bounded(type_, batch) for type_ in (x, indices, weights)),
            outputs=tuple(_bounded(type_, batch) for type_ in outputs),
        ),
    )


def _bounded(type_, batch):
    """Cover the declared routing envelope in one batch, including count-only buffers."""
    prefix = {axis: f"d{axis}" for axis in range(batch)}
    return BoundaryRelation(
        reached_at(
            batch + 1,
            type_,
            type_,
            prefix,
            free=tuple(range(batch, len(type_.shape))),
        )
    )


@register_access_relation(AllToAllCombine)
def _combine_access(call, ctx):
    """Bound the expert slots selected by source indices; the source supplies only shape."""
    x, indices, counts, source = (ctx.type_of(arg) for arg in call.args)
    output = combine_type(call, ctx, check_scope=False)
    batch = len(source.shape) - 2
    domain = ", ".join(f"d{i}" for i in range(batch + 1))
    zeros = ", ".join("0" for _ in source.shape)
    shape_only = AffineAccess(isl.map(f"{{ [{domain}] -> [{zeros}] : false }}"))
    return iterating(
        (*source.shape[:batch], 1),
        AccessRelations(
            inputs=(
                *(_bounded(type_, batch) for type_ in (x, indices, counts)),
                BoundaryRelation(shape_only),
            ),
            outputs=(_bounded(output, batch),),
        ),
    )
