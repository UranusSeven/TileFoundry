"""Explicit communication over one axis of a device mesh."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types.shard import Layout, Mesh, c_order_strides, canonical_shard_layout
from tilefoundry.ir.types.shard.scope_match import covered_by_scope
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
    split_target_axes,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


def device_mesh_shape(mesh: Mesh) -> tuple[int, ...]:
    """Validate the concrete, single-level device mesh supported by collectives."""
    if len(mesh.topologies) != 1 or mesh.topologies[0].name != "gpu":
        raise ValueError("collectives require a single-level gpu mesh")
    if not isinstance(mesh.layout, Layout):
        raise ValueError("collectives require a concrete Layout mesh")
    shape = mesh.layout.shape
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("collectives require positive concrete mesh extents")
    if mesh.layout.strides != c_order_strides(shape):
        raise ValueError("collectives require a contiguous mesh in coordinate order")
    capacity = mesh.topologies[0].size
    if type(capacity) is not int or prod(shape) > capacity:
        raise ValueError("collective mesh exceeds its concrete topology extent")
    return shape


@register_op
class AllReduce(Op):
    """Complete a partial reduction and replicate it along one mesh axis."""

    x = ParamDef(kind="input", pattern=Tensor)
    mesh_axis = ParamDef(kind="attribute", annotation=int)


@register_op
class AllGather(Op):
    """Gather equal shards in coordinate order along one mesh axis."""

    x = ParamDef(kind="input", pattern=Tensor)
    mesh_axis = ParamDef(kind="attribute", annotation=int)


@register_op
class ReduceScatter(Op):
    """Complete a partial reduction and equally partition one logical axis."""

    x = ParamDef(kind="input", pattern=Tensor)
    mesh_axis = ParamDef(kind="attribute", annotation=int)
    tensor_axis = ParamDef(kind="attribute", annotation=int)


def _collective_type(call, ctx):
    source = ctx.type_of(call.args[0])
    layout = source.layout
    if not isinstance(layout, ShardLayout):
        ctx.error(call, "collective input must have a ShardLayout")
    try:
        shape = device_mesh_shape(layout.mesh)
    except ValueError as error:
        ctx.error(call, str(error))
    if ctx.current_mesh is None or not covered_by_scope(layout.mesh, ctx.current_mesh):
        ctx.error(call, "collective mesh must be bound by the current mesh scope")
    axis = call.target.mesh_axis
    if type(axis) is not int or not 0 <= axis < len(shape):
        ctx.error(call, "mesh_axis must index the input mesh")
    if len(layout.attrs) != len(shape):
        ctx.error(call, "collective input needs one shard attribute per mesh axis")
    if not all(isinstance(attr, (Broadcast, Split, Partial)) for attr in layout.attrs):
        ctx.error(call, "collectives support only Broadcast, Split and Partial ownership")
    attrs = list(layout.attrs)
    targets = split_target_axes(layout, source.shape)
    mapping = layout_axis_to_tensor_axis(layout.layout.shape, source.shape)
    for attr, target in zip(attrs, targets):
        if isinstance(attr, Split):
            preceding = layout.layout.shape[mapping.index(target):attr.axis]
            if any(size != 1 for size in preceding):
                ctx.error(call, "collectives require contiguous partitions at leading layout factors")
    attrs = [Split(target) if target is not None else attr for target, attr in zip(targets, attrs)]
    split_axes = [attr.axis for attr in attrs if isinstance(attr, Split)]
    if len(set(split_axes)) != len(split_axes):
        ctx.error(call, "collectives support one mesh split per logical tensor axis")
    selected = attrs[axis]
    if isinstance(call.target, AllGather):
        if not isinstance(selected, Split):
            ctx.error(call, "AllGather requires Split on mesh_axis")
    elif not isinstance(selected, Partial) or selected.reduction not in ("sum", "max", "min"):
        ctx.error(call, "reduction collective requires Partial(sum, max or min) on mesh_axis")
    if isinstance(call.target, ReduceScatter):
        tensor_axis = call.target.tensor_axis
        if type(tensor_axis) is not int or not 0 <= tensor_axis < len(source.shape):
            ctx.error(call, "tensor_axis must index the logical tensor shape")
        if tensor_axis in split_axes:
            ctx.error(call, "ReduceScatter tensor_axis is already split by another mesh axis")
        attrs[axis] = Split(tensor_axis)
    else:
        attrs[axis] = Broadcast()
    try:
        result = canonical_shard_layout(source.shape, layout.mesh, tuple(attrs))
    except ValueError as error:
        ctx.error(call, str(error))
    return replace(source, layout=result)


def _eval_collective(ctx):
    raise EvalError("device collectives require distributed evaluation")


for _op in (AllReduce, AllGather, ReduceScatter):
    register_typeinfer(_op)(_collective_type)
    register_eval(_op)(_eval_collective)
    register_access_relation(_op)(identity_relations(1))
