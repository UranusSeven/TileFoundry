"""Simulate device-local tensors and explicit communication on one torch device."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import torch

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue, Value
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.sharding.alltoall import AllToAllCombine, AllToAllDispatch
from tilefoundry.ir.hir.sharding.collective import (
    AllGather,
    AllReduce,
    AllToAll,
    ReduceScatter,
    device_mesh_shape,
)
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import Layout, Mesh
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)
from tilefoundry.ir.types.substitute import substitute_dims


@dataclass(frozen=True)
class DistributedValue(Value):
    """Hold concrete local tensors in lexicographic mesh-coordinate order."""

    shards: tuple[torch.Tensor, ...]
    type: TensorType
    mesh: Mesh


def coordinates(mesh):
    """Enumerate independent participants, with the last mesh axis fastest."""
    return tuple(product(*(range(size) for size in device_mesh_shape(mesh))))


def participant_groups(mesh, axis):
    """Group rank indices by all coordinates except the selected mesh axis."""
    groups = {}
    for rank, coord in enumerate(coordinates(mesh)):
        key = coord[:axis] + coord[axis + 1:]
        groups.setdefault(key, []).append(rank)
    return tuple(tuple(ranks) for ranks in groups.values())


def _shape(type_, bindings):
    return tuple(resolve_dim(size, bindings) for size in type_.shape)


def _attrs(type_, mesh):
    layout = type_.layout
    if layout is None or isinstance(layout, Layout):
        return (Broadcast(),) * len(mesh.layout.shape)
    if not isinstance(layout, ShardLayout) or layout.mesh != mesh:
        raise EvalError("distributed evaluation requires a matching device ShardLayout")
    if len(layout.attrs) != len(mesh.layout.shape):
        raise EvalError("distributed evaluation requires one attribute per mesh axis")
    if not all(isinstance(attr, (Split, Partial, Broadcast)) for attr in layout.attrs):
        raise EvalError("distributed evaluation does not support Dynamic ownership")
    mapping = layout_axis_to_tensor_axis(layout.layout.shape, type_.shape)
    targets = tuple(
        mapping[attr.axis] if isinstance(attr, Split) else None for attr in layout.attrs
    )
    attrs = tuple(
        Split(target) if target is not None else attr for target, attr in zip(targets, layout.attrs)
    )
    split_axes = [attr.axis for attr in attrs if isinstance(attr, Split)]
    if len(set(split_axes)) != len(split_axes):
        raise EvalError("distributed evaluation supports one mesh split per logical tensor axis")
    for mesh_axis, attr in enumerate(layout.attrs):
        if isinstance(attr, Split):
            extent = layout.layout.shape[attr.axis]
            preceding = layout.layout.shape[mapping.index(targets[mesh_axis]):attr.axis]
            if any(size != 1 for size in preceding):
                raise EvalError(
                    "distributed evaluation requires contiguous partitions at leading layout factors"
                )
            if isinstance(extent, int) and extent % mesh.layout.shape[mesh_axis]:
                raise EvalError("split layout factor is not divisible by its mesh extent")
    return attrs


def _slices(type_, mesh, coord, bindings):
    shape = _shape(type_, bindings)
    attrs = _attrs(type_, mesh)
    slices = []
    for tensor_axis, size in enumerate(shape):
        position, count = 0, 1
        for mesh_axis, attr in enumerate(attrs):
            if isinstance(attr, Split) and attr.axis == tensor_axis:
                extent = mesh.layout.shape[mesh_axis]
                position = position * extent + coord[mesh_axis]
                count *= extent
        if size % count:
            raise EvalError(f"logical axis {tensor_axis} extent {size} is not divisible by {count}")
        local = size // count
        slices.append(slice(position * local, (position + 1) * local))
    return tuple(slices)


def distribute(value, type_, bindings):
    """Bind an entry tensor to declared ownership without inventing partial values."""
    if tuple(value.data.shape) != _shape(type_, bindings):
        raise EvalError("logical input shape does not match its distributed declaration")
    if not isinstance(type_.layout, ShardLayout):
        return value
    mesh = type_.layout.mesh
    attrs = _attrs(type_, mesh)
    if any(isinstance(attr, Partial) for attr in attrs):
        raise EvalError("a logical input cannot supply Partial values")
    return DistributedValue(
        tuple(value.data[_slices(type_, mesh, coord, bindings)] for coord in coordinates(mesh)),
        substitute_dims(type_, bindings),
        mesh,
    )


def _local_type(type_, mesh, bindings):
    slices = _slices(type_, mesh, coordinates(mesh)[0], bindings)
    return replace(type_, shape=tuple(s.stop - s.start for s in slices), layout=None)


def reconstruct(value):
    """Assemble split outputs and check replicas; never complete a partial reduction."""
    if isinstance(value, TupleValue):
        return TupleValue(tuple(reconstruct(item) for item in value.elements))
    if not isinstance(value, DistributedValue):
        return value
    if any(isinstance(attr, Partial) for attr in _attrs(value.type, value.mesh)):
        raise EvalError(
            "distributed output is Partial; an explicit reduction collective is required"
        )
    output = torch.empty(
        value.type.shape, dtype=value.shards[0].dtype, device=value.shards[0].device
    )
    written = set()
    for coord, shard in zip(coordinates(value.mesh), value.shards):
        slices = _slices(value.type, value.mesh, coord, {})
        if tuple(shard.shape) != tuple(s.stop - s.start for s in slices):
            raise EvalError("distributed output does not match the declared shard shape")
        key = tuple((s.start, s.stop) for s in slices)
        if key in written:
            try:
                torch.testing.assert_close(output[slices], shard, rtol=0, atol=0, equal_nan=True)
            except AssertionError as error:
                raise EvalError("distributed output has inconsistent Broadcast replicas") from error
        else:
            output[slices] = shard
            written.add(key)
    return TensorValue(output, value.type)


def _reshard(ctx, mesh, value):
    source_attrs = _attrs(value.type, mesh)
    dest_attrs = _attrs(ctx.result_type, mesh)
    if any(isinstance(attr, Partial) for attr in dest_attrs) and dest_attrs != source_attrs:
        raise EvalError("Reshard cannot create or complete Partial ownership; use a collective")
    shards = (
        value.shards
        if isinstance(value, DistributedValue)
        else (value.data,) * len(coordinates(mesh))
    )
    result = []
    for coord, shard in zip(coordinates(mesh), shards):
        source = _slices(value.type, mesh, coord, ctx.dim_bindings)
        dest = _slices(ctx.result_type, mesh, coord, ctx.dim_bindings)
        for before, after in zip(source_attrs, dest_attrs):
            if isinstance(before, Partial) and before != after:
                raise EvalError("Reshard cannot complete Partial ownership; use a collective")
        if any(new.start < old.start or new.stop > old.stop for old, new in zip(source, dest)):
            raise EvalError("Reshard requires cross-device data; use an explicit collective")
        selection = tuple(
            slice(new.start - old.start, new.stop - old.start) for old, new in zip(source, dest)
        )
        result.append(shard[selection])
    return DistributedValue(tuple(result), substitute_dims(ctx.result_type, ctx.dim_bindings), mesh)


def _collective(ctx, value):
    mesh = value.mesh
    axis = ctx.op.mesh_axis
    coords = coordinates(mesh)
    result_type = substitute_dims(ctx.result_type, ctx.dim_bindings)
    _attrs(result_type, mesh)
    source_attr = _attrs(value.type, mesh)[axis]
    result = [None] * len(coords)
    for ranks in participant_groups(mesh, axis):
        shards = [value.shards[rank] for rank in ranks]
        if isinstance(ctx.op, AllToAll):
            sends = [torch.tensor_split(shard, len(ranks), dim=ctx.op.tensor_axis) for shard in shards]
            for destination, rank in enumerate(ranks):
                result[rank] = torch.cat([send[destination] for send in sends], dim=source_attr.axis)
        elif isinstance(ctx.op, AllGather):
            gathered = torch.cat(shards, dim=source_attr.axis)
            for rank in ranks:
                result[rank] = gathered
        else:
            combined = shards[0].clone()
            for shard in shards[1:]:
                if source_attr.reduction == "sum":
                    combined = combined + shard
                elif source_attr.reduction == "max":
                    combined = torch.maximum(combined, shard)
                else:
                    combined = torch.minimum(combined, shard)
            if isinstance(ctx.op, ReduceScatter):
                pieces = torch.tensor_split(combined, len(ranks), dim=ctx.op.tensor_axis)
                for rank, piece in zip(ranks, pieces):
                    result[rank] = piece
            else:
                for rank in ranks:
                    result[rank] = combined
    return DistributedValue(tuple(result), result_type, mesh)


def evaluate_distributed_op(ctx, handler):
    """Dispatch collectives collectively and supported local handlers per participant."""
    if isinstance(ctx.op, TupleGetItem):
        return handler(ctx)
    distributed = [arg for arg in ctx.args if isinstance(arg, DistributedValue)]
    mesh = ctx.mesh or (distributed[0].mesh if distributed else None)
    if mesh is None:
        result_layout = getattr(ctx.result_type, "layout", None)
        if isinstance(result_layout, ShardLayout):
            mesh = result_layout.mesh
    if mesh is None:
        return handler(ctx)
    device_mesh_shape(mesh)
    if any(arg.mesh != mesh for arg in distributed):
        raise EvalError("distributed operands must use the current device mesh")
    if isinstance(ctx.op, Reshard):
        return _reshard(ctx, mesh, ctx.args[0])
    if isinstance(ctx.op, (AllToAllDispatch, AllToAllCombine)):
        return handler(ctx)
    if isinstance(ctx.op, (AllReduce, AllGather, ReduceScatter, AllToAll)):
        if not isinstance(ctx.args[0], DistributedValue):
            raise EvalError("collective requires distributed input")
        return _collective(ctx, ctx.args[0])
    if not isinstance(ctx.op, (Binary, Unary, MatMul, Cast, Transpose, Reduce, ReLU)):
        raise EvalError(
            f"distributed local semantics are not supported for {type(ctx.op).__name__}"
        )
    if isinstance(ctx.op, Reduce):
        reduced = {axis % len(ctx.args[0].type.shape) for axis in ctx.op.axes}
        if any(
            isinstance(attr, Split) and attr.axis in reduced
            for attr in _attrs(ctx.args[0].type, mesh)
        ):
            raise EvalError(
                "distributed Reduce over a Split axis needs explicit local reduction semantics"
            )
    result_type = _local_type(ctx.result_type, mesh, ctx.dim_bindings)
    shards = []
    for rank in range(len(coordinates(mesh))):
        args = tuple(
            TensorValue(arg.shards[rank], _local_type(arg.type, mesh, ctx.dim_bindings))
            if isinstance(arg, DistributedValue)
            else arg
            for arg in ctx.args
        )
        result = handler(replace(ctx, args=args, result_type=result_type))
        if not isinstance(result, TensorValue) or tuple(result.data.shape) != result_type.shape:
            raise EvalError("local operation result does not match the declared shard shape")
        shards.append(result.data)
    return DistributedValue(tuple(shards), substitute_dims(ctx.result_type, ctx.dim_bindings), mesh)
