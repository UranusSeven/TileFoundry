"""Concrete device ownership projected onto logical tensor shapes."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

from tilefoundry.ir.hir.sharding.collective import device_mesh_shape
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.shard import Layout, ShardLayout
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    Split,
    layout_axis_to_tensor_axis,
    shard_layout_of,
    split_target_axes,
)

from .errors import AnalysisError


def mesh_ranks(mesh):
    """Return physical rank IDs and their coordinates in a contiguous device mesh."""
    try:
        shape = device_mesh_shape(mesh)
    except ValueError as error:
        raise AnalysisError(f"engine: {error}") from error
    return tuple(enumerate(product(*(range(size) for size in shape))))


def groups(mesh, axis):
    grouped = {}
    for rank, coord in mesh_ranks(mesh):
        key = coord[:axis] + coord[axis + 1 :]
        grouped.setdefault(key, []).append(rank)
    return tuple(tuple(group) for group in grouped.values())


def logical_attrs(type_):
    layout = type_.layout
    if not isinstance(layout, ShardLayout):
        return ()
    if not all(isinstance(attr, (Broadcast, Partial, Split)) for attr in layout.attrs):
        raise AnalysisError("engine: Dynamic ownership requires a supported bounded representation")
    targets = split_target_axes(layout, type_.shape)
    mapping = layout_axis_to_tensor_axis(layout.layout.shape, type_.shape)
    selected = [target for target in targets if target is not None]
    if len(set(selected)) != len(selected):
        raise AnalysisError("engine: only one mesh split per logical tensor axis is supported")
    for attr, target in zip(layout.attrs, targets):
        if isinstance(attr, Split) and any(
            size != 1 for size in layout.layout.shape[mapping.index(target):attr.axis]
        ):
            raise AnalysisError("engine: contiguous partitions at leading layout factors are required")
    return tuple(
        Split(target) if target is not None else attr for target, attr in zip(targets, layout.attrs)
    )


def local_type(type_, rank):
    """Project one rank's logical tensor shapes without materializing any tensor data."""
    if isinstance(type_, TupleType):
        fields = tuple(local_type(field, rank) for field in type_.fields)
        if any(field is None for field in fields):
            return None
        return TupleType(fields=fields)
    if not isinstance(type_, TensorType):
        return type_
    if any(type(size) is not int or size < 0 for size in type_.shape):
        raise AnalysisError("engine: tensor dimensions must be concrete nonnegative integers")
    if type_.layout is None or isinstance(type_.layout, Layout):
        return replace(type_, layout=None)
    if not isinstance(type_.layout, ShardLayout):
        if shard_layout_of(type_.layout) is None:
            return replace(type_, layout=None)
        raise AnalysisError("engine: composed tensor placement is not supported")
    mesh = type_.layout.mesh
    participants = mesh_ranks(mesh)
    if rank >= len(participants):
        return None
    shape = list(type_.shape)
    for axis, attr in enumerate(logical_attrs(type_)):
        if isinstance(attr, Split):
            extent = mesh.layout.shape[axis]
            if shape[attr.axis] % extent:
                raise AnalysisError("engine: tensor shard is not divisible by its mesh extent")
            shape[attr.axis] //= extent
    return replace(type_, shape=tuple(shape), layout=None)


def local_selection(type_, rank):
    """State each contiguous logical interval owned by a rank."""
    attrs = logical_attrs(type_)
    if not attrs:
        return tuple((0, size) for size in type_.shape)
    mesh = type_.layout.mesh
    coordinates = dict(mesh_ranks(mesh))
    if rank not in coordinates:
        return None
    positions = [0] * len(type_.shape)
    divisors = [1] * len(type_.shape)
    for axis, attr in enumerate(attrs):
        if isinstance(attr, Split):
            positions[attr.axis] = (
                positions[attr.axis] * mesh.layout.shape[axis] + coordinates[rank][axis]
            )
            divisors[attr.axis] *= mesh.layout.shape[axis]
    return tuple(
        (position * (size // divisor), (position + 1) * (size // divisor))
        for size, position, divisor in zip(type_.shape, positions, divisors)
    )
