"""Rank-local reference execution of bounded expert routing."""

from __future__ import annotations

from itertools import product

import torch

from tilefoundry.evaluator.distributed import (
    DistributedValue,
    _local_type,
    participant_groups,
)
from tilefoundry.evaluator.value import EvalError, TupleValue, to_torch_dtype
from tilefoundry.ir.types.substitute import substitute_dims


def _operands(ctx):
    if not all(isinstance(arg, DistributedValue) for arg in ctx.args):
        raise EvalError("routed all-to-all requires distributed operands")
    mesh = ctx.args[0].mesh
    for arg in ctx.args:
        if arg.mesh != mesh:
            raise EvalError("routed all-to-all operands must share a device mesh")
        expected = _local_type(arg.type, mesh, ctx.dim_bindings).shape
        if any(tuple(shard.shape) != expected for shard in arg.shards):
            raise EvalError("routed all-to-all operand does not match its declared shard shape")
    return mesh


def _allocate(type_, mesh, bindings, source, fill=0):
    shape = _local_type(type_, mesh, bindings).shape
    return tuple(
        torch.full(shape, fill, dtype=to_torch_dtype(type_.dtype), device=shard.device)
        for shard in source.shards
    )


def dispatch(ctx):
    """Send each active route to its expert, preserving source and top-k order."""
    mesh = _operands(ctx)
    x, indices, weights = ctx.args
    experts, capacity = ctx.op.num_experts, ctx.op.capacity
    degree = mesh.layout.shape[ctx.op.mesh_axis]
    local_experts = experts // degree
    for shard in indices.shards:
        if torch.any((shard < -1) | (shard >= experts)).item():
            raise EvalError(f"expert index must be -1 or in [0, {experts})")
    types = tuple(substitute_dims(field, ctx.dim_bindings) for field in ctx.result_type.fields)
    buffers = tuple(
        _allocate(type_, mesh, ctx.dim_bindings, x, -1 if i == 2 else 0)
        for i, type_ in enumerate(types)
    )
    recv_x, recv_weights, source_indices, counts = buffers
    for ranks in participant_groups(mesh, ctx.op.mesh_axis):
        for source_coord, source_rank in enumerate(ranks):
            data, routes, scales = (
                x.shards[source_rank],
                indices.shards[source_rank],
                weights.shards[source_rank],
            )
            for batch in product(*(range(size) for size in data.shape[:-2])):
                for token in range(data.shape[-2]):
                    for choice in range(routes.shape[-1]):
                        expert = int(routes[(*batch, token, choice)].item())
                        if expert == -1:
                            continue
                        destination = ranks[expert // local_experts]
                        local_expert = expert % local_experts
                        expert_position = (*batch, local_expert)
                        slot = int(counts[destination][expert_position].item())
                        if slot >= capacity:
                            raise EvalError(
                                f"all-to-all dispatch capacity {capacity} exceeded for expert {expert} "
                                f"on rank {destination}, batch {batch}"
                            )
                        row = (*expert_position, slot)
                        recv_x[destination][row] = data[(*batch, token)]
                        recv_weights[destination][(*row, 0)] = scales[(*batch, token, choice)]
                        source_indices[destination][row] = source_coord * data.shape[-2] + token
                        counts[destination][expert_position] = slot + 1
    return TupleValue(
        tuple(DistributedValue(shards, type_, mesh) for shards, type_ in zip(buffers, types))
    )


def combine(ctx):
    """Accumulate active, already-weighted expert rows at their source token owners."""
    mesh = _operands(ctx)
    x, indices, counts, source = ctx.args
    type_ = substitute_dims(ctx.result_type, ctx.dim_bindings)
    output = _allocate(type_, mesh, ctx.dim_bindings, source)
    capacity = x.shards[0].shape[-2]
    local_tokens = output[0].shape[-2]
    total_tokens = local_tokens * mesh.layout.shape[ctx.op.mesh_axis]
    for rank_counts in counts.shards:
        if torch.any((rank_counts < 0) | (rank_counts > capacity)).item():
            raise EvalError(f"expert counts must be in [0, {capacity}]")
    for ranks in participant_groups(mesh, ctx.op.mesh_axis):
        for expert_rank in ranks:
            data = x.shards[expert_rank]
            for batch in product(*(range(size) for size in data.shape[:-3])):
                for expert in range(data.shape[-3]):
                    expert_position = (*batch, expert)
                    count = int(counts.shards[expert_rank][expert_position].item())
                    for slot in range(count):
                        row = (*expert_position, slot)
                        token = int(indices.shards[expert_rank][row].item())
                        if not 0 <= token < total_tokens:
                            raise EvalError(f"active source index must be in [0, {total_tokens})")
                        destination = ranks[token // local_tokens]
                        target = (*batch, token % local_tokens)
                        output[destination][target] += data[row]
    return DistributedValue(output, type_, mesh)
