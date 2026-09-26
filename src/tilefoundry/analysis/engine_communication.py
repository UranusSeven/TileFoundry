"""Explicit payload and local-service models for device collectives."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

from tilefoundry.ir.hir.sharding.alltoall import AllToAllCombine, AllToAllDispatch
from tilefoundry.ir.hir.sharding.collective import AllGather, AllReduce, AllToAll, ReduceScatter
from tilefoundry.ir.types import FloatDType, numel, tensor_bytes

from .engine_geometry import groups, local_type
from .engine_metadata import EngineTransfer, EngineWork
from .engine_profile import RoutingProfile
from .errors import AnalysisError

COLLECTIVES = (AllGather, AllReduce, ReduceScatter, AllToAll, AllToAllDispatch, AllToAllCombine)


@dataclass(frozen=True)
class CommunicationPlan:
    """One collective's payload schedule and declared routing certainty."""

    work: tuple[EngineWork, ...]
    transfers: tuple[EngineTransfer, ...]
    scratch: tuple[tuple[int, int], ...]
    traffic_kind: str = "exact"
    routing_safe: bool | None = True
    diagnostics: tuple[str, ...] = ()


def _reduction_work(type_, amount):
    if isinstance(type_.dtype, FloatDType):
        return ((type_.dtype.name, amount),), ()
    return (), (("integer", amount),)


def communication_plan(call, ranks, rank_count, profile=None):
    """Price a primitive from its typed payload and group, independently of composed kernels."""
    if isinstance(call.target, (AllToAllDispatch, AllToAllCombine)):
        return _routed(call, ranks, rank_count, profile)
    source = call.args[0].type
    mesh = source.layout.mesh
    work, transfers, scratch = [], [], []
    for peers in groups(mesh, call.target.mesh_axis):
        degree = len(peers)
        for position, rank in enumerate(peers):
            held, output = local_type(source, rank), local_type(call.type, rank)
            read, written = tensor_bytes(held), tensor_bytes(output)
            reductions = 0
            if isinstance(call.target, (AllReduce, ReduceScatter)):
                reductions = ceil(numel(held) * (degree - 1) / degree)
            flops, operations = _reduction_work(held, reductions)
            work.append(EngineWork(rank, flops, operations, read, written))
            scratch.append((rank, max(read, written)))
            for phase in range(degree - 1):
                if isinstance(call.target, AllToAll):
                    destination = peers[(position + phase + 1) % degree]
                    amount = ceil(numel(held) // degree * held.dtype.bit_width / 8)
                else:
                    destination = peers[(position + 1) % degree]
                    if isinstance(call.target, AllGather):
                        amount = read
                    else:
                        base, extra = divmod(numel(held), degree)
                        chunk = (position - phase) % degree
                        amount = ceil((base + (chunk < extra)) * held.dtype.bit_width / 8)
                if amount:
                    transfers.append(EngineTransfer(rank, destination, amount, phase))
            if isinstance(call.target, AllReduce):
                base, extra = divmod(numel(held), degree)
                for phase in range(degree - 1):
                    chunk = (position + 1 - phase) % degree
                    amount = ceil((base + (chunk < extra)) * held.dtype.bit_width / 8)
                    if amount:
                        transfers.append(
                            EngineTransfer(
                                rank, peers[(position + 1) % degree], amount, degree - 1 + phase
                            )
                        )
    return CommunicationPlan(tuple(work), tuple(transfers), tuple(scratch))


def _validate_profile(
    call, ranks, rank_count, profile, peers, expert_local_types, source_local_types
):
    if len(profile.peer_tokens) != rank_count:
        raise AnalysisError("engine: routing profile rank count does not match the deployment")
    member_group = {rank: group for group in peers for rank in group}
    for source in range(rank_count):
        if source not in ranks:
            if any(profile.peer_routes[source]) or profile.expert_counts[source]:
                raise AnalysisError("engine: routing profile assigns work to an inactive rank")
            continue
        tokens = prod(source_local_types[source].shape[:-1])
        for destination in range(rank_count):
            unique = profile.peer_tokens[source][destination]
            if unique and destination not in member_group[source]:
                raise AnalysisError("engine: routing profile crosses independent mesh groups")
            if unique > tokens:
                raise AnalysisError("engine: peer token count exceeds source tokens")
        if isinstance(call.target, AllToAllDispatch):
            choices = call.args[1].type.shape[-1]
            if sum(profile.peer_routes[source]) > tokens * choices:
                raise AnalysisError("engine: routing profile exceeds available top-k choices")
            for unique, expanded in zip(profile.peer_tokens[source], profile.peer_routes[source]):
                if expanded > unique * choices:
                    raise AnalysisError(
                        "engine: peer routes exceed the selected tokens' top-k choices"
                    )
    safe = True
    for rank in ranks:
        held = expert_local_types[rank]
        expected = prod(held.shape[:-2])
        if len(profile.expert_counts[rank]) != expected:
            raise AnalysisError("engine: expert_counts do not match local batch/expert dimensions")
        safe &= all(count <= held.shape[-2] for count in profile.expert_counts[rank])
    return safe


def _routed(call, ranks, rank_count, profile: RoutingProfile | None):
    dispatch = isinstance(call.target, AllToAllDispatch)
    original = call.args[0].type if dispatch else call.args[3].type
    mesh = original.layout.mesh
    peers = groups(mesh, call.target.mesh_axis)
    source_types = {rank: local_type(original, rank) for rank in ranks}
    expert_global = call.type.fields[0] if dispatch else call.args[0].type
    expert_types = {rank: local_type(expert_global, rank) for rank in ranks}
    capacity = expert_global.shape[-2]
    diagnostics = []
    if profile is not None:
        safe = _validate_profile(
            call, ranks, rank_count, profile, peers, expert_types, source_types
        )
        tokens, routes = profile.peer_tokens, profile.peer_routes
        if not safe:
            diagnostics.append("routing profile exceeds an expert's declared dispatch capacity")
    else:
        tokens = [[0] * rank_count for _ in range(rank_count)]
        routes = [[0] * rank_count for _ in range(rank_count)]
        choices = call.args[1].type.shape[-1] if dispatch else None
        for group in peers:
            for source in group:
                available = prod(source_types[source].shape[:-1])
                for destination in group:
                    slots = prod(expert_types[destination].shape[:-1])
                    tokens[source][destination] = min(available, slots)
                    routes[source][destination] = (
                        min(available * choices, slots) if dispatch else slots
                    )
        safe = True if dispatch and capacity >= original.shape[-2] * choices else None
        diagnostics.append(
            "routing traffic uses independent peer upper bounds; live token/expert counts were not supplied"
        )
        if safe is None:
            diagnostics.append(
                "routing capacity is unresolved without expert counts or a worst-case-safe capacity"
            )
    transfers, work, scratch = [], [], []
    for group in peers:
        for source in group:
            for destination in group:
                if source == destination:
                    continue
                if dispatch:
                    hidden_bytes = ceil(original.shape[-1] * original.dtype.bit_width / 8)
                    metadata_bytes = ceil(call.args[1].type.dtype.bit_width / 8) + ceil(
                        call.args[2].type.dtype.bit_width / 8
                    )
                    amount = (
                        tokens[source][destination] * (hidden_bytes + 8)
                        + routes[source][destination] * metadata_bytes
                    )
                    sender, receiver = source, destination
                else:
                    amount = tokens[source][destination] * (
                        ceil(expert_global.shape[-1] * expert_global.dtype.bit_width / 8)
                        + ceil(call.args[1].type.dtype.bit_width / 8)
                    )
                    sender, receiver = destination, source
                if amount:
                    transfers.append(EngineTransfer(sender, receiver, amount, 0))
    for rank in ranks:
        outputs = local_type(call.type, rank)
        inputs = [local_type(arg.type, rank) for arg in call.args]
        if dispatch:
            read = sum(tensor_bytes(type_) for type_ in inputs)
            written = tensor_bytes(outputs)
            flops, operations = (), (("integer", numel(inputs[1])),)
        else:
            live = (
                sum(profile.expert_counts[rank]) if profile else prod(expert_types[rank].shape[:-1])
            )
            read = live * (
                ceil(expert_global.shape[-1] * expert_global.dtype.bit_width / 8)
                + ceil(inputs[1].dtype.bit_width / 8)
            ) + tensor_bytes(inputs[2])
            written = tensor_bytes(outputs)
            local_routes = sum(row[rank] for row in routes)
            local_tokens = sum(row[rank] for row in tokens)
            incoming = sum(tokens[rank])
            additions = (max(local_routes - local_tokens, 0) + incoming) * expert_global.shape[-1]
            flops, operations = _reduction_work(expert_types[rank], additions)
        work.append(EngineWork(rank, flops, operations, read, written))
        scratch.append((rank, max(sum(tensor_bytes(type_) for type_ in inputs[:3]), written)))
    return CommunicationPlan(
        tuple(work),
        tuple(transfers),
        tuple(scratch),
        "profile" if profile else "upper-bound",
        safe,
        tuple(diagnostics),
    )
