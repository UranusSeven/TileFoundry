"""Compose primitive device work, communication and conservative HBM allocation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from tilefoundry.ir.core import attach_metadata, describe_expr
from tilefoundry.ir.hir.sharding.alltoall import AllToAllCombine, AllToAllDispatch
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import TensorType, TupleType, tensor_bytes, tensor_types
from tilefoundry.ir.types.shard.shard_layout import Partial, ShardLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.target import UnsupportedCapabilityError
from tilefoundry.visitor_registry.contexts import CostContext
from tilefoundry.visitor_registry.visitors import CostEvaluator

from .engine_communication import COLLECTIVES, CommunicationPlan, communication_plan
from .engine_geometry import groups, local_selection, local_type, logical_attrs, mesh_ranks
from .engine_metadata import (
    EngineBuffer,
    EngineMetadata,
    EngineOperation,
    EngineRank,
    EngineRates,
    EngineWork,
)
from .engine_profile import EngineOptions
from .errors import AnalysisError
from .facts import MemoryHierarchyFacts, PerformanceServiceFacts


def _maximum(values):
    values = tuple(values)
    return None if any(value is None for value in values) else max(values, default=0)


def _add(left, right):
    return None if left is None or right is None else left + right


def _conjunction(values):
    values = tuple(values)
    return False if False in values else None if None in values else True


def _aliases_input(call):
    return isinstance(call.target, (Slice, Reshape, Local)) or (
        isinstance(call.target, Reshard) and call.args[0].type.storage == call.type.storage
    )


@dataclass(eq=False)
class _Buffer:
    name: str
    kind: str
    sizes: dict[int, int]
    begin: int
    end: int
    pinned: bool = False


@dataclass
class _Value:
    type: object
    buffers: tuple[_Buffer, ...] = ()
    fields: tuple[_Value, ...] = ()
    dependencies: tuple[int, ...] = ()
    routing: str | None = None

    def roots(self):
        return set(self.buffers).union(*(value.roots() for value in self.fields))


@dataclass
class _Event:
    index: int
    operation: str
    source: str
    ranks: tuple[int, ...]
    dependencies: tuple[int, ...]
    work: tuple[EngineWork, ...]
    repeats: int
    communication: CommunicationPlan | None = None
    groups: tuple[tuple[int, ...], ...] = ()


@dataclass
class _Loop:
    body: list
    trips: int


class _Planner(ExprVisitor):
    """Build one static event tree and bounded allocation intervals."""

    def __init__(self, function, rank_count, options):
        super().__init__(root_function=function)
        self.rank_count, self.options = rank_count, options
        self.ranks = tuple(range(rank_count))
        self.buffers, self.events = [], []
        self.block = []
        self.root = self.block
        self.position, self.repeats = 0, 1
        self.counts = defaultdict(int)
        self.routing = {profile.operation: profile for profile in options.routing}
        self.used_profiles, self.route_safety = set(), {}
        self.diagnostics = []
        declared = {param.name: param for param in function.params}
        for name in options.workload.state_inputs:
            if name not in declared or declared[name].is_const:
                raise AnalysisError(f"engine: state input {name!r} is not a non-constant parameter")
        for param in function.params:
            self.bind(param, self.parameter(param))

    def bind(self, expr, value):
        self._memo[id(expr)] = (expr, value)

    def allocate(self, type_, name, ranks, kind="temporary", pinned=False):
        if isinstance(type_, TupleType):
            return _Value(
                type_,
                fields=tuple(
                    self.allocate(field, f"{name}[{i}]", ranks, kind, pinned)
                    for i, field in enumerate(type_.fields)
                ),
            )
        if isinstance(type_, TensorType) and isinstance(type_.layout, ShardLayout):
            if len(mesh_ranks(type_.layout.mesh)) > self.rank_count:
                raise AnalysisError(f"engine: placement of {name!r} exceeds the deployment")
        sizes = {}
        for rank in ranks:
            held = local_type(type_, rank)
            if isinstance(held, TensorType) and held.storage is StorageKind.GMEM:
                sizes[rank] = tensor_bytes(held)
        if any(sizes.values()):
            buffer = _Buffer(name, kind, sizes, self.position, self.position, pinned)
            self.buffers.append(buffer)
            return _Value(type_, (buffer,))
        return _Value(type_)

    def parameter(self, var):
        kind = (
            "weights"
            if var.is_const
            else "state"
            if var.name in self.options.workload.state_inputs
            else "input"
        )
        value = self.allocate(var.type, var.name, self.ranks, kind, True)
        for buffer in value.roots():
            buffer.begin = 0
        return value

    def visit_leaf_Var(self, var, _operands, ctx):
        if not var.is_const:
            raise AnalysisError(f"engine: unbound variable {var.name!r}")
        return self.parameter(var)

    def visit_leaf_Constant(self, const, _operands, ctx):
        return self.allocate(const.type, "constant", self.ranks, "constant", True)

    def visit_leaf_Tuple(self, value, fields, ctx):
        dependencies = tuple(dict.fromkeys(dep for field in fields for dep in field.dependencies))
        return _Value(value.type, fields=fields, dependencies=dependencies)

    def visit_MeshRegion(self, region, mesh):
        args = tuple(self.visit(arg, mesh) for arg in region.args)
        participants = mesh_ranks(region.mesh)
        if participants and participants[-1][0] >= self.rank_count:
            raise AnalysisError("engine: mesh exceeds deployment rank count")
        if mesh is not None and mesh != region.mesh:
            raise AnalysisError("engine: nested different device meshes are not supported")
        for param, value in zip(region.params, args):
            self.bind(param, value)
        return self.visit(region.body, region.mesh)

    def visit_LoopRegion(self, region, mesh):
        initial = tuple(self.visit(arg, mesh) for arg in region.init_args)
        if (
            not all(type(value) is int for value in (region.start, region.extent, region.step))
            or region.step <= 0
        ):
            raise AnalysisError(
                "engine: loops require uniform concrete start, stop and positive step"
            )
        trips = len(range(region.start, region.extent, region.step))
        if not trips:
            if not initial:
                raise AnalysisError("engine: empty loop without carried results")
            return initial[0] if len(initial) == 1 else _Value(region.type, fields=initial)
        parent, repeats, begin = self.block, self.repeats, self.position
        loop = _Loop([], trips)
        parent.append(loop)
        self.block, self.repeats = loop.body, repeats * trips
        self.bind(region.induction_var, _Value(region.induction_var.type))
        for phi, value in zip(region.carried_args, initial):
            self.bind(phi, value)
        results = (
            tuple(self.visit(value, mesh) for value in region.yield_values)
            if initial
            else (self.visit(region.body, mesh),)
        )
        if trips > 1:
            roots = set().union(*(value.roots() for value in results))
            for buffer in roots:
                if buffer.begin >= begin and not buffer.pinned:
                    self.buffers.append(
                        _Buffer(
                            buffer.name + ":carry-spare",
                            "loop-carry",
                            dict(buffer.sizes),
                            begin,
                            self.position,
                        )
                    )
        self.block, self.repeats = parent, repeats
        dependencies = tuple(dict.fromkeys(dep for value in results for dep in value.dependencies))
        return (
            results[0]
            if len(results) == 1
            else _Value(region.type, fields=results, dependencies=dependencies)
        )

    def _reshard(self, call, ranks):
        source, destination = call.args[0].type, call.type
        before, after = logical_attrs(source), logical_attrs(destination)
        if before != after and any(isinstance(attr, Partial) for attr in (*before, *after)):
            raise AnalysisError(
                f"engine: {describe_expr(call)}: Reshard cannot create or complete Partial ownership"
            )
        for rank in ranks:
            old, new = local_selection(source, rank), local_selection(destination, rank)
            if new is not None and (
                old is None or any(b[0] < a[0] or b[1] > a[1] for a, b in zip(old, new))
            ):
                raise AnalysisError(
                    f"engine: {describe_expr(call)}: Reshard requires explicit cross-device communication"
                )

    def visit_leaf_Call(self, call, args, mesh):
        if isinstance(call.target, TupleGetItem):
            return args[0].fields[call.target.index]
        placed = call.type.fields[0] if isinstance(call.type, TupleType) else call.type
        scope = mesh or (
            placed.layout.mesh if isinstance(getattr(placed, "layout", None), ShardLayout) else None
        )
        ranks = tuple(rank for rank, _ in mesh_ranks(scope)) if scope is not None else self.ranks
        if any(rank >= self.rank_count for rank in ranks):
            raise AnalysisError(f"engine: {describe_expr(call)}: execution scope exceeds the deployment")
        self.position += 1
        for value in args:
            for buffer in value.roots():
                buffer.end = max(buffer.end, self.position)
        name = type(call.target).__name__
        operation = f"{name}:{self.counts[name]}"
        self.counts[name] += 1
        if isinstance(call.target, Reshard):
            self._reshard(call, ranks)
        plan = None
        if isinstance(call.target, COLLECTIVES):
            inherited = args[1].routing if isinstance(call.target, AllToAllCombine) else None
            profile = self.routing.get(operation) or self.routing.get(inherited)
            if profile:
                self.used_profiles.add(profile.operation)
            plan = communication_plan(call, ranks, self.rank_count, profile)
            if (
                isinstance(call.target, AllToAllCombine)
                and inherited in self.route_safety
                and profile is None
            ):
                plan = replace(
                    plan,
                    routing_safe=self.route_safety[inherited],
                    diagnostics=tuple(
                        note for note in plan.diagnostics if "capacity is unresolved" not in note
                    ),
                )
            self.route_safety[operation] = plan.routing_safe
            self.diagnostics.extend(f"{operation}: {note}" for note in plan.diagnostics)
            work = plan.work
            for rank, size in plan.scratch:
                self.buffers.append(
                    _Buffer(
                        operation + ":scratch",
                        "communication",
                        {rank: size},
                        self.position,
                        self.position,
                    )
                )
        else:
            work = tuple(self._local_cost(call, rank) for rank in ranks)
        alias = _aliases_input(call)
        result = (
            _Value(call.type, buffers=args[0].buffers, fields=args[0].fields)
            if alias
            else self.allocate(call.type, operation, ranks)
        )
        dependencies = tuple(dict.fromkeys(dep for value in args for dep in value.dependencies))
        event = _Event(
            len(self.events),
            operation,
            describe_expr(call),
            ranks,
            dependencies,
            work,
            self.repeats,
            plan,
            groups(call.args[0].type.layout.mesh, call.target.mesh_axis) if plan else (),
        )
        self.events.append(event)
        self.block.append(event)

        def ready(value):
            value.dependencies = (event.index,)
            if isinstance(call.target, AllToAllDispatch):
                value.routing = operation
            for field_ in value.fields:
                ready(field_)

        ready(result)
        return result

    def _local_cost(self, call, rank):
        inputs = [local_type(arg.type, rank) for arg in call.args]
        output = local_type(call.type, rank)
        if output is None or any(type_ is None for type_ in inputs):
            raise AnalysisError(
                f"engine: {describe_expr(call)}: operand unavailable on rank {rank}"
            )
        selected = {id(arg): type_ for arg, type_ in zip(call.args, inputs)}
        selected[id(call)] = output
        try:
            cost = CostEvaluator().visit(
                call, CostContext(selected_types=selected, selected_output_type=output)
            )
        except (ValueError, TypeError) as error:
            raise AnalysisError(f"engine: {describe_expr(call)}: {error}") from error
        read, written = 0, 0
        for type_, traffic in zip((*inputs, output), cost.traffic):
            leaves = tensor_types(type_)
            if leaves and all(leaf.storage is StorageKind.GMEM for leaf in leaves):
                read += traffic.read
                written += traffic.write
        if (
            not _aliases_input(call)
            and inputs
            and isinstance(inputs[0], TensorType)
            and isinstance(output, TensorType)
        ):
            source = inputs[0]
            if (
                source.shape == output.shape
                and source.dtype == output.dtype
                and source.storage is StorageKind.GMEM
                and output.storage is StorageKind.GMEM
                and cost.traffic[-1].write < tensor_bytes(output)
            ):
                read += tensor_bytes(source)
                written += tensor_bytes(output) - cost.traffic[-1].write
        if cost.sent:
            raise AnalysisError(
                f"engine: {describe_expr(call)}: implicit communication needs an explicit device collective"
            )
        return EngineWork(
            rank,
            tuple((dtype.name, count) for dtype, count in cost.flops.items()),
            tuple(cost.service.items()),
            read,
            written,
        )


class _Schedule:
    """Serialize shared rank/link resources and compact uniform loop repetition."""

    def __init__(self, rank_count, services, options):
        self.rank_count, self.services, self.options = rank_count, services, options
        self.clocks = {f"rank:{rank}": 0 for rank in range(rank_count)}
        self.finished, self.records, self.diagnostics = {}, {}, []
        self.routes = {
            (route.source, route.destination): route for route in options.deployment.routes
        }

    def duration(self, work):
        values = []
        for name, amount in work.flops:
            if not amount:
                continue
            rate = (
                next((rate for dtype, rate in self.services.unit_flops if dtype.name == name), None)
                if self.services
                else None
            )
            values.append(self.rate(amount, rate, f"compute rate for {name}"))
        for name, amount in work.other_ops:
            if amount:
                values.append(
                    self.rate(
                        amount,
                        self.services.ops(name) if self.services else None,
                        f"service rate for {name}",
                    )
                )
        compute = None if None in values else sum(values)
        amount = work.read_bytes + work.write_bytes
        memory = self.rate(
            amount, self.services.bandwidth("gmem") if self.services else None, "HBM bandwidth"
        )
        return _maximum((compute, memory))

    def rate(self, amount, rate, label):
        if not amount:
            return 0
        if rate is None:
            self.diagnostics.append(f"missing {label}")
            return None
        if type(rate) is not int or rate <= 0:
            raise AnalysisError(f"engine: {label} must be a positive integer")
        return (amount * 1_000_000_000 + rate - 1) // rate

    def run(self, block):
        for item in block:
            if isinstance(item, _Loop):
                start = _maximum(self.clocks.values())
                self.clocks = {key: start for key in self.clocks}
                previous = set(self.records)
                self.run(item.body)
                end = _maximum(self.clocks.values())
                extra = None if start is None or end is None else (end - start) * (item.trips - 1)
                self.clocks = {key: _add(value, extra) for key, value in self.clocks.items()}
                for index in self.records.keys() - previous:
                    self.finished[index] = {
                        rank: _add(value, extra) for rank, value in self.finished[index].items()
                    }
                continue
            self.operation(item)

    def operation(self, event):
        starts, ends = {}, {}
        work = {item.rank: item for item in event.work}
        for rank in event.ranks:
            starts[rank] = _maximum(
                (
                    self.clocks[f"rank:{rank}"],
                    *(self.finished[index].get(rank, 0) for index in event.dependencies),
                )
            )
        if event.communication:
            for group in event.groups:
                start = _maximum(starts[rank] for rank in group)
                for rank in group:
                    starts[rank] = start
        for rank in event.ranks:
            ends[rank] = _add(starts[rank], self.duration(work[rank]))
        transfers = []
        if event.communication:
            for phase in sorted({item.phase for item in event.communication.transfers}):
                base = dict(ends)
                for transfer in event.communication.transfers:
                    if transfer.phase != phase:
                        continue
                    route = self.routes.get((transfer.source, transfer.destination))
                    paths = (
                        tuple(f"network:{name}" for name in route.resources)
                        if route and route.resources
                        else (f"path:{transfer.source}:{transfer.destination}",)
                    )
                    resources = (f"tx:{transfer.source}", f"rx:{transfer.destination}", *paths)
                    start = _maximum(
                        (
                            base[transfer.source],
                            base[transfer.destination],
                            *(self.clocks.get(key, 0) for key in resources),
                        )
                    )
                    if (
                        route is None
                        or route.bandwidth_bytes_per_second is None
                        or route.latency_ns is None
                    ):
                        self.diagnostics.append(
                            f"missing network facts for {transfer.source}->{transfer.destination}"
                        )
                        duration = None
                    else:
                        duration = route.latency_ns + self.rate(
                            transfer.bytes, route.bandwidth_bytes_per_second, "link bandwidth"
                        )
                    end = _add(start, duration)
                    for key in resources:
                        self.clocks[key] = end
                    ends[transfer.source] = _maximum((ends[transfer.source], end))
                    ends[transfer.destination] = _maximum((ends[transfer.destination], end))
                    transfers.append(
                        replace(transfer, resources=resources, start_ns=start, end_ns=end)
                    )
                for group in event.groups:
                    end = _maximum(ends[rank] for rank in group)
                    for rank in group:
                        ends[rank] = end
            for group in event.groups:
                end = _maximum(ends[rank] for rank in group)
                for rank in group:
                    ends[rank] = end
        for rank in event.ranks:
            self.clocks[f"rank:{rank}"] = ends[rank]
        self.finished[event.index] = ends
        self.records[event.index] = EngineOperation(
            event.operation,
            event.source,
            event.ranks,
            event.work,
            tuple(transfers),
            event.repeats,
            event.communication.traffic_kind if event.communication else "exact",
            None if None in starts.values() else min(starts.values(), default=0),
            _maximum(ends.values()),
        )


def analyze_engine(function, context):
    """Report one checked device program under the declared buffer and service model."""
    options = context.options if context.options is not None else EngineOptions()
    if not isinstance(options, EngineOptions):
        raise AnalysisError("engine: options must be EngineOptions")
    if context.topology_level not in (None, "gpu"):
        raise AnalysisError("engine: the selected topology must be gpu")
    topology = next(
        (item for item in context.module.effective_topologies() if item.name == "gpu"), None
    )
    if topology is None or type(topology.size) is not int or topology.size <= 0:
        raise AnalysisError("engine: the module must declare a concrete gpu topology")
    count = topology.size
    if any(device.rank >= count for device in options.deployment.devices) or any(
        route.source >= count or route.destination >= count for route in options.deployment.routes
    ):
        raise AnalysisError("engine: deployment profile names a rank outside the program topology")
    services, capacity, diagnostics = None, None, []
    try:
        services = context.target.get_facts(PerformanceServiceFacts, "gpu")
        if services.unit != "gpu":
            raise AnalysisError("engine: performance rates must be stated per gpu")
    except UnsupportedCapabilityError as error:
        diagnostics.append(str(error))
    try:
        level = context.target.get_facts(MemoryHierarchyFacts).explicit("gmem")
        capacity = level.capacity_bytes if level and level.scope in ("gpu", "device") else None
    except UnsupportedCapabilityError as error:
        diagnostics.append(str(error))
    planner = _Planner(function, count, options)
    result = planner.visit(function.body, None)
    final = planner.position + 1
    returned = result.roots()
    for buffer in planner.buffers:
        if buffer.pinned or buffer in returned:
            buffer.end = final
    unused = planner.routing.keys() - planner.used_profiles
    if unused:
        raise AnalysisError(f"engine: unused routing profiles: {', '.join(sorted(unused))}")
    schedule = _Schedule(count, services, options)
    schedule.run(planner.root)
    records = tuple(schedule.records[index] for index in sorted(schedule.records))
    rank_records = []
    for rank in range(count):
        budget = next((item for item in options.deployment.devices if item.rank == rank), None)
        limit = budget.capacity_bytes if budget and budget.capacity_bytes is not None else capacity
        reserve = budget.reserve_bytes if budget else 0
        buffers = [buffer for buffer in planner.buffers if buffer.sizes.get(rank, 0)]
        points = {0, *(buffer.begin for buffer in buffers), *(buffer.end for buffer in buffers)}
        peak, live_at_peak = reserve, ()
        for point in sorted(points):
            live = tuple(buffer for buffer in buffers if buffer.begin <= point <= buffer.end)
            size = reserve + sum(buffer.sizes[rank] for buffer in live)
            if size > peak:
                peak, live_at_peak = size, live
        kinds = {
            kind: sum(buffer.sizes[rank] for buffer in buffers if buffer.kind == kind)
            for kind in ("weights", "state", "input")
        }
        flops, reads, writes, sent, received = defaultdict(int), 0, 0, 0, 0
        for record in records:
            for work in record.work:
                if work.rank != rank:
                    continue
                reads += work.read_bytes * record.repeats
                writes += work.write_bytes * record.repeats
                for dtype, amount in work.flops:
                    flops[dtype] += amount * record.repeats
            for transfer in record.transfers:
                if transfer.source == rank:
                    sent += transfer.bytes * record.repeats
                if transfer.destination == rank:
                    received += transfer.bytes * record.repeats
        rank_records.append(
            EngineRank(
                rank,
                kinds["weights"],
                kinds["state"],
                kinds["input"],
                sum(buffer.sizes.get(rank, 0) for buffer in returned),
                peak,
                reserve,
                limit,
                None if limit is None else peak <= limit,
                tuple(sorted(flops.items())),
                reads,
                writes,
                sent,
                received,
                tuple(
                    EngineBuffer(buffer.name, buffer.kind, buffer.sizes[rank])
                    for buffer in live_at_peak
                ),
            )
        )
    predicted = _maximum(schedule.clocks.values())
    workload = options.workload
    capacity_fits = _conjunction(item.fits for item in rank_records)
    routing_safe = _conjunction(planner.route_safety.values())
    slo = (
        None
        if workload.latency_budget_ns is None or predicted is None
        else predicted <= workload.latency_budget_ns
    )
    feasible = _conjunction(
        (
            capacity_fits,
            routing_safe,
            True if predicted is not None else None,
            slo if workload.latency_budget_ns is not None else True,
        )
    )
    throughput = (
        workload.work_items * 1_000_000_000 / predicted
        if workload.work_items is not None and predicted
        else None
    )
    if capacity_fits is False:
        diagnostics.append("modeled HBM peak plus reserve exceeds a rank's capacity")
    if capacity_fits is None:
        diagnostics.append("HBM capacity is unknown for at least one rank")
    assumptions = (
        "device-serial primitive roofline services; blocking ring/pairwise/direct collectives",
        "network phases follow local HBM/compute services; shared paths and rank endpoints serialize",
        "parameters stay resident; views retain backing allocations; other results are out of place",
        "one payload-sized communication scratch buffer; repeated carries reserve a spare generation",
        "uniform loops repeat compactly with resource barriers; no loop-invariant hoisting is assumed",
        "HBM only: local storage constraints, backend workspace beyond reserve, launch and queueing are excluded",
        "useful work is supplied per invocation and is never multiplied by replica count",
    )
    attach_metadata(
        function,
        EngineMetadata(
            "device-serial-ring",
            options.deployment.source,
            tuple(rank_records),
            records,
            predicted,
            workload.work_items,
            workload.work_unit,
            throughput,
            workload.latency_budget_ns,
            capacity_fits,
            routing_safe,
            slo,
            feasible,
            assumptions,
            tuple(dict.fromkeys((*diagnostics, *planner.diagnostics, *schedule.diagnostics))),
            options=options,
            rates=EngineRates(
                tuple((dtype.name, rate) for dtype, rate in services.unit_flops),
                services.unit_ops,
                services.bandwidth("gmem"),
            )
            if services
            else None,
        ),
    )
