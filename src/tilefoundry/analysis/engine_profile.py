"""Portable deployment and workload inputs for device-level analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _integer(name, value, minimum=0, optional=False):
    if value is None and optional:
        return
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}" + (" or null" if optional else "")
        )


def _names(name, values):
    if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must be a tuple of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")


def _records(name, values, type_):
    if not isinstance(values, tuple) or any(not isinstance(value, type_) for value in values):
        raise ValueError(f"{name} must be a tuple of {type_.__name__} values")


@dataclass(frozen=True)
class DeviceBudget:
    """Override one rank's HBM capacity and reserve bytes outside authored HIR."""

    rank: int
    capacity_bytes: int | None = None
    reserve_bytes: int = 0

    def __post_init__(self):
        _integer("rank", self.rank)
        _integer("capacity_bytes", self.capacity_bytes, optional=True)
        _integer("reserve_bytes", self.reserve_bytes)


@dataclass(frozen=True)
class NetworkRoute:
    """Describe an effective directed path and the resources it shares with other paths."""

    source: int
    destination: int
    bandwidth_bytes_per_second: int | None = None
    latency_ns: int | None = None
    resources: tuple[str, ...] = ()

    def __post_init__(self):
        _integer("source", self.source)
        _integer("destination", self.destination)
        if self.source == self.destination:
            raise ValueError("network routes must connect different ranks")
        _integer("bandwidth_bytes_per_second", self.bandwidth_bytes_per_second, 1, optional=True)
        _integer("latency_ns", self.latency_ns, optional=True)
        _names("resources", self.resources)


@dataclass(frozen=True)
class EngineDeployment:
    """Add deployment-specific capacity and connectivity to target-owned device facts."""

    devices: tuple[DeviceBudget, ...] = ()
    routes: tuple[NetworkRoute, ...] = ()
    source: str = "unspecified"

    def __post_init__(self):
        _records("devices", self.devices, DeviceBudget)
        _records("routes", self.routes, NetworkRoute)
        if len({device.rank for device in self.devices}) != len(self.devices):
            raise ValueError("duplicate device rank")
        if len({(route.source, route.destination) for route in self.routes}) != len(self.routes):
            raise ValueError("duplicate directed network route")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("deployment source must be a nonempty provenance string")


@dataclass(frozen=True)
class EngineWorkload:
    """State useful work, an invocation latency budget, and persistent state inputs."""

    work_items: int | None = None
    work_unit: str = "tokens"
    latency_budget_ns: int | None = None
    state_inputs: tuple[str, ...] = ()

    def __post_init__(self):
        _integer("work_items", self.work_items, optional=True)
        _integer("latency_budget_ns", self.latency_budget_ns, optional=True)
        if not isinstance(self.work_unit, str) or not self.work_unit:
            raise ValueError("work_unit must be a nonempty string")
        _names("state_inputs", self.state_inputs)


@dataclass(frozen=True)
class RoutingProfile:
    """Describe one dispatch's peer traffic and live expert rows, including self routes."""

    operation: str
    peer_tokens: tuple[tuple[int, ...], ...]
    peer_routes: tuple[tuple[int, ...], ...]
    expert_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("routing operation must be a nonempty operation identifier")
        count = len(self.peer_tokens)
        for name in ("peer_tokens", "peer_routes", "expert_counts"):
            matrix = getattr(self, name)
            if not isinstance(matrix, tuple) or len(matrix) != count or not count:
                raise ValueError(f"{name} must have one row per rank")
            for row in matrix:
                if not isinstance(row, tuple) or (name != "expert_counts" and len(row) != count):
                    raise ValueError(f"{name} must contain rank-ordered tuples")
                for value in row:
                    _integer(name, value)
        for source in range(count):
            for destination in range(count):
                if self.peer_tokens[source][destination] > self.peer_routes[source][destination]:
                    raise ValueError("peer_tokens cannot exceed peer_routes")
                if bool(self.peer_tokens[source][destination]) != bool(
                    self.peer_routes[source][destination]
                ):
                    raise ValueError("peer tokens and routes must agree on empty transfers")
        for destination in range(count):
            if sum(row[destination] for row in self.peer_routes) != sum(
                self.expert_counts[destination]
            ):
                raise ValueError(
                    "incoming peer_routes must equal the destination expert_counts sum"
                )


@dataclass(frozen=True)
class EngineOptions:
    """Collect the external assumptions used by one reproducible engine analysis."""

    deployment: EngineDeployment = field(default_factory=EngineDeployment)
    workload: EngineWorkload = field(default_factory=EngineWorkload)
    routing: tuple[RoutingProfile, ...] = ()

    def __post_init__(self):
        if not isinstance(self.deployment, EngineDeployment) or not isinstance(
            self.workload, EngineWorkload
        ):
            raise ValueError("engine options require deployment and workload records")
        _records("routing", self.routing, RoutingProfile)
        if len({profile.operation for profile in self.routing}) != len(self.routing):
            raise ValueError("duplicate routing operation profile")

    def to_dict(self) -> dict:
        """Return JSON-serializable assumptions without inferred facts."""
        return asdict(self)


def _from_dict(data: dict) -> EngineOptions:
    def fields(value, allowed, label):
        if not isinstance(value, dict) or set(value) - set(allowed):
            raise ValueError(f"{label}: expected an object with fields {tuple(allowed)}")
        return dict(value)

    def array(value, label):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{label} must be an array")
        return value

    root = fields(data, ("deployment", "workload", "routing"), "engine profile")
    deployment = fields(root.get("deployment", {}), ("devices", "routes", "source"), "deployment")
    devices = tuple(
        DeviceBudget(**fields(item, ("rank", "capacity_bytes", "reserve_bytes"), "device"))
        for item in array(deployment.pop("devices", ()), "devices")
    )
    routes = []
    for item in array(deployment.pop("routes", ()), "routes"):
        route = fields(
            item,
            ("source", "destination", "bandwidth_bytes_per_second", "latency_ns", "resources"),
            "route",
        )
        route["resources"] = tuple(array(route.get("resources", ()), "resources"))
        routes.append(NetworkRoute(**route))
    workload = fields(
        root.get("workload", {}),
        ("work_items", "work_unit", "latency_budget_ns", "state_inputs"),
        "workload",
    )
    workload["state_inputs"] = tuple(array(workload.get("state_inputs", ()), "state_inputs"))
    routing = []
    for item in array(root.get("routing", ()), "routing"):
        profile = fields(
            item, ("operation", "peer_tokens", "peer_routes", "expert_counts"), "routing"
        )
        for name in ("peer_tokens", "peer_routes", "expert_counts"):
            profile[name] = tuple(tuple(array(row, name)) for row in array(profile[name], name))
        routing.append(RoutingProfile(**profile))
    return EngineOptions(
        EngineDeployment(devices, tuple(routes), **deployment),
        EngineWorkload(**workload),
        tuple(routing),
    )


def engine_options_from_dict(data: dict) -> EngineOptions:
    """Read the portable profile, rejecting unknown fields and malformed values."""
    try:
        return _from_dict(data)
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid engine profile: {error}") from error


def load_engine_options(path: str | Path) -> EngineOptions:
    """Load deployment and workload assumptions from one JSON file."""
    return engine_options_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
