"""Device-level resource, communication and workload findings."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.metadata import IRMetadata

from .engine_profile import EngineOptions


@dataclass(frozen=True)
class EngineRates:
    """Retain the exact target service rates consumed by the engine model."""

    flops_per_second: tuple[tuple[str, int], ...]
    operations_per_second: tuple[tuple[str, int], ...]
    hbm_bytes_per_second: int | None


@dataclass(frozen=True)
class EngineWork:
    """One rank's primitive work and HBM traffic for one operation occurrence."""

    rank: int
    flops: tuple[tuple[str, int], ...] = ()
    other_ops: tuple[tuple[str, int], ...] = ()
    read_bytes: int = 0
    write_bytes: int = 0


@dataclass(frozen=True)
class EngineTransfer:
    """One directed payload in a collective phase."""

    source: int
    destination: int
    bytes: int
    phase: int
    resources: tuple[str, ...] = ()
    start_ns: int | None = None
    end_ns: int | None = None


@dataclass(frozen=True)
class EngineOperation:
    """Provenance and estimated execution of one static HIR operation."""

    operation: str
    source: str
    ranks: tuple[int, ...]
    work: tuple[EngineWork, ...]
    transfers: tuple[EngineTransfer, ...] = ()
    repeats: int = 1
    traffic_kind: str = "exact"
    start_ns: int | None = None
    end_ns: int | None = None


@dataclass(frozen=True)
class EngineBuffer:
    """One allocation contributing to a rank's modeled HBM peak."""

    name: str
    kind: str
    bytes: int


@dataclass(frozen=True)
class EngineRank:
    """Resident ownership, operation totals and peak storage for a physical rank."""

    rank: int
    weights_bytes: int
    state_bytes: int
    input_bytes: int
    output_bytes: int
    peak_hbm_bytes: int
    reserve_bytes: int
    capacity_bytes: int | None
    fits: bool | None
    flops: tuple[tuple[str, int], ...]
    read_bytes: int
    write_bytes: int
    sent_bytes: int
    received_bytes: int
    peak_buffers: tuple[EngineBuffer, ...]


@dataclass(frozen=True)
class EngineMetadata(IRMetadata):
    """One invocation's predicted resources under an explicit deployment and workload."""

    model: str
    deployment_source: str
    ranks: tuple[EngineRank, ...]
    operations: tuple[EngineOperation, ...]
    predicted_ns: int | None
    work_items: int | None
    work_unit: str
    throughput_per_second: float | None
    latency_budget_ns: int | None
    capacity_fits: bool | None
    routing_safe: bool | None
    slo_met: bool | None
    feasible: bool | None
    assumptions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    options: EngineOptions | None = None
    rates: EngineRates | None = None
