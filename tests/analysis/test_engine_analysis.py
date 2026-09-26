"""Hand-computable distributed work, ownership, timing and uncertainty."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests._source import import_dsl
from tests.fixtures.distributed.engine import RepeatedState, ShardedProjection, SyntheticTarget
from tests.fixtures.distributed.moe import ExpertParallel
from tests.fixtures.distributed.projection import TensorParallel
from tilefoundry.analysis import analyze
from tilefoundry.analysis.engine_metadata import EngineMetadata
from tilefoundry.analysis.engine_profile import (
    DeviceBudget,
    EngineDeployment,
    EngineOptions,
    EngineWorkload,
    NetworkRoute,
    RoutingProfile,
)
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.inspection.analysis_report import render_analysis, render_text
from tilefoundry.ir.core import get_metadata


def options(*, bandwidth=1_000_000_000, shared=False, capacity=None, reserve=0):
    resources = ("shared-fabric",) if shared else ()
    deployment = EngineDeployment(
        (DeviceBudget(0, capacity, reserve), DeviceBudget(1, capacity, reserve)),
        (
            NetworkRoute(0, 1, bandwidth, 10, resources),
            NetworkRoute(1, 0, bandwidth, 10, resources),
        ),
        "synthetic facts",
    )
    return EngineOptions(deployment, EngineWorkload(4, "tokens", 2000, ("state",)))


def record(module, profile=None):
    result = analyze(module, module.entry_function(), analysis="engine", options=profile)
    return get_metadata(result.function, EngineMetadata), result


def test_projection_matches_work_payload_peak_and_latency_by_hand():
    found, result = record(ShardedProjection, options(capacity=600))
    assert found.predicted_ns == 1044
    assert found.throughput_per_second == pytest.approx(4e9 / 1044)
    assert found.feasible is True
    for rank in found.ranks:
        assert rank.weights_bytes == 96
        assert rank.state_bytes == 96
        assert rank.input_bytes == 64
        assert rank.flops == (("f32", 252),)
        assert rank.sent_bytes == rank.received_bytes == 96
        assert rank.peak_hbm_bytes == 544
        assert sum(buffer.bytes for buffer in rank.peak_buffers) == 544
    rendered = render_analysis(result)
    assert rendered.data["function_records"]["engine"]["predicted_ns"] == 1044
    assert (
        rendered.data["function_records"]["engine"]["rates"]["flops_per_second"]["f32"]
        == 1000000000
    )
    assert (
        rendered.data["function_records"]["engine"]["options"]["deployment"]["routes"][0][
            "latency_ns"
        ]
        == 10
    )
    assert "engine" in render_text(rendered)


def test_links_reserves_and_slo_change_the_relevant_verdicts():
    baseline, _ = record(ShardedProjection, options(capacity=600))
    slower, _ = record(ShardedProjection, options(bandwidth=500_000_000, capacity=600))
    shared, _ = record(ShardedProjection, options(shared=True, capacity=600))
    reserved, _ = record(ShardedProjection, options(capacity=600, reserve=57))
    assert slower.predicted_ns == baseline.predicted_ns + 96
    assert shared.predicted_ns == baseline.predicted_ns + 116
    assert reserved.capacity_fits is False and reserved.feasible is False
    assert reserved.ranks[0].peak_hbm_bytes == 601
    tight = replace(options(), workload=EngineWorkload(4, "tokens", 1043, ("state",)))
    assert record(ShardedProjection, tight)[0].slo_met is False


def test_unplaced_weight_backing_stays_resident_after_a_view():
    replicated = replace(TensorParallel, target=SyntheticTarget())
    found, _ = record(replicated, options(capacity=600))
    assert all(rank.weights_bytes == 192 for rank in found.ranks)
    assert all(rank.peak_hbm_bytes == 704 for rank in found.ranks)
    assert found.capacity_fits is False


def test_unknown_network_facts_do_not_become_zero_time_or_a_pass():
    found, _ = record(
        ShardedProjection, EngineOptions(workload=EngineWorkload(4, latency_budget_ns=2000))
    )
    assert found.predicted_ns is None
    assert found.throughput_per_second is None
    assert found.capacity_fits is True
    assert found.slo_met is None and found.feasible is None
    assert any("missing network facts" in note for note in found.diagnostics)


def test_loop_aggregates_work_without_materializing_iterations():
    found, _ = record(
        RepeatedState, EngineOptions(workload=EngineWorkload(state_inputs=("state",)))
    )
    assert len(found.operations) == 1
    assert found.operations[0].repeats == 1000000
    assert found.predicted_ns == 192000000
    assert found.ranks[0].flops == (("f32", 16000000),)
    assert found.ranks[0].peak_hbm_bytes == 256


def test_moe_profile_counts_deduplicated_tokens_separately_from_expert_rows():
    module = replace(ExpertParallel, target=SyntheticTarget(capacity=100000))
    tokens = ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0))
    routes = ((0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 2), (0, 0, 0, 0))
    counts = ((0, 0), (1, 1), (0, 0), (1, 1))
    routing = RoutingProfile("AllToAllDispatch:0", tokens, routes, counts)
    links = tuple(
        NetworkRoute(s, d, 1_000_000_000, 10) for s in range(4) for d in range(4) if s != d
    )
    profile = EngineOptions(EngineDeployment(routes=links), routing=(routing,))
    found, _ = record(module, profile)
    dispatch = next(op for op in found.operations if op.operation == "AllToAllDispatch:0")
    combine = next(op for op in found.operations if op.operation == "AllToAllCombine:0")
    assert sorted((item.source, item.destination, item.bytes) for item in dispatch.transfers) == [
        (0, 1, 48),
        (2, 3, 48),
    ]
    assert sorted((item.source, item.destination, item.bytes) for item in combine.transfers) == [
        (1, 0, 24),
        (3, 2, 24),
    ]
    assert found.routing_safe is True
    assert dispatch.traffic_kind == combine.traffic_kind == "profile"


def test_hidden_redistribution_is_refused_by_costing_too():
    module = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tests.fixtures.distributed.engine import SyntheticTarget
@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 2),))
class Hidden:
    @func
    def run(x: Tensor[(4, 8), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as m:
            rows = tf.reshard(x, (4 @ m.tp, 8), "gmem")
            return tf.reshard(rows, (4, 8), "gmem")
""",
        "Hidden",
    )
    with pytest.raises(AnalysisError, match="explicit cross-device communication"):
        record(module)


def test_resident_weight_capacity_is_independent_of_selected_rows():
    module = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf
from tests.fixtures.distributed.engine import SyntheticTarget
@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 2),))
class SparseRead:
    @func
    def selected(w: ConstTensor[(100, 4), "f32"], ids: Tensor[(2,), "i64"]):
        return tf.index_select(w, ids, dim=0)
    @func
    def run(ids: Tensor[(2,), "i64"], w: ConstTensor[(100, 4), "f32"]):
        return selected(w, ids)
""",
        "SparseRead",
    )
    found, _ = record(module)
    assert all(rank.weights_bytes == 1600 for rank in found.ranks)
    assert all(rank.read_bytes == 48 for rank in found.ranks)
    assert all(rank.peak_hbm_bytes == 1648 for rank in found.ranks)
    assert found.predicted_ns == 80


def test_missing_capacity_and_invalid_profile_bindings_remain_explicit():
    unknown = replace(ShardedProjection, target=SyntheticTarget(capacity=None))
    found, _ = record(unknown, options())
    assert found.capacity_fits is None and found.feasible is None
    with pytest.raises(AnalysisError, match="outside the program topology"):
        record(ShardedProjection, EngineOptions(EngineDeployment((DeviceBudget(2),))))
    with pytest.raises(AnalysisError, match="state input"):
        record(ShardedProjection, EngineOptions(workload=EngineWorkload(state_inputs=("missing",))))


def test_out_of_place_state_update_prices_preserved_backing_bytes():
    module = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, Topology, tf
from tests.fixtures.distributed.engine import SyntheticTarget
@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 1),))
class Update:
    @func
    def run(cache: Tensor[(100, 4), "f32"], ids: Tensor[(2,), "i64"], update: Tensor[(2, 4), "f32"]):
        return tf.index_copy(cache, ids, update, dim=0)
""",
        "Update",
    )
    found, _ = record(module, EngineOptions(workload=EngineWorkload(state_inputs=("cache",))))
    assert found.ranks[0].state_bytes == 1600
    assert found.ranks[0].peak_hbm_bytes == 3248
    assert found.ranks[0].read_bytes == 1648
    assert found.ranks[0].write_bytes == 1600
    assert found.predicted_ns == 3248


def test_uneven_ring_chunks_charge_only_the_reductions_received_by_each_rank():
    module = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, Mesh, Topology, tf
from tests.fixtures.distributed.engine import SyntheticTarget
@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 2),))
class OddPayload:
    @func
    def run(x: Tensor[(1, 2), "f32"], w: Tensor[(2, 3), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as mesh:
            lhs = tf.reshard(x, (1, 2 @ mesh.tp), "gmem")
            rhs = tf.reshard(w, (2 @ mesh.tp, 3), "gmem")
            return tf.allreduce(tf.matmul(lhs, rhs), mesh_axis=0)
""",
        "OddPayload",
    )
    found, _ = record(module, replace(options(), workload=EngineWorkload()))
    reduction = next(op for op in found.operations if op.operation == "AllReduce:0")
    assert [dict(work.flops)["f32"] for work in reduction.work] == [1, 2]
    assert [rank.sent_bytes for rank in found.ranks] == [12, 12]
