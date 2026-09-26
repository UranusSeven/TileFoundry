"""All-to-all ownership and numerical expert routing through authored HIR."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests._source import import_dsl
from tests.fixtures.distributed import moe
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import select
from tilefoundry.runtime import DictResource


def moe_inputs(pattern):
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(moe.D, moe.N, moe.H, generator=generator)
    gates = torch.randn(moe.D, moe.N, moe.K, generator=generator)
    routes = torch.randint(0, moe.E, (moe.D, moe.N, moe.K), generator=generator)
    if pattern == "mixed":
        routes[0, 0] = torch.tensor([0, 1, 3])
        routes[0, 1] = -1
        routes[1, :4] = -1
        routes[1, 4] = torch.tensor([2, 2, -1])
    elif pattern == "skewed":
        routes[:] = 0
    elif pattern == "empty":
        routes[:] = -1
        gates[:] = float("nan")
    weights = {
        "w1": torch.randn(moe.E, moe.H, moe.F, generator=generator),
        "w2": torch.randn(moe.E, moe.F, moe.H, generator=generator),
    }
    return x, routes, gates, weights


def torch_moe(x, routes, gates, weights):
    result = torch.zeros_like(x)
    for batch in range(x.shape[0]):
        for token in range(x.shape[1]):
            for slot in range(routes.shape[-1]):
                expert = int(routes[batch, token, slot])
                if expert >= 0:
                    hidden = torch.relu(x[batch, token] @ weights["w1"][expert])
                    result[batch, token] += (hidden @ weights["w2"][expert]) * gates[
                        batch, token, slot
                    ]
    return result


@pytest.mark.parametrize("pattern", ["mixed", "skewed", "empty"])
def test_moe_dispatch_experts_and_combine(pattern):
    x, routes, gates, weights = moe_inputs(pattern)
    expected = torch_moe(x, routes, gates, weights)
    resource = DictResource(weights)
    reference = evaluate(moe.Reference.load(resource), x, routes, gates)
    actual = evaluate(moe.ExpertParallel.load(resource), x, routes, gates, distributed=True)
    torch.testing.assert_close(reference, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_dispatch_metadata_preserves_route_multiplicity_and_padding():
    x, routes, gates, _ = moe_inputs("mixed")
    packed, scales, indices, counts = evaluate(
        select(moe.ExpertParallel, "dispatch").load(DictResource({})),
        x,
        routes,
        gates,
        distributed=True,
    )
    for batch in range(moe.D):
        for expert in range(moe.E):
            expected = [
                (token, slot)
                for token in range(moe.N)
                for slot in range(moe.K)
                if routes[batch, token, slot] == expert
            ]
            count = len(expected)
            assert counts[batch, expert] == count
            for row, (token, slot) in enumerate(expected):
                torch.testing.assert_close(packed[batch, expert, row], x[batch, token])
                assert scales[batch, expert, row, 0] == gates[batch, token, slot]
                assert indices[batch, expert, row] == token
            assert torch.all(packed[batch, expert, count:] == 0)
            assert torch.all(scales[batch, expert, count:] == 0)
            assert torch.all(indices[batch, expert, count:] == -1)


def test_capacity_overflow_never_drops_routes():
    source = Path(moe.__file__).read_text().replace("CAPACITY = N * K", "CAPACITY = 1")
    candidate = import_dsl(source, "ExpertParallel")
    x, routes, gates, weights = moe_inputs("skewed")
    with pytest.raises(EvalError, match="capacity 1 exceeded for expert 0"):
        evaluate(candidate.load(DictResource(weights)), x, routes, gates, distributed=True)


def test_zero_capacity_accepts_empty_routing_and_returns_zero_counts():
    candidate = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 2),))
class Empty:
    @func
    def run(x: Tensor[(4, 3), "f32"], ids: Tensor[(4, 2), "i64"], gates: Tensor[(4, 2), "f32"]):
        with Mesh(("gpu",), (2,), names=("ep",)) as devices:
            source = tf.reshard(x, (4 @ devices.ep, 3), "gmem")
            routes = tf.reshard(ids, (4 @ devices.ep, 2), "gmem")
            weights = tf.reshard(gates, (4 @ devices.ep, 2), "gmem")
            packed, scales, origins, counts = tf.alltoall_dispatch(source, routes, weights, num_experts=4, capacity=0, mesh_axis=0)
            output = tf.alltoall_combine(packed, origins, counts, source, mesh_axis=0)
            return output, packed, origins, counts
""",
        "Empty",
    )
    x = torch.ones(4, 3)
    routes = torch.full((4, 2), -1, dtype=torch.int64)
    gates = torch.full((4, 2), float("nan"))
    output, packed, origins, counts = evaluate(
        candidate.entry_function(), x, routes, gates, distributed=True
    )
    assert packed.shape == (4, 0, 3)
    assert origins.shape == (4, 0)
    assert torch.all(counts == 0)
    torch.testing.assert_close(output, torch.zeros_like(x))


@pytest.mark.parametrize("invalid", [-2, moe.E])
def test_invalid_expert_ids_are_rejected(invalid):
    x, routes, gates, weights = moe_inputs("mixed")
    routes[0, 0, 0] = invalid
    with pytest.raises(EvalError, match="expert index must be -1 or"):
        evaluate(moe.ExpertParallel.load(DictResource(weights)), x, routes, gates, distributed=True)


def test_regular_alltoall_preserves_values_and_round_trips():
    module = import_dsl(
        """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 4),))
class Exchange:
    @func
    def run(x: Tensor[(2, 4, 6), "f32"], w: Tensor[(6, 3), "f32"]):
        with Mesh(("gpu",), (2, 2), names=("dp", "ep")) as devices:
            rows = tf.reshard(x, (2 @ devices.dp, 4 @ devices.ep, 6), "gmem")
            columns = tf.alltoall(rows, mesh_axis=1, tensor_axis=2)
            returned = tf.alltoall(columns, mesh_axis=1, tensor_axis=1)
            local_w = tf.reshard(w, (6 @ devices.ep, 3), "gmem")
            projection = tf.allreduce(tf.matmul(columns, local_w), mesh_axis=1)
            return columns, returned, projection
""",
        "Exchange",
    )
    x = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)
    w = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    exchanged, restored, projection = evaluate(module.entry_function(), x, w, distributed=True)
    torch.testing.assert_close(exchanged, x)
    torch.testing.assert_close(restored, x)
    torch.testing.assert_close(projection, x @ w)


_COMBINE = """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 2),))
class Reverse:
    @func
    def run(x: Tensor[(4, 3, 5), "f32"], ids: Tensor[(4, 3), "i64"], counts: Tensor[(4,), "i64"], source: Tensor[(6, 2), "f32"]):
        with Mesh(("gpu",), (2,), names=("ep",)) as devices:
            payload = tf.reshard(x, (4 @ devices.ep, 3, 5), "gmem")
            origins = tf.reshard(ids, (4 @ devices.ep, 3), "gmem")
            live = tf.reshard(counts, (4 @ devices.ep,), "gmem")
            original = tf.reshard(source, (6 @ devices.ep, 2), "gmem")
            return tf.alltoall_combine(payload, origins, live, original, mesh_axis=0)
"""


@pytest.mark.parametrize(
    "malformed", [None, "negative_count", "overflow_count", "missing_origin", "outside_origin"]
)
def test_combine_validates_active_metadata_and_ignores_padding(malformed):
    candidate = import_dsl(_COMBINE, "Reverse")
    payload = torch.full((4, 3, 5), float("nan"))
    payload[0, 0], payload[1, 0], payload[2, 0] = 2, -3, 5
    origins = torch.full((4, 3), 99, dtype=torch.int64)
    origins[0, 0], origins[1, 0], origins[2, 0] = 4, 0, 4
    counts = torch.tensor([1, 1, 1, 0])
    source = torch.full((6, 2), float("nan"))
    if malformed:
        if malformed == "negative_count":
            counts[0] = -1
        elif malformed == "overflow_count":
            counts[0] = 4
        elif malformed == "missing_origin":
            origins[0, 0] = -1
        else:
            origins[0, 0] = 6
        with pytest.raises(EvalError, match="expert counts|active source index"):
            evaluate(candidate.entry_function(), payload, origins, counts, source, distributed=True)
    else:
        output = evaluate(
            candidate.entry_function(), payload, origins, counts, source, distributed=True
        )
        expected = torch.zeros(6, 5)
        expected[0], expected[4] = -3, 7
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("tokens", [0, 2, 3, 6])
def test_moe_dynamic_source_batches(tokens):
    source = (
        Path(moe.__file__)
        .read_text()
        .replace(
            "CAPACITY = N * K",
            'from tilefoundry.dsl import DimVar\nN = DimVar("tokens", 0, 8)\nCAPACITY = 24',
        )
    )
    candidate = import_dsl(source, "ExpertParallel")
    x, routes, gates, weights = moe_inputs("mixed")
    x, routes, gates = x[:, :tokens], routes[:, :tokens], gates[:, :tokens]
    if tokens % 2:
        with pytest.raises(EvalError, match="not divisible"):
            evaluate(candidate.load(DictResource(weights)), x, routes, gates, distributed=True)
    else:
        output = evaluate(candidate.load(DictResource(weights)), x, routes, gates, distributed=True)
        torch.testing.assert_close(
            output, torch_moe(x, routes, gates, weights), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize(
    "before, after, message",
    [
        ("num_experts=E", "num_experts=3", "num_experts"),
        ("capacity=CAPACITY", "capacity=-1", "capacity"),
        ("routes,", "scales,", "expert_indices must have dtype"),
        ("mesh_axis=1", "mesh_axis=0", "source tokens must be Split"),
    ],
)
def test_dispatch_rejects_incompatible_routing_contracts(before, after, message):
    source = Path(moe.__file__).read_text().replace(before, after)
    with pytest.raises(VerifyError, match=message):
        import_dsl(source, "ExpertParallel")
