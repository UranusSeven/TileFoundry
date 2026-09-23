"""Numerical contracts of distributed HIR through public evaluator entry points."""

from __future__ import annotations

import pytest
import torch

from tests._source import import_dsl
from tests.fixtures.distributed.projection import ReduceScatterProjection, Reference, TensorParallel
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.ir.core import VerifyError
from tilefoundry.runtime import DictResource


@pytest.mark.parametrize("candidate", [TensorParallel, ReduceScatterProjection])
def test_projection_and_repeated_state(candidate):
    generator = torch.Generator().manual_seed(81)
    weight = torch.randn(8, 6, generator=generator)
    reference_state = torch.randn(4, 6, generator=generator)
    candidate_state = reference_state.clone()
    resource = DictResource({"w": weight})
    for _ in range(3):
        x = torch.randn(4, 8, generator=generator)
        expected = x @ weight + reference_state
        reference_output, reference_state = evaluate(Reference.load(resource), x, reference_state)
        output, candidate_state = evaluate(
            candidate.load(resource), x, candidate_state, distributed=True
        )
        torch.testing.assert_close(reference_output, -expected)
        torch.testing.assert_close(output, -expected)
        torch.testing.assert_close(candidate_state, expected)


_PROJECTION = """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 4),))
class Projection:
    @func
    def run(x: Tensor[(4, 8), "f32"], w: Tensor[(8, 6), "f32"]):
        with Mesh(("gpu",), (2, 2), names=("dp", "tp")) as devices:
            a = tf.reshard(x, (4 @ devices.dp, 8 @ devices.tp), "gmem")
            b = tf.reshard(w, (8 @ devices.tp, 6), "gmem")
            partial = tf.matmul(a, b)
            complete = tf.allreduce(partial, mesh_axis=1)
            return tf.allgather(complete, mesh_axis=0)
"""


def test_multiaxis_groups_gather_in_coordinate_order():
    module = import_dsl(_PROJECTION, "Projection")
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    w = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    torch.testing.assert_close(evaluate(module.entry_function(), x, w, distributed=True), x @ w)


@pytest.mark.parametrize("batch", [0, 2, 3, 6])
def test_dynamic_partitions_check_divisibility_and_empty_payloads(batch):
    source = (
        _PROJECTION.replace(
            "from tilefoundry import module, func",
            "from tilefoundry import module, func\n"
            'from tilefoundry.ir.types.dim import DimVar\nN = DimVar("batch", 0, 8)',
        )
        .replace("Tensor[(4, 8)", "Tensor[(N, 8)")
        .replace("(4 @ devices.dp,", "(N @ devices.dp,")
    )
    module = import_dsl(source, "Projection")
    x = torch.arange(batch * 8, dtype=torch.float32).reshape(batch, 8)
    w = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    if batch % 2:
        with pytest.raises(EvalError, match="not divisible"):
            evaluate(module.entry_function(), x, w, distributed=True)
    else:
        torch.testing.assert_close(evaluate(module.entry_function(), x, w, distributed=True), x @ w)


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("return partial", "output is Partial"),
        ("return tf.reshard(partial, (4 @ devices.dp, 6), 'gmem')", "cannot complete Partial"),
        ("return tf.reshard(complete, (4, 6), 'gmem')", "cross-device data"),
    ],
)
def test_missing_communication_is_not_repaired(replacement, message):
    source = _PROJECTION.replace("return tf.allgather(complete, mesh_axis=0)", replacement)
    module = import_dsl(source, "Projection")
    with pytest.raises(EvalError, match=message):
        evaluate(module.entry_function(), torch.ones(4, 8), torch.ones(8, 6), distributed=True)


@pytest.mark.parametrize(
    "source, message",
    [
        (
            _PROJECTION.replace(
                "allreduce(partial, mesh_axis=1)", "allreduce(partial, mesh_axis=0)"
            ),
            "requires Partial",
        ),
        (
            _PROJECTION.replace(
                "allgather(complete, mesh_axis=0)", "allgather(complete, mesh_axis=1)"
            ),
            "requires Split",
        ),
        (
            _PROJECTION.replace(
                "allreduce(partial, mesh_axis=1)", "allreduce(partial, mesh_axis=3)"
            ),
            "mesh_axis",
        ),
    ],
)
def test_illegal_collective_contracts_are_rejected(source, message):
    with pytest.raises(VerifyError, match=message):
        import_dsl(source, "Projection")


def test_collectives_require_distributed_execution():
    module = import_dsl(_PROJECTION, "Projection")
    with pytest.raises(EvalError, match="require distributed evaluation"):
        evaluate(module.entry_function(), torch.ones(4, 8), torch.ones(8, 6))


@pytest.mark.parametrize("axis", [0, 1])
def test_local_composition_preserves_transposed_ownership(axis):
    source = """
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 2),))
class LocalComposition:
    @func
    def run(x: Tensor[(4, 6), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as devices:
            local = tf.reshard(x, (4 @ devices.tp, 6), "gmem")
            transposed = tf.transpose(local, perm=(1, 0))
            rounded = tf.cast(transposed, dtype="bf16")
            return tf.reduce(rounded, axes=(AXIS,), keepdim=False, kind="sum")
""".replace("AXIS", str(axis))
    module = import_dsl(source, "LocalComposition")
    x = torch.randn(4, 6, generator=torch.Generator().manual_seed(42))
    if axis == 1:
        with pytest.raises(EvalError, match="Reduce over a Split axis"):
            evaluate(module.entry_function(), x, distributed=True)
    else:
        expected = x.T.bfloat16().sum(dim=0)
        torch.testing.assert_close(evaluate(module.entry_function(), x, distributed=True), expected)


def test_collectives_survive_function_and_loop_boundaries():
    module = import_dsl(
        """
from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
@module(entry="run", topologies=(Topology("gpu", 2),))
class Repeated:
    @func(mesh=Mesh(("gpu",), (2,), names=("tp",)))
    def project(x: Tensor[(4, 8 @ mesh.tp), "f32"], w: Tensor[(8 @ mesh.tp, 6), "f32"]):
        return tf.allreduce(tf.matmul(x, w), mesh_axis=0)

    @func
    def run(x: Tensor[(4, 8), "f32"], w: Tensor[(8, 6), "f32"], state: Tensor[(4, 6), "f32"]):
        with Mesh(("gpu",), (2,), names=("tp",)) as devices:
            a = tf.reshard(x, (4, 8 @ devices.tp), "gmem")
            b = tf.reshard(w, (8 @ devices.tp, 6), "gmem")
            current = tf.reshard(state, (4, 6), "gmem")
            for i in range(3):
                current = tf.add(current, project(a, b))
            return current
""",
        "Repeated",
    )
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(4, 8, generator=generator)
    w = torch.randn(8, 6, generator=generator)
    state = torch.randn(4, 6, generator=generator)
    output = evaluate(module.entry_function(), x, w, state, distributed=True)
    torch.testing.assert_close(output, state + 3 * (x @ w))
