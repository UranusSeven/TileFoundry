"""Analytically tractable device programs and a synthetic, explicitly stated target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tilefoundry import func, module
from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    PerformanceServiceFacts,
)
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf
from tilefoundry.ir.types import DType
from tilefoundry.target import Target
from tilefoundry.target.facts import TopologyFacts, TopologyLimitFacts


@dataclass(frozen=True)
class SyntheticTarget(Target):
    """Supply simple rates for independent arithmetic checks, not a hardware prediction."""

    name: ClassVar[str] = "synthetic.engine"
    capacity: int | None = 4096
    bandwidth: int = 1_000_000_000
    flops: int = 1_000_000_000

    def get_facts(self, facts_type, query=None):
        if facts_type is TopologyFacts:
            return TopologyFacts((TopologyLimitFacts("gpu", 4, from_target=True),))
        if facts_type is MemoryHierarchyFacts:
            return MemoryHierarchyFacts(
                (ExplicitMemoryLevelFacts("gmem", self.capacity, "gpu", "gpu"),), (), ()
            )
        if facts_type is PerformanceServiceFacts:
            return PerformanceServiceFacts(
                ((DType.f32, self.flops),),
                tuple((kind, self.flops) for kind in ("integer", "predicate", "select", "special")),
                (("gmem", self.bandwidth),),
                "gpu",
            )
        return super().get_facts(facts_type, query)


@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 2),))
class ShardedProjection:
    @func(mesh=Mesh(("gpu",), (2,), names=("tp",)))
    def run(
        x: Tensor[(4, 8 @ mesh.tp), "f32"],  # noqa: F821
        w: ConstTensor[(8 @ mesh.tp, 6), "f32"],  # noqa: F821
        state: Tensor[(4, 6), "f32"],
    ):
        complete = tf.allreduce(tf.matmul(x, w), mesh_axis=0)
        updated = tf.add(complete, state)
        return tf.neg(updated), updated


@module(entry="run", target=SyntheticTarget(), topologies=(Topology("gpu", 2),))
class RepeatedState:
    @func(mesh=Mesh(("gpu",), (2,), names=("tp",)))
    def run(
        x: Tensor[(4 @ mesh.tp, 8), "f32"],  # noqa: F821
        state: Tensor[(4 @ mesh.tp, 8), "f32"],  # noqa: F821
    ):
        current = state
        for i in range(1000000):
            current = tf.add(current, x)
        return current
