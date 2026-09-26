# Price parallel strategies under a memory and latency budget

Use `check` to validate a candidate, then `analyze --engine` to estimate its
per-rank resources and useful throughput. This page compares three authored
placements of the same projection on a fixed four-device budget. Selection is
ordinary Python over the report, so TileFoundry does not prescribe a strategy
catalog or search policy.

The rates and capacities below are synthetic arithmetic inputs, not GPU
measurements. The program, device facts, deployment links and workload budget
are separate inputs. Save this page as `engine-analysis.md` and extract them:

```bash
set -euo pipefail
for name in candidates.py configure.py choose.py; do
  awk -v tag="<!-- tilefoundry-source: $name -->" '
    $0 == tag { block=1; next }
    block && /^```python$/ { in_python=1; next }
    in_python && /^```$/ { in_python=0; block=0; next }
    in_python { print }
  ' engine-analysis.md > "$name"
done
```

Device facts supply one device's rates. The deployment profile supplies
connectivity and reserves, and the workload states 16 useful tokens and a
45,000 ns invocation budget. No work count is multiplied by replica count.

<!-- tilefoundry-source: configure.py -->

```python
# example
import json
from pathlib import Path

Path("device.json").write_text(json.dumps({
    "capacity": 64000, "bandwidth": 1000000000, "flops": 1000000000,
}, indent=2))

profile = {
    "deployment": {
        "source": "synthetic tutorial facts",
        "devices": [{"rank": rank, "reserve_bytes": 48000} for rank in range(4)],
        "routes": [
            {"source": source, "destination": destination,
             "bandwidth_bytes_per_second": 1000000000, "latency_ns": 10}
            for source in range(4) for destination in range(4) if source != destination
        ],
    },
    "workload": {"work_items": 16, "work_unit": "tokens", "latency_budget_ns": 45000},
}
Path("engine.json").write_text(json.dumps(profile, indent=2))
```

```bash
set -euo pipefail
python configure.py
```

`DP4` replicates weights over four data-parallel ranks. `TP2` uses two groups
with tensor parallel degree two. `TP4` uses tensor parallel degree four. Weight
ownership is declared on the parameters: selecting a later view cannot free
an already replicated backing allocation.

<!-- tilefoundry-source: candidates.py -->

```python
# example
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tilefoundry import func, module
from tilefoundry.analysis.facts import ExplicitMemoryLevelFacts, MemoryHierarchyFacts, PerformanceServiceFacts
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, Topology, tf
from tilefoundry.ir.types import DType
from tilefoundry.target import Target
from tilefoundry.target.facts import TopologyFacts, TopologyLimitFacts


@dataclass(frozen=True)
class ProfileTarget(Target):
    """Synthetic device rates supplied by the example's external input file."""

    name: ClassVar[str] = "synthetic.tutorial"
    capacity: int
    bandwidth: int
    flops: int

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
                tuple((name, self.flops) for name in ("integer", "predicate", "select", "special")),
                (("gmem", self.bandwidth),), "gpu",
            )
        return super().get_facts(facts_type, query)


DEVICE = ProfileTarget(**json.loads(Path(__file__).with_name("device.json").read_text()))

TOKENS = DimVar("tokens", 1, 65536)


@module(entry="run")
class StrategyReference:
    @func
    def run(x: Tensor[(TOKENS, 64), "f32"], w: ConstTensor[(64, 64), "f32"]):
        return tf.matmul(x, w)


@module(entry="run", target=DEVICE, topologies=(Topology("gpu", 4),))
class DP4:
    @func(mesh=Mesh(("gpu",), (4,), names=("dp",)))
    def run(x: Tensor[(TOKENS @ mesh.dp, 64), "f32"], w: ConstTensor[(64, 64), "f32"]):  # noqa: F821
        return tf.matmul(x, w)


@module(entry="run", target=DEVICE, topologies=(Topology("gpu", 4),))
class TP2:
    @func(mesh=Mesh(("gpu",), (2, 2), names=("dp", "tp")))
    def run(
        x: Tensor[(TOKENS @ mesh.dp, 64 @ mesh.tp), "f32"],  # noqa: F821
        w: ConstTensor[(64 @ mesh.tp, 64), "f32"],  # noqa: F821
    ):
        return tf.allreduce(tf.matmul(x, w), mesh_axis=1)


@module(entry="run", target=DEVICE, topologies=(Topology("gpu", 4),))
class TP4:
    @func(mesh=Mesh(("gpu",), (4,), names=("tp",)))
    def run(
        x: Tensor[(TOKENS, 64 @ mesh.tp), "f32"],  # noqa: F821
        w: ConstTensor[(64 @ mesh.tp, 64), "f32"],  # noqa: F821
    ):
        return tf.allreduce(tf.matmul(x, w), mesh_axis=0)
```

```bash
set -euo pipefail
for name in DP4 TP2 TP4; do
  tilefoundry check "candidates.py:$name" --reference candidates.py:StrategyReference \
    --distributed --dim tokens=16 --inputs random --weights random --device cpu \
    --out output --fn allclose --atol 1e-4 --rtol 1e-4 > "$name-check.txt"
  tilefoundry analyze "candidates.py:$name" "$name.json" \
    --engine --dim tokens=16 --engine-profile engine.json --json
done
```

Every preceding correctness command must succeed before selection. The report
retains primitive work, directed transfers, source provenance, per-rank peaks,
input assumptions and the target rates actually consumed. Rank the candidates
whose modeled capacity, routing and latency constraints all pass:

<!-- tilefoundry-source: choose.py -->

```python
# example
import json
from pathlib import Path

findings = {
    name: json.loads(Path(name + ".json").read_text())["function_records"]["engine"]
    for name in ("DP4", "TP2", "TP4")
}
print("candidate  latency_ns  peak_HBM_B  capacity_fit  SLO_met  tokens/s")
for name, report in findings.items():
    peak = max(rank["peak_hbm_bytes"] for rank in report["ranks"])
    rate = report["throughput_per_second"]
    shown = "unknown" if rate is None else f"{rate:.0f}"
    print(f"{name:9}  {str(report['predicted_ns']):10}  {peak:10}  "
          f"{str(report['capacity_fits']):12}  {str(report['slo_met']):7}  {shown}")
feasible = [
    name for name, report in findings.items()
    if report["feasible"] is True and report["throughput_per_second"] is not None
]
if feasible:
    selected = max(feasible, key=lambda name: findings[name]["throughput_per_second"])
    print("Selected:", selected)
else:
    print("No feasible candidate under the supplied assumptions.")
```

```bash
set -euo pipefail
python choose.py
```

```text
candidate  latency_ns  peak_HBM_B  capacity_fit  SLO_met  tokens/s
DP4        32768            66432  False         True     488281
TP2        38932            63360  True          True     410973
TP4        47164            65408  False         False    339242
Selected: TP2
```

The reserve leaves 16,000 bytes for authored allocations on each device.
The faster fully replicated candidate does not fit; tensor parallel degree two
meets both constraints. The model charges ring communication to the tensor
parallel candidates.

Now keep the program, rates, links, useful work and latency budget fixed while
removing the reserve:

```bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
path = Path('engine.json')
profile = json.loads(path.read_text())
for device in profile['deployment']['devices']:
    device['reserve_bytes'] = 0
path.write_text(json.dumps(profile, indent=2))
PY
for name in DP4 TP2 TP4; do
  tilefoundry analyze "candidates.py:$name" "$name.json" \
    --engine --dim tokens=16 --engine-profile engine.json --json
done
python choose.py
```

```text
candidate  latency_ns  peak_HBM_B  capacity_fit  SLO_met  tokens/s
DP4        32768            18432  True          True     488281
TP2        38932            15360  True          True     410973
TP4        47164            17408  True          False    339242
Selected: DP4
```

With sufficient memory, the fully replicated candidate wins on modeled
throughput. This is the best candidate in the evaluated set under these
assumptions; it is not a global optimality claim or a serving benchmark.

For a batch/context sweep, bind each program dimension with `--dim`, supply
that invocation's useful work and budget, and retain a report per point. A KV
or recurrent-state tensor is an explicit parameter; identify it in
`workload.state_inputs`. Its shape and placement determine resident bytes.
The model preserves input buffers and uses out-of-place results, so returned
state can require another allocation; no unproven in-place reuse is assumed.

For routed MoE, `routing` profiles identify operations such as
`AllToAllDispatch:0` and supply distinct peer tokens, expanded expert routes and
per-expert live counts. Without them, traffic is an explicitly labeled upper
bound, and insufficient capacity bounds leave routing feasibility unknown.
Missing link facts also leave timing and SLO feasibility unknown.

See [engine analysis](../spec/analysis.md#5-device-level-engine-analysis) and
[profile fields](../spec/analysis.md#3-portable-engine-profiles) for the complete
contract, and use `tilefoundry check --help` for numerical predicates.
The existing local `performance` analysis retains its meaning. GPU backend
lowering, local resource validation, runtime overhead calibration and physical
execution remain separate from this device-level estimate.
