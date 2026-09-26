# Check a distributed projection against its reference

Use `check --reference SOURCE --distributed` to evaluate a transformed HIR
with simulated device ranks and compare its logical outputs against another
HIR. The simulation runs on the torch device supplied by `--device`, so the
example below runs on a CPU.

Save this page as `distributed-check.md`, then extract its program:

```bash
set -euo pipefail
awk -v tag="<!-- tilefoundry-source: projection.py -->" '
  $0 == tag { block=1; next }
  block && /^```python$/ { in_python=1; next }
  in_python && /^```$/ { in_python=0; block=0; next }
  in_python { print }
' distributed-check.md > projection.py
```

<!-- tilefoundry-source: projection.py -->

```python
# example
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf


@module(entry="step")
class Reference:
    @func
    def step(
        x: Tensor[(4, 8), "f32"],
        w: ConstTensor[(8, 6), "f32"],
        state: Tensor[(4, 6), "f32"],
    ):
        updated = tf.add(tf.matmul(x, w), state)
        return tf.neg(updated), updated


@module(entry="step", topologies=(Topology("gpu", 2),))
class TensorParallel:
    @func
    def step(
        x: Tensor[(4, 8), "f32"],
        w: ConstTensor[(8, 6), "f32"],
        state: Tensor[(4, 6), "f32"],
    ):
        with Mesh(("gpu",), (2,), names=("tp",)) as devices:
            a = tf.reshard(x, (4, 8 @ devices.tp), "gmem")
            b = tf.reshard(w, (8 @ devices.tp, 6), "gmem")
            partial = tf.matmul(a, b)
            complete = tf.allreduce(partial, mesh_axis=0)
            replicated_state = tf.reshard(state, (4, 6), "gmem")
            updated = tf.add(complete, replicated_state)
            return tf.neg(updated), updated
```

Each rank computes half the contraction. Its result is a partial sum;
`allreduce` completes that sum before adding state. The second output explicitly
carries the state for the next invocation.

```bash
set -euo pipefail
tilefoundry check projection.py:TensorParallel \
  --reference projection.py:Reference --distributed \
  --inputs random --weights random --device cpu \
  --out 'output[0]' --fn allclose --atol 1e-5 --rtol 1e-5 \
  --out 'output[1]' --fn allclose --atol 1e-5 --rtol 1e-5
```

```text
projection.py:TensorParallel
  reference: projection.py:Reference
  evaluation: distributed
  inputs:    random (seed 0); activations actual f32, f32 (declared f32, f32)

  output[0]   f32[4,6]   ref_norm 17.3306
    allclose(atol=1e-05 rtol=1e-05)    max_violation 0            PASS
  output[1]   f32[4,6]   ref_norm 17.3306
    allclose(atol=1e-05 rtol=1e-05)    max_violation 0            PASS

PASS
```

Both programs receive the same logical activations and weight resource. The
chosen tolerances are explicit for this small floating-point example; choose
them for the numerical policy of the actual program. Add `--json report.json`
to retain a machine-readable result identifying the candidate, reference and
evaluation mode.

Returning the partial sum raises an error. Replacing `allreduce` with a
`Reshard` to broadcast ownership also raises an error: a layout annotation
cannot silently perform cross-device communication. `allgather` assembles an
existing split, while `reducescatter` completes a partial reduction and leaves
its result split. The evaluator reconstructs split outputs for comparison.

For a stateful check, call `evaluate(candidate.load(resource), x, state,
distributed=True)` repeatedly and pass the returned state into the next call.
Keep an independent reference state and compare both the visible output and
the next state at every step. The CLI checks one invocation per dimension point;
it does not infer a state feedback loop.

The [evaluator contract](../spec/evaluator.md#7-distributed-evaluation) lists
supported local operations and partition shapes. This checks collective value
semantics, not a communication backend or performance. `analyze --engine`
models their device-level resources and communication; see the
[engine-analysis tutorial](./engine-analysis.md). Existing local analysis
families retain their separate operation-support boundaries.


Use `tilefoundry check --help` for comparison predicates and
`tilefoundry spec evaluator` for the evaluation contract.
