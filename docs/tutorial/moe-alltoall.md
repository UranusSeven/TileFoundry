# MoE all-to-all: dispatch, experts, and combine

MoE communication routes each token to its selected experts, then restores source
order and sums their weighted outputs. This example uses two independent data
parallel groups, each with two expert parallel ranks, all simulated on a CPU.
The reference and candidate take the same logical tokens, routes and weights.

Save this page as `moe-alltoall.md` and extract its programs:

```bash
set -euo pipefail
for name in moe.py inputs.py; do
  awk -v tag="<!-- tilefoundry-source: $name -->" '
    $0 == tag { block=1; next }
    block && /^```python$/ { in_python=1; next }
    in_python && /^```$/ { in_python=0; block=0; next }
    in_python { print }
  ' moe-alltoall.md > "$name"
done
```

The reference computes each expert in ordinary HIR. The candidate dispatches
into expert-major buffers, runs the same MLP with batched matrix multiplications,
applies the transported router weights once, and combines the results.
`source_indices` and `counts` are ordinary HIR tensors, so the return route is
explicit and can be checked.

<!-- tilefoundry-source: moe.py -->

```python
# example
"""A nonlinear MoE expressed as reference HIR and bounded expert parallel HIR."""

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Mesh, Tensor, Topology, tf

D, N, H, F, E, K = 2, 8, 4, 6, 4, 3
CAPACITY = N * K


@module(entry="run")
class Reference:
    @func
    def run(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
        w1: ConstTensor[(E, H, F), "f32"],
        w2: ConstTensor[(E, F, H), "f32"],
    ):
        result = tf.full_like(x, value=0.0)
        zeros = tf.full_like(weights, value=0.0)
        for expert in range(E):
            up = tf.slice(w1, (expert, 0, 0), sizes=(1, H, F), strides=(1, 1, 1))
            down = tf.slice(w2, (expert, 0, 0), sizes=(1, F, H), strides=(1, 1, 1))
            expert_result = tf.matmul(tf.relu(tf.matmul(x, up)), down)
            selected = tf.where(tf.cmp_eq(indices, expert), weights, zeros)
            scale = tf.reduce(selected, axes=(-1,), keepdim=True, kind="sum")
            result = tf.add(result, tf.mul(expert_result, scale))
        return result


@module(entry="run", topologies=(Topology("gpu", 4),))
class ExpertParallel:
    @func
    def dispatch(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
    ):
        with Mesh(("gpu",), (2, 2), names=("dp", "ep")) as devices:
            source = tf.reshard(x, (D @ devices.dp, N @ devices.ep, H), "gmem")
            routes = tf.reshard(indices, (D @ devices.dp, N @ devices.ep, K), "gmem")
            scales = tf.reshard(weights, (D @ devices.dp, N @ devices.ep, K), "gmem")
            return tf.alltoall_dispatch(
                source,
                routes,
                scales,
                num_experts=E,
                capacity=CAPACITY,
                mesh_axis=1,
            )

    @func
    def run(
        x: Tensor[(D, N, H), "f32"],
        indices: Tensor[(D, N, K), "i64"],
        weights: Tensor[(D, N, K), "f32"],
        w1: ConstTensor[(E, H, F), "f32"],
        w2: ConstTensor[(E, F, H), "f32"],
    ):
        packed, scales, source_indices, counts = dispatch(x, indices, weights)  # noqa: F821
        with Mesh(("gpu",), (2, 2), names=("dp", "ep")) as devices:
            source = tf.reshard(x, (D @ devices.dp, N @ devices.ep, H), "gmem")
            up = tf.reshard(w1, (E @ devices.ep, H, F), "gmem")
            down = tf.reshard(w2, (E @ devices.ep, F, H), "gmem")
            expert_result = tf.matmul(tf.relu(tf.matmul(packed, up)), down)
            weighted = tf.mul(expert_result, scales)
            return tf.alltoall_combine(weighted, source_indices, counts, source, mesh_axis=1)
```

Each expert has `CAPACITY` allocated rows, with its actual row count returned by
dispatch. This example chooses the worst-case bound `N * K`, which also covers
repeated expert choices. Smaller bounds are allowed; exceeding one raises an
error rather than dropping tokens. Expert IDs are global within an EP group;
`-1` disables a choice.

Routes are structured inputs. Generate valid IDs explicitly, including an
inactive token, several experts on one destination, and repeated selections:

<!-- tilefoundry-source: inputs.py -->

```python
# example
import torch
from moe import D, N, H, E, K

generator = torch.Generator().manual_seed(31)
tokens = torch.randn(D, N, H, generator=generator)
indices = torch.randint(0, E, (D, N, K), generator=generator)
weights = torch.softmax(torch.randn(D, N, K, generator=generator), dim=-1)
indices[0, 0] = torch.tensor([0, 1, 3])
indices[0, 1] = -1
indices[1, 0] = torch.tensor([2, 2, -1])
for name, value in (
    ("tokens.pt", tokens), ("expert_ids.pt", indices), ("router_weights.pt", weights),
):
    torch.save(value, name)
```

```bash
set -euo pipefail
python inputs.py
tilefoundry check moe.py:ExpertParallel --reference moe.py:Reference \
  --distributed --inputs files:tokens.pt,expert_ids.pt,router_weights.pt \
  --weights random --device cpu \
  --out output --fn allclose --atol 2e-5 --rtol 2e-5 --fn nan_inf
```

```text
moe.py:ExpertParallel
  reference: moe.py:Reference
  evaluation: distributed
  inputs:    files:tokens.pt,expert_ids.pt,router_weights.pt; activations actual f32, i64, f32 (declared none); files tokens.pt: 1 tensor(s) f32[2, 8, 4]; expert_ids.pt: 1 tensor(s) i64[2, 8, 3]; router_weights.pt: 1 tensor(s) f32[2, 8, 3]

  output   f32[2,8,4]   ref_norm 20.3158
    allclose(atol=2e-05 rtol=2e-05)    max_violation 0            PASS
    nan_inf                            nan 0 inf 0                PASS

PASS
```

Inspecting dispatch separately exposes the live counts and bounded storage:
weights, padding and source indices stay available as outputs of the same HIR.

```bash
set -euo pipefail
python - <<'PY'
import torch
from moe import ExpertParallel
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core.module import select
from tilefoundry.runtime import DictResource

inputs = [torch.load(path, weights_only=True) for path in
          ('tokens.pt', 'expert_ids.pt', 'router_weights.pt')]
packed, scales, origins, counts = evaluate(
    select(ExpertParallel, 'dispatch').load(DictResource({})),
    *inputs, distributed=True,
)
print('Live rows per batch and expert:', counts.tolist())
print('Capacity per expert:', packed.shape[-2])
print('Packed token shape:', tuple(packed.shape))
PY
```

```text
Live rows per batch and expert: [[7, 5, 4, 5], [6, 4, 8, 5]]
Capacity per expert: 24
Packed token shape: (2, 4, 24, 4)
```

The semantic reference is [DeepEP expanded dispatch and combine](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/elastic.py#L855)
and the [vLLM integration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py).
vLLM applies router weighting before calling combine. Here that multiplication
is visible in HIR, and explicit indices/counts provide the role of routing
metadata retained in a communication handle.

These bounded expert-major buffers describe expanded expert work. A backend
can deduplicate transport when several experts share a destination rank;
allocated slots and live expert-row counts do not by themselves give network
bytes. This HIR representation is independent of DeepEP's buffer ABI, stream
management and quantized transport formats.

`tf.alltoall` also supports regular equal-size redistribution between split
tensor axes. The routed operations above handle data-dependent expert choices.
The [HIR contract](../spec/hir.md#31-bounded-moe-dispatch-and-combine) defines
ownership, ordering, metadata validation and overflow behavior. Use
`tilefoundry check --help` for predicates and `tilefoundry spec hir` for the
operation definitions.

This example validates HIR value semantics in the rank simulator. The [engine-analysis tutorial](./engine-analysis.md)
shows device-level resource estimates through `analyze --engine`. Routing
profiles refine its traffic bounds. Physical GPU lowering remains separate
work, and local analysis families keep their existing support boundaries.
