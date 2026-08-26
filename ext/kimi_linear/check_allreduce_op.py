"""The `tf.all_reduce` op, exercised end to end.

A two-rank row-split matmul: `x` and `w` carry `Split` layouts over a
`gpu x 2` mesh on their contraction dims, so `matmul` propagation hands the
product a `Partial("sum")` state ([shard §8](docs/spec/shard.md#8-layout-propagation))
and `all_reduce` is the op that converges it to `Broadcast` -- the layout
transition being validated, in one authored function.

What is asserted:

1. **typeinfer**: `all_reduce` on a `Partial("sum")` input yields the same
   type with that axis `Broadcast`; on a `Split`-carrying input it errors
   (that is an all-gather); on an already-converged sharded input it errors
   (nothing to converge); on an unsharded input it passes through (the
   single-device case).
2. **eval**: identity on the logical value -- `evaluate(converge, x, w)`
   equals `x @ w`, because a one-process allreduce changes nothing.
3. **access relation**: registered and exercised by the analysis the check
   below runs.
4. **the op in an authored function, checked**: the twin below computes the
   converged value directly (`x @ w`), so

       tilefoundry check ext/kimi_linear/check_allreduce_op.py:AllReduceDemoTwin.converge \
           --inputs random --out output --fn allclose --atol 1e-5 --rtol 1e-5

   runs the authored body -- matmul propagation producing the Partial,
   `all_reduce` converging it, the evaluator's identity -- against torch.

Reproduce (container dev-yingshan-7cf9dbcf45-xtm8p):

    cd /root/develop/yingshan/TileFoundry && \
      python3 ext/kimi_linear/check_allreduce_op.py && \
      tilefoundry check ext/kimi_linear/check_allreduce_op.py:AllReduceDemoTwin.converge \
        --inputs random --out output --fn allclose --atol 1e-5 --rtol 1e-5
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ops  # noqa: E402,F401 -- registers tf.all_reduce
import torch  # noqa: E402
from ops import AllReduce  # noqa: E402

from tilefoundry import func, module  # noqa: E402
from tilefoundry.dsl import Tensor, tf  # noqa: E402,F401 -- tf used by the @func body
from tilefoundry.dsl.tf import *  # noqa: E402,F403 -- bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.ir.core.errors import VerifyError  # noqa: E402
from tilefoundry.ir.types import DType, make_shard_tensor_type, make_tensor_type  # noqa: E402
from tilefoundry.ir.types.shard import Topology, make_mesh  # noqa: E402
from tilefoundry.ir.types.shard.shard_layout import (  # noqa: E402
    Broadcast,
    Partial,
    Split,
    canonical_shard_layout,  # noqa: E402
)
from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

#: A two-rank device mesh. `make_mesh` defaults the topology name to "gpu",
#: which is what a torchrun pair is.
MESH = make_mesh((2,))

#: `x`: the (1, 16) activation, its K axis split across the mesh.
#: `w`: the (16, 32) weight, its K axis split the same way -- matmul requires
#: the contraction dim to be split on the same mesh axes for both operands.
X_SPLIT_K = canonical_shard_layout((1, 16), MESH, (Split(1),))
W_SPLIT_K = canonical_shard_layout((16, 32), MESH, (Split(0),))


@module(entry="converge", topologies=(Topology("gpu", 2),))
class AllReduceDemo:
    """The smallest authored function with a collective in it."""

    @func
    def converge(
        x: Tensor[(1, 16), "f32", X_SPLIT_K],
        w: Tensor[(16, 32), "f32", W_SPLIT_K],
    ) -> Tensor[(1, 32), "f32"]:
        # The matmul's K dim is mesh-split on both sides, so the product is
        # Partial("sum") over the mesh axis; all_reduce converges it.
        return all_reduce(matmul(x, w))


@runtime_module(AllReduceDemo)
class AllReduceDemoTwin:
    """The converged value, computed directly (what rank-pair NCCL buys)."""

    @runtime_func
    def converge(self, x, w):
        return torch.matmul(x, w)


def main() -> int:
    from tests.ops.typeinfer_utils import infer_call  # noqa: PLC0415

    ok = True

    # -- 1. typeinfer: the Partial -> converged transition -------------------
    psum = make_shard_tensor_type((16, 32), DType.f32, mesh=MESH, attrs=(Partial("sum"),))
    out = infer_call(AllReduce(), psum)
    from tilefoundry.ir.types.shard.shard_layout import shard_layout_of  # noqa: PLC0415

    out_attrs = shard_layout_of(out.layout).attrs
    ok &= isinstance(out_attrs[0], Broadcast)
    print(f"  typeinfer Partial('sum') -> {type(out_attrs[0]).__name__}"
          f"  {'PASS' if isinstance(out_attrs[0], Broadcast) else 'FAIL'}")

    for name, ty in (
        ("split input refused",
         make_shard_tensor_type((16, 32), DType.f32, mesh=MESH, attrs=(Split(1),))),
        ("converged input refused",
         make_shard_tensor_type((16, 32), DType.f32, mesh=MESH, attrs=(Broadcast(),))),
    ):
        try:
            infer_call(AllReduce(), ty)
        except VerifyError as error:
            print(f"  typeinfer {name}: refused as intended  PASS  ({error})")
        else:
            ok = False
            print(f"  typeinfer {name}: accepted, should not  FAIL")

    plain = make_tensor_type((16, 32), DType.f32)
    same = infer_call(AllReduce(), plain)
    ok &= same.layout is None
    print(f"  typeinfer unsharded passthrough"
          f"  {'PASS' if same.layout is None else 'FAIL'}")

    # -- 2. eval: identity on the logical value ------------------------------
    torch.manual_seed(0)
    x = torch.randn(1, 16)
    w = torch.randn(16, 32)
    got = evaluate(AllReduceDemo.lookup("converge"), x, w, device="cpu")
    want = x @ w
    d = (got - want).abs().max().item()
    ok &= d == 0.0
    print(f"  eval: evaluate(converge) == x @ w, max|d|={d:.3e}"
          f"  {'PASS' if d == 0.0 else 'FAIL'}")

    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
