"""Small type-inference and evaluator smoke for ``tf.kda_prefill``."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ops  # noqa: E402,F401
import torch  # noqa: E402

from tilefoundry import func  # noqa: E402
from tilefoundry.dsl import Tensor, tf  # noqa: E402
from tilefoundry.evaluator import evaluate  # noqa: E402


@func
def kda_smoke(
    q: Tensor[(1, 4, 2, 3), "f32"],
    k: Tensor[(1, 4, 2, 3), "f32"],
    v: Tensor[(1, 4, 2, 3), "f32"],
    g: Tensor[(1, 4, 2, 3), "f32"],
    beta: Tensor[(1, 4, 2), "f32"],
) -> Tensor[(1, 4, 2, 3), "f32"]:
    return tf.kda_prefill(q, k, v, g, beta, scale=3 ** -0.5)


def main() -> int:
    torch.manual_seed(0)
    args = [torch.randn(1, 4, 2, 3) for _ in range(3)]
    args += [-torch.rand(1, 4, 2, 3), torch.sigmoid(torch.randn(1, 4, 2))]
    out = evaluate(kda_smoke, *args)
    print("type:", kda_smoke.return_type)
    print("eval:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    return 0 if out.shape == (1, 4, 2, 3) and torch.isfinite(out).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
