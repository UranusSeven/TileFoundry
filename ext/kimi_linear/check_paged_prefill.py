"""Check paged MLA externally and real prefill logits against the HF oracle."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
for path in (_HERE, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import ops  # noqa: E402,F401

from tilefoundry import func, module  # noqa: E402
from tilefoundry.dsl import Tensor  # noqa: E402
from tilefoundry.dsl.tf import *  # noqa: E402,F401,F403
from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

VENV_HF = "/root/develop/yingshan/venv_hf/bin/python"
DEFAULT_DUMP = "/root/develop/yingshan/prefill_phase2_hf_logits.pt"
S, H, DQK, DV, PAGE, PAGES, LIVE = 100, 32, 192, 128, 16, 8, 7


@module(entry="attend")
class PagedPrefillDemo:
    @func
    def attend(
        q: Tensor[(S, H, DQK), "bf16"],
        k_pages: Tensor[(PAGES, PAGE, H, DQK), "bf16"],
        v_pages: Tensor[(PAGES, PAGE, H, DV), "bf16"],
        block_table: Tensor[(LIVE,), "i32"],
    ) -> Tensor[(S, H, DV), "bf16"]:
        return paged_mla_prefill(q, k_pages, v_pages, block_table)


@runtime_module(PagedPrefillDemo)
class PagedPrefillDemoTwin:
    @runtime_func
    def attend(self, q, k_pages, v_pages, block_table):
        import prefill  # noqa: PLC0415

        return prefill.flash_paged(
            q.cuda(),
            k_pages.cuda(),
            v_pages.cuda(),
            block_table.cuda().reshape(1, -1).contiguous(),
            q.shape[0],
        )


def check_op() -> bool:
    import torch  # noqa: PLC0415
    from ops import PagedMlaPrefill  # noqa: PLC0415

    from tests.ops.typeinfer_utils import infer_call  # noqa: PLC0415
    from tilefoundry.evaluator import evaluate  # noqa: PLC0415
    from tilefoundry.ir.core.errors import VerifyError  # noqa: PLC0415
    from tilefoundry.ir.types import DType, make_tensor_type  # noqa: PLC0415

    q_ty = make_tensor_type((S, H, DQK), DType.bf16)
    k_ty = make_tensor_type((PAGES, PAGE, H, DQK), DType.bf16)
    v_ty = make_tensor_type((PAGES, PAGE, H, DV), DType.bf16)
    bt_ty = make_tensor_type((LIVE,), DType.i32)
    out = infer_call(PagedMlaPrefill(), q_ty, k_ty, v_ty, bt_ty)
    ok = tuple(out.shape) == (S, H, DV) and out.dtype == DType.bf16
    for args in (
        (make_tensor_type((H, DQK), DType.bf16), k_ty, v_ty, bt_ty),
        (q_ty, make_tensor_type((PAGES, PAGE, 16, DQK), DType.bf16), v_ty, bt_ty),
        (q_ty, k_ty, v_ty, make_tensor_type((LIVE,), DType.f32)),
    ):
        try:
            infer_call(PagedMlaPrefill(), *args)
        except (VerifyError, ValueError):
            pass
        else:
            ok = False

    torch.manual_seed(0)
    q = torch.randn(S, H, DQK, dtype=torch.bfloat16)
    k = torch.randn(PAGES, PAGE, H, DQK, dtype=torch.bfloat16)
    v = torch.randn(PAGES, PAGE, H, DV, dtype=torch.bfloat16)
    blocks = torch.randperm(PAGES)[:LIVE].to(torch.int32)
    got = evaluate(PagedPrefillDemo.lookup("attend"), q, k, v, blocks, device="cpu")
    ref = PagedPrefillDemoTwin().attend(q, k, v, blocks).cpu()
    max_abs = (got.float() - ref.float()).abs().max().item()
    rel_l2 = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    good = max_abs < 2e-2 and rel_l2 < 2e-2
    print(f"paged MLA op: max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}")
    return ok and good


def ensure_hf_dump(path: str) -> None:
    if os.path.exists(path):
        return
    script = os.path.join(_HERE, "hf_prefill_logits.py")
    subprocess.run([VENV_HF, script, "--out", path], check=True, env=dict(os.environ))


def check_logits(path: str) -> bool:
    import torch  # noqa: PLC0415
    from prefill import PrefillRunner  # noqa: PLC0415

    oracle = torch.load(path, weights_only=False)
    runner = PrefillRunner()
    ok = True
    for length in (8, 32, 100):
        expected = oracle[length]
        logits = runner(expected["input_ids"])[0].float().cpu()
        ref = expected["logits"].float()
        finite = bool(torch.isfinite(logits).all().item())
        shape = tuple(logits.shape) == tuple(ref.shape) == (163840,)
        argmax = int(logits.argmax()) == expected["argmax"]
        rel_l2 = ((logits - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        passed = finite and shape and argmax and rel_l2 <= 6e-2
        ok &= passed
        print(
            f"S={length}: finite={finite} shape={tuple(logits.shape)} "
            f"argmax={int(logits.argmax())}/{expected['argmax']} "
            f"rel_l2={rel_l2:.6e} {'PASS' if passed else 'FAIL'}"
        )
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    args = parser.parse_args(argv)
    ok = check_op()
    if args.oracle:
        ensure_hf_dump(args.dump)
        ok &= check_logits(args.dump)
    print("PAGED PREFILL PASS" if ok else "PAGED PREFILL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
