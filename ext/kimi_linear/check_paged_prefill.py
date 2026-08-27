"""The M3-B paged prefill check.

The `tf.paged_mla_prefill` op, then the whole
paged prefill against the HF oracle's own prefill, on the real checkpoint.

Part A (fast): the op's four pieces, in the shape `check_allreduce_op.py` set.

  1. typeinfer: output shape/dtype from the inputs; malformed inputs refused.
  2. eval: `evaluate(attend)` -- the op's stated torch semantics -- against
     the production FA3 paged kernel on the same random pages, with a
     permuted block table so the indirection is exercised.
  3. access relation: exercised by the CLI check below (it runs the
     analyses over the authored function).
  4. the op in an authored function, checked against its runtime twin, which
     calls the production kernel:

       tilefoundry check ext/kimi_linear/check_paged_prefill.py:PagedPrefillDemoTwin.attend \
           --inputs random --out output --fn allclose --atol 2e-2 --rtol 2e-2

Part B (real weights, one GPU): paged prefill logits vs the official
checkpoint implementation's prefill logits, last position, prompts of 8 and
100 tokens. The HF side runs under venv_hf (transformers 4.57.1 -- the
container's 5.15 cannot import the vendored modeling code); this script
drives it as a subprocess with `--hf-dump`, which writes the logits the main
run then diffs. Both sides are fed the same token ids (the M1 lesson: check
the ids, not the strings).

    CUDA_VISIBLE_DEVICES=3 python3 ext/kimi_linear/check_paged_prefill.py            # part A
    CUDA_VISIBLE_DEVICES=3 python3 ext/kimi_linear/check_paged_prefill.py --oracle   # part B
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT = os.environ.get(
    "KIMI_LINEAR_CKPT", "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct"
)
VENV_HF = "/root/develop/yingshan/venv_hf/bin/python"

#: Part A demo dims: S=100 over 8 pages of 16 (7 live), heads and dims are
#: the published MLA's.
_S, _H, _DQK, _DV, _PS, _PAGES, _LIVE = 100, 32, 192, 128, 16, 8, 7


#: The HF-dump subprocess runs under venv_hf, where TileFoundry is not
#: importable (its isl binding predates the system lib); skip the op section
#: there -- that mode only needs torch/transformers/fla below.
_HF_DUMP = "--hf-dump" in sys.argv

# ---------------------------------------------------------------------------
# The authored demo function and its twin (part A, piece 4).
# ---------------------------------------------------------------------------

if not _HF_DUMP:
    import ops  # noqa: E402,F401 -- registers tf.paged_mla_prefill

    from tilefoundry import func, module  # noqa: E402
    from tilefoundry.dsl import Tensor  # noqa: E402
    from tilefoundry.dsl.tf import *  # noqa: E402,F401,F403 -- bare op bindings
    from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

    @module(entry="attend")
    class PagedPrefillDemo:
        """Causal prefill attention over a paged cache, as an authored function."""

        @func
        def attend(
            q: Tensor[(_S, _H, _DQK), "bf16"],
            k_pages: Tensor[(_PAGES, _PS, _H, _DQK), "bf16"],
            v_pages: Tensor[(_PAGES, _PS, _H, _DV), "bf16"],
            block_table: Tensor[(_LIVE,), "i32"],
        ) -> Tensor[(_S, _H, _DV), "bf16"]:
            return paged_mla_prefill(q, k_pages, v_pages, block_table)

    @runtime_module(PagedPrefillDemo)
    class PagedPrefillDemoTwin:
        """The production call: FA3 over the pages the block table names."""

        @runtime_func
        def attend(self, q, k_pages, v_pages, block_table):
            # The CLI removes the file's own directory from sys.path after
            # loading it, so a call-time import has to put it back.
            if _HERE not in sys.path:
                sys.path.insert(0, _HERE)
            import prefill as pf  # noqa: PLC0415

            return pf.flash_paged(
                q.cuda(),
                k_pages.cuda(),
                v_pages.cuda(),
                block_table.cuda().reshape(1, -1).contiguous(),
                q.shape[0],
            )


def _random_pages(seed=0):
    """Random pages with a *permuted* block table: the indirection is real."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    q = torch.randn(_S, _H, _DQK, dtype=torch.bfloat16)
    k_pages = torch.randn(_PAGES, _PS, _H, _DQK, dtype=torch.bfloat16)
    v_pages = torch.randn(_PAGES, _PS, _H, _DV, dtype=torch.bfloat16)
    block_table = torch.randperm(_PAGES)[:_LIVE].to(torch.int32)
    return q, k_pages, v_pages, block_table


def check_op() -> bool:
    """Part A: typeinfer cases, then eval against the FA3 production kernel."""
    from ops import PagedMlaPrefill  # noqa: PLC0415

    from tests.ops.typeinfer_utils import infer_call  # noqa: PLC0415
    from tilefoundry.evaluator import evaluate  # noqa: PLC0415
    from tilefoundry.ir.core.errors import VerifyError  # noqa: PLC0415
    from tilefoundry.ir.types import DType, make_tensor_type  # noqa: PLC0415

    ok = True

    q_ty = make_tensor_type((_S, _H, _DQK), DType.bf16)
    k_ty = make_tensor_type((_PAGES, _PS, _H, _DQK), DType.bf16)
    v_ty = make_tensor_type((_PAGES, _PS, _H, _DV), DType.bf16)
    bt_ty = make_tensor_type((_LIVE,), DType.i32)
    out = infer_call(PagedMlaPrefill(), q_ty, k_ty, v_ty, bt_ty)
    good = tuple(out.shape) == (_S, _H, _DV) and out.dtype == DType.bf16
    ok &= good
    print(f"  typeinfer: [S,H,Dqk]xpages -> {tuple(out.shape)} {out.dtype}"
          f"  {'PASS' if good else 'FAIL'}")

    for name, args in (
        ("q rank", (make_tensor_type((_H, _DQK), DType.bf16), k_ty, v_ty, bt_ty)),
        ("head mismatch", (q_ty, make_tensor_type((_PAGES, _PS, 16, _DQK), DType.bf16), v_ty, bt_ty)),
        ("block_table dtype", (q_ty, k_ty, v_ty, make_tensor_type((_LIVE,), DType.f32))),
    ):
        try:
            infer_call(PagedMlaPrefill(), *args)
        except (VerifyError, ValueError):
            print(f"  typeinfer {name}: refused as intended  PASS")
        else:
            ok = False
            print(f"  typeinfer {name}: accepted, should not  FAIL")

    # eval (the op's torch semantics) vs the FA3 kernel it declares.
    import prefill as pf  # noqa: PLC0415

    q, k_pages, v_pages, block_table = _random_pages()
    got = evaluate(PagedPrefillDemo.lookup("attend"), q, k_pages, v_pages,
                   block_table, device="cpu")
    ref = pf.flash_paged(
        q.cuda(), k_pages.cuda(), v_pages.cuda(),
        block_table.cuda().reshape(1, -1).contiguous(), _S,
    ).cpu()
    d = (got.float() - ref.float()).abs().max().item()
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    # bf16 in, bf16 out, two attention implementations over unit-gaussian
    # data: sub-1e-2 absolute, matching the FA3-vs-f32 probe floor.
    good = d < 2e-2 and rel < 2e-2
    ok &= good
    print(f"  eval vs FA3 kernel: max|d|={d:.3e} rel_l2={rel:.3e}"
          f"  {'PASS' if good else 'FAIL'}")

    print()
    print("part A VERDICT:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Part B: paged prefill vs the HF oracle's prefill, real checkpoint.
# ---------------------------------------------------------------------------

def hf_dump(path: str, prompts: dict[str, list[int]]) -> None:
    """The HF side, under venv_hf: prefill logits at the last position.

    Same three shims as hf_greedy.py (auto_docstring, the fla 0.5.2 gate
    signature, eager attention), then one forward per prompt.
    """
    import torch  # noqa: PLC0415
    import transformers.utils as _tu  # noqa: PLC0415
    import transformers.utils.auto_docstring as _ad  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    def _noop_docstring(obj=None, **kwargs):
        return (lambda o: o) if obj is None else obj

    _tu.auto_docstring = _noop_docstring
    _ad.auto_docstring = _noop_docstring

    import fla.ops.kda.gate as _fla_gate  # noqa: PLC0415
    from fla.ops.kda.gate import naive_kda_gate  # noqa: PLC0415

    def _gate_shim(g, A_log, head_dim, g_bias=None):
        *lead, hk = g.shape
        h = hk // head_dim
        return naive_kda_gate(
            g.view(*lead, h, head_dim), A_log.reshape(-1), g_bias
        )

    _fla_gate.fused_kda_gate = _gate_shim

    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.config._attn_implementation = "eager"
    model.eval()

    out = {}
    with torch.no_grad():
        for name, ids in prompts.items():
            t = torch.tensor([ids], dtype=torch.long, device="cuda")
            logits = model(t).logits
            out[name] = logits[0, -1].float().cpu()
            print(f"  hf prefill {name}: {len(ids)} tokens done", flush=True)
    torch.save(out, path)
    print(f"wrote {path}")


def check_oracle(dump: str, prompts: dict[str, list[int]]) -> bool:
    """The TileFoundry side: paged prefill logits, diffed two ways.

    Against the HF oracle's dump (two full implementations apart: HF's
    fused_recurrent/chunk and eager/flash attention versus fla chunk_kda and
    FA3 -- the M1 per-mixer floor between any two legs was already 3e-2, so
    this is a smoke gate at rel_l2 < 1e-1 plus argmax), and against the same
    session's loop prefill (the M1 decode kernels, the validated production
    decode path -- the tight semantic gate, rel_l2 < 5e-2 plus argmax).
    """
    import prefill as pf  # noqa: PLC0415
    import runtime_model as rt  # noqa: PLC0415
    import torch  # noqa: PLC0415

    theirs = torch.load(dump)
    # `max` is the DSL op here (the tf star-import shadows builtins), so sort.
    capacity = sorted(len(ids) for ids in prompts.values())[-1] + 8
    session = rt.Session(capacity=capacity)
    ok = True
    for name, ids in prompts.items():
        session.reset()
        loop = None
        for token in ids:
            loop = session.step(token)
        loop = loop.float().reshape(-1).cpu()
        session.reset()
        paged = pf.prefill(session, ids).float().reshape(-1).cpu()
        ref = theirs[name].reshape(-1)

        def report(tag, a, b, gate):
            d = (a - b).abs().max().item()
            rel = ((a - b).norm() / b.norm().clamp_min(1e-9)).item()
            same = int(a.argmax()) == int(b.argmax())
            good = same and rel < gate
            print(f"  {name} ({len(ids)} tokens) paged vs {tag}: "
                  f"max|d|={d:.3e} rel_l2={rel:.3e} "
                  f"argmax {'agree' if same else 'DIFFER'}"
                  f"  {'PASS' if good else 'FAIL'}")
            return good

        ok &= report("loop prefill", paged, loop, 5e-2)
        ok &= report("HF oracle", paged, ref, 1e-1)
        ok &= report("HF oracle (loop leg)", loop, ref, 1e-1)
    print()
    print("part B VERDICT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", action="store_true",
                   help="run part B (real weights, HF oracle subprocess)")
    p.add_argument("--hf-dump", metavar="PATH", default=None,
                   help="(venv_hf) write HF prefill logits to PATH and exit")
    p.add_argument("--prompt-ids", default=None,
                   help="with --hf-dump: 'name:id,id,...;name:id,id,...'")
    p.add_argument("--dump", default="/root/develop/yingshan/m3_scratch/hf_prefill_logits.pt")
    args = p.parse_args()

    if args.hf_dump is not None:
        prompts = {}
        for entry in args.prompt_ids.split(";"):
            name, _, ids = entry.partition(":")
            prompts[name] = [int(x) for x in ids.split(",")]
        hf_dump(args.hf_dump, prompts)
        return 0

    if not args.oracle:
        return 0 if check_op() else 1

    # Fix the token ids on this side first, then hand them to the oracle.
    import weights as wt  # noqa: PLC0415

    tok = wt.tokenizer()
    base = tok.encode(
        "The history of the transformer architecture begins with attention. "
        "Before 2017, sequence models were dominated by recurrent networks, "
        "which processed tokens one at a time and struggled with long range "
        "dependencies. The key insight of the transformer was to replace "
        "recurrence entirely with self attention, allowing every position to "
        "attend to every other position in parallel during training. This "
        "change unlocked much larger models, because the training step no "
        "longer serialized along the sequence axis, and it created the "
        "inference problem this project measures: a fast decode loop needs "
        "the prompt consumed by a different, wider kernel first."
    )
    prompts = {"s8": base[:8], "s100": base[:100]}
    print("prompt ids (must match on both sides):")
    for name, ids in prompts.items():
        print(f"  {name}: {ids[:8]}... ({len(ids)} tokens)")

    if not os.path.exists(args.dump):
        spec = ";".join(f"{n}:{','.join(str(i) for i in ids)}"
                        for n, ids in prompts.items())
        import subprocess  # noqa: PLC0415

        env = dict(os.environ)
        r = subprocess.run(
            [VENV_HF, os.path.abspath(__file__), "--hf-dump", args.dump,
             "--prompt-ids", spec],
            env=env,
        )
        if r.returncode != 0:
            print("hf dump failed", file=sys.stderr)
            return 2
    return 0 if check_oracle(args.dump, prompts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
