"""The runtime twins vs the authored evaluator, on real checkpoint weights.

The kernel lines validated each tilelang kernel against the authored IR on
*random* weights; this script validates the M1 integration: the twins'
plumbing (ConstTensor filling, orchestration) and `weights.py`'s loading
(alias table, converters, expert stacking) on *real* weights. A wrong alias
or a misplaced transpose shows up here as a large diff, not as gibberish 98
GB later.

Coverage:
  1. the root's own functions -- embed / final_rms_norm / lm_head -- twin
     against `evaluate` of the authored body, weights passed explicitly;
  2. one layer of each kind -- 0 (KDA+dense), 1 (KDA+MoE), 3 (MLA+MoE) --
     whole-layer forward, twin against a LoadedModule over the same tensors.

The MLA layer runs the way the driver runs it: fixed-capacity cache (bucket
128) with 24 live positions, `pos_ids = [24]`, against the authored path's
exact-length cache and `pos_ids = [0]` (the identity rope makes the results
comparable -- that equivalence is part of what is being checked).

Reproduce (container dev-yingshan-7cf9dbcf45-xtm8p):

  cd /root/develop/yingshan/TileFoundry && \
    CUDA_VISIBLE_DEVICES=5 python3 ext/kimi_linear/check_twin.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as sem  # noqa: E402
import torch  # noqa: E402
import weights as wt  # noqa: E402

from tilefoundry.evaluator import evaluate  # noqa: E402

DEVICE = "cuda"
BF16 = torch.bfloat16
TOL = 2e-2

_HID = 2304
_NH, _DK = 32, 128
_KP = _NH * _DK
_NOPE, _ROPE = 128, 64
_QK = _NOPE + _ROPE
_V = 128

MLA_LIVE = 24  # live positions in the MLA layer's bucket-128 cache


def report(name, a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        print(f"  {name:24s} SHAPE  {tuple(a.shape)} vs {tuple(b.shape)}  FAIL")
        return False
    d = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=TOL, rtol=TOL)
    print(f"  {name:24s} max|d|={d:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def loaded_view(twin_node):
    """A LoadedModule over a twin's own bound weights (no re-reading)."""
    from tilefoundry.ir.core.module import LoadedModule  # noqa: PLC0415

    return LoadedModule(
        module=twin_node.module,
        constants=twin_node._bound,
        modules=tuple(loaded_view(c) for c in twin_node.modules),
    )


def drawn(*shape, sigma=0.1, seed=1):
    return (torch.randn(*shape, device=DEVICE, generator=_gen.manual_seed(seed))
            * sigma).to(BF16)


_gen = torch.Generator(device=DEVICE)


def check_root_funcs(resource):
    """Embed / final_rms_norm / lm_head: twin bodies vs the authored ones."""
    import runtime_model as rt  # noqa: PLC0415

    print("== root functions ==")
    ok = True
    table = resource.load("table")
    gamma_final = resource.load("gamma_final")
    w_head = resource.load("w_head")

    ids = torch.tensor([1000, 42, 163000], dtype=torch.int64, device=DEVICE)
    for tid in ids:
        one = tid.reshape(1)
        ok &= report(
            f"embed[{int(tid)}]",
            rt._embed(None, table, one),
            evaluate(sem.KimiLinear48BA3B.lookup("embed"), table, one, device=DEVICE),
        )

    hidden = drawn(1, 1, _HID, sigma=0.5, seed=11)
    ok &= report(
        "final_rms_norm",
        rt._final_rms_norm(None, hidden, gamma_final),
        evaluate(
            sem.KimiLinear48BA3B.lookup("final_rms_norm"), hidden, gamma_final,
            device=DEVICE,
        ),
    )
    ok &= report(
        "lm_head",
        rt._lm_head(None, hidden, w_head),
        evaluate(sem.KimiLinear48BA3B.lookup("lm_head"), hidden, w_head, device=DEVICE),
    )
    return ok


def check_layer(resource, index, scale, routed_scale):
    """One layer's whole forward: twin vs LoadedModule over the same weights."""
    import runtime_model as rt  # noqa: PLC0415

    layer_res = resource.subtree(f"layer{index}")
    kind = sem.LAYER_KINDS[index]
    twin = rt._LAYER_TWIN[kind]()
    twin.load(layer_res)
    authored = loaded_view(twin)

    hidden = drawn(1, 1, _HID, sigma=0.3, seed=100 + index)
    mixer, _ffn = kind
    if mixer == "kda":
        cache = tuple(drawn(1, 3, _KP, sigma=0.2, seed=200 + index) for _ in range(3))
        cache += (drawn(1, _NH, _DK, _DK, sigma=0.05, seed=300 + index),)
        mixer_args = (*cache, scale["kda"])
    else:
        # The driver's call shape: bucket-128 buffers, MLA_LIVE live slots,
        # pos_ids = [MLA_LIVE]; the authored call: exact 24-long caches,
        # pos_ids = [0], one-row identity rope.
        cap = 128
        k_buf = torch.zeros(1, cap, _NH, _QK, dtype=BF16, device=DEVICE)
        v_buf = torch.zeros(1, cap, _NH, _V, dtype=BF16, device=DEVICE)
        k_buf[:, :MLA_LIVE] = drawn(MLA_LIVE, _NH, _QK, sigma=0.2, seed=400 + index)
        v_buf[:, :MLA_LIVE] = drawn(MLA_LIVE, _NH, _V, sigma=0.2, seed=500 + index)
        cos_cap = torch.ones(cap, _ROPE, dtype=BF16, device=DEVICE)
        sin_cap = torch.zeros(cap, _ROPE, dtype=BF16, device=DEVICE)
        pos_cap = torch.tensor([MLA_LIVE], dtype=torch.int32, device=DEVICE)
        mixer_args = (cos_cap, sin_cap, pos_cap, k_buf, v_buf, scale["mla"])

        cos1 = torch.ones(1, _ROPE, dtype=BF16, device=DEVICE)
        sin1 = torch.zeros(1, _ROPE, dtype=BF16, device=DEVICE)
        pos1 = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        authored_args = (cos1, sin1, pos1, k_buf[:, :MLA_LIVE].contiguous(),
                         v_buf[:, :MLA_LIVE].contiguous(), scale["mla"])

    print(f"== layer{index} ({mixer}, {kind[1]}) ==")
    out_t, state_t = twin(hidden, mixer_args, routed_scale)
    out_a, state_a = authored(
        hidden, mixer_args if mixer == "kda" else authored_args, routed_scale
    )
    ok = report("layer out", out_t, out_a)
    for j, (st, sa) in enumerate(zip(state_t, state_a)):
        ok &= report(f"state[{j}]", st, sa)
    return ok


def main() -> int:
    resource, tally = wt.decoder_resource(sem.KimiLinear48BA3B, device=DEVICE)
    scale = {
        "kda": torch.full((1, 1, 1), _DK ** -0.5, dtype=BF16, device=DEVICE),
        "mla": torch.full((1, 1, 1, 1), _QK ** -0.5, dtype=BF16, device=DEVICE),
    }
    routed_scale = torch.full(
        (1, 1), sem.config.routed_scaling_factor, dtype=BF16, device=DEVICE
    )

    ok = check_root_funcs(resource)
    for index in (0, 1, 3):
        ok &= check_layer(resource, index, scale, routed_scale)

    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
