"""Check the independent prefill HIR forest and runtime weight ABI."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model  # noqa: E402


def main() -> int:
    failures = []
    root = model.KimiLinear48BA3BPrefill
    expected = tuple(model.LAYER_KINDS)
    actual = tuple(
        (
            "kda" if any(fn.name == "kda_prefill" for fn in child.mixer.functions) else "mla",
            "moe" if any(grandchild.name == "moe" for grandchild in child.modules) else "dense",
        )
        for child in root.modules
    )
    if len(root.modules) != 27:
        failures.append(f"layer count {len(root.modules)} != 27")
    if actual != expected:
        failures.append("layer mixer/FFN forest differs from LAYER_KINDS")

    names = {fn.name for node in _subtree(root) for fn in node.functions}
    required = {
        "embed_prefill",
        "kda_prefill",
        "mla_prefill",
        "moe_prefill",
        "mlp_prefill",
        "final_rms_norm_prefill",
        "lm_head_prefill",
    }
    missing = required - names
    if missing:
        failures.append(f"missing prefill functions: {sorted(missing)}")
    forbidden = {name for name in names if "decode" in name or "step" in name}
    if forbidden:
        failures.append(f"decode functions remain: {sorted(forbidden)}")

    moe = next(child.moe for child in root.modules if hasattr(child, "moe"))
    if "w_gate_up" not in moe.weights or "w_gate" in moe.weights or "w_up" in moe.weights:
        failures.append("MoE does not expose only packed w_gate_up")
    else:
        print(f"packed gate_up: {tuple(moe.weights['w_gate_up'].shape)}")

    print(f"model forest: {len(root.modules)} layers, {len(names)} function names")
    print("prefill-only names:", ", ".join(sorted(required)))
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print("PREFILL FOREST PASS")
    return 0


def _subtree(node):
    yield node
    for child in node.modules:
        yield from _subtree(child)


if __name__ == "__main__":
    raise SystemExit(main())
