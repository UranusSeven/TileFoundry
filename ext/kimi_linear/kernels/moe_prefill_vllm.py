"""vLLM fused-MoE adapter for Kimi prefill."""
from __future__ import annotations

import importlib
import os
import sys
import types

# The installed fused_moe package initializer imports optional NIXL/DeepEP
# extensions whose protobuf ABI conflicts with TileFoundry OR-Tools. The fused
# kernel module itself has no such dependency, so load it as a namespace child.
_PACKAGE = "vllm.model_executor.layers.fused_moe"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = ["/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe"]
    package.__package__ = _PACKAGE
    package.get_config = lambda: None
    sys.modules[_PACKAGE] = package
_fused_moe = importlib.import_module(f"{_PACKAGE}.fused_moe")
fused_experts = _fused_moe.fused_experts
MoEActivation = importlib.import_module(f"{_PACKAGE}.activation").MoEActivation

_IMPL_ENV = "KIMI_MOE_PREFILL_IMPL"


def implementation() -> str:
    """Selected prefill expert implementation (vLLM by default)."""
    impl = os.environ.get(_IMPL_ENV, "vllm").lower()
    if impl not in {"vllm", "tilelang"}:
        raise ValueError(f"{_IMPL_ENV} must be vllm or tilelang, got {impl!r}")
    return impl


def prepare_fused_weights(twin, kinds) -> None:
    """Validate that every MoE already carries the unified packed ABI."""
    if implementation() != "vllm":
        return
    for index, (_mixer_kind, ffn_kind) in enumerate(kinds):
        if ffn_kind == "moe" and "w_gate_up" not in twin.modules[index].moe._bound:
            raise KeyError(f"layer {index} is missing packed w_gate_up")

def fused_routed(tokens, weights, indices, gate_up, down):
    """Run custom GPU routing through vLLM unquantized fused experts."""
    return fused_experts(
        tokens.contiguous(), gate_up, down, weights.contiguous(), indices.contiguous(),
        activation=MoEActivation.SILU, apply_router_weight_on_input=False,
    )


__all__ = [
    "fused_routed", "implementation", "prepare_fused_weights",
]
