"""vLLM fused-MoE adapter for Kimi prefill."""
from __future__ import annotations

import importlib
import os
import sys
import types

import torch

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
    """Pack gate/up once after load, reusing the allocation through views."""
    if implementation() != "vllm":
        return
    for index, (_mixer_kind, ffn_kind) in enumerate(kinds):
        if ffn_kind != "moe":
            continue
        bound = twin.modules[index].moe._bound
        if "w_gate_up" in bound:
            continue
        gate, up = bound["w_gate"], bound["w_up"]
        gate_up = torch.cat((gate, up), dim=1).contiguous()
        intermediate = gate.shape[1]
        bound["w_gate_up"] = gate_up
        bound["w_gate"] = gate_up[:, :intermediate]
        bound["w_up"] = gate_up[:, intermediate:]
    # Concatenation temporarily holds old and packed expert tensors together.
    # Return the replaced allocations before vLLM requests its workspaces.
    torch.cuda.empty_cache()


def restore_decode_weights(bound) -> None:
    """Restore the decode kernels packed expert-major ABI after prefill."""
    gate_up = bound.pop("w_gate_up", None)
    if gate_up is None:
        return
    intermediate = gate_up.shape[1] // 2
    bound["w_gate"] = gate_up[:, :intermediate].contiguous()
    bound["w_up"] = gate_up[:, intermediate:].contiguous()


def fused_routed(tokens, weights, indices, gate_up, down):
    """Run custom GPU routing through vLLM unquantized fused experts."""
    return fused_experts(
        tokens.contiguous(), gate_up, down, weights.contiguous(), indices.contiguous(),
        activation=MoEActivation.SILU, apply_router_weight_on_input=False,
    )


__all__ = [
    "fused_routed", "implementation", "prepare_fused_weights",
    "restore_decode_weights",
]
