"""Binding the published Kimi-Linear checkpoint to what `model.py` declares.

Same pattern as the qwen3.5 example's `weights.py`: `HFResource` is
`Module.prepare` without the prepare directory -- the same walk, the same
per-weight converters from `model.py` (the only place the raw->declared
transform is written), run on demand, on the GPU, at load time.

Two differences from the example, both from this checkpoint:

* **Everything is bf16.** The shell declares `_DT = bf16` for every weight
  (the checkpoint stores bf16; `A_log` / `dt_bias` are stored f32 but their
  converters cast down). There is no f32 exception set.
* **The expert stacks are one-to-many aliases.** `w_gate_up` names interleaved
  `experts.{i}.w1/w3.weight` tensors and is packed to `[E, 2I, H]` while
  `w_down` stacks `experts.{i}.w2.weight` to `[E, H, I]`. These are the sole
  declared expert ABI tensors.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as sem  # noqa: E402
import torch  # noqa: E402

from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402

#: The checkpoint this runs against, unless the caller says otherwise.
CKPT = os.environ.get(
    "KIMI_LINEAR_CKPT", "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct"
)

#: What every weight is declared (and stored) at.
DTYPE = torch.bfloat16


class PrefillWeightResource:
    """A `RuntimeResource` over the raw checkpoint that answers canonical names.

    Implements the protocol of runtime §1.5 -- `load` / `load_group` /
    `subtree` -- so the authored prefill tree can be loaded directly from it.
    """

    #: (raw resource, name, tensor) of the last raw read, shared across
    #: subtree views because a view is a new object per child.
    _last = None

    def __init__(self, node, raw, *, dtype=DTYPE, report=None, _converters=None):
        self._node = node
        self._raw = raw
        self._dtype = dtype
        self._report = report
        self._converters = (
            _converter_map(node) if _converters is None else _converters
        )

    def _fetch(self, name: str) -> torch.Tensor:
        """One raw tensor, with a one-entry cache.

        Converter parameters that share a raw tensor are asked for back to
        back; one entry is enough because the reuse is always immediate.
        """
        cached = PrefillWeightResource._last
        if cached is not None and cached[0] is self._raw and cached[1] == name:
            return cached[2]
        value = self._raw.load(name)
        PrefillWeightResource._last = (self._raw, name, value)
        return value

    def load(self, name: str) -> torch.Tensor:
        conv = self._converters.get(name)
        if conv is None:
            parts = self._raw.load_group(name)
            if parts is not None:
                if name == "w_gate_up":
                    if len(parts) % 2:
                        raise ValueError("w_gate_up alias must contain gate/up pairs")
                    value = torch.stack([
                        torch.cat((parts[i], parts[i + 1]), dim=0).to(self._dtype)
                        for i in range(0, len(parts), 2)
                    ])
                else:
                    value = torch.stack([part.to(self._dtype) for part in parts])
            else:
                value = self._fetch(name).to(self._dtype)
        else:
            # The converter's parameters are raw-checkpoint names; the alias
            # table maps them. `evaluate` materialises the converter's
            # declared return dtype, which here is bf16 already.
            raws = [self._fetch(p.name).to(self._dtype) for p in conv.params]
            value = evaluate(conv, *raws).to(self._dtype)
        value = value.contiguous()
        if self._report is not None:
            self._report(name, value)
        return value

    def load_group(self, name: str):
        # Every group weight of this model is stacked by `load` above.
        return None

    def subtree(self, seg: str) -> "PrefillWeightResource":
        for child in self._node.modules:
            if child.name == seg:
                return PrefillWeightResource(
                    child,
                    self._raw.subtree(seg),
                    dtype=self._dtype,
                    report=self._report,
                )
        raise KeyError(f"{self._node.name!r} has no child module {seg!r}")


def _converter_map(node) -> dict:
    """This node's per-weight converters, unioned over its functions.

    Same rule as `Module._prepare_into`: a converter is registered on the
    function that declares the weight, and one weight may be declared by
    several functions of one Module, so the map is per Module.
    """
    found: dict[str, object] = {}
    for fn in node.functions:
        for weight_name, conv in getattr(fn, "converters", ()):
            found[weight_name] = conv
    return found


def raw_resource(ckpt=CKPT, cfg=None, *, device="cuda"):
    """The published checkpoint, alias table attached, nothing converted yet."""
    return SafetensorsResource(str(ckpt), device=device, alias=sem.hf_alias(cfg))


def prefill_resource(node=None, ckpt=CKPT, cfg=None, *, device="cuda",
                     dtype=DTYPE, verbose=False):
    """What `KimiLinear48BA3BPrefill.load(...)` reads.

    *node* is the authored root the resource walks -- the published one by
    default, or another prefill-only tree returned by `model.build(cfg)`. It has to
    match *cfg*, since the alias table is generated per layer index.
    """
    node = sem.KimiLinear48BA3BPrefill if node is None else node
    total = {"bytes": 0, "n": 0, "t0": time.perf_counter()}

    def report(name, value):
        total["bytes"] += value.numel() * value.element_size()
        total["n"] += 1
        if verbose and total["n"] % 200 == 0:
            gb = total["bytes"] / 1e9
            dt = time.perf_counter() - total["t0"]
            print(
                f"  loaded {total['n']:5d} tensors  {gb:6.2f} GB  "
                f"{dt:5.1f}s  ({gb / max(dt, 1e-9):.2f} GB/s)",
                flush=True,
            )

    return PrefillWeightResource(
        node,
        raw_resource(ckpt, cfg, device=device),
        dtype=dtype,
        report=report,
    ), total



__all__ = [
    "CKPT",
    "DTYPE",
    "PrefillWeightResource",
    "prefill_resource",
    "raw_resource",
]
