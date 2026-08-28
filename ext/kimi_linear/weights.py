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
from safetensors import safe_open  # noqa: E402

from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402
from tilefoundry.runtime.resource import _reject_group, _resolve_key  # noqa: E402

#: The checkpoint this runs against, unless the caller says otherwise.
CKPT = os.environ.get("KIMI_LINEAR_CKPT", "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct")

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

    def __init__(self, node, raw, *, dtype=DTYPE, report=None, tp_rank=None, _converters=None):
        self._node = node
        self._raw = raw
        self._dtype = dtype
        self._report = report
        self._tp_rank = tp_rank
        self._converters = _converter_map(node) if _converters is None else _converters

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
            parts = self._load_group(name)
            if parts is not None:
                if name == "w_gate_up":
                    if len(parts) % 2:
                        raise ValueError("w_gate_up alias must contain gate/up pairs")
                    value = torch.stack(
                        [
                            torch.cat((parts[i], parts[i + 1]), dim=0).to(self._dtype)
                            for i in range(0, len(parts), 2)
                        ]
                    )
                else:
                    value = torch.stack([part.to(self._dtype) for part in parts])
            else:
                value = self._load_raw(name).to(self._dtype)
        else:
            # The converter's parameters are raw-checkpoint names; the alias
            # table maps them. `evaluate` materialises the converter's
            # declared return dtype, which here is bf16 already.
            raws = [self._load_converter_raw(name, p.name).to(self._dtype) for p in conv.params]
            if self._tp_rank is not None and self._converter_is_sharded(name):
                value = self._convert_shard(name, raws[0]).to(self._dtype)
            else:
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
                    tp_rank=self._tp_rank,
                )
        raise KeyError(f"{self._node.name!r} has no child module {seg!r}")

    def _span(self, width):
        half = width // 2
        return slice(self._tp_rank * half, (self._tp_rank + 1) * half)

    def _load_group(self, name):
        if self._tp_rank is None:
            return self._raw.load_group(name)
        if name == "w_gate_up":
            slices = [(self._span(1024), slice(None))] * (256 * 2)
        elif name == "w_down":
            slices = [(slice(None), self._span(1024))] * 256
        else:
            return self._raw.load_group(name)
        return self._load_group_slices(name, slices)

    def _load_raw(self, name):
        return self._raw.load(name)

    def _slice_accessor(self, name):
        resolved, read = _resolve_key(self._raw._alias, self._raw._prefix, name)
        raw_key = _reject_group("PrefillWeightResource._load_slice", name, resolved)
        if read is not None:
            raise TypeError("sliced Preprocessed aliases are unsupported")
        shard = self._raw._index()[raw_key]
        handle = self._raw._handles.get(shard)
        if handle is None:
            path = os.path.join(self._raw._ckpt_dir, shard)
            device = self._raw._device
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"
            handle = safe_open(path, framework="pt", device=device)
            self._raw._handles[shard] = handle
        return handle.get_slice(raw_key)

    def _load_slice(self, name, slices):
        return self._slice_accessor(name)[slices]

    def _load_group_slices(self, name, slices):
        resolved, read = _resolve_key(self._raw._alias, self._raw._prefix, name)
        if read is not None or not isinstance(resolved, tuple):
            raise TypeError(f"{name!r} is not an unprocessed alias group")
        return tuple(
            self._load_slice(raw[len(self._raw._prefix) :], sl)
            for raw, sl in zip(resolved, slices, strict=True)
        )

    def _converter_is_sharded(self, name):
        node = self._node.name
        return (
            node == "mixer"
            and name
            in {
                "w_q",
                "w_k",
                "w_v",
                "w_f_b",
                "w_g_b",
                "w_kv_b",
                "conv_w_q",
                "conv_w_k",
                "conv_w_v",
                "dt_bias",
                "a_log",
                "w_b",
                "w_o",
            }
        ) or (
            node in {"moe", "mlp"}
            and name
            in {
                "w_gate",
                "w_up",
                "w_down",
                "sh_gate",
                "sh_up",
                "sh_down",
            }
        )

    def _convert_shard(self, name, raw):
        if name.startswith("conv_w_"):
            return raw.reshape(raw.shape[0], raw.shape[-1]).transpose(0, 1)
        if name == "a_log":
            return raw.reshape(-1)
        if raw.ndim == 1:
            return raw
        return raw.transpose(0, 1).unsqueeze(0)

    def _load_converter_raw(self, name, raw_name):
        if self._tp_rank is None:
            return self._fetch(raw_name)
        node = self._node.name
        axis = None
        if node == "mixer":
            if name in {"w_q", "w_k", "w_v", "w_f_b", "w_g_b", "w_kv_b"}:
                axis = 0
            elif name in {"conv_w_q", "conv_w_k", "conv_w_v", "dt_bias", "w_b"}:
                axis = 0
            elif name == "a_log":
                axis = 2
            elif name == "w_o":
                axis = 1
        elif node in {"moe", "mlp"}:
            if name in {"w_gate", "w_up", "sh_gate", "sh_up"}:
                axis = 0
            elif name in {"w_down", "sh_down"}:
                axis = 1
        if axis is None:
            return self._fetch(raw_name)
        shape = self._slice_accessor(raw_name).get_shape()
        slices = [slice(None)] * len(shape)
        slices[axis] = self._span(shape[axis])
        return self._load_slice(raw_name, tuple(slices))


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


def prefill_resource(
    node=None, ckpt=CKPT, cfg=None, *, device="cuda", dtype=DTYPE, verbose=False, tp_rank=None
):
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
        tp_rank=tp_rank,
    ), total


__all__ = [
    "CKPT",
    "DTYPE",
    "PrefillWeightResource",
    "prefill_resource",
    "raw_resource",
]
