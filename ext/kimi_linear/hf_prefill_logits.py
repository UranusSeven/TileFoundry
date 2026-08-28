#!/usr/bin/env python3
"""Dump fixed-token HF last-position prefill logits for correctness checks."""
from __future__ import annotations

import argparse
import os
import sys
import types

import torch
import transformers.utils as transformers_utils
import transformers.utils.auto_docstring as auto_docstring_module

CKPT = os.environ.get(
    "KIMI_LINEAR_CKPT", "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct"
)
ORACLE_PARENT = "/root/develop/yingshan/oracle"
FIXED_IDS = [1, 15043, 299, 17722, 295, 1296, 374, 264]


def fixed_ids(length: int) -> list[int]:
    return (FIXED_IDS * ((length + len(FIXED_IDS) - 1) // len(FIXED_IDS)))[:length]


def _install_shims() -> None:
    def noop_docstring(obj=None, **_kwargs):
        return (lambda value: value) if obj is None else obj

    transformers_utils.auto_docstring = noop_docstring
    auto_docstring_module.auto_docstring = noop_docstring

    if "transformers.utils.output_capturing" not in sys.modules:
        output_capturing = types.ModuleType("transformers.utils.output_capturing")

        class OutputRecorder:
            def __init__(self, module_class, index=0):
                self.module_class = module_class
                self.index = index

        output_capturing.OutputRecorder = OutputRecorder
        sys.modules[output_capturing.__name__] = output_capturing

    import fla.ops.kda.gate as fla_gate  # noqa: PLC0415
    from fla.ops.kda.gate import naive_kda_gate  # noqa: PLC0415

    def gate_shim(g, a_log, head_dim, g_bias=None):
        *leading, hidden = g.shape
        heads = hidden // head_dim
        return naive_kda_gate(
            g.view(*leading, heads, head_dim), a_log.reshape(-1), g_bias
        )

    fla_gate.fused_kda_gate = gate_shim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="8,32,100")
    args = parser.parse_args(argv)
    lengths = [int(value) for value in args.lengths.split(",")]

    _install_shims()
    if ORACLE_PARENT not in sys.path:
        sys.path.insert(0, ORACLE_PARENT)
    from kimi_ref.configuration_kimi import KimiLinearConfig  # noqa: PLC0415
    from kimi_ref.modeling_kimi import KimiLinearForCausalLM  # noqa: PLC0415

    config = KimiLinearConfig.from_pretrained(CKPT)
    config._attn_implementation = "eager"
    model = KimiLinearForCausalLM.from_pretrained(
        CKPT, config=config, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model.config._attn_implementation = "eager"

    dump = {}
    with torch.no_grad():
        for length in lengths:
            ids = fixed_ids(length)
            tensor = torch.tensor([ids], dtype=torch.long, device="cuda")
            logits = model(tensor, use_cache=False).logits[0, -1].float().cpu()
            dump[length] = {
                "input_ids": ids,
                "logits": logits,
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "argmax": int(logits.argmax().item()),
                "checksum": float(logits.sum().item()),
            }
            print(
                f"HF S={length}: shape={tuple(logits.shape)} "
                f"argmax={dump[length]['argmax']} checksum={dump[length]['checksum']:.6e}",
                flush=True,
            )
    torch.save(dump, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
