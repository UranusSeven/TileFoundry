#!/usr/bin/env python3
"""Benchmark one Kimi prefill and report last-position logits and TTFT."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

_DEFAULT_IDS = [1, 15043, 299, 17722, 295, 1296, 374, 264]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ids", help="comma-separated token ids")
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1, choices=[1])
    parser.add_argument("--out", help="write the report as JSON")
    return parser.parse_args(argv)


def input_ids(args):
    if args.input_ids:
        ids = [int(value) for value in args.input_ids.split(",")]
        if args.length != 8 and args.length != len(ids):
            raise ValueError("--length must match --input-ids when both are supplied")
        return ids
    if args.length < 1:
        raise ValueError("--length must be positive")
    return (_DEFAULT_IDS * ((args.length + len(_DEFAULT_IDS) - 1) // len(_DEFAULT_IDS)))[: args.length]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("--repeat must be positive and --warmup non-negative")
    ids = input_ids(args)

    from prefill import PrefillRunner  # noqa: PLC0415

    runner = PrefillRunner()
    for _ in range(args.warmup):
        runner(ids)
    torch.cuda.synchronize()

    elapsed = []
    logits = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        logits = runner(ids)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)

    row = logits[0].float()
    median = statistics.median(elapsed)
    report = {
        "input_tokens": len(ids),
        "max_tokens": args.max_tokens,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_device": str(logits.device),
        "logits_finite": bool(torch.isfinite(row).all().item()),
        "argmax": int(row.argmax().item()),
        "checksum": float(row.sum().item()),
        "median_ttft_ms": median * 1e3,
        "input_tok_s": len(ids) / median,
        "repeat": args.repeat,
        "warmup": args.warmup,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
