#!/usr/bin/env python3
"""Check TP2 CUDA-graph prefill against eager TP2 and HF logits."""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from runtime_tp2 import PrefillRunnerTP2  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="/root/develop/yingshan/prefill_phase2_hf_logits.pt")
    args = parser.parse_args(argv)
    oracle = torch.load(args.dump, weights_only=False)
    runner = PrefillRunnerTP2()
    ok = True
    for length in (8, 32, 100):
        expected = oracle[length]
        ids = expected["input_ids"]
        eager = runner(ids).clone()
        state = runner.capture(ids, warmup=2)
        runner.set_graph_input(ids)
        graph = runner.replay(length)
        torch.cuda.synchronize()
        gathered = [torch.empty_like(graph) for _ in range(2)]
        dist.all_gather(gathered, graph)
        same_ranks = torch.equal(gathered[0], gathered[1])
        same_eager = torch.equal(graph, eager)
        logits = graph[0].float().cpu()
        ref = expected["logits"].float()
        rel_l2 = ((logits - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        argmax = int(logits.argmax())
        passed = same_ranks and same_eager and argmax == expected["argmax"] and rel_l2 <= 6e-2
        ok &= passed
        if runner.rank == 0:
            print(
                f"S={length}: graph/eager_bit_exact={same_eager} "
                f"rank_bit_exact={same_ranks} argmax={argmax}/{expected['argmax']} "
                f"rel_l2={rel_l2:.6e} warmup_ms={state.warmup_seconds * 1e3:.3f} "
                f"capture_ms={state.capture_seconds * 1e3:.3f} "
                f"graph_memory_gib={state.memory_bytes / 2**30:.3f} "
                f"{'PASS' if passed else 'FAIL'}",
                flush=True,
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
