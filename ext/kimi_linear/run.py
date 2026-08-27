#!/usr/bin/env python
"""Decode real tokens from Kimi-Linear-48B-A3B-Instruct and report how fast.

    python run.py --prompt "The capital of France is" --max-new-tokens 32

Switches:

    --seed N     seeds the sampler (with --sample)
    --sample     sample (temperature/top-k/top-p) instead of argmax; the
                 published generation_config.json sets no sampling, so greedy
                 is the default and is what two runs get compared through
    --prefill X  'loop' (default): the decode step per prompt token;
                 'paged': the M3-B paged prefill (prefill.py), then decode
    --check [N]  before the run, replay N steps through the authored
                 evaluator over the same weights and diff the logits
    --tp 2       tensor-parallel over two GPUs; run under
                 `torchrun --nproc_per_node=2` (see runtime_tp2.py)
    --out PATH   write the produced ids as json (--compare reads these)

What the printed rate means
---------------------------
`tok/s` is **decode** throughput: new tokens divided by the time to produce
them, measured after the prompt is consumed. It excludes weight loading and
kernel compilation, which happen once, and it excludes prefill, which is
reported separately (prefill here is the decode step looped over the prompt
-- the kernels are S=1 decode-shaped; a real prefill path is a later
milestone). This is one token per step, batch one -- the regime `model.py`
declares -- so it is a latency number.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

EOS = {163586}  # generation_config.json's eos_token_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=None, help="seed the sampler")
    p.add_argument("--sample", action="store_true", help="sample instead of argmax")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--ckpt", default=None,
        help="the checkpoint directory (default: $KIMI_LINEAR_CKPT or the "
        "container's /models path)",
    )
    p.add_argument(
        "--prefill", choices=["loop", "paged"], default="loop",
        help="how the prompt is consumed: 'loop' runs the decode step per "
        "prompt token (the S=1 kernels); 'paged' runs the M3-B prefill path "
        "(paged FA3 MLA + fla chunk_kda, see prefill.py) and hands its state "
        "over to the same decode driver",
    )
    p.add_argument(
        "--check", type=int, nargs="?", const=8, default=None, metavar="N",
        help="first: replay min(N, steps) steps through the authored evaluator "
        "over the same weights and diff logits + greedy picks",
    )
    # Everything below is for working on this, not for using it.
    p.add_argument(
        "--layers", type=int, default=None,
        help="run only the first N layers (a fast loop, not the published model)",
    )
    p.add_argument(
        "--impl", default=None,
        help="TF_IMPL override, e.g. 'torch' or 'moe:torch,kda_attention:torch'",
    )
    p.add_argument(
        "--profile", action="store_true",
        help="after the run, torch-profiler a few decode steps",
    )
    p.add_argument(
        "--compare", metavar="PATH", default=None,
        help="diff the greedy token ids against an hf_greedy.py json",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--tp", type=int, default=1, choices=[1, 2],
        help="tensor parallelism over this many GPUs; 2 must run under "
        "torchrun --nproc_per_node=2 (see runtime_tp2.py's docstring)",
    )
    p.add_argument(
        "--out", metavar="PATH", default=None,
        help="write the produced token ids as json (hf_greedy.py's format, "
        "usable as another run's --compare)",
    )
    return p.parse_args(argv)


def truncated(n: int):
    """The published config cut to its first *n* layers (a debug loop)."""
    import model as sem  # noqa: PLC0415

    cfg = sem.published()
    la = dict(cfg.linear_attn_config)
    la["kda_layers"] = [i for i in la["kda_layers"] if i <= n]
    la["full_attn_layers"] = [i for i in la["full_attn_layers"] if i <= n]
    cfg.linear_attn_config = la
    cfg.num_hidden_layers = n
    return cfg


def sample(logits, *, greedy, temperature, top_k, top_p, generator):
    """One token from one row of logits (see the qwen3.5 example's run.py)."""
    row = logits.reshape(-1)
    if greedy:
        return int(torch.argmax(row).item())

    k = min(top_k or row.numel(), row.numel())
    values, ids = torch.topk(row.float(), k)
    if temperature != 1.0:
        values = values / max(temperature, 1e-6)
    probs = torch.softmax(values, dim=-1)
    if top_p and top_p < 1.0:
        keep = (torch.cumsum(probs, dim=-1) - probs) < top_p
        keep[0] = True
        probs = probs * keep
    probs = probs / probs.sum()
    return int(ids[torch.multinomial(probs, 1, generator=generator)].item())


def check_against_authored(session, prompt_ids, n: int) -> bool:
    """Replay *n* steps on both paths over the same weights and diff them.

    Twin (tilelang) against the authored evaluator: per-step logits max|d|
    and whether argmax agrees. The authored path is slow (an interpreter,
    exact-length caches), which is why this is a flag and not the run.
    """
    loaded = session.authored()
    ids = torch.tensor(prompt_ids, dtype=torch.int64)
    caches = session.twin.init_caches(device=session.device)
    session.reset()

    ok = True
    steps = min(n, len(prompt_ids))
    for step in range(steps):
        logits_t = session.step(prompt_ids[step])
        logits_a, caches = session.authored_step(loaded, ids, step, caches)
        a, b = logits_t.float(), logits_a.float()
        d = (a - b).abs().max().item()
        # Logits run to +/-30, where one bf16 ulp is 0.25 and the two paths
        # round 27 layers of fused-vs-interpreted arithmetic differently:
        # max|d| and per-element allclose are both meaningless at this
        # amplitude. The load-bearing comparisons are argmax (what a greedy
        # run emits) and the relative L2 of the whole row: measured noise is
        # 1-2e-2 here (the per-mixer checks' evaluator-vs-oracle floor is
        # itself 3e-2); a wiring error sits orders higher.
        rel_l2 = ((a - b).norm() / b.norm().clamp_min(1e-9)).item()
        same = int(logits_t.argmax()) == int(logits_a.argmax())
        ok &= same and rel_l2 < 5e-2
        print(
            f"  step {step:3d}  max|d|={d:.3e}  rel_l2={rel_l2:.3e}  "
            f"argmax {'agree' if same else 'DIFFER'}"
            f"  ({'PASS' if same and rel_l2 < 5e-2 else 'FAIL'})",
            flush=True,
        )
    session.reset()
    print(f"check: {'PASS' if ok else 'FAIL'} ({steps} steps vs authored evaluator)")
    return ok


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.impl is not None:
        os.environ["TF_IMPL"] = args.impl

    rank, device = 0, "cuda"
    dist = None
    if args.tp == 2:
        import torch.distributed as dist  # noqa: PLC0415

        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
        if rank != 0:
            # Replicated ranks compute but do not speak.
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115

    # Imported after TF_IMPL is set: the twins bind their implementations at
    # import time so a per-call dispatch never appears in the step.
    import runtime_model as rt  # noqa: PLC0415
    import weights as wt  # noqa: PLC0415

    if args.tp == 2:
        import runtime_tp2 as rt2  # noqa: PLC0415

        if args.check is not None:
            print("--check replays the authored evaluator, which is the "
                  "full-width computation; under --tp 2 the comparison is "
                  "TP2 tokens vs a TP1/HF json (--compare).", file=sys.stderr)
            return 2

    ckpt = args.ckpt or wt.CKPT
    cfg = None
    if args.layers is not None:
        cfg = truncated(args.layers)
        print(
            f"note: --layers {args.layers} runs a {args.layers}-layer prefix of "
            f"the published 27-layer stack. The tokens are not the model's."
        )

    tok = wt.tokenizer(ckpt)
    prompt_ids = list(tok.encode(args.prompt))
    if not prompt_ids:
        print("empty prompt after tokenisation", file=sys.stderr)
        return 2

    need = len(prompt_ids) + args.max_new_tokens + 1

    print(f"loading {ckpt} ...", flush=True)
    if args.tp == 2:
        session = rt2.SessionTP2(
            cfg, ckpt=ckpt, capacity=need, device=device, verbose=args.verbose
        )
    else:
        session = rt.Session(cfg, ckpt=ckpt, capacity=need, verbose=args.verbose)
    print(
        f"loaded {session.loaded_bytes / 1e9:.1f} GB in {session.load_seconds:.1f}s"
        f"  ({session.loaded_bytes / 1e9 / session.load_seconds:.2f} GB/s)",
        flush=True,
    )

    t0 = time.perf_counter()
    n_buckets = rt.precompile(
        session.capacity, nh=rt._NH // args.tp if hasattr(rt, "_NH") else 32
    )
    print(
        f"attention buckets ready ({n_buckets}) in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    if args.check is not None and not check_against_authored(
        session, prompt_ids, args.check
    ):
        return 1

    generator = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    picks = dict(
        greedy=not args.sample, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, generator=generator,
    )

    # -- prefill -------------------------------------------------------------
    if args.prefill == "paged":
        import prefill as pf  # noqa: PLC0415

        # Warm-up: the prefill kernels (FA3, fla chunk_kda) and the decode
        # kernels alike pay a one-time compilation/module load on first use.
        # Loop prefill absorbs the decode half inside its own timed loop; a
        # server pays all of it once at startup, so the timed sections below
        # run warm.
        warm_logits = pf.prefill(session, prompt_ids)
        session.step(int(warm_logits.argmax()))
        session.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.prefill == "paged":
        logits = pf.prefill(session, prompt_ids)
    else:
        # one step per prompt token, because the decode step is one token
        logits = None
        for token in prompt_ids:
            logits = session.step(token)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # One untimed draw: the first topk/multinomial of a process loads its CUDA
    # kernels lazily, which would otherwise land on the reported rate.
    sample(logits, **picks)
    torch.cuda.synchronize()

    # -- decode --------------------------------------------------------------
    produced: list[int] = []
    if dist is not None:
        dist.barrier()
        rt2.comm_reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        nxt = sample(logits, **picks)
        produced.append(nxt)
        if nxt in EOS:
            break
        logits = session.step(nxt)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    text = tok.decode(produced)
    print()
    print("=" * 72)
    print(args.prompt + text)
    print("=" * 72)
    print()
    print(f"prompt        {len(prompt_ids)} tokens   {prefill_s * 1e3:.1f} ms"
          f"   ({len(prompt_ids) / prefill_s:.1f} tok/s)")
    print(f"generated     {len(produced)} tokens   {decode_s * 1e3:.1f} ms")
    print(f"              {decode_s / len(produced) * 1e3:.2f} ms/token")
    print()
    print(f"  {len(produced) / decode_s:.1f} tok/s")
    print()

    # The floor: per token the run reads every non-expert weight, top_k of 256
    # experts per MoE layer, and the growing MLA caches. ~6.9 GB at bf16 for
    # the published stack.
    expert_gb = 26 * 8 * 3 * 1024 * 2304 * 2 / 1e9
    dense_gb = (49.1e9 - 26 * 256 * 3 * 1024 * 2304) * 2 / 1e9
    active_gb = expert_gb + dense_gb
    ms = decode_s / len(produced) * 1e3
    print(f"note: ~{active_gb:.2f} GB of weights are read per token, so 4.8 TB/s"
          f" of HBM is a {active_gb / 4.8:.2f} ms/token floor"
          f"  ({4.8 / active_gb * 1e3:.0f} tok/s).")
    print(f"      this run moves them at {active_gb / (ms / 1e3) / 1e3:.2f} TB/s"
          f" effective ({active_gb / (ms / 1e3) / 1e3 / 4.8 * 100:.0f}% of peak).")

    if args.out and rank == 0:
        import json  # noqa: PLC0415

        with open(args.out, "w") as fh:
            json.dump({"prompt": args.prompt, "greedy_ids": produced}, fh)
        print(f"wrote {len(produced)} ids to {args.out}")

    if dist is not None:
        torch.cuda.synchronize()
        if rank == 0:
            print(rt2.comm_report(len(produced)))
        dist.destroy_process_group()

    if args.profile:
        profile_decode(session, produced[-1])

    if args.compare:
        import json  # noqa: PLC0415

        with open(args.compare) as fh:
            oracle = json.load(fh)
        theirs = oracle["greedy_ids"][: len(produced)]
        same = sum(1 for a, b in zip(produced, theirs) if a == b)
        first = next(
            (i for i, (a, b) in enumerate(zip(produced, theirs)) if a != b), None
        )
        print()
        print(f"vs {args.compare}: {same}/{len(theirs)} ids equal"
              + ("" if first is None else f", first difference at index {first}"))
        print(f"  mine   {produced[:16]}")
        print(f"  theirs {theirs[:16]}")
        return 0 if first is None else 1
    return 0


def profile_decode(session, token_id: int, steps: int = 8):
    """A few decode steps under the torch profiler, printed by CUDA time."""
    for _ in range(3):
        session.step(token_id)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(steps):
            session.step(token_id)
        torch.cuda.synchronize()
    print()
    print(f"== profile: {steps} decode steps ==")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    raise SystemExit(main())
