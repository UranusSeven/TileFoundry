"""Greedy token ids from the official checkpoint implementation.

For `run.py --compare`. Loads `KimiLinearForCausalLM` from the checkpoint's own modeling code
(`auto_map` + trust_remote_code) and greedily decodes a handful of tokens.
Writes {"prompt", "greedy_ids"} as json; `run.py --compare FILE` diffs its
own greedy stream against it. This is the end-to-end leg of the M1 evidence:
the per-mixer checks established authored == HF at one layer; this
establishes twin == HF at 27 layers of accumulated state.

    CUDA_VISIBLE_DEVICES=5 /root/develop/yingshan/venv_hf/bin/python \
        ext/kimi_linear/hf_greedy.py \
        --prompt "The capital of France is" --max-new-tokens 16 \
        --out /root/develop/yingshan/kimi_hf_greedy.json

Runs under the `venv_hf` interpreter (transformers 4.57.1, the version the
checkpoint's vendored code was written against; the container's 5.15 cannot
import it). Three shims, all at import boundaries, none touching math:

* `auto_docstring` no-op'd: 4.57.1's decorator trips on a PEP-604
  annotation in the vendored `KimiLinearModel.forward`.
* `fla.ops.kda.gate.fused_kda_gate` shimmed to `naive_kda_gate`: installed
  fla is 0.5.2, whose signature lacks `g_bias` (same shim as
  `check_kda_kernel.py` / the KDA oracle harness).
* `_attn_implementation` forced back to "eager" after loading: the vendored
  model hard-codes `flash_attention_2` and flash-attn is not installed. The
  config object is shared by every layer and read at forward time.
"""
import argparse
import json
import os

import torch

CKPT = os.environ.get(
    "KIMI_LINEAR_CKPT", "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # -- shim 1: the docstring decorator (see module docstring) -------------
    import transformers.utils as _tu  # noqa: PLC0415
    import transformers.utils.auto_docstring as _ad  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    def _noop_docstring(obj=None, **kwargs):
        return (lambda o: o) if obj is None else obj

    _tu.auto_docstring = _noop_docstring
    _ad.auto_docstring = _noop_docstring

    # -- shim 2: the fla 0.5.2 gate signature -------------------------------
    import fla.ops.kda.gate as _fla_gate  # noqa: PLC0415
    from fla.ops.kda.gate import naive_kda_gate  # noqa: PLC0415

    def _gate_shim(g, A_log, head_dim, g_bias=None):
        *lead, hk = g.shape
        h = hk // head_dim
        return naive_kda_gate(
            g.view(*lead, h, head_dim), A_log.reshape(-1), g_bias
        )

    _fla_gate.fused_kda_gate = _gate_shim

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    # -- shim 3: eager attention instead of the hard-coded flash ------------
    model.config._attn_implementation = "eager"
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    produced = out[0, ids.shape[1]:].tolist()
    with open(args.out, "w") as fh:
        json.dump({"prompt": args.prompt, "greedy_ids": produced}, fh)
    print(f"prompt ids : {ids[0].tolist()}")
    print(f"greedy ids : {produced}")
    print(f"text       : {tok.decode(produced)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
