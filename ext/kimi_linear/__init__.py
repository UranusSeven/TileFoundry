
"""Kimi-Linear-48B-A3B-Instruct, authored outside the shipped tree.

`model.py` is the full decode shell in HIR: the vendored config class, the
three shipped mixers (KDA / MLA / MoE) plus the dense layer-0 MLP, the
27-layer stack with embed / final norm / lm_head, and `hf_alias` for the
published checkpoint. `kernels/` holds the tilelang implementation of the
decode-step mixers plus their plain-torch reference.

The Milestone-1 runtimes: `weights.py` binds the published checkpoint to the
shell's declared weights (converters on demand, expert stacks grouped);
`runtime_model.py` is the `@runtime_module` twins plus the `Session` decode
driver; `run.py` decodes real tokens and reports decode tok/s;
`check_twin.py` validates twins against the authored evaluator on real
weights. `check_{kda,mla,moe}_kernel.py` are the per-kernel validations.
"""
