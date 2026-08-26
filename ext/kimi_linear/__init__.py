
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

Milestone 2 (TP2): `ops.py` registers the external HIR op `tf.all_reduce`
(the Partial -> converged boundary of docs/spec/shard.md);
`runtime_tp2.py` is the two-GPU twins -- weights sliced per rank, one
NCCL all-reduce per mixer/FFN output -- and its `SessionTP2` driver;
`run.py --tp 2` runs them under torchrun. `check_allreduce_op.py` and
`check_tp2_shards.py` are their validations.
"""
