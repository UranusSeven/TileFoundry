"""Kimi-Linear-48B-A3B-Instruct, authored outside the shipped tree.

`model.py` is the full decode shell in HIR: the vendored config class, the
three shipped mixers (KDA / MLA / MoE) plus the dense layer-0 MLP, the
27-layer stack with embed / final norm / lm_head, and `hf_alias` for the
published checkpoint. `kernels/` holds the tilelang implementation of the
decode-step mixers plus their plain-torch reference.
"""
