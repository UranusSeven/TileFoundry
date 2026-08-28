"""Prefill-only Kimi-Linear-48B-A3B extension.

`model.py` owns the independent S-symbolic 27-layer authored HIR. `ops.py`
registers HIR-only KDA recurrence and paged MLA prefill operations. `weights.py`
binds the checkpoint using the packed expert ABI `[E, 2I, H]` / `[E, H, I]`.
The retained runtime prefill code is transitional and is not yet a complete
end-to-end prefill pipeline.
"""
