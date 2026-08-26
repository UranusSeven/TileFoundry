"""TP2: the M1 runtime sharded over two GPUs with torch.distributed + NCCL.

Sharding (rank r of 2), per the milestone plan:

* **KDA and MLA mixers: heads 32 -> 16 per rank.** The q/k/v (and MLA
  kv_b) projections are column-sliced by head, o_proj is row-sliced, and
  the per-head state follows the heads: each rank's KDA recurrent state is
  [1, 16, 128, 128], its conv windows [1, 3, 2048], its MLA KV cache holds
  only its own 16 heads. MLA's w_kv_a and latent norm stay replicated --
  the 512-wide latent couples every head (each rank recomputes it: 2.7 MB
  of weights against a second norm launch's worth of traffic).
* **MoE: expert-internal TP.** Every rank keeps all 256 experts; gate/up
  lose half their intermediate columns (1024 -> 512), down loses half its
  input rows; the shared expert likewise. The router is replicated, so both
  ranks select identical experts.
* **Layer 0's dense MLP:** the same halving at 9216 -> 4608.
* **embed / lm_head / every norm gamma:** replicated.

Each sharded block is a column-parallel projection followed by a
row-parallel one, so its output is a `Partial("sum")` over the 2-GPU mesh
(docs/spec/shard.md). The twins below converge it with
`dist.all_reduce` on the *un-rounded f32* partial (the kernels' `keep_f32`
entry points), landing bf16 once, on the sum -- one collective per mixer
and per FFN output, 54 per decode step, 4.6-9 KB each. The IR-level
statement of the same boundary is `ops.py`'s `tf.all_reduce` op (checked
by `check_allreduce_op.py`); the slicing math itself is proven on one GPU
by `check_tp2_shards.py`. This file is the production half -- the place
NCCL is reachable.

Launch:

    torchrun --nproc_per_node=2 ext/kimi_linear/run.py --tp 2 ...
"""
from __future__ import annotations

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as sem  # noqa: E402
import runtime_model as rt1  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from kernels import attn as _attn  # noqa: E402
from kernels import kda as _kda  # noqa: E402
from kernels import moe as _moe  # noqa: E402
from kernels import torch_ref as _tr  # noqa: E402

from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

_BF16 = torch.bfloat16

# ---------------------------------------------------------------------------
# The collective, with accounting.
# ---------------------------------------------------------------------------

_COMM = {"calls": 0, "events": []}


def _converge(out: torch.Tensor) -> torch.Tensor:
    """Partial -> replicated: all-reduce, landing bf16 once on the sum.

    The fast paths hand over their f32 accumulator (`keep_f32`); the torch
    fallbacks (bisection only) return bf16 already. One 4.6-9 KB
    latency-bound call either way.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    dist.all_reduce(out)
    end.record()
    _COMM["calls"] += 1
    _COMM["events"].append((start, end))
    return out if out.dtype is _BF16 else out.to(_BF16)


def comm_report(steps: int) -> str:
    """The all-reduce accounting of the last run. Call after a synchronize."""
    ms = sum(s.elapsed_time(e) for s, e in _COMM["events"])
    calls = _COMM["calls"]
    per_step = calls / max(steps, 1)
    return (
        f"all-reduce: {calls} calls ({per_step:.0f}/step), {ms:.2f} ms GPU total, "
        f"{ms / max(steps, 1):.3f} ms/step, {ms / max(calls, 1) * 1e3:.1f} us/call"
    )


def comm_reset() -> None:
    _COMM["calls"] = 0
    _COMM["events"] = []


# ---------------------------------------------------------------------------
# The TP twins: the M1 twins with one output boundary converged.
# ---------------------------------------------------------------------------


def _tp_kda_attention(self, *args):
    out, *rest = rt1._impl(
        "kda_attention", functools.partial(_kda.kda_step, keep_f32=True), _tr.kda_step
    )(*args)
    return _converge(out), *rest


def _tp_mla_attention(self, *args):
    out, k_new, v_new = rt1._impl(
        "mla_attention",
        functools.partial(_attn.mla_attention_cap, keep_f32=True),
        rt1._t_mla_attention,
    )(*args)
    return _converge(out), k_new, v_new


def _tp_moe(self, *args):
    out = rt1._impl(
        "moe", functools.partial(_moe.moe_block, keep_f32=True), rt1._t_moe
    )(*args)
    return _converge(out)


def _tp_mlp(self, *args):
    out = rt1._impl(
        "mlp", functools.partial(_moe.dense_mlp, keep_f32=True), rt1._t_mlp
    )(*args)
    return _converge(out)


def _tp_twin(sem_module, base_cls, **overrides):
    """*base_cls* (an M1 twin) again, with *overrides* substituted.

    The runtime marks survive the copy (`runtime_func` returns the function
    itself), so re-decorating the rebuilt namespace gives a twin of the same
    authored Module.
    """
    ns = {
        name: value
        for name, value in vars(base_cls).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
    ns.update(overrides)
    return runtime_module(sem_module)(type(f"{base_cls.__name__}TP", (), ns))


KimiKdaTP = _tp_twin(sem.KimiKda, rt1.KimiKda,
                     kda_attention=runtime_func(_tp_kda_attention))
KimiMlaTP = _tp_twin(sem.KimiMla, rt1.KimiMla,
                     mla_attention=runtime_func(_tp_mla_attention))
KimiMoeTP = _tp_twin(sem.KimiMoe, rt1.KimiMoe, moe=runtime_func(_tp_moe))
KimiDenseMlpTP = _tp_twin(sem.KimiDenseMlp, rt1.KimiDenseMlp,
                          mlp=runtime_func(_tp_mlp))

KimiKdaDenseLayerTP = rt1._layer_twin(
    sem.KimiKdaDenseLayer, KimiKdaTP, "mlp", KimiDenseMlpTP
)
KimiKdaMoeLayerTP = rt1._layer_twin(sem.KimiKdaMoeLayer, KimiKdaTP, "moe", KimiMoeTP)
KimiMlaMoeLayerTP = rt1._layer_twin(sem.KimiMlaMoeLayer, KimiMlaTP, "moe", KimiMoeTP)

_LAYER_TWIN_TP = {
    ("kda", "dense"): KimiKdaDenseLayerTP,
    ("kda", "moe"): KimiKdaMoeLayerTP,
    ("mla", "moe"): KimiMlaMoeLayerTP,
}


def _root_twin_tp(sem_root):
    """`runtime_model._root_twin` with the TP layer twins substituted."""
    namespace = {
        child.name: _LAYER_TWIN_TP[rt1._kind_of(child)] for child in sem_root.modules
    }
    namespace.update(
        embed=runtime_func(rt1._embed),
        final_rms_norm=runtime_func(rt1._final_rms_norm),
        lm_head=runtime_func(rt1._lm_head),
    )
    return runtime_module(sem_root)(type("KimiLinear48BA3BTP", (), namespace))


KimiLinear48BA3BTP = _root_twin_tp(sem.KimiLinear48BA3B)


# ---------------------------------------------------------------------------
# Weight slicing. None = replicated on every rank.
# ---------------------------------------------------------------------------

#: Per leaf kind, the dim each weight halves along. Every bound weight must
#: be listed -- an unmapped name is a bug, not a replication.
_TABLES = {
    "kda": {
        "gamma_in": None, "gamma_o": None, "w_f_a": None, "w_g_a": None,
        "w_q": -1, "w_k": -1, "w_v": -1,
        "conv_w_q": -1, "conv_w_k": -1, "conv_w_v": -1,
        "w_f_b": -1, "w_g_b": -1, "dt_bias": -1, "a_log": 0, "w_b": -1,
        "w_o": -2,
    },
    "mla": {
        "gamma_in": None, "w_kv_a": None, "gamma_kv_a": None,
        "w_q": -1, "w_kv_b": -1, "w_o": -2,
    },
    "moe": {
        "gamma_post": None, "w_router": None, "bias": None,
        "w_gate": 1, "w_up": 1, "w_down": 2,
        "sh_gate": -1, "sh_up": -1, "sh_down": -2,
    },
    "dense": {
        "gamma_post": None,
        "w_gate": -1, "w_up": -1, "w_down": -2,
    },
    "root": {"table": None, "gamma_final": None, "w_head": None},
}


def _kind_of_bound(names) -> str:
    """Which slicing table a module's bound weights answer to."""
    s = set(names)
    for key, probe in (
        ("moe", "w_router"), ("mla", "w_kv_a"), ("kda", "w_f_a"),
        ("dense", "w_gate"), ("root", "w_head"),
    ):
        if probe in s:
            return key
    raise KeyError(f"no TP2 slicing table for weights {sorted(s)}")


def shard_weights(twin_root, rank: int, world_size: int) -> tuple[int, int]:
    """Halve every sharded weight in place; return (bytes before, after)."""

    def visit(node):
        before = after = 0
        if node._bound:
            table = _TABLES[_kind_of_bound(node._bound)]
            for name, tensor in list(node._bound.items()):
                before += tensor.numel() * tensor.element_size()
                dim = table[name]  # KeyError here means an unmapped weight
                if dim is None:
                    after += tensor.numel() * tensor.element_size()
                    continue
                n = tensor.shape[dim] // world_size
                node._bound[name] = (
                    tensor.narrow(dim, rank * n, n).contiguous()
                )
                del tensor
                after += node._bound[name].numel() * node._bound[name].element_size()
        for child in node.modules:
            b, a = visit(child)
            before += b
            after += a
        return before, after

    return visit(twin_root)


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


class SessionTP2(rt1.Session):
    """One rank of the TP2 pair.

    Weights load whole, then `shard_weights` keeps this rank's half; caches
    come up at the sharded shapes.
    """

    def __init__(self, cfg=None, *, ckpt=None, device, capacity=1024, verbose=False):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._cache_heads = rt1._NH // self.world_size
        import weights as wt  # noqa: PLC0415

        super().__init__(
            cfg, ckpt=ckpt or wt.CKPT, device=device, capacity=capacity,
            verbose=verbose,
        )

    def _twin_cls(self):
        if self.cfg is sem.config:
            return KimiLinear48BA3BTP
        return _root_twin_tp(self.sem)

    def _post_load(self) -> None:
        before, after = shard_weights(self.twin, self.rank, self.world_size)
        torch.cuda.empty_cache()
        self.sharded_bytes = after
        if self.rank == 0:
            print(
                f"sharded weights: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB "
                f"per rank",
                flush=True,
            )

    def _shard_kda_entry(self, entry):
        """The KDA cache is per-head: this rank keeps its heads' slices."""
        *convs, state = entry
        r, ws = self.rank, self.world_size
        nh = state.shape[1] // ws
        kp = convs[0].shape[-1] // ws
        return tuple(
            [c[..., r * kp : (r + 1) * kp].contiguous() for c in convs]
            + [state[:, r * nh : (r + 1) * nh].contiguous()]
        )


__all__ = [
    "KimiLinear48BA3BTP",
    "SessionTP2",
    "comm_report",
    "comm_reset",
    "shard_weights",
]
