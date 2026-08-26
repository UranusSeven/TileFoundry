"""KDA tilelang kernels vs the official checkpoint implementation and the authored IR.

Four-way comparison on cuda, bf16, real config dims (hidden 2304, 32 heads x
128, conv kernel 4), shared seeded random weights -- the same construction as
`/root/develop/yingshan/oracle/check_kda_oracle.py`, which established that the
authored IR is ground truth:

  A. HF oracle: kimi_ref.modeling_kimi.KimiDeltaAttention (checkpoint's own
     modeling code) + fla 0.5.2 fused_recurrent_kda, decode mode.
  B. Authored IR: KimiLinear48BA3B.kda.lookup("kda_attention") via
     tilefoundry.evaluator.evaluate.
  K. The tilelang kernels in `kernels/kda.py` (the fused `kda_step`).
  R. The plain-torch reference in `kernels/torch_ref.py`.

Single step, then a chained 8-step recurrence (states and conv windows fed back
on every side) -- a single step proves one transition; chaining proves the
replace contract. Latency at the end: wall time per fused step, 100 iterations
after warmup.

Reproduce (container dev-yingshan-7cf9dbcf45-xtm8p):

  cd /root/develop/yingshan/TileFoundry && \
    CUDA_VISIBLE_DEVICES=2 python3 ext/kimi_linear/check_kda_kernel.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/root/develop/yingshan/oracle")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ext"))

import torch  # noqa: E402
from fla.ops.kda.gate import naive_kda_gate  # noqa: E402
from kimi_ref import configuration_kimi, modeling_kimi  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16
CONFIG_JSON = "/models/moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json"
WEIGHT_SEED = 0
ACT_SEED = 1
ATOL = RTOL = 2e-2
STEPS = 8


def fused_kda_gate_shim(g, A_log, head_dim, g_bias=None):
    # Same shim as check_kda_oracle.py: the checkpoint modeling calls the
    # fla-core signature; installed fla 0.5.2 differs. Same math, f32, and the
    # vendored file stays byte-identical to the checkpoint.
    *lead, hk = g.shape
    h = hk // head_dim
    return naive_kda_gate(g.view(*lead, h, head_dim), A_log.reshape(-1), g_bias)


modeling_kimi.fused_kda_gate = fused_kda_gate_shim

cfg = configuration_kimi.KimiLinearConfig(**json.load(open(CONFIG_JSON)))
KDA = cfg.linear_attn_config
H, D, W = KDA["num_heads"], KDA["head_dim"], KDA["short_conv_kernel_size"]
KP = H * D
HIDDEN = cfg.hidden_size
EPS = cfg.rms_norm_eps
assert cfg.is_kda_layer(0), "layer 0 must be a KDA layer"

# ── shared weights (identical seeds and order to check_kda_oracle.py) ────────

torch.manual_seed(WEIGHT_SEED)
with torch.device(DEVICE):
    kda = modeling_kimi.KimiDeltaAttention(cfg, layer_idx=0)

torch.manual_seed(WEIGHT_SEED)
with torch.no_grad():
    for name, p in kda.named_parameters():
        if name.endswith("A_log"):
            p.copy_(torch.log(torch.empty_like(p).uniform_(1.0, 16.0)))
        elif name.endswith("o_norm.weight"):
            p.normal_(1.0, 0.1)
        else:
            p.normal_(0.0, 0.02)
kda = kda.eval().to(DTYPE)


def drawn(*shape, sigma=0.05):
    return (torch.randn(*shape, device=DEVICE) * sigma).to(DTYPE)


def make_hn(hidden, gamma_in):
    h32 = hidden.float()
    return (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + EPS)).to(DTYPE) * gamma_in


def lw(linear):  # nn.Linear [out, in] -> authored [1, in, out]
    return linear.weight.detach().t().unsqueeze(0).contiguous()


def cw(conv):  # ShortConvolution [D, 1, W] -> authored [W, D]
    return conv.weight.detach()[:, 0, :].t().contiguous()


def to_hf_conv(cs):
    # HF cache is [N, D, W]; slot 0 is stale, slots 1..W-1 hold the window.
    buf = torch.zeros(1, KP, W, device=DEVICE, dtype=DTYPE)
    buf[:, :, 1:] = cs[0].t()
    return buf


def hf_conv_back(buf):  # [1, KP, W] -> authored [1, W-1, KP]
    return buf[:, :, 1:].transpose(1, 2)


def step_args(hidden, gamma_in, wins, st):
    return (
        hidden, gamma_in,
        lw(kda.q_proj), lw(kda.k_proj), lw(kda.v_proj),
        cw(kda.q_conv1d), cw(kda.k_conv1d), cw(kda.v_conv1d),
        wins[0], wins[1], wins[2],
        lw(kda.f_a_proj), lw(kda.f_b_proj),
        kda.dt_bias.detach().contiguous(),
        kda.A_log.detach().reshape(H).contiguous(),
        lw(kda.b_proj),
        lw(kda.g_a_proj), lw(kda.g_b_proj),
        kda.o_norm.weight.detach().contiguous(),
        lw(kda.o_proj),
        st,
        torch.full((1, 1, 1), D ** -0.5, device=DEVICE, dtype=DTYPE),
    )


def report(name, a, b):
    a, b = a.float(), b.float()
    adiff = (a - b).abs()
    denom = b.abs().clamp_min(1e-3)
    rdiff = adiff / denom
    ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    print(f"  {name:30s} shape={tuple(a.shape)}  max|d|={adiff.max().item():.3e}  "
          f"max rel={rdiff.max().item():.3e}  allclose(2e-2)={'PASS' if ok else 'FAIL'}")
    return ok


# ── the kernels under test ───────────────────────────────────────────────────

from kimi_linear.kernels import kda as kk  # noqa: E402
from kimi_linear.kernels import torch_ref  # noqa: E402

from tests.models.kimi_linear_48b_a3b.model import KimiLinear48BA3B  # noqa: E402
from tilefoundry.evaluator import evaluate  # noqa: E402

fn = KimiLinear48BA3B.kda.lookup("kda_attention")

# ── single step ──────────────────────────────────────────────────────────────

torch.manual_seed(ACT_SEED)
hidden = drawn(1, 1, HIDDEN, sigma=0.1)
gamma_in = (torch.randn(HIDDEN, device=DEVICE) * 0.1 + 1.0).to(DTYPE)
conv_states = [drawn(1, W - 1, KP) for _ in range(3)]
state_auth = drawn(1, H, D, D)
hn = make_hn(hidden, gamma_in)

cache = modeling_kimi.KimiDynamicCache(cfg)
cache.conv_states[0] = tuple(to_hf_conv(cs) for cs in conv_states)
cache.recurrent_states[0] = state_auth.transpose(-1, -2).contiguous()
with torch.no_grad():
    out_hf = kda(hn, attention_mask=None, cache_params=cache)

args = step_args(hidden, gamma_in, conv_states, state_auth)
out_b, state_b, cq_b, ck_b, cv_b = evaluate(fn, *args, device=DEVICE)

t0 = time.perf_counter()
out_k, state_k, cq_k, ck_k, cv_k = kk.kda_step(*args)
torch.cuda.synchronize()
print(f"(first kda_step call incl. compile: {time.perf_counter() - t0:.1f}s)")

out_r, state_r, cq_r, ck_r, cv_r = torch_ref.kda_step(*args)

ok = True
print("== single step: tilelang kernels (K) vs HF oracle (A) ==")
ok &= report("out", out_k, out_hf)
ok &= report("state_next", state_k, cache.recurrent_states[0].transpose(-1, -2))
ok &= report("conv_q_next", cq_k, hf_conv_back(cache.conv_states[0][0]))
ok &= report("conv_k_next", ck_k, hf_conv_back(cache.conv_states[0][1]))
ok &= report("conv_v_next", cv_k, hf_conv_back(cache.conv_states[0][2]))

print("== single step: tilelang kernels (K) vs authored IR (B) ==")
ok &= report("out", out_k, out_b)
ok &= report("state_next", state_k, state_b)
ok &= report("conv_q_next", cq_k, cq_b)
ok &= report("conv_k_next", ck_k, ck_b)
ok &= report("conv_v_next", cv_k, cv_b)

print("== single step: torch_ref (R) vs HF oracle (A) ==")
ok &= report("out", out_r, out_hf)
ok &= report("state_next", state_r, cache.recurrent_states[0].transpose(-1, -2))
ok &= report("conv_q_next", cq_r, hf_conv_back(cache.conv_states[0][0]))

print("== single step: torch_ref (R) vs authored IR (B) ==")
ok &= report("out", out_r, out_b)
ok &= report("state_next", state_r, state_b)

# ── standalone op boundaries ─────────────────────────────────────────────────

print("== standalone ops: tilelang vs torch_ref ==")
torch.manual_seed(ACT_SEED + 7)
x_conv = drawn(1, 1, KP, sigma=0.3)
y_tl, cn_tl = kk.short_conv(x_conv, cw(kda.q_conv1d), conv_states[0])
y_rf, cn_rf = torch_ref.short_conv(x_conv, cw(kda.q_conv1d), conv_states[0])
ok &= report("short_conv out", y_tl, y_rf)
ok &= report("short_conv window", cn_tl, cn_rf)
x_l2 = drawn(1, 1, H, D, sigma=0.3)
ok &= report("l2_normalize", kk.l2_normalize(x_l2), torch_ref.l2_normalize(x_l2))

# ── chained recurrence: STEPS decode steps, states fed back on all sides ─────

print(f"== chained {STEPS}-step recurrence ==")
cache2 = modeling_kimi.KimiDynamicCache(cfg)
cache2.conv_states[0] = tuple(to_hf_conv(cs) for cs in conv_states)
cache2.recurrent_states[0] = state_auth.transpose(-1, -2).contiguous()
st_b, wins_b = state_auth, list(conv_states)
st_k, wins_k = state_auth, list(conv_states)
for step in range(STEPS):
    torch.manual_seed(ACT_SEED + 101 + step)
    hid_t = (torch.randn(1, 1, HIDDEN, device=DEVICE) * 0.1).to(DTYPE)
    hn_t = make_hn(hid_t, gamma_in)
    with torch.no_grad():
        o_hf = kda(hn_t, attention_mask=None, cache_params=cache2)
    args_t = step_args(hid_t, gamma_in, wins_b, st_b)
    o_b, st_b, w0, w1, w2 = evaluate(fn, *args_t, device=DEVICE)
    wins_b = [w0, w1, w2]
    o_k, st_k, w0, w1, w2 = kk.kda_step(*step_args(hid_t, gamma_in, wins_k, st_k))
    wins_k = [w0, w1, w2]
    ok &= report(f"step {step} out   (K vs A)", o_k, o_hf)
    ok &= report(f"step {step} out   (K vs B)", o_k, o_b)
    ok &= report(f"step {step} state (K vs A)", st_k,
                 cache2.recurrent_states[0].transpose(-1, -2))
    ok &= report(f"step {step} cq    (K vs A)", wins_k[0],
                 hf_conv_back(cache2.conv_states[0][0]))

# ── latency ──────────────────────────────────────────────────────────────────

print("== latency (torch timers, 100 iterations after 20 warmup) ==")
bench_args = step_args(hidden, gamma_in, conv_states, state_auth)


def bench(call, iters=100, warmup=20):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


us_k = bench(lambda: kk.kda_step(*bench_args))
us_r = bench(lambda: torch_ref.kda_step(*bench_args), iters=20, warmup=3)
print(f"  kda_step tilelang   {us_k:8.2f} us/step")
print(f"  kda_step torch_ref  {us_r:8.2f} us/step")

print()
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
