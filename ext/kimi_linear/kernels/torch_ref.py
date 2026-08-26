"""The KDA decode step in plain torch.

Not dead weight: `TF_IMPL=torch` routes every `kernels.kda` entry point here,
which is what makes a wrong output bisectable to one kernel instead of one
step, and `check_kda_kernel.py` scores it as its own leg. The semantics are the
authored IR's (`tests/models/kimi_linear_48b_a3b/model.py:KimiKda`), which the
oracle harness validated against the official checkpoint implementation; the
f32 placements follow it (matmuls in bf16 with f32 accumulate, the norm/gate/
delta internals in f32, one bf16 rounding at each authored tensor boundary).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

H = 2304
NH = 32
DK = 128
KP = NH * DK
CW = 4
WS = CW - 1

EPS = 1e-5  #: rms_norm_eps
L2_EPS = 1e-6  #: l2_normalize, inside the root, over the sum of squares


def short_conv(x, conv_w, conv_state):
    """`silu` of the 4-tap causal depthwise conv, plus the shifted window.

    Returns `(out, window_next)`: the window with its oldest position dropped
    and this token appended.
    """
    window = torch.cat([conv_state, x], dim=1)
    acc = (window.float() * conv_w.float().unsqueeze(0)).sum(1, keepdim=True)
    return F.silu(acc).to(x.dtype), window[:, 1:CW, :]


def l2_normalize(x):
    """`x / sqrt(sum(x^2) + 1e-6)` per head -- the sum, not the mean."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + L2_EPS)).to(x.dtype)


def kda_step(
    hidden, gamma_in,
    w_q, w_k, w_v,
    conv_w_q, conv_w_k, conv_w_v,
    conv_state_q, conv_state_k, conv_state_v,
    w_f_a, w_f_b, dt_bias, a_log,
    w_b, w_g_a, w_g_b,
    gamma_o, w_o,
    state, scale,
):
    """One KDA decode step, same contract as `kernels.kda.kda_step`."""
    hn32 = hidden.float()
    # Round the normalised value to bf16 *before* the learned scale multiplies
    # (KimiRMSNorm semantics), then let the bf16 product round again.
    hn = (hn32 * torch.rsqrt(hn32.pow(2).mean(-1, keepdim=True) + EPS)).to(
        hidden.dtype
    ) * gamma_in

    q_c, conv_q_next = short_conv(torch.matmul(hn, w_q), conv_w_q, conv_state_q)
    k_c, conv_k_next = short_conv(torch.matmul(hn, w_k), conv_w_k, conv_state_k)
    v_c, conv_v_next = short_conv(torch.matmul(hn, w_v), conv_w_v, conv_state_v)

    q_n = l2_normalize(q_c.view(1, 1, NH, DK))
    k_n = l2_normalize(k_c.view(1, 1, NH, DK))
    # The scale multiplies q after the l2 norm.
    q_s = q_n.float().reshape(1, NH, 1, DK) * scale.float().reshape(1, 1, 1, 1)

    # Per-channel forget gate: -exp(A_log) * softplus(g + dt_bias).
    low = torch.matmul(hn, w_f_a)
    g_raw = (torch.matmul(low, w_f_b) + dt_bias).view(1, 1, NH, DK).float()
    g = -a_log.float().exp().reshape(1, 1, NH, 1) * F.softplus(g_raw)
    beta = torch.sigmoid(torch.matmul(hn, w_b).float()).reshape(1, NH, 1)

    # The delta rule on the v-major [1, H, V, K] state; decay is per K channel.
    decay = g.exp().reshape(1, NH, 1, DK)
    k_r = k_n.float().reshape(1, NH, 1, DK)
    h_decayed = state.float() * decay
    kv_mem = (h_decayed * k_r).sum(-1)
    delta = (v_c.float().reshape(1, NH, DK) - kv_mem) * beta
    state_next = (h_decayed + delta.unsqueeze(-1) * k_r).to(state.dtype)
    attn = (state_next.float() * q_s).sum(-1)  # retrieval reads the update

    # Gated output norm: rms_norm(attn) * sigmoid(g2) -- sigmoid, not swish.
    g2 = torch.matmul(torch.matmul(hn, w_g_a), w_g_b).view(1, NH, DK).float()
    on = attn * torch.rsqrt(attn.pow(2).mean(-1, keepdim=True) + EPS)
    on = on * gamma_o.float() * torch.sigmoid(g2)
    out = torch.matmul(on.reshape(1, 1, KP).to(w_o.dtype), w_o)
    return out, state_next, conv_q_next, conv_k_next, conv_v_next


__all__ = ["kda_step", "l2_normalize", "short_conv"]
