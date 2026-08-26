"""tilelang kernels for the Kimi-Linear decode step.

    kda        KDA: short_conv / l2_normalize standalone boundaries and the
               fused whole-mixer step (`kda_step`), three launches
    attn       MLA: the `mla_attention` decode step (fused input RMSNorm +
               q/kv projections, latent norm + kv_b expansion, online-softmax
               attention over the cache plus this token, log-sum-exp merge +
               o_proj), five launches
    moe        the MoE block (sigmoid router with selection-only bias, 256
               routed experts + one shared expert in two fused kernels) and
               layer 0's dense SwiGLU MLP
    basic      the shared basic ops: embedding-row gather, RMSNorm with the
               authored round-before-gamma placement, residual add, and the
               lm_head GEMV
    torch_ref  the KDA functions in plain torch

Every kernel is decode-shaped: sequence length 1, all matmuls are GEMV. The
KDA recurrent state is `[1, heads, v_dim, k_dim]` (the authored layout) and its
convolution windows are caller-owned and replaced, never appended to; the MLA
cache is read-only inside the step and its new k/v come back for the caller to
append -- both exactly the authored contracts.

`TF_IMPL=torch` in the environment switches the KDA entry points to
`torch_ref`, which is what makes a wrong output bisectable to one kernel
instead of one step.
"""
