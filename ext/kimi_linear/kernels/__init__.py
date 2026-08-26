"""tilelang kernels for the Kimi-Linear KDA decode step.

    kda        short_conv / l2_normalize standalone boundaries and the fused
               whole-mixer step (`kda_step`), three launches
    torch_ref  the same functions in plain torch

Every kernel is decode-shaped: sequence length 1, all matmuls are GEMV, the
recurrent state is `[1, heads, v_dim, k_dim]` (the authored layout), and the
convolution windows are caller-owned and replaced, never appended to.

`TF_IMPL=torch` in the environment switches every entry point to `torch_ref`,
which is what makes a wrong output bisectable to one kernel instead of one step.
"""
