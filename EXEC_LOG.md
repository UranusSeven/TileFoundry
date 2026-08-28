

## 2026-08-28 grouped MoE prefill correction

The grouped TileLang kernel was numerically wrong because both intermediate gate/up output and final routing-weight lookup used the unpadded expert-relative index (`start + i`) against padded buffers. For experts after the first, this read/write crossed padded expert regions. The gate/up result is now stored at the padded block row (`p0 + i`), and the down output multiplies by the padded row's weight (`weights[p0 + i]`). Expert weights remain `[E,I,H]` for gate/up and `[E,H,I]` for down, matching `transpose_B=True`.

Focused deterministic CUDA comparison on GPU 2, E=4/I=32/H=64/top-8, against a float32 PyTorch loop:
- S=2: max_abs=3.483e-03, rel_l2=1.464e-03, argmax=True
- S=8: max_abs=6.012e-03, rel_l2=1.407e-03, argmax=True
- S=32: max_abs=1.157e-02, rel_l2=1.657e-03, argmax=True

The test exercised `grouped_routed` and compiled the TileLang grouped kernel; no per-token weight-gather fallback is called.

