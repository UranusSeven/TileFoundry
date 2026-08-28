

## 2026-08-28 grouped MoE prefill correction

The grouped TileLang kernel was numerically wrong because both intermediate gate/up output and final routing-weight lookup used the unpadded expert-relative index (`start + i`) against padded buffers. For experts after the first, this read/write crossed padded expert regions. The gate/up result is now stored at the padded block row (`p0 + i`), and the down output multiplies by the padded row's weight (`weights[p0 + i]`). Expert weights remain `[E,I,H]` for gate/up and `[E,H,I]` for down, matching `transpose_B=True`.

Focused deterministic CUDA comparison on GPU 2, E=4/I=32/H=64/top-8, against a float32 PyTorch loop:
- S=2: max_abs=3.483e-03, rel_l2=1.464e-03, argmax=True
- S=8: max_abs=6.012e-03, rel_l2=1.407e-03, argmax=True
- S=32: max_abs=1.157e-02, rel_l2=1.657e-03, argmax=True

The test exercised `grouped_routed` and compiled the TileLang grouped kernel; no per-token weight-gather fallback is called.


## 2026-08-28 grouped MoE follow-up verification

All commands below ran inside pod `dev-yingshan-7cf9dbcf45-xtm8p` in namespace `default`, using GPU 2 (`CUDA_VISIBLE_DEVICES=2`), on branch `kimi-linear-run` at commit `029c7d4`.

### 1. Real-size grouped kernel CUDA comparison

Deterministic float32 per-assignment Torch reference versus `grouped_routed` with `E=256`, `I=1024`, `H=2304`, `topk=8`; output showed TileLang compilation of `_kernel` and the call completed through the grouped implementation.

- `S=8`: `max_abs=1.557891e+02`, `rel_l2=1.673228e-03`, `argmax=True` (`8/8`)
- `S=32`: `max_abs=1.874062e+02`, `rel_l2=1.680781e-03`, `argmax=True` (`32/32`)

The comparison directly called `grouped_routed`; no per-token weight-gather fallback was used.

### 2. Paged prefill check

`CUDA_VISIBLE_DEVICES=2 python3 ext/kimi_linear/check_paged_prefill.py`: **PASS**. FA3 comparison reported `max|d|=1.562e-02`, `rel_l2=1.895e-03`.

### 3. Paged generation and loop reference

The requested paged run generated 32 tokens and printed coherent matrix-multiplication explanation text, with no repeated garbage. Its output showed repeated `TileLang begins/completes to compile kernel main` messages. The same prompt and token budget with `--prefill loop` produced identical ids: `TOKEN_EQUALITY True`, `32/32`, `mismatches=0`.

Artifacts: `/tmp/grouped_paged.json` and `/tmp/grouped_loop.json` (container-local).

### 4. S=1024 profile status

A CUDA profiler run was attempted after warmup, but the generated prompt contained `S=961` rather than requested `S=1024`; this is incomplete and must not be treated as the requested S=1024 result. The measured S=961 trace reported `Self CUDA time total: 67.483ms`; grouped TileLang entries were `main_kernel_1` `14.068ms` (20.85%) and `main_kernel` `13.085ms` (19.39%), 26 calls each.

The trace **did contain** old `void at::native::vectorized_gather_kernel<16, long>(...)` (`1.823ms`, 67 calls), so the required no-old-kernel condition failed. Because step 4 did not pass, S=4096 was not attempted.

**Overall:** steps 1--3 passed; the final profile gate failed due to both the accidental `S=961` length and the presence of `vectorized_gather_kernel` in the trace.

## 2026-08-28 grouped MoE profile follow-up (final evidence)

- Profile script now asserts fixed input lengths and explicitly constructs a 1-D int64 token tensor, so completed traces are exact S=1024 and S=4096; each shape was warmed before profiling.
- Full paged traces: S=1024 wall 2494.8 ms, 4511 CUDA kernels, summed kernel duration 81.904 ms, vectorized_gather_kernel 67 calls / 1.928 ms. S=4096 wall 9515.1 ms, 4511 CUDA kernels, summed kernel duration 199.814 ms, vectorized_gather_kernel 67 calls / 7.779 ms.
- grouped_routed in ext/kimi_linear/kernels/moe_prefill.py uses sort/pad indexing followed by one TileLang function containing two grouped GEMM kernels (gate/up and down); it has no expert-weight gather. Full traces expose generated nvjet_sm90_tst_* names, not a source-level grouped label.
- Isolated grouped-MoE profiling was attempted but could not start: an existing vLLM worker PID 61146 occupied GPU 2 and model construction failed OOM. No algorithm change was made. Source inspection and full-trace evidence classify the remaining gathers as non-MoE bookkeeping (paged MLA/page or block-table repack, embedding/indexing/routing), not per-token expert-weight gathers.
- Fix commit 029c7d4 and validation commit 5bc190c remain on local kimi-linear-run (ahead 3); no push.


## MoE prefill 全 GPU dispatch（vLLM fused_moe）

All implementation and validation ran inside the chengfeng pod on GPU 2 with vLLM 0.27.1. The selected API is `vllm.model_executor.layers.fused_moe.fused_moe.fused_experts`: unquantized BF16, `MoEActivation.SILU` (fused SiLU+mul), `w1=[E,2I,H]`, `w2=[E,H,I]`, custom CUDA `topk_weights/topk_ids=[S,8]`, and `apply_router_weight_on_input=False`. The installed API accepts BF16 or FP32 routing weights and int32 or int64 ids. A synthetic E=4/H=64/I=32/S=8 probe reported `rel_l2=3.397e-3`, argmax agreement, and BF16 output.

`ext/kimi_linear/kernels/moe_prefill_vllm.py` loads only vLLM's fused kernel module (the package initializer's optional NIXL/DeepEP protobuf extensions conflict with TileFoundry's OR-Tools in this image). Session initialization concatenates gate/up once into `[256,2048,2304]`, retaining decode-compatible names as views while prefill runs; final paged prefill restores contiguous decode tensors layer-by-layer. `KIMI_MOE_PREFILL_IMPL=tilelang` keeps commit 029c7d4's grouped kernel as the explicit fallback. The default hot call has no `.cpu()`, `.numpy()`, `bincount`, host sort, Python expert loop, or per-token expert-weight gather. Authored routing remains unchanged: f32 sigmoid logits, top-k on score+bias, selected-bias subtraction, top-k renormalization, BF16 scale 2.446, and unscaled shared expert.

Focused real-shape E=256/H=2304/I=1024/topk=8 comparisons against an authored-rounding Torch reference:
- S=8: max_abs `7.8125e-3`, rel_l2 `3.473e-3`, argmax `8/8`.
- S=32: max_abs `7.8125e-3`, rel_l2 `3.363e-3`, argmax `32/32`.

Correctness:
- `CUDA_VISIBLE_DEVICES=2 python3 ext/kimi_linear/check_paged_prefill.py`: **PASS** (`max|d|=1.562e-2`, `rel_l2=1.895e-3`).
- Fixed prompt paged vs loop generation: `greedy_ids` exact equality, **32/32**, zero mismatches. The paged output used the default vLLM fused path; loop retained decode kernels.

Steady-state prefill-only wall after same-shape warmup (`max_tokens=1` equivalent final-logit work, exact token tensors):
- S=512: **51.772 ms** (vLLM TP2 baseline 23.7 ms; 2.18x).
- S=1024: **74.866 ms** (baseline 50.8 ms; 1.47x).
- S=4096: **135.898 ms** (baseline 58.0 ms; 2.34x).

Exact-shape Torch/CUDA traces (`/root/develop/yingshan/traces/tf_vllm_moe/`), each after same-shape warmup:
- S=1024: profiled CPU wall 79.064 ms, CPU event span 78.928 ms, summed GPU kernels+memcpy 51.090 ms, MoE CUDA 23.331 ms (52 `fused_moe_kernel`, 26 `moe_sum`, 26 GPU `moe_align_block_size`). Previous TileLang full trace was 81.904 ms GPU / 2494.8 ms wall; vLLM TP2 GPU baseline 21.4 ms.
- S=4096: profiled CPU wall 159.341 ms, CPU event span 159.210 ms, summed GPU kernels+memcpy 132.923 ms, MoE CUDA 37.573 ms with the same 104 MoE launches. Previous TileLang full trace was 199.814 ms GPU / 9515.1 ms wall; vLLM TP2 GPU baseline 49.4 ms.
- Trace CPU-op search found no `.cpu()`, `bincount`, NumPy, argsort, or host sort operation in either trace. MoE metadata is produced by vLLM's CUDA `moe_align_block_size_kernel`.

The remaining gap is not host MoE dispatch. At S=1024, GPU busy is 29.7 ms above the vLLM TP2 rank baseline and MoE alone is 23.3 ms; TP1 reads full expert weights while the baseline is TP2. At S=4096, GPU busy exceeds baseline by 83.5 ms while MoE is 37.6 ms, so KDA/MLA/dense TP1 work and the 26-layer Python launch span dominate the residual. The installed package has no tuned H200 config for E=256/N=1024 and emits the default-config warning, leaving additional fused-MoE tuning headroom.
