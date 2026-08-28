

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
