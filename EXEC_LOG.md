

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


## Prefill-only HIR 收敛（阶段1）

- Removed the decode shell, decode runtimes, decode kernels, decode checks, and HF greedy driver under `ext/kimi_linear/`.
- Replaced `ext/kimi_linear/model.py` with an independent `DimVar("seq_len", 1, model_max_length)` prefill model containing the published 27-layer stack, KDA/MLA mixers, MoE/dense FFNs, embedding, final norm, and last-position LM head.
- Added HIR-only `tf.kda_prefill(q, k, v, g, beta, scale=...) -> output` with whole-sequence inputs, zero initial recurrent state, no final-state handoff, type inference, access relation, and a sequential Torch evaluator matching causal chunk-KDA delta-rule semantics. `tf.causal_depthwise_conv1d` keeps KDA's causal projection convolution in HIR.
- Unified routed-expert weights to packed `w_gate_up [E, 2I, H]` plus `w_down [E, H, I]`; authored HIR uses static slices for gate/up, and checkpoint loading packs per-expert `w1/w3` pairs directly.
- Checks: all retained extension Python files compile; the authored model imports and has 27 layers; model forest/function counts and KDA evaluator/type inference were inspected; paged MLA smoke, Ruff, repository hygiene lints, and `git diff --check` were run before commit.
- Limitation: this is phase-1 HIR convergence, not a complete runnable prefill pipeline. `run.py`, `prefill.py`, and legacy broad check drivers still contain decode-session assumptions and are deferred to the phase-2 runtime rewrite.

## Prefill-only runtime（阶段2）

All work and validation ran inside pod `dev-yingshan-7cf9dbcf45-xtm8p` on branch `kimi-linear-run`, using GPU 2 (`CUDA_VISIBLE_DEVICES=2`).

- Replaced the transitional session/handoff driver with `PrefillRunner`, which loads and owns the prefill-only weight tree and executes embedding, all 27 layers, final RMS norm, and the last-position LM head. MLA pages are allocated per request and freed in `finally`; KDA calls FLA `chunk_kda` with `initial_state=None` and `output_final_state=False`; MoE calls vLLM `fused_experts` over packed `w_gate_up [256, 2048, 2304]` and `w_down`.
- Reduced `weights.py` to the prefill resource API and removed tokenizer/rotary/decode-resource helpers. Rewrote `run.py` as a prefill-only benchmark with `--input-ids`, `--length`, `--repeat`, `--warmup`, `--max-tokens 1`, and `--out`. Added `hf_prefill_logits.py` using the checkpoint `modeling_kimi`, `use_cache=False`, fixed token ids, the known `/root/develop/yingshan/venv_hf`, and compatibility-only import/gate/eager-attention shims.
- Replaced legacy decode/handoff checks with a prefill forest/packed-ABI check and a paged-MLA external-op plus real-checkpoint HF-logit check.

Correctness on the fixed ids:

- S=8: runtime/HF argmax `3971/3971`, relative L2 `3.406950e-02`.
- S=32: runtime/HF argmax `1/1`, relative L2 `3.971295e-02`.
- S=100: runtime/HF argmax `295/295`, relative L2 `3.983630e-02`.
- Every runtime row had shape `(163840,)` and all finite values; all comparisons passed the required `rel_l2 <= 6e-2` gate.
- Paged MLA evaluator versus FA3: max absolute `1.562500e-02`, relative L2 `1.894684e-03`.

Benchmark contract, S=32, one warmup and three repeats: logits shape `[1, 163840]`, BF16/CUDA, argmax `1`, checksum `-560219.75`, median TTFT `48.867 ms`, input throughput `654.834 tok/s`. `--max-tokens 2` was rejected by argparse and `--max-tokens 1` completed and wrote the JSON report.

Validation passed: Python compilation for all retained extension files, 27-layer model forest and packed-weight ABI, `check_kda_prefill_op.py`, rewritten fast checks, HF/runtime real-checkpoint comparisons at S=8/32/100, Ruff on changed files, comment/English/forward-reference/machine-path hygiene lints, and `git diff --check`.

Phase 3 remains intentionally out of scope: TP2 prefill and the broader performance sweep/tuning.


## Prefill-only TP2 与 vLLM baseline

Phase 3 ran in the fixed chengfeng container with `torchrun`, NCCL, GPUs 0+1, BF16, batch 1, `max_tokens=1`, fixed baseline token IDs, and three repeats after one same-shape warmup. The runtime directly slices safetensors during checkpoint loading: KDA/MLA heads 32→16 per rank, all 256 routed experts retained with intermediate 1024→512, shared experts and layer-0 dense MLP similarly split, output projections row-split, and each mixer/FFN output all-reduced. Embedding and LM head remain replicated. Rank-local resident and load peak were both 47.213 GiB, confirming no transient full 98 GB model was loaded before slicing.

Correctness on fixed IDs:
- S=8: TP2/TP1 argmax 3971/3971, rel-L2 1.7015e-2; TP2/HF argmax 3971/3971, rel-L2 3.6607e-2.
- S=32: TP2/TP1 argmax 1/1, rel-L2 4.8300e-2; TP2/HF argmax 1/1, rel-L2 4.6144e-2.
- S=100: TP2/TP1 argmax 295/295, rel-L2 4.5029e-2; TP2/HF argmax 295/295, rel-L2 4.8565e-2.
- Rank-0/rank-1 logits were bit-identical at all lengths. HF gates passed; TP1 differences are BF16 collective-order noise with exact argmax.

Quick gate medians: S=512 57.925 ms / 23.7 ms = 2.444x; S=1024 59.498 ms / 50.8 ms = 1.171x; S=4096 98.795 ms / 58.0 ms = 1.703x. The full sweep was skipped because 512 and 4096 miss 1.5x.

Exact traces are `/root/develop/yingshan/traces/prefill_tp2_phase3_s1024_rank0.json` and `/root/develop/yingshan/traces/prefill_tp2_phase3_s4096_rank0.json`, with rank-1 siblings. S=1024 profiler wall/summed GPU events were 196.037/221.033 ms: MoE 41.048 ms, KDA 36.684 ms, NCCL 25.709 ms, MLA 5.010 ms. S=4096 was 181.115/257.545 ms: KDA 43.386 ms, MoE 35.282 ms, MLA 6.044 ms, NCCL 4.793 ms. The dominant residual bottleneck at the failing long shape is the 20-layer KDA path, not communication. Next action: fuse its projection-conv-gate elementwise path or use a TP-aware fused KDA prefill kernel; then revisit fixed 54-collective latency at S=512.
