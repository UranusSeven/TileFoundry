---
type: FEAT
component: distributed-hir
target_repo: tilefoundry
---

# [FEAT][distributed-hir] All-to-all and bounded MoE dispatch/combine

## Description

Add equal-size all-to-all redistribution and routed MoE dispatch/combine to
distributed HIR checking. Expert computation remains ordinary HIR. The semantic
reference is DeepEP V2's expanded dispatch and the pinned vLLM integration:
dispatch transports route weights; expert outputs are weighted before combine;
combine restores source-token order and sums contributions.

Semantic sources: [DeepEP ElasticBuffer](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/buffers/elastic.py#L855),
the [requested vLLM manager](https://github.com/vllm-project/vllm/blob/c9611215195eaaaa49a9ef6b65c1b28abd15e8ed/vllm/distributed/device_communicators/all2all.py#L1012),
and its [dispatch/combine integration](https://github.com/vllm-project/vllm/blob/c9611215195eaaaa49a9ef6b65c1b28abd15e8ed/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py).

### Current state

- `src/tilefoundry/ir/hir/sharding/collective.py` defines reductions and gathering, but no all-to-all.
- `src/tilefoundry/evaluator/distributed.py` holds local tensors per mesh participant and reconstructs split results.
- `src/tilefoundry/cli/check.py` binds one logical input/weight set to explicit candidate and reference HIRs.
- `tests/fixtures/distributed/projection.py` exercises regular partitions; routed and bounded variable-count payloads have no executable example.

### Decisions

- D1 Regular exchange -- AllToAll changes ownership from one logical Split axis to another while preserving logical shape and values; each peer receives a slice from every sender.
- D2 Routed representation -- AllToAllDispatch returns expert-major padded tokens, route weights, source-token indices and actual expert counts as ordinary tensors. A static capacity per expert gives a declared allocation bound; overflow fails rather than dropping tokens.
- D3 Expert placement -- experts are contiguous equal partitions along the selected mesh axis. Other mesh axes may split leading batch dimensions, preserving independent DP groups. Arbitrary expert maps and backend-specific buffer layouts are outside this contract.
- D4 Route semantics -- each nonnegative expert ID contributes one route; -1 disables a slot. Multiple routes to one rank retain their individual expert contributions. Repeated expert selections retain multiplicity. Inactive output rows are zero-padded with -1 source indices.
- D5 Combine -- consume already-weighted expert outputs plus explicit indices/counts and a source-shaped tensor. This keeps router weighting in HIR and avoids hidden or repeated weighting. Only active rows participate; source indices restore the original token owner and order.
- D6 Shape contract -- inputs are [..., tokens, hidden] and [..., tokens, topk]; outputs are [..., experts, capacity, hidden/1], [..., experts, capacity], and [..., experts]. Leading dimensions carry optional independent groups, and symbolic token counts remain runtime-bound.
- D7 Evaluator organization -- use a dedicated routed-exchange evaluator module, sharing participant groups with regular collectives. Reuse existing MatMul, pointwise operations and ReLU for expert MLPs; do not add an opaque expert kernel.
- D8 Analysis -- register logical access relations for the new primitives; keep missing distributed cost models explicit errors, as for the existing device collectives.
- D9 Evidence -- test numerical MoE dispatch/expert/combine end to end with skew, empty destinations, disabled routes, same-rank experts, duplicate selections and overflow. Demonstrate the existing check CLI using file-backed route inputs.

## Milestones

### Milestone M0: Regular and routed exchange semantics

#### Depends
- None

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/ir/hir/sharding/collective.py
+class AllToAll(Op): ...
# src/tilefoundry/ir/hir/sharding/alltoall.py
+class AllToAllDispatch(Op): ...
+class AllToAllCombine(Op): ...
# src/tilefoundry/evaluator/distributed.py
+def participant_groups(mesh, axis): ...
# src/tilefoundry/evaluator/alltoall.py
+def dispatch(ctx): ...
+def combine(ctx): ...
# tests/fixtures/distributed/moe.py
+class Reference: ...
+class ExpertParallel: ...
```

##### Accepted by

New public evaluator tests and a reusable MoE fixture cover communication and
nonlinear expert computation, which existing projection tests cannot exercise.
Wrong ownership, route multiplicity, weighting or source restoration changes
their numerical results. Existing distributed and analysis-invariant suites
remain unchanged regression coverage.

Validation: 498 CPU tests passed across evaluator, operation, mesh/parser, CLI
and analysis coverage. One existing cross-device test was skipped and one
existing CUDA-default CLI test was excluded. The 22 all-to-all cases cover
independent groups, an ownership-sensitive contraction after regular exchange,
nonlinear MoE expert compute, route multiplicity, skew, disabled routes,
dynamic/empty batches, zero-capacity dispatch/combine, malformed metadata,
padding contamination and capacity overflow. Static zero-capacity expert MLPs
encounter an existing empty-MatMul type-inference limitation; the zero-capacity
route case therefore checks dispatch and combine without expert computation.

- [x] AllToAll redistributes and round-trips values across independent groups.
- [x] Routed dispatch and weighted combine agree with an independent MoE oracle.
- [x] Actual counts, padding, inactive tokens and capacity overflow are explicit.
- [x] Invalid route IDs, metadata and layouts fail at public boundaries.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/ir/hir/sharding/collective.py`
- `src/tilefoundry/ir/hir/sharding/alltoall.py`
- `src/tilefoundry/evaluator/distributed.py`
- `src/tilefoundry/evaluator/alltoall.py`
- `tests/evaluator/test_alltoall.py`
- `tests/fixtures/distributed/moe.py`
- `docs/spec/architecture.md`
- `docs/spec/hir.md`
- `docs/spec/evaluator.md`

### Milestone M1: Agent-facing example and CLI validation

#### Depends
- M0

#### Target State Design

##### Delivered
```diff
# tests/cli/test_cli_check.py
+def test_routed_moe_reference_with_file_inputs(...): ...
# docs/tutorial/moe-alltoall.ipynb
+# Extract and execute reference/candidate HIR and structured input generation.
# docs/tutorial/moe-alltoall.md
+# Generated tutorial, including the passing check output.
# src/tilefoundry/cli/tutorial.py
+PAGES = (*existing_pages, "moe-alltoall")
```

##### Accepted by

The existing check command must compare the two HIRs with real routing inputs;
the tutorial executes its extracted source and records the result. This covers
binding and metadata transport beyond the evaluator tests without adding a new
command or duplicated numerical engine.

Validation: the file-backed check passes and records the explicit reference
and distributed evaluation mode in JSON. The notebook executed its extracted
programs and produced the committed Markdown output. All four analysis
selectors reject the missing AllToAllDispatch cost model, and routing access
relations have bounded images for every field. Ruff and repository specification,
reference, comment, annotation, language and path checks pass.

- [x] File-backed routes and shared weights pass check with explicit predicates.
- [x] The installed tutorial describes the expanded layout and explicit weighting.
- [x] Documentation states backend and cost-model boundaries and cites the referenced semantics.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `tests/cli/test_cli_check.py`
- `docs/tutorial/moe-alltoall.ipynb`
- `docs/tutorial/moe-alltoall.md`
- `docs/tutorial/index.md`
- `src/tilefoundry/cli/tutorial.py`
- `docs/spec/cli.md`

## Final Gate

<!-- final_gate:start -->
<!-- final_gate:end -->
