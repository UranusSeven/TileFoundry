---
type: FEAT
component: distributed-hir
target_repo: tilefoundry
---

# [FEAT][distributed-hir] Execute and check explicit device collectives

## Description

Implement the first executable slice of the engine-scope RFC: rank-local
evaluation of distributed projections, explicit collectives, and candidate HIR
comparison through the existing `check` command. Distributed costing and ragged
expert exchange remain subsequent RFC work.

### Current state

- `src/tilefoundry/evaluator/interpreter.py:122` evaluates a mesh body once on logical tensors.
- `src/tilefoundry/ir/hir/sharding/reshard.py:277` evaluates a layout change without communication.
- `src/tilefoundry/ir/hir/nn/matmul.py:204` derives partial output ownership for split contractions.
- `src/tilefoundry/cli/check.py:306` compares runtime twins or saved outputs, with no second HIR selection.
- `src/tilefoundry/ir/types/shard/shard_layout.py:182` maps factored layout positions to logical axes.

### Decisions

- D1 Participant groups -- reuse a concrete single-level device Mesh and one mesh-axis attribute per collective; remaining coordinates identify independent groups. This preserves existing placement ownership.
- D2 Primitive slice -- implement AllReduce, AllGather, and ReduceScatter over equal partitions; reject unsupported distributions explicitly. Ragged exchange requires a separate payload contract.
- D3 Ordering -- use the existing deterministic operand-order DAG walk, with function calls and uniform structured loops inlined conceptually. All ranks execute each collective occurrence together; downstream execution must preserve that group order.
- D4 Simulation -- add an opt-in distributed evaluator context with rank-local values, reuse local primitive handlers, and reconstruct complete outputs only at the public boundary. No automatic completion of Partial outputs or communication through Reshard.
- D5 Supported local computation -- initially support pointwise arithmetic, MatMul, Cast, Transpose, tuple projection, and reductions over unsplit axes; fail closed on other device-local operations rather than pretend global evaluation is distributed execution.
- D6 HIR comparison -- add --reference SOURCE and --distributed to check. Bind activations positionally with matching logical shape/dtype and constants by the same resource paths. Reject incompatible signatures or conflicting reference modes.
- D7 Evidence -- add a portable projection/state example and numerical public tests; no new eval command, strategy registry, or composed-kernel cost function. Existing analysis must refuse collectives until communication costs are implemented.
- D8 Partition support -- use canonical or unfactored contiguous partitions and one mesh split per logical tensor axis; other layouts fail explicitly because their gathering may require additional redistribution.
- D9 Bound dimensions -- accept leading layout factors across mixed factored/static and unfactored/dynamic axes. Preserve zero-sized logical axes during factor-to-axis mapping; otherwise an empty shard's ownership shifts to another dimension. Refines D8.

## Milestones

### Milestone M0: Collective semantics and rank simulation

#### Depends
- None

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/ir/hir/sharding/collective.py
+class AllReduce(Op): ...
+class AllGather(Op): ...
+class ReduceScatter(Op): ...
# src/tilefoundry/evaluator/interpreter.py
-def evaluate(target, *inputs):
+def evaluate(target, *inputs, distributed=False):
# src/tilefoundry/evaluator/distributed.py
+class DistributedValue(Value): ...
+def distribute(value, type_, bindings): ...
+def reconstruct(value): ...
+def evaluate_distributed_op(ctx, handler): ...
# src/tilefoundry/ir/types/shard/shard_layout.py
+# Pseudocode: consume factors of a zero-sized logical axis through its zero factor.
```

##### Accepted by

New numerical projection/state tests exercise real DSL parsing, type inference,
local evaluation and communication. Incorrect subgroup membership, hidden
reductions, or a dropped rank breaks observable results. Existing evaluator,
sharding and parser suites remain regression coverage unchanged.

Validation: the combined evaluator, IR operation, mesh/parser, CLI and analysis
run passed 466 tests on CPU. One existing cross-device check was skipped and
one existing CLI test selecting CUDA by default was excluded after confirming
that this machine has no CUDA device. Numerical cases include dynamic and empty
partitions, function/loop boundaries, and repeated explicit state updates.

- [x] Partitioned projections agree with independently calculated torch results.
- [x] Multi-axis groups preserve separate replicas and partition order.
- [x] Partial outputs and communication-requiring Reshard fail explicitly.
- [x] State outputs can be supplied to a following invocation with matching results.
- [x] Local evaluation without distributed mode preserves existing behavior.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/ir/hir/sharding/collective.py`
- `src/tilefoundry/evaluator/`
- `src/tilefoundry/ir/types/shard/shard_layout.py`
- `tests/evaluator/test_distributed.py`
- `tests/fixtures/distributed/projection.py`
- `docs/spec/architecture.md`
- `docs/spec/hir.md`
- `docs/spec/evaluator.md`
- `docs/spec/shard.md`

### Milestone M1: Explicit reference HIR checking

#### Depends
- M0

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/cli/check.py
 def add_arguments(parser):
+    parser.add_argument("--reference", metavar="SOURCE")
+    parser.add_argument("--distributed", action="store_true")
 def check_concrete(request):
+    # Pseudocode: bind the same logical inputs/resource to both selected HIRs.
+    reference = evaluate_reference if request.reference is not None else existing_reference
# docs/tutorial/distributed-check.md
+# Check a distributed projection against its reference
# docs/tutorial/distributed-check.ipynb
+# Executable source, extraction command, comparison command and recorded output.
# src/tilefoundry/cli/tutorial.py
+PAGES = (*existing_pages, "distributed-check")
```

##### Accepted by

Extend the existing CLI behavioral suite with explicit HIR selection and the
distributed fixture. This is the public agent workflow; a false numerical pass,
independently drawn weights, or an ambiguous reference selection breaks it.

Validation: both distributed projections pass explicit-reference CLI checks;
an altered reference fails numerically and incompatible weight paths are
refused. The tutorial notebook was executed through the existing renderer,
including source extraction, and its Markdown records the passing output.
All four analysis selectors were also exercised on a two-device deployment;
each refuses the unregistered collective cost evaluator explicitly. Ruff,
specification, comment, reference, language and path checks passed.

- [x] CLI compares reference and distributed HIR using caller-provided predicates.
- [x] JSON reports identify both selected programs and the evaluation mode.
- [x] Mismatched bindings and conflicting reference flags fail before comparison.
- [x] Existing expected-file and runtime-twin checks retain their behavior.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/cli/check.py`
- `src/tilefoundry/cli/tutorial.py`
- `tests/cli/test_cli_check.py`
- `tests/fixtures/distributed/projection.py`
- `docs/spec/cli.md`
- `docs/tutorial/distributed-check.md`
- `docs/tutorial/distributed-check.ipynb`
- `docs/tutorial/index.md`

## Final Gate

<!-- final_gate:start -->
<!-- final_gate:end -->
