---
type: FEAT
component: engine-analysis
target_repo: tilefoundry
---

# [FEAT][engine-analysis] Price distributed HIR under deployment and workload constraints

## Description

Continue the engine-scope RFC with compositional distributed resource analysis,
portable deployment/workload profiles and executable strategy comparison.
Each milestone ends in a tested commit. Existing local analysis meanings stay
unchanged; the new `engine` analysis reads the same checked, inlined HIR.

### Current state

- `src/tilefoundry/analysis/api.py` checks and inlines HIR, resolves analyzer dependencies, and serializes owned metadata.
- `src/tilefoundry/visitor_registry/op_cost.py` prices primitive work and operand traffic from typed shapes, independently of kernel names.
- `src/tilefoundry/analysis/liveness.py` establishes structured definition/use order; engine deployment ownership is not represented by its intervals.
- `src/tilefoundry/target/cuda/facts.py` supplies per-device rates and HBM capacity, but no deployment links or runtime reserve.
- `src/tilefoundry/ir/hir/sharding/alltoall.py` states bounded routing and metadata semantics while deliberately leaving cost evaluators unregistered.
- `src/tilefoundry/cli/analyze.py` exposes local analysis families with no engine profile input.

### Decisions

- D1 Public entry -- add an `engine` selector under analyze, not a new command. Device timing has a separate record/model identity so local performance retains its meaning.
- D2 Facts -- use target-provided per-GPU rates/capacity; an immutable deployment profile adds per-rank capacity/reserve and directed effective routes with bandwidth, startup latency and shared resource names. No implicit link bandwidth is invented.
- D3 Workload -- explicit useful work per invocation, work unit, latency budget and persistent-state parameter names. Throughput never multiplies work by replica count implicitly.
- D4 Primitive reuse -- project supported device-level HIR onto local logical tensor shapes and call the existing CostEvaluator. Communication gets explicit operation models rather than a cost function for each composed strategy.
- D5 Network schedule -- ring reduction/gather and pairwise all-to-all use explicit peer transfers. A deterministic schedule serializes operations using a rank and transfers sharing link resources; unrelated ranks/resources may overlap. Timing includes declared compute, HBM and communication services.
- D6 Memory -- parameters remain live through the invocation, with weights and named state reported separately. Known views alias backing buffers; other results are out of place. Communication reserves one payload-sized scratch buffer, and loop carries use bounded spare storage. Report the declared buffer model and limiting live allocations.
- D7 Routed traffic -- absent a route profile, use explicit upper bounds and flag whether capacity covers arbitrary routes. Optional per-operation peer-token, peer-route and per-expert counts provide stated traffic assumptions; distinguish unique transported tokens from expanded expert rows and padded allocation.
- D8 Unknowns -- missing rates, paths, capacities or insufficient routing bounds remain explicit unknowns/diagnostics. They never become zero-cost transfers or a feasible SLO conclusion.
- D9 Scope -- start with concrete single-level device meshes and uniform structured loops. Unsupported geometry fails with provenance. Loop work is aggregated rather than unrolled per token; local storage constraints remain the downstream local workflow's responsibility.
- D10 Examples -- parameter-sharded examples expose resident ownership at the function boundary. An unplaced input is replicated; selecting a view with Reshard cannot free its backing allocation.
- D11 Partial updates -- the out-of-place allocation policy charges initialization of preserved backing bytes for partially written results; otherwise cache-update traffic would implicitly assume in-place reuse while capacity models a separate result.

## Milestones

### Milestone M2: Portable engine inputs and records

#### Depends
- None

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/analysis/engine_profile.py
+class DeviceBudget: ...
+class NetworkRoute: ...
+class EngineDeployment: ...
+class EngineWorkload: ...
+class RoutingProfile: ...
+class EngineOptions: ...
+def load_engine_options(path): ...
# src/tilefoundry/analysis/engine_metadata.py
+class EngineMetadata(IRMetadata): ...
+class EngineRank: ...
+class EngineOperation: ...
```

##### Accepted by

New profile tests validate public JSON/Python inputs and reject malformed
deployment/routing assumptions that could otherwise produce false feasibility.
They cover the portable boundary used by later analysis and CLI workflows.

- [x] Deployment and workload inputs are immutable, serializable and unit-explicit.
- [x] Missing facts remain optional rather than acquiring numerical defaults.
- [x] Invalid ranks, routes, counts and budgets are rejected before analysis.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/analysis/engine_profile.py`
- `src/tilefoundry/analysis/engine_metadata.py`
- `tests/analysis/test_engine_profile.py`
- `docs/spec/architecture.md`
- `docs/spec/analysis.md`
- `docs/spec/target.md`

### Milestone M3: Compositional device resources and timing

#### Depends
- M2

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/analysis/engine.py
+def analyze_engine(function, context): ...
# src/tilefoundry/analysis/engine_communication.py
+def communication_plan(call, profile, ranks): ...
# src/tilefoundry/analysis/registry.py
+if selector == "engine": ...
# src/tilefoundry/analysis/report.py
+declare_record(EngineMetadata, family="engine")
# src/tilefoundry/inspection/analysis_report.py
+# Render the same engine findings in human and machine-readable reports.
```

##### Accepted by

New public analyze tests use independently calculable projections, reductions
and routed programs. Altering capacity, reserve, link rate or ownership changes
the corresponding result. Existing local analysis suites remain unchanged.

Validation: 25 profile/engine cases pass, including exact projection work,
544-byte per-rank peak, a 1044 ns synthetic timeline, shared-link contention,
unknown facts, sparse reads of resident weights, functional state-copy traffic,
routed-token deduplication, and a compact million-iteration loop. A broader
analysis/evaluator/operation/CLI run passed 493 tests with one existing GPU-only
skip and one existing CUDA-default test excluded. Reports retain both supplied
profiles and consumed target rates for reproduction.

- [x] Primitive compute/traffic compose over checked HIR without kernel-specific registration.
- [x] Per-rank resident allocation and peak storage distinguish weights, state and views.
- [x] Communication reflects group membership, payloads and shared resources.
- [x] Uniform loops are compact and dependent resources determine elapsed time.
- [x] Unknown facts and routing assumptions are visible in feasibility and timing.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/analysis/engine.py`
- `src/tilefoundry/analysis/engine_communication.py`
- `src/tilefoundry/analysis/engine_geometry.py`
- `src/tilefoundry/analysis/engine_metadata.py`
- `src/tilefoundry/analysis/registry.py`
- `src/tilefoundry/analysis/report.py`
- `src/tilefoundry/analysis/check.py`
- `src/tilefoundry/inspection/analysis_report.py`
- `src/tilefoundry/inspection/values.py`
- `tests/analysis/test_engine_analysis.py`
- `tests/fixtures/distributed/engine.py`
- `docs/spec/analysis.md`
- `docs/spec/inspection.md`
- `docs/spec/hir.md`

### Milestone M4: CLI profiles and strategy comparison example

#### Depends
- M3

#### Target State Design

##### Delivered
```diff
# src/tilefoundry/cli/analyze.py
+EVIDENCE["engine"] = "per-rank memory, communication and workload throughput"
+def run_authored_analysis(..., engine_profile=None): ...
# src/tilefoundry/cli/__init__.py
+analyze.add_argument("--engine-profile", metavar="PATH")
# docs/tutorial/engine-analysis.ipynb
+# Execute reference checking and price alternative placements under an external profile.
# docs/tutorial/engine-analysis.md
+# Render the executable strategy comparison.
```

##### Accepted by

Public CLI checks and an executed notebook compare feasible candidates under
fixed workload/hardware inputs. Profile/schema failures are exercised through
the command, and source/metadata provenance remains reviewable.

- [ ] analyze --engine accepts a portable deployment/workload profile.
- [ ] Text and JSON expose the same capacity, timing and throughput assumptions.
- [ ] A reusable example checks candidates and selects throughput under a supplied SLO.
- [ ] The roadmap records remaining backend/handoff and strategy-coverage work accurately.

<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
- [ ] A milestone that changes a public contract MUST list the owning `docs/spec/*.md` path in its `#### Related Files`; one that changes none lists no spec path. <!-- policy_ac: spec_impact-0 -->
<!-- policy_ac:end -->

#### Related Files
- `src/tilefoundry/cli/analyze.py`
- `src/tilefoundry/cli/__init__.py`
- `src/tilefoundry/cli/tutorial.py`
- `tests/cli/test_cli_engine.py`
- `tests/fixtures/distributed/engine.py`
- `docs/tutorial/engine-analysis.ipynb`
- `docs/tutorial/engine-analysis.md`
- `docs/tutorial/index.md`
- `docs/rfcs/engine-scope-optimization.md`
- `docs/spec/cli.md`

## Final Gate

<!-- final_gate:start -->
<!-- final_gate:end -->
