# STEP-005 SuperOp retirement design

Date: 2026-08-27

## Goal

Make `execute_steps` and its mesh variants the sole production forward executor.
Retire `SuperOp`, `ForwardBindings`, `LayerProgram`, `run_layer_program`,
`run_layer_program_ep`, architecture-owned SuperOp handlers, and the duplicate
hand loops and selection toggles that support them.

The migration preserves model behavior, kernel selection, graph/capture
optimization, state ownership, and admitted topology. It changes orchestration,
not model algorithms, quantization, kernels, or capability admission.

## Scope

- Inventory every production/default-on SuperOp route and its legacy hand-loop
  oracle.
- Extend the typed `Step` vocabulary only for semantic operations that current
  production families cannot represent.
- Migrate Gemma4 and LFM2 first, review the resulting executor contract, then
  migrate MiniMax, Qwen35 Single/EP, and DeepSeek4 Single/EP.
- Express EP/TP/PP collectives as Step execution semantics derived from the
  admitted `DeviceMesh`.
- Preserve pre-resolved kernel keys and graph/capture as executor backends below
  the Step contract.
- Delete parity switches and duplicate family routes immediately after each
  family passes its cutover gates.
- Delete the shared SuperOp substrate after its final caller is removed.

## Non-goals

- STEP-006 Muse/Glimmer migration or other decoder families absorbed after the
  STEP-004 inventory.
- New PP/TP/EP capability cells, production admission changes, or physical
  topology closure. Those remain in GEN, AXIS, and HW tasks.
- Kernel rewrites, new quant formats, routing-policy changes, or performance
  claims.
- AR/speculative lifecycle unification; SPEC-001 owns that work.
- A generic escape Step, architecture callback Step, integer-opcode Step, or
  compatibility wrapper around SuperOp.
- Flattening model-specific geometry or state policy into the common executor.

## Current problem

The merged branch has two forward orchestration substrates:

1. typed `Step` execution through `execute_steps`, `execute_steps_mesh`,
   `execute_steps_parallel`, and `execute_steps_tp`; and
2. coarse SuperOp programs interpreted through architecture implementations of
   `ForwardBindings` and, for EP, `run_layer_program_ep`.

Production/default-on SuperOp routes remain in Qwen35, DeepSeek4, MiniMax,
LFM2, and Gemma4. Qwen35 and DeepSeek4 EP additionally use the separate EP
SuperOp executor. Gemma4 and LFM2 encode architecture-local operation numbers
inside generic bindings. These routes duplicate ordering, error, capture, and
collective policy and allow an architecture to remain outside the Step spine
while appearing lowered.

STEP-004 remains complete for its historical inventory, but it does not cover
these post-merge routes. STEP-005 is the corrective execution-spine gate.

## Target architecture

Production decoder execution has one direction:

```text
architecture weights and state
        |
        v
borrowed typed Step program
        |
        v
execute_steps / execute_steps_mesh / execute_steps_parallel
        |
        v
dispatch families and pre-resolved kernel backends
        |
        v
HIP launches, collectives, and capture/replay
```

Architecture crates retain model semantics: layer geometry, layer type,
attention policy, routing policy, recurrent state, and the choice and order of
Steps. The executor owns validation of executable programs, sequencing,
dispatch, rank coordination, synchronization, collective placement, and
contextual errors.

## Executor invariants

### One production executor

- No production route calls `run_layer_program` or `run_layer_program_ep`.
- No Step execution failure falls back to a hand loop or SuperOp route.
- `superop.rs` and the EP SuperOp executor are deleted after their final callers.
- A family cutover is incomplete while a production selection toggle or
  duplicate layer loop remains.

### Typed semantics

Each Step identifies a semantic operation with explicit input, output,
dimensions, row count, state effects, and collective behavior where applicable.
Architecture opcodes may not be hidden in `WeightSlot`, integer tags,
`EscapeKind`, callbacks, trait methods, or a generic custom-operation variant.

A fused Step is permitted only when it is a stable executor capability with a
clear semantic contract. It must not mean “run the old block for family X.”
Fusion and pre-resolved kernel keys remain interchangeable backend choices for
the same semantic program.

### Deep architecture ownership

Architecture lowering decides which Steps implement a layer and binds the
model-owned weights and request-owned state. It does not implement a second
interpreter. Model-specific epilogues may remain architecture-owned only when
they execute as typed Steps or as a stable executor-owned fused Step pattern.

### Explicit mesh semantics

Collective intent is carried by a Step or a sealed lowered program. The
executor derives the concrete group from `DeviceMesh`. Architecture handlers do
not directly select peer versus RCCL behavior. Single-rank execution uses the
same semantic program with identity collectives where that is valid.

### Allocation discipline

Decode must not allocate a fresh heap program per token when a program skeleton
can be resolved at load or entry time. Prefer borrowed Step slices, reusable
program storage, or sealed lowered programs whose resource lifetimes are owned
by the model. Request binding supplies only changing tensor/state references and
scalar dimensions.

## Components

### Shared Step layer

`crates/hipfire-dispatch/src/pipeline/steps.rs` remains the semantic execution
owner. Migration work falls into three classes:

1. reuse existing Steps directly;
2. add a typed Step for a genuine semantic gap; or
3. add an executor-owned fused pattern for a stable backend operation.

Expected gaps include:

- Gemma4 hybrid sliding/full attention and interleaved or partial RoPE;
- LFM2 convolution and recurrent-state transitions;
- Qwen35 DeltaNet operations not already Step-backed;
- DeepSeek4 MLA, compressor, and indexer operations; and
- explicit routed-expert collective phases currently hidden by the EP SuperOp
  executor.

The inventory, not this expectation list, determines the final Step additions.
No Step is added until a production caller and its required semantics are named.

### Architecture lowering

Each architecture exposes a narrow lowering boundary that:

- inspects architecture configuration and layer type;
- binds manifest-backed weights and request-owned state;
- produces the ordered typed operation program;
- declares semantic collective intent; and
- rejects unsupported geometry or missing resources before the first GPU
  operation.

Concrete Rust signatures may use borrowed slices, fixed-capacity storage, or
sealed family adapters to satisfy lifetimes and avoid allocations. The logical
contract must not vary by family.

### Executor backends

Existing direct dispatch, pre-resolved kernel selection, graph capture, and
replay remain below the Step boundary. Backend selection cannot change Step
ordering, dimensions, state effects, or collective semantics. Captured kernel
arguments must remain owned for the full replay lifetime.

## Execution flow

### Model construction or execution-entry resolution

1. Resolve manifest-backed weights and state ownership.
2. Validate shapes, dtypes, admitted mesh, collective requirements, and scratch
   capacity.
3. Resolve reusable layer variants, operation skeletons, and kernel capability
   metadata.
4. Reject unsupported programs before publishing an executable model owner.

### Request binding and execution

1. Bind current rows, positions, KV state, recurrent state, and scratch.
2. Materialize a borrowed semantic Step view without avoidable per-token heap
   allocation.
3. Execute through the appropriate common executor.
4. Advance architecture request state only after the defined successful
   execution boundary.
5. Propagate failure through the existing generation error path without retrying
   the partially executed layer through another executor.

## Error contract

Lowering and execution errors remain distinct.

A lowering error names the family, layer, unsupported geometry or missing
resource, and mesh/collective mismatch. It occurs before GPU mutation.

An execution error names the family, layer, Step identity, rank/device, and
underlying dispatch, synchronization, capture, or collective failure. New paths
may not use `unwrap`, silently skip an operation, substitute an identity, or
fall back to legacy execution.

## Migration strategy

Use family-first vertical cutovers. Each increment performs inventory,
implements only necessary Step gaps, proves deterministic parity, flips
production, and removes that family’s migration scaffolding before the next
increment.

### Increment 0: authoritative inventory and fixture readiness

- Enumerate every production/default-on SuperOp caller, selection toggle,
  binding implementation, and duplicate hand loop.
- Classify each route by family and Single/EP/TP/PP mode.
- Record the Step replacement or missing semantic gap for every operation.
- Name a deterministic model, prompt, state oracle, capture route, lifecycle
  fixture, and emulated-mesh fixture where applicable.
- Block a family increment if no numerical or state oracle exists.

### Increment 1: Gemma4

Replace `g4_op` integer opcodes, `Gemma4Bindings`, and the default-on
`run_layer_program` route. Preserve sliding/full attention, layer-specific RoPE,
dense/MoE blocks, and the currently admitted eager E-series and drafter
boundaries.

This increment discovers the typed hybrid-attention vocabulary. It must not
pull STEP-006 vision or unsupported family admission into scope.

### Increment 2: LFM2

Replace `lfm2_op`, `ForwardBindings`, `run_layer_program`, and the
capture-dependent route selection. Preserve convolution state transitions,
dense/MoE distinctions, and the existing sealed `lower_moe_steps` routed-expert
program.

This increment establishes recurrent/conv Step semantics and proves that direct
and captured execution share the same executor.

### Contract gate

Before MiniMax:

- review every new Step for architecture leakage;
- reject generic escape, callback, and integer-opcode mechanisms;
- confirm reusable programs avoid per-token heap allocation;
- confirm lowering and execution errors carry sufficient context;
- confirm single and mesh executors share semantic collective rules; and
- simplify or combine Step additions that encode the same operation under
  different family names.

### Increment 3: MiniMax

Migrate standard attention and layer sequencing around the already Step-backed
sealed MoE program. Remove `MinimaxBindings`, its SuperOp lowering, the selection
toggle, and the duplicate hand loop after parity.

### Increment 4: Qwen35 Single

Migrate hybrid attention and DeltaNet sequencing. Preserve CASK/adaptive-KV,
recurrent and convolution state, mRoPE refusal or support boundaries, and
capture/replay behavior. Remove the Single-device SuperOp path and toggle after
parity.

### Increment 5: Qwen35 EP

Replace `run_layer_program_ep` with per-rank Step programs and explicit
Step-owned EP collectives derived from the admitted mesh. Preserve expert
ownership, shared-expert behavior, rank-local recurrent state, and reset/unload
semantics.

### Increment 6: DeepSeek4 Single and EP

Represent MLA, compressor, indexer, routing, and HC epilogues through typed
Steps or stable executor-owned fused patterns. Replace both Single and EP
SuperOp execution while preserving DSML/tool behavior and MTP-adjacent state
boundaries.

### Increment 7: substrate deletion

After the final family cutover:

- delete `SuperOp`, `SuperOpKind`, `LayerProgram`, `ForwardBindings`,
  `run_layer_program`, and `run_layer_program_ep`;
- delete architecture-local handlers, opcodes, toggles, parity switches, and
  duplicate hand loops;
- remove obsolete modules, exports, comments, examples, and tests that describe
  SuperOp as a production executor; and
- run a final source inventory proving no fallback executor remains.

## Cutover discipline

Each family follows this temporary sequence:

```text
legacy baseline fixture
        |
        v
Step route behind parity-only selection
        |
        v
deterministic numerical/state and capture comparison
        |
        v
production flips to Step
        |
        v
legacy switch, bindings, and duplicate loop are deleted
```

Dual-run scaffolding must use isolated state or perform the architecture’s total
reset contract between runs. It may not execute both routes against shared
mutated state and call the result parity evidence. The parity route is removed
in the same family increment after the production flip succeeds.

## Verification strategy

There is no universal coherence gate. `docs/VALIDATION.md` remains authoritative
for claim-to-route selection.

### Gate 0: inventory and oracle readiness

For each family/mode, record:

- production caller and selection default;
- legacy oracle and Step candidate;
- deterministic artifact and prompt digests;
- path-specific numerical or state oracle;
- capture/replay route;
- reset, abort, unload, and reload fixture; and
- emulated topology fixture where applicable.

`serve_harness.py` is a user-facing semantic smoke, not a substitute for a
forward numerical/state oracle.

### Per-family structural checks

- Host-side lowering tests cover every admitted layer variant.
- Step ordering, shapes, rows, dtypes, KV positions, recurrent transitions, and
  collective anchors are explicit.
- Unsupported geometry and missing resources fail before GPU mutation.
- No architecture opcode, callback, or fallback is present.

### Per-family numerical and state parity

Use the same model, prompt, seed, temperature, initial state, and environment for
legacy and Step paths. Prefer first-divergence-sensitive evidence: per-layer
state/logit hashes or bounded numerical deltas, followed by committed-token
parity. Recurrent families require multi-token and reset/reuse evidence; token
identity alone is insufficient.

### Capture and graph parity

Direct execution and the applicable captured/replay route must produce the same
observable result. Verify retained kernel-argument ownership and prove that
capture-disabled execution does not select a legacy executor.

### Lifecycle evidence

Exercise normal completion, abort, reset, unload, and reload. Require no
cross-request state bleed and no monotonic VRAM growth after the first warm
lifecycle. COR-002 and COR-003 remain the reset and finalization authorities.

### User-facing smoke

After numerical/state parity, run `scripts/serve_harness.py` with the exact
family model and settings. Read the decoded output. Harness counters alone do
not establish coherence.

### Mesh evidence

Run emulated EP/PP/TP parity where the family supports it. Emulation proves
orchestration and ownership only. STEP-005 does not close or weaken the physical
`HW-*` gates.

### Family emphasis

- Gemma4: sliding/full attention state, RoPE variants, dense/MoE ordering, and
  prefill/decode consistency.
- LFM2: convolution history, recurrent reset, dense/MoE behavior, and capture.
- MiniMax: attention plus sealed MoE integration, multi-turn reset, and capture.
- Qwen35 Single: hybrid attention/DeltaNet state, CASK/adaptive-KV, and replay.
- Qwen35 EP: per-rank state, collective placement, emulated parity, reset, and
  unload.
- DeepSeek4: MLA/compressor/indexer state, DSML/tool behavior, Single/EP parity,
  and MTP-adjacent paths.

## Final acceptance

STEP-005 completes only when:

- every family and mode named by the tracker executes production forward through
  typed Steps;
- no production or fallback reference remains to `run_layer_program`,
  `run_layer_program_ep`, `ForwardBindings`, `LayerProgram`, `SuperOpKind`, or
  architecture-owned SuperOp handlers;
- collectives are explicit Step semantics owned by the common executor;
- graph/capture and pre-resolved kernel optimization remain backend details;
- every family has recorded deterministic parity, lifecycle, applicable
  emulated-mesh, and user-facing smoke evidence;
- obsolete toggles, parity switches, duplicate loops, modules, exports, and
  comments are deleted;
- applicable focused tests, changed-file rustfmt, narrow clippy, `cargo build`,
  and `cargo test` pass under the project’s Rust rules;
- `graphify update .` has refreshed the repository graph after implementation;
  and
- the authoritative tracker and PR checklist are updated from observed evidence,
  with physical support still delegated to the named `HW-*` tasks.

## Planning structure

This design is implemented through one master plan with separately reviewable,
gated increments. The plan must preserve the order above, but each family
increment receives its own concrete file/symbol inventory, parity oracle, and
verification commands. Later family steps may be refined after the contract
gate; the invariants and final acceptance criteria in this design do not change.
