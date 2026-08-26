# Redline PM4 as the Default Dispatch Transport — Implementation Plan

> **For agentic workers:** execute this plan task-by-task. Every promotion gate
> requires the route-specific runtime evidence in `docs/VALIDATION.md`; compile
> success is not acceptance.

**Status:** proposed  
**Lifecycle:** `active` implementation plan  
**Goal:** Make Redline PM4 the default kernel-launch transport for current and
future models that use hipfire's common typed launch API, without model- or
quant-specific PM4 lowering. Retain HIP as the correctness oracle and fallback.

## 1. Problem statement

Today three concerns are too easy to couple:

1. model lowering chooses the operator graph;
2. kernel dispatch chooses the compiled kernel for an op/quant/architecture;
3. the launch transport submits that already-selected kernel through HIP or
   Redline PM4.

Redline should own only the third concern. A PM4 backend should not know that a
kernel implements MQ4, FP16 attention, a recurrent model, or a future quant
format. If architecture code uses the common typed launcher and all referenced
allocations are registered, the same launch should work through HIP and PM4.

New models and quant formats will still need operator lowering and kernels. They
must not need bespoke Redline packet code, per-model PM4 tapes, or quant-type
allowlists.

## 2. Non-goals

- Do not turn PM4 into a kernel compiler.
- Do not remove HIP or weaken HIP parity coverage.
- Do not promote an architecture before multi-position output parity.
- Do not special-case a quant ID in the Redline transport.
- Do not make environment variables the permanent product selection surface.
- Do not require future architecture crates to understand PM4 registers,
  residency lists, queue packets, or fences.

## 3. Target architecture

### 3.1 One transport-neutral launch descriptor

Every launch MUST lower to one owned descriptor before either backend sees it:

```rust
struct KernelLaunch {
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_lds_bytes: u32,
    kernargs: OwnedKernargBlob,
    allocations: SmallAllocationSet,
    dependencies: SmallDependencySet,
    completion: Option<CompletionSignal>,
}
```

Names are illustrative; the implementation MUST reuse or cleanly replace the
existing dispatch descriptor rather than introduce a parallel launch model.

Required invariants:

- kernargs own their bytes for the entire asynchronous launch lifetime;
- buffer references resolve to registered allocation handles, not raw
  caller-assembled BO lists;
- geometry and LDS validation happen before backend selection;
- kernel identity resolves once at load/registration, not by string in the
  per-token hot path;
- graph capture, direct launch, and PM4 tape capture consume the same descriptor.

### 3.2 One dispatch backend boundary

HIP and Redline implement the same narrow interface:

```rust
trait DispatchBackend {
    fn submit(&self, launch: &KernelLaunch) -> Result<Submission>;
    fn submit_batch(&self, launches: &[KernelLaunch]) -> Result<Submission>;
    fn wait(&self, submission: &Submission) -> Result<()>;
}
```

The exact trait MAY differ to match current code, but architecture/model code
MUST NOT branch on HIP versus PM4.

### 3.3 Automatic PM4 metadata

At kernel registration, Redline MUST derive the launch metadata from the loaded
code object or existing ROCR executable metadata:

- code entry address;
- `COMPUTE_PGM_RSRC1/2/3`;
- kernel code properties and user-SGPR layout;
- kernarg size/alignment;
- static LDS;
- wave mode and architecture capability.

Eligibility is metadata/capability based. It MUST NOT be an op-name, model-name,
or quant-type allowlist.

### 3.4 Automatic residency

The GPU allocator registry MUST map every launch allocation to its HIP/ROCR/KMD
handle. Redline derives and caches the residency/BO list from the descriptor.
Callers MUST NOT pass `extra_bos` manually.

A launch with an unregistered raw pointer fails closed to the HIP backend and
emits a precise diagnostic naming the argument/allocation that prevented PM4.

### 3.5 Default policy

After an architecture class passes its acceptance gate:

- PM4 is the default transport for eligible launches;
- HIP remains selectable through typed developer configuration;
- unsupported launch features fall back to HIP with a reason code;
- no silent PM4 failure may fall through to a different kernel;
- production logs expose selected transport and fallback counts.

## 4. Performance plan

The primary opportunity is submission amortization, not hashing.

### 4.1 Remove the false allocation-free path

`FastDispatch::dispatch` currently creates a heap-backed `CommandBuffer` even
though the path is documented as allocation-free. Replace that temporary vector
with direct bounded packet writes into the persistently mapped IB.

Acceptance:

- zero heap allocations per steady-state launch;
- packet bytes match the existing encoder byte-for-byte;
- overflow fails before writing beyond the mapped IB;
- no new crate is required unless a fixed array proves insufficient.

### 4.2 Kernarg ring

Replace the single shared kernarg staging slot with a bounded ring:

- alignment derived from kernel metadata;
- one slot remains live until its completion signal retires;
- wrap waits only on the slot being reused;
- graph/tape replay patches pointers/scalars in place.

### 4.3 PM4 templates

Cache the invariant packet prefix per registered kernel. At launch, patch only:

- code/descriptor values that are genuinely dynamic;
- grid geometry;
- kernarg address;
- dynamic LDS;
- completion/fence values.

### 4.4 Batch submission

Add `submit_batch` so a layer or decode step can use one IB and one queue submit.
Preserve required barriers and dependencies, but eliminate per-kernel ioctl and
host synchronization.

Promotion order:

1. packet template parity;
2. multiple independent launches in one IB;
3. one layer tape;
4. one complete decode-step tape;
5. captured forward/verify tapes where state contracts permit.

A hasher change is explicitly out of scope until profiling shows name/map lookup
inside the steady-state launch path. Current visible Redline kernel lookup is a
linear setup-time scan; PM4 submission already receives a resolved kernel.

## 5. Implementation phases

### Phase A — launch inventory and contract freeze

- [ ] Enumerate direct HIP launch callsites with CodeGraph and LSP references.
- [ ] Identify the current closest typed descriptor and backend interfaces.
- [ ] Classify every launch field as invariant, per-load, per-request, or
      per-token.
- [ ] Record unsupported features: dynamic LDS, multi-stream ordering, peer
      memory, graph capture, host callbacks, cooperative launch, and external
      synchronization.
- [ ] Add a repository gate that rejects new architecture/model direct raw HIP
      launches outside the backend/bridge implementation.

**Gate:** no new launch abstraction beside an existing equivalent; ownership and
lifetime invariants have focused tests.

### Phase B — common descriptor cutover

- [ ] Move direct and graph-aware helpers onto the common owned descriptor.
- [ ] Migrate every existing architecture caller as a clean cutover.
- [ ] Remove obsolete raw-launch helpers and aliases.
- [ ] Add diagnostics for unregistered pointers and malformed kernargs.

**Gate:** user-facing HIP generation is byte-identical before/after; graph
capture retains valid owned kernargs.

### Phase C — automatic metadata and residency

- [ ] Parse/register PM4 metadata once at kernel load.
- [ ] Cache resolved kernel handles; remove steady-state string lookup.
- [ ] Connect allocator registrations to residency handles.
- [ ] Derive cached BO lists from launch allocation sets.
- [ ] Add capability reasons for every HIP fallback.

**Gate:** an arbitrary registered kernel with supported ABI dispatches through
PM4 without op/model/quant code changes.

### Phase D — allocation-free single launch

- [ ] Write PM4 directly into persistent mapped IB memory.
- [ ] Add bounded packet writer and exact encoder parity tests.
- [ ] Add kernarg ring and completion retirement.
- [ ] Remove per-launch `Vec`, memcpy, and BO-list construction.

**Gate:** zero steady-state host allocations; raw HIP/PM4 output parity on the
same kernel fixtures; device-loss-free repeated dispatch.

### Phase E — batched IB and tape replay

- [ ] Submit multiple launch descriptors in one IB.
- [ ] Insert only required release/wait/cache barriers.
- [ ] Patch a prebuilt per-kernel template.
- [ ] Build layer and decode-step tapes from the same descriptors.
- [ ] Preserve cancellation, quiescence, and state-lifecycle boundaries.

**Gate:** fewer submits/synchronizations with no output delta; record host timing
and JSON Redline harness evidence.

### Phase F — PM4 default promotion

Promote one architecture class at a time. A class is eligible only after:

- [ ] stable capture through `scripts/redline_daemon_harness.py`;
- [ ] valid AQL/PM4 contracts and quiescence evidence;
- [ ] multi-position HIP/PM4 output parity;
- [ ] repeated user-facing decoded output inspection;
- [ ] graph/replay and cancellation smoke;
- [ ] no regression under the exact model/quant fixtures claimed;
- [ ] typed config and diagnostics documented.

Raw HIP remains the fallback/oracle after promotion.

## 6. Future-model contract

A new model automatically inherits PM4 when it:

1. selects registered kernels through the normal dispatch registry;
2. allocates GPU state through registered allocator APIs;
3. launches only through the common descriptor;
4. declares dependencies through the descriptor/tape contract;
5. does not request a backend capability Redline lacks.

No future model checklist may contain “write PM4 lowering for this model.” If it
does, the abstraction boundary is still wrong.

## 7. Correctness and failure policy

- PM4 never silently substitutes a kernel or changes launch geometry.
- A metadata/residency/capability failure falls back before submission.
- A submission failure after GPU-visible state mutation is terminal unless the
  backend proves rollback/quiescence.
- HIP/PM4 dual-run is a certification/debug route, not a production double-run.
- Stateful recurrent/KV work requires explicit state transaction boundaries;
  transport promotion does not waive them.
- Device loss, garbage decoded output, or parity drift blocks promotion even if
  throughput improves.

## 8. Success criteria

The plan is complete when:

- architecture/model crates contain no transport-specific PM4 logic;
- adding a new quant kernel requires no Redline code;
- eligible launches use PM4 by default on promoted architectures;
- HIP fallback reasons are typed and observable;
- steady-state single launch allocates nothing;
- decode-step batching reduces submits/synchronizations measurably;
- runtime validation proves coherent HIP/PM4 parity on each promoted class.

## 9. Coordination

Open multi-slot PR #609 changes the concurrent serve/runtime path and should be
integrated rather than duplicated. PR #626 is superseded; PR #636 is the
separate VL delta. The dispatch transport work should keep its model-facing API
stable enough that those branches consume the same launch descriptor after
rebase.
