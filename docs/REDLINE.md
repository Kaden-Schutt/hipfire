# Redline Contributor Guide

Canonical **shipped / ref-pinned** procedure for Redline retained replay in
hipfire. Use it to construct, validate, measure, review, and decide whether a
retained route may be promoted. Runtime source is authoritative for executable
behavior. Dated reports are evidence for the fixtures they name only.

| Field | Value |
|---|---|
| Page state | **shipped / ref-pinned** (see [`INDEX.md`](INDEX.md)) |
| Inventory date | 2026-07-22 |
| Audited source ref | `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Comparison base | `origin/beta` @ `202282de8759dfa6963ea5184ad2bf2b9259cef6` |
| Validation routes | [`VALIDATION.md`](VALIDATION.md) |
| Thin skill hook (non-normative) | [`.agents/skills/redline-retained-replay/SKILL.md`](../.agents/skills/redline-retained-replay/SKILL.md) |

This guide is normative contributor policy on the audited beta ref.

## 1. Scope, authority, and evidence classes

Use this guide when changing any of:

- model admission to retained AQL or retained PM4;
- recorder coverage, launch identity, kernarg capture, artifacts, resource
  effects, or dynamic bindings;
- retained-plan construction, PM4 lowering, queue policy, waits, fences, or
  acquire policy;
- model reset, pointer lifetime, replay failure, or fallback behavior;
- a kernel, fusion, Radiowave transformation, or scheduling overlay on a
  retained route;
- a benchmark or product claim attributed to Redline.

**In scope shape:** ordinary sequential single-token autoregressive
continuation.

**Out of scope unless separately certified:** prefill; speculative
verification; MTP reseed or proposal; mutable multi-token replay; a new model
path merely because it can be captured; the experimental direct-KMD
`crates/redline` crate as a serving transport.

### Source-of-truth order

1. Runtime source and tests — executable admission, state transitions, failure
   handling, encodings.
2. This guide — contributor workflow and certification policy before promotion
   or a Redline-attributed performance claim.
3. Dated performance checkpoints and raw reports — observations for exact
   fixtures.
4. `crates/redline-dispatch/HIPFIRE-GRAFT.md` and
   `crates/redline-rocr/PROVENANCE.md` — graft/ABI provenance only; not current
   procedure.

### Evidence classes

| Class | Meaning | Examples |
|---|---|---|
| **Runtime fact** | Defined by current source or tests. | Automatic default predicate; `ReplayState` transitions; replay failure returns an error. |
| **Certification policy** | Evidence this guide requires before promotion. May be stricter than runtime enforcement. | Multi-position shadow parity; positive timed-arm route proof; matched stationary performance. |
| **Dated evidence** | Observation tied to date, source/binary/model identity, and fixture. | A 2026-07-11 gfx1201 PM4-versus-HipGraph measurement. |

`ReplayState::Ready` is a runtime state. It is **not** repository certification.

### Capability vs admission (do not collapse)

| Classification | Meaning |
|---|---|
| **Implementation capability** | Code can lower/prepare/replay something when explicitly driven. |
| **Opt-in availability** | An explicit backend/manual path can exercise that capability. |
| **Model performance evidence** | Dated tok/s (or similar) on a fixture; route may be unspecified. |
| **Retained-route certification** | Full Section 7 ladder, including timed-arm route proof. |
| **Automatic product default** | Runtime predicate requests Auto without explicit backend/manual bypass. |
| **Registry admission** | Row in [`admissions.yml`](admissions.yml). Schema v2 holds exactly one evidence-bound record (LFM2.5-350M MQ4 gfx1201 retained-PM4); every other route stays unadmitted. |

Runtime admission ≠ guide certification ≠ product admission.

### Terms

| Term | Definition |
|---|---|
| **Recorder-aware HIP launch** | Ordinary HIP launch that also contributes exact typed launch metadata to the retained tape. Must not alter direct-HIP semantics. |
| **Retained tape** | Stable ordered launch sequence plus exact artifacts, padded kernargs, geometry, resource contracts, dependencies, and dynamic bindings. |
| **PM4 command body** | Architecture-specific indirect-buffer command stream lowered from a certified tape. |
| **PM4-IB AQL packet** | Public ROCr/HSA vendor packet that submits a retained PM4 command body. |
| **Runtime admission** | Current code deciding whether a forward may capture, prepare, or consume replay. |
| **Certification gate** | Evidence this guide requires before promotion or a Redline-attributed claim. |
| **External adapter boundary** | Work kept outside the retained body and explicitly counted (e.g. current Qwen token embedding and position preparation). |

## 2. Mental model and ownership

Active retained-PM4 path:

```text
ordinary recorder-aware HIP launches
    │  preserve the ordinary HIP call and copy exact typed launch metadata
    ▼
stable retained tape
    │  validate artifacts, ABI, geometry, effects, dependencies, bindings
    ▼
architecture-specific PM4 lowering
    │  gfx10/gfx11 and gfx12 use distinct register and acquire contracts
    ▼
retained public-HSA command memory
    │
    ▼
one PM4-IB AQL packet
    │
    ▼
public ROCr/HSA queue → release-ordered doorbell → signal → completion
```

Redline retains the real model forward. It does not inherently replace or fuse
kernels. A kernel change is a separate optimization that must be re-certified.

### Three-crate ownership

| Crate | Owns | Does not own |
|---|---|---|
| `redline` | Experimental direct-KMD/bare-libdrm device, memory, queue, PM4, sync. | Active product serving route. |
| `redline-dispatch` | Dispatch-DAG recording/validation, artifact/kernarg identity, dependency/visibility policy, plan compilation/selection, retained AQL/PM4 graph construction. | ROCr ABI lifetime mechanics or model-specific admission. |
| `redline-rocr` | Dynamically loaded public ROCr/HSA ABI; agent, queue, memory, packet, signal, doorbell, completion lifetimes; AQL encoding; architecture PM4 builders. | Model scheduling, dispatch-DAG policy, or backend admission. |

`rdna-compute::replay::ReplayController` is the product integration controller
held by `Gpu`. The model adapter owns the eligible forward boundary and
model-owned state/lifetimes; the controller records, prepares, routes, poisons,
and resets. That role does **not** make `rdna-compute` a fourth Redline
transport crate.

### Mechanisms that must remain distinct

| Mechanism | Submission shape | Proves | Does not prove |
|---|---|---|---|
| Ordinary serial HIP | One `hipModuleLaunchKernel` path per launch | Baseline model execution | Any retained route |
| HipGraph | HIP stream capture → graph launch | HIP graph capture/replay for an eligible forward | Retained AQL or PM4 |
| Per-dispatch retained AQL | Many public-HSA kernel-dispatch AQL packets | AQL preparation/replay when positively routed | Single retained PM4 submission |
| Retained PM4-IB | One vendor AQL packet → retained PM4 IB | Retained PM4 only when prep + observed replay are proven | Direct-KMD dispatch |
| Experimental `crates/redline` | Direct DRM outside HIP/ROCr serving | Direct-KMD experimentation | Current serving transport |
| Launch fusion | Fewer/different kernels under any host path | Changed device graph | Redline, route selection, or wall-time win |
| Stable partial recorder fingerprint | Repeatable recorded subsequence | Discovery progress | Complete tape or installed route |

## 3. Runtime lifecycle and the model boundary

### Model load and automatic default

After a successful model load, the daemon may request Auto via
`ReplayController::configure_model_default` when the MQ4R runtime-default
predicate matches. **Current automatic runtime selection is only**
`mq4r_redline_default` for single-GPU `.mq4r` on the exact GPU-arch allowlist.
Existing LFM `.mq4` registry evidence is not auto-selected because it is not
`.mq4r` (not because LFM is categorically model-family exempt).

#### MQ4R runtime default

`mq4r_redline_default` — source:
`crates/hipfire-runtime/src/config.rs`.

```text
exact GPU arch is gfx1100 OR gfx1151 OR gfx1201
AND pipeline parallelism == 1
AND tensor parallelism == 1
AND model extension is .mq4r, case-insensitive
```

Model-family agnostic: there is no `arch_id` gate. This is a **runtime**
automatic product default only: when the predicate matches and no explicit
backend/manual bypass wins, load-time `configure_model_default` requests Auto
(retained PM4 when transport is unset). It is **not** a registry admission,
**not** Section 7 certification, and **not** a claim that every model/arch
pair has a sealed golden or `admissions.yml` row. Unsupported retained routes
still fail closed. `gfx1200` and every other GPU arch remain explicit opt-in
only.

Disable the automatic default by selecting the built-in `hip` profile in the
config wizard, choosing an explicit backend there, or setting
`HIPFIRE_REPLAY_BACKEND=hip`. That opt-out does **not** relax certification or
admission requirements for any later promotion claim.

#### LFM registry evidence (not a runtime default)

Registry owner: [`admissions.yml`](admissions.yml) row
`lfm25-350m-mq4-gfx1201-retained-pm4-plain-ar` (schema v2). That sealed row is
product-route / registry **evidence and admission metadata only**. It does
**not** wire automatic runtime selection, does **not** name a current source
predicate, and does **not** opt the sealed `.mq4` fixture into
`configure_model_default`. The sealed LFM artifact is **not** automatically
selected because it is `.mq4`, not `.mq4r`; model family itself no longer
blocks `.mq4r`. Any usable non-default retained route is **explicit opt-in**
and must still prove route support. The prior LFM screen is
relaxed-stationarity evidence until a separate 1% stationary certification
run completes (see §11.5).

Exact sealed evidence scope for that row: sealed same immutable snapshot
bytes (identity MD5 `cb5284b8ad5c6f9e4ca859c0aff0bcd0`, length 229,474,032);
gpu `gfx1201`; `arch_id` 11; weight quant MQ4; KV Q8; `max_seq` 2048; plain AR;
`pp=tp=1`; transport PM4; retained plan 161 dispatches / 15 kernels / sequence
hash `04305cf1254244f6`.

#### Shared bypass and residual HIP

An explicit `HIPFIRE_REPLAY_BACKEND` selection, config wizard / profile
backend selection (including the built-in `hip` opt-out), or enabled
`HIPFIRE_REPLAY_MANUAL_CAPTURE` bypasses the MQ4R runtime default.
`HIPFIRE_REPLAY_TRANSPORT` does **not**: it changes only the
transport. Therefore an eligible automatic-default `.mq4r` model still
requests `Auto` when only the transport is explicit (using that transport);
with transport unset, it uses `Pm4Ib`. When `mq4r_redline_default` is false
and no backend/manual selection applies, the backend remains ordinary HIP.

Explicit opt-in can exercise broader implementation capability (including LFM
retained routes when the implementation supports them). Opt-in availability is
not certification. Wizard/backend disable of a runtime default is also not
certification and does not create or remove a registry admission.

Model reset clears recorded launches, certified observations, prepared AQL/PM4
objects, and the fallback reason, then sets `forward_eligible = true` before
admission restarts. Allocation identity is part of the replay contract.

### Eligible forward boundary

Only an ordinary sequential single-token continuation may arm, record, or
consume the plain-AR tape. The model adapter must set a one-forward eligibility
decision before the forward body.

Must neither contaminate nor consume the plain-AR tape:

- model load and initialization;
- prefill;
- speculative reseed, proposal, or verification;
- MTP reseed, proposal, or verification;
- a graph-capture or batched path with a different mutable-state contract;
- model swap or allocation reconstruction;
- any other non-sequential call.

In the current Qwen adapter, token embedding and the position-buffer H2D update
stay outside the retained layer-stack body. That boundary is valid only because
it is explicit and counted.

HipGraph and Redline share the plain-AR eligibility decision but are separate
backends. While Redline is enabled and not in sticky fallback, the Qwen HipGraph
path is mutually excluded. After preparation failure poisons Redline, later
eligible forwards may use HIP-side policy (which can include HipGraph).

### Automatic product lifecycle

```text
successful eligible model load/default or explicit Auto
    ▼
Armed
    │ begin_auto_capture_if_armed on the first eligible continuation
    ▼
RecordingWarmup
    │ ordinary HIP layer stack + record exact launches
    │ synchronize, then finish_capture
    ▼
Captured
    │ prepare_linear_aql or prepare_pm4_prefix
    ├─ success ───────────────────────────────────────────────┐
    ▼                                                        │
Ready ◄───────────────────────────────────────────────────────┘
    │ eligible continuation + matching transport
    ▼
observed retained AQL or PM4 replay
```

The automatic MQ4R product path goes **directly** from `Captured` to `Ready`
when preparation succeeds. It does **not** traverse `ShadowValidated`, call
`observe_shadow`, or call `install_prepared_plan`.

### Manual controller and certification lifecycle

The manual controller API can collect `ShadowValidation` observations. Two
accepted observations can move it to `ShadowValidated`; `install_prepared_plan`
then moves a non-Shadow request to `Ready`. Manual capture alone ends at
`Captured` and changes no launch route.

Repository certification is stricter than either automatic transition:

```text
stable complete capture
  → artifact/ABI/guard validation
  → multi-position HIP vs PM4 vs HIP-kernarg-blob parity
  → successful preparation
  → Ready plus observed replay at multiple positions
  → production serve validation
  → matched stationary performance
  → long-context and reset/model-swap validation
  → promotion
```

The controller’s internal `1.03` shadow field is not a durable contributor
promotion threshold and is not used by the automatic MQ4R admission path. Each
campaign must state its promotion rule in advance.

### Exact fail-closed semantics

| Failure phase | What has already happened | Current forward | Later forwards | Required wording |
|---|---|---|---|---|
| Preparation after automatic warmup | Eligible ordinary-HIP warmup completed and produced output | Success for that completed HIP forward; controller poisoned | Sticky `Fallback`; retained routing disabled; HIP-side policy may include HipGraph | “Preparation failed after a successful HIP warmup; later forwards fall back.” |
| Retained replay execution | Prepared AQL/PM4 selected; execution errored | Controller poisoned; forward returns error. **No same-forward HIP retry** | Sticky `Fallback` | “Replay failed; this forward errored; later forwards fall back.” |
| Manual shadow observation | Observation failed parity, ABI, timing, or configured gate | No product-route promise | Sticky fallback | “Manual shadow validation failed.” |
| Recorder capacity or capture contract | Capture could not remain valid | Fail closed; no partial route install | Sticky fallback until reset | Name the exact capture failure |
| Successful model load/reconfiguration | New model/allocation identity | Old retained objects not reusable | Tape, prepared queues, command buffers, fallback state reset | “Model reconfiguration invalidated the old plan.” |

Never describe a replay-execution error as “the same token retried through HIP.”
That behavior does not exist.

## 4. Retained tape, ABI, lifetime, and dynamic-state contract

A retained route is valid only when all of the following hold.

### Launch and kernarg identity

- Recorder copies exact naturally aligned, tail-padded kernarg bytes used by
  HIP. Never retain a pointer to a caller stack frame or temporary argument
  array.
- Loaded HSA code object that owns `{kernel}.kd` must match the captured launch
  symbol and loader metadata.
- Runtime-specialized launch names and aliases must resolve to the exact owning
  artifact. Absent or ambiguous alias blocks preparation.
- Unsupported scratch, implicit SGPR, loader-kernarg, workgroup, grid, shared
  memory, or symbol metadata blocks installation before `Ready`.
- Artifact path, artifact digest, symbol, loader kernarg size/alignment, padded
  blob, grid, block, and shared memory are one identity contract. A stable
  kernel-name hash alone is insufficient.

### Geometry and dynamic bindings

- Capture-time geometry is immutable unless a named replay binding defines a
  bounded patch surface.
- Current position-derived grid narrowing uses
  `ReplayGridBinding::PositionCeilDiv`; the recorded grid remains a hard maximum.
- Dynamic scalar or pointer fields must be named, offset-bounded, type-sized,
  and associated with one owner. Hidden mutation of arbitrary kernarg bytes is
  forbidden.
- A dynamic grid can be certified through a bounded binding. A dynamic block
  dimension or changing shared-memory assumption requires a replay-stable
  fixed/tiled design.
- Multi-queue phases must not share a dynamic patch key whose meaning depends on
  one global capture order.

### Resources, dependencies, and state

- Allocation-wide reads and writes are explicit. Unknown effects remain
  serialized.
- Initial plan preserves all dependencies conservatively. Independence must be
  proven before waits or acquires are removed.
- Shadow oracle must cover logits, KV cache, recurrent state, convolution state,
  guard regions, exact captured blobs, and every other mutable model-specific
  state touched by the forward.
- Model reset, request reset, and model swap must reset or invalidate all
  retained mutable-state assumptions.
- Incomplete recorder coverage is not a smaller valid tape unless every omitted
  launch belongs to a named external adapter boundary and launch accounting
  reconciles exactly.

### Lifetime

Prepared AQL packets and PM4 commands contain device pointers. Model
allocations, scratch, KV/recurrent state, loaded code objects, kernarg storage,
queues, command memory, and completion objects must retain identity and lifetime
until the plan is destroyed. Allocation teardown, rebinding, model swap, or
incompatible topology change invalidates the plan before further replay.

## 5. Reproducible model and architecture porting recipe

Do these stages in order. Do not optimize waits or kernels while the route
contract is still moving.

| Stage | Required action | Binary exit criterion | Stop gate |
|---:|---|---|---|
| 1. Freeze the fixture | Record UTC date, branch/commit, clean state, binary digest, GPU/ROCm identity, model path/digest, architecture id, quantization, topology, KV mode, continuation API, exact prompt/token stream, baseline route | Another contributor can select the same bytes, binary, model, device, and route | Any identity field missing or baseline not coherent |
| 2. Define admission | One fail-closed predicate for an ordinary sequential generated-token forward; explicit negatives for prefill, spec/MTP, graph capture, batching, model swap, incompatible topology | Positive and every negative case observable; no ineligible call can record or consume the tape | Boundary depends on incidental call order or unrecorded heuristic |
| 3. Census the full compute body | rocprof or equivalent complete kernel trace; name launches intentionally outside the tape | `compute launches = retained dispatches + explicitly external launches` at each certified position | Counts do not reconcile or a “external” launch mutates retained state |
| 4. Migrate recorder coverage | Route every in-body raw launch through the typed recorder while preserving ordinary HIP call and result | Retained count equals reconciled target; ordinary-HIP outputs unchanged | Missing, duplicated, or reordered in-body launch |
| 5. Prove tape stability | Compare launch count, unique-kernel set, ordered sequence hash, geometry, owning artifact identity across positions and fresh processes | Immutable fields stable; every intended dynamic difference has a named binding | Stable partial subsequence, unnamed position-dependent mutation, unexplained fresh-process drift |
| 6. Bind artifact and ABI | Resolve exact loaded HSACO and symbol; validate loader metadata, padded kernargs, geometry, shared memory, resources, dynamic fields | Every launch passes ABI/artifact probe with no alias ambiguity | Missing/wrong HSACO, loader incompatibility, scratch/implicit-SGPR uncertainty, kernarg mismatch |
| 7. Stabilize geometry | Fix/tile changing block/shared-memory shapes; only certified bounded patches for dynamic grids/scalars | Supported positions fit recorded maximum and intended launch shape | Replaying capture-time block/shared-memory shape at a different context |
| 8. Build the state oracle | Reset/prime, logits, KV, recurrent/convolution, guard, snapshot/restore for multi-position comparison | HIP, exact HIP-kernarg-blob, and candidate retained execution compare every mutable state surface | Touched state surface cannot be observed or restored |
| 9. Lower conservatively | One queue, explicit dependency waits, architecture-correct register encoding, conservative acquire/fence policy | Prepared route reaches `Ready`, executes, completes, passes parity without hazard elision | Unsupported architecture, unresolved dependency, queue/completion lifetime failure, parity fault |
| 10. Certify then promote | Complete ladder in Section 7; archive benchmark record in Section 8 | Positive route proof, serve health, matched stationary performance, long-context state, reset, model-swap, failure-path gates all pass | Capture fingerprint, microbenchmark, silent fallback, or unmatched speed ratio as only positive evidence |

Diagnostic sequence for a known supported Qwen fixture (not full certification):

```bash
HIPFIRE_REPLAY_MANUAL_CAPTURE=1 \
HIPFIRE_REPLAY_BACKEND=shadow \
python3 scripts/redline_daemon_harness.py \
  --model "$MODEL" --skip-prefill --pm4 --shadow-iterations 15

python3 -m tools.redline bench \
  --model "$MODEL" \
  --daemon target/release/examples/daemon \
  --context 128 --iterations 100 --warmups 3 --runs 10 \
  --transport pm4 --max-seq 2048 \
  --work-dir .redline-work/product \
  --out .redline-work/product/report.json
```

Set `MODEL` to the exact digested fixture. These commands are diagnostics only.
They do **not** emit the full Section 7 timed-arm route-proof ledger.

### Route-proof product harness

`python3 -m tools.redline bench` is the route-proof-capable product harness. It
records controller `Ready`, fallback reason, observed replay positions,
prepared submission identity (packets, queue_id, queues, phases, PM4 command
dwords), tape identity, and anti-fallback evidence on both timed arms. Manual
shadow/capture via `scripts/redline_daemon_harness.py --pm4` remains discovery
evidence only and still cannot be stitched into positive timed-arm proof
([`INDEX.md`](INDEX.md), [`VALIDATION.md`](VALIDATION.md)).

A candidate report is route-proof-capable only when its
`lifecycle_route_proof.valid` and `route_proof.valid` fields are true under the
Section 7 ledger. The diagnostic commands above still do **not** emit that full
ledger; use the product bench (or the Golden wrapper around it) for certification
evidence.

### Steady-state dispatch profiler (attribution-only diagnostic)

`scripts/redline_dispatch_profile.py` is a **named manual diagnostic** for
steady-state, exactly-once retained-PM4 per-dispatch span attribution on an
instrumented GFX12 tape. It is **not** route proof, **not** Section 7
certification or registry admission, and **not** absolute or pure kernel timing.

Reasons it cannot prove those claims:

- timestamp `COPY_DATA` packets change the tape (dword count and sequence hash)
  and introduce observer overhead on every dispatch;
- span *i* is every PM4 command after timestamp *i* through dispatch *i*, so
  spans include preceding boundary packets (entry acquire on span 0; waits and
  mid-tape acquires on later spans), not kernel-only time.

Use product bench / Golden for certification evidence. Use this profiler only
to attribute relative stall shape across dispatches under the instrumented tape.


## 6. PM4 lowering and hazard policy

`Pm4Architecture::from_device` selects distinct gfx10, gfx11, and gfx12
lowering. Gfx10/gfx11 and gfx12 do not share one universal register recipe.
Register addresses, dispatch setup, acquire behavior, cache policy, and
supported phase synchronization must be derived for the selected architecture
and checked against current runtime support.

First accepted route is deliberately boring:

1. one queue;
2. original launch order;
3. explicit waits for recorded dependencies;
4. conservative entry, intermediate, and terminal acquire/fence policy;
5. one clear completion boundary;
6. no CU-mask, register-policy, or phase-parallel overlay.

Remove a wait or acquire only when both hold:

- an exact resource argument proves no producer/consumer, write/write,
  cache-visibility, or completion dependency across the boundary; and
- multi-position parity, guard checks, serve behavior, and stationary
  PM4-versus-PM4-plus-change performance all pass.

Sibling kernels are not independent merely because they have different names or
output pointers. Consider allocation-wide aliasing, shared scratch, indirect
addressing, KV/recurrent state, and terminal visibility. Unknown access metadata
means serialize.

Multi-queue or phase parallelism is a later optimization. It additionally
requires a dependency-derived phase partition; explicit cross-queue fan-in
before any consumer or terminal read; completion and queue lifetimes covering
every phase; dynamic bindings with unambiguous keys per phase/queue;
architecture-supported synchronization; and PM4-base versus PM4-plus-multi-queue
evidence.

Teach the dependency argument, not a copied packet sequence. A valid gfx1201
packet trace is not automatically a valid gfx1100 or gfx1151 lowering.

## 7. Certification and route-proof ladder

Pass these gates in order. Failure at a gate blocks promotion even if a later
metric looks attractive.

| Gate | Required evidence | Rejection condition |
|---:|---|---|
| 1. Baseline correctness | Ordinary HIP produces stable, coherent output on the exact fixture | Baseline unstable, incoherent, or fixture identity incomplete |
| 2. Capture completeness | Compute count, external-launch count, retained count, unique kernels, ordered sequence hash, fresh-process stability reconcile | Missing/extra launch, partial capture, unexplained drift, unowned external boundary |
| 3. ABI/artifact validation | Every symbol resolves to the exact loaded artifact and loader metadata; padded blobs, geometry, effects, dynamic bindings validate | Wrong/missing artifact, ambiguous alias, unsupported ABI/resource contract, hidden mutation |
| 4. Multi-position shadow parity | Ordinary HIP, retained PM4, and the exact HIP-kernarg-blob oracle agree for logits, KV, recurrent/model state, guards, and captured blobs | Any unexplained state mismatch. Bit exactness required for an execution-preserving route; tolerance only for an explicitly changed numerical kernel contract with justification |
| 5. Route proof | Backend request, transport, successful preparation, `Ready`, observed replay at multiple positions, dispatch/packet/queue/dword identity, sequence hash, no replay fault, and fallback reason recorded for every arm | Timed arm may have silently used HIP/HipGraph, or reports only capture/`Ready` without observed replay |
| 6. Production serve | User-facing generation with exact model/settings has healthy decoded output, finish state, repetition/attractor behavior, and model-specific framing | Emitted tokens without semantic/framing health, runaway, empty response, or route ambiguity |
| 7. Stationary matched performance | Certified baseline and retained route use identical binary, model, prompt/token stream, settings, process policy, and clocks; tok/s and ms/token reported | Cross-harness, cross-binary, cross-prompt, nonstationary, silent-fallback, or microkernel-only comparison |
| 8. Long-context and lifecycle | Dynamic position, geometry, KV growth, recurrent/convolution state, request reset, failure behavior, and model swap pass throughout the supported range | Context drift, state leak, stale pointer, wrong reset, or same-forward fallback claim |

Harness success without route proof is insufficient. A ratio from two nominal
arms that both executed ordinary HIP is invalid evidence. A stable manual-capture
fingerprint is discovery evidence; it does not install a plan or prove that a
user-facing forward selected retained AQL or PM4.

`python3 -m tools.redline bench` now emits the minimum Gate 5 ledger for
both timed arms. A candidate report is route-proof-capable only when its
`lifecycle_route_proof.valid` and `route_proof.valid` fields are true, every
required retained row reports `state=ready`, `fallback_reason=null`, a stable
prepared/tape identity, and positive replay-position deltas. Older reports
without those fields remain discovery evidence and cannot be stitched to a
separate daemon-harness fingerprint.

`python3 -m tools.redline golden` is a narrow sealed-fixture wrapper around that
product harness. It pins the MQ4R model, TG128 parameters, PM4 policy,
architecture-specific prepared/tape identity, registry sampling snapshot, and
stationary acceptance floor recorded in `registry/redline-golden-v1.json`.
Its pass means that exact existing fixture reproduced; it does not certify a
new route, replace any gate in this section, or create an admission.

### Minimum route-proof record per arm

| Field | Required content |
|---|---|
| Request and selection | Backend request, transport, exact eligibility predicate/result, negative gates active |
| Preparation | Capture summary; preparation success; controller state; fallback reason |
| Tape identity | Dispatch count, external count, unique-kernel count, ordered sequence hash, owning artifacts, geometry, dynamic bindings |
| Submission identity | Observed transport, packet count, queue identity/count, phase count, PM4 command dwords, completion |
| Observation span | Multiple generated positions, named |
| Fault evidence | Replay error/fault absence, guard status, any controller poison/reset event |
| Competing routes | Evidence the measured arm did not execute ordinary HIP or HipGraph |

## 8. Benchmark record schema and claim language

Every report supporting a retained-replay claim must contain:

| Group | Required fields |
|---|---|
| Time and source | UTC date/time; branch; source commit; clean/dirty state; exact daemon/binary digest |
| Device | GPU product; PCI identity; gfx architecture; visible-device/topology; ROCm, runtime, driver identity; clock/governor policy |
| Model | Full model path; architecture id; quantization; artifact size/digest; sidecar names and digests |
| Workload | Harness path/revision; full command; prompt bytes or deterministic token-stream path and digest; sampler and seed; KV mode; context; prefill/generated counts; parallel topology; graph/spec/MTP settings |
| Configuration | Every route-affecting `HIPFIRE_*` variable and whether unset; explicit backend and transport request |
| Sampling | Warmup policy; fresh-process or resident-daemon policy; run order; run count; per-run raw values; min/median/max; raw report path |
| Tape | Compute/external/retained counts; unique kernels; sequence hash; owning artifacts; geometry; resource effects; dynamic bindings |
| Route proof | Preparation success; `Ready`; observed replay positions; transport; packets; queues/phases; PM4 dwords; fault absence; fallback reason; proof against silent HIP/HipGraph |
| Correctness | Shadow/oracle report; logits/KV/recurrent/guard/blob results; tolerance and rationale if not bit exact; decoded-output health |
| Result | Tok/s and ms/token per arm; ratio and absolute delta; predeclared promotion rule; disposition |

### Comparison rules

- Byte-identical prompts or deterministic token streams; same binary, model,
  topology, KV mode, context, generation length, sampler, and clocks.
- State whether each sample used a fresh process or a resident daemon. Do not
  mix policies within an A/B.
- Interleave run order when practical; archive raw values; report
  min/median/max rather than one best run.
- Automatic clocks are the default unless clock policy itself is the experiment.
- Compare both tok/s and milliseconds per token.
- Numbers from different harnesses, binaries, prompts, or route-proof states are
  not one A/B.
- A dated number is evidence for its exact fixture, not a permanent minimum
  speedup or current product guarantee.
- General perf protocol:
  [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).
  Historical non-Redline tables:
  [`BENCHMARKS.md`](BENCHMARKS.md).

### Direct value versus enabling value

1. **Direct transport value:** one retained submission can remove variable host
   launch and packet-publication work. Measure with a matched baseline-versus-PM4
   comparison in which the device kernel stack is otherwise identical.
2. **Enabling value:** reducing the host floor and host-side variance makes later
   device-side kernel, traffic, fence, overlap, and composite transformations
   easier to resolve. Measure each later lever as base PM4 versus PM4 plus that
   one lever.

Do not merge these claims. The historical Qwen MQ4R progression from roughly 110
to 204 tok/s combined changing kernel stacks, graph shape, and Redline work. It
is useful dated engineering history, not a pure PM4 transport A/B.

## 9. Post-Redline kernel and Radiowave loop

After the conservative retained route passes Section 7:

1. Freeze a stationary retained-PM4 baseline and archive its complete record.
2. Profile the new device-bound timeline rather than assuming host launch cost
   still dominates.
3. Choose one device-side lever: kernel change, fusion, traffic reduction,
   fence/acquire policy, queue overlap, or Radiowave/composite transformation.
4. Keep the base tape, fixture, binary inputs, route proof, and all unrelated
   overlays fixed.
5. Compare **base PM4** against **PM4 plus exactly that overlay**.
6. Retain the overlay only if state parity, route proof, serve health,
   long-context behavior, and reproducible wall time all pass.
7. Compose independently proven overlays one at a time; refresh the stationary
   PM4 baseline after each accepted change.
8. Rerun the full gates after any hazard-policy, geometry, artifact,
   dynamic-binding, or mutable-state boundary change.

Launch-count reduction alone is not proof of a wall-time win. Under retained
PM4, fusion earns value by reducing device work, memory traffic, or
synchronization—not by claiming avoidance of host API calls the retained
transport already removed. Radiowave transformations belong in this
post-certification loop, not in the definition of Redline.

## 10. Failure atlas

| Symptom or mistake | Failure phase / diagnosis | Required response |
|---|---|---|
| Capture starts during model load or prefill | Admission boundary wrong | Reset; arm on first eligible continuation; recapture; rerun every gate |
| Raw in-body launch absent from recorder | Capture completeness failure | Migrate to typed recorder or formally externalize and reconcile counts |
| Recorder retains stack-backed kernargs | ABI/lifetime failure | Copy exact padded bytes into owned storage; invalidate the tape |
| HSACO missing, wrong, or loader-incompatible | Artifact/ABI preparation failure | Resolve exact owning artifact; fail before `Ready` |
| Specialized launch alias resolves to another artifact | Launch/artifact identity failure | Correct alias mapping; regenerate fingerprint and parity |
| Capture-time block/shared-memory shape replayed at new context | Geometry contract failure | Fix/tile or add bounded binding; recertify range |
| Token, KV, recurrent, or conv state absent from oracle | State-certification gap | Add snapshot/restore before interpreting parity or speed |
| Model swap retains pointer-keyed state | Lifetime/reset failure | Destroy prepared objects; reset admission on every successful reconfiguration |
| Scratch, implicit SGPR, or loader-kernarg unsupported | Preparation cannot safely lower | Reject installation until runtime validates the exact contract |
| Wait/acquire removed because launches “look independent” | Hazard-policy failure | Restore conservative ordering; exact resource argument + multi-position evidence |
| Timed `auto` and `hip` arms both execute HIP | Route-proof failure | Discard the ratio; instrument request, state, transport, packets/queues/dwords, replay positions, fallback |
| Replay execution described as same-token HIP fallback | Failure-phase misreport | Report current forward error and later sticky fallback exactly |
| Stable partial capture called a complete route | Capture classification failure | Reconcile full compute/external/retained counts |
| Launch graph shrinks but wall time does not improve | Structural result without product value | Keep experimental or reject under campaign rule |
| Results from different harnesses/binaries/prompts/clocks compared | Benchmark comparability failure | Rerun stationary matched experiment |
| Manual capture fingerprint called routed PM4 | Discovery over-promoted | Add preparation, `Ready`, observed multi-position replay, submission identity, anti-fallback proof |
| Production framing fails while numerical replay parity passes | User-facing contract failure may be framing | Diagnose framing separately; block product promotion until serve health passes |

## 11. Worked cases

Same classification fields throughout. Historical measurements are quoted only
with their dates and fixtures. None of these cases is a universal admission.

### 11.1 Qwen3.5 0.8B dense `.mq4` on gfx1201: positive explicit opt-in evidence

| Field | Evidence |
|---|---|
| Intent | Compact dense-model capture, exact-state parity, retained-PM4 example |
| Baseline route | HipGraph ordinary AR for the dated comparison |
| Candidate route | Explicitly selected retained PM4. `arch_id=5`, `.mq4` — **not** the automatic `.mq4r` product default (then and now; `.mq4` is outside `mq4r_redline_default`) |
| Fixture | Qwen3.5 0.8B dense on gfx1201; dated 2026-07-11 in `crates/redline-dispatch/HIPFIRE-GRAFT.md`. Recoverable row does not pin full model digest or daemon binary digest |
| Immutable contract | 356 retained dispatches, 21 unique kernels, ordered sequence hash `55f99a58cb4b9363` |
| Validation | Fifteen consecutive positions bit exact for logits, KV, and recurrent state vs ordinary HIP and exact HIP-kernarg-blob oracle |
| PM4 and route evidence | Dated record preserves retained-PM4 fingerprint, PM4 shadow/parity, and a nominal HipGraph-versus-PM4 product comparison. Recoverable timed-arm report **lacks** the full Section 7 ledger |
| Performance observation | 2026-07-11, automatic clocks, resident 10×100: HipGraph 363.682 tok/s vs requested PM4 392.248 tok/s (`1.07855×`). Incomplete timed-arm ledger → fixture-bound positive performance evidence, **not** fully certified direct-transport A/B |
| Serve | Five-prompt greedy battery: no runaways, avg 384.7 decode tok/s; response-framing gate failed (independent of PM4 parity; still fails full production-serve gate) |
| Disposition | Preserve as positive explicit-opt-in capture/parity/PM4/performance evidence. **Not fully certified under this guide**, not a passed full serve case, not automatic-default admission |
| Reusable lesson | Capability, positive transport evidence, guide certification, production-serve health, and product default are separate classifications |

### 11.2 Qwen3.6 35B-A3B MQ4R across gfx1100, gfx1151, and gfx1201

Current automatic predicate `mq4r_redline_default` is model-family agnostic
and narrower than full implementation capability: exact GPU arch `gfx1100`,
`gfx1151`, or `gfx1201`; single GPU (`pp=tp=1`); case-insensitive `.mq4r`.
No `arch_id` gate. `gfx1200` and every other arch remain explicit opt-in.
Explicit backend selection (including the config wizard / profile backend)
can request broader implemented paths or disable the automatic default.
Explicit transport changes only the transport. Runtime default ≠ Section 7
certification ≠ registry admission; sealed golden / fixture rows are not
implied for every default-eligible arch.

| Architecture | Implementation capability | Model performance evidence | Explicit opt-in | Recoverable retained-PM4 evidence under Section 7 | Automatic default |
|---|---|---|---|---|---|
| gfx1100 | Yes: gfx11 PM4 lowering exists | Sealed fixture: HIP median 219.649 tok/s, PM4 median 251.798 tok/s (`1.146×`) | Via explicit backend request | Golden row `qwen36-a3b-mq4r-gfx1100-tg128-q8-pm4`: exact model/tool identities, retained route fingerprint, positions 128/255 observed, stationary floor | Yes when complete `mq4r_redline_default` holds and no wizard/backend/manual-capture bypass wins; measured validation is not admission |
| gfx1151 | Yes: gfx11 PM4 lowering and gfx1151 experiment knobs | Sealed fixture: HIP median 102.348 tok/s, PM4 median 115.290 tok/s (`1.126×`); coherent Silver high-water 114.209 vs Golden floor 115.021 (-0.812 tok/s, 0.71%, approximately 1 tok/s); ROCm 7.14 is only a hypothesis — neither causality nor absence of a Redline route regression is proven | Via explicit backend request | Golden row `qwen36-a3b-mq4r-gfx1151-tg128-q8-pm4`: exact model/tool identities, retained route fingerprint, positions 128/255 observed; Silver route proof valid | Yes when complete `mq4r_redline_default` holds and no wizard/backend/manual-capture bypass wins; measured validation is not admission |
| gfx1200 | Yes where gfx12 PM4 lowering applies | Model performance may exist on related RDNA4 SKUs; not this automatic row | Via explicit backend request | Not an automatic-default arch | **No** — remains opt-in only |
| gfx1201 | Yes: gfx12 PM4 lowering and Qwen adapter | Sealed fixture: HIP median 174.670 tok/s, PM4 median 202.460 tok/s (`1.159×`) | Available | Golden row `qwen36-a3b-mq4r-gfx1201-tg128-q8-pm4`: exact model/tool identities, retained route fingerprint, positions 128/255 observed, stationary floor | Yes when complete `mq4r_redline_default` holds and no wizard/backend/manual-capture bypass wins; measured validation is not admission |

Primary gfx1201 case:

| Field | Evidence |
|---|---|
| Intent | Single-GPU ordinary-AR Qwen3.6 35B-A3B MQ4R retained-PM4 route |
| Baseline route | Tuned HipGraph ordinary AR |
| Candidate route | `redline-dispatch`/`redline-rocr` PM4-IB through the Qwen adapter |
| Fixture | `qwen3.6-35b-a3b.mq4r`, `arch_id=6`, Q8 KV, no MTP/DFlash, gfx1201 Radeon AI PRO R9700; see `docs/perf-checkpoints/2026-07-11-redline-qwen36-a3b-ar.md` |
| Immutable contract | Initial tape: 833 launches, 26 kernels, sequence hash `8d5620ca2ca8a536`; PM4 body 34,563 dwords; one vendor AQL packet. Later graph reshapes have checkpoint-specific fingerprints |
| Validation | Fifteen-position bit-exact logits hash `9874244965e2c7d6`, KV `fa5f3bb2b32fffcd`, recurrent `609db41ffad8ceb6` |
| PM4 and route evidence | Dated shadow proves successful retained-PM4 execution and multi-position parity. Product benchmark records requested `transport=pm4` and throughput but **not** per-timed-arm preparation/`Ready`/fallback/observed replay/anti-fallback proof |
| Conservative PM4 observation | HipGraph 164.220 vs requested retained PM4 174.087 tok/s (`1.06009×`) — closest recoverable conservative observation; not fully certified without timed-arm ledger |
| Selective fence/wait composition | Tuned HipGraph 165.839 vs retained PM4 plus removal of one proven-independent boundary wait 178.320 tok/s (`1.07526×`) — **combined** PM4 + fence/wait observation, not direct transport alone |
| Later progression | `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md` — productized no-env campaign including TG128 median 203.93 tok/s. Changed kernel stack and graph shape; not a direct transport A/B |
| Automatic runtime selection | Runtime requests `Auto` on complete `mq4r_redline_default` (exact `gfx1100`/`gfx1151`/`gfx1201`, `pp=tp=1`, `.mq4r`; no `arch_id` gate) unless config wizard/backend selection or enabled manual capture bypasses. At the dated campaign the then-current predicate was A3B-only (`a3b_mq4r_redline_default`, `arch_id=6`). Runtime default ≠ guide certification ≠ registry admission |
| Disposition | Strongest recoverable gfx1201 PM4 capture/parity/submission/performance evidence. **Do not call fully certified under this guide** until a positive timed-arm route-proof ledger is archived |

### 11.3 Rejected gfx1030 Qwen3.6 MQ2 lowering

Repository-visible rejection record (performance-checkpoint tree not extended).

| Field | Evidence |
|---|---|
| Intent | Port Qwen3.6 35B-A3B MQ2G256Lloyd prefill and retained-PM4 decode to RX 6950 XT gfx1030 for a product wall-time win |
| Baseline route | Ordinary HIP decode |
| Candidate route | Product `auto` with `transport=pm4`, plus Radiowave off/on overlay |
| Source identity | Host `hipx`; branch `feat/mq2g256-gfx1030-prefill-redline`; prefill commit `0f3444f8cecf9976ced483237a8fc26028f3b94d`; measured candidate `e017f83ceb9d41d4be0d6665161615c9ae74d89b`; daemon SHA-256 `47585859295f44a5cc2aab090e7fc43ef342d2932e1d7d402a4d569cbf53acaf` |
| Model and device | `/home/kaden/bench/models/qwen3.6-35b-a3b.mq2`, SHA-256 `48b3f84614c46eb8b5ffb494f7a75c15216664afcbb47c3e78dd80c4ce7eb0a3`; RX 6950 XT gfx1030; `ROCR_VISIBLE_DEVICES=2`; Q8 KV; automatic clocks; dated 2026-07-18 |
| Exact PM4 evidence | Radiowave-off/on exact reports: 942 launches, 24 kernels, sequence hash `becff4a4f1849d1e`, one PM4-IB packet, queue id 2, 21,783 command dwords; bit-exact parity passed |
| Matched product result | Same e017 binary/model: Radiowave-off HIP median 101.431 vs `auto` 83.711 tok/s (`0.82529×`); Radiowave-on similar ~17.5% slower; overlay did not recover |
| Route-proof limitation | Product JSON records requested `transport=pm4` but omits controller `Ready`, fallback reason, observed multi-position replay |
| Disposition | **Rejected** as retained-route/Radiowave product promotion |
| Reusable lesson | Stable sequence + bit-exact shadow cannot replace positive product-arm route proof plus matched wall time |

Raw remote evidence (not checked in):
`ssh://hipx/home/kaden/redline-results/gfx1030-radiowave-ab-20260718/` and
`ssh://hipx/home/kaden/redline-results/gfx1030-radiowave-precontract-ab-20260718/`.

### 11.4 Rejected LFM Stage A

Repository-visible rejection record. **Not Redline** — serial-HIP fusion only.
Exact scope: **LFM2.5-350M dense MQ4** on **gfx1201**, candidate path with
explicit `HIPFIRE_LFM2_DECODE_FUSION=1`, ordinary serial HIP only (no AQL/PM4/
HipGraph product route). **Rejected / not shipped.**

**2026-07-19 Stage-A disposition:** LFM was not admitted to a retained route
in that campaign — no Redline admission predicate then covered `arch_id==11`.
That remains the historical Stage A rejection (serial-HIP fusion only; not an
admitted-then-rejected Redline candidate). It does **not** deny the later exact
retained-PM4 sealed registry evidence/admission record documented in §11.5 and
[`admissions.yml`](admissions.yml) (evidence only; no automatic runtime wiring).

| Field | Evidence |
|---|---|
| Intent | Reduce LFM2.5-350M dense MQ4 gfx1201 serial-HIP decode launches by fusing RMSNorm plus MQ rotation activation preparation |
| Baseline route | Serial HIP lowered decode, `HIPFIRE_LFM2_DECODE_FUSION=0`, graph off, Q8 KV |
| Candidate route | Same serial-HIP path with `HIPFIRE_LFM2_DECODE_FUSION=1`; **not** Redline, AQL, or PM4 |
| Retained-route admission (2026-07-19 Stage A) | **None — never admitted in that campaign.** The then-current Qwen automatic default (`a3b_mq4r_redline_default`) required `arch_id==6`; LFM is `arch_id==11` and was excluded from that historical predicate. (Current policy is model-family agnostic `mq4r_redline_default` on `.mq4r` only; this sealed LFM fixture remains `.mq4` and is still not auto-selected.) `HIPFIRE_REPLAY_MANUAL_CAPTURE=1` bypassed admission and installed no plan. The sole arch-11 predicate then in play, `plan_lfm_decode_fusion`, gated serial-HIP decode **fusion**, not a retained route. The later sealed registry evidence/admission record is §11.5 / [`admissions.yml`](admissions.yml) — it does not reopen or promote Stage A, and it does not wire automatic runtime selection. |
| Source identity | Baseline `lfm-redline` @ `e8831ae8347f04ac821077ee159c86423b4bf88a`, daemon MD5 `9ee43d2673866775786d8075fb5b6e76`; candidate `feat/lfm-gfx1201-mq4-decode-fusion` @ `518c221756a1065a7560449165bc8817c2ad6176`, daemon MD5 `07d62bbd915416b07ce7783969126dd7` |
| Fixture | `lfm2.5-350m.mq4` (dense) MD5 `cb5284b8ad5c6f9e4ca859c0aff0bcd0`; prompt fixture MD5 `18cb45e00d424bef16fa9b097d02caf3`; gfx1201; HIP/ROCm 7.2; dated 2026-07-19 |
| Correctness | Frozen twelve-step decode parity exact; Stage A negatives (default/eager-prefill/spec/graph/capture) did not admit; serve content equality passed |
| Launch reconciliation | Baseline 281 compute launches/token; Stage A 221 compute / recorder-visible 204 launches, 9 kernels, hash `67dcc9e17e00ed8f` |
| Fresh-process ABBA | Pooled tg128 medians +2.11%; tg512 +1.04% — both missed predeclared ≥5% wall gates |
| Route proof | Absent by design: no PM4 plan installed; harness stopped at Qwen-only shadow requirement |
| Disposition | **Rejected** as standalone Stage A promotion. Classified as serial-HIP activation-preparation fusion, **not** Redline. Not shipped. |
| Reusable lesson | Fewer launches are neither retained replay nor a wall-time win. **As of the 2026-07-19 Stage A campaign, LFM was never admitted to a Redline route** — "not Redline" is upstream of the then-absent PM4/shadow work. Preserve that rejection. The later exact gfx1201 retained-PM4 row in §11.5 / [`admissions.yml`](admissions.yml) is a separate sealed registry evidence/admission record without automatic runtime wiring and does not imply generic LFM Redline, automatic product default, or promotion beyond that sealed fixture. |

**Local evidence pointers (session/workstation-local — not checked into this
repo):**

| Kind | Location |
|---|---|
| Source ledger / narrative report | `local://lfm-stage-a-measurement-report.md` |
| Raw artifact root | `/home/kaden/ClaudeCode/autorocm/lfm-stage-a-measurement-20260719/` |
| Phase command ledgers | `run_preflight.sh`, `run_serve_gate.sh`, `run_abba.sh`, `run_recorder.sh`, `run_direct_capture.sh`, `run_rocprof.sh` under that root |
| Phase raw dirs | `preflight/`, `serve/`, `abba/`, `recorder/`, `rocprof/` under that root |
| GPU / ROCm identity | live `rocminfo` gfx1201 UUID `GPU-6125bfcd5e216e52`; daemon HIP 7.2; `hipcc` `7.2.26015-fc0010cf6a`; ROCm clang `roc-7.2.0 26014` (see `preflight/rocminfo.stdout`, `preflight/hipcc-version.stdout`) |

**Locality warning:** those raw artifacts and the `local://` ledger are
session/workstation-local evidence for the named 2026-07-19 campaign. They are
**not** portable repository fixtures. Absence of the local tree on another
machine does not reopen the rejection; it only means the raw dumps are not
recoverable there. Do not treat missing local paths as missing rejection.

### 11.5 LFM2.5-350M dense MQ4 on gfx1201: positive retained-PM4 evidence

This case supersedes only the missing-retained-route conclusion in Section
11.4. It does not retroactively promote Stage A, does not wire automatic LFM
runtime selection, and does not admit another LFM size, quant, architecture,
topology, speculative route, or graph route. The sealed [`admissions.yml`](admissions.yml)
record remains registry evidence/admission only; current LFM retained replay is
explicit opt-in and must still prove route support.

| Field | Evidence |
|---|---|
| Date and fixture | 2026-07-20; `/home/kaden/.hipfire/models/lfm2.5-350m.mq4`, 229,474,032 bytes, MD5 `cb5284b8ad5c6f9e4ca859c0aff0bcd0`, base HFQ with no REAP overlay; gfx1201; Q8 KV; `max_seq=physical_cap=2048`; context position 127; plain AR; `pp=tp=1`; graph off |
| Exact admission | On Linux, the trusted loader first rejects REAP overlays, then copies exactly 229,474,032 bytes from the already-open HFQ into an anonymous memfd while computing the content MD5 over that same stream. It applies and verifies write/grow/shrink/final seals, reparses the HFQ from the sealed fd, and rebinds later mmap, pread, and path-based reads to that immutable snapshot before minting the opaque exact-identity token. Short reads fail; non-Linux targets mint no retained provenance. The token is then combined with the frozen LFM structure. Retained replay additionally requires gfx1201, active decode fusion, lowered forward, sequential `position == n_tokens`, non-HIP replay request, plain AR, `pp=tp=1`, Q8 KV with the exact 2048-token state surface, and graph off. Any identity, overlay, route, or geometry miss selects ordinary decode. MD5 is fixture identity here, not a cryptographic authorization boundary. |
| Tape identity | 161 retained dispatches, 15 unique kernels, ordered sequence hash `04305cf1254244f6`; one PM4-IB packet on queue 2; 3,916 command dwords |
| Shadow equivalence | Artifact `.redline-work/lfm-pm4-wide-sealed-final.json` reports `backend=pm4_ib`: 161 dispatches across 15 kernel identities collapsed into one 3,916-dword IB on queue 2. The ordered dispatch sequence hash was `04305cf1254244f6`; logits, KV, recurrent state, scratch state, and cursor state were bit-exact across 4 compared decode steps. PM4-IB host time was 7,296.400 µs versus 14,283.526 µs for fresh HIP. |
| Product route proof | Fresh post-snapshot artifact `.redline-work/lfm-product-pm4-sealed-final.json`: `auto`/PM4 reached `ready`, reported no fallback, and observed retained replay on every timed row from positions 127 through 158. The timed report carries the same 161/15/`04305cf1254244f6` tape and 1/2/3916 submission identity as the shadow report; the ordinary-HIP control recorded zero retained rows. |
| Fresh product A/B | Artifact `.redline-work/lfm-product-pm4-sealed-final.json` reports ordinary HIP median `807.772834 tok/s`, retained PM4 median `1,023.616692 tok/s`, and ratio `1.267208612`. `auto.route_proof` reports `backend=auto`, `transport=pm4`, `rows=5`, `require_complete_replay=true`, and `retained_rows=5`, with observed positions `[127, 158]`; its timed rows report `redline_route.state=ready`, `retained_replay_observed=true`, and `observed.count_delta=32`. The HIP control's route proof reports `backend=hip` and `retained_rows=0`; its timed rows report `redline_route.state=hip`, `retained_replay_observed=false`, and `observed.count_delta=0`. |
| Measurement caveat | This campaign used deliberately relaxed stationarity screening (`settle_window=3`, max slope 10%, max spread 20%, max median drift 20%). The final three-row spreads were 0.555% for HIP and 0.247% for PM4, but this is a screening result, not the canonical 1% stationarity/certification floor. Do not relabel `1,023.616692 tok/s` as certified. |
| Serve health | The production server returned `Paris Berlin Rome`, `finish_reason=stop`, 3 generated tokens, with no empty-output, runaway, or repeated-attractor failure (`.redline-work/lfm-serve-health-sealed-final.json`); the matching runtime transcript is `.redline-work/lfm-serve-health-sealed-final.log`. |
| Disposition | Positive, self-contained retained-PM4 product-route evidence for the exact fixture. Gate 5 is cleared for this dated run. Full Section 7 certification remains blocked at Gate 7 until the matched automatic-clock arms satisfy the canonical stationarity contract. |

Reproduce the exact shadow and submission proof from a release daemon:

```bash
cargo build -p hipfire-runtime --release --example daemon --features deltanet

HIPFIRE_LFM2_GFX1201_DECODE_FUSION=1 \
HIPFIRE_REPLAY_LOWERED_FORWARD=1 \
HIPFIRE_REPLAY_MANUAL_CAPTURE=1 \
HIPFIRE_REPLAY_BACKEND=shadow \
HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
HIPFIRE_REPLAY_PM4_QUEUES=1 \
HIPFIRE_REPLAY_PM4_STATEFUL=stateful \
HIPFIRE_REPLAY_PM4_GCR_TRIM=1 \
python3 scripts/redline_daemon_harness.py \
  --model ~/.hipfire/models/lfm2.5-350m.mq4 \
  --daemon target/release/examples/daemon \
  --kv-mode q8 --skip-prefill --decode-context 127 \
  --capture-repeats 2 --measure-repeats 5 \
  --decode-iterations 4 --shadow-iterations 4 --pm4 \
  --out .redline-work/lfm-pm4-wide-exact.json
```

The required pass record is `sequence_stable=true`, `pass=true`,
`backend=pm4_ib`, all state/blob equality fields true, and the exact tape and
submission identity above. A different hash is not automatically a failure,
but it invalidates this dated identity and requires fresh ABI, parity, route,
and performance evidence.

Run the product arm separately; do not stitch its result to the shadow report:

```bash
HIPFIRE_LFM2_GFX1201_DECODE_FUSION=1 \
HIPFIRE_REPLAY_LOWERED_FORWARD=1 \
HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
HIPFIRE_REPLAY_PM4_QUEUES=1 \
HIPFIRE_REPLAY_PM4_STATEFUL=stateful \
HIPFIRE_REPLAY_PM4_GCR_TRIM=1 \
python3 -m tools.redline bench \
  --model ~/.hipfire/models/lfm2.5-350m.mq4 \
  --daemon target/release/examples/daemon \
  --context 127 --transport pm4 --kv-mode q8 \
  --dpm-warmup-secs 10 \
  --out .redline-work/lfm-product-pm4-stationary.json
```

Leave the benchmark's stationarity defaults intact for a certification claim.
The report must end with `valid=true`; the retained arm must have complete
positive `lifecycle_route_proof` and `route_proof`; and the HIP arm must show
zero retained rows. A faster row from a failed or relaxed campaign remains
screening evidence.

## 12. Copyable new-route checklist

### Admission and boundary

- [ ] Exact positive admission predicate names model, architecture, quantization, topology, continuation shape, and route
- [ ] Explicit negative gates cover prefill, spec/MTP, batching, graph/capture conflicts, model swap, and every non-sequential call
- [ ] Compute / external / retained launches reconcile exactly
- [ ] Every external adapter launch is named with state/lifetime justification

### Tape, ABI, and state

- [ ] Every in-body launch uses the typed recorder while preserving ordinary-HIP behavior
- [ ] Count, unique-kernel set, ordered sequence hash, geometry, and owning artifact identity are stable across positions and fresh processes
- [ ] Each symbol resolves to the exact loaded artifact and loader kernarg metadata
- [ ] Exact padded kernarg bytes are owned; no stack-backed argument storage survives capture
- [ ] Resource reads/writes and dependencies are conservative and explicit
- [ ] Each dynamic value uses a named, bounded binding; no hidden kernarg mutation
- [ ] Reset/prime and snapshot/restore cover logits, KV, recurrent/convolution state, guards, and all model-specific mutable state
- [ ] Model swap/allocation teardown invalidates every retained pointer and prepared object

### Lowering and certification

- [ ] PM4 register, acquire, fence, and completion policy is correct for the selected architecture
- [ ] First route is single-queue and conservatively ordered
- [ ] Ordinary HIP, exact HIP-kernarg-blob, and retained PM4 pass multi-position state parity
- [ ] Route proof records request, transport, preparation, `Ready`, observed replay positions, dispatches, packets, queues/phases, dwords, faults, and fallback reason
- [ ] Timed-arm ledger includes complete lifecycle/route-proof fields (controller, observed replay, transport, anti-fallback). Missing those fields leaves **that report** experimental / fail-closed for positive timed-arm route proof — do not stitch incomplete reports into certification
- [ ] Timed retained arm is proven not to be ordinary HIP or HipGraph
- [ ] Production serve output, finish state, repetition/attractor health, and response framing pass
- [ ] Dynamic position, growing context, request reset, failure behavior, and model swap pass
- [ ] Stationary matched performance reports tok/s, ms/token, raw samples, and the predeclared disposition rule
- [ ] Dated raw evidence has an immutable path and complete identity manifest

## 13. Copyable reviewer checklist

- [ ] I can distinguish implementation capability, model performance, opt-in availability, retained-route certification, and automatic-default admission in every claim
- [ ] I verified the current runtime predicate and state transitions against the source symbols below
- [ ] Automatic `Captured → Ready` is not mislabeled as automatic shadow certification
- [ ] Preparation failure and replay-execution failure use the exact phase-specific semantics in Section 3
- [ ] Full compute/external/retained launch equation reconciles; no stable partial tape is promoted
- [ ] Artifact, kernarg, geometry, resource, binding, lifetime, reset, and model-swap contracts are explicit
- [ ] Parity spans multiple positions and every mutable state surface, not logits alone
- [ ] Positive route proof excludes silent HIP and HipGraph for every timed arm
- [ ] Benchmark arms match binary, model, prompt/token bytes, topology, KV mode, clocks, harness, and process policy
- [ ] Direct transport A/B is separate from any changing-kernel historical progression
- [ ] Kernel/Radiowave/hazard overlays are compared as base PM4 versus PM4 plus one overlay
- [ ] Launch-count reductions are not presented as wall-time wins without stationary evidence
- [ ] Every throughput or ratio is dated and fixture-bound, never a timeless floor
- [ ] Rejected cases preserve the established falsification and do not invent a missing failure
- [ ] Raw reports and source/binary/model digests are recoverable
- [ ] No stitched product-bench + daemon-harness “route proof”
- [ ] No generic LFM promotion from Stage A alone; every current LFM claim stays inside the exact admitted fixture and carries its own route proof

## 14. Stable source-path and symbol index

Prefer paths and symbols over line numbers.

| Concern | Stable source path and symbols |
|---|---|
| Automatic product predicates | `crates/hipfire-runtime/src/config.rs` — `mq4r_redline_default` only (`.mq4r` + exact `gfx1100`/`gfx1151`/`gfx1201` + `pp=tp=1`; model-family agnostic, no `arch_id` gate). Existing LFM `.mq4` sealed evidence has no automatic selector. |
| Model-load application and diagnostic handlers | `crates/hipfire-runtime/examples/daemon.rs` — load-time `configure_model_default`; `redline_capture`; `redline_shadow_aql`; `redline_shadow_pm4` |
| Qwen model boundary and route | `crates/hipfire-arch-qwen35/src/qwen35.rs` — `forward_scratch`; `prepare_scratch_inputs`; `set_forward_eligible`; `should_route_aql`; `should_route_pm4`; `finish_capture`; `prepare_*` |
| LFM decode path and registry evidence | `crates/hipfire-arch-lfm2moe/src/forward.rs` — `decode_step`; `decode_step_with_graph`; lowered path under `HIPFIRE_FORWARD_LOWERED`. Sealed LFM retained-PM4 evidence/admission: [`admissions.yml`](admissions.yml) row `lfm25-350m-mq4-gfx1201-retained-pm4-plain-ar` (no current runtime selector). |
| LFM generate / spec path | `crates/hipfire-runtime/examples/daemon.rs` — `generate_lfm2moe`; `crates/hipfire-arch-lfm2moe/src/spec_impl.rs` — `Lfm2MoeBundle` / `SpecTarget` |
| Controller, tape, lifecycle, routing | `crates/rdna-compute/src/replay.rs` — `ReplayController`; `ReplayState`; `RecordedHipLaunch`; `ReplayGridBinding`; `configure_model_default`; `reset_for_model`; `begin_auto_capture_if_armed`; `finish_capture`; `prepare_linear_aql_prefix`; `prepare_pm4_prefix`; `replay_linear_aql`; `replay_pm4`; `observe_shadow`; `install_prepared_plan`; `should_route_aql`; `should_route_pm4`; `poison` |
| Central HIP recording and artifact aliases | `crates/rdna-compute/src/dispatch.rs` — `Gpu::replay`; typed HIP launch recording |
| DAG, identity, ABI, visibility | `crates/redline-dispatch/src/lib.rs` — `Recorder`; `CompiledPlan`; `KernelArtifactIdentity`; `KernargAbi`; `derive_aql_visibility` |
| Retained AQL and PM4 graph objects | `crates/redline-dispatch/src/aql/replay.rs` — `SingleQueueBatchGraph`; `SingleQueuePm4Ib`; `PhasedMultiQueuePm4Ib` |
| Public ROCr/HSA ownership | `crates/redline-rocr/src/lib.rs` |
| PM4-IB vendor packet | `crates/redline-rocr/src/packet.rs` — `PacketImage::pm4_indirect_buffer` |
| Architecture PM4 builders | `crates/redline-rocr/src/pm4.rs`; `crates/redline-rocr/src/pm4_gfx10.rs` |
| Manual capture/shadow diagnostic | `scripts/redline_daemon_harness.py` |
| Product stationary comparison | `python3 -m tools.redline bench` |
| Sealed MQ4R fixture reproduction | `python3 -m tools.redline golden`; `registry/redline-golden-v1.json`; `docs/GOLDEN-REDLINE.md` |
| Claim → validation route selector | [`VALIDATION.md`](VALIDATION.md) |
| Graft and ABI provenance | `crates/redline-dispatch/HIPFIRE-GRAFT.md`; `crates/redline-rocr/PROVENANCE.md` |
| Positive dated gfx1201 evidence | `docs/perf-checkpoints/2026-07-11-redline-qwen36-a3b-ar.md`; `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md` |

A change to any referenced route, state, artifact, geometry, resource, or
lifetime symbol requires rechecking the applicable certification gates. Update
this guide when contributor procedure changes; leave dated case records
immutable for their original fixtures.
