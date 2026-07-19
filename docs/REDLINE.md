# Redline Contributor Guide

This is the canonical, normative contributor procedure for Redline retained replay in hipfire. It defines how to construct, validate, measure, review, and promote a retained route. Runtime source remains authoritative for executable behavior. Dated reports remain evidence for the exact fixtures they record; they are not current defaults or timeless performance floors.

## 1. Scope, authority, and evidence classes

Use this guide when changing any of the following:

- model admission to retained AQL or retained PM4;
- recorder coverage, launch identity, kernarg capture, artifacts, resource effects, or dynamic bindings;
- retained-plan construction, PM4 lowering, queue policy, waits, fences, or acquire policy;
- model reset, pointer lifetime, replay failure, or fallback behavior;
- a kernel, fusion, Radiowave transformation, or scheduling overlay on a retained route;
- a benchmark or product claim attributed to Redline.

This guide covers ordinary sequential single-token autoregressive continuation. It does not certify prefill, speculative verification, MTP reseed or proposal work, mutable multi-token replay, or a new model path merely because those paths can be captured. It does not make the experimental direct-KMD `crates/redline` crate the serving transport.

### Source-of-truth order

1. Runtime source and tests define executable admission, state transitions, failure handling, and currently implemented encodings.
2. This guide defines contributor workflow and the certification policy required before promotion or a performance claim.
3. Dated performance checkpoints and raw reports preserve observations for exact fixtures.
4. `crates/redline-dispatch/HIPFIRE-GRAFT.md` and `crates/redline-rocr/PROVENANCE.md` preserve graft and ABI provenance; they do not define the current procedure.

Every Redline statement should be recognizable as one of these evidence classes:

| Class | Meaning | Examples |
|---|---|---|
| **Runtime fact** | Directly defined by current source or tests. | The current automatic default predicate; `ReplayState` transitions; replay failure returns an error. |
| **Certification policy** | Evidence this guide requires before a route or optimization is promoted. It may be stricter than runtime enforcement. | Multi-position shadow parity; positive timed-arm route proof; matched stationary performance. |
| **Dated evidence** | An observation tied to a named date, source/binary/model identity, and fixture. | A 2026-07-11 gfx1201 PM4-versus-HipGraph measurement. |

`ReplayState::Ready` is a runtime state. It is not, by itself, repository certification.

### Terms

| Term | Definition |
|---|---|
| **Recorder-aware HIP launch** | An ordinary HIP launch that also contributes exact typed launch metadata to the retained tape. Recorder awareness must not alter direct-HIP semantics. |
| **Retained tape** | The stable ordered launch sequence plus exact artifacts, padded kernargs, geometry, resource contracts, dependencies, and dynamic bindings needed for replay. |
| **PM4 command body** | An architecture-specific indirect-buffer command stream lowered from a certified tape. |
| **PM4-IB AQL packet** | The public ROCr/HSA vendor packet that submits a retained PM4 command body. |
| **Runtime admission** | Current code that decides whether a forward may capture, prepare, or consume replay. |
| **Certification gate** | Evidence required by this guide before promotion or a performance claim. |
| **External adapter boundary** | Work intentionally kept outside the retained body and explicitly counted as such, such as current Qwen token embedding and position preparation. |

## 2. Mental model and ownership

The active retained-PM4 path is:

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

Redline retains the real model forward. It does not inherently replace or fuse its kernels. A kernel change is a separate optimization whose numerical, resource, and hazard contracts must be re-certified.

### Three-crate ownership

| Crate | Owns | Does not own |
|---|---|---|
| `redline` | Experimental direct-KMD/bare-libdrm device, memory, queue, PM4, and synchronization machinery. | The active product serving route. |
| `redline-dispatch` | Dispatch-DAG recording and validation, artifact and kernarg identity, dependency and visibility policy, plan compilation/selection, and retained AQL/PM4 graph construction. | ROCr ABI lifetime mechanics or model-specific admission. |
| `redline-rocr` | Dynamically loaded public ROCr/HSA ABI; agent, queue, memory, packet, signal, doorbell, and completion lifetimes; AQL packet encoding; architecture PM4 command builders. | Model scheduling, dispatch-DAG policy, or backend admission. |

`rdna-compute::replay::ReplayController` is the product integration controller held by `Gpu`. The model adapter defines the eligible forward boundary and the model-owned state/lifetimes; the controller records, prepares, routes, poisons, and resets. This integration role does not turn `rdna-compute` into a fourth Redline transport crate.

### Mechanisms that must remain distinct

| Mechanism | Submission shape | What it proves | What it does not prove |
|---|---|---|---|
| Ordinary serial HIP | One `hipModuleLaunchKernel` path per launch. | Baseline model execution. | Any retained route. |
| HipGraph | HIP stream capture followed by a HIP graph launch. | HIP graph capture/replay for an eligible forward. | Retained AQL or retained PM4. |
| Per-dispatch retained AQL | Many public-HSA kernel-dispatch AQL packets on one retained queue. | AQL packet preparation and replay when positively routed. | A single retained PM4 submission. |
| Retained PM4-IB | One vendor AQL packet points to a retained architecture PM4 indirect buffer. | Retained PM4 only when preparation and observed replay are proven. | Direct-KMD dispatch. |
| Experimental `crates/redline` | Direct DRM command submission outside HIP/ROCr serving. | Direct-KMD experimentation. | The current serving transport. |
| Launch fusion | Fewer or different kernels under any host path. | A changed device graph. | Redline, route selection, or a wall-time win. |
| Stable partial recorder fingerprint | A repeatable recorded subsequence. | Discovery progress. | A complete tape or installed route. |

## 3. Runtime lifecycle and the model boundary

### Model load and automatic default

After a successful model load, the daemon evaluates `gfx12_mq4r_redline_default` and calls `ReplayController::configure_model_default`. The exact automatic-default predicate is:

```text
gpu architecture name starts with "gfx12"
AND model arch_id == 6
AND pipeline parallelism == 1
AND tensor parallelism == 1
AND model extension is .mq4r, case-insensitive
```

An explicit `HIPFIRE_REPLAY_BACKEND` selection or `HIPFIRE_REPLAY_MANUAL_CAPTURE` selection overrides the model default. When the narrow model default applies and no transport override is present, the controller requests `Auto` with `Pm4Ib`. Otherwise the default backend is ordinary HIP. Explicit opt-in can exercise broader implementation capability, but opt-in availability is not certification.

A successful reconfiguration or model swap clears recorded launches, certified observations, prepared AQL/PM4 objects, fallback reason, and forward eligibility before admission restarts. This reset is required because model allocation identity is part of the replay contract.

### Eligible forward boundary

Only an ordinary sequential single-token continuation may arm, record, or consume the plain-AR tape. The model adapter must set a one-forward eligibility decision before the forward body.

The following calls must neither contaminate nor consume the plain-AR tape:

- model load and initialization;
- prefill;
- speculative reseed, proposal, or verification;
- MTP reseed, proposal, or verification;
- a graph-capture or batched path with a different mutable-state contract;
- model swap or allocation reconstruction;
- any other non-sequential call.

In the current Qwen adapter, token embedding and the position-buffer host-to-device update stay outside the retained layer-stack body. That boundary is valid because it is explicit and counted. It is not permission to omit arbitrary raw launches.

HipGraph and Redline share the plain-AR eligibility decision but are separate backends. While Redline is enabled and not in sticky fallback, the Qwen HipGraph path is mutually excluded. After preparation failure poisons Redline, later eligible forwards may use the HIP-side policy, which can include HipGraph.

### Automatic product lifecycle

```text
successful eligible model load/default or explicit Auto
    ▼
Armed
    │ begin_auto_capture_if_armed on the first eligible continuation
    ▼
RecordingWarmup
    │ run the ordinary HIP layer stack and record exact launches
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

The automatic Qwen product path goes directly from `Captured` to `Ready` when preparation succeeds. It does **not** traverse `ShadowValidated`, call `observe_shadow`, or call `install_prepared_plan`.

### Manual controller and certification lifecycle

The separate manual controller API can collect `ShadowValidation` observations. Two accepted observations can move it to `ShadowValidated`; `install_prepared_plan` then moves a non-Shadow request to `Ready`. Manual capture alone ends at `Captured` and changes no launch route.

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

The controller's internal `1.03` shadow field is not a durable contributor promotion threshold and is not used by the automatic Qwen admission path. Each campaign must state its promotion rule in advance.

### Exact fail-closed semantics

| Failure phase | What has already happened | Current forward | Later forwards | Required wording |
|---|---|---|---|---|
| Preparation after automatic warmup | The eligible ordinary-HIP warmup forward completed successfully and produced its output. | Returns success for that already-completed HIP forward; the controller is poisoned. | Sticky `Fallback`; retained routing is disabled. Eligible HIP-side policy may include HipGraph. | “Preparation failed after a successful HIP warmup; later forwards fall back.” |
| Retained replay execution | A prepared AQL or PM4 route was selected and execution returned an error. | Controller is poisoned and the forward returns an error. **There is no same-forward HIP retry.** | Sticky `Fallback`; retained routing remains disabled. | “Replay failed; this forward errored; later forwards fall back.” |
| Manual shadow observation | A manual observation failed parity, ABI, timing, or its configured observation gate. | No product-route promise. | Controller enters sticky fallback. | “Manual shadow validation failed.” |
| Recorder capacity or capture contract | Capture could not remain valid. | The controller fails closed rather than installing a partial route. | Sticky fallback until reset. | Name the exact capture failure. |
| Successful model load/reconfiguration | A new model/allocation identity has replaced the old one. | Old retained objects are not reusable. | Tape, prepared queues, command buffers, and fallback state are reset; admission starts over. | “Model reconfiguration invalidated the old plan.” |

Never describe a replay-execution error as “the same token retried through HIP.” That behavior does not exist.

## 4. Retained tape, ABI, lifetime, and dynamic-state contract

A retained route is valid only when all of the following invariants hold.

### Launch and kernarg identity

- The recorder copies the exact naturally aligned, tail-padded kernarg bytes used by HIP. It must never retain a pointer to a caller stack frame or temporary argument array.
- The loaded HSA code object that owns `{kernel}.kd` must match the captured launch symbol and loader metadata.
- Runtime-specialized launch names and aliases must resolve to the exact owning artifact. An absent or ambiguous alias blocks preparation; it is not permission to pick a similarly named HSACO.
- Unsupported scratch, implicit SGPR, loader-kernarg, workgroup, grid, shared-memory, or symbol metadata blocks installation before `Ready`.
- Artifact path, artifact digest, symbol, loader kernarg size/alignment, padded blob, grid, block, and shared memory are one identity contract. A stable kernel-name hash alone is insufficient.

### Geometry and dynamic bindings

- Capture-time geometry is immutable unless a named replay binding explicitly defines a bounded patch surface.
- Current position-derived grid narrowing uses `ReplayGridBinding::PositionCeilDiv`; the recorded grid remains a hard maximum.
- Dynamic scalar or pointer fields must be named, offset-bounded, type-sized, and associated with one owner. Hidden mutation of arbitrary kernarg bytes is forbidden.
- A dynamic grid can be certified through a bounded binding. A dynamic block dimension or changing shared-memory assumption requires a replay-stable fixed/tiled design, not reuse of capture-time values.
- Multi-queue phases must not share a dynamic patch key whose meaning depends on one global capture order.

### Resources, dependencies, and state

- Allocation-wide reads and writes are explicit. Unknown effects remain serialized.
- The initial plan preserves all dependencies conservatively. Independence must be proven before waits or acquires are removed.
- The shadow oracle must cover logits, KV cache, recurrent state, convolution state, guard regions, exact captured blobs, and every other mutable model-specific state touched by the forward.
- Model reset, request reset, and model swap must reset or invalidate all retained mutable-state assumptions.
- Incomplete recorder coverage is not a smaller valid tape unless every omitted launch belongs to a named external adapter boundary and launch accounting reconciles exactly.

### Lifetime

Prepared AQL packets and PM4 commands contain device pointers. The model allocations, scratch buffers, KV/recurrent state, loaded code objects, kernarg storage, queues, command memory, and completion objects they reference must retain identity and lifetime until the plan is destroyed. Allocation teardown, rebinding, model swap, or incompatible topology change invalidates the plan before any further replay.

## 5. Reproducible model and architecture porting recipe

Do these stages in order. Do not optimize waits or kernels while the route contract is still moving.

| Stage | Required action | Binary exit criterion | Stop gate |
|---:|---|---|---|
| 1. Freeze the fixture | Record UTC date, branch/commit, clean state, binary digest, GPU/ROCm identity, model path/digest, architecture id, quantization, topology, KV mode, continuation API, exact prompt/token stream, and baseline route. | Another contributor can select the same bytes, binary, model, device, and route. | Any identity field is missing or the baseline is not coherent. |
| 2. Define admission | Write one fail-closed predicate for an ordinary sequential generated-token forward. Preserve explicit negatives for prefill, spec/MTP, graph capture, batching, model swap, and incompatible topology. | Positive and every negative case are observable; no ineligible call can record or consume the tape. | The boundary depends on incidental call order or an unrecorded heuristic. |
| 3. Census the full compute body | Use rocprof or an equivalent complete kernel trace. Name launches intentionally outside the tape. | `compute launches = retained dispatches + explicitly external launches` for each certified position. | Counts do not reconcile or a supposedly external launch mutates retained state. |
| 4. Migrate recorder coverage | Route every in-body raw launch through the typed recorder while preserving the ordinary HIP call and its result. | Retained count equals the reconciled target and ordinary-HIP outputs remain unchanged. | Any raw in-body launch is missing, duplicated, or reordered. |
| 5. Prove tape stability | Compare launch count, unique-kernel set, ordered sequence hash, geometry, and owning artifact identity across positions and fresh processes. | All immutable fields are stable; every intended dynamic difference has a named binding. | A stable partial subsequence, position-dependent unnamed mutation, or unexplained fresh-process drift. |
| 6. Bind artifact and ABI | Resolve the exact loaded HSACO and symbol; validate loader metadata, padded kernargs, geometry, shared memory, resources, and every dynamic field. | Every launch passes the ABI/artifact probe with no alias ambiguity or unsupported contract. | Missing/wrong HSACO, loader incompatibility, scratch/implicit-SGPR uncertainty, or kernarg mismatch. |
| 7. Stabilize geometry | Fix/tile changing block/shared-memory shapes; use only certified bounded patches for dynamic grids/scalars. | All supported positions fit the recorded maximum and produce the intended launch shape. | Replaying a capture-time block/shared-memory shape at a different context. |
| 8. Build the state oracle | Add reset/prime, logits, KV, recurrent/convolution state, guard, and snapshot/restore support needed for multi-position comparison. | HIP, exact HIP-kernarg-blob, and candidate retained execution compare every mutable state surface. | A touched state surface cannot be observed or restored. |
| 9. Lower conservatively | Start with one queue, explicit dependency waits, architecture-correct register encoding, and conservative acquire/fence policy. | A prepared route reaches `Ready`, executes, completes, and passes parity without hazard elision. | Unsupported architecture, unresolved dependency, queue/completion lifetime failure, or parity fault. |
| 10. Certify and only then promote | Pass the complete ladder in Section 7 and archive the benchmark record in Section 8. | Positive route proof, serve health, matched stationary performance, long-context state, reset, model-swap, and failure-path gates all pass. | A capture fingerprint, microbenchmark, silent fallback, or unmatched speed ratio is the only positive evidence. |

A typical diagnostic sequence for a known supported Qwen fixture is:

```bash
HIPFIRE_REPLAY_MANUAL_CAPTURE=1 \
HIPFIRE_REPLAY_BACKEND=shadow \
python3 scripts/redline_daemon_harness.py \
  --model "$MODEL" --skip-prefill --pm4 --shadow-iterations 15

python3 scripts/redline_product_bench.py \
  --model "$MODEL" \
  --daemon target/release/examples/daemon \
  --context 128 --iterations 100 --warmups 3 --runs 10 \
  --transport pm4 --max-seq 2048 \
  --work-dir .redline-work/product \
  --out .redline-work/product/report.json
```

Set `MODEL` to the exact digested fixture before running. The manual harness is a fingerprint/shadow diagnostic; it does not establish that the user-facing product arm selected retained PM4. The product benchmark must therefore be paired with the explicit route-proof record below.

## 6. PM4 lowering and hazard policy

`Pm4Architecture::from_device` selects distinct gfx10, gfx11, and gfx12 lowering. Gfx10/gfx11 and gfx12 do not share one universal register recipe. Register addresses, dispatch setup, acquire behavior, cache policy, and supported phase synchronization must be derived for the selected architecture and checked against current runtime support.

The first accepted route is deliberately boring:

1. one queue;
2. original launch order;
3. explicit waits for recorded dependencies;
4. conservative entry, intermediate, and terminal acquire/fence policy;
5. one clear completion boundary;
6. no CU-mask, register-policy, or phase-parallel overlay.

Only remove a wait or acquire when both conditions hold:

- an exact resource argument proves there is no producer/consumer, write/write, cache-visibility, or completion dependency across the boundary; and
- multi-position parity, guard checks, serve behavior, and stationary PM4-versus-PM4-plus-change performance all pass.

Sibling kernels are not independent merely because they have different names or output pointers. Consider allocation-wide aliasing, shared scratch, indirect addressing, KV/recurrent state, and terminal visibility. Unknown access metadata means serialize.

Multi-queue or phase parallelism is a later optimization. It additionally requires:

- a dependency-derived phase partition;
- explicit cross-queue fan-in before any consumer or terminal read;
- completion and queue lifetimes that cover every phase;
- dynamic bindings whose keys are unambiguous per phase/queue;
- architecture-supported synchronization; and
- PM4-base versus PM4-plus-multi-queue evidence.

Teach the dependency argument, not a copied packet sequence. A valid gfx1201 packet trace is not automatically a valid gfx1100 or gfx1151 lowering.

## 7. Certification and route-proof ladder

Pass these gates in order. Failure at a gate blocks promotion even if a later-looking metric is attractive.

| Gate | Required evidence | Rejection condition |
|---:|---|---|
| 1. Baseline correctness | Ordinary HIP produces stable, coherent output on the exact fixture. | Baseline is unstable, incoherent, or fixture identity is incomplete. |
| 2. Capture completeness | Compute count, external-launch count, retained count, unique kernels, ordered sequence hash, and fresh-process stability reconcile. | Missing/extra launch, partial capture, unexplained drift, or unowned external boundary. |
| 3. ABI/artifact validation | Every symbol resolves to the exact loaded artifact and loader metadata; padded blobs, geometry, effects, and dynamic bindings validate. | Wrong/missing artifact, ambiguous alias, unsupported ABI/resource contract, or hidden mutation. |
| 4. Multi-position shadow parity | Ordinary HIP, retained PM4, and the exact HIP-kernarg-blob oracle agree for logits, KV, recurrent/model state, guards, and captured blobs. | Any unexplained state mismatch. Bit exactness is required for an execution-preserving route; a tolerance is allowed only for an explicitly changed numerical kernel contract and must be justified. |
| 5. Route proof | Backend request, transport, successful preparation, `Ready`, observed replay at multiple positions, dispatch/packet/queue/dword identity, sequence hash, no replay fault, and fallback reason are recorded for every arm. | Timed arm may have silently used HIP/HipGraph, or reports only capture/`Ready` without observed replay. |
| 6. Production serve | User-facing generation with exact model/settings has healthy decoded output, finish state, repetition/attractor behavior, and model-specific framing. | Emitted tokens without semantic/framing health, runaway output, empty response, or route ambiguity. |
| 7. Stationary matched performance | Certified baseline and retained route use identical binary, model, prompt/token stream, settings, process policy, and clocks; tok/s and ms/token are reported. | Cross-harness, cross-binary, cross-prompt, nonstationary, silent-fallback, or microkernel-only comparison. |
| 8. Long-context and lifecycle | Dynamic position, geometry, KV growth, recurrent/convolution state, request reset, failure behavior, and model swap pass throughout the supported range. | Context drift, state leak, stale pointer, wrong reset, or same-forward fallback claim. |

Harness success without route proof is insufficient. A ratio from two nominal arms that both executed ordinary HIP is invalid evidence. A stable manual-capture fingerprint is discovery evidence; it does not install a plan or prove that a user-facing forward selected retained AQL or PM4.

### Minimum route-proof record per arm

| Field | Required content |
|---|---|
| Request and selection | Backend request, transport, exact eligibility predicate/result, and negative gates active. |
| Preparation | Capture summary; preparation success; controller state; fallback reason. |
| Tape identity | Dispatch count, external count, unique-kernel count, ordered sequence hash, owning artifacts, geometry, and dynamic bindings. |
| Submission identity | Observed transport, packet count, queue identity/count, phase count, PM4 command dwords, and completion. |
| Observation span | At least multiple generated positions, with positions named. |
| Fault evidence | Replay error/fault absence, guard status, and any controller poison/reset event. |
| Competing routes | Evidence the measured arm did not execute ordinary HIP or HipGraph. |

## 8. Benchmark record schema and claim language

Every report supporting a retained-replay claim must contain the following schema.

| Group | Required fields |
|---|---|
| Time and source | UTC date/time; branch; source commit; clean/dirty state; exact daemon/binary digest. |
| Device | GPU product; PCI identity; gfx architecture; visible-device/topology selection; ROCm, runtime, and driver identity; clock/governor policy. |
| Model | Full model path; model architecture id; quantization; artifact size/digest; sidecar names and digests. |
| Workload | Harness path/revision; full command; prompt bytes or deterministic token-stream path and digest; sampler and seed; KV mode; context; prefill/generated counts; parallel topology; graph/spec/MTP settings. |
| Configuration | Every route-affecting `HIPFIRE_*` variable and whether it was unset; explicit backend and transport request. |
| Sampling | Warmup policy; fresh-process or resident-daemon policy; run order; run count; per-run raw values; minimum, median, and maximum; raw report path. |
| Tape | Compute/external/retained counts; unique kernels; sequence hash; owning artifacts; geometry; resource effects; dynamic bindings. |
| Route proof | Preparation success; `Ready`; observed replay positions; transport; packets; queues/phases; PM4 dwords; fault absence; fallback reason; proof against silent HIP/HipGraph. |
| Correctness | Shadow/oracle report; logits/KV/recurrent/guard/blob results; tolerance and rationale if not bit exact; decoded-output health. |
| Result | Tok/s and ms/token for each arm; ratio and absolute delta; predeclared promotion rule; disposition. |

### Comparison rules

- Use byte-identical prompts or deterministic token streams, the same binary, model, topology, KV mode, context, generation length, sampler, and clocks.
- State whether each sample used a fresh process or a resident daemon. Do not mix policies within an A/B.
- Interleave run order when practical and archive raw values; report minimum/median/maximum rather than one best run.
- Automatic clocks are the default unless clock policy itself is the experiment.
- Compare both tok/s and milliseconds per token.
- Numbers from different harnesses, binaries, prompts, or route-proof states are not one A/B.
- A dated number is evidence for its exact fixture, not a permanent minimum speedup or current product guarantee.

### Direct value versus enabling value

Redline has two distinct performance roles:

1. **Direct transport value:** one retained submission can remove variable host launch and packet-publication work. Measure this with a matched baseline-versus-PM4 comparison in which the device kernel stack is otherwise identical.
2. **Enabling value:** reducing the host floor and host-side variance makes later device-side kernel, traffic, fence, overlap, and composite transformations easier to resolve and compose. Measure each later lever as base PM4 versus PM4 plus that one lever.

Do not merge these claims. In particular, the historical Qwen MQ4R progression from roughly 110 to 204 tok/s combined changing kernel stacks, graph shape, and Redline work. It is useful dated engineering history, not a pure PM4 transport A/B. The direct PM4 A/B must remain the matched, fixed-stack comparison recorded for its own checkpoint.

## 9. Post-Redline kernel and Radiowave loop

After the conservative retained route passes Section 7:

1. Freeze a stationary retained-PM4 baseline and archive its complete record.
2. Profile the new device-bound timeline rather than assuming host launch cost still dominates.
3. Choose one device-side lever: a kernel change, fusion, traffic reduction, fence/acquire policy, queue overlap, or Radiowave/composite transformation.
4. Keep the base tape, fixture, binary inputs, route proof, and all unrelated overlays fixed.
5. Compare **base PM4** against **PM4 plus exactly that overlay**.
6. Retain the overlay only if state parity, route proof, serve health, long-context behavior, and reproducible wall time all pass.
7. Compose independently proven overlays one at a time; refresh the stationary PM4 baseline after each accepted change.
8. Rerun the full gates after any hazard-policy, geometry, artifact, dynamic-binding, or mutable-state boundary change.

Launch-count reduction alone is not proof of a wall-time win. Under retained PM4, fusion earns value by reducing device work, memory traffic, or synchronization—not by claiming avoidance of host API calls that the retained transport already removed. Radiowave transformations belong in this post-certification loop, not in the definition of Redline.

## 10. Failure atlas

| Symptom or mistake | Failure phase / diagnosis | Required response |
|---|---|---|
| Capture starts during model load or prefill. | Admission boundary is wrong; setup work contaminated the ordinary-AR tape. | Reset, move arming to the first eligible continuation, recapture, and rerun every gate. |
| A raw in-body launch is absent from the recorder. | Capture completeness failure. | Migrate it to the typed recorder or formally place it at an external adapter boundary and reconcile counts. |
| Recorder retains stack-backed kernargs. | ABI/lifetime failure; replay can dereference invalid bytes or pointers. | Copy exact padded bytes into owned storage; invalidate the tape. |
| HSACO is missing, wrong, or loader-incompatible. | Artifact/ABI preparation failure. | Resolve the exact owning artifact and metadata; fail before `Ready`. Do not substitute by name similarity. |
| Specialized launch alias resolves to another artifact. | Launch/artifact identity failure. | Correct and prove the alias mapping; regenerate fingerprint and parity evidence. |
| Capture-time block or shared-memory shape is replayed at a new context. | Geometry contract failure. | Fix/tile the shape or add a bounded supported binding; recertify the range. |
| Token, KV, recurrent, or convolution state is absent from the oracle. | State-certification gap. | Add snapshot/restore and comparison before interpreting parity or speed. |
| Model swap retains pointer-keyed state. | Lifetime/reset failure. | Destroy prepared objects and reset admission on every successful reconfiguration. |
| Scratch, implicit SGPR, or loader-kernarg contract is unsupported. | Preparation cannot safely lower the launch. | Reject installation until the runtime can validate the exact contract. |
| A wait/acquire is removed because launches “look independent.” | Hazard-policy failure. | Restore conservative ordering; produce an exact resource argument and multi-position evidence before retrying. |
| Timed `auto` and `hip` arms both execute HIP. | Route-proof failure. | Discard the ratio; instrument request, state, transport, packets/queues/dwords, replay positions, and fallback. |
| Replay execution fails and is described as same-token HIP fallback. | Failure-phase misreport. | Report the current forward error and later sticky fallback exactly. |
| Stable partial capture is called a complete route. | Capture classification failure. | Reconcile full compute, external, and retained counts; do not prepare or promote the partial tape. |
| Launch graph shrinks but wall time does not improve. | Structural result without product value. | Keep it experimental or reject it under the campaign rule; do not market it as a speedup. |
| Results from different harnesses, binaries, prompts, or clocks are compared. | Benchmark comparability failure. | Rerun a stationary matched experiment; keep old rows only as separate dated evidence. |
| Manual capture reports a stable fingerprint and is called routed PM4. | Discovery evidence was promoted beyond its meaning. | Add preparation, `Ready`, observed multi-position replay, submission identity, and anti-fallback proof. |
| Production framing fails while numerical replay parity passes. | User-facing contract failure may be framing rather than transport corruption. | Diagnose framing separately, preserve the parity classification, and block product promotion until serve health passes. |

## 11. Worked cases

The following cases use the same classification fields. Historical measurements are quoted only with their dates and fixtures.

### 11.1 Qwen3.5 0.8B dense `.mq4` on gfx1201: positive explicit opt-in

| Field | Evidence |
|---|---|
| Intent | Establish a compact dense-model capture, exact-state parity, and retained-PM4 example. |
| Baseline route | HipGraph ordinary AR for the dated matched transport comparison. |
| Candidate route | Explicitly selected retained PM4. This `arch_id=5`, `.mq4` model is not the automatic `.mq4r`, `arch_id=6` product default. |
| Fixture | Qwen3.5 0.8B dense on gfx1201; dated 2026-07-11 in `crates/redline-dispatch/HIPFIRE-GRAFT.md`. The recoverable row does not pin the full model digest or daemon binary digest. |
| Immutable contract | 356 retained dispatches, 21 unique kernels, ordered sequence hash `55f99a58cb4b9363`; exact artifacts/kernargs and ordinary-AR state. |
| Validation | Fifteen consecutive positions were bit exact for logits, KV, and recurrent state against ordinary HIP and the exact HIP-kernarg-blob oracle. |
| Route proof | Dated retained-PM4 evidence includes the fingerprint, PM4 transport, multi-position parity, and a matched HipGraph-versus-PM4 product comparison. Per-dispatch AQL was measured separately and was neutral. |
| Matched performance | On 2026-07-11, automatic clocks, resident 10×100 comparison: HipGraph 363.682 tok/s versus PM4 392.248 tok/s, `1.07855×`. This is fixture-bound evidence, not a floor. |
| Serve | A five-prompt greedy battery had no runaways and averaged 384.7 decode tok/s; a response-framing gate failed independently. Numerical route parity must not be relabeled as framing success. |
| Disposition | Positive retained-PM4 certification example for explicit opt-in; **not** automatic-default admission. |
| Reusable lesson | Capability, opt-in certification, and product default are separate classifications. |

Evidence limitation: the recoverable graft row lacks the full artifact/binary identity now required by Section 8. Use it as a known-good historical example, not as a complete modern benchmark manifest.

### 11.2 Qwen3.6 35B-A3B MQ4R across gfx1100, gfx1151, and gfx1201

The current automatic predicate is narrower than implementation capability: gfx12, `arch_id=6`, single GPU (`pp=tp=1`), and `.mq4r`. Explicit backend/transport selections can request broader implemented paths, but they do not confer certification.

| Architecture | Implementation capability | Model performance evidence | Explicit opt-in | Positive retained-PM4 certification in recoverable docs | Automatic default |
|---|---|---|---|---|---|
| gfx1100 | Yes: gfx11 PM4 lowering exists. | Yes: the dated README/CHANGELOG MQ4R row reports TG128 median 253.3 tok/s, route unspecified. | Available. | **No positive retained-route proof recovered.** A stationarity-test comment is insufficient. | No; runtime tests reject gfx1100 for the gfx12 default. |
| gfx1151 | Yes: gfx11 PM4 lowering and gfx1151 experiment knobs exist. | Yes: the dated README/CHANGELOG MQ4R row reports TG128 median 115.1 tok/s, route unspecified. | Available. | **No positive retained-route proof recovered.** Experiment knobs are capability, not certification. | No. |
| gfx1201 | Yes: gfx12 PM4 lowering and Qwen adapter. | Yes, with dated model and serve checkpoints. | Available. | **Yes**, including retained-PM4 route proof and multi-position parity. | Yes when the complete narrow predicate holds and no explicit override wins. |

Primary gfx1201 case:

| Field | Evidence |
|---|---|
| Intent | Certify and productize the single-GPU ordinary-AR Qwen3.6 35B-A3B MQ4R retained-PM4 route. |
| Baseline route | HipGraph ordinary AR for the initial fixed-stack transport comparison. |
| Candidate route | `redline-dispatch`/`redline-rocr` PM4-IB through the Qwen adapter. |
| Fixture | `qwen3.6-35b-a3b.mq4r`, `arch_id=6`, Q8 KV, no MTP/DFlash, gfx1201 Radeon AI PRO R9700; see `docs/perf-checkpoints/2026-07-11-redline-qwen36-a3b-ar.md`. |
| Immutable contract | Initial tape: 833 launches, 26 kernels, sequence hash `8d5620ca2ca8a536`; PM4 body 34,563 dwords; one vendor AQL packet. Later graph reshapes have checkpoint-specific fingerprints and must not reuse this hash. |
| Validation | Fifteen-position bit-exact logits hash `9874244965e2c7d6`, KV hash `fa5f3bb2b32fffcd`, recurrent hash `609db41ffad8ceb6`; `bit_exact`, `blob_bit_exact`, logits/KV/recurrent equality all passed. |
| Route proof | Positive gfx1201 PM4 preparation, one-packet submission identity, command dwords, matched product arm, and multi-position shadow evidence are recorded in the 2026-07-11 checkpoint. |
| Matched performance | On the initial 2026-07-11 fixed-stack fixture, HipGraph 165.839 tok/s versus PM4 178.320 tok/s, `1.07526×`, after independently justified boundary-wait removal. Dated, not a universal floor. |
| Later progression | `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md` records the later productized no-env campaign, including a TG128 median 203.93 tok/s and long-turn serve evidence. That progression changed kernel stack and graph shape; it is not the direct transport A/B above. |
| Disposition | Positive retained-PM4 certification and automatic default only under the narrow runtime predicate. |
| Reusable lesson | Separate architecture capability, model performance, opt-in availability, retained-route certification, and automatic admission in every matrix. |

Evidence limitations: initial raw `.redline-work` JSON is referenced by dated markdown but is not checked into the repository. The initial checkpoint also records a composition/transplant provenance caveat; later product measurements supersede it for the productized path but do not turn the full historical progression into a pure transport experiment. The recoverable gfx1100 and gfx1151 performance rows do not state whether HIP, HipGraph, or retained PM4 was selected.

### 11.3 Rejected gfx1030 Qwen3.6 MQ2 lowering

| Field | Evidence |
|---|---|
| Intent | Port Qwen3.6 35B-A3B MQ2G256Lloyd prefill and retained-PM4 decode to an RX 6950 XT gfx1030 and earn a product wall-time win. |
| Baseline route | Ordinary HIP decode. |
| Candidate route | Product `auto` with `transport=pm4`, plus a Radiowave off/on overlay. |
| Fixture | Branch `feat/mq2g256-gfx1030-prefill-redline`; prefill commit `0f3444f8cecf9976ced483237a8fc26028f3b94d`; measured candidate `e017f83ceb9d41d4be0d6665161615c9ae74d89b`; daemon SHA-256 `47585859295f44a5cc2aab090e7fc43ef342d2932e1d7d402a4d569cbf53acaf`; model SHA-256 `48b3f84614c46eb8b5ffb494f7a75c15216664afcbb47c3e78dd80c4ce7eb0a3`; dated 2026-07-18 on host hipx. |
| Immutable contract | Launch identity/order, owning HSACO, exact padded kernargs, effects, bindings, capture boundary, output/state parity, and positive product-arm route proof. |
| Validation | Exact harness off/on reported 942 launches, 24 kernels, sequence hash `becff4a4f1849d1e`, one PM4-IB packet, 21,783 command dwords, and bit-exact logits/KV/recurrent/blob parity. Owning artifact paths and kernel names matched. Intermediate `HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS` HSACO-load errors occurred during bring-up but were not the final exact-report failure. |
| Falsification | Radiowave-on changed every captured kernarg hash, 942/942, despite stable launch-name sequence and artifact paths. This clobbered the exact padded-kernarg/dynamic-binding tape surface for the overlay comparison. |
| Route-proof limitation | Exact shadow proves that PM4-IB could execute bit exactly. Product JSON records `transport=pm4` but omits controller `Ready`, fallback reason, and observed multi-position replay fields, so it does not independently exclude silent HIP for the timed arm. |
| Matched performance | Same e017 binary/model, automatic clocks, ctx 32, 32 iterations, 10 warmups, 8 runs: Radiowave-off HIP 101.431 versus auto 83.711 tok/s (`0.82529×`); Radiowave-on HIP 101.460 versus auto 83.653 (`0.82449×`). The candidate was about 17.5% slower. |
| Disposition | **Rejected** as a retained-route/Radiowave product promotion. A commit-message or alternate-bench `+3.1%` is not the matched product result. |
| Reusable lesson | Stable sequence hash plus bit-exact shadow can hide full kernarg mutation and cannot replace product-arm route proof plus matched wall time. |

Authoritative raw evidence is recoverable under `ssh://hipx/home/kaden/redline-results/gfx1030-radiowave-ab-20260718/` and the corresponding `gfx1030-radiowave-precontract-ab-20260718/` directory. No checked-in performance checkpoint defines an official single-enum rejection label. Therefore this guide states only the established clobber surfaces: universal Radiowave kernarg-hash drift and the matched product wall regression. It does not claim that final HSACO identity, launch order, or output parity failed; the final exact reports show the opposite.

### 11.4 Rejected LFM Stage A

| Field | Evidence |
|---|---|
| Intent | Reduce LFM2.5-350M MQ4 gfx1201 serial-HIP decode launches by fusing RMSNorm plus MQ rotation activation preparation. |
| Baseline route | Serial HIP lowered decode, fusion disabled, graph off, Q8 KV. |
| Candidate route | Serial HIP with `HIPFIRE_LFM2_DECODE_FUSION=1`; not Redline, AQL, or PM4. |
| Fixture | Baseline `e8831ae8347f04ac821077ee159c86423b4bf88a`; candidate `518c221756a1065a7560449165bc8817c2ad6176`; model `lfm2.5-350m.mq4` MD5 `cb5284b8ad5c6f9e4ca859c0aff0bcd0`; dated 2026-07-19. See `docs/design/lfm2moe-gfx1201-decode-architecture.md` and the archived measurement ledger named in the campaign. |
| Immutable contract | Bit-exact production decode; fail-closed non-admission on prefill/spec/graph/capture/default; campaign structural targets; predeclared campaign wall gates. PM4 was an explicit non-goal. |
| Validation | Correctness and five-prompt serve checks passed. Rocprof count moved 281→221. Recorder-visible tape was 204 launches, 9 kernels, hash `67dcc9e17e00ed8f`. |
| Launch reconciliation | Baseline recorder target 264 = 281 compute − 1 embedding − 10 direct-HIP conv − 6 direct-HIP attention. Stage A tape 204 = 221 − 1 − 10 − 6. The value 220 = 221 − 1 is a **future** full-tape target after conv/attention recorder migration, not a Stage A pass bar. |
| Route proof | Absent by design: no PM4 command, shadow, prepared plan, `Ready`, or observed replay was run. The LFM harness reached a Qwen-only shadow endpoint and could not install an LFM route. |
| Matched performance | Authoritative fresh-process ABBA: tg128 `+2.114%`, tg512 `+1.041%`; both missed that campaign's predeclared `≥5%` wall gates, and low-sample guards also failed. The 5% value belongs to this dated campaign, not a timeless Redline floor. |
| Disposition | **Rejected** as a standalone Stage A promotion despite exactness and structural launch reduction. Classified as serial-HIP activation-preparation fusion, not Redline. |
| Reusable lesson | Fewer launches are neither retained replay nor a wall-time win. Complete recorder coverage, replay-stable attention geometry, LFM mutable-state shadow support, prepared-plan installation, and positive PM4 route proof remain prerequisites. |

## 12. Copyable new-route checklist

Copy this checklist into the route's dated evidence record.

### Admission and boundary

- [ ] The exact positive admission predicate names model, architecture, quantization, topology, continuation shape, and route.
- [ ] Explicit negative gates cover prefill, spec/MTP, batching, graph/capture conflicts, model swap, and every non-sequential call.
- [ ] Compute launches, external launches, and retained dispatches reconcile exactly.
- [ ] Every external adapter launch is named with a state/lifetime justification.

### Tape, ABI, and state

- [ ] Every in-body launch uses the typed recorder while preserving ordinary-HIP behavior.
- [ ] Count, unique-kernel set, ordered sequence hash, geometry, and owning artifact identity are stable across positions and fresh processes.
- [ ] Each symbol resolves to the exact loaded artifact and loader kernarg metadata.
- [ ] Exact padded kernarg bytes are owned; no stack-backed argument storage survives capture.
- [ ] Resource reads/writes and dependencies are conservative and explicit.
- [ ] Each dynamic value uses a named, bounded binding; no hidden kernarg mutation remains.
- [ ] Reset/prime and snapshot/restore cover logits, KV, recurrent/convolution state, guards, and all model-specific mutable state.
- [ ] Model swap/allocation teardown invalidates every retained pointer and prepared object.

### Lowering and certification

- [ ] PM4 register, acquire, fence, and completion policy is correct for the selected architecture.
- [ ] The first route is single-queue and conservatively ordered.
- [ ] Ordinary HIP, exact HIP-kernarg-blob, and retained PM4 pass multi-position state parity.
- [ ] Route proof records request, transport, preparation, `Ready`, observed replay positions, dispatches, packets, queues/phases, dwords, faults, and fallback reason.
- [ ] The timed retained arm is proven not to be ordinary HIP or HipGraph.
- [ ] Production serve output, finish state, repetition/attractor health, and response framing pass.
- [ ] Dynamic position, growing context, request reset, failure behavior, and model swap pass.
- [ ] Stationary matched performance reports tok/s, ms/token, raw samples, and the predeclared disposition rule.
- [ ] Dated raw evidence has an immutable path and complete identity manifest.

## 13. Copyable reviewer checklist

- [ ] I can distinguish implementation capability, model performance, opt-in availability, retained-route certification, and automatic-default admission in every claim.
- [ ] I verified the current runtime predicate and state transitions against the source symbols below.
- [ ] Automatic `Captured → Ready` behavior is not mislabeled as automatic shadow certification.
- [ ] Preparation failure and replay-execution failure use the exact phase-specific semantics in Section 3.
- [ ] The full compute/external/retained launch equation reconciles; no stable partial tape is promoted.
- [ ] Artifact, kernarg, geometry, resource, binding, lifetime, reset, and model-swap contracts are explicit.
- [ ] Parity spans multiple positions and every mutable state surface, not logits alone.
- [ ] Positive route proof excludes silent HIP and HipGraph for every timed arm.
- [ ] Benchmark arms match binary, model, prompt/token bytes, topology, KV mode, clocks, harness, and process policy.
- [ ] Direct transport A/B is separate from any changing-kernel historical progression.
- [ ] Kernel/Radiowave/hazard overlays are compared as base PM4 versus PM4 plus one overlay.
- [ ] Launch-count reductions are not presented as wall-time wins without stationary evidence.
- [ ] Every throughput or ratio is dated and fixture-bound, never a timeless floor.
- [ ] Rejected cases preserve the established falsification and do not invent a missing failure.
- [ ] Raw reports and source/binary/model digests are recoverable.

## 14. Stable source-path and symbol index

Prefer these paths and symbols over line numbers; line numbers drift.

| Concern | Stable source path and symbols |
|---|---|
| Automatic product predicate | `crates/hipfire-runtime/src/config.rs` — `gfx12_mq4r_redline_default` |
| Model-load application and diagnostic handlers | `crates/hipfire-runtime/examples/daemon.rs` — load-time `configure_model_default`; `redline_capture`; `redline_shadow_aql`; `redline_shadow_pm4`; prefix/profile/probe handlers |
| Qwen model boundary and route | `crates/hipfire-arch-qwen35/src/qwen35.rs` — `forward_scratch`; `prepare_scratch_inputs`; calls to `set_forward_eligible`, `should_route_aql`, `should_route_pm4`, `finish_capture`, and `prepare_*` |
| Controller, tape, lifecycle, and routing | `crates/rdna-compute/src/replay.rs` — `ReplayController`; `ReplayState`; `RecordedHipLaunch`; `ReplayGridBinding`; `configure_model_default`; `reset_for_model`; `begin_auto_capture_if_armed`; `finish_capture`; `prepare_linear_aql_prefix`; `prepare_pm4_prefix`; `replay_linear_aql`; `replay_pm4`; `observe_shadow`; `install_prepared_plan`; `should_route_aql`; `should_route_pm4`; `poison` |
| Central HIP recording and artifact aliases | `crates/rdna-compute/src/dispatch.rs` — `Gpu::replay`; central typed HIP launch recording and owning-artifact alias map |
| DAG, identity, ABI, and visibility policy | `crates/redline-dispatch/src/lib.rs` — `Recorder`; `CompiledPlan`; `KernelArtifactIdentity`; `KernargAbi`; `derive_aql_visibility` |
| Retained AQL and PM4 graph objects | `crates/redline-dispatch/src/aql/replay.rs` — `SingleQueueBatchGraph`; `SingleQueuePm4Ib`; `PhasedMultiQueuePm4Ib` |
| Public ROCr/HSA ownership | `crates/redline-rocr/src/lib.rs` — ROCr symbols, device/queue/pool/kernarg ownership exports |
| PM4-IB vendor packet | `crates/redline-rocr/src/packet.rs` — `PacketImage::pm4_indirect_buffer` |
| Architecture PM4 builders | `crates/redline-rocr/src/pm4.rs`; `crates/redline-rocr/src/pm4_gfx10.rs` |
| Manual capture/shadow diagnostic | `scripts/redline_daemon_harness.py` |
| Product stationary comparison | `scripts/redline_product_bench.py` |
| Graft and ABI provenance | `crates/redline-dispatch/HIPFIRE-GRAFT.md`; `crates/redline-rocr/PROVENANCE.md` |
| Positive dated gfx1201 evidence | `docs/perf-checkpoints/2026-07-11-redline-qwen36-a3b-ar.md`; `docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md` |

A change to any referenced route, state, artifact, geometry, resource, or lifetime symbol requires rechecking the applicable certification gates. The guide should be updated when contributor procedure changes; dated case records should remain immutable evidence for their original fixtures.
