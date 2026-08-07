<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DeepSeek V4 harmonic execution on gfx1100 + gfx1151

Status: H0-H1 complete; H2 CPU protocol complete and the PCI-scoped worker
survivability oracle passed 11 forced deaths on the authorized gfx1010 guinea
pig. Stable HIP and ROCr PCI selection is source-complete, but the exact-pair
H2 hardware exit remains open. The process-local gfx1151 H4 expert-service
source seam and routed residency receipt are implemented but have not executed
a model or passed numerical parity. Independent H3/H4 work remains in
progress. GPU product execution remains prohibited.

Branch: `ds4-beta-staging`

This specification supersedes the forward-work portions of
[`2026-08-06-deepseek4-heterogeneous-gfx1100-gfx1151.md`](2026-08-06-deepseek4-heterogeneous-gfx1100-gfx1151.md).
The older document remains the historical record for the transport, loading,
correctness, and performance experiments already completed.

## 1. Replacement goal

Replace the unsafe stalled heterogeneous DeepSeek V4 Flash 0731 MQ2R route
with a fault-contained **harmonic execution** system:

- `gfx1100` owns routing, dense and shared compute, canonical state, and its
  own exact-architecture kernels and retained tape.
- `gfx1151` owns the resident routed-expert weights, routed-expert execution,
  and its already optimized exact-architecture expert kernels.
- The devices exchange only compact, typed, generation-counted activation,
  route, and routed-result packets.
- Neither device may wait indefinitely on progress that only the other device
  can make.
- A fault or loss of either device must not make the unaffected device
  unusable or require a host reboot.

The frozen ordinary-AR targets on the canonical fixture are:

| Gate | Target | Meaning |
|---|---:|---|
| T1 | at least 50 tok/s | native gfx1100 non-expert execution pays without requiring heroic overlap |
| T2 | at least 60 tok/s | useful gfx1151 expert work is substantially hidden by gfx1100 work |
| T3 | at least 62 tok/s | rate-matched exact-architecture retained execution approaches the all-gfx1100 counterfactual |

T1 is the minimum successful architecture. T2 is the product target. T3 is the
stretch target. These are measured results to pursue, not claims made by this
document.

No speculative decoding is in scope. DSpark may be reconsidered only after T2
ordinary AR is certified and only under a separately approved goal.

## 2. Why "harmonic" is a distinct execution model

A wavefront cannot span GPUs. Registers, LDS, caches, queues, and instruction
state are local to one agent. Harmony therefore occurs at the subgraph and
queue level, not inside one physical wavefront.

Ordinary heterogeneous execution merely uses different devices. Asymmetric
execution assigns them unequal work. Harmonic execution adds three mandatory
properties:

1. **Native voices.** Each device executes a work decomposition designed for
   its own CU count, occupancy, cache, memory, and launch behavior. Compiling a
   gfx1151-shaped HIP source for `gfx1100` is not sufficient.
2. **Rate matching.** Work placement minimizes the layer critical path rather
   than equalizing bytes, launches, or tensor count. The desired time is the
   maximum of concurrent branches, not their sum.
3. **Bounded resolution.** Every cross-device epoch either joins successfully
   or expires into an owner-correct recovery path. A hardware queue cannot be
   left in an unbounded wait on a peer-owned word.

The intended per-layer shape is:

```text
gfx1100  attention -> router/top-k -> dense/shared branch ------------> join
                                      |                                ^
                                      | compact route/activation       | partial
                                      v                                |
gfx1151                         resident routed experts ----------------'
```

The scheduler optimizes:

```text
T_layer = T_serial_gfx1100
        + max(T_shared_gfx1100, T_packet_out + T_experts_gfx1151)
        + T_packet_back
        + T_join
```

## 3. Frozen identity and acceptance fixture

Model and quality remain unchanged:

- Model: DeepSeek V4 Flash 0731 MQ2R, `arch_id = 9`.
- Artifact SHA-256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`.
- Routed experts: `MQ2G256Lloyd`.
- Dense/non-routed weights: `MFP4G32E8SOA`.
- Embed: `Q8_0`.
- Norms: `F16`.
- Experts per token: 6.
- KV request mode: Q8.
- Batch: 1.
- Sampling: greedy, temperature 0.

Canonical acceptance fixture:

- Prompt: `benchmarks/prompts/ds4_heterogeneous_code_2048.txt`.
- Prompt MD5: `593234a767e71b97a3a4dad6431b47ce`.
- Prompt tokens: 2,048.
- Generated tokens: 512.
- Expected decoded bytes: 2,491.
- Expected decoded MD5: `ee05ab4f07393fb7d624d966a7dde4af`.

No change to weights, quantization, expert count, sampling, prompt, KV policy,
or arithmetic may manufacture a performance result.

## 4. Measured basis and target model

Preserved evidence establishes:

- Single-gfx1151 MQ2R retained-PM4 AR is approximately 28.87-29 tok/s.
- The accepted direct-HIP split route reached 32.0029 tok/s after gfx1100
  attention overlap and grouped O-LoRA, but retained the unsafe transport.
- The apparent 33.3-33.6 tok/s route was five capped 16-token serving
  diagnostics, not the canonical 2,048/512 fixture.
- The sealed Qwen 3.6 35B-A3B MQ4R retained-PM4 rows are 251.798 tok/s on
  gfx1100 and 115.290 tok/s on gfx1151, a 2.184x architecture ratio for that
  resident workload.
- The measured DS4 routed expert gate/up plus down tier is approximately
  8.54 ms/token at the 2,048-depth profile.
- The selected G0 packet chain moved one approximately 16 KiB packet in each
  direction per layer and measured 0.589 ms for the complete 43-layer B=1
  chain.

At 29 tok/s, the token budget is 34.48 ms. A first-order decomposition is:

```text
expert tier on gfx1151                 8.54 ms
remaining work                        25.94 ms
remaining work / Qwen arch ratio      11.88 ms
```

Without overlap, that is a roughly 49 tok/s hypothesis before synchronization
and transport costs. Reaching 60-63 tok/s requires hiding approximately
half to three-fifths of the expert tier while keeping transport and joins
close to the measured sub-millisecond chain. H1 must replace this hypothesis
with an occurrence-weighted DS4 bill.

Forwarding expert weights to gfx1100 is prohibited as a product design. The
active expert tier is approximately 1.83 GB/token; at the measured peer path it
would dominate the token budget. The 72+ GiB expert payload remains resident
on gfx1151. Only activations, routing metadata, and routed partials cross the
link.

## 5. Safety invariants

These invariants supersede prior no-host-wait-at-any-cost rules:

1. The current reciprocal HIP signal path is quarantined and cannot be used by
   a product or performance harness.
2. No `hipStreamWaitValue32`, HSA barrier, PM4 wait, polling kernel, or other
   device-side dependency may wait forever on peer progress.
3. Cross-device buffers carry an owner, architecture, allocation generation,
   epoch, layer, slot, byte extent, and completion state. Foreign or stale
   generations fail closed before submission.
4. Teardown is bounded and ordered. `Drop` must not perform an unbounded stream
   or device synchronization.
5. GPU execution remains prohibited until synthetic kill/fault injection
   proves that terminating either worker leaves the other device usable.
6. After any kernel fault, timeout, bus loss, failed queue removal, or stale
   VRAM accounting, stop. Do not launch a diagnostic on either GPU and do not
   probe KFD repeatedly.
7. Single-gfx1151 DS4 and gfx1100 Qwen routes remain independent and unchanged.
8. No new policy is keyed only on a broad gfx11 family match.

The preferred containment boundary is one long-lived worker process per GPU,
each with an independent KFD process/PASID, supervised by a Rust controller.
If runtime constraints make that impossible, a single-process design must
provide equivalent unaffected-device recovery evidence before it can pass H2.

## 6. Typed route contract

Model routing, work routing, and transport routing are separate concepts.
The cross-device request is a typed protocol, not a collection of peer
pointers hidden inside `forward.rs`.

The logical request contains at least:

```text
RoutePacket {
    route_identity,
    model_identity,
    epoch,
    layer,
    slot,
    source_owner,
    destination_owner,
    allocation_generation,
    expert_ids[6],
    route_weights[6],
    activation_extent,
    result_extent,
    deadline,
}
```

Payload slots are persistent and double-buffered or deeper only when measured.
The control protocol uses monotonically increasing epochs with explicit stale,
cancelled, completed, and failed terminal states. No buffer is reclaimed until
its owner observes a terminal state or the worker process is isolated and
destroyed.

## 7. Roadmap and gates

### H0 - Preserve and quarantine - COMPLETE

- Preserve every accepted and rejected G0-G5 artifact and ledger row.
- Keep the accepted exact-gfx1100 attention-overlap and grouped O-LoRA commits
  reachable.
- Keep the shared-jobs product wiring reverted.
- Mark the reciprocal HIP product transport unavailable to harness and serving
  admission.
- Add static tests that reject reintroduction of the unsafe route.
- Record the two wedge chronologies and the collateral gfx1151 mechanism.

Exit: source and tests make accidental execution of the unsafe product path
impossible. No GPU execution is required or permitted.

### H1 - Critical-path bill and native ownership map - COMPLETE

- Use the preserved canonical 2,048/512 trace; do not rerun merely to rebuild
  already durable data.
- Classify every occurrence into serial gfx1100 work, overlap-eligible gfx1100
  work, gfx1151 expert work, transfer, or required join.
- Record source bytes, dispatches, achieved bandwidth, architecture target,
  and whether the implementation is truly native or only exact-compiled.
- Produce explicit budgets for 50, 60, and 62 tok/s: 20.000, 16.667, and
  16.129 ms/token.
- Identify the minimum overlap required after measured native gfx1100 costs.

Exit: the roadmap has an occurrence-weighted critical path and no unpriced
"unknown equals zero" term.

### H2 - Fault-contained transport and lifecycle

- Build the bounded host-supervised transport first as the correctness oracle.
- Build isolated per-device workers with persistent resources and typed packet
  slots.
- Reuse the selected public ROCr SDMA mechanism where it remains correct, but
  do not inherit its old cross-device wait policy.
- Measure controller wakeups and eliminate per-layer host round trips only
  after containment passes.
- Inject worker exit, timeout, stale epoch, malformed owner, mid-copy cancel,
  and producer loss in both directions.
- After each injection, prove the unaffected GPU can run its existing
  single-device oracle without reboot or reset.

Exit: 10,000 exact synthetic chains, bounded teardown, and the full
unaffected-device fault battery pass. This gate requires explicit user
authorization before its first GPU execution.

### H3 - Architecture-native voices

gfx1151:

- Retain the certified DS4 MQ2 routed-expert kernels and dispatch semantics.
- Change only when the H1 bill identifies a measured expert-service need.

gfx1100:

- Build exact-native dense, attention, compressor, router, shared-expert, HC,
  and head families for the occurrence-weighted DS4 shapes.
- Treat exact compilation as necessary but insufficient; select workgroup,
  vector, reduction, cache, grouping, and launch plans for gfx1100.
- Prohibit generic-gfx11 or gfx1151-biased hot-path fallback.
- Screen under the intended retained/direct regime; a single HIP micro is not
  a ceiling or promotion result.

Exit: every hot family has an exact architecture identity, bit-exact oracle,
resource contract, and measured in-regime cost.

### H4 - Resident expert service

- Keep the entire routed payload resident on gfx1151.
- Deliver route metadata as soon as gfx1100 top-K completes.
- Start useful gate/up work as early as dependencies allow; prefer useful
  computation over a duplicate-read prefetch kernel.
- Evaluate cache-conscious selected-expert ordering within unchanged
  arithmetic and expert order constraints.
- Return only the routed F32 partial.
- Keep routing, RMSNorm, FWHT, and canonical state on gfx1100.

Exit: the service is byte-identical, persistent, bounded, and its isolated
timeline is fully attributable.

### H5 - Harmonic AR composition, T1

- Compose routing, gfx1100 shared work, packet transport, gfx1151 experts, and
  the ordered join.
- Record per-layer start/end times, overlap, idle time, transfer time, and
  required waits on both devices.
- Prove the canonical 512 output tokens and decoded bytes are identical.
- Run the appropriate kernel/dispatch route validation and serving lifecycle
  checks.
- Screen one fresh process first. Promote with at least three fresh processes
  only after correctness and the projected 2% threshold pass.

Exit: at least 50 tok/s median on the canonical fixture with no safety,
identity, or correctness failure.

### H6 - Rate matching, T2

Optimize only measured imbalance:

- advance route publication;
- group or fuse gfx1100 projections;
- coarsen packet boundaries where the arithmetic DAG permits;
- overlap useful expert work with the shared branch;
- reduce expert-service idle gaps;
- tune queue depth and packet slots;
- reshape the graph rather than merely count fewer launches.

Do not forward expert weights, reduce top-k, change quantization, or add
speculation.

Exit: at least 60 tok/s median, with the measured overlap and branch balance
explaining the result.

### H7 - Exact-architecture retained tapes, T3

- Lower one owner-local retained tape for gfx1100 and one for gfx1151.
- Give each tape its own architecture, symbol, launch, resource, and sequence
  identity.
- Define a combined route identity over both tapes and the typed packet
  protocol.
- Retain the H2 timeout, cancellation, and fault-containment contract.
- Compare both device timelines and the bounded control; do not attribute
  kernel enabling value to transport alone.
- Run the Redline shadow and route validation required by `docs/REDLINE.md`
  and `docs/VALIDATION.md`.

Exit: at least 62 tok/s median or a measured, decomposed remaining gap with no
unpriced component. A number alone cannot exit the gate.

### H8 - Product admission

- Add typed Rust configuration and automatic fail-closed admission for the
  exact model, device pair, and certified route identities.
- Never require an environment variable for correct default behavior.
- Preserve explicit single-device selection.
- Certify load, unload, cancellation, client disconnect, worker failure, and
  immediate single-device recovery.
- Prove gfx1100 Qwen route identity/performance and gfx1151 DS4 golden
  identity/performance remain unchanged.
- Commit the final performance, correctness, lifecycle, and evidence report.

Exit: the route is user-facing, recoverable, and reproducible without manual
source deployment or hand-driven environment state.

## 8. Measurement and promotion rules

- Use repo-native bench, serving, and Redline harnesses; no hand-timed product
  loop.
- Every comparable run uses the committed prompt and records its MD5.
- Every number names HIP, HipGraph, ROCr/AQL, retained PM4, or another exact
  regime.
- Screening may use one fresh process to reject a large loss.
- Promotion requires at least three fresh processes, median and spread,
  byte-identical output, and complete binary/model/device/route identity.
- A kernel micro is mechanism evidence, never an end-to-end claim.
- A performance result below the single-gfx1151 golden is a failed
  accelerator route.
- The old 33.x short-generation diagnostics are not acceptance evidence.
- Any GPU fault ends the session's GPU work immediately.

## 9. Commit, evidence, and reporting

- Work only on `ds4-beta-staging`.
- Keep safety/quarantine, transport, native kernels, scheduler, and product
  admission in separately reviewable commits.
- Format with `scripts/fmt-changed.sh`; never use bare `cargo fmt`.
- Commit accepted gates with DCO signoff and push immediately.
- Preserve rejected experiments and evidence; revert rejected product code.
- Evidence lives under a durable hipx DS4 evidence directory, never `/tmp`.
- Each gate report records fixture, prompt MD5, model SHA, source commit,
  binary SHA, both device identities, both tape identities where applicable,
  transport, sample count, output identity, performance, failure behavior,
  skipped work, and next gate.

Report after every gate, after two consecutive failed checkpoint bundles, and
immediately on any runtime or hardware fault.

## 10. Immediate action

H0 and H1 are complete and recorded in
[`2026-08-06-ds4-harmonic-h0-quarantine.md`](../investigations/2026-08-06-ds4-harmonic-h0-quarantine.md)
and
[`2026-08-06-ds4-harmonic-h1-critical-path.md`](../investigations/2026-08-06-ds4-harmonic-h1-critical-path.md).
H2 is the first unmet gate. Its transport and fault injection are developed
CPU-first. The authorized gfx1010 survivability canary passed; the exact
gfx1100/gfx1151 fault battery remains unrun. Stable HIP PCI binding is recorded
in
[`2026-08-07-ds4-harmonic-h2-hip-pci-binding.md`](../investigations/2026-08-07-ds4-harmonic-h2-hip-pci-binding.md).
In parallel, the source-only H4 expert-service seam is recorded in
[`2026-08-07-ds4-harmonic-h4-expert-service-source.md`](../investigations/2026-08-07-ds4-harmonic-h4-expert-service-source.md).
