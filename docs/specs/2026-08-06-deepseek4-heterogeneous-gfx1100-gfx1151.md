<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DeepSeek V4 heterogeneous dense/expert compute on gfx1100 + gfx1151

> [!CAUTION]
> This document is retained as the historical G0-G5 record. Its direct-HIP
> product transport is quarantined: reciprocal cross-device
> `hipStreamWriteValue32` / `hipStreamWaitValue32` dependencies and blocking
> teardown can strand both devices when either queue faults. Do not execute
> that product route. Forward work is governed by
> [`2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md`](2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md).

| Field | Value |
|---|---|
| State | **G0-G4 complete; G5 direct-HIP heterogeneous optimization next** |
| Date | 2026-08-06 |
| Model | DeepSeek V4 Flash 0731 MQ2R (`arch_id=9`) |
| Artifact | `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` |
| Capacity device | `gfx1151` / Radeon 8060S / 103.1 GB (96 GiB) addressable VRAM |
| Dense accelerator | `gfx1100` / Radeon RX 7900 XTX / 25.8 GB (24 GiB) VRAM |
| Initial mode | Batch-1 autoregressive decode, top-k 6, no drafter |
| Validation owner | [`docs/VALIDATION.md`](../VALIDATION.md) |
| Quant identity | [`2026-07-23-deepseek4-mq2r-e8-recipe.md`](2026-07-23-deepseek4-mq2r-e8-recipe.md) |

## 1. Decision

Add a DeepSeek-owned heterogeneous compute route that uses complementary
tensor residency rather than ordinary pipeline or expert parallelism:

- the `gfx1100` device owns the non-routed model tier and is the canonical
  execution device for embeddings, attention, compressors, the router,
  shared-expert projections, Hyper-Connections, norms, and the output head;
- the `gfx1151` device owns every routed-expert payload and evaluates the six
  selected experts for each layer;
- the devices exchange one compact FFN input packet and one F32 routed result
  per layer, while the shared and routed branches execute concurrently;
- the final shared-plus-routed accumulation remains on `gfx1100` in the same
  arithmetic order as the single-device route.

This is **not** the existing EP implementation with different device counts.
Current EP replicates all dense weights and state on every rank, shards experts,
and all-reduces partials. This design stores each large tensor family only on
its owning device and performs a point-to-point branch join.

The initial route is direct HIP and explicitly selected. It does not replace or
mutate the certified single-`gfx1151` MQ2R/PM4 route. Automatic admission and
multi-device retained replay are later gates.

## 2. Motivation and measured feasibility

### 2.1 The split fits the actual cards

The 0731 MQ2R tensor index was classified using the loader's current GPU-byte
accounting on the exact golden artifact:

| Tier | Tensor records | Estimated GPU residency |
|---|---:|---:|
| Routed experts (`layers.L.ffn.experts.E.*`) | 33,024 | 72.562 GiB |
| Every non-routed tensor | 1,199 | 4.042 GiB |

The host's current topology reports:

| Logical device | Architecture | VRAM bytes | Role |
|---:|---|---:|---|
| 0 | `gfx1100` | 25,753,026,560 | non-routed/dense accelerator |
| 1 | `gfx1151` | 103,079,215,104 | routed-expert capacity device |

The non-routed tier therefore has substantial room on the 24 GiB card for its
weights, ordinary short-context state, and scratch. The expert tier fits the
96 GiB device with room for its branch scratch. These are planning numbers,
not a substitute for the transactional loader's exact post-allocation budget.

### 2.2 The branch boundary already exists

The production decode path already exposes the correct dependency seam:

- `ffn_prepare` produces the normalized plain and FWHT-rotated activation;
- `ffn_shared_project` evaluates the dense shared expert;
- `ffn_hash_routed` / `ffn_routed` evaluate the selected routed experts into
  a separate partial;
- `ffn_shared_routed_overlap` joins the branches before
  `hc_ffn_mix`, preserving serial accumulation order.

See `crates/hipfire-arch-deepseek4/src/forward.rs:4374` and
`:5806`. The heterogeneous route should preserve this graph and replace the
same-device side stream with a device-resident branch worker.

### 2.3 Peer access exists, but latency is the gate

On 2026-08-06, `hipDeviceCanAccessPeer` returned true in both directions and
the existing `hip-bridge` peer smoke copied byte-identical data. ROCm reports a
two-hop PCIe path with topology weight 40.

The same cold, synchronous 1 MiB smoke reported only 88.7 MB/s. That sample
includes setup and synchronization overhead and must not be extrapolated into
a bandwidth ceiling, but it is a warning: this route wins only if persistent
streams/events make the 86 small per-token boundaries cheap. H0 therefore
measures the exact 16 KiB decode packet chain before model code is changed.

## 3. Goals

1. Use `gfx1100` bandwidth and compute for the complete non-routed MQ2R tier
   without copying dense weights per token or per layer.
2. Keep the 72.562 GiB routed-expert tier resident on `gfx1151` and run the
   existing certified expert arithmetic there.
3. Overlap each layer's shared expert on `gfx1100` with its routed experts on
   `gfx1151`.
4. Preserve output, routing, KV, recurrent-state, and Hyper-Connection
   semantics exactly.
5. Preserve the current single-`gfx1151` MQ2R and DSpark routes as unchanged
   controls and fallbacks.
6. Make device selection and route admission typed, inspectable, and
   DeepSeek-specific so Qwen and single-device gfx1100 behavior cannot inherit
   DS4 placement policy.

## 4. Non-goals

- No weight, dtype, quant recipe, top-k, sampling, or model-artifact change.
- No DSpark, MTP, DFlash, DDTree, or other speculative execution in the first
  implementation.
- No attempt to reuse current EP collectives or to replicate dense weights.
- No cross-device fine-grained kernel launch for every projection.
- No transparent host-staged production fallback. Host staging is a
  correctness diagnostic only; an admitted performance route requires the
  measured peer channel.
- No initial 1M-context claim. Long-context placement is a later phase because
  the 24 GiB canonical attention device, not total system memory, becomes the
  binding capacity.
- No initial PM4 claim. Current retained replay is a single-device route.
- No global `HIPFIRE_ALLOW_MIXED_ARCH=1` requirement and no new environment
  variable as the user-facing control plane.

## 5. Why existing parallel modes are insufficient

### 5.1 Expert parallelism

`crates/hipfire-runtime/src/ep.rs` runs every non-MoE super-op on every rank,
stores full dense weights and KV on every rank, and all-reduces routed partials.
That is correct for symmetric expert sharding, but wastes the 7900 XTX's scarce
capacity on a replica and pays a collective when only one result consumer is
needed.

The heterogeneous route instead has one canonical residual stream, one dense
owner, one expert owner, and one point-to-point result.

### 5.2 Pipeline parallelism

The generic model loader assigns whole layer bands to devices. It cannot place
the non-routed and routed tensors of the same layer on different devices, and a
layer-band split leaves each card serially idle for batch-1 decode.

### 5.3 Generic mixed-architecture admission

`Gpus::preflight_vram_with_opts` currently permits different architectures
only through the process-wide `allow_mixed_arch` escape hatch. That setting is
too broad for a certified model route. The new route must admit exactly one
`gfx1100` dense role plus one `gfx1151` expert role using `ArchCaps::is_gfx1100`
and `ArchCaps::is_gfx1151`; no family-wide `Legacy` or `is_rdna3` test is
sufficient.

## 6. Ownership contract

### 6.1 Product placement

| State or tensor family | Owner | Reason |
|---|---|---|
| Token embedding, output norm, LM head, head HC | `gfx1100` | Always-used non-routed work; only one final consumer |
| Attention norms, Q/KV/O projections | `gfx1100` | Keep the complete attention subgraph resident |
| Main compressor and sparse indexer weights | `gfx1100` | Avoid intra-attention device ping-pong |
| KV, SWA, compressor, indexer, recurrent and HC state | `gfx1100` initially | Canonical state follows attention compute |
| FFN norm, router, route bias/table | `gfx1100` | Produces the compact expert dispatch packet |
| Shared expert w1/w2/w3 and scratch | `gfx1100` | Dense branch overlaps routed branch |
| Routed expert blobs and pointer tables | `gfx1151` | 72.562 GiB capacity tier |
| Routed gate/up/down scratch and output | `gfx1151` | Keeps the expert subgraph resident |
| Canonical residual and final shared+routed sum | `gfx1100` | One state owner and unchanged accumulation order |
| DSpark/MTP sidecars | Neither in v1 | Explicitly out of scope |

The loader must derive these classes from tensor names and the frozen P3
policy. Unknown or newly added tensor classes fail closed; they are not placed
on whichever device has space.

### 6.2 Decode transfer packet

For hidden size 4096 and top-k 6, one persistent input packet contains:

```text
ffn_x_rot[4096]       F32    16,384 bytes
expert_ids[6]         U32        24 bytes
route_weights[6]      F32        24 bytes
token/layer metadata  fixed      <= 64 bytes
```

The return packet is `routed_out[4096]` F32, 16,384 bytes. Exact alignment is
chosen by the implementation, but the packet must remain contiguous so each
direction uses one peer copy per layer.

At 43 layers, the lower-bound payload is 1,409,024 bytes per generated token,
plus at most a few KiB of metadata. This is bandwidth-small and latency-large:
the design must not allocate events, streams, tensors, or host packets in the
token loop.

### 6.3 Event and buffer lifetime

Introduce a persistent, generic peer-channel primitive owned by the DS4
heterogeneous state:

```text
Ds4HeteroPeerChannel
  dense_to_expert_stream
  expert_to_dense_stream
  prepare_ready[2]
  expert_ready[2]
  dense_packet[2]       // gfx1100
  expert_packet[2]      // gfx1151
  expert_result[2]      // gfx1151
  dense_result[2]       // gfx1100
```

Double buffering prevents token/layer reuse from racing an outstanding copy.
Events are created once after all allocations and destroyed on unload. The
implementation may generalize this into `hipfire-runtime`, but its policy and
role selection remain in the DS4 loader/architecture crate.

`Gpus::boundary_copy` is useful proof of the required HIP operations, but its
per-call event creation/destruction is not the target decode implementation.

## 7. Decode execution DAG

For each layer:

1. On `gfx1100`, execute attention and `ffn_prepare` as today.
2. On `gfx1100`, execute router/top-k and write the contiguous expert packet.
3. Record `prepare_ready` once the rotated activation and routing data are
   complete.
4. Fork from that event:
   - `gfx1100` evaluates `ffn_shared_project`;
   - the transfer stream copies the packet to `gfx1151`.
5. `gfx1151` waits for the copy, evaluates the existing hash-routed or
   score-routed expert path, and writes a routed-only F32 result.
6. Copy the routed result to `gfx1100` and signal `expert_ready`.
7. `gfx1100` waits for both branches, then performs the existing
   `ffn_out += routed_partial` and `hc_ffn_mix` sequence.
8. Continue to the next layer with canonical state still on `gfx1100`.

The host does not synchronize between steps. Only device events establish
dependencies. A host wait inside the 43-layer loop is a correctness fallback,
not an admissible performance implementation.

### 7.1 Arithmetic invariants

- `ffn_prepare` runs once, on the dense owner.
- The exact rotated activation consumed by the single-device routed kernels is
  copied; `gfx1151` must not recompute RMSNorm or FWHT.
- The expert IDs and route weights are copied after the existing router logic;
  routing is not recomputed on the expert device.
- `gfx1151` produces a routed-only partial initialized exactly as the current
  overlap path initializes it.
- `gfx1100` performs the final add in the existing order and then runs HC mix.
- No peer-visible remote load is used inside a GEMV. Kernels read only local
  device allocations.

## 8. Loader and type design

### 8.1 Separate state, not `EpState`

Add a distinct loader-owned state, provisionally:

```rust
pub struct Ds4HeterogeneousState {
    pub devices: Ds4HeterogeneousDevices,
    pub dense_weights: DeepseekV4DenseWeights,
    pub expert_weights: DeepseekV4ExpertWeights,
    pub dense_state: DeepseekV4State,
    pub expert_state: DeepseekV4ExpertState,
    pub peer: Ds4HeteroPeerChannel,
}
```

The final names may change, but the type split is mandatory. A
`DeepseekV4Weights` containing aliases from two devices is forbidden because
`free_gpu` currently assumes a single owning `Gpu`.

### 8.2 Transactional load

The loader opens one artifact and routes records directly to the owning
device. It must not load the 77 GiB artifact onto one device and then migrate
tensors.

Required sequence:

1. Resolve stable device selectors and instantiate both `Gpu` objects.
2. Verify exact architecture roles and artifact P3 identity.
3. Census tensor classes and projected bytes before any large allocation.
4. Upload non-routed records to `gfx1100`; upload routed blobs/pointer tables
   to `gfx1151` using the current contiguous-expert construction.
5. Allocate all state, scratch, packet buffers, streams, and events.
6. Enable bidirectional peer access **after** peer-visible allocations exist,
   matching the ordering required by `Gpus::enable_peer_all`.
7. Run a byte-exact packet round trip and pointer-owner audit.
8. Publish the model state only after every step succeeds.

On failure, the staging guard frees each tensor with its owning GPU, destroys
events/streams, and leaves the previous loaded model intact. The existing EP
constructor-mid-failure leak is not copied into this route.

### 8.3 Typed user-facing configuration

Add a Rust config/CLI model placement, not an environment-only switch:

```text
Deepseek4ComputePlacement::Single
Deepseek4ComputePlacement::DenseExpertSplit {
    dense: DeviceSelector,
    experts: DeviceSelector,
}
```

`DeviceSelector` should prefer PCI BDF or device unique ID. An exact-arch
selector is allowed only when exactly one visible device matches. Logical
ROCR indices are logged but are not stable identity.

Admission rules:

- exact model architecture `arch_id=9` and frozen MQ2R P3 policy;
- exact `gfx1100` dense device and exact `gfx1151` expert device;
- two distinct physical devices;
- bidirectional peer access and a passing warmed channel preflight;
- PP=1, TP=1, EP off, drafter off for v1;
- requested max sequence fits the measured `gfx1100` state budget;
- every required dense kernel compiles/selects for `gfx1100` and every expert
  kernel selects the certified `gfx1151` implementation.

Explicit `DenseExpertSplit` fails with a useful error when any condition is
not met. A future `Auto` mode may fall back to the single-device route, but it
must log the rejected condition. The broad `allow_mixed_arch` flag neither
enables nor bypasses this route.

## 9. Kernel and dispatch contract

### 9.1 gfx1100 dense side

The route must use per-device `ArchCaps`; it must not run a gfx1151-specific
kernel merely because both devices are RDNA3-family. H0 inventories every
non-routed dispatch shape on `gfx1100`, including:

- all MFP4G32E8SOA GEMV and batched GEMM shapes;
- Q/KV/O LoRA grouped paths;
- compressor and indexer paths;
- shared expert gate/up/down;
- router and LM head;
- norms, FWHT, HC and state kernels.

Missing or portable fallback kernels are recorded before performance work.
Dense-side tuning, if needed, follows ordinary `gfx1100` kernel ownership and
must not modify the certified `gfx1151` selectors.

### 9.2 gfx1151 expert side

Reuse the exact MQ2-Lloyd expert blobs, packing, pointer-table construction,
router-selected expert order, gate/up/down kernels, and route scaling from the
single-device MQ2R path. Only the input/output transport and state ownership
change.

The complete routed-expert tier remains resident on gfx1151 even when a later
rate-matched replica set is admitted on gfx1100. The gfx1100 set is a cache of
whole `(layer, expert)` triples (`w1`, `w2`, `w3`), never a shard that makes
the canonical gfx1151 route incomplete.

### 9.3 Rate-matched hot-expert replicas

After the full non-routed tier is resident, remaining gfx1100 capacity may hold
a ranked set of complete routed-expert triples. Selection is learned from
prefill routes and evaluated on subsequent decode routes. Capacity is an upper
bound, not the objective: the selected fraction must equalize the measured
gfx1100 local branch and gfx1151 remote branch. Filling all free memory after
that point lengthens the critical path and is forbidden.

Each of the six canonical route positions has exactly one owner per layer.
Both devices produce separate F32 rows for the positions they own. The
`DS4HARM3` join consumes positions 0 through 5 in original order with the
original raw-bit route weight at the original fused multiply-add point. Owner
masks must reject overlap, gaps, and unknown bits; stale unowned result rows
are ignored. No owner may return an early aggregate whose floating-point
association differs from single-device `moe_down_combine_k8_batched`.

### 9.4 No hidden global route state

Backend selection is carried by the owning weight/state type. Loading this
route cannot change process-global behavior for Qwen, MiniMax, another DS4
model, or a later model swap.

## 10. Phased implementation

### H0 — Feasibility and hard stop gates

Before changing model execution:

1. Add a persistent peer-channel microbench for both 16 KiB directions,
   43-layer chains, and batched payloads `{1, 16, 128, 512, 1024} × hidden`.
2. Report warmed p50/p95 one-way latency, chain time, effective bandwidth,
   event cost, and byte exactness. Separate cold setup from steady state.
3. Measure the single-gfx1151 2048/512 decode timeline and attribute the
   non-routed, shared, and routed portions. Unknown time is not zero.
4. Compile and raw-bit screen the complete gfx1100 dense kernel inventory.
5. Replace the planning census with exact post-load bytes including duplicate
   packed blobs, pointer tables, KV/state, pools, and scratch.

**Exit:** the measured peer chain plus projected split graph is at least 2%
faster end to end, both devices fit with 2 GiB safety margin, and no required
kernel is missing. Otherwise stop and record the route as capacity-feasible
but transport- or kernel-infeasible.

The existing cold 1 MiB result is not sufficient to pass or fail H0.

### H1 — Shared-expert branch prototype

Move only `shared_w1/w2/w3` and their branch scratch to `gfx1100`; retain the
single-device canonical state on `gfx1151`. Use the same compact packet and
ordered join intended for the final route.

This prototype necessarily uses the bidirectional channel in the reverse
semantic direction from H2: activation moves from the `gfx1151` canonical
state to the `gfx1100` shared branch, and shared output returns. The reusable
asset is the persistent peer channel and ordering proof; the product packet
layout and dense-to-expert direction begin at H2.

This prototype proves:

- transactional split loading and owner-correct unload;
- persistent cross-device stream/event ordering;
- raw-bit branch inputs and results;
- useful overlap against the unchanged routed branch.

It is an engineering gate, not the product topology. Reject it if it does not
reduce the measured FFN critical path; do not compensate with unrelated
kernel tuning.

### H2 — Full non-routed residency

Make `gfx1100` the canonical state owner and place the full 1,199-record
non-routed tier there. Keep only routed experts and routed scratch on
`gfx1151`. This is the first product-candidate topology described by this
spec.

Run direct-HIP AR only. Do not add PM4 or speculation while ownership and
ordering are still changing.

### H2.1 — Rate-matched routed-expert residency

Micro-screen the exact DS4 MQ2-Lloyd gate/up and deterministic expanded-down
families on gfx1100 and gfx1151 over the observed 0–6 local-slot distribution.
Every candidate must be raw-bit exact and must retain its architecture-local
resource contract. Use the measured architecture ratio to select a
branch-balanced whole-slot budget, then upload only those complete triples to
gfx1100 behind a typed, stable residency identity. Keep the full expert tier on
gfx1151 and fail before publication on any dtype, extent, capacity, or pointer
ownership mismatch.

Implement the `DS4HARM3` per-route-slot result contract and CPU exact-order
oracle before any two-device product run. Product validation remains gated on
a combined projection of at least 2%; a single-shape kernel win is not enough.

### H3 — Batched prefill

Extend the packet channel to `[B, hidden]` activations, batched expert IDs and
route weights, and `[B, hidden]` routed results. Reuse production DS4 batched
prefill and its real shape distribution; do not substitute row-at-a-time
decode as a prefill benchmark.

At `B=1024`, each activation or result is 16 MiB. Measure transfer/computation
overlap and chunk-size sensitivity. Prefill is promoted only if TTFT improves
at 2K and 20K without regressing 85K; no isolated GEMM number is sufficient.

### H4 — Context and VMM policy

The initial canonical KV/compressor state lives on the 24 GiB device, so the
loader computes a context ceiling from exact non-routed residency plus VMM
growth. Requests above the certified ceiling use the existing single-gfx1151
route; they do not silently spill attention state across PCIe.

Only after H2/H3 win may a second topology be evaluated for long context:

- VMM-backed attention state on `gfx1100` up to its physical budget; or
- coarse attention-state ownership on `gfx1151` with a separately measured
  subgraph boundary.

Remote KV loads, per-attention-kernel copies, and host/GTT spill are excluded
until a full dependency and bandwidth model predicts a gain.

### H5 — Retained replay and optional features

After direct HIP is correct and faster:

1. Define a separately versioned heterogeneous route identity.
2. Capture one per-device typed tape with explicit peer-event edges between
   them; never bake cross-device pointers into the wrong device's code object.
3. Run Redline shadow/certification for both tapes and their join.
4. Consider DSpark and MTP only after AR replay is promoted.

Single-device `.mq2r` automatic PM4 admission remains unchanged. The
heterogeneous route must not claim PM4 merely because its dense device or
expert device can individually replay a tape.

## 11. Performance model and admission thresholds

The relevant per-layer critical path is:

```text
T_layer = T_attention_dense_gfx1100
        + max(T_shared_gfx1100,
              T_packet_to_gfx1151 + T_routed_gfx1151 + T_result_to_gfx1100)
        + T_join_and_hc_gfx1100
```

Do not add isolated speedups arithmetically. Measure the composed graph.

### 11.1 Fixtures

- Screening: [`benchmarks/prompts/ds4-gfx942-ar-2048.txt`](../../benchmarks/prompts/ds4-gfx942-ar-2048.txt),
  MD5 `25e22faef15a20ae53501f1956e62b79`, greedy, top-k 6, AR, 2048 prompt
  tokens and 128 generated tokens. The historical filename does not scope the
  fixture to gfx942.
- Product acceptance: the same prompt bytes at 2048/512, batch 1, greedy,
  top-k 6, with decoded output retained.
- Prefill: synthetic-but-production-batched `hipfire bench` rows at 2K, 20K,
  and 85K with prefix caching defeated.

The comparison arms are:

1. certified single-gfx1151 direct HIP;
2. certified single-gfx1151 retained PM4, the shipping waterline;
3. heterogeneous direct HIP candidate.

### 11.2 Promotion bar

- H1 continues only with a measured FFN critical-path reduction and a
  projected end-to-end gain of at least 2%.
- H2 is worth shipping only if heterogeneous direct HIP beats the current
  single-gfx1151 PM4 waterline by at least 5% on 2048/512. Merely beating the
  slower single-device HIP control is insufficient.
- A default-on `Auto` placement requires at least 10% end-to-end gain, stable
  load/unload, and all correctness/non-regression gates.
- Any result below the single-gfx1151 shipping route is a failed accelerator
  candidate, not an improvement.

Use `hipfire bench` and the repository harnesses. Do not hand-roll a timing
loop. Cheap H0/H1 screens may use 1-3 samples for decisive effects; promotion
uses the current project measurement contract and reports every arm.

## 12. Correctness and validation

### 12.1 Primitive and state oracles

Before performance claims:

- packet round trip is byte-identical in both directions;
- copied `ffn_x_rot`, expert IDs, and route weights match the single-device
  boundary raw bits at multiple layers and positions;
- routed-only output matches the same gfx1151 kernel fed the same frozen input;
- joined FFN output, HC state, residual, logits, KV, compressor and recurrent
  state match the single-device route at certification positions;
- decoded output is byte-identical on the fixed greedy fixture.

The implementation changes placement and scheduling, not arithmetic. A
failure of exactness blocks promotion; it is not dismissed as acceptable
cross-device noise.

### 12.2 Route validation

For H1/H2 kernel, dispatch, or graph changes, extend
`scripts/redline_daemon_harness.py` with a heterogeneous direct-HIP oracle
mode. Redline replay itself remains disabled until H5. Preserve the JSON
report.

For user-facing lifecycle behavior, run `scripts/serve_harness.py`:

- two fresh-process batteries;
- an eight-turn session;
- explicit load, generate, unload, reload;
- cancellation/error injection while one branch is in flight;
- no runaway, empty response, stale-state, or wrong-device free.

Read the decoded text. Bench numbers alone do not prove coherence.

### 12.3 Anti-bleed matrix

Any promotion owes:

| Control | Required proof |
|---|---|
| DS4 MQ2R single gfx1151 HIP | unchanged output and performance within noise |
| DS4 MQ2R single gfx1151 PM4 | unchanged route hash, launch count, contracts and golden performance |
| DS4 MQ2R single gfx1100 load rejection/fallback | no gfx1151 kernel selection; useful error if model does not fit |
| Qwen on gfx1100 | unchanged route and performance if shared runtime/loader code changed |
| Qwen on gfx1151 | unchanged route and output if daemon/config dispatch changed |
| Unsupported mixed pair | fail closed without enabling a process-global policy |

Use `scripts/fmt-changed.sh`, not bare `cargo fmt`. Compile the directly
affected crates and run owner-specific unit tests before hardware validation.

## 13. Observability and evidence

Every load logs, in one structured record:

- model SHA and frozen recipe identity;
- stable physical identities, logical IDs, PCI BDFs, architectures and roles;
- exact weight/state/scratch bytes per device;
- peer-access matrix and peer-channel preflight result;
- selected dense and expert kernel-route identities;
- context ceiling and the reason for any fallback;
- direct HIP vs heterogeneous vs retained-replay mode.

Every benchmark artifact records per-device kernel time, transfer time,
overlap, idle time, and the count/bytes of peer copies. A headline tok/s number
without this decomposition cannot promote the route.

Evidence lives under a durable campaign root, not `/tmp`, and the final route
gets a registry manifest separate from the single-gfx1151 MQ2R route.

## 14. Stop conditions

Stop or park the route when any of the following holds:

1. Warmed 43-layer peer-channel cost consumes the measured dense-side saving.
2. The exact dense-side allocation plus certified context state cannot retain
   a 2 GiB safety margin on gfx1100.
3. A required gfx1100 dense kernel falls back to a path that removes the
   projected end-to-end gain, and no existing proven kernel family covers it.
4. H1 cannot overlap the shared branch with routed compute.
5. H2 fails byte-exact state/output parity.
6. H2 does not beat single-gfx1151 PM4 by 5% at 2048/512.
7. Any shared runtime change regresses Qwen or the single-gfx1151 golden route.

Do not route around a failed gate by changing top-k, weights, sampling, KV
semantics, or enabling speculation.

## 15. Expected implementation surface

The likely code surface is deliberately narrow:

- `crates/hipfire-config`: typed placement and stable device selectors;
- `crates/hipfire-loader`: transactional split load and model-state variant;
- `crates/hipfire-runtime/src/multi_gpu.rs`: persistent generic peer channel,
  only if it cannot live entirely in the DS4 crate;
- `crates/hipfire-arch-deepseek4/src/arch.rs`: tensor-role upload split;
- `crates/hipfire-arch-deepseek4/src/deepseek4.rs`: owner-specific weights and
  state;
- `crates/hipfire-arch-deepseek4/src/forward.rs`: heterogeneous FFN DAG and
  later full non-routed execution;
- `crates/hipfire-cli`: native config/status surface;
- validation harness adapters and a new route manifest after promotion.

No Qwen-owned forward function should contain a DS4 branch. Generic peer
transport may be shared, but model placement and architecture pairing remain
DeepSeek-owned.

## 16. Open questions resolved by H0, not assumption

Transport question 1 is now resolved by
[`2026-08-06-ds4-heterogeneous-g0-transport.md`](../investigations/2026-08-06-ds4-heterogeneous-g0-transport.md):
explicit public-ROCr SDMA is selected at 588.737 us p50 / 603.854 us p95
for the complete 86-copy B=1 chain, with zero corruption in the required
10,000-chain stress. G1 is resolved by
[`2026-08-06-ds4-heterogeneous-g1-cooperative.md`](../investigations/2026-08-06-ds4-heterogeneous-g1-cooperative.md):
the exact-target, double-buffered 43-layer DAG is raw-bit exact and overlaps
all 43 shared/expert dispatch pairs. G2 is resolved by
[`2026-08-06-ds4-heterogeneous-g2-loading.md`](../investigations/2026-08-06-ds4-heterogeneous-g2-loading.md):
the frozen artifact loads transactionally with 1,198 dense allocations owned
by gfx1100 and 172 packed routed allocations owned by gfx1151, all six injected
failure points reclaim in-process, and failed replacement preserves the old
model. Questions 2 through 6 remain model-execution unknowns for G4 and later
gates.

G3 is resolved by
[`2026-08-06-ds4-heterogeneous-g3-scheduler.md`](../investigations/2026-08-06-ds4-heterogeneous-g3-scheduler.md):
the generic exact-target scheduler is raw-bit exact, overlaps all 43
shared/routed forks without host synchronization inside the graph, reaches
1.851x over serialized execution at B=1024, and holds about 653 synthetic
rows/s from 2K through 86K depth. This is scheduling evidence only; production
DeepSeek lowering begins at G4.

1. What are warmed 16 KiB one-way and 43-layer chain latencies on this exact
   two-hop PCIe topology?
2. How much of the current token is non-routed work after the latest gfx1151
   kernels and PM4 route?
3. Does gfx1100 have optimized E8 coverage for all 554 dense tensors and their
   actual decode/prefill shapes?
4. Is the best H1 fork point before or after router/top-k when event latency is
   included?
5. What maximum context fits when the full attention state is canonical on the
   24 GiB device?
6. Does batched prefill benefit after transferring `[B, hidden]` in both
   directions per layer?

Until measured, each answer is **unknown**, not zero and not a projected win.
