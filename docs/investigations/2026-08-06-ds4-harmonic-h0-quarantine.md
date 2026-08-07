<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H0: preserve and quarantine

Status: **complete; source-only; no GPU execution performed**

Branch: `ds4-beta-staging`

Roadmap:
[`2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md`](../specs/2026-08-06-deepseek4-harmonic-gfx1100-gfx1151.md)

## Decision

The former DeepSeek V4 gfx1100+gfx1151 product route is fail-closed before
artifact hashing, HIP discovery, allocation, stream creation, or queue
submission. There is no environment-variable bypass. Its accepted ownership,
kernel, scheduler, and performance work remains reachable in Git and its source
remains available for the H1 bill, but the reciprocal direct-HIP transport
cannot be admitted by the loader or entered by an already constructed model.

The stable rejection is:

```text
deepseek4 harmonic execution is quarantined: the former split-owner route used reciprocal indefinite cross-device HIP waits; use single placement until H2 fault-containment certification removes this guard
```

Single-device DS4 and Qwen execution are outside this guard.

## Preserved work

The quarantine does not discard the useful results from G0-G5:

- public-ROCr SDMA chain evidence remains in
  [`2026-08-06-ds4-heterogeneous-g0-transport.md`](2026-08-06-ds4-heterogeneous-g0-transport.md);
- the exact-target cooperative DAG remains in
  [`2026-08-06-ds4-heterogeneous-g1-cooperative.md`](2026-08-06-ds4-heterogeneous-g1-cooperative.md);
- transactional tensor ownership and loading remain in
  [`2026-08-06-ds4-heterogeneous-g2-loading.md`](2026-08-06-ds4-heterogeneous-g2-loading.md);
- scheduler evidence remains in
  [`2026-08-06-ds4-heterogeneous-g3-scheduler.md`](2026-08-06-ds4-heterogeneous-g3-scheduler.md);
- exact-gfx1100 attention overlap remains reachable at `2d82b2516`;
- grouped gfx1100 O-LoRA remains reachable at `1f4f4c558`;
- the rejected ragged projection candidate remains recorded by `ea380c35f`;
- the later shared-projection job candidate remains rejected by `4f8ca925f`.

The last accepted canonical heterogeneous product result remains 32.0029
tok/s. The 33.x rows were 16-token diagnostics and are not recast as canonical
evidence.

## Unsafe chain retained for diagnosis

The first product lowering entered at `4130b311c`. The current historical
source shows the complete reciprocal chain:

1. gfx1100 issues three `memcpy_peer_async` operations and then
   `stream_write_value32` to publish an epoch to gfx1151;
2. gfx1151 submits `stream_wait_value32` on that peer-owned epoch;
3. gfx1151 runs selected experts, copies the result back, and publishes a
   second epoch;
4. gfx1100 submits a second `stream_wait_value32` on the gfx1151-owned epoch;
5. teardown calls unbounded `stream_synchronize` on both devices before
   destroying their streams and allocations.

Neither device wait has a deadline, cancellation state, allocation generation,
or host-supervised terminal transition. A producer fault therefore leaves the
consumer queue waiting on progress that can never occur, while the old
destructor waits for that queue to drain before reclaiming its resources.

Removing only the synchronizations would permit use-after-free. Removing only
the waits would violate ordering. The route must remain unreachable until H2
replaces the lifecycle as a whole.

## Fault chronology

### First incident: gfx1100 fault and stuck teardown

The ragged heterogeneous projection candidate faulted on gfx1100 with HIP
error 700. During diagnostic cleanup the owning process entered
`drm_sched_entity_flush` rather than reaching a bounded teardown. The candidate
was rejected and reverted in `ea380c35f`.

This established the first safety failure: after a kernel fault, the old route
could not prove that peer-dependent queues and their owning process would
terminate without waiting on the failed device.

### Second incident: gfx1100 loss with collateral gfx1151 wedge

A later product attempt again left gfx1100 unrecoverable; the observed kernel
record included device/bus loss and MES queue-removal failure. This time the
same process also owned the gfx1151 streams, peer allocations, and reciprocal
signals. gfx1151 had queue work whose completion depended on the failed
gfx1100 producer, and process teardown synchronously waited on both devices.
The internal gfx1151 consequently became unusable with the external device and
both recovered only after the operator rebooted the host.

The collateral mechanism does not require a faulty gfx1151 kernel: one KFD
process owned both device contexts, the iGPU queue waited on a peer-owned word,
and cleanup could not independently destroy the unaffected side. This is why
H2 requires per-device containment plus proof that either worker may die while
the other device remains usable.

No reset, unbind, GPU probe, model load, or product smoke was attempted while
closing H0. Both devices were reported healthy after the operator's reboot.

## Source quarantine

The guard is centralized in
`hipfire_arch_deepseek4::ensure_harmonic_execution_admitted` and is applied at
three points:

1. `Deepseek4Carrier::load` rejects every non-single DS4 placement before the
   frozen 82 GiB artifact verification and any GPU work;
2. every public `DeepseekV4HeterogeneousModel` constructor and transactional
   replacement funnels through `load_inner`, whose first operation is the
   guard;
3. `ensure_execution` repeats the guard before creating the old streams,
   events, peer signals, or execution scratch.

The low-level historical functions remain compiled so H1 can inspect and price
the accepted implementation. They are not a product admission surface.

## CPU/static validation

The focused test command was:

```text
cargo test -p hipfire-arch-deepseek4 --lib harmonic_quarantine -- --nocapture
```

Result: 2 passed, 0 failed, 252 filtered. The tests prove:

- the error text is stable and names both the unsafe mechanism and H2;
- a nonexistent model path returns the quarantine error, proving rejection
  precedes path access and HIP initialization.

The remaining CPU-only checks also passed:

```text
cargo check -p hipfire-loader -p hipfire-arch-deepseek4
cargo test -p hipfire-arch-deepseek4 --lib
```

The full DS4 library result was 253 passed, 0 failed, and 1 ignored (the
pre-existing gfx942/rocBLAS hardware test). `git diff --check` also passed.

## Re-admission rule

Deleting or bypassing the guard is not H2. Re-admission requires the roadmap's
complete H2 exit:

- typed, generation-counted, owner-checked packets;
- bounded deadlines and terminal states;
- isolated per-device workers or equivalent demonstrated containment;
- 10,000 exact synthetic chains;
- producer-loss, worker-exit, timeout, stale-epoch, malformed-owner, and
  mid-copy-cancel injection in both directions;
- after every injected failure, an authorized proof that the unaffected GPU
  can run its existing single-device oracle without reboot or reset.

The first H2 GPU execution still requires explicit user authorization.

## H0 exit

H0 exits: accidental product execution of the reciprocal HIP route is blocked
at carrier admission, model construction, and execution creation; useful work
and negative evidence remain preserved; the two fault sequences and collateral
mechanism are durable; and the quarantine was validated without touching a
GPU. H1 may proceed from preserved traces only.
