<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic H4 process-local expert service source checkpoint

Date: 2026-08-07

Branch: `ds4-beta-staging`

Status: source and CPU checks only. No DS4 model was loaded and no product GPU
was touched. This checkpoint is not H2 exact-pair fault containment, H3
architecture-native certification, H4 numerical parity, route admission, or a
performance result.

## Implemented seam

`DeepseekV4HarmonicExpertService` is a process-local gfx1151 selected-expert
service for the frozen DS4 0731 MQ2R P3 shape. It:

- admits only exact `gfx1151`, 43 layers, 256 routed experts, top-k 6, hidden
  4096, MoE intermediate 2048, MQ2R enabled, MQ2RXT and DSpark disabled;
- owns one local stream and only the selected-expert scratch required by the
  existing MQ2-Lloyd MoE family;
- validates the epoch, owners, allocation generations, extents, deadline,
  route metadata, activation fingerprint, and fixed payload layout before any
  GPU submission;
- uploads one 16 KiB rotated activation plus 48 bytes of routing metadata;
- invokes the existing gfx1151 `moe_family().run_selected` arithmetic without
  changing expert order or numerical operations;
- synchronizes only its local stream to return one 16 KiB F32 routed partial;
- exposes no peer pointer, peer copy, foreign GPU handle, cross-device signal,
  stream wait/write-value operation, or blocking `Drop` path.

The activation payload layout is fixed at 16,448 bytes: 16,384 bytes of F32
`x_rot`, six little-endian expert IDs, six raw F32 route-weight bit patterns,
and 16 reserved zero bytes. The result payload is exactly 16,384 bytes.

`DeepseekV4RoutedWeights::audit_local_owner` produces a process-local residency
receipt containing architecture, stable PCI identity, diagnostic HIP ordinal,
tensor count, and resident bytes after querying HIP pointer ownership for every
routed tensor. The ordinal is never the physical identity.

The feature-gated `deepseek4-harmonic-expert-worker` binary now owns the
process boundary around that service. It:

- accepts only a canonical PCI BDF, never a HIP or ROCr ordinal;
- verifies the exact 0731 MQ2R artifact SHA before initializing either runtime
  or allocating model weights;
- resolves HIP and ROCr independently by the same BDF, requires both to report
  exact `gfx1151`, and rejects any physical-identity mismatch;
- opens the already-created typed shared ring and requires its destination
  allocation generation to equal the worker generation;
- loads only routed weights, publishes the ownership receipt over a private
  Unix-domain control socket, and services explicit epoch commands;
- explicitly releases scratch and weights on every returned error and graceful
  shutdown; a GPU stall remains the supervisor's process-kill boundary.

The binary has no product entry point or automatic admission. Building it
requires the `harmonic-worker` feature.

## Redline identity provenance

The independent `warpfront/redline` repository was inspected at master
`b505a72df59e0203d467c08081c6ba313e49cb5c`. Its `redline-rocr` owns the
normalized ROCr `PciBusId` derived from `AMD_AGENT_INFO_DOMAIN` and
`AMD_AGENT_INFO_BDFID`. Hipfire's `GpuSelector::PciBusId` and PCI-scoped
survivability oracle are integration extensions over that identity type;
neither hipfire beta nor upstream Redline's current C API selected a GPU by BDF.

The two repositories have diverged. No wholesale Redline source copy was made:
hipfire retains its own async-copy, device-local allocation, and PM4 changes,
while the stable physical identity is lifted selectively and fail closed.

## Checks

- `cargo check -p hipfire-arch-deepseek4`
- `cargo check -p hipfire-arch-deepseek4 --features harmonic-worker --bin deepseek4-harmonic-expert-worker`
- `cargo test -p hipfire-arch-deepseek4 --lib`
- `scripts/fmt-changed.sh` against `origin/ds4-beta-staging`
- `git diff --check`

The crate-level static tests reject cross-device primitives and ordinal
fallbacks in both the service and worker process, and require exact HIP/ROCr
PCI selectors plus artifact verification. These checks prove source shape and
CPU protocol invariants only.

## Unmet exits

Before H4 can close, the supervisor and hardware gates must:

1. spawn the exact-BDF worker, validate every phase/ready receipt, and reject
   any unexpected child exit or protocol event;
2. enforce a host deadline by terminating the exact child process, never by placing
   a reciprocal indefinite wait on either GPU;
3. confirm process exit before calling `isolate_owner`, advancing the
   generation, or recycling an abandoned slot;
4. pass exact gfx1100/gfx1151 fault injection and then byte-exact selected-MoE
   parity against the existing single-gfx1151 route.

Until those exits pass, product execution remains quarantined.
