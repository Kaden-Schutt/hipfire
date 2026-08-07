<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic independent role residency checkpoint

Status: source and CPU validation complete; no GPU execution performed; H0
product quarantine remains unconditional.

## Finding

The retained G2 loader already separated ownership into
`DeepseekV4DenseWeights` and `DeepseekV4RoutedWeights`, but constructed both
inside one process holding both `Gpu` objects. That type split prevented an
owner-wrong `hipFree`; it did not prevent one PASID from owning queues and
allocations on both devices. The latter is the failure domain that allowed a
gfx1100 fault plus reciprocal peer waits to strand gfx1151.

The harmonic route now has two independent low-level load entries:

- `DeepseekV4::load_weights_harmonic_dense_gfx1100` uploads globals,
  attention, compressor/indexer, routing, shared experts, HC, and the head to
  one exact-gfx1100 owner while forcibly skipping every routed-expert upload;
- `DeepseekV4::load_weights_harmonic_experts_gfx1151` uploads only the 43
  layers of packed routed-expert blobs and pointer tables to one exact-gfx1151
  owner. It never constructs dense weights, canonical state, routing, RMSNorm,
  FWHT, or a foreign `Gpu`.

The existing one-process historical loader remains compiled for evidence but
quarantined before artifact or GPU access. Neither new entry is wired to
serving, a benchmark, or product admission.

## Fail-closed role contract

Each owner now carries an exact architecture identity:

| Owner | Required architecture | Resident payload |
|---|---|---|
| `DenseGfx1100` | exactly `gfx1100` | complete non-routed tower |
| `ExpertGfx1151` | exactly `gfx1151` | complete routed-expert tier |

Admission also requires the frozen MQ2R P3 configuration: 43 layers, 256
routed experts, top-k 6, no MQ2RXT, no DSpark, and no REAP keep map. A broad
gfx11 match, a partial expert layer cap, or the developer upload-experts switch
cannot alter either harmonic worker's residency.

The expert-only loader has its own transactional staging guard. If any layer
upload fails, it walks only the gfx1151-owned routed tensors already installed.
The dense-only path reuses the dense staging guard and validates that no
routed allocation entered the returned type.

## Validation

```text
scripts/fmt-changed.sh
cargo test -q -p hipfire-arch-deepseek4 --lib
269 passed; 0 failed; 1 ignored

cargo check -q -p hipfire-loader -p hipfire-arch-deepseek4
pass
```

The new CPU test proves role-specific exact architecture admission, including
case-insensitive identity normalization and rejection of gfx1151 for the dense
owner or another gfx11 architecture for the expert owner.

No HIP runtime, model artifact, GPU queue, allocation, daemon, or hardware
probe was used for this checkpoint.

## Remaining proof

This is necessary source separation, not residency certification. After the
user explicitly authorizes H2 hardware work, each loader still owes:

1. its own process and KFD/PASID;
2. frozen artifact SHA verification before allocation;
3. pointer-owner and byte-count receipts for its local payload;
4. bounded supervisor teardown and allocation-generation replacement; and
5. the unaffected-device oracle after the peer process is killed.

Only after those H2 exits may the process-local expert service and canonical
gfx1100 dense execution be composed through the typed harmonic ring.
