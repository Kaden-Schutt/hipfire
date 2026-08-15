<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 heterogeneous G2 — transactional asymmetric loading

## Verdict

G2 is complete for the frozen DeepSeek V4 Flash 0731 MQ2R artifact. The loader
opens one HFQ file, classifies every record before large allocation, uploads
the non-routed and routed tiers directly to their final owners, and publishes
the model only after state, scratch, post-load budgets, and an exact HIP
pointer-owner audit pass.

The successful split is:

| Owner | Records | Projected weights | Actual pooled residency | Free after load |
|---|---:|---:|---:|---:|
| gfx1100 dense/non-routed | 1,199 (1 host-only, 1,198 GPU allocations) | 4,272,562,988 B / 3.979 GiB | 7,887,388,672 B / 7.346 GiB | 17,672,699,904 B / 16.459 GiB |
| gfx1151 routed experts | 33,024 (172 packed allocations) | 77,913,567,232 B / 72.563 GiB | 77,915,488,256 B / 72.536 GiB | 24,844,959,744 B / 23.139 GiB |

Both devices retain far more than the required 2 GiB safety margin. The dense
actual includes the 3,045,289,480-byte projected prefill scratch inventory and
allocator/kernel pools; the measured non-weight dense pool was
3,614,825,684 bytes in the final certification process.

## Identity

- Branch: `ds4-beta-staging`
- Artifact SHA256:
  `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce`
- Loader commit: `dc33a712e`
- Typed placement commit: `d678e3b54`
- Immediate rollback fix and certification harness: `9416a597c`
- Dense role: exactly one visible `gfx1100`
- Expert role: exactly one visible `gfx1151`
- ROCm/HIP: 7.14

The typed public syntax is:

```text
single
dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)
```

Logical HIP ordinals are not accepted as selectors. Exact-architecture
selection fails closed unless exactly one visible device matches. PCI BDF and
UUID are represented in the typed selector but remain fail-closed until the
HIP discovery layer can resolve them.

The typed setting is not yet a user-runnable heterogeneous generation route.
Connecting it to the production loader and daemon is G4 work, after G3 proves
the generic cross-device scheduler.

## Ownership and transaction design

The heterogeneous route does not place mixed-device aliases inside ordinary
`DeepseekV4Weights`. It has three distinct owners:

- `DeepseekV4DenseWeights` contains a validated ordinary weight tree with no
  routed allocations;
- `DeepseekV4RoutedWeights` contains only the 43 packed expert layer tiers and
  their pointer-table storage;
- `DeepseekV4HeterogeneousWeights` coordinates exact-owner audit and release.

`DeepseekV4HeterogeneousStaging` owns both devices, split weights, canonical
state, and prefill scratch until all fallible work succeeds. Its destructor
releases scratch, state, each weight owner, graph/cache state, and both pools.
`PrefillBatchScratch::new` likewise stages every tensor independently and
publishes only after the complete inventory allocates.

Failed routed-weight construction bypasses the normal successful-model reuse
pool and immediately releases pointer tables and packed expert allocations on
their exact owning device. This is required for a late layer-42 failure: the
first harness version showed that normal deferred pool cleanup could leave
tens of GiB resident until process exit.

The artifact census is fail-closed at exactly 1,199 non-routed records, one
host-only record, and `43 × 256 × 3 = 33,024` routed expert records. In-band
MTP and DSpark payloads are refused for this AR-only gate.

## Hardware validation

The release harness was run under the hipx GPU lock from the local ext4/NVMe
artifact, not NAS. Successful load/unload was first repeated twice in one
process. Both cycles reported zero pointer-owner violations and converged on
the same post-load free bytes (`17,672,699,904` dense and `24,844,959,744`
routed). The final one-process fault matrix then reused one verified artifact
receipt for all injected failures, loaded successfully, and exercised failed
replacement. Exact-target kernel/runtime caches remain process-resident; no
model allocation survived any failed staging transaction.

Injected failure points:

| Failure point | Expected error | Immediate in-process used VRAM after rollback |
|---|---|---|
| after dense weights | observed | gfx1100 192,937,984 B; gfx1151 165,675,008 B |
| after routed layer 0 | observed | gfx1100 192,937,984 B; gfx1151 318,767,104 B |
| after routed layer 42 | observed | gfx1100 192,937,984 B; gfx1151 318,767,104 B |
| after ownership audit | observed | gfx1100 192,937,984 B; gfx1151 318,767,104 B |
| after state | observed | gfx1100 192,937,984 B; gfx1151 318,767,104 B |
| after scratch | observed | gfx1100 192,937,984 B; gfx1151 318,767,104 B |

A final successful load follows the complete injected-failure sequence. A
failed transactional replacement is also checked to leave the previously
published model identity and pointer-owner audit intact. It fails preflight
because the already resident model leaves insufficient expert-device margin,
then reports the original artifact SHA and an empty ownership-violation list.

## Software validation

- `cargo test -p hipfire-arch-deepseek4 --lib`: 251 passed, 1 gfx942-only
  hardware test ignored.
- `cargo test -p hipfire-config --lib`: 54 passed.
- `cargo check -p hipfire-arch-deepseek4 --example ds4_heterogeneous_load`:
  passed.
- `cargo check -p hipfire-cli`: passed.
- `scripts/fmt-changed.sh` and `git diff --check`: passed.

## Evidence

Durable root on hipx:

```text
/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g2/
```

Important files:

- `success-01.log`
- `success-repeat.log`
- `fault-matrix-and-recovery-v2.log`

Sealed identities:

- certification binary SHA256:
  `a0b421d7189b53d6f154ed4d176cffa3f12982625d7705fd9551462630885171`;
- fault/recovery log SHA256:
  `d081f386768190b412f5378a928f0ef2dd48213139194dc1b5583688ea67aa24`;
- first success log SHA256:
  `d0c29641c1ba103f1972f52ece104b07c2fbd063d7d7d9074d3501d271df00c2`;
- repeated success log SHA256:
  `7a7412c8748d798c2ccfcc8bb59ad289472f65bef699d40d066697cf3eb355be`.

This gate records no throughput number and makes no model-execution claim.
G3 is the generic 43-layer cross-device prefill/decode scheduler fixture; DS4
lowering remains forbidden until that generic graph is correct and overlapping.
