<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 heterogeneous gfx1100 + gfx1151 G0 transport verdict

G0 is complete. The selected transport is public ROCr
`hsa_amd_memory_async_copy_on_engine` with persistent HSA dependency/completion
signals and explicit SDMA engine selection:

- gfx1100 to gfx1151: engine bit `0x2`;
- gfx1151 to gfx1100: engine bit `0x1`.

It is correct across the five required shapes, survives 10,000 uniquely
re-seeded B=1 chains across four fresh processes, contains no host wait inside
the 43-layer chain, and is the fastest correct measured arm. Model loading and
lowering did not begin during G0.

## Identity

| Field | Value |
|---|---|
| Commit | `ae3b38b771b2921014743573e393b91c6b90ec8e` |
| Benchmark SHA256 | `dd9a224da52f31515725e92cca8bf0503e4e5cb8269310acf4251966c2153acc` |
| ROCm/HIP | `7.14.60850-0000000` |
| Logical 0 | `gfx1100`, PCI `0000:66:00.0` |
| Logical 1 | `gfx1151`, PCI `0000:bf:00.0` |
| Topology | PCIe, two hops, weight 40 |
| 0 to 1 engines | available `0x3`, preferred `0x0`, selected `0x2` |
| 1 to 0 engines | available `0x1`, preferred `0x0`, selected `0x1` |
| Cold process-to-persistent-ready | 227.067 ms on the final cold process |

## Selected-route matrix

`gpu` below is host wall time through the terminal system-acquire HSA signal.
`enqueue` is time until all 86 copies have been submitted; it excludes the
terminal wait. ROCr has no payload-free dependency operation, so the fixed-cost
control is an 86-link chain of one-byte async copies.

| B | one-way 0 to 1 p50/p95 us | one-way 1 to 0 p50/p95 us | fixed p50 us | full chain p50/p95 us | enqueue p50 us | effective GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.011 / 12.163 | 7.143 / 7.598 | 438.476 | 588.737 / 603.854 | 33.167 | 2.393 |
| 16 | 43.702 / 48.774 | 46.277 / 64.208 | 438.471 | 3,752.919 / 3,767.848 | 31.253 | 6.007 |
| 128 | 308.191 / 616.158 | 349.103 / 382.209 | 436.958 | 26,198.316 / 26,239.645 | 28.629 | 6.884 |
| 512 | 1,357.596 / 1,387.010 | 1,237.125 / 1,242.366 | 652.997 | 102,612.752 / 102,675.813 | 31.559 | 7.031 |
| 1024 | 2,531.999 / 2,539.103 | 2,466.106 / 2,477.315 | 659.279 | 204,166.281 / 204,251.809 | 31.799 | 7.067 |

The B=1 payload increment over the irreducible one-byte control is 150.261 us.
The complete selected chain is 38.08% faster than the corrected HIP signal
arm (950.856 us) and 53.75% faster than system-scope HIP events
(1,272.875 us).

At the 28.8678 tok/s single-gfx1151 waterline, the 588.737 us transport is
1.70% of a 34.641 ms token. A gross non-routed saving of 1.282 ms is sufficient
to retain a net 2% gain after transport. Prior 2048-depth profiling attributes
well over that amount to non-routed E8 work, so G1 has credible admission
headroom; G1 must measure the composed producer/consumer DAG rather than treat
that projection as a result.

## Correctness

- Five of five batch shapes passed 11 exact chains each in the final matrix.
- Four fresh processes checked seed ranges `0..2499`, `2500..4999`,
  `5000..7499`, and `7500..9999`.
- Including each process's pre-stress correctness chain, 10,004 B=1 chains
  passed with zero mismatch, timeout, stale payload, or negative HSA signal.
- Buffers and all 86 completion signals were reused; each checked chain used a
  unique payload and alternated the two gfx1100 buffers.
- Only the terminal HSA signal is host-polled. No host synchronization occurs
  inside the chain.

## Other arms

| Arm | B=1 chain p50 | Verdict |
|---|---:|---|
| Host stream synchronization | 2,027.991 us | Correctness/latency control only |
| HIP system-scope events | 1,272.875 us | Correct, slower |
| HIP signal memory | 950.856 us | Correct, slower |
| RCCL grouped send/recv | No completed chain | Mixed gfx1100/gfx1151 communicator fails `invalid device function`; both cards pass independently |
| ROCr auto engine | 587.514 us on the earlier matched binary | Correct; explicit engine bit `0x2` was marginally better and deterministic |
| ROCr explicit SDMA | 588.737 us final | **Selected** |
| ROCr plus dual AQL barriers | 902.639 us | Correct, rejected; 303.233 us is the barrier-only chain |

The dual-AQL result is useful for G6 design: the payload component remains
close to raw SDMA, but adding one system-scoped AQL barrier at every copy
boundary gives back most of the raw-ROCr win. G6 must batch or coarsen queue
ownership transitions rather than repeat this 86-barrier construction.

## Evidence

Durable root on hipx:

`/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-hetero-g0/`

| Artifact | SHA256 |
|---|---|
| `peer-chain-final-metadata.txt` | `ef82956b0705bc64b4ecb7741147ec9d7733e54ef9e5fc512f19b914f09a47d6` |
| `peer-chain-rocr-sdma-final-matrix.txt` | `ae882da25d19038f6efdc805d445249825a17561a456ac8fe971f3e1e0e7643a` |
| `peer-chain-rocr-sdma-final-stress-10k.txt` | `b9090765ffbb52197a6be5011868402db8f1e979219e1a48365ee9fb53270b46` |
| `peer-chain-rocr-dual-aql-matrix.txt` | `08ee1e718d1dfa4ed4538150ee8574c081bf20c03c8d06a202023780752ace95` |
| `peer-chain-rocr-dual-aql-stress-10k.txt` | `f85c9398eeb973d9cfd5cbeb954fcaa2e6182056b34cad3f80c53f5d4ef8d3a2` |
| `peer-chain-rocr-auto-stress-10k.txt` | `1cef0cb96eb509482b6fa58bc0213e9e70ae4b79b3dc8b1f889b3fd7c413eb73` |
| `peer-chain-rocr-sdma-engine0-stress-10k.txt` | `027e081cd718cd05cdbe30020f733cb06b6a2c5abd06192e8c77f04b6b00ee6f` |

The next gate is G1: exact-target producer/consumer kernels, local expert-side
compute, double-buffered packets, shared-branch overlap, and a raw-bit oracle.
