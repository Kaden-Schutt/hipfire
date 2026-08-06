# DS4 heterogeneous G5: ragged side-projection rejection

Date: 2026-08-06  
Branch: `ds4-beta-staging`  
Fixture: committed 2,048-token code prompt, 512 generated tokens, batch 1,
greedy, top-k 6, Q8 KV, speculation off  
Accepted predecessor: `b2856ce8d` / 32.002912953 tok/s median

## Candidate

The candidate collapsed the gfx1100-owned `wkv` and compressor projections
which consume the same 4,096-element input into one ragged launch:

- ratio-128 layers: `[512, 512, 512]`, 20 occurrences/token;
- ratio-4 layers: `[512, 1024, 1024, 256, 256]`, 21 occurrences/token.

The micro-only kernel was introduced by `c1a979244`; product wiring was
introduced by `d4981a8f8`. Both were exact-gfx1100 gated. The artifact index
confirmed the product tensors use the same rows, K=4096, qt=35, and the same
2,192-byte packed row stride exercised by the micro.

## Micro result

Seven alternating trials used a working set larger than the 96 MiB cache and
compared every output bit against the incumbent sequential launches.

| Family | Sequential | Ragged | Kernel speedup | Projected token saving |
|---|---:|---:|---:|---:|
| ratio-128 | 0.032199 ms | 0.014444 ms | 2.2292x | 0.355093 ms |
| ratio-4 | 0.045534 ms | 0.016169 ms | 2.8162x | 0.616668 ms |

The occurrence-weighted projection was 0.971761 ms/token, or 3.110% against
the accepted 32.002912953 tok/s line. This was screening evidence only.

## Product verdict

Rejected before timing. The first canonical product process loaded the frozen
artifact and admitted the exact gfx1100/gfx1151 ownership map, then failed with
HIP error 700. Repeated `hipSetDevice` calls and the final logits D2H copy
reported an illegal memory access. No output artifact or throughput result was
accepted.

A follow-up single-token run with `AMD_SERIALIZE_KERNEL=3` never reached a
diagnostic launch report. The process remained in uninterruptible kernel wait
at `drm_sched_entity_flush` for nine minutes after the prior fault and was
terminated by explicit PID. No device reset or reboot was performed.

The exact first-faulting instruction was therefore not localized. Matching
metadata dimensions and a passing isolated kernel do not override the product
failure. Commits `c1a979244` and `d4981a8f8` were reverted by `ea380c35f`.

## Evidence

- `/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/gfx1100-e8-ragged-micro/`
- `/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/gfx1100-e8-ragged-product-screen/`
- `/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g5/gfx1100-e8-ragged-fault-localization/`

Candidate binary SHA-256:
`9070e5e7102c824af700d01329648a73ef4c46a4761b268e8603d9c881a51ac8`.
Product failure log SHA-256:
`a4d6dbc1ffaa28a6f0a3777047f5fdcd84f636874f9bfdf7513b94ddbcf1c14d`.
Process-state evidence SHA-256:
`8b570794134feb17807e17395619d3a70b701e563e85084bf31016d1544863de`.

## Gate state

G5 remains in progress. The accepted product line remains the grouped gfx1100
O-LoRA bundle at 32.002912953 tok/s. G6 dual-queue retained replay remains
blocked until T1 (70 tok/s direct heterogeneous HIP) is reached.
