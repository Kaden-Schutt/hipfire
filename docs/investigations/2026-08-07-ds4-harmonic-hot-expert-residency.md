<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic hot-expert residency sizing

Status: **CPU sizing accepted; typed artifact projection implemented; GPU
execution not started**

Branch: `ds4-beta-staging`

## 1. Decision

Add a rate-matched expert-residency tier to the harmonic route:

- all 43 x 256 routed experts remain resident on gfx1151;
- a selected set of exact `(layer, expert)` w1/w2/w3 payloads is additionally
  loaded from the frozen MQ2R artifact onto otherwise-unused gfx1100 memory;
- hot route slots execute locally on gfx1100 and cold slots execute on
  gfx1151;
- expert weights never cross the device link in the token loop;
- each owner returns one result per canonical route slot, and gfx1100 combines
  slots 0 through 5 in the original order.

This is exact artifact replication, not a weight, quant, or routing change.
It also is not ordinary expert parallelism: gfx1151 remains a complete fallback
owner, while the replicated set is selected to minimize the maximum concurrent
branch time rather than maximize byte occupancy.

The implementation in this checkpoint is deliberately CPU-only. It adds a
capacity-bounded typed residency plan and an artifact projection receipt, but
does not upload replicas, change dispatch, or admit a product route.

## 2. Inputs and durable evidence

| Item | Value |
|---|---|
| Artifact SHA-256 | `cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce` |
| Route dump | `hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-06-ds4-heterogeneous-g4/route-dump-combined.bin` |
| Records per repeated half | 110,037 = 2,559 tokens x 43 layers |
| Training region | first 2,048 prompt tokens |
| Evaluation region | following 511 decode transitions |
| Route width | 6 experts per record |
| Exact w1+w2+w3 bytes per slot | 7,077,904 |
| Measured gfx1100 free bytes after dense load | 17,672,699,904 |
| Analyzer | `scripts/reap/deepseek4_route_hotset.rs` |

Durable CPU-analysis directory:

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-h4-hotset-capacity/`

Evidence identities:

| File | SHA-256 |
|---|---|
| `prefill-train-decode-eval-15gb.txt` | `898a5d705c971128c793fe131d77fee970e98aa6922cb747ceee6cb78768b2ec` |
| `prefill-train-decode-eval-full-free.txt` | `495bf6dff689d24d8fb1cd729b5aca5e9728f8b1d7f4e9437625fe382d0849dd` |
| `deepseek4_route_hotset` | `0e059c23bbb51782a59f269fd34692aee1e4c8951d067d39b1cd23c2807e4411` |

The combined dump's two halves have identical residency counts. The analyzer
rejects an odd or non-identical paired dump before ranking it.

## 3. Measured route locality

The ranking is learned only from prefill, then evaluated on subsequent decode.
This avoids the invalid oracle of choosing the set from the decode routes it is
supposed to predict.

| Replica budget | Slots | Prefill coverage | Decode coverage | Decode oracle |
|---:|---:|---:|---:|---:|
| 15,000,000,000 bytes | 2,119 | 71.424002% | **64.831081%** | 76.733567% |
| 17,672,699,904 bytes | 2,496 | 75.739424% | **69.338127%** | 81.300536% |

The 15 GB plan therefore predicts almost two thirds of decode expert
selections without observing decode. The 11.90 percentage-point gap to the
15 GB decode oracle prices prompt-to-decode distribution shift; it is not
silently counted as local work.

This is one canonical technical prompt. Multi-genre route dumps are required
before automatic product admission. The result is sufficient to justify the
exact-native MQ2 micro and the load-plan implementation, not a universal
residency table.

## 4. Critical-path sizing

The occurrence-weighted H1 inputs are:

```text
gfx1100 serial useful work before the fork    15.3657 ms/token
gfx1100 shared branch                           1.6483 ms/token
routed gate/up + down work                      9.5106 ms/token
gfx1151 worker fixed activation/rotate/publish  0.9539 ms/token
```

For hot fraction `h` and exact-native gfx1100 expert speed ratio `r`:

```text
local branch  = 1.6483 + h * 9.5106 / r
remote branch = 0.9539 + (1 - h) * 9.5106
useful union  = 15.3657 + max(local branch, remote branch)
```

At the independently observed Qwen architecture ratio `r = 2.184`:

| Plan | Local branch | Remote branch | Useful union | Useful ceiling |
|---|---:|---:|---:|---:|
| 15 GB / h=64.831% | 4.471 ms | 4.299 ms | 19.837 ms | 50.41 tok/s |
| Full free / h=69.338% | 4.668 ms | 3.870 ms | 20.033 ms | 49.92 tok/s |

The calculated branch-balanced fraction at that ratio is 63.585%. Filling all
available memory is therefore counterproductive: after balance, every extra
local slot lengthens the critical branch. The scheduler must choose a
rate-matched set, not a maximum-size set.

The ratio is not assumed to transfer from Qwen. Exact gfx1100 MQ2-Lloyd expert
throughput remains **unmeasured**. The useful-ceiling table only selects the
next micro gate:

| Exact MQ2 ratio | Balanced hot fraction | Useful union | Useful ceiling |
|---:|---:|---:|---:|
| 1.50x | 55.619% | 20.540 ms | 48.68 tok/s |
| 1.75x | 58.990% | 20.220 ms | 49.46 tok/s |
| 2.00x | 61.799% | 19.953 ms | 50.12 tok/s |
| 2.184x | 63.585% | 19.783 ms | 50.55 tok/s |
| 2.50x | 66.213% | 19.533 ms | 51.20 tok/s |
| 3.00x | 69.524% | 19.218 ms | 52.03 tok/s |

This is a device-useful ceiling, not a product claim. The prior unsafe product
trace had an additional 6.036 ms/token outside the useful union; retaining that
residual would limit the 2.184x/15 GB case to about 38.65 tok/s. T1 therefore
requires both the ownership revision and elimination or overlap of nearly all
host, launch, and transport residual. T2 cannot come from residency alone; it
requires reducing the 15.3657 ms serial gfx1100 tier as H3/H6 already specify.

## 5. Exactness contract

Splitting the six selected experts into two owner-local aggregates would change
floating-point association. That is forbidden. The product design instead
requires:

1. Both owners compute a separate F32 output for every route slot they own.
2. Missing slots are represented by ownership metadata, not by a fabricated
   expert or reordered list.
3. gfx1100 consumes route slots in the canonical original order 0 through 5.
4. Each slot uses the original raw-bit route weight at the same multiplication
   point as the single-device kernel.
5. The result payload grows from one 16 KiB aggregate to at most six 16 KiB
   slot results per layer. A fixed six-row result plus the activation is at
   most about 4.70 MiB/token across 43 layers, and avoids transferring 7.1 MB
   expert payloads per selection. A later compaction may transmit only the
   remote-owned rows after it proves that the packing cost pays.

The typed `HarmonicExpertResidencyPlan` stores a canonical bitmap and stable
identity, rejects duplicate/out-of-range slots and insufficient capacity, and
partitions each route slot to an explicit owner. The CPU-only artifact
projection rejects any selected w1/w2/w3 tensor that is absent, is not
`MQ2G256Lloyd` (`qt=19`), has a different exact slot extent, or exceeds the
declared budget.

The follow-on DS4HARM3 result contract is now versioned separately from the
existing aggregate-result DS4HARM2 route. It assigns every slot to exactly one
owner, rejects overlap/gaps/unknown mask bits, ignores stale unowned rows, and
folds the six selected rows with the same ordered fused multiply-add sequence
as `moe_down_combine_k8_batched`. The CPU split-vs-monolithic oracle is raw-bit
identical across all 4,096 output columns.

## 6. Preceding experiments closed at this checkpoint

### gfx1100 E8 prefetch4

The exact micro improved the occurrence-weighted 511 dense launches from
7.630 to 6.596 ms/token, a 1.034 ms or projected 2.920% end-to-end mechanism
win. The one-process product screen was byte-identical but fell from the
accepted harmonic 27.7100 tok/s to 27.5620 tok/s (-0.534%) because exposed
gfx1151 expert wait increased. The product dispatch was reverted. The kernel
and evidence remain preserved for reconsideration after branch balance moves.

### ROCr IPC signal as a GPU dependency

The first bounded 64-cycle probe failed on cycle zero when gfx1151 attempted to
use the attached cross-process IPC signal as an AQL completion dependency. KFD
reported a node-2 page-not-present/supervisor-privilege memory fault at the
imported signal backing. The supervisor inactivated the queue and terminated
the exact child within its two-second bound. The implementation was reverted,
ledgered, and no further GPU command was submitted in that session.

ROCr IPC signals and mapped cross-device atomics are not retry candidates for
the steady-state dependency. Owner-local compute plus host-supervised bounded
publication remains the admitted control structure.

## 7. Next gates

1. Collect or recover multi-genre 2,048+decode route dumps and cross-validate a
   static or prefill-derived plan. No GPU is needed if dumps already exist.
2. In a fresh GPU session, micro-screen the exact DS4 MQ2-Lloyd gate/up and down
   shapes on gfx1100. Measure the in-model shape distribution, raw-bit parity,
   resource contract, and ratio against gfx1151.
3. Select the branch-balanced hot fraction from the measured ratio; do not fill
   VRAM by default.
4. Implement artifact-local subset upload behind the typed plan. Preserve the
   complete gfx1151 residency and fail before allocation on any projection
   mismatch.
5. Implement per-route-slot result packets and a CPU exact-order oracle before
   any two-device product run.
6. Revisit the banked E8 prefetch only after the new branch balance predicts it
   can reduce the maximum branch or serial path.
7. Run the two-token, 128-token, then canonical 2,048/512 correctness and
   product gates only after the combined projection exceeds 2%.

No GPU product number, promotion sample, or T1 claim was produced in this
checkpoint.
