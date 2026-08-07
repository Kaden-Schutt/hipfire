<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev> -->

# DS4 harmonic hot-expert residency sizing

Status: **exact HARM3 product checkpoint; historical performance gate open**

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

The implementation now carries that plan through a typed DS4HARM3 product
route. It uploads the exact selected replicas, executes the split owners, and
joins the six route slots in canonical order. The resulting route is an exact
functional checkpoint; its measured speed only recovers the regression of the
fault-contained HARM2 path and does not beat the historical pre-safety
approximately 32 tok/s waterline.

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

### 4.1 Rate-matched candidate and route-count distribution

At the provisional 2.184x ratio, 2,025 ranked slots (14,332,755,600 bytes)
produce 63.615194% decode coverage, within 0.031 percentage points of the
calculated 63.5848% branch balance. The decode-side local-slot histogram is:

```text
hot slots       0     1     2     3     4     5     6
occurrences   369  1102  2341  4381  6185  5368  2227
records                                           21,973
mean local slots                                    3.816912
```

The complementary gfx1151 histogram is the same row reversed. This is the
shape distribution the exact-native micro must occurrence-weight. Measuring
only top-k 6 would overstate both branches and is not an admission result.

The 2,025-slot plan is provisional. The measured gfx1100/gfx1151 MQ2 ratio
selects the final hot fraction and therefore the final budget; the loader must
not hard-code this prompt-derived set as a universal registry asset.

### 4.2 Exact-gfx1100 routed-down candidate prepared

Source inspection found that gate/up is already the K4+LDS implementation
explicitly ported from the gfx1100 MQ3 pattern. Deterministic routed-down still
loads and converts all four FP16 codebook entries redundantly in every lane.
An exact-gfx1100-only micro candidate now loads each K4 codebook cooperatively
into 64 bytes of LDS while preserving index loads, K4 accumulator assignment,
FMA expression order, wave reduction, and expanded result layout.

Offline ROCm compilation for gfx1100 gives:

| Resource | Incumbent down | LDS candidate |
|---|---:|---:|
| VGPR | 93 | 73 |
| SGPR | 37 | 20 |
| fixed LDS | 0 B | 64 B |
| spills | 0 | 0 |
| wavefront | 32 | 32 |
| disassembled instructions | 685 | 517 |
| compiler-reported waves/EU | 16 | 16 |

The candidate therefore does not buy its shorter instruction path by reducing
occupancy or spilling. This is still only an ISA/resource mechanism check. It
does not become product dispatch unless the fresh-device micro proves raw-bit
identity and an occurrence-weighted projection of at least 2% end to end.

### 4.3 Exact-device complete-branch micro

The fresh-device micro at commit `5fd9d587b` resolved each marketing-name
selector through live HIP discovery to a PCI identity, then opened only that
exact device under the GPU lock. The occurrence histograms are the 2,025-slot
plan's complementary decode distributions from section 4.1.

| Owner and histogram | Gate/up | Activation | Rotation | Down | Complete expert branch |
|---|---:|---:|---:|---:|---:|
| gfx1100 local, incumbent down | 31.786 us | 4.252 us | 4.161 us | 27.436 us | 67.635 us/layer |
| gfx1100 local, LDS down | 31.786 us | 4.252 us | 4.161 us | **16.775 us** | **56.974 us/layer** |
| gfx1151 remote, incumbent | 36.617 us | 1.990 us | 2.260 us | 29.807 us | **70.675 us/layer** |

Every gfx1100 down shape from local top-k 1 through 6 was raw-bit identical.
The LDS candidate reduces the complete local expert branch by 18.71%, or
0.4584 ms/token across 43 layers. Including the fixed 1.6483 ms shared branch,
the local side falls from 4.5566 to 4.0982 ms/token: an 11.19% fork reduction
and a 2.15% useful-path projection. It therefore clears the campaign's 2%
micro admission gate.

The measured effective per-selected-slot architecture ratio is about 2.30x,
but the product scheduler must use the measured whole-branch histograms rather
than linearizing that ratio. Under the provisional 2,025-slot plan the local
branch is now longer than the remote branch, so the final plan must use fewer
replicas than this provisional set.

Evidence:

| File | SHA-256 |
|---|---|
| `gfx1100-complete-micro.txt` | `0f0f86ced7d97ba64a135b18a8f67a3634bcf7e9e48e7227d67f385c7b25c639` |
| `gfx1151-complete-micro.txt` | `a4de717fde8c7eef446bd82463df563688028ed85627c0f6da0e53934b788f76` |

Both are under:

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-h4-exact-mq2-complete-branch/`

### 4.4 Six-row result transport and revised ceiling

The split route moves the deterministic six-slot combine to gfx1100. Its
measured cost is 10.741 us/layer = 0.4619 ms/token. A separate page-backed,
HIP-registered mapping probe at commit `c54cab05d` sized the full 98,304-byte
DS4HARM3 result without changing the accepted DS4HARM2 ring:

| Exact owner | 16 KiB write | 16 KiB read | 96 KiB write | 96 KiB read |
|---|---:|---:|---:|---:|
| gfx1151 | 4.392 us | 4.363 us | **5.805 us** | 5.796 us |
| gfx1100 | 10.989 us | 9.537 us | 19.863 us | **20.946 us** |

For the direction used by the product, gfx1151 publication is 0.2496 ms/token
and gfx1100 acquisition is 0.9007 ms/token. The provisional whole-path device
accounting is therefore:

```text
serial gfx1100 work before fork       15.3657 ms/token
max(local 4.0982, remote 3.0390+0.2496) 4.0982
gfx1100 96 KiB acquire + final combine  1.3625
device-useful union                    20.8264 ms/token
device-useful ceiling                   48.016 tok/s
```

This is not a product ceiling. It deliberately charges the most conservative
fixed six-row transport. The packet already carries canonical route ownership,
so the production candidate should send only the packed remote-owned rows and
have one exact-order split-combine kernel select the packed local or remote row
for slots 0 through 5. That removes unused mapped bytes without changing one
FMA. Its occurrence-weighted transport and exact GPU combine are the next
micro gate. The prior 6.036 ms/token host/queue residual also remains a product
integration target; it is not included as unavoidable work.

Transport evidence in the same durable directory:

| File | SHA-256 |
|---|---|
| `gfx1151-split-transport.txt` | `5d290af5ab67de7324a1e80b397f2beb79d110c8e5f918ac1ad6738380901a10` |
| `gfx1100-split-transport.txt` | `f07e169994e665f1662aefda3054c253bd8b546d271a6657801bb8d9bba20a79` |

### 4.5 Packed transport, exact GPU join, and final replica count

The follow-up micro at commit `67f110cc5` measured every packed remote result
extent from one through six 16 KiB rows. It also introduced a gfx1100-only
split-combine candidate that consumes local rows from device memory and remote
rows directly from the registered mapped result allocation. The kernel keeps
canonical route-slot order and the original six FMA sequence.

The exactness gate covered every one of the 64 possible six-slot ownership
masks and all 4,096 output columns: **zero raw-bit mismatches** against
`moe_down_combine_k8_batched`. Direct mapped combine measured 8.252, 10.314,
12.431, 14.494, 16.611, and 18.743 us/layer for one through six remote rows.
At three remote rows it replaces a separate 14.005 us mapped read plus about
5.16 us combine with one 12.431 us kernel.

The occurrence-weighted capacity sweep selects **1,400 resident slots** rather
than the provisional 2,025-slot maximum:

| Resident slots | Bytes | Eval local-count histogram | Mean local experts | Projected tok/s |
|---:|---:|---|---:|---:|
| 1,300 | 9,201,275,200 | `[1342,1789,3931,5646,5311,3108,846]` | 3.115 | 50.180 |
| 1,350 | 9,555,170,400 | `[1263,1705,3776,5595,5448,3286,900]` | 3.170 | 50.359 |
| 1,375 | 9,732,118,000 | `[1228,1682,3679,5480,5512,3427,965]` | 3.206 | 50.473 |
| **1,400** | **9,909,065,600** | **`[1190,1581,3625,5491,5578,3508,1000]`** | **3.238** | **50.483** |
| 1,425 | 10,086,013,200 | `[1181,1544,3525,5438,5660,3559,1066]` | 3.265 | 50.480 |
| 1,450 | 10,262,960,800 | `[1154,1506,3461,5426,5701,3620,1105]` | 3.288 | 50.474 |

For 1,400 slots the folded accounting is 3.9319 ms/token on the local expert
branch, 3.8933 ms/token on the remote branch including its packed publication,
and 0.5110 ms/token for direct mapped canonical join. Added to the previously
measured 15.3657 ms/token serial prefix, that is 19.8086 ms/token or 50.483
tok/s. This remains a **device-useful projection, not a product claim**; it
admits product integration because it clears the 2% campaign floor and crosses
T1 without assuming the old host/queue residual disappears for free.

Evidence is under:

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-h4-packed-transport/`

### 4.6 DS4HARM3 product integration

Commits `b6440037d`, `1bff4f5df`, and `14f116afa` instantiate the typed
hotset route in the product-shaped parent/worker path. The frozen manifest is
`benchmarks/routes/ds4_0731_harmonic_hotset_1400.ds4hot`, SHA-256
`af643539ff01acf706a14073dc4898c058bc1b8d241279b3cff7719151eca7b5`.
It accounts for each selected expert's exact w1/w2/w3 payload plus its two
owner-local pointer-table entries; the first bounded load caught and fixed the
otherwise-hidden 16-byte-per-slot distinction before any kernel dispatch.

The two-token gate matched the preserved canonical output prefix byte for
byte. The 128-token gate matched all 592 decoded bytes of the canonical
512-token output prefix. The full product gate then produced:

| Fixture | top-k | Samples | Decode | Layer wall | Expert wait | Exactness |
|---|---:|---:|---:|---:|---:|---|
| 2,048 prompt / 512 generated, greedy | 6 | 1 fresh process | **31.5721 tok/s** | 30.9810 ms/token | 7.7294 ms/token | 2,491 bytes, MD5 `ee05ab4f07393fb7d624d966a7dde4af`, byte-identical |

This is **+13.94%** over the fault-contained DS4HARM2 implementation at
27.7100 tok/s, but that is only a local diagnostic comparison. The campaign's
real pre-safety waterline was approximately **32 tok/s**. At 31.5721 tok/s,
DS4HARM3 recovers the safety-path regression while remaining slightly below
that historical baseline. It is therefore an exact functional checkpoint,
**not a campaign performance win**, not completion of the 50 tok/s goal, and
not a repeated publication sample. Device identities were resolved at runtime
to exact gfx1100 and gfx1151 owners; no PCI address is encoded in the route.

The micro projection did not fully translate. Measured layer wall is 30.9810
ms/token versus the 19.8086 ms/token device-useful projection, leaving an
**11.1724 ms/token unclassified product residual**. The resident split did
reduce expert wait from 12.5167 to 7.7294 ms/token, but the remaining residual
must be stage-profiled in the product path before another kernel or scheduling
lever is ranked. Unknown is not zero.

Product evidence is under:

`hipx:/home/kaden/ds4-gfx1151-evidence/2026-08-07-harmonic-h4-hotset-product/`

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

1. Replace the per-layer gfx1151 HIP expert chord with an owner-local retained
   submission. The paired-stage accounting places about 3.25 ms/token between
   the measured remote HIP path and its occurrence-weighted device work.
2. Retain the gfx1100 dense-owner composition after the remote chord is exact;
   the current path still issues about 2,676 dense launches per token.
3. Revisit the banked E8 prefetch only after the new branch balance predicts it
   can reduce the maximum branch or serial path.
4. Admit only a candidate that projects at least another 2% end to end; repeat
   the two-token, 128-token, and canonical 2,048/512 exactness sequence.

The one-sample 31.5721 tok/s result is an exact baseline-recovery checkpoint,
not a performance promotion or publication number. It does not exceed the
historical approximately 32 tok/s line. T1 remains open at 50 tok/s.
