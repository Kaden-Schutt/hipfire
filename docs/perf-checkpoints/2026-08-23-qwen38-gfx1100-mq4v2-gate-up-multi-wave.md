# Qwen3.8-27B — gfx1100 MQ4V2 gate/up multi-wave LDS (exact gfx1100)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **accepted** for the exact source defaults below on exact-`gfx1100`
only. Acceptance is **Class A + C + D**. This is fixture-bound measured evidence
under the method below. It is **not** a product baseline, SKU claim, registry
throughput number, or transferable cross-arch result. Newest file ≠ automatic
product baseline.

| accepted default (exact `gfx1100` only) | value |
|---|---|
| Ordinary Qwen prefill chunk | **512** (unchanged from prior stages) |
| Adaptive MQ4V2 **QKVZA** / **QKV** batch-tile | **BT4** `96..191`; **BT12** `≥192`; **base** below 96 (unchanged from residual-MW / full-BT) |
| Adaptive MQ4V2 **residual** multi-wave | **MW4** for `N = 416..463`; **MW8** for `N ≥ 464`; residual BT/base below **416** (unchanged from residual-MW) |
| Adaptive MQ4V2 **gate/up** below **384** | prior gate **BT6** for `N = 96..383` (and prior base outside that band) — **unchanged** below the new MW band |
| Adaptive MQ4V2 **gate/up multi-wave** (`N ≥ 384`) | `tiles64 = ceil(N / 64)`; **even → MW8**, **odd → MW4** |
| Every other arch string | **unchanged** (no port of these defaults) |
| Capture / replay path | **excluded** — exact gfx1100 gate/up multi-wave outside capture/replay only |
| Adaptive-KV outer chunk hard cap | **256** (unchanged) |
| Eviction internal chunk hard cap | **256** (unchanged) |

**Related (context only; not a mutation):**

- [`2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md)
  — **prior accepted residual multi-wave stage** (residual MW4/MW8 on top of
  full adaptive batch-tile). That ledger remains **unchanged**. This file is the
  next gate/up stage only: multi-wave LDS gate/up for `N ≥ 384` on top of that
  residual-MW + full-BT policy. Cite incremental lifts against that residual-MW
  arm; cite total lifts against the pre-BT base recorded there (pp2048 base
  **528.5** tok/s).
- [`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md)
  — prior accepted full adaptive batch-tile stage; still unchanged.
- [`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md)
  — earlier gate/up-only BT stage; still unchanged.
- [`2026-08-23-qwen38-gfx1100-prefill-screen.md`](./2026-08-23-qwen38-gfx1100-prefill-screen.md)
  — same-day rejected chunk/MMQ screen; not amended by this ledger.
- [`2026-08-23-qwen38-gfx1201-prefill-chunk384.md`](./2026-08-23-qwen38-gfx1201-prefill-chunk384.md)
  — accepted exact-`gfx1201` chunk default. **Do not** transfer magnitudes or
  policy onto gfx1100.

---

## 1 · Question

On exact **gfx1100**, with the already-accepted residual multi-wave + full
adaptive batch-tile stage held fixed for residual / QKVZA / QKV and for gate/up
shapes below **384**, does promoting an exact-gfx1100 gate/up **multi-wave LDS**
policy (`tiles64 = ceil(N/64)`; even → **MW8**, odd → **MW4**) recover large-N
gate/up throughput where residual-MW left gate/up as the primary ceiling, without
moving other-arch defaults, without claiming a Redline capture/replay route pass,
and without over-claiming MW12/MW16 as hard failures?

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked **healthy** |
| Base revision | `31d965dba` |
| Model path | `/home/kaden/qcal/ladder-v2-gfx1100/artifacts/qwen3.8-27b.mq4v2.xt.hfq` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |
| Spec / KV | **spec off**; **kv-mode q8**; **kv-backend vmm** |
| Final CLI (`hipfire`) md5 | `ffa28455cf302e0902be286fc2f0c107` |
| Final daemon md5 | `7f3d339b23d1be91bfdbfdd9d5b9c148` |

Post-run device health: overall **idle**, KFD present, healthy idle path. No
recovery action required.

Machine-local (non-repo) evidence root:

- `/home/kaden/qcal/q38-gfx1100-mwgate-20260823/` — raw reports/logs
- `/home/kaden/qcal/q38-gfx1100-mwgate-20260823/summary.json` — structured summary
  (machine-local)

These paths are **host-local run artifacts**, not durable in-tree fixtures.
Repo research capsule:
`.codeinsight+research/q38-gfx1100-mq4v2-gate-up-multi-wave-20260823.json`.

## 3 · Evidence taxonomy

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix (final arm)** | three fresh processes × `hipfire bench --matrix` with full five-sample arrays under gate/up multi-wave + prior residual-MW policy | product SKU SLA; Redline PM4/AQL pass; other arches |
| **B. Wave-count screens** | single-process forced MW4/8/12/16 (+ base) and fine threshold screens used to choose even/odd `tiles64` MW bands | three-process final admission by themselves |
| **C. Numerical parity** | GPU base/BT vs GPU gate/up MW outputs under hardened routing/overwrite harness (distinct gate/up salts, boundary-crossing partial row tiles, quiet-NaN output init); **hard raw f32 bit equality** on **gate** and **up** | host oracle; wall-clock tok/s |
| **D. User-facing serve / long prompt** | cold JIT first run + warm identical-prompt serve coherence | open-ended EOS quality; Redline |
| **E. Serialized profile** | post-port wall / serialized / family share attribution | 1000 tok/s claim |
| **F. Redline disclosure** | new gate/up multi-wave route is capture/replay-excluded; inherited stable capture/AQL/PM4 limitations from residual-MW / full-BT | acceptance of a capture/replay product route |

Acceptance for this ledger is **Class A + C + D**. Class F is disclosed and **does not pass**. The promoted gate/up multi-wave path **explicitly excludes** capture/replay.

---

## 4 · Class A — final synthetic matrix

### Method

Guarded exclusive device; three **fresh processes** under the accepted gate/up
multi-wave policy layered on the prior residual-MW + full-BT defaults:

```bash
hipfire bench --matrix \
  --pp 128,192,256,320,384,400,416,432,448,464,480,512,2048 \
  --ctx 128 \
  --tg 1 \
  --runs 5 \
  --warmups 10 \
  --spec off \
  --kv-mode q8 \
  --kv-backend vmm \
  --json
```

- Each process emits five samples per pp length; process median is the sample
  median; headline is the **median of the three process medians**.
- Full five-sample arrays below are from the JSON evidence (not medians only).

### Headline (median-of-process-medians, tok/s)

Incremental vs prior **residual multi-wave** promoted arm
([`2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md)):

| pp | prior residual-MW | gate/up multi-wave | Δ |
|---:|---:|---:|---:|
| 128 | **680.2** | **679.1** | **−0.16%** (neutral by design) |
| 192 | **804.3** | **801.2** | **−0.39%** (neutral by design) |
| 256 | **743.3** | **741.9** | **−0.19%** (neutral by design) |
| 320 | **743.0** | **742.2** | **−0.11%** (neutral by design) |
| 384 | **762.5** | **784.0** | **+2.82%** |
| 448 | **795.7** | **816.8** | **+2.65%** |
| 480 | **815.7** | **851.4** | **+4.38%** |
| 512 | **834.2** | **883.9** | **+5.96%** |
| 2048 | **814.6** | **861.4** | **+5.75%** |

Shapes **below the gate/up MW band** (`pp < 384`) remain on the prior gate BT6
policy; measured deltas there are noise / continuity checks, **not** claimed
wins. Shapes **≥ 384** are on the new even/odd `tiles64` MW policy.

Total vs **pre-BT base** (same residual-MW / full-BT lineage’s pre-BT base arm;
pp2048 base **528.5**):

| pp | pre-BT base | gate/up multi-wave | Δ |
|---:|---:|---:|---:|
| 2048 | **528.5** | **861.4** | **+63.0%** |

Decode **tg1** process medians from evidence: **47.712 / 47.624 / 47.452** tok/s.
Recorded for continuity with the residual-MW stage (~47 tok/s). This record is a
**prefill** win claim only for the gate/up MW band — **no** decode regression
claim and **no** decode win claim.

### Gate/up multi-wave arm — process medians (all measured shapes)

| pp | p0 | p1 | p2 | median-of-medians |
|---:|---:|---:|---:|---:|
| 128 | 684.8 | 679.1 | 679.0 | **679.1** |
| 192 | 808.4 | 801.2 | 801.1 | **801.2** |
| 256 | 748.1 | 741.7 | 741.9 | **741.9** |
| 320 | 746.3 | 742.2 | 740.6 | **742.2** |
| 384 | 788.3 | 784.0 | 783.2 | **784.0** |
| 400 | 673.7 | 669.2 | 667.1 | **669.2** |
| 416 | 768.9 | 767.4 | 767.3 | **767.4** |
| 432 | 794.0 | 791.0 | 792.0 | **792.0** |
| 448 | 817.6 | 816.5 | 816.8 | **816.8** |
| 464 | 832.5 | 830.9 | 830.7 | **830.9** |
| 480 | 852.3 | 851.0 | 851.4 | **851.4** |
| 512 | 884.2 | 883.9 | 883.2 | **883.9** |
| 2048 | 862.0 | 861.4 | 860.1 | **861.4** |

### Gate/up multi-wave arm — raw five-sample prefill arrays

| pp | process | samples (n=5) | process median |
|---:|---:|---|---:|
| 128 | 0 | 683.1, 684.8, 686.3, 686.7, 684.4 | 684.8 |
| 128 | 1 | 676.8, 679.1, 679.2, 679.8, 677.9 | 679.1 |
| 128 | 2 | 675.7, 678.6, 679.9, 679.0, 679.1 | 679.0 |
| 192 | 0 | 800.4, 808.9, 808.9, 808.4, 808.1 | 808.4 |
| 192 | 1 | 799.2, 801.6, 801.2, 801.8, 800.8 | 801.2 |
| 192 | 2 | 799.5, 802.0, 801.1, 799.8, 801.1 | 801.1 |
| 256 | 0 | 745.9, 748.1, 748.6, 750.0, 746.7 | 748.1 |
| 256 | 1 | 741.7, 740.4, 742.8, 739.5, 742.5 | 741.7 |
| 256 | 2 | 741.9, 741.5, 742.5, 738.9, 744.0 | 741.9 |
| 320 | 0 | 746.3, 745.4, 747.2, 747.2, 746.0 | 746.3 |
| 320 | 1 | 740.1, 740.7, 742.2, 742.3, 742.2 | 742.2 |
| 320 | 2 | 739.5, 743.1, 739.9, 742.4, 740.6 | 740.6 |
| 384 | 0 | 788.8, 789.5, 787.0, 788.3, 786.3 | 788.3 |
| 384 | 1 | 786.1, 785.8, 780.5, 784.0, 781.6 | 784.0 |
| 384 | 2 | 783.2, 783.9, 783.2, 781.7, 779.5 | 783.2 |
| 400 | 0 | 674.6, 675.1, 673.6, 673.6, 673.7 | 673.7 |
| 400 | 1 | 668.4, 671.6, 670.1, 669.2, 668.9 | 669.2 |
| 400 | 2 | 667.4, 669.9, 667.1, 667.1, 667.1 | 667.1 |
| 416 | 0 | 768.0, 769.1, 769.5, 768.9, 768.8 | 768.9 |
| 416 | 1 | 767.7, 767.4, 766.5, 767.9, 767.3 | 767.4 |
| 416 | 2 | 767.1, 767.5, 766.9, 767.3, 768.5 | 767.3 |
| 432 | 0 | 793.7, 793.0, 794.3, 794.4, 794.0 | 794.0 |
| 432 | 1 | 790.3, 790.7, 791.5, 791.0, 791.7 | 791.0 |
| 432 | 2 | 790.7, 792.0, 792.0, 791.4, 792.4 | 792.0 |
| 448 | 0 | 814.2, 818.3, 818.2, 817.6, 816.9 | 817.6 |
| 448 | 1 | 814.1, 816.5, 816.1, 817.0, 818.1 | 816.5 |
| 448 | 2 | 814.7, 817.1, 817.1, 816.8, 816.7 | 816.8 |
| 464 | 0 | 832.4, 832.6, 833.0, 832.5, 832.3 | 832.5 |
| 464 | 1 | 830.9, 830.4, 831.3, 830.7, 831.6 | 830.9 |
| 464 | 2 | 828.2, 830.0, 830.7, 831.4, 832.0 | 830.7 |
| 480 | 0 | 851.3, 852.3, 851.0, 852.5, 852.6 | 852.3 |
| 480 | 1 | 848.6, 851.0, 850.7, 851.3, 852.1 | 851.0 |
| 480 | 2 | 851.4, 851.2, 850.3, 851.6, 851.6 | 851.4 |
| 512 | 0 | 882.8, 883.7, 884.2, 884.4, 885.4 | 884.2 |
| 512 | 1 | 882.4, 883.6, 885.5, 884.2, 883.9 | 883.9 |
| 512 | 2 | 881.2, 881.8, 884.5, 883.2, 884.0 | 883.2 |
| 2048 | 0 | 861.5, 862.1, 861.5, 862.0, 862.0 | 862.0 |
| 2048 | 1 | 861.5, 861.8, 860.8, 861.4, 861.1 | 861.4 |
| 2048 | 2 | 859.9, 860.3, 860.1, 860.1, 859.5 | 860.1 |

### Gate/up multi-wave arm — raw decode tg1 samples (not a win/regression claim)

| process | samples (n=5) | process median |
|---:|---|---:|
| 0 | 47.726550537787546, 47.62180514833621, 47.72630225630389, 47.71231610898046, 47.54877957973916 | 47.71231610898046 |
| 1 | 47.66688019840099, 47.600866354808, 47.61205998243496, 47.74588798088942, 47.62448814986055 | 47.62448814986055 |
| 2 | 47.4318026726967, 47.53777161181188, 47.35632375510643, 47.6036353753849, 47.45236778773835 | 47.45236778773835 |

---

## 5 · Class B — wave-count / threshold screens

Single-process supporting screens (tok/s medians from evidence). Used to choose
the gate/up multi-wave even/odd `tiles64` policy on top of the already-accepted
residual-MW stack; not a substitute for the three-process final matrix.

### Main screen (base / MW4 / MW8 / MW12 / MW16)

| arm | 384 | 416 | 448 | 480 | 512 | 2048 |
|---|---:|---:|---:|---:|---:|---:|
| base (prior residual-MW stack; gate BT at large N) | 774.4 | 756.0 | 798.1 | 818.9 | 838.8 | 818.4 |
| MW4 | 757.6 | **767.4** | **817.0** | 807.3 | 841.7 | 821.9 |
| MW8 | **787.5** | 765.7 | 815.0 | **850.2** | **882.7** | **861.2** |
| MW12 | 774.4 | 722.5 | 767.7 | 802.0 | 837.6 | 818.8 |
| MW16 | 709.6 | 760.5 | 801.5 | 823.9 | 842.8 | 823.1 |

Reading:

- **MW8** wins the large even-`tiles64` shapes (**pp480** / **pp512** / **pp2048**
  and **pp384** on this grid).
- **MW4** is competitive / stronger on several mid-band odd-`tiles64` points
  (**pp416** / **pp448** here).
- **MW12 / MW16 are not hard failures.** Both remain at or above base on several
  shapes and are not correctness rejects. They are **slower than MW8** on the
  large-N winners that matter for promotion, so they are **not** promoted. Do not
  describe them as broken or rejected-for-correctness.

### Fine screen (base / MW4 / MW8 around the even/odd handoff)

| arm | 384 | 400 | 416 | 432 | 448 | 464 | 480 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 763.8 | 659.7 | 751.4 | 774.9 | 796.2 | 804.8 | 818.0 |
| MW4 | 753.1 | **669.4** | **767.2** | **792.5** | **815.9** | 787.5 | 805.8 |
| MW8 | **783.5** | 669.1 | 764.8 | 791.7 | 815.5 | **830.9** | **851.0** |

Reading that drove production even/odd `tiles64` routing:

- `tiles64 = ceil(N / 64)` pads N up to a 64-token compute tile count so the
  multi-wave LDS cooperative layout stays aligned to fixed tile geometry.
- **Even** `tiles64` → **MW8** (e.g. N=384 → 6; N=464 → 8; N=480 → 8; N=512 → 8).
- **Odd** `tiles64` → **MW4** (e.g. N=400 → 7; N=416 → 7; N=432 → 7; N=448 → 7).
- Padding exists so partial final tiles still participate in the same-row
  multi-wave schedule without a separate ragged kernel; the even/odd split then
  picks the wave count that won the corresponding band on this fixture rather
  than forcing one wave count across both parities.
- **pp400** is an awkward padded shape on this grid (absolute tok/s dips vs
  neighbors under base and both MW arms); it is retained in the final matrix as
  a continuity / policy-stress point, **not** as a headline win shape.

### Selected production gate/up policy (exact gfx1100)

| gate/up band | policy |
|---|---|
| `N < 384` | prior gate adaptive **BT6** (`N = 96..383`) / base outside that band |
| `N ≥ 384` | `tiles64 = ceil(N / 64)`; **even → MW8**, **odd → MW4** |

Residual multi-wave (MW4 `416..463` / MW8 `≥464`), QKVZA/QKV BT, chunk **512**,
adaptive-KV/eviction caps **256**, and all non-`gfx1100` arches remain as in the
prior residual-MW ledger.

---

## 6 · Class C — numerical parity

**Status: PASS** under the hardened harness. Comparison is **GPU base vs GPU
candidate**, not a host oracle.

**Comparison mode (not a host oracle):** GPU historical gate/up base/BT kernel
vs GPU gate/up multi-wave candidate on the same device; hard-gate
`got.to_bits() == want.to_bits()` on every downloaded element for **gate** and
**up** outputs.

**Hardened harness contract**
(`crates/rdna-compute/examples/test_mq4v2_gate_up_bt_gfx1100.rs`):

- Distinct gate/up **projection salts** → distinct packed weight bytes (swapped
  projection routing hard-fails `to_bits`). Distinct salts caught routing.
- Row geometry `gate_m=40`, `up_m=53`: unequal M so the gate/up boundary row
  **40** lies inside the 16-row tile covering **[32, 48)**; total **93** rows →
  **13-row** partial final tile (not clean 16-tiling).
- Before every base / BT candidate / MW candidate launch, gate and up F32
  outputs are filled with **distinct quiet-NaN** payloads via HtoD. Unwritten
  tail elements or accidental `+=` leave NaN and hard-fail **finite**.
- Existing finite/nondegenerate + raw `to_bits` checks then prove full overwrite,
  completeness, and no accidental `+=`.

| surface | shapes | variants | semantics | raw `f32::to_bits` |
|---|---|---|---|---|
| gate/up BT continuity | N ∈ {128, 256} | tiles ∈ {6, 12} | **gate** and **up**; hardened routing / overwrite contract | **exact** |
| gate/up multi-wave | N ∈ {383, 384, 511, 512} | waves ∈ {4, 8} | **gate** and **up**; hardened routing / overwrite contract | **exact** |

**Observed (hardened GPU rerun):**
`PASS: all N={128,256} × BT{6,12} and N={383,384,511,512} × MW{4,8} × {gate,up}
raw-bit equal (f32::to_bits) under distinct gate/up salts + quiet-NaN overwrite
init (gate_m=40/up_m=53 boundary-in-tile32..47, final-tile 13); finite/nondegenerate
proves full overwrite (no leftover NaN / no accidental +=)`.

This proves GPU candidate output identity against the historical GPU gate/up
surface on the exercised N×tile / N×wave set under the hardened
routing/overwrite contract. It is **not** a separate host-side dequant/GEMM
oracle.

---

## 7 · Class D — user-facing checks

| check | result |
|---|---|
| First cold serve run | prefill **85.3** tok/s; decode **12.5** tok/s — **JIT**, not steady-state |
| Second warm run (identical prompt) | prefill **823.6** tok/s; decode **46.8** tok/s |
| Long committed prompt md5 | `7ef3f606d2826a08e161bbeb4848b11e` |
| Warm output start | `**Technical Review: Halyard Distributed Cache (v3)**` |
| Coherence | decoded coherently (same Halyard prefix class as prior residual-MW stage; non-degeneration under the exercised prompt) |

Serve/long-prompt checks are acceptance evidence for the promoted direct path.
They are **not** a natural-stop / open-ended EOS quality claim. Cold first-run
**12.5** decode is JIT compile cost, **not** a decode regression claim relative
to the prior stage’s ~12.5–12.6 cold / ~47 warm decode band.

---

## 8 · Class E — post-port serialized profile

Single post-port profile under the gate/up multi-wave policy (machine-local log
`/home/kaden/qcal/q38-gfx1100-mwgate-20260823/post-mwgate-profile.log`):

| metric | value |
|---|---:|
| wall | **2441.32** ms |
| serialized | **2247.34** ms |
| prefill | **838.9** tok/s |
| kernel | **911.3** tok/s |
| startup overhead | **193.98** ms |
| gate/up | **912.1** ms (**40.6%**) |
| residual | **661.4** ms (**29.4%**) |
| QKVZA | **356.1** ms (**15.8%**) |
| QKV | **99.5** ms (**4.4%**) |

Gate/up remains the **primary** serialized family share after this port
(**40.6%**), ahead of residual (**29.4%**). Residual stays near the residual-MW
profile band; this stage’s lift is gate/up-side, not a residual re-tune.

**State:** **1000 tok/s not reached.** For a 2048-token prefill wall budget at
1000 tok/s (**2048 ms**), remaining wall gap is **~393 ms**
(`2441.32 − 2048`). This profile is attribution only — not a product throughput
claim. Gate is still the primary lever on the remaining gap.

---

## 9 · Class F — Redline disclosure (does **not** pass)

The promoted gate/up multi-wave path **explicitly excludes** capture/replay.
Serve/long-prompt (Class D), Class C hardened GPU raw-bit parity, and the
measured Class A performance screen are the acceptance evidence. Do **not**
cite this record as a Redline route pass.

Inherited stable capture / AQL / PM4 limitations from the prior residual-MW and
full-BT checkpoints still apply to the broader exact-gfx1100 MQ4V2 surface. This
ledger does **not** re-run or re-claim those steps and does **not** claim a pass.

| step | outcome |
|---|---|
| Path scope | **excludes** capture/replay — exact gfx1100 gate/up multi-wave outside that route only |
| New MW gate/up route | **not** admitted into capture/replay |
| Inherited decode capture / AQL / PM4 status | see residual-MW §9 / full-BT §9 (stable capture enumeration; AQL still not exact; PM4 still blocked by inherited scratch limitation) — **not** re-litigated here as a pass |

Raw report/log paths (machine-local only): under
`/home/kaden/qcal/q38-gfx1100-mwgate-20260823/` including machine-local
`summary.json`.

---

## 10 · Mechanism (accepted source defaults)

Exact-`gfx1100` only:

1. Gate/up multi-wave kernel mechanism (promoted path):
   - **same-row cooperative block** for gate/up
   - **static 8 KiB** multi-wave LDS
   - **one accumulator per wave**
   - **raw exact routing / overwrite** contract on gate and up outputs (not
     residual `Y +=`); hardened-harness Class C GPU rerun **PASS** (see §6)
2. Production exact-gfx1100 **non-capture** gate/up multi-wave policy:
   - for `N ≥ 384`: `tiles64 = ceil(N / 64)`; **even → MW8**, **odd → MW4**
   - for `N < 384`: prior gate **BT6** (and prior base outside the BT band)
3. Residual multi-wave / residual BT below 416, QKVZA/QKV, chunk **512**,
   adaptive-KV/eviction caps **256**: **unchanged** from residual-MW.
4. Other arches: **no change**.
5. Capture/replay: **out of scope** for the promoted gate/up multi-wave path.

### Padding rationale (brief)

`tiles64 = ceil(N / 64)` is the compute tile count after padding N up to a
multiple of 64. Padding keeps the multi-wave LDS schedule on fixed 64-token tile
geometry instead of a ragged final partial. The **parity of that padded tile
count** then selects MW8 (even) vs MW4 (odd), matching the screen winners on
this fixture rather than forcing a single wave count across both parities.

### Resource footprint (promoted MW arms)

| arm | VGPR | spills | private bytes | LDS |
|---|---:|---:|---:|---:|
| MW4 | **90** | **0** | **0** | **8192** (8 KiB) |
| MW8 | **90** | **0** | **0** | **8192** (8 KiB) |

This is a **gate/up multi-wave LDS scheduling** change on the exact-gfx1100
MQ4V2 gate/up surface, layered on the prior residual-MW ledger — not a mutation
or soft rewrite of
[`2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md).

## 11 · Verdict

1. **Accepted (exact gfx1100 only):** gate/up **MW** for `N ≥ 384` via
   `tiles64 = ceil(N/64)` with **even → MW8** / **odd → MW4**, prior gate BT6
   below 384, and prior residual-MW + QKVZA/QKV + chunk **512** held fixed;
   other arches unchanged; adaptive-KV/eviction caps stay **256**;
   capture/replay excluded.
2. **Headline prefill lifts (median-of-three-process-medians):**
   - incremental vs residual-MW: pp384 **+2.82%**, pp448 **+2.65%**, pp480
     **+4.38%**, pp512 **+5.96%**, pp2048 **+5.75%** (**861.4** tok/s); smaller
     shapes neutral by design
   - total vs pre-BT base: pp2048 **+63.0%** (from **528.5** → **861.4**)
3. **Decode ~47.5–47.7 tok/s process medians** — recorded, **not** claimed as a
   decode win or regression; serve cold first turn **12.5** is JIT.
4. **Screen history:** MW8 wins large even-`tiles64` shapes; MW4 wins/holds
   odd-`tiles64` mid-band points; MW12/MW16 are slower-than-MW8 on the large-N
   winners, **not** hard failures.
5. **Parity Class C PASS** on screened gate/up BT N128/256 tiles 6/12 and MW
   N383/384/511/512 waves 4/8 for **gate** and **up**: distinct salts caught
   routing; unequal `gate_m=40` / `up_m=53` crosses the boundary inside tile
   **[32, 48)** with total **93** → **13-row** final-tile tail; distinct
   quiet-NaN sentinels plus finite/nondegenerate prove overwrite, completeness,
   and no accidental `+=`; raw `f32::to_bits` equal on every element —
   **not** a host-oracle claim (GPU base vs GPU candidate only).
6. **Serve cold JIT + warm identical prompt** coherent under the exercised
   prompt; Halyard output prefix (`7ef3…` prompt md5).
7. **Profile:** wall **2441.32** ms / serialized **2247.34** ms; gate **912.1** ms
   (**40.6%**) still primary; residual **661.4** ms (**29.4%**); **1000 tok/s not
   reached** (~**393** ms remaining wall gap vs 2048 ms @ 1000 tok/s).
8. **Redline route did not pass**; promoted gate/up multi-wave path excludes
   capture/replay; inherit residual-MW / full-BT capture/AQL/PM4 limitations
   without claiming a pass.
9. **Historical / fixture-bound.** Cite only with date, host, BDF, HIP, base
   revision, binary md5s, model path/md5, method flags, and evidence class.

## 12 · Non-claims

- Not a product, registry, or multi-SKU throughput number.
- Not transferable to gfx1201 / gfx1151 / CDNA / other arches.
- Not a mutation of the prior residual-MW ledger, full-BT ledger, gate-only
  ledger, or the rejected same-day chunk/MMQ screen.
- Not a decode throughput win or decode regression claim.
- Not a Redline PM4/AQL capture/replay pass.
- Not a claim that MW12/MW16 are correctness failures (they are slower than MW8
  on the large-N winners on this fixture).
- Not a host-oracle claim for gate/up MW/BT (Class C is GPU-base vs GPU-candidate
  raw-bit compare on gate and up under the hardened salts / boundary-crossing /
  quiet-NaN overwrite contract; **PASS** on the exercised set).
- Not a natural-stop / open-ended generation-quality claim.
- Not a 1000 tok/s claim.
- Not a claim that smaller-shape pp128–320 deltas are multi-wave wins (those
  shapes stay on prior gate BT6 policy by design).
- Not a claim that pp400’s absolute dip is a regression introduced by MW (base
  also dips there on the fine screen; it is an awkward padded continuity shape).
- Not a claim that machine-local `/home/kaden/qcal/...` paths are durable repo
  artifacts.
