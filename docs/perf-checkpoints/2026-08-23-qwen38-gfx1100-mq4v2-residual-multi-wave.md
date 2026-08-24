# Qwen3.8-27B — gfx1100 MQ4V2 residual multi-wave LDS (exact gfx1100)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **accepted** for the exact source defaults below on exact-`gfx1100`
only. This is fixture-bound measured evidence under the method below. It is **not**
a product baseline, SKU claim, registry throughput number, or transferable
cross-arch result. Newest file ≠ automatic product baseline.

| accepted default (exact `gfx1100` only) | value |
|---|---|
| Ordinary Qwen prefill chunk | **512** (unchanged from prior stages) |
| Adaptive MQ4V2 **gate/up** batch-tile | **BT6** for `N = 96..383`; **BT12** for `N ≥ 384` (unchanged) |
| Adaptive MQ4V2 **QKVZA** / **QKV** batch-tile | **BT4** `96..191`; **BT12** `≥192`; **base** below 96 (unchanged from full-BT) |
| Adaptive MQ4V2 **residual** below **416** | prior residual adaptive BT/base (**BT4** `96..223`; **BT6** `224..287`; **BT8** `288..415`; **base** otherwise) |
| Adaptive MQ4V2 **residual multi-wave** | **MW4** for `N = 416..463`; **MW8** for `N ≥ 464` |
| Every other arch string | **unchanged** (no port of these defaults) |
| Capture / replay path | **excluded** — exact gfx1100 residual multi-wave outside capture/replay only |
| Adaptive-KV outer chunk hard cap | **256** (unchanged) |
| Eviction internal chunk hard cap | **256** (unchanged) |

**Related (context only; not a mutation):**

- [`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md)
  — **prior accepted full adaptive batch-tile stage** (gate/up + residual BT +
  QKVZA/QKV). That ledger remains **unchanged**. This file is the next residual
  stage only: multi-wave LDS residual for large N on top of that full-BT policy.
  Cite incremental lifts against that full-BT arm; cite total lifts against the
  pre-BT base recorded there (pp2048 base **528.5** tok/s).
- [`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md)
  — earlier gate/up-only stage; still unchanged.
- [`2026-08-23-qwen38-gfx1100-prefill-screen.md`](./2026-08-23-qwen38-gfx1100-prefill-screen.md)
  — same-day rejected chunk/MMQ screen; not amended by this ledger.
- [`2026-08-23-qwen38-gfx1201-prefill-chunk384.md`](./2026-08-23-qwen38-gfx1201-prefill-chunk384.md)
  — accepted exact-`gfx1201` chunk default. **Do not** transfer magnitudes or
  policy onto gfx1100.

---

## 1 · Question

On exact **gfx1100**, with the already-accepted full adaptive batch-tile stage
held fixed for shapes below the residual multi-wave band, does promoting an
exact-gfx1100 residual **multi-wave LDS** policy (**MW4** / **MW8**) recover
large-N residual throughput where residual BT had deliberately fallen back to
base, without moving other-arch defaults, without claiming a Redline
capture/replay route pass, and without over-claiming MW12/MW16 as hard failures?

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked **healthy** |
| Base revision | `ba15ae469` |
| Model path | `/home/kaden/qcal/ladder-v2-gfx1100/artifacts/qwen3.8-27b.mq4v2.xt.hfq` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |
| Spec / KV | **spec off**; **kv-mode q8**; **kv-backend vmm** |
| Final CLI (`hipfire`) md5 | `f7122ffecfef8e6c507f2a3bb5dd7a73` |
| Final daemon md5 | `bd34857002fa108a31b949e54b13e14d` |

Post-run device health: overall **idle**, KFD present (`true`), `dstate=0`,
link **active**. No recovery action required.

Machine-local (non-repo) evidence root:

- `/home/kaden/qcal/q38-gfx1100-mwresid-20260823/` — raw reports/logs
- `/home/kaden/qcal/q38-gfx1100-mwresid-20260823/summary.json` — structured summary

These paths are **host-local run artifacts**, not durable in-tree fixtures.
Repo research capsule:
`.codeinsight+research/q38-gfx1100-mq4v2-residual-multi-wave-20260823.json`.

## 3 · Evidence taxonomy

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix (final arm)** | three fresh processes × `hipfire bench --matrix` with full five-sample arrays under residual multi-wave + prior full-BT policy | product SKU SLA; Redline PM4/AQL pass; other arches |
| **B. Wave-count screens** | single-process forced MW2/4/6/8/12/16 and threshold screens used to choose MW4/MW8 bands | three-process final admission by themselves |
| **C. Numerical parity** | GPU base vs GPU residual MW/BT outputs; **hard raw f32 bit equality** on residual surfaces | host oracle; wall-clock tok/s |
| **D. User-facing serve / long prompt** | cold JIT first run + warm identical-prompt serve coherence | open-ended EOS quality; Redline |
| **E. Serialized profile** | post-port wall / serialized / family share attribution | 1000 tok/s claim |
| **F. Redline disclosure** | new residual multi-wave route is capture/replay-excluded; inherited stable capture/AQL/PM4 limitations from full-BT | acceptance of a capture/replay product route |

Acceptance for this ledger is **Class A + C + D**. Class F is disclosed and
**does not pass**. The promoted residual multi-wave path **explicitly excludes**
capture/replay.

---

## 4 · Class A — final synthetic matrix

### Method

Guarded exclusive device; three **fresh processes** under the accepted residual
multi-wave policy layered on the prior full-BT defaults:

```bash
hipfire bench --matrix \
  --pp 128,192,256,320,384,416,448,480,512,2048 \
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

Incremental vs prior **full-BT** promoted arm
([`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md)):

| pp | prior full-BT | multi-wave residual | Δ |
|---:|---:|---:|---:|
| 128 | **679.6** | **680.2** | **+0.09%** (neutral by design) |
| 192 | **801.5** | **804.3** | **+0.35%** (neutral by design) |
| 256 | **742.1** | **743.3** | **+0.16%** (neutral by design) |
| 320 | **741.9** | **743.0** | **+0.15%** (neutral by design) |
| 384 | **763.8** | **762.5** | **−0.17%** (neutral by design) |
| 448 | **704.5** | **795.7** | **+12.95%** |
| 512 | **722.4** | **834.2** | **+15.48%** |
| 2048 | **707.4** | **814.6** | **+15.15%** |

Shapes **below the MW band** (`pp ≤ 384`) remain on the prior residual BT/base
policy; measured deltas there are noise / continuity checks, **not** claimed
wins.

Total vs **pre-BT base** (same prior full-BT ledger’s pre-BT base arm; pp2048
base **528.5**):

| pp | pre-BT base | multi-wave residual | Δ |
|---:|---:|---:|---:|
| 2048 | **528.5** | **814.6** | **+54.13%** |

Decode **tg1** process medians from evidence: **47.782 / 47.660 / 47.649** tok/s.
Recorded for continuity with the full-BT stage (~47 tok/s). This record is a
**prefill** win claim only for the MW residual band — **no** decode regression
claim and **no** decode win claim.

### Multi-wave residual arm — process medians (all measured shapes)

| pp | p0 | p1 | p2 | median-of-medians |
|---:|---:|---:|---:|---:|
| 128 | 683.6 | 680.2 | 679.6 | **680.2** |
| 192 | 809.0 | 804.3 | 801.3 | **804.3** |
| 256 | 748.1 | 743.3 | 739.0 | **743.3** |
| 320 | 748.1 | 743.0 | 739.4 | **743.0** |
| 384 | 768.9 | 762.5 | 760.6 | **762.5** |
| 416 | 752.5 | 751.3 | 751.9 | **751.9** |
| 448 | 795.7 | 794.8 | 795.8 | **795.7** |
| 480 | 815.7 | 814.9 | 816.9 | **815.7** |
| 512 | 834.2 | 832.1 | 835.6 | **834.2** |
| 2048 | 814.6 | 813.8 | 815.6 | **814.6** |

### Multi-wave residual arm — raw five-sample prefill arrays

| pp | process | samples (n=5) | process median |
|---:|---:|---|---:|
| 128 | 0 | 681.9, 683.6, 683.3, 684.0, 683.7 | 683.6 |
| 128 | 1 | 679.6, 681.0, 680.5, 680.0, 680.2 | 680.2 |
| 128 | 2 | 679.2, 679.6, 679.5, 680.6, 679.9 | 679.6 |
| 192 | 0 | 807.3, 808.5, 809.0, 810.8, 811.0 | 809.0 |
| 192 | 1 | 799.7, 806.9, 804.3, 803.3, 804.4 | 804.3 |
| 192 | 2 | 798.3, 800.6, 801.3, 801.8, 804.1 | 801.3 |
| 256 | 0 | 747.7, 748.1, 749.6, 750.9, 745.4 | 748.1 |
| 256 | 1 | 741.6, 744.1, 742.8, 745.6, 743.3 | 743.3 |
| 256 | 2 | 739.0, 741.5, 738.7, 740.6, 736.6 | 739.0 |
| 320 | 0 | 748.6, 747.2, 748.3, 748.1, 748.0 | 748.1 |
| 320 | 1 | 743.6, 743.0, 743.4, 742.7, 742.7 | 743.0 |
| 320 | 2 | 739.4, 740.1, 738.3, 739.0, 740.2 | 739.4 |
| 384 | 0 | 771.6, 767.8, 768.5, 768.9, 769.2 | 768.9 |
| 384 | 1 | 764.7, 760.5, 766.2, 762.5, 761.1 | 762.5 |
| 384 | 2 | 760.7, 756.0, 760.6, 762.1, 760.6 | 760.6 |
| 416 | 0 | 752.0, 751.7, 753.4, 752.9, 752.5 | 752.5 |
| 416 | 1 | 751.1, 750.8, 751.3, 752.2, 753.4 | 751.3 |
| 416 | 2 | 751.9, 751.6, 753.8, 751.9, 752.6 | 751.9 |
| 448 | 0 | 794.8, 794.4, 795.7, 796.7, 795.9 | 795.7 |
| 448 | 1 | 792.8, 793.0, 795.6, 794.9, 794.8 | 794.8 |
| 448 | 2 | 793.9, 795.0, 796.0, 796.0, 795.8 | 795.8 |
| 480 | 0 | 814.0, 815.7, 816.5, 816.1, 815.7 | 815.7 |
| 480 | 1 | 812.8, 814.9, 815.2, 814.9, 815.8 | 814.9 |
| 480 | 2 | 815.5, 816.9, 817.4, 817.4, 816.4 | 816.9 |
| 512 | 0 | 832.7, 833.5, 834.2, 834.9, 834.4 | 834.2 |
| 512 | 1 | 831.6, 832.1, 834.5, 833.6, 831.7 | 832.1 |
| 512 | 2 | 831.0, 835.8, 835.8, 835.6, 834.4 | 835.6 |
| 2048 | 0 | 814.8, 815.8, 814.5, 814.2, 814.6 | 814.6 |
| 2048 | 1 | 815.0, 813.5, 814.0, 813.8, 813.6 | 813.8 |
| 2048 | 2 | 816.0, 816.0, 815.0, 815.3, 815.6 | 815.6 |

### Multi-wave residual arm — raw decode tg1 samples (not a win/regression claim)

| process | samples (n=5) | process median |
|---:|---|---:|
| 0 | 47.592737900272176, 47.86285874229713, 47.83984902509086, 47.66665071422756, 47.7821828172117 | 47.7821828172117 |
| 1 | 47.65959231508138, 47.57610904786742, 47.83968882004651, 47.55908467214931, 47.74467294779669 | 47.65959231508138 |
| 2 | 47.74785085729595, 47.547145015036065, 47.64874186117797, 47.69277383414179, 47.3454248813938 | 47.64874186117797 |

---

## 5 · Class B — wave-count / threshold screens

Single-process supporting screens (tok/s medians from evidence). Used to choose
the residual multi-wave bands on top of the already-accepted full-BT residual
policy below 416; not a substitute for the three-process final matrix.

### First screen (pp448 / 512 / 2048)

| arm | 448 | 512 | 2048 |
|---|---:|---:|---:|
| base (prior residual fallback) | 708.4 | 726.8 | 711.5 |
| MW2 | 735.6 | 756.3 | 741.9 |
| MW4 | **794.6** | 816.4 | 797.0 |
| MW6 | 791.1 | 803.6 | 785.8 |
| MW8 | 790.0 | **836.0** | **816.5** |

Reading:

- **MW8** wins the large-N pair (**pp512** / **pp2048**).
- **MW4** is the strongest mid-large arm at **pp448** on this grid and remains
  competitive at 512/2048.
- **MW2** lifts over base but loses to MW4/MW8.

### Extended screen (MW8 / MW12 / MW16 vs base)

| arm | 448 | 512 | 2048 |
|---|---:|---:|---:|
| base | 709.7 | 727.2 | 711.6 |
| MW8 | **790.6** | **836.6** | **817.1** |
| MW12 | 755.9 | 805.9 | 788.7 |
| MW16 | 774.7 | 804.6 | 786.4 |

**MW12 / MW16 are not hard failures.** Both remain above base on every measured
shape here. They are **slower than MW8** on this fixture, so they are not
promoted. Do not describe them as broken or rejected-for-correctness.

### Threshold screen (band cut between MW4 and MW8)

| arm | 416 | 432 | 448 | 480 | 512 |
|---|---:|---:|---:|---:|---:|
| base | 684.6 | 695.8 | 704.4 | 710.9 | 722.0 |
| MW4 | **751.7** | **775.4** | **796.2** | 795.7 | 817.7 |
| MW8 | 746.8 | 769.5 | 789.2 | **817.1** | **834.8** |

Reading that drove production bands:

- **MW4** wins **416..448** on this grid (and is accepted through **463** as the
  production handoff before MW8).
- **MW8** wins from **480** upward (production **≥464**).
- Below **416**, residual stays on the prior full-BT adaptive BT/base policy.

### Selected production residual policy (exact gfx1100)

| residual band | policy |
|---|---|
| `N < 416` | prior residual adaptive BT/base (**BT4** `96..223`; **BT6** `224..287`; **BT8** `288..415`; **base** otherwise) |
| `N = 416..463` | **MW4** |
| `N ≥ 464` | **MW8** |

Gate/up, QKVZA, QKV, chunk **512**, adaptive-KV/eviction caps **256**, and all
non-`gfx1100` arches remain as in the prior full-BT ledger.

---

## 6 · Class C — numerical parity

**Comparison mode (not a host oracle):** GPU historical residual base/BT kernel
vs GPU residual multi-wave candidate on the same device; hard-gate
`got.to_bits() == want.to_bits()` on every downloaded element. Evidence capsule
reports `parity.raw_bit_equal: true` and `parity.nonzero_y: true`.

| surface | shapes | variants | semantics | raw `f32::to_bits` |
|---|---|---|---|---|
| residual BT continuity | N ∈ {128, 256} | tiles ∈ {4, 6, 8} | output; **nonzero-Y additive** residual (`Y +=`) | **exact** |
| residual multi-wave | N ∈ {511, 512} | waves ∈ {4, 8} | output; **nonzero-Y additive** residual (`Y +=`) | **exact** |

This proves GPU candidate output identity against the historical GPU residual
surface on the exercised N×tile / N×wave set. It is **not** a separate
host-side dequant/GEMM oracle.

---

## 7 · Class D — user-facing checks

| check | result |
|---|---|
| First cold serve run | prefill **85.1** tok/s; decode **12.6** tok/s — **JIT**, not steady-state |
| Second warm run (identical prompt) | prefill **795.4** tok/s; decode **46.8** tok/s |
| Long committed prompt md5 | `7ef3f606d2826a08e161bbeb4848b11e` |
| Warm output start | `**Technical Review: Halyard Distributed Cache (v3)**` |
| Coherence | decoded coherently (same Halyard prefix class as prior full-BT stage; non-degeneration under the exercised prompt) |

Serve/long-prompt checks are acceptance evidence for the promoted direct path.
They are **not** a natural-stop / open-ended EOS quality claim. Cold first-run
**12.6** decode is JIT compile cost, **not** a decode regression claim relative
to the prior stage’s ~12.5 cold / ~47 warm decode band.

---

## 8 · Class E — post-port serialized profile

Single post-port profile under the residual multi-wave policy (machine-local log
`/home/kaden/qcal/q38-gfx1100-mwresid-20260823/post-mw-profile.log`):

| metric | value |
|---|---:|
| wall | **2570.05** ms |
| serialized | **2376.96** ms |
| prefill | **796.9** tok/s |
| kernel | **861.6** tok/s |
| startup overhead | **193.09** ms |
| gate/up share | **44.2%** |
| residual share | **27.6%** |
| QKVZA share | **14.9%** |
| QKV share | **4.2%** |

Residual family time moved **1006.4 ms → 656.1 ms** (~**1.53×** residual-only
speedup on this profile). Residual is no longer the largest serialized share;
**gate/up is now the primary ceiling lever** on this fixture.

**State:** **1000 tok/s not reached.** This profile is attribution only — not a
product throughput claim.

---

## 9 · Class F — Redline disclosure (does **not** pass)

The promoted residual multi-wave path **explicitly excludes** capture/replay.
Direct kernel parity (Class C) and serve/long-prompt (Class D) are the
acceptance evidence. Do **not** cite this record as a Redline route pass.

Inherited stable capture / AQL / PM4 limitations from the prior full-BT
checkpoint
([`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md)
§9) still apply to the broader exact-gfx1100 MQ4V2 surface. This ledger does
**not** re-run or re-claim those steps and does **not** claim a pass.

| step | outcome |
|---|---|
| Path scope | **excludes** capture/replay — exact gfx1100 residual multi-wave outside that route only |
| New MW residual route | **not** admitted into capture/replay |
| Inherited decode capture / AQL / PM4 status | see prior full-BT §9 (stable capture enumeration; AQL still not exact; PM4 still blocked by inherited scratch limitation) — **not** re-litigated here as a pass |

Raw report/log paths (machine-local only): under
`/home/kaden/qcal/q38-gfx1100-mwresid-20260823/` including `summary.json`.

---

## 10 · Mechanism (accepted source defaults)

Exact-`gfx1100` only:

1. Residual multi-wave kernel mechanism (promoted path):
   - **same-row cooperative block**
   - **static 8 KiB tile-major LDS**
   - **128** logical staging jobs
   - **one accumulator per wave**
   - **two uniform barriers per group**
   - exact **K2 / MQ4V2 / interleaved / `Y +=`** residual contract
2. Production exact-gfx1100 **non-capture** residual multi-wave policy:
   - **MW4** for `N = 416..463`
   - **MW8** for `N ≥ 464`
3. Residual below **416**: prior full-BT residual adaptive BT/base — **unchanged**.
4. Gate/up, QKVZA, QKV, chunk **512**, adaptive-KV/eviction caps **256**:
   **unchanged** from full-BT.
5. Other arches: **no change**.
6. Capture/replay: **out of scope** for the promoted residual multi-wave path.

### Resource footprint (promoted MW arms)

| arm | VGPR | spills | private bytes | LDS |
|---|---:|---:|---:|---:|
| MW4 | **90** | **0** | **0** | **8192** (8 KiB) |
| MW8 | **90** | **0** | **0** | **8192** (8 KiB) |

This is a **residual multi-wave LDS scheduling** change on the exact-gfx1100
MQ4V2 residual surface, layered on the prior full-BT ledger — not a mutation or
soft rewrite of
[`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md).

## 11 · Verdict

1. **Accepted (exact gfx1100 only):** residual **MW4** for `N = 416..463` and
   **MW8** for `N ≥ 464`, with prior residual BT/base below 416 and prior
   full-BT gate/up + QKVZA/QKV + chunk **512** held fixed; other arches
   unchanged; adaptive-KV/eviction caps stay **256**; capture/replay excluded.
2. **Headline prefill lifts (median-of-three-process-medians):**
   - incremental vs full-BT: pp448 **+12.95%**, pp512 **+15.48%**, pp2048
     **+15.15%**; smaller shapes neutral by design
   - total vs pre-BT base: pp2048 **+54.13%** (from **528.5** → **814.6**)
3. **Decode ~47.6–47.8 tok/s process medians** — recorded, **not** claimed as a
   decode win or regression; serve cold first turn **12.6** is JIT.
4. **Screen history:** MW8 wins N512/pp2048; MW4 wins 416..448; MW12/MW16 are
   slower-than-MW8 lifts over base, **not** hard failures.
5. **Parity PASS** on screened residual BT N128/256 tiles 4/6/8 and MW N511/512
   waves 4/8: raw `f32::to_bits` exact with nonzero-Y additive semantics —
   **not** a host-oracle claim.
6. **Serve cold JIT + warm identical prompt** coherent under the exercised
   prompt; Halyard output prefix.
7. **Profile:** residual **1006.4 → 656.1 ms** (~1.53×); gate/up is now the
   primary lever; **1000 tok/s not reached**.
8. **Redline route did not pass**; promoted residual multi-wave path excludes
   capture/replay; inherit full-BT capture/AQL/PM4 limitations without claiming
   a pass.
9. **Historical / fixture-bound.** Cite only with date, host, BDF, HIP, base
   revision, binary md5s, model path/md5, method flags, and evidence class.

## 12 · Non-claims

- Not a product, registry, or multi-SKU throughput number.
- Not transferable to gfx1201 / gfx1151 / CDNA / other arches.
- Not a mutation of the prior full-BT ledger, the gate-only ledger, or the
  rejected same-day chunk/MMQ screen.
- Not a decode throughput win or decode regression claim.
- Not a Redline PM4/AQL capture/replay pass.
- Not a claim that MW12/MW16 are correctness failures (they are slower than MW8
  on this fixture).
- Not a host-oracle claim for residual MW/BT (Class C is GPU-base vs GPU-candidate
  raw-bit compare).
- Not a natural-stop / open-ended generation-quality claim.
- Not a 1000 tok/s claim.
- Not a claim that smaller-shape pp128–384 deltas are multi-wave wins (those
  shapes stay on prior residual policy by design).
- Not a claim that machine-local `/home/kaden/qcal/...` paths are durable repo
  artifacts.
