# Qwen3.8-27B — gfx1100 MQ4V2 full adaptive batch-tile (residual + QKVZA + QKV)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **accepted** for the exact source defaults below on exact-`gfx1100`
only. This is fixture-bound measured evidence under the method below. It is **not**
a product baseline, SKU claim, registry throughput number, or transferable
cross-arch result. Newest file ≠ automatic product baseline.

| accepted default (exact `gfx1100` only) | value |
|---|---|
| Ordinary Qwen prefill chunk | **512** (unchanged from prior gate/up stage) |
| Adaptive MQ4V2 **gate/up** batch-tile | **BT6** for `N = 96..383`; **BT12** for `N ≥ 384` (unchanged prior stage) |
| Adaptive MQ4V2 **residual** batch-tile | **BT4** for `N = 96..223`; **BT6** for `N = 224..287`; **BT8** for `N = 288..415`; **base** otherwise |
| Adaptive MQ4V2 **QKVZA** batch-tile | **BT4** for `N = 96..191`; **BT12** for `N ≥ 192`; **base** below 96 |
| Adaptive MQ4V2 **QKV** batch-tile | same as QKVZA: **BT4** `96..191`; **BT12** `≥192`; **base** below 96 |
| Every other arch string | **unchanged** (no port of these defaults) |
| Capture / replay path | **excluded** — exact gfx1100 BT outside capture/replay only |
| Adaptive-KV outer chunk hard cap | **256** (unchanged) |
| Eviction internal chunk hard cap | **256** (unchanged) |

**Related (context only; not a mutation):**

- [`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md)
  — **prior accepted gate/up stage** (chunk **512** + adaptive gate/up **BT6 /
  BT12** only). That ledger remains **unchanged**. This file is the next
  stage: residual + QKVZA + QKV adaptive BT on top of that gate/up policy.
  Cite incremental lifts against that gate-only promoted arm; cite total lifts
  against the pre-BT base recorded there.
- [`2026-08-23-qwen38-gfx1100-prefill-screen.md`](./2026-08-23-qwen38-gfx1100-prefill-screen.md)
  — same-day rejected chunk/MMQ screen; not amended by either BT ledger.
- [`2026-08-23-qwen38-gfx1201-prefill-chunk384.md`](./2026-08-23-qwen38-gfx1201-prefill-chunk384.md)
  — accepted exact-`gfx1201` chunk default. **Do not** transfer magnitudes or
  policy onto gfx1100.

---

## 1 · Question

On exact **gfx1100**, with the already-accepted gate/up BT + chunk512 stage
held fixed, does promoting adaptive MQ4V2 **residual / QKVZA / QKV** batch-tile
policies improve synthetic and native prefill without moving other-arch
defaults, without claiming a Redline capture/replay route pass, and without
forcing residual BT where the screen shows large-N regression?

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked **healthy** |
| Base revision | `585ecf4d3` |
| Model path | `/home/kaden/qcal/ladder-v2-gfx1100/artifacts/qwen3.8-27b.mq4v2.xt.hfq` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |
| Spec / KV | **spec off**; **kv-mode q8**; **kv-backend vmm** |
| Final CLI (`hipfire`) md5 | `848b1134ea28e9fb56d2d53702657225` |
| Final daemon md5 | `3ce255cc8c67ebddbae47cc402c8f324` |

Post-run device health: overall **idle**, KFD present (`true`), `dstate=0`,
link **active**. No recovery action required.

Machine-local (non-repo) evidence root:

- `/home/kaden/qcal/q38-gfx1100-fullbt-20260823/` — raw reports/logs
- `/home/kaden/qcal/q38-gfx1100-fullbt-20260823/summary.json` — structured summary

These paths are **host-local run artifacts**, not durable in-tree fixtures.
Repo research capsule:
`.codeinsight+research/q38-gfx1100-mq4v2-full-batch-tile-20260823.json`.

## 3 · Evidence taxonomy

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix (final arm)** | three fresh processes × `hipfire bench --matrix` with full five-sample arrays under full adaptive policy | product SKU SLA; Redline PM4/AQL pass; other arches |
| **B. Family / fine / combo screens** | single-process forced-tile screens used to choose residual/QKVZA/QKV adaptive bands | three-process final admission by themselves |
| **C. Numerical parity** | GPU base vs GPU BT outputs; **hard raw f32 bit equality** on residual / QKVZA / QKV surfaces | host oracle; wall-clock tok/s |
| **D. User-facing serve / long prompt** | five-prompt battery + one committed long prompt coherence | open-ended EOS quality; Redline |
| **E. Serialized profile** | post-port wall / serialized / family share attribution | 1000 tok/s claim |
| **F. Redline disclosure** | capture attempt outcomes and blockers | acceptance of a capture/replay product route |

Acceptance for this ledger is **Class A + C + D**. Class F is disclosed and
**does not pass**. The promoted BT path **explicitly excludes** capture/replay.

---

## 4 · Class A — final synthetic matrix

### Method

Guarded exclusive device; three **fresh processes** under the full adaptive
policy (gate/up stage + residual/QKVZA/QKV policies):

```bash
hipfire bench --matrix \
  --pp 128,192,256,320,384,448,512,2048 \
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

Incremental vs prior **gate-only** promoted arm
([`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md)):

| pp | gate-only | full port | Δ |
|---:|---:|---:|---:|
| 128 | **600.8** | **679.6** | **+13.12%** |
| 512 | **653.3** | **722.4** | **+10.58%** |
| 2048 | **640.0** | **707.4** | **+10.53%** |

Total vs **pre-BT base** (same prior ledger’s base arm):

| pp | pre-BT base | full port | Δ |
|---:|---:|---:|---:|
| 128 | **526.4** | **679.6** | **+29.10%** |
| 512 | **541.3** | **722.4** | **+33.46%** |
| 2048 | **528.5** | **707.4** | **+33.85%** |

Decode **tg1** process medians: **47.773 / 47.633 / 47.410** tok/s. Recorded
for continuity with the gate-only stage (~47 tok/s). This record is a
**prefill** win claim only — **no** decode regression claim and **no** decode
win claim.

### Full-port arm — process medians (all measured shapes)

| pp | p0 | p1 | p2 | median-of-medians |
|---:|---:|---:|---:|---:|
| 128 | 686.3 | 679.1 | 679.6 | **679.6** |
| 192 | 812.5 | 801.5 | 801.5 | **801.5** |
| 256 | 750.3 | 742.1 | 740.7 | **742.1** |
| 320 | 750.5 | 741.9 | 741.9 | **741.9** |
| 384 | 772.3 | 763.8 | 761.6 | **763.8** |
| 448 | 707.8 | 704.5 | 703.3 | **704.5** |
| 512 | 724.3 | 721.8 | 722.4 | **722.4** |
| 2048 | 709.1 | 707.4 | 706.6 | **707.4** |

### Full-port arm — raw five-sample prefill arrays

| pp | process | samples (n=5) | process median |
|---:|---:|---|---:|
| 128 | 0 | 685.3, 686.8, 689.0, 686.3, 686.2 | 686.3 |
| 128 | 1 | 679.4, 679.1, 678.2, 679.6, 678.5 | 679.1 |
| 128 | 2 | 677.2, 679.5, 679.7, 680.1, 679.6 | 679.6 |
| 192 | 0 | 808.4, 814.2, 812.9, 807.6, 812.5 | 812.5 |
| 192 | 1 | 798.9, 801.5, 802.1, 803.2, 801.2 | 801.5 |
| 192 | 2 | 797.8, 805.1, 801.8, 801.0, 801.5 | 801.5 |
| 256 | 0 | 749.7, 751.7, 750.8, 749.2, 750.3 | 750.3 |
| 256 | 1 | 742.2, 743.5, 740.4, 742.1, 740.1 | 742.1 |
| 256 | 2 | 739.7, 739.7, 741.5, 741.6, 740.7 | 740.7 |
| 320 | 0 | 751.0, 748.8, 749.4, 750.5, 750.6 | 750.5 |
| 320 | 1 | 742.4, 739.5, 742.3, 738.7, 741.9 | 741.9 |
| 320 | 2 | 741.9, 742.0, 741.0, 741.0, 741.9 | 741.9 |
| 384 | 0 | 774.1, 771.3, 770.3, 774.5, 772.3 | 772.3 |
| 384 | 1 | 764.4, 761.2, 763.8, 764.6, 763.1 | 763.8 |
| 384 | 2 | 761.6, 764.9, 761.0, 760.7, 762.8 | 761.6 |
| 448 | 0 | 706.7, 707.8, 707.5, 708.3, 708.2 | 707.8 |
| 448 | 1 | 703.1, 704.5, 704.6, 704.3, 704.7 | 704.5 |
| 448 | 2 | 702.6, 702.5, 704.4, 703.3, 704.1 | 703.3 |
| 512 | 0 | 724.2, 723.6, 724.3, 725.1, 724.6 | 724.3 |
| 512 | 1 | 719.9, 721.8, 721.8, 722.5, 722.5 | 721.8 |
| 512 | 2 | 719.2, 722.4, 722.0, 722.8, 723.5 | 722.4 |
| 2048 | 0 | 709.7, 709.0, 709.1, 709.3, 708.6 | 709.1 |
| 2048 | 1 | 708.0, 707.4, 707.4, 706.6, 707.0 | 707.4 |
| 2048 | 2 | 706.6, 707.2, 706.6, 706.3, 706.6 | 706.6 |

### Full-port arm — raw decode tg1 samples (not a win/regression claim)

| process | samples (n=5) | process median |
|---:|---|---:|
| 0 | 47.83555132434159, 47.908451165765925, 47.74623905262294, 47.773093095187654, 47.66465815741301 | 47.773093095187654 |
| 1 | 47.700424543318526, 47.52633053764446, 47.5324864343472, 47.63277039337523, 47.64011135399628 | 47.63277039337523 |
| 2 | 47.39523602993427, 47.47408887965992, 47.3601970790409, 47.473275275733094, 47.41043043692741 | 47.41043043692741 |

---

## 5 · Class B — family / fine / combo screens

Single-process supporting screens (tok/s medians from evidence). Used to choose
the residual / QKVZA / QKV adaptive bands on top of the already-accepted
gate/up stage; not a substitute for the three-process final matrix.

Screen baseline here is the **gate-only promoted** surface (chunk512 + gate/up
BT), not the pre-BT base.

### Baseline (gate-only stage on this screen host)

| pp | baseline |
|---:|---:|
| 128 | 602.6 |
| 256 | 654.0 |
| 384 | 667.9 |
| 512 | 658.6 |
| 2048 | 646.1 |

### Seven-tile family screens (forced single family + tile)

Each cell is forced residual **or** QKVZA **or** QKV at the named tile, with
other families left at the gate-only baseline policy.

#### Residual

| pp | base | bt4 | bt6 | bt8 | bt10 | bt12 | bt14 | bt16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 602.6 | **633.1** | 568.8 | 551.9 | 598.2 | 598.4 | 598.5 | 598.8 |
| 256 | 654.0 | 686.0 | **693.7** | 681.7 | 665.2 | 653.8 | 635.1 | 632.9 |
| 384 | 667.9 | 671.4 | 656.4 | **683.1** | 648.2 | 626.0 | 620.6 | 626.1 |
| 512 | 658.6 | 635.7 | 619.8 | 626.0 | 598.1 | 609.9 | 614.5 | 593.8 |
| 2048 | 646.1 | 622.9 | 609.2 | 615.0 | 590.0 | 599.3 | 603.5 | 586.0 |

**Explicit negative:** residual BT at **N ≥ 448** (pp512 / pp2048 on this grid)
**regresses** vs baseline for every measured tile. Base fallback above the
accepted residual bands is **deliberate**, not an oversight.

#### QKVZA

| pp | base | bt4 | bt6 | bt8 | bt10 | bt12 | bt14 | bt16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 602.6 | **631.1** | 618.1 | 627.2 | 621.0 | 613.7 | 560.7 | 552.5 |
| 256 | 654.0 | 692.3 | 677.6 | 676.7 | 690.7 | **687.3** | 645.5 | 656.0 |
| 384 | 667.9 | 708.9 | 708.5 | 702.4 | 697.2 | **715.0** | 692.5 | 682.9 |
| 512 | 658.6 | 698.9 | 693.1 | 689.7 | 681.4 | **703.0** | 681.5 | 688.7 |
| 2048 | 646.1 | 683.6 | 679.2 | 675.9 | 668.0 | **688.8** | 667.8 | 674.5 |

#### QKV

| pp | base | bt4 | bt6 | bt8 | bt10 | bt12 | bt14 | bt16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 602.6 | **609.5** | 604.8 | 608.7 | 608.1 | 608.0 | 601.1 | 598.9 |
| 256 | 654.0 | 659.7 | 654.9 | 658.3 | 658.9 | **659.2** | 651.3 | 660.4 |
| 384 | 667.9 | 676.6 | 671.8 | 672.1 | 671.9 | **680.4** | 673.8 | 672.8 |
| 512 | 658.6 | 665.3 | 659.3 | 662.6 | 658.8 | **666.1** | 660.2 | 663.6 |
| 2048 | 646.1 | 651.9 | 647.2 | 648.9 | 646.8 | **652.6** | 647.1 | 650.2 |

### Fine threshold screens (pp192 / 320 / 448)

Fine baseline (gate-only):

| pp | baseline |
|---:|---:|
| 192 | 650.4 |
| 320 | 630.1 |
| 448 | 642.2 |

| arm | 192 | 320 | 448 |
|---|---:|---:|---:|
| resid_bt4 | **727.8** | 639.5 | 634.3 |
| resid_bt6 | 682.6 | 640.1 | 611.3 |
| resid_bt8 | 649.7 | **676.8** | 625.9 |
| qkvza_bt4 | 690.6 | 670.3 | 683.0 |
| qkvza_bt8 | 659.2 | 662.6 | 673.2 |
| qkvza_bt12 | **695.8** | **674.3** | **682.3** |
| qkv_bt4 | 662.9 | 636.9 | 642.5 |
| qkv_bt12 | **666.9** | **639.8** | **644.1** |

Fine reading that drove residual bands:

- **BT4** wins mid-small N (pp128 / pp192).
- **BT6** wins around pp256; **BT8** wins around pp320–384.
- At **pp448**, every residual BT arm is **below** fine baseline **642.2** —
  same large-N residual regression as the main grid; keep **base**.

Fine reading that drove QKVZA/QKV bands:

- **BT4** is the practical mid-small winner at pp128.
- From **pp192 upward**, **BT12** is the durable large-N winner for both QKVZA
  and QKV on the measured grid (with BT4 remaining competitive only below that
  threshold).

### Combined adaptive screens (policy composition checks)

| combo label | measured pp → tok/s |
|---|---|
| n128 | 128 → **678.0** |
| n192 | 192 → **796.2** |
| n256 | 256 → **740.9** |
| n320-384 | 320 → **737.9**; 384 → **760.9** |
| n448-plus | 448 → **700.9**; 512 → **720.8**; 2048 → **704.4** |

Combo screens are single-process composition checks under the selected adaptive
bands; they agree directionally with the three-process final matrix but are
not the admission numbers.

### Selected production policy (exact gfx1100)

| family | band | tile |
|---|---|---|
| residual | `N = 96..223` | **BT4** |
| residual | `N = 224..287` | **BT6** |
| residual | `N = 288..415` | **BT8** |
| residual | otherwise (incl. `N ≥ 448` and `N < 96`) | **base** |
| QKVZA | `N = 96..191` | **BT4** |
| QKVZA | `N ≥ 192` | **BT12** |
| QKVZA | `N < 96` | **base** |
| QKV | same as QKVZA | same as QKVZA |
| gate/up | prior accepted stage | **BT6** `96..383` / **BT12** `≥384` |
| chunk | prior accepted stage | **512** |

---

## 6 · Class C — numerical parity

**Comparison mode (not a host oracle):** GPU historical base kernel vs GPU
batch-tile candidate on the same device; hard-gate
`got.to_bits() == want.to_bits()` on every downloaded element.

Evidence capsule reports `parity.raw_bit_equal: true` for the screened surfaces:

| family | shapes | tiles | tensors / semantics | raw `f32::to_bits` |
|---|---|---|---|---|
| **residual** | N ∈ {128, 256} | BT ∈ {4, 6, 8} | output; **nonzero-Y additive** residual semantics | **exact** |
| **QKVZA** | N ∈ {128, 256} | BT ∈ {4, 12} | qkv, z, beta, alpha | **exact** |
| **QKV** | N ∈ {128, 256} | BT ∈ {4, 12} | q, k, v | **exact** |

Screened tile set in the capsule also lists BT ∈ {4, 6, 8, 10, 12, 14, 16} as
the family screen surface; the hard raw-bit admission claim above is scoped to
the production tiles actually promoted (residual BT4/6/8; QKVZA/QKV BT4/12) at
N128/256. This proves GPU candidate output identity against the historical GPU
base kernel on the exercised surface; it is **not** a separate host-side
dequant/GEMM oracle.

Gate/up parity for the prior stage remains documented in
[`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md)
and is not re-litigated here.

---

## 7 · Class D — user-facing checks

| check | result |
|---|---|
| Built-in five-prompt serve battery | coherent; **empty = 0**; **attractor = 0** |
| Battery cold first-turn decode | **12.5** tok/s — **JIT**, matches prior gate-only **~12.4**; **not** steady-state decode |
| Battery warm decode | **47.2** tok/s (matches prior ~47) |
| Battery average decode | **40.3** tok/s (matches prior; cold first turn pulls the average down) |
| Long committed prompt md5 | `7ef3f606d2826a08e161bbeb4848b11e` |
| Long-prompt measured prefill | **700.8** tok/s |
| Long-prompt measured decode | **46.8** tok/s |
| Long-prompt output start | `**Technical Review: Halyard Distributed Cache (v3)**` |
| Coherence | decoded coherently (same Halyard output prefix as prior stage; non-degeneration under the exercised prompt) |

Serve/long-prompt checks are acceptance evidence for the promoted direct path.
They are **not** a natural-stop / open-ended EOS quality claim. Cold first-turn
**12.5** is JIT compile cost, not a decode regression relative to the prior
stage’s **12.4**.

---

## 8 · Class E — post-port serialized profile

Single post-port profile under the full adaptive policy (machine-local log
`/home/kaden/qcal/q38-gfx1100-fullbt-20260823/post-fullbt-profile.log`):

| metric | value |
|---|---:|
| wall | **2955.2** ms |
| serialized | **2760.9** ms |
| prefill | **693.0** tok/s |
| kernel | **741.8** tok/s |
| gate/up share | **38.6%** |
| residual share | **36.5%** |
| QKVZA share | **13.2%** |
| QKV share | **3.7%** |
| GDN share | **3.9%** |

**State:** **1000 tok/s not reached.** Residual and gate/up remain the next
ceiling levers on this fixture (largest serialized shares). This profile is
attribution only — not a product throughput claim.

---

## 9 · Class F — Redline disclosure (does **not** pass)

The promoted full-BT path **explicitly excludes** capture/replay. Direct kernel
parity (Class C) and serve/long-prompt (Class D) are the acceptance evidence.
Do **not** cite this record as a Redline route pass.

Observed / inherited disclosure (same fixture class as the prior gate-only
stage; PM4 still blocked by the inherited scratch limitation):

| step | outcome |
|---|---|
| Path scope | **excludes** capture/replay — exact gfx1100 BT outside that route only |
| Decode capture | **stable**: **1043** launches, **16** kernels, sequence hash `ad87296b93f5c32f` |
| Contracts | enumerated **16** kernels |
| AQL shadow | **still not exact** |
| PM4 shadow | **remains blocked** by the inherited scratch limitation from the prior checkpoint (`gemv_mq4g256v2_residual` / private scratch unsupported) |

Raw report/log paths (machine-local only): under
`/home/kaden/qcal/q38-gfx1100-fullbt-20260823/` including `summary.json`.

---

## 10 · Mechanism (accepted source defaults)

Exact-`gfx1100` only:

1. Ordinary Qwen prefill chunk default **512** — **unchanged** from prior
   accepted gate/up stage.
2. Adaptive MQ4V2 **gate/up** batch-tile — **unchanged** prior stage:
   - **BT6** for `N = 96..383`
   - **BT12** for `N ≥ 384`
3. Adaptive MQ4V2 **residual** batch-tile (**new** this stage):
   - **BT4** for `N = 96..223`
   - **BT6** for `N = 224..287`
   - **BT8** for `N = 288..415`
   - **base** otherwise (explicitly including large-N where residual BT
     regressed)
4. Adaptive MQ4V2 **QKVZA** and **QKV** batch-tile (**new** this stage):
   - **BT4** for `N = 96..191`
   - **BT12** for `N ≥ 192`
   - **base** below 96
5. Other arches: **no change**.
6. Adaptive-KV and eviction hard caps: remain **256**.
7. Capture/replay: **out of scope** for the promoted path.

This is a **full-family adaptive batch-tile** scheduling change on the MQ4V2
F16-WMMA surface, layered on the prior gate/up ledger — not a mutation or soft
rewrite of
[`2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-batch-tile.md).

## 11 · Verdict

1. **Accepted (exact gfx1100 only):** residual **BT4/6/8** bands + QKVZA/QKV
   **BT4/BT12** bands on top of prior gate/up **BT6/BT12** + chunk **512**;
   other arches unchanged; adaptive-KV/eviction caps stay **256**; capture/replay
   excluded.
2. **Headline prefill lifts (median-of-three-process-medians):**
   - incremental vs gate-only: pp128 **+13.12%**, pp512 **+10.58%**, pp2048
     **+10.53%**
   - total vs pre-BT base: pp128 **+29.10%**, pp512 **+33.46%**, pp2048
     **+33.85%**
3. **Decode ~47.4–47.8 tok/s process medians** — recorded, **not** claimed as a
   decode win or regression; serve cold first turn **12.5** is JIT.
4. **Residual BT at N ≥ 448 regressed** on the screen; base fallback there is
   deliberate.
5. **Parity PASS** on screened residual / QKVZA / QKV N×BT surfaces: raw
   `f32::to_bits` exact (residual with nonzero-Y additive semantics) — **not** a
   host-oracle claim.
6. **Serve battery + long prompt** coherent under the exercised prompts; same
   Halyard output prefix.
7. **Profile:** residual + gate/up still dominate serialized time; **1000 tok/s
   not reached**.
8. **Redline route did not pass**; promoted path excludes capture/replay; PM4
   still blocked by inherited residual scratch limitation.
9. **Historical / fixture-bound.** Cite only with date, host, BDF, HIP, base
   revision, binary md5s, model path/md5, method flags, and evidence class.

## 12 · Non-claims

- Not a product, registry, or multi-SKU throughput number.
- Not transferable to gfx1201 / gfx1151 / CDNA / other arches.
- Not a mutation of the prior gate-only ledger or the rejected same-day
  chunk/MMQ screen.
- Not a decode throughput win or decode regression claim.
- Not a Redline PM4/AQL capture/replay pass.
- Not a claim that residual BT should run at N ≥ 448 (screen negative).
- Not a host-oracle claim for residual/QKVZA/QKV BT (Class C is GPU-base vs
  GPU-BT raw-bit compare).
- Not a natural-stop / open-ended generation-quality claim.
- Not a 1000 tok/s claim.
- Not a claim that machine-local `/home/kaden/qcal/...` paths are durable repo
  artifacts.
