# Qwen3.8-27B — gfx1100 MQ4V2 prefill batch-tile (chunk512 + adaptive BT)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **accepted** for the exact source defaults below on exact-`gfx1100`
only. This is fixture-bound measured evidence under the method below. It is **not**
a product baseline, SKU claim, registry throughput number, or transferable
cross-arch result. Newest file ≠ automatic product baseline.

| accepted default (exact `gfx1100` only) | value |
|---|---|
| Ordinary Qwen prefill chunk | **512** |
| Adaptive MQ4V2 gate/up batch-tile | **BT6** for `N = 96..383`; **BT12** for `N ≥ 384` |
| Every other arch string | **unchanged** (no port of these defaults) |
| Adaptive-KV outer chunk hard cap | **256** (unchanged) |
| Eviction internal chunk hard cap | **256** (unchanged) |

**Related (context only; not a mutation):**

- [`2026-08-23-qwen38-gfx1100-prefill-screen.md`](./2026-08-23-qwen38-gfx1100-prefill-screen.md) —
  same-day exact-`gfx1100` prefill screen of **chunk size alone** and a
  default-off **Q8_1 MMQ** gate/up candidate. That record’s disposition is
  **rejected** (no source default change; remains chunk 256 / F16-WMMA on that
  base). **This file is a new mechanism and a new ledger entry** — adaptive
  MQ4V2 gate/up **batch-tile** plus ordinary chunk **512** — not an amendment,
  rewrite, or soft flip of that rejected result. Do not collapse the two
  dispositions.
- [`2026-08-23-qwen38-gfx1201-prefill-chunk384.md`](./2026-08-23-qwen38-gfx1201-prefill-chunk384.md) —
  accepted exact-`gfx1201` chunk default 384. Linked so same-day Qwen3.8 prefill
  ledgers are not conflated. **Do not** transfer gfx1201 bt12/384 magnitudes or
  policy onto gfx1100.

---

## 1 · Question

On exact **gfx1100** (7900 XTX class), does promoting ordinary prefill chunk
**512** together with an adaptive MQ4V2 **gate/up** batch-tile policy
(**BT6** mid-N, **BT12** large-N) improve synthetic and native prefill without
moving adaptive-KV / eviction hard caps off **256**, and without claiming a
Redline capture/replay route pass?

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked **healthy** |
| Base revision | `aee774858` |
| Model path | `/home/kaden/qcal/ladder-v2-gfx1100/artifacts/qwen3.8-27b.mq4v2.xt.hfq` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |
| Spec / KV | **spec off**; **kv-mode q8**; **kv-backend vmm** |
| Final CLI (`hipfire`) md5 | `35b4e05ac82ee65fdf3511c0c58231c3` |
| Final daemon md5 | `6dc7967d40ed97173655597ac5a8d716` |

Post-run device health: overall **idle**, KFD present (`true`), `dstate=0`,
link **active**. No recovery action required.

Machine-local (non-repo) evidence root:

- `/home/kaden/qcal/q38-gfx1100-bt-screen-20260823/` — raw reports/logs
- `/home/kaden/qcal/q38-gfx1100-bt-screen-20260823/summary.json` — structured summary

These paths are **host-local run artifacts**, not durable in-tree fixtures.
Repo research capsule:
`.codeinsight+research/q38-gfx1100-mq4v2-batch-tile-20260823.json`.

## 3 · Evidence taxonomy

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix (final arms)** | three fresh processes × `hipfire bench --matrix` with full five-sample arrays | product SKU SLA; Redline PM4/AQL pass; other arches |
| **B. pp2048 BT × chunk screens** | single-process forced chunk + forced BT screens used to choose adaptive policy | three-process final admission by themselves |
| **C. Numerical parity** | GPU base gate/up vs GPU BT6/BT12 outputs; **hard raw f32 bit equality** plus tolerance reporting for screened N×BT | host oracle; wall-clock tok/s |
| **D. User-facing serve / long prompt** | five-prompt battery + one committed long prompt coherence | open-ended EOS quality; Redline |
| **E. Redline disclosure** | capture attempt outcomes and blockers | acceptance of a capture/replay product route |

Acceptance for this ledger is **Class A + C + D**. Class E is disclosed and
**does not pass**. The promoted BT path **explicitly excludes** capture/replay.

---

## 4 · Class A — final synthetic matrix

### Method

Guarded exclusive device; three **fresh processes** per final arm (base vs
promoted):

```bash
hipfire bench --matrix \
  --pp 128,512,2048 \
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

| pp | base | promoted | Δ |
|---:|---:|---:|---:|
| 128 | **526.4** | **600.8** | **+14.13%** |
| 512 | **541.3** | **653.3** | **+20.69%** |
| 2048 | **528.5** | **640.0** | **+21.10%** |

Decode **tg1** stayed about **47 tok/s** on both arms. This record is a
**prefill** win claim only — **not** a decode win.

### Base arm — process medians and raw five-sample arrays

Process medians (tok/s):

| pp | p0 | p1 | p2 | median-of-medians |
|---:|---:|---:|---:|---:|
| 128 | 527.3 | 524.3 | 526.4 | **526.4** |
| 512 | 541.6 | 541.3 | 541.1 | **541.3** |
| 2048 | 527.9 | 528.5 | 528.6 | **528.5** |

Raw prefill samples (tok/s):

| pp | process | samples (n=5) | process median |
|---:|---:|---|---:|
| 128 | 0 | 525.5, 522.6, 527.4, 528.0, 527.3 | 527.3 |
| 128 | 1 | 525.6, 524.3, 524.3, 523.0, 521.5 | 524.3 |
| 128 | 2 | 522.4, 522.3, 526.8, 526.4, 526.9 | 526.4 |
| 512 | 0 | 542.2, 541.6, 541.9, 539.6, 541.4 | 541.6 |
| 512 | 1 | 541.7, 540.3, 541.7, 541.3, 540.6 | 541.3 |
| 512 | 2 | 541.1, 539.6, 541.3, 541.0, 541.7 | 541.1 |
| 2048 | 0 | 528.4, 528.9, 527.9, 527.9, 526.9 | 527.9 |
| 2048 | 1 | 528.7, 529.2, 528.5, 528.4, 528.1 | 528.5 |
| 2048 | 2 | 528.8, 529.3, 527.6, 528.6, 528.0 | 528.6 |

Raw decode tg1 samples (tok/s; not a win claim):

| process | samples (n=5) | process median |
|---:|---|---:|
| 0 | 47.0579, 47.0488, 46.9448, 46.9464, 47.0876 | 47.0488 |
| 1 | 46.8702, 46.9777, 47.0753, 47.0011, 47.1099 | 47.0011 |
| 2 | 47.1721, 46.9263, 47.1839, 46.9674, 46.9660 | 46.9674 |

### Promoted arm — process medians and raw five-sample arrays

Process medians (tok/s):

| pp | p0 | p1 | p2 | median-of-medians |
|---:|---:|---:|---:|---:|
| 128 | 604.3 | 600.8 | 600.2 | **600.8** |
| 512 | 657.8 | 653.3 | 652.8 | **653.3** |
| 2048 | 644.0 | 640.0 | 639.9 | **640.0** |

Raw prefill samples (tok/s):

| pp | process | samples (n=5) | process median |
|---:|---:|---|---:|
| 128 | 0 | 604.7, 604.2, 604.3, 604.4, 603.7 | 604.3 |
| 128 | 1 | 600.1, 600.4, 600.8, 602.1, 601.0 | 600.8 |
| 128 | 2 | 600.0, 600.2, 600.7, 600.3, 599.0 | 600.2 |
| 512 | 0 | 657.8, 658.5, 657.6, 658.1, 657.0 | 657.8 |
| 512 | 1 | 654.3, 653.3, 653.1, 652.2, 654.9 | 653.3 |
| 512 | 2 | 651.1, 653.7, 652.8, 653.3, 652.1 | 652.8 |
| 2048 | 0 | 645.2, 644.6, 644.0, 643.7, 642.9 | 644.0 |
| 2048 | 1 | 640.8, 640.9, 640.0, 640.0, 639.8 | 640.0 |
| 2048 | 2 | 640.0, 639.9, 640.6, 639.4, 639.7 | 639.9 |

Raw decode tg1 samples (tok/s; not a win claim):

| process | samples (n=5) | process median |
|---:|---|---:|
| 0 | 47.6528, 47.5396, 47.5613, 47.4370, 47.4483 | 47.5396 |
| 1 | 47.5382, 47.4656, 47.4916, 47.5496, 47.4610 | 47.4916 |
| 2 | 47.4785, 47.4320, 47.5245, 47.4639, 47.5034 | 47.4785 |

---

## 5 · Class B — pp2048 chunk × BT screens

Single-process supporting screens (tok/s medians from evidence). Used to choose
the adaptive BT policy and ordinary chunk **512**; not a substitute for the
three-process final matrix.

### Main chunk grid (every measured BT arm)

| chunk | base | bt4 | bt6 | bt8 | bt10 | bt12 | bt14 | bt16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 533.9 | 616.0 | 632.7 | 629.3 | 610.3 | 620.8 | 552.0 | 615.0 |
| 384 | 534.3 | 607.8 | 632.7 | 633.1 | 588.2 | 635.4 | 618.4 | 594.2 |
| **512** | 535.6 | 611.8 | 626.0 | 624.2 | 578.7 | **641.0** | 603.7 | 623.0 |
| 1024 | 536.7 | 596.7 | — | 582.8 | — | 613.0 | — | 581.3 |

### BT12 fine chunk sweep

| chunk | bt12 pp2048 |
|---:|---:|
| 320 | 632.9 |
| 448 | 629.0 |
| 576 | 636.0 |
| 640 | 613.1 |
| 768 | 626.4 |
| 896 | 619.3 |

Screen reading that drove acceptance: at chunk **512**, **BT12** is the best
measured large-N tile on this grid; mid-N favors **BT6** over forcing BT12
everywhere. Ordinary chunk **512** pairs with the adaptive gate (BT6 below 384,
BT12 at ≥384) rather than a single global tile.

### BT16 question (explicit)

**BT12 was never a hardware maximum.** **BT16** was compiled and matched the
base GPU outputs under the Class C tolerance gates (parity log also reports
observed relL2/maxAbs 0 and cosine 1 for BT16 on N=128/256), but it was
**slower** than BT12 on the measured chunk grid (e.g. chunk512: BT12 **641.0**
vs BT16 **623.0**; chunk384: BT12 **635.4** vs BT16 **594.2**). Resource report
from the screen:

| tile | VGPR | VGPR spills | private |
|---|---:|---:|---:|
| **BT6** | 119 | 0 | 0 B |
| **BT12** | 256 | 13 | 56 B |
| **BT16** (screen only) | 256 | 3 | 16 B |

Practical winner is **adaptive BT6 / BT12**, not BT16. BT16 is retained only as
a negative screen / compile proof in this ledger.

---

## 6 · Class C — numerical parity

Harness: `crates/rdna-compute/examples/test_mq4v2_gate_up_bt_gfx1100.rs`
(machine-local summary also in
`/home/kaden/qcal/q38-gfx1100-mq4v2-bt-parity-20260823.log`).

**Comparison mode (not a host oracle):**

1. **Reference:** historical base GPU kernel `gemm_gate_up_mq4g256v2_wmma`,
   forced by arming `capture_mode` for that launch only so production adaptive
   BT is skipped.
2. **Candidate:** direct GPU launcher
   `gemm_gate_up_mq4g256v2_wmma_gfx1100_bt` for **BT6** and **BT12**.
3. Compare downloaded **gate** and **up** F32 tensors independently.

There is **no** separate host-side dequant/GEMM oracle. The harness hard-gates
**raw f32 bit equality** on every downloaded gate/up element via
`got.to_bits() == want.to_bits()` (including +0/−0 and any other bit
difference). Finite/nondegenerate/relL2/cosine/maxAbs remain reported and are
still required, but **cannot substitute** for the raw-bit gate:

| gate | threshold |
|---|---|
| **raw bit equality** (`f32::to_bits`) | **hard required** — all elements |
| finite + nondegenerate | required (variance > 1e-12) |
| rel-L2 | `≤ 1e-5` |
| cosine | `≥ 1 − 1e-6` |
| maxAbs | reported; not a hard fail threshold |

Shapes **N ∈ {128, 256}** × tiles **BT ∈ {6, 12}** × {gate, up}. The
post-review rerun exercised the hard `f32::to_bits` gate on every downloaded
element:

| metric | observed |
|---|---|
| gate rel-L2 | **0** |
| up rel-L2 | **0** |
| cosine | **1** |
| maxAbs | **0** |
| raw-bit equality (`to_bits`) | **true for every gate/up element across all 8 shape/tile/projection arms** |
| disposition | **raw-bit PASS** |

This proves GPU candidate output identity against the historical GPU base
kernel on the exercised surface; it is not a separate host-side dequant/GEMM
oracle.

---

## 7 · Class D — user-facing checks

| check | result |
|---|---|
| Built-in five-prompt serve battery | coherent; **empty = 0**; **attractor = 0** |
| Long committed prompt md5 | `7ef3f606d2826a08e161bbeb4848b11e` |
| Long-prompt measured prefill | **640.7** tok/s |
| Long-prompt measured decode | **46.9** tok/s |
| Long-prompt output start | `**Technical Review: Halyard Distributed Cache (v3)**` |
| Coherence | decoded coherently (non-degeneration under the exercised prompt) |

Serve/long-prompt checks are acceptance evidence for the promoted direct path.
They are **not** a natural-stop / open-ended EOS quality claim.

---

## 8 · Class E — Redline disclosure (does **not** pass)

The promoted BT path **explicitly excludes** capture/replay. Direct kernel
parity (Class C) and serve/long-prompt (Class D) are the acceptance evidence.
Do **not** cite this record as a Redline route pass.

Observed on the disclosure attempt:

| step | outcome |
|---|---|
| Prefill capture | hit the harness’s existing missing `redline_capture` response / `KeyError` |
| Decode capture | **stable**: **1043** launches, **16** kernels, sequence hash `ad87296b93f5c32f` |
| Contracts | enumerated **16** kernels |
| PM4 shadow | **blocked** by pre-existing `gemv_mq4g256v2_residual` scratch unsupported (`private12`) |
| AQL shadow | **not exact** while blob matched HIP |

Raw report/log paths (machine-local only): under
`/home/kaden/qcal/q38-gfx1100-bt-screen-20260823/` including `summary.json`.

---

## 9 · Mechanism (accepted source defaults)

Exact-`gfx1100` only:

1. Ordinary Qwen prefill chunk default **512** (was 256 on the prior rejected
   chunk/MMQ screen’s baseline; that screen remains historically rejected and
   unamended).
2. Adaptive MQ4V2 **gate/up** batch-tile:
   - **BT6** for `N = 96..383`
   - **BT12** for `N ≥ 384`
3. Other arches: **no change**.
4. Adaptive-KV and eviction hard caps: remain **256**.
5. Capture/replay: **out of scope** for the promoted path.

This is a **batch-tile + chunk** scheduling change on the MQ4V2 F16-WMMA family,
not the rejected MMQ experiment and not a silent edit of
`2026-08-23-qwen38-gfx1100-prefill-screen.md`.

## 10 · Verdict

1. **Accepted (exact gfx1100 only):** chunk **512** + adaptive gate/up
   **BT6 (96..383) / BT12 (≥384)**; other arches unchanged; adaptive-KV/eviction
   caps stay **256**.
2. **Headline prefill lifts (median-of-three-process-medians):** pp128
   **+14.13%**, pp512 **+20.69%**, pp2048 **+21.10%**.
3. **Decode ~47 tok/s** — recorded, **not** claimed as a decode win.
4. **BT16** compiled and matched base GPU outputs under the same tolerance
   gates (observed zeros on the screen) but was slower; adaptive **BT6/BT12**
   is the practical winner.
5. **Parity PASS** on screened N×BT gate/up: GPU base vs GPU BT, tolerance-
   gated; observed relL2/maxAbs 0 and cosine 1 — **not** a host-oracle claim.
6. **Serve battery + long prompt** coherent under the exercised prompts.
7. **Redline route did not pass**; promoted path excludes capture/replay.
8. **Historical / fixture-bound.** Cite only with date, host, BDF, HIP, base
   revision, binary md5s, model path/md5, method flags, and evidence class.

## 11 · Non-claims

- Not a product, registry, or multi-SKU throughput number.
- Not transferable to gfx1201 / gfx1151 / CDNA / other arches.
- Not a mutation or soft acceptance of the rejected same-day chunk/MMQ screen.
- Not a decode throughput win.
- Not a Redline PM4/AQL capture/replay pass.
- Not a BT16 promotion.
- Not a host-oracle or raw device-bit identity claim for gate/up BT (Class C is
  GPU-base vs GPU-BT float compare).
- Not a natural-stop / open-ended generation-quality claim.
- Not a claim that machine-local `/home/kaden/qcal/...` paths are durable repo
  artifacts.
