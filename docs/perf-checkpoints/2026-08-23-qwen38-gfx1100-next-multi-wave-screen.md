# Qwen3.8-27B — gfx1100 next multi-wave screen (hybrid gate two-acc + QKVZA MW)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **rejected** — no source default change. Both candidate
families (hybrid gate/up two-accumulator multi-wave; QKVZA multi-wave LDS) were
screened on exact-`gfx1100` and **rejected**. Final cleanup **removes** both
candidate kernels, envs, and launchers. The hardened QKVZA BT parity harness
retains distinct-weights / unequal-dims / quiet-NaN overwrite hardening. Newest
file ≠ product baseline. This is a **historical single-process screen**, fixture-
bound measured evidence under the method below. It is **not** a product baseline,
SKU claim, registry throughput number, or transferable cross-arch result.

**Related (context only; no mutation of prior ledgers):**

- [`2026-08-23-qwen38-gfx1100-mq4v2-gate-up-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-gate-up-multi-wave.md)
  — **prior accepted** exact-`gfx1100` gate/up multi-wave LDS stage (even/odd
  `tiles64` MW4/MW8 on top of residual-MW + full-BT). That ledger remains
  **unchanged** and is the production baseline this screen tried (and failed) to
  beat with the next gate/QKVZA multi-wave candidates.
- [`2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-residual-multi-wave.md)
  — prior accepted residual multi-wave stage; still unchanged.
- [`2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md`](./2026-08-23-qwen38-gfx1100-mq4v2-full-batch-tile.md)
  — prior accepted full adaptive batch-tile stage; still unchanged.
- [`2026-08-23-qwen38-gfx1100-prefill-screen.md`](./2026-08-23-qwen38-gfx1100-prefill-screen.md)
  — same-day earlier rejected chunk/MMQ screen; not amended by this ledger.

---

## 1 · Question

On exact **gfx1100**, with the already-accepted gate/up multi-wave + residual-MW
+ full-BT stack held as the production baseline, do either of the following
clear a full-model prefill admission bar without a correctness regression?

1. **Hybrid gate/up two-accumulator multi-wave** — alternate gate/up MW design
   that halves the grid (and thus concurrency) while doubling per-wave issue
   work across two accumulators, forced at MW4 / MW8 / MW12 / MW16.
2. **QKVZA multi-wave LDS** — port the residual/gate multi-wave LDS pattern onto
   the QKVZA prefill GEMM, forced at MW4 / MW8 / MW12 / MW16.

Neither candidate is admitted by micro parity alone. Throughput must improve
materially vs the live baseline on large-N shapes without mixed regressions that
undercut the prior accepted policy.

---

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked **healthy** |
| Base revision | `9998995e2` |
| Candidate revision | `3d324c1c1` |
| Model path | `/home/kaden/qcal/ladder-v2-gfx1100/artifacts/qwen3.8-27b.mq4v2.xt.hfq` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |
| Spec / KV | **spec off**; **kv-mode q8**; **kv-backend vmm** |

Post-run device health: overall **idle**, KFD present, healthy idle path. No
recovery action required.

Machine-local (non-repo) evidence root:

- `/home/kaden/qcal/q38-gfx1100-nextmw-20260823` — raw reports/logs
- summary JSON is **machine-local** under that root (not a durable in-tree fixture)

Repo research capsule:
`.codeinsight+research/q38-gfx1100-next-multi-wave-rejection-20260823.json`.

---

## 3 · Evidence taxonomy

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Single-process forced-arm screen** | one fresh process per base / forced MW arm; matrix medians over the listed pp lengths | three-process final admission; product SKU SLA; Redline PM4/AQL; other arches |
| **B. Numerical parity** | GPU base vs GPU candidate under hardened routing/overwrite harness; **hard raw f32 bit equality** | wall-clock tok/s; admission by itself |
| **C. Resource occupancy** | VGPR, spill, private, LDS footprint of candidate kernels | that clean occupancy implies a win |

This ledger is **Class A negative evidence** supported by Class B PASS and Class C
clean occupancy. Class B and C are **not** admission. No Redline / decode /
cross-arch claim is made.

---

## 4 · Method (Class A screen)

Guarded exclusive device. **Single fresh process** per arm (base and each forced
candidate wave count). Not a three-process final matrix.

```bash
hipfire bench --matrix \
  --pp 192,256,320,384,400,416,432,448,464,480,512,2048 \
  --ctx 128 \
  --tg 1 \
  --runs 3 \
  --warmups 5 \
  --spec off \
  --kv-mode q8 \
  --kv-backend vmm \
  --json
```

- Spec off; Q8 KV; VMM backend.
- Arms: **base** (accepted gate/up MW + residual-MW + full-BT stack at
  `9998995e2`), then forced **QKVZA MW4/8/12/16** and forced **hybrid gate
  two-acc MW4/8/12/16** under candidate `3d324c1c1`.
- Values below are **single-process screen medians** (tok/s) from the research
  capsule — not median-of-three-process-medians.

---

## 5 · Class B — numerical parity (not admission)

**Status: PASS** for both candidates under the hardened harness. Comparison is
**GPU base vs GPU candidate**, not a host oracle. Hard gate:
`got.to_bits() == want.to_bits()` on every downloaded element.

| candidate | raw-bit equal | N shapes | waves | hardened routing / overwrite |
|---|---|---|---|---|
| hybrid gate two-acc | **true** | 383, 384, 511, 512 | 4, 8, 12, 16 | **true** |
| QKVZA multi-wave | **true** | 191, 192, 511, 512 | 4, 8, 12, 16 | **true** |

Parity PASS does **not** admit either route. Throughput screens reject both.

---

## 6 · Class C — resource occupancy (not admission)

| candidate | VGPR | spills | private bytes | LDS bytes |
|---|---:|---:|---:|---:|
| hybrid gate MW4/8/12/16 | **96** | **0** | **0** | **8192** (8 KiB) |
| QKVZA MW4/8/12/16 | **90** | **0** | **0** | **8192** (8 KiB) |

Both candidates are resource-clean (no spills; fixed 8 KiB LDS; no private
scratch). Clean occupancy is **not** a throughput win and is **not** admission.

---

## 7 · Class A — full screen medians (tok/s)

Single-process medians from
`.codeinsight+research/q38-gfx1100-next-multi-wave-rejection-20260823.json`.

### 7.1 · All arms × all pp

| pp | base | qkvza MW4 | qkvza MW8 | qkvza MW12 | qkvza MW16 | gate2 MW4 | gate2 MW8 | gate2 MW12 | gate2 MW16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | **809.5** | 803.0 | 780.9 | 807.0 | 787.3 | 802.1 | 799.2 | 798.4 | 801.6 |
| 256 | **749.8** | 752.0 | 763.0 | 722.7 | 753.1 | 742.1 | 740.9 | 741.3 | 740.5 |
| 320 | **749.8** | 741.4 | 734.1 | 734.3 | 701.5 | 740.0 | 738.8 | 740.1 | 741.0 |
| 384 | **792.8** | 778.2 | 791.2 | 784.1 | 761.0 | 721.7 | 700.6 | 670.4 | 667.2 |
| 400 | **678.4** | 670.2 | 671.0 | 657.7 | 669.1 | 642.7 | 655.5 | 556.8 | 620.4 |
| 416 | **771.8** | 766.6 | 764.7 | 749.7 | 763.3 | 724.1 | 731.0 | 612.7 | 702.2 |
| 432 | **795.7** | 792.4 | 790.1 | 774.7 | 786.9 | 745.4 | 739.0 | 634.9 | 709.8 |
| 448 | **819.4** | 817.8 | 816.6 | 800.1 | 811.1 | 754.8 | 740.3 | 649.1 | 708.5 |
| 464 | **832.5** | 817.7 | 836.0 | 817.4 | 829.7 | 767.5 | 751.4 | 672.6 | 710.4 |
| 480 | **852.3** | 842.4 | 857.5 | 838.3 | 847.9 | 780.4 | 755.8 | 688.2 | 708.5 |
| 512 | **884.3** | 879.7 | **894.2** | 875.8 | 877.3 | 792.7 | 761.6 | 722.9 | 707.8 |
| 2048 | **863.6** | 856.5 | **871.0** | 854.6 | 854.7 | 774.8 | 744.8 | 706.2 | 693.7 |

### 7.2 · Hybrid gate two-acc vs base (large-N focus)

| pp | base | gate2 MW4 | Δ MW4 | gate2 MW8 | Δ MW8 | gate2 MW12 | gate2 MW16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 792.8 | 721.7 | −9.0% | 700.6 | −11.6% | 670.4 | 667.2 |
| 400 | 678.4 | 642.7 | −5.3% | 655.5 | −3.4% | 556.8 | 620.4 |
| 416 | 771.8 | 724.1 | −6.2% | 731.0 | −5.3% | 612.7 | 702.2 |
| 432 | 795.7 | 745.4 | −6.3% | 739.0 | −7.1% | 634.9 | 709.8 |
| 448 | 819.4 | 754.8 | −7.9% | 740.3 | −9.7% | 649.1 | 708.5 |
| 464 | 832.5 | 767.5 | −7.8% | 751.4 | −9.7% | 672.6 | 710.4 |
| 480 | 852.3 | 780.4 | −8.4% | 755.8 | −11.3% | 688.2 | 708.5 |
| 512 | 884.3 | 792.7 | −10.4% | 761.6 | −13.9% | 722.9 | 707.8 |
| **2048** | **863.6** | **774.8** | **−10.3%** | **744.8** | **−13.8%** | **706.2** | **693.7** |

### 7.3 · QKVZA multi-wave vs base (headline shapes)

| pp | base | best QKVZA arm | best tok/s | Δ |
|---:|---:|---|---:|---:|
| 512 | 884.3 | MW8 | **894.2** | **+1.12%** |
| 2048 | 863.6 | MW8 | **871.0** | **+0.86%** |

Other QKVZA wave counts are flat-to-worse on those shapes (MW4/12/16 all below
base at pp512 and pp2048). Mid-band shapes show mixed regressions under every
QKVZA MW arm (e.g. MW8 pp192 780.9 vs base 809.5; MW12 pp256 722.7 vs 749.8;
MW16 pp320 701.5 vs 749.8).

---

## 8 · Verdicts

### 8.1 · Hybrid gate two-acc multi-wave — **rejected**

Every large-N arm (`N ≥ 384`) is **substantially slower** than the accepted
gate/up multi-wave baseline. Headline pp2048:

| arm | pp2048 tok/s | Δ vs base 863.6 |
|---|---:|---:|
| base | **863.6** | — |
| gate2 MW4 | **774.8** | **−10.3%** |
| gate2 MW8 | **744.8** | **−13.8%** |
| gate2 MW12 | **706.2** | −18.2% |
| gate2 MW16 | **693.7** | −19.7% |

**Why the design did not pay:** the two-accumulator hybrid halves the launch
grid (and therefore concurrent waves on the device) while doubling per-wave
issue work. On this fixture that trade was pure loss: occupancy stayed clean
(96 VGPR, 0 spills, 8 KiB LDS), raw-bit parity passed, and throughput still fell
hard across the entire large-N band. Concurrency lost to the halved grid was not
recovered by the extra work issued per remaining wave. **Reject and remove.**

### 8.2 · QKVZA multi-wave LDS — **rejected**

Best arm is **MW8**, and even that is below a meaningful admission bar with
mixed regressions elsewhere:

| shape | base | QKVZA MW8 | Δ |
|---:|---:|---:|---:|
| 512 | 884.3 | 894.2 | **+1.12%** |
| 2048 | 863.6 | 871.0 | **+0.86%** |

MW4 / MW12 / MW16 are worse than base on those headline shapes. Several smaller
pp points regress under MW8 and the other wave counts. Resource footprint is
clean (90 VGPR, 0 spills, 8 KiB LDS) and raw-bit parity passed — again, **not
admission**. Lifts under +1.2% on a single-process screen with mixed regressions
do not justify replacing the accepted QKVZA batch-tile policy. **Reject and
remove.**

### 8.3 · Cleanup disposition (explicit)

| item | disposition |
|---|---|
| Hybrid gate two-acc MW kernel(s) | **removed** in final cleanup |
| QKVZA multi-wave LDS kernel(s) | **removed** in final cleanup |
| Candidate envs / forced-arm launchers | **removed** in final cleanup |
| Accepted gate/up MW + residual-MW + full-BT defaults | **unchanged** (prior ledgers stand) |
| Hardened QKVZA BT harness | **retained** — keeps distinct weights, unequal dims, and quiet-NaN overwrite hardening from the screen harness work |
| This checkpoint | **historical rejected** evidence only |

---

## 9 · Non-claims

- **Not** a product baseline, SKU SLA, or registry throughput number.
- **Not** a three-process final admission matrix (single fresh process per arm).
- **Not** a Redline / capture / PM4 / AQL claim.
- **Not** a decode (tg1) win or regression claim.
- **Not** a cross-arch result — exact-`gfx1100` fixture only; do not transfer
  magnitudes or policy to gfx1151 / gfx1201 / other arches.
- **Not** admission by parity or by clean VGPR/LDS occupancy.
- Prior accepted gate/up multi-wave ledger is **not** amended by this rejection.

---

## 10 · Pointers

| kind | path |
|---|---|
| Repo research capsule | `.codeinsight+research/q38-gfx1100-next-multi-wave-rejection-20260823.json` |
| Machine-local raw logs | `/home/kaden/qcal/q38-gfx1100-nextmw-20260823` |
| Prior accepted gate/up MW | [`2026-08-23-qwen38-gfx1100-mq4v2-gate-up-multi-wave.md`](./2026-08-23-qwen38-gfx1100-mq4v2-gate-up-multi-wave.md) |
