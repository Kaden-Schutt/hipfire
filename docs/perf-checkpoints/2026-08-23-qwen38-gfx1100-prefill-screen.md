# Qwen3.8-27B — gfx1100 prefill screen (chunk + MMQ)

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **rejected** — no source default change. Exact-`gfx1100`
prefill remains **chunk 256** on the existing **F16-WMMA** path. Candidate
routes that were default-off for this screen were removed; tree restored to
base `e8f586166` aside from this checkpoint. Newest file ≠ current baseline.
This is fixture-bound measured evidence under the method below. It is **not** a
product baseline, SKU claim, or transferable throughput number.

**Related (context only; no cross-arch inference):**

- [`2026-08-23-qwen38-gfx1201-prefill-chunk384.md`](./2026-08-23-qwen38-gfx1201-prefill-chunk384.md) —
  accepted exact-`gfx1201` chunk default 384. Linked so the two same-day Qwen3.8
  prefill screens are not conflated. **Do not** transfer that acceptance, its
  bt12 mechanism, or its win magnitudes onto gfx1100.
- [`2026-08-21-mq4v2-prefill-mmq-screen.md`](./2026-08-21-mq4v2-prefill-mmq-screen.md) —
  kernel-only MQ4V2 F16-WMMA vs MMQ screen that motivated a full-model gfx1100
  gate/up MMQ experiment. This record closes that experiment on full-model
  evidence; it does not amend the kernel-only numbers.

---

## 1 · Question

On exact **gfx1100** (7900 XTX class), does either of the following clear a
predeclared full-model prefill gate of **≥ 1.5%** vs the current default, without
a correctness regression?

1. Raising the Qwen prefill chunk size above the default **256**.
2. Routing dense MQ4V2 **gate + up** prefill (N ≥ 256) through a default-off
   generic **Q8_1 MMQ** path (one activation quant + two SET launches), alone or
   stacked with a larger chunk.

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hipx** |
| Device | HIP device **0**, **gfx1100**, BDF `0000:66:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs; locked healthy |
| Base revision | `e8f586166` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |

Post-run device health: overall **idle**, KFD present (`true`), `dstate=0`,
link **active**. No recovery action required.

## 3 · Evidence taxonomy

Three **distinct** evidence classes. Do not collapse them into one “tok/s”
claim or treat micro parity as a full-model win.

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix / chunk screens** | `hipfire bench --matrix` fixed pp lengths (baseline) and single-process forced-chunk pp2048 screens | natural language quality; product SKU SLA; MMQ admission |
| **B. Micro parity (MMQ candidate)** | standalone gate/up numerical parity of the default-off MMQ path vs F16-WMMA reference | wall-clock full-model tok/s; launch amortization; product default |
| **C. Internal serialized profile + full-model MMQ** | relative GPU-time shares on the default path; full-model pp2048 with MMQ alone and MMQ+chunk512 | absolute µs portability; kernel-only screen ratios; gfx1201/chunk384 transfer |

Predeclared admission gate for any candidate that would change source defaults:
full-model pp2048 improvement **≥ 1.5%** vs the Class A baseline median, with
Class B parity PASS. Neither candidate cleared it.

---

## 4 · Class A — baseline matrix and chunk screens

### Method (baseline)

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

- Spec off; Q8 KV; VMM backend.
- **Three fresh processes**; each process reports a median over its measured
  runs (5 samples); headline is the **median of the three process medians**.
- Default prefill chunk **256**, F16-WMMA (no MMQ).

### Fresh-process baseline medians (tok/s)

| pp | process medians | median-of-medians |
|---:|---|---:|
| 128 | 531.6 / 528.1 / 523.9 | **528.1** |
| 512 | 545.6 / 541.0 / 541.9 | **541.9** |
| 2048 | 533.4 / 528.4 / 529.6 | **529.6** |

### Chunk-size screens (same matrix class; single-process; supporting)

Forced chunk on the same host/device class; pp2048 only. These are screens, not
product configs and not three-process medians.

| forced chunk | pp2048 tok/s (screen) | Δ vs baseline median 529.6 |
|---:|---:|---:|
| **256** (default) | (matrix median **529.6**) | — |
| 384 | 534.4 | +0.9% |
| 512 | 538.5 | +1.7% |
| 1024 | 535.8 | +1.2% |

**Max observed chunk lift: +1.7%** at forced 512. That is at the edge of the
predeclared **≥ 1.5%** gate on a weaker (single-process) protocol and does
**not** justify a source default change on its own. **Rejected** as a gfx1100
chunk default move. gfx1100 remains **chunk 256**.

---

## 5 · Class B — MMQ micro parity (candidate only)

Default-off generic dense **gate + up** MQ4V2 MMQ experiment under test:

| constraint | value |
|---|---|
| Arch gate | exact `gfx1100` only |
| Shape gate | `N >= 256` |
| Path | one Q8_1 activation quant + **two** SET launches (gate, up) |
| Default | **off** for the entire screen; never product-on |

### Parity (PASS)

| metric | value |
|---|---|
| gate rel-L2 | `2.876890e-3` |
| up rel-L2 | `2.880719e-3` |
| cosine | `0.999996` |
| disposition | **PASS** |

Correctness cleared the micro gate. Parity alone is **not** admission.

---

## 6 · Class C — profile and full-model MMQ

### Internal serialized profile (default path; attribution only)

One long prefill profile on the default chunk-256 / F16-WMMA path. Shares are
**serialized** internal attribution, not wall-clock product SLAs.

| bucket | serialized share |
|---|---:|
| Projection GEMMs (sum) | **92.7%** |
| — gate_up | 43.0% |
| — residual | 29.7% |
| — qkvza | 15.5% |
| — qkv | 4.5% |
| GDN | 3.2% |
| remainder | ~4.1% |

gate_up is the largest single bucket, which is why the MMQ experiment targeted
that family. Profile share is not a promise that an alternate gate_up kernel
moves full-model tok/s by the same fraction.

### Full-model MMQ screens (pp2048)

Same matrix-class measurement surface as Class A; candidate default-off MMQ
enabled only for the arms below.

| arm | pp2048 tok/s | notes |
|---|---:|---|
| baseline default256 F16-WMMA | **529.6** | Class A median-of-medians |
| default256 + MMQ gate/up | 532.0 | **+0.45%** vs baseline |
| chunk512 screen (no MMQ) | 538.5 | Class A single-process screen |
| MMQ + chunk512 | 538.7 | **neutral** vs chunk512 alone (+0.04%) |

Full-model MMQ alone is well under the **≥ 1.5%** gate. Stacking MMQ on the
best chunk screen is **neutral** — no additive win.

---

## 7 · Verdict

1. **Rejected.** No gfx1100 source default change. Prefill remains **chunk 256**
   and **F16-WMMA**.
2. **Chunk screens:** max +1.7% (single-process pp2048 @ 512). Not admitted as a
   new default; weaker protocol than the three-process baseline and no mechanism
   win comparable to gfx1201’s bt12/384 case.
3. **MMQ candidate:** micro parity **PASS**; full-model default256 **+0.45%**;
   MMQ+chunk512 **neutral** vs chunk512. Failed the predeclared **≥ 1.5%**
   full-model gate.
4. **Candidate removed.** Default-off generic gate+up MQ4V2 MMQ experiment paths
   restored to base `e8f586166`; only this historical checkpoint remains from
   the screen.
5. **Projection-bound still:** ~92.7% serialized share in projection GEMMs.
   Further gfx1100 prefill work needs a lever that actually moves full-model
   wall time, not another un-gated kernel-only extrapolation.
6. **Historical / fixture-bound.** Cite only with date, host, BDF, HIP, base
   revision, model md5, method flags, and evidence class (A/B/C).

## 8 · Non-claims

- Not a performance win. Not a product or registry throughput number.
- Not a source default, SKU, or admission decision for chunk size or MMQ.
- Not transferable to gfx1201 (see same-day chunk384 acceptance) or any other
  arch. **No gfx1151 results** are part of this record.
- Not an amendment of the 2026-08-21 kernel-only MMQ screen numbers; this is the
  full-model close-out on gfx1100 only.
- Not a claim that parity PASS implies production readiness.
- Not a 1000 tok/s claim or ideal-roofline achievement.
