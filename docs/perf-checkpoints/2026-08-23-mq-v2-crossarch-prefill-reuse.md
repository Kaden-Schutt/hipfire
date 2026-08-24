# MQ V2 — cross-arch prefill weight-reuse / QKV BT8 screen

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **fixture-bound measured evidence** under the method below.
Candidate screen commit `4d7a26f8`; promotion commit `b7159076`. This record
documents the same-day cross-arch prefill reuse / QKV BT screen and the final
no-env medians on the promotion binary. It is **not** a current product
baseline, automatic registry throughput number, SKU SLA, or standing admission
decision. Newest file ≠ current baseline. **MQ2V2 is measured here as a wire /
runtime arm only — do not call MQ2 shipping** (product quality rejection is
recorded separately on the 2026-08-20 ladder).

| field | value |
|---|---|
| Candidate commit | `4d7a26f8` (A/B screen binary) |
| Promotion commit | `b7159076` (final no-env binary) |
| HIP / ROCm | **7.14** |
| Model family | dense **Qwen3.8-27B** XT ladder artifacts (mq2–mq6 V2) |
| Measurement surface | synthetic `hipfire bench --matrix` only |

**Binary digest disclosure (required):**

- **Candidate A/B binary md5 was not captured.** The `4d7a26f8` screen arms have
  no retained `hipfire` / `daemon` md5 in this ledger. Treat those arms as
  commit-bound only.
- **Final no-env digests were captured** on the promotion commit `b7159076` and
  are recorded per host below.

**Related (context only; not a mutation of those records):**

- [`2026-08-20-qwen38-mq-v2-product-ladder.md`](./2026-08-20-qwen38-mq-v2-product-ladder.md)
  — quality ladder; **MQ2V2 product-rejected**. This file does not reopen that
  quality disposition.
- Same-day gfx1100 / gfx1201 prefill chunk and batch-tile checkpoints — linked
  only so prefill ledgers are not conflated. **Do not** transfer those defaults
  or magnitudes onto the arms below without their own fixtures.

---

## 1 · Question

On the measured hosts, does MQ V2 prefill **weight-reuse** (gfx1151) and/or
**QKV batch-tile BT8** (gfx1201 ABBA) improve synthetic matrix prefill tok/s
across the MQ2/3/5/6 V2 XT fixtures relative to the then-base path, and what
final no-env medians land on promotion commit `b7159076`?

---

## 2 · Fixture

### Model artifacts (Qwen3.8-27B XT)

| codec | qt (register) | md5 |
|---|---:|---|
| mq2v2 | 50 | `12d3cafef619b552f78e9ae557b899ed` |
| mq3v2 | 49 | `80bb9198e6a565fc006b2ae1b7c89eca` |
| mq4v2 | 44 | `e45d15bfe0c9a87132697101d17cbed6` |
| mq5v2 | 48 | `e7c17382812f33df90f5138eb7cf9973` |
| mq6v2 | 47 | `de6eb059a10577b80f0a162e5c89249e` |

Throughput tables below report **mq2 / mq3 / mq5 / mq6** only (the arms exercised
for base→reuse and final no-env). mq4v2 is part of the fixture set and parity
surface; no mq4 base→reuse or final no-env median series is recorded in this
file.

### Method (all matrix arms)

```bash
hipfire bench --matrix \
  --pp 128,256,512,2048 \
  --ctx 128 \
  --tg 1 \
  --runs 3 \
  --warmups 5 \
  --spec off \
  --kv-mode q8 \
  --kv-backend vmm \
  --json
```

- Spec **off**; KV **q8**; backend **vmm**.
- Synthetic matrix: **no prompt file and no prompt md5** (fixed pp lengths only).
- Headline numbers below are **medians** (tok/s) at pp **128 / 256 / 512 / 2048**
  in that order.
- Decode **tg1** is collected by the matrix command but is **not** a win claim
  in this record.

### Hosts / final binaries (promotion `b7159076`)

| host | arch class | final `hipfire` md5 | final `daemon` md5 |
|---|---|---|---|
| **hipx** (gfx1151 final no-env) | gfx1151 | `4e50322cfddda9dd55f45b3c8d49336e` | `9d981075694d850733de376bc7ff8ce4` |
| **hiptrx** (gfx1201 final no-env) | gfx1201 | `479be96736a157cadde20ba67844cb42` | `201ccecedf5a6362ee3414fbabbac354` |

Candidate-screen (`4d7a26f8`) binary digests: **not captured**.

### Remote log pointers (non-durable)

| path | role |
|---|---|
| `/home/kaden/qcal/crossarch-20260823/` | discovery pointer — full JSON logs |
| `/home/kaden/qcal/crossarch-gfx1151-20260823/` | discovery pointer — gfx1151 JSON logs |

These paths are **host-local discovery pointers**, not durable in-tree fixtures
and not reproduction authority once they rotate or disappear. This markdown
record is the immutable ledger entry.

---

## 3 · gfx1151 — base → weight-reuse (candidate screen)

**Host class:** gfx1151  
**Commit:** candidate `4d7a26f8`  
**Binary md5:** **not captured**  
**Mechanism under test:** prefill **weight-reuse** defaults (all ops, bits 2–6
on the screen surface that produced the rows below).

Medians tok/s at pp **128 / 256 / 512 / 2048**:

| codec | base | reuse | notes |
|---|---|---|---|
| mq2v2 | 207.3 / 203.0 / 202.2 / 195.6 | 342.6 / 312.2 / 310.1 / 296.6 | runtime arm only; **not** product shipping |
| mq3v2 | 178.2 / 175.7 / 173.2 / 170.3 | 248.2 / 217.4 / 214.1 / 208.7 | |
| mq5v2 | 136.0 / 135.0 / 134.7 / 133.3 | 207.9 / 165.9 / 164.0 / 161.7 | |
| mq6v2 | 132.1 / 127.9 / 127.6 / 126.4 | 235.5 / 193.6 / 191.9 / 187.5 | |

Screen reading: reuse lifts every reported codec/pp cell on this fixture vs the
same-day base path. Magnitudes are fixture-bound to this method and host class.

---

## 4 · gfx1151 — final no-env (promotion `b7159076`)

**Host:** hipx  
**Commit:** `b7159076`  
**Binaries:** `hipfire` `4e50322cfddda9dd55f45b3c8d49336e` · `daemon`
`9d981075694d850733de376bc7ff8ce4`  
**Env:** final **no-env** matrix (no forced candidate env overrides).

Medians tok/s at pp **128 / 256 / 512 / 2048**:

| codec | final no-env |
|---|---|
| mq2v2 | 348.3 / 316.1 / 314.6 / 306.1 |
| mq3v2 | 249.9 / 218.4 / 216.8 / 210.4 |
| mq5v2 | 209.9 / 166.3 / 166.1 / 162.0 |
| mq6v2 | 235.8 / 194.0 / 193.0 / 187.4 |

Final no-env lands in the same band as the candidate reuse arm (slightly higher
on several cells). This is a **historical measurement snapshot** on the
promotion binary — not a standing SLA.

---

## 5 · gfx1201 — ABBA conclusion and final no-env

**Host:** hiptrx  
**Commit (final):** `b7159076`  
**Binaries:** `hipfire` `479be96736a157cadde20ba67844cb42` · `daemon`
`201ccecedf5a6362ee3414fbabbac354`  
**Candidate A/B binary md5:** **not captured**

### ABBA conclusion (this record)

- **Promote** QKV **BT8** for bits **2 / 5 / 6** (mq2v2 / mq5v2 / mq6v2 on the
  measured surface).
- **Leave MQ3 on base** (do not promote BT8 for bit 3 / mq3v2 on this evidence).

That conclusion is the disposition of the gfx1201 ABBA screen as recorded here.
It is not a claim about every future binary, SKU, or arch string.

### Final no-env medians (promotion `b7159076`)

Medians tok/s at pp **128 / 256 / 512 / 2048**:

| codec | final no-env |
|---|---|
| mq2v2 | 754.5 / 775.9 / 833.0 / 832.5 |
| mq3v2 | 670.7 / 687.7 / 798.8 / 816.2 |
| mq5v2 | 598.0 / 636.8 / 753.3 / 783.3 |
| mq6v2 | 731.8 / 688.5 / 907.5 / 935.2 |

mq2 rows are **throughput measurement only**. They do **not** reverse the
MQ2V2 product quality rejection.

---

## 6 · Parity evidence (supporting)

| surface | result |
|---|---|
| gfx1100 + gfx1151 | **32 / 32** raw-bit arms PASS |
| gfx1201 | **24 / 24** projections PASS |

Parity PASS is numerical identity evidence on the exercised arms. It is **not**
by itself a product admission, SKU claim, or quality claim (especially not for
MQ2V2).

Capture / replay contracts are **out of scope** for this record; they retain
their fixed contracts elsewhere and are not re-opened here.

---

## 7 · Verdict

1. **Historical / fixture-bound.** Cite only with date, commits (`4d7a26f8` /
   `b7159076`), HIP 7.14, XT md5s, method flags, host, and (where present)
   final binary md5s.
2. **gfx1151:** candidate weight-reuse lifted mq2/3/5/6 synthetic prefill vs
   base; final no-env on `b7159076` matches the reuse band.
3. **gfx1201:** ABBA conclusion — promote QKV BT8 for bits **2/5/6**; **leave
   MQ3 base**. Final no-env medians recorded on hiptrx promotion binaries.
4. **MQ2V2 does not ship** from this file. Runtime/wire measurement ≠ product
   admission.
5. **Candidate A/B binary md5 missing; final no-env digests present.** Do not
   invent or backfill candidate digests.
6. **Not** a current baseline, registry throughput row, or cross-arch transfer
   license. Newest file ≠ current baseline.

---

## 8 · Non-claims

- Not a product, registry, or marketing throughput number.
- Not an automatic source-default or SKU admission beyond the historical ABBA
  conclusion stated in §5.
- Not a claim that MQ2V2 is quality-admitted or shipping.
- Not transferable to unmeasured arches, prompts, KV modes, spec routes, or
  HIP versions without a new record.
- Not a durable archive of `/home/kaden/qcal/crossarch*` JSON trees — those are
  discovery pointers only.
- Not a capture/replay, DFlash, or MTP claim.
- Not an amendment or soft rewrite of the 2026-08-20 quality ladder.
