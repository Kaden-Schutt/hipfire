# Qwen3.8-27B — gfx1201 prefill chunk default 384

**Date:** 2026-08-23  
**Lifecycle:** `historical`  
**Disposition:** **accepted** on the quant/quality branch as the exact-`gfx1201`
source default for Qwen prefill chunking (`PREFILL_DEFAULT_BATCH_GFX1201 = 384`).
This is fixture-bound measured evidence under the method below. It is **not** a
current product baseline, automatic admission decision, SKU claim, or transferable
throughput number. Newest file ≠ current baseline. Ideal 1000 tok/s was **not**
reached.

**Related (context only; unchanged):**

- [`2026-08-21-mq4v2-prefill-mmq-screen.md`](./2026-08-21-mq4v2-prefill-mmq-screen.md) —
  kernel-only MQ4V2 F16-WMMA vs MMQ screen. On gfx1201 F16-WMMA wins every screened
  N; this chunk-size change stays on that F16-WMMA path and does **not** reopen MMQ.
- [`2026-08-22-qwen38-mqv2-mtp-prefix-cache-AMENDMENT-4.md`](./2026-08-22-qwen38-mqv2-mtp-prefix-cache-AMENDMENT-4.md) —
  Q8 query-tiled verify candidate **rejected**. No relation to prefill chunk size;
  linked so the two closed gfx1201 screens are not conflated.

---

## 1 · Question

On exact **gfx1201** (R9700), does raising the Qwen prefill chunk default from
256 → **384** improve long synthetic and native prompt prefill without breaking
hidden-ring / adaptive-KV / eviction safety caps that must remain at 256?

## 2 · Host, binaries, model

| field | value |
|---|---|
| Host | **hiptrx** |
| Device | HIP device **3**, Radeon AI PRO **R9700**, **gfx1201**, BDF `0000:13:00.0` |
| HIP / ROCm | **7.14** |
| Guard | exclusive device lock/guard for all runs |
| Base revision | `047032ef2` |
| Model md5 | `e45d15bfe0c9a87132697101d17cbed6` |

| binary class | role | md5 |
|---|---|---|
| final CLI | post-change `hipfire` | `58ebc8d9a062bcc3f4f64e272773d64f` |
| final daemon | post-change daemon | `53e537c85019223783506ca8f634d68d` |
| baseline rebuild CLI | pre-change / 256-default rebuild | `3876eb6a3cd41672658ad0e16d0cdd0a` |
| baseline rebuild daemon | pre-change rebuild | `356df04cf6616fbc71768ffa65e2b6ad` |

Post-run device health: GPU ended **`runtime_status=suspended`** (healthy idle).
No recovery action required.

## 3 · Evidence taxonomy

Three **distinct** evidence classes. Do not collapse them into one “tok/s” claim.

| class | what it measures | what it does **not** measure |
|---|---|---|
| **A. Synthetic matrix** | `hipfire bench --matrix` fixed pp lengths, synthetic token streams, fresh processes | natural language quality, decode quality, product SKU SLA |
| **B. Native generate** | same final binary A/B on a real prompt file (`adaptive_kv_long_prefill.txt`) | multi-arch portability; ideal 1000 tok/s; stop-token behavior |
| **C. Internal serialized profile** | relative share of serialized GPU time across named ops for one long prefill | wall-clock end-to-end product number; absolute µs portability |

---

## 4 · Class A — synthetic matrix

### Method

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

- Three **fresh processes** per arm (baseline rebuild vs final).
- Spec off; Q8 KV; VMM backend.
- Each process reports a median over its measured runs; the headline number is
  the **median of the three process medians**.

### Synthetic token stream identity (u32le)

| pp | token-stream md5 (u32le payload) |
|---:|---|
| 128 | `b988825426d8d0777c5c707b012f3815` |
| 512 | `d86de7a8e62f65ae54f3ed9007c15588` |
| 2048 | `719576fb6ea13fe404e57bddc6d87cc6` |

### Fresh-process medians (tok/s)

**Baseline (chunk default 256):**

| pp | process medians | median-of-medians |
|---:|---|---:|
| 128 | 738.9 / 736.9 / 706.0 | **736.9** |
| 512 | 727.8 / 725.7 / 727.5 | **727.5** |
| 2048 | 706.3 / 706.2 / 707.0 | **706.3** |

**Final (chunk default 384 on exact gfx1201):**

| pp | process medians | median-of-medians | Δ vs baseline median |
|---:|---|---:|---:|
| 128 | 713.4 / 708.3 / 706.4 | **708.3** | −3.9% (see note) |
| 512 | 858.0 / 848.1 / 847.3 | **848.1** | **+16.6%** |
| 2048 | 875.7 / 865.8 / 866.9 | **866.9** | **+22.7%** |

**pp128 note (non-claim):** at N=128 the final and baseline sources are
**source-identical** for chunking (both emit a single ≤128 chunk; 384 never
splits or widens N=128). Observed spread is treated as **non-actionable noise**.
This record does **not** claim a pp128 win.

### Chunk-size screens (same matrix class; supporting)

On the same host/device class, absolute pp2048 screens:

| forced chunk | pp2048 tok/s (screen) | disposition vs 384 |
|---:|---:|---|
| **384** | (matrix median **866.9**) | **won** |
| 512 | 644.6 | regresses |
| 768 | 804.7 | below 384 |

512/768 are negative screens for the default choice, not product configs.

---

## 5 · Class B — native generate A/B

Same **final** binary; only the effective prefill chunk differs
(`HIPFIRE_PREFILL_MAX_BATCH=256` override vs default 384).

| field | value |
|---|---|
| Prompt | `benchmarks/prompts/adaptive_kv_long_prefill.txt` |
| Prompt md5 | `7ef3f606d2826a08e161bbeb4848b11e` |

| arm | prefill tok/s | TTFT (ms) | decode tok/s |
|---|---:|---:|---:|
| override 256 | 699.9 | 1337.4 | 35.1 |
| default **384** | **831.6** | **1125.6** | 35.2 |
| Δ | **+18.8%** | **−15.8%** | ~0 (noise) |

Decode is neutral. Prefill/TTFT move together as expected for a chunk-bound
prompt path. This is native generate evidence, not the synthetic matrix number.

### Coherence (native; not a stop claim)

Long prompt, context length **936**, produced a **coherent structured review**:
no empty completion, no attractor loop, no retrieval miss on the exercised
prompt. Output remained **length-limited even at max_tokens=1024**, so this is
**not** a natural-stop or EOS-quality claim — only a non-degeneration check under
a hard length cap.

---

## 6 · Class C — internal serialized profile

One long prefill profile on the 256-default baseline identified the lever.
Shares are **serialized** internal attribution, not wall-clock product SLAs.

| bucket | serialized share |
|---|---:|
| Projection GEMMs (sum) | **89.3%** |
| — gate_up | 43.7% |
| — residual | 29.7% |
| — qkvza | 11.9% |
| — qkv | 4.0% |
| GDN | 5.1% |
| remainder | ~5.6% |

At default chunk **256**, pp2048 splits into **eight** 256-token chunks. At
default **384**, the same 2048-token prompt becomes **five** 384-token chunks
plus a **128**-token tail (`5×384 + 128 = 2048`).

## 7 · Mechanism

gfx1201 HFQ4/MQ4 F16-WMMA prefill GEMMs use adaptive batch-tile (`bt`) selection
shared across gate_up / residual / qkvza (env `HIPFIRE_GATE_UP_BT`, default on):

```text
B = 12  if N % 192 == 0   (else if N ≥ 192 without tighter modulus match)
B =  8  if N % 128 == 0
B =  4  if N %  64 == 0
…
```

- **N=384:** `384 % 192 == 0` → **bt12**. Five full chunks hit the highest
  batch-tile fast path; weight loads are reused across more batch columns per
  launch; launch/prologue amortization improves vs eight×256.
- **N=512:** `512 % 128 == 0` (and not `% 192`) → **bt8**. Larger chunk but
  **weaker tile**; measured pp2048 **644.6** tok/s — worse than both 256 and 384.
- **N=128 tail:** still a valid residual chunk; does not undo the five bt12 bodies.

So 384 is not “bigger is always better”; it is the chunk size that keeps the
hot path on **bt12** while cutting chunk count on long prompts. This is an
**exact-gfx1201** F16-WMMA scheduling effect. It does not contradict the
2026-08-21 MMQ screen (MMQ remains slower on gfx1201 at these N).

## 8 · Safety contract (accepted source defaults)

| path | bound | notes |
|---|---|---|
| Exact `gpu.arch == "gfx1201"` default | **384** | `PREFILL_DEFAULT_BATCH_GFX1201`; not gfx1200 / other gfx12 |
| Every other arch string | **256** | `PREFILL_MAX_BATCH` |
| Explicit override | `HIPFIRE_PREFILL_MAX_BATCH≥2` | still honored |
| Hidden-ring staging | **min(configured, ring)** | 256-row ring cannot receive a 384-row write |
| Adaptive-KV outer chunk | **capped 256** | controller margin / downshift boundaries unchanged |
| Eviction internal chunk | **capped 256** | outer eviction window / `maybe_evict` cadence unchanged |

No cross-arch default change. No product-doc 1000 tok/s claim.

## 9 · Verdict

1. **Accepted (quant/quality branch, exact gfx1201 only):** default prefill chunk
   **384** for Qwen on gfx1201; **256** elsewhere; safety caps above retained.
2. **Wins are long-prompt / multi-chunk:** synthetic **pp512 +16.6%**, **pp2048
   +22.7%** (median-of-three-process-medians); native long prompt **+18.8%**
   prefill / **−15.8%** TTFT at neutral decode.
3. **No pp128 win claimed** — source-identical chunking at N=128; spread is noise.
4. **512 is a trap** on this path (selects bt8); do not “round up” past 384 without
   a new screen.
5. **Ideal 1000 tok/s not reached.** Projection GEMMs still ~89% of serialized
   prefill; further gains are GEMM/algorithm work (see MMQ screen — not admitted
   on gfx1201), not another blind chunk bump.
6. **Historical / fixture-bound.** Not a product baseline. Cite only with date,
   host, binary md5s, model md5, method flags, and evidence class (A/B/C).

## 10 · Non-claims

- Not a current product or registry throughput number.
- Not a cross-arch (gfx1100 / gfx1151 / CDNA / gfx1200) inference.
- Not a pp128 improvement claim.
- Not a natural-stop, EOS, or open-ended generation-quality claim.
- Not an MMQ admission; does not amend the 2026-08-21 MMQ rejection on gfx1201.
- Not related to the Amendment-4 query-tile verify rejection.
- Not a 1000 tok/s achievement.
