# Benchmarks

Published **historical** tables only. Every number here is a retained,
fixture-bound observation from an earlier campaign. None of these rows is a
live floor, product default, admission, or route certificate. Per
[`INDEX.md`](INDEX.md), a **measured** claim requires a named fixture, binary
and model identity, and a measurement date on the same report — these tables
do not carry that full identity manifest, so this page is **historical** only
(the inventory date below is not a measurement date).

| Field | Value |
|---|---|
| Page state | **historical** (see [`INDEX.md`](INDEX.md)) |
| Inventory date | 2026-07-19 |
| Audited source ref | `692a726dde53508cb53de1a74c720e75a7c9f33e` |
| Comparison base | `origin/beta` @ `9ffb18da9d1377dfbf759db82641ea039b2e522e` |

## How to read this page

1. Treat every table on this page as **historical** only. Do not upgrade a row
   to **measured** without a complete per-table date, fixture, and
   binary/model-identity evidence manifest **and** a fresh protocol-compliant
   run under
   [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md).
   Measurement and admission are independent: a complete **measured** row is
   still not a product default or route certificate. Any product-default or
   admission claim **additionally** requires an explicit row in
   [`admissions.yml`](admissions.yml) (schema v2; exactly one earned record — fail closed otherwise).
2. Many older rows used the then-default `asym3` KV mode. Current clean configs
   resolve `kv_cache=auto` through the model registry and otherwise fall back to
   `q8`. Do not compare asym3 rows to q8 rows as one A/B.
3. Speed floors used by tooling live in `tests/speed-baselines/<arch>.txt` and
   are exercised by [`scripts/speed-gate.sh`](../scripts/speed-gate.sh) when that
   path’s policy applies. Those files are **not** reproduced here.
4. Validation and promotion routes live only in [`VALIDATION.md`](VALIDATION.md).
   Retired batteries are **historical reproduction only** — never current
   acceptance for a bench claim.
5. Redline-attributed numbers require the certification ladder in
   [`REDLINE.md`](REDLINE.md). Throughput without timed-arm route proof is not a
   Redline certification.

## Claim language (fail closed)

| Allowed | Forbidden without fresh evidence |
|---|---|
| “On \<date\>, fixture \<id\>, median X tok/s” | “Current baseline is X” |
| “Historical DFlash genre table (asym3, max_tokens=120)” | “DFlash is 4× on 27B” as a present product fact |
| “Speed-gate floor in `tests/speed-baselines/…`” | Treating any table below as that floor |
| Link to a dated `perf-checkpoints/` file | Stitching harness exits into an admission |

A bench number without protocol + identity hashes is **rejected** as promotion
evidence ([`VALIDATION.md`](VALIDATION.md)).

## Reproducing measurements

Protocol owner: [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md)
(warmup, fresh-process, noise band, prompt MD5 discipline).

```bash
# Canonical CLI surface (pp/decode matrix depends on flags and model)
hipfire bench qwen3.5:9b

# Optional speed-floor check when that path’s policy applies
./scripts/speed-gate.sh --fast
```

For DFlash genre work, use prompt-MD5-pinned fixtures under
[`benchmarks/prompts/`](../benchmarks/prompts/). Prompt structure swings τ;
byte-identical prompts are mandatory across sessions.

Optional A/B tooling: [`scripts/probe_commits.sh`](../scripts/probe_commits.sh)
(also reachable from `scripts/gates.sh --perf`). Neither script creates an
admission row.

---

## Historical: autoregressive decode (no spec) — 7900 XTX (gfx1100)

**Truth state:** historical
**Fixture notes:** then-default engine config (asym3 KV, FlashAttention auto,
`prompt_normalize=on`). Medians across 5 runs unless noted. Not a current q8
methodology row. No per-row binary/model hash manifest on this page.

| Model | decode | prefill (peak) | effective BW |
|---|---:|---:|---:|
| Qwen 3.5 0.8B MQ4 | **391 tok/s** | **7383 tok/s** | 200 GiB/s |
| Qwen 3.5 4B MQ4 | **180 tok/s** | **2487 tok/s** | 433 GiB/s |
| Qwen 3.5 9B MQ4 | **132 tok/s** | **1663 tok/s** | **654 GiB/s** |
| Qwen 3.5 27B MQ4 | **47 tok/s** | **478 tok/s** | **651 GiB/s** |

Engineering note retained with the snapshot: 9B and 27B decode saturated
~650 GiB/s of the 7900 XTX’s 960 GB/s peak (~68% BW-efficient end-to-end across
weights + KV + activations). Prefill on the smaller sizes was WMMA-bound on the
MQ4 fused projections **under that fixture**.

## Historical: DFlash speculative decode by genre — 7900 XTX

**Truth state:** historical (**superseded methodology**)
**Do not use as a current DFlash baseline.**

This table used `asym3` KV and `max_tokens=120`. Current DFlash performance
claims require the protocol in
[`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) (including
q8 where that is the active KV path, `max_tokens=256` when that is the campaign
contract, ≥3 fresh-process runs, prompt and binary hashes) plus the claim-class
route in [`VALIDATION.md`](VALIDATION.md). A retired battery pass is **not**
acceptance evidence. No per-row
binary/model hash manifest on this page.

DFlash speedup in this snapshot was **genre-conditional**. Code prompts whose
target distribution matched the draft won; long-form prose where high-entropy
continuations diverged could net-lose.

5-run medians under the historical configuration: asym3 KV, `--no-chatml`,
`max_tokens=120`, `prompt_normalize=true`:

| Model | genre | AR tok/s | DFlash tok/s | speedup | τ |
|---|---|---:|---:|---:|---:|
| Qwen 3.5 27B | code (HumanEval/53) | 44.1 | **196.0** (peak 218.6) | **4.45×** | 9.82 |
| Qwen 3.5 27B | prose (Rome essay) | 44.0 | 49.6 | 1.13× | 1.67 |
| Qwen 3.5 27B | instruct (sky-color) | 44.6 | 44.7 | 1.00× | 1.39 |
| Qwen 3.5 9B | code (HumanEval/53) | 124.0 | **329.1** (peak 346.7) | **2.65×** | 6.76 |
| Qwen 3.5 9B | code (HumanEval/0) | 121.9 | **372.9** | **3.06×** | 8.23 |
| Qwen 3.5 9B | instruct (sky-color) | 124.4 | **246.9** | **1.99×** | 4.76 |
| Qwen 3.5 9B | prose (federalist) | **125.3** | 99.4 | 0.79× ✗ | 1.20 |
| Qwen 3.5 9B | prose (Rome) | **122.7** | 98.3 | 0.80× ✗ | 1.20 |
| Qwen 3.6 27B | code (HumanEval/53) | 44.2 | **185.5** | **4.19×** | 9.25 |

**Config context for the snapshot (not a timeless default):** CLI default
`dflash_mode` is `"off"` (`crates/hipfire-config/src/lib.rs`); DFlash is opt-in until a campaign
proves a broader win. Enable globally with `hipfire config set dflash_mode auto`
(dense Qwen 3.5+ on, A3B off unless overridden) or per model with
`hipfire config qwen3.5:27b set dflash_mode on`. The numbers above were measured
with DFlash forced on.

## Historical: vs ollama (Q4_K_M GGUF) — 7900 XTX

**Truth state:** historical
Same-machine snapshot: hipfire MQ4 with asym3 KV and FlashAttention versus
ollama Q4_K_M through llama.cpp’s ROCm backend. Matched ~140-token and
~530-token prompts and matched 128-token generation lengths. Ollama numbers from
its `prompt_eval_duration` / `eval_duration` reporting via `/api/generate` with
`num_predict=128`. No per-row binary/model hash manifest on this page.

| Model | hf pp128 | oll pp128 | hf pp512 | oll pp512 | hf decode | oll decode | decode× |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 0.8B | **10,861** | 4,622 | **12,962** | 7,117 | **353** | 168 | **2.10×** |
| Qwen 3.5 4B | **3,304** | 1,972 | **3,321** | 2,670 | **165** | 93 | **1.78×** |
| Qwen 3.5 9B | **1,920** | 1,428 | 1,919 | **1,970** | **122** | 71 | **1.71×** |

The retired comparison harness remains available in git history; the table is
historical evidence, not a currently runnable route.

## Historical: other arches (decode tok/s)

**Truth state:** historical
Then-default configuration; not a cross-arch speed-gate matrix and not an
admission of gfx12 product routes. No per-row binary/model hash manifest on
this page.

| Arch | Examples | 0.8B | 4B | 9B | 27B |
|---|---|---:|---:|---:|---:|
| RDNA2 (gfx1030) | V620 Pro, RX 6800 XT | 250 | — | 65 | 22 |
| RDNA1 (gfx1010) | RX 5700 XT | 190 | 61 | 43 (HF4) | OOM |
| APU (gfx1013) | BC-250 | 207 | 77 | 47 | OOM |
| GCN5 (gfx906) | MI50 / MI60 | 231 | 61 | 59 | 21 |
| MI300X (gfx942) | datacenter | 850 | 480 | 320 | 130 |

Supporting dated notes (still fixture-bound):

- MI300X is wave64 + MFMA — different kernel family from RDNA WMMA paths.
- RDNA4 (gfx1200 / gfx1201) has gfx12-specific WMMA paths across fused GEMM,
  attention, and MoE kernels; operations without a gfx12 sibling still fall
  through typed dispatch tables. Presence of a kernel path ≠ Redline admission
  and ≠ a row in [`admissions.yml`](admissions.yml).
- gfx906 (Vega 20) prefill batch≥16 used the nwarps=4 dp4a MMQ kernel
  (`docs/plans/gfx906-mmq-prd.md`). Decode batch=1 notes from the 2026-05-05
  investigation
  (`docs/perf-checkpoints/2026-05-05-gfx906-decode-investigation.md`): residual
  GEMV software-pipelined ILP variant (+4.8% on 9B under that fixture) and fused
  projections pre-quantizing x to Q8_1 with `v_dot4_i32_i8` (+9.3% on 9B).
  Combined historical row: 50.7 → 58.9 tok/s (+16.2%) on Qwen 3.5 9B. Stock
  llama.cpp Q4_K_M on the same hardware in that note: 61.55 tok/s; skyne98/iacopPBK
  fork: 63.48.

## Where new numbers go

| Kind of result | Owner |
|---|---|
| How to measure | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) |
| Bench-suite layout | [`methodology/bench-suite.md`](methodology/bench-suite.md) |
| Immutable campaign checkpoints | [`perf-checkpoints/`](perf-checkpoints/) (new dated file; do not rewrite old bodies) |
| Claim → validation route | [`VALIDATION.md`](VALIDATION.md) |
| Product admission | [`admissions.yml`](admissions.yml) only (schema v2; exactly one earned record) |
| Redline-attributed claims | [`REDLINE.md`](REDLINE.md) |
| Speculation capability inventory | [`speculation-support-inventory.md`](speculation-support-inventory.md) (verify in source) |

Do not paste mutable inventory matrices into this page. Do not promote a
historical row by recency alone.
