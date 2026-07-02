# gfx1201 Native Decode Surface — A3B mq4r (Phase-A gating deliverable)

**Purpose.** Synthesize the measured A3B `mq4r` decode kernel surface-map into a
go/no-go gate for **Phase B** (a dp4a / int-dot decode-GEMV port on gfx1201).
Answers three questions: what dominates decode, how close the hot kernels already
run to the DRAM roofline, and what end-to-end tok/s an Amdahl-honest dp4a win can
actually reach.

## Provenance

| Field | Value |
|---|---|
| Model / tag | `qwen3.6:35b-a3b-mq4r` (35B total, ~3B active MoE) |
| Command | `scripts/rocprof-wrap.sh …/2026-07-02-a3b-mq4r-decode-surface-arplain/rocprof -- bun cli/index.ts run qwen3.6:35b-a3b-mq4r --kv-mode q8 --spec off -n 200 "Explain the quicksort algorithm."` |
| Mechanism | `rocprofv3 --kernel-trace --stats -S --output-format csv`, injected into the `bun` parent → captures the spawned `target/release/examples/daemon` child |
| GPU / stack | gfx1201 / Radeon AI PRO R9700 (32 GB), HIP 7.2, `power_dpm_force_performance_level` warm |
| Commit | `cb8c9e26` (worktree `perfmaxx-gfx1201-pkf16-gemv`) |
| Window | 200 generated tokens, **`decode_dominated=true`**, pure AR (`[hipfire] DFlash disabled (dflash_mode=off)`) |
| Coherence | On-topic, fluent quicksort explanation; no attractor / repetition / special-token leak (eyeballed) |
| Roofline denom | `tests/chip-profiles/gfx1201.json`: DRAM peak **611.8 GB/s**, cache tier **1352.4 GB/s** @ 64 MiB Infinity Cache |

Two profiles were captured. The `--spec off` (`arplain`) run above is the clean
AR surface used here. The default-`auto` run is **contaminated** by an
auto-engaged MTP/WMMA path (`gemm_hfq4g256_moe_grouped_wmma_gfx12` at 28.6%,
`attention_dflash_wmma_f32_gfx12` present) and is **not** used — see follow-ups.

**Basis note.** The `Percentage` column is the share of measured **GPU kernel
time**. The full kernel-time budget for the 200-token window is
`183949.656 µs / 0.1309 ≈ 1.4053 s` (cross-checked against three other kernels,
all agree to ±0.05%). That is **142.3 tok/s of pure GPU-kernel time**
(200 / 1.4053 s) — the correct Amdahl baseline. The instrumented wall rate
(23.9 tok/s) is rocprof injection overhead and is discarded; the un-instrumented
AR decode for this tier is ~124 tok/s (fleet memory) — used below as the
conservative wall-anchored baseline.

---

## (a) Ranked per-kernel-family % of decode time

Families are grouped by implementation surface. Listed top kernels cover 97.83%
of decode kernel time; the ~2.2% remainder is the sub-cutoff long tail (incl.
`attention_flash_q8_0_reduce` ≈ 0.40%).

| Rank | Family | % decode time | Members (calls, % each) |
|---:|---|---:|---|
| 1 | **HFQ4G256 4-bit weight-streaming GEMV / projection** *(the hot family)* | **58.38%** | `fused_qkvza_hfq4g256` 13.09 · `gemv_…_moe_gate_up_k8_indexed` 11.43 · `gemv_…_moe_down_k8_indexed_batched_expanded` 9.41 · `gemv_…_residual` 8.12 · `gemv_…_residual_sigmoid_scaled_gpu` 7.28 · `gemv_…_multirow_r2` (lm_head, vocab 248320) 6.29 · `fused_qkv_hfq4g256` 2.76 |
| 2 | Q8 FlashAttention | ~11.95% | `attention_flash_q8_0_tile` 11.55 · `attention_flash_q8_0_reduce` ~0.40 |
| 3 | Fused norm / rotate / RoPE glue | ~15.75% | `fused_rmsnorm_mq_rotate` 6.47 · `fused_silu_mul_mq_rotate` 1.86 · `fused_qk_l2_norm_scale_f32` 1.47 · `softmax_f32` 1.45 · `gated_norm_f32` 1.04 · `mq_rotate_x` 0.82 · `rope_partial_halfsplit_f32` 0.74 · `rmsnorm_f32` 0.71 · `fused_sigmoid_alpha_gate_f32` 0.68 · `repeat_interleave_qk_f32` 0.50 |
| 4 | MoE router / combine glue | ~5.03% | `moe_topk_renorm_k8` 3.48 · `moe_down_combine_k8_batched` 1.55 |
| 5 | DeltaNet recurrent-state update (30 linear-attn layers) | ~4.27% | `gated_delta_net_q8_fast` 3.56 · `conv1d_silu_split_f32` 0.71 |
| 6 | Greedy top-k sampling | ~2.02% | `sample_topk_partial` 1.23 · `sample_topk_finalize` 0.79 |
| 7 | WMMA prefill residue (not decode) | 0.84% | `gemm_hfq4g256_moe_grouped_wmma_gfx12` (80 calls, prefill-scale) |

**Read:** classic bandwidth-bound single-token decode. Many small (10–30 µs)
4-bit weight GEMVs dominate; the only WMMA kernel present (0.84%) is prefill
residue. There is **essentially zero decode tensor-core presence** to accelerate
— the compute surface that matters is the 4-bit GEMV family.

---

## (b) Hot 4-bit decode GEMV — % decode time and roofline utilization

**Hot family true decode share: 58.38%** (52.09% excluding the inferred
`multirow_r2` lm_head attribution). The single largest kernel is only **13.09%**
(`fused_qkvza_hfq4g256`) — the mass is spread across ≥7 GEMV variants, so this is
a **family**, not one kernel.

Roofline denominator = committed measured **611.8 GB/s DRAM** (gfx1201.json,
~96% of the R9700's 256-bit GDDR6@20 Gbps theoretical). The 1352.4 GB/s cache
tier (64 MiB Infinity Cache) is **not exploitable for decode weight streaming**:
at batch=1 each 4-bit weight is read exactly once with no reuse, so the working
set is DRAM-resident and the cache tier only benefits the (small) activation/KV
traffic — the weight GEMVs are bound by the 611.8 GB/s DRAM plateau.

Per-token budget: 1.4053 s / 200 = **7.026 ms/token**; hot family =
0.5838 × 7.026 = **4.102 ms/token**.

Achieved bandwidth (estimated; assumes ~3B active params × ~4.25 bits/param at
`hfq4g256`, lm_head = 248320 × 2048 × 4.25 bits):

| Surface | µs/token | Est. weight bytes/token | Achieved BW | % of 611.8 DRAM peak |
|---|---:|---:|---:|---:|
| **lm_head** (`multirow_r2`, ~442 µs, 1 call/tok) | 442 | ~0.27 GB | **~611 GB/s** | **~100%** (DRAM-saturated) |
| Non-lm_head hot GEMVs | 3660 | ~1.59 GB | ~434 GB/s | ~71% |
| **Hot family (blended)** | 4102 | ~1.86 GB | **~453 GB/s** | **~74%** |

**Interpretation (load-bearing for the verdict):** the hot family already runs
at **~74% of DRAM peak**, and lm_head is **at the roofline (~100%)**. dp4a
accelerates the *int8 dot-product compute*, not DRAM reads. On a kernel already
near the DRAM plateau, dp4a delivers ~nothing (lm_head: **0 headroom** —
**exclude from scope**). The non-lm_head GEMVs have ~1.41× headroom *to* the DRAM
roofline (611.8 / 434) — and dp4a can only capture that gap **if those kernels
are currently ALU/dequant-bound below the plateau**, not already DRAM-bound.
Bounding the whole hot family to the DRAM roofline gives a **max hot-family
speedup of ≈1.35×** (non-lm_head → roofline, lm_head unchanged). That is the
physical ceiling on any dp4a decode-GEMV win here.

> Byte counts are estimated from the A3B active-param headline + 4.25 bits/param,
> not measured per-kernel. The utilization band is therefore ±; treat ~71–74% as
> "clearly bandwidth-bound with modest sub-roofline headroom," not an exact
> figure. Confirming the ALU-vs-mem split per kernel (rocprofv3 `--pmc`
> VALU-busy vs mem-busy + `gfx-kernel-metadata` occupancy) is the cheap gate
> before committing Phase B engineering.

---

## (c) Amdahl table — end-to-end decode tok/s vs hot-GEMV speedup

Amdahl on the hot-family fraction **p = 0.5838**. Total speedup
`f(s) = 1 / ((1−p) + p/s)`. Two baselines shown: **kernel-time 142.3 tok/s** (the
% decompose against this — primary) and **wall-anchored 124 tok/s** (real
un-instrumented AR — conservative).

| Hot-GEMV speedup `s` | Total speedup `f(s)` | tok/s @ 142.3 (kernel) | tok/s @ 124 (wall) |
|---:|---:|---:|---:|
| 1.0× (baseline) | 1.00× | 142.3 | 124.0 |
| 1.35× *(roofline ceiling)* | 1.18× | **167.7** | 146.1 |
| 1.5× | 1.242× | 176.7 | 154.0 |
| 2.0× | 1.412× | 201.0 | 175.1 |
| 3.0× | 1.637× | 233.0 | 203.0 |
| ∞ (hot family → 0 time) | 2.403× | **341.9** | 297.9 |

**Minimum hot-GEMV speedup to hit a target:**

| Target | On kernel basis (142.3) | On wall basis (124) |
|---|---|---|
| ~200 tok/s | s ≈ **1.98×** | s ≈ **2.87×** |
| ~300 tok/s | s ≈ **10.0×** | **unreachable** (∞-asymptote = 297.9 < 300) |

The **hard Amdahl ceiling** (infinite hot speedup) is 2.40× → **341.9 tok/s**
(kernel) / **297.9 tok/s** (wall). Both 200-tok/s targets and (especially) 300
demand hot-family speedups **well above the ~1.35× DRAM-roofline ceiling** from
(b). The physically reachable end-to-end ceiling is the s=1.35× row:
**~168 tok/s (kernel) / ~146 tok/s (wall), i.e. +15–18% end-to-end.**

---

## (d) VERDICT — Phase B (dp4a decode GEMV): **QUALIFIED GO, small win only**

**Go/no-go: conditional GO for family-wide, NO-GO for one-kernel or as a
"solve A3B decode" claim.**

The decode surface is genuinely concentrated in one implementation family — the
HFQ4G256 4-bit weight-streaming GEMVs are **58.38%** of decode kernel time — so a
**family-wide** dp4a port (targeting `moe_gate_up`, `moe_down`, `residual`,
`residual_sigmoid_scaled`, `fused_qkvza`, `fused_qkv`) is the right unit of work.
This matches the Codex verdict (`refuted:false`, conf 0.82): GO **only** if
scoped family-wide; NO-GO for a one-kernel experiment marketed as solving decode
(the largest single kernel is 13.09% → a 2× on it alone is only +7% end-to-end).

**But the realistic ceiling is small, and this must be said plainly.** Two
independent caps agree:

1. **Amdahl:** 58.38% hot mass → even *infinite* hot speedup tops out at
   2.40× (341.9 tok/s kernel / 297.9 wall). 200 tok/s needs ~1.98× (kernel) or
   ~2.87× (wall) on the hot family; 300 tok/s needs ~10× or is outright
   unreachable on the wall basis.
2. **Roofline:** the hot family already runs at **~74% of the 611.8 GB/s DRAM
   peak**, lm_head at **~100%**. dp4a cannot exceed DRAM bandwidth by speeding up
   compute. The roofline-bounded max hot-family speedup is **≈1.35×**, which
   lands *below* even the 1.5× row — so the physically reachable end-to-end gain
   is **~+15–18% (≈168 tok/s kernel / ≈146 wall)**, not 200 and never 300.

**Honest bottom line.** Phase B is **not Amdahl-dead** (58% is a real,
worth-attacking surface) but it **is roofline-limited**: this is a
bandwidth-bound decode, and dp4a's lever is compute. The expected payoff is a
**modest ~15–18% end-to-end** win, contingent on the target GEMVs actually being
ALU/dequant-bound below the DRAM plateau. If a cheap pre-check (rocprofv3
`--pmc` VALU-busy vs mem-busy on 2–3 representative hot GEMVs +
`gfx-kernel-metadata` occupancy) shows them already DRAM-saturated like lm_head,
**Phase B should be rejected** — there is no compute headroom to capture.

**Recommendation:** GO to the **one-day dp4a spike on `fused_qkvza_hfq4g256` +
`moe_gate_up` only**, gated behind the ALU-vs-mem pre-check, targeting the
~1.35× roofline ceiling and validated on `./scripts/coherence-gate.sh`. Do
**not** greenlight full family engineering, and do **not** attach any 200/300
tok/s target — the roofline forbids it. Exclude lm_head (`multirow_r2`) from dp4a
scope entirely (already at DRAM peak).

---

## Follow-ups (out of scope here)

1. **MTP auto-engages on A3B `mq4r` despite `--spec` help text.** The default
   (`auto`) run showed a speculative WMMA/MTP path even though the CLI help says
   "A3B → off" (registry `cli/registry.json:171-179` ships an `.mtp` sidecar for
   this tag). File as a CLI-help / spec-auto-default discrepancy.
2. **lm_head attribution is inferred** (`multirow_r2`, 6.29%, ~442 µs, 201 calls,
   vocab 248320) from call count + shape, not a name-labeled log line —
   high-confidence but not ground-truth.
3. **DPM read caveat:** `power_dpm_force_performance_level` returned
   "Device or resource busy" on read (pre-existing box quirk); `auto` was written
   back (exit 0) but unverifiable by read.

---

## Pre-check resolution (ALU-vs-mem) — 2026-07-02

The `(d)` verdict above gated the dp4a spike behind a cheap ALU-vs-mem
pre-check. That pre-check ran as three parallel probes (gfx1100 rocprofv3 PMC
split, no-GPU static occupancy, precise per-kernel byte-model BW) adjudicated by
Codex. **Resolved bound-class: `mixed` — and the adjudicated decision is NO-GO
for a broad Phase-B dp4a build.** Details below.

### 1. gfx1100 PMC ALU-vs-mem split — **could not be obtained** (tool-limited, not methodology)

The direct VALU-busy-vs-mem-busy split failed on hipx's gfx1100 (7900 XTX) —
not from a methodology error (every step through kernel dispatch worked) but
from two stacked ROCm 7.2.2 / rocprofv3 limitations found live:

- **Daemon path OOM'd (anticipated).** `qwen3.6-35b-a3b.mq4r` (18.7 GB) + q8 KV
  OOM'd on the 25.8 GB-reported card. The MQ4 "MMQ safety screening" flagged
  68/190 weight matrices UNSAFE and **duplicated them into a WMMA-precision
  fallback copy in addition to the hfq4g256 copy**, pushing effective footprint
  over budget — not simply "18.7 GB doesn't fit in 24 GB." Fell back to a
  `bench_gemv_pmc_a3b` microbench driving `gemv_hfq4g256` at the two real A3B
  decode shapes read off the OOM log (gate_up-proxy M=8192,K=2048;
  down-proxy M=2048,K=4096).
- **`VALUBusy` does not exist for gfx1100 in this metrics DB.** In this
  multi-arch box, `rocprofv3 --list-avail` registers `VALUBusy` **only** under
  the gfx1201 block — contrary to the framing that gfx1100 counters work
  generically. `MemUnitBusy`, `TA_BUSY_avr`, `SQ_INSTS_VALU`, `GRBM_GUI_ACTIVE`,
  `SQ_BUSY_CYCLES`, `Wavefronts` are present.
- **Every %-busy / occupancy-style counter read exactly 0.0** across all 640
  sampled dispatches per counter (`MemUnitBusy`, `TA_BUSY_avr`, `GRBM_GUI_ACTIVE`,
  `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_ANY`, `SQ_INSTS_VALU` all zero), while the two
  simplest raw accumulators (`SQ_BUSY_CYCLES` ~0.9–1.0 M cycles/dispatch,
  `Wavefronts` = M exactly) read sane values in the **same** runs — proving PMC
  collection was functioning, not silently no-op'ing. This is a genuine ROCm
  7.2.2 dispatch-scoped-counter capture floor on very short (~9–10 µs),
  low-occupancy (single wave32/workgroup) dispatches on gfx1100 — distinct from
  the known "gfx12 PMC reads zero" issue.
- **Anti-hang note (reconfirmed):** requesting >2 counters spanning different HW
  blocks in one `--pmc` pass SIGABRTs rocprofv3 (`error code 38`) but the wrapped
  child spins at 150–200% CPU forever and must be manually killed. Valid combos:
  ≤2 counters, or counters within one HW block.

**Net:** no usable VALU-busy% vs mem-busy% signal on either shape. The direct
ALU-vs-mem confirmation this doc asked for **never materialized** — it provides
neither positive ALU-bound evidence nor a DRAM-saturation refutation. Raw CSVs
left at `/tmp/pmc-a3b-gemv/*.csv` on hipx; GPU perf level restored to `auto`.

### 2. Occupancy verdict — **100% occupancy, wave-slot-cap bound, NOT VGPR/SGPR/LDS/spill limited**

No-GPU static extraction (unbundled `.hsaco` → `llvm-readelf --notes`), hardware
constants from `tests/chip-profiles/gfx1201.json` (`vgprs_per_simd=1536`,
`max_waves_per_simd=16`, wave32). Two kernels from committed gfx1201 fixtures;
`fused_qkvza` + `gemv_hfq4g256_residual` compiled fresh via the runtime JIT
invocation (dispatch traced to confirm gfx1201 falls through to the generic
kernel bodies, not the RDNA3 `_gfx1100` variant).

| Kernel | VGPR | SGPR | LDS | Scratch | VGPR-limited waves/SIMD | Waves/SIMD (capped@16) | Occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fused_qkvza_hfq4g256` | 72 | 22 | 0 | 0 | 21 | 16 | **100%** |
| `gemv_hfq4g256_moe_gate_up_k8_indexed_batched` | 80 | 26 | 0 | 0 | 19 | 16 | **100%** |
| `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` | 80 | 22 | 0 | 0 | 19 | 16 | **100%** |
| `gemv_hfq4g256_residual` | 72 | 22 | 0 | 0 | 21 | 16 | **100%** |

**Zero spills across all four** (`private_segment_fixed_size=0`). The limiter is
the architectural 16-waves/SIMD wave-slot cap, **not** register/LDS pressure: the
VGPR-only ceiling (19–21 waves/SIMD) already exceeds the hardware max of 16, so
latency-hiding is saturated and shrinking VGPR buys **zero** additional
occupancy. This **falsifies the "occupancy-limited" hypothesis** for the sub-
roofline gap — but it cannot distinguish (1) achieved-vs-theoretical DRAM
ceiling from (2) VALU/dequant issue-rate bound; only the (failed) PMC split
could, and dp4a helps only in case (2). (Skill-doc nit surfaced: the
`gfx-kernel-metadata` cheat-sheet's "128 KB LDS for gfx1200/1201" is the
per-WGP figure shared by 2 CUs; 65536 is the correct per-CU value — immaterial
here since all four kernels use 0 LDS.)

### 3. Precise per-kernel achieved-BW table (real shapes, real dispatch counts)

Byte model rebuilt from the **live `.mq4r` checkpoint header** (not the "~3B
active" headline): `hidden=2048`, 40 layers (30 linear-attn + 10 full-attn),
256 experts / 8 per-tok, `moe_intermediate=512`, all HFQ4G256. gfx1201 DRAM peak
= 611.8 GB/s.

| Kernel | bytes/call | µs/call | achieved GB/s | % of 611.8 peak | read |
|---|---:|---:|---:|---:|---|
| `fused_qkvza_hfq4g256` (blended, 2 call-sites) | 6,555,977 | 13.139 | **499.0** | **81.6%** | near-roofline; mixed bucket |
| `gemv_hfq4g256_moe_gate_up_k8_indexed` | 8,912,896 | 20.087 | **443.7** | **72.5%** | headroom + good grid → **best dp4a candidate** |
| `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` | 4,456,448 | 16.529 | **269.6** | **44.1%** | small-transaction (272 B/block) → **granularity, not ALU** |
| `gemv_hfq4g256_residual` (wo/out_proj) | 4,456,448 | 14.263 | **312.4** | **51.1%** | small grid (2048 blocks) → **occupancy/batching, not dp4a** |
| `gemv_hfq4g256_multirow_r2` (lm_head) | 270,172,160 | 441.957 | **611.3–613.6** | **99.9–100.3%** | DRAM-saturated → dp4a dead, EXCLUDE |

Key correctness check: the `fused_qkvza` 14000-call bucket is **two structurally
different call-sites** under one rocprof name — attention QKVZA (6000 calls,
13.4 MB/call) + MoE gate-fusion router/shared-expert (8000 calls, 1.4 MB/call);
6000+8000=14000 exact. Treating all 14000 as the attention shape yields
**1023 GB/s = 167% of DRAM peak (physically impossible)**; the correctly-blended
total gives a sane **499.0 GB/s**. This refines the doc's flat "~74% blended"
into a **heterogeneous family**: only `moe_gate_up` looks genuinely
ALU-capturable; `moe_down`/`residual` have larger numeric gaps but for
transaction-granularity / grid-occupancy reasons dp4a likely won't fix; lm_head
is at peak.

### 4. Codex adjudication

Codex adjudicated the three probes: **`bound_class = "mixed"`, `dp4a_go = false`,
confidence 0.73.**

> Probe 1 (PMC) is non-decisive — the split failed, so no positive ALU-bound
> evidence. Probe 2 (occupancy) is useful **negative** evidence — not
> VGPR/SGPR/LDS/spill or static-occupancy limited, so dp4a will not unlock more
> residency. Probe 3 (byte model) carries the most weight — a **heterogeneous**
> family, not one broad ALU-bound bucket: lm_head fully DRAM-saturated
> (excluded); `fused_qkvza` already ~82% of peak and a mixed call-site bucket;
> `moe_gate_up` at ~72.5% the only credible dp4a candidate; `moe_down`/`residual`
> look like transaction-granularity / launch / latency inefficiency, not compute
> issue. The Probe-2-vs-3 "conflict" is terminology: static occupancy is full,
> yet short/small-memory-shape latency and transaction efficiency can still cap
> achieved BW. Since **gfx1201 has less bandwidth than gfx1100 (612 vs 809
> GB/s), any DRAM-ish result transfers *against* dp4a**, and the direct gfx1100
> ALU-vs-mem confirmation never materialized. Net: mixed bound-class, but not
> enough ALU-bound surface to justify Phase-B dp4a as a build priority.

### FINAL Phase-B decision: **NO-GO** (broad dp4a decode-GEMV rejected)

**Resolved bound-class: `mixed`, but DRAM/latency/transaction-dominated, not
ALU-bound.** This supersedes the `(d)` "QUALIFIED GO." **Phase-B dp4a is a
NO-GO as a build priority.**

**Why.** The one thing that could have justified the spike — direct evidence
that the hot GEMVs are VALU/dequant-issue-bound *below* the DRAM plateau — never
materialized (Probe 1 failed). Everything that *did* resolve points the other
way: occupancy is already 100% (dp4a unlocks no residency), and the precise byte
model shows the family is heterogeneous with only **one** kernel (`moe_gate_up`,
72.5%) plausibly ALU-capturable — the rest are DRAM-saturated (lm_head ~100%,
`fused_qkvza` ~82%) or bounded by transaction granularity / small grids
(`moe_down` 44%, `residual` 51%) that dp4a cannot address. Critically, **gfx1201
has *less* DRAM bandwidth than the gfx1100 we probed (612 vs 809 GB/s), so any
memory-bound result transfers against dp4a, making the target arch *more*
mem-bound, not less.** The +15–18% Amdahl ceiling from `(d)` required *most* of
the hot family to be ALU-capturable; these probes do not support that
premise.

**Realistic ceiling if pursued anyway:** ~**0–3% end-to-end** if only
`moe_gate_up` benefits; perhaps **~5–6%** in an optimistic narrow spike that also
captures part of `fused_qkvza`'s attention sub-component. The `(d)` +15–18%
figure is **not reachable** and should not be quoted.

**Recommended alternative levers** (better ROI than a decode-GEMV port):

1. **Wire A3B MTP into the daemon** — the single highest-value lever. Per the
   ROCmFP4 gfx1201 H2H (`project_rocmfp4_h2h_gfx1201_2026_07_01`), on R9700 the
   competitor leads hipfire on **raw AR** decode (131–147 vs our mq4r ~124
   tok/s); hipfire only wins via **MTP (152.7 tok/s, τ=3.02 coherent)**, which is
   currently **`mtp_only_demo`-only and NOT daemon-wired for A3B** (ships
   AR-only). Closing that gap is a multiplicative decode win the dp4a spike's
   best case (+5–6%) cannot approach, and it targets the axis where we actually
   trail on this arch.
2. **Attack the non-GEMV surface the map already exposed** — the decode surface
   is not only the 58% GEMV family: **norm/glue ≈ 16%** and **attention ≈ 12%**
   are unattacked and are not roofline-pinned the way the weight-stream GEMVs
   are. Fusion / launch-count reduction there sidesteps the DRAM wall entirely.
3. **If dp4a is ever revisited, scope it to `moe_gate_up` ONLY** (not the `(d)`
   `fused_qkvza + moe_gate_up` pairing), and only after obtaining the real
   VALU-vs-mem split — via the Probe-1 follow-ups (inflate per-dispatch work so
   GRBM/SQ/TA counters latch above the ROCm 7.2.2 sampling floor; try
   aggregate/PC-sampling mode; or an older rocprof with working dispatch-scoped
   gfx1100 capture). Do not build on the assumption of ALU-headroom that the
   pre-check failed to establish.

---

## moe_down re-tile check + Phase-B final decision — 2026-07-02

**Framing.** The `FINAL Phase-B decision: NO-GO` above rejected a *broad dp4a
decode-GEMV port*. This section resolves a distinct, narrower question raised by
Probe-3's byte model: the two sub-roofline outliers `moe_down` (44.1% of peak)
and `residual` (51.1%) — are they capturable by a **source-supported multirow
re-tile** (a block/grid remap, **not** dp4a, **not** LDS tiling)? Two probes (a
kernel-source audit + a gfx1100 inflated-dispatch PMC run) were adjudicated by
Codex.

### 1. Kernel-source findings (moe_down / residual re-tile feasibility)

- **Current tiling (both kernels)** — `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded`
  (`kernels/src/…moe_down….hip`, grid `[M,K_TOP,batch]`) and
  `gemv_hfq4g256_residual` (grid `[M,1,1]`): block `[32,1,1]`, **one output row
  per 32-thread wave**, pure register-accumulator GEMV, **0 LDS**, one 4-byte
  (dword) scalar load per thread (already wave-coalesced *within* a row; no
  `dwordx4`/`uint4`). No arch branch — the same kernel runs on every arch incl.
  gfx1201.
- **272 B/block is model-shape-forced, not a coalescing bug.** A3B
  `moe_intermediate=512` → `K=512` → `groups_per_row = K/256 = 2` → the kernel
  runs its **2-group TAIL-ONLY path** (never enters the 4-group "quads" main
  loop); `2 × 136 B = 272 B` is simply how little weight one MoE-down row *is* at
  this expert width.
- **Occupancy confirmed** (no-GPU static extract): moe_down 80 VGPR / 22 SGPR /
  0 LDS / 0 spill; residual 72 VGPR / 22 SGPR / 0 LDS / 0 spill — both already at
  the **100% hardware 16-wave/SIMD cap**, not register-limited. Because both are
  0-LDS register GEMVs, the only source-supported lever is a **VGPR-resident
  multirow** (fold `R` adjacent rows into one wave, hoist the x-vector loads once,
  per-block A traffic → `R×272 B` contiguous) — **not** LDS tiling.
- **Mechanism already exists in-tree:** `gemv_hfq4g256_multirow_r{2,4,8}`
  (`kernels/src/gemv_hfq4g256_multirow.hip`, `.gfx1100.hip`) + residual sibling,
  numerics-preserving by construction (same `DOG`/`TAIL_DOG` macros, same
  accumulator order, the tail-must-stay-in-`acc[r][g%4]` discipline from commit
  `5302926`). **Coherence risk: low** — a pure block/grid remap; the dequant
  math, the indexed-expert gather (`topk_indices[…]` / `expert_ptrs[expert_id]`),
  and the K8-expanded output layout are all untouched.
- **Real risk flagged:** both kernels sit only ~16 VGPR under the 96-VGPR /
  16-wave threshold (80 / 72 used vs 96 ceiling). Multirow's extra
  accumulators/row-pointers could push VGPR over that line and *cost* occupancy
  even as it grows the per-block transaction — this needs a `gfx-kernel-metadata`
  VGPR check on the compiled R=2 variant **before** on-device trust, not an
  accept-on-faith port.
- **Prior-attempt scan:** `remotes/hiptrx/feat/gfx1201-kernel-tuning` (`ea69403b`,
  never merged; not an ancestor of master) already ported
  `multirow_r{2,4,8}.gfx1201` + the residual sister and returned a **NULL
  verdict** (`39a670f4`, real R9700): R=1/2/4 within 0.1% on 9B/27B **dense AR +
  DFlash**, "BW-saturated ~500 GiB/s." **That null does NOT falsify the current
  case** — it tested near-saturated dense-model large-K rows and (critically)
  **never touched `moe_down` at all** (which has zero multirow variant on any
  arch), and the residual multirow path is hard-gated to `is_rdna3_dgpu()` so
  gfx1201 never reaches it even with `HIPFIRE_GEMV_ROWS` set. The one regime
  where multirow *did* win (gfx1010, +2.7% at R=2, 48.5% of peak) is the
  low-utilization analogue to where moe_down/residual sit now.

### 2. gfx1100 inflated-dispatch PMC — **ROCm-limited, not a latch floor**

Following `(d)`-section recommendation 3 ("inflate per-dispatch work so counters
latch above the ROCm 7.2.2 floor"), a long-dispatch microbench
(`bench_gemv_moe_down_pmc_loop`, `ITERS=4000`, 32-block × 4.46 MB weight rotation
= 136 MB to defeat L2/Infinity-Cache residency, moe_down proxy M=2048/K=4096) ran
**24–25 ms per dispatch** (host wall + rocprof `Start/End_Timestamp` agree, e.g.
24.97 ms) — **~2500× longer** than the ~9–10 µs floor blamed for the prior
zero-reads. **The short-dispatch-floor hypothesis is FALSIFIED:**

| Counter | Value on the 24–25 ms dispatch |
|---|---|
| `MemUnitBusy` | **0.0** |
| `GPUBusy` | **0.0** |
| `WriteUnitStalled` (nearest `MemUnitStalled` analog available) | **0.0** |
| `SQ_BUSY_CYCLES` (control) | 2,900,237,047 — scales sanely vs 43.6 M @ 300 µs warmup |
| `Wavefronts` (control) | 2048 — exactly == grid M |

All %-busy metrics still read **exactly zero** on the 2500×-longer dispatch,
while the raw accumulators scaled correctly — **PMC collection works; the derived
%-busy-metric computation is broken for gfx1100 in ROCm 7.2.2**, independent of
dispatch duration (the duration variable is now eliminated). New failure mode:
`SQ_INSTS_VALU` alone **SIGSEGVs rocprofv3** (exit 139), worse than the prior
SIGABRT-hang, so no VALU-side signal is obtainable at all.

**Verdict: inconclusive by tool limitation — the ALU-vs-mem split is unobtainable
on this box/ROCm; per the task's fallback rule, the source analysis is the best
available evidence.** This is the **second** independent PMC attempt (short + now
long dispatch) to fail on the same limitation → further rocprofv3 PMC on this
box/ROCm is unproductive without a ROCm upgrade or a different tool
(rocprof-compute / Omniperf). CSVs left at `hipx:/tmp/pmc-moe-down-longdispatch/`
(ephemeral); GPU lock cleanly released, DPM untouched (`--setperflevel` lacked
permission, ran at `auto`).

### 3. Codex decision

> `lever_real=true, recommend_attempt=true, coherence_risk="low".` Attempt a
> **bounded R=2 spike, only for the unfalsified shape**:
> `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` on A3B `mq4r` / gfx1201.
> The lever is real — K=512 forces 272 B/row, the wave already coalesces within a
> row, and multirow is the one source-supported way to enlarge per-block
> contiguous A traffic while reusing x without changing dequant math. The prior
> gfx1201 multirow null does not apply (it tested dense AR/DFlash already near
> saturation, not the small-K MoE-down kernel at 44.1% of peak). **Do not** spend
> time on dp4a, LDS tiling, or wider per-thread vector loads — the source rules
> those out. **Residual is weaker** (the diagnosed problem is a small grid, and
> rows-per-block makes the grid *smaller*) — include it only as a cheap follow-on
> if moe_down R=2 metadata stays under the VGPR/occupancy threshold. **Hard
> stops:** R=2 VGPR > 96, spills, no clear per-kernel BW/timing lift on the A3B
> fixture, or any coherence/channel anomaly. Worth a **half-day experiment, not a
> multi-day pivot.**
> **Expected gain:** best case ~**+3–4% end-to-end** from moe_down if its 44%
> peak-BW rises toward ~75–80%; residual likely **0–2%** and possibly
> neutral/negative (multirow shrinks an already-small grid). **Combined realistic
> expectation ~1–4%, not a new perf tier.**

### FINAL CALL: **GO — bounded moe_down R=2 re-tile spike (half-day, hard-stopped)**

**Decision: GO**, narrowly scoped. This does **not** reopen the broad-dp4a NO-GO
above (that stands) — it is a **different lever** (a block/grid multirow remap of
one kernel, zero dequant-math change) against the single unfalsified sub-roofline
shape, `moe_down`.

**Execute exactly this, in order:**

1. Port a `gemv_hfq4g256_multirow_r2` variant of
   `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded` (which currently has **no**
   multirow variant on any arch), preserving the `DOG`/`TAIL_DOG` tail-in-
   `acc[r][g%4]` discipline verbatim.
2. `gfx-kernel-metadata` VGPR check on the compiled R=2 `.hsaco` — **abort if
   VGPR > 96 or any spill** (the ~16-VGPR headroom is the binding risk).
3. Bench on the **actual A3B `.mq4r` K=512 moe_down shape** (not the dense-27B
   null), per-kernel BW/timing on gfx1201/R9700.
4. `./scripts/coherence-gate.sh` before any claim.
5. Optional cheap follow-on: admit gfx1201 into the `is_rdna3_dgpu()`-gated
   residual multirow path — **only** if step 2 passes and moe_down showed a lift.

**Expected payoff: ~+1–4% end-to-end (best case ~+3–4% from moe_down alone);
success is not guaranteed given the narrow occupancy headroom.** This is a
half-day spike, not a perf tier — the Amdahl/roofline ceiling on the whole
decode-GEMV surface remains as bounded as `(d)` and the NO-GO established.

**Strategic note (the re-tile does not change this).** Even in its best case this
is a single-digit-% decode tweak. The **real competitive lever on gfx1201 is
wiring A3B MTP into the daemon.** Per the ROCmFP4 H2H
(`project_rocmfp4_h2h_gfx1201_2026_07_01`), on R9700 the competitor's llama.cpp
fork **leads hipfire on raw AR** (131–147 vs our `mq4r` ~124 tok/s) and hipfire
wins **only via MTP** (152.7 tok/s, τ=3.02 coherent) — which is currently
**`mtp_only_demo`-only and NOT daemon-wired for A3B** (the daemon ships AR-only
for this tag). That is a *multiplicative* decode win the re-tile cannot approach,
and it targets the axis where we actually trail on this arch. But it is an
**engineering task needing user greenlight** — and carries the **A3B R̄≈0.39
acceptance caveat** (`feedback_a3b_r_not_acceptable`: default A3B = AR-only + no
eviction until R̄ improves) — **NOT another kernel gamble.** Prioritize the MTP
wiring conversation over, or in parallel with, the half-day re-tile spike.
