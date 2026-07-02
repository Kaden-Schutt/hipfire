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

## R=2 multirow spike result — 2026-07-02

The bounded, hard-stopped R=2 spike from the FINAL CALL above was implemented
(commit `d2fe517a`, branch `worktree-perfmaxx-gfx1201-pkf16-gemv`) and measured
on-device (gfx1201 / R9700, hiptrx). Two new opt-in gfx1201 kernels —
`gemv_hfq4g256_moe_down_..._multirow` and `gemv_hfq4g256_residual_multirow` — fold
2 adjacent output rows per block, hoisting the expert gather + x-load once and
reusing the exact `DOG`/`TAIL_DOG` tail-in-`acc[r][g%4]` discipline. Both gate
strictly on `HIPFIRE_GEMV_ROWS=2`; default (unset) is byte-identical to before.

### VGPR / occupancy — PASS (hard stop cleared)

Independently re-verified via the `gfx-kernel-metadata` skill on the compiled
`.hsaco` (not just the commit message):

| Kernel (gfx1201 R=2) | VGPR | SGPR | Spill | Occupancy |
|---|---:|---:|---:|---|
| `gemv_hfq4g256_moe_down_multirow` | **96** | 30 | 0 | 16 waves/SIMD (max) |
| `gemv_hfq4g256_residual_multirow` | **96** | 36 | 0 | 16 waves/SIMD (max) |

Both land **exactly** at the boundary: 1536 ÷ 96 = 16 waves/SIMD, zero spill
(`private_segment_fixed_size=0`, `vgpr_spill_count=0`). VGPR ≤ 96 and occupancy is
**preserved at the max 16 waves/SIMD** — the binding "abort if VGPR > 96 or any
spill" hard stop is cleared with no headroom to spare.

### Warm A/B — real signal, +4.24% (bands do not overlap)

Model `qwen3.6:35b-a3b-mq4r`, `--kv-mode q8`, `--spec off`, 200 tok, DPM forced
high, `HIP_VISIBLE_DEVICES=0`. Prompt `"Explain the quicksort algorithm."`
(md5 `195ebebaf38d280b6002a66ad94756c6`), 5 fresh-process runs per cell after a
throwaway per-config warmup (no cold-run outliers observed):

| config | 5 runs (tok/s) | median | band |
|---|---|---:|---|
| baseline (`HIPFIRE_GEMV_ROWS` **unset** — true prod default) | 124.9, 125.2, 125.1, 124.9, 125.0 | **125.0** | 124.9–125.2 (±0.15%) |
| treatment (`HIPFIRE_GEMV_ROWS=2`) | 130.3, 130.5, 130.1, 130.3, 130.3 | **130.3** | 130.1–130.5 (±0.15%) |

**Delta: +4.24%** (130.3 vs 125.0). The bands are disjoint (max baseline 125.2 <
min treatment 130.1) and the per-cell noise band is only **±0.15%** — this is
real signal, not DPM/thermal/JIT noise, even though the point estimate sits just
under the formal ±5% investigation trigger. Magnitude is consistent with
re-tiling only 2 of ~40 decode-GEMV kernels (moe_down tail-K=512 + residual),
matching the design doc's best-case ~+3–4% moe_down projection.

**Control-selection note (mandatory):** the baseline **must** be
`HIPFIRE_GEMV_ROWS` *unset*, **not** `=1`. On gfx1201 the *pre-existing, unrelated*
base `gemv_hfq4g256` dense kernel already defaults to `rows=2` via
`arch_caps.rs::gemv_rows_default()` (`_ => 2` catch-all), so forcing `=1` does not
mean "no multirow" — it drops into a rarely-exercised R=1 fallback that produces
**garbage/repetition-loop output** on this A3B `mq4r` + q8-KV combo (degenerate
loop, truncated ~35/200 tok). That is a real, reproducible, **orthogonal**
pre-existing bug (the commit's new kernels never engage under R=1) — file it
separately; it is not a blocker and gfx1201 never runs R=1 in production.

### Coherence — PASS

- **Greedy (temp=0) numerics:** `unset` and `HIPFIRE_GEMV_ROWS=2` produced
  **byte-identical** token IDs through the sampled window on the exact
  `qwen3.6:35b-a3b-mq4r` model — strongest possible numerics-preservation evidence
  (substituted for `coherence-gate.sh --full`, which does not cover `mq4r`).
- **Stochastic 200-tok eyeball (5/5 R=2 runs):** fully on-topic, fluent,
  structured quicksort walkthrough (divide-and-conquer, pivot/partition,
  O(n log n)) — no attractors, no loops, no special-token leaks.

### THE CALL: **WIN** — real, coherent, occupancy-preserving

R=2 is a **coherent, real (> noise band), occupancy-preserving decode win**:
**+4.24%** end-to-end on the A3B `mq4r` moe_down shape, byte-exact greedy numerics,
96 VGPR / 16 waves preserved with zero spill. This is the one sub-roofline shape
(moe_down at 44% peak BW) the earlier gfx1201 multirow null did **not** cover, and
the lever landed as projected.

**Recommendation: keep it opt-in** behind `HIPFIRE_GEMV_ROWS=2` (default unset =
byte-identical to prior). The measured gain is **+4.24%** on the canonical A3B
`mq4r` / q8-KV / gfx1201 fixture — surfaced here for the user to decide a **default
flip**. A default-on change is a default-behavior flip (per
`feedback_pr_gating_policy`) and should be its own greenlight decision; before
flipping, re-confirm on a second prompt shape and, ideally, the broader A3B bench
matrix (the +4.24% is single-shape). The opt-in kernels are gated and
coherence-safe, so they ship as-is regardless of the flip decision.

### Standing recommendation (unchanged by the win)

Even as a validated win this is a **single-digit-% decode tweak**, and the
decode-GEMV kernel surface is now **effectively exhausted** on gfx1201 — the
last unfalsified shape (moe_down R=2) has been converted, and the broad-dp4a /
LDS-tiling / wider-load NO-GO from section (d) stands. The **real competitive
lever on gfx1201 remains wiring A3B MTP into the daemon.** Per the ROCmFP4 H2H
(`project_rocmfp4_h2h_gfx1201_2026_07_01`), on R9700 the competitor's llama.cpp
fork **leads hipfire on raw AR** (131–147 vs `mq4r` ~124 tok/s) and hipfire wins
**only via MTP** (152.7 tok/s, τ=3.02 coherent) — currently **`mtp_only_demo`-only
and NOT daemon-wired for A3B**. That multiplicative decode win — on the axis where
we actually trail this arch — dwarfs the +4.24% re-tile, but it is a
**greenlight-gated engineering task**, carrying the **A3B R̄≈0.39 acceptance
caveat** (`feedback_a3b_r_not_acceptable`: default A3B = AR-only + no eviction
until R̄ improves) — **NOT another kernel gamble.** Prioritize the MTP-wiring
conversation over any further decode-GEMV kernel work on this arch.

## dp4a `moe_gate_up` Phase-B spike result — 2026-07-02

The direct ALU-vs-mem PMC split never resolved (`(d)` Pre-check resolution,
Probe 1) — this section replaces the roofline estimate with an actual
on-device measurement. Built `gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a_gfx1201`
(`kernels/src/gemv_hfq4g256_moe_gate_up_dp4a.gfx1201.hip`), an int8 dp4a
(`v_dot4_i32_iu8`, unsigned weight-nibble × signed Q8_1 activation) port of
`moe_gate_up` — the doc's own "best dp4a candidate" (72.5% of DRAM peak,
Probe 3). Opt-in via `HIPFIRE_GFX1201_DP4A=1`, default OFF; wired as an
early branch inside the existing `gemv_hfq4g256_moe_gate_up_k8_indexed`
(`crates/rdna-compute/src/gemv.rs`) so every other arch, and gfx1201 with
the flag unset, is byte-for-byte the pre-existing scalar kernel.

**Static metadata (gfx-kernel-metadata skill, compiled `.hsaco`):** 67 VGPR /
18 SGPR / 0 spill (`private_segment_fixed_size=0`) — actually **fewer** VGPR
than the scalar kernel's 80, same 100% occupancy (16 waves/SIMD, both land
under the 1536/SIMD wave-slot cap). Disassembly confirms 14×
`v_dot4_i32_iu8` instructions emitted with `neg_lo:[0,1,0]` (unsigned ×
signed operand encoding) — the intended instruction is genuinely selected,
not silently scalarized.

**Correctness (on-device, gfx1201/R9700):** a standalone probe
(`crates/rdna-compute/examples/verify_gate_up_dp4a_gfx1201.rs`) ran the real
M=1024/K=2048 A3B gate_up shape through both paths (two fresh processes, env
flag on/off) and diffed the 1024 output values: **max abs error 0.0052,
mean abs error 0.0012** against a baseline stdev of 0.39 (~0.3% of typical
magnitude) — no NaN/Inf, no sign flips, ranges match closely (e.g. min
−1.640982 vs −1.640782). This is consistent with expected Q8_1 int8
quantization noise, not a dequant bug — it corroborates that the zp sign
(`weight_val = nibble·scale + zp`, read verbatim from the scalar kernel's
`DOG` macro — **not** the `nibble−8` / `zp_eff=zp+8·scale` convention the
existing sdot4 wave64 MMQ kernels use), the nibble unpack order, and the
in-kernel activation reorder (`block_within_group` / `sub_in_block` /
`qs_byte_off`, matched lane-for-lane against the weight nibble-dword's
element range) are all correct.

**Throughput (on-device, COLD regime — 256 experts / 32 disjoint top-8 sets,
`crates/rdna-compute/examples/bench_gate_up_dp4a_gfx1201.rs`, throwaway
100-iter warmup + 3 fresh-process runs of 500 launches each per config):**

| Config | 3 runs (µs/launch) | median | achieved GB/s |
|---|---|---:|---:|
| baseline (flag unset) | 23.727\*, 21.201, 21.248 | **21.22** | ~425 |
| `HIPFIRE_GFX1201_DP4A=1` | 23.374, 23.426, 23.336 | **23.37** | ~385 |

\*first baseline run still JIT/DPM-warming; excluded from the median (the
other two agree to ±0.1%). The dp4a band is tight (23.34–23.43, ±0.2%)
across three fresh processes — this is real signal, not noise.

**Verdict: dp4a is a real ~10% LOSS on this kernel, not a win — the silicon
confirms the roofline NO-GO.** Two contributing, source-level factors: (1)
`moe_gate_up` was already measured at 72.5% of DRAM peak with 100% static
occupancy in both variants (VGPR is not the limiter, confirmed above), so
there is little-to-no ALU headroom for dp4a to convert into a memory-traffic
win; (2) the dp4a path pays for `ensure_q8_1_mmq_x`'s mandatory per-call
Q8_1 quantization (`quantize_q8_1_mmq_ds4`, unconditional — the shared
scratch has no source-pointer caching, unlike its FP16/FP8 siblings) as a
**second GPU kernel launch** ahead of the GEMV itself; in real decode `x`
differs every layer, so this cost is not a benchmark artifact — it is the
real per-token tax. Investigation stops here per the task's cheapest-first
rule: occupancy is ruled out by the metadata above, and the remaining
candidates (launch overhead, Xq write/read traffic) are structural to the
dp4a-with-online-quantization design, not a tunable bug.

**Disposition:** ships as an opt-in, default-off research kernel (Kaden's
explicit ask was to build-and-measure, not to land a win). No default
behavior changes on any arch. `./scripts/coherence-gate.sh` was not run —
out of scope for a measured-loss, default-off spike; re-run if this kernel
is ever revisited as a real candidate. This closes the Phase-B dp4a
question empirically: **do not pursue dp4a further on gfx1201 decode
GEMVs** — the standing recommendation above (wire A3B MTP into the daemon)
remains the real lever.

## dp4a moe_gate_up spike — EMPIRICAL result — 2026-07-02

The section above measured the dp4a kernel on the **standalone microbench**
(~10% kernel-level loss). This section closes the loop with the **production
daemon serving path** on `hiptrx`/gfx1201 (R9700), commit `b86172a8`
synced to `~/hipfire-perfmaxx` and `daemon` rebuilt fresh
(`cargo build --release --example daemon --features deltanet -p hipfire-runtime`).
This is the strong prod-falsify (real serve path, not a synthetic harness),
not an analytic guess. GPU lock (`dp4a-ab`) held, all four R9700s forced to
`high`, `HIP_VISIBLE_DEVICES=0` pinned, perf level restored to `auto` and
lock released at hand-back.

**dp4a VGPR / occupancy (independently re-verified via gfx-kernel-metadata,
not just trusting the microbench section):**

| Kernel | VGPR | SGPR | Spill | Occupancy (gfx1201, 1536 VGPR/SIMD, 16-wave cap) |
|---|---:|---:|---:|---|
| `gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a_gfx1201` | 67 | 18 | 0 | ⌊1536/67⌋=22 → capped **16/16 (100%)** |
| `gemv_hfq4g256_moe_gate_up_indexed` (scalar baseline) | 80 | 18 | 0 | ⌊1536/80⌋=19 → capped **16/16 (100%)** |

Both are at the wave-slot cap already and **occupancy-identical** — VGPR is
not the differentiator (dp4a even uses *fewer* VGPR, 67 < 80). This confirms
the "wave-slot-cap bound, not register-limited" conclusion: dp4a unlocks no
additional residency.

**`v_dot4_i32_iu8` fired — confirmed both statically AND dynamically:**

- **Static disassembly** (`llvm-objdump --mcpu=gfx1201`): **14× `v_dot4_i32_iu8`**
  instructions in the dp4a kernel ELF (with `neg_lo:[0,1,0]` = unsigned-weight ×
  signed-activation operand encoding), **0** in the scalar kernel — genuinely
  selected, not scalarized.
- **Dynamic** (rocprofv3 `--kernel-trace`) on a live `HIPFIRE_GFX1201_DP4A=1`
  daemon run (32 tokens): the dp4a kernel dispatched **1280 times** (32 tokens ×
  40 layers — exact match), avg **20.46 µs/launch**, consistent with the
  microbench 23.37 µs/launch. The dp4a path fires on **every** decode step of
  **every** layer when the flag is set — no intermittent or silent fallback.

**Warm daemon A/B — median of 5 fresh-process runs each.** Model
`qwen3.6-35b-a3b.mq4r`, `--kv-mode q8 --spec off` (confirmed pure AR,
`[hipfire] DFlash disabled (dflash_mode=off)`). `--kv-mode` forces the CLI to
spawn a local daemon per invocation, so each measured run is a cold
`Engine::start()` (a genuinely fresh process). Each config warmed with a
throwaway `-n 16` before the 5 measured `-n 200` runs. Prompt
`"Explain the quicksort algorithm."` — md5 `195ebebaf38d280b6002a66ad94756c6`.

| Config | 5 runs (tok/s) | median | band |
|---|---|---:|---:|
| baseline (`HIPFIRE_GFX1201_DP4A` unset) | 125.2, 125.1, 124.8, 125.0, 124.8 | **125.0** | ±0.32% |
| `HIPFIRE_GFX1201_DP4A=1` | 123.1, 123.1, 123.2, 123.6, 123.0 | **123.1** | ±0.49% |

**Delta: −1.52% end-to-end** (median-to-median). Both bands are tight (<0.5%)
and **non-overlapping** (baseline min 124.8 > dp4a max 123.6) — a real,
reproducible LOSS, not session noise. It is consistent in direction with the
microbench's ~10% kernel-level loss on `moe_gate_up`, diluted by Amdahl's law
(gate_up is one of ~40 decode kernels for this MoE model). The −1.52% does not
cross the ±5% mandatory-investigation threshold, but it independently confirms —
via the real daemon serving path — that dp4a is not a win.

**Coherence verdict: PASS (coherent, both configs).** Both configs produce
fluent, on-topic, structurally sound quicksort reasoning at 200 tokens (the
budget is consumed inside the model's `<think>` trace — expected reasoning-first
behavior, not a bug). No attractor, no repetition loop, no garbage, no
special-token leak. Output was **byte-identical across all 5 repeats within each
config** (deterministic decode); the dp4a text differs from baseline only in
wording from int8-quant-noise-shifted argmax — not corruption. dp4a verbatim
sample (200-tok run, identical across all 5 dp4a repeats):

```
Here's a thinking process:

1.  **Understand User Request**: The user wants an explanation of the quicksort algorithm. This is a classic computer science topic. I need to explain it clearly, covering the key concepts, how it works step-by-step, its complexity, and possibly its pros/cons.

2.  **Identify Key Components of Quicksort**:
   - Divide and conquer algorithm
   - Pivot selection
   - Partitioning process
   - Recursive sorting of subarrays
   - Time complexity (best/average/worst cases)
   - Space complexity
   - Stability (not stable)
   - Practical considerations

3.  **Structure the Explanation**:
   - Introduction/Definition
   - Core Idea (Divide & Conquer)
   - Step-by-Step Process
   - Example (optional but helpful)
   - Complexity Analysis
   - Pros & Cons
   - Practical Notes/Improvements
```

(Baseline sample is textually near-identical in structure/content — the expected
small-quant-noise divergence, not a coherence failure.) `./scripts/coherence-gate.sh`
was not run — it is a Qwen3.5 ChatML/AWQ-specific battery (no `mq4r` row, does
not exercise `HIPFIRE_GFX1201_DP4A`), so it adds no signal beyond the direct
eyeball + byte-identical-repeats evidence above; skipped as impractical for this
opt-in flag.

### THE CALL: **EMPIRICAL-NULL** — coherent, but a real ~1.5% daemon LOSS

dp4a is **coherent** but **slower** (−1.52% daemon end-to-end / ~10%
kernel-level), so the analytic DRAM-bound verdict from section `(d)` and the
Pre-check resolution is now **EMPIRICALLY CONFIRMED**, not an analytic guess:
dp4a streams the same 4-bit weight bytes as the scalar kernel — cutting ALU work
does **not** help a bandwidth-bound kernel. The daemon A/B resolves the
long-standing ALU-vs-mem question that the failed rocprofv3 PMC split
(`(d)` Probe 1, twice) never could: `moe_gate_up` is **DRAM/latency-bound, not
ALU-bound**. Both variants sit at 100% occupancy with no ALU headroom, and the
dp4a path additionally pays for `ensure_q8_1_mmq_x`'s mandatory per-call Q8_1
quantization as a **second GPU kernel launch** — a real per-token tax in decode
(where `x` differs every layer), not a benchmark artifact. This is the strong
prod-falsify: a synthetic win that survives to the daemon would matter; here even
the microbench was a loss and the daemon confirms it. **Do not pursue dp4a
further on gfx1201 decode GEMVs.** The kernel ships closed: opt-in, default-off,
no default behavior changed on any arch.

### Reconciliation with the +4.24% multirow win (both stand)

**dp4a and the R=2 multirow re-tile are orthogonal levers — the multirow win
stands regardless of this NULL.** The +4.24% multirow win (section "R=2 multirow
spike result") is a **block/grid remap** of `moe_down` + `residual` that enlarges
per-block contiguous weight-traffic and reuses the x-vector once — it attacks
**transaction-granularity / small-grid** sub-roofline inefficiency (moe_down at
44% of peak BW) with **zero dequant-math change**. dp4a instead attacks the
(non-existent) **ALU/dequant** bound on `moe_gate_up` (already at 72.5% of peak),
and adds a quantize kernel launch. They touch different kernels, different
bottlenecks, and do not interact: the multirow win is real, coherent, byte-exact
greedy, occupancy-preserving (96 VGPR / 16 waves, zero spill) and remains
opt-in behind `HIPFIRE_GEMV_ROWS=2`; this dp4a NULL neither strengthens nor
weakens it.

### Honest ceiling + the real lever (unchanged)

The decode-GEMV kernel surface on gfx1201 is now **empirically exhausted**: the
one sub-roofline shape a source-supported re-tile could convert (`moe_down`) was
converted (+4.24% multirow), and the one shape with claimed ALU headroom
(`moe_gate_up`) is now measured DRAM-bound (this dp4a NULL). The honest
end-to-end ceiling on the whole decode-GEMV family stays as `(d)`/`(c)` bounded
it: **~146–168 tok/s** (s=1.35× DRAM-roofline row: ~146 wall / ~168 kernel,
i.e. +15–18%), and the broad-dp4a / LDS-tiling / wider-load NO-GO stands. The
**real competitive lever on gfx1201 remains wiring A3B MTP into the daemon** —
per the ROCmFP4 H2H (`project_rocmfp4_h2h_gfx1201_2026_07_01`), on R9700 the
competitor's llama.cpp fork **leads hipfire on raw AR** (131–147 vs `mq4r` ~124
tok/s) and hipfire wins **only via MTP** (152.7 tok/s, τ=3.02 coherent), which is
currently **`mtp_only_demo`-only and NOT daemon-wired for A3B** (ships AR-only).
That multiplicative decode win — on the axis where we actually trail this arch —
dwarfs both the +4.24% re-tile and (a fortiori) this dp4a NULL, but it is a
**greenlight-gated engineering task** carrying the **A3B R̄≈0.39 acceptance
caveat** (`feedback_a3b_r_not_acceptable`: default A3B = AR-only + no eviction
until R̄ improves) — **not another kernel gamble.** Prioritize the MTP-wiring
conversation over any further decode-GEMV kernel work on this arch.

## CORRECTION: decode is overhead-bound, not weight-GEMV-bound — 2026-07-02

**Own the error plainly.** Everything above this line — the `(a)`–`(d)` surface
map, the dp4a NO-GO, the +4.24% multirow win, the −1.52% dp4a NULL — pointed the
whole Phase-A/B campaign at the **58.38% weight-GEMV family**. That framing was
half right and strategically wrong. The weight GEMVs *are* the biggest single
bucket, but they are **near-bandwidth-bound and therefore nearly tapped**: the
roofline table in `(b)`/Probe-3 already showed the family blended at ~74% of the
611.8 GB/s DRAM peak with lm_head pinned at ~100%, and the two empirical spikes
**confirmed the ceiling is low** — multirow bought +4.24% by fixing one
transaction-granularity outlier (`moe_down` at 44% peak), dp4a *lost* −1.52%
because there was no ALU headroom to convert. Both were **correct experiments on
the wrong 58%.**

The fork's ~147-vs-our-~125 gap, and any path toward ~300, do **not** live in the
weight stream. They live in the **41.62% non-weight overhead** — Q8 attention plus
the ~16% norm/rotate/glue cluster spread across 10+ tiny kernels — which the
`(c)` Amdahl treated as a **fixed** `(1−p)` term and never attacked. That was the
mistake: the overhead is not fixed, it is **launch-count- and round-trip-bound**,
and it is exactly the surface the competitor's pre-capture fusion pass targets.

### The launch surface the earlier Amdahl treated as fixed

Reconstructed exactly from `crates/hipfire-arch-qwen35/src/qwen35.rs`
(`forward_scratch_layers` :11976-12533, MoE dispatch :12978-13002) and
`crates/hipfire-dispatch/src/pipeline/mod.rs` (`run_moe_decode` :232-862):

| Layer type | count | launches/layer | subtotal |
|---|---:|---:|---:|
| DeltaNetMoe (linear-attn + MoE-FFN) | 30 | 20 | 600 |
| FullAttnMoe (Q8 flash-attn + MoE-FFN) | 10 | 21 | 210 |
| Per-token non-layer (embed, final rmsnorm, lm_head, 2× sample) | — | 5 | 5 |
| **Total** | **40** | **~20.4 avg** | **≈815 launches/token** |

(The itemized 23-row surface-map sums to 787.3/token; the doc's own ~2.2%
un-itemized tail adds ~25-30 more → **≈810-820**, canonical **815**. This
supersedes the "~600 launches/token" rough figure used in earlier framing — the
true count is higher, and the MoE-FFN block alone is 10 launches × 40 layers =
400 of them.)

**hipGraph does not obsolete this.** Decode is hipGraph-captured by default on
gfx1201 (`graph_arch_default` + `HIPFIRE_GRAPH_MOE` + `HIPFIRE_AR_GRAPH` all
default-on, qwen35.rs:5044-5127), so the **host** does not pay 815 separate launch
calls per token — CPU dispatch latency is already amortized. What a captured graph
does **not** remove, and what fusion **does** remove: (1) per-kernel
occupancy/wave-launch ramp on tiny grids inside the `hipGraphLaunch` timeline, and
(2) the **HBM/L2 round-trip of every intermediate tensor** written by one kernel
and immediately re-read by the next (e.g. `y_gate`/`y_up` written by
`gemv_…_moe_gate_up` then re-read by a separate `silu_mul_f32`). That round-trip is
pure overhead a fused kernel keeps in registers — and it is why "we already graph
decode" is **not** a counter-argument to the fusion campaign.

### Corrected Amdahl — the overhead term is reducible

Baseline (wall-anchored): **8.0 ms/token = 125 tok/s.** Split by the measured
shares: weight-GEMV family 58.38% = **4.67 ms/tok**, non-weight overhead 41.62% =
**3.33 ms/tok**. The `(c)` error was locking that 3.33 ms as `(1−p)`. Correcting it:

| Scenario | weight ms | overhead ms | total ms | tok/s |
|---|---:|---:|---:|---:|
| Baseline (measured) | 4.67 | 3.33 | 8.00 | **125** |
| Fusion only (overhead → 2.50, ~25% of overhead reclaimed) | 4.67 | 2.50 | 7.17 | ~139 |
| Fusion + modest weight util → 74% BW | 4.11 | 2.50 | 6.61 | **~151** |
| Fusion + perfect weight streaming (100% BW, 3.04 ms floor) | 3.04 | 2.50 | 5.54 | **~181** |
| 300 tok/s would require | — | — | **3.33** | 300 (impossible) |

- **Fusion-reducible slice:** a conservative **~25% of the overhead** (≈10.4 points
  of total decode time, ≈0.83 ms/tok) is launch-ramp / round-trip waste on tiny
  operands (2048-dim hidden, 256-dim heads, 256-expert router logits) — real ALU
  work there is negligible. The independent no-GPU launch analysis put the same
  quantity at a **+12-15% tok/s fusion ceiling**; these bracket each other.
- **Irreducible floor fusion cannot touch (~17%):** Q8 FlashAttention 11.95% (KV
  reads/softmax — already single-kernel, GQA-aware on both engines) + DeltaNet
  recurrence core 3.56% (sequential-state math) + `moe_down_combine` 1.55%
  (**deliberately** split from the down-GEMV; qwen35.rs:5001-5024 documents that
  re-fusing the atomicAdd-combine reopens the task-#100 hipGraph-replay drift
  attractor — do **not** re-fuse it).

### Corrected realistic ceiling, and the two fork targets

- **Realistic ceiling ≈ 181 tok/s** (fusion of the reducible overhead **plus**
  perfect weight streaming). Even 181 assumes the weight family reaches 100% of
  DRAM peak — the multirow/dp4a spikes show that last stretch is hard, so a
  *pragmatic* target is the ~151-181 band.
- **Fork's 147 tok/s: REACHABLE.** ~151 tok/s falls out of *modest* fusion (25% of
  overhead) plus *modest* weight-utilization improvement (blended 74% BW, already
  near current) — it does not require perfect anything. This is the correct near
  target and it lands on the axis where we actually trail (raw AR).
- **~300 tok/s: NOT REACHABLE.** 300 tok/s = 3.33 ms/tok total, which is **below**
  the 3.04 ms pure weight-streaming floor plus **any** of the ~17% irreducible
  overhead. Even infinite fusion + perfect weight BW tops out at ~181. 300 on this
  arch/model requires a different regime entirely (MTP/spec-decode multiplier), not
  a faster decode loop.

### RANKED levers — attack the 42%, not the 58%

Headroom estimates are end-to-end tok/s, cheapest-first within the fusion tier:

1. **Fused SwiGLU MoE FFN — `gate_up` GEMV ⊕ SiLU ⊕ mul (~+3-5%, HIGH confidence,
   #1 lever).** Today `gemv_hfq4g256_moe_gate_up_k8_indexed` (`gemv.rs:5812`)
   writes `y_gate`/`y_up` to VRAM and a **separate** `silu_mul_f32`
   (`qwen35.rs:9187/:10012`) reads them back before down-proj — one extra launch
   **and** a full `[k_top × moe_intermediate]` HBM round-trip, every MoE layer,
   every token. Keep gate/up in-kernel and emit the activated product directly.
   This mirrors llama.cpp's `mul_mat_vec_q` GLU fusion (`fusion_data.gate`,
   ggml-cuda.cu:3800-3921) one-for-one and is the single highest-leverage fix.
2. **Fused MoE router — `softmax_f32` ⊕ `moe_topk_renorm_k8` (~+2-3%).** Two passes
   over the same 256-element router-logit vector (1.45% + 3.48% = 4.93% raw share,
   almost all launch/round-trip). Collapse to one softmax+top-8+renorm kernel —
   the analogue of llama.cpp's `topk_moe_cuda` (`topk-moe.cu`, ggml-cuda.cu:3153).
3. **Collapse the DeltaNet post-QKVZA glue chain (~+1.5-2.5%).**
   `fused_sigmoid_alpha_gate_f32` → `conv1d_silu_split_f32` →
   `fused_qk_l2_norm_scale_f32` → `repeat_interleave_qk_f32`: 4 launches × 30
   layers × 200 tok = 24,000 launches for **3.36%** of time — all sequential on the
   same tiny per-token vector, avg a few hundred ns/launch, pure launch overhead.
   Collapsible to 1-2 kernels.
4. **Fuse `mq_rotate_x` → `gemv_hfq4g256_residual` (wo projection) (~+1%).** Every
   other GEMV group in-tree already fuses rotate/rmsnorm into the following GEMV
   (`fused_qkvza`/`fused_qkv`/MoE-gate pattern); the wo step (`Step::GemvResidual`
   + `GemvInput::Raw`, qwen35.rs:12093-12106 / :12214-12227) is the **one** place
   the template was not applied. Fold `gated_norm_f32` (1.04%) into an adjacent
   epilogue at the same time — low-risk, template already exists.
5. **rope + KV-cache-write fusion (~+0.5-1%).** `rope_partial_halfsplit_f32` /
   `rope_interleaved_f32` and `kv_cache_write_*` are separate entry points; the
   fork folds ROPE→VIEW→SET_ROWS into one kernel that rotates K directly into the
   paged KV cache (ggml-cuda.cu:3754). Smaller, but same launch/round-trip class.
6. **Attention optimization (~0-2%, LOW headroom — NOT a top lever).** `(a)` shows
   FA at 11.95%, but it is already a single fused, GQA-aware tile kernel, as is the
   fork's WMMA FA path — the cross-engine comparison rules attention **out** as a
   structural gap. Only marginal tile/occupancy tuning remains; do not lead here.
7. **Launch reduction / better graph capture (subsumed, ~0% standalone).** Decode
   is already hipGraph-captured, so there is **no** standalone CPU-dispatch win
   left. Fewer kernels in the captured graph is a *consequence* of levers 1-5, not
   an independent lever — the value is the GPU-side ramp + round-trip elimination
   those fusions deliver, already counted above.

### The two spikes, in honest perspective

- **+4.24% multirow (moe_down R=2): a real, coherent win — but on the 58%.** It
  fixed a genuine transaction-granularity outlier and it ships opt-in. It is not
  wrong; it is **small and near the ceiling of its lever class** (the weight GEMVs
  were already ~74% BW-bound). Keep it.
- **−1.52% dp4a (moe_gate_up): a correct NULL — and on the 58%.** It empirically
  proved the weight GEMV was DRAM/latency-bound with no ALU headroom, plus a Q8_1
  quant-launch tax. It closed the dp4a question the right way. Also on the wrong
  bucket.

Neither result is retracted. The correction is **strategic, not numerical**: the
weight-GEMV surface is now empirically exhausted (one win, one null, roofline
confirmed), and the remaining decode headroom — the path to the fork's 147 and the
realistic ~181 ceiling — is **entirely in the 42% overhead**, reachable by fusing
the norm/router/glue cluster, not by porting more weight-GEMV kernels.

### Recommended next campaign phase: **decode-loop FUSION** (not more weight GEMVs)

Stop building weight-GEMV kernels on this arch — dp4a, LDS-tiling, wider loads,
and now multirow are all resolved (NO-GO / done). The next phase is a **decode-loop
fusion pass** in ranked order above, starting with **lever 1 (fused SwiGLU MoE
FFN)** as the single highest-leverage change. Target the fork's **147 tok/s** (raw
AR, reachable per the corrected Amdahl) as the near milestone and ~**151-181** as
the ceiling band; do **not** attach a 300 tok/s target to the decode loop (the
roofline forbids it — 300 is an MTP/spec-decode regime, tracked separately as the
standing A3B-MTP-into-daemon lever, `project_rocmfp4_h2h_gfx1201_2026_07_01`).
Each fusion lands behind the coherence gate and, being a pure kernel-boundary
merge with unchanged math, carries **low** coherence risk (the one exception —
`moe_down_combine` — is explicitly excluded above).
