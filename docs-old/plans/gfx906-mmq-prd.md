# gfx906 MMQ adaptation — PRD (post-redesign)

**Status:** Closed (2026-05-05). Goal achieved: 95% of stock llama.cpp
pp512 on Qwen 3.5 9B (713 vs 750 tok/s). Outstanding follow-ups
documented below; not currently scheduled.

**Owner:** hipfire research
**Hardware:** AMD Instinct MI50 (gfx906, Vega 20)
**See also:** `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
(the dev log for the executed work)

This PRD consolidates three earlier planning docs that drove the gfx906
MMQ work:

- `plans/gfx906_mmq_plan.md` — Phase 0/1 adaptation plan (original
  direction: nwarps=2 dp4a substitution into the existing 128×128 tile)
- `plans/p1.2_dp4a_mmq_design.md` — pre-redesign kernel design doc
- `plans/gfx906_mmq_l2.md` — L2 prefetch investigation
  (rejected by rocprof attribution)

The original Phase 0/1 design was superseded mid-effort by the redesign
(commit `c022682` onwards) when stock-comparison work showed the
nwarps=2 topology was the structural bottleneck, not just the lack of
dp4a. The PRD below preserves the durable findings — gfx906
architectural reference, layered decision rationale, negative results
on prefetch — without re-litigating the superseded design points.

---

## 1. Problem

**Pre-redesign baseline (2026-05-03):** hipfire's gfx906 prefill on
Qwen 3.5 9B pp128 was **137 tok/s** (FP16 wave64 hybrid). llama.cpp on
the same MI50 hit **244 tok/s pp512** — a 1.7–3.3× gap depending on
batch size. Decode was within 1.24× (60 vs 49 tok/s) and not the
bottleneck.

The architectural diagnosis (in `docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`):

- **hipfire's old approach:** row-parallel FP16 wave64 — one warp per
  output row, no LDS, no data reuse. 64 large kernels (113 ms avg).
- **llama.cpp's approach:** tiled MMQ with LDS staging, Q8_1
  pre-quantized activations, int8 dot-products, 8× weight reuse per
  tile. 1182 small kernels (1.77 ms avg), heavy async overlap.

Phase 0 rocprof showed the FP16 wave64 path is **compute-leaning**
(VALUBusy 58–65%, MemBusy 75%, MemStall <2%, L2 hit ~90%). MMQ's
value on this hardware is **arithmetic density** (dp4a 4 MACs per
instruction vs FP16 v_fma 2 MACs) — *not* bandwidth savings.

## 2. gfx906 architectural reference (durable)

From skyne98/wiki-gfx906 (measured on real MI50):

| Property | Value |
|---|---|
| CUs | 60 (MI50) / 64 (MI60) |
| Clock | 1725 MHz |
| Wavefront size | 64 |
| LDS/CU | 64 KiB, 32 banks × 4 B |
| L1 Instruction Cache/CU | 32 KiB |
| L2 | 4 MiB |
| HBM2 | 1024 GB/s |

**Per-CU limits at nwarps=4 (256 threads/WG, target 2 WGs/CU):**
- LDS: ≤32 KiB/WG
- VGPRs: ≤128/thread
- Stock fits at 128 VGPRs/thread (zero margin) and 28.5 KiB/WG LDS.
- **We don't get more headroom than stock.** Anywhere we exceed these,
  occupancy halves.

**Integer dot instructions on gfx906 (verified on real MI50):**

| Instruction | Throughput (peak) | Notes |
|---|---|---|
| `v_dot4_i32_i8` (dp4a) | 43–44 TOPS w/ ILP | 4 MACs per instruction |
| `v_dot8_i32_i4` (dot8) | 85–86 TOPS w/ ILP | 8 MACs per instruction; 4-bit raw nibbles |
| QDQ-amortized dp4a | 21.7 TOPS | When dequant overhead is amortized across tile reuse |
| LDS bandwidth (b128 reads) | 9.5–11.2 TB/s | per-CU, with 16-B aligned ds_read_b128 |

Ruled out: `__builtin_amdgcn_wmma_*` (gfx906 has no WMMA — RDNA3+ only).

## 3. What was tried & resolved

### 3.1 Phase 0 (2026-05-03): nwarps=2 dp4a kernel ([RESOLVED, superseded])

**Plan:** Drop dp4a into the existing 128×128 tiled MMQ kernel
structure, gated by an arch check, keeping the nwarps=2 topology
inherited from the unverbraucht fork's `gfx906-config.h`.

**Outcome:** Built, debugged, eventually shipped a correct nwarps=2 dp4a
kernel in `feb8e08`. Hit **148 tk/s pp128** (1.05× over FP16 wave64
baseline, but only 60% of llama.cpp's 244). Investigation revealed
the unverbraucht config was stale and the actual stock topology was
nwarps=4 — see §3.4.

**Durable findings preserved in `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md` §2:**
- Symmetric weight unpacking trick (`(n - 8)`)
- F32 LDS staging for x_dm (half2 → float2)
- mmq_screen_weight infrastructure
- Diagnostic flags (`HIPFIRE_MMQ_*`)
- Row 3994 dump-data quirk

### 3.2 P1.2 design (2026-05-03): unpacked-at-load LDS layout ([RESOLVED])

Decision: unpack HFQ4 nibbles to one nibble per byte (int8) at LDS
load time, vs llama.cpp's "keep packed in LDS, unpack during dp4a"
strategy. Both approaches survive in the redesigned kernel:
the redesign's `load_hfq4_tile_streaming` does the unpack, dp4a then
reads the prepared bytes.

**Trade-off recorded:** packed-in-LDS saves ~16 KB but requires
interleaved Q8_1 layout from the quantize kernel. Not worth the
complexity at LDS budgets the redesign achieves (27 KiB/WG at mmq_x=64
fits 2 WGs/CU comfortably).

### 3.3 L2 prefetch ([RESOLVED — REJECTED])

llama.cpp-gfx906 emits explicit L2 prefetch for the next k-block
(`AS_S_LOAD_DWORD` 1024 B ahead). v1 plan proposed the same — gated on
≥10% prefill gain.

**rocprof attribution (`docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`)
killed the lever:**
- VMEM_WR 7.9× FP16 per call (spill-write traffic dominant)
- L2 hit 65% vs FP16's 85%
- FLAT loads tiny (0.04× FP16) — **HBM/L2 prefetch on weight path
  would do nothing**
- Global loads aren't on the critical path; spills are

The L2 prefetch optimization is **not applicable** to our shape — even
though stock uses it, our kernel was bottlenecked elsewhere. After the
redesign, MemUnitStalled dropped to ≤0.25 across all MMQ kernels (see
final report §3.5), confirming HBM is even less of a factor post-redesign.

**Lesson:** The mismatch was a kernel-design issue (nwarps=2 + small
mmq_x produced spill pressure), not an HBM bandwidth issue. Prefetch
masks symptoms; the redesign fixed the cause.

### 3.4 The redesign (2026-05-04 → 2026-05-05) ([COMPLETED])

Stock-comparison rocprof (`docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`)
showed our nwarps=2 kernel was **per-call 1.22× slower than stock** at
the residual K=4096 shape, but the topology gap was the bigger
opportunity:

- Stock: nwarps=4, 256 threads/WG, runtime mmq_x dispatch ∈ {8..64}
- Hipfire: nwarps=2, 128 threads/WG, hardcoded mmq_x=8

Redesign fully described in
`docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`. Final
result: **95% of stock pp512 (713 vs 750 tok/s)**, 4.24× pp128
speedup, 5.02× pp512 speedup over pre-redesign baseline.

## 4. What was *not* tried (open lever inventory)

These are levers either deferred from the redesign or rejected mid-flight.
Each entry has an estimated impact + cost ratio, captured at branch
close (2026-05-05). The list is **not** scheduled work.

### 4.1 Path B: true fused 4-output qkvza MMQ kernel

**Status:** Deferred. The redesign uses Path A (split qkvza → qkv+z
through MMQ, beta+alpha tail through fused FP16 wave64 with
`qkv_m=z_m=0`). Path B would write a single fused 4-output MMQ kernel
mirroring the existing FP16 wave64 design.

**Estimated impact (post-redesign):** 0.3% of GEMM time is the
beta+alpha tail still on FP16 wave64. Path B's marginal value is
small.

**Cost:** 3–5 days (per original plan estimate).

**Trigger:** if a future workload exposes the beta+alpha tail as a
larger share, e.g. a model with thicker linear-attention.

### 4.2 dot8 (`v_dot8_i32_i4`) MMQ

**Status:** Speculative. gfx906 has `v_dot8_i32_i4` (8 MACs per
instruction vs dp4a's 4) which would in principle close more of the
remaining 5% gap to stock.

**Estimated impact:** llama.cpp doesn't use dot8 in the main MMQ path
(it uses dp4a + Q4_K), so this would be moving past stock's design
point, not catching up to it. Realistic gain unclear without a probe.

**Cost:** New kernel — 8–12 days (per original plan estimate). Requires
new Q4_1 quantize layout to feed dot8 (vs current dp4a + Q8_1).

**Trigger:** if there's appetite to push past stock parity. Probably
not until the dispatch / qkvza tail / KV-cache levers are exhausted.

### 4.3 Reduce sync 4 → 2 per HFQ4 group

**Status:** Speculative — rejected at the time. Stock uses 2 syncs/group;
our redesign uses 4 (see final report §11.5). Going to 2 would require
**256-K-resident x_qs**, which forces 1 WG/CU (Phase 2a Gate 3 fail at
mmq_x=64).

**Estimated impact:** unclear. Tighter sync but halved occupancy.

**Cost:** 2–3 days probe + redesign at single-WG/CU.

**Trigger:** if a future LDS-budget breakthrough (e.g. compressed weight
layout with smaller LDS footprint) makes 256-K-resident x_qs fit at 2
WGs/CU.

### 4.4 attention_q8_0_kv_batched per-call doubling

**Status:** Out-of-scope for MMQ branch. The attention kernel's
per-call time doubles from pp256 (0.66 ms) to pp512 (1.29 ms). Small
share (1.75% of GEMM time at pp512) but suspicious 2× scaling pattern.

**Estimated impact:** if scaling is sub-linear instead of 2×, marginal.

**Trigger:** worth investigating as a separate task before any KV-cache
or attention-kernel work.

### 4.5 Test harness mmq_x parameterization

**Status:** Polish.
`crates/rdna-compute/examples/test_gfx906_mmq_correctness.rs` currently
takes `M K N` and dispatches via the production greedy ladder. A future
contributor running per-mmq_x correctness sweeps would benefit from
adding an explicit `--mmq-x` flag.

**Cost:** ~1 hour. Trigger: next time someone needs it.

## 5. Decisions log (durable)

These were settled during the redesign and aren't re-opened by the
follow-ups above.

| Decision | Rationale |
|---|---|
| nwarps=4 (vs nwarps=2) | Stock topology; per-warp register budget allows mmq_x=64 |
| Runtime mmq_x dispatch ∈ {8..64} step 8 | Match stock's `mmq_x_max_device()`; mmq_x=128 doesn't exist on gfx906 |
| Option C window streaming (4 syncs/group) | LDS budget forces ≤128-K resident; Option B (8 syncs) was first-land but Option C +10% wallclock |
| Per-mmq_x X_STRIDE (33 small / 40 large) | Bank-conflict vs b128-alignment trade-off; small kernels lack j0 iters to amortize 4-way conflict |
| `mmq_screen_threshold = 0.50` on gfx906 | Recalibrated post-redesign; old 0.10 was set when kernel was buggy |
| min_batch=16 default-on for gfx906 | Below pp16, Q8_1 quantize + per-output launch overhead dominates |
| qkvza split: qkv+z to MMQ, beta+alpha to FP16 wave64 tail | beta/alpha M=32 << MMQ_Y=128; tail kernel handles small M efficiently |
| ds_read_b128 only at mmq_x ≥ 64 | Smaller kernels regress -2.6% pp32 under b128 due to int4 unpack overhead |

## 6. Lessons learned

(Copied from the dev log for completeness; the canonical source is
`docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md` §11.)

1. **PMC counters > ELF metadata for LDS-layout changes.** `LDSBankConflict`
   and `ALUStalledByLDS` should be checked first; ELF "looks clean" can
   hide a 47% bank-conflict regression. See
   `~/.claude/projects/.../memory/feedback_lds_bank_conflict.md`.

2. **Stride choice trades alignment vs bank conflict.** Per-mmq_x stride
   templating is the right shape when small and large kernels prefer
   different points on the trade-off curve.

3. **Screening threshold needs recalibration on kernel changes.** The
   0.10 default was set when the kernel was buggy; carrying it forward
   masked a 14% pp128 gain.

4. **"200% CPU max" ≠ CPU-bound.** Per-process %CPU samples need
   sub-second granularity and process-state context (D-state vs R).

5. **L2 prefetch is the wrong lever when you're not HBM-bound.** Stock
   uses it, but verify the bottleneck before copying their design.

6. **MMQ's value on gfx906 is arithmetic density, not bandwidth.** dp4a
   gives 4× the FMA density of FP16 v_fma at the same issue rate. This
   is why the kernel beats FP16 wave64 even though the latter has
   higher VALUBusy.

## 7. References

- Final report: `docs/perf-checkpoints/2026-05-05-gfx906-mmq-redesign-final.md`
- Pre-redesign attribution: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-attribution.md`
- Stock comparison: `docs/perf-checkpoints/2026-05-04-llamacpp-stock-comparison.md`
- Pre-redesign j0-unroll: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-junroll.md`
- Pre-redesign spill reduction: `docs/perf-checkpoints/2026-05-04-gfx906-mmq-spill-reduction.md`
- iacopPBK/llama.cpp-gfx906 (original gfx906 fork; canonical reference
  for warp-coop GEMV, Y-tile prefetch, load-defer pipelining):
  https://github.com/iacopPBK/llama.cpp-gfx906
- skyne98/llama.cpp-gfx906 (fork-of-iacopPBK that ports iacop opts +
  tracks upstream): https://github.com/skyne98/llama.cpp-gfx906
- skyne98/wiki-gfx906 (gfx906 ISA reference — LDS bank-conflict
  patterns, dp4a issue rate, Q8_1 layout):
  https://skyne98.github.io/wiki-gfx906/intro.html
