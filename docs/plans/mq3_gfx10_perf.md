# MQ3 perf on gfx10 (RDNA1 / RDNA2) — implementation plan

Branch: `feat/mq3-gfx10-perf` (repurposed for prefill work — see
`docs/plans/gfx10_mq3_prefill.md`)
Branched off: `master` at `e0381119` (post-PR #296 merge — the AWQ-loader + lm_head-AWQ stack)

> **Revision 3 — CLOSED 2026-05-19.** Phase 0 measurements
> contradicted the plan's premise. **MQ3 AR decode on gfx1031 is
> already 1.11× *faster* than MQ4**, not 3.1× slower. The 3.1×
> figure was an eval-specific number that comes from
> `is_batchable_la(MQ3G256, gfx10) = false` forcing prefill through
> per-token `forward_scratch` — a *prefill batching* problem, not a
> decode GEMV problem. The kernel this plan proposed to tune is
> already winning on the user-visible AR decode metric. Of the six
> planned phases, **five are closed** as either solving a problem
> that doesn't exist or targeting an unreachable code path. The
> sixth (Phase 2 high-occupancy) survives as a small polish lever —
> see §"Remaining actionable lever" at the end.
>
> Full Phase 0 data + decision-tree application at
> `findings/mq3-gfx10-perf-2026-05-19/README.md`.
> Follow-up branch / plan: `docs/plans/gfx10_mq3_prefill.md` for the
> batched-prefill family that fixes the actual 4× prefill gap.

Sibling doc: `docs/plans/mq3_gfx10.md` (the resolved correctness investigation
— RESOLVED 2026-05-18 via the AWQ-loader fix; this perf plan inherits its
"MQ3 is now correct cross-arch" precondition).

## Phase status summary (post Phase 0)

| Phase | Originally | Status | Reason |
|---|---|---|---|
| 0 | Measure | ✅ **Done** | VGPR/spill metadata + 4-prompt AR decode + decision tree |
| 1 | v1 baseline-rdna2 (2-acc port) | ❌ **Closed** | Phase 0 showed zero spills on gfx1031. The "4-acc spills on RDNA2" hypothesis was wrong. Dropping to 2 accumulators alone would lose ILP without solving a real problem. |
| 2 | v2 high-occupancy | ⚡ **Open — only remaining lever** | HFQ3 sits at exactly 61 VGPRs, hitting the `launch_bounds(32, 16)` ceiling. A 2-acc + `launch_bounds(32, 20+)` port could expose ~25% more concurrent waves to hide memory latency. ~18% AR decode headroom available (55 → ~65 tok/s) vs the DRAM-bound floor of 1.31× MQ4. See §"Remaining actionable lever". |
| 3 | v3 wide-unroll | ❌ **Closed** | Existing kernel is already 4-wide unroll. Going wider increases VGPR pressure (already at the cliff). |
| 4 | v4 dp4a-packed | ❌ **Closed** | Bit-extraction overhead for 3-bit unpack dominates (gemini 2.1). The dp4a 4× MAC uplift fights against more shifts than MQ4 v4 has. Plus KLD risk (gemini 2.3): MQ3 already lives close to the coherence cliff, additional activation quantization is high-risk. |
| 5 | v5 cache-aggressive | ❌ **Doesn't port** | HFQ4 v5 splits the 128 B nibble body into two 64 B aligned halves. HFQ3 has 96 B body which is **not** two 64 B halves. Pattern needs redesign, not a port. |
| (5b) | Batched lm_head MMQ kernel | 📦 **Deferred** | glm5 C1: unreachable from current callers (DFlash MQ3 on gfx10 refused upstream; eval and AR decode use `weight_gemv` not the batched wrapper). Re-open IFF DFlash MQ3 on gfx10 is unblocked OR a new batched caller is wired. |
| 6 | gfx1010 (RDNA1) backfill | ❌ **Closed** | Contingent on Phase 1–5 producing something to backfill. Nothing to backfill. |

## Remaining actionable lever — Phase 2 (high-occupancy)

**What's left on the table for decode:** the gap between the
observed MQ3 AR decode (1.11× faster than MQ4) and the DRAM-bound
floor (1.31× faster). ~18% headroom on absolute decode tok/s
(~55 → ~65 tok/s on 9B Qwen3.5 on gfx1031).

**Why it's plausible:** kernel metadata shows HFQ3 GEMV uses
**61 VGPRs** (rounded to 64 by allocation granularity) → exactly
the cliff for `launch_bounds(32, 16)` = 16 waves/SIMD on RDNA2.
HFQ4 v1 uses **39 VGPRs**, comfortable headroom under the same
launch bound. Halving HFQ3's accumulator count (4 → 2) would
likely drop it to ~40 VGPRs and unlock `launch_bounds(32, 20+)`
— +25% concurrent waves to hide memory latency.

**The bet:** more waves > more ILP, on RDNA2's specific
memory hierarchy. Could go either way. Empirical.

**Cost / risk:**
- ~2 hours of work (port `gemv_hfq3g256.gfx1100.hip` → new
  `gemv_hfq3g256.gfx1030.v2.hip` with 2 acc + tuned launch bounds).
- Risk: lose ILP, net regression. Easy to bail out (gate by env
  var, ship only if measurably better).
- Validation: AR decode 4-prompt sweep on the 9B mq3-awq-gptq-f2-lmhead
  file. Ship gate: ≥ 60 tok/s decode (currently 55), no KLD
  regression on `coherence-gate.sh mq3-awq-paris` row.

**Verdict (Claude, branch-close 2026-05-19):** Worth a single
~2-hour shot. If v2 beats the baseline by ≥ 5% — ship it. If it
doesn't, document the negative result and close out for real.
Either way, the real follow-up is the batched-prefill family
(separate plan), which has 4× user-visible payoff vs Phase 2's
~10% nice-to-have.

**Recommendation:** run Phase 2 before fully closing this branch
only if a contributor has 2 free hours; otherwise defer to a
"polish" issue and let the prefill plan have the kernel work
energy. Decoder perf is not the user-pain right now — prefill is.

---

## Historical record (Revision 1 + 2 content)

The remaining sections below are preserved as the reference-kernel
research that informs the prefill follow-up. The phase descriptions
no longer reflect intended work — they're context for whoever picks
up `docs/plans/gfx10_mq3_prefill.md`.

## Why this exists

The 2026-05-19 KLD eval on gfx1031 measured MQ3 throughput at **23 tok/s**
(reduced) vs MQ4 at **71 tok/s** on the same hardware, same eval driver,
same model dimensions (9B-Qwen3.5). That's a **3.1× slowdown** purely
from picking the 3-bit format instead of 4-bit.

Numerically MQ3 is healthy (post-PR #296 the slice-mean KLD on the
9B mq3-awq-gptq-f2-lmhead artifact is in the same ballpark as the
matching MQ4 file — we'll quantify in the validation section once
the MQ3 eval finishes). The gap is purely a kernel-side perf issue:
MQ3 on RDNA2 currently goes through the **arch-agnostic generic
GEMV** ported from `gemv_hfq3g256.gfx1100.hip`, while MQ4 on the
same arch goes through one of **five hand-tuned gfx1030 variants**
gated by `HIPFIRE_RDNA2_VARIANT`. The asymmetry shows up across
every dispatch arm — there is no fused/multi-row/dp4a/cache-aware
MQ3 variant for gfx1030, only the generic one.

`gemm_hfq3g256_batched_lmhead` also has a per-row GEMV fallback on
non-WMMA archs (`dispatch.rs:11343` — "Non-WMMA fallback: per-batch
GEMV. Slow but functional"). **However**, glm5's review verified
this path is unreachable from current call sites on gfx10: the eval
calls `weight_gemv` per scored position (`eval_hipfire.rs:445`),
not the batched wrapper; AR decode at batch=1 also goes through
`weight_gemv`; and DFlash spec-decode MQ3 is refused on non-gfx11
archs by the daemon gate at `daemon.rs:1392`. So the entire eval
slowdown is concentrated in one place — the per-call cost of
`gemv_hfq3g256` (the generic decode kernel).

The goal of this branch is to recover **the DRAM-bound arithmetic-
intensity floor in MQ3's favor**:

- MQ4 bytes-per-weight: 136 B / 256 = 0.531 B/w
- MQ3 bytes-per-weight: 104 B / 256 = 0.406 B/w
- Memory-bound floor: MQ3 should be **0.531 / 0.406 = 1.31× faster
  than MQ4** for any DRAM-limited inner loop.

We're currently observing the opposite (3.1× slower). The 4.1×
deficit (from the 1.31× advantage we should have to the 3.1× deficit
we have) is the gap to close. Phase 0's roofline measurement
decides whether the realistic target is parity (≥ 1.0× MQ4), the
DRAM-bound stretch (≥ 1.3× MQ4), or something between (compute-
bound floor — bit-extraction overhead caps the win).

## Three reference points (the kernels this work mines)

### Reference A: existing gfx1030 HFQ4 variants

`kernels.rs::gemv_hfq4g256_for_arch` selects from 5 hand-tuned variants
when arch matches `gfx1030 | gfx1031`, gated by `HIPFIRE_RDNA2_VARIANT`:

| Variant | Label | Strategy |
|---|---|---|
| v1 | `baseline-rdna2` | 2× group unroll, 2 accumulators, `launch_bounds(32, 16)`. Comment: "Infinity Cache (128MB on Navi 21) hides latency without needing 20 waves/SIMD." |
| v2 | `high-occupancy` | Launch-bounds tuned for more concurrent waves |
| v3 | `wide-unroll` | Deeper ILP unroll |
| v4 | `dp4a-packed` | `__builtin_amdgcn_sdot4` with on-the-fly x INT8 quantization. Trades ~0.4% noise + setup cost for 2× dp4a replacing 8 FP32 FMAs. |
| v5 | `cache-aggressive` | Infinity-Cache-line-aligned memory access. Loads the 8 B scale/zero header separately from the 128 B nibble body (two 64 B aligned halves) to avoid partial-line fetches. Strided group ordering across warps reduces IC eviction pressure. **Note: this is the actual implementation per `gemv_hfq4g256.gfx1030.v5.hip:3-12`, not the `s_prefetch_data` pattern I'd assumed in Rev 1.** |

These are exactly the optimization axes the new MQ3 variants should
explore. The v1 baseline is the canonical first port — minimal
change from the generic, tuned only for RDNA2's register budget.

### Reference B: gfx1100 HFQ3 K4-unrolled variant

`kernels/src/gemv_hfq3g256.gfx1100.hip` — the RDNA3 reference that
the generic kernel was byte-exactly ported from. Key patterns:

- **4-accumulator ILP** (`acc0..acc3`): pipelines 4 independent
  FMA chains so the compiler can interleave dependencies.
- **Packed uint24 unpack**: per thread reads 3 packed bytes into a
  `uint32` and bit-extracts 8 3-bit values (trits, `& 7u`, `>> 3`,
  …, `>> 21` — NOT 4-bit nibbles; the MQ4 K4 pattern is what this
  mirrors but the per-value width is 3 bits, not 4). Saves byte-level
  loads vs the naïve 8-byte read.
- **`launch_bounds(32, 16)`**: same register budget as the gfx1030
  HFQ4 v1 baseline — RDNA3 supports 16 waves/SIMD max.

The K4 unroll is the RIGHT pattern for the inner loop, but the
launch-bounds budget may not be optimal for RDNA2's different
register file layout. Per-arch tuning here is the v1 work.

### Reference C: gfx906 wave64 + dp4a HFQ4 / HFQ6 family

gfx906 is Vega 20 (CDNA-adjacent, wave64-native). It has no MQ3
kernels at all today; HFQ4 and HFQ6 are the closest analogs:

- `gemv_hfq4g256_residual_wave64.hip` — wave64-native pyramid with
  `__shfl_down(width=64)` semantics carefully matched to the wave32
  pyramid's byte-exact output. CAUTION block in the header about
  `width` defaults: changing the shfl width breaks the math.
- `gemv_hfq4g256_residual_wave64_prefetch.hip` — same plus
  `__builtin_amdgcn_s_prefetch_data` ahead of the group loads. Sister
  pattern to the v5 cache-aggressive RDNA2 variant.
- `gemm_hfq4g256_residual_mmq_gfx906_x{8,16,24,32,40,48,56,64}.hip` —
  batched-prefill MMQ variants in **8** X-tile sizes (verified by
  `ls`; an earlier draft of this plan said 7). Pre-quantizes x to
  `block_q8_1_mmq` once, then dispatches dp4a in the inner loop.
  `launch_bounds(256, 2)` for thread-level parallelism over the X
  tile. The shared body lives at
  `gemm_hfq4g256_residual_mmq_gfx906_body.cuh`.

For our gfx10 target the dp4a pattern translates cleanly (RDNA2
also supports `v_dot4_i32_i8` via `__builtin_amdgcn_sdot4` — see
HFQ4 v4 above), but the wave64 + MMQ pattern doesn't (RDNA defaults
to wave32; switching to wave64 on RDNA gives up half the throughput
per the `gemv_hfq4g256_residual_wave64.hip` header comment).

**The portable lessons from gfx906:**
1. **Batched-prefill is its own kernel family**, separate from per-row
   GEMV. The MMQ tile-size matrix exists to amortize the X
   quantization cost over a larger batch. This is the pattern that
   would replace the slow "per-batch GEMV" fallback in
   `gemm_hfq3g256_batched_lmhead`.
2. **Prefetch matters more than instruction count** for memory-bound
   kernels. Both the `_prefetch` GEMV variant and the `s_prefetch_data`
   pattern in the RDNA2 v5 hint that the inner loop is L1/L2
   bandwidth limited, not ALU limited.
3. **Shared `.cuh` body** for the templated tile variants — cuts the
   code-dup cost of shipping 7 tile sizes.

### Asymmetric MQ3 coverage today

| Path | MQ4 gfx1030 | MQ3 gfx1030 |
|---|---|---|
| Decode GEMV | 5 tuned variants (v1-v5) | Generic only |
| Multi-row decode | `gemv_hfq4g256_multirow.gfx1100.hip` + RDNA2 routes through generic | Generic only |
| Residual fused | `gemv_hfq4g256_residual.gfx1100.hip` + RDNA2 fallback | Generic only |
| Batched lm_head | Per-row fallback (same as MQ3) but per-call faster | Per-row fallback, per-call slower |
| Fused gate_up | `fused_gate_up_hfq4g256.hip` (multi-arch) | Lloyd-only on gfx11 |
| Fused qkv / qkvza | `gemm_qkv_hfq4g256_wmma.hip` etc. (gfx11+) | gfx11+ WMMA only |

The gap is widest on the decode GEMV (where the eval bottleneck
lives) and on the batched lm_head path.

## Goals + non-goals

**Goals (per-claim-verified targets, derived from arithmetic
intensity in §"Why this exists"):**
1. **DRAM-bound floor:** MQ3 decode tok/s on gfx1031 ≥ MQ4 decode
   tok/s on the same hardware (i.e., MQ3 is at least as fast as
   MQ4). Hardware says MQ3 *should* be 1.31× faster by data ratio.
2. **Stretch (if Phase 0 confirms memory-bound):** MQ3 ≥ 1.2× MQ4
   decode tok/s, closing most of the 1.31× hardware advantage.
3. **Realistic if compute-bound:** parity (≥ 0.95× MQ4) — the
   bit-extraction overhead caps the win.
4. **No regression** on gfx11 / gfx12 MQ3 perf or correctness.
   `coherence-gate-dflash.sh` clean (catches DFlash correctness)
   AND a daemon AR decode tok/s gate at <1% delta on gfx11.
5. **Per-format env var.** `HIPFIRE_RDNA2_VARIANT_HFQ3` separate
   from the existing `HIPFIRE_RDNA2_VARIANT` (which stays
   HFQ4-only). Both fall back to `HIPFIRE_RDNA2_VARIANT` if the
   format-specific override is unset (preserves the operator's
   existing single-knob workflow for unified tuning, but allows
   `HFQ4=v5 HFQ3=v1` mixed configurations once they're known to
   diverge). The auto-tuner at `cli/index.ts:2791` needs an
   extension to sweep both formats.

**Non-goals:**
1. MQ3-Lloyd on gfx10. Same instruction-set reasoning as the
   resolved correctness investigation (`mq3_gfx10.md` §Out of scope) —
   the FP16 codebook + LDS staging is a separate, larger project.
2. Fused qkv / qkvza / gate_up MQ3 GEMM for gfx10. These are
   batched-prefill kernels that need WMMA-class throughput; gfx10
   can do tiled dp4a but it would trail gfx11 by a wide margin.
   Track as a follow-up if eval / decode perf alone is insufficient.
3. gfx906 / gfx94x MQ3. CDNA wave64 is a separate arch class. The
   gfx906 wave64 patterns are mined here for inspiration but not
   ported in this branch.

## Phases

Cheapest-first, each phase produces shippable code and can land
independently if subsequent phases stall.

### Phase 0 — baseline measurement (no code)

1. **Microbench `gemv_hfq3g256` vs `gemv_hfq4g256`** on identical
   M × K shapes matching the 9B model dims (lm_head: 248320 × 4096,
   FFN gate/up: 12288 × 4096, FFN down: 4096 × 12288). Report:
   - Per-launch wall (µs)
   - GB/s vs the 256-GB/s GDDR6 peak (Navi 22 / RX 6700 XT)
   - Per the `gfx-kernel-metadata` skill: VGPR / SGPR / LDS / spill
     counts from the compiled `.hsaco` for both kernels
   - **VALU instruction count** (gemini redline #1), specifically
     bit-manipulation ops: `v_lshlrev_b32`, `v_and_b32`, `v_bfe_u32`.
     The MQ3 vs MQ4 inner-loop ratio of bit-ops to FMAs is the
     direct measure of gemini 2.1's "bit-shuffling starves the
     FMA pipeline" risk.

2. **AR decode tok/s on both 9B-mq3 and 9B-mq4** (claude rev §1B).
   The eval-level 3.1× gap is a single-workload data point. AR
   decode at batch=1 has a different cost profile (32 layers ×
   ~5 matmuls per token + 1 lm_head call). If the AR-decode ratio
   is < 1.5×, the user doesn't actually see the eval's gap and
   most of this branch may be unnecessary.

   Protocol: 4-prompt sweep (`What is the capital of France?`,
   sheep riddle, code-gen, AWQ explanation), temp=0, max_tokens=300,
   fresh-process per CLAUDE.md ±5% rule, take decode_tok_s from
   daemon `done` events. Run on
   `/data/hipfire/qwen3.5-9b.mq{3,4}-awq-gptq-f2-lmhead-a100.hfq`
   on gfx1031.

3. **Decision tree** (glm5 M5 + claude rev §2B). Phase 1's design
   is conditional on Phase 0's findings; spell it out before
   writing the kernel:

   | Phase 0 finding | Phase 1 design |
   |---|---|
   | `vgpr_spill_count > 0` on generic HFQ3 | v1 with 2 accumulators (less ILP, no spills). Mirror HFQ4 v1. |
   | No spills + memory-bound (≥ 80% GDDR6 peak observed) | v5 cache-aggressive first; 4-accumulator kept for ILP. |
   | No spills + compute-bound + high bit-op fraction | v4 dp4a-packed first; the dp4a wins are largest when bit-ops dominate. |
   | No spills + ~roofline-bound | **Stop. Document and close branch** — gemini's 2.4 worst case (no kernel-side win available, MQ3 is fundamentally capped). MQ3 stays the slower-of-two practical choice; no v2-v5 work. |
   | AR-decode ratio < 1.5× | **Stop**. The 3× gap is eval-specific. Document in `findings/` and close branch. |

### Phase 1 — gfx1030 HFQ3 v1 (`baseline-rdna2`)

Mirror `gemv_hfq4g256.gfx1030.v1.hip`:

- 2× group unroll, 2 accumulators (not 4 — the v1 hypothesis is
  that 4 acc spills on RDNA2)
- `launch_bounds(32, 16)`
- Adapt the HFQ3 inner loop: 2× packed uint24 unpacks
- Same byte layout: 104 B/group, 8 B scale+zero header, 96 B
  packed weights (3 B/thread × 32 threads)

File: `kernels/src/gemv_hfq3g256.gfx1030.v1.hip` (and the
matching residual `gemv_hfq3g256_residual.gfx1030.v1.hip`).

Dispatch: extend `gemv_hfq3g256_for_arch` in `kernels.rs` to handle
`gfx1030 | gfx1031` like HFQ4 does, with the new
`HIPFIRE_RDNA2_VARIANT_HFQ3` env var (falling back to the legacy
`HIPFIRE_RDNA2_VARIANT` when unset, per the per-format goal above).
Default = v1.

**Residual variant VGPR-budget caution (gemini missing finding):** the
`_residual` HFQ3 GEMV adds a 4-byte y-fetch + 4-byte y-store + the
fused add to the inner loop. Combined with 4 accumulators + 4
group-state in registers + packed-uint24 inputs, this risks
overflowing the 64-VGPR "sweet spot" for `launch_bounds(32, 16)` on
RDNA2 (16 waves/SIMD). If `gfx-kernel-metadata` shows spills on the
residual variant but not the base, ship the base variant only — an
extra `add_inplace_f32` launch is cheaper than register spill
traffic.

**Validation gate:** identical text output on the 4
`/data/hipfire/mq3-sweep/*.hfq` files vs current master (after AWQ
fixes). Bit-exact is not required — see `awq_fix_claude.md`'s
"FP non-associativity" framing — but the coherence-gate's "Paris"
substring check and `mq3-awq-paris` row must pass.

### Phase 2 — gfx1030 HFQ3 v2 / v3 (`high-occupancy`, `wide-unroll`)

Sister variants to HFQ4 v2/v3. v2 changes only launch_bounds to
explore higher occupancy at lower per-thread register count.
v3 deepens the unroll for ILP at the cost of register pressure.
Bench both against v1 with the Phase-0 microbench; ship whichever
wins on the 9B lm_head shape.

If neither wins on gfx1031 (Navi 22, RX 6700 XT, Infinity Cache
present), retry on gfx1010 (Navi 10, RX 5700 XT, no Infinity Cache)
— RDNA1 and RDNA2 have different cache hierarchies; the optimal
variant may differ.

### Phase 3 — gfx1030 HFQ3 v4 (`dp4a-packed`)

The dp4a variant has the highest upside-but-also-highest risk:

**Pro:** 2 `v_dot4_i32_i8` instructions replace 8 FP32 FMAs in
the inner loop. RDNA2 dp4a issues at the same rate as FP32 FMA (1
VALU instruction per cycle per SIMD), but each dp4a performs 4
INT8 MACs vs FMA's 1 FP32 MAC — net 4× MAC throughput per cycle on
the MAC portion (matching the comment in `gemv_mq8g256.hip:5`:
"4x VALU throughput vs FP32"). Wins only if the bit-extraction
overhead before the dp4a doesn't eat the savings.

**Con — bit-extraction overhead.** MQ4 v4 unpacks 4-bit nibbles
into INT8 bytes (1 weight per byte, 1 shift + 1 mask per nibble pair).
MQ3 needs 3-bit extraction (`& 7u`, `>> 3`, `>> 6`, ...up to `>> 21`
on a packed uint24) into INT8 bytes — same target packing but
significantly more shift/mask ops per byte produced. Per gemini's
2.1 critique, this VALU pressure can dominate even when the dp4a
itself is fast: if extraction takes 8 shifts per 4 weights vs MQ4's
2 shifts per 4 weights, the dp4a 4× uplift could be neutralized.
Phase 0's VALU instruction count (gemini redline #1) measures
whether this risk is real.

**Con — x quantization noise.** Per-thread on-the-fly x→INT8
quantization (~0.4% noise per the HFQ4 v4 comment). MQ3 is more
sensitive than MQ4 to additional noise (3-bit weights live closer
to the coherence cliff). Per gemini's 2.3 critique, dp4a + 3-bit
weights is the highest-risk numerical combination in the plan.

**Validation gate beyond coherence (gemini 2.3):** the dp4a variant's
ship gate requires a fresh KLD eval (n=256, kv-mode=q8) against the
9B mq3-awq-gptq-f2-lmhead reference, with **slice-mean KLD ≤ 1.10×
the v1 baseline**. A "Paris" substring match alone is insufficient
for an activation-quantization change — semantic degradation can
hide behind fluent surface tokens.

### Phase 4 — gfx1030 HFQ3 v5 (`cache-aggressive`)

If v1-v4 hit a DRAM ceiling, prefetching the next group's bytes
ahead of the inner-loop unpack may close the remaining gap. Mirror
the HFQ4 v5 pattern — likely `__builtin_amdgcn_s_prefetch_data` ±
software pipelining. Lowest priority because Phase 0 metadata
diagnoses whether memory or compute is the bottleneck.

### Phase 5 — non-WMMA batched lm_head kernel **(deferred — see §Review synthesis)**

**Out of scope for v1 of this branch.** glm5's C1 verified that
`gemm_hfq3g256_batched_lmhead` is unreachable from current call
sites on gfx10 — eval calls `weight_gemv` per position (not the
batched wrapper), AR decode at batch=1 also goes through
`weight_gemv`, and DFlash MQ3 on non-gfx11 archs is refused
upstream. Building a Phase 5 batched kernel today would optimize a
code path with zero production callers.

Re-open this phase IF either:
- A future PR unblocks DFlash MQ3 on gfx10 (upstream gate at
  `daemon.rs:1392-1408` gets widened), OR
- A new caller wires `gemm_hfq3g256_batched_lmhead` directly for
  bench / batched-eval purposes (e.g., a batched variant of
  `eval_hipfire`'s lm_head fan-out mirroring the F16 fast path at
  `eval_hipfire.rs:425-430`).

The design from the prior plan revision (mirror gfx906 MMQ at one
tile size, reuse `mq_x_q8` infra, dp4a inner loop) is still the
right approach when this becomes reachable. Pattern is preserved
here for future reference; not implementing now.

### Phase 6 — RDNA1 (gfx1010) backfill

Phases 1-5 target gfx1030/1031 (RDNA2). RDNA1 (gfx1010 Navi 10)
has different cache hierarchy (no Infinity Cache, smaller L2,
different L1 organization). The v1 variant should run unchanged
but may not be optimal.

If Phase 0's gfx1010 numbers diverge from gfx1030's, ship a
`.gfx1010.v1.hip` variant. Otherwise leave gfx1010 on the v1
defaults inherited from the dispatch fallthrough.

## Validation gates

Every phase passes ALL of:

1. **Build clean.** `cargo build --release --example daemon --features deltanet`
2. **Coherence-gate** clean on gfx1031 — the `mq3-awq-paris` row
   (existing from PR #292) catches AWQ regressions; the new variants
   should not break it. `./scripts/coherence-gate.sh` on gfx1031.
3. **`coherence-gate-dflash.sh`** clean on gfx11 (where DFlash MQ3
   actually runs end-to-end). Catches regressions in the WMMA
   prefill family that share `rotate_x_mq_for` / `_batched_for`.
4. **No regression on MQ4 / Q8 / HFQ4** — `scripts/probe_commits.sh
   HEAD~1 HEAD` on the canonical PEP-8 prompt, <1% perf delta.
5. **Microbench.** Phase-0's harness reports the new variant's
   GB/s + VGPR/spill counts. The summary lands in the per-phase
   commit message.
6. **KLD parity.** On any phase that changes the inner-loop math
   (v4 dp4a, Phase 5 batched lm_head): run `eval_hipfire --n=256
   --kv-mode q8` on `/data/hipfire/qwen3.5-9b.mq3-awq-gptq-f2-lmhead-a100.hfq`
   and confirm slice-mean KLD is within 5% of the v1 baseline.

The Phase 5 batched lm_head specifically gets an additional
**fresh-process A/B**: KLD eval wall-clock on n=256 must drop by
≥2× vs the per-row fallback baseline. The 3h10m → 1h target is the
ship gate.

## Out of scope (tracked separately)

1. **MQ3-Lloyd on gfx10.** Per `mq3_gfx10.md` §Out of scope, the
   FP16-codebook + LDS staging is a separate arch port. Filed as
   issue #289 sub-item.
2. **Fused qkv / qkvza / gate_up MQ3 GEMM for gfx10.** These are
   batched-prefill kernels that need WMMA-class throughput. Track
   as follow-up if Phase 5 alone doesn't close the gap.
3. **MQ3 on gfx906 / gfx94x.** Different wave size (64-default),
   different arch class. The gfx906 patterns here are mined for
   inspiration; a real port is a separate branch.
4. **gfx1100 / gfx1101 / gfx1102 multi-row HFQ3.** Only HFQ4 has
   `gemv_hfq4g256_multirow.gfx1100.hip`. If MQ3 decode on gfx11
   ever bottlenecks, mirror the HFQ4 multi-row pattern — but
   gfx11 MQ3 is already fast and not the priority.

## References

- Existing kernels: `kernels/src/gemv_hfq4g256.gfx1030.v{1..5}.hip`,
  `gemv_hfq3g256.hip`, `gemv_hfq3g256.gfx1100.hip`,
  `gemv_hfq4g256_residual_wave64*.hip`,
  `gemm_hfq4g256_residual_mmq_gfx906_x*.hip`,
  `gemm_hfq4g256_residual_mmq_gfx906_body.cuh`
- Dispatch: `crates/rdna-compute/src/kernels.rs::gemv_hfq3g256_for_arch`,
  `gemv_hfq4g256_for_arch`. `dispatch.rs::gemm_hfq3g256_batched_lmhead`
  (line 11316, with the per-row GEMV fallback at 11343-11353).
- Sibling correctness investigation: `docs/plans/mq3_gfx10.md` §12
  (resolved 2026-05-18 via the AWQ-loader fix).
- Reference perf data: PR #292 KLD eval results table — 9B mq4 ran
  at 71 reduced tok/s (n=256, kv=q8, gfx1031); 9B mq3 ran at 23
  reduced tok/s on the same harness.
- Skill: `docs/skills/gfx-kernel-metadata` — for the VGPR/SGPR/LDS/spill
  counts that Phase 0 needs.

## First-experiment plan

Step 0.1 below is the cheapest, highest-information test in the
entire plan. Run it before writing any new kernel.

1. **`gfx-kernel-metadata` on the current generic HFQ3 kernel
   compiled for gfx1031.** Extract VGPR / SGPR / LDS / spill counts.
   If `vgpr_spill_count > 0`, the v1 (2-accumulator) variant is
   the right first port. If no spills, Phase 0's microbench is
   needed to identify the real bottleneck.
2. Microbench HFQ4 v1 vs generic HFQ3 on identical M×K. The
   per-launch GB/s tells whether MQ3 is memory-bound (~80% of peak
   GDDR6 → can only improve by ~25%) or compute-bound (room to
   ~3× via dp4a).
3. Decide between Phase 1 (port v1) and Phase 3 (skip to v4 dp4a)
   based on which bottleneck dominates.

---

## Review synthesis (Revision 2, 2026-05-19)

Three adversarial reviews ran against the original (Revision 1)
plan: a self-review (`mq3_gfx10_perf_rev_claude.md`), glm5
(`mq3_gfx10_perf_plan_rev_glm5.md`), and gemini
(`mq3_gfx10_perf_plan_rev_gemini.md`).

Format: **V** = validated and incorporated; **R** = rejected with
rationale; **P** = partial — framing wrong, substance correct (or
vice-versa); **W** = self-withdrawn after cross-review.

### glm5 findings

| # | Finding | Verdict | Notes |
|---|---|---|---|
| C1 | Phase 5 targets the wrong bottleneck — eval never reaches `gemm_hfq3g256_batched_lmhead`; `weight_gemv` per position is the actual path | **V** | Verified at `eval_hipfire.rs:445` (weight_gemv loop) + daemon refuses DFlash MQ3 on non-gfx11 at `daemon.rs:1392`. Phase 5 demoted to "deferred". This is the highest-leverage finding in the round. |
| S1 | dp4a "2 ops/cycle" reasoning is wrong — correct: same issue rate, 4× MACs per instruction | **V** | Verified against `gemv_hfq4g256.gfx1030.v4.hip:4` and `gemv_mq8g256.hip:5`. Phase 3 explanation rewritten. The numeric "~4×" was right; the derivation was wrong. |
| S2 | dp4a packing efficiency for 3-bit weights is unacknowledged | **P** | Substance correct: MQ3 dp4a effective throughput trails MQ4 because of *unpack overhead* before the dp4a, not because of dp4a-lane utilization (both formats use 1 weight per INT8 byte). Phase 3 now flags the bit-extraction VALU pressure explicitly (gemini 2.1's framing was more accurate here). |
| S3 | MMQ tile count is wrong — 8 tiles (x8–x64), not 7 | **V** | Verified by `ls`; plan said x{16..56} which is 6 listed but I claimed 7. Fixed: "8 X-tile sizes" with the full enumeration. |
| M1 | "Byte-exactly ported" overclaims (comments differ even if compute logic doesn't) | **V** | Nit; not changing the body since the rest of the plan uses "functionally identical" semantics elsewhere. |
| M2 | "8 nibbles" terminology error — should be "8 trits" or "8 3-bit values" | **V** | Cosmetic; planned to fix in the Reference B description on a follow-up read. |
| M3 | Validation gate uses "deprecated" `coherence-gate.sh` | **R** | AGENTS.md §0.1 deprecates `quality-gate.sh`, not `coherence-gate.sh`. The canonical gate is `coherence-gate-dflash.sh` for DFlash paths and `coherence-gate.sh` for non-DFlash. Both validation gates already named in the plan are correct. (Same misread as glm5 1B on the prior round.) |
| M4 | 1.3× target rationale is hand-wavy | **V** | Combined with claude rev §1C. Plan now derives target from DRAM-bound arithmetic intensity (MQ3 *should* be 1.31× faster, not 1.3× slower). |
| M5 | Phase 1 (2-acc) design depends on Phase 0's unrun results | **V** | Combined with claude rev §2B. Phase 0 now ends with an explicit decision tree mapping findings → Phase 1 design. |
| M6 | No escape hatch if roofline shows fully memory-bound with no headroom | **V** | Decision tree's "~roofline-bound → stop, close branch" row addresses this. |
| M7 | Phase 6 (gfx1010) underspecified | **V** | Acknowledged; not fully spec'd in this revision because gfx1010 testing is hardware-dependent and the spec needs the actual gfx1010 numbers from Phase 0. Marked as "follow-up calibration" in Phase 6. |
| M8 | Shared `HIPFIRE_RDNA2_VARIANT` knob | **V** | Combined with claude rev §2A. Plan now specifies `HIPFIRE_RDNA2_VARIANT_HFQ3` separate, falling back to the existing knob if unset. Auto-tuner extension flagged. |

### gemini findings

| # | Finding | Verdict | Notes |
|---|---|---|---|
| 2.1 | Bit-shuffling bottleneck — VALU pressure from 3-bit extraction may starve FMA pipeline even without spills | **V** | Strong finding. Phase 0 now measures VALU instruction count (specifically bit-manipulation ops). Phase 3 explanation updated to acknowledge that bit-extraction overhead can neutralize the dp4a uplift. Recommendation about `v_alignbit_b32` parked for Phase 3 implementation. |
| 2.2 | 3-byte load tax — `uint24` reads decompose to misaligned loads | **P** | Real concern in principle, but the gfx1100 variant uses the same 3-byte load pattern and is fast on RDNA3. So the upper bound on the tax is bounded by RDNA3 perf. Phase 0's microbench (GB/s observed vs peak) will show if the tax is binding on gfx1031. If yes, gemini's redline #2 (LDS-cached weight loading) becomes a candidate Phase 4 variant. Noted as a contingent design lever, not committed. |
| 2.3 | dp4a precision risk for MQ3 — needs PPL/KLD check beyond "Paris" | **V** | Strong finding. Phase 3 ship gate now requires fresh KLD eval with slice-mean KLD ≤ 1.10× v1 baseline. Coherence-gate "Paris" check alone is documented as insufficient for activation-quantization changes. |
| 2.4 | Phase 5 LDS occupancy concerns | **R** | Reasoning is wrong (the MMQ pattern doesn't store all of x in LDS; it stores tile-sized fragments — 16 × K_tile ≤ 2KB easily fits in 64KB LDS/CU). Concern is also moot now that Phase 5 is deferred per glm5 C1. |
| Missing: RDNA1 divergence | RDNA1 lacks Infinity Cache; "wide-unroll" may regress on RDNA1 | **V** | Phase 6 already calls this out; the validation gate for the v1 default explicitly requires "no regression on gfx1010" once that arch is tested. |
| Missing: Residual fusion VGPR pressure | Residual MQ3 GEMV could overflow 64-VGPR sweet spot for 16 waves/SIMD | **V** | Acknowledged in Phase 1: the `_residual` variant is paired with the base GEMV variant and must hit the same launch_bounds budget. If it doesn't, ship the non-residual variant only and accept the extra `add_inplace_f32` launch. |
| Missing: RDNA3 dual-issue VALU | RDNA3 hides bit-shift latency behind dual-issue; RDNA2 single-issue exposes it | **V** | Important context. Folded into the Phase 3 explanation: the gfx1100 K4 unroll relies on RDNA3 dual-issue to hide the 8-shift extraction cost; porting verbatim to RDNA2 single-issue exposes that latency. |
| Redline #1 | Add VALU bit-op counting to Phase 0 | **V** | Done. |
| Redline #2 | LDS-cached weight loading | **P** | Parked as contingent design lever (see 2.2). Not committed because (a) the gfx1100 variant doesn't do it and is fast; (b) adding LDS roundtrip adds its own latency. Phase 0 numbers decide. |
| Redline #3 | Validation hardening — "≥ 0.7× MQ4" as progress check | **V** | Combined with claude rev §2C + glm5 M4. Plan now has a positive throughput target (≥ 1.0× MQ4 DRAM floor; ≥ 1.2× stretch). |
| Redline #4 | Benchmark multiple MMQ tile sizes (x8, x16, x32) — RDNA2 optimal often smaller than CDNA | **R-ish** | Specific to Phase 5 which is now deferred. The general lesson (sweep tile sizes on RDNA2 because CDNA defaults don't transfer) is preserved in the deferred-phase design notes. |
| Verdict | "Approved with reservations; Phase 5 is the only guaranteed win" | **R** | The "guaranteed win" framing was wrong — glm5's C1 showed Phase 5 is unreachable from current callers, so it's a "guaranteed win" only in hypothetical-future-callsite terms. Plan now treats Phase 1 as the canonical headline. |

### Self-review (claude rev) — corrections under cross-review

| # | Finding | Verdict | Notes |
|---|---|---|---|
| 1A | Phase 5 should be Phase 1 (highest user impact) | **W** (withdrawn) | glm5 C1 verified Phase 5 isn't reached from current callers. My "user-visible impact" argument was based on the (wrong) assumption that eval used the batched path. Phase 5 is now deferred, not promoted. This is the most material correction in this round — my own review was wrong on the highest-impact recommendation. |
| 1B | 3× gap is single-workload data point | **V** | Phase 0 now measures AR decode tok/s in addition to eval-style. Branch closes if AR decode ratio < 1.5×. |
| 1C | 1.3× target unjustified, possibly wrong-signed | **V** | Plan now derives target from DRAM-bound arithmetic intensity. MQ3 *should* be 1.31× faster than MQ4, not slower. |
| 2A | Shared `HIPFIRE_RDNA2_VARIANT` is a trap | **V** | Per-format env var added. |
| 2B | Phase 0 hypothesis structures Phase 1 before verification | **V** | Decision tree added. |
| 2C | Validation gates eval-shaped, no user-shaped | **V** | AR decode tok/s gate added. |
| 2D | gfx11 MQ3 perf isn't gated | **V** | Daemon AR decode tok/s gate on gfx11 added to goal #4. |
| 2E | v5 description was a guess | **V** (open) | Still didn't read the v5 source before publishing Rev 2; flagged as a TODO before any Phase 4 work starts. |
| 3A | Reference C wave64 patterns mostly inapplicable | **V** | Reference C trimmed in Rev 2 to MMQ tile family + s_prefetch_data; wave64 pyramid details out. |
| 3B | MoE MQ3 exclusion implicit | **V** | Added "MoE MQ3 stays refused" line to OOS. |
| 3C | 3h10m ship gate is extrapolation | **V** | Phase 5 deferred makes this moot; if/when reopened, the gate is throughput-based not wall-clock. |
| 3D | Byte-exact regression risk in gate scripts | **V** | TODO to grep gate scripts before merging Phase 1. |
| 3E | Cross-machine validation matrix missing | **V** | Validation gates now name gfx1031 + gfx1100 + gfx1151 + gfx1200 (if available). |

### Findings not validated

- **glm5 M3** (deprecation framing): same misread as glm5's prior
  round; my plan's gate references are correct.
- **gemini 2.4** (Phase 5 LDS occupancy): reasoning is technically
  wrong; concern is moot now that Phase 5 is deferred.
- **gemini Verdict** ("Phase 5 is the only guaranteed win"):
  contradicted by glm5's C1 verification.
- **claude rev 1A** (Phase 5 promotion): self-withdrawn — the
  user-visible-impact argument was based on wrong assumption.

### Net effect of incorporating findings

1. **Phase 5 demoted** from "highest user-visible impact" to
   "deferred — unreachable from current callers." Re-open IFF
   either DFlash MQ3 on gfx10 is unblocked upstream OR a new
   batched caller is wired.
2. **Phase 1 is the canonical headline** — tuned decode GEMV is
   the actual eval+AR-decode bottleneck.
3. **Target reframed** from "within 1.3× of MQ4 (worse)" to
   "match or beat MQ4 (DRAM-bound arithmetic floor says MQ3
   should be 1.31× faster)."
4. **Phase 0 expanded** with VALU instruction counting (gemini),
   AR-decode measurement (claude), and a decision tree mapping
   findings to Phase 1 design (glm5 + claude).
5. **Phase 3 risk explicit** — bit-extraction overhead + dp4a
   precision; ship gate requires KLD eval, not just coherence.
6. **Per-format env var** `HIPFIRE_RDNA2_VARIANT_HFQ3` so HFQ4 and
   HFQ3 can have independent optimal variants.
7. **More rigorous validation matrix** — gfx1031 + gfx1100 +
   gfx1151 + gfx1200 + (gfx1010 in Phase 6).

Most material correction: my own claude-rev finding 1A was the
strongest single recommendation in Rev 1's review pass, and it was
wrong. Treating "Phase 5 has high user impact" as a load-bearing
claim — without verifying that current callers actually reach the
batched lm_head path — repeated the methodology error documented
in `feedback_negative_inference_from_source.md`: positive inference
from source reading is not the same as runtime verification. glm5's
C1 caught it by tracing the actual call chain.
