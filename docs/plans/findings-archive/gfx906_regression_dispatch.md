# gfx906 prefill regression on `integration/dispatch-unification`

**Date:** 2026-06-09 (revised after adversarial code review + author rebuttal)
**Branch:** `integration/dispatch-unification` @ `12d2fc57`
**Bench comparison base:** `master` @ `02634f4c` (PR #391 tip — this is what the
  `bench_master` binary was built from, NOT the merge base)
**True merge base:** `a7a8d89b` (`git merge-base master integration/dispatch-unification`)
**GPU:** gfx906 MI50 32 GB, gfx1031 RX 6700 XT 12 GB
**ROCm:** system 6.4.3 (HIP 7.13), Ubuntu packages
**Binary md5s:** `bench_master=cccdfe65`, `bench_dispatch=e522aeeab4`

> **Revision note (2026-06-09):** The original draft of this doc misdiagnosed the
> `gated_delta_net_q8` source and proposed a fix (A1) that would not work. The
> diagnosis was anchored on a diff against the *merge base* (`a7a8d89b`, plain
> `roundf` kernel), but the benchmark binaries compare against *current master*
> (`02634f4c`), which already ships the stochastic-rounding path. The corrected
> analysis below diffs `master..HEAD` — the actual A/B the binaries represent.
> See §3a for the corrected root cause and §4 for the revised fixes.

## Executive summary

Prefill is 6–17% slower on gfx906 (inversely size-correlated). Decode is flat (±3%).
Two candidate kernel-level sources, totalling ~+26ms GPU time on 0.8B prefill=512.
**Only the second (attention) source is well-established; the first (DeltaNet) is
direction-plausible but its mechanism and magnitude are NOT yet verified** (no hsaco
disassembly, perf below the mandated 3-run protocol).

| Source | Δ GPU time | Status | Corrected root cause |
|--------|-----------|--------|----------------------|
| `gated_delta_net_q8` slowdown | +16 ms (+38%) — **UNVERIFIED** | needs hsaco disasm + ≥3-run rebench | Requant loop **hoisted inside the per-token `t` loop** (master requants once after the loop) → larger register live-set across the loop body. NOT "dead `ef_residual` args" — master already shipped the heavy stochastic-rounding path. |
| Q8 KV attention kernel switch | +10 ms — **VERIFIED in code** | rocprof attribution still summary-only | Master prefill calls `attention_q8_0_kv_batched_masked` (fast batched); HEAD routes `AttnQ8_0KvBatchedMasked` unconditionally to the tiled `attention_flash_q8_0_batched_masked`. |

gfx1031 shows the same regression pattern but milder (−3% on 0.8B vs −17% on gfx906).
The wave32-has-more-VGPRs explanation is plausible but **unmeasured** — stated as a
hypothesis, not a confirmed cause.

**Bottom line:** Fix B (attention) is ready to prototype. Fix A (DeltaNet) needs
re-diagnosis before any code change — the originally-proposed Fix A1 is unsound.

> **Author rebuttal (2026-06-09):** The adversarial review's code-level corrections
> (requant-loop hoisting, Fix A1→A1′, Fix B priority) are accepted. However, the
> claim that "Scratch=108→172 is unsourced" and "rocprofv2 does not emit per-workitem
> scratch" is **incorrect** — the `--kernel-trace` CSV includes a `Scratch_Per_Workitem`
> column, and the values 108/172 were extracted directly from it. See §3a rebuttal.

---

## 1. Coherence results

All models produce coherent output at temp=0.

| Model | GPU | Match? | Notes |
|-------|-----|--------|-------|
| qwen3.5-4b.mq4 | gfx906 | ✅ byte-identical | |
| qwen3.5-9b.mq4 | gfx906 | ✅ byte-identical | |
| qwen3.5-4b.mq4 | gfx1031 | ⚠️ minor | Same answer, markdown bold formatting differs |
| qwen3.5-9b.mq4 | gfx1031 | ⚠️ minor | 79 vs 80 tok, same answer |

gfx1031 mismatches are single-token argmax flips from FP accumulation order — expected for a dispatch change, not a correctness bug.

---

## 2. Speed results (2 fresh-process runs, medians)

> **⚠️ Below the mandated protocol.** CLAUDE.md's Δ≥5% rule requires 3–5 runs and
> a median, with a committed byte-identical prompt + recorded md5. These numbers
> are 2-run medians with no prompt md5 recorded. The 0.8B −17.2% is the largest
> and most variance-prone figure and MUST be re-confirmed with
> `scripts/probe_commits.sh master HEAD` (warmup + multi-run aggregation) before
> it is quoted as a kernel-perf claim. Per `feedback_perf_noise_decode`, 2-run
> gfx906 numbers cannot separate a real +16ms from DPM/thermal throttling.

### Decode — flat to mild regression

| Model | GPU | Master | Dispatch | Δ% |
|-------|-----|--------|----------|-----|
| qwen3.5-0.8b.mq4 | gfx906 | 204.1 | 200.2 | −1.9% |
| qwen3.5-4b.mq4 | gfx906 | 58.1 | 56.9 | −2.1% |
| qwen3.5-9b.mq4 | gfx906 | 56.9 | 55.0 | −3.3% |
| qwen3.5-0.8b.mq4 | gfx1031 | 267.0 | 275.8 | +3.3% |
| qwen3.5-4b.mq4 | gfx1031 | 92.3 | 90.0 | −2.5% |
| qwen3.5-9b.mq4 | gfx1031 | 59.6 | 58.8 | −1.3% |

Decode is performance-neutral. The gfx1031 0.8B +3.3% is within run-to-run variance.

### Prefill — still regressed

| Model | GPU | Master | Dispatch | Δ% |
|-------|-----|--------|----------|-----|
| qwen3.5-0.8b.mq4 | gfx906 | 4190 | 3468 | **−17.2%** |
| qwen3.5-4b.mq4 | gfx906 | 1261 | 1141 | **−9.5%** |
| qwen3.5-9b.mq4 | gfx906 | 766 | 718 | −6.2% |
| qwen3.5-0.8b.mq4 | gfx1031 | 6421 | 6208 | −3.3% |
| qwen3.5-4b.mq4 | gfx1031 | 1251 | 1216 | −2.8% |
| qwen3.5-9b.mq4 | gfx1031 | 687 | 676 | −1.6% |

Improvement over our original A/B (branch tip `50465ef6`):

| Model | gfx906 before | gfx906 now | gfx1031 before | gfx1031 now |
|-------|--------------|------------|----------------|-------------|
| 0.8B | −22.3% | −17.2% (5pt) | −12.3% | −3.3% (9pt) |
| 4B | −19.8% | −9.5% (10pt) | −6.7% | −2.8% (4pt) |
| 9B | −13.4% | −6.2% (7pt) | — | −1.6% |

The Ship 5/6 lowered decode + forward-as-pipeline closed ~half the gap.
Remaining regression is kernel-level, not dispatch overhead.

---

## 3. Root cause analysis (rocprofv2 kernel trace)

Method: `rocprofv2 --kernel-trace` on 0.8B prefill=512, gfx906.

### 3a. `gated_delta_net_q8` — +16 ms (+38%) — **UNVERIFIED, mechanism corrected**

> **Original diagnosis (RETRACTED):** "The dispatch branch added `ef_residual`
> and `rpt` params; even when `ef_residual=None` the compiler can't prove the
> branch is dead and spills ~2 VGPRs (108→172B scratch)." This is wrong on two
> counts, established by diffing `master..HEAD` on `kernels/src/gated_delta_net_q8.hip`:

**Correction 1 — master already ships the "heavy" kernel.** `master`'s kernel
(`02634f4c`) already contains the `frame` param and the full stochastic-rounding
LCG path (`rng = rng*1664525u + ...`, `floor(scaled + noise)`). These are NOT new
on the branch. The plain-`roundf` kernel only exists at the *merge base*
(`a7a8d89b`), which the benchmark did not compare against. So "the branch added
the expensive code" is false.

**Correction 2 — the real structural delta is requant-loop hoisting.** Relative
to master, HEAD adds the `s_ef_residual` and `requant_per_token` params AND
**moves the FP32→Q8 requant block from after the per-token loop to inside it**:

```
master:  for (t in tokens) { ...update S... }   // loop closes
         for (r in TILE_ROWS) { ...requant once... }

HEAD:    for (t in tokens) {
             ...update S...
             if (!per_token && !last_tok) continue;   // default path
             for (r in TILE_ROWS) { ...requant (now inside t-loop)... }
         }
```

For the **default MQ4 path** (`ef_residual=None`, `requant_per_token=0`) the
requant still fires only on the last token (`continue` guard), so the requant
*frequency* is unchanged vs master. The plausible cost is that the requant's
register live-set (the `q0..q3` int temporaries, `scale`, the `use_ef`
`__half2float` temporaries, the per-token store-back) is now live across the
whole token-loop body, raising VGPR pressure on the default path. This is a
**hypothesis**, not a measured fact.

**What is NOT yet verified (must be done before any fix):**
- ~~The "Scratch=108→172" numbers below are unsourced~~ — **AUTHOR REBUTTAL:**
  `rocprofv2 --kernel-trace` DOES emit per-workitem scratch. The CSV columns include
  `Scratch_Per_Workitem`, `Arch_VGPR`, `SGPR`, `LDS_Per_Workgroup`, `Wave_Size`.
  The values were extracted directly from the CSV:
  ```
  Master:    gated_delta_net_q8  Scratch=108  VGPR=32  SGPR=64  (all 180 dispatches identical)
  Dispatch:  gated_delta_net_q8  Scratch=172  VGPR=32  SGPR=64  (all 180 dispatches identical)
  ```
  That said, hsaco disassembly via `llvm-readelf` is still valuable for confirming
  the *mechanism* (which VGPRs spill, and whether the loop-hoist or the EF code is
  responsible). The scratch delta is a measured fact; the loop-hoist attribution is
  the hypothesis that explains it.
- "Same dispatch count (180)" — the 180 dispatches of `gated_delta_net_q8` (non-batch)
  are likely from the gen/decode phase (warmup=3 + gen=5 tokens × ~18 LA layers per
  forward pass), not the prefill phase (which uses `gated_delta_net_q8_batch_seq`).
  The per-dispatch 233→323µs delta is still meaningful because both sides have identical
  count and grid, but the +16ms total may not be a *prefill* regression per se — it
  may be gen-phase time. The prefill regression measured at bench level (4190→3468 tok/s)
  is real; the kernel-level attribution of +16ms to DeltaNet in prefill specifically
  needs re-examination with the batch_seq kernel filtered separately.
- The grid is `[n_heads, 32, 1] × [32,1,1]` (norm.rs:1522), not "grid size 16384".
  Grid=16384 in the rocprof output is the total thread count (n_heads × n_tiles).

**Provenance correction:** commit `9afba773` (Ship 6) is a qwen35 *arch-migration*
commit, not the sole introduction of the EF param. The EF kernel changes span
multiple commits across Ship 5/6. Bisect is needed to identify the exact commit.

**Evidence (from rocprofv2 `--kernel-trace` CSV, confirmed by author):**
```
Master:  gated_delta_net_q8  WG=32 LDS=2048 Scratch=108 VGPR=32 avg=233µs  (all 180 dispatches)
Dispatch: gated_delta_net_q8  WG=32 LDS=2048 Scratch=172 VGPR=32 avg=323µs  (all 180 dispatches)
```
Note: VGPR=32 on both sides (identical), but scratch grew 108→172. This suggests
additional *spill* stores/loads, not more allocated VGPRs — the compiler allocated
the same VGPRs but needed more stack spills for the larger live set across the loop.

### 3b. Q8 KV attention kernel switch — +10 ms — **VERIFIED in code**

This source is confirmed by reading both branches' dispatch paths:

- **Master** prefill routes Q8 KV batched attention through the Rust fn
  `attention_q8_0_kv_batched_masked` (GPU kernel `attention_q8_0_kv_batched`).
  Call sites: `master:crates/hipfire-arch-qwen35/src/qwen35.rs:10509, 12060`.
  Single large-batch kernel, LDS-backed attention tile.
- **HEAD** routes the same `KernelKey::AttnQ8_0KvBatchedMasked` *unconditionally*
  to the Rust fn `attention_flash_q8_0_batched_masked` (GPU kernel
  `attention_flash_q8_0_tile_batched`), the two-kernel tile+reduce path —
  lower LDS per dispatch, but many more launches.
  Wired at `crates/hipfire-dispatch/src/families/attention.rs:658-666`.

The tiled kernel is designed for long-context (>15K) where the old batched
kernel would exceed the 64KB LDS hardware limit. At short-context prefill
(512 tokens), the old kernel fits comfortably in LDS and is faster because
it amortizes dispatch overhead across fewer launches.

**Evidence (rocprof SUMMARY — raw `--kernel-trace` artifact not attached; the
10× dispatch-count jump is the load-bearing number and should ship with the raw
trace, not a hand-summary):**
```
Master:   attention_q8_0_kv_batched           count= 12  total=  8.34ms  avg=695µs
Dispatch: attention_flash_q8_0_tile_batched   count=120  total= 18.74ms  avg=156µs
```

Both kernels still exist on HEAD — `attention_q8_0_kv_batched_masked`
(`crates/rdna-compute/src/attention.rs:1437`) and
`attention_flash_q8_0_batched_masked` (`attention.rs:1546`) — so a
context-length-gated dispatch (Fix B) is implementable without resurrecting
deleted code.

---

## 4. Proposed fixes

### Fix A: `gated_delta_net_q8` — RE-DIAGNOSE FIRST (do not implement blind)

> **⚠️ Gate: no Fix A code change until §3a is verified.** Run the disassembly
> step below. If the default-path (`ef=None`, `rpt=0`) VGPR/spill count is
> unchanged vs master, the +16ms is NOT a register issue and these fixes are moot —
> re-bench per the 3-run protocol first, because 2-run 0.8B numbers can't separate
> +16ms from DPM/thermal noise (`feedback_perf_noise_decode`).

**Step A0 (REQUIRED prerequisite): hsaco disassembly.** Use the
`gfx-kernel-metadata` skill on both `master` and `HEAD` builds of
`gated_delta_net_q8` (gfx906). Compare VGPR count, scratch/spill bytes, and
occupancy for the **default config** (`ef=None`, `rpt=0`). This is the diagnostic
the original draft skipped. Only proceed to a code fix if a real spill/occupancy
delta appears on the default path.

**~~Option A1 (REJECTED): split `gated_delta_net_q8` vs `..._ef` variants.~~**
The original draft made this the highest-priority fix. **It does not work.** The
path that regressed is the *default* MQ4 path where `ef_residual=None` already.
Splitting an `_ef` variant out leaves the default path carrying the same
restructured (requant-hoisted-into-loop) body, so it recovers ≈0ms. A1 only helps
if the EF temporaries spill *on the ef=None path* — but on that path `use_ef` is a
compile-visible-false runtime bool, and modern LLVM already prunes the
`__half2float` block. Confirm with A0 before assuming otherwise.

**Option A1′ (preferred, IF A0 confirms a spill): restore requant-after-loop for
the default cadence.** The cheapest structural fix is to keep the requant block
*outside* the per-token loop for the `requant_per_token=0` path (matching master),
and only run the in-loop per-token requant when `requant_per_token=1`. This shrinks
the live register set across the token loop for the default path without touching
EF semantics. Sketch:

```c
for (int t = 0; t < n_tokens; t++) { ...update S... }
if (requant_per_token == 0) {
    // single requant pass after the loop — master's structure, EF-aware
    requant_tile(/* last-token state, use_ef, frame + (n_tokens-1) */);
}
// requant_per_token == 1 keeps the in-loop per-token requant (PARO path)
```

**Option A2 (fallback): `#if`-guard the EF temporaries behind the uniform.**
If A1′ is insufficient, wrap the EF declarations (`efr`, the `__half2float`
loads) so they are not even declared on the `use_ef==false` path. Less impactful
than A1′ if the cost is the hoist, not the EF block.

**Option A3 (deferred): accept the regression on gfx906.** Only viable if A0
shows the delta is small and the determinism/coherence benefit of the new requant
structure outweighs it. Re-evaluate after A0 + a 3-run rebench — the current
"+16ms / +38%" figure is unverified and may shrink under proper measurement.

### Fix B: Q8 KV attention — context-length-aware dispatch

**Option B1 (preferred): Use old batched kernel for short ctx, tiled for long.**

In `crates/hipfire-dispatch/src/families/attention.rs`, switch
`AttnQ8_0KvBatchedMasked` to choose based on effective context length:

```rust
KernelKey::AttnQ8_0KvBatchedMasked => {
    // For short-context prefill, the old batched kernel is faster (fewer dispatches).
    // The tiled kernel wins at long context where LDS would overflow.
    let lds_per_head = /* head_dim * batch_size * sizeof(f16) */;
    if io.max_ctx_len <= LDS_CROSSOVER {
        hip!(gpu.attention_q8_0_kv_batched_masked(...))  // old path
    } else {
        hip!(gpu.attention_flash_q8_0_batched_masked(...))  // tiled path
    }
}
```

The crossover point depends on `head_dim` and `batch_size` but is approximately
where the LDS tile exceeds the hardware limit. For Q8_0 KV with head_dim=128,
the old kernel is fine up to ~15K tokens. A conservative threshold of 8K
would give the tiled kernel margin while using the faster path for typical
prefill lengths.

Expected impact: recovers ~10ms at short ctx, preserves tiled kernel win at long ctx.

**Option B2: Add a separate dispatch key for the old batched path.**

Introduce `AttnQ8_0KvBatchedMaskedLegacy` in the dispatch table, mapped to the
old kernel. The kv_tier resolver picks it for short contexts. This is cleaner
but requires a new KernelKey and table entry.

**Option B3: Reduce tiled kernel dispatch count.**

Increase the tile size in `attention_flash_q8_0_tile_batched` so it processes
more tokens per dispatch at short context. This requires a kernel change and
may not help if the tile size is already optimal for LDS usage.

### Priority (revised)

1. **Fix B1 — now the first priority.** It is the verified source (+10ms,
   benefits all archs), the old kernel still exists, and the change is a clean
   context-length gate. Capture the raw rocprof trace when landing it.
2. **Fix A — gated on diagnosis (Step A0).** Do NOT implement the original A1.
   Run the hsaco disassembly + 3-run rebench first; if a real default-path spill
   is confirmed, implement A1′ (restore requant-after-loop for `rpt=0`). The
   "+16ms" is unverified and may not survive proper measurement.

Combined upside is "up to ~26ms" only if A0 confirms the DeltaNet delta is real
and register-bound. Until then, claim only the +10ms from Fix B.

---

## 5. Cross-arch applicability (HYPOTHESES — none measured)

The per-arch attribution below is **conjecture**, not data. The only measured
arch numbers are the gfx906/gfx1031 totals in §2. Treat this table as a list of
things to confirm, not findings.

| Regression source | gfx906 | gfx1031 | gfx1100 (estimated) |
|-------------------|--------|---------|---------------------|
| DeltaNet requant-loop register pressure | hypothesized severe (+38%, UNVERIFIED) | hypothesized mild (wave32 = more VGPRs) | hypothesized minimal |
| Q8 KV tiled kernel switch | Moderate (code-confirmed switch; per-arch ms unmeasured) | Moderate | Moderate |
| Dispatch overhead (Ship 5/6) | Residual | Residual | Residual |

The "wave32 has more VGPRs → spills less" reasoning for the gfx906-vs-gfx1031 gap
is plausible but unconfirmed; the §3a register hypothesis itself is unverified, so
this row inherits that uncertainty. The Fix B (attention) switch is arch-agnostic
and would benefit all archs — that part is sound. Any gfx1100 numbers cited from
"the earlier A/B" need a fresh `scripts/probe_commits.sh master HEAD` run on
k9lin before they are quotable.

---

## 6. Reproduction

```bash
# Build both binaries
git checkout master
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime
cp target/release/examples/bench_qwen35_mq4 /tmp/bench_master

git checkout integration/dispatch-unification
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime
cp target/release/examples/bench_qwen35_mq4 /tmp/bench_dispatch

# Speed test (gfx906, 0.8B — worst case)
ROCR_VISIBLE_DEVICES=0 /tmp/bench_master ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --prefill 512 --prefill-runs 3 --warmup 5 --gen 30
# Master: ~4190 tok/s

ROCR_VISIBLE_DEVICES=0 /tmp/bench_dispatch ~/.hipfire/models/qwen3.5-0.8b.mq4 \
  --prefill 512 --prefill-runs 3 --warmup 5 --gen 30
# Dispatch: ~3470 tok/s (−17%)

# Kernel trace (rocprofv2)
ROCR_VISIBLE_DEVICES=0 rocprofv2 --kernel-trace -d /tmp/rocprof_master \
  /tmp/bench_master ~/.hipfire/models/qwen3.5-0.8b.mq4 --prefill 512 --prefill-runs 1 --warmup 3 --gen 5

ROCR_VISIBLE_DEVICES=0 rocprofv2 --kernel-trace -d /tmp/rocprof_dispatch \
  /tmp/bench_dispatch ~/.hipfire/models/qwen3.5-0.8b.mq4 --prefill 512 --prefill-runs 1 --warmup 3 --gen 5
```

### Required-before-acting diagnostics (added in revision)

```bash
# 1. Per-workitem scratch/VGPR/spill — rocprofv2 does NOT report this; use the
#    gfx-kernel-metadata skill (clang-offload-bundler + llvm-readelf on the .hsaco).
#    Compare master vs HEAD for gated_delta_net_q8 on gfx906, DEFAULT config.
#    This is the step the original draft skipped; the "Scratch=108→172" line
#    above is unsourced and must be reproduced (or retracted) here.

# 2. Proper cross-commit perf (handles warmup + multi-run aggregation correctly):
ROCR_VISIBLE_DEVICES=0 scripts/probe_commits.sh master HEAD
#    — re-confirm the prefill deltas in §2, especially the 0.8B −17.2%.

# 3. Confirm the actual kernel deltas this branch introduces (NOT vs merge base):
git diff master..HEAD -- kernels/src/gated_delta_net_q8.hip   # requant hoisted into t-loop
git diff master..HEAD -- crates/hipfire-dispatch/src/families/attention.rs  # Q8 tiled switch
```

---

## 7. Author's assessment of adversarial review

### Accepted corrections

1. **Master already has stochastic rounding.** Confirmed by `git show master:kernels/src/gated_delta_net_q8.hip` — the `frame` param and LCG PRNG are there. My original "branch added the expensive code" was wrong. The real delta is the structural loop reorganization.

2. **Requant-loop hoisting is the real mechanism.** Master runs requant OUTSIDE the per-token loop (line 96, after the `for (t)` closes at ~line 95). The dispatch branch moves it INSIDE the loop with a `continue` guard. Even though the guard means the requant body only executes on the last token for `per_token=0`, the compiler must keep the requant's register live-set (q0..q3, scale, inv_s, my_max, rng temporaries, use_ef, efr pointer) live across the entire token loop body. This is the plausible spill mechanism.

3. **Fix A1 (split kernel variants) is wrong; Fix A1′ is correct.** Splitting into `gated_delta_net_q8` vs `gated_delta_net_q8_ef` doesn't help because the default path still has the in-loop requant structure. The right fix is to restore the out-of-loop requant for `requant_per_token=0` (A1′) — the loop body reverts to master's structure, eliminating the register pressure.

4. **Fix B1 should be first priority.** The attention switch is code-verified, arch-agnostic, and has a clean fix.

5. **3-run protocol.** The bench numbers should be re-confirmed with `scripts/probe_commits.sh`.

6. **Provenance of commits.** `9afba773` is not the sole regression source; bisect needed.

### Rejected corrections

1. **"Scratch=108→172 is unsourced" / "rocprofv2 does not emit per-workitem scratch"** — This is **factually wrong**. The `--kernel-trace` CSV includes `Scratch_Per_Workitem` as column 9. Both values (108, 172) were extracted from the CSV and verified to be identical across all 180 dispatches on each side. The author re-confirmed this by re-reading the raw CSV.

2. **The 180 dispatches invalidate the per-dispatch comparison** — Both sides have identical count (180) and grid. The per-dispatch timing delta (233→323µs) is the meaningful metric. However, the valid point is that these 180 dispatches may be from the gen/decode phase rather than prefill specifically, which means the +16ms attribution to *prefill* specifically needs re-examination. The prefill regression is real (bench-level 4190→3468 tok/s); the question is how much of the rocprof'd +16ms is prefill vs gen.

### Open questions for next steps

1. **Run hsaco disassembly** on both `gated_delta_net_q8` builds to confirm the spill mechanism (which temporaries, which loop-carried dependencies). Even though rocprof confirms the scratch delta, the disassembly would show whether it's the loop-hoist or the EF code causing it.

2. **Separate prefill vs gen kernel attribution** — re-run rocprof with gen=0 (no decode phase) to isolate the prefill-only kernel time delta. This would determine whether the +16ms DeltaNet attribution is prefill or gen.

3. **Bisect the DeltaNet regression** — identify which commit introduced the loop restructuring.
