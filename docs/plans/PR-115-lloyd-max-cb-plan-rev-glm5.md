# Adversarial Review: PR-115 Lloyd-Max Codebook Plan

**Reviewer:** glm-5 (adversarial)
**Date:** 2026-05-06
**Subject:** `docs/plans/PR-115-lload-max-codebooks-mq3.md`

---

## Verdict: Plan is mostly sound but has 7 issues, 2 of which are blocking.

---

## BLOCKING ISSUES

### B1. HFQ3 tail has the same bug the plan warns about — "mirror byte-for-byte" inherits it

The plan says (line 39): "Single-accumulator drift breaks coherence on long
generations whenever `groups_per_row` isn't a multiple of 4" and instructs the
implementer to "mirror HFQ3 file lines 76-120 byte-for-byte modulo lookup
substitution."

The HFQ3 reference kernel (`gemv_hfq3g256.gfx1100.hip:90-98`) has exactly this
bug. Its tail block uses `TAIL_DOG3(pk, sc, zp, base, acc0)` for `tail >= 1`,
`acc1` for `tail >= 2`, `acc2` for `tail >= 3`. The comment on line 76 says
"must accumulate into their own accumulator (acc[g % 4])" but the actual code
uses *fixed* acc0/acc1/acc2 for tail positions 0/1/2 regardless of the quad
alignment. When `groups_per_row = 5` (quads=1, tail=1), the tail group should
go into `acc[1 % 4] = acc1`, not `acc0`.

The plan inherits this bug by telling the implementer to mirror the HFQ3 tail
code. The `acc[g % 4]` comment is aspirational, not actual.

**Fix:** Either (a) fix the HFQ3 tail first in a separate commit and then mirror
the *fixed* version, or (b) note in the plan that the HFQ3 reference has a
latent tail bug on certain model dimensions and the Lloyd port must compute
`g % 4` correctly at tail entry (not blindly use acc0/acc1/acc2 by ordinal).

### B2. `gemv_mq3g256_lloyd` dispatch uses raw `self.hip.launch_kernel` — no graph-capture safety

`dispatch.rs:2073` calls `self.hip.launch_kernel(...)` directly instead of
`self.launch_maybe_blob(...)`. This is exactly the pattern CLAUDE.md §6 pitfall
table warns about: "Dangling stack-pointer kernargs from raw `self.hip.launch_kernel`
calls [...] captured pointers dangle past `end_graph_capture`."

The HFQ3 dispatch at `dispatch.rs:2626` already uses `launch_maybe_blob`. The
plan's §4 step 3 says to "replace the hard-coded `ensure_kernel(...)`" but
doesn't mention migrating the launch call. If this kernel ever runs under
`HIPFIRE_GRAPH=1` (and the forward-scratch-layers path uses it), it will
produce plausible tok/s numbers with garbage output — the exact silent-
corruption scenario from CLAUDE.md.

**Fix:** Add a step 3b to the plan: migrate the launch call in
`gemv_mq3g256_lloyd` from `self.hip.launch_kernel` to `self.launch_maybe_blob`
with a `KernargBlob` closure, matching the pattern at `dispatch.rs:2626-2640`.

---

## HIGH-SEVERITY ISSUES

### H1. Arch selector is too narrow — misses gfx115x and gfx12

The HFQ3 `gemv_hfq3g256_for_arch` at `kernels.rs:451-458` matches only
`gfx1100 | gfx1101 | gfx1102`. The plan proposes to mirror this exactly for
the Lloyd variant. But AGENTS.md §v0.1.9-alpha says MQ3 is production on
gfx1100/1101/1102/1150/1151/1151 and gfx1200/1201. The HFQ3 arch selector
itself is already stale — it doesn't include gfx115x or gfx12. Mirroring a
stale selector propagates the gap.

gfx115x (RDNA 3.5 / Strix Halo) has the same wave32 WMMA and LDS bank width
as gfx1100. The K4 unroll + LDS codebook pattern should work identically.
gfx12 has `_w32_gfx12` WMMA but the GEMV path (not WMMA) uses the same
scalar ops and LDS layout — likely portable without changes.

**Fix:** Extend the arch match to `"gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1152"`. Note gfx12 separately as a candidate for the same kernel (no WMMA in GEMV, so no builtin swap needed). Alternatively, file a follow-up to fix the HFQ3 selector first and inherit the corrected set.

### H2. LDS bank conflict analysis is wrong — fp16 doesn't cause 4-way conflicts

The plan (lines 47-50) claims that at fp16 (2 B), "8 entries map to 2 banks →
up to 4-way bank conflict when 32 threads read different `q`." This is
incorrect.

RDNA3 LDS is 32 banks × 4 bytes/bank. A 2-byte fp16 address `base + 2*i`
hits bank `(base + 2*i) / 4 = (base/4) + floor(i/2)`. So 8 consecutive fp16
entries span 4 banks, and 32 threads each reading a different `q` from the
same group hit the *same* 4 banks — 8-way per bank. At fp32, 8 entries span 8
banks — 4-way per bank (32 threads / 8 banks). Both are worse than the plan
claims.

However, this doesn't change the *conclusion* — fp32 is still strictly better
than fp16 (4-way vs 8-way conflict, and 128 B fp32 is trivial on 64 KB LDS).
The recommendation is correct but the arithmetic justifying it is wrong.

**Fix:** Correct the bank-conflict paragraph. At fp16: 8 entries → 4 unique
banks → 8 threads/bank worst case. At fp32: 8 entries → 8 unique banks →
4 threads/bank worst case. Or just delete the specific conflict numbers and
keep the "fp32 is strictly better, 128 B is nothing" conclusion.

---

## MEDIUM-SEVERITY ISSUES

### M1. Validation order runs `coherence-gate.sh` — should be `coherence-gate-dflash.sh` for DFlash models

Step 3 (line 99) says to run `./scripts/coherence-gate.sh`. But the test
target is 9B MQ3 which, per the plan's own goal (line 5), targets the DFlash
decode-perf path. AGENTS.md §3.5 and CLAUDE.md both state that any change to
kernels or dispatch must pass the coherence gate, and spec-decode/DFlash changes
must pass `coherence-gate-dflash.sh`. The plan's step 4 also benchmarks DFlash
decode.

The Lloyd kernel is a *GEMV* (decode-only) change, not a DFlash structural
change, so `coherence-gate.sh` is technically sufficient for the kernel
correctness gate. But if the perf gate (step 4) is measured via
`dflash_spec_demo`, the coherence gate should match — run `coherence-gate-dflash.sh`
to catch the token-attractor class of bugs that `coherence-gate.sh` doesn't
test for.

**Fix:** Run both gates. Step 3: `coherence-gate.sh` (kernel correctness).
Add step 3b: `coherence-gate-dflash.sh` (DFlash decode path correctness,
if a draft model is available for the Lloyd-quantized target).

### M2. Coherence-gate.sh does not auto-run on kernel change — plan's claim is wrong

Line 100: "Pre-commit hook runs it automatically on kernel change." The
`.githooks/pre-commit` file exists (per CLAUDE.md §Coherence Gate setup), but
the plan should verify it actually triggers for *new* kernel files in
`kernels/src/`. If the hook uses a path-based glob that doesn't include the
new `gemv_mq3g256_lloyd.gfx1100.hip`, the gate won't fire. More importantly,
`cargo check` (step 1) doesn't run the hook — only `git commit` does. The
validation order implies the gate is a hard gate between steps 2 and 4, but
it only fires on commit, not on build.

**Fix:** Explicitly run `./scripts/coherence-gate.sh` as a manual step. Don't
rely on the pre-commit hook for validation ordering.

---

## LOW-SEVERITY / NIT ISSUES

### L1. `__syncthreads()` in single-wave workgroup is a CUDA-ism — use `__syncthreads()` or `barrier()` but document why

Line 53: "single-wave so an `s_waitcnt lgkmcnt(0)` suffices, but keep the
barrier for readability." This is fine, but `__syncthreads()` is a CUDA alias
that HIP supports. On AMD, `barrier()` is the native intrinsic. Both compile
to the same ISA on single-wave workgroups. Not a bug, but worth a comment in
the new kernel file noting this is intentional HIP compatibility.

### L2. "32 threads × 1 fp16 each = exactly the 32 codebook entries for the 4 groups"

Line 51: This math checks out (4 groups × 8 entries = 32, 32 threads each
load 1). But the plan doesn't specify *which* thread loads *which* entry.
The cooperative load pattern needs to be explicit: thread `tid` loads
`cb_h[tid]` from group `tid / 8`, which means each thread loads from a
*different group pointer*. The 4 group pointers are already computed in the
quad unroll, so this is natural, but the plan should say it explicitly to
avoid a "all 32 threads load from gp0" bug.

### L3. Module name `gemv_mq3g256_lloyd_rdna3` is inconsistent with the existing naming

The HFQ3 module is named `gemv_hfq3g256_rdna3` but the kernel function is
`gemv_hfq3g256` (not `gemv_hfq3g256_rdna3`). The plan proposes module
`gemv_mq3g256_lloyd_rdna3` with presumably function name
`gemv_mq3g256_lloyd`. This follows the existing pattern (module != function
name for arch variants). But the plan should state the function name
explicitly — the `ensure_kernel` call in dispatch.rs needs it.

### L4. Quality number "9B Lloyd-MQ3 ppl 18.52 vs MQ4 10.34 — 1.79× MQ4" is alarming

Line 12-13: 18.52 ppl for a 3-bit format vs 10.34 for 4-bit is a massive
quality gap. The plan frames this as "the closest sub-4-bit format hipfire has
hit" but doesn't address whether 18.52 ppl is actually usable for anything
beyond a research demo. If the perf gate clears (≥120 tok/s) but quality
remains 1.79× worse than MQ4, is this shippable? The plan doesn't state the
quality acceptance criteria — only the perf gate.

---

## SUMMARY TABLE

| ID | Severity | Issue | Plan line(s) |
|----|----------|-------|--------------|
| B1 | BLOCKING | HFQ3 tail bug inherited via "mirror byte-for-byte" | 39, 116 |
| B2 | BLOCKING | Raw `launch_kernel` not migrated to `launch_maybe_blob` | 77-83, dispatch.rs:2073 |
| H1 | HIGH | Arch selector too narrow (misses gfx115x, gfx12) | 73-75 |
| H2 | HIGH | LDS bank conflict arithmetic is wrong (conclusion correct) | 47-50 |
| M1 | MEDIUM | Wrong coherence gate for DFlash validation path | 99-101 |
| M2 | MEDIUM | Pre-commit hook claim is misleading for new files | 100 |
| L1 | LOW | `__syncthreads()` CUDA-ism — document | 53 |
| L2 | LOW | Cooperative load thread→entry mapping unspecified | 51 |
| L3 | LOW | Function name not stated for `ensure_kernel` | 67-68 |
| L4 | LOW | 18.52 ppl quality gap not assessed for shipability | 12-13 |

---

## RECOMMENDED CHANGES TO THE PLAN

1. Fix the tail accumulator logic: compute `g % 4` at tail entry, don't
   blindly use acc0/acc1/acc2 by ordinal. Add a test case with
   `groups_per_row` not divisible by 4.
2. Add dispatch migration from `self.hip.launch_kernel` to
   `self.launch_maybe_blob` as an explicit step.
3. Widen arch match to include gfx115x (and note gfx12 candidacy).
4. Correct the LDS bank conflict paragraph or remove specific numbers.
5. Run `coherence-gate-dflash.sh` in addition to `coherence-gate.sh`.
6. Don't claim the pre-commit hook covers validation ordering — run the gate
   scripts explicitly.
