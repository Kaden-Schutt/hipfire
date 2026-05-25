# PR #115 — K4-unroll + LDS-resident codebook fix for Lloyd-MQ3 GEMV

**Branch:** `lloyd-max-mq3-spike`
**PR:** https://github.com/Kaden-Schutt/hipfire/pull/115
**Target:** clear the open decode-perf ship gate — 9B MQ3-Lloyd from 44 → ≥120 tok/s on gfx1100.
**Hardware:** gfx1100 (7900 XTX). Same arch the PR's tok/s table is measured on.
**Date:** 2026-05-06 (rev 2 — folded findings from gemini, glm5, and Claude adversarial reviews; see `PR-115-lloyd-max-cb-plan-rev-claude.md`).

## Goal

PR #115 lands Lloyd-Max fp16 codebooks for MQ3/MQ2. Quality result is real
(9B Lloyd-MQ3 ppl 18.52 vs MQ4 10.34 — 1.79× MQ4, the closest sub-4-bit
format hipfire has hit). The blocker is decode perf: the naive 8-way switch
in `gemv_mq3g256_lloyd.hip` regresses 9B decode 3.2× vs uniform MQ3
(44 vs 141 tok/s). Mirror the K4-unroll pattern from
`gemv_hfq3g256.gfx1100.hip` and replace the per-thread switch with an
LDS-resident codebook table.

## Root-cause recap (verified via disassembly, 2026-05-06)

`kernels/src/gemv_mq3g256_lloyd.hip:67-71` chains 7 ternaries to map
`q ∈ [0,8)` to `cbN`. Disassembly of the gfx1100 build (see
`benchmarks/results/devlog_20260506_lloyd_mq4_extension.md`,
"2026-05-06 cont. — Step 0a disassembly preflight") shows the
compiler emits a **divergent-execution decision tree**, not a
branchless cmp/cndmask chain or a vector LUT.

Per-group inner-body counts on gfx1100 with `-O3`:

| Instruction class | Count |
|---|---:|
| `v_cmpx_*` + `v_cmp_*` | 62 |
| `s_or_b32 exec_lo, ...` (EXEC restore) | 50 |
| `s_cbranch_execz` | 43 |
| `v_cndmask_b32` | 11 |
| `v_fmac_f32` + `v_dual_mul_f32` (useful) | 8 |

That's roughly 166 dispatch instructions for 8 useful FMAs per group
inner body — a **~21:1 overhead-to-work ratio**, worse than the plan's
original estimate.

Why: `q` is inherently divergent (every thread in a wave holds a
different packed-index byte triple). The compiler can't prove
wave-uniformity, can't use a `v_perm_b32` LUT for VGPR-resident
sources, and falls back to a binary-decision tree using
`v_cmpx_lt_i32 + s_cbranch_execz + s_or_b32 exec` for each of the 8
lookups. Combined with the single-accumulator no-ILP loop body (vs
HFQ3's 4 accumulators on this arch), the two together account for
the 3.2× gap.

**Implication for the LDS half of the rewrite:** dropping Change 2
(LDS staging) is NOT viable. K4 unroll alone removes the
single-accumulator stall but leaves the divergent-execution lookup
tree intact. Replacing the tree with `ds_read_b32` indexed by `q` is
where the bulk of the 2.7× perf gate has to come from. If K4-alone
clears the gate, that's a happy surprise; assume it won't.

## Two changes, one new kernel file

The K4 unroll (Change 1) and LDS staging (Change 2) attack different
mechanisms (single-accumulator dep chain vs cmp/cndmask tree). To
keep perf attribution bisectable, **prefer landing K4 first as one
commit and LDS second as a follow-up commit** in the same PR. Bundle
only if Change 1 alone falls short of the ≥120 tok/s gate.

### Change 1 — K4 unroll (mirror HFQ3 control flow exactly)

- 4 accumulators `acc0..acc3`, `quads = groups_per_row >> 2`,
  `tail = groups_per_row & 3`.
- Per quad: 4 group pointers `gp0..gp3 = row_ptr + (g+i) * 112`, load 4
  packed-uint24 indices `pk0..pk3`.
- Tail loop must accumulate into `acc[g % 4]` — *not* always `acc0`.
  HFQ3 file line 76 has the explanatory comment; same trap applies to
  Lloyd. Single-accumulator drift breaks coherence on long generations
  whenever `groups_per_row` isn't a multiple of 4. Note that under the
  invariant `quads*4 % 4 == 0`, HFQ3's literal `acc0/acc1/acc2` for
  tail iterations 0/1/2 **is** `acc[g % 4]` — both formulations are
  equivalent. Either spelling is fine; prefer the explicit
  `acc[(g) & 3]` index in code for clarity.

### Change 2 — LDS-resident codebook lookup

- Allocate `__shared__ float cb_lds[8 * 4]` (128 B per workgroup, 4
  groups × 8 entries, fp16 → f32 converted at load time so the GEMV
  body indexes float). **8-entry hardcoding is intentional for MQ3-
  Lloyd; MQ4-Lloyd extension** (`devlog_20260506_lloyd_mq4_extension.md`)
  **will need 16-entry layout** — consider parameterizing via
  `#define CB_ENTRIES 8` so the MQ4 follow-up is a header swap.
- **fp32 not fp16 in LDS.** RDNA LDS is 32 banks × 4 B/bank.
  - At fp16 (2 B): 8 entries span 4 banks → up to 8 threads/bank
    worst case (8-way conflict) when 32 threads pick distinct `q`s.
  - At fp32 (4 B): 8 entries span 8 banks → up to 4 threads/bank
    worst case (4-way conflict).
  - Same-`q` accesses broadcast (1 cycle), independent of fp width.
  - Conclusion: fp32 is strictly better; 128 B is nothing on 64 KB
    LDS.
- Cooperative load at top of each quad: 32 threads each load **1
  fp16** from one of the 4 codebooks. Explicit mapping:
  - thread `tid` reads `cb_h_gp[tid >> 3][tid & 7]` (group pointer
    `gp[tid / 8]`, codebook entry `tid % 8`)
  - converts via `__half2float`
  - writes `cb_lds[tid]`
  - The wave issues 4 distinct ~16 B reads from 4 cachelines that
    are 112 B apart (one per group). Not a single coalesced 64 B
    transaction, but RDNA3 services this fine. The structural win
    is in the lookup body, not the load.
- `__syncthreads()` after the load — single-wave so an
  `s_waitcnt lgkmcnt(0)` would suffice, but keep the barrier for
  readability and any future block-size change.
- Body: `acc0 += cb_lds[0*8 + q0] * x[base+0] + ... + cb_lds[0*8 + q7] * x[base+7]`,
  groups 1/2/3 → acc1/2/3 at LDS offsets 8/16/24.
- **Tail iterations use a per-group cooperative load** (NOT the
  4-group K4 load):
  - For each tail group present (1, 2, or 3 of them), only the first
    8 lanes load that group's 8 fp16 codebook entries; barrier;
    everyone reads `cb_lds[0..8]`.
  - Critically: do NOT execute the 4-group cooperative load in the
    tail path with only 1-3 groups present — `gp1/gp2/gp3` would
    point past the row's allocated bytes (true OOB).
  - Write the tail group accumulation into `acc[g & 3]`.

Net per-group inner work: 8 `ds_read_b32` (one per lookup, no switch) +
8 FMAs vs current ~10×8 cmp/cndmask + 8 FMAs. The order-of-magnitude
reduction in lookup overhead is what motivates the kernel rewrite;
the actual win must be measured cross-process, not predicted.

## File-level diff plan

1. **New file** `kernels/src/gemv_mq3g256_lloyd.gfx1100.hip` — K4 + LDS
   body. **Module name** `gemv_mq3g256_lloyd_rdna3` (parallels
   `gemv_hfq3g256_rdna3`). **Kernel function name** stays
   `gemv_mq3g256_lloyd` (unsuffixed), matching HFQ3's
   module-vs-function naming convention (see
   `kernels/src/gemv_hfq3g256.gfx1100.hip:13`). Preserve
   `__launch_bounds__(32, 16)`.
2. **Leave** the existing `kernels/src/gemv_mq3g256_lloyd.hip` as the
   gfx1010 / baseline fallback. It's slow but correct, and gfx1010
   isn't the perf target.
3. **`crates/rdna-compute/src/kernels.rs`**:
   - Add `pub const GEMV_MQ3G256_LLOYD_GFX1100_SRC: &str = include_str!(...)`.
   - Add `pub fn gemv_mq3g256_lloyd_for_arch(arch: &str) -> (&'static str, &'static str)`
     mirroring `gemv_hfq3g256_for_arch` (lines 451-458). Match
     `"gfx1100" | "gfx1101" | "gfx1102"` initially. (Widening to
     gfx115x / gfx12 is a separate commit that should also widen
     HFQ3 — out of scope for this PR.)
4. **`crates/rdna-compute/src/dispatch.rs:2063-2074`**:
   - Replace the hard-coded
     `ensure_kernel("gemv_mq3g256_lloyd", GEMV_MQ3G256_LLOYD_SRC, "gemv_mq3g256_lloyd")`
     with the arch-selector pair, using `self.arch` (the field used
     by `gemv_hfq3g256_for_arch` at `dispatch.rs:2612`).
   - `gemv_mq3g256_lloyd_with_rotate` is unchanged (calls the new
     dispatcher).
4b. **Graph-capture safety migration (`dispatch.rs:2063-2083`):**
    Migrate `gemv_mq3g256_lloyd` from raw `self.hip.launch_kernel` to
    `self.launch_maybe_blob` with a `KernargBlob` closure, mirroring
    the HFQ3 pattern at `dispatch.rs:2626-2640`. The bench harness
    runs under `HIPFIRE_GRAPH=1`; CLAUDE.md is explicit that
    stack-resident `Vec<*mut c_void>` kernargs dangle past
    `end_graph_capture` → silent corruption.
    - Apply the same migration to `gemv_mq2g256_lloyd` for
      consistency with the existing Lloyd surface (it shares the
      raw-launch pattern).
5. **MQ2-Lloyd kernel rewrite:** out of scope. PR keeps MQ2 research-
   only because bit-width (not codebook) is binding at 2 bpw.
   (4b's launch-blob migration above is independent of this — it's
   a graph-safety fix, not a perf rewrite.)

## Validation order

Run in this order — each step gates the next. Steps that need a
Lloyd model must export `HIPFIRE_ALLOW_MQ3_LLOYD=1` (the runtime/
quantizer guard, see `crates/hipfire-quantize/src/main.rs:2011-2013`;
verify also at runtime load).

### Step 0 — Bench harness exists

`scripts/probe_commits.sh` is hardcoded to `bench_qwen35_mq4`. Before
any of the perf gate is meaningful, **either**:

- (a) Add `crates/hipfire-runtime/examples/bench_qwen35_mq3_lloyd.rs`
  mirroring `bench_qwen35_mq4.rs`, point it at
  `~/.hipfire/models/qwen3.5-9b.mq3-lloyd`, and teach
  `probe_commits.sh` to take an example name (or fork
  `probe_commits_mq3_lloyd.sh`).
- (b) Verify that `bench_qwen35_mq4.rs` auto-detects the dtype from
  the loaded weights — if so, point it at the `.mq3-lloyd` file
  directly. (Read the example to confirm before relying on this.)

Commit the bench prompt as a file (NOT a heredoc inside the script);
log its md5 alongside any reported tok/s number per the CLAUDE.md
prompt-md5 rule.

### Step 0a — Disassemble the existing kernel (DONE 2026-05-06)

Compiled the existing kernel for gfx1100 with `--save-temps` and
inspected the inner-loop assembly. **Bottleneck attribution
verified** — the compiler emits a divergent-execution decision tree,
not a vector LUT. See `benchmarks/results/devlog_20260506_lloyd_mq4_extension.md`
"Step 0a disassembly preflight" for the full instruction-class
count + structural pattern. Both K4 and LDS halves of the rewrite
are justified.

To reproduce:

```
mkdir -p /tmp/lloyd_disasm && cd /tmp/lloyd_disasm
hipcc -O3 --offload-arch=gfx1100 -c \
  /path/to/kernels/src/gemv_mq3g256_lloyd.hip --save-temps
# inspect: gemv_mq3g256_lloyd-hip-amdgcn-amd-amdhsa-gfx1100.s
```

### Step 1 — Build clean

`cargo check -p rdna-compute -p hipfire-runtime`.

### Step 2 — Correctness

Two anchors, both required:

a. **4B Lloyd-MQ3 ppl reproduction** (primary numerical anchor):
   run the perplexity harness on Qwen 3.5+ 4B Lloyd-MQ3, expect
   ppl ≈ **22.56 ±0.3** (PR's table). 4B is the smallest size where
   ppl is a real quality signal — 0.8B (ppl=155.22) is text-collapsed
   and will pass-by-luck or fail-by-flutter for subtle bugs.
b. **Logits-Δ vs the existing kernel:** run a single short-prompt
   forward pass through the OLD kernel (`gemv_mq3g256_lloyd.hip`)
   and the NEW gfx1100 kernel against the same model + prompt; diff
   the logit vector. Expect max-abs Δ within fp32-summation-order
   noise (~1e-5 to 1e-3 for 9B). A drift > 1e-2 indicates an
   indexing / tail / fp16-conversion bug that ppl would smooth over.
   The K4 split changes summation order, so bit-identity is not the
   gate — magnitude is.

If 2a passes but 2b shows large drift: bug in tail-group accumulator
rotation, fp16→f32 LDS conversion order, or cb_lds indexing.

### Step 2.5 — VGPR budget check

Dump `--save-temps` for the new kernel; read the `.s` and confirm
`.vgpr_count` ≤ HFQ3's count under the same `__launch_bounds__(32,
16)`. A spill into LDS-spill-VGPR area would regress the kernel
instead of helping. If VGPR count is over budget, drop unused
intermediates or split the body to reduce live ranges before
proceeding.

### Step 2.6 — Tail K-sweep correctness

Qwen 3.5+ projection K values may all land on multiples of 1024 (=
4-quad clean), in which case the tail path never executes during
production inference. Add a standalone test
(`crates/rdna-compute/tests/gemv_mq3g256_lloyd_tail.rs` or similar)
that calls the kernel directly with `groups_per_row ∈ {4, 5, 6, 7,
8}` (clean + each tail size + clean+1 quad). Compare results against
the existing baseline kernel. fp32 reconstruction should match within
summation-order noise.

### Step 3 — Coherence-gate

Run explicitly:

```
HIPFIRE_ALLOW_MQ3_LLOYD=1 ./scripts/coherence-gate.sh
```

(specify the Lloyd model via whatever the gate's model selection
mechanism is; verify before running). Watch the report for the
standard 4-prompt battery — same battery the PR's gate 2 references.

The pre-commit hook in `.githooks/pre-commit` runs the gate on `git
commit` for staged kernel changes. Treat this as a backup catch, NOT
as part of the validation ordering — the hook fires on commit, not
on `cargo check`, so it can't gate steps 1-2 by itself.

### Step 4 — Cross-process perf gate (the ship gate)

`scripts/probe_commits.sh <baseline> HEAD` against the bench example
landed in Step 0. Per CLAUDE.md, within-session A/B has ±10–15%
drift — DO NOT use one-shell numbers. Target: **≥120 tok/s** on 9B
Lloyd-MQ3 decode. Use byte-identical prompts (CLAUDE.md prompt-md5
rule); record the prompt md5 alongside the result.

If the K4-only commit clears ≥120, stop there. Layer the LDS commit
only if it doesn't.

### Step 5 — Larger-model ppl reproduction

9B Lloyd-MQ3 ppl reproduction within ±0.5 of PR table (18.52). This
is the PR's quality eyeball — should already follow from Step 2a +
Step 2b passing, but a full 9B run is the closing sanity check.

## Risks and watch-items

- **Tail accumulator bug.** Easy to write `acc0 +=` in the tail and
  ship a kernel that's correct on `groups_per_row % 4 == 0` and
  silently wrong on long generations otherwise. Mirror HFQ3's K4 +
  tail control-flow structure (lines 76-120); substitute the 16 B
  fp16 codebook header parse + LDS cooperative stage + `cb_lds[q]`
  lookup for HFQ3's 8 B (scale, zero) parse + affine recon. Do
  **not** treat the port as byte-for-byte: the per-group offset
  shifts (HFQ3 data starts at gptr+8, Lloyd data starts at gptr+16)
  and the codebook header is 2× the size.
- **Tail OOB on K4 cooperative load.** Reusing the 4-group
  cooperative load for tail iterations would read `gp1/gp2/gp3`
  past the row's allocated bytes when only 1-3 tail groups exist.
  Use a per-group load in the tail path (only first 8 lanes read,
  barrier, everyone indexes).
- **LDS bank conflict at fp16 — 8-way, not 4-way.** Storing fp16 in
  LDS to save 64 B is worse than the original framing implied: 8
  entries span 4 banks → up to 8 threads/bank. fp32 (4-way worst
  case) is strictly better. Don't save the 64 B.
- **Single-wave barrier elision.** `__syncthreads()` on a 32-thread
  block compiles to little on RDNA wave32, but keep it for
  readability and any future block-size change.
- **VGPR pressure under K4.** 4 accumulators + 4 group pointers + 4
  packed indices + LDS staging temps push VGPR count up vs the
  existing single-accumulator kernel. Step 2.5 validates the budget.
- **Graph-capture corruption** (CLAUDE.md). The existing dispatch at
  `dispatch.rs:2073` uses raw `self.hip.launch_kernel` with
  stack-resident kernargs. Under `HIPFIRE_GRAPH=1` (which the bench
  harness exports) those pointers dangle past `end_graph_capture`,
  yielding plausible tok/s with garbage output. Step 4b above fixes
  this for the touched dispatch arms.
- **Stream / `active_stream` gotcha** (CLAUDE.md): not relevant here.
  That note is about the memset helper, not GEMV dispatch.
- **WMMA prefill kernel for Lloyd:** not in scope for this gate. PR
  only ships GEMV; prefill goes through the dense path or a separate
  WMMA kernel that doesn't yet exist for the Lloyd variants. Defer to
  a later PR.
- **Codebook prefetching** (future opt). Loading the next quad's
  codebook into registers during the current quad's FMAs, then
  staging into LDS at the next quad's barrier, hides the global-load
  latency. Out of scope here; revisit after Change 2 lands.

## What this plan does NOT do

- No MQ2-Lloyd **kernel rewrite** (research-only, no perf gate). The
  4b launch-blob migration above DOES touch `gemv_mq2g256_lloyd` for
  graph safety — that's a one-line dispatch change, not a kernel
  rewrite.
- No residual variant for Lloyd-MQ3 (PR doesn't ship one; defer until
  a use site appears). The format is `--allow-mq3-lloyd`-gated and
  has no production caller using a residual GEMV; adding the variant
  speculatively is scope creep.
- No rewrite of the generic baseline kernel — it's the gfx1010
  fallback and works.
- No widening of the arch selector beyond gfx1100/1101/1102.
  Extending to gfx115x (RDNA 3.5) and gfx12xx (RDNA 4) is a separate
  commit that should also widen the HFQ3 selector, not part of this
  PR.
- No removal of the `--allow-mq3-lloyd` guard — that's a separate
  commit per the PR's test plan, after both gates clear.

## Quality acceptance criteria

The format ships `--allow-mq3-lloyd`-gated, research-only. The quality
gate is "don't regress vs PR's published 18.52 ppl on 9B (within
noise)" — NOT "is 18.52 production-shippable" (the latter is out of
scope for this PR; production gating is decided when/if the
`--allow-mq3-lloyd` guard is removed in a future commit).

## References

- PR: https://github.com/Kaden-Schutt/hipfire/pull/115
- `kernels/src/gemv_mq3g256_lloyd.hip` — current generic Lloyd kernel
  (slow switch dispatch, baseline fallback).
- `kernels/src/gemv_hfq3g256.gfx1100.hip` — K4-unroll reference; mirror
  its 4-accumulator + tail-rotation structure (control flow only,
  not byte-for-byte; header semantics differ).
- `crates/rdna-compute/src/kernels.rs:451-458` — `gemv_hfq3g256_for_arch`
  selector; mirror for `gemv_mq3g256_lloyd_for_arch`.
- `crates/rdna-compute/src/dispatch.rs:2063-2083` — current Lloyd-MQ3
  dispatch path to retarget through the arch selector.
- `crates/rdna-compute/src/dispatch.rs:2610-2635` — HFQ3 dispatch
  using `launch_maybe_blob` (the graph-safe pattern to mirror in 4b).
- `crates/hipfire-quantize/src/main.rs:2011-2013` — `--allow-mq3-lloyd`
  / `HIPFIRE_ALLOW_MQ3_LLOYD=1` guard at quantizer.
- `benchmarks/results/devlog_20260506_lloyd_mq4_extension.md` — the
  session devlog this plan slots into; flags MQ4-Lloyd as the next
  follow-up that will reuse this LDS path with 16-entry codebooks.
- `benchmarks/results/lloyd_max_findings_20260501.md` — PR's full
  empirical writeup.
- `docs/plans/PR-115-lloyd-max-cb-plan-rev-claude.md` — adversarial
  review (this plan, rev 2) consolidating gemini + glm5 + Claude
  findings; includes rejected false alarms with rationale.
- `PR-115-lloyd-max-cb-plan-rev-gemini.md` — gemini's review.
- `docs/plans/PR-115-lloyd-max-cb-plan-rev-glm5.md` — glm5's review.
