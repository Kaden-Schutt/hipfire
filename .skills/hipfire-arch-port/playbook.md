# hipfire arch-port playbook

When you (or an agent helping you) want to add a new GPU arch to
hipfire, read this end-to-end before touching code. Most of the
mistakes other arch ports have hit are documented here so you don't
spend a week chasing them.

## When to use this skill

- A user reports a HIP codegen / kernel-select failure on a new arch
  (e.g. issue #54: `Cannot select: intrinsic %llvm.amdgcn.wmma...`
  on gfx1201).
- You're adding `gfx1200` / `gfx1201` / `gfx1151` / `gfx1152` /
  `gfx94x` / `gfx950` to the supported list.
- You see the `(EXPERIMENTAL — opt-in only)` flag on a feature and
  want to mainstream it on a new arch.
- You're refactoring `dispatch.rs`'s arch-conditional branches.

## What's in this skill

| File | Purpose |
|---|---|
| `playbook.md` (this) | Top-level workflow, when-to-use, contributor pointer |
| `wmma-matrix.md` | WMMA operand-shape × builtin × lane-layout table per arch |
| `validation.md` | The three gates every port must pass before merge |
| `contributor-onboarding.md` | If you have hardware and want to help — start here |

## The arch-port workflow (the load-bearing 6 steps)

### 1. Read `wmma-matrix.md` first

Most arch ports involve at least one matrix kernel (GEMM/WMMA/MFMA).
The matrix doc lists the operand shapes, builtin names, and lane
layouts for every arch hipfire currently knows about. **The single
biggest pitfall is assuming an `#ifdef` macro swap of the builtin
name is enough — it isn't, because operand vector lengths and
per-lane K-packing differ between archs.**

### 2. Check `dispatch.rs` for the existing arch-conditional sites

Every arch-aware GEMM dispatch in `crates/rdna-compute/src/dispatch.rs`
has the shape:

```rust
if has_<feature>(&self.arch) {
    return self.<kernel>_<feature>(...);
}
if has_<fallback_feature>(&self.arch) {
    return self.<kernel>_<fallback_feature>(...);
}
return self.<kernel>_baseline(...);
```

Style for new dispatch branches: match the surrounding code. Some
sites use `arch.starts_with(...)` inline, others factor into
`has_<feature>(arch)` helpers (`has_dot2_f32_f16`, etc.). Either
shape is fine in principle. The existing helpers are co-located
near the top of `dispatch.rs` (around lines 30–80) — extend them
when the same predicate would be tested in 3+ places. Add new
inline checks when one site needs the test once.

When adding a new arch's dispatch branch, the only invariant is
**no unreachable code**. The dispatch sites are layered fast-paths
with fallthrough — multiple archs legitimately share a downstream
predicate (e.g. `has_dot2_f32_f16` matches RDNA1.5 / RDNA2 / RDNA3
/ RDNA4 all together for the dot2 fallback), and that's correct
because they each route there only after the higher-priority
branches return-or-fall-through.

What you DO need to check after adding a new branch: did your new
branch make any of the literal conditions in lower-priority
branches redundant? Specifically:

- A literal `... || starts_with("gfxN")` clause where every gfxN-
  prefixed arch is now matched by your earlier branch → drop the
  `|| starts_with("gfxN")` clause in the same diff.
- An entire branch whose predicate is now strictly subsumed by an
  earlier branch (rare, usually a sign your new branch is too
  broad).

Predicate-style helpers (`has_dot2_f32_f16(arch)`, etc.) typically
do NOT need narrowing when you add a new arch — the helper
intentionally covers a broad family, and downstream sites rely on
that. Edit the helper only if its definition is genuinely wrong
for the new arch, not because your new branch overlaps with part
of its set.

Run the speed-gate on the baseline arch (gfx1100 on the local
7900 XTX bench) after the change. If the gate regresses,
root-cause it (see `validation.md` troubleshooting) rather than
splitting the diff to dodge the regression.

### 3. Open root-cause: "predicate-vs-inline" gfx11 perf observation

In commit `a048544` (reverted in `1f3bad3`) I replaced six inline

```rust
if self.arch.starts_with("gfx11") || self.arch.starts_with("gfx12") {
```

calls with a single `has_wmma_f16(&self.arch)` predicate. The
post-commit speed-gate measured a ~50% prefill regression on
gfx1100. The post-revert speed-gate restored 4b prefill to the
baseline (1014→1047 tok/s).

**This observation is unverified.** Possible explanations:

- A real inlining / register-allocator interaction in the dispatch
  hot loop (would manifest as a function-call frame in the GEMM
  prefill path).
- Cached build artifact during my "stash and re-run" verification:
  the speed-gate's `cargo build --release` may have been a no-op
  if the source didn't change between runs, leaving the regressed
  binary in place when I thought I was testing master.
- GPU thermal / DPM state drift over the long debugging session.
- Firmware shadowing (`feedback_firmware_shadowing_perf_trap.md`)
  pre-existing on the host.

I did NOT isolate the cause before committing the revert. The
"predicate refactor regresses gfx11" hypothesis is one
possibility among several and **should not be encoded as standard
practice** until reproduced from a clean checkout with explicit
build-cache invalidation.

**For contributors today:** if you hit a perf regression after a
dispatch.rs edit that "should be a no-op", root-cause it before
working around it. `cargo clean -p rdna-compute && cargo build
--release ...` to rule out cache, GPU `pp_dpm_sclk` to rule out
thermal/DPM drift, `dmesg | tail` to rule out firmware mismatch,
THEN look at the diff. Don't pre-emptively avoid helper functions
based on the observation above.

### 4. Author the new arch's kernel(s) as separate `.hip` files

Naming convention: `<existing_kernel_name>_gfx12.hip` (or
`_gfx1201.hip` if the variant is sub-arch-specific). Examples in
the existing tree:

- `kernels/src/gemv_hfq4g256.gfx1030.v1.hip` — gfx1030 RDNA2 variant
- `kernels/src/gemm_hfq4g256_residual_wmma_k4.hip` — gfx11 K4 WMMA

Single-file `#ifdef __gfx12__` is fine *only* when:
- The operand types are identical across archs (rare for WMMA/MFMA)
- The lane layout is identical (rare)
- The tuning constants are identical (rare)

For WMMA in particular, **the gfx11 → gfx12 port is NOT a single-file
ifdef**; operand vector lengths differ (`<16 x fp16>` vs `<8 x fp16>`)
and per-lane K-packing differs. Use a separate file.

### 5. Wire the include + dispatch

In `crates/rdna-compute/src/kernels.rs`:

```rust
pub const GEMM_X_WMMA_GFX12_SRC: &str = include_str!(
    "../../../kernels/src/gemm_x_wmma_gfx12.hip"
);
```

In `crates/rdna-compute/src/dispatch.rs`, add the dispatch branch
ABOVE the existing gfx11 inline check (per step 3).

### 6. Validate against all three gates (see `validation.md`)

A new arch port is merge-ready ONLY when:

1. **Channel-test passes** on real hardware (the contributor's
   target arch). This is correctness — `cargo run --release -p
   engine --example test_kernels` (or the QA variant) emits "OK"
   for every dispatched kernel on the new arch.
2. **Coherence-gate passes** (`./scripts/coherence-gate.sh`). No
   panics, no zero-tokens, no timeouts on the canonical
   small-prompt battery.
3. **Speed-gate passes** on the regression-baseline arch
   (`./scripts/speed-gate.sh --fast`). The new code path **cannot
   regress gfx1100** (or whichever arch the baseline lives on).

If you don't have hardware for the target arch, you cannot merge
— flag it in the PR and find a contributor with hardware (see
`contributor-onboarding.md` and #45 watchers).

## Quick reference

- WMMA matrix → `wmma-matrix.md`
- Validation procedure → `validation.md`
- Contributing without privileged repo access → `contributor-onboarding.md`
- Memory pin on this topic: `memory/project_wmma_correctness_fix.md`
  (the gfx11 `acc[j] = C[2*j + (tid>>4)][tid & 15]` mapping was
  silently wrong for 6 weeks before being caught — assume any
  per-lane mapping for a new arch is wrong until proven by channel-
  test on hardware).

## Known traps

| Trap | Symptom | Memory |
|---|---|---|
| WMMA C-mapping wrong | All-WMMA models emit garbage / fail correctness | `project_wmma_correctness_fix.md` |
| Removing "dead" WMMA kernels | Per-cycle GEMM cost ~2× on dispatch path that secretly uses it | `project_27b_dflash_perf_analysis_2026_04_22.md` (PR #32 → 9a2c667 recovery) |
| Bypassing speed-gate | Local-env regression masked by `--no-verify` lands on master | feedback (this session, commit `a048544` → reverted in `1f3bad3`) |
| "Should-be-no-op" dispatch.rs refactor regresses speed-gate | Most often a stale build cache during re-verification; less often a real codegen difference. Always run the gate after any dispatch change | This session, root cause unverified — see `validation.md` |
| Greedy degenerate decode | "Engine bug" smoke tests halt; turns out `--temp 0` + `<think>` exhaust max_tokens | `feedback_quality_gate_baselines_degenerate.md` |
| Firmware shadowing | `/lib/firmware/updates/amdgpu` overrides kernel firmware → SMU IF mismatch → 50% prefill drop, looks like code regression | `feedback_firmware_shadowing_perf_trap.md` |

## Skill discoverability

This skill lives at `.skills/hipfire-arch-port/`. Triggers in
`skill.json` cover the obvious phrases. Future agents asking
"how do I support gfx1XYZ?" should land here directly.
