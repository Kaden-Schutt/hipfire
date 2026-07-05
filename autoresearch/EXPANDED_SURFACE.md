# Autoresearch surface: expand from `.hip`-only to `.hip` + hot-path launch config

## The realization (2026-07-05)
The autoresearch certify loop swaps ONLY `kernels/src/<k>.hip` and measures. That surface is **structurally
blind to half the lever space.** Everything it gained (traffic-cut, attention-ring, ~10-22% cumulative) was
`.hip`-expressible; then cache / occupancy / float4 / de-unroll / global_load_lds all died — and the last
several died *because the remaining levers don't live in the `.hip` file*. They live in the **Rust launch
config** (`crates/rdna-compute/src/gemm.rs` / `gemv.rs`): block/grid shape, **split-K**, fusion↔de-fusion,
tiling, LDS budget, dispatch-variant selection. A source-only loop cannot reach any of them.

**So the "gfx1100 kernel-only ceiling" we hit is a HARNESS-SCOPE artifact, not a hardware ceiling.** The
ratio-proven ~25% headroom (see levers/gfx1100.md) is almost certainly reachable only on the expanded surface.
Concrete proof: split-K PRECHECK_FAIL'd in the loop with `learned=source-only certify swaps only the .hip while
the runtime launch is fixed block=[32,1,1]` — the lever needed `[32]→[S*32]` in gemm.rs, which the loop can't touch.

## The scoped expansion (thread the needle — NOT "let it edit gemm.rs")
Expand the certify "change" surface to exactly two things per kernel, no more:
1. `kernels/src/<k>.hip` — the kernel body (as today).
2. **The launch TUPLE at that kernel's launch site** — `block` / `grid` / dynamic-LDS / launch args. NOT the
   surrounding dispatch routing, kernel selection, or any correctness-critical logic.

Scope to the **hot path only** (the decode/prefill kernel launches), never the runtime orchestration
(KV cache, sampler, daemon, serving).

### Three-layer safety (defense in depth)
1. **Prompt scope** — codex told it may touch only the kernel `.hip` + that kernel's launch tuple.
2. **Diff-scope GUARD (mechanical, not trust)** — reject any variant whose `git diff` touches lines/files
   outside {the `.hip`, the specific launch tuple}. This is the load-bearing guardrail.
3. **Coherence gate** — the ultimate net: a bad block size / wrong reduction → wrong tokens → COHERENCE_FAIL
   → rejected. Already catches this class. Build failure also rejects (Rust won't compile → out).

## Harness change (certify_v3, TODO — gated on split-K prototype proving the surface)
- Reset worktree to baseline; apply variant = `.hip` + launch-tuple edit.
- Build the workspace daemon (rebuilds rdna-compute → picks up the dispatch change). Same coherence + A/B as v2.
- Precheck adds launch-config verification (block/grid emitted as intended) alongside the .hsaco VGPR/occ check.
- The FOLD/promote-fork model still applies: arch-specific launch configs fork per arch (a gfx1100 split-K
  block size is wrong for gfx1201's 64 CU — fork the launch tuple too, not just the .hip).

## Why this is the unlock
The `.hip`-only space is mapped and largely exhausted for gfx1100 decode. The launch-config surface is where
split-K, block/grid tuning, and fusion decisions live — a strictly larger space that includes the structural
levers the measurement isolated (occupancy via cooperative multi-wave, which REQUIRES the block-size change).
Prototype = the manual split-K build (kernel + gemm.rs together). If it's viable, generalize into certify_v3.
