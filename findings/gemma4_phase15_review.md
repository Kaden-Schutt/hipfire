# Adversarial Review — §4.5 Phase 1.5 (Long-context: sliding-window + ring-buffer)

**Reviewer:** pi (adversarial mode)
**Date:** 2026-06-08
**Section:** `docs/plans/gemma4_dispatch.md` §4.5 (lines 546–686)
**Branch:** `feat/dispatch-unification-gemma4` @ `41bd5d87`

---

## Verdict: Plan is mostly sound but has 4 material issues

The plan's decomposition (Step A = correctness via windowing, Step B = memory via
ring buffer) is the right split. The kernel provenance claim is mostly accurate.
But there are four issues ranging from merge-blocking to silent-correctness-risk.

---

## Issue 1: MERGE-BLOCKING — `feat/gemma4-128k-ring-buffer` diverges from dispatch branch

**Severity: High — blocks Step B execution**

The plan states (§4.5.2, line 602):

> The proven kernels exist on `feat/gemma4-128k-ring-buffer` (current branch
> kernels == master for these files, so they apply cleanly)

The kernel HIP diffs apply fine (184 lines across 3 files). But the **Rust side**
of the ring-buffer branch diverges catastrophically from the dispatch branch:

1. `hipfire-dispatch/` does not exist on the ring-buffer branch — it was created
   on the dispatch branch. The ring-buffer branch **deleted** `kv_tier.rs`
   (confirmed: `git diff HEAD feat/gemma4-128k-ring-buffer -- crates/hipfire-dispatch/`
   shows the file as deleted). The plan's Step B references "derive `cache_capacity`
   once in `KvTierPlan::derive`" — that struct doesn't exist on the ring-buffer branch.

2. The GPU method signatures differ. The ring-buffer branch adds `cache_capacity`
   directly to existing methods like `kv_cache_write_asym3_fused`. The plan
   proposes a **sibling method** strategy (`_cap` variants). These are incompatible
   approaches — cherry-picking the kernel changes and applying the plan's Rust
   strategy requires writing the Rust side from scratch anyway.

3. The ring-buffer branch **reverts** the fp32 `attention_flash.hip` window fix
   we just committed (`41bd5d87`). A merge would lose our fix. The plan doesn't
   mention this.

**Impact:** The "proven kernels apply cleanly" claim is true for 3 HIP files but
false for the Rust integration. The plan's §4.5.2 Rust strategy and the
ring-buffer branch's Rust strategy are different designs targeting different
codebases. Someone implementing Step B needs to cherry-pick the 3 HIP diffs and
write the Rust wiring fresh per the plan's sibling-method design.

**Recommendation:** Amend the plan to state: "Kernel HIP changes cherry-picked
from `feat/gemma4-128k-ring-buffer`; Rust integration written fresh per §4.5.2
sibling-method strategy. Do NOT merge the ring-buffer branch."

---

## Issue 2: fp32 path ring-buffer gap — unhandled write-side wrapping

**Severity: Medium — affects oracle/debug paths**

The plan's §4.5.2 only discusses the **asym3** ring-buffer path:
- `kv_cache_write_asym_k_givens3.hip` (K write)
- `kv_cache_write_q8_0.hip` (V write)
- `attention_flash_asym3_tile.hip` (read)

It does not mention the **fp32** KV path:
- `kv_cache_write.hip` — used by `gpu.kv_cache_write()` (the fp32 write)
- `attention_flash.hip` — used by `gpu.attention_flash()` (the fp32 read, which
  we just added `kv_window` to in `41bd5d87`)

On the current branch, the daemon allocates the sliding KV as **fp32** via
`KvCache::new_gpu` (line 4006 in daemon.rs). This means:

- The daemon's sliding KV write at line 1938 uses the fp32 `kv_cache_write`
- The daemon's sliding KV read at line 1963 uses the fp32 `attention_flash`
- NEITHER has ring-buffer wrapping (`slot = pos % cap`)

If Step B is implemented only for asym3, the daemon (which uses fp32 sliding KV)
still writes `slot = pos` and would OOB at pos >= 1024.

**Wait — is this actually a problem?** The plan says the daemon will switch to
asym3 for the sliding path. Let me check:

The daemon currently allocates:
```rust
// daemon.rs ~4000
let kv_sliding = llama::KvCache::new_gpu(  // fp32
    gpu, config.n_layers, config.sliding_n_kv_heads,
    config.sliding_head_dim, config.sliding_window,
)
let kv_full = llama::KvCache::new_gpu_asym3(  // asym3
    gpu, config.n_layers, config.full_n_kv_heads,
    config.full_head_dim, max_seq,
)
```

The sliding cache is **fp32**, not asym3. The plan never specifies switching it
to asym3. So the ring-buffer Step B MUST also handle the fp32 path, or the plan
must add a step to switch the sliding cache to asym3.

**Impact:** Without fp32 ring-buffer support or a switch to asym3, Step B leaves
the daemon's sliding path broken at >1024. The plan's gate ("ring == window-only")
would pass if tested with the oracle (which can use either path), but the daemon
would still OOB.

**Recommendation:** Add one of:
1. A sub-step to switch `kv_sliding` to `KvCache::new_gpu_asym3` in the daemon
   (aligns with production path, simpler)
2. A fp32 ring-buffer kernel for `kv_cache_write.hip` + read-side `slot` in
   `attention_flash.hip`

Option 1 is simpler and aligns the daemon with the oracle-on-asym3 intent
described in §4.5.1.

---

## Issue 3: `cache_capacity` plumbing is incomplete — the plan understates the gap

**Severity: Medium — implementation risk, not correctness**

The plan says (§4.5.2, line 647):

> `cache_capacity` already exists on `KvTierInputs` → `KvTierPlan` → `AttnParams`
> (Phase 0a, struct-level only). Wire it the rest of the way.

Confirmed: the struct fields exist in `kv_tier.rs` (line 39, 60, 108) and
`attention.rs` (line 50). But "wire it the rest of the way" involves:

1. **`KvTierPlan::derive`** — must set `cache_capacity = sliding_window` for
   gemma4 sliding tiers. Currently `derive` is a skeleton; it doesn't produce
   gemma4-specific values. The plan doesn't show how `derive` knows which model
   is loaded or which tiers are sliding. This needs an arch-level hook or config.

2. **gemma4 → dispatch migration** — the plan says "Near-term, gemma's
   `gemma4_ext` wrappers read the cap from the plan." But gemma4 is still on
   old-style dispatch (Phase 1), not `AttentionFamily`. The wrappers in
   `gemma4_ext.rs` already accept `cache_capacity: u32` and stub it out with
   `let _ = cache_capacity`. The "near-term" wiring is just removing that stub
   and passing it to the kernel. The plan should explicitly state this — it's
   the cheapest Step B increment.

3. **Fan-out audit** — the plan discusses 46 callers of `kv_cache_write_q8_0`
   and ~7 of `kv_cache_write_asym3_fused`. But it doesn't count the
   `kv_cache_write_asym3_hd512` wrapper in `gemma4_ext.rs` (line 158). That
   method also needs the sibling/extension. Similarly,
   `kv_cache_write_q8_0_v_hd512` is a separate kernel write for hd512 V cache.

**Impact:** Implementor might underestimate the plumbing surface. The struct
fields are done but the derivation logic and the per-method threading have more
touch points than listed.

**Recommendation:** Add a concrete touch-point list:
- `kv_cache_write_asym_k_givens3` → add `cache_capacity` param
- `kv_cache_write_q8_0` → sibling `_cap`
- `kv_cache_write_asym_k_givens3_hd512` → add param (or confirm full layers
  don't need it — they don't, since full layers don't wrap)
- `attention_flash_asym3_tile` → already has window, needs `cache_capacity`
- `attention_flash_asym3_tile_hd512` → full layers, no wrap needed
- `gemma4_ext.rs` all `_window` wrappers → remove `let _ = cache_capacity`
- `gemma4.rs` sliding write calls → pass `cap = sliding_window`

---

## Issue 4: Stale kernel cache caution is incomplete

**Severity: Low — operational, easy to fix at runtime**

The plan mentions (§4.5.4, line 682):

> Stale-kernel-cache caution: changing `kernels/src/*.hip` requires clearing
> `.hipfire_kernels/{arch}/`

This is correct but underspecified. After adding `cache_capacity` to the
asym3 tile kernel, the kernel's **name** doesn't change but its parameter list
does. The hsaco cache keys on the **kernel name** (not source hash or parameter
signature). If a cached hsaco from the old 14-param version is loaded for a
15-param launch, the kernarg buffer will be misaligned — silent corruption, not
a clean error.

**Impact:** Previous sessions (Session 16) already hit this exact failure mode
with the `kv_window` addition. It's a known hazard but worth strengthening the
warning: any kernel parameter change (not just source changes) requires cache
clearing.

**Recommendation:** Add: "Adding parameters to existing kernel names is the
highest-risk cache invalidation scenario — the old hsaco loads silently but
reads garbage kernargs. Either clear the cache or rename the kernel when
changing its parameter signature."

---

## Minor observations (non-blocking)

1. **§4.5.0 accuracy:** The plan says "the debug oracle allocated the sliding KV
   via `KvCache::new_gpu` = the fp32 KV path, whose `attention_flash` branch has
   no window masking." This was accurate when written but is now stale — we fixed
   the fp32 path in `41bd5d87`. The plan should note this fix landed.

2. **§4.5.1 Step A.2 says "size the sliding cache at `max_seq` (window-only)".**
   This is the simplest correct approach. But the daemon currently sizes at
   `sliding_window`. Changing to `max_seq` means the fp32 sliding cache grows
   from 1024 rows to whatever `max_seq` is. At 128k context: 48 layers × 8 heads
   × 256 dim × 4 bytes × 128k = ~6 GB for just the sliding cache. Step B's ring
   buffer then shrinks it back to 1024 rows. This "grow then shrink" cycle is
   fine architecturally but the plan should note the transient memory spike.

3. **Gate clarity:** The plan's Step C gate says "ring == window-only" logits.
   This is sharp. But it should also specify that the comparison is at >1024
   tokens (the interesting region) not just at any sequence length.

4. **The plan says `cache_capacity = 0` is the identity.** This is correct in
   the HIP kernels (`slot = cap>0 ? pos%cap : pos`). But in Rust, `u32` default
   is 0, so any caller that forgets to set it gets the identity — which is the
   safe default. Good design.

---

## Summary

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| 1 | Ring-buffer branch Rust diverges; don't merge, cherry-pick HIP only | High | Amend plan |
| 2 | fp32 sliding KV path (daemon's path!) has no ring-buffer support | Medium | Add sub-step or switch to asym3 |
| 3 | Plumbing surface larger than stated; derive logic missing | Medium | Add touch-point list |
| 4 | Stale cache caution underspecified | Low | Strengthen wording |

The plan's core design (window = correctness, ring = memory; sibling methods for
high fan-out, extend for low fan-out) is sound. The issues are all in the
execution details — particularly the fp32/asym3 split for the daemon's sliding
cache, which is the one path that actually runs in production.
