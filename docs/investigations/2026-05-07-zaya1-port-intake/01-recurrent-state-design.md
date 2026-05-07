# 01 - CCA Recurrent-State Design

**Date:** 2026-05-07
**Branch:** `feat/zaya1-port-intake`
**Status:** REQUIRES-KADEN-DECISION (see MANUAL_REVIEW.md)
**Predicates:** 00-cca-disambiguation.md (verdict: RECURRENT), 02-mod-design.md (no
runtime changes needed), 03-eda-identification.md (no recurrent state)

## What this doc decides

Where the CCA recurrent state lives in hipfire. Two coherent options
follow; neither is wrong, but their long-term shape differs.

The state being placed:

| Buffer | Per-layer shape (per seq) | Dtype | Update semantic |
|---|---|---|---|
| `conv_states` | `[in_out_ch=1280, conv_kernel_size=2]` | fp16 | roll(-1) + write at [-1] |
| `prev_hs` | `[hidden_size=2048]` | fp16 | overwrite (1-step lag of input hs) |

For ZAYA1-8B: ~370 KB per sequence at fp16. (40 ATT layers carry CCA;
the other 40 are MLP layers and have no CCA state. ZAYA1's
`num_hidden_layers=80` is total sub-layer count, alternating ATT/MLP;
per Phase 1 layer-structure probe.) No KV-pager-scale allocation;
comfortably HBM-resident.

This is the **first per-layer recurrent state in hipfire** that lives
across decode steps (qwen35's DeltaNet recurrence is folded into the LA
forward signature; never externalized as a runtime concept).

## Two options

### Option A: per-arch State carries the recurrent buffers

ZayaState owns the buffers; allocation, update, reset are
arch-private. The runtime sees only an opaque `Self::State` per the
`Architecture` trait. No new abstractions in hipfire-runtime.

**Pros:**
- Zero hipfire-runtime churn. PR fits in `hipfire-arch-zaya/` plus the
  per-arch HFQ format growth (or a new metadata field) for the conv
  state shapes.
- Doesn't lock in a shape for a recurrent-cache abstraction before we
  see what other recurrent archs (Mamba2, SSM hybrids, RWKV) actually
  need. Keeps the experimentation surface in the per-arch crate.
- Easy to delete if ZAYA1 is retired; runtime never grew the slot.

**Cons:**
- Reuses no infrastructure. Spec-decode, paging, multi-GPU sharding
  all become per-arch concerns instead of per-runtime.
- Two recurrent arches (this one and a hypothetical next) duplicate
  the buffer-management code.
- The arch crate ends up doing GPU allocation it would otherwise
  delegate to a pager.

### Option B: first-class recurrent-cache abstraction in hipfire-runtime

Mirror the existing KV-cache abstraction with a parallel
"recurrent cache" primitive: per-sequence per-layer typed slots,
allocation via the runtime's pager, update-semantic registered by the
arch crate (roll-and-write here, but other patterns possible), reset on
session end, paging-aware, sharding-aware.

**Pros:**
- Spec-decode, paging, sharding gain a uniform place to handle
  recurrent state. When DFlash/MTP comes online for ZAYA1 it can use
  the existing fork/merge primitives by design.
- Future recurrent arches (Mamba2, SSM) cost zero new runtime code.
- Reset / migration / serialization for sessions becomes a runtime
  concern, not arch-private.

**Cons:**
- Larger PR, more design surface to get right. The first abstraction
  attempt almost always shapes the next one wrong; we'd be designing
  for a population of one.
- Locks in a shape before we know what other recurrent archs need.
  Mamba2's selective scan has different update semantics; CCA's
  roll-and-write is a special case.
- Slower to ship.

## Recommendation

**Option A first. Migrate to Option B after the second recurrent arch
arrives.** Concretely:

1. Phase 6.A (this PR; estimated 1-2 weeks): land Option A with the
   recurrent buffers in `ZayaState`. Restrict spec-decode for ZAYA1
   to AR-only on first ship (small, explicit comment in the daemon).
   Paging: hold ZAYA1 sequences in HBM in their entirety (no paging
   mode) on first ship; the recurrent state is small enough that
   paging adds complexity without payoff.

2. Phase 6.B (deferred until a second recurrent arch): when Mamba2
   or SSM lands, pull the buffers and update semantics out of
   `ZayaState` into a runtime primitive whose first user is ZAYA1.
   By then we will have seen what semantics generalize.

This is the same shape the engine modularization PRD took for
`speculative.rs` and `pflash.rs`: live in the arch crate today,
generalize when the second user arrives.

## Detailed Phase 6.A plan (Option A, this branch)

### State allocation

`ZayaState::new(gpu, cfg)` allocates two device buffers per sequence
during session creation:

```rust
let conv_states = gpu.alloc_zero_fp16(
    cfg.num_hidden_layers * cfg.batch_size *
    (cfg.cca_num_q_heads + cfg.num_query_groups) * cfg.head_dim *
    max(cfg.cca_time0, cfg.cca_time1)
)?;
let prev_hs = gpu.alloc_zero_fp16(
    cfg.num_hidden_layers * cfg.batch_size * cfg.hidden_size
)?;
```

Shape choice: keep both as flat fp16 slabs with explicit per-layer
strides. The `roll(-1) + write[-1]` semantic for conv_states becomes a
single uint32 swap per channel if `conv_kernel_size = 2` is laid out
as two consecutive fp16 slots.

### Per-step update kernel

One fused HIP kernel per layer's CCA forward, called from the per-arch
forward path:

```c
// pseudo: per-channel update
__global__ void cca_advance_state(
    const half2* qk_packed_new,   // [B, in_out_ch/2]   new step's QK proj
    half2*       conv_states,     // [B, in_out_ch/2]   was [B, ch, 2] viewed as packed
    const half*  hs_input,        // [B, hidden_size]   current step's input hidden state
    half*        prev_hs,         // [B, hidden_size]
    int B, int in_out_ch, int hidden_size
) {
    // For each lane in wave32 (gfx1201):
    //   - Load 2 fp16 channels packed as half2 from conv_states (= old [t-1, t])
    //   - Shift: new_state.x = old_state.y; new_state.y = qk_packed_new
    //   - Store back: conv_states[lane] = new_state
    // For prev_hs: copy current hs into prev_hs (separate kernel or fused tail).
}
```

Wave32-friendly because `in_out_ch/2 = 640` divides cleanly into
`640 / 32 = 20` lane-groups. fp16 packed math is two ops per lane per
update.

The conv1d output is computed in a separate kernel that consumes
`conv_states` directly (read the [t-1, t] pair, multiply by depthwise
weights, accumulate). The grouped second conv operates on the result.

### Spec-decode story

First ship: **gate ZAYA1 to AR-only.** Add `Self::supports_spec_decode()
-> false` (or equivalent) on the trait, default true, override here
returning false.

Why: drafter and target sharing CCA conv state across N parallel
candidate tokens means N parallel possible state advances; rollback on
rejection requires reverting the conv state. The roll(-1) operation is
not cheap to invert without a save/restore. In Phase 6.B / future PRs,
add a `state.fork(N)` / `state.commit(idx)` primitive for general
spec-decode of recurrent archs.

This is a real cost: ZAYA1 loses the 1.5-3x DFlash multiplier. The
acceptable trade is "AR-only ship now, recurrent spec-decode later."

### Paging story

First ship: **no paging for ZAYA1 sequences.** The recurrent state is
small (~720 KB per sequence) but its update semantic doesn't compose
with the LRU eviction the existing KV pager does. A paged-out
sequence's conv_states can't roll forward without being resident.

Long-term: pin the recurrent state in HBM even when the KV cache is
paged out. The 720 KB is ~0.07% of an R9700's 32 GB; even 256
sequences only consumes 184 MB pinned.

### Multi-GPU sharding

Pipeline parallel (PP): natural fit. Each GPU owns a layer range; the
recurrent state for those layers lives on that GPU. The `prev_hs`
boundary travel between PP stages is one extra `[B, hidden_size]`
fp16 send per decode step (negligible vs the residual stream that
already crosses).

Tensor parallel (TP) within a layer: `conv_states` shards along its
channel dim (`in_out_ch=1280`); each shard owns its 1/N slice of the
1280 channels. The depthwise+grouped conv is locality-friendly (each
output channel is a function of its own channel's history). TP shards
the prev_hs along the residual stream's existing shard axis.

### Migration path

Existing arches (qwen35, llama, gemma) gain a no-op `State` slot for
the recurrent buffer; their `new_state` continues to allocate zero
recurrent bytes. The `Architecture` trait does NOT grow new methods
(forward stays static, recurrent state is owned per-arch). Zero impact
on existing arch crates.

When Phase 6.B comes, the recurrent-cache primitive moves into
`hipfire-runtime::recurrent_cache` and `ZayaState` grows a borrow into
it. Other arches still allocate zero bytes there; the primitive is
opt-in by per-arch crate.

### Reset semantics

`ZayaState::reset_cca(&mut self)` zeroes both buffers and sets
`has_previous_state = false`. The runtime's session-end / clear-cache
path calls this alongside KV reset. No new runtime entry point needed;
the existing `Architecture::reset_state(state)` (TODO: confirm this
exists; if not, add as part of this PR) covers it.

## Effort estimate

| Sub-task | Effort |
|---|---|
| ZayaState allocation + reset (Option A) | 1-2 days |
| CCA conv-state advance kernel + tests | 3-5 days |
| CCA prev_hs writeback (fused or separate) | 1 day |
| ZAYA1 spec-decode opt-out (1-line override) | 1 hour |
| Per-layer NRMSE pass on CCA output (vs ref) | 2-3 days |
| Documentation + PR review iteration | 2-3 days |
| **Total Phase 6.A** | **~2 weeks** |

This assumes Phase 0-5 deliverables already landed (which they will
when this PR's predecessors merge). It does NOT include:
- HFQ writer for ZAYA1 (separate work, ~3-5 days).
- Final ZayaAttention plumbing for the 8q->16q head rebalance (Phase 1
  open question; ~1 day once read).
- MoD impl (~3-5 days, Phase 4 plan).
- EDA impl (~half-day, Phase 5 plan).

End-to-end ZAYA1 first-token: ~3-4 weeks calendar from this branch's
state. End-to-end coherent decode + bench: ~5-6 weeks.

## RDNA-aware design points (per user clarification)

The user explicitly called out: respect the prescribed model arch AND
target RDNA hardware respecting its ISA. Specific commitments:

1. **Wave32 first.** All CCA kernels target gfx1201's wave32 default,
   not wave64. The depthwise k=2 conv with `in_out_ch=1280` divides
   cleanly into 1280/32 = 40 wave-groups for the per-channel update.

2. **Packed fp16 (`v_pk_fma_f16`).** The conv1d's k=2 multiply-add per
   channel is exactly the shape of one packed fp16 FMA. Two fp16 weights
   in `<2 x fp16>` × two history slots in `<2 x fp16>` = one
   `v_pk_fma_f16` issue, accumulating into the output.

3. **LDS-resident state for hot loop.** During a multi-token decode
   chunk (e.g. spec-decode candidate evaluation, when re-enabled), the
   conv_states for the layers being processed can sit in LDS with bank
   conflict-free strides. 720 KB total state for ZAYA1-8B fits in a
   single CU's LDS (gfx1201 has 96 KB LDS per CU; sequence-state per
   layer is ~9 KB so a few layers can live in LDS at once).

4. **No SHL+OR composition tricks.** The roll(-1)+write[-1] semantic
   is a single uint32 swap when conv_states are laid out as
   `(channel, time)` packed pairs. No extra ALU.

5. **gfx1201 wmma-SAFE.** No matrix-unit code in CCA; the conv is too
   small for wmma. The downstream attention (post-CCA) reuses the
   existing FA path which already targets gfx1201.

These are sketches, not commitments. The actual kernel layout falls
out during impl; this section is here to set the bar.

## REQUIRES-KADEN-DECISION

Two decisions to confirm before Phase 6.A merges:

1. **Option A vs B, this PR**. Recommendation Option A; happy to flip
   to B if you want to invest now. (Cost delta: roughly +1 week.)

2. **First-ship spec-decode policy: AR-only or block-mode-spec?**
   Recommendation AR-only first. The recurrent-spec-decode design
   landed properly takes care; cheap stub today buys ZAYA1 ship.

3. **Paging policy**: HBM-pinned per-sequence vs paged. Recommendation
   HBM-pinned, given the small per-sequence footprint (~720 KB at
   bf16 / fp16) and the un-page-friendly update semantic.

Other items in the design above (kernel choices, Wave32 vs Wave64,
LDS layout, multi-GPU shard axis) are implementer's call; this doc
captures the proposal but does not require approval before code
starts on those items.
