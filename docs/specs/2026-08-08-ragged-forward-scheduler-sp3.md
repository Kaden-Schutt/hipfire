# Ragged multi-slot forward and scheduler (SP3)

- **Date:** 2026-08-08
- **Base:** `feat/batched-attn-impl` (SP1 + SP2)
- **Status:** design

## 1. Goal

SP1 gave the attention kernels a slot dimension. SP2 built the state around them
— `SlotPool`, slot-aware KV write, DeltaNet slot stride, per-slot sampling.
**Neither runs a model.** Every production call site still passes null
descriptors.

SP3 assembles them: a forward pass that advances N slots in one step, and a
scheduler that decides what to run. **This is the first sub-project that
produces a visible result** — N sequences generating concurrently.

## 2. The integration surface

Everything funnels through one function:

```rust
pub fn forward_prefill_batch_with_pbs_opts(
    gpu, weights, config,
    tokens: &[u32],              // ONE sequence
    start_pos: usize,            // ONE position
    kv_cache: &mut llama::KvCache,   // ONE sequence
    dn_state: &mut DeltaNetState,    // ONE sequence
    scratch, hidden_rb, per_token_hidden_out, gdn_tape,
    tree_verify, pbs_in, mask_override, max_layer, needs_last_token_logits,
) -> HipResult<()>
```
`crates/hipfire-arch-qwen35/src/qwen35.rs:6554`.

Four of its parameters are singular where SP3 needs plural. That is the whole
job, and it is why SP3 is integration rather than kernel work.

## 3. Approach: a parallel entry point, not a rewrite

**`forward_prefill_batch_with_pbs_opts` is not modified.** SP3 adds
`forward_batch_slots`, which takes a `SlotBatch` and drives the same per-layer
sequence with slot-aware calls. The existing function keeps working untouched,
so every current caller — chat, spec decode, MTP, the TUI — is unaffected while
the multi-slot path matures.

Rejected alternative: generalise the existing function so `n_slots == 1` is the
old behaviour. It is tempting (no duplication) but wrong here — that function
carries hipGraph capture eligibility, tree-verify, GDN tape and MTP interactions
whose interaction with a slot axis is unknown, and breaking AR decode is a much
worse outcome than some duplication. SP1's experience is the argument: the
null-descriptor path was preserved bitwise precisely so a regression could not
hide, and that discipline caught a 25% occupancy loss.

Consolidation is a later decision, once the multi-slot path has run.

## 4. `SlotBatch` — one step's work

```rust
pub struct SlotBatch {
    /// Per-slot token counts for this step. 0 = slot idle.
    pub m_per_slot: Vec<usize>,
    /// Flat token ids, packed across slots in slot order.
    pub tokens: Vec<u32>,
    /// Per-row absolute position. Global-row indexed, authoritative for the
    /// causal bound (never desc.seq_len — SP1's only Critical defect).
    pub positions: Vec<i32>,
    /// Slot index per flat row.
    pub row_slot: Vec<i32>,
}
```

A step therefore mixes freely: slot 0 verifying 8 draft tokens, slot 1
chunk-prefilling 256, slots 2-3 decoding 1 each. That raggedness is what SP1's
kernels were built for and it is not re-derived here.

## 5. Scheduler

Deliberately simple, because the interesting scheduling questions cannot be
answered until something runs.

- **Chunked prefill.** A long prompt is split into chunks so it cannot block
  other slots for its whole duration. Chunk size reuses the existing
  `HIPFIRE_PREFILL_MAX_BATCH` (default 256).
- **Prefill and decode mix in one step.** No separate prefill phase; a slot that
  is prefilling contributes a large `m`, a decoding slot contributes 1.
- **No preemption, no priorities, no fairness policy.** Slots are served
  round-robin. SP4 owns admission control and residency.

**Measured constraint that shapes this:** the batched-vs-sequential win on the
attention term is ~1.36× at 8 slots, and batching recovers under a tenth of the
available bandwidth headroom. So the scheduler should not contort itself to
maximise batch width — the returns are modest and it is not where the remaining
performance is. Correctness and not blocking other slots matter more.

## 6. Per-slot speculative decode

hipfire's headline decode numbers come from DFlash spec decode, so the
multi-slot path must not silently drop it.

**SP3 scope: per-slot draft lengths and independent acceptance**, expressed as
each slot contributing `m = 1 + n_draft_accepted` rows to the batch. The ragged
`m` is exactly what §4 already carries, so this needs no new mechanism.

**Explicitly out of scope: tree-verify with descriptors.** SP1 asserts
`tree_bias.is_none()` when descriptors are present, and that assertion stays.
Combining tree attention with multi-slot is a separate piece of work with no
current contract.

## 7. What SP3 must not assume

Carried from SP1 and SP2, each learned the hard way:

- **`positions[]` is authoritative for the causal bound, never `desc.seq_len`.**
  They differ whenever a slot has more than one query row.
- **The KV write resolves both arenas through `k_base`.** Correct under the Q8
  ABI (`v_base == k_base`, enforced by `SlotPool`), but **asym3 cannot use that
  path** — its strides differ. SP3 must not treat the write kernel as
  mode-agnostic.
- **DeltaNet takes one launch per slot.** The recurrence is sequential within a
  slot; one launch advances one slot's state. Mixing slots in a launch would
  interleave independent recurrences.
- **RoPE is slot-agnostic** (SP2 Task 2, verified) — no work needed.
- **The LDS attention path lost to tile in 62 of 63 multi-slot trials**, which
  contradicts the shipped `LDS_CTX_LIMIT = 15000`. SP3 owns that routing
  decision; SP1 deliberately left the constant alone.

## 8. Success criteria

1. `forward_batch_slots` advances N slots one step and produces per-slot logits.
2. **Golden equivalence:** N slots run together produce the same tokens as each
   run alone through the existing single-sequence path, within tolerance.
3. Chunked prefill and decode mix in one step without a slot starving.
4. Per-slot spec decode works with differing draft lengths and acceptance.
5. Existing single-sequence paths are untouched — legacy numeric fingerprint and
   kernel resource gate both unchanged.
6. **A demo runs N sequences generating concurrently** and prints their output.
   This is the first user-visible result of the whole programme.

## 9. Risks

- **This is the largest sub-project.** The forward function is long and carries
  graph capture, MTP and spec-decode interactions. A parallel entry point
  contains the blast radius but duplicates structure.
- **Golden equivalence is harder than SP1's.** Sampling makes runs diverge, so
  equivalence must be checked greedily, or on logits rather than tokens.
- **The demo needs the GPU free.** Development on this box must coexist with the
  user's work; SP2's Tasks 3-5 were compile-only for exactly this reason.
- **The batching win is modest (~1.36×).** SP3 should not be judged on
  throughput; its job is that concurrency works at all.
