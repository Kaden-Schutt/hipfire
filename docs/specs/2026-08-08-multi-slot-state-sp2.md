# Multi-slot state and per-slot ops (SP2)

- **Date:** 2026-08-08
- **Base:** `feat/batched-attn-impl` (SP1)
- **Status:** design

## 1. Goal

SP1 gave the attention kernels a slot dimension: one launch serves N independent
sequences. **Nothing creates or maintains those slots.** Every production call
site still passes null descriptors.

SP2 builds the state and the per-slot operations around them, so SP3 can
assemble a ragged multi-slot forward pass without also having to invent the
allocator.

**Scope decision (agreed): components only, wired in SP3.** SP2 does not run a
model forward and does not produce an end-to-end token. Each component is
verified at unit level against its existing single-sequence counterpart.

## 2. The shape of the work

Every family SP2 touches already has a *multi-token, single-sequence* batched
variant — exactly the position SP1's attention kernels were in:

| family | existing batched entry point | what it batches today |
|---|---|---|
| KV write | `kv_cache_write_q8_0_batched` | N token positions, one sequence (already takes a device `positions[]`) |
| RoPE | `rope_batched_f32`, `rope_partial_interleaved_f32_batched` | N token positions, one sequence |
| DeltaNet | `gated_delta_net_q8_batch_seq` | N tokens, **one** `s_q8` state |
| Sampling | `argmax_f32_batched`, `sample_top_p` | N rows of logits |

So SP2 is the same move SP1 made, applied four more times: **add the slot axis
to families that already have the token axis.** The descriptor machinery already
exists and is not redesigned — `KvSlotDesc`, `build_tiles`, `kv_offset_for_k/v`
and `preflight_alloc` all ship in SP1.

## 3. Components

### 3.1 `SlotPool` — the allocator (the genuinely hard part)

Owns, for `n_slots`:

- per-layer K and V arenas, sliced into per-slot slabs
- per-slot DeltaNet `s_matrices`, `s_scales`, `conv_states`
- the host-side `Vec<KvSlotDesc>` and its device mirror

Responsibilities: `acquire(slot_cap_tokens) -> SlotId`, `release(SlotId)`,
`reset(SlotId)`, and `descriptors() -> &[KvSlotDesc]` with change-detected
upload (skip when unchanged, following the ds4 precedent SP1 already cites).

**Fixed-size slabs, not a free-list.** Every slot gets the same `cap`, chosen at
pool construction. Variable-size slabs would fragment and buy nothing at 2-8
slots, and the paged upgrade (SP4 §15) replaces this wholesale rather than
extending it. YAGNI applies hard here.

**Two invariants the pool enforces host-side**, because SP1 deliberately removed
the device asserts (they shipped in release and cost 64 B/lane of scratch):

1. `seq_len <= cap` for every descriptor.
2. **`v_base == k_base` for Q8_0 arenas** — an ABI constraint from SP1: the Q8
   flash-prefill kernel uses one shared slab offset, because keeping both bases
   live across its staging loop costs 23 VGPRs and 25% of its occupancy. asym3
   is exempt and must keep both bases; its K and V strides genuinely differ.

### 3.2 Batched KV write and RoPE

`kv_cache_write_q8_0_batched` already takes a device `positions[]`; it needs the
same treatment attention got — a `KvSlotDesc` table plus a per-row `row_slot[]`,
with addressing routed through `kv_offset_for_k/v`.

**RoPE is expected to need no slot awareness at all.** It transforms q/k in the
flat row layout *before* the cache write, and `positions[]` is already indexed by
global row. This must be confirmed rather than assumed; if it holds, RoPE is
zero work and the plan says so.

### 3.3 Batched DeltaNet state update

Despite being three quarters of the layers, this is expected to be the *easiest*
component, and it is worth stating why so the effort is not mis-budgeted:

- The state is **fixed-size** — it does not grow with context.
- Slots are **completely independent**: no shared arena, no ragged lengths, no
  offset descriptors. Just a slot stride on the state pointer.

| | per slot | 4 slots |
|---|---|---|
| 27B (48 linear layers, 48 v-heads, 128×128) | 151 MB | 0.60 GB |
| 35B-A3B (30 linear layers, 32 v-heads, 128×128) | 63 MB | 0.25 GB |

Against 4.56 GB/slot of KV at 128K, DeltaNet state is ~3% of the cost.

The recurrence runs *within* a slot across tokens and never across slots, so
adding the slot axis cannot introduce a cross-slot dependency.

### 3.4 Batched sampling, with per-slot parameters

`argmax_f32_batched` already handles N rows. The real design content is that
**each agent may have different sampling parameters** — temperature, top-p,
top-k, penalties. A single scalar temperature across the batch is wrong the
moment two agents differ.

Sampling therefore takes a per-slot parameter table, in the same style as the
descriptor table. Greedy/argmax remains a fast path when every slot is greedy.

## 4. Non-goals

- No forward pass, no scheduler, no daemon change (SP3/SP4).
- No paged or swappable KV — SP4 §15, and it replaces `SlotPool`'s addressing
  rather than extending it.
- No variable-size slabs, no free-list, no defragmentation.
- No new KV quant modes: Q8_0 and asym3, matching SP1.

## 5. Verification

SP1's gates carry over unchanged and are not re-derived:

- **`scripts/attn_legacy_baseline.sh`** — numeric legacy fingerprint. Any SP2
  change that touches a shared kernel must leave it bitwise identical.
- **`scripts/kernel_resource_gate.sh`** — VGPR/scratch/occupancy, both arches.
  Added in SP1 after a 25% occupancy regression passed nine numeric checks. Any
  SP2 kernel edit re-runs it, and a deliberate cost updates the baseline in the
  same commit with the reason.
- **Per-component golden equivalence**: each slot's result from the multi-slot
  op must match the existing single-sequence op run on that slot alone.
- **Cross-slot isolation**: poison every other slot's state with NaN and confirm
  the target is unchanged and finite. This was the sharpest instrument in SP1
  and it transfers directly to DeltaNet state and KV write.
- **A negative control per component.** SP1's lesson: a suite that has never
  failed is not evidence, and a control that corrupts *both* arms proves
  nothing. Each control must produce a genuine numeric mismatch.

## 6. Memory discipline — binding

Non-negotiable, and the reasons are measured rather than theoretical:

- **The cgroup does NOT contain amdgpu GTT.** A gated run still invoked the
  *global* OOM killer and killed the user's Slack and three steamwebhelper
  processes. `MemoryMax` bounds host RSS only.
- **Refuse to start** unless `MemAvailable >= cap + 10 GiB`. This is the primary
  protection, not the backstop.
- **`kv_slots::preflight_alloc(total, budget, what)`** before allocating, with
  the total held live at once — **device and host**. A review found harness
  totals omitting host `Vec`s and undercounting by ~30%.
- **Never run a GPU harness while a `hipfire serve` daemon holds a model.** Check
  `pgrep -f 'hipfire/bin/daemon'`. One resident model takes MemAvailable from
  ~58 GiB to ~19 GiB.
- **Never write to `~/.hipfire/`.** Everything needed is an environment
  variable. An agent edited the user's config and died before restoring it,
  leaving `max_seq` at 16384 instead of 131072.
- A live `free` shows nothing after the fact. Damage appears only in
  `journalctl -k` with `constraint=CONSTRAINT_NONE`.

## 7. Success criteria

1. `SlotPool` acquires, releases and resets slots; descriptors upload with
   change detection; both invariants enforced host-side with tests.
2. Batched KV write, DeltaNet update and sampling each pass golden equivalence
   against their single-sequence counterparts for slot counts 1-8.
3. Cross-slot isolation passes for KV write and DeltaNet state.
4. Each component has a negative control that produces a real mismatch.
5. Legacy numeric fingerprint and kernel resource gate both unchanged.
6. RoPE's slot-independence is confirmed or refuted explicitly.

## 8. Risks

- **The allocator is the real work**, not DeltaNet. Effort should be budgeted
  accordingly; an earlier estimate had this backwards on layer count alone.
- **Per-slot sampling parameters are easy to under-scope.** A single scalar
  temperature across the batch is wrong as soon as two agents differ, and the
  bug is invisible in a uniform-parameter test.
- **gfx1151 is not gfx1201.** Same caveat as SP1: no tuned constant in a `const`.
- **GPU verification is gated on the box being free.** SP2's unit tests are
  small, but they are still GPU work and must wait for headroom.
