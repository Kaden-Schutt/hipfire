# OwnedTensor: RAII GPU scratch (eliminating the manual-free leak class)

Status: landed (core + transient-forward migration). Validation on `halo` pending.

## Problem

`GpuTensor` wraps a `DeviceBuffer` that has **no `Drop`**. GPU memory is reclaimed
only by an explicit `gpu.free_tensor(t)` (returns the buffer to the pool free-list;
real `hipFree` happens only in `drain_pool`). Three forces made this deliberate:

1. `free_tensor(&mut self, t)` needs `&mut Gpu` — a `Drop` impl can't reach the pool.
2. `sub_offset()`/`alias()` produce non-owning **views** sharing a buffer; if
   `GpuTensor` freed on drop, every view would double-free.
3. The hot path must issue no real `hipMalloc`/`hipFree` during hipGraph capture.

Consequence: every forward had to free scratch on **all** paths by hand, and Rust's
`?` early-returns skipped those frees → a leak on every error path. Closing it by
hand scaled as `#allocs x #fallible-calls` (e.g. the gemma3-vl `vision_forward`
band-aid was ~+380 lines).

## Design: `OwnedTensor` (owning/view split + deferred, graph-gated reclaim)

In `crates/rdna-compute/src/dispatch/mod.rs`. `GpuTensor` is unchanged (still the
no-op-drop view / kernel-arg type).

- `OwnedTensor` wraps a uniquely-owned pooled buffer + a clone of a per-`Gpu`
  free-mailbox (`Arc<Mutex<Vec<DeviceBuffer>>>`). `Deref<Target = GpuTensor>` (and
  `.view()`) so kernels keep taking `&GpuTensor` unchanged.
- `Drop` only **enqueues** the raw buffer into the mailbox — no `&mut Gpu`, no pool
  access, no `hipFree`, no fallible work. Runs on success, `?`-error, and panic.
- Allocators: `gpu.alloc_owned` / `zeros_owned` / `upload_owned_f32`. There is **no**
  `into_owned`/wrap-a-view (an interior `sub_offset`/`alias` pointer must never
  reach `pool.free`).
- `gpu.reclaim_pending()` returns enqueued buffers to the pool. It **self-gates on
  `graph_state_live()`** (capturing, or any captured/replayable graph cached): while
  live it is a no-op, so a buffer whose pointer may be baked into a live/replayable
  graph is never handed back to a later alloc. This mirrors the existing
  `*_staging_scratch_keepalive` rule. Nothing else drains the mailbox (in particular
  `alloc_tensor` does not), so a view borrowed mid-forward stays valid until the next
  boundary reclaim. `drain_pool` empties the mailbox at teardown.

Call `reclaim_pending()` at forward boundaries — and at the bottom of a per-layer
loop (with an explicit `drop()` of that iteration's last scratch just before it) to
keep peak VRAM flat instead of accumulating across layers.

### Why this design (vs rejected alternatives)

Evaluated four designs adversarially:
- **scope-guard bag** (track→shared mailbox, reap in `alloc_tensor`): FATAL —
  reaps a still-live buffer mid-forward → read/write aliasing corruption.
- **`GpuTensor` gets `Drop`** (deferred-free queue): rejected — inverts drop
  semantics codebase-wide; deliberate no-op-drop sites (graph-baked staging,
  `peer_ar_tmp`, kernarg pointers) become silent UAF; ~40 `hip.free` sites need
  defusing.
- **per-forward arena** (bump + RAII reset): best raw perf, but `reset-to-0`
  clobbers a parent forward's live residual under nesting (the deeply-nested
  qwen35/deepseek4 forwards are the real targets), and the arena isn't freed by
  `drain_pool` → prefill-sized teardown leak. Elegant for leaf forwards only.
- **owning/view split** (this): no fatal flaw; zero decode/hot-path cost (not
  applied to persistent scratch); does not change `GpuTensor`'s drop semantics.

## Migration

Converted per-call **pooled** scratch in the transient forwards; deleted their
manual frees + error-path band-aid scaffolding:

- gemma3-vl: `vision_forward` (324→189 LOC), `project`, `serve_with_embeds`.
- qwen35-vl: `vision_forward`, `linear_f16`.
- zaya: `gpu_forward_serve/_calib/_decode/_prefill` (via `zo`/`z2o` helpers).
- gemma3: `forward_prefill_batch`.
- llama (runtime): `prefill_forward`, `forward`, `forward_logits_gpu`.
- qwen35: `rq_apply_*`, `forward_scratch_layers`, `grouped_moe_prefill_*_final_logits`,
  `capture_pflash_block_scores`, `run_dflash_draft_for_logits/_topk_gpu`,
  `compute_scores_batched_gpu`.
- deepseek4: `prefill_with_mtp_fill` (`snap`).

Intentionally **kept manual** (not pooled single tensors, or escaping):
- Raw `gpu.hip.malloc` (`pos_buf`, `pos_buf_tmp`) — not pooled; freed on all paths.
- Scratch **structs** (`PrefillBatchScratch`/`own_pbs`, `Qwen35Scratch`, etc.) and
  `Vec<DeviceBuffer>` (`KvCacheRowsSnapshot`) — freed via their own `free_gpu`.
- Escaping return tensors (`vision_forward` `out`, `forward_logits_gpu` `logits`,
  qwen35 `forward`'s by-value `x`) — stay plain `alloc_tensor` so they aren't
  enqueued; the consumer owns them.
- Persistent hoisted decode/state scratch, KV caches, views, weights.

## Validation TODO (run on halo — gfx1151)

- `./tests/coherence-gate-dflash.sh` — confirm numerics unchanged (the migration is
  allocation-lifetime only; kernel order/dtypes/pooled-alloc-order preserved).
- Peak-VRAM under a **cached replay graph**: `reclaim_pending` is a deliberate no-op
  while `graph_state_live()`, so transient scratch allocated during a graph-cached
  session is held until graphs invalidate. Bounded, but measure on the 8 GB-class
  targets; if it bites, those specific sites can keep manual frees, or drain at
  graph-invalidation points.
- Per-iteration reclaim holds ~2 scratch sets in flight (constant, not depth-growing);
  acceptable, tighten with explicit pre-reclaim `drop` only if profiling shows it.

## Follow-ups

- deepseek4 `attention_block_*` `debug_max`/`debug_sumexp` (under
  `HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1`) is a pre-existing dev-gated pool leak;
  out of scope for this pass.
- New per-call GPU scratch should use `alloc_owned`/`zeros_owned`/`upload_owned_f32`
  + `reclaim_pending()` at the boundary — not `alloc_tensor` + manual `free_tensor`.
