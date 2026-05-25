# Path D — Stale-Context Overlap Pipelining

Implementation plan for issue [#38](https://github.com/Kaden-Schutt/hipfire/issues/38).
Branch: `feat/38-ddtree-pipeline` off master `80330c3`.

> **Revision note (2026-05-02):** plan v2, after consolidated adversarial
> review (`plans/path_d_plan_rev_*.md`). v1 treated D3 as a single
> ~250 LOC change layered on existing primitives. Verification against
> the codebase showed the relevant primitives (`commit_staging_to_ring`,
> `scatter_hidden_block_to_interleaved`, `gpu.active_stream` routing,
> `verify_scratch` reuse) are explicitly designed against concurrent
> execution. Path D now consists of a **Phase D0** of primitive
> refactors landed as separate PRs, then the original D1–D5 work as the
> Path D PR proper. Env var renamed `HIPFIRE_DDTREE_PIPELINE` →
> `HIPFIRE_DFLASH_PIPELINE` (DDTree-prefix was misleading; pipelining
> applies to base DFlash too).

> **Status note (2026-05-06):** Path D is **empirically dominated** on every
> tested model regime. See [`findings/path-d-vs-path-c.md`](../../findings/path-d-vs-path-c.md)
> + sibling `findings/path-d-*.md` (Qwen3.5-27B / gfx1100 / RX 7900 XT, PR #149)
> and [`findings/path-c-qwen36.md`](../../findings/path-c-qwen36.md) (Qwen3.6-27B
> / gfx1151 / Strix Halo APU, #151). PR #131 closed not-planned; no code from §3
> below landed on master. The plan is preserved as historical context for future
> revivals (e.g. if a Qwen3.6-A3B-matching DFlash draft becomes available, or on
> a different hardware class), not as an active implementation target. §1
> quantitative target rebaselined below per #150.

## 1. Goal

Launch DFlash draft N+1 **before** verify N completes, using cycle N-1's hidden
state. Trade some τ for full-cycle parallelism.

Quantitative target (rebaselined 2026-05-06 per #150): **cycle 75 ms → 75 ms
(0 %), no perf claim.** Original target was `60 ms (-20 %)` if τ stays ≥ 90 %
of the PEP-8 + norm baseline (`τ=10.36`, ~199 tok/s on 7900 XTX); empirical
investigations (PR #131 / #151) showed both chain-mode model B and Path C
tree-mode pipelining are dominated by chain mode on the 27B-3.5 LRU regime
(gfx1100) AND the 27B-3.6 + matching draft regime (gfx1151). The target is
unreachable on every tested hardware/model pair via the approaches in §3.

**Caveat on the 20 % cycle target.** Bandwidth contention on shared GDDR6
(7900 XTX 576 GB/s) erodes the overlap: when verify reads ~350 GB/s of target
weights and draft wants ~100 GB/s, the draft slows under contention. Worst-case
analytical model says effective savings could be as low as ~4 %, not 20 %. The
target therefore depends on whether draft and verify are bandwidth-bound vs
compute-bound *concurrently*. **Pre-D0 risk-check (see §3 below) runs a
bandwidth-contention micro-bench on master to refit this number before any
Path D code lands.** Ship-gate: speed gate at ≥ 210 tok/s median on code, plus
per-class gates for instruct and prose; if observed cycle savings drop below
8 % we pause and rescope.

Acceptance bars (verbatim from issue, plus per-class refinements):
- `coherence-gate-dflash.sh` passes with `HIPFIRE_DFLASH_PIPELINE=1`.
- Mean τ within 90 % of pre-pipeline baseline **per prompt class** (code,
  instruct, prose). Per-class because CLAUDE.md prompt-structure rule shows τ
  varies ~17 % across classes and a single average can mask a code-class
  regression behind a prose-class win.
- Speed (`max=120 --no-chatml --kv-mode asym3`, PEP-8 + norm prompt) on
  27B-3.5 LRU code: **≥ 210 tok/s median** (current 199–202).
- Speed on instruct: **≥ 195 tok/s median**; prose: **≥ 180 tok/s median**.
  These are the existing per-class baselines × ~0.98, allowing for τ
  regression while still requiring net wins.
- `HIPFIRE_DFLASH_PIPELINE=0` (default) within ±2 % of pre-change perf on all
  three classes.
- Default OFF until 5 full-matrix benches pass all gates.

## 2. The staleness model

Today (sequential), cycle N is:

```
  [ draft N (uses hidden after commit N-1) ] → [ verify N ] → commit hidden N
                                ^ depends on
```

Path D shifts the dependency by one cycle:

```
  cycle N-1: [ draft N-1 ] → [ verify N-1 ] → commit hidden N-1
  cycle N:   [ draft N (sees commit N-1) ]
                                            ‖ overlap
             [ verify N ] → commit hidden N
  cycle N+1: [ draft N+1 (sees commit N-1, NOT N) ]   ← THIS is the staleness
```

Draft N+1 launches as soon as draft N finishes (on `draft_stream`), reading
`hidden_rb`'s ring head as it stood after commit N-1. Verify N runs in parallel
on `gpu.active_stream` (which is the verify stream). By the time draft N+1
actually needs the context layers the kernels read, only commit N-1 is visible
— commit N is mid-flight.

Why this is OK: draft predictions are already a noisy approximation; missing
one cycle of context costs τ but not correctness. Cycle N's commit lands on
`hidden_rb` before draft N+2 ever reads it, so the staleness window is
exactly one cycle. Multi-step pipelining (N+2, N+3) is **out of scope** —
mis-prediction blast radius compounds.

Verify N still feeds the commit logic on the main stream — accepted-prefix
length, RoPE position increment, and target-stream KV cache writes all run
unchanged. Only the *draft* leg moves to a side stream.

### 2.1 Snapshot mechanism (the missing half of v1's diagram)

The "draft N+1 sees commit N-1" guarantee has two halves: a GPU-side half
(stream events) and a **CPU-side half** (snapshot of the ring-buffer
bookkeeping). The CPU-side half is load-bearing because `HiddenStateRingBuffer`
holds `head: usize` and `written: usize` as plain CPU mutable state
(speculative.rs:1124-1125). `commit_staging_to_ring` advances both
host-side at line 1335-1336.

```
At cycle N entry, on the host (before any verify-leg work begins):
  snap = (
    hidden_rb.head,                       // copy
    hidden_rb.written,                    // copy
    hidden_rb.target_hidden_abs_positions.clone(),  // shallow clone
  )

The draft N+1 launch uses ONLY snap.* values for slot indices and
position arrays. The verify leg may freely advance the live head/written
during its commit because the draft never reads them.
```

This snapshot is what makes "draft sees pre-commit state" deterministic.
Without it, the draft and verify race on a single mutable `usize`.

### 2.2 GPU-side ordering: events on stream-async primitives only

The pre-commit-event trick assumes every memory op involved is async-on-stream.
HIP cross-stream events do **not** order operations on the null stream. Today,
both `commit_staging_to_ring` and `scatter_hidden_block_to_interleaved` use
sync `memcpy_dtod_at` on the null stream. Phase D0 (§3) refactors both to
async variants taking an explicit `&Stream`, before any pipelining logic lands.

The cycle structure assumes those refactors are in place:

```text
verify_stream (= gpu.active_stream):
  pre_commit_evt = record()           // event captures pre-commit position
  commit_staging_to_ring_async(..., stream=verify_stream)

draft_stream:
  stream_wait_event(pre_commit_evt)   // pin draft launch to pre-commit snap
  scatter_hidden_block_to_interleaved_async(..., stream=draft_stream)
  draft_forward(scratch_cur, ..., stream_override=Some(draft_stream))
  draft_done_evt = record()

verify_stream:
  stream_wait_event(draft_done_evt)   // verify N+1 cannot start until draft N+1 done
  ...
```

### 2.3 Bypass conditions (when pipelining is disabled per cycle)

Pipelining requires the previous cycle to have left a clean hidden state on
`hidden_rb`. The following per-cycle conditions force fall-through to
sequential `spec_step_dflash`:

- **Cycle 0** (first decode iteration after prefill): no commit N-1 to observe;
  `hidden_rb` was seeded directly by `seed_target_hidden_from_prompt`, not by
  a prior cycle's commit. Run sequentially.
- **Previous cycle used PLD**: per `dflash_spec_demo.rs:1108-1122`, when PLD
  provides a spine the draft leg is a CPU-only lookup (no GPU draft forward,
  no `hidden_rb` GPU-side reads). The pipelined "draft N+1" has nothing to
  overlap. Run cycle N+1 sequentially.
- **Graph capture is enabled** (`HIPFIRE_GRAPH=1`): two concurrent capture-mode
  streams aren't supported by HIP. At session init, if both `HIPFIRE_GRAPH=1`
  and `HIPFIRE_DFLASH_PIPELINE=1` are set, log a clear warning and force
  pipelining off for the session. Graph wins.

These checks are cheap (a handful of bools at cycle entry).

## 3. Phase breakdown

> Path D now ships in two stages. **Phase D0** is a set of primitive refactors
> that land as separate PRs *before* Path D — each is byte-exact at default
> config, runs the existing coherence-gate and speed-gate, and reduces D3 from
> a 700-LOC mega-commit to a focused orchestration commit. **Phase D1–D5** is
> the Path D PR proper, on top of D0.

### Phase D0 — Primitive refactors (pre-PR work)

These are independent, byte-exact at default config, and each runs
coherence-gate + speed-gate before merge. Order doesn't matter; run in parallel
if convenient.

#### D0a — `commit_staging_to_ring` async-on-stream

**File:** `crates/engine/src/speculative.rs:1302-1338`

Today the function host-blocks on `stream_synchronize(active_stream)` (line
1311) and then issues sync `memcpy_dtod_at` calls (lines 1316-1333), all on
the null stream. New signature:

```rust
pub fn commit_staging_to_ring_on_stream(
    &mut self,
    gpu: &Gpu,
    n: usize,
    stream: &Stream,
) -> HipResult<()>;
```

Implementation: drop the host-blocking `stream_synchronize` (the caller is
responsible for stream ordering via events); replace each `memcpy_dtod_at`
with `memcpy_dtod_async_at(..., stream)` (ffi.rs:810). Update host-side
`head`/`written` mutation to happen *after* the async enqueues (so callers
that snapshot before calling see the pre-commit values).

Keep the existing `commit_staging_to_ring` as a thin wrapper that picks
`gpu.active_stream` and calls the new function — no caller change at default.

Audit callers (speculative.rs:2057 inside `verify_dflash_block_inner`,
qwen35.rs:3058 in chunked prefill) to confirm none rely on the old host-sync
semantics. If any does, mark explicitly and add a `gpu.hip.stream_synchronize`
call at the call site instead of inside the helper.

**Acceptance:** byte-exact output on coherence-gate and coherence-gate-dflash;
within ±2 % on speed-gate (per-class).

#### D0b — `scatter_hidden_block_to_interleaved` async-on-stream

**File:** `crates/engine/src/speculative.rs:2238-2273`

Today the function uses sync `memcpy_dtod_at` (line 2263) inside a 2D loop.
Add a new variant:

```rust
pub fn scatter_hidden_block_to_interleaved_on_stream(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
    head_snapshot: usize,                 // explicit, not read from hidden_rb.head
    stream: &Stream,
) -> HipResult<()>;
```

Implementation: replace `memcpy_dtod_at` with `memcpy_dtod_async_at(..., stream)`.
Take `head_snapshot` as an explicit parameter so the pipelined caller can pass
the pre-commit head; the existing sync wrapper passes `hidden_rb.head`
unchanged.

**Acceptance:** byte-exact at default; same gates as D0a.

#### D0c — Thread `stream_override: Option<&Stream>` through `draft_forward`

**File:** `crates/engine/src/dflash.rs:663` plus every kernel/memset callee.

Today `draft_forward(gpu: &mut Gpu, ...)` rides on `gpu.active_stream`
implicitly. Add a `stream_override: Option<&Stream>` parameter and route every
kernel launch / async memcpy / async memset that currently reads
`gpu.active_stream` through the override when provided. When `None`,
behavior is unchanged (use `gpu.active_stream`).

This is the largest of the three D0 commits. Touch points (non-exhaustive,
to be expanded during implementation):

- `dflash.rs:663` — `draft_forward` signature.
- `dflash.rs:699-700` — `upload_slice_f32` already takes a stream-aware path
  via `active_stream`; either thread the override or accept it falls through
  to the override-aware lower layer.
- All `gpu.dispatch_*` and `gpu.launch_*` callsites inside `draft_forward`.
- The `memset_async` helper (CLAUDE.md memset-pressure note: gated on
  `active_stream.is_some()`) — needs an override-aware variant.

**Acceptance:** byte-exact at `stream_override = None` (the default everywhere
post-D0c); coherence-gate green; speed-gate within ±2 %.

#### D0-bandwidth — Bandwidth-contention micro-bench (validation, no merge)

Before D1 lands: run a synthetic bandwidth-contention micro-bench on master
that mimics the real overlap (verify-sized weight read on one stream + draft-
sized weight read on a second stream, both on draft model). Report observed
overlap factor. If observed cycle savings under simulated overlap are < 8 %,
**pause and rescope Path D** — the 210 tok/s target is unreachable on this
hardware regardless of correct stream wiring.

Output: `findings/path-d-bandwidth-contention.md`. Not a merged commit; a
gating measurement.

### D1 — Lazy stream allocation (~30 LOC, byte-no-op)

**File:** `crates/rdna-compute/src/dispatch.rs:226-227, 434-435`

Phase A added `pub draft_stream: Option<Stream>` and
`pub verify_stream: Option<Stream>` on `Gpu`, both initialized to `None`.

**Update from v1:** *delete* `verify_stream` from `Gpu`. The verify stream is
just `gpu.active_stream` by convention; introducing a separate field is
impossible (the `Stream` type at ffi.rs:937 is not `Clone` and not safely
duplicable — it owns the underlying `HipStream` and has single-ownership
destroy semantics) and unnecessary (cross-stream events between `draft_stream`
and `active_stream` express what we need).

Add:

- `Gpu::init_pipeline_streams(&mut self) -> HipResult<()>` — idempotent;
  creates `draft_stream` if `HIPFIRE_DFLASH_PIPELINE=1` is set, leaves it
  `None` otherwise. Name matches the existing doc-comment at dispatch.rs:224.
- Call site: top of `spec_step_dflash` next to the existing
  `active_stream.is_none()` check at speculative.rs:2429.
- After creating `draft_stream`, run a one-pass DPM warmup on it (mirror of
  the existing warmup loop on `active_stream`). This ensures the draft stream
  is not the first to hit DPM throttling on cycle 1. Cheap (one extra warmup
  pass).
- Add `Drop` impl on `Gpu` that destroys `draft_stream` (and `active_stream`)
  on drop, plus an explicit cleanup hook called from `unload_model` so model
  swaps in the long-running daemon don't leak streams.

Acceptance for D1 alone: existing benches unchanged with env unset; with env
set, `gpu.draft_stream.is_some()` becomes true after the first
`spec_step_dflash` call. No kernel routing yet — purely scaffolding.

### D2 — DflashScratchPair + draft lm_head scratch

**File:** `crates/engine/src/dflash.rs:304-393`

Path D needs two `DflashScratch` instances so cycle N can write while cycle
N+1 reads. Add:

```rust
pub struct DflashScratchPair {
    pub a: DflashScratch,
    pub b: DflashScratch,
    /// Even cycles use `a`, odd use `b`. Toggled by spec_step_dflash_pipelined.
    pub parity: bool,
    /// Dedicated lm_head scratch for the draft leg, sized to
    /// max_block × max(vocab, q_dim). Avoids racing on
    /// `verify_scratch.logits` / `verify_scratch.rot` when draft N+1's
    /// lm_head runs concurrently with verify N.
    pub draft_lm_head_logits: GpuTensor,
    pub draft_lm_head_rot: GpuTensor,
}

impl DflashScratchPair {
    pub fn new(gpu, cfg, max_block_size, max_ctx_len, vocab) -> HipResult<Self>;
    pub fn current(&mut self) -> &mut DflashScratch;
    pub fn previous(&mut self) -> &mut DflashScratch;
    pub fn flip(&mut self);
    /// Free `b` when bypass activates for the rest of a request.
    pub fn drop_unused_half(&mut self);
}
```

**Why dedicated lm_head scratch:** the existing `spec_step_dflash` reuses
`verify_scratch.logits` and `verify_scratch.rot` for the draft's lm_head
(speculative.rs around 2634-2648). In the sequential path this is safe because
draft completes before verify starts. In the pipelined path, both legs would
write the same buffers concurrently — a flat data race. Dedicated draft-side
scratch eliminates the race; the cost on 27B is roughly
`max_block × vocab × 4 B = 16 × 152k × 4 = 9.7 MB` for logits plus a few MB
for rot — small relative to the rest of the pair.

**Memory cost — config-sensitive, not a single number.** Per-scratch size
scales linearly with `max_ctx_len × layers × hidden`. For the plan's example
config (5-layer draft, B=16, L=512, h=2048, kvd=256, ne=5) one scratch is
~30 MB; the pair is ~64 MB. For a 16-layer draft at max_ctx=4096 (cited as
the worst case in the existing in-code comment at dflash.rs:437-438), one
scratch can exceed 500 MB and the pair pushes 1 GB. **Mitigation:**

1. **Gate allocation behind the env flag** — `DflashScratchPair::b` (the
   second scratch) is allocated only when `HIPFIRE_DFLASH_PIPELINE=1`.
   At default, only `a` exists and memory cost matches the pre-Path D
   `DflashScratch`.
2. **Startup VRAM headroom check** — before allocating `b`, query free VRAM
   and compare against pair size + safety margin (200 MB). If insufficient,
   refuse pipelining at runtime (force env=0 with a warning) rather than
   OOM-crash mid-request.
3. **Bypass deactivation frees the second half** — when D4's adaptive bypass
   triggers "for the rest of the request", call `drop_unused_half()` to
   reclaim the 30 MB+ that pair `b` is holding.

`k_ctx_cached` and `v_ctx_cached` are the largest line items
(`Vec<GpuTensor>` of `[max_ctx, kv_dim]` per layer). For the pipelined path,
each scratch keeps its own cache — they diverge by exactly one cycle's
worth of appended rows, so we can't share.

`uploaded_target_hidden_rows` and `draft_ctx_cached_rows` (dflash.rs:352, 392)
are per-scratch state already; nothing to refactor.

Single-scratch callers (`spec_step_dflash` non-pipelined) keep working with
plain `DflashScratch` — the pair is additive.

### D3 — Pipelined orchestration (split into D3a/D3b)

The original v1 D3 was a single ~250 LOC commit. After verification (the
mirrored function would be ~700 LOC, plus several blocking primitive issues),
this is split into two PR commits on top of the D0 refactors.

#### D3a — Hoist commit out of `verify_dflash_block_inner`

**File:** `crates/engine/src/speculative.rs:2057`

The commit at speculative.rs:2057 lives **inside** `verify_dflash_block_inner`,
not at the top-level cycle scope. The pipelined cycle structure needs to
record `pre_commit_evt` *between* the verify forward and the commit, which is
impossible while the commit is buried inside the verify helper.

Change: add a `skip_internal_commit: bool` parameter to
`verify_dflash_block_inner`. When true, the function returns without calling
`commit_staging_to_ring` and the caller is responsible. Both
`spec_step_dflash` (non-pipelined) and the pipelined path benefit: the
pipelined path needs the explicit hoist; the non-pipelined path keeps the
default (skip_internal_commit=false) and is unchanged.

This is a small, byte-exact-at-default refactor. Land it before D3b.

#### D3b — `spec_step_dflash` mode flag (refactor, not mirror)

**File:** `crates/engine/src/speculative.rs:2384`

**Update from v1:** *do not mirror* `spec_step_dflash` into a sibling 700 LOC
function. Instead, refactor `spec_step_dflash` to take a `pipeline_mode: bool`
flag that gates the cross-stream behavior. One function, two modes.

Reasoning: the function is 779 lines. A mirror produces a ~700 LOC duplicate
that would drift; CLAUDE.md mandates coherence-gate on any kernel/dispatch
change, and gate failures from one variant lagging the other are a known
failure mode (e.g., 6c84b13 / f9c920a attractors). Refactor-with-flag is
strictly cheaper than two parallel functions with a deferred dedup commit.

Cycle structure when `pipeline_mode = true`:

```text
ENTRY (cycle N):
  // Bypass conditions per §2.3:
  if cycle == 0 || prev_used_pld || graph_capture_active { run sequential; return }

  scratch_cur  = pair.current()    // fresh-write target
  scratch_prev = pair.previous()   // last cycle's draft state, still valid

  // Snapshot ring-buffer bookkeeping BEFORE any verify-leg work (§2.1)
  snap = (hidden_rb.head, hidden_rb.written,
          hidden_rb.target_hidden_abs_positions.clone())

  [verify_stream = gpu.active_stream]
    1. verify N forward (skip_internal_commit=true)
    2. pre_commit_evt = record()
    3. commit_staging_to_ring_on_stream(verify_stream)

  [draft_stream]                       ← runs concurrent with verify
    4. stream_wait_event(pre_commit_evt)
    5. scatter_hidden_block_to_interleaved_on_stream(
         dst=scratch_cur.target_hidden, head_snapshot=snap.head, stream=draft_stream)
    6. draft N+1 forward on `scratch_cur`, with stream_override=Some(draft_stream),
       writing to pair.draft_lm_head_logits / pair.draft_lm_head_rot
    7. draft_done_evt = record()

  [verify_stream]
    8. stream_wait_event(draft_done_evt)
       — next cycle's verify cannot start until this cycle's draft N+1 done
    9. CASK eviction mirror, deferred behind draft_done_evt (§A6)
   10. commit accepted prefix; build SpecStepResult
   11. pair.flip()
```

**Read-after-write hazards covered:**

- `hidden_rb.layer_bufs`: draft_stream reads via async scatter pinned behind
  `pre_commit_evt`. Verify_stream's commit writes to disjoint slots
  (snap.head .. snap.head + n) in stream order after pre_commit_evt is
  recorded. Cross-stream events order both sides; D0a/D0b ensure both sides
  use stream-async copies (not the null stream).
- `hidden_rb.head`/`hidden_rb.written`: the draft uses `snap.head` /
  `snap.written` only. The live values may freely advance during verify's
  commit because the draft never reads them.
- `target_hidden_abs_positions`: snapshotted at cycle entry; CASK eviction
  mirror is deferred behind `draft_done_evt` so it cannot compact the array
  while draft is reading.
- Draft lm_head buffers: `pair.draft_lm_head_logits` / `.rot` are dedicated
  to the draft leg; verify uses `verify_scratch.logits` / `.rot`. No aliasing.

**Branching.** Mode is selected by the caller based on session state. Read
`HIPFIRE_DFLASH_PIPELINE` once at session init, cache on the session, pass
into `spec_step_dflash` as a parameter or carry on a per-session config
struct. Avoid per-cycle env reads.

**FFI.** `hip_bridge::Gpu::stream_wait_event` (ffi.rs:755) and
`event_record` (load at ffi.rs:268, method nearby) are already exposed
(commit `8e9b9aa`). No new FFI.

**Estimated touch:** D3b is the orchestration logic itself — ~150–250 LOC
inserted into `spec_step_dflash` as a `pipeline_mode` branch, mostly the
event/wait choreography and the snapshot-and-flip plumbing. Not a duplicate
of the 779 LOC core.

### D4 — Adaptive bypass (τ guard)

**File:** `crates/engine/src/speculative.rs` (within `spec_step_dflash`,
pipelined branch).

Telemetry already tracks accept-length per cycle (drives adaptive-B). Add a
3-cycle rolling window:

```
if rolling_tau_3 < expected_pipelined_floor for 3 consecutive cycles:
    pair.drop_unused_half()                        // free 30 MB+ of VRAM
    fall back to sequential for the rest of the request
    log structured event for bench harness
```

**Baseline keying — refined.** v1 keyed by `(target_hash, draft_hash)`,
which averages over prompt classes that differ by ~17 % (CLAUDE.md
prompt-structure rule). Replace with:

- Key: `(target_hash, draft_hash, prompt_class)` where `prompt_class ∈
  {code, instruct, prose}` is detected at request start (cheap heuristic on
  prompt text) or supplied by the caller.
- Seed from **5 cycles**, not 20. Per Gemini §4: 20 × 75 ms = 1.5 s of every
  request lost to seeding before pipelining activates; 5 cycles brings that
  to ~375 ms, tolerable for short agentic turns.
- Cache the baseline **across requests** for the same model — the first
  request pays the seeding cost, subsequent requests inherit. Reset on model
  swap.

**Adaptive-B coupling — τ-debt adjustment.** Adaptive-B and pipelining
bypass both react to rolling τ. To prevent thrashing:

- During pipelined operation, adaptive-B's τ target is `baseline × 0.90`
  (the pipelining floor), not the sequential baseline. This way adaptive-B
  shrinks B only if real τ drops below the *expected* pipelined floor, not
  every time τ dips below the sequential reference.
- Bypass triggers only after adaptive-B has had 3 cycles to react (i.e.,
  bypass requires 3 consecutive cycles of low τ *after* adaptive-B has
  stabilized). Worst case: ~6 cycles before bypass — for ≤ 60-token
  generations, bypass may never fire; that's acceptable, cost is bounded by
  the 10 % τ floor.
- B-changes lag one cycle in pipelined mode (cycle N+1's draft was launched
  with cycle N's B, before cycle N's verify could decide to shrink). Document
  this; not a bug.

Reset on accept of a new prompt boundary (in long-running daemon, "new prompt
boundary" = new chat turn / new request id).

### D5 — Gates & ship

Sequence (each step blocks on the previous):

1. Run `./scripts/coherence-gate-dflash.sh` with `HIPFIRE_DFLASH_PIPELINE=0`
   — confirm no regression vs master `80330c3` post-D0.
2. Run with `HIPFIRE_DFLASH_PIPELINE=1` — must pass all three tiers
   (first-128 unique-token-ratio ≥ 0.15 + max-freq ≤ 0.50; last-128 ≥ 0.30
   + ≤ 0.50; full-output 3gram-density flag for human review). Per CLAUDE.md
   coherence rules. **Run 5×** and collect:
   - byte-exact diff across runs (must be byte-equal — staleness mechanism is
     deterministic; non-determinism here is a bug)
   - τ stddev across runs vs sequential's τ stddev — narrowing > 30 %
     is a hard fail (CLAUDE.md spec-decode rule: tight stddev is suspicious,
     not reassuring; it's a known attractor signature).
   - **Threshold-relaxation is forbidden.** If the gate fails by < 5 %, do
     NOT widen thresholds. Investigate root cause OR revert the chain.
     Decision owner: PR author + one reviewer.
3. `./scripts/probe_commits.sh master HEAD` for cross-process speed
   verification at env=0 (within-session A/B drifts ±10–15 % per CLAUDE.md
   perf-benchmarking rule). This validates speed parity at default, not the
   pipeline correctness — that's covered by step 2.
4. Bench matrix (5 runs each, fresh process, byte-identical PEP-8 + norm
   prompt — CLAUDE.md prompt-md5 rule):
   - 27B-3.5 LRU code (target ≥ 210 tok/s)
   - 27B-3.5 LRU instruct (target ≥ 195 tok/s)
   - 27B-3.5 LRU prose (target ≥ 180 tok/s)
   - 8B-3.5 (smaller draft sanity)
   - 27B-3.5 with `HIPFIRE_DFLASH_PIPELINE=0` (regression check, ±2 %)
   - 27B-3.5 with `max=2000` (ring-buffer wrap exercise — `max=120` does not
     wrap a 2048-slot buffer; wrap interaction with the staleness window is
     invisible at default bench length)
5. Update `docs/methodology/perf-benchmarking.md` with the new baseline
   numbers + a negative-result entry if any of D2/D3/D4 didn't land as
   projected. Include the bandwidth-contention finding from the D0
   risk-check.
6. Default `HIPFIRE_DFLASH_PIPELINE=0` in code; flip to `1` only after the
   matrix above is green AND the GPU lock protocol confirms no contention
   regressions (since pipelining changes stream usage patterns).

## 4. Concrete touch list

### Phase D0 (separate PRs, byte-exact at default)

| File | Change |
|------|--------|
| `crates/engine/src/speculative.rs:1302-1338` | D0a — `commit_staging_to_ring_on_stream(&Stream)` async variant; thin wrapper preserves old API |
| `crates/engine/src/speculative.rs:2238-2273` | D0b — `scatter_hidden_block_to_interleaved_on_stream(&Stream, head_snapshot)` async variant |
| `crates/engine/src/dflash.rs:663` | D0c — `stream_override: Option<&Stream>` parameter on `draft_forward`; thread through callees |
| `crates/engine/src/dflash.rs` (helpers) | D0c — propagate `stream_override` to upload_slice_f32 path, memset_async helper, kernel launches |

### Phase D1–D5 (Path D PR, on top of D0)

| File | Change |
|------|--------|
| `crates/rdna-compute/src/dispatch.rs:226-227` | D1 — delete `verify_stream` field; keep only `draft_stream` |
| `crates/rdna-compute/src/dispatch.rs:434-435` | D1 — `init_pipeline_streams()`, DPM warmup hook for draft stream, `Drop` cleanup |
| `crates/engine/src/dflash.rs:304-393` | D2 — `DflashScratchPair` struct (gated env=1 alloc, drop_unused_half, draft_lm_head buffers) + ctor |
| `crates/engine/src/speculative.rs` (verify_dflash_block_inner) | D3a — `skip_internal_commit: bool` parameter |
| `crates/engine/src/speculative.rs:2384` | D3b — `pipeline_mode: bool` branch in `spec_step_dflash`; snapshot-head, async scatter, cross-stream events |
| `crates/engine/src/speculative.rs` (top) | D4 — rolling-τ helper, per-class baseline cache, bypass state |
| `crates/engine/examples/dflash_spec_demo.rs` | D1–D5 — thread env-flag to scratch alloc + step selection |
| `crates/engine/examples/daemon.rs` | D1–D5 — same threading; daemon path also calls `spec_step_dflash` (line 1717) and was missed in v1 touch list |
| `crates/hip-bridge/src/ffi.rs` | (none — APIs already exposed) |
| `docs/methodology/perf-benchmarking.md` | D5 — bench-matrix results, baseline update, bandwidth-contention finding |

### Wait to touch

- **Phase B oracle leftovers** (speculative.rs:25-90 `SEED_ORACLE_*` globals,
  oracle proxy logic at 2948-2990, `HIPFIRE_DFLASH_SEED_ORACLE` env). These
  become dead code if Path D ships, but cleaning them up is not on the
  critical path for this PR. File a follow-up issue and link it from the
  Path D PR description; remove in a separate commit so the diff stays
  reviewable.
- **MoE on non-default stream.** MoE/A3B is refused at load time per
  AGENTS.md §0; the pipelined verify-on-active-stream and draft-on-draft-
  stream are untested for MoE targets. Defer to a follow-up issue. Don't
  block this PR on it.

## 5. Risks & open questions

1. **Memory headroom.** `DflashScratchPair` size is config-sensitive: ~64 MB
   at the plan's example (5-layer / max_ctx=512), can exceed 1 GB at
   16-layer / max_ctx=4096 (per the in-code comment at dflash.rs:437-438).
   Mitigation:
   - Gate `DflashScratchPair::b` allocation behind env=1 (D2 implements this).
   - Startup VRAM headroom check before allocating `b`; refuse pipelining at
     runtime if `available_vram - target_model_size < pair_size + 200 MB`
     safety margin.
   - On bypass activation, `pair.drop_unused_half()` reclaims 30 MB+.
   On a 24 GB 7900 XTX with 27B target loaded, the typical pair size is
   fine; on 16 GB W7700 the runtime check matters.

2. **Bandwidth contention erodes overlap.** Worst-case analytical model
   (verify reading ~350 GB/s of target weights, draft wanting ~100 GB/s on
   shared 576 GB/s GDDR6) suggests effective cycle savings could fall to
   ~4 % under contention rather than the 20 % theoretical max. Validated by
   the **D0 bandwidth-contention micro-bench** before any Path D code lands.
   If observed cycle savings under simulated overlap are < 8 %, pause Path D
   and rescope. Either Path D ships smaller perf wins or a different
   approach (e.g., async-prefetch + sequential overlap, not full pipelining)
   is preferred.

3. **Event/stream-wait overhead.** Each cycle adds 2 × `hipEventRecord` +
   1 × `hipStreamWaitEvent`. These are µs-scale on RDNA3 (~5–15 µs each,
   total 15–45 µs/cycle = 0.075 % of a 60 ms cycle) — negligible. Validated
   in D3b by running with `HIPFIRE_DFLASH_PIPELINE=1` but pipelining
   logically disabled (single scratch, both legs on `gpu.active_stream`) —
   the delta to baseline is the pure FFI overhead.

4. **τ regression sensitivity per prompt class.** The 90 % τ acceptance is
   tight, and one-cycle staleness costs more on code than on prose (recent
   context is more informative — a function definition just written is
   high-signal). The D4 adaptive bypass may trigger immediately on code
   prompts, negating the overlap benefit on the very class we most want to
   accelerate. Per-class speed gates in §1 acceptance bars enforce that
   *each* class must clear its target — averages cannot hide a code-class
   regression behind a prose-class win.

5. **PLD spine interaction.** `spec_step_dflash` shrinks `b` when
   `pld_spine` is `Some` (speculative.rs:2416). Path D reads context as of
   N-1; if PLD spines change rapidly between cycles, the staleness window
   may compound with PLD-driven block-size changes. **Resolved by §2.3
   bypass:** if cycle N used PLD, cycle N+1 runs sequentially. No overlap is
   gained on PLD cycles; that's acceptable since PLD already eliminates the
   draft GPU forward (the thing pipelining would overlap).
   Integration test: pipelined path + PLD spine over **200 cycles**
   (per CLAUDE.md attractor manifestation horizon ~150 cycles), assert no
   panic + τ within 85 % of non-pipelined PLD path. (This is an integration
   test, not a unit test — unit tests in this codebase don't have GPU.)

6. **Coherence-gate determinism.** Pipelining introduces a non-deterministic
   scheduling element (HIP stream interleaving). Output should remain
   deterministic because the staleness mechanism pins the draft's input to
   the pre-commit snapshot regardless of scheduling — but only if event
   ordering is correct. **Verification: run coherence-gate-dflash 5× with
   the same seed and diff outputs byte-exact.** A correct implementation
   must produce byte-identical output across runs. Additionally, compare
   τ stddev to the sequential 5×: narrowing > 30 % is a hard fail
   (CLAUDE.md: tight stddev is an attractor signature).

7. **RDNA3 cross-stream L2 visibility.** Likely safe in practice — the
   disjoint-slot argument (draft reads slots [0..pre_commit_head), commit
   writes [pre_commit_head..pre_commit_head+n)) means no single cache line
   is read+written by both streams. But "likely" isn't proof. The 5×
   byte-exact determinism check above doubles as the L2-coherence
   verification; if there's a coherence bug, byte-exact will fail.

8. **Adaptive bypass interaction with adaptive-B.** Both D4 and the
   existing adaptive-B logic key off rolling τ. **Resolved by τ-debt
   adjustment in D4:** during pipelined operation, adaptive-B's τ target is
   `baseline × 0.90` (the pipelining floor), so it shrinks B only when real
   τ drops below the *expected* pipelined floor. Bypass triggers only after
   adaptive-B has had 3 cycles to react. Worst case: ~6 cycles before
   bypass; for ≤ 60-token generations, bypass may never fire — acceptable,
   τ floor bounds the cost.

9. **CASK eviction races with pipelined draft.** CASK-driven `target_hidden`
   compaction runs from the verify leg's post-commit path; the pipelined
   draft N+1 reads `target_hidden` on draft_stream. **Resolved by D3b:**
   eviction mirror is deferred behind `draft_done_evt` on verify_stream, so
   compaction cannot run concurrently with the draft read. Tested by D5
   matrix's CASK-active configs.

10. **Graph capture incompatible.** Two concurrent capture-mode streams
    aren't supported. **Resolved by §2.3 bypass:** at session init, if
    `HIPFIRE_GRAPH=1` and `HIPFIRE_DFLASH_PIPELINE=1` are both set, log a
    warning and force pipelining off for the session. Graph wins.

## 6. Out of scope

- Multi-step pipelining (N+2, N+3).
- Cross-arch validation beyond gfx1100. BC-250 / MI300X is a separate PR.
- Removing Phase B oracle instrumentation (separate follow-up issue).
- MoE/A3B on the pipelined path (deferred — MoE refused at load time today
  per AGENTS.md §0; pipelined verify-on-active-stream is untested for MoE).
- Vulkan / RADV path. Out of scope per CLAUDE.md (issue #44 closed).
- Refactoring `spec_step_dflash` to dedupe pipelined and sequential code
  paths beyond the `pipeline_mode` flag. The flag covers it; deeper dedup
  isn't needed.
- Tree-mode pipelining on `spec_step_ddtree_path_c`. Empirically dominated
  by chain mode on Qwen3.5-27B / gfx1100 (`findings/path-d-vs-path-c.md`)
  and Qwen3.6-27B / gfx1151 (`findings/path-c-qwen36.md`). The original
  hedge "(e.g. Qwen3.6 + matching draft per #41 comment data) might let
  tree mode win" has been empirically refuted on the testable variants.
  Only worth pursuing if a *new* model regime is shown by gates to have
  Path C beating chain mode on chain-mode default — Qwen3.6-A3B is the
  only untested target regime, blocked on DFlash draft availability.

## 7. PR shape

### Phase D0 — three separate PRs (byte-exact at default)

Each runs coherence-gate + speed-gate at default config. None touch behavior
at `stream_override = None` / sync API.

1. `infra(spec): commit_staging_to_ring async-on-stream variant` — D0a.
2. `infra(spec): scatter_hidden_block_to_interleaved async-on-stream variant` — D0b.
3. `infra(dflash): thread stream_override through draft_forward` — D0c.

Plus the **non-merged** D0 bandwidth-contention micro-bench that lands as
`findings/path-d-bandwidth-contention.md` and gates whether D1 begins.

### Path D PR proper — 7 commits aligned to D1, D2, D3a, D3b, D4, D5a, D5b

1. `ddtree(path-d D1): lazy pipeline stream init + Drop cleanup` —
   dispatch.rs only. Drops `verify_stream` field; renames helper to
   `init_pipeline_streams`.
2. `ddtree(path-d D2): DflashScratchPair with env-gated 2nd half + draft lm_head scratch` —
   dflash.rs only.
3. `ddtree(path-d D3a): hoist commit out of verify_dflash_block_inner` —
   speculative.rs only. Byte-exact at default.
4. `ddtree(path-d D3b): spec_step_dflash pipeline_mode flag + cross-stream orchestration` —
   speculative.rs + demo/daemon dispatch.
5. `ddtree(path-d D4): adaptive τ-guard bypass + per-class baseline + τ-debt adjustment` —
   speculative.rs.
6. `ddtree(path-d D5a): bench matrix run + perf-benchmarking.md update` —
   docs only.
7. `ddtree(path-d D5b): flip default to env=1 after green matrix` —
   one-line config change, separable so a regression discovery in 6 doesn't
   require reverting the entire chain.

Each commit must independently compile and pass `cargo test -p engine`.
Coherence-gate runs at commits 3, 4, 5, and 6 (CLAUDE.md pre-commit hook
catches all dispatch/spec-decode files automatically). Speed-gate at 6.

**Rollback criteria.** If commits 3–5 fail any gate, do NOT widen
thresholds (CLAUDE.md spec-decode rule). Investigate root cause OR revert
the chain. Decision: PR author + one reviewer. If commit 6 (default flip)
shows a regression in the wild, revert 6 alone and leave 1–5 as
opt-in scaffolding.
