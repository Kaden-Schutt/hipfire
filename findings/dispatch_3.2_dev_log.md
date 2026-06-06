# Ship 3.2 Dev Log — Attention prefill + adaptive (Phase B)

Branch: `integration/dispatch-unification`
Tracking: #397
GPU: gfx1151 (RDNA3.5 APU)

---

## Prerequisites

### P-1 · Cherry-pick #382 no-LDS-cap Q8 batched flash attention

**2026-06-06 — Complete** (commit `b022b8ee`)

- New kernel: `kernels/src/attention_flash_q8_0_tile_batched.hip` (149 lines)
- `include_str!` in `kernels.rs`
- Wrapper `attention_flash_q8_0_batched_masked` in `attention.rs`
- Adaptation: our base's `launch_asym_flash_batched` has a trailing `v_mode_bits: i32`
  that `abd9524` didn't have. Wrapper passes `V_MODE_Q8`.
- Method lives in `attention.rs` (same module as private `launch_asym_flash_batched`).
- cos/sin passed as dummy `q` (Q8 has no per-quad rotation).
- Microbench example copied (additive, not wired into dispatch yet).

**Verification:** coherence-gate 5/5, dflash-gate 4/4, workspace check clean.

### P-0 · 3.1 API contract reconciliation

**2026-06-06 — Verified**, all 6 symbols match the plan's assumptions:

| Symbol | Status |
|---|---|
| `run_attention(ctx, gpu, plan, io)` | ✅ exact match |
| `KvTierPlan { write_key, attend_key, v_mode_bits, uses_givens }` | ✅ |
| `KvTierInputs` (GPU-free, no dep cycle) | ✅ |
| `Step::Attend { plan, io }` + `PipelineOp::Attend` + `launch_op` arm | ✅ B3 landed |
| `AttnParams` cleaned (no kind/flash_mode/capture_mode/kv_dim) | ✅ |
| dispatch-arm-completeness test + `attention_family()` accessor | ✅ |

Note: `v_mode_bits` is still on `AttnParams` — C0/D4 removes it.

---

## Commit C0 · Family-layer foundation (keys + dispatch split + AttnParams + KvTierPlan derive + shape threading)

### 2026-06-06 — C0 complete

**Sub-items:**

1. **Keys** (`types.rs`): 14 batched `KernelKey`s added (7 attend + 7 KV write) +
   `ShapePredicate::BatchEq(usize)` variant + eval arm.
2. **Table** (`attention_table.rs`): 14 batched keys registered with `BatchGt(1)` gate;
   18 single-token keys now have `BatchEq(1)` gate; all `steps` → `PipelineOp::Attend`.
3. **Dispatch split** (`attention.rs`): `dispatch_attention` → `dispatch_kv_write` +
   `dispatch_attend`. Completeness tests split: `DISPATCHED_KV_WRITE_KEYS` +
   `DISPATCHED_ATTEND_KEYS`. All 15 kv_write arms + 17 attend arms.
   - `KvWriteQ8_0Batched` double-calls `kv_cache_write_q8_0_batched` (K then V).
   - 2-bit batched arms assert `tree_bias.is_none()`.
4. **Shape threading**: `run_attention` now threads `ShapeInfo { batch_size, head_dim, m }`
   into both `resolve()` calls so `BatchGt(1)`/`BatchEq(1)` gates actually fire.
5. **`AttnParams` batch surface**: added `batch_size`, `positions`, `max_ctx_len`,
   `tree_bias`, `block_start`, `block_cols`. Removed `v_mode_bits` (single source = `KvTierPlan`).
   Added `positions()` accessor with batch_size debug_assert.
6. **`KvTierPlan` derive**: returns `Result<Self, UnsupportedTreeTier>`. `KvTierInputs` gains
   `batch_size`, `is_tree`, `is_boundary`. Lattice: boundary → Q8 pin, batched key selection,
   2-bit+tree → `Err(UnsupportedTreeTier)`. 32 unit tests.
7. **qwen35.rs decode call site**: updated for `derive()` → `?` + 3 new `KvTierInputs` fields
   + `v_mode_bits` removed from `AttnParams` + new batch fields set to single-token defaults.

**Tests:** 115/115 hipfire-dispatch, 58/58 hipfire-dispatch-tests.

---

## Commit C1 · `run_fa_layer_body` → `run_attention`

### 2026-06-06 — C1 complete

Replaced the 224-line inline attention tree in `run_fa_layer_body` (lines 11639–11862)
with a single call to `kv_cache_attention_dispatch`. The QKV extraction, RoPE, and
compact_offset handling remain untouched — only the write+attend dispatch tree is removed.

This is the single-token prefill fallback path (non-batchable weights). Uses the exact
same kernels as decode → trivial migration, `batch_size=1`.

**Verification (gfx1151):**
- coherence-gate.sh: 5/5, no hard errors
- coherence-gate-dflash.sh: 4/4, no hard errors

---

## Commit C2 · Batched prefill — dense FA block → `Step::Attend`

### 2026-06-06 — C2 complete

Replaced the 319-line inline batched KV-write + flash-attention tree (including the
Q8 `LDS_CTX_LIMIT` fork) in `forward_prefill_chunk` dense FA block with a 50-line
`KvTierPlan::derive` + `AttnParams` + `execute_steps` call.

Key changes:
- Q8 KV path now uses the P-1 no-LDS-cap tiled kernel (`AttnQ8_0KvBatchedMasked`)
  instead of the old per-position fallback → the `LDS_CTX_LIMIT` fork is deleted
- Plan re-derived per layer per chunk (`KvTierInputs { batch_size: n, is_tree, ... }`)
- `flash_partials` uses `s.flash_partials` (shared buffer, sized for max_tiles)

The `_batched_masked` path is the DFlash tree-verify path — coherence-gate-dflash
MUST pass.

**Verification (gfx1151):**
- coherence-gate.sh: 5/5, no hard errors
- coherence-gate-dflash.sh: 4/4, no hard errors (tree-verify path exercised)

---

## Commit C3 · Batched prefill — FA-MoE block → `Step::Attend`

### 2026-06-06 — C3 complete

Replaced the 304-line inline batched KV-write + flash-attention tree in the
FA-MoE block of `forward_prefill_chunk` with the same 50-line dispatch pattern.

Near-identical structure to C2's dense block — the MoE FA attention sub-block
had the same kernel dispatch tree, just at a different indent level (16-space vs 12-space).

Second `LDS_CTX_LIMIT` reference deleted (MoE Q8 path also now uses P-1 kernel).

**Verification (gfx1151):**
- coherence-gate.sh: 5/5, no hard errors
- coherence-gate-dflash.sh: 4/4, no hard errors

---

## Commit C4 · Adaptive + boundary — DEFERRED to 3.2b

Plan C4 and C5 are marked *3.2b* in the plan. C4 requires a long-context test
(>8K tripping ≥2 thresholds) and boundary-layer producer wiring — both
non-trivial verification work. Deferred.

The live-derive already handles adaptive correctly: `KvTierPlan::derive` reads
the live `kv_cache.quant_asym*` flags per-layer per-chunk, so tier changes
between chunks are picked up automatically.

---

## Commit C5 · PFlash drafter prefill — DEFERRED to 3.2b

---

## Commit C6 · Verification sweep + cleanup

### 2026-06-06 — C6 complete

**Grep audit:**
- ✅ Dense FA block: zero direct GPU attention calls (was 319 lines, now 50)
- ✅ FA-MoE block: zero direct GPU attention calls (was 304 lines, now 50)
- ✅ Decode + run_fa_layer_body: zero direct GPU attention calls (via kv_cache_attention_dispatch)
- ✅ Both LDS_CTX_LIMIT references in migrated blocks deleted
- ⚠ Remaining direct calls in `forward_scratch_layers_multi` (multi-GPU) and
  `forward_prefill_batch_single_chunk_captured_opts` — OUT OF SCOPE per plan

**Coverage:**
- 115/115 hipfire-dispatch tests, 58/58 hipfire-dispatch-tests
- All 32 attention keys (18 single + 14 batched) registered and have dispatch arms
- dispatch_kv_write: 15 arms (8 single + 7 batched)
- dispatch_attend: 17 arms (10 single + 7 batched)

**Verification (gfx1151):**
- coherence-gate.sh: 5/5, no hard errors (all 4 commits verified)
- coherence-gate-dflash.sh: 4/4, no hard errors (tree-verify path exercised)

**Ship 3.2 core (C0–C3 + C6) complete.** qwen35 prefill batched attention
path fully migrated to `AttentionFamily` via `Step::Attend`. The Q8 >15k ctx
LDS cliff is eliminated (P-1 tiled kernel). Deferred: C4 (adaptive verification +
boundary), C5 (PFlash confirmation).
