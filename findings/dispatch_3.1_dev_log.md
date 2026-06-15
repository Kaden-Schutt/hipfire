# Ship 3.1 Dev Log — Attention decode wire-up (Phase A)

**Branch:** `integration/dispatch-unification`
**Owner:** Kevin / unverbraucht
**Plan:** `docs/plans/dispatch_3.1.md`
**Started:** 2026-06-06

---

## Commit B0 · Close the non-flash Q8 gap + dispatch-arm completeness guard

### 2026-06-06 — B0 implementation started

**Files touched:**

| File | Change |
|---|---|
| `crates/hipfire-dispatch/src/types.rs` | Added `KernelKey::AttnQ8_0Kv` variant (append-only, after `AttnFlashQ8_0`) |
| `crates/hipfire-dispatch/src/tables/attention_table.rs` | Registered `AttnQ8_0Kv → ArchPredicate::Always` |
| `crates/hipfire-dispatch/src/families/attention.rs` | Added `AttnQ8_0Kv` dispatch arm → `gpu.attention_q8_0_kv(…)` (no `flash_partials`); refactored `dispatch_attention` to take explicit `key: KernelKey` arg (D2 prep — `run()` passes `params.kind`); populated catch-all error fields with key name for forensic value |

**Still TODO for B0:**
- [ ] GPU-free dispatch-arm completeness test: iterate every `Attn*`/`KvWrite*` key, assert each has a dedicated match arm (not the catch-all)
- [ ] Grow coverage gate (`hipfire-dispatch-tests` or `coverage_tests.rs`) with Q8 non-flash attention row
- [ ] `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests` green
- [ ] `cargo check --workspace --all-targets` clean

**Notes:**
- Refactored `dispatch_attention` signature from `(gpu, params)` to `(gpu, key, params)` now (rather than deferring to B1) because the completeness test needs a way to check "does key X have a dedicated arm?" and keeping the old `params.kind`-reading form would mean changing it twice.
- The `run()` method now passes `params.kind` explicitly: `dispatch_attention(gpu, params.kind, params)`. B1 will remove `params.kind` entirely when `KvTierPlan` takes over.
- `AttnQ8_0Kv` maps to `gpu.attention_q8_0_kv(q, k_cache, v_cache, output, pos_buf, seq_len, n_heads, n_kv_heads, head_dim, physical_cap)` — same as `AttnFlashQ8_0` minus `flash_partials`. Parameter mapping verified against `rdna-compute/src/attention.rs:4424`.
- Catch-all error now says `"unhandled key — missing dispatch arm"` instead of empty strings.
- Added `DISPATCHED_ATTENTION_KEYS` const array (18 entries) + `dispatch_attention_has_arms_for_all_attention_keys` test — cross-checks table registrations vs dispatched arms bidirectionally.
- Added `attention_keys_resolve_on_fleet_archs` coverage test — all 18 attention keys resolve on their target archs (ALL for Always-gated, WAVE32 for HasWmma-gated).

**Test results:** `cargo test -p hipfire-dispatch` → 86/86 pass (was 85). `cargo test -p hipfire-dispatch-tests` → 59/59 pass. `cargo check --workspace --all-targets` → clean (pre-existing warnings only).

---

## Commit B1 · `KvTierPlan` + `attention_family()` accessor + paired `run_attention` + `AttnParams` cleanup

### 2026-06-06 — B1 implementation complete

**Files touched:**

| File | Change |
|---|---|
| `crates/hipfire-dispatch/src/families/kv_tier.rs` | **New file.** `KvTierInputs` (9 scalar fields, GPU-free), `KvTierPlan` (write_key + attend_key + v_mode_bits + uses_givens), `KvTierPlan::derive()` with q8 `use_flash` heuristic moved verbatim from `qwen35.rs:12885`, `tiers_match()` guard, 16 unit tests covering all tiers |
| `crates/hipfire-dispatch/src/families/attention.rs` | Removed `kind: KernelKey`, `flash_mode: Option<usize>`, `capture_mode: bool`, `kv_dim: usize` from `AttnParams`. Added `run_attention()` paired method. `KvWriteF32` now computes `kv_dim` locally from `n_kv_heads * head_dim`. `run()` now takes explicit `key: KernelKey` param. Added `pos` doc comment (0-based, caller passes `pos` not `pos+1`). |
| `crates/hipfire-dispatch/src/families/mod.rs` | Registered `kv_tier` module |
| `crates/hipfire-dispatch/src/coverage_tests.rs` | Added `WMMA_ARCHS` constant + doc distinguishing WMMA from wave32 (RDNA1/2 are wave32 but lack WMMA). Renamed `WAVE32` doc to warn about the historical misnomer. Attention coverage test uses `WMMA_ARCHS` for `AttnGqaFused`. |
| `crates/hipfire-runtime/src/llama.rs` | Added `attention_family()` accessor (OnceLock pattern). Re-exported `AttnParams`, `KvTierPlan`, `KvTierInputs`. |

**Test results:** `cargo test -p hipfire-dispatch` → 102/102 pass (was 86, +16 kv_tier tests). `cargo test -p hipfire-dispatch-tests` → 59/59. `cargo check --workspace --all-targets` → clean.

---

## Commit B2 · qwen35 decode → `run_attention` (direct paired call)

### 2026-06-06 — B2 complete

Replaced the 130-line `kv_cache_attention_dispatch` inline match tree with:
`KvTierPlan::derive(KvTierInputs { ... }) + AttnParams { ... } + attention_family().run_attention(...)`.
44 lines total. Call sites at `:12301` and `:12510` untouched.

**Verification (gfx1151):**
- coherence-gate.sh: 5/5 cells, no hard errors
- coherence-gate-dflash.sh: 4/4 cells, no hard errors

---

## Commit B3 · `Step::Attend` — route through `execute_steps`/`launch_op`

### 2026-06-06 — B3 complete

- `types.rs`: added `PipelineOp::Attend`
- `kv_tier.rs`: added `Clone, Copy` derives to `KvTierPlan` + `KvTierInputs`
- `pipeline/steps.rs`: added `Step::Attend { plan, io }`, `op_kind` arm, `launch_op` arm
  (local `OnceLock<AttentionFamily>` — same pattern as GEMV/ROTATION statics)
- `qwen35.rs`: replaced direct `attention_family().run_attention()` with
  `execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])`
- No `FUSED_TABLE` row for Attend (coupled pair, not fusible)

**Verification (gfx1151):**
- coherence-gate.sh: 5/5 cells, no hard errors
- coherence-gate-dflash.sh: 4/4 cells, no hard errors
- 102/102 hipfire-dispatch tests pass

---

## Commit B5 · Verification sweep + cleanup

### 2026-06-06 — B5 complete

**Checklist:**

- [x] **Coverage golden:** `attention_keys_resolve_on_fleet_archs` — all 18 attention keys resolve
  across ALL + WMMA_ARCHS fleet. `dispatch_attention_has_arms_for_all_attention_keys` —
  bidirectional cross-check passes.
- [x] **Grep audit (goal #1):** zero `gpu.kv_cache_write_*` / `gpu.attention_flash_*` /
  `gpu.attention_f32` / `gpu.attention_q8_0_kv` calls remain in `kv_cache_attention_dispatch`.
  Function is 44 lines (was 130). All dispatch goes through `run_attention`.
- [x] **Dispatch-arm-completeness test:** green.
- [x] **Prefill paths untouched:** `run_fa_layer_body` (`:11446`),
  `forward_scratch_layers_multi` (`:12844`), all `forward_prefill_*` variants still have
  inline GPU dispatch (Ship 3.2). `forward_scratch_layers` delegates entirely to
  `kv_cache_attention_dispatch` (2 call sites) with zero direct GPU attention calls.
- [x] **Tests:** 102/102 hipfire-dispatch, 59/59 hipfire-dispatch-tests.

**Dev-log fixtures:**
- GPU: gfx1151 (RDNA3.5 APU)
- Binary md5: `3b94777f13ada6cd0bc9b2693f2cb8f4` (`target/release/examples/daemon`)
- coherence-gate.sh: 5/5 cells pass (0.8b-cap, 4b-code, 9b-reason, 9b-tool-call, 9b-reason-mq3)
- coherence-gate-dflash.sh: 4/4 cells pass (27b-dflash-prose, 27b-dflash-code,
  27b-ddtree-b12-prose, 27b-ddtree-b12-code)
- **A/B perf gate:** `probe_commits.sh dc0a5adb HEAD` → baseline 45.9 tok/s, HEAD 45.9 tok/s
  (±0.0%, model qwen3.5-9b.mq4, KV asym3, gfx1151). Resolve-cost 2×n_layers/token
  is in the noise band — well under ±1–3% gate.

**Ship 3.1 complete.** qwen35 single-token decode attention path fully migrated to
`AttentionFamily` via `Step::Attend` / `execute_steps`. A new KV quant tier is now a
registry entry + kernel file + `KvTierPlan::derive` arm — no per-model dispatch tree.
