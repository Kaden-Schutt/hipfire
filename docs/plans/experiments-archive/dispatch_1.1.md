# Ship 1.1 — qwen35 DeltaNet shim migration

**Branch:** `feature/dispatch-unification`
**Tracking:** #397 (ship 1, item 1.1)
**Goal:** Every projection in qwen35's DeltaNet layers goes through
`execute_steps`. No direct `weight_gemv`, `rmsnorm_rotate_dispatch`, or
`fused_qkvza_dispatch` calls remain in any single-token forward path.

**Nomenclature note:** Roadmap 1.1's title says "DeltaNet **prefill**"
but the work is on the **decode** path (`forward_scratch_layers`) and
the test/demo path (`forward_from_x_gpu`). True batched prefill
(`forward_prefill_chunk`) uses GEMM and lands in Ship 5 (GemmFamily).

**Review:** `docs/plans/dispatch_1.1_combined_review.md`

---

## Current state

### Two single-token forward paths contain DeltaNet layers

| Function | Lines | Purpose | DeltaNet status |
|---|---|---|---|
| `forward_from_x_gpu` | 4702–5733 | Test/demo single-token (examples only) | **All bare `weight_gemv`** (24 calls) |
| `forward_scratch_layers` | 12522–13400 | **Production** daemon decode (pre-allocated scratch) | **Legacy helpers** (`rmsnorm_rotate_dispatch` + `fused_qkvza_dispatch`), `execute_steps` for wo/gate+up already |

**Call graph verified:** `forward_from_x_gpu` is reachable only from
`test_inference.rs`, `test_inferenceQA.rs`, `infer.rs`,
`profile_deltanet.rs`, `bench_qwen35_forward.rs`, and the VL
integration (`forward_with_embedding`). The **production daemon decode
path** is `forward_scratch_layers` → called from
`hipfire-runtime/src/llama.rs:2884` and `qwen35.rs:5882/5901/5913`.

### `forward_prefill_chunk` (batched prefill)

Uses batched GEMM calls (`gemm_qkvza_*`, `gemm_gate_up_*`,
`gemm_hfq4g256_residual`, etc.) — **not in scope for 1.1**. Lands in
Ship 5 (GemmFamily prefill expansion).

### `forward_scratch_layers_multi` (multi-GPU)

Uses `weight_gemv_prerotated` (not the helpers being deleted) —
unaffected by 1.1, tracked in Ship 5.

### What's already migrated in `forward_scratch_layers`

- **FullAttn QKV**: `qkv_via_execute_steps` (Step::RmsnormAutomatic + 3× Step::Gemv)
- **FullAttn gate+up**: `gate_up_via_execute_steps` (Step::RmsnormAutomatic + 2× Step::Gemv)
- **wo projection**: `Step::GemvResidual`
- **lm_head**: `Step::Gemv`
- **DeltaNet wo**: `Step::GemvResidual` (already done)
- **DeltaNet gate+up**: `gate_up_via_execute_steps` (already done)

### What remains unmigrated

**`forward_scratch_layers` (2 DeltaNet sites: DeltaNet + DeltaNetMoe):**
1. `rmsnorm_rotate_dispatch` (L12548, L12754) → replace with `Step::RmsnormAutomatic`
2. `fused_qkvza_dispatch` (L12554, L12759) → replace with new `qkvza_via_execute_steps` helper

**`forward_from_x_gpu` (4 layer types, 24 bare `weight_gemv` calls):**

| Layer type | Count | Line numbers |
|---|---|---|
| DeltaNet | 8 | 4740, 4745, 4750, 4752, 4885, 4894, 4895, 4899 |
| FullAttn | **7** | 4919, 4944, 4945, 5023, 5032, 5033, 5037 |
| DeltaNetMoe | 5 | 5057, 5061, 5065, 5067, 5186 |
| FullAttnMoe | **4** | 5209, 5228, 5229, 5301 |

---

## Plan — three commits

### Commit 1 · `qkvza_via_execute_steps` helper + `FUSED_TABLE` QKVZA entries + `launch_fused` arms

**Scope:** Dispatch crate + `forward_scratch_layers` DeltaNet sites.

The existing `qkv_via_execute_steps` handles 3-way QKV. We need a new
`qkvza_via_execute_steps` that handles the 4-way QKVZA pattern for
DeltaNet layers. This mirrors `fused_qkvza_dispatch` but goes through
the pipeline interpreter.

**Dispatch crate changes:**

1. Add `QKVZA4` op-pattern constant (= 5 ops: RmsnormAutomatic + 4× Gemv)
   in `steps.rs`, alongside existing `QKV3` and `GATE_UP2`.

2. Add QKVZA guards in `steps.rs`. Each guard **must open with**
   `if ctx.flags.force_unfused { return false; }` — every existing
   guard follows this pattern (`steps.rs:93,98,105,124,129,134`) and
   omitting it would make `HIPFIRE_FORCE_UNFUSED=1` silently fuse the
   QKVZA path, breaking the byte-parity verification.
   - `guard_qkvza_mq4g256lloyd` — 5 ops, 4× Gemv uniform MQ4G256Lloyd
   - `guard_qkvza_mq3g256lloyd` — same for MQ3G256Lloyd
   - `guard_qkvza_hfq4g256` — same for MQ4G256/HFQ4G256
   - `guard_qkvza_hfq6g256` — same for HFQ6G256/MQ6G256 (dp4a gated)

3. Add 4 `FUSED_TABLE` entries:
   ```
   FusedPattern { ops: QKVZA4, key: FusedQkvzaMq4G256Lloyd, guard: guard_qkvza_mq4g256lloyd }
   FusedPattern { ops: QKVZA4, key: FusedQkvzaMq3G256Lloyd, guard: guard_qkvza_mq3g256lloyd }
   FusedPattern { ops: QKVZA4, key: FusedQkvzaHfq4G256,     guard: guard_qkvza_hfq4g256     }
   FusedPattern { ops: QKVZA4, key: FusedQkvzaHfq6G256,     guard: guard_qkvza_hfq6g256     }
   ```

4. **Add QKVZA match arms to `launch_fused`** (`steps.rs`).
   The existing `launch_fused` has arms for 3-way QKV (extracts
   `steps[1..=3]`) and 2-way GateUp, but **no arms for `FusedQkvza*`
   keys** — it falls through to `_ => Err(DispatchError::MissingImpl)`.
   `FusedQkvFamily::run` has the 4-way dispatch bodies
   (`families/fused_qkv.rs:101–124`), but `launch_fused` never reaches
   them. Add a new arm extracting 4 weights + 4 outputs from
   `steps[1..=4]` and calling `fused_qkv.run` with 4-element arrays:
   ```rust
   KernelKey::FusedQkvzaHfq4G256
   | KernelKey::FusedQkvzaMq3G256Lloyd
   | KernelKey::FusedQkvzaMq4G256Lloyd
   | KernelKey::FusedQkvzaHfq6G256 => {
       let (wqkv, qkv) = gemv_weight_out(&steps[1]);
       let (wz, z)     = gemv_weight_out(&steps[2]);
       let (wb, beta)  = gemv_weight_out(&steps[3]);
       let (wa, alpha) = gemv_weight_out(&steps[4]);
       fused_qkv.run(ctx, gpu, &FusedQkvParams {
           kind: key,
           weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
           x: activated,
           outputs: &[qkv, z, beta, alpha],
           m: &[wqkv.m, wz.m, wb.m, wa.m],
           k: wqkv.k,
       })
   }
   ```
   ~12 lines, modelled on the existing 3-way arm.

5. Add GPU-free (arch × dtype) coverage golden in
   `hipfire-dispatch-tests` asserting the resolved `KernelKey` for QKVZA
   across (gfx1100, gfx1201) × (MQ4G256Lloyd, MQ3G256Lloyd, HFQ4G256,
   MQ4G256, HFQ6G256, MQ6G256, ParoQ4G128, Q8_0). The QKVZA table
   entries gate on `HasWmmaW32` which already ORs gfx12
   (`tables/mod.rs:81`), so both arches should resolve correctly.

**qwen35 changes:**

6. New helper `qkvza_via_execute_steps` in `qwen35.rs`:
   ```rust
   fn qkvza_via_execute_steps(
       gpu, ctx,
       wqkv, wz, w_beta, w_alpha,        // 4 weights
       attn_norm, x, tmp, x_rot,          // norm inputs + scratch
       dn_qkv, dn_z, dn_beta, dn_alpha,  // 4 outputs
       eps,
   )
   ```
   Builds `[RmsnormAutomatic, Gemv{wqkv}, Gemv{wz}, Gemv{w_beta}, Gemv{w_alpha}]`
   and calls `execute_steps`. All dtypes go through the same step slice:
   - MQ4/MQ3/MQ6/HFQ4/HFQ6: `FUSED_TABLE` matches QKVZA4 → fused kernel
   - ParoQ4G128: no fused entry, interpreter falls through to per-op
     `launch_op` → `GemvFamily::run_auto` (handles internal Givens
     rotation). This is semantically identical to the current
     `fused_qkvza_dispatch` Paro branch which calls `weight_gemv`
     four times — same GPU launches, no regression.
   - Q8_0 / others: per-op fallback, same as unfused path

   **Rotation-tag contract** (must replicate identically from
   `qkv_via_execute_steps`, lines 13146–13180):
   - Derive `rotation = dtype_rotation_plan(wqkv.gpu_dtype)`.
   - **Two-branch pairing:**
     - `rotation == Givens` → `RmsnormAutomatic { rotation: None }`
       (plain rmsnorm) + `GemvInput::Raw(x_rot)` (per-weight Givens
       handled inside `run_auto`)
     - `rotation != Givens` → `RmsnormAutomatic { rotation }` (rmsnorm
       + FWHT) + `GemvInput::Prerotated(x_rot)` (no re-rotation —
       avoids the double-FWHT trap at `qwen35.rs:13096–13099`)
     - `rotation == None` → `RmsnormAutomatic { rotation: None }`
       (plain rmsnorm) + `GemvInput::Prerotated(x_rot)` (no rotation
       needed, but Prerotated avoids redundant check in `run_auto`)

7. Replace both `rmsnorm_rotate_dispatch` + `fused_qkvza_dispatch` call
   sites in `forward_scratch_layers` with `qkvza_via_execute_steps`:
   - DeltaNet site (L12548–12554)
   - DeltaNetMoe site (L12754–12759)

8. Delete `rmsnorm_rotate_dispatch` and `fused_qkvza_dispatch` functions
   **in their entirety**. The ParoQ4G128 fallback inside
   `fused_qkvza_dispatch` (four bare `weight_gemv` calls) is subsumed
   by the interpreter's per-op fallback — no retention needed.

**Verify:**
- `coherence-gate.sh` on gfx1100 (MQ4 + MQ3 + MQ6 weights)
- `HIPFIRE_FORCE_UNFUSED=1` byte-identical committed-token streams via
  `HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0, on fixed prompt set
- GPU-free (arch × dtype) golden passes for QKVZA including RDNA4 row
- `cargo test -p hipfire-dispatch` — new guard tests pass

**Estimated delta:** ~140 lines added (helper + guards + table entries +
`launch_fused` arm + golden), ~150 lines deleted (two legacy functions).

---

### Commit 2 · Collapse `forward_from_x_gpu` to a thin wrapper

**Rationale:** `forward_from_x_gpu` is test/demo-only (called from
`examples/test_inference*.rs`, `infer.rs`, `profile_deltanet.rs`,
`bench_qwen35_forward.rs`, and VL integration). The production daemon
uses `forward_scratch_layers`. After Commit 1, `forward_scratch_layers`
is fully pipeline-migrated. Rather than individually migrating all 24
`weight_gemv` calls in `forward_from_x_gpu` (which would be deleted
immediately afterward), collapse it to a wrapper that:

1. Builds a `Qwen35Scratch` (or equivalent temporary scratch state)
2. Delegates to `forward_scratch_layers`

This removes the second dispatch style from the file in one step and
ensures test/demo paths exercise the same pipeline code as production.

**Investigate:** Whether `Qwen35Scratch` can be constructed outside
`forward_scratch` (it may need `Gpu` allocators, dims from config,
etc.). If construction is non-trivial, factor out a
`Qwen35Scratch::new(gpu, config)` constructor and use it from both
`forward_scratch` (existing path) and `forward_from_x_gpu` (new
wrapper). The constructor must allocate all the same buffers that
`forward_scratch` currently allocates inline.

**Public API preserved:** `forward()`, `forward_gpu()`,
`forward_with_embedding()` remain unchanged — they call
`forward_from_x_gpu`/`forward_from_x` which now delegates. No external
caller changes needed.

**Verify:**
- `cargo test -p hipfire-arch-qwen35` (if any) passes
- `test_inference.rs` produces identical output to pre-migration
- `coherence-gate.sh` still passes (production path unchanged)

**Estimated delta:** ~800 lines deleted (entire body of
`forward_from_x_gpu`), ~30 lines added (scratch constructor + wrapper).

---

### Commit 3 · Verification + cleanup

- [ ] `coherence-gate.sh --full` on gfx1100 (MQ4 + MQ3 + MQ6 weights)
- [ ] `coherence-gate-dflash.sh` — DeltaNet layers participate in spec-decode
- [ ] `HIPFIRE_FORCE_UNFUSED=1` byte-identical committed-token streams
      via `HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0, on fixed prompt set
      (MQ4 + MQ3 + MQ6)
- [ ] `probe_commits.sh` A/B ±1–3% on decode tok/s (gfx1100)
- [ ] Grep audit: `grep -n 'weight_gemv(' qwen35.rs` should return 0
      hits outside `forward_prefill_chunk` (Ship 5)
- [ ] Grep audit: `grep -n 'rmsnorm_rotate_dispatch\|fused_qkvza_dispatch'`
      returns 0 hits (fully deleted)
- [ ] Unit tests: `hipfire-dispatch` guard tests cover QKVZA4 pattern
      including `force_unfused` rejection
- [ ] GPU-free golden: `hipfire-dispatch-tests` QKVZA (arch × dtype)
      coverage including RDNA4 row passes

**Note:** `w_down` (`weight_gemv_swiglu_residual`) is not migrated in
1.1 — it is dispatch-internal (calls `GemvFamily::run` with
`WithSwiGLUResidual`) but bypasses the interpreter. Tracked in
`dispatch_todo.md` for post-1.1 cleanup.

---

## Out of scope (tracked in `dispatch_todo.md` and/or Ship 5+)

| Item | Where |
|---|---|
| `forward_prefill_chunk` batched GEMM migration | Ship 5 (GemmFamily) |
| `forward_scratch_layers_multi` multi-GPU migration | Ship 5 |
| ParoQ4G128 fused QKVZA entries | Ship 1.2 |
| Q4K fused QKV entries | Ship 1.2 |
| Q8_0 fused gate+up gap | Ship 1.2 |
| `w_down` `Step::GemvSwigluResidual` variant | `dispatch_todo.md` (D-1) |
| gfx1201 hardware testing | `dispatch_todo.md` (D-3) |
| Same-binary `HIPFIRE_DISPATCH_OLD/_NEW` selector | `dispatch_todo.md` (D-4) |
| Attention/KV pipeline migration | Ship 3 |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-05 | — | Plan written | — |
| 2026-06-05 | — | Gemini + Claude reviews; combined review at `dispatch_1.1_combined_review.md` | See review doc |
| 2026-06-05 | — | Plan revised: restructure 4→3 commits, add `launch_fused` arm, fix counts, strengthen verification | — |
