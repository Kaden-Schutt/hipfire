# Ship 4.2 — qwen35 grouped-GEMM MoE prefill → `MoeFamily::run_prefill` (Step 8)

**Branch:** `feature/ship-4.2-moe-prefill` off `integration/dispatch-unification`.
**Tracking:** [#397](https://github.com/Kaden-Schutt/hipfire/issues/397) Step 4.2 / Step 8.
**Owner:** Kevin (Ship-4 takeover from Nick, aligned). Edits `families/moe.rs` + `pipeline/mod.rs`
+ `qwen35.rs`.
**Depends on:** 4.1 landed (`31738389`); PR #428 (ds4 `run_bias_aware_prefill` template,
`MoeGroupedGemm`/`MoeGroupedI8` coverage-key precedent); **Ship 3.3 landed** (qwen35.rs released —
verify no textual overlap with the MoE-prefill region ~6940-7790 before V1).
**Phase 0:** 0.2 (scatter scratch model-owned, family takes refs), 0.5 (typed `run*()`, family owns
dispatch decision), 0.6 (byte-parity + prefill tok/s probe, gfx1100 + gfx1201).

**Reviews folded:** `dispatch_4.2.plan_rev_claude.md` (self, R1–R8 + N1–N8), and external
gemini / ds4_arch / ds4 / glm5 — see [adjudication](#review-adjudication). Finding tags `[Rn]`/
`[Nn]`/`[ext]` mark where each lands.

**Goal:** the qwen35 batched MoE-prefill **routed-expert block** stops deciding *how* to dispatch.
`prefill_moe_ffn_body_batched` (qwen35.rs:6940) inlines the Path 0/1/2 selection, arch/env gating,
and a per-dtype×i8×k8 grouped match. After 4.2 that lives in `MoeFamily::run_prefill`, mirroring
ds4's `run_bias_aware_prefill`. Last open piece of Ship 4.

---

## The model ↔ family boundary (pinned) `[R1, R2, ext: ds4_arch F1 / ds4 F1 / glm5 #2]`

The four reviews + my self-review **unanimously** rejected the draft's "SwiGLU-rotate stays
model-owned" — ds4's family executor owns it, and a single `run_prefill` is only coherent if the
family owns the whole routed pipeline. Pinned split (mirrors ds4):

- **Model owns** (stays in `prefill_moe_ffn_body_batched`): RMSNorm, router GEMV + softmax top-k →
  `topk_indices`/`topk_weights`, and the **shared expert** (incl. its `gate.gpu_dtype`-matched
  `gemv_*_residual_sigmoid_scaled_gpu_batched` **down** — a model-side dtype match this ship does
  **not** remove; consistent with ds4; a future `GemvFamily::run_residual_sigmoid_scaled` cleanup).
  The shared-expert down already adds into the residual `x_batch`.
- **Family owns** (`run_prefill`): scatter → grouped/indexed **gate_up** → unscatter → **SwiGLU +
  FWHT/Givens rotate** → grouped/indexed/atomic **down** → combine, accumulating into `x_batch`.

This is a **function split**, not a wholesale move `[R2]`.

---

## Grounded starting state

`prefill_moe_ffn_body_batched` (qwen35.rs:6940; **two callers** — :10367 dense, :10855 FA-MoE —
both benefit from one migration `[R5]`; no per-token MoE-FFN fallback). The routed block
(`:7281`–end):

- **gate_up:** `path2_eligible = HIPFIRE_MOE_GROUPED_GEMM(default on) && (gfx11||gfx12)`.
  - **Path 2:** `moe_scatter_fused_k8` → per-dtype grouped-WMMA (`MQ4`→`gemm_hfq4g256_moe_grouped_wmma_k2`,
    `MQ6`→`gemm_hfq6g256_moe_grouped_wmma`, `Paro`→ **`givens_rotate_to` preamble** then
    `gemm_paro_q4g128_moe_grouped_{mmq_k8_gfx1151 | mmq_gfx1151 | wmma_k2}` gated by
    `HIPFIRE_MOE_PARO_I8`/`_K8`, **gfx1151-only**) → `moe_gate_up_unscatter_k8`.
  - **Path 1 (else):** per-token `gemv_*_moe_gate_up_k8_indexed_batched`.
- **silu+rotate:** AWQ-aware select — `fused_silu_mul_givens_rotate_f32` (Paro) vs
  `fused_silu_mul_rotate_mq_batched_for` (MQ).
- **down:** Path 2 grouped (same dtype match, **no** Paro preamble — `rot_batch` already rotated)
  → combine; **else** `use_atomic_free_down = !gfx9`: **Path 1** (atomic-free expanded + combine,
  RDNA incl. gfx1030) **or Path 0** (CDNA gfx9: residual-scaled atomic GEMV) `[N1]`.
- combine writes to **`pbs.x_batch`** (the residual) — **no separate `ffn_out`** `[N2]`.
- `m_total_max = moe_grouped_m_total_bound(total_slots, n_exp)` (qwen35.rs:5844) =
  `align_up(total_slots + min(total_slots,n_exp)*(BLOCK_M-1), BLOCK_M)`, `BLOCK_M=16`; computed
  once, reused for both halves (dtoh-skip, :7347) — **differs from ds4's** `batch*k_top + n_exp*BLOCK_M`
  `[N4]`.

**Perf:** Path 2 is **+114% (gfx1100) / +192% (gfx1201)** on A3B mq4 prefill=256 — non-regression
is the **hard gate**.

ds4 template: `MoeBiasAwarePrefillParams` (all fields `&'a GpuTensor` `[N3]`),
`run_moe_prefill_bias_aware` (dispatches **by dtype/env internally**, `dispatch_grouped_lloyd`;
registered keys are **coverage-only** `[R4]`).

---

## Design decisions

### D1 · `MoePrefillParams` (qwen35 softmax-top-k flavor) `[R3, N2, N3, N4, ext: ds4 F5]`

Distinct from `MoeBiasAwarePrefillParams`. All tensor refs **`&'a GpuTensor`** (shared, not
`&mut` — GpuTensor is Copy `[N3]`). Fields:
- `dtypes: MoeDtypes`, `batch_size: usize` (= n), `mi/down_m/down_k/gate_up_k/k_top/n_exp`,
  `m_total_max: usize` (model computes via `moe_grouped_m_total_bound` `[N4]`).
- routing inputs (model-produced): `topk_indices`, `topk_weights`.
- **destination = `x_batch`** (residual; combine accumulates here) — **not `ffn_out`** `[N2]`.
- activation/rotate buffers: `x_norm_batch`, `x_rot_batch`, `gate_batch`, `up_batch`, `rot_batch`.
- scatter scratch (model-owned, enumerated `[ds4 F5]`): `expert_token_counts`, `expert_offsets`,
  `sorted_slot_index`, `expert_tile_ids`, `inverse_perm`, `y_gate_up_grouped`, `y_down_grouped`.
- paro sidecars: `gate_up`/`down` `Option<GivensRef>`.

### D2 · `MoePrefillResolution` + `run_prefill` — family owns Path0/1/2 + dtype + i8/k8 `[R1, R8, N1, N6, N7, ext]`

Separate `MoePrefillResolution` (decode's `MoeResolution` axes don't fit) `[R8]`. Pure-ish fn of
`MoeDtypes` + arch + `FeatureFlags`. Selects: **gate_up path** (Path 1 vs Path 2), **down path**
(**Path 0 / Path 1 / Path 2** — `gfx9`→Path 0 `[N1]`), per-dtype grouped kernel, **Paro i8/k8**
(`gfx1151`-only — **verify `ArchCaps` exposes gfx1151 granularity** before V0; `HasWmma` is too
coarse `[N7]`), and the **AWQ silu-rotate select** (`fused_silu_mul_givens_rotate_f32` vs
`_mq_batched_for`) `[R1]`. Env levers (`HIPFIRE_MOE_GROUPED_GEMM`, `_PARO_I8`, `_K8`) read from
**`ctx.flags`/`FeatureFlags`** (add to `FeatureFlags`, parsed at `Gpu::init`) — **not** `std::env`
in the resolver `[N6]`; document that mid-prefill env mutation is no longer honored.

`run_prefill(ctx, gpu, &MoePrefillParams)` mirrors `run_bias_aware_prefill`: scatter → gate_up →
unscatter → silu+rotate → down → combine, **one call**. **`ctx` is decision-only** (arch/env) —
the raw `gpu.gemm_*`/`gpu.gemv_*` kernel calls do **not** take it (unlike 4.1 decode's
`gemv.run_auto(ctx)`) `[ext: ds4_arch F4]`. Build **one ctx in `forward_prefill_chunk`** (once per
chunk, not per layer) and thread it down `[ext: gemini F1]`.

### D3 · `dispatch_grouped_gemm` helper — grouped-GEMM **selection only** `[ext: ds4_arch F2 / ds4 F2 / glm5 #2]`

Dedups the per-dtype×i8×k8 grouped-kernel match (eliminates the gate_up/down duplication of the
`use_paro_i8`/`_k8` reads). **The Paro gate_up `givens_rotate_to` preamble stays in the gate_up
block, above the helper** (down has none — `rot_batch` pre-rotated). Helper takes `x` explicitly
(gate_up reads `x_rot_batch` `[N×dim]`; down reads `rot_batch` `[N*k_top×mi]` — different tensors)
`[glm5 #2]`.

### D4 · Registry keys = coverage-only; dispatch is dtype-driven `[R4, ext: all]`

`run_prefill` dispatches **by dtype/env internally** (the `run_moe_decode` shape; ds4 prefill does
the same). Register coarse coverage keys (`MoeGroupedHfq4`/`Hfq6`/`Paro` — append-only) in
`moe_table`; **`shape_gate: Some(BatchGt(1))` is documentation, NOT runtime-evaluated** (the family
never calls `resolve()` for dispatch). This **deviates from issue #397's "`ShapeInfo.batch_size`
gating"** — intentional: the {dtype × i8 × k8 × arch} cross-product can't be `ShapeInfo`-expressed
`[glm5 #10]`. (MoE has no dispatch-arm completeness test, so a coverage-only key is safe.)

### D5 · Byte-identical + perf-neutral `[N4, N5, ext]`

Verbatim transcription → same kernels/args/order/scatter pipeline → byte-identical. **Preserve the
qwen35 `moe_grouped_m_total_bound`** — do NOT substitute ds4's formula `[N4]`. Combine order
untouched. Byte-parity is **not** provable by `EMIT_TOKEN_IDS` alone (that's the first decode
token's argmax) — use a **prefill hidden-state / logits diff** `[N5]`.

---

## Plan

> **V0 + V1 land as one commit** (the `MoeParams`→`MoePrefillParams` + new-fn dependency spans
> crates; matches 4.1 W0+W1) `[ext: ds4_arch F5]`. V2 is the verification sweep.

### Commit V0+V1 · family executor + qwen35 split

**V0 (dispatch crate):**
1. `families/moe.rs`: `MoePrefillParams` (D1); `MoePrefillResolution` (D2); `MoeFamily::run_prefill`.
2. `pipeline/mod.rs`: `run_moe_prefill` — verbatim transcription of the routed block (scatter →
   gate_up [Path1/2] → unscatter → silu+rotate → down [**Path0/1/2**] → combine→`x_batch`),
   `dispatch_grouped_gemm` helper (D3), Paro gate_up Givens preamble in-line, qwen35 `m_total`
   bound (N4). Add MoE env levers to `FeatureFlags` (N6).
3. `tables/moe_table.rs`: coverage-only grouped keys (D4).
4. `FeatureFlags`/`ArchCaps`: ensure gfx1151 granularity (N7) + MoE env levers.
5. GPU-free tests: `MoePrefillResolution` cells — Path2 (gfx11/12 × MQ4/MQ6/Paro × i8/k8 env),
   Path1 (gfx1030), **Path0 (gfx906)** `[N1]`; AWQ-select cells.

**V1 (qwen35):** split `prefill_moe_ffn_body_batched` — keep router/softmax-topk/shared-expert
(incl. `_residual_sigmoid_scaled_` down) model-owned; replace the routed block with
`MoePrefillParams { dtypes, batch_size: n, m_total_max, topk_*, x_batch, <scatter scratch>, … }` +
`moe_family().run_prefill(&ctx, gpu, &params)` (ctx threaded from `forward_prefill_chunk`). Grep
audit: zero `gpu.gemm_*_moe_grouped_*` / `gpu.moe_scatter_*` / `gpu.gemv_*_moe_gate_up_k8_indexed_batched`
/ down combine in the qwen35 routed block.

**Verify:** `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests`; `cargo check --workspace`.

### Commit V2 · verification sweep (gfx1100 + gfx1201)

- [ ] **Prefill byte-parity** vs pre-4.2: `HIPFIRE_DUMP_HIDDEN` diff (or lm_head logits-diff) of
      the MoE-prefill output on a 32-tok prompt — **bit-exact** `[N5]`. (`EMIT_TOKEN_IDS` first
      token is a smoke, not the gate.)
- [ ] **`probe_commits.sh <pre-4.2> HEAD` prefill tok/s ±1–3%** on gfx1100 + gfx1201, **≥256-token
      prompt** with non-trivial per-expert counts `[glm5 PG1]` — the +114/192% Path-2 lift holds.
- [ ] `coherence-gate.sh --full` (A3B cells).
- [ ] **Path-1 force-smoke:** `HIPFIRE_MOE_GROUPED_GEMM=0` on gfx11 (tests Path 1 logic without
      gfx10 hw) `[N8]`; gfx906 Path-0 + gfx1030 Path-1 if reachable, else document.
- [ ] **A3B MoE DFlash** pinned fixture (target `edde51ec`, draft `8254bbe1`) or document the gap
      `[ext: ds4 F8]`.
- [ ] Paro/MQ6 A3B fixtures if available (exercise the i8/k8 + HFQ6 arms); gfx1151-only for
      Paro-i8/k8; document cross-arch residual `[R7]`.
- [ ] `dispatch_4.2_dev_log.md`: fixtures + prompt/binary md5 + both-arch tok/s.

---

## Risks

1. **Perf non-regression (hard gate, not byte-parity) `[D5]`.** Verbatim transcription incl. the
   `m_total` dtoh-skip + the qwen35 bound formula (N4); ±1–3% probe on both arches.
2. **Path 0 (gfx9) must be preserved `[N1]`** — the down half has three paths; resolution selects
   Path0/1/2. Missing it mis-dispatches CDNA. GPU-free resolution test covers gfx906.
3. **Paro gate_up Givens preamble must NOT be deduped into the helper `[D3]`** — silent garbage if
   lost. Helper = grouped-GEMM selection only.
4. **env-lever / gfx1151 resolution `[N6, N7]`** — read from `ctx.flags`; verify `ArchCaps`
   gfx1151 granularity before V0.
5. **`x_batch` vs `ffn_out` `[N2]`** — combine writes the residual directly; no seed/accumulate
   buffer. Wrong dst = wrong output.
6. **Cross-lane edit `[R2/4.1 precedent]`** — Kevin owns `moe.rs`/`pipeline.rs`/`qwen35.rs`; align
   with Nick; verify no Ship-3.3 textual overlap in 6940-7790 first.

---

## Out of scope

| Item | Where |
|---|---|
| MoE decode (`moe_ffn_decode_impl`) | 4.1 (done) |
| ds4 MoE prefill (`run_bias_aware_prefill`) | 4.3 (done) |
| **Factor shared scatter→grouped→unscatter→combine skeleton** between `run_moe_prefill` + `run_moe_prefill_bias_aware` `[ext: ds4 F4]` | future (keep self-contained now — cross-contamination = the N4 risk) |
| Shared-expert `_residual_sigmoid_scaled_` down → `GemvFamily` `[ext: gemini F2]` | future GemvFamily cleanup |
| `Step::Moe`; multi-GPU MoE prefill | deferred |
| `HIPFIRE_DISPATCH_OLD/_NEW` selector `[ext: gemini F4 — REJECTED]` | program-level Phase-0.6 (git-checkout parity used) |

---

## Review adjudication

Full per-claim validate/reject for all five reviews (self R1–R8 + N1–N8; gemini, ds4_arch, ds4,
glm5) is in **`findings/dispatch_4.2.plan_rev_claude.md`**. Headlines: SwiGLU-rotate ownership
(quadruple convergence → family owns it, R1); Path 0 omission (N1); `ffn_out`→`x_batch` (N2);
`&mut`→`&` (N3); preserve qwen35 `m_total` formula (N4); prefill byte-parity needs hidden-state
diff (N5); env levers → `FeatureFlags` (N6); verify `is_gfx1151` (N7); D4 keys coverage-only
(unanimous); `HIPFIRE_DISPATCH` selector rejected (program-level).

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | — | Plan drafted (fork ii, ds4-prefill template). | — |
| 2026-06-06 | — | Self-review (R1–R8) + 4 external reviews folded. **Pinned the model↔family boundary** (family owns silu+rotate); added **Path 0** (down has 3 paths, N1); fixed `ffn_out`→`x_batch` (N2), `&mut`→`&` (N3); preserve qwen35 `m_total` bound (N4); prefill byte-parity via hidden-state/logits diff (N5); env levers → `FeatureFlags` (N6); verify gfx1151 granularity (N7); Path-1 force-smoke (N8); keys coverage-only / dtype-driven (D4); V0+V1 co-land; `HIPFIRE_DISPATCH` selector rejected. | — |
| 2026-06-06 | V0+V1 | V0+V1 co-land: `MoePrefillParams` (D1), `MoePrefillResolution` (D2), `MoeFamily::run_prefill` → `pipeline::run_moe_prefill` + `dispatch_grouped_gemm` (D3), 3 MoE prefill env levers → `FeatureFlags` (N6), `forward_prefill_chunk` ctx threading (gemini F1). qwen35 `prefill_moe_ffn_body_batched` routed block replaced with family delegation. 10 GPU-free resolution tests (Path2 gfx11/gfx12/gfx1151 × MQ4/MQ6/Paro × i8/k8 env, Path1 gfx1030, Path0 gfx906/gfx942, grouped-gemm opt-out, Paro i8 opt-out). D4 coverage keys deferred (dtype-driven dispatch, never calls resolve()). | — |
