# Ship 3.1 — Attention decode wire-up (Phase A)

**Branch:** `integration/dispatch-unification`
**Tracking:** #397 (Ship 3 · Attention + KV cache unification, Phase A)
**Owner:** Kevin / unverbraucht (Ship 3 lane — see [Ship 3 ⊥ Ship 4 boundary contract](https://github.com/Kaden-Schutt/hipfire/issues/397#issuecomment-4634433036))
**Depends on:** Ships 1.x + 2.x (landed): `execute_steps` interpreter, `Step`/`launch_op`,
`DispatchCtx`, the `gemv_family()`/`fused_qkv_family()`/`moe_family()` runtime accessor
pattern, Phase 0.4 `HasWmma`.
**Phase 0 contracts in force:** 0.1 (KV-tier families exempt from init resolve-cache —
live-resolve per token), 0.3 (`KvTierPlan` derived once per attention step, paired
write-then-attend, `debug_assert` write-tier == attend-tier), 0.6 (same-binary
byte-parity oracle, RDNA3 **and** RDNA4 probe).

**Goal:** the qwen35 single-token **decode** attention path (KV-write + flash-attention)
goes through the centralized `AttentionFamily` instead of the inline 5-branch
`kv_cache_attention_dispatch` match tree. A new KV quant tier becomes a registry entry +
kernel file — no per-model `kv_cache_write_*` / `attention_flash_*` dispatch tree. Phase A
is **decode only, single-GPU, non-adaptive KV path**; prefill / adaptive / DFlash WMMA
tiles are Phases B/C (Ship 3.2 / 3.3).

---

## Grounded starting state (verified on the branch tip)

**The family is real, not a stub.** `crates/hipfire-dispatch/src/families/attention.rs`:
- `AttnParams<'a>` (lines 9–33) — 22-field borrow struct: `kind: KernelKey`, `q/k/v`,
  `k_cache/v_cache`, `k_scales/v_scales`, `pos_buf`, `pos`, `n_heads/n_kv_heads/head_dim/
  kv_dim/physical_cap`, `flash_partials`, `givens_cos/sin`, `flash_mode`, `capture_mode`,
  `v_mode_bits`, `output`.
- `AttentionFamily::run()` (lines 60–68) resolves `params.kind` then calls
  `dispatch_attention` (lines 83–225) — a **real 17-arm match** with live `gpu.*` calls
  for every registered key.
- Table `tables/attention_table.rs` registers **17 keys** (8 `KvWrite*` + 9 `Attn*`), all
  `ArchPredicate::Always` except `AttnGqaFused` = `HasWmma`.
- **Zero external callers today** (grep `AttentionFamily|attention_family|AttnParams`).
  `KvTierPlan` does **not** exist yet. `ResourceManager` is an empty stub (Phase 0.2 — it
  stays empty; scratch stays arch-owned, passed `&mut`).

**The decode site to migrate** is `qwen35.rs:12789–12918` `fn kv_cache_attention_dispatch`
(called from `forward_scratch_layers` at `:12299`, single-token decode). It is a 5-way
branch on `kv_cache.{quant_asym4,quant_asym3,quant_asym2,quant_q8, else F32}` × a
`quant_fwht` sub-branch, each arm doing **write-then-attend** as a hard-coupled pair —
exactly the Phase 0.3 shape.

### Gap found while grounding — the non-flash Q8 path is unregistered

The q8 arm (`:12885–12903`) picks between two attend kernels at runtime:

```rust
let use_flash = gpu.graphs.capture_mode || s.flash_mode == 2
    || (s.flash_mode == 1 && pos + 1 >= 2048) || pos + 1 > 15000;
if use_flash { gpu.attention_flash_q8_0(…) } else { gpu.attention_q8_0_kv(…) }
```

The family only has `AttnFlashQ8_0` → `gpu.attention_flash_q8_0`. The **non-flash**
`gpu.attention_q8_0_kv` (the default short-context decode path — the common case) has **no
KernelKey and no dispatch arm**. Migrating the q8 branch naively would silently force every
short-context q8 decode onto the flash kernel — a perf regression, possibly a correctness
delta. **This must be fixed in 3.1** (Commit B0), not deferred.

---

## Design decisions (this ship)

> **Incorporates the two adversarial plan reviews** (`findings/dispatch_3.1_plan_dev_gemini.md`,
> `findings/dispatch_3.1_plan_dev_glm5.md`). Per-finding adjudication is in the
> [Review adjudication](#review-adjudication-2026-06-06) section; accepted items are folded
> directly into the design and commits below.

### D1 · `KvTierPlan` — the Phase 0.3 paired-plan type (new)

Add `KvTierPlan` **and** the GPU-free `KvTierInputs` to the dispatch crate
(`families/attention.rs` or a new `families/kv_tier.rs`). Derived **once per attention
step** from the live KV-cache state:

```rust
pub struct KvTierPlan {
    pub write_key:  KernelKey,   // KvWrite*
    pub attend_key: KernelKey,   // Attn*  (Flash or non-flash Q8 or F32 or GQA-fused)
    pub v_mode_bits: i32,        // shared sub-plan: 8=Q8, 2/3/4=Lloyd-V
    pub uses_givens: bool,       // shared sub-plan: needs givens_cos/sin
}

// GPU-free scalar inputs — NO runtime types (avoids the dep cycle, see below).
pub struct KvTierInputs {
    pub quant_asym4: bool, pub quant_asym3: bool, pub quant_asym2: bool,
    pub quant_q8: bool, pub quant_fwht: bool, pub v_mode_bits: i32,
    pub pos: usize, pub flash_mode: usize, pub capture_mode: bool, // q8 use_flash inputs
}
```

- Pure constructor `KvTierPlan::derive(inputs: KvTierInputs) -> KvTierPlan`. The q8
  `use_flash` heuristic (qwen35.rs:12885) moves **into** `derive` (selecting `AttnFlashQ8_0`
  vs the new `AttnQ8_0Kv`), not the kernel arm. GPU-free, unit-testable.
- **Circular-dep fix (Gemini F4, validated):** `hipfire-dispatch` **cannot** depend on
  `hipfire-runtime`, so the `KvCache → KvTierInputs` conversion lives in the **runtime/arch
  side** (a `fn kv_tier_inputs(kv_cache: &KvCache, pos, flash_mode, capture_mode) ->
  KvTierInputs` near `qwen35.rs`/`llama.rs`). Dispatch only ever sees plain scalars. There is
  **no** `from_kv_cache` inside the dispatch crate.
- **`debug_assert!`** inside `derive` that `write_key` and `attend_key` agree on tier — e.g.
  `KvWriteAsym3Fwht` ⇔ `AttnFlashAsym3Fwht`. Phase 0.3 #30-class drift guard — a first-class
  assert on a first-class type, not a convention.
- Phase 0.1: `derive` runs **live, every attention step** — never cached at init. Keys off
  `kv_cache` mutable mode state which adaptively downshifts. (Adaptive downshift itself is
  Phase B; Phase A only needs the live-derive plumbing so B is a no-restructure extension.)

### D2 · Paired family entry point

The current `AttentionFamily::run()` runs **one** key. Add a paired method so the
write+attend coupling is enforced in one call, consuming a `KvTierPlan`:

```rust
impl AttentionFamily {
    pub fn run_attention(
        &self, ctx: &DispatchCtx, gpu: &mut Gpu,
        plan: &KvTierPlan, io: &AttnParams,   // io.kind removed (D-clean)
    ) -> Result<(), DispatchError> {
        debug_assert!(tiers_match(plan.write_key, plan.attend_key));
        let w = self.resolve(plan.write_key,  ctx, None)?.key;  // arch-gate check via `?`
        dispatch_attention(gpu, w, io)?;
        let a = self.resolve(plan.attend_key, ctx, None)?.key;
        dispatch_attention(gpu, a, io)
    }
}
```

- `dispatch_attention` is refactored to take an explicit `key: KernelKey` arg (instead of
  reading `io.kind`) so it can be called twice. The single-key `run()` is kept for tests.
- **On Gemini F3 (rejected as a hazard, idiom adopted):** this registry's `resolve()` is
  keyed by the requested key (`table.get(&key)`) and returns a same-key variant or errors —
  it does **not** remap a key to a different fallback key (that lives in the
  `KernelKey::for_gemv_*` helpers, not `resolve()`). So `resolved.key == requested key`
  always; there is no fallback-bypass bug. We still bind `.key` from the resolve result (the
  form Gemini suggested) defensively at zero cost — the real value of the `resolve()?` is the
  **arch-predicate gate** (it returns `MissingImpl` if no arch-passing variant exists).
- **Pos semantics (Gemini F6, validated):** `AttnParams.pos` is the **0-based physical
  index**; `dispatch_attention` internally computes `seq_len = pos + 1`. The caller passes
  `pos`, **never** `pos + 1`. Pinned here so the migration doesn't double-increment.
- Both `resolve` calls are the cheap path Phase 0.1 documents (one `HashMap::get` + ≤9-entry
  `Always` scan); the perf gate confirms 2×n_layers resolves/token stays in ±1–3%.

### D3 · Routing — staged: direct paired call first, then `Step::Attend`

The issue's pipeline model is "a paired `Step::Attend` … through `launch_op` in
`execute_steps`." We reach it in two verifiable steps (mirrors how 2.1 treated `fused.run`
as a stepping stone to `execute_steps`):

- **B2 — direct paired call** from qwen35 decode: replace the body of
  `kv_cache_attention_dispatch` with `derive plan` + `attention_family().run_attention(…)`.
  Isolates "does the family reproduce the inline tree byte-for-byte" from pipeline plumbing.
  Lowest-risk byte-parity checkpoint.
- **B3 — `Step::Attend`**: add `Step::Attend { plan: KvTierPlan, io: AttnParams<'a> }` — it
  **owns `AttnParams` directly, no parallel `AttnIo` struct** (Gemini F5 / glm5 F9,
  validated; `AttnParams` is constructed once on the stack per layer and the `&[Step]` holds
  it). Add `op_kind` arm → `PipelineOp::Attend`, `launch_op` arm →
  `attention_family().run_attention`. `FUSED_TABLE` deliberately does **not** match it
  (write+attend are coupled, not independently fusible — per the issue's pipeline-model
  note). The decode site then expresses attention as `execute_steps(gpu, &ctx, &[Step::Attend
  { … }])`, fleet-consistent with QKV/gate-up.

Both B2 and B3 must be byte-identical to master and to each other; B3 is a pure routing
wrap with no kernel-selection change.

---

## Plan

### Commit B0 · Close the non-flash Q8 gap + dispatch-arm completeness guard (family-layer, GPU-free)

1. `types.rs`: add `KernelKey::AttnQ8_0Kv` (append-only — coordinate the enum block per the
   boundary contract; Nick appends MoE keys, we append `Attn*`/`KvWrite*`).
2. `tables/attention_table.rs`: register `AttnQ8_0Kv` → `ArchPredicate::Always`.
3. `families/attention.rs` `dispatch_attention`: add the arm → `gpu.attention_q8_0_kv(q,
   k_cache, v_cache, output, pos_buf, seq_len, n_heads, n_kv_heads, head_dim,
   physical_cap)` (no `flash_partials`).
4. **Catch-all footgun (glm5 F2, validated):** `dispatch_attention` ends in a `_ =>
   UnsupportedVariant{…}` with empty forensic fields. A new registered key with a forgotten
   arm (the 953ea648 dead-gate defect class) is silently swallowed. Mitigate:
   - Populate the error fields — at minimum `variant: <key name>` — for forensic value.
   - Add a **GPU-free test** that iterates every `Attn*`/`KvWrite*` `KernelKey` and asserts
     `dispatch_attention` has a dedicated (non-catch-all) arm. (Drives a dummy `AttnParams`
     per key behind a flag, or matches each key against an explicit dispatched-key list.)
     The existing coverage gate checks **table registration**, not **dispatch-arm presence** —
     this closes that hole.
5. Grow the `(op × dtype × arch)` coverage gate (`hipfire-dispatch-tests`) with the q8
   non-flash row across the fleet incl. RDNA4.

**Verify:** `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests` green; coverage gate
resolves `AttnQ8_0Kv` on every arch row; the new dispatch-arm-completeness test passes.

### Commit B1 · `KvTierPlan` + `attention_family()` accessor + paired `run_attention` + `AttnParams` cleanup

1. **`KvTierPlan` + `KvTierInputs` + `KvTierPlan::derive`** (D1) in the dispatch crate, with
   the q8 `use_flash` heuristic folded in and the tier-match `debug_assert`. Pure unit tests
   for every tier (asym4/3/2 × fwht/givens, q8 flash vs non-flash by pos/flash_mode/capture,
   F32) asserting the `(write_key, attend_key)` pair — GPU-free.
2. **`run_attention` paired method** (D2); refactor `dispatch_attention` to take `key` arg
   and pin `pos` as 0-based (F6).
3. **`AttnParams` field cleanup** (glm5 F3/F4, validated — do it now, before any caller binds
   the old shape):
   - **Remove `flash_mode: Option<usize>` and `capture_mode: bool`** — the flash/non-flash
     decision now lives in `KvTierPlan::derive` (via `KvTierInputs`). Leaving them is the
     #30-class divergence trap (a future arm re-reading them = a second flash-selection path
     the plan doesn't control).
   - **Remove `kv_dim: usize`** — redundant with `n_kv_heads * head_dim`; the only consumer
     (`KvWriteF32`) computes it locally, matching the original (`qwen35.rs:12906`). Avoids the
     stale-stride silent-bug.
   - **Remove `kind: KernelKey`** — superseded by `KvTierPlan`'s two keys (run via `key` arg).
   - `k_scales`/`v_scales` (glm5 F7): unused by qwen35 — keep, with a `// TODO(ship 3.1b):
     llama HFQ8/INT8 attend scales` comment tying them to the deferred llama work.
4. **`attention_family()` accessor** in `hipfire-runtime::llama`, mirroring `gemv_family()`
   exactly (`OnceLock<AttentionFamily>` + `get_or_init`). Re-export `AttnParams` /
   `KvTierPlan` / `KvTierInputs` via `hipfire_runtime::llama` for the arch crates (matches the
   `FusedQkvParams` re-export precedent).

**Verify:** `cargo check --workspace --all-targets`; new unit tests green. No behavior
change yet (no caller).

### Commit B2 · qwen35 decode → `run_attention` (direct paired call)

Rewrite `kv_cache_attention_dispatch` (`qwen35.rs:12789`) body to:

```rust
let plan = KvTierPlan::derive(KvTierInputs::from_kv_cache(kv_cache, pos,
    s.flash_mode, gpu.graphs.capture_mode));
let ctx = DispatchCtx::new(gpu);
let io = AttnParams { /* q=&s.fa_q, k/v=&s.fa_k/&s.fa_v, k_cache/v_cache=&kv_cache
    .k_gpu/v_gpu[layer_idx], pos_buf=&s.pos_buf, givens_cos/sin, flash_partials=
    &s.flash_partials, v_mode_bits, physical_cap, n_heads/n_kv_heads/head_dim, … */ };
hipfire_runtime::llama::attention_family()
    .run_attention(&ctx, gpu, &plan, &io)
    .map_err(|e| HipError::new(0, &e.to_string()))
```

The 5×2-branch inline tree is deleted. `kv_cache` is `&mut` only because the old signature
was; `run_attention` takes `gpu: &mut` + immutable cache refs — keep the borrow shape that
compiles (cache tensors are read for write-target + attend-source; that's fine).

**Verify (on-GPU, the linchpin — explicit matrix per glm5 F5 / Gemini F7, validated):**
**byte-identical committed-token IDs vs master** (`HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0) on
**gfx1100 and gfx1201**, across an explicit KV-tier × flash-path matrix, each with a
**committed prompt file** in `benchmarks/prompts/` and its **md5 recorded** (per the
prompt-md5 bench rule), plus **binary md5 recorded** in the commit message:

| Case | KV mode | Path exercised | Prompt condition |
|---|---|---|---|
| q8 non-flash | `q8` | `AttnQ8_0Kv` (B0 — the **default** short-ctx path) | `pos < 2048`, `flash_mode=0`, no capture |
| q8 flash | `q8` | `AttnFlashQ8_0` | `pos > 2048` **or** `flash_mode=2` **or** capture |
| asym4-fwht | `fwht4` | `KvWriteAsym4Fwht`+`AttnFlashAsym4Fwht` | any |
| asym3-fwht | `fwht3` | `…Asym3Fwht` | any |
| F32 | f32 | `KvWriteF32`+`AttnF32` | any |

- `probe_commits.sh master HEAD` ±1–3% on gfx1100 + gfx1201 against a **named model + prompt
  fixture** (resolve-cost gate — 2×n_layers cheap resolves/token must not move the band).
- `./scripts/coherence-gate.sh` (the current mandatory gate — it replaced the removed
  byte-exact `quality-gate.sh`; it is **not** deprecated).
- **`./scripts/coherence-gate-dflash.sh` IS required here (glm5 F5, validated; rejects Gemini
  F7's "coherence-gate.sh is deprecated" framing but accepts the dflash need):**
  `kv_cache_attention_dispatch` is the **same** function the spec-decode/DFlash accepted-token
  path runs (via `forward_scratch_layers`), so a regression in the migrated attention is
  invisible to `coherence-gate.sh` alone. Run the dflash gate even though 3.1 touches no
  spec-decode *code*.

### Commit B3 · `Step::Attend` — route through `execute_steps`/`launch_op`

1. `pipeline/steps.rs`: add `Step::Attend { plan, io }` (+ a small `AttnIo` borrow bundle),
   `op_kind` arm, `launch_op` arm → `attention_family().run_attention`. Exhaustive-match
   compiler check confirms no Step is unhandled.
2. `types.rs`: add `PipelineOp::Attend` (append-only).
3. qwen35 decode: replace the direct B2 call with `execute_steps(gpu, &ctx,
   &[Step::Attend { plan: &plan, io: &io }])`.
4. **Do not** add a `FUSED_TABLE` row for `Attend` (coupled pair, not fusible — documented).

**Verify:** byte-identical to **B2 and master** on the same matrix (pure routing wrap);
`coherence-gate.sh`; coverage goldens unchanged.

### Commit B4 · llama + qwen2 decode attention — DEFERRED to Ship 3.1b (own PR)

**Both reviews independently graded this a separate ship's worth of work (Gemini F1+F2
Critical, glm5 F1 High), and they're right** — grounded, this is not a "scope call", it's a
registry expansion:

- **llama** (`arch.rs:198–284`) has **6** KV tiers; only F32 (+ q8 attend via B0) overlap the
  current table. Missing **8 keys + arms + table rows**: `kv_cache_write_hfq4`/
  `attention_hfq4_kv`, `kv_cache_write_hfq8`/`attention_hfq8_kv`, `kv_cache_write_int8c_f16`/
  `attention_int8c_f16_kv`, `kv_cache_write_q4`/`attention_q4kv`. HFQ8/INT8 attend need the
  `k_scales`/`v_scales` `AttnParams` fields actually wired (today declared-but-unused).
- **qwen2** (`qwen2.rs:855–897`) uses F32 KV with **4** attend kernels; only
  `attention_flash_gqa_fused` (`AttnGqaFused`) is registered. Missing **3 keys + arms**:
  `attention_flash` (split-K F32), `attention_flash_gqa`, `attention_gqa_warp`. Selection is
  **env-gated by `HIPFIRE_GQA_FUSED`** (glm5 F10) — that axis must move into `KvTierInputs`
  for qwen2, same pattern as the q8 `use_flash` heuristic.

Migrating these naively (the original B4) would `UnsupportedVariant`-crash or silently fall
back to the slow non-flash F32 path (a real perf regression on OCR/long-context qwen2).

**Disposition:** **3.1 ships qwen35-only (B0–B3, B5).** The llama/qwen2 expansion lands as
**Ship 3.1b**, its own PR, with the 11 new keys + arms + `k_scales`/`v_scales` wiring +
`HIPFIRE_GQA_FUSED` input axis fully planned. qwen35's `kv_cache_attention_dispatch`
(`:12789`) is the only single-GPU decode site that uses **exclusively** the
currently-registered keys — making B0–B3 a clean, self-contained vertical cut.

### Commit B5 · Verification sweep + cleanup

- [ ] Coverage golden incl. RDNA4 + non-WMMA rows for all 18 attention keys (17 + new
      `AttnQ8_0Kv`).
- [ ] **Grep audit (goal #1 gate):** zero `gpu.kv_cache_write_*` / `gpu.attention_flash_*` /
      `gpu.attention_f32` / `gpu.attention_q8_0_kv` calls remain in the qwen35 **decode** path
      (`kv_cache_attention_dispatch` and its `forward_scratch_layers` call sites) — all go
      through `run_attention`. (Prefill/multi-GPU sites are deferred — see Out of scope.)
- [ ] Dispatch-arm-completeness test (B0) green; `KvTierPlan` tier-match `debug_assert` never
      fires in a real run (instrument once).
- [ ] Dev-log fixtures used (model + KV mode + prompt md5 + binary md5) per the bench rule.
- [ ] Prefill paths (`forward_prefill_*`, `run_fa_layer_body`, batched/masked attend) and the
      multi-GPU loop untouched — still pass coherence (they are Ship 3.2 / later).

---

## Risks

1. **Non-flash Q8 gap (B0) is correctness/perf-critical.** The common short-context q8
   decode uses `attention_q8_0_kv`, not the flash kernel. Forgetting B0 silently reroutes
   it. Mitigation: B0 lands first, with the `use_flash` heuristic moved verbatim into
   `KvTierPlan::derive` and a unit test asserting flash vs non-flash key for
   `pos<2048/flash_mode=0` vs `pos≥2048`/`flash_mode∈{1,2}`/`capture_mode`.
2. **Write/attend tier drift (Phase 0.3).** Independent re-derivation of the two halves is
   the #30 shape. Mitigation: single `KvTierPlan` derives both keys together; `debug_assert`
   tiers match before dispatch; the pair is one type threaded through one call.
3. **Resolve-cost per token per layer (Phase 0.1 exemption cost).** KV-tier families
   live-resolve (no init cache), so 2×n_layers `resolve()`/token. Mitigation: the resolve
   is the documented cheap path; `probe_commits.sh` A/B ±1–3% gate on gfx1100 **and**
   gfx1201 is the backstop. If it moves the band, revisit (still no `invalidate()` surface —
   that cross-crate footgun is rejected by 0.1).
4. **Borrow shape at the call site.** `AttnParams`/`AttnIo` carry ~10 refs into the cache
   arrays + scratch; the `&mut kv_cache` + `&s` (immutable scratch) + `&mut gpu` borrows
   must coexist. Mitigation: read cache tensors by index into locals before the call; B2
   compiles before B3 wraps it — if the borrow fights `execute_steps`, that surfaces at B3,
   not silently.
5. **q8 `use_flash` reads `gpu.graphs.capture_mode`** — a GPU/runtime field, not a KV-cache
   field. `KvTierInputs` must capture it at derive time (passed in), keeping `KvTierPlan`
   GPU-free and testable. Don't reach into `gpu` from inside `derive`.
6. **Append-only enum discipline.** `KernelKey` + `PipelineOp` are co-edited with Nick's
   Ship 4. Append `AttnQ8_0Kv` / `PipelineOp::Attend` at the end of their blocks, never
   reorder, ping before adding (boundary contract).

---

## Review adjudication (2026-06-06)

Two adversarial reviews of this plan (`findings/dispatch_3.1_plan_dev_gemini.md`,
`findings/dispatch_3.1_plan_dev_glm5.md`). Disposition of every finding:

| # | Finding (source) | Sev | Verdict | Disposition |
|---|---|---|---|---|
| 1 | llama missing 8 KV/attend keys; qwen2 missing 3 flash/GQA keys → B4 can't compile / regresses (Gemini F1+F2, glm5 F1) | Crit/High | **VALIDATED** | B4 **deferred to Ship 3.1b** (own PR). 3.1 = qwen35-only. |
| 2 | `run_attention` dispatches `plan.write_key` directly, "bypassing resolved fallback key" (Gemini F3) | Crit | **REJECTED as hazard / idiom adopted** | This registry's `resolve(key)` is keyed by the requested key and returns a **same-key** variant or errors — it does **not** remap to a fallback key (confirmed at `tables/mod.rs:32–57`). `resolved.key == key` always; no bypass bug. Adopted the `.key`-from-resolve form anyway (zero cost). Real value of `resolve()?` = the arch-gate. |
| 3 | `KvTierInputs::from_kv_cache` in dispatch crate = circular dep on runtime (Gemini F4) | High | **VALIDATED** | D1: conversion lives runtime-side; dispatch sees only plain scalars. |
| 4 | `AttnIo` duplicates `AttnParams` (Gemini F5, glm5 F9) | Med/Low | **VALIDATED** | D3/B3: `Step::Attend` owns `AttnParams` directly; no `AttnIo`. |
| 5 | `pos` vs `seq_len` off-by-one (`dispatch_attention` does `+1`) (Gemini F6) | Med | **VALIDATED** | D2: `AttnParams.pos` pinned 0-based; caller never passes `pos+1`. |
| 6 | Use `coherence-gate-dflash.sh`; `coherence-gate.sh` "deprecated" (Gemini F7) | Low | **SPLIT** | "deprecated" **REJECTED** (`coherence-gate.sh` is current, replaced `quality-gate.sh`). dflash-gate need **VALIDATED** via glm5 F5 (decode dispatch is shared with the spec-decode path). B2 runs **both**. |
| 7 | Catch-all `_ =>` arm silently swallows a key with a forgotten arm (glm5 F2) | High | **VALIDATED** | B0: populate error fields + GPU-free dispatch-arm-completeness test. |
| 8 | Dead `flash_mode`/`capture_mode` fields post-migration (glm5 F3) | Med | **VALIDATED** | B1: removed from `AttnParams` (decision lives in `KvTierInputs`). |
| 9 | `kv_dim` redundant with `n_kv_heads*head_dim`, stale-stride risk (glm5 F4) | Med | **VALIDATED** | B1: removed; `KvWriteF32` computes locally. |
| 10 | B2 verification underspecified vs bench rules (glm5 F5, Gemini F7) | Med | **VALIDATED** | B2: explicit tier×flash matrix, committed prompts + md5, binary md5, dflash gate. |
| 11 | Undocumented 2nd inline site `run_fa_layer_body` + multi-GPU/batched (glm5 F6) | Low | **VALIDATED** | Out-of-scope: known-sites table added. |
| 12 | `k_scales`/`v_scales` declared-but-unused, needed for llama (glm5 F7) | Low | **VALIDATED** | B1: `TODO(ship 3.1b)` comment; wired in 3.1b. |
| 13 | `HIPFIRE_GQA_FUSED` env axis for qwen2 GQA selection (glm5 F10) | Info | **VALIDATED** | Noted in B4/3.1b scope (moves into `KvTierInputs`). |
| 14 | Key-count "17" self-consistency nit (glm5 F8) | Info | **ACK, no action** | Count correct; becomes 18 after B0 (already noted in B5). |

**Net effect on scope:** B4 leaves 3.1 (→ 3.1b); B0 gains a completeness test; B1 gains an
`AttnParams` cleanup (3 fields removed) + the circular-dep-safe input split; B2 gains an
explicit verification matrix + the dflash gate.

## Out of scope (tracked elsewhere)

| Item | Ship / Phase |
|---|---|
| **llama + qwen2 decode attention** (11 new keys, `k_scales`/`v_scales` wiring, `HIPFIRE_GQA_FUSED` axis) | **Ship 3.1b (own PR)** — see Commit B4 |
| Batched/masked **prefill** attend, PFlash | 3.2 (Phase B) |
| **Adaptive** KV re-resolution (downshift mid-sequence), boundary-layer Q8 forks | 3.2 (Phase B) |
| DFlash WMMA tile attention variants (~18, `shape_gate`), qwen2 dots-ocr pattern | 3.3 (Phase C) |
| MoE path | Ship 4 (Nick) |
| `Step::Rmsnorm` / `Step::SiluMul` pipeline vocab | Ship 6 |

**Known inline attention/KV dispatch sites NOT touched by 3.1** (glm5 F6 — enumerated so a
future author doesn't discover a hidden site mid-migration):

| Site | qwen35.rs | Path | Deferred to |
|---|---|---|---|
| `kv_cache_attention_dispatch` | `:12789` (called `:12299`/`:12508`) | single-GPU **decode** | **migrated by 3.1** |
| `run_fa_layer_body` inline KV dispatch | `:11444` (`:11642–11847`) | single-GPU **prefill** | 3.2 |
| batched prefill attend | `:9318`, `:10954` | batched/masked prefill | 3.2 |
| `forward_scratch_layers_multi` | `:12921+` | **multi-GPU** decode | later |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | — | Plan drafted. Grounded the family as real (17-arm `dispatch_attention`, zero callers, no `KvTierPlan` yet) and the decode site as `qwen35.rs:12789` `kv_cache_attention_dispatch`. **Found the non-flash `attention_q8_0_kv` gap** (only `AttnFlashQ8_0` registered) → added Commit B0. Staged routing B2 (direct paired) → B3 (`Step::Attend`). | — |
| 2026-06-06 | — | Folded two adversarial reviews (gemini + glm5). 12 findings validated, 1 rejected (Gemini F3 — `resolve()` doesn't remap keys; confirmed `tables/mod.rs:32–57`), 1 split (coherence-gate not "deprecated", but dflash gate needed since decode dispatch is shared with spec-decode). **Deferred B4 (llama/qwen2, 11 missing keys) → Ship 3.1b.** Added: B0 dispatch-arm completeness test, B1 `AttnParams` cleanup (drop `flash_mode`/`capture_mode`/`kv_dim`/`kind`) + circular-dep-safe `KvTierInputs` (runtime-side conversion), B2 explicit verification matrix + dflash gate, known-inline-sites table. | — |
