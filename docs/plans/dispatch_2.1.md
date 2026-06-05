# Ship 2.1 — Dense fleet unified (qwen2 + llama through the pipeline)

**Branch:** `feature/dispatch-unification`
**Tracking:** #397 (ship 2)
**Depends on:** Ships 1.1 + 1.2 (treated as landed) — `execute_steps`, `FUSED_TABLE`,
guards, `launch_fused` QKV/gate-up/QKVZA arms, `gemv_steps_uniform[_raw]`.
**Phase 0 contracts:** PR #402 — 0.4 (`HasWmmaW32 → HasWmma`), 0.6 (verification:
`HIPFIRE_FORCE_UNFUSED`, RDNA4 non-optional, byte-identical token streams).

**Goal:** All three dense-only paths (qwen35 done in Ship 1; **qwen2 + llama** here)
route every projection through `execute_steps`. After 2.1, a new dense quant is a
`FUSED_TABLE` entry + kernel file across the dense fleet — no model code changes.

> **Folded in:** the Q4K / Q8_0 fused-table work deferred from Ship 1.2. Per the 1.2
> lesson (no GPU-unexecuted dispatch glue), **each fused entry lands in the same slice
> as the model that exercises it on GPU**: Q8_0 gate+up ↔ qwen2; Q4K QKV/gate-up ↔
> llama. This is why llama is in scope here rather than postponed — without it, Q4K
> would ship untested again.

---

## Grounded state (verified at `715f966c` + 1.1/1.2)

### qwen2 — `crates/hipfire-arch-qwen2/src/qwen2.rs`

- Decode is a **single dense path**: `forward_step_after_x` (qwen2.rs:800–944). No
  DeltaNet/MoE/scratch variants (simpler than qwen35). Prefill =
  `forward_prefill_batch_embeds` → Ship 5.
- Already constructs `DispatchCtx::new(gpu)` (814) and uses `gemv.run_auto` for
  o_proj (907), w_down (932), lm_head (940), and the QKV/gate-up **fallback** arms.
  So qwen2 already calls `GemvFamily` — it just doesn't use `execute_steps`; it has
  inline **dtype `if/else` fast paths**:
  - **QKV (822–841):** `if all HFQ4G256 → gpu.fused_qkv_hfq4g256` else 3× `run_auto`.
    Producer = plain `rmsnorm_f32(x→tmp)` (820).
  - **gate+up (914–930):** `if w_gate/w_up == Q8_0 → gpu.fused_gate_up_q8_0` else 2×
    `run_auto`. Producer = plain `rmsnorm_f32(x→tmp)` (911).
- o_proj (907) = `run_auto` **then separate** `add_inplace_f32` (908). w_down (932) =
  `run_auto` then separate residual (935). lm_head (940) = `run_auto`.
- **Cargo:** qwen2 does **not** depend on `hipfire-dispatch` directly (only
  `hipfire-runtime` + `rdna-compute`); it reaches dispatch types via runtime
  re-exports. Needs a direct dep (cleanliness + future-proofing).

> **Correction to the draft Ship 2 text:** it says "Qwen2: Gate+up … →
> FusedGateUpHfq4G256". **Wrong — qwen2 gate+up is Q8_0** (`fused_gate_up_q8_0`,
> qwen2.rs:920), so it maps to the **new** `FusedGateUpQ8_0` entry, not the HFQ4 one.
> qwen2 is precisely the model that closes the Q8_0 gate+up gap.

### llama — `crates/hipfire-arch-llama/src/arch.rs`

- Decode = `forward_scratch_layers` (arch.rs:134), pre-allocated scratch, `ctx` at 147.
- **Already depends on `hipfire-dispatch`** directly (Cargo:10) — no Cargo change.
- QKV is **three-way branched** by dtype: `if Q4K → gpu.fused_qkv_q4k` (161); MQ-family
  prerotated path using `scratch.x_rot` (183–185); plain `rmsnorm→tmp` + 3× `run_auto`
  (187–190). gate+up mirrors it: `if Q4K → gpu.fused_gate_up_q4k` (315) else MQ/plain
  `run_auto` (337–342). o_proj (310), lm_head (360) = `run_auto`.
- So llama is the **only** arch that exercises Q4K (confirmed: `fused_qkv_q4k` /
  `FusedQkvQ4K` are referenced only by `hipfire-arch-llama` + `hipfire-runtime/llama.rs`).

### Dispatch table state (`steps.rs` / `types.rs` / `fused_qkv_table.rs`)

| Entry | key | family arm | table reg | `FUSED_TABLE` + guard + launch_fused arm | verifier |
|---|---|---|---|---|---|
| `FusedQkvHfq4G256` | ✅ | ✅ | ✅ | ✅ (steps.rs:223, 348) | qwen2 QKV (+ qwen35) |
| `FusedGateUpHfq4G256` | ✅ | ✅ | ✅ | ✅ (steps.rs:233, 364) | (qwen35; qwen2 fallback) |
| `FusedGateUpQ8_0` | ❌ | ❌ | ❌ | ❌ **add (full stack)** | **qwen2 gate+up** |
| `FusedQkvQ4K` | ✅ (types.rs:187) | ✅ | ✅ (`Always`) | ❌ **add interpreter** | **llama QKV** |
| `FusedGateUpQ4K` | ✅ (types.rs:198) | ✅ | ✅ (`Always`) | ❌ **add interpreter** | **llama gate+up** |

---

## Verification-reachability principle (non-negotiable, from the 1.2 lesson)

No fused entry merges without a same-PR forward that executes it on GPU under
byte-parity:
- **Q8_0 gate+up** is reached only by qwen2 FFN → land it **with** the qwen2
  migration (Slice A), verified on a qwen2 model whose FFN is Q8_0.
- **Q4K** is reached only by llama → land it **with** the llama migration (Slice B),
  verified on a Q4K llama model.
- GPU-free goldens (resolve + `match_prefix` + `force_unfused` reject incl. RDNA4 row)
  are **necessary but not sufficient** — they don't execute the kernel.

---

## Producer-step decision (resolves the draft's "Step::Rmsnorm vs bare-Gemv")

Use **`RmsnormAutomatic(rotation=None)`** for Q4K and Q8_0 — it lowers to a plain
`gpu.rmsnorm_f32` (steps.rs:258–261), exactly what these non-rotated kernels want
(both take a pre-normed `x`, no internal rmsnorm). **Do not** add a `Step::Rmsnorm`
variant: it would duplicate the `rotation=None` branch. `Step::Rmsnorm` belongs to the
Ship 6 forward-as-pipeline capstone (glm5 F9 on the 1.2 review) — deferred, not
rejected. Q4K/Q8_0 are not rotated, so their Gemv steps use **`GemvInput::Prerotated`**
(the buffer holds plain rmsnorm), matching the existing HFQ4 guards
(`gemv_steps_uniform`, Prerotated) — **no Raw-guard needed** (that was Paro-only).

---

## Plan

### Slice A · qwen2 + `FusedGateUpQ8_0` (verified on qwen2)

#### Commit A1 · `FusedGateUpQ8_0` full dispatch stack

1. **types.rs:** add `FusedGateUpQ8_0`.
2. **fused_qkv.rs:** add `run` arm → `gpu.fused_gate_up_q8_0(wg.buf, wu.buf, x, gate,
   up, wg.m, wu.m, wg.k)` (single `x`, 2 weights, 2 outputs, **no scratch**; matches
   qwen2.rs:920). Assert `wg.k == wu.k` (the kernel/qwen2 precondition at 918).
3. **fused_qkv_table.rs:** register with `ArchPredicate::Always` — confirm
   `fused_gate_up_q8_0` uses no WMMA/dp4a-gated instruction (Q8_0 GEMV is `Always`);
   if it does, gate accordingly.
4. **steps.rs:** `GATE_UP2` `FusedPattern` for `FusedGateUpQ8_0` + `guard_gate_up_q8_0`
   (`force_unfused` early-return → `window_gemv_dtype == Q8_0` → `gemv_steps_uniform`
   (Prerotated)). Extend the existing 2-way `launch_fused` gate+up arm match list to
   include the key.

**Verify:** GPU-free golden (resolve + select + force_unfused reject, incl. RDNA4).
GPU byte-parity comes with A2 (qwen2 executes it).

#### Commit A2 · qwen2 → `execute_steps`; delete inline dtype branches

1. **Cargo.toml:** add `hipfire-dispatch = { path = "../hipfire-dispatch", features =
   ["from-hip-error"] }` (mirror llama).
2. **forward_step_after_x:** replace the inline fast/fallback branches with
   `execute_steps`:
   - **QKV (delete 822–841):** `execute_steps([RmsnormAutomatic(None){x→tmp},
     Gemv(Prerotated tmp)→q, →k, →v])`. Matcher picks `FusedQkvHfq4G256` for HFQ4,
     per-op otherwise. Bias (852–854), RoPE, KV write, attention stay inline
     (attention = Ship 3).
   - **gate+up (delete 914–930):** `execute_steps([RmsnormAutomatic(None){x→tmp},
     Gemv(Prerotated tmp)→gate, →up])`. Matcher picks `FusedGateUpQ8_0` for Q8_0,
     `FusedGateUpHfq4G256` for HFQ4, per-op otherwise.
   - **o_proj (907–908):** `execute_steps([GemvResidual{attn_out, residual=x}])` —
     fuses the separate `add_inplace_f32`.
   - **w_down (932,935):** `execute_steps([GemvResidual{ffn_hidden, residual=x}])`.
   - **lm_head (940):** `execute_steps([Gemv{tmp→logits}])`.
3. The explicit `if all_hfq4g256` / `if Q8_0` dtype checks **disappear** — the
   interpreter does dtype dispatch. This is the "model says *what*, not *how*" win.

**Verify (qwen2, on-GPU):**
- Byte-identical committed token IDs **vs master** (`HIPFIRE_EMIT_TOKEN_IDS=1`, temp
  0.0, fixed prompt + md5), gfx1100 **and gfx1201**, on a qwen2 model with **HFQ4G256
  QKV + Q8_0 FFN** (exercises both `FusedQkvHfq4G256` and `FusedGateUpQ8_0`). If the
  fleet has no such recipe, state the gap and pick the nearest (HFQ4 FFN exercises the
  HFQ4 gate+up entry; Q8_0 then needs a dedicated fixture).
- `HIPFIRE_FORCE_UNFUSED` byte-parity (these are Prerotated/non-rotated paths — unlike
  Paro, fused vs per-op should be byte-identical; if a delta appears, treat as a real
  divergence, not noise).
- `probe_commits.sh master HEAD` ±1–3% (o_proj/w_down residual fusion should be neutral
  or a small win); `coherence-gate.sh` on qwen2 weights.

### Slice B · Q4K interpreter wiring + llama (verified on llama)

#### Commit B1 · `FusedQkvQ4K` + `FusedGateUpQ4K` interpreter wiring

Keys/family/table already exist — only the interpreter is missing.
1. **steps.rs:** `QKV3` `FusedPattern` for `FusedQkvQ4K`, `GATE_UP2` for
   `FusedGateUpQ4K`; `guard_qkv_q4k` / `guard_gate_up_q4k` (`force_unfused` →
   `window_gemv_dtype == Q4K` → `gemv_steps_uniform` Prerotated). Extend the existing
   3-way QKV and 2-way gate+up `launch_fused` arms to include the Q4K keys (single-`x`
   extraction, no scratch — same shape as HFQ4).

**Verify:** GPU-free golden (incl. RDNA4). GPU parity with B2.

#### Commit B2 · llama Q4K call sites → `execute_steps`

1. **arch.rs QKV (161, 187–190):** replace the `if Q4K → fused_qkv_q4k` branch (and,
   for a fully-migrated function, the plain `rmsnorm→tmp` branch) with
   `execute_steps([RmsnormAutomatic(None){x→tmp}, Gemv(Prerotated)→q,→k,→v])`. Matcher
   picks `FusedQkvQ4K` for Q4K, per-op otherwise.
2. **arch.rs gate+up (315, 340–342):** `execute_steps([RmsnormAutomatic(None),
   Gemv×2])` → `FusedGateUpQ4K` / per-op.
3. **o_proj (310), lm_head (360):** `execute_steps([GemvResidual])` / `([Gemv])`.
4. **MQ-family prerotated branch (183–185, 337–338):** this is the FWHT/`x_rot` path
   (rotation ≠ None). Two options:
   - **B2-min:** migrate **only** the Q4K + plain branches; leave the MQ prerotated
     branch as-is. Lower risk, but leaves a half-migrated function (the 1.1-review
     anti-pattern).
   - **B2-full (recommended if time):** migrate the MQ branch too —
     `execute_steps([RmsnormAutomatic(rotation=<plan>), Gemv(Prerotated x_rot)×3])`
     picks the fused MQ QKV/gate-up kernels (a perf *win* for llama MQ, currently 3×
     `run_auto`). Reuses the qwen35 producer-rotation contract from 1.1.
   Decide per the A/B and the half-migration tradeoff; default B2-full.

**Verify (llama, on-GPU):** byte-identical vs master on a **Q4K llama** model (gfx1100
+ gfx1201); `HIPFIRE_FORCE_UNFUSED` byte-parity (non-rotated Q4K → byte-identical
expected; MQ branch under B2-full → see 1.1/1.2 rotation parity rules);
`probe_commits.sh master HEAD` ±1–3%; `coherence-gate.sh`.

### Commit C · Verification sweep + cleanup

- [ ] `(op × dtype × arch)` coverage golden incl. RDNA4 row for `FusedGateUpQ8_0`,
      `FusedQkvQ4K`, `FusedGateUpQ4K` (Phase 0.4 gate).
- [ ] qwen2 + llama coherence gates green on gfx1100 + gfx1201.
- [ ] Grep audit: no inline `gpu.fused_qkv_hfq4g256` / `gpu.fused_gate_up_q8_0` /
      `gpu.fused_qkv_q4k` / `gpu.fused_gate_up_q4k` call sites remain in qwen2.rs /
      arch.rs decode paths (all reach the kernel via `FusedQkvFamily::run`).
- [ ] Confirm `forward_prefill_batch_embeds` (qwen2) and llama prefill untouched
      (Ship 5) still pass coherence.
- [ ] Dev-log which qwen2 / llama model+quant fixtures were used (with prompt md5).

---

## Risks

1. **Silent perf no-op.** As in 1.2: if a guard fails to fire, output stays correct but
   the fused kernel never runs. Backstop: `probe_commits.sh` gain-vs-parent (qwen2
   gate+up Q8_0 and llama Q4K should show the fused win) + a debug-build assert that the
   intended `launch_fused` arm was reached ≥ once/forward.
2. **qwen2 verification fixture for Q8_0.** The byte-parity check is only meaningful if
   the test model's FFN is actually Q8_0. If no fleet qwen2 has Q8_0 FFN, `FusedGateUpQ8_0`
   reverts to the 1.2 "unverified" problem — must source/confirm a fixture before A1
   merges. (This is the linchpin; verify it exists first.)
3. **o_proj / w_down residual fusion** (run_auto + separate add → `GemvResidual`) is a
   numerics change on paper; for F32 elementwise add it should be byte-identical —
   confirm under the master byte-parity check, don't assume.
4. **llama half-migration (B2-min).** Leaving the MQ branch un-migrated reproduces the
   two-dispatch-styles smell the 1.1 review flagged. Prefer B2-full unless A/B blocks.
5. **`Always` predicate for Q8_0.** Verify `fused_gate_up_q8_0` truly has no
   arch-gated instruction before registering `Always` (it's the one new predicate
   choice here; a wrong `Always` would panic on an arch missing the path).

---

## Out of scope (tracked elsewhere)

| Item | Ship |
|---|---|
| qwen2 / llama **prefill** (`forward_prefill_batch_embeds`, llama batched) | Ship 5 |
| Attention + KV cache (qwen2 flash/GQA, llama KV) | Ship 3 |
| `Step::Rmsnorm` variant | Ship 6 (forward-as-pipeline) |
| qwen2 QKV **bias** fusion (option (c) batched bias-add, qwen2.rs:843–851) | not in dispatch scope |
| MoE archs (qwen35 MoE, deepseek4) | Ship 4 |
| Phase 0.4 `HasWmmaW32 → HasWmma` collapse | Phase 0 cleanup |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-05 | — | Plan written; folded Q4K/Q8_0 from 1.2; llama pulled into scope so Q4K has a GPU verifier; corrected draft's qwen2-gate+up-is-Q8_0 error | — |
