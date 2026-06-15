# Ship 2.1 — Dense fleet unified (qwen2 + llama through `execute_steps`)

**Branch:** `feature/dispatch-unification`
**Tracking:** #397 (ship 2)
**Depends on:** Ships 1.1 + 1.2 (landed on this branch: `execute_steps`,
`FUSED_TABLE`, guards, `launch_fused` arms, `gemv_steps_uniform[_raw]`, Phase 0.4
`HasWmma`).
**Parallel work to reconcile:** `upstream/integration/dispatch-migration` (a different
dev, divergent off `715f966c`, **not in our history**): `6ded4332` ("migrate llama Q4K
+ qwen2 Q8_0 fused launches onto FusedQkvFamily") and `adfcbc6` ("migrate qwen2
HFQ4G256 QKV onto FusedQkvFamily + widen dead-gate"). Both use the direct-`fused.run`
style → stepping stones to convert to `execute_steps`.
**Target architecture (decided):** **full `execute_steps`**, fleet-consistent with
qwen35 — *not* the direct-`FusedQkvFamily::run` style `6ded4332` used.

**Goal:** qwen2 + llama route every projection through `execute_steps` (`[Step]` →
`FUSED_TABLE` → kernel), with **no model-side dtype branching and no model naming a
kernel key**. After 2.1, a new dense quant is a `FUSED_TABLE` entry + kernel file
across the whole dense fleet — goals #1 and #3 met.

---

## How `6ded4332` changes the starting point

`6ded4332` did a **behavior-preserving** rewire of the existing fused fast-path calls
onto `FusedQkvFamily::run`, **keeping the inline `if dtype == … { fused.run(kind) }
else { run_auto }` branches**. It centralizes the *kernel implementation* but the
model still says *how* (picks the key, does the dtype match) — so it does **not** meet
goals #1/#3 and leaves qwen2/llama inconsistent with qwen35. Under the "full
execute_steps" decision it is a **stepping stone**, not the destination.

**Reuse, don't redo** (merge these family-layer additions in; they're correct and
arch-agnostic):
- `KernelKey::FusedGateUpQ8_0` + its `Always` table row + `FusedQkvFamily::run` arm.
- `fused_qkv_family()` runtime helper + `FusedQkvParams`/`KernelKey` re-exports via
  `hipfire_runtime::llama`.

**Discard / supersede** (the parts that conflict with execute_steps):
- The inline `fused.run(kind: …)` call sites in llama `arch.rs` and qwen2 `qwen2.rs`
  — replaced by `execute_steps([Step])` (A2/A3 below). The end state has *zero*
  `fused.run` or `gpu.fused_*` calls in the decode paths.

> Whether we land 2.1 on top of the integration branch (start from `fused.run`) or on
> our own branch (start from raw `gpu.fused_*`), the destination is identical:
> `execute_steps`. The intermediate state doesn't change the work below.

---

## The HFQ4 predicate dead-gate — QKV fixed by `adfcbc6`, gate+up residual

`6ded4332` originally deferred qwen2 `fused_qkv_hfq4g256` because the family arm
`FusedQkvHfq4G256` was `HasWmma`-gated while the kernel is cross-arch precompiled
(generic wave32 + CDNA wave64), so routing it through the family would resolve to
`UnsupportedVariant` on non-WMMA archs (RDNA1/RDNA2/CDNA) — CI-enforced via the coverage
gate's `["gfx1100", "gfx1030", "gfx906"]` rows (`hipfire-dispatch-tests/src/qwen2.rs:12`).
Phase 0.4's `HasWmma` does **not** fix it (still excludes non-WMMA).

**`adfcbc6` resolved this for QKV:** `FusedQkvHfq4G256` → `ArchPredicate::Always`
(mirrors the `FusedQkvQ4K` row; correct, since the kernel runs everywhere). So A0-QKV
is **done** (on the integration branch — pending merge).

**Residual:** `adfcbc6` did **not** widen `FusedGateUpHfq4G256` — still `HasWmma`. The
gate+up kernel also has cross-arch variants (`fused_gate_up_hfq4g256` +
`…_dp4a`), so it has the same latent dead-gate. It only bites once a model routes HFQ4
gate+up through the interpreter on a non-WMMA arch (qwen2 gate+up is Q8_0, llama is
Q4K/MQ/plain — so not 2.1's primary verifiers, but qwen35 HFQ4 gate+up + the coverage
matrix make it worth fixing for consistency). → folded into A0 below as the remaining
item.

**Merge note:** the `FusedQkvHfq4G256` predicate now has a 3-way divergence — base
`HasWmmaW32`, our branch `HasWmma` (0.4 `dfe7231e`), integration `Always` (`adfcbc6`).
**Resolve to `Always`.**

---

## Plan

### Commit A0 · HFQ4 fused predicate — QKV done (`adfcbc6`); finish gate+up

**QKV: done** — `adfcbc6` set `FusedQkvHfq4G256` → `Always`. On merge, resolve the
3-way predicate divergence to `Always` (see merge note above).

**Remaining — gate+up:** widen `FusedGateUpHfq4G256` the same way. Confirm
`gpu.fused_gate_up_hfq4g256` (+ `…_dp4a`) is cross-arch precompiled (it mirrors the QKV
kernel), then set its predicate to `Always` (or a dp4a-ladder + `Always` fallback if
the dp4a sibling should be preferred on dp4a archs). This keeps HFQ4 gate+up resolvable
on non-WMMA archs once it's reached through the interpreter (qwen35 + coverage matrix).

**Verify:** coverage golden green for HFQ4 QKV **and gate-up** across `gfx1100,
gfx1030, gfx906` **and RDNA4** — resolves to a valid path on every row, fused where
supported.

### Commit A1 · Interpreter wiring for Q8_0 + Q4K (family layer already exists)

The `FusedGateUpQ8_0` (from `6ded4332`) and `FusedQkvQ4K`/`FusedGateUpQ4K`
(pre-existing) keys/arms/table rows exist; only the **interpreter** layer is missing.

1. **steps.rs:** add `FUSED_TABLE` rows — `GATE_UP2`→`FusedGateUpQ8_0`,
   `QKV3`→`FusedQkvQ4K`, `GATE_UP2`→`FusedGateUpQ4K`.
2. **guards:** `guard_gate_up_q8_0`, `guard_qkv_q4k`, `guard_gate_up_q4k` —
   `force_unfused` early-return → `window_gemv_dtype == {Q8_0|Q4K}` →
   `gemv_steps_uniform` (Prerotated; these are non-rotated, so plain rmsnorm output
   feeds the kernel — **no Raw-guard**, that was Paro-only).
3. **launch_fused:** extend the existing `QKV3` and `GATE_UP2` arms' key match-lists to
   include the three keys (single-`x` extraction, no scratch — same shape as HFQ4).
4. Producer for all: `RmsnormAutomatic(rotation=None)` (lowers to plain rmsnorm;
   the kernels take a pre-normed `x`). **No `Step::Rmsnorm`** — deferred to Ship 6.

**Verify:** GPU-free coverage goldens (resolve + `match_prefix` select + force_unfused
reject) incl. RDNA4. GPU byte-parity lands with A2/A3 (the models execute them).

### Commit A2 · qwen2 → `execute_steps`; delete inline branches

`crates/hipfire-arch-qwen2/src/qwen2.rs::forward_step_after_x`. Add
`hipfire-dispatch` as a **direct** dep (qwen2 currently reaches it only via
`hipfire-runtime` re-exports). Replace each projection:

- **QKV (delete the `if all_hfq4g256 {…} else {…}`):**
  `execute_steps([RmsnormAutomatic(None){x→tmp}, Gemv(Prerotated tmp)→q,→k,→v])`.
  Matcher → `FusedQkvHfq4G256` (now resolvable on all archs via A0) or per-op. Bias
  (852–854), RoPE, KV write, attention stay inline (attention = Ship 3).
- **gate+up (delete the `if Q8_0 {…} else {…}`):**
  `execute_steps([RmsnormAutomatic(None){x→tmp}, Gemv(Prerotated tmp)→gate,→up])`.
  Matcher → `FusedGateUpQ8_0` / `FusedGateUpHfq4G256` / per-op.
- **o_proj (907–908):** `execute_steps([GemvResidual{attn_out, residual=x}])` (fuses
  the separate `add_inplace_f32`).
- **w_down (932,935):** `execute_steps([GemvResidual{ffn_hidden, residual=x}])`.
- **lm_head (940):** `execute_steps([Gemv{tmp→logits}])`.

The inline dtype `if` chains disappear — the interpreter does dtype dispatch.

**Verify (qwen2, on-GPU):** byte-identical token IDs **vs master**
(`HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0, fixed prompt + md5) on gfx1100 + gfx1201, on a
qwen2 model with **HFQ4G256 QKV + Q8_0 FFN** (exercises both new fused entries — **the
fixture must exist; confirm before A2**, see Risk 2). `HIPFIRE_FORCE_UNFUSED`
byte-parity (non-rotated → byte-identical expected). `probe_commits.sh master HEAD`
±1–3%. `coherence-gate.sh`.

### Commit A3 · llama → `execute_steps`; delete inline branches

`crates/hipfire-arch-llama/src/arch.rs::forward_scratch_layers` (already deps
`hipfire-dispatch`). llama QKV/gate-up are **three-way branched** (Q4K / MQ-rotated /
plain) — migrate all three for full coverage (avoid the 1.1-review half-migration smell):

- **QKV (161 Q4K, 183–185 MQ-prerotated, 187–190 plain):**
  `execute_steps([RmsnormAutomatic(rotation=<plan>){…}, Gemv×3])`. `rotation=None` for
  Q4K/plain (Prerotated input over plain rmsnorm); `rotation=<MQ plan>` for the MQ
  branch (reuse the qwen35 1.1 producer-rotation + Prerotated contract). Matcher →
  `FusedQkvQ4K` / fused MQ QKV (a **win** vs today's 3× `run_auto` on the MQ branch) /
  per-op.
- **gate+up (315 Q4K, 337–342 MQ/plain):** same shape, 2-way →
  `FusedGateUpQ4K` / fused MQ / per-op.
- **o_proj (310):** `execute_steps([GemvResidual])`. **lm_head (360):**
  `execute_steps([Gemv])`.

**Verify (llama, on-GPU):** byte-identical vs master on a **Q4K llama** model (gfx1100
+ gfx1201); MQ branch parity per the 1.1/1.2 rotation rules (fused-vs-master
byte-identical; force-unfused coherence/cosine if the MQ fused vs per-op rotation paths
differ); `probe_commits.sh master HEAD` ±1–3%; `coherence-gate.sh`.

### Commit A4 · Verification sweep + cleanup + merge reconciliation

- [ ] Coverage golden incl. RDNA4 + non-WMMA rows (`gfx1030`, `gfx906`) for HFQ4 (A0),
      Q8_0, Q4K, and the MQ entries newly reachable from llama.
- [ ] **Grep audit (the goal-#1 gate):** zero `gpu.fused_*` and zero
      `FusedQkvFamily::run` / `fused.run(` calls remain in qwen2/llama **decode** paths
      — everything goes through `execute_steps`.
- [ ] qwen2 + llama coherence gates green on gfx1100 + gfx1201.
- [ ] Prefill paths (`forward_prefill_batch_embeds`, llama batched) untouched (Ship 5),
      still pass coherence.
- [ ] **Merge reconciliation** with `integration/dispatch-migration`: fold its
      family-layer additions (`FusedGateUpQ8_0`, `fused_qkv_family()`) and its qwen35
      MTP/multi-GPU commits (`c87eb0a8`, `4033d594`, `81f4609c`); ensure its
      `fused.run` call sites (`6ded4332` llama Q4K + qwen2 Q8_0; `adfcbc6` qwen2 HFQ4
      QKV) are converted to `execute_steps`, not double-applied. **Resolve the
      `FusedQkvHfq4G256` predicate 3-way conflict (base `HasWmmaW32` / ours `HasWmma`
      / integration `Always`) → `Always`.**
- [ ] Dev-log the qwen2/llama fixtures used (model + quant + prompt md5).

---

## Risks

1. **A0 mostly landed (`adfcbc6` fixed QKV `Always`); the gate+up residual still
   precedes any interpreter path that hits HFQ4 gate+up on a non-WMMA arch.** Don't
   migrate HFQ4 gate+up through the interpreter before widening `FusedGateUpHfq4G256`,
   or `gfx1030`/`gfx906` coverage goes red.
2. **Q8_0 verification fixture (linchpin) — RESOLVED (option a implemented 2026-06-05).**
   Key insight: the runtime hfq loader already maps **quant_type 3 (`Q8F16`) → `DType::Q8_0`**
   (`hfq.rs:632`, "Q8F16 — same block format as GGML Q8_0"), so no new `QuantType` was needed.
   Added **`--format hfq4-q8ffn`** to `hipfire-quantize` (inverse of `hfq-mixed`): HFQ4 for
   attn (q/k/v/o), `Q8F16`→`Q8_0` for `mlp.*` + embed/lm_head. This is the exact qwen2 A2
   recipe — HFQ4 QKV → `FusedQkvHfq4G256`, Q8_0 FFN → `FusedGateUpQ8_0`. *(Earlier "BLOCKED /
   no QuantType::Q8_0" analysis missed the loader's code-3→Q8_0 mapping; corrected.)*

   **Fixtures built (2026-06-05, dense safetensors → `hipfire-quantize`):**
   - `/data/hipfire/qwen2-1.5b.hfq4-q8ffn.hfq` (`--format hfq4-q8ffn`, Qwen2-1.5B) → **A2
     linchpin**: HFQ4 q/k/v/o + Q8_0 mlp → exercises `FusedQkvHfq4G256` **and**
     `FusedGateUpQ8_0`. Dense Qwen2 (no QK-norm) for the qwen2 crate. mean err 2e-4.
   - `/data/hipfire/qwen3-0.6b.hfq4.hfq` (`--format hfq4`, Qwen3-0.6B) → `FusedQkvHfq4G256` +
     `FusedGateUpHfq4G256`; q_norm/k_norm present (`has_qk_norm`), llama arch.
   - `/data/hipfire/qwen3-0.6b.q4k.hfq` (`--format q4k`, Qwen3-0.6B) → all-Q4_K → **A3**
     `FusedQkvQ4K` + `FusedGateUpQ4K`. *(Q4K **is** producible via `--format q4k`/`use_q4k_all`.)*
   All dense (arch_id 1), small enough for fast on-GPU smoke. (A `hfq4-q8ffn` Qwen3-0.6B can be
   built the same way for the llama-arch HFQ4+Q8 combo if needed.)
3. **Silent perf no-op.** If a guard fails to fire, output stays correct but the fused
   kernel never runs. Backstop: `probe_commits.sh` gain-vs-parent + a debug assert that
   the intended `launch_fused` arm was reached ≥ once/forward.
4. **Merge order with the integration branch.** If `6ded4332` merges first, A2/A3 must
   *replace* its `fused.run` sites (not leave both). If our branch merges first, we go
   raw→execute_steps and the integration branch's `fused.run` sites become conflicts to
   resolve toward execute_steps. Either way, end state = no `fused.run` in decode.
5. **o_proj/w_down residual fusion** (`run_auto`+separate add → `GemvResidual`): F32
   elementwise add should be byte-identical — confirm under the master byte-parity
   check, don't assume.
6. **llama MQ-branch rotation parity** (A3-full): fused MQ vs per-op may differ in FP
   order (cf. 1.2 Paro) — use fused-vs-master as the byte oracle, coherence/cosine for
   force-unfused.

---

## Out of scope (tracked elsewhere)

| Item | Ship |
|---|---|
| qwen2 / llama **prefill** | Ship 5 |
| Attention + KV cache (qwen2 flash/GQA, llama KV) | Ship 3 |
| `Step::Rmsnorm` variant | Ship 6 (forward-as-pipeline) |
| qwen2 QKV **bias** fusion (`qwen2.rs:843–851`) | not in dispatch scope |
| MoE archs (qwen35 MoE, deepseek4) | Ship 4 |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-05 | — | Rewrote around parallel work `6ded4332` (family layer done on integration branch) + decision = full execute_steps. Added A0 (HFQ4 predicate re-arch) as prerequisite; A1 now interpreter-only wiring; A2/A3 replace `fused.run`+inline-if with `execute_steps`; added merge reconciliation. | — |
| 2026-06-05 | — | A0-QKV resolved by `adfcbc6` (`FusedQkvHfq4G256` → `Always`); A0 reduced to the `FusedGateUpHfq4G256` gate+up residual. Noted the 3-way predicate merge conflict (→ `Always`) and `adfcbc6` as another `fused.run` stepping stone to convert. | — |
