# Ship 4.1 — qwen35 MoE decode → `MoeFamily` owns resolution + ctx (fork ii)

**Branch:** `feature/ship-4.1-moe-family-resolution` off `integration/dispatch-unification`
(per `feedback_branch_per_feature`).
**Tracking:** [#397](https://github.com/Kaden-Schutt/hipfire/issues/397) Step 4.1.
**Owner:** Kevin — **taken over from Nick** (Kevin aligns with Nick out-of-band). Fork (ii)
edits `families/moe.rs` **and** `pipeline/mod.rs` (the MoE dispatch core, normally Nick's Ship-4
surface), so this is a deliberate cross-lane takeover, not a qwen35-only cut. See Ownership note.
**Depends on:** **Ship 3.3 landed** (shipped, incl. most tri-review bug fixes; F-1 WMMA grid fix
validated on gfx11 — `findings/dispatch_3.3_f1_wmma_validation.md`). **PR #428 merged**
(`MoeFamily::run` real, `moe_family()`, restored CPU-top-K fallback, MoE coverage row,
`MoeResolution`/`MoeDtypes` extracted). No 3.3-validation precondition gates 4.1 — 3.3 is done.
**Phase 0 contracts:** 0.1 (KV-tier exempt — N/A to MoE), 0.2 (scratch arch-owned, family takes
`&mut`), 0.5 (family API checklist — typed `run*()`, **no** model-side dispatch decision), 0.6
(same-binary byte-parity, gfx1100 + gfx1201 probe).

**Goal (fork ii — the substantive Goal-#2 win):** the qwen35 MoE decode FFN stops **deciding how
to dispatch**. Today the *model* computes the fused-vs-fallback routing
(`MoeResolution::resolve`) and passes the verdict in `MoeParams.res`; `MoeFamily::run` is a
ctx-ignoring delegate. After 4.1 the **family** owns resolution (model passes only its weight
`MoeDtypes` + `k` — the "what"), and a single `DispatchCtx` is threaded through the executor to
every inner GEMV (eliminating 6+ per-token `DispatchCtx::new` reconstructions). A new MoE routing
config becomes family logic, not a model edit — Goal #2 for MoE decode.

> Folds three adversarial plan reviews: `dispatch_4.1.plan_rev_claude.md` (A1–A10),
> `dispatch_4.1.plan_rev_gemini.md` (G-F1…F7), `dispatch_4.1.plan_rev_ds4.md` (D-F1…F12).
> Per-finding disposition in the [Review adjudication](#review-adjudication) table. Accepted items
> are folded into the design + commits below; finding tags `[Fn]` mark where.

---

## Grounded starting state (verified on the branch tip)

> Function names are the stable anchors; line numbers drift.

### Resolution is already a pure function — the model just calls it in the wrong place `[A1]`

- `MoeResolution::resolve(d: &MoeDtypes, k: usize) -> MoeResolution` (`moe.rs:63`) is a **pure,
  ctx-free** function whose own doc says it *"IS the routing-config logic, relocated … into one
  typed, testable place (review finding #1)."* So the *logic* already lives in the dispatch crate.
- But the **call site is in the model**: `qwen35.rs` `moe_ffn_decode_impl` builds `moe_dtypes`
  (`:4579`), calls `MoeResolution::resolve(&moe_dtypes, k)` (`:4600`), and passes the verdict as
  `MoeParams.res` (`:4616`). The model thus makes the dispatch decision — the exact "model says
  *how*" that Goal #2 forbids. `moe_res` is **only** consumed by passing it in (no model branch on
  it — verified), so the call relocates cleanly.
- `MoeFamily::run(&self, _ctx, gpu, p)` (`moe.rs:314`) ignores `_ctx` and calls
  `run_moe_decode(gpu, p)`. `run_moe_decode` reads `let res = p.res;` (`pipeline/mod.rs:147`).

### `DispatchCtx` is reconstructed 6+ times per MoE layer per token `[G-F1,G-F2,G-F3,D-F8,A3]`

- `run_moe_decode` builds `DispatchCtx::new(gpu)` at **5 sites** (`pipeline/mod.rs:198, 267, 429,
  449, 459`). `449`/`459` are **inside the per-expert `for` loop** of
  `run_moe_decode_cpu_fallback` (`:361`) — for the k≠8/non-indexable fallback that is K×2 = 16
  ctx/layer (≈480/token on a 30-layer model). `DispatchCtx::new` runs `FeatureFlags::from_env()`
  (env reads) + `ArchCaps::new()` each time.
- At the qwen35 call site, the layer's `ctx` is **block-scoped and drops before** the MoE call
  (`qwen35.rs:12969-12973` — `{ let ctx…; execute_steps(…); }` then `moe_ffn_decode_with_scratch`).
  So there is no reusable ctx in scope; D3a-as-originally-planned is not available `[G-F1,D-F3]`.

### Runtime-guard precedent for batch_size already exists `[D-F2,D-F5,G-F4]`

`run_moe_decode_bias_aware` (`pipeline/mod.rs:485`) guards the *identical* invariant with a
**runtime** error, not a `debug_assert`:
```rust
if p.batch_size != 1 {
    return Err(DispatchError::UnsupportedVariant { family: "moe",
        variant: "bias-aware-decode-requires-batch-1", arch: "", quant: "" });
}
```
`MoeParams` has **no** `batch_size` field (only `MoeBiasAwareParams` does). `run_moe_decode`
hardcodes `1` as a literal in its kernel calls (e.g. `moe_down_combine_k8_batched(…, p.k, 1)`) —
synchronized to the decode-only assumption by convention only `[D-F4]`.

### Verification surface gaps

- `coherence-gate-dflash.sh` targets **dense 27B** only — it has **no MoE model**, so it never
  exercises `moe_ffn_decode_impl` `[D-F1,A6]`. A3B MoE DFlash *does* work (AGENTS.md §"Pinned A3B
  MoE DFlash fixtures": 3.5-A3B τ=4.91; target md5 `edde51ec`, draft md5 `8254bbe1`). The pinned
  **draft** model `qwen36-35b-a3b-dflash-mq4.hfq` is **missing locally** `[G-F7]`.
- `coherence-gate.sh` A3B cells are gated on `--full` (`FULL_EXTRA`, only when `FULL=1`) `[D-F6]`.
- A3B is **k=8 → `use_gpu_topk`** (`moe.rs:83`) — an A3B-only fixture never exercises the
  **k≠8 CPU-top-K fallback** the change also routes `[A5,G-F6]`.
- `routed_experts: Vec<…>` is allocated per call (`qwen35.rs:4606`) — pre-existing, not 4.1's `[D-F7]`.

---

## Design decisions (fork ii)

### D1 · The family owns resolution; the model passes only `MoeDtypes` `[A1, Goal #2]`

- `MoeParams`: **replace `res: MoeResolution` with `dtypes: MoeDtypes`** (the model already builds
  `moe_dtypes` at `qwen35.rs:4579` — it just passes it instead of the resolved verdict). `k` stays.
- `run_moe_decode` computes `let res = MoeResolution::resolve(&p.dtypes, p.k);` at the top
  (replacing `let res = p.res;`). Because resolution moves into the **executor**, all three callers
  (`MoeFamily::run`, `execute_pipeline`'s `PipelineParams::Moe` arm, `dispatch_fused`'s arm) get it
  for free; the model stops resolving.
- The model deletes the `MoeResolution::resolve` call (`:4600`) and passes `dtypes: moe_dtypes`.

### D2 · Thread one `DispatchCtx` end-to-end; delete the 6 internal reconstructions `[G-F2,G-F3,D-F8,A3]`

This is the change that makes `run()`'s `ctx` parameter *mean* something and kills the per-expert
cliff:

- `run_moe_decode(ctx: &DispatchCtx, gpu, p)` and `run_moe_decode_cpu_fallback(ctx, …)` take a
  `ctx` and **forward it to every `gemv.run_auto`**. Delete all 5 internal `DispatchCtx::new` and
  the per-expert-loop constructions.
- `MoeFamily::run` forwards its `ctx` (stop ignoring it). `execute_pipeline`'s Moe early-return
  (`:49`) forwards its `ctx` param. `dispatch_fused`'s Moe arm (`:766`): add a `ctx` param to
  `dispatch_fused` and forward (verify reachability — MoE early-returns in `execute_pipeline`
  before `find_fused`, so this arm may be defensive; thread ctx regardless for correctness).
- Net: **one** ctx per MoE decode invocation (built once at the qwen35 call site), down from 6+.
  This fixes G-F3 for free and dissolves D3 (no block-scope problem — the family builds/receives
  the single ctx).

### D3 · Runtime batch guard matching the bias-aware precedent `[D-F2,D-F5,G-F4,A2]`

`run_moe_decode` opens with a **runtime** guard (not `debug_assert` — 6436bd1 was silent-in-prod):
```rust
if p.batch_size != 1 {
    return Err(DispatchError::UnsupportedVariant { family: "moe",
        variant: "decode-requires-batch-1", arch: "", quant: "" });
}
```
matching `run_moe_decode_bias_aware`. Add `MoeParams.batch_size` (model sets `1`). Add
`// FIXME(Step 8): replace hardcoded 1 with p.batch_size when grouped prefill lands` at each
hardcoded `1` kernel literal — the field is a real plumbing guard for Step 8, **documented as
inert today** (decode-only path) `[D-F4,A2]`.

### D4 · Entry point `moe_family().run`; no `Step::Moe` `[D-F10,A1]`

The qwen35 site calls `moe_family().run(&ctx, gpu, &params)` — now a real typed entry (ctx used,
resolution owned). `Step::Moe` stays **deferred** — and the deferral rationale is corrected: not
"B2→B3 staging like attention" but **MoE decode fuses with no neighbor**, so a `Step::Moe` would
be pure vocabulary with zero fusion value `[D-F10]`. `FUSED_TABLE` untouched.

### D5 · Byte-identical, fixed-order combine preserved `[D-F prefix, A2]`

Resolution relocation + ctx threading select the **same kernels in the same order** — output must
be byte-identical to pre-4.1. The fixed-order MoE combine is untouched (reorder = ULP drift =
attractor). Unlike the discarded fork (i), this is **not** a trivial no-op (resolution call moved,
ctx path rebuilt), so byte-parity is a **load-bearing** check, not a formality `[A7 withdrawn]`.

---

## Plan

### Commit W0 · Family owns resolution + ctx threading + runtime guard (dispatch crate, GPU-free)

1. `families/moe.rs`: `MoeParams` — drop `res: MoeResolution`, add `dtypes: MoeDtypes` and
   `batch_size: usize`. `MoeFamily::run` forwards `ctx` to `run_moe_decode`.
2. `pipeline/mod.rs`:
   - `run_moe_decode(ctx, gpu, p)` + `run_moe_decode_cpu_fallback(ctx, …)`: compute
     `res = MoeResolution::resolve(&p.dtypes, p.k)`; thread `ctx` to all `gemv.run_auto`; delete
     the 5 internal `DispatchCtx::new`.
   - Runtime `batch_size != 1` guard (D3); `FIXME(Step 8)` at each hardcoded `1` (D-F4).
   - `execute_pipeline` (`:49`) + `dispatch_fused` (`:766`) Moe arms forward `ctx` (add a `ctx`
     param to `dispatch_fused`; fix its callers).
3. Tests (GPU-free): `MoeResolution::resolve` unit cells (k=8 indexable → `use_gpu_topk`; k≠8 →
   fallback; non-indexable routed dtype → fallback) `[D-F9 fix: test the resolved verdict, not a
   debug_assert]`; a unit asserting `run_moe_decode` returns `Err(UnsupportedVariant)` for
   `batch_size=2` (real test now the guard is runtime) `[D-F9]`; existing MoE coverage row stays.

**Verify:** `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests`; `cargo check --workspace
--all-targets` (the `MoeParams.res→dtypes` change breaks the qwen35 builder — W1 fixes it; W0+W1
land together if the workspace must stay green between commits).

### Commit W1 · qwen35 decode site → family-owned dispatch

1. `qwen35.rs` `moe_ffn_decode_impl`: delete the `MoeResolution::resolve` call (`:4600`); in the
   `MoeParams` literal pass `dtypes: moe_dtypes` (not `res`) + `batch_size: 1`.
2. Build **one** `DispatchCtx::new(gpu)` at the MoE site `[A3,D-F3 resolution]` and call
   `hipfire_runtime::llama::moe_family().run(&ctx, gpu, &moe_params)`.
3. Grep audit: zero `pipeline::run_moe_decode(` and zero `MoeResolution::resolve(` in `qwen35.rs`
   — the model neither calls the free executor nor resolves.

**Verify (on-GPU, linchpin — Phase 0.6, gfx1100 + gfx1201) `[A5,A7,D-F1,D-F6,G-F6,G-F7]`:**
- **Byte-identical committed-token IDs vs the pre-4.1 tip** (`HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0)
  on A3B (`qwen3.6-35b-a3b.mq4`, k=8) — load-bearing (resolution moved + ctx rebuilt). Prompt md5
  + binary md5 recorded. Same-binary parity via git-checkout toggle (no `HIPFIRE_DISPATCH` selector
  — that's a program gap, not 4.1's `[G-F5]`).
- **k≠8 / non-indexable fixture** byte-parity to exercise the **CPU-top-K fallback** the change
  re-plumbs (A3B alone is k=8 only) `[A5,G-F6]`.
- `coherence-gate.sh **--full**` (A3B cells are `--full`-gated) `[D-F6]`.
- **A3B MoE DFlash** pinned-fixture run (the dflash gate script has no MoE model `[D-F1]`):
  acquire the missing draft `qwen36-35b-a3b-dflash-mq4.hfq` (md5 `8254bbe1`) `[G-F7]`, run the
  AGENTS.md pinned fixture, record prompt+binary md5 + τ + acceptance. If the draft can't be
  obtained, **document the gap and gate on the AR `--full` A3B cells** — do not silently skip.
- `probe_commits.sh <pre-4.1> HEAD` ±1–3% on gfx1100 **and** gfx1201 (ctx-threading should *reduce*
  ctx-construction cost; confirm no regression).

### Commit W2 · Verification sweep + dev-log

- [ ] Grep audit (W1) green; runtime guard never fires in a real A3B run.
- [ ] CPU-fallback path covered by a real on-GPU fixture (k≠8) — closes the long-standing test gap
      `[G-F6,D-F9]`.
- [ ] `findings/dispatch_4.1_dev_log.md`: every fixture (model + k + prompt md5 + binary md5 +
      gfx1100 & gfx1201 numbers + A3B DFlash τ/accept or documented gap).
- [ ] Out-of-scope untouched: grouped prefill (`moe_grouped`, `qwen35.rs:7280+`), ds4 MoE,
      multi-GPU MoE.

---

## Risks

1. **Cross-lane edit (`moe.rs` + `pipeline/mod.rs`) `[A4]`.** Fork (ii) touches Nick's Ship-4 core.
   Mitigation: explicit takeover (owner line); align with Nick before W0 lands; one person in
   those files.
2. **Resolution relocation changes a path `[D5]`.** Mitigation: byte-parity is load-bearing on W1
   (A3B + k≠8); `MoeResolution::resolve` is already pure and unit-tested in W0.
3. **ctx threading touches `dispatch_fused` signature `[D2]`.** Mitigation: small, compiler-checked;
   verify the Moe arm's reachability; `cargo check --workspace`.
4. **CPU-fallback only validated if a k≠8 fixture exists `[A5,G-F6]`.** Mitigation: W1 names a
   non-indexable fixture; if none ships, build a minimal one or document the residual.
5. **A3B DFlash draft model missing `[G-F7,D-F1]`.** Mitigation: acquire `8254bbe1`, else document
   the dflash-coverage gap and gate on AR `--full`.
6. **Combine-order drift → attractor `[D5]`.** Mitigation: D5 touches no executor combine logic;
   `coherence-gate.sh --full`; A3B DFlash eyeball.

---

## Out of scope (tracked elsewhere)

| Item | Where |
|---|---|
| **Step 8** — grouped-GEMM MoE prefill: register HFQ4G256/MQ6/Paro/MQ2-Lloyd grouped keys; **plumb `p.batch_size` through the `run_moe_decode` hardcoded `1` literals** (the `FIXME(Step 8)`s) `[D-F4]` | Step 8 |
| `Step::Moe` in `execute_steps` (no fusion value) `[D-F10]` | deferred / Ship 6 vocab |
| ds4 MoE decode/prefill | done (PR #428) |
| `HIPFIRE_DISPATCH_OLD/_NEW` same-binary selector `[G-F5]` | program-level (Phase 0.6), not 4.1 |
| Multi-GPU MoE decode | later |
| `routed_experts` per-call `Vec` alloc `[D-F7]` | pre-existing perf cleanup |

---

## Review adjudication

Dispositions of all findings across the three plan reviews (claude/gemini/ds4). Re-grounded on the
tip.

| Finding (source) | Sev | Verdict | Disposition |
|---|---|---|---|
| Swap is cosmetic; doesn't centralize (claude A1) | High | **VALIDATED** | **Fork (ii) chosen** — D1 moves resolution into the family. |
| `batch_size` guards an impossible case / inert today (claude A2, ds4 F4) | High/Med | **VALIDATED** | D3 keeps it as a **documented inert** Step-8 plumbing guard + FIXME at literals. |
| D3a ctx unavailable — block-scoped (claude A3, gemini F1, ds4 F3) | Med | **VALIDATED** | D2 dissolves it — family builds/receives one ctx; call site builds it once. |
| Ownership / lane (claude A4) | Med | **VALIDATED** | Owner line: Kevin takes over from Nick, aligns. Risk #1. |
| Fixture under-covers CPU fallback (claude A5, gemini F6) | Med | **VALIDATED** | W1 adds a k≠8 fixture; W2 closes the test gap. |
| dflash gate misses A3B MoE (claude A6, ds4 F1) | Med/Crit | **VALIDATED (claude A6 softened)** | A3B DFlash exists; W1 runs the pinned fixture (acquire draft) or documents the gap. |
| Verification disproportionate (claude A7) | Med | **WITHDRAWN for fork (ii)** | Resolution+ctx move → byte-parity is load-bearing. |
| 3.3 dependency overstated (claude A8) | Low | **MOOT** | 3.3 has shipped; dependency satisfied. |
| Over-staging / vague W0 test (claude A9, A10, ds4 F9) | Low | **VALIDATED** | W0/W1 may co-land; W0 tests the resolved verdict + the runtime `Err`, not a `debug_assert`. |
| Signature facade; forward ctx to GEMVs (gemini F2) | High | **VALIDATED** | D2 — core of the work. |
| CPU-fallback ctx cliff (gemini F3) | Med | **VALIDATED (scoped to k≠8)** | D2 deletes the per-expert ctx. |
| `debug_assert` stripped in release (gemini F4, ds4 F2/F5) | Med/Crit | **VALIDATED** | D3 — runtime guard matching bias-aware. |
| `HIPFIRE_DISPATCH_OLD/_NEW` missing (gemini F5) | Med | **fact VALID / claim REJECTED** | Program-level Phase-0.6 gap, absent since Ship 1; git-checkout parity used instead. Out of scope. |
| CPU fallback no GPU test (gemini F6) | Med | **VALIDATED** | W1/W2 k≠8 fixture. |
| Env: A3B draft model missing (gemini F7) | Low | **VALIDATED** | W1 acquires `8254bbe1` or documents. |
| 6 hardcoded `1` literals (ds4 F4) | High | **VALIDATED** | `FIXME(Step 8)` in W0; Step 8 plumbs `p.batch_size`. |
| `coherence-gate.sh` needs `--full` (ds4 F6) | Med | **VALIDATED** | W1 uses `--full`. |
| `routed_experts` per-call alloc (ds4 F7) | Med | **VALIDATED (pre-existing)** | Out-of-scope cleanup. |
| ctx reconstruction has internal precedent (~20%) (ds4 F8) | Med | **VALIDATED** | Moot under D2 (all collapse to one). |
| `Step::Moe` analogy imperfect — no fusion (ds4 F10) | Info | **VALIDATED** | D4 deferral rationale corrected. |
| `run` ignores `_ctx`; passing zero-cost (ds4 F11) | Info | **VALIDATED** | D2 makes ctx used. |
| `TileImpl` has no `Moe` variant (ds4 F12) | Info | **ACK** | MoE uses `batch_size`/resolution, not `TileImpl`. |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | — | Plan drafted (fork i — call-site swap). | — |
| 2026-06-06 | — | **Rewritten to fork (ii)** after folding 3 plan reviews. Resolution moves model→family (D1); one `ctx` threaded end-to-end, 6 `DispatchCtx::new` deleted (D2); runtime batch guard matching bias-aware + Step-8 FIXMEs (D3). Ownership taken over from Nick (align). 3.3 landed → dependency satisfied; A7 withdrawn (byte-parity now load-bearing). Verification: A3B byte-parity + k≠8 CPU-fallback fixture + `coherence-gate.sh --full` + A3B DFlash pinned fixture (acquire `8254bbe1`) + gfx1100/gfx1201 probe. | — |
