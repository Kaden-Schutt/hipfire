# Ship 3.2 — Attention prefill + adaptive (Phase B)

**Branch:** `integration/dispatch-unification`
**Tracking:** #397 (Ship 3 · Attention + KV cache unification, Phase B)
**Owner:** Kevin / unverbraucht (Ship 3 lane)
**Depends on:** **Ship 3.1 (Phase A) landed** — see the **3.1 API contract** section below for
the exact consumed surface (pin + reconcile against the real tip before starting; 3.1 is
landing in parallel under us).
**Phase 0 contracts still in force:** 0.1 (KV-tier families live-resolve per token — no
init-cache), 0.3 (`KvTierPlan` derived once per attention step, paired write-then-attend,
tier-match `debug_assert`), 0.6 (same-binary byte-parity oracle, RDNA3 **and** RDNA4 probe).

**Goal:** the **prefill** attention/KV path (per-token fallback + batched/masked + tree-verify)
and the **adaptive** mid-sequence precision path both go through `AttentionFamily`. After
3.2, a new KV quant tier is a registry entry + kernel file across decode **and** prefill; the
inline batched dispatch trees in `forward_prefill_chunk` are gone; adaptive downshift and
boundary-layer Q8 forks resolve correctly via the live per-step `KvTierPlan` derive.

> This plan folds the consolidated adversarial review
> (`findings/dispatch_3.2_plan_rev_claude.md` — my self-review + the gemini/glm5 externals,
> F1–F19). Finding tags `[Fn]` mark where each is addressed.

---

## Prerequisites (before C0)

### P-0 · 3.1 API contract — pin and reconcile `[F1, glm5 F8]`

3.2 hard-depends on 3.1's surface, which is **being implemented in parallel**. Before C0,
**diff the real 3.1 tip against this contract and reconcile any drift** (treat the first 3.2
commit as a rebase-onto-actual-3.1, not greenfield). Consumed surface 3.2 assumes:

| Symbol | Assumed shape | If 3.1 landed it differently |
|---|---|---|
| `AttentionFamily::run_attention(ctx, gpu, plan: &KvTierPlan, io: &AttnParams)` | paired write-then-attend | adapt the C-commit call sites |
| `KvTierPlan { write_key, attend_key, v_mode_bits, uses_givens }` | struct, derived per step | extend (don't re-shape) |
| `KvTierInputs` (GPU-free) + runtime-side `kv_tier_inputs(kv_cache, …)` conversion | scalars only, no dep cycle | keep the conversion runtime-side |
| `Step::Attend { plan, io }` + `PipelineOp::Attend` + `launch_op` arm | **3.1 B3 must be landed** | if B3 deferred, C0 adds it |
| `AttnParams` cleaned (no `flash_mode`/`capture_mode`/`kv_dim`/`kind`) | base for D4 additions | if `kind` kept, reconcile D4 |
| dispatch-arm-completeness test + `attention_family()` accessor | extend in C0 | — |

**`Step::Attend`/`PipelineOp::Attend` are a hard precondition** — C0 *extends* `Step::Attend`
with batch fields; it does not create it. If 3.1 B3 hasn't landed, C0 must land it first.

### P-1 · Cherry-pick the #382 no-LDS-cap Q8 kernel `[F3, F3.1]`

The batched Q8 attend path is **not** a single kernel today: at 5 sites
(`qwen35.rs:6107, 9437, 9576, 11063, 11202`) it forks on `max_ctx_len > LDS_CTX_LIMIT (15000)`
into a per-token `attention_flash_q8_0` fallback (and **hard-errors** for tree-verify >15000).
**We have already solved this** — PR #382 commit **`abd9524`** (*"no-LDS-cap batched-masked Q8
flash attention for long ctx"*) ships a tiled kernel (`LDS = tile_size` only) that handles
arbitrary ctx.

**Cherry-pick only the additive `rdna-compute` asset** (parent `ac02a1f6`, predates the PP/MTP
stack — zero PP/MTP files):
- `kernels/src/attention_flash_q8_0_tile_batched.hip` (new)
- `rdna-compute/src/dispatch.rs` wrapper `attention_flash_q8_0_batched_masked` (appended)
- `rdna-compute/src/kernels.rs` `include_str!` line
- (optional) `q8_batched_attn_microbench.rs` for the >15k C2 verify

**Do NOT take** `abd9524`'s `qwen35.rs`/`llama.rs` hunks — they rewire the *old inline*
LDS-forked dispatch that C2 deletes wholesale. Instead, **C0 registers `AttnQ8_0KvBatchedMasked
→ gpu.attention_flash_q8_0_batched_masked`** — the new arm is one clean kernel call with **no
`LDS_CTX_LIMIT` branch**. (This is why Gemini F1's "port the per-position loop into the
dispatch arm" fix is **rejected** — superseded here; no context-shaped fork enters dispatch.)
The 3 sibling commits (`594738` falsified/off, `7ced6ca`+`8c48c08` gfx906 dp4a, gated off) are
**not** needed for 3.2.

---

## Grounded starting state (verified on the branch tip)

> Line numbers are as-of the grounding commit; **function names are the stable anchors**
> (they will drift as 3.1 + Ship 4 land) `[glm5 F13]`.

### Prefill dispatch sites in qwen35 (`fn`-anchored)

| Site | `fn` / lines | Shape | Kernels |
|---|---|---|---|
| `run_fa_layer_body` inline tree | `:11444` (tree `:11638–11860`) | **single-token** | the *same* `_fused` write + single-token flash kernels as decode (`kv_cache_write_fwht4_fused`, `attention_flash_fwht4`, …, `attention_q8_0_kv`, `attention_f32`) |
| `forward_prefill_chunk` — **dense FA** | `:9318–9630` | **batched** | `kv_cache_write_*_batched` + `attention_flash_*_batched(_masked)` + `attention_q8_0_kv_batched_masked` |
| `forward_prefill_chunk` — **FA-MoE** | `:10954–11253` | **batched** | near-identical second copy of the dense block |
| per-token fallback (non-batchable wts) | `:10015–10059` | gather→`run_fa_layer_body`→scatter | single-token path |

**Two structural classes:** (1) **single-token** (`run_fa_layer_body`) — reuses 3.1's decode
kernels verbatim → trivial migration; (2) **batched/masked** — distinct kernels with
`positions[n]`, `n` (batch), `max_seq` (= `physical_cap`) **and** `max_ctx_len` (= start_pos+n),
plus the tree-verify triple (`tree_bias[n×n]`, `block_start`, `block_cols`). The
`_batched_masked` kernels serve **both** plain causal (`tree_bias=None`) **and** spec-decode
tree-verify (`tree_bias=Some`) — so the batched migration **touches the DFlash tree path** and
must pass the dflash gate.

**Batched kernel surface** (`rdna-compute/src/attention.rs`): asym4/3 + fwht4/3 have
`_batched_masked`; **asym2/fwht2 have only `_batched` (no `_masked`)** — a real gap: 2-bit
tiers cannot tree-verify today. Q8: the `>15000` LDS fork → **replaced by the P-1 kernel**.
Batched KV writes (`kv_cache_write_{asym4,fwht4,asym3,fwht3,asym2,fwht2}_batched`) are `_fused`
(K+V one call); **`kv_cache_write_q8_0_batched` is the odd one out — called twice (K, then V)**
`[glm5 F5/F15]`.

### Registry / shape-gate readiness

- `ShapeInfo { batch_size, head_dim, m }` and `ShapePredicate::BatchGt(usize)` **exist**;
  `resolve()` evaluates `variant.shape_gate` **only when `shape` is `Some`** (`tables/mod.rs:
  41–52`: *"shape is None → bypass shape gating"*). **All 18 attention keys are single-token.**
- **6436bd1 lesson:** `GemvFamily::run` hardcodes `ShapeInfo { batch_size: 1, … }`
  (`gemv.rs:264`). For 3.2 the guard only works if `run_attention` actually **passes a real
  `ShapeInfo`** — see D2 `[F2]`.

### Adaptive KV machinery (`kv_adaptive.rs`, `daemon.rs`, `llama.rs`)

- `KvAdaptive::maybe_downshift` (`kv_adaptive.rs:178`) mutates `KvCache.v_mode` (via
  `transcode_v_step`) and `KvCache.quant_asym{4,3,2}` (via `transcode_k_step`) — **global, not
  per-layer**. `quant_fwht`/`layer_is_boundary` are **static-at-load**. Both transcodes
  **re-quantize the whole existing cache `[0, seq_pos)`** to the new tier (uniform-tier — the
  load-bearing invariant for one-key-per-step, see C4 `[F6]`).
- Called from `daemon.rs` **between** forward passes (chunks `:8743`, post-prefill `:8813`, per
  decode token `:9178`) — never mid-attention. So within one step the tier is stable → Phase
  0.3 derive-once already guarantees write-tier == attend-tier; adaptivity is **cross-step**.
- Because 3.1 live-derives `KvTierInputs` from `kv_cache.{quant_asym*, v_mode_bits()}` every
  step (Phase 0.1, no init-cache), **decode adaptivity already works** when 3.1 lands. 3.2's
  adaptive work is (a) make the **batched** path re-derive **per layer per chunk** `[F18]`, and
  (b) a long-context test that trips downshifts and verifies the kernel each step.
- `transcode_*_step` is a `KvCache` op in `hipfire-runtime` — stays there; dispatch only sees
  the resulting scalar mode via the runtime-side `kv_tier_inputs()` conversion. No new coupling.

### Boundary-layer Q8 (`llama.rs:4242–4257`)

`boundary_layers`, `layer_is_boundary: Vec<bool>`, `is_boundary(kv_ordinal)` **exist but are a
placeholder** — `layer_is_boundary` is `vec![]` in every load path, **read by nothing**.
"Boundary-layer Q8 forks" is **new functionality** (consumer wiring in C4); the **producer**
(populating the flags at load) is non-trivial — a Q8-pinned layer needs a **Q8-sized KV buffer**
while others are floor-sized, touching per-layer allocation + the adaptive threshold model
`[F7]`. Producer is a separate prerequisite for the feature being "live".

### PFlash (`hipfire-arch-qwen35/src/pflash.rs`)

Drafter prefill calls `forward_prefill_batch` (Q8-locked: `assert!(kv.quant_q8, …)`); its
attention rides the C2 batched path for free. Its **scoring** uses `pflash_score_q8_kv_blocks`
— a K-only importance scorer, **not write+attend** → **out of `AttentionFamily` scope**
(possible future `ScoringFamily`; left as a direct call).

---

## Design decisions (this ship)

### D1 · Batched keys are distinct keys `[F4, F9]`

Registry is key-dispatched and Phase 0.5 says "one `KernelKey` per (quant tier × variant)";
batched is a variant axis → **distinct keys**, 14 new:

- **Attend (7):** `AttnFlashAsym4BatchedMasked`, `AttnFlashAsym4FwhtBatchedMasked`,
  `AttnFlashAsym3BatchedMasked`, `AttnFlashAsym3FwhtBatchedMasked`, `AttnFlashAsym2Batched`,
  `AttnFlashAsym2FwhtBatched` (⚠ no-masked — 2-bit tree gap), `AttnQ8_0KvBatchedMasked`
  (→ the **P-1** no-LDS-cap kernel, not the old `attention_q8_0_kv_batched_masked`).
- **KV write (7):** `KvWriteAsym4Batched`, `KvWriteAsym4FwhtBatched`, `KvWriteAsym3Batched`,
  `KvWriteAsym3FwhtBatched`, `KvWriteAsym2Batched`, `KvWriteAsym2FwhtBatched`,
  `KvWriteQ8_0Batched`.

**F32-batched is NOT added** — qwen35 F32 prefill uses the per-token fallback; a `// TODO(3.3):
F32-batched keys for models with F32 KV + batchable weights` note marks the gap so a future
author doesn't re-discover it `[F9, glm5 F3]`. (`attention_causal_batched` exists but is unused
on qwen35 — used by qwen2/llama; don't register dead here.)

`is_tree` (D5) affects key selection **only** for the 2-bit / q8-long-ctx branches — for 4/3-bit
the **same** `*BatchedMasked` key serves both causal (`tree_bias=None`) and tree
(`tree_bias=Some`); the non-masked `_batched` 4/3-bit kernels are intentionally unregistered
`[F4]`.

### D2 · Shape threading + the real batch guard `[F2, F14, glm5 F1, Gemini F4]`

**The `BatchGt(1)` shape-gate is inert under 3.1's `resolve(…, None)`.** To make it fire (and
to honor the 6436bd1 lesson), `run_attention` must pass a real `ShapeInfo`:

```rust
let shape = ShapeInfo { batch_size: plan.batch_size, head_dim: io.head_dim, m: 0 };
let w = self.resolve(plan.write_key,  ctx, Some(&shape))?.key;
…
```

This requires `KvTierPlan` to carry **`batch_size`** (D5) `[F14]`. Then:
- batched keys register `shape_gate: Some(BatchGt(1))` → a batched key resolved at
  `batch_size==1` → `MissingImpl` (not a silent single-token kernel on a batch).
- single-token keys register `shape_gate: Some(BatchEq(1))` (new `ShapePredicate::BatchEq`
  variant) → a single-token key resolved at `batch_size>1` → `MissingImpl` too, symmetric
  defense `[Gemini F4]`.

The **primary** correctness comes from `KvTierPlan::derive` picking the right key from
`batch_size`; the shape-gate is the resolve-time backstop. (If we choose **not** to thread
`ShapeInfo`, the gate must be deleted and replaced by a `debug_assert` in `derive` — but
threading is preferred, it makes the guard real and is reusable for batched GEMM later.)

### D3 · `dispatch_attention` splits into `dispatch_kv_write` + `dispatch_attend` `[F11, F15, glm5 F11]`

Adding 14 batched arms pushes the single function past maintainability. **Split** into
`dispatch_kv_write(gpu, key, io)` + `dispatch_attend(gpu, key, io)`, each exhaustive per-family.
`run_attention` calls write then attend. **The completeness test + its key constant split too**:
`DISPATCHED_ATTENTION_KEYS` → `DISPATCHED_KV_WRITE_KEYS` + `DISPATCHED_ATTEND_KEYS`, one test
each. The `KvWriteQ8_0Batched` arm calls `kv_cache_write_q8_0_batched` **twice (K, then V)**,
mirroring the single-token `KvWriteQ8_0` arm (asym/fwht batched writes are `_fused`) `[F15]`.

### D4 · `AttnParams` batch surface `[F8, F12, F16, F10, glm5 F2]`

`AttnParams` (cleaned in 3.1) gains:

```rust
pub batch_size: usize,                 // REQUIRED; decode/per-token = 1
pub positions: Option<&'a GpuTensor>,  // [n] i32 for batched; None → use pos_buf (single)
pub max_ctx_len: usize,                // batched attend loop bound (start_pos + n)
pub tree_bias: Option<&'a GpuTensor>,  // [n×n]; Some → tree-verify, None → causal
pub block_start: usize,                // tree window start (0 for plain causal)
pub block_cols: usize,                 // tree window cols (0 for plain causal)
```

- **Position type coexistence `[F8, glm5 F2]`:** `pos_buf: &DeviceBuffer` (single-token) and
  `positions: &GpuTensor` (batched) are **different types** — keep both (**option 1**, lower
  risk: the 18 single arms keep reading `pos_buf`). Invariant: `positions.is_some() ⟺
  batch_size > 1`; `pos_buf` ignored when batched. `debug_assert` it; document on the fields.
- **`v_mode_bits` single source `[F12, Gemini F2]`:** **remove `v_mode_bits` from `AttnParams`**
  — `KvTierPlan` is the single source; `dispatch_*` reads `plan.v_mode_bits`. (Eliminates the
  divergent-value drift class.)
- **`max_seq` `[F16, glm5 F6]`:** batched kernels take `max_seq` **and** `max_ctx_len`.
  `AttnParams.physical_cap` already carries `max_seq`; for batched qwen35 they're equal — pass
  `physical_cap` as `max_seq` with a `debug_assert(physical_cap == max_seq)` in the batched
  arms. No new field beyond `max_ctx_len`.
- No parallel `AttnBatchParams` struct (the 3.1 anti-duplication rule).

### D5 · `KvTierPlan::derive` selects batched / masked / boundary keys `[F13, F14, F17]`

`KvTierPlan` gains **`batch_size`** (for D2's `ShapeInfo`). `KvTierInputs` (GPU-free) gains
`batch_size`, `is_tree`, `is_boundary`. `derive` returns **`Result<KvTierPlan,
DispatchError>`** (the typed error below) — **this changes the 3.1 decode call site**, which
must be updated to `?`/`map_err` `[F13, Gemini F3]`. Lattice (pure, unit-tested):

1. `is_boundary` → force `KvWriteQ8_0[Batched]` + `AttnFlashQ8_0`/`AttnQ8_0Kv[BatchedMasked]`
   (boundary layers pin Q8 regardless of the global downshifted tier).
2. else tier from live `quant_asym{4,3,2}`/`quant_q8`/`v_mode_bits` (3.1 logic).
3. `batch_size > 1` → the `*Batched(Masked)` key for that tier; else the single-token key.
4. tier-match `debug_assert` extended: write/attend agree on tier **and** batched-ness.
5. ⚠ `batch_size > 1 && is_tree && tier ∈ {asym2, fwht2}` → **no masked kernel** → `Err(
   UnsupportedTreeTier { tier, batch_size })`. **Caller behavior `[F17, glm5 F9]`:** the prefill
   path maps it to a `HipError` (message carries tier + batch + model config) — matching today's
   `assert!` semantics but surfaced earlier; **no silent fallback to a wrong kernel**.

---

## Plan

> **Commit structure note `[F19, glm5 structural]`:** C0 merges the former "keys+dispatch" and
> "derive" commits into **one** family-layer foundation — registering batched keys without the
> `derive` that selects them would leave keys that pass the completeness test but are
> unreachable (confusing for bisect). 3.2-core = **P-1 + C0–C3 + C6**; 3.2b = **C4 + C5**.

### Commit C0 · Family-layer foundation (keys + dispatch split + `AttnParams` + `KvTierPlan` derive + shape threading) — GPU-free

1. **Keys** `types.rs`: append the 14 batched `KernelKey`s + `ShapePredicate::BatchEq`
   (append-only — coordinate the enum block with Nick) `[D1, D2]`.
2. **Table** `attention_table.rs`: register the 14 (`Always` arch — verify cross-arch
   precompiled) with `shape_gate: Some(BatchGt(1))`; add `BatchEq(1)` gates to the 18
   single-token keys `[D2]`.
3. **Dispatch split** `families/attention.rs`: `dispatch_attention` → `dispatch_kv_write` +
   `dispatch_attend` `[D3]`; add the 14 batched arms (`AttnQ8_0KvBatchedMasked →
   gpu.attention_flash_q8_0_batched_masked` from **P-1**; `KvWriteQ8_0Batched` double-call).
4. **`run_attention`** threads `ShapeInfo` into both `resolve()` calls `[D2/F2]`.
5. **`AttnParams`** batch surface `[D4]`: add batch fields; **remove `v_mode_bits`**; keep
   `pos_buf` + add `positions`; `physical_cap`-as-`max_seq` assert.
6. **`KvTierPlan` + `derive`** `[D5]`: add `batch_size`; `KvTierInputs` gains
   `batch_size`/`is_tree`/`is_boundary`; lattice + `UnsupportedTreeTier`; runtime-side
   `kv_tier_inputs(…)` extended (passes `is_boundary = kv_cache.is_boundary(kv_ordinal)`).
7. **Tests** (GPU-free): split completeness test (`DISPATCHED_KV_WRITE_KEYS` /
   `DISPATCHED_ATTEND_KEYS`); coverage gate batched rows incl. RDNA4; `BatchGt(1)`/`BatchEq(1)`
   resolve-guard tests; every (tier × batched × tree × boundary) `derive` cell asserts the
   `(write_key, attend_key)` pair; boundary→Q8; 2-bit-tree→`Err`.

**Verify:** `cargo test -p hipfire-dispatch -p hipfire-dispatch-tests` green; `cargo check
--workspace --all-targets` (incl. the updated 3.1 decode call site for the `Result` `[F13]`).

### Commit C1 · `run_fa_layer_body` (single-token prefill fallback) → `run_attention`

Lowest-risk cut: `run_fa_layer_body`'s inline tree (`:11638–11860`) uses the **exact decode
kernels** 3.1 migrated. Replace its body with the same `Step::Attend` / `run_attention` call
(`batch_size=1`, `positions=None`). Delete the inline tree.

**Verify:** byte-identical token IDs vs master (`HIPFIRE_EMIT_TOKEN_IDS=1`, temp 0.0, committed
prompt + md5) on gfx1100 + gfx1201 with a **non-batchable-weights** model (forces the per-token
fallback); `coherence-gate.sh`.

### Commit C2 · Batched prefill — dense FA block → `Step::Attend`

Migrate the `forward_prefill_chunk` dense block (`:9318–9630`). **Scope `[F5, F7, glm5 F7]`:
only the inline KV-write+attend dispatch tree is replaced by a `run_attention` call** — the
layer loop, weight access, FFN dispatch, and hidden_rb extraction stay. `plan` is **re-derived
per layer** (`KvTierInputs { batch_size: n, is_tree: tree_bias.is_some(), is_boundary:
is_boundary(kv_ordinal), … }`); `io` carries `positions`, `max_ctx_len`, `tree_bias`,
`block_start`, `block_cols`. The q8 arm now goes through the **P-1 no-LDS-cap kernel** — the
`LDS_CTX_LIMIT` fork is **deleted, not ported** `[F3]`.

**Verify (linchpin):** byte-identical vs master on gfx1100 + gfx1201 across the prefill tier
matrix (fwht4 / fwht3 / asym4 / asym3 / **q8 incl. >15k ctx via P-1** / F32-fallback) with
committed long prompts (prompt md5 + binary md5 recorded). **`coherence-gate-dflash.sh`
REQUIRED** (the `*_batched_masked` path is the tree-verify attend). `probe_commits.sh master
HEAD` ±1–3% (prefill tok/s — perf-sensitive). **2-bit tree:** confirm no gated fixture silently
hits `UnsupportedTreeTier`.

### Commit C3 · Batched prefill — FA-MoE block → `Step::Attend`

Migrate the second block (`:10954–11253`). **Re-ground at-tip first `[F5]`:** Nick's Ship-4
MoE-prefill work edits this same region and may relocate/refactor the attention block — migrate
*whatever the FA-MoE attention sub-block looks like post-Ship-4-prefill*, and verify it's still
near-identical to C2's dense block before folding both onto one `run_attention` path (the
fold-win assumes they haven't diverged). MoE FFN half is Nick's lane — one person in the file at
a time (boundary contract).

**Verify:** same matrix as C2 on an A3B MoE fixture; `coherence-gate.sh --full` +
`coherence-gate-dflash.sh`.

### Commit C4 · Adaptive re-resolution + boundary-layer Q8 forks (correctness-critical) — *3.2b*

**Adaptive (verification — the live-derive does the work):**
- **Uniform-tier precondition `[F6]`:** verify `transcode_v_step`/`transcode_k_step` rewrite the
  **full `[0, seq_pos)` cache** to the new tier (so one attend key per step is correct). Name it
  as the load-bearing invariant; if it ever stops holding, single-key-per-step breaks.
- **Per-layer per-chunk re-derive `[F18, glm5 F10]`:** the migrated `forward_prefill_chunk` calls
  `KvTierPlan::derive` **per layer** (not once at chunk entry), so all layers of chunk N+1 see
  the post-`maybe_downshift` tier. State as the critical path.
- **Long-context test (>8K)** tripping ≥2 thresholds (q8→lloyd4→lloyd3 / fwht4→fwht3): assert
  (a) tier changes at expected positions, (b) write-tier == attend-tier every step across the
  boundary (`debug_assert` never fires), (c) byte-parity vs master.

**Boundary-layer Q8 (consumer wiring; producer is a prerequisite) `[F7]`:**
- Wire `is_boundary(kv_ordinal)` into `kv_tier_inputs()` (C0) so boundary layers pin Q8.
- **Ship the consumer INERT** — `layer_is_boundary` is `vec![]`, so `is_boundary` is always
  false → byte-identical, zero behavior change. **Do not claim the feature live** until a
  separate producer (per-layer buffer sizing + adaptive-threshold interaction — non-trivial)
  + a boundary-Q8 byte test land.

**Verify:** long-context adaptive byte-parity gfx1100 + gfx1201; `coherence-gate.sh` +
`coherence-gate-dflash.sh`; boundary wiring proven inert (byte-identical, empty boundary set).

### Commit C5 · PFlash drafter prefill — confirm free ride; scope out scoring — *3.2b*

- Confirm the drafter's `forward_prefill_batch` attention rides the C2 batched path (Q8 batched
  keys) with **no pflash-specific dispatch change**.
- **Scope `pflash_score_*` OUT of `AttentionFamily`** (K-only scorers, not write+attend) —
  direct calls; note a possible future `ScoringFamily`.

**Verify:** `dflash_spec_demo --pflash` coherent on gfx1100; drafter prefill byte-parity vs
master.

### Commit C6 · Verification sweep + cleanup + grep audit

- [ ] Coverage golden incl. RDNA4 + non-WMMA rows for all 32 attention keys (18 single + 14
      batched), both dispatch functions.
- [ ] **Grep audit (goal #1):** zero `gpu.kv_cache_write_*` / `gpu.attention_flash_*` /
      `gpu.attention_q8_0_kv*` / `gpu.attention_f32` calls **and zero `LDS_CTX_LIMIT`
      references** remain in the qwen35 **prefill** paths — all through `run_attention`.
- [ ] Dense + MoE prefill blocks folded to **one** code path (duplicate tree deleted).
- [ ] All `debug_assert`s (tier-match, position-coexistence, `max_seq`==`physical_cap`) never
      fire in a real run.
- [ ] Dev-log every fixture (model + KV mode + batch + prompt md5 + binary md5).
- [ ] Multi-GPU prefill (`forward_scratch_layers_multi`) + PFlash scoring untouched.

---

## Risks

1. **Batched migration touches the DFlash tree-verify path.** `coherence-gate-dflash.sh`
   mandatory on C2/C3.
2. **shape-gate inert if `ShapeInfo` not threaded `[F2]`.** Mitigation: D2 threads it +
   `KvTierPlan.batch_size`; `BatchGt(1)`/`BatchEq(1)` resolve-guard tests. If we skip threading,
   delete the gate claim and guard in `derive` instead — don't ship a decorative gate.
3. **Position type coexistence `[F8]`** (`&DeviceBuffer` vs `&GpuTensor`). Mitigation: option 1
   (keep both) + invariant `debug_assert`.
4. **2-bit tree-verify gap.** Mitigation: typed `UnsupportedTreeTier` → `HipError` (surfaced),
   never a wrong kernel; C2 confirms no gated fixture hits it. (Real 2-bit tree kernel = 3.3.)
5. **Adaptive correctness rests on the uniform-tier transcode invariant `[F6]`.** Mitigation:
   verify + name it in C4; the per-step tier-match `debug_assert` + the long-context test.
6. **Adaptive re-derive must be per-layer per-chunk `[F18]`**, not at chunk entry. Mitigation:
   C4 states it as the critical path; the long-context test crosses real chunk boundaries.
7. **Boundary-layer Q8 producer is non-trivial `[F7]`** (per-layer buffer sizing + adaptive
   thresholds), not just a flag array. Mitigation: ship consumer inert; producer is a separate
   scoped follow-up; don't claim the feature live.
8. **MoE-prefill region moves under Nick `[F5]`.** Mitigation: re-ground C3 at-tip; sequence
   after his edits; one person in the file at a time.
9. **`derive` → `Result` breaks the 3.1 decode call site `[F13]`.** Mitigation: C0 updates it in
   the same commit; `cargo check --workspace` is the gate.
10. **Prefill is perf-sensitive.** `probe_commits.sh` prefill-tok/s ±1–3% is a hard gate on
    C2/C3, not just byte-parity.

---

## Out of scope (tracked elsewhere)

| Item | Ship / Phase |
|---|---|
| **DFlash WMMA tile attention variants** (~18, `shape_gate`), qwen2 dots-ocr pattern | 3.3 (Phase C) |
| **2-bit tree-verify kernel** (`attention_flash_asym2_batched_masked`) — new kernel | 3.3 / kernel work |
| **F32-batched keys** (`attention_causal_batched`) for F32-KV + batchable-weights models | 3.3 (TODO marker in `KernelKey`) |
| `pflash_score_*` scoring kernels → possible `ScoringFamily` | future (not attention) |
| Boundary-layer **producer** (per-layer buffer sizing + threshold interaction) | prerequisite for C4 "live" |
| gfx906 dp4a Q8 attention perf (`7ced6ca`, `8c48c08`) | separate gfx906 perf item |
| llama + qwen2 prefill attention | Ship 3.1b (decode) then their prefill follow-up |
| Multi-GPU prefill attend (`forward_scratch_layers_multi`) | later |
| MoE FFN path | Ship 4 (Nick) |

---

## Dev log

| Date | Commit | What | Result |
|---|---|---|---|
| 2026-06-06 | — | Plan drafted (assumes 3.1 landed). Grounded the prefill/adaptive/PFlash/shape-gate surfaces; decisions D1–D5. | — |
| 2026-06-06 | — | **Folded consolidated review (F1–F19).** Added P-0 (3.1 API contract pin) + P-1 (#382 `abd9524` no-LDS-cap Q8 kernel cherry-pick — additive rdna-compute only; **rejects Gemini F1's loop-in-arm fix**). D2 now threads `ShapeInfo` so the `BatchGt(1)` gate actually fires (+`BatchEq(1)` symmetric guard; +`KvTierPlan.batch_size`). D4: drop `v_mode_bits` from `AttnParams` (single source = plan), keep `pos_buf`+add `positions` (type coexistence), `physical_cap`-as-`max_seq` assert. D5: `derive`→`Result` (updates 3.1 decode site), `is_tree` scoped to 2-bit/q8-longctx, `UnsupportedTreeTier`→`HipError`. **Merged C0+C1** (F19, no transient dead-keys). Added Q8 double-call (F15), per-layer-per-chunk re-derive (F18), uniform-tier transcode invariant (F6), boundary producer non-triviality (F7), dense/MoE fold scope clarification (F7-glm5), completeness-test split (F11). Renumbered C2–C7 → C1–C6. | — |
