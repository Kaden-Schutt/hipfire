# Greenfield engine architecture — the daemon↔model↔decode layer

**Date:** 2026-06-13 (revised after a 3-reviewer adversarial pass)
**Branch:** `feature/paro-transparent-loading`
**Status:** Concept design (no code). Companion to `2026-06-13-archspec-and-dyn-boundary.md`.
**Scope:** How hipfire's model-loading / dispatch / decode layer *should* be structured if rebuilt
from scratch — split into a load-bearing **core** (ship) and three **deferred bets** (separate docs).

---

## 0. Scope & honest framing (read this first)

A 3-reviewer adversarial pass (2026-06-13) on the first draft of this doc reached one verdict:
**the ownership diagnosis is right and load-bearing; the rest was over-scoped** (≈12 new types
bundled for narrative completeness, two of which were falsified by real code). This revision splits
accordingly:

- **CORE (this doc, ships):** the ownership refactor — `Model` (owned, immutable) + `Session`
  (mutable) + `Runtime` **enum** with per-arch state nested. Plus exhaustive teardown and a
  *pure-helper* primitive dedup. **Forward stays statically dispatched** (no dyn on the hot path).
- **BET 1 — spec-decode unification** (mistral.rs-shaped): own design doc, gated on ≥2 arches
  migrated. **BET 2 — `ArchSpec`** authoring surface: companion doc `archspec-and-dyn-boundary.md`.
  **BET 3 — a dyn `ArchInstance` forward boundary**: only if a perf microbench (§9 R3) shows <1%.

What the reviews **falsified** in the first draft, now removed: `Arc<Model>` sharing (the weights
hold a `RefCell<WeightPager>` mutated per-token → `!Sync`; the daemon is single-model anyway; and
`Arc` + GPU-free leaks by construction), a `dyn ArchInstance::forward` boundary (reverses an explicit
measured decision in `hipfire-runtime/src/arch.rs:30-46` + `hipfire-arch-llama/src/arch.rs:10-11`
without measuring), a `dyn SpecTarget` (verify reaches too many
target internals per-op), and a single `Proposer::propose` unifying linear+tree+MTP.

---

## 1. Why greenfield, and the reframe

Three review rounds (`archspec-and-dyn-boundary.md` §2) killed every attempt to *retrofit* a uniform
model-dispatch contract. Root cause: **the ownership model**, not the contract shape. The daemon
conflates *immutable weights* with *per-conversation mutable state*, and stores the speculative-decode
draft as a **sibling** of the target. That forces the ~45-field `LoadedModel` god-struct
(`hipfire-loader/src/lib.rs:243-297`), the dispatch ladders, and the per-spec-call **transient
`ModelSlot` rebuild that re-mmaps the HfqFile every step** (`daemon.rs:3837-3920`, incl. error-path
re-boxing). You cannot bolt a clean contract onto that incrementally.

**The reframe:** fix ownership first; the rest follows. The shape is validated by a shipping Rust peer
(`mistral.rs`, §11) — but the *delivery* is the ownership core, with spec/ArchSpec/dyn deferred.

---

## 2. CORE — the ownership model

Split state by the **two axes `LoadedModel` conflates**, keeping only the one that pays: *mutability*.
(The first draft also split on *sharing* via `Arc`; the reviews falsified that — see §0/§9 R4 — so
`Model` is single-owner.)

```
Engine                                  // process-global, SINGLE model today (daemon.rs:1067: Option<LoadedModel>)
 ├─ gpu:   Gpu                          // &mut threaded into forward/free; owned here
 └─ model: Option<Model>               // OWNED, not Arc. (multi-model later = a registry actor
                                        //   owning Vec<Model> + the Gpu — still single-owner, never Arc)

Model                                   // immutable RESIDENT weights only — honestly read-only
 └─ { weights, config, tokenizer, mmap: Hfq /* read source for the pager */, caps: Capabilities }
   // NO RefCell, NO pager, NO scratch here (the pager is per-session — see below).

Session                                 // ONE conversation; arch-AGNOSTIC mutable state
 ├─ cursor: Cursor                      // seq_pos, conversation_tokens, checkpoints, ngram window
 ├─ evict:  Option<Eviction>
 └─ rt:     Runtime                     // the ONLY arch-dispatch point — a CLOSED ENUM (see §3)

Runtime (enum)                          // per-arch mutable state lives INSIDE each variant
 ├─ Dense(DenseRt   { kv, scratch, pager? })
 └─ Qwen35(Qwen35Rt { kv, dn, scratch,
                      pager:  Option<WeightPager>,   // HOISTED out of Qwen35Weights; plain &mut, no RefCell
                      spec:   Option<Spec>,           // draft+rings+tape NESTED here, not a sibling
                      vision: Option<VisionState> })
//                    ep:     Option<EpRanks>         ← FUTURE (rehome C, §6.1): folds today's sibling
//                                                       EpState/EpArch (lib.rs:431-449) in; gated on THIS core

Spec  └─ { draft: DraftModel, rings: DflashRings, tape: GdnTape }   // ex-DflashState, co-located
```

**Consequences (each verified against real code):**

- **The four-way spec borrow becomes legal.** Today `spec_step_dflash` needs `&mut target` +
  `&mut df.{draft_scratch,hidden_rb,target_snap,gdn_tape}` simultaneously (`daemon.rs:4416`), forcing
  target to be a freshly-built local. As a method on `Qwen35Rt`, those are disjoint `&mut self.field`
  borrows — legal Rust. **The transient `ModelSlot` rebuild deletes entirely** (mmap lives in
  `model.mmap`; target pieces are long-lived `self.kv/dn/scratch`).
- **The pager is per-session mutable state, not a weight.** `Qwen35Weights` currently holds
  `pager: Option<RefCell<WeightPager>>`, `borrow_mut`-mutated during forward to page experts
  (`qwen35.rs:677-695`; `weight_pager.rs:424/537` `ensure_resident`/`evict`). Hoisting it into
  `Qwen35Rt` (plain `&mut`, no `RefCell`) makes `Model.weights` honestly read-only and removes the
  `!Sync` that would block any future threading.
- **Teardown is compiler-enforced** (closes the C2 silent-leak class — today `unload_model` frees
  neither `lfm2moe`/`minimax`/`dots_ocr`, `lib.rs:1166-1204`):
  ```rust
  pub trait GpuOwned { fn free(self, gpu: &mut Gpu); }
  impl GpuOwned for Runtime {                       // EXHAUSTIVE — new variant won't compile until handled
      fn free(self, gpu: &mut Gpu) { match self {
          Runtime::Dense(r)  => r.free(gpu),
          Runtime::Qwen35(r) => r.free(gpu),
      } }
  }
  impl GpuOwned for Qwen35Rt {
      fn free(self, gpu: &mut Gpu) {
          let Qwen35Rt { kv, dn, scratch, pager, spec, vision } = self;   // destructure: every field named
          kv.free(gpu); dn.free(gpu); scratch.free(gpu);
          if let Some(p) = pager  { p.free(gpu); }
          if let Some(s) = spec   { s.free(gpu); }
          if let Some(v) = vision { v.free(gpu); }
      }
  }
  ```
  `Model::free(self, gpu)` consumes the owned model (today's `unload_model(m, gpu)`, relocated) — no
  `Arc::into_inner`, no implicit-`Drop` free, no leak. A debug `Drop for Model` that panics if dropped
  un-freed turns the *silent* leak into a *loud* test failure (single owner ⇒ panic = real bug).

**Non-goals, stated explicitly:** multi-session weight sharing (the daemon is single-model;
paged models can't share a residency map anyway). If multi-model ever lands, weights move behind a
registry **actor that owns them + the Gpu** — single-owner, never `Arc`.

---

## 3. CORE — the dispatch boundary is a `Runtime` enum (forward stays static)

The first draft proposed a `dyn ArchInstance` with `forward`/`sample` on it. **Rejected for v1:**
the code records a *deliberate, measured* decision — the `Architecture` trait carries only the
bring-up triple (`config_from_hfq`/`load_weights`/`new_state`) with the "why forward isn't on the
trait" rationale at `hipfire-runtime/src/arch.rs:30-46`, and `hipfire-arch-llama/src/arch.rs:10-11`
states "forward passes stay direct `llama::*` calls — the hot path doesn't pay dyn dispatch overhead."
Putting `forward`/`sample` behind
`dyn` reverses that, per-token, **unmeasured**, in a repo where a 3% delta is real signal.

So the dispatch point is the **closed `Runtime` enum**:
```rust
impl Session {
    fn step(&mut self, gpu: &mut Gpu) -> StepStatus {
        match &mut self.rt {                        // ONE match per token; arms call concrete forwards
            Runtime::Dense(rt)  => rt.step(gpu, &self.model, &mut self.cursor),
            Runtime::Qwen35(rt) => rt.step(gpu, &self.model, &mut self.cursor),   // static llama::*/qwen35::* calls inside
        }
    }
}
```
This delivers the ownership win + exhaustive teardown with **zero vtable in the token loop**. Adding
an arch is a one-line enum variant + one `REGISTRY` array line (`lib.rs:53-77`, a `const` array, hand-
edited — there is no `inventory`/open-world today, and the single-model daemon doesn't need it). **The
"touches zero shared files / out-of-tree arches" framing from the first draft is retracted** — adding
an arch is a *small, reviewable, one-line-per-shared-site* edit, not zero.

> **BET 3 (deferred, perf-gated):** if open-world / out-of-tree arches ever become a real requirement,
> a `Box<dyn ArchInstance>` forward boundary is the move — but only after a microbench (§9 R3) shows
> the per-token vtable costs <1% on a small model. Until then: enum.

---

## 4. BET 1 (deferred) — speculative decoding, mistral.rs-shaped

This is the subsystem that broke three reviews. It gets its **own design doc + PR**, gated on ≥2
arches migrated behind `Runtime`. Recorded here so the core doesn't accidentally preclude it.

- **The spec target is CONCRETE, monomorphized — not `dyn`.** Verify (`verify_dflash_block_inner`,
  `speculative.rs:2385`) reaches `target.weights.{output, token_embd, embd_format}`,
  `target.config.{vocab_size, dim, num_experts}`, `target.scratch.logits`, `&mut target.kv_cache`,
  `&mut target.dn_state`, and does **per-slot** lm_head/embedding dispatch. A `dyn` trait would need
  ~15 methods and force per-op virtual calls (banned). So: `Speculative<T: TargetModel, P: SpecForm>`
  with `target`/proposer monomorphized; a `Speculative` is just one more `Runtime` enum arm. The
  **target-concrete (not `dyn`)** decision matches `archspec-and-dyn-boundary.md` §2; the one place
  the docs differ is the *top-level* boundary — companion §2's `dyn ArchStep`/`advance(ctx, cursor)`
  is the **deferred BET-1/BET-3** form (perf-gated, §9 R3), whereas the **core ships the `Runtime`
  enum** with forward static (§3). The companion doc's banner now records this supersession.
- **Three forms behind a `SpecForm` SELECTOR, not one `propose()`.** `DraftModel` and `MtpHead` both
  **borrow the target's embedder** (`dflash.rs:14` "single embedding table shared"; MTP runs the
  target's trunk+lm_head, `mtp_spec.rs:1265,1351`; the draft embeds via `target.weights.token_embd`,
  `speculative.rs:3095`) — so the no-rebuild story is real for both. But the **verify shapes diverge**:
  linear left-to-right (`verify_dflash_block`) vs **tree multi-branch** (`verify_dflash_block_tree`,
  needs `TreeVerifyCtx{positions, attn_bias}`, commit picks a *path* not a prefix) vs MTP's batched-K
  trunk collapse. Each `SpecForm` variant owns its own `step()` body (the existing `spec_step_*` moved
  verbatim); there are **two verify shapes** (linear + tree), acknowledged like the two grammar shapes.
- **Module split (mistral.rs's literal layout):** `proposer / verifier / staging / cache`. The
  staging invariant — *staged tokens valid for exactly the next verify pass, else drop+rollback* —
  applies to **accept/reject within verify**, where `DeltaNetSnapshot::save_from/restore_to`
  (`speculative.rs:2214/2274`) gives clean rollback. It does **not** cover grammar rejection (§5).
- **Reference:** mistral.rs `mistralrs-core/src/speculative/{driver,proposer,config,staging,cache,
  target}.rs`; `SpeculativeConfig::Mtp`; the target-side mixin is **`SpeculativeTargetMixin`** (no
  `: Pipeline` supertrait — the first draft misnamed it `SpeculativePipelineExt`).

---

## 5. BET 1 (deferred) — result & telemetry; and the grammar reality

Lands with the spec subsystem (until strategies unify, AR telemetry is fine as-is). Recorded for
correctness:

- **Per-strategy rich results stay LOCAL.** `SpecStepResult` (`speculative.rs:813`) and `MtpSpecResult`
  (`mtp_spec.rs:791`, disjoint extra fields) never cross the shared seam; only a 2-field
  `Committed { tokens, position_delta }` (AR is N=1) and a `StepStatus { hit_eos }` do.
- **Telemetry rides a `CycleEvent` enum with one variant PER spec result type** — `Ar`,
  `Spec{&SpecStepResult}`, `Mtp{&MtpSpecResult}` (and a reserved `Tree`). The invariant is **"no
  shared *projection* struct"** (the lossy `record_tau` mistake), **not** "one variant" — there are
  already ≥2 because the result types have disjoint fields. By-reference payloads avoid field-copy
  regrowth; τ/histograms are derived sink-side (`SpecStats::record`, `speculative.rs:2154`).
- **Crate layering:** `CycleEvent`/`TelemetrySink` must live in the **daemon/serve top layer**, not a
  "shared crate below the arches" — `hipfire-runtime` and `hipfire-arch-qwen35` already mutually
  depend (`Cargo.toml`), so a type below the arches cannot name `SpecStepResult` without a cycle.
- **Grammar is N-shaped (per-arch), and rejection DETONATES, it does not roll back.** Each arch family
  has its own `Matcher` (`deepseek4::grammar` vs `hipfire_arch_qwen35::grammar`, `daemon.rs:3050` vs
  `4182`). On a spec grammar violation the code sets `grammar_violated = hit_eos = true; break`
  (`daemon.rs:4471-4478`) then **nukes the session**: clears `conversation_tokens`, frees checkpoints,
  zeroes every DeltaNet `s_matrix/s_scale/conv_state`, resets `seq_pos = 0` and
  `kv_cache.compact_offset = 0` (`daemon.rs:4628-4647`, "rather reset than emit garbage"). So the
  contract is: `sample_masked(logits, mask)` (host pre-mask) vs `validate_committed(ids) -> RetainMask`
  (spec) **plus an explicit `CommitEffect::SessionPoisoned` (full reset)** — *not* a `rollback_to(pos)`.
  Building real partial rollback (re-roll `compact_offset` + restore `DeltaNetState` + truncate draft
  rings) is net-new work, not a rename; defer it with the spec subsystem. **Do not force-unify the
  per-arch matchers** — that's a coherence-regression trap (R3).

---

## 6. CORE (partial) — the loop, split honestly into "now" vs "needs the redesign"

The 9 `generate_*` in `daemon.rs` are **1 AR loop copy-pasted 6×** (qwen2/deepseek4/lfm2moe/minimax/
plain/ep — drifting: 4 of the 6 lack the budget-alert thread), **2 different control flows** (dflash
spec, VL), **1 transport variant** (pp>1). The eventual shape is 3 strategies (`Autoregressive`,
`Speculative`, `VisionPrefix`) selected by capability, with EP/PP as AR *transport variants* (a
different `forward` primitive), **as an enum, not a `dyn` trait** (only 3). But the dedup splits in two
by what it can touch **today**:

- **Ships now, truly redesign-agnostic (~2–4k lines, strangler-fig).** Pure helpers that take slices/
  tokenizer refs, **not** `&mut dyn ArchInstance`: `build_prompt` (render + `maybe_normalize_prompt` +
  LCP cache), `emit` (detok + stream), `stop_check`, and the `fwht256` setup dedup (the 6 inlined
  `gen_fwht_signs` pairs in `weight_backend.rs`). The **old `generate_*` adopt them immediately** →
  single source of truth before any rebuild. This is the honest "ship now" subset.
- **Needs the ownership layer first.** `prefill`, `sample_masked`/`validate_committed`, `kv_append` —
  these touch arch-specific forward/KV/grammar state reached today via `m.arch_id ==` branches
  (`daemon.rs:5844-5990`) and per-crate grammar types. They unify only *after* `Runtime` lands. The
  first draft's "~10k lines, ship now" conflated the two; the realistic now-number is ~2–4k.

Selection replaces the `arch_id ==` ladder with a pure function of the load-time `Capabilities` +
request (`arch_id == 7/9/10/11` vanish — those arches just want plain AR).

### 6.1 EP rehome — current state and staged future steps (qwen35 `Forward::Custom` prerequisite)

`qwen35` can't be declared `Forward::Custom` (§7 / BET 2) while EP is a sibling ownership universe.
Correcting the record before the work:

**Shape (correction).** EP is **not** a pre-forward `Stage`. Its collective is a **per-MoE-layer
all-reduce interleaved in the layer loop** (`hipfire-runtime/src/ep.rs:73` `run_layer_program_ep`:
zero → owned experts → `all_reduce_sum_f32` → add-back, per `Moe` op). Layer *N+1*'s replicated
attention must read the cross-rank-summed residual from layer *N*, so it cannot be hoisted. EP is a
**multi-rank transport variant wrapping the per-layer program loop** — same bucket as PP, not as VL
splice. (The companion doc's "awkward 10%" bullet is corrected to match.)

**Current state (verified).**
- The per-layer collective seam is **already shared**: `run_layer_program_ep<B: ForwardBindings>`
  owns the whole zero/experts/all-reduce/add lifecycle; qwen35/ds4/minimax each supply only a
  `ForwardBindings` impl. Not duplicated.
- What **is** duplicated 3× is the **outer driver** (embed-per-rank → loop layers building N
  bindings + call the shared executor → rank-0 norm+lm_head → sync): `qwen35.rs:11087`
  (`forward_ep`) + `forward_prefill_batch_ep`, deepseek4 `forward.rs:~2187`, minimax
  `forward.rs:~1144`.
- **qwen35 EP is substrate-only — not reachable from the daemon.** `EpArch` (`lib.rs:436`) has only
  `Ds4`/`Minimax`; qwen35's EP functions are exercised only by `ep_decode_parity.rs`.
- EP weights are **replicated**, only MoE experts sharded; the only cross-rank divergence is at
  `Moe` (ep.rs:19-23) — which is why EP needs no attention-sharding seam.

**Staged steps (only C unblocks `Forward::Custom`).**
- **A — generic EP driver dedup (doable NOW, no core dependency).** Extract the duplicated outer
  driver from qwen35/ds4/minimax into one generic driver in `hipfire-runtime` over a small per-arch
  trait, reusing `run_layer_program_ep`. Shrinks the future-C surface (3 drivers → 1). Lifetime-heavy
  (per-layer `Vec<Bindings>` borrowing `kv_per_rank.iter_mut()`); touches live ds4/minimax EP → must
  hold byte-parity (`ep_decode_parity.rs` + multi-GPU coherence on hiptrx).
- **B — wire qwen35 EP to the daemon (orthogonal product value).** Add `EpArch::Qwen35` +
  `load_model_ep`/`generate_ep` arms so qwen35-A3B EP is usable across hiptrx (4×gfx1201).
  Mechanical, but **adds** EP surface — moves *away* from C; validate on hiptrx (ssh), not locally.
- **C — full rehome (the actual unblock; gated on the core Runtime-enum refactor).** Fold
  `EpState`/`EpArch` into `Runtime::Qwen35Rt.ep` (§2) so EP becomes one *transport mode* of the
  arch's own runtime; delete the sibling state + `generate_ep` fork. Only then is "the qwen35
  forward" one thing that `Forward::Custom(qwen35_forward)` can name. Large; can't start until the
  core lands (and its unified-contract form was already rejected 3× — see
  `archspec-and-dyn-boundary.md` §2).

**Dependency:** core Runtime-enum (NOT landed) → C → coherent qwen35 `Forward::Custom` (BET 2).
A is independent prep; B is orthogonal.

---

## 7. BET 2 (deferred) — `ArchSpec` authoring surface

Has its own treatment in the companion doc (`archspec-and-dyn-boundary.md` §3–§5). One-line summary:
config-field rows + one interpreter (kills the ×2 hand-walked `config_from_hfq`/`_safetensors`),
`Forward::DenseTransformer` over the existing `Step` op-list (zero arch code; per-op stays static via
`execute_steps`), `Forward::Custom + Block`s for hybrids (DeltaNet/MoE/MLA stay hand-written HIP,
referenced by name). Adding a dense arch → one file. **`QuantCodec`/`fwht256_inplace` is orthogonal**
(below `WeightBackend`; do the fwht extraction first, independently). Gated on ≥2 dense arches
validating `DenseTransformer` with byte-identical token-ids. **Risk R2 (escape-hatch proliferation)
is real** — a hard conformance budget, not convention.

---

## 8. CORE — preserve & generalize the existing per-layer cache split

The first draft proposed vLLM-style "per-layer cache specs." **hipfire already solves this** — not one
global KV layout, but `KvCache` (per-layer `Vec<GpuTensor>` + `layer_is_boundary` + `_filtered`
constructors taking `is_kv_layer: &[bool]`, `llama.rs:4369,5123`) **plus** a separate `DeltaNetState`,
chosen per layer by `config.layer_types[i]` (`speculative.rs:585`). The ownership refactor must
**preserve** this split. The one modest generalization worth taking: replace the proliferating
`new_gpu_*_filtered` constructor family with a `CacheKind` enum per layer (FullAttn / LinearAttn / MLA-
latent / conv1d-ring), so a 3rd cache kind is a variant, not a new sibling struct + constructor. Not
a vLLM-style rewrite.

---

## 9. Risks & mitigations

- **R1 — telemetry enum regrows into the fat struct.** *Mitigation:* `CycleEvent::Spec` carries
  `&SpecStepResult` by reference (no field copy); new accounting goes in the arch-crate struct. New
  *variants* are unavoidable (one per spec result type) but bounded by the number of strategies.
- **R2 — `ArchSpec` escape hatches proliferate** (BET 2). *Mitigation:* hard conformance budget +
  `cargo test` byte-identical gate; a 2nd arch sharing a `Custom` means that shape was a missing enum
  variant (data), not a closure.
- **R3 — per-token indirection perf / grammar mis-unification.** *Mitigation A (perf):* forward stays
  **static** (enum) in the core; the dyn boundary is BET 3, gated on a microbench — wrap *only*
  `forward`+`sample` behind `Box<dyn>` on a throwaway branch, A/B a **small** model (TinyLlama-class,
  warmed, temp=0, byte-identical prompt, median of 5); **≥5% disqualifies, <1% & stable required.**
  *Mitigation B (grammar):* keep per-arch matchers; `sample_masked`/`validate_committed` non-
  interchangeable; add a constraint-validity assertion to the DFlash coherence gate.
- **R4 — GPU teardown lifecycle.** `Model` is single-owner `Option<Model>`; `free(self, gpu)` consumes
  it; debug `Drop` panics if un-freed. No `Arc`, no `into_inner`, no shared-weights leak.
- **R5 — migration divergence.** Strangler-fig + delete-on-green, CI-enforced (§10).

---

## 10. Migration order (strangler-fig, not big-bang)

1. **Now, truly redesign-agnostic (~2–4k lines):** the **pure-helper subset** (`build_prompt`/`emit`/
   `stop_check`/`fwht256` dedup) as functions over slices/tokenizer refs. **The old `generate_*` adopt
   them immediately** → one source of truth before any rebuild. Ship alongside **L1** (VRAM leak fix)
   and **L2** (fold the non-core arches' `Option` sibling fields into the `ModelState`/`Runtime` enum
   → exhaustive teardown). These are the blessed near-term items and need zero commitment to the rest.
2. **Ownership fix:** `Option<Model>` + `Session` + `Runtime` **enum** + exhaustive `GpuOwned::free`;
   hoist the pager out of `Qwen35Weights`; move the mmap to `Model.mmap` and make the target pieces
   long-lived `self.kv/dn/scratch` — **this is what deletes the transient-`ModelSlot` rebuild**
   (`daemon.rs:3837-3920`), a consequence of the ownership split, not of L2's enum fold alone.
   **Forward stays statically dispatched.**
3. **Port llama first** (simplest; already on `execute_steps`) behind `Runtime`; gate byte-identical
   token-ids + coherence + perf. **Delete the old llama path the moment it's green** — never keep an
   arch on both paths past one merge (CI gate: fail if a `generate_*` and its `Runtime` arm coexist for
   the same `arch_id`). This bounds the two-path window *per arch*, not globally.
4. **Migrate arch-by-arch** (qwen2 → qwen35 dense → qwen35 hybrid → ds4/minimax/lfm2moe → VL),
   deleting each old path on green. Grammar stays per-arch. **EP folds in here (§6.1 rework C):
   `EpState`/`EpArch` collapse into the arch's `Runtime` variant as the ds4/minimax/qwen35 arms
   migrate; the §6.1 step-A driver dedup may land independently beforehand to shrink that fold.**
5. **BET 1 — spec unification** (own doc): `Speculative<T>` + `SpecForm` selector + linear/tree verify
   + `proposer/verifier/staging/cache`. After ≥2 arches on `Runtime`.
6. **BET 2 — `ArchSpec`** (companion doc). After ≥2 dense arches validate `DenseTransformer`.
7. **BET 3 — dyn `ArchInstance` forward** — only if the R3 microbench clears <1% and open-world arches
   become a real need. Default: stay enum.

---

## 11. Prior-art grounding

`mistral.rs` is the reference to read while building BET 1 (Rust, single-process, closest peer):
`dyn Pipeline` boundary, a target-side **`SpeculativeTargetMixin`** trait (not `SpeculativePipelineExt`
— first-draft misname), `SpeculativeConfig::Mtp`, spec split into
`proposer/verifier/staging/cache/target`, the proposer borrowing the target embedder (no rebuild —
verified real). **Steal:** that module split + the borrow-the-embedder move (BET 1). **From
vLLM/SGLang:** the draft-can-be-independently-quantized insight; the staged-valid-for-one-pass
invariant *within verify*. **Avoid:** llama.cpp's flat GGUF manifest + hand-coded `build_<arch>()`
(defeats hipfire's FWHT/DeltaNet/INT4-WMMA moat); TGI's gRPC router↔backend split (hot-path
serialization, forbids the zero-copy embedder borrow); SGLang's worker-per-draft processes (here it's
a borrow + a method). Grammar stays in the sampler/verifier layer in every engine — never in the
model — and in hipfire it's per-arch, so don't force-unify the matchers.

**Sources:** mistral.rs `mistralrs-core/src/speculative/{driver,proposer,config,staging,cache,
target}.rs` + `pipeline/{normal,multimodal}.rs` (file split, `SpeculativeConfig::Mtp`, and the
`target_embedder` borrow verified against master); vLLM ModelRunner/Worker/Scheduler + V1 spec;
SGLang EAGLE/MTP/DFLASH/NGRAM; TGI router↔backend; llama.cpp `build_<arch>()`.
