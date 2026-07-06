# ArchSpec + the daemon↔model step contract

**Date:** 2026-06-13 (N1 revised same day, post-adversarial-review + ADHD)
**Branch:** `feature/paro-transparent-loading`
**Status:** §2 (the unified step contract) is **SUPERSEDED** — kept as a documented dead-end per
Rule 3. §3–§5 (the `ArchSpec` authoring surface) remain **live as BET 2** of the greenfield design.
**Scope:** How to minimize the effort of adding a new model architecture to hipfire.

> **⚠️ Supersession (2026-06-13).** A *third* review round found the `advance(ctx, cursor)` +
> `Speculative<T,D>` contract below was still over-built and rested on a misread lifecycle (the
> daemon rebuilds a transient `ModelSlot` per spec call). The conclusion: the unified-dispatch
> contract is the wrong *first* move — the root problem is **ownership**, not the contract shape.
> The shipping design is in **`2026-06-13-greenfield-engine-architecture.md`**: a `Runtime`
> **enum** with **forward statically dispatched** (no per-token vtable), the ownership split, and
> the spec/contract work deferred as gated BETs. Read §2 below as the rejected exploration that
> motivated the greenfield doc; the `dyn ArchStep`/`advance(ctx, cursor)` form here is the
> **deferred BET-1 (spec subsystem) / BET-3 (dyn boundary)** shape — *not* the shipping boundary,
> and explicitly **perf-gated** (greenfield §9 R3 microbench) before it could ever land.

> **N1 history (preserved per Rule 3):** this doc originally proposed a single
> `ArchInstance::decode_step` dyn boundary. **Round 1** — adversarial review shot it down (RED:
> two return contracts, a triple-`&mut` borrow, 6 spec signatures, big-bang); an ADHD pass
> found the keystone (sampling behind a ctx) and a three-layer `StepCtx`/`Progress` contract.
> **Round 2** — a second review found that contract still had three gaps (spec *accounting*
> didn't fit `Progress`; the *two-model* state needed a downcast; nothing *validated* it short
> of a big-bang); a second ADHD pass resolved all three into the `advance(ctx, cursor)` +
> `Speculative<T,D>` design in §2 below. **Round 3** — superseded (see banner above): the
> contract ships as gated BETs, not as the primary boundary. The naive sketch is kept as the
> documented strawman.

This doc is the written form of a 4-agent review of the unified-loading work on this
branch. It is the source-of-truth reasoning behind todo items **N1–N6 + C1–C3** in the
project memory (`unified-loading-review-todos`).

---

## 1. Context and the question

The branch landed two stacked abstractions that were supposed to make adding an
architecture cheap:

- **`WeightBackend`** (`crates/hipfire-runtime/src/weight_backend.rs`) — hides the quant
  matrix (`HfqBackend`/`ParoBackend`, `proj`/`norm`/`raw_f32`) from per-arch callers.
- **Carrier registry** (`crates/hipfire-loader/src/carriers.rs` + `lib.rs`) — a
  machine-checked, fail-loud dispatch table that replaced the old `match arch_id` ladder.

Both are genuinely good. The disjointness test (`carriers_are_disjoint`), the fail-loud
ambiguity error, and the `WeightSource` orchestrator in `model_load.rs` are the kind of
design we want more of.

**The problem:** these abstractions hide cost at the *caller* but concentrate it into a
few hand-maintained chokepoints. Adding an architecture is still a cross-cutting change
that edits files shared by every other arch. The review converged — independently, across
3 of 4 reviewers — on a single root cause.

---

## 2. The root cause (todo N1)

A causal chain, each link forced by the previous one:

```
forward() is deliberately kept OFF the Architecture trait   (runtime/src/arch.rs:30-46)
        │  (rationale: avoid dyn-dispatch cost in the hot loop)
        ▼
the runtime cannot hold a model opaquely
        ▼
LoadedModel becomes a ~45-field god-struct with ~20 per-arch Option<…> fields
        │   (hipfire-loader/src/lib.rs — deepseek4_weights, minimax_state, lfm2moe_config, …)
        ▼
hipfire-loader now structurally depends on every arch crate
        ▼
daemon.rs (~11k lines) hand-writes ~70 forward-dispatch ladders:
        if let Some(ref mut s) = m.deepseek4_weights { deepseek4::decode_step(s, …) }
```

So the *actual* cost of adding an arch — the part the `hipfire-arch-toy` README hides — is:

1. Add 3–5 `Option<…>` fields to `LoadedModel`.
2. Add a `None` initializer to `skeleton` / `skeleton_pp`.
3. Add a free branch to `unload_model` — **not compiler-enforced**, so forgetting it is a
   silent VRAM leak (this is exactly how **C2** below was found: `dots_ocr_weights` /
   `lfm2moe_weights` / `minimax_weights` appear to free only their `_state`).
4. Add a forward arm at every one of the ~70 daemon dispatch sites the arch participates in
   (decode, prefill, bench, spec-decode, EP).

Items 1–4 are edits to central structures that good design would leave *closed for
modification, open for extension*.

### Why the performance objection is weak

The trait keeps `forward` off itself to avoid dyn-dispatch cost. But dyn cost is
**per-call**, and `forward_step` / `forward_prefill` is called **once per token**. A single
vtable lookup amortized over a full transformer forward — thousands of kernel launches
across the layer stack — is unmeasurable. Nobody is proposing to dyn-dispatch individual
GEMVs; only the top-level entry point. The trait's own evidence for "measurable tok/s loss"
is *inner-loop* (per-op) dispatch, which is a different thing.

### The naive fix — and why it failed review

The obvious fix is an object-safe trait with one forward entry point:

```rust
// NAIVE — does NOT survive contact with the call sites. Kept here as the
// strawman the adversarial review (2026-06-13) shot down. See below.
pub trait ArchInstance: Send {
    fn decode_step(&mut self, ctx: &mut DecodeCtx) -> Result<StepOut, String>;
    fn prefill(&mut self, ctx: &mut PrefillCtx) -> Result<PrefillOut, String>;
    fn free(&mut self, gpu: &mut Gpu);
    fn as_spec_decode(&mut self) -> Option<&mut dyn SpecDecode> { None }
    fn as_ep(&mut self) -> Option<&mut dyn EpServe> { None }
}
```

An adversarial review of this sketch came back **RED**, with cited evidence:

1. **Two irreconcilable *return* contracts.** Host-logit arches
   (deepseek4/lfm2moe/minimax) return `Vec<f32>` and the daemon samples host-side
   (`daemon.rs:9325,9660,10029`); qwen35 spec-decode samples **on-GPU inside the tree
   step** and returns already-committed tokens (`SpecStepResult`). A single
   `decode_step(ctx) -> StepOut` cannot be both "return logits, caller samples" and "I
   already sampled and committed B tokens."
2. **A concrete simultaneous-`&mut` borrow.** `daemon.rs:3837-3977` holds `target` (the
   model) + `m.dflash` + `m.dflash_checkpoints` mutably **at the same time**. Move the model
   behind `Box<dyn>` and the qwen35-only spec state must move inside the box — at which point
   `seed_target_hidden_suffix` can no longer take them as separate `&mut` args. `Box<dyn>`
   coarsens borrow granularity and breaks every spec site.
3. **`as_spec_decode` swallows the win.** There are 6 flag-selected spec variants with
   different signatures; one downcast object can't span them, so the per-arch matching the
   trait claimed to delete reappears behind `Any`.
4. **Big-bang.** `generate()` hard-branches on `arch_id` into 9 structurally different
   pipelines *before* any unified call, so you cannot put one arch behind the trait and leave
   the rest — untestable until 100% converted, against the byte-identical-token gate.

The naive trait is the manifest trap (option d, §3) relocated onto the **borrow axis**.

### The contract (resolved across two review rounds)

The naive trait failed because the divergence is *four* problems, not one. A first divergent
pass (`/adhd`) found the keystone — **move sampling behind a capability so models commit tokens
instead of returning logits** — but a *second* adversarial review showed that keystone alone
still left three gaps: the spec **accounting** return (`{accepted, bonus_token, drafted}`)
didn't fit a uniform result; the spec **two-model** state (a draft model + rings) couldn't live
in a generic ctx without a downcast; and nothing **validated** any of it short of a big-bang. A
second ideation pass resolved all three. The settled contract:

```rust
pub trait ArchStep {            // object-safe; one vtable lookup per *token step*
    fn advance(&mut self, ctx: &mut StepCtx, cursor: &mut DaemonCursor) -> StepStatus;
}
pub struct StepStatus { pub hit_eos: bool }   // control flow only — never accounting
```

Two seams, deliberately split:

- **`StepCtx` = inputs + capabilities** the model *reads* — `gpu`, plus sampling
  (`sample_masked` / `validate_committed`, below).
- **`DaemonCursor` = the effects sink** the model *writes* — `push_token(TokenId)`,
  `advance_position(n)`, `set_seed(TokenId)`, `record_tau(drafts, accepted)` (default no-op).
  The arch applies its **own** accounting to the cursor; the daemon never reads arch-shaped
  result fields.

The daemon is thus generic over **behaviour** (what gets pushed to the cursor), not over
**shape** (a result struct). The four divergences become four rows, each resolving cleanly:

| Divergence | How it resolves | Seam |
|---|---|---|
| **commit** (1 vs N tokens) | `cursor.push_token` × N — AR is the N=1 case | `DaemonCursor` |
| **accounting** (position/seed/τ) | the step *self-applies* via `cursor.advance_position`/`set_seed`/`record_tau`; its fat result (`MtpSpecResult`/`SpecStepResult`) stays **private** | `DaemonCursor` |
| **constraint** (grammar) | genuinely two-shaped: `ctx.sample_masked` (host pre-mask) vs `ctx.validate_committed` (spec post-hoc reject) | `StepCtx` |
| **two-model state** | spec is its *own arch* (`Speculative<T,D>`), owning both models concretely — below | the arch, *not* the ctx |

**Why not the prettier forms.** (a) An associated result `type Step: Commit` is **not
object-safe** — you cannot `Box<dyn ArchStep>` when `advance` returns `Self::Step` by value. So
the self-application is folded *into* `advance` via the passed-in `cursor`: zero per-token
alloc, object-safe, arch crates stay out-of-tree. (b) A uniform `Progress { committed, accepted,
bonus, rolled_back, … }` would grow a spec-shaped tail plain decode never uses — the fat struct
the first review warned about, relocated. The cursor avoids both: accounting is a *sequence of
effect calls*, not a returned shape. (Ergonomic escape hatch that survives: a blanket
`impl<S: Commit> ArchStep for S` lets an arch author *write* a private `Step: Commit` and the
blanket bridges it to the object-safe method — no alloc, no object-safety break.)

#### Spec-decode is an arch, not a daemon mode (resolves the two-model knot)

The second review's sharpest finding: putting spec state in `StepCtx` does **not** dissolve the
borrow hazard — it makes `StepCtx` arch-specific or forces a `Box<dyn SpecState>` downcast, and
the real hazard is a **four-way** borrow involving a **second model** (the draft) at
`daemon.rs:4416`, partly in prefill/seeding. The resolution is composition:

```rust
pub struct Speculative<T: TargetModel, D: DraftModel> {
    target: T,            // CONCRETE, not dyn — verify needs target internals
    draft:  D,
    rings:  DflashState,  // the 11 ex-sibling fields, now wrapper-private
}
impl<T: TargetModel, D: DraftModel> ArchStep for Speculative<T, D> { /* advance = spec_step body */ }
```

- `Speculative` **is** an `ArchStep`. It owns target + draft + rings as its own fields, so the
  four-way borrow becomes **disjoint `&mut self.field` borrows** (legal Rust). The daemon holds
  one `dyn ArchStep` and never sees the split — the dyn boundary sits at `Speculative` itself
  (per-token, fine).
- **`target` must be concrete, not `dyn`.** Verify reaches target internals a bare `ArchStep`
  can't surface — `target.{scratch.logits, final_hidden, dn_snapshot/restore, kv_mut}` — so
  `Speculative` is *generic* over a `TargetModel` super-surface. Making target `dyn` would push
  per-op virtual calls into the verify inner loop (the banned cost). Monomorphised target/draft
  *inside*; dyn boundary *outside*.
- **Seeding is a `Speculative::seed()` method, not on `ArchStep`** — a one-shot pre-loop phase
  on the same `TargetModel` surface, so it never co-occurs with the steady-state borrow.

hipfire already has the precedent (`SpecPair { target, draft }`, `speculative.rs:720`); this
generalises it and folds `DflashState` in. **Consequence:** the ~70 `if m.dflash.is_some()`
ladders, the ~10 spec entrypoints, and `generate_dflash` collapse into the *same*
`while …advance(ctx, cursor)` loop as plain decode — spec-decode stops being a concept and
becomes "an arch whose `advance` commits N and whose constructor took two models."

#### Dispatch + the awkward 10% (manifest, stages)

- **Dispatch by capability manifest, not `arch_id`.** Each arch publishes `Capabilities` at
  load (`{ spec, samples_on_gpu, needs_vision_tower, is_ep, … }`); the 9 `generate_*` pipelines
  collapse to one strategy-parameterised shell (`m.dflash.is_some() && temp<=eps && arch_id∈{5,6}`
  → `m.caps.spec.is_dflash() && temp<=eps`). Prefer **capabilities-as-code** (`sampler: &dyn
  SamplerStrategy` over a bool — declaring == implementing) + a `cargo test` conformance check
  (`caps.spec==DFlash ⟹ dflash present`) to stop drift; `mut_state: &[StateKind]` is
  documentation, *not* compile-time borrow safety (the actual `&mut` in `advance` is).
- **The awkward 10% as DATA — but VL and EP are different shapes.** *VL splice* genuinely *is*
  pre-forward: the vision tower runs once and embeddings are spliced in before any layer, so a
  daemon-composed pre-forward `Stage` fits and `advance` sees assembled hidden states. *EP is
  NOT pre-forward* — its all-reduce is a **per-MoE-layer collective interleaved inside the layer
  loop** (`hipfire-runtime/src/ep.rs:73`), so it is a **multi-rank transport variant wrapping the
  per-layer program loop**, not a gather. See the greenfield doc §6.1 for the EP-rehome staging.
  The ~10 spec entrypoints become a `StepPlan` value over the **`execute_steps` op-list llama
  already runs** (N6); novel kernels (DeltaNet/MoE/MLA) stay hand-written, referenced by name —
  data wires, code does the math.

#### Constraint is genuinely two-shaped — and must not be faked

The one divergence that does **not** unify. Host path masks logits *before* sampling
(`daemon.rs:3149-3151`, then matcher-advance `3177-3178`); the DFlash path **cannot** reach the
verifier tree's per-slot logits and degrades to **post-hoc rejection** (`4127-4143`, mirror
`res.retain_mask` `4216-4220`). A single `ctx.sample()` would hide this — risking a silent
DFlash grammar-behaviour change (a coherence regression dressed as a refactor). So `StepCtx`
names *both*: `sample_masked(logits, mask)` (host pre-mask) and `validate_committed(ids) ->
RetainMask` (spec post-hoc); the cursor's `advance_position` already absorbs the rollback count.
**The commit/accounting contracts unify; the constraint contract is honestly two-shaped — that
distinction is the design, not a wart.**

#### De-risking — two `cargo check` spikes before any big-bang

The third gap was *validation*: the clean arches (lfm2moe/minimax) can't exercise variable-width
commit, the two-model borrow, or the accounting, so a runtime prototype on them proves nothing.
The cheap, informative tests are compile-/host-level and need **no GPU**:

1. **Borrow spike (two-model knot).** Define `ArchStep` + `TargetModel` + `Speculative<T>` in
   `hipfire-arch-qwen35`, paste the existing `spec_step_dflash` body into `advance` renaming
   params to `self.*`, `cargo check`. The one question: do the four disjoint borrows +
   `T::forward_hidden(&mut self.target)` coexist without `E0499`? (A returned `&GpuTensor`
   colliding with a `&mut self.rings` write → fix is `forward_hidden_into(dst)`, found at
   compile time — matches how the code already copies into `target_hidden_host`.)
2. **Contract spike + property test (accounting).** A `#[allow(dead_code)]` fn that destructures
   the real `SpecStepResult` and binds the real `spec_step_dflash` borrows (never runs, must
   compile — so any future reshape fails CI). Plus a `#[cfg(test)]` cursor-mirror driven by a
   **scripted** degenerate draft (fixed `drafted` vector) + scripted target oracle accepting the
   first `k ∈ {0, 2, b-1}` — hitting rollback / partial / full-accept+bonus-seed, asserting
   `position == Σ(accepted+1)` and `τ == Σaccepted/cycles`. **Trap:** `draft == target` trivially
   accepts everything and proves nothing; the draft must be *scripted* so `accepted < drafted` is
   reachable.

Both can gate the pre-commit hook *before* a GPU boots, and become the permanent contract
regression harness.

### Net effect and effort

Adding a *dense* arch touches **one `REGISTRY` line + one spec**; the daemon never names it. All
four divergences resolve — commit/accounting via the cursor, constraint as two declared sampling
methods, the two-model knot as a wrapper-arch — and spec-decode/VL/EP stop being daemon code
paths.

**Effort: L**, reversing an explicit commented decision. But it is **incremental and gated**:
the two compile spikes prove the hard parts (borrow + accounting) with no GPU and no big-bang;
the leak-driven `ModelState` fold (the plan's Phase 1) ships independently; only then does the
contract land on a real arch. See the plan
(`docs/superpowers/plans/2026-06-13-noncore-teardown-and-step-contract.md`).

---

## 3. Could a DSL help? — data vs. code (todo N4/N5)

### What is already declarative (and good)

hipfire is ~70% of the way to a data-driven arch model and doesn't advertise it:

| Seam | Where | What it already does |
|------|-------|----------------------|
| `WeightBackend` | `weight_backend.rs` | quant matrix (~25 formats) in one arch-agnostic place |
| `load_layer<B>` | qwen35 `layer_driver.rs` | per-layer weight **table** — `b.proj("self_attn.q_proj", …)` struct literals |
| `Step` op-list | `hipfire-dispatch/.../steps.rs` | the forward pass is **already** an interpreted op-list (`Step::{Gemv, RmsnormAutomatic, Attend}`) with a fusion engine |
| `WeightAugmentor` | `augmentor.rs` | transparent ParoQuant plugin keyed on `QuantConfig` |
| `Architecture` | `arch.rs` | bring-up contract (`config_from_hfq`/`load_weights`/`new_state` + override structs) |

### What is needlessly code

For a transformer *family*, the following are **data** but currently hand-written:

| Concern | Today | Lines/arch | Reducible |
|---------|-------|-----------:|----------:|
| config field → metadata-key map (HFQ **and** safetensors, ×2) | hand-walked `serde_json` | ~250 | ~96% |
| per-layer weight schema | `load_layer` struct literals (already near-data) | ~110 | unify |
| tokenizer / chat-template / skeleton / `pp>1` wiring | `carriers.rs` boilerplate | ~80 | ~90% |
| KV-mode selection ladder | copy-pasted `match` (×4–7) | ~50 | one helper |
| RoPE style / norm convention / `norm_bias` / qk-norm | scattered constants | ~20 | data row |
| dense forward graph | `Step` list — already interpreted | templated | `Forward::DenseTransformer` |

A **dense** transformer arch is ~95% data. A **hybrid/MoE/MLA** arch is ~60% data plus a
bounded set of named, hand-written *blocks*.

### The irreducible core — what stays hand-written

- Novel kernels: DeltaNet recurrence `S_t = decay·S_{t-1} + β·v·kᵀ`
  (`gated_delta_net_*.hip`), MLA, conv1d-in-attention, new WMMA/dot paths. This is the
  FWHT / INT4-native moat — it is HIP, not data.
- Custom forward control flow for hybrids: LA-vs-FA layer scheduling, MoE expert dispatch,
  VL conditioning, spec-decode/DFlash tree logic.
- `new_state` scratch allocation for recurrent/hybrid models.
- Genuinely derived config semantics (per-layer type arrays, derived rotary factors) — a
  small `derive: fn(&mut Config)` hook, not a config row.

**Rule:** *data describes structure and wiring; code implements novel math.*

### DSL spectrum — decision

| Option | Verdict | Why |
|--------|---------|-----|
| (a) Better Rust factoring (CarrierKit, ConfigSchema rows, generalized layer table) | ✅ first step | zero new machinery, kills duplication now, net-negative cost on current arch count |
| (b) In-Rust `ArchSpec` aggregate + generic drivers | ✅ **recommended** | single source of truth per arch; type-checked & inlinable (respects hot-path + no-Python); novel attention as *named blocks* so hybrids fit |
| (c) Macro-DSL (`declare_arch!{…}`) | ❌ | sugar with zero capability gain over a struct; hostile errors; opaque expansion in a repo that bisects perf to a single newline |
| (d) External manifest (TOML/JSON), llama.cpp-style | ❌ | **structurally cannot describe the moat** (FWHT, DeltaNet, MLA, INT4 WMMA); adds runtime string-dispatch in the load path; buys no-recompile flexibility hipfire explicitly does not want (arches ship *with* the engine, perf-validated against specific kernels) |

llama.cpp gets away with a flat manifest because GGUF arches are near-homogeneous dense/MoE
transformers over a fixed kernel library. hipfire's differentiators are exactly the parts a
manifest can't express. Choose (b), reached via the (a) refactors.

---

## 4. The `ArchSpec` sketch (todo N5)

Generic drivers, built **once** in `hipfire-runtime`:

- `interpret_config(schema, source) -> Config` — one parser over the existing `ModelSource`
  abstraction (replaces both `config_from_hfq` and `config_from_safetensors`).
- `interpret_layers(layer_schema, &mut dyn WeightBackend, cfg)` — generalizes today's
  `load_layer`.
- `run_forward(forward_template, ctx)` — feeds the existing `Step` interpreter.
- `CarrierKit` — absorbs tokenizer / template / skeleton / `pp>1` / KV-mode (see §6, the
  prototype).

A **new dense arch** becomes one file:

```rust
pub static SMOLLM: ArchSpec = ArchSpec {
    name: "smollm",
    arch_ids: &[12],
    norm: Norm::Rms { bias: 0.0, qk_norm: false },
    rope: Rope::Llama { theta_key: "rope_theta" },

    // CONFIG: field ← metadata key (+ default). Replaces 2× hand-walked parsers.
    config: &[
        cfg!(dim,        "hidden_size"),
        cfg!(n_layers,   "num_hidden_layers"),
        cfg!(n_heads,    "num_attention_heads"),
        cfg!(n_kv_heads, "num_key_value_heads"),
        cfg!(hidden_dim, "intermediate_size"),
        cfg!(norm_eps,   "rms_norm_eps", default = 1e-5),
        cfg!(vocab_size, "vocab_size"),
    ],

    // LAYER: weight table. Generalizes today's load_layer struct literals.
    layer: LayerSchema::Dense(&[
        slot!(attn_norm, Norm, "input_layernorm.weight",         [dim]),
        slot!(wq,        Proj, "self_attn.q_proj", q_out_dim,     dim),
        slot!(wk,        Proj, "self_attn.k_proj", kv_dim,        dim),
        slot!(wv,        Proj, "self_attn.v_proj", kv_dim,        dim),
        slot!(wo,        Proj, "self_attn.o_proj", dim,           o_in),
        slot!(ffn_norm,  Norm, "post_attention_layernorm.weight", [dim]),
        slot!(w_gate,    Proj, "mlp.gate_proj",    hidden_dim,    dim),
        slot!(w_up,      Proj, "mlp.up_proj",      hidden_dim,    dim),
        slot!(w_down,    Proj, "mlp.down_proj",    dim,           hidden_dim),
    ]),

    // FORWARD: standard dense block — emits the existing Step list. No bespoke code.
    forward: Forward::DenseTransformer,

    kv: KvPolicy::Standard,        // CarrierKit's shared asym3/q8/fwht ladder
    overrides: Overrides { prompt: Raw, ..DEFAULT },
};

// Registration is one line — no Carrier struct, no claims_arch_id/load impl:
register_arch(&SMOLLM);
```

A **hybrid / novel arch** (qwen35-class) uses the *same* spec, swapping the forward and
layer schema to reference hand-written blocks:

```rust
layer: LayerSchema::PerType(&[              // LA vs FA chosen by config.layer_types
    (LayerType::LinearAttention, &DELTANET_SLOTS),
    (LayerType::FullAttention,   &FULL_ATTN_SLOTS),
]),
forward: Forward::Custom(qwen35_forward),   // ← escape hatch: hand-written hybrid graph
blocks:  &[Block::DeltaNet, Block::Moe { experts: 256 }],  // named kernels stay Rust
```

The DeltaNet recurrence, MoE routing, and VL tower stay exactly as hand-written kernels /
closures — the spec only *names and wires* them. Nothing about the moat moves to data;
only the boilerplate around it does. `Forward::Custom` is a first-class citizen, not a
grudging exception — hybrids are hipfire's whole point.

### Payoff

| Component (per dense arch) | Today | Under ArchSpec | Saved |
|---|---:|---:|---:|
| `config_from_hfq` + `config_from_safetensors` | ~250 | ~10 | ~96% |
| Carrier (`claims_arch_id`/`load`/tokenizer/template/skeleton/KV) | ~90 | ~1 + spec fields | ~90% |
| Layer schema | ~110 | ~10 | unified |
| Forward (dense) | ~170 | 0 | ~100% |
| **Dense arch total** | **~620** | **~60** | **~90%** |

Build cost ~1–1.5 weeks. The first two pieces (CarrierKit, ConfigSchema) are
**net-negative cost** on the current 7-carrier set. Full break-even at the 2nd–3rd new
dense arch — a threshold the project roadmap (Qwen2/3/3.5/3.6, Llama, DeepSeek, MiniMax,
LFM2, dots-ocr, "any model") crosses immediately.

---

## 5. The qwen35 33k lines, demystified (todo N6)

Don't read 33k as complexity. Rough budget:

- **~18%** genuinely novel arch logic (DeltaNet, MoE, MLA, hybrid scheduling) — *stays*.
- **~49%** co-located spec-decode / MTP / PFlash / grammar feature stack — **not "the
  architecture"** at all; a new arch needs none of it for a forward pass.
- **~32%** plumbing the trait split was supposed to factor out and didn't.

llama looks "cheap" (396 lines) only because its 8.3k-line body is shelved in
`runtime/llama.rs` — an accounting artifact, not a design win. **N6**: qwen35 hand-rolls a
~2–3k-line `SuperOp` / `lower_variant` / `run_fused_*_key` kernel-lowering layer that llama
**already deleted** by adopting `hipfire-dispatch::execute_steps`. Porting qwen35 to it
removes ~2–3k lines with no behavior change, attacking the 32%.

---

## 6. Sequencing

Ordered by *cost-adjusted value* (do net-negative-cost items first):

| # | Change | Effort | Note |
|---|--------|:------:|------|
| **N2** | **CarrierKit** — collapse the 5 byte-identical non-core carriers into a generic `HfqCarrier{id,name,load_fn}`; extract one `build_kv_cache()` for the ×4–7 KV-mode ladder (which has 3 *disagreeing* defaults) | S–M | **net-negative cost. Prototype first — validates the direction.** |
| **N1** | **step contract** — `advance(&mut self, ctx, cursor) -> StepStatus` (sampling reads `StepCtx`, commits/accounting write `DaemonCursor` → unifies commit+accounting, no fat struct); `Speculative<T,D>` wrapper-arch owns the two-model state (no downcast); capability-manifest dispatch; stages/`StepPlan` for VL/EP; constraint honestly two-shaped (`sample_masked` vs `validate_committed`); two `cargo check` spikes de-risk it. Exhaustive `free()` closes C2 | L | root cause; highest payoff; incremental + gated |
| **N3** | **`QuantCodec` registry** — one data table replaces the 3–4 lockstep `match quant_type` tables; extract `fwht256_inplace` (inlined **6×** in attractor-critical math, `weight_backend.rs:608,689,864,911,956,1032`) | L (+S for fwht) | do the fwht extraction first, independently |
| **N4** | **`ConfigSchema`** rows replace the ×2 hand-walked config parsers | M | |
| **N5** | **`ArchSpec`** aggregate + `Forward::DenseTransformer` over the Step interpreter | M | completes the declarative skin |
| **N6** | port qwen35 to `hipfire-dispatch::execute_steps` | M–L | deletes ~2–3k lines, no behavior change |

**N2 is the validation probe** for this whole direction and is implemented as a prototype
alongside this doc.

---

## 7. Correctness flags found in passing

File these regardless of whether the redesign proceeds:

- **C1** — `derive_arch_id` silently defaults an unknown `model_type` → `arch_id = 5`
  (Qwen35) at `safetensors_source.rs:244-249`. An unrecognized safetensors dir mis-routes
  to `Qwen35Carrier` and dies deep in weight loading with a confusing error instead of a
  clean "no carrier." This punches a hole through the otherwise-robust namespace guard.
  **Fix:** return an explicit unclaimed sentinel.
- **C2** — `unload_model` (`lib.rs:1170-1184`) appears to free only the `_state` for
  `dots_ocr` / `lfm2moe` / `minimax`, not their `_weights` — a possible VRAM leak. N1's
  exhaustive `match`/`free()` closes this whole class.
- **C3** — `bf16_loader.rs` is dead scaffold (`load_bf16_model` = `unimplemented!()`; only
  `is_gptq_target` is live). Inflates the surface under review. Pre-existing — flag, don't
  delete unasked.

---

## 8. Guardrails (project idiom)

- The spec layer is **load-time and config-time only**. It must not touch the forward hot
  path beyond feeding the existing `Step` interpreter, which already runs.
- Keep `Forward::Custom` first-class. A spec that can only express dense transformers would
  be the manifest trap (option d) in a nicer hat.
- Behavior-preserving refactors (N2, N6, the fwht extraction) must produce **byte-identical
  token-id streams** on the coherence-gate models before landing — same bar R2/R3 used.
- N1 and N3 touch dispatch/teardown and the quant cores, so they require the full
  coherence-gate (and cross-arch gates per #397: gfx1201 non-optional) before merge.
