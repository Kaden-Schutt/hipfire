# Handover — `build_speculator` registry + n-gram drafter

> **STATUS 2026-06-22 (implemented):** Both items landed.
> `spec_build::build_speculator` registry dispatches DFlash → else opt-in
> `NgramSpeculator` (new `crates/hipfire-loader/src/spec_ngram.rs`), gated
> `HIPFIRE_NGRAM_DRAFT=1` for qwen35 arch 5/6 with no draft. The daemon gate
> needed **no** change (greedy `Some(speculator)` already routes through
> `generate_dflash`; n-gram is greedy).
>
> **Validation:** `serve-multiturn-gate.sh` PASS (AR + DFlash arms — registry
> refactor no-regression). n-gram arm **coherent on 9B** (`coherence_probe`
> verdict OK on code + copy prompts).
>
> **Perf is situational (the headline finding).** The DeltaNet verify runs the
> GDN recurrence *sequentially* over the `b`-token block, so a window costs ~`b`×
> the DeltaNet part for `accept+1` tokens. It WINS only when PLD acceptance is
> high (high prompt-copy: edit/refactor/verbatim) — measured **+15% decode** on a
> 9B copy task (46.3 vs 40.2 tok/s). On low-copy "write" prompts the verify cost
> dominates and it LOSES to AR (~22 vs 40 tok/s on the 9B lru-cache prompt).
> Implemented two cheap-path optimizations: skip the rewind on full-accept, and
> batched (not per-token) replay on partial-accept. The tiny 0.8B loops under the
> arm (batched-verify greedy ≠ per-token AR; the model is too fragile) — opt-in
> + 9B-coherent makes that acceptable, but it's why the flag stays off by default.
> A broad win needs a draft *model* (DFlash: block-parallel verify + GDN-tape
> rollback), which the model-free arm structurally cannot match on DeltaNet.

**Branch:** `feature/speculator-abstraction`
**Written:** 2026-06-22 (after the daemon DFlash fold landed)
**Goal of next session:** (1) introduce a generic `build_speculator` registry that
picks a drafter at load time, and (2) add a **model-free n-gram / PLD drafter**
as the second `Speculator` arm — proving the abstraction with a real second
implementation and giving spec-decode speedup to qwen35 models that have **no
DFlash draft model** loaded.

---

## 1. Where things stand (read this first)

The arch-generic speculative-decode seam is **live and folded**:

- `crates/hipfire-runtime/src/spec.rs` — the `Speculator` trait, the unified
  `SpecStep` result, `SpecTarget` (borrowed verifier), `PrefillOutcome`,
  `EvictRetain`, `SpecGrammar` (marker). Read the whole file — it's ~250 lines
  and documents the contract precisely.
- `crates/hipfire-loader/src/spec_build.rs` — `Qwen35SlotGuard` (RAII target
  borrow), `DflashSpeculator` (the **only** impl today), `build_dflash_speculator`
  (load-time constructor, reads env).
- `crates/hipfire-arch-qwen35/src/spec_impl.rs` — `impl SpecTarget for ModelSlot`.
- `crates/hipfire-runtime/examples/daemon.rs::generate_dflash` — the daemon's
  DFlash decode loop, now driving `&mut dyn Speculator` (no inline `spec_step_*`).

Relevant commits on this branch: `8a2f3ed8` (the daemon fold), `256a111a`
(doc-header refresh), `8054500f` (`build_dflash_speculator`), `d338437e` /
`86dc9225` / `c5cbcd54` (earlier seam stages).

**Load site (where the registry plugs in)** — `crates/hipfire-loader/src/lib.rs:703`:
```rust
let speculator =
    dflash.map(|s| crate::spec_build::build_dflash_speculator(s, eviction.is_none()));
```
`LoadedModel.speculator: Option<Box<dyn Speculator>>` (lib.rs ~316).

**Daemon routing gate** — `examples/daemon.rs` (~line 6020):
```rust
if m.speculator.is_some() && temp <= 1e-6 && (m.arch_id == 5 || m.arch_id == 6)
    && !budgeted_thinking_needs_ar && !force_ar_chat { generate_dflash(...) } else { generate(...) }
```

---

## 2. The big surprise: n-gram / PLD infra ALREADY EXISTS

Do **not** build drafting from scratch. `crates/hipfire-arch-qwen35/src/speculative.rs`
already has:

- `NgramCache` (line ~1960): a bigram `(a,b) → next-token histogram` predictor
  with `observe`/`observe_many`/`predict(a,b) -> Option<(tok,count)>` and a
  `min_count` trust threshold.
- **PLD (Prompt Lookup Decoding, Saxena 2023)** (line ~2007+): context-suffix
  self-match drafting. Comment cites 2–18× higher acceptance than bigram.
- `spec_step_dflash` already takes `ngram_cache: Option<&NgramCache>` and
  `pld_spine: Option<&[u32]>` params (speculative.rs ~2981/2984) and has the
  accept logic wired (~3461). **The daemon currently passes `None` for both**
  (see `DflashSpeculator::step` in spec_build.rs — `None, // ngram_cache` /
  `None, // pld_spine`).

So the n-gram drafter is mostly an **integration + new-Speculator-arm** task, not
a kernel task. Two viable shapes (decide in session 1, lean toward B):

- **A (hybrid):** keep DFlash as the drafter but enable `NgramCache`/`pld_spine`
  as an override/augmentation. Smaller change, but doesn't prove the registry
  with a distinct arm.
- **B (clean second arm, RECOMMENDED):** a standalone `NgramSpeculator` that is
  **model-free** — it drafts from `NgramCache`/PLD over the committed-token
  history and **verifies with the target only** (no DFlash draft weights). This
  is the arm that justifies the registry and works when no draft model is loaded.

---

## 3. Mapping the n-gram drafter onto the `Speculator` trait (shape B)

The verifier-side primitive you need is "run the target on a token block and read
per-position argmax": use `hipfire_arch_qwen35::qwen35::forward_prefill_batch`
(qwen35.rs:6027) which writes per-position logits the caller can argmax. Method by
method:

- `prefill(gpu, target, prompt, prefill, start, cache_hit, resume_from, abort)`:
  advance the **target** over the prompt (KV + DeltaNet) and return first-token
  argmax. ⚠️ The current `DflashSpeculator::prefill` is entangled with DFlash's
  `target_hidden` ring / scatter. The n-gram drafter needs only the *plain target
  advance*. **Factor out a small helper** (e.g. reuse
  `seed_target_hidden_from_prompt_abortable` and ignore the hidden ring, or add a
  leaner `seed_target_prompt` that skips hidden extraction). Seed the `NgramCache`
  from the prompt via `observe_many`.
- `step(gpu, target, position, seed, emitted, grammar)`:
  1. Build a draft block: PLD self-match on `emitted` suffix (preferred) else
     `NgramCache::predict` chained K times.
  2. Run the target on `[seed, draft..]` via `forward_prefill_batch`; argmax each
     position.
  3. Accept the longest matching prefix; bonus = target argmax at divergence.
  4. Return `SpecStep::new(emit = accepted_drafts ++ [bonus], next_seed = bonus,
     proposed = K, accepted = matched)`. **`emit.len()` MUST equal `accepted+1`**
     — this is the load-bearing loop contract (see the `emit_len_drives_advance`
     test in spec.rs). After committing, `NgramCache::observe_many` the new tokens.
- `on_evict`: **default no-op** (no drafter-local GPU cache).
- `reset(gpu)`: clear the `NgramCache` (CPU). No GPU frees.
- `checkpoint` / `rewind_to` / `checkpoint_positions`: **defaults** (no
  divergent-render resume in v1; cold-prefill on divergence is acceptable).
- `block_size()`: the n-gram draft length K. `ctx_capacity()`: target capacity.
- `free(self, gpu)`: nothing to free; just consume the box.
- `requires_greedy()`: `true` for v1 (greedy verify).

**Where it lives:** `NgramSpeculator` needs qwen35 forward symbols, so put it in
`hipfire-loader` (alongside `DflashSpeculator`) or a new
`crates/hipfire-loader/src/spec_ngram.rs`. (Going arch-generic later would mean
adding a `verify_block(...)->Vec<u32>` method to `SpecTarget`; out of scope for v1.)

---

## 4. The registry

Replace the single `build_dflash_speculator` call at lib.rs:703 with a
`build_speculator(...)` that picks an arm:

```rust
// pseudocode — exact inputs TBD from the load context at lib.rs:674-704
pub fn build_speculator(
    arch_id: u8,
    dflash: Option<DflashState>,
    eviction_is_none: bool,
    // + whatever the n-gram arm needs (it needs no draft model)
) -> Option<Box<dyn Speculator>> {
    if let Some(df) = dflash { return Some(build_dflash_speculator(df, eviction_is_none)); }
    if ngram_enabled() && arch_is_qwen35(arch_id) { return Some(build_ngram_speculator(...)); }
    None
}
```

- Gate the n-gram arm behind an env flag (e.g. `HIPFIRE_NGRAM_DRAFT=1`) for v1 so
  it's opt-in until validated.
- **Per-request greedy routing:** the speculator is built once at load, so it
  can't see per-request `temp`. The daemon gate must consult the trait instead of
  hardcoding `temp <= 1e-6`. Change the gate (~daemon.rs:6020) to something like
  `m.speculator.is_some() && (temp <= 1e-6 || !spec.requires_greedy()) && ...`.
  For v1 (n-gram is greedy) the existing `temp <= 1e-6` already suffices, but make
  the gate trait-driven so a future sampling drafter doesn't need a daemon edit —
  that is the whole point of the abstraction.

---

## 5. Suggested step plan (each step → verify)

1. **Registry shell.** Add `build_speculator` wrapping the existing
   `build_dflash_speculator`; swap the lib.rs:703 call. → verify: build + the
   existing serve-multiturn DFlash arm still PASS (pure refactor, no behavior
   change).
2. **Leaner target prefill helper** (decouple plain target-advance from the DFlash
   hidden ring). → verify: DflashSpeculator still works (it can keep its own
   prefill; the helper is for n-gram).
3. **`NgramSpeculator` skeleton** implementing the trait with defaults + a trivial
   `step` that proposes 0 drafts (always falls back to 1 target token). → verify:
   build; wire behind `HIPFIRE_NGRAM_DRAFT=1`; run a single prompt, confirm
   coherent (it's just AR-equivalent at this point).
4. **Real PLD/bigram drafting in `step`.** → verify: τ > 1 on a repetitive code
   prompt; coherence holds.
5. **Coherence + multi-turn gates** (see §6).

---

## 6. Validation (mandatory) + the fmt rule

- `./scripts/serve-multiturn-gate.sh` — AR + DFlash arms must stay green (proves
  no regression). Add/observe an n-gram session if practical.
- `./scripts/coherence-gate.sh` — general daemon coherence.
- `./scripts/coherence-gate-dflash.sh` — only if you touch `spec_step_*`/kernels
  (the n-gram arm shouldn't).
- Build/clippy/test as usual.
- **FORMATTING: never `cargo fmt` in this repo** (it rewrites historical
  rustfmt-debt files). Use `scripts/fmt-changed.sh` to apply and
  `scripts/ci-rustfmt-changed.sh` to check (changed files only). This bit me on
  the fold — I had to revert 3 untouched files.
- Local box has a matched DFlash pair: `~/.hipfire/models/qwen3.6-27b.mq4` +
  `qwen36-27b-dflash-mq4.hfq`, plus `qwen3.5-0.8b.mq4` (AR). gfx card present;
  GPU lock via `source scripts/gpu-lock.sh && gpu_acquire`.

---

## 7. Gotchas

- **`SpecStep` contract:** `position += emit.len()` (NOT `accepted`); `emit` is the
  committed tail with the seed re-echo stripped; `next_seed == emit.last()`. The
  `emit_len_drives_advance_not_accepted` test in spec.rs pins this.
- **Borrow choreography in `generate_dflash`:** the slot guard borrows `m.state`,
  the speculator borrows `m.speculator`, and `m.seq_pos`/`conversation_tokens`/
  `m.eviction` are accessed as disjoint fields. Keep that discipline if you touch
  the loop.
- **Checkpoint ring** lives inside `DflashSpeculator` now; `m.dflash_checkpoints`
  is a vestigial always-empty field, and the 5 conversation-reset sites call
  `spec.reset(gpu)` to free the real ring. The n-gram arm has no ring (defaults).
- **MTP / DeepSeek4 and `SpecGrammar` Stage 3 are out of scope** (re-scoped on the
  MTP-stub discovery). `generate_deepseek4` keeps its own spec path.
