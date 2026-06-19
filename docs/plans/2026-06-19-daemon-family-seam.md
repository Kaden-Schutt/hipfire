# Daemon de-qwen-ification — the model-family serving seam

Status: **active** — started 2026-06-19. Prereq for [gemma3 bring-up](2026-06-19-gemma3-bringup.md).
Owner: chaingun.

## Why

Adding a new model family today means editing the 18k-line `hipfire-daemon`
`main.rs` in ~1000 places. The daemon is structurally welded to qwen35:

- **`LoadedModel` is an Option-soup** (`main.rs:3214`): a separate
  `Option<config>/Option<weights>/Option<state>` field group bolted on **per
  arch** — `q35_*`, `llama_*`, `qwen2_*`, `deepseek4_*`, `minimax_*`,
  `lfm2moe_*`, `dots_ocr_*`, `vision_*`. Adding gemma3 = add another field
  group + an `if let Some(gemma3_…)` arm at every site that branches on arch.
- **`generate()` (`main.rs:13986`) doubles as the qwen35 path AND the central
  arch dispatcher** — its body (≈14024–14290) delegates to `generate_qwen2`,
  `generate_deepseek4`, `generate_minimax`, `generate_lfm2moe`,
  `generate_multi`, `generate_dflash`, `generate_mtp`, then falls through to the
  qwen35/llama AR path. VL is dispatched even earlier (`8696`).
- **`hipfire-arch-qwen35` is a de-facto base crate**: 5 other arch crates
  (`llama`, `qwen35-vl`, `dots-ocr`, `qwen2`, `minimax`) depend on it for shared
  infra (`Qwen35ScratchSet`, `DeltaNetState`, session-state types), and
  `KvCache` lives in `hipfire-runtime::llama`. The "neutral" serving primitives
  are scattered across arch crates.
- **1008 `qwen35`/`DeltaNet` references in `main.rs`.**

The `Architecture` trait (`hipfire-runtime::arch`) was scaffolded (bring-up
triple + policy overrides) but **deliberately never abstracted the
forward/serving path**, so the daemon was never migrated onto it.

## Target architecture

Collapse the Option-soup into one boxed backend; route the per-arch
`generate_*` functions through an **object-safe serving trait**.

```
LoadedModel {
    arch_id: u32,
    backend: Box<dyn ServingBackend>,   // owns its own typed config/weights/state
    tokenizer: ...,                     // shared serving infra stays in the daemon
    seq_pos / max_seq / physical_cap / session bookkeeping ...
}
```

### Two-tier trait

`Architecture` (existing, **typed**, associated `Config/Weights/State`) stays for
load/bring-up. It is NOT object-safe, so the daemon loop cannot hold
`Box<dyn Architecture>`. Add an **object-safe** serving trait that erases the
associated types; each arch crate implements it for a struct that internally
owns its typed state.

```rust
/// Object-safe serving surface the daemon's generation loop drives.
pub trait ServingBackend: Send {
    fn arch_id(&self) -> u32;
    fn caps(&self) -> ArchCaps;
    fn eos_token(&self) -> u32;
    fn seq_pos(&self) -> usize;

    // Bespoke per-arch full-response loop (qwen35 dflash/mtp, deepseek4 mtp,
    // VL splice). Dense AR archs get a default impl via SimpleAr below.
    fn generate(&mut self, gpu: &mut Gpu, ctx: &mut GenerateCtx) -> GenerateResult;

    // Session lifecycle (multi-turn KV reuse).
    fn reset_session(&mut self, gpu: &mut Gpu, session_id: &str) -> Result<(), String>;
    fn drop_session(&mut self, session_id: &str);

    fn unload(self: Box<Self>, gpu: &mut Gpu);
}

/// Optional fast-path capabilities. The daemon checks these instead of
/// branching on arch_id; archs that lack a path return false and the daemon
/// uses the AR fallback.
#[derive(Clone, Copy, Default)]
pub struct ArchCaps {
    pub dflash: bool,        // DDTree / spec-decode draft+verify
    pub mtp: bool,           // multi-token-prediction head
    pub pipeline_parallel: bool,
    pub grouped_moe_batch: bool,
    pub vision: bool,
    pub paged_kv: bool,
}
```

`GenerateCtx` carries the serving-infra params the daemon owns (prompt,
system_prompt, sampling knobs, penalties, think_mode, tools, messages_history,
stop_sequences, streaming sink, evidence_dir) — extracted from today's
`generate()` 28-arg signature. Sampling, streaming, EOS-filter, and loop-guard
**stay in the daemon** and are reused by every backend.

```rust
/// Dense autoregressive archs implement only this; a blanket impl gives them
/// `ServingBackend::generate` (shared prefill→sample→stream→decode_step loop).
pub trait SimpleAr {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<Logits, String>;
    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<Logits, String>;
}
```

So a **new dense family (gemma3, llama, qwen2) implements `SimpleAr` only** and
inherits the entire serving loop. Bespoke families override
`ServingBackend::generate`.

## Migration order (strangler-fig; commit each ✓)

The existing qwen35 code is correct and perf-tuned — do NOT rewrite it up front.
Introduce the seam, onboard simple archs to prove it, route gemma3 through it
clean, then peel qwen35 onto it last.

- **P0 ✓** — Define `ServingBackend` + `ArchCaps` + `SimpleAr` + `GenerateCtx` in
  `hipfire-runtime::arch` (`ef4f4f30`). Compiles, zero behavior change. Also
  added `StopReason`/`ServeOutcome` and the shared `run_simple_ar` driver
  (`14c9de28`), later split into a reusable `decode_loop(gpu, backend, tok, eos,
  ctx, start_pos, prompt_tokens)` so non-token-stream prefills (the VL splice)
  can share the one streaming/stop loop (`ec54d01f`).
- **P1** — Relocate the neutral serving primitives (`KvCache`, generic scratch,
  session-state) out of `hipfire-arch-qwen35`/`llama` into a neutral home so arch
  crates stop depending on qwen35. (Can be deferred if too invasive; P0/P2 don't
  strictly need it.) **Deferred** — not needed for the backend impls below.
- **P2 (crate side ✓, daemon routing pending)** — qwen2 (arch_id 7) onboarded
  onto `SimpleAr` + `ServingBackend` as the proof-of-seam (`a6f6d1d9`). The
  *crate-side* backend exists and is tested; **routing the daemon's qwen2 path
  through it is the pending half** (see "Decision point" below).
- **P3 (crate side ✓)** — gemma3 text (arch_id 12, `3fc36d2e`) and gemma3-vl
  (arch_id 13, `ec54d01f`) implement `ServingBackend`. gemma3-vl overrides
  `serve` for the SigLIP→projector→image-token splice prefill, then hands off to
  the shared `decode_loop`; gemma3 text is a plain `run_simple_ar` delegate.
  Added `eos_token_id` to `Gemma3Config` (gemma3-it stops on `<end_of_turn>`=106,
  scalar-or-array) and `preprocess_image_bytes` for the daemon's raw image form.
- **P4** — Migrate the dense llama AR path and the qwen35 AR path onto the trait.
- **P5** — Migrate qwen35 fast paths (dflash/mtp/pp/grouped-moe) behind
  `ArchCaps`, overriding `ServingBackend::serve`.
- **P6** — Collapse `LoadedModel` Option-soup into the single `backend` field;
  delete the per-arch `generate_*` free functions and the `arch_id` match ladder.

## Decision point — daemon wiring (needs direction before proceeding)

All three seam archs (qwen2/gemma3/gemma3-vl) have committed, tested
`ServingBackend` impls. The remaining work all lives in the 18k-line
`hipfire-daemon/src/main.rs` and touches the **production hot path**, so it
warrants an explicit call rather than an autonomous edit:

1. **`load_model` has no arch_id 12/13 blocks yet** (`main.rs:10221` ladder stops
   at 11) — gemma3/medgemma can't be loaded by the daemon at all today. Need a
   load block per arch building the backend from the bring-up triple.
2. **`generate`'s ~2000-line dispatch** (`main.rs:~8000–10000`) owns the daemon's
   sampler, sessions, eviction, tool-calls and streaming protocol. Our
   `ServingBackend::serve`/`decode_loop` currently does its **own** greedy +
   JSONL `{"type":"token"}` streaming. Two routes:
   - **Additive (recommended for medgemma now):** add a `serving_backend:
     Option<Box<dyn ServingBackend>>` field + a `serve`-based branch for 12/13
     only, leaving qwen35/deepseek4/minimax fast paths untouched. Ships medgemma
     vision fast; greedy-only sampling at first.
   - **Full collapse (P4–P6):** thread the daemon sampler/sessions through
     `decode_loop`, migrate every arch, delete the Option-soup. Larger, riskier,
     better end state.
3. **Image bytes through the request protocol** — `GenerateCtx::image_bytes`
   exists; the daemon-protocol crate's request type and the `hipfire serve`
   path need a field to carry the encoded image to the backend.

## Testing

- `./tests/no-gpu-ci.sh` after each structural step (trait/type changes).
- Coherence gate + MQ4 speed gate where the forward/serving path is touched.
  **NB:** never run perf gates while a CPU-heavy quantize job runs — this
  gfx1151 box is a UMA APU (CPU/GPU share bandwidth); contention produced a
  false MQ4 "regression" on 2026-06-19. Quiet box only.

## Risks

- `GenerateCtx` surface: today's `generate()` takes 28 args wired through
  closures/streaming state. Extracting a clean ctx without changing streaming
  semantics is the fiddly part of P2/P4.
- qwen35 fast paths (dflash/mtp) interleave draft/verify/sample in ways that
  don't fit `SimpleAr`; they MUST stay as `ServingBackend::generate` overrides
  (P5), not be forced through decode_step.
- `Box<dyn>` at the serving boundary is per-token (coarse) → dispatch cost
  negligible vs thousands of per-layer kernel launches; the per-layer forward
  stays monomorphized inside each backend.
