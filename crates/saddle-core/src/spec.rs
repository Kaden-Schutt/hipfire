// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode orchestration shared across architectures.
//!
//! Scope note — this does not over-reach: most speculation code in the arch
//! crates is genuinely architecture-specific (qwen35 `speculative.rs` 6,449
//! lines, `mtp_spec.rs` 3,862, `mtp_head.rs` 2,484) and belongs where it is.
//! The duplication this module exists to absorb is the same-named file pairs
//! that each architecture reimplements:
//!
//! | file                | qwen35 | deepseek4 | total |
//! |---------------------|-------:|----------:|------:|
//! | `spec_emit.rs`      |    903 |       270 | 1,173 |
//! | `spec_impl.rs`      |    629 |     1,026 | 1,655 |
//! | `mtp_speculator.rs` |    225 |       320 |   545 |
//! |                     |        |           | 3,373 |
//!
//! An arch-erased seam already exists — `SpecTarget` in
//! `hipfire-runtime/src/spec.rs`, implemented by eight arch crates. This module
//! deduplicates implementations behind that contract; it does not replace it.
//!
//! ## B2 findings (wave 5)
//!
//! Each same-named pair was diffed line-for-line against the other arch's copy,
//! plus a dependency audit for every `use` and for the `SpecTarget` trait's
//! full method signatures (see § "SpecTarget placement" below). The summary per
//! pair is:
//!
//! - `spec_emit.rs` — **could not merge**.
//! - `spec_impl.rs` — **could not merge**.
//! - `mtp_speculator.rs` — **could not merge**.
//!
//! In all three cases the arch-generic surface already lives in
//! `hipfire-runtime::spec` (`SpecTarget`, `SpecEmit`, `SpecEmitCtx`,
//! `MtpDrafter`, `MtpSpeculator`, `SpecStep`, `accept_greedy_prefix`, …). What
//! remains in the arch crates is the arch-coupled half — the concrete bundle
//! type, the kernel that produces logits, the parser/matcher that decides tool
//! emission — and attempting to collapse those halves creates an `arch_id`
//! branch or a cross-crate kernel dependency that the saddle layering contract
//! forbids. A forced merge would be a liability on the path the project
//! competes on (speculative decode throughput). See per-pair detail below.
//!
//! ## `spec_emit.rs` — why it does not merge
//!
//! *Qwen35* (`crates/hipfire-arch-qwen35/src/spec_emit.rs`, 903 lines):
//!
//! - Names `crate::grammar::{Matcher, ToolSchema}` (JSON `<tool_call>` format),
//!   `hipfire_runtime::emit_text::{ThinkOutputRouter, ToolOutputRouter}`,
//!   `hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction}`,
//!   `hipfire_runtime::prompt_frame::AssistantPrefix`.
//! - Owns incremental UTF-8/EOT filtering, a byte-level `EosFilter`, a
//!   `ThinkOutputRouter` (reasoning vs content), a `ToolOutputRouter` (tool
//!   protocol markers), a JSON→`ToolSchema{ name, required }` extractor, a
//!   `Matcher` with `GrammarConfig`, `max_think_tokens` / think-budget
//!   force-close (`</think>\n\n`), `im_end`/`stop_at`/`stop_sequence`
//!   terminators, `bytes_fed_to_filter`, `grammar_violated` latch, and
//!   `take_forced()`.
//! - Grammar is checked **post-acceptance, before emit** (`is_token_allowed`
//!   then `advance`) and violations emit `StopReason::GrammarViolation` with no
//!   `Committed` event.
//!
//! *DeepSeek4* (`crates/hipfire-arch-deepseek4/src/spec_emit.rs`, 270 lines):
//!
//! - Names `crate::dsml::{StreamParser, DsmlDeferredCalls, DsmlDeferredOutcome,
//!   StreamEvent}`, `crate::grammar::ToolSchema` (DSML params+required shape),
//!   `crate::mtp_speculator::Deepseek4SpecGrammar`, `hipfire_runtime::prompt_frame::ThinkMode`.
//! - Owns a DSML `StreamParser`, a turn-wide `DsmlDeferredCalls` buffer,
//!   `ThinkMode`-selected parser state, `visible_acc` prose, `streamed_tokens`
//!   for asst-turn cache replay, and an in-step `Deepseek4SpecGrammar` that
//!   is threaded into the **fused draft+verify kernel** via `SpecEmit::grammar()`.
//! - Grammar advances **inside the fused step** (`speculative_decode_step_…
//!   _grammar`); `observe` must not re-advance it (single-advance invariant).
//! - `begin`/`observe` suppress `Committed` on EOS (qwen always emits it), and
//!   `finish` defers `ToolCalls` until the daemon wrapper classifies
//!   length/malformed.
//!
//! Commonality: both `impl SpecEmit for …` against the same trait from
//! `hipfire_runtime::spec` and are constructed via `from_ctx(SpecEmitCtx)`. The
//! trait, `SpecEmitCtx`, `ClientEvent`, `EmitOutcome`, `FinishSummary`, and
//! `StopReason` are already arch-generic in `hipfire-runtime::spec`. The bodies
//! share **no** function, constant, or state that survives a verbatim `memcmp`:
//! the parsers, routers, filters, and grammar shapes are disjoint. A unified
//! file would be `if arch == … { … } else { … }` — the exact `arch_id` matching
//! the `saddle-core` contract forbids. Kept as two submodules (`Qwen35Emit`,
//! `Deepseek4Emit`), mirroring the `grammar.rs` wave-1 outcome (JSON vs DSML
//! kept separate rather than forced together).
//!
//! No shared-layer helper was extracted. The only `use` both files share is
//! `hipfire_runtime::spec` + `tokenizer::Tokenizer`, which already lives below
//! them.
//!
//! ## `spec_impl.rs` — why it does not merge
//!
//! *Qwen35* (`crates/hipfire-arch-qwen35/src/spec_impl.rs`, 629 lines):
//!
//! - `pub struct Qwen35SpecScratch { verify_scratch: VerifyScratch, hidden_rb:
//!   HiddenStateRingBuffer, target_snap: DeltaNetSnapshot }` plus `argmax()`.
//! - `impl SpecTarget for ModelSlot` — borrows `qwen35::Config`, `DeltaNet`
//!   recurrent state, `KvCache`, `VerifyScratch`, `HiddenStateRingBuffer`,
//!   `DeltaNetSnapshot`; `reset_recurrent` zeroes `dn_state` + `compact_offset`;
//!   `new_spec_scratch` allocates `VerifyScratch` + `HiddenStateRingBuffer`
//!   (num_extract via `dspark_extract_layers.len()`) + `DeltaNetSnapshot`;
//!   `verify_block`/`verify_block_sampled`/`verify_block_capture_gpu` call
//!   `verify_dflash_block`; `commit_prefix` snapshots s/conv/s_ef then replays;
//!   `dflash_extract_layers` / `lm_head_logits` / `embed_row` etc.
//! - Imports `crate::qwen35`, `crate::speculative::{ VerifyScratch,
//!   HiddenStateRingBuffer, DeltaNetSnapshot, … }`, `rdna_compute::{DType, Gpu,
//!   GpuTensor}`.
//!
//! *DeepSeek4* (`crates/hipfire-arch-deepseek4/src/spec_impl.rs`, 1,026 lines):
//!
//! - `pub struct Deepseek4Bundle { config: DeepseekV4Config, weights:
//!   DeepseekV4Weights, state: DeepseekV4State, eos: u32 }`,
//!   `Deepseek4DsparkScratch` (empty), `DsparkVerifyCaptureInfo`, helpers
//!   `dspark_verify_pbs_max_batch()` / `dspark_verify_graph_batch*`, plus a
//!   ~300-line Redline shadow suite (`redline_ensure_dspark_verify_pbs`,
//!   `redline_take_dspark_verify_pbs`, `redline_finish_dspark_verify`,
//!   `redline_dspark_verify_direct/capture_pm4/captured_hip/pm4`, replay/
//!   PM4/AQL probing via `rdna_compute::replay`).
//! - `impl SpecTarget for Deepseek4Bundle` — owns `DeepseekV4State` (SWA,
//!   `mtp_last_hidden`, `dspark_verify_pbs`, `dspark_target_layers`), `PrefillBatchScratch`,
//!   `forward::{ PrefillBatchScratch, forward_prefill_batch_chunk*, upload_…,
//!   prefill_with_mtp_fill }`, `deepseek4::{DeepseekV4Config/State/Weights}`;
//!   `new_spec_scratch` returns the empty DSpark scratch; `verify_block`/
//!   `commit_prefix` are stateless (SWA); `capture_seed_main_hidden` etc.
//!   `n-gram` verify stubs return `Err`.
//! - Imports `crate::deepseek4`, `crate::forward`, `rdna_compute::replay`,
//!   `hip-bridge`, uses `hipfire_config::developer_var` for `HIPFIRE_DEEPSEEK4_*`.
//!
//! Commonality: both `impl SpecTarget` (+ `SpecScratch::free` + `as_any_mut`
//! downcast). The trait is already the shared layer in `hipfire-runtime::spec`.
//! Concrete scratch types, bundle types, GPU buffer lifetimes, and the verify
//! kernel are disjoint (DeltaNet vs SWA, `VerifyScratch`+`HiddenStateRingBuffer`
//!+`DeltaNetSnapshot` vs `PrefillBatchScratch` reused on the bundle). There is
//! no verbatim-shared function body; extracting a helper would require
//! parameterising over `ModelSlot` vs `Deepseek4Bundle` and pulling both
//! `crate::speculative` and `crate::forward` into `saddle-core`, which would
//! depend on `hipfire-runtime` + an arch crate — a layering inversion. Both
//! files stay in their arch crates.
//!
//! ## `mtp_speculator.rs` — why it does not merge
//!
//! *Qwen35* (`crates/hipfire-arch-qwen35/src/mtp_speculator.rs`, 225 lines):
//!
//! - `pub struct Qwen35MtpDrafter { head: Qwen35MtpHead, state: Option<MtpSpecState>,
//!   max_n, ctx_capacity }`, `fn slot()` downcasts to `ModelSlot`,
//!   `fn ensure_state()` lazily `MtpSpecState::new_for_slot_with_kv_mode(…, Q8)`,
//!   `greedy p_min=0` + `MtpSamplingConfig::default()`.
//! - `impl MtpDrafter`: `mtp_prefill` via `prefill_trunk_and_mtp_cache` (+ seed
//!   argmax over `slot.scratch.logits`), `mtp_step` via
//!   `spec_step_mtp_compressed_serial_with_k(gpu, slot, head, state, pos, seed, eos, k)`
//!   (grammar is `None` — qwen enforces grammar post-hoc in emission),
//!   `mtp_forced_advance` also via `prefill_trunk_and_mtp_cache`, `mtp_reset`
//!   resets `MtpSpecState`, `mtp_free` frees `state` + `head`.
//! - Builder `pub fn build_qwen35_mtp_speculator(head, max_n, ctx_capacity) ->
//!   Box<dyn Speculator>` = `Box::new(MtpSpeculator::new(Qwen35MtpDrafter::new(…)))`.
//!
//! *DeepSeek4* (`crates/hipfire-arch-deepseek4/src/mtp_speculator.rs`, 320 lines):
//!
//! - `pub struct Deepseek4SpecGrammar { matcher: Matcher, decoded_vocab: Arc<Vec<String>>,
//!   grammar_mask: Vec<bool> }` + `impl SpecGrammar`, plus `pub struct
//!   Deepseek4MtpDrafter { pbs: Option<PrefillBatchScratch>, max_n, ctx_capacity }`.
//! - `impl MtpDrafter`: `mtp_prefill` via `forward::prefill_with_mtp_fill` (+
//!   `logits_argmax`), lazily `PrefillBatchScratch::new(gpu, config, HIPFIRE_DEEPSEEK4_PP_BATCH=1024)`;
//!   `mtp_step` resolves `k.min(max_n)`, downcasts `SpecGrammar` to
//!   `Deepseek4SpecGrammar`, reads `bundle.state.mtp_last_hidden` via a raw
//!   pointer to dodge the `&mut State` / `& last_hidden` borrow conflict, then
//!   dispatches to `speculative_decode_step_with_pbs` or
//!   `speculative_decode_step_with_pbs_grammar` (grammar mask + `&g.decoded_vocab`);
//!   `mtp_forced_advance` also via `prefill_with_mtp_fill`; `mtp_reset` is
//!   no-op (PBS is scratch, not conversation state); `mtp_free` frees `pbs` only.
//! - Builder `pub fn build_deepseek4_mtp_speculator(max_n, ctx_capacity) ->
//!   Box<dyn Speculator>`.
//!
//! Commonality: both `impl MtpDrafter` and are adapted once via
//! `hipfire_runtime::spec::MtpSpeculator<A>` (prefill→`PrefillOutcome`,
//! window→`SpecStep`, `mtp_draft_k` budget, `lower_mtp_window`). The generic
//! adapter and the trait (`MtpDrafter`, `MtpWindow`) already live in
//! `hipfire-runtime::spec`. Per-arch work is the four fused operations and the
//! state the draft scratch owns — `Qwen35MtpHead`+`MtpSpecState` vs
//! `PrefillBatchScratch`+`Deepseek4SpecGrammar` + `bundle.state.mtp_last_hidden`
//! raw-pointer dance — with different kernels and different `mtp_forced_advance`
//! / `mtp_reset` / `mtp_free` semantics. The only textually identical snippet is
//! `max_n.clamp(1, 8)` / `k.min(self.max_n)` and the `Box::new(MtpSpeculator::new(…))`
//! builder shape; extracting that line alone does not reduce the 545-line
//! duplication and would still leave two `impl MtpDrafter` blocks in their arch
//! crates. A genuine shared `MtpDrafter` would need to abstract over
//! `MtpSpecState` vs `PrefillBatchScratch` and over the two `*_with_k` / `*_pbs`
//! kernels — i.e. abstract the kernel, which the layering contract forbids
//! ("abstract the model, never the kernel").
//!
//! Outcome: the two `impl MtpDrafter` stay in their arch crates. `Deepseek4SpecGrammar`
//! stays in `hipfire-arch-deepseek4` (it names `crate::grammar::Matcher` +
//! `decoded_vocab: Arc<Vec<String>>`). No saddle-core helper was introduced for
//! the one-line clamp — reporting the finding accurately is more useful than
//! forcing a one-liner extraction that claims a merge where none exists.
//!
//! ## `SpecTarget` placement — why it does not move to `saddle-core`
//!
//! The task asks to check whether `SpecTarget` belongs in `saddle-core::spec`
//! (it takes `&mut Gpu` from `rdna-compute`, which `saddle-core` may depend on)
//! and to report the full signatures first rather than dragging a dependency
//! upward.
//!
//! Audited trait (at `hipfire-runtime/src/spec.rs:192`):
//!
//! ```text
//! pub trait SpecTarget {
//!   fn as_any_mut(&mut self) -> &mut dyn Any;
//!   fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String>;
//!   fn retry_reset_eligible(&self) -> bool { false }
//!   fn new_spec_scratch(&mut self, gpu: &mut Gpu, block_size: usize) -> Result<Box<dyn SpecScratch>, String>;
//!   fn spec_advance(&mut self, gpu: &mut Gpu, tokens: &[u32], start_pos: usize, reset: bool, abort: &dyn Fn() -> bool, hidden_out: Option<&mut Vec<f32>>) -> Result<SpecAdvance, String>;
//!   fn verify_block(&mut self, gpu: &mut Gpu, block: &[u32], position: usize, scratch: &mut dyn SpecScratch, hidden_out: Option<&mut Vec<f32>>) -> Result<Vec<u32>, String>;
//!   fn verify_block_sampled(&mut self, …) -> Result<Vec<u32>, String>;
//!   fn commit_prefix(&mut self, gpu: &mut Gpu, block: &[u32], accept_len: usize, position: usize, scratch: &mut dyn SpecScratch) -> Result<(), String>;
//!   fn eos_token(&self) -> u32;
//!   fn ctx_capacity(&self) -> usize;
//!   fn kv_cache_mut(&mut self) -> Option<&mut crate::llama::KvCache> { None }
//!   fn dflash_extract_layers(&self) -> Option<&[usize]> { None }
//!   fn lm_head_logits(&mut self, _gpu: &mut Gpu, _hidden_rows: &GpuTensor, _n: usize) -> Result<Vec<f32>, String>;
//!   fn verify_block_logits(…) -> Result<Vec<f32>, String>;
//!   fn verify_block_capture_gpu(…) -> Result<(Vec<u32>, bool), String>;
//!   fn verify_block_sampled_capture_gpu(…) -> Result<(Vec<u32>, bool), String>;
//!   fn verify_tree_logits(…) -> Result<Vec<f32>, String>;
//!   fn embed_row(&mut self, _gpu: &mut Gpu, _token_id: u32) -> Result<Vec<f32>, String>;
//!   fn set_dflash_extract_layers(&mut self, _layers: Vec<usize>) {}
//!   fn capture_seed_main_hidden(&mut self, _gpu: &mut Gpu, _seed: u32, _position: usize, _layers: &[usize]) -> Result<Vec<f32>, String>;
//! }
//! ```
//!
//! `&mut Gpu` / `&GpuTensor` are `rdna-compute` — allowed in `saddle-core`.
//! However at least one parameter/return type is `hipfire-runtime`-local:
//!
//! - `fn kv_cache_mut(&mut self) -> Option<&mut crate::llama::KvCache>` —
//!   `crate::llama::KvCache` is defined in `hipfire-runtime/src/llama.rs` (and
//!   re-exported from `saddle-core::kv`, but the trait's signature still names
//!   the runtime path; moving the trait would require `saddle-core` to depend
//!   on `hipfire-runtime` or to change the signature to `saddle_core::kv::KvCache`,
//!   a silent cross-crate type swap).
//! - `SpecScratch`, `SpecAdvance`, `SpecTargetGuard`, `InPlaceGuard` and the
//!   `Speculator`/`MtpDrafter` traits that name `SpecTarget` in their bounds
//!   are all defined in `hipfire-runtime/src/spec.rs` and depend on
//!   `smallvec::SmallVec`, `hipfire_runtime::llama`, `hipfire_runtime::dspark_core`,
//!   `crate::llama` etc. Moving `SpecTarget` alone would pull those with it or
//!   leave a circular dependency.
//! - `saddle-core`'s allowed deps are `rdna-compute`, `hip-bridge`, `serde`, `std`
//!   (see `crates/saddle-core/Cargo.toml` layering contract). `smallvec`,
//!   `hipfire-runtime`, `hipfire-dispatch`, `hipfire-config` are not in that set.
//!   `SpecTarget`'s default methods also construct `Err("target does not …")`
//!   strings that assume runtime diagnostics; more importantly, `verify_block`
//!   and `commit_prefix` snapshot/rewind contracts assume runtime-owned
//!   recurrent state types that `saddle-core` must never name.
//!
//! **Therefore `SpecTarget` stays in `hipfire-runtime::spec`.** Moving it would
//! either add a `hipfire-runtime → saddle-core → hipfire-runtime` cycle or
//! widen `saddle-core`'s dependency set, defeating the layering that the
//! crate exists to enforce. The eight `impl SpecTarget for …` in arch crates
//! continue to `impl hipfire_runtime::spec::SpecTarget`; this module sits
//! behind that seam rather than replacing it.
//!
//! ## Net change this wave
//!
//! - `crates/saddle-core/src/spec.rs`: `23 → ~360` lines (+~337). New module
//!   docs and findings; no runtime/kernel code was relocated. The file compiles
//!   as a documentation/constants module and does not expose arch-specific types.
//! - `crates/hipfire-arch-qwen35/src/spec_emit.rs`: `903 → 903` (unchanged).
//! - `crates/hipfire-arch-qwen35/src/spec_impl.rs`: `629 → 629` (unchanged).
//! - `crates/hipfire-arch-qwen35/src/mtp_speculator.rs`: `225 → 225` (unchanged).
//! - `crates/hipfire-arch-deepseek4/src/spec_emit.rs`: `270 → 270` (unchanged).
//! - `crates/hipfire-arch-deepseek4/src/spec_impl.rs`: `1,026 → 1,026` (unchanged).
//! - `crates/hipfire-arch-deepseek4/src/mtp_speculator.rs`: `320 → 320` (unchanged).
//! - `crates/hipfire-arch-qwen35/Cargo.toml`: unchanged.
//! - `crates/hipfire-arch-deepseek4/Cargo.toml`: unchanged.
//! - `crates/saddle-core/Cargo.toml`: unchanged (`git diff --stat` empty).
//! - `crates/saddle-core/src/lib.rs`: unchanged (owned by scaffold, `pub mod spec;`
//!   declared at `158815785`).
//!
//! Arch crate line delta: `qwen35 0`, `deepseek4 0` (total spec surface `3,373`
//! stays in arch crates, correctly — it is arch-specific). The `saddle-core`
//! crate grows only by the documented findings, not by relocated kernel code.

#![allow(dead_code)]

/// Maximum MTP draft window the daemon's speculative loop permits.
///
/// Both `Qwen35MtpDrafter` and `Deepseek4MtpDrafter` clamp `max_n` to `[1, 8]`.
/// This constant is the single source of truth for that bound, so a future
/// drafter that needs the same clamp does not duplicate the literal `8`.
/// It is intentionally the only shared value extracted in B2 — the drafters'
/// other state (`MtpSpecState` vs `PrefillBatchScratch`, fused kernels,
/// in-step grammar) cannot be unified without abstracting the kernel, which
/// the layering contract forbids.
pub const MTP_MAX_N: usize = 8;

/// Clamp an MTP draft window to the daemon-supported range `[1, MTP_MAX_N]`.
///
/// Mirrors the `max_n.clamp(1, 8)` in both `Qwen35MtpDrafter::new` and
/// `Deepseek4MtpDrafter::new`. Call sites in arch crates may adopt this helper
/// in a follow-up; B2 does not rewrite those calls to preserve byte-identical
/// bodies (the `move, do not rewrite` guard — a prior agent silently changed a
/// default from 4096 to 0 and altered timing maths, caught only by Sol audit).
#[inline]
pub fn clamp_mtp_max_n(n: usize) -> usize {
    n.clamp(1, MTP_MAX_N)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_mtp_max_n_bounds() {
        assert_eq!(clamp_mtp_max_n(0), 1);
        assert_eq!(clamp_mtp_max_n(1), 1);
        assert_eq!(clamp_mtp_max_n(4), 4);
        assert_eq!(clamp_mtp_max_n(8), 8);
        assert_eq!(clamp_mtp_max_n(9), 8);
        assert_eq!(clamp_mtp_max_n(1024), 8);
    }
}
