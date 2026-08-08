// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transparent speculative-decode seam.
//!
//! The daemon's decode loop drives a `&mut dyn Speculator` and never learns
//! which drafter/mode (DFlash chain, DDTree tree, DeepSeek4 MTP, future
//! n-gram / EAGLE) is in use. Adding a drafter is a bounded-context change:
//! implement [`Speculator`] and register one arm in the loader's
//! `build_speculator` — no daemon edits.
//!
//! This is the arch-generic boundary anticipated by the `hipfire-arch-qwen35`
//! crate docs ("speculative.rs will become arch-generic"): the trait and the
//! unified result live here in the arch-agnostic runtime, while the
//! arch-coupled impls (which need `qwen35::*` / `deepseek4::*` symbols) stay in
//! their arch crates and `impl` this trait under the orphan rule.
//!
//! Status: the trait, the unified [`SpecStep`] result, and the borrowed-target /
//! erased-grammar interfaces are live. The daemon's DFlash decode loop drives a
//! `&mut dyn Speculator` (`examples/daemon.rs::generate_dflash`), with the
//! loader's `DflashSpeculator` as the sole impl. Still future work: a generic
//! `build_speculator` registry (dispatch on arch/draft kind) and additional
//! drafters (n-gram, MTP, EAGLE) — the AR one-token path still runs through
//! `generate()`, not this trait.

use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;

/// Outcome of one speculative-decode acceptance window, drafter-agnostic.
///
/// # Pending-seed contract (daemon `generate_spec`)
///
/// After `Speculator::step(position, seed)`:
/// - target/drafter state has processed the prior pending seed plus accepted
///   drafts;
/// - `emit` is accepted drafts plus the bonus (seed re-echo already stripped);
/// - `next_seed` is emitted but **unprocessed** (the new pending seed);
/// - `position` (caller-side) advances by `emit.len()` to the next write slot.
///
/// A forced suffix must first commit the current pending seed, then every
/// forced token except the last; the last forced token becomes the new pending
/// seed. Safe terminal completion flushes the final pending seed exactly once.
///
/// The two arch result types lower onto this:
/// - qwen35 `SpecStepResult` → `emit = committed[1..]` (the seed re-echo is
///   dropped), `next_seed = bonus_token`.
/// - deepseek4 MTP → `emit = accepted_tokens`, `next_seed = accepted_tokens.last()`.
///
/// `committed[1..].len()` equals `accepted + 1` for the chain drafters and
/// `accepted_tokens.len()` equals the MTP position advance, so a single
/// `position += emit.len()` is correct for both. `proposed`/`accepted` are τ
/// accounting only; they do NOT drive position math.
#[derive(Debug, Clone)]
pub struct SpecStep {
    /// Tokens to emit this window, in order, with any seed re-echo already
    /// stripped. `position += emit.len()`. Non-empty on `Ok` (forward progress).
    ///
    /// `SmallVec` so a future one-token-per-step drafter (e.g. n-gram) stays
    /// heap-alloc-free; only large spec windows spill to the heap.
    pub emit: SmallVec<[u32; 8]>,
    /// Seed for the next window — the verifier's preferred token at the
    /// divergence point (qwen35 `bonus_token`; MTP `accepted_tokens.last()`).
    pub next_seed: u32,
    /// Drafts offered this window (τ denominator).
    pub proposed: usize,
    /// Drafts accepted this window (τ numerator).
    pub accepted: usize,
}

impl SpecStep {
    /// Build a step from an `emit` iterator, hiding the `SmallVec` backing from
    /// caller crates that don't depend on `smallvec` (the per-arch lowering
    /// adapters `lower_qwen35` / `lower_mtp` live in the loader / arch crates).
    pub fn new(
        emit: impl IntoIterator<Item = u32>,
        next_seed: u32,
        proposed: usize,
        accepted: usize,
    ) -> Self {
        Self {
            emit: emit.into_iter().collect(),
            next_seed,
            proposed,
            accepted,
        }
    }

    /// Cap `emit` to at most `max_emit` tokens **after** construction.
    ///
    /// Prefer shrinking the draft/verify window *before* GPU commit (see
    /// [`Speculator::step`]'s `max_emit`). This helper is the last-resort
    /// parity clamp used when a window still overshoots: it keeps the prefix,
    /// reseeds from the last kept token, and shrinks `accepted` so τ accounting
    /// stays consistent with the truncated emit.
    pub fn cap_emit(mut self, max_emit: usize) -> Self {
        if max_emit == 0 {
            self.emit.clear();
            self.accepted = 0;
            return self;
        }
        if self.emit.len() <= max_emit {
            return self;
        }
        self.emit.truncate(max_emit);
        // emit = accepted drafts + bonus ⇒ accepted drafts kept = len-1 (or 0).
        self.accepted = self.accepted.min(max_emit.saturating_sub(1));
        if let Some(&last) = self.emit.last() {
            self.next_seed = last;
        }
        self
    }
}

/// Outcome of the shared greedy accept-prefix rule ([`accept_greedy_prefix`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreedyAccept {
    /// Accepted drafts in order, followed by the bonus token — UNLESS an
    /// accepted draft was itself the EOS token (then it stops there, no bonus).
    /// This is the committed tail the caller emits (callers that track a seed
    /// prepend it themselves).
    pub committed: Vec<u32>,
    /// Number of drafts accepted (excludes the bonus). `0..=drafts.len()`.
    pub accepted: usize,
    /// Whether decoding hit EOS inside this window (only possible when `eos`
    /// was `Some`): either an accepted draft or the bonus was the EOS token.
    pub hit_eos: bool,
}

/// The one greedy speculative-accept rule, shared by every drafter (DFlash,
/// MTP, deepseek4 non-grammar, n-gram). **Precompute-then-match**: the caller
/// supplies `target_pick[i]` — the verifier's chosen token at slot `i`, computed
/// however that arch needs (plain argmax, grammar-masked argmax, n-gram /
/// repeat-penalty-overridden argmax) — and this does ONLY the arch-invariant
/// part: accept the longest prefix where `target_pick[i] == drafts[i]`, then take
/// the bonus `target_pick[accepted]`.
///
/// `eos = Some(id)` enables EOS-early-stop *inside* the prefix (MTP semantics):
/// if an accepted draft equals `id`, stop there and emit no bonus. `eos = None`
/// (DFlash / n-gram) never early-stops and always appends the bonus.
///
/// Non-greedy rules (DFlash `temp>0` rejection sampling, MTP residual
/// acceptance) and stateful per-position grammar masking (deepseek4 tool-call
/// path, where `target_pick[i+1]` depends on the token accepted at `i`) are NOT
/// expressible here and stay at their call sites.
///
/// Requires `target_pick.len() >= drafts.len() + 1` (one extra slot for the
/// bonus at full acceptance).
pub fn accept_greedy_prefix(drafts: &[u32], target_pick: &[u32], eos: Option<u32>) -> GreedyAccept {
    debug_assert!(
        target_pick.len() >= drafts.len() + 1,
        "accept_greedy_prefix: target_pick (len {}) needs drafts+1 (len {})",
        target_pick.len(),
        drafts.len() + 1,
    );
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    let mut hit_eos = false;
    for (i, &d) in drafts.iter().enumerate() {
        if target_pick[i] == d {
            committed.push(d);
            accepted += 1;
            if eos == Some(d) {
                hit_eos = true;
                break;
            }
        } else {
            break;
        }
    }
    if !hit_eos {
        let bonus = target_pick[accepted];
        committed.push(bonus);
        if eos == Some(bonus) {
            hit_eos = true;
        }
    }
    GreedyAccept {
        committed,
        accepted,
        hit_eos,
    }
}

/// The verifier (target) model's GPU state, borrowed by [`Speculator::step`]
/// for the duration of one window. A `Speculator` impl recovers its concrete
/// target via `as_any_mut().downcast_mut::<T>()` (e.g. the qwen35 `ModelSlot`).
///
/// The borrowed-not-owned shape lets one decode loop hold the target across all
/// windows (taken from the model bundle once, via the loader's RAII slot guard)
/// while the speculator borrows it per step — no per-step ownership transfer.
pub trait SpecTarget {
    /// Downcast hook: `target.as_any_mut().downcast_mut::<ModelSlot>()`.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Zero the target's recurrent (DeltaNet) state and reset the KV eviction
    /// offset — used by the daemon's mid-generation abort path in place of its
    /// current inline memset loop. Returns `Err` when any HIP memset/bind fails
    /// so production rollback can attest `rolled_back:false`.
    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String>;

    /// Whether this target's architecture reset-core is complete enough for
    /// serve-hardening retry. Default `false` (explicitly ineligible). Retry
    /// candidates override to `true` only when recurrent/EF/KV/cache/graph/
    /// drafter/adaptive residuals are covered (see `reset_core` inventory).
    fn retry_reset_eligible(&self) -> bool {
        false
    }
    // ── Arch-generic speculation primitives ─────────────────────────────────
    //
    // These let a *model-free* speculator (n-gram / PLD) drive any arch's target
    // without knowing its internals: the target owns ALL verify mechanics (the
    // batched forward, the per-position lm_head, the recurrent snapshot/rewind,
    // and the arch-specific scratch), while the speculator owns only policy
    // (drafting + acceptance). The arch-specific verify scratch is created by the
    // target via [`new_spec_scratch`](Self::new_spec_scratch) and handed back on
    // every call as an erased `&mut dyn SpecScratch`, so no arch type leaks into
    // the speculator and the speculator owns the scratch's lifetime.

    /// Allocate arch-specific verify scratch sized to `block_size` (the max
    /// speculation window). The speculator owns the returned box for its lifetime
    /// and frees it via [`SpecScratch::free`].
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String>;

    /// Advance the target over `tokens` from absolute `start_pos` (chunked,
    /// abortable), returning the greedy argmax at the LAST position. When
    /// `reset` is true, the target zeroes recurrent + KV state first
    /// (cache-miss prefill); `false` continues from the current state
    /// (cache-hit suffix, or the partial-accept replay).
    ///
    /// `hidden_out`: when `Some` and `dflash_extract_layers()` is `Some(layers)`,
    /// the target appends, per processed position, the concat of residual hidden
    /// at `layers` (`layers.len() × dim` f32/row) to the provided `Vec`. Ignored
    /// (no-op) when `None` or when `dflash_extract_layers()` returns `None`.
    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String>;

    /// Advance the target for speculative prefill with an explicit cold-start
    /// signal. The signal is lifecycle metadata, not a second reset authority:
    /// the model owner has already performed the total reset on a cache miss.
    /// Targets that have a numerically distinct cold prefill may override this
    /// hook; the default preserves the ordinary advance path.
    fn spec_advance_cold_start(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        cold_start: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        let _ = cold_start;
        self.spec_advance(gpu, tokens, start_pos, false, abort, hidden_out)
    }

    /// Run the target over `block` at absolute `position`, returning the greedy
    /// argmax at each of the `block.len()` positions (`argmax[i]` is the target's
    /// next-token prediction after consuming `block[0..=i]`). Leaves target state
    /// advanced by `block.len()`.
    ///
    /// CONTRACT: this MUST first snapshot whatever recurrent state
    /// [`commit_prefix`](Self::commit_prefix) needs to rewind (e.g. the DeltaNet
    /// S/conv state AND the Q8 error-feedback residual) INTO `scratch`, *before*
    /// running the forward that advances it. Stateless (pure-attention) arches
    /// snapshot nothing.
    ///
    /// `hidden_out`: when `Some` and `dflash_extract_layers()` is `Some(layers)`,
    /// the target appends, per processed position, the concat of residual hidden
    /// at `layers` (`layers.len() × dim` f32/row). Ignored when `None` or when
    /// `dflash_extract_layers()` returns `None`.
    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String>;

    /// Like [`verify_block`](Self::verify_block) but returns per-position SAMPLED
    /// tokens (sample `s_i ~ p_T(top_k,top_p)` from the target logits) instead of
    /// argmax. For a point-mass n-gram draft this makes the shared
    /// `accept_greedy_prefix(draft, picks)` flow faithful temp-T speculation:
    /// "accept the guess iff it equals the target sample" is exactly
    /// distribution-preserving (the committed token is always the target sample).
    /// Same snapshot CONTRACT as `verify_block`.
    ///
    /// Default: `Err` — a target that has NOT implemented sampled verify must
    /// never silently fall back to greedy (that would decode temp>0 at argmax).
    /// The drafter's `requires_greedy()`/`build_speculator` gate is what keeps a
    /// temp>0 request off this path for such a target; the `Err` is the
    /// belt-and-suspenders guard if one ever slips through.
    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        temp: f32,
        top_p: f32,
        top_k: usize,
        rng_state: &mut u64,
    ) -> Result<Vec<u32>, String> {
        let _ = (gpu, block, position, scratch, temp, top_p, top_k, rng_state);
        Err("verify_block_sampled: this target does not support sampled verify".into())
    }

    /// Fix target state to reflect exactly the committed prefix
    /// `block[..accept_len + 1]` (after [`verify_block`](Self::verify_block)
    /// over-advanced it by `block.len()`). Cases:
    /// - full accept (`accept_len == block.len() - 1`): no-op — verify already
    ///   left state at the right position;
    /// - recurrent + partial: restore the snapshot saved in `scratch` (incl. the
    ///   s_ef residual) and replay `block[..accept_len + 1]` with the SAME batched
    ///   forward `verify_block` used (numerics must match the accepted argmax);
    /// - stateless + partial: no-op — the accepted-prefix KV the verify wrote is
    ///   already correct, and the rejected tail is overwritten by the next verify.
    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String>;

    /// The target's EOS token id (for the daemon's decode-loop terminator check).
    fn eos_token(&self) -> u32;

    /// The target's usable context capacity (decode-loop overflow guard).
    fn ctx_capacity(&self) -> usize;

    /// The target's KV cache, for the daemon's FlashCASK eviction — but ONLY for
    /// arches that store KV in the shared [`crate::llama::KvCache`] (qwen35
    /// `ModelSlot`, llama `LlamaBundle`). Arches with their own KV representation
    /// (e.g. qwen2's `Qwen2State`) return `None` (the default): they don't support
    /// eviction, and the daemon's eviction sites are `if let Some(ev)`-gated so
    /// this is never reached for a non-evicting target. Keeping it `Option` is
    /// what lets a `Qwen2State`-backed target implement `SpecTarget` at all
    /// (it has no `llama::KvCache` to hand back).
    fn kv_cache_mut(&mut self) -> Option<&mut crate::llama::KvCache> {
        None
    }

    // ── DFlash drafter primitives (default no-op) ───────────────────────────
    //
    // These let a hidden-conditioned drafter (DFlash / EAGLE) be built on top of
    // ANY dense-attention target without arch-specific coupling. The default
    // implementations return `None` / `Err` so that `build_speculator`'s DFlash
    // arm declines gracefully on targets that don't expose hidden states (e.g.
    // minimax, cohere2moe). A target that DOES expose them (llama, qwen3) overrides
    // both `dflash_extract_layers` (returning the layer ids) and captures hidden
    // rows into `hidden_out` inside `spec_advance` / `verify_block` (Task 2b).

    /// The layer indices whose residual hidden states the drafter wants captured,
    /// in order. When `Some(layers)`, the target should, on each processed
    /// position, append `layers.len() × dim` f32 values to the `hidden_out` sink
    /// passed to [`spec_advance`](Self::spec_advance) / [`verify_block`](Self::verify_block).
    ///
    /// `None` (default) means this target cannot feed a hidden-conditioned drafter.
    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        None
    }

    /// Apply the target's lm_head to `n` rows of residual hidden states
    /// (shape `[n, dim]`, stored row-major) on `gpu`, returning `n × vocab`
    /// host-side logits (not argmax), so the caller has access to the full
    /// distribution for sampling. Returns `Err` by default.
    fn lm_head_logits(
        &mut self,
        _gpu: &mut Gpu,
        _hidden_rows: &GpuTensor,
        _n: usize,
    ) -> Result<Vec<f32>, String> {
        Err("target does not expose lm_head over hidden".into())
    }

    /// Like [`verify_block`](Self::verify_block), but returns the FULL per-position
    /// target logits (`block.len() × vocab`, row-major) instead of the per-position
    /// argmax. The caller uses the full logits to draw from the target distribution
    /// (e.g. distribution-exact sampling at temp>0).
    ///
    /// Same contract as `verify_block` otherwise: snapshots whatever
    /// [`commit_prefix`](Self::commit_prefix) needs *before* advancing, leaves
    /// target state advanced by `block.len()`, and captures per-extract-layer
    /// residual hidden into `hidden_out` when `Some` and
    /// [`dflash_extract_layers`](Self::dflash_extract_layers) is `Some`.
    ///
    /// Returns `Err` by default.
    fn verify_block_logits(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        Err("target does not expose verify_block_logits".into())
    }

    /// Like [`verify_block`](Self::verify_block), but captures the per-position
    /// extract-layer residual hidden into the caller-owned GPU buffer
    /// `hidden_gpu` (position-major `[n_pos × dflash_extract_layers().len() ×
    /// dim]` F32) instead of a host `Vec` — keeping the DSpark accepted-prefix-
    /// hidden reuse entirely on-device (no D2H+H2D per window; ~free on UMA, a
    /// real saving on a discrete-VRAM GPU).
    ///
    /// Returns `(per-position argmax, captured)`. `captured` is `true` iff all
    /// `block.len()` positions' hidden were written to `hidden_gpu`; a target
    /// whose batched capture path can't run for this block (e.g. llama with
    /// `block.len() < 4`) returns `false` and leaves `hidden_gpu` untouched, and
    /// the caller re-bootstraps the next window. `hidden_gpu` must be
    /// `≥ block.len() × dflash_extract_layers().len() × dim` F32.
    ///
    /// Same snapshot/advance contract as [`verify_block`](Self::verify_block).
    /// Returns `Err` by default (target has no GPU-resident hidden capture).
    #[allow(clippy::too_many_arguments)]
    fn verify_block_capture_gpu(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        Err("target does not expose verify_block_capture_gpu".into())
    }

    /// Sampled (temp>0) counterpart of
    /// [`verify_block_capture_gpu`](Self::verify_block_capture_gpu): the target
    /// SAMPLES `t_i ~ p_T(temp, top_p, top_k)` per position (advancing `rng`)
    /// instead of taking argmax, still capturing the per-position hidden into
    /// `hidden_gpu`. Returns `(per-position sampled tokens, captured)`.
    ///
    /// For a point-mass drafter, `accept_greedy_prefix(drafts, picks)` on these
    /// sampled `picks` is exactly distribution-preserving temp-T speculation (the
    /// committed token is always the target sample). Same contract/`captured`
    /// semantics as the greedy variant. Returns `Err` by default.
    #[allow(clippy::too_many_arguments)]
    fn verify_block_sampled_capture_gpu(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
        _temp: f32,
        _top_p: f32,
        _top_k: usize,
        _cactus_delta: f32,
        _rng_state: &mut u64,
        _hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        Err("target does not expose verify_block_sampled_capture_gpu".into())
    }

    /// Single-pass TREE-masked verify: run the target over a linearized draft tree
    /// in ONE batched forward and return the FULL per-node target logits
    /// (`tokens.len() × vocab`, row-major).
    ///
    /// `tokens` is the slot-ordered token sequence (slot 0 = seed). `mask_block`
    /// is the `[n × n]` additive `0.0`/`-inf` ancestor-visibility bias that encodes
    /// the tree topology: a node's logits equal a causal verify of that node's
    /// root-to-node chain. `depth_positions` are the per-slot RoPE positions
    /// (`position + node.depth`); the target rotates Q/K at these positions so
    /// parent→child distance is 1, while KV writes and the mask stay on contiguous
    /// slots. The forward runs at contiguous positions `[position .. position + n)`.
    ///
    /// `hidden_out` captures per-extract-layer residual rows (same contract as
    /// [`verify_block`](Self::verify_block) / [`dflash_extract_layers`](Self::dflash_extract_layers)).
    ///
    /// Leaves target state advanced by `n`; for stateless targets
    /// [`commit_prefix`](Self::commit_prefix) is a no-op. Returns `Err` by default.
    #[allow(clippy::too_many_arguments)]
    fn verify_tree_logits(
        &mut self,
        _gpu: &mut Gpu,
        _tokens: &[u32],
        _mask_block: &[f32],
        _depth_positions: &[i32],
        _position: usize,
        _scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        Err("target does not expose verify_tree_logits".into())
    }

    /// Look up the target's embedding row for `token_id`, dequantized to F32
    /// (length `dim`). Used by hidden-conditioned drafters to obtain the noise
    /// embedding broadcast across masked block positions. Returns `Err` by default.
    fn embed_row(&mut self, _gpu: &mut Gpu, _token_id: u32) -> Result<Vec<f32>, String> {
        Err("target does not expose embed_row".into())
    }

    /// Configure which residual-hidden layer indices the target captures into
    /// the `hidden_out` sink of [`spec_advance`](Self::spec_advance) /
    /// [`verify_block`](Self::verify_block). Called by a hidden-conditioned drafter
    /// at build time so capture indices match its `fc` expectation. Default no-op
    /// for targets that do not expose hidden states (their [`dflash_extract_layers`](Self::dflash_extract_layers)
    /// stays `None`).
    fn set_dflash_extract_layers(&mut self, _layers: Vec<usize>) {}

    /// Capture the target's residual hidden states at `layers` for a freshly-
    /// committed `seed` token at absolute `position`, returning the concatenated
    /// `[layers.len() * hidden]` F32 vector (the DSpark `main_hidden`).
    ///
    /// The generic [`crate::dspark_core::DsparkDrafter`] calls this once per
    /// window (the "bootstrap" forward) to materialise the seed's hidden before
    /// the DSpark draft block runs. The target runs a 1-token forward with capture
    /// armed at `layers`, assembles the concat, and returns it. The generic drafter
    /// then uploads it to GPU for [`crate::dspark_core::DsparkBody::draft_block`].
    ///
    /// Default: returns `Err` (unsupported). Targets that provide a DSpark body
    /// (deepseek4, qwen3) override this in Tasks 5 / 9.
    fn capture_seed_main_hidden(
        &mut self,
        _gpu: &mut Gpu,
        _seed: u32,
        _position: usize,
        _layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        Err("capture_seed_main_hidden: target does not support DSpark capture".to_string())
    }
}

/// RAII borrow of the spec-decode target as `&mut dyn SpecTarget`, dispatched
/// per-arch by the loader's `spec_target_guard()`.
///
/// The guard restores any moved-out model state on Drop — the qwen35 arm moves
/// its bundle out of `ModelState`, reopens its `HfqFile`, and rebuilds the bundle
/// on *every* exit path (return / `?` / panic), which is what structurally
/// eliminates the #462 cross-request state-bleed class; a pure-attention arm
/// borrows its bundle in place (no reopen needed). `generate_dflash` only ever
/// sees `Box<dyn SpecTargetGuard>` and never learns which arch it drives — this
/// folds the old hand-written `SpecSlotGuard` enum into a single trait object.
pub trait SpecTargetGuard {
    /// Borrow the target. The qwen35 arm opens its `HfqFile` lazily on first
    /// call (an AR-only caller never pays the mmap); a reopen failure returns
    /// `Err` with the bundle still parked for `Drop` to restore, so `m.state` is
    /// never left `None`.
    fn slot(&mut self) -> Result<&mut dyn SpecTarget, String>;
}

/// In-place spec-target borrow for arches whose bundle *is* a [`SpecTarget`]
/// directly — the pure-attention family (LLaMA / plain Qwen3, Qwen2, DeepSeek V4).
/// Unlike qwen35 there is no `HfqFile` to reopen and nothing to move out of
/// `ModelState`: the `&mut` is held for the guard's life and released on `Drop`
/// like any borrow, with no restore step. Generic over the bundle type so this
/// single impl replaces the per-arch `LlamaSlotGuard` / `Qwen2SlotGuard` /
/// `Deepseek4SlotGuard` (which differed only by bundle type).
pub struct InPlaceGuard<'m, B: SpecTarget> {
    pub bundle: &'m mut B,
}

impl<B: SpecTarget> SpecTargetGuard for InPlaceGuard<'_, B> {
    fn slot(&mut self) -> Result<&mut dyn SpecTarget, String> {
        Ok(&mut *self.bundle as &mut dyn SpecTarget)
    }
}

/// Erased, arch-specific verify scratch owned by a model-free speculator.
///
/// The concrete scratch (qwen35: `VerifyScratch` + `DeltaNetSnapshot` + s_ef
/// backup + hidden ring; llama: lm_head/argmax buffers) is crate-local to each
/// arch and this crate depends on none of them — so it is threaded through the
/// trait erased, mirroring [`SpecGrammar`]. The arch's [`SpecTarget`] impl
/// recovers it via `scratch.as_any_mut().downcast_mut::<T>()`.
pub trait SpecScratch: Send {
    /// Downcast hook for the owning [`SpecTarget`] impl.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Release all GPU buffers the scratch owns. Explicit because `GpuTensor` /
    /// `DeviceBuffer` have no `Drop` in this codebase — a bare `drop` of the box
    /// would orphan device memory. Called from the speculator's `free`.
    fn free(self: Box<Self>, gpu: &mut Gpu);
}

/// Outcome of [`SpecTarget::spec_advance`].
#[derive(Debug, Clone)]
pub enum SpecAdvance {
    /// Advanced to the end. `last_argmax` is the greedy token at the final
    /// position (the first decode seed on prefill when sampling is disabled;
    /// ignored on replay). `last_logits`, when `Some`, is the host-side vocab
    /// row already materialized for that position so a sampling-capable
    /// speculator can draw the first token without a second D2H / lm_head pass.
    /// Targets that only expose a GPU argmax leave it `None`.
    Ready {
        last_argmax: u32,
        last_logits: Option<Vec<f32>>,
    },
    /// Client cancelled mid-advance; the target reset its own state.
    Aborted,
}

/// Erased grammar-mask interface for tool-call-constrained spec-decode.
///
/// The concrete per-arch grammar `Matcher` types
/// (`hipfire_arch_qwen35::grammar`, `hipfire_arch_deepseek4::grammar`) are
/// crate-local and distinct, and this crate depends on neither — so grammar is
/// threaded through the trait as an erased `&mut dyn SpecGrammar`, not a shared
/// concrete struct (a shared struct would invert the crate dependency graph; an
/// associated type would break `Box<dyn Speculator>`). Marker for now; the
/// mask-fill / accept method set will be defined when a grammar-consuming
/// drafter (MTP / EAGLE) first needs it. `DflashSpeculator` ignores grammar —
/// qwen35 enforces tool-call grammar post-hoc in the daemon.
///
/// `as_any_mut` lets an in-step grammar consumer (the deepseek4 MTP drafter)
/// downcast the erased handle back to its concrete arch grammar type
/// (`hipfire_arch_deepseek4::mtp_speculator::Deepseek4SpecGrammar`) to reach the
/// `Matcher` + decoded-vocab + mask the fused grammar step needs.
pub trait SpecGrammar {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Outcome of [`Speculator::prefill`].
#[derive(Debug, Clone)]
pub enum PrefillOutcome {
    /// Prompt prefilled; `first_token` is the target's next-token draw at the
    /// last prompt position (the seed for the first decode window). At temp≈0
    /// this is the historical argmax; at temp>0 a sampling-capable speculator
    /// MUST draw from the target distribution configured via [`Speculator::set_sampling`].
    Ready { first_token: u32 },
    /// Client cancelled mid-prefill. The caller resets conversation state and
    /// emits the aborted/done events; the slot guard restores the target bundle.
    Aborted,
}

/// Eviction-retain descriptor for [`Speculator::on_evict`] — lets the drafter
/// compact its cached target-hidden rows to match the target KV after a
/// FlashCASK eviction the daemon already applied to the target.
#[derive(Debug, Clone)]
pub struct EvictRetain {
    /// Per-physical-slot retain mask from the eviction policy.
    pub retain_mask: Vec<u32>,
    /// Physical fill before the eviction (rows to compact).
    pub pre_phys: usize,
}

/// Live post-reset evidence for serve-fault-inject / parity snapshots.
///
/// Produced only by Speculators that own DFlash-style scratch. Snapshot writers
/// MUST treat missing evidence (`None` from [`Speculator::reset_state_evidence`])
/// as fail-closed dirty — never invent clean from host vestigial checkpoint rings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecResetEvidence {
    /// Drafter-local scratch is empty (thlog watermarks zero; host shadow empty).
    pub drafter_reset: bool,
    /// Divergent-render / live checkpoint ring is empty.
    pub checkpoint_empty: bool,
}

/// A speculative-decode drafter+verifier, owned by the loaded model behind a
/// `Box<dyn Speculator>`. The daemon's decode loop holds `&mut dyn Speculator`
/// and is agnostic to whether the impl is a DFlash chain, a DDTree tree, an MTP
/// head, or a future n-gram / EAGLE drafter — chain-vs-tree, K, budget, and
/// topk are all resolved at build time and stored inside the impl.
pub trait Speculator: Send {
    /// Prefill the prompt: seed the target's hidden state (advancing its KV +
    /// recurrent state) and prime the drafter's cached target-hidden buffer,
    /// returning the target's first token. `prefill_tokens` is the suffix to
    /// seed on a cache hit (from `prefill_start`) or the full prompt on a miss;
    /// `prompt_tokens` is the full rendered prompt (used to size the drafter
    /// cursor). `resume_from`, when set, drops the drafter projection cursor to a
    /// divergent-render checkpoint position.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String>;

    /// Whether this speculator's verify is distribution-correct at temp>0 (so the
    /// daemon may route temp>0 requests through it for the spec speedup). Default
    /// `false` — greedy-only drafters (n-gram, chain DFlash, MTP) keep temp>0 on
    /// the AR sampler. The qwen35 DFlash ddtree path overrides this to `true`
    /// (its SWOR verify samples the target distribution exactly).
    fn supports_temp_verify(&self) -> bool {
        false
    }

    /// Short, human-readable drafter identity for per-request debug output
    /// (e.g. "dspark", "mtp", "dflash", "ngram"). The daemon logs this at
    /// request end so it's unambiguous which drafter actually ran (several
    /// drafters share the same generate function). Default "spec".
    fn name(&self) -> &'static str {
        "spec"
    }

    /// Run one acceptance window starting from `seed` at absolute `position`.
    /// `target` is the borrowed verifier; `emitted` is the prior committed
    /// tokens (repeat-penalty / n-gram context); `grammar` constrains both the
    /// draft and verify logits (`None` = unconstrained). `temp` is the request
    /// sampling temperature — ignored by greedy-only drafters; the ddtree path
    /// uses it to switch the verify into distribution-preserving SWOR at temp>0.
    ///
    /// `max_emit` is the remaining client-visible output budget (`max_tokens -
    /// generated`). Implementations MUST NOT verify/commit more tokens than this
    /// budget can absorb: shrink the draft window (and accept walk) so
    /// `emit.len() <= max_emit` before GPU/KV/drafter state advances past that
    /// prefix. Callers pass `usize::MAX` when uncapped (benches).
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String>;

    /// Advance an emitter-injected continuation after a completed speculative
    /// window. The default keeps target state correct for model-free drafters;
    /// hidden-conditioned drafters override it to advance their target-hidden
    /// cache and projection cursor in lockstep with the target.
    fn advance_forced(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        position: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        target.spec_advance(gpu, tokens, position, false, abort, None)
    }

    /// Compact drafter-local cached state after a target KV eviction the daemon
    /// already applied. Default no-op for drafters with no target-hidden cache.
    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        let _ = (gpu, retain);
        Ok(())
    }

    /// Advance the target over a pending-seed GPU transaction (forced
    /// continuation or terminal flush) while keeping drafter-local per-position
    /// state in sync.
    ///
    /// Daemon callers pass the **commit** slice from `SpecPendingSeedTx`:
    /// - forced suffix: optional current pending seed (omitted when
    ///   non-committable, e.g. DS4 empty-event EOS) then `forced[..n-1]`
    ///   (last forced stays the unprocessed next seed);
    /// - terminal flush: `[pending_seed]` exactly once before host bake.
    ///
    /// The plain [`SpecTarget::spec_advance`] moves KV + recurrent state only and
    /// performs no hidden extraction, so a drafter that caches one row of target
    /// hidden state per position would be left with an UNWRITTEN hole at the
    /// forced positions — uninitialized memory, i.e. NaN, which poisons every
    /// later draft forward and silently collapses acceptance to zero for the rest
    /// of the session (it also survives a prompt-cache HIT).
    ///
    /// Returns `true` when the speculator advanced the target itself (the caller
    /// must then NOT also call `spec_advance` — the recurrent state would advance
    /// twice). Default returns `false`: no drafter-local per-position state, so
    /// the caller's plain advance is correct and this is byte-identical to the
    /// pre-hook behavior.
    fn on_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        let _ = (gpu, target, tokens, start_pos, abort);
        Ok(false)
    }

    /// Rewind drafter-LOCAL state for a fresh conversation. The target's KV /
    /// recurrent state is the daemon's concern (it owns the bundle); this clears
    /// only the drafter's own scratch + checkpoint ring.
    ///
    /// Returns `Err` when any HIP free/memset that must succeed for a clean
    /// drafter fails so production rollback can attest `rolled_back:false`.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String>;

    /// Live post-reset evidence for serve-fault-inject snapshots.
    ///
    /// `None` means the live Speculator does not expose DFlash-style evidence
    /// (or it is unavailable) — snapshot writers MUST fail closed
    /// (`drafter_reset`/`checkpoint_empty` = false) rather than inventing
    /// clean from host vestigial rings. DFlash overrides with truthful
    /// thlog + checkpoint-ring probes.
    fn reset_state_evidence(&self) -> Option<SpecResetEvidence> {
        None
    }

    /// Fallible reset used by model-lifetime reset owners. The legacy
    /// [`Self::reset`] hook remains for daemon compatibility; stateful
    /// speculators override this when GPU teardown can fail.
    fn reset_checked(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)?;
        Ok(())
    }

    /// Snapshot drafter-local recurrent state at `position` for divergent-render
    /// prompt-cache reuse. Default no-op for stateless drafters (n-gram).
    fn checkpoint(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<(), String> {
        let _ = (gpu, target, position);
        Ok(())
    }

    /// Restore drafter-local state to the nearest checkpoint `<= position`,
    /// returning the position actually restored to. Default no-op (returns
    /// `position`) for stateless drafters.
    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        let _ = (gpu, target);
        Ok(position)
    }

    /// The drafter's speculation window (DFlash block size / MTP K). The daemon
    /// uses it for capacity checks and the decode-loop overflow guard.
    fn block_size(&self) -> usize;

    /// The target's usable context capacity (for the loop overflow guard).
    fn ctx_capacity(&self) -> usize;

    /// Divergent-render checkpoint positions (ascending), for prompt-cache
    /// resume planning. Default empty for drafters with no checkpoint ring.
    fn checkpoint_positions(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Release all GPU buffers the drafter owns. Called from `unload_model`,
    /// so a drafter that forgets to free is a missing-trait-method compile
    /// error rather than a silent VRAM leak.
    fn free(self: Box<Self>, gpu: &mut Gpu);

    /// Whether this drafter requires greedy verification (temperature 0).
    /// A greedy-only drafter (n-gram chain, the MTP-via-`Speculator` wrapper)
    /// returns `true`; a sampling-capable drafter (DFlash with lossless rejection
    /// sampling) returns `false`. The daemon dispatch consults this per request:
    /// a temp>0 request may take the spec path only against a drafter that returns
    /// `false`; against a `true` drafter it falls through to AR rather than being
    /// silently decoded greedy.
    fn requires_greedy(&self) -> bool {
        true
    }

    /// Configure per-request sampling for the next acceptance window(s). Called
    /// by the daemon's spec wrapper (`generate_dflash`) once before the step
    /// loop, threading the request's resolved temp/top_p/top_k/cactus down to
    /// the drafter. Default no-op: a greedy-only drafter ignores it and keeps
    /// decoding at argmax. A sampling-capable drafter (DFlash) stores these and
    /// applies the IDENTICAL (top_k,top_p) nucleus truncation to draft + target
    /// inside `step` (lossless == AR-at-(top_k,top_p)). `temp <= 0` ⇒ greedy.
    fn set_sampling(&mut self, _temp: f32, _top_p: f32, _top_k: usize, _cactus_delta: f32) {}
}

// ─── Multi-token-prediction (MTP) drafter core ──────────────────────────────
//
// Every MTP drafter (qwen35 MTP head, deepseek4 MTP layer) shares one shape: a
// prompt prefill that primes the arch's MTP history + recurrent/KV state and
// returns a greedy seed, then a per-window draft+verify+accept step returning
// the committed tail (accepted prefix + bonus, seed excluded). The arches differ
// only in the fused kernels they run — that difference is [`MtpDrafter`], and
// [`MtpSpeculator`] adapts any `MtpDrafter` to the generic [`Speculator`]
// interface (prefill→`PrefillOutcome`, window→`SpecStep`) ONCE, so a new MTP arch
// implements only `MtpDrafter` (+ `SpecTarget`), never a whole `Speculator`.

/// One acceptance window's committed tokens: the accepted draft prefix plus the
/// verifier's bonus, EXCLUDING the seed. Identical in meaning to qwen35's
/// `MtpSpecResult.committed` and deepseek4's `accepted_tokens`.
#[derive(Debug, Clone)]
pub struct MtpWindow {
    /// Tokens committed this window (accepted drafts + bonus, seed excluded).
    /// `committed.len()` is what drives the daemon's `position += emit.len()`.
    pub committed: Vec<u32>,
    /// Drafts accepted, excluding the bonus (τ numerator).
    pub accepted: usize,
    /// Drafts offered this window, ≤ `k` (τ denominator).
    pub drafts_generated: usize,
}

/// Per-arch MTP draft+verify core. The impl owns its draft scratch (qwen35:
/// `MtpSpecState` + head; deepseek4: the relocated `PrefillBatchScratch`) and
/// downcasts `target` (`&mut dyn SpecTarget`) to its concrete model type to run
/// the fused head-draft + trunk-verify kernels. Everything arch-INvariant
/// (prefill outcome, window→step lowering, position/seed advance) lives once in
/// [`MtpSpeculator`].
pub trait MtpDrafter: Send {
    /// Prefill `fill_tokens` from absolute `start_pos`: advance the target's
    /// KV/recurrent state AND the MTP head's position-aligned cache, returning
    /// the greedy seed (argmax at the last prefilled position). The central
    /// speculative lifecycle performs the authoritative total reset before a
    /// `cache_hit=false` call; implementations only choose full versus suffix
    /// tokens here and must not reset target state again.
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String>;

    /// Abort-aware prefill entry point. Existing drafters keep their original
    /// API and behavior through this default; drafters with interruptible
    /// target work override it to pass the callback into that work.
    fn mtp_prefill_abortable(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: Option<&dyn Fn() -> bool>,
    ) -> Result<Option<u32>, String> {
        if abort.is_some_and(|check| check()) {
            return Ok(None);
        }
        let first_token = self.mtp_prefill(gpu, target, fill_tokens, start_pos, cache_hit)?;
        if abort.is_some_and(|check| check()) {
            Ok(None)
        } else {
            Ok(Some(first_token))
        }
    }

    /// One acceptance window: seed at absolute `position`; draft up to `k`,
    /// verify, greedily accept the longest matching prefix + bonus. `k` is the
    /// per-call draft budget from [`MtpSpeculator::step`] (already clamped by
    /// remaining `max_emit`); implementations MUST honor it and MUST NOT draft
    /// or commit more than `k` candidates (emit ≤ k+1 including bonus). `k == 0`
    /// means verify `[seed]` only and emit the single bonus token.
    /// `grammar` is the IN-STEP grammar (deepseek4 masks draft+verify logits
    /// with it; qwen35 ignores it and relies on post-hoc grammar in the
    /// emission layer).
    #[allow(clippy::too_many_arguments)]
    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String>;

    /// Advance drafter-local MTP state over emitter-forced tokens so the head
    /// KV / prev_hidden / multi-slot context stay aligned with the target.
    /// Returns `true` when the drafter advanced the target itself (caller must
    /// not also `spec_advance`). Default: no drafter-local position state.
    fn mtp_forced_advance(
        &mut self,
        _gpu: &mut Gpu,
        _target: &mut dyn SpecTarget,
        _tokens: &[u32],
        _start_pos: usize,
        _abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        Ok(false)
    }

    /// Reset drafter-local state for a fresh conversation (MTP cache + any
    /// captured graphs). The target's KV/recurrent reset is the daemon's job.
    /// Returns `Err` when any HIP step required for a clean drafter fails.
    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String>;

    /// Fallible MTP reset used by model-owned context reset. Legacy MTP
    /// implementations remain source-compatible through the default wrapper.
    fn mtp_reset_checked(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.mtp_reset(gpu);
        Ok(())
    }

    /// Release all GPU buffers the drafter owns.
    fn mtp_free(self: Box<Self>, gpu: &mut Gpu);

    /// Draft window size (K).
    fn k(&self) -> usize;

    /// Target context capacity (for the loop overflow guard).
    fn ctx_capacity(&self) -> usize;

    /// Whether verification is greedy-only (temp≈0). qwen35 MTP → `true`.
    fn requires_greedy(&self) -> bool;

    /// Short drafter identity surfaced by [`MtpSpeculator::name`] for per-request
    /// debug output. Plain MTP drafters keep the default "mtp"; the DSpark
    /// drafter overrides to "dspark".
    fn name(&self) -> &'static str {
        "mtp"
    }

    /// Stash the request sampling params for the next `mtp_step`. The
    /// [`MtpSpeculator`] forwards `set_sampling` + the per-step `temp` here so a
    /// temp>0-capable drafter (DSpark) can drive a sampled verify. Default no-op
    /// (greedy-only drafters ignore it). `top_p==0` means "disabled" (→ 1.0).
    fn set_sampling(&mut self, _temp: f32, _top_p: f32, _top_k: usize, _cactus_delta: f32) {}

    /// Whether this drafter's verify is distribution-correct at temp>0 (so the
    /// daemon may route temp>0 requests through it). Default `false`.
    fn supports_temp_verify(&self) -> bool {
        false
    }
}

/// Generic adapter driving any [`MtpDrafter`] through the [`Speculator`]
/// interface. One impl serves every MTP arch; the arch-specific work is the
/// `MtpDrafter` (+ `SpecTarget`) impl in the arch crate.
pub struct MtpSpeculator<A: MtpDrafter> {
    arch: A,
    /// Request sampling (top_p/top_k from `set_sampling`, temp from `step`).
    /// Forwarded to the drafter via `arch.set_sampling` before each `mtp_step`
    /// so a temp>0-capable drafter (DSpark) can sample its verify. `top_p==0`
    /// means disabled; greedy drafters ignore all three.
    top_p: f32,
    top_k: usize,
    /// CACTUS acceptance-boost δ (0 = lossless). Forwarded to the drafter with
    /// top_p/top_k; only a CACTUS-capable sampled verify (deepseek4 DSpark) uses it.
    cactus: f32,
}

impl<A: MtpDrafter> MtpSpeculator<A> {
    pub fn new(arch: A) -> Self {
        Self {
            arch,
            top_p: 1.0,
            top_k: 0,
            cactus: 0.0,
        }
    }
}

/// Lower an [`MtpWindow`] to the generic [`SpecStep`]. `committed` already
/// excludes the seed and includes the bonus, so it maps 1:1 to `emit`; the next
/// window's seed is the last committed token (the daemon's `position +=
/// emit.len()` / `seed = next_seed` contract). An empty `committed` would stall
/// the loop (no position/`generated` advance) — surface it as an error so the
/// loop breaks instead of spinning.
fn lower_mtp_window(w: MtpWindow) -> Result<SpecStep, String> {
    let next_seed = *w
        .committed
        .last()
        .ok_or("MtpSpeculator: drafter committed 0 tokens (would stall the decode loop)")?;
    Ok(SpecStep::new(
        w.committed.iter().copied(),
        next_seed,
        w.drafts_generated,
        w.accepted,
    ))
}

/// Per-call draft count from remaining client budget. `max_emit == 0` is rejected
/// by the caller before this helper runs. emit = accepted drafts + bonus, so we
/// propose at most `max_emit - 1` drafts (`max_emit == 1` → k=0 → verify seed only).
fn mtp_draft_k(arch_k: usize, max_emit: usize) -> usize {
    max_emit.saturating_sub(1).min(arch_k)
}

impl<A: MtpDrafter> Speculator for MtpSpeculator<A> {
    fn name(&self) -> &'static str {
        self.arch.name()
    }

    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        _resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        if abort() {
            return Ok(PrefillOutcome::Aborted);
        }
        // Cache hit ⇒ warm suffix prefill from `prefill_start`; miss ⇒ full
        // prefill from 0 (the drafter resets recurrent + MTP cache on miss).
        let (fill_tokens, start_pos): (&[u32], usize) = if cache_hit {
            (prefill_tokens, prefill_start)
        } else {
            (prompt_tokens, 0)
        };
        let first_token = self.arch.mtp_prefill_abortable(
            gpu,
            target,
            fill_tokens,
            start_pos,
            cache_hit,
            Some(abort),
        )?;
        match first_token {
            Some(first_token) => Ok(PrefillOutcome::Ready { first_token }),
            None => Ok(PrefillOutcome::Aborted),
        }
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        // Remaining budget caps how many tokens this window may emit
        // (accepted drafts + bonus). Reject 0 before any trunk/head/KV mutation.
        // k drafts ⇒ emit ≤ k+1; clamp k so drafts + bonus ≤ max_emit.
        // max_emit == 1 → k = 0 → verify [seed] only, commit the single bonus.
        if max_emit == 0 {
            return Err("MtpSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        let k = mtp_draft_k(self.arch.k(), max_emit);
        let eos = target.eos_token();
        // Forward the per-step temp + the request top_p/top_k (stashed by
        // `set_sampling`) to the drafter. Greedy-only drafters ignore it; a
        // temp>0-capable one (DSpark) uses it to sample its verify.
        self.arch
            .set_sampling(temp, self.top_p, self.top_k, self.cactus);
        let window = self
            .arch
            .mtp_step(gpu, target, position, seed, k, eos, grammar)?;
        // Pre-commit budget is authoritative; cap_emit is defensive only.
        let step = lower_mtp_window(window)?;
        debug_assert!(
            step.emit.len() <= max_emit,
            "MtpSpeculator: drafter committed past max_emit (emit={}, max_emit={})",
            step.emit.len(),
            max_emit
        );
        Ok(step.cap_emit(max_emit))
    }

    fn on_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        self.arch
            .mtp_forced_advance(gpu, target, tokens, start_pos, abort)
    }

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.arch.mtp_reset(gpu)
    }

    fn reset_checked(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.arch.mtp_reset_checked(gpu)
    }

    fn block_size(&self) -> usize {
        self.arch.k()
    }

    fn ctx_capacity(&self) -> usize {
        self.arch.ctx_capacity()
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // Move the drafter out of the box and hand it its own boxed-self free.
        Box::new(self.arch).mtp_free(gpu);
    }

    fn requires_greedy(&self) -> bool {
        self.arch.requires_greedy()
    }

    fn supports_temp_verify(&self) -> bool {
        self.arch.supports_temp_verify()
    }

    fn set_sampling(&mut self, _temp: f32, top_p: f32, top_k: usize, cactus_delta: f32) {
        // Stash top_p/top_k/cactus; temp arrives per-step via `step`. Forwarded to
        // the drafter inside `step` (before `mtp_step`).
        self.top_p = top_p;
        self.top_k = top_k;
        self.cactus = cactus_delta;
    }
}

// ─── Per-token emission seam (SpecEmit) ─────────────────────────────────────
//
// The daemon's spec-decode loops emit committed tokens twice over: qwen35/llama
// in `generate_dflash` (an `EosFilter` byte stream + post-hoc grammar matcher +
// max-think force-close + user stop-sequence match), and deepseek4 in
// `generate_deepseek4` (a DSML `StreamParser`). `SpecEmit` is the arch-generic
// boundary between "the loop committed token N" and "what the client sees" so a
// future single decode loop can drive any arch's emission without learning its
// quirks. The trait returns SEMANTIC events ([`ClientEvent`]) — the daemon owns
// the JSONL rendering and timing, and the loop-state bookkeeping (KV/recurrent
// resets, position/seed advance, conversation-token bake, eviction) stays in the
// decode loop; only the per-token emit decisions move behind this seam.

/// One client-visible emission produced by a [`SpecEmit`] step. The daemon
/// renders these to its JSONL wire format (`{"type":"token",...}` etc.) and
/// supplies its own timing; this carries only the semantic payload.
#[derive(Debug, Clone)]
pub enum ClientEvent {
    /// Visible answer text (post-filter, post-think-strip). → `{"type":"token"}`.
    Token(String),
    /// Reasoning/think-block text surfaced separately (DSML reasoning channel).
    /// → `{"type":"reasoning"}`. Unused by the qwen35 emitter (it strips think
    /// only when configured and never emits a separate reasoning channel), but
    /// part of the shared vocabulary the deepseek4 emitter will populate.
    Reasoning(String),
    /// Parsed tool calls for this turn. → `{"type":"tool_calls"}`.
    ToolCalls(Vec<crate::prompt_frame::ToolCall>),
    /// A committed token id at output index `idx`. → `{"type":"committed"}`
    /// (gated by `HIPFIRE_EMIT_TOKEN_IDS=1` at the daemon). The daemon attaches
    /// its own `t_ms` timestamp when rendering.
    Committed { id: u32, idx: usize },
}

/// Why a [`SpecEmit`] decided this token ends generation. The daemon maps this
/// to its `done` envelope's `finish_reason` and to the post-loop cleanup
/// (grammar-violation forces a full KV/recurrent reset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A natural end-of-turn terminator (EOS / `<|im_end|>` / tokenizer
    /// terminator). `finish_reason` resolves to `stop` (or `tool_calls` when the
    /// turn parsed any).
    Eos,
    /// `max_think_tokens` reached inside an open `<think>` block — the emitter
    /// force-closed the block. `finish_reason` = `stop`.
    ThinkCap,
    /// A user-supplied stop sequence matched the decoded suffix. `finish_reason`
    /// = `stop`.
    StopSequence,
    /// A committed token violated the active tool-call grammar. The daemon
    /// treats this as EOS for the turn AND forces a full KV/recurrent reset so
    /// the next turn starts clean.
    GrammarViolation,
}

/// Outcome of a single [`SpecEmit::begin`] / [`SpecEmit::observe`] step: the
/// client events to render (in order) plus an optional stop signal. A non-`None`
/// `stop` means the decode loop must stop AFTER rendering `events`.
#[derive(Debug, Clone, Default)]
pub struct EmitOutcome {
    /// Events to render this step, in order.
    pub events: Vec<ClientEvent>,
    /// Whether this outcome admitted a client-visible generation token.  This
    /// is deliberately explicit: diagnostics/events are not a generation
    /// counter (terminal and held tokens may have either).
    pub generation_advanced: bool,
    /// If set, generation stops after this step's events.
    pub stop: Option<StopReason>,
}

impl EmitOutcome {
    /// An empty outcome (filter held all bytes; no stop). The common case when
    /// the `EosFilter` is buffering a partial UTF-8 codepoint or marker prefix.
    pub fn held() -> Self {
        Self::default()
    }
}

/// Terminal flush of a [`SpecEmit`], consumed by value at end of turn. Carries
/// any final events (e.g. a held `tool_calls` event) plus the resolved
/// `finish_reason` and parsed tool-call count for the daemon's `done` envelope.
/// The daemon still owns the length-cap decision (`generated >= max_tokens`
/// without decoded EOT ⇒ `length`), so `finish_reason` here is the emitter's
/// provisional view (`stop` / `tool_calls` / `malformed_protocol` /
/// `open_think`); the caller may override with `length` when the loop hit the
/// token cap without EOT.
///
/// Structured `ClientEvent::ToolCalls` on [`Self::events`] are **held** by
/// `generate_spec` (`hold_tool_calls=true`) until the arch wrapper classifies a
/// tool-safe terminal.
#[derive(Debug, Clone, Default)]
pub struct FinishSummary {
    /// Final events to render (e.g. trailing visible Token + held ToolCalls).
    pub events: Vec<ClientEvent>,
    /// The emitter's finish reason (`stop`, `tool_calls`, `malformed_protocol`,
    /// or `open_think`); the caller may override with `length` on a pure
    /// token-cap exit.
    pub finish_reason: &'static str,
    /// Number of tool calls parsed this turn (for the `done` envelope).
    /// Zero when malformed or when calls must stay suppressed.
    pub tool_calls: usize,
    /// The sealed transcript, when this emitter owns one.  Qwen's ChatML
    /// emitter is the current cache/replay authority; other emitters retain
    /// their legacy terminal summaries until they adopt the same owner.
    pub finalized: Option<crate::spec_transcript::FinalizedAssistantTurn>,
    /// Producer-authorized visible prose accumulated this turn (think-stripped,
    /// marker-free). Used for asst-turn cache fingerprinting; empty when the
    /// emitter does not track a visible channel.
    pub visible_text: String,
    /// Emitter-authoritative decoded end-of-turn verdict (token-id EOT and/or
    /// filter stop_at spanning split byte fragments). Carried into terminal
    /// lowering so wrappers do not recompute solely from the last token id.
    pub decoded_eot: bool,
    /// Unclosed thinking region at finish — nonretryable unsafe terminal.
    pub open_think: bool,
}

/// Model-independent context for constructing a turn's [`SpecEmit`].
///
/// Built by `generate_spec` *after* it has acquired the spec target (so `eos`
/// is `slot.eos_token()`), then handed to the arch's carrier, whose
/// `make_spec_emitter` constructs the concrete `Box<dyn SpecEmit>`. Every field
/// is a model-independent request/render output — tool definitions are passed
/// as **raw JSON** (`tools`) because the two arch emitters extract different
/// schema shapes from them, and the only lossless rep both accept is the
/// original JSON. The arch carrier owns the JSON→its-own-`ToolSchema`
/// conversion, so no arch type appears here.
pub struct SpecEmitCtx<'a> {
    /// Tokenizer for decoding committed tokens to text (byte filter, grammar,
    /// think-scan, tool-call extraction).
    pub tokenizer: &'a crate::tokenizer::Tokenizer,
    /// The target's EOS token (`slot.eos_token()`), known only after slot
    /// acquisition — which is why the emitter is built here, not by the caller.
    pub eos: u32,
    /// Secondary terminator (e.g. `<|im_end|>`), if the arch uses one.
    pub im_end: Option<u32>,
    /// Raw tool definitions from the request (OpenAI-shape JSON). Each carrier
    /// extracts its own grammar `ToolSchema` from these; `None`/empty ⇒ no
    /// tool-call grammar.
    pub tools: Option<&'a [serde_json::Value]>,
    /// User stop sequences matched against the decoded suffix.
    pub stop: Vec<String>,
    /// `max_think_tokens` budget (0 ⇒ no think force-close).
    pub max_think: usize,
    /// The turn's `max_tokens` cap. Used by arches whose emitter sizes a
    /// think-token reserve against it (cohere2moe's think-budget force-close);
    /// ignored by emitters without a generation-side think guard.
    pub max_tokens: usize,
    /// Whether the prompt opened a `<think>` span via the assistant prefix.
    pub assistant_prefix: crate::prompt_frame::AssistantPrefix,
    /// Requested reasoning-effort level (arch interprets the frame mapping).
    pub think_mode: crate::prompt_frame::ThinkMode,
    /// Pre-decoded vocab for arches whose grammar masks per-token (DeepSeek V4).
    /// The daemon builds/caches this Arc before the call so the neutral ctx
    /// never has to mutate `LoadedModel`.
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
}

/// Per-token emission policy for a spec-decode turn. The decode loop calls
/// [`begin`](Self::begin) once for the prefill's first token, then
/// [`observe`](Self::observe) for each subsequently-committed token, rendering
/// the returned [`EmitOutcome::events`] and stopping when a step returns a
/// [`StopReason`]. At end of turn it calls [`finish`](Self::finish) for the
/// terminal flush.
///
/// The emitter OWNS the per-turn emission state (byte filter, grammar matcher,
/// think counter, decoded-token history) so the decode loop stays arch-agnostic.
/// It does NOT own loop/cache state (position, seed, KV/recurrent resets,
/// conversation-token bake) — those stay in the loop.
pub trait SpecEmit {
    /// Emit the prefill's first token. Returns the events to render and any
    /// immediate stop (the first token can itself be a terminator).
    fn begin(&mut self, first_token: u32) -> EmitOutcome;

    /// Emit one subsequently-committed token. Returns the events to render and
    /// any stop signal (EOS / grammar / stop-sequence / think-cap).
    fn observe(&mut self, token: u32) -> EmitOutcome;

    /// Terminal flush at end of turn (parse tool calls, resolve finish reason).
    fn finish(self: Box<Self>) -> FinishSummary;

    /// In-step grammar mask interface, when the emitter applies grammar to the
    /// verifier's logits. The qwen35 emitter applies grammar POST-hoc inside
    /// `observe` and returns `None` here; a future in-step grammar consumer
    /// returns its erased matcher.
    fn grammar(&mut self) -> Option<&mut dyn SpecGrammar> {
        None
    }

    /// The full committed-token stream (incl. the first token), for the decode
    /// loop's post-turn bookkeeping (the qwen35 asst-turn cache store). Default
    /// empty for emitters whose wrapper does no token-replay cache.
    fn streamed_tokens(&self) -> &[u32] {
        &[]
    }

    /// Whether a committed token tripped the grammar matcher — the decode loop
    /// forces a full KV/recurrent reset for the next turn when true. Default
    /// `false` for emitters without post-hoc grammar enforcement.
    fn grammar_violated(&self) -> bool {
        false
    }

    /// Emitter-authoritative decoded EOT (token id and/or filter stop marker).
    /// Default false; qwen35 overrides so terminal lowering need not re-decode.
    fn decoded_eot(&self) -> bool {
        false
    }

    /// Hint the emitter of the decode loop's current `generated` count so an
    /// attractor-detect log message can report the same number it did inline.
    /// Default no-op.
    fn set_generated_hint(&mut self, _generated: usize) {}

    /// Generation-intervention hook: tokens the emitter wants the loop to FORCE
    /// into the stream after the just-observed token, suppressing this step's
    /// terminator. The decode loop, when this returns non-empty, advances the
    /// target over each token, re-feeds it through [`observe`](Self::observe),
    /// and continues WITHOUT honoring the current step's `stop` — used by arches
    /// whose bespoke AR loop does generation-side recovery a pure emitter cannot
    /// express (e.g. cohere2moe's empty-turn guard force-injects `<|START_TEXT|>`
    /// when the model ends thinking with no visible output, and its think-budget
    /// force-close injects `<|END_THINKING|><|START_TEXT|>`). The emitter must
    /// drain the queue (return-and-clear) and bound its own re-entry (e.g. a max
    /// suppression count) so forcing terminates. Default empty ⇒ the loop never
    /// enters the force path, so every other emitter is byte-identical no-op.
    fn take_forced(&mut self) -> Vec<u32> {
        Vec::new()
    }
}

// ─── Model-free drafting sources (arch-agnostic, pure CPU) ──────────────────
//
// Moved here from `hipfire-arch-qwen35::speculative` so the arch-generic
// `NgramSpeculator` can use them without an arch-crate dependency. qwen35's
// `spec_step_dflash` still uses them via a `pub use` re-export in that crate.

/// Rolling bigram n-gram cache. Keyed by the last two committed tokens
/// `(a, b)`; value is a small map from possible next-token to count.
///
/// Populated incrementally from the committed output stream. Used as a
/// "free" second opinion on top of the DFlash draft: if the cache has
/// seen a (a, b) → c transition with high enough count, and the DFlash
/// draft proposed something else at that position, the n-gram's `c`
/// often turns out to match the target's argmax.
///
/// Scales: the cache size is bounded by the number of distinct bigrams
/// in the committed output — typically a few hundred per session, so
/// no eviction policy needed.
pub struct NgramCache {
    /// `(a, b) → { next: count, ... }` with the next-token histogram.
    pub bigram: std::collections::HashMap<(u32, u32), std::collections::HashMap<u32, u32>>,
    /// Minimum count before we trust the prediction. Smaller = more
    /// aggressive (more overrides), larger = more conservative. 3 is a
    /// reasonable default on hot-loop code / repetitive text.
    pub min_count: u32,
}

impl NgramCache {
    pub fn new(min_count: u32) -> Self {
        Self {
            bigram: std::collections::HashMap::new(),
            min_count,
        }
    }

    /// Record the triple `(a, b) → c` in the cache.
    #[inline]
    pub fn observe(&mut self, a: u32, b: u32, c: u32) {
        *self.bigram.entry((a, b)).or_default().entry(c).or_insert(0) += 1;
    }

    /// Predict `c` from last-two `(a, b)` if the max-count next-token
    /// reaches `min_count`. Returns (token, count).
    #[inline]
    pub fn predict(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let map = self.bigram.get(&(a, b))?;
        let (&tok, &cnt) = map.iter().max_by_key(|(_, &c)| c)?;
        if cnt >= self.min_count {
            Some((tok, cnt))
        } else {
            None
        }
    }

    /// Record every consecutive triple in a slice of committed tokens.
    /// Caller supplies the full token stream; this walks it in-place.
    pub fn observe_many(&mut self, tokens: &[u32]) {
        if tokens.len() >= 3 {
            for w in tokens.windows(3) {
                self.observe(w[0], w[1], w[2]);
            }
        }
    }
}

/// Prompt Lookup Decoding (Saxena 2023): training-free deterministic draft
/// built from context suffix self-match. If the last N tokens of context
/// appeared earlier in context, the tokens that followed that earlier
/// occurrence are a high-quality continuation guess.
///
/// Used as the draft source in Goose bypass mode (Jin et al. 2026,
/// arXiv:2604.02047 §4.3): PLD-matched tokens have 2–18× higher acceptance
/// than bigram (TR) tokens (median 6× across 5 models × 5 benchmarks).
/// When PLD confidence is high, the spine — a deep linear chain of
/// PLD-matched tokens — is verified in one target forward pass without
/// tree construction. That's exactly what we need on Qwen3.5 hybrid
/// (24 DeltaNet + 8 FullAttention): linear verify sidesteps the
/// state-forking problem that tree verify imposes on recurrent LA layers.
pub struct PldMatcher {
    /// n-gram suffix lengths to try, longest first. Paper uses {5,4,3}.
    /// Longer matches are more selective; if the longest fails we fall
    /// back to shorter. Order matters: we return the first (longest) hit.
    pub ngram_lens: Vec<usize>,
    /// Hard cap on spine length. Paper uses 8 — sufficient for typical
    /// block sizes and avoids running off the end of a match into drift.
    pub max_extract: usize,
    /// Minimum extracted length to count as a usable spine. Very short
    /// spines aren't worth the PLD path (bigram covers 1-token lookahead
    /// at lower risk); require at least this many continuation tokens.
    pub min_extract: usize,
}

impl Default for PldMatcher {
    fn default() -> Self {
        Self {
            ngram_lens: vec![5, 4, 3],
            max_extract: 8,
            min_extract: 3,
        }
    }
}

/// Result of a successful PLD lookup.
#[derive(Debug, Clone)]
pub struct PldMatch {
    /// The extracted spine (continuation tokens after the matched suffix).
    pub tokens: Vec<u32>,
    /// The suffix length that produced this match (the longest that hit).
    pub n: usize,
    /// Number of tried n-gram lengths that agreed on `tokens[0]`. Paper
    /// §4.3 uses this as part of the bypass-mode confidence signal;
    /// higher consensus = more reliable spine. Ranges 1..=ngram_lens.len().
    pub consensus: usize,
}

impl PldMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a spine continuation for `context`. Returns `None` if no tried
    /// n-gram length produces a match of length ≥ `self.min_extract`.
    ///
    /// For each n in `self.ngram_lens`: take the last-n tokens as the
    /// suffix, search for its last occurrence earlier in context, and
    /// extract the `max_extract` tokens that followed it (stopping before
    /// the suffix itself so we don't include tokens that would be about
    /// to be re-predicted). Returns the longest-n match with a usable
    /// spine; consensus counts how many alternate n's produced the same
    /// first continuation token.
    pub fn lookup(&self, context: &[u32]) -> Option<PldMatch> {
        if self.ngram_lens.is_empty() {
            return None;
        }
        // Per-n continuation, collected to compute consensus across lengths.
        let mut firsts: Vec<u32> = Vec::with_capacity(self.ngram_lens.len());
        let mut best: Option<(usize, Vec<u32>)> = None; // (n, spine)
        for &n in &self.ngram_lens {
            if context.len() <= n {
                continue;
            }
            let suffix_start = context.len() - n;
            let suffix = &context[suffix_start..];
            let haystack = &context[..suffix_start];
            if haystack.len() < n {
                continue;
            }
            // Last occurrence (freshest) of `suffix` in `haystack`.
            let mut found: Option<usize> = None;
            for i in (0..=haystack.len() - n).rev() {
                if &haystack[i..i + n] == suffix {
                    found = Some(i);
                    break;
                }
            }
            let start = match found {
                Some(s) => s,
                None => continue,
            };
            let cont_start = start + n;
            let cont_end = (cont_start + self.max_extract).min(suffix_start);
            if cont_end <= cont_start {
                continue;
            }
            let spine: Vec<u32> = context[cont_start..cont_end].to_vec();
            if spine.len() < self.min_extract {
                continue;
            }
            firsts.push(spine[0]);
            if best.is_none() {
                best = Some((n, spine));
            }
        }

        let (n, tokens) = best?;
        let consensus = firsts.iter().filter(|&&t| t == tokens[0]).count();
        Some(PldMatch {
            tokens,
            n,
            consensus,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Locks the load-bearing loop contract: the daemon advances `position` by
    // `emit.len()`, NOT by `accepted`. For a chain window that accepted 2 of 4
    // drafts, emit is `[d0, d1, bonus]` (len 3 = accepted + 1) and the next
    // seed is the bonus = `emit.last()`. The adversarial review certified this
    // equivalence against `speculative.rs:3737-3744`; this test pins it so a
    // future lowering change can't silently break the position math.
    #[test]
    fn emit_len_drives_advance_not_accepted() {
        let step = SpecStep {
            emit: SmallVec::from_slice(&[10, 11, 12]),
            next_seed: 12,
            proposed: 4,
            accepted: 2,
        };
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(*step.emit.last().unwrap(), step.next_seed);
    }

    // ── accept_greedy_prefix ────────────────────────────────────────────────

    #[test]
    fn accept_greedy_none_partial() {
        // drafts [11,12,13]; target picks [11,12,99,..] → accept 11,12; diverge
        // at slot 2 (99 != 13); bonus = target_pick[2] = 99.
        let r = accept_greedy_prefix(&[11, 12, 13], &[11, 12, 99, 0], None);
        assert_eq!(r.accepted, 2);
        assert_eq!(r.committed, vec![11, 12, 99]);
        assert!(!r.hit_eos);
    }

    #[test]
    fn accept_greedy_none_full() {
        // all drafts match; bonus = target_pick[3].
        let r = accept_greedy_prefix(&[11, 12, 13], &[11, 12, 13, 77], None);
        assert_eq!(r.accepted, 3);
        assert_eq!(r.committed, vec![11, 12, 13, 77]);
        assert!(!r.hit_eos);
    }

    #[test]
    fn accept_greedy_none_zero() {
        // first draft rejected; accept nothing, bonus = target_pick[0].
        let r = accept_greedy_prefix(&[11, 12], &[42, 0, 0], None);
        assert_eq!(r.accepted, 0);
        assert_eq!(r.committed, vec![42]);
        assert!(!r.hit_eos);
    }

    #[test]
    fn accept_greedy_eos_stop_midprefix() {
        // eos=2; drafts [11,2,13]; accept 11, then 2==eos → stop, NO bonus.
        let r = accept_greedy_prefix(&[11, 2, 13], &[11, 2, 13, 0], Some(2));
        assert_eq!(r.accepted, 2);
        assert_eq!(r.committed, vec![11, 2]);
        assert!(r.hit_eos);
    }

    #[test]
    fn accept_greedy_eos_as_bonus() {
        // eos=2; drafts [11] accepted; bonus = target_pick[1] = 2 == eos.
        let r = accept_greedy_prefix(&[11], &[11, 2], Some(2));
        assert_eq!(r.accepted, 1);
        assert_eq!(r.committed, vec![11, 2]);
        assert!(r.hit_eos);
    }

    // ── lower_mtp_window (MtpWindow → SpecStep) ─────────────────────────────

    #[test]
    fn lower_mtp_window_maps_committed_and_last_seed() {
        // 4 drafts offered, 2 accepted + bonus ⇒ committed = [a, b, bonus].
        let step = lower_mtp_window(MtpWindow {
            committed: vec![10, 11, 12],
            accepted: 2,
            drafts_generated: 4,
        })
        .unwrap();
        assert_eq!(step.emit.as_slice(), &[10, 11, 12]);
        assert_eq!(step.next_seed, 12); // committed.last() == bonus
        assert_eq!(step.proposed, 4);
        assert_eq!(step.accepted, 2);
        // The load-bearing loop contract still holds for the MTP lowering.
        assert_eq!(step.emit.len(), step.accepted + 1);
    }

    #[test]
    fn lower_mtp_window_single_bonus_only() {
        // 0 accepted ⇒ committed = [bonus] (still non-empty).
        let step = lower_mtp_window(MtpWindow {
            committed: vec![99],
            accepted: 0,
            drafts_generated: 4,
        })
        .unwrap();
        assert_eq!(step.emit.as_slice(), &[99]);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.accepted, 0);
    }

    #[test]
    fn lower_mtp_window_empty_is_error() {
        // An empty window would stall the daemon loop — must be an error.
        assert!(lower_mtp_window(MtpWindow {
            committed: vec![],
            accepted: 0,
            drafts_generated: 4,
        })
        .is_err());
    }

    // ── SpecEmit seam types ─────────────────────────────────────────────────

    #[test]
    fn emit_outcome_held_is_empty_no_stop() {
        let o = EmitOutcome::held();
        assert!(o.events.is_empty());
        assert!(o.stop.is_none());
    }

    #[test]
    fn spectarget_hidden_default_is_unsupported() {
        // A SpecTarget that doesn't override the DFlash hooks reports no extract
        // layers and refuses capture — so build_speculator's DFlash arm declines
        // gracefully on arches without hidden capture (e.g. minimax).
        struct Bare;
        impl SpecTarget for Bare {
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
            fn reset_recurrent(&mut self, _gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
                Ok(())
            }
            fn new_spec_scratch(
                &mut self,
                _gpu: &mut rdna_compute::Gpu,
                _block_size: usize,
            ) -> Result<Box<dyn SpecScratch>, String> {
                unimplemented!()
            }
            fn spec_advance(
                &mut self,
                _gpu: &mut rdna_compute::Gpu,
                _tokens: &[u32],
                _start_pos: usize,
                _reset: bool,
                _abort: &dyn Fn() -> bool,
                _hidden_out: Option<&mut Vec<f32>>,
            ) -> Result<SpecAdvance, String> {
                unimplemented!()
            }
            fn verify_block(
                &mut self,
                _gpu: &mut rdna_compute::Gpu,
                _block: &[u32],
                _position: usize,
                _scratch: &mut dyn SpecScratch,
                _hidden_out: Option<&mut Vec<f32>>,
            ) -> Result<Vec<u32>, String> {
                unimplemented!()
            }
            fn commit_prefix(
                &mut self,
                _gpu: &mut rdna_compute::Gpu,
                _block: &[u32],
                _accept_len: usize,
                _position: usize,
                _scratch: &mut dyn SpecScratch,
            ) -> Result<(), String> {
                unimplemented!()
            }
            fn eos_token(&self) -> u32 {
                0
            }
            fn ctx_capacity(&self) -> usize {
                0
            }
        }
        let b = Bare;
        assert!(b.dflash_extract_layers().is_none());
    }

    #[test]
    fn emit_outcome_carries_events_and_stop() {
        let o = EmitOutcome {
            events: vec![
                ClientEvent::Committed { id: 42, idx: 7 },
                ClientEvent::Token("hi".to_string()),
            ],
            generation_advanced: true,
            stop: Some(StopReason::Eos),
        };
        assert_eq!(o.events.len(), 2);
        assert_eq!(o.stop, Some(StopReason::Eos));
        // Ordering is load-bearing: committed before token, matching the
        // daemon's emit_committed_event-then-token-write order.
        assert!(matches!(
            o.events[0],
            ClientEvent::Committed { id: 42, idx: 7 }
        ));
        assert!(matches!(&o.events[1], ClientEvent::Token(t) if t == "hi"));
    }

    #[test]
    fn spec_summary_struct_literals_remain_source_compatible() {
        let outcome = EmitOutcome {
            events: Vec::new(),
            generation_advanced: false,
            stop: None,
        };
        let summary = FinishSummary {
            events: Vec::new(),
            finish_reason: "stop",
            tool_calls: 0,
            finalized: None,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        assert!(outcome.events.is_empty());
        assert!(summary.events.is_empty());
    }

    // ── Send bounds (PendingQwen35Unload needs Speculator: Send) ─────────────

    /// Helper: compile-time Send assertion.
    fn _assert_send<T: Send>() {}

    /// `Box<dyn Speculator>` must be Send for loader-private PendingQwen35Unload.
    #[test]
    fn box_dyn_speculator_is_send() {
        _assert_send::<Box<dyn Speculator>>();
    }

    /// `Box<dyn SpecScratch>` must be Send (nested inside GenericDflashSpeculator).
    #[test]
    fn box_dyn_spec_scratch_is_send() {
        _assert_send::<Box<dyn SpecScratch>>();
    }

    /// ChainSpeculator<NgramDrafter> — CPU-only drafter; no GPU fields.
    #[test]
    fn chain_speculator_ngram_is_send() {
        _assert_send::<crate::spec_ngram::ChainSpeculator<crate::spec_ngram::NgramDrafter>>();
    }

    /// MtpSpeculator<DsparkDrafter> — generic MTP wrapping the DSpark drafter.
    #[test]
    fn mtp_speculator_dspark_is_send() {
        _assert_send::<MtpSpeculator<crate::dspark_core::DsparkDrafter>>();
    }

    /// GenericDflashSpeculator — arch-agnostic DFlash.
    #[test]
    fn generic_dflash_speculator_is_send() {
        _assert_send::<crate::dflash_generic::GenericDflashSpeculator>();
    }
}
