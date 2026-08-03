// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Generic model-free chain speculator + the n-gram / PLD block drafter.
//!
//! [`ChainSpeculator<D>`] is the arch-agnostic [`Speculator`] skeleton for any
//! **token-only block drafter**: propose a block from token ids → verify with
//! the target (`SpecTarget::verify_block`) → [`accept_greedy_prefix`] → fix
//! target state (`SpecTarget::commit_prefix`). It owns the target-side verify
//! scratch and the whole verify/accept/commit flow; the drafter (a
//! [`BlockDrafter`]) owns only *propose + bookkeeping*. A future classic
//! "small model drafts for big model" drafter implements `BlockDrafter` and gets
//! this skeleton for free — no new accept/rewind code.
//!
//! [`NgramDrafter`] is the model-free drafter: Prompt Lookup Decoding (Saxena
//! 2023, context-suffix self-match) with a rolling bigram fallback
//! ([`NgramCache`]), over the committed token history (prompt + emitted).
//! Verification is exact, so an over-eager draft only costs τ, never coherence.
//!
//! **Perf is situational on recurrent (DeltaNet) targets** — opt-in
//! (`HIPFIRE_NGRAM_DRAFT=1`): the verify forward runs the recurrence
//! sequentially over the block there, so it only wins on high PLD acceptance
//! (high prompt-copy). On pure-attention targets the verify is block-parallel,
//! so model-free spec wins broadly.

use crate::spec::{
    accept_greedy_prefix, NgramCache, PldMatcher, PrefillOutcome, SpecAdvance, SpecGrammar,
    SpecScratch, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// A token-only block drafter: it proposes a continuation block from the
/// committed token history and maintains whatever CPU state it needs. The GPU
/// verify/accept/commit is [`ChainSpeculator`]'s job, not the drafter's.
pub trait BlockDrafter: Send {
    /// Seed drafter state from the full rendered prompt (called at prefill).
    fn prefill_seed(&mut self, prompt_tokens: &[u32]);

    /// Propose up to `max_draft` continuation tokens after `seed` (whose absolute
    /// history is the prompt seeded above ++ `emitted`, with `emitted.last() ==
    /// seed`). Empty ⇒ a pure AR step (block is just `[seed]`).
    fn propose(&mut self, emitted: &[u32], seed: u32, max_draft: usize) -> Vec<u32>;

    /// Grow drafter state with the tokens committed this window. `emitted` is the
    /// pre-commit history; `committed` is the newly-emitted tail.
    fn observe(&mut self, emitted: &[u32], committed: &[u32]);

    /// Clear drafter-local state for a fresh conversation.
    fn reset(&mut self);
}

/// Model-free n-gram / PLD drafter. See module docs.
pub struct NgramDrafter {
    /// Bigram fallback predictor; seeded from the prompt, grown from output.
    ngram: NgramCache,
    /// PLD self-match matcher (primary draft source).
    pld: PldMatcher,
    /// The rendered prompt, kept so PLD/bigram see the full context (prompt +
    /// emitted) — PLD's biggest wins are copies from the prompt.
    prompt: Vec<u32>,
}

impl NgramDrafter {
    /// `block_size` sizes the PLD spine cap (`block_size - 1`); `min_count` is the
    /// bigram trust threshold.
    pub fn new(min_count: u32, block_size: usize) -> Self {
        let block_size = block_size.max(2);
        // min_extract = 1 keeps even short self-matches usable — exact verify
        // gates them anyway.
        let pld = PldMatcher {
            ngram_lens: vec![5, 4, 3],
            max_extract: block_size - 1,
            min_extract: 1,
        };
        Self {
            ngram: NgramCache::new(min_count),
            pld,
            prompt: Vec::new(),
        }
    }
}

impl BlockDrafter for NgramDrafter {
    fn prefill_seed(&mut self, prompt_tokens: &[u32]) {
        self.prompt.clear();
        self.prompt.extend_from_slice(prompt_tokens);
        self.ngram.observe_many(prompt_tokens);
    }

    fn propose(&mut self, emitted: &[u32], _seed: u32, max_draft: usize) -> Vec<u32> {
        // ctx = prompt ++ emitted; its last token is the seed.
        let mut ctx = Vec::with_capacity(self.prompt.len() + emitted.len());
        ctx.extend_from_slice(&self.prompt);
        ctx.extend_from_slice(emitted);
        // PLD first (longest self-match), bigram chain as fallback.
        if let Some(m) = self.pld.lookup(&ctx) {
            let mut d = m.tokens;
            d.truncate(max_draft);
            if !d.is_empty() {
                return d;
            }
        }
        if ctx.len() >= 2 {
            let mut d = Vec::with_capacity(max_draft);
            let mut a = ctx[ctx.len() - 2];
            let mut b = ctx[ctx.len() - 1];
            while d.len() < max_draft {
                match self.ngram.predict(a, b) {
                    Some((c, _)) => {
                        d.push(c);
                        a = b;
                        b = c;
                    }
                    None => break,
                }
            }
            return d;
        }
        Vec::new()
    }

    fn observe(&mut self, emitted: &[u32], committed: &[u32]) {
        // Grow the bigram cache with the new tokens, including the triples
        // spanning the previous-context boundary (last 2 of prompt++emitted).
        let plen = self.prompt.len();
        let total = plen + emitted.len();
        let mut window: Vec<u32> = Vec::with_capacity(2 + committed.len());
        for i in total.saturating_sub(2)..total {
            window.push(if i < plen {
                self.prompt[i]
            } else {
                emitted[i - plen]
            });
        }
        window.extend_from_slice(committed);
        self.ngram.observe_many(&window);
    }

    fn reset(&mut self) {
        self.ngram = NgramCache::new(self.ngram.min_count);
        self.prompt.clear();
    }
}

/// Arch-agnostic chain speculator over any [`BlockDrafter`]. Owns the target
/// verify scratch (built lazily on first `prefill` via
/// [`SpecTarget::new_spec_scratch`]) and the verify/accept/commit flow.
pub struct ChainSpeculator<D: BlockDrafter> {
    drafter: D,
    /// Max block size including the seed: a window verifies `[seed, draft..]`
    /// with `draft.len() <= block_size - 1`, so `b <= block_size`.
    block_size: usize,
    ctx_capacity: usize,
    scratch: Option<Box<dyn SpecScratch>>,
    /// Whether the TARGET arch implements `SpecTarget::verify_block_sampled`
    /// (set by `build_speculator` from arch_id). Drives `requires_greedy()`:
    /// a target without sampled verify keeps the n-gram path greedy-only
    /// (temp>0 → AR), so `step` never calls a verify that would `Err`.
    samples: bool,
    /// Per-request sampling, set via `set_sampling`. Default greedy (temp 0).
    sample_temp: f32,
    sample_top_p: f32,
    sample_top_k: usize,
    /// Sampled-verify RNG stream; reset to a fixed seed per request in
    /// `set_sampling` so a sampled request is deterministic given its seed.
    rng_state: u64,
}

impl<D: BlockDrafter> ChainSpeculator<D> {
    pub fn new(drafter: D, block_size: usize, ctx_capacity: usize, samples: bool) -> Self {
        Self {
            drafter,
            block_size: block_size.max(2),
            ctx_capacity,
            scratch: None,
            samples,
            sample_temp: 0.0,
            sample_top_p: 1.0,
            sample_top_k: 0,
            rng_state: 0x13579BDF,
        }
    }
}

impl<D: BlockDrafter> Speculator for ChainSpeculator<D> {
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
        // Lazily build the arch-specific verify scratch (target available now).
        if self.scratch.is_none() {
            self.scratch = Some(target.new_spec_scratch(gpu, self.block_size)?);
        }

        // Advance the target over the prompt (miss → full) or just the new suffix
        // (hit → from prefill_start). The central speculative lifecycle already
        // performed the authoritative miss reset.
        let start = if cache_hit { prefill_start } else { 0 };
        let adv =
            target.spec_advance_cold_start(gpu, prefill_tokens, start, !cache_hit, abort, None)?;
        let first_token = match adv {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready { last_argmax } => last_argmax,
        };

        self.drafter.prefill_seed(prompt_tokens);
        Ok(PrefillOutcome::Ready { first_token })
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        _temp: f32, // n-gram verify is greedy-only
    ) -> Result<SpecStep, String> {
        // Propose the draft (pure CPU) BEFORE borrowing scratch.
        let draft = self.drafter.propose(emitted, seed, self.block_size - 1);

        // block = [seed, draft..] ; b = block.len() in 1..=block_size.
        let mut block = Vec::with_capacity(draft.len() + 1);
        block.push(seed);
        block.extend_from_slice(&draft);

        // Sampling decision (set per request via `set_sampling`). `samples` is the
        // target's capability; `sample_temp > 0` is the request asking for it.
        let sampled = self.samples && self.sample_temp > 1e-6;
        let (s_temp, s_top_p, s_top_k) = (self.sample_temp, self.sample_top_p, self.sample_top_k);
        // Copy the RNG stream to a local so the verify call can take `&mut` on it
        // alongside the `&mut scratch` borrow (both are `self` fields).
        let mut rng_state = self.rng_state;

        let scratch = self
            .scratch
            .as_deref_mut()
            .ok_or("ChainSpeculator: step before prefill")?;

        // Verify: target snapshots its pre-state into `scratch`, runs the block,
        // leaves state advanced by b. Greedy → per-position argmax; sampled →
        // per-position sample at temp/(top_k,top_p) (faithful for a point-mass
        // n-gram draft via the same accept-prefix below). The trailing `None`
        // hidden_out sink is the DFlash hidden-capture arg the n-gram path never
        // uses.
        let picks = if sampled {
            target.verify_block_sampled(
                gpu,
                &block,
                position,
                scratch,
                s_temp,
                s_top_p,
                s_top_k,
                &mut rng_state,
            )?
        } else {
            target.verify_block(gpu, &block, position, scratch, None)?
        };

        // Shared accept-prefix (eos=None: EOS handled downstream by the daemon
        // decode loop). `committed` = accepted drafts ++ bonus. Faithful in both
        // modes: greedy accepts on argmax match, sampled on target-sample match.
        let acc = accept_greedy_prefix(&draft, &picks, None);
        let bonus = *acc
            .committed
            .last()
            .expect("eos=None always yields a bonus");

        // Fix target state to the committed prefix block[..accept_len+1] (the
        // target decides full-accept-skip vs rewind+replay vs no-op internally).
        target.commit_prefix(gpu, &block, acc.accepted, position, scratch)?;
        // Persist the advanced sampled-verify RNG stream (no-op in greedy mode).
        self.rng_state = rng_state;

        self.drafter.observe(emitted, &acc.committed);

        Ok(SpecStep::new(
            acc.committed.iter().copied(),
            bonus,
            draft.len(),
            acc.accepted,
        ))
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Drafter-local reset; the verify scratch is reusable GPU state — kept.
        self.drafter.reset();
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn set_sampling(&mut self, temp: f32, top_p: f32, top_k: usize, _cactus_delta: f32) {
        // n-gram drafts are a point mass (no draft distribution), so cactus (the
        // acceptance bump) does not apply. Store temp/top_p/top_k and reset the
        // sampled-verify RNG stream to a fixed seed per request (deterministic
        // given the seed). `step` only takes the sampled path when `samples`.
        self.sample_temp = temp;
        self.sample_top_p = top_p;
        self.sample_top_k = top_k;
        self.rng_state = 0x13579BDF;
    }

    fn requires_greedy(&self) -> bool {
        // Sampled n-gram needs the target's `verify_block_sampled`; only arches
        // that implement it are built with `samples = true` (build_speculator).
        // When false, a temp>0 request is kept off this drafter by the daemon
        // dispatch (`spec_can_sample`) and routes to AR — never silent-greedy.
        !self.samples
    }

    fn free(mut self: Box<Self>, gpu: &mut Gpu) {
        if let Some(scratch) = self.scratch.take() {
            scratch.free(gpu);
        }
    }
}

// ── Send-bound assertions ──────────────────────────────────────────────
#[cfg(test)]
mod send_assertions {
    use super::*;

    /// Helper: compile-time Send assertion.
    fn _assert_send<T: Send>() {}

    /// `Box<dyn BlockDrafter>` — for future use if drafter is stored boxed.
    #[test]
    fn box_dyn_block_drafter_is_send() {
        _assert_send::<Box<dyn BlockDrafter>>();
    }
}
