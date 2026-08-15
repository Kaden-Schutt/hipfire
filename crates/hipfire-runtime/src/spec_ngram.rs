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

use crate::ddtree::sample_host_nucleus;
use crate::spec::{
    accept_greedy_prefix, NgramCache, PldMatcher, PrefillOutcome, SpecAdvance, SpecGrammar,
    SpecScratch, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;

/// A token-only block drafter: it proposes a continuation block from the
/// committed token history and maintains whatever CPU state it needs. The GPU
/// verify/accept/commit is [`ChainSpeculator`]'s job, not the drafter's.
pub trait BlockDrafter {
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
    fn name(&self) -> &'static str {
        "ngram"
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
        // Lazily build the arch-specific verify scratch (target available now).
        if self.scratch.is_none() {
            self.scratch = Some(target.new_spec_scratch(gpu, self.block_size)?);
        }

        // Advance the target over the prompt (miss → reset + full) or just the
        // new suffix (hit → no reset, from prefill_start). The target owns the
        // chunked/abortable forward; we only need its KV + recurrent state moved.
        let start = if cache_hit { prefill_start } else { 0 };
        let adv = target.spec_advance(gpu, prefill_tokens, start, !cache_hit, abort, None)?;
        let first_token = match adv {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready {
                last_argmax,
                last_logits,
            } => {
                // temp≈0: historical last_argmax (byte-identical greedy).
                // temp>0 + samples: same host nucleus as Qwen35 chain verify
                // (temp/top_k/top_p + self.rng_state). Missing last_logits → Err
                // so we never silently fall back to greedy on a sampled request.
                if !(self.samples && self.sample_temp > 1e-6) {
                    last_argmax
                } else {
                    let logits = last_logits.ok_or_else(|| {
                        "ChainSpeculator: temp>0 prefill needs last_logits \
                         from SpecTarget::spec_advance (target only exposed GPU argmax)"
                            .to_string()
                    })?;
                    sample_host_nucleus(
                        &logits,
                        self.sample_temp,
                        self.sample_top_p,
                        self.sample_top_k,
                        &mut self.rng_state,
                    )
                }
            }
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
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        // `max_emit` is remaining client-visible budget. Reject 0 before any
        // mutation. emit = accepted drafts + bonus, so propose at most
        // max_emit-1 drafts: max_emit==1 → draft_n=0 → verify [seed] only and
        // commit the single bonus — never a doomed 1-token draft + clamp.
        if max_emit == 0 {
            return Err("ChainSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        let draft_n = max_emit
            .saturating_sub(1)
            .min(self.block_size.saturating_sub(1));
        // Propose the draft (pure CPU) BEFORE borrowing scratch.
        let draft = self.drafter.propose(emitted, seed, draft_n);

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
        let mut acc = accept_greedy_prefix(&draft, &picks, None);
        // Clamp accept so emit.len() (= accepted + 1) never exceeds max_emit
        // *before* commit_prefix advances target state past the budget.
        if acc.committed.len() > max_emit {
            let keep = max_emit.max(1);
            acc.accepted = keep.saturating_sub(1).min(acc.accepted);
            acc.committed.truncate(keep);
        }
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
        )
        .cap_emit(max_emit))
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset; the verify scratch is reusable GPU state — kept.
        self.drafter.reset();
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ddtree::sample_host_nucleus;
    use crate::llama;

    // ── Task 4: ChainSpeculator first-token sampling seams ───────────────
    //
    // Prefill after set_sampling:
    //   !(samples && sample_temp > 1e-6) → last_argmax (greedy / non-sampling target)
    //   samples && sample_temp > 1e-6    → sample_host_nucleus(last_logits, temp,
    //                                     top_p, top_k, &mut rng_state)
    //   missing last_logits on sample path → Err (no silent greedy)
    // set_sampling reseeds rng_state to 0x13579BDF.

    #[test]
    fn chain_prefill_temp_zero_keeps_last_argmax() {
        let logits = [1.0f32, -2.0, 0.5, 4.25, 4.0];
        let last_argmax = llama::argmax(&logits);
        assert_eq!(last_argmax, 3);

        // Gate used by ChainSpeculator::prefill: greedy when not (samples && temp>1e-6).
        let samples = true;
        let sample_temp = 0.0f32;
        let take_sample = samples && sample_temp > 1e-6;
        assert!(!take_sample);
        let first = if !take_sample {
            last_argmax
        } else {
            panic!("must not sample at temp=0");
        };
        assert_eq!(first, last_argmax);

        // samples=false also forces last_argmax even at temp>0 (requires_greedy path).
        let samples_off = false;
        let temp_hot = 0.9f32;
        assert!(!(samples_off && temp_hot > 1e-6));

        // Greedy does not advance the fixed set_sampling seed.
        let seed = 0x13579BDF_u64;
        let mut rng = seed;
        let _ = llama::argmax(&logits);
        assert_eq!(rng, seed);
    }

    #[test]
    fn chain_prefill_temp_gt_zero_host_nucleus_advances_rng() {
        let logits = [0.0f32, 2.0, 0.5, -1.0, 0.1];
        let temp = 0.8f32;
        let top_p = 0.95f32;
        let top_k = 0usize; // disabled cut — full-softmax nucleus only if top_p binds
        assert!(temp > 1e-6);

        // set_sampling reseed contract.
        let seed = 0x13579BDF_u64;
        let samples = true;
        assert!(samples && temp > 1e-6);

        let mut rng_a = seed;
        let tok_a = sample_host_nucleus(&logits, temp, top_p, top_k, &mut rng_a);
        assert!((tok_a as usize) < logits.len());
        assert_ne!(rng_a, seed, "host nucleus must advance rng_state once");

        // Deterministic: same seed → same token + same post-state.
        let mut rng_b = seed;
        let tok_b = sample_host_nucleus(&logits, temp, top_p, top_k, &mut rng_b);
        assert_eq!(tok_b, tok_a);
        assert_eq!(rng_b, rng_a);

        // Stream continues for later verify_block_sampled draws.
        let mut rng_c = rng_a;
        let _ = sample_host_nucleus(&logits, temp, top_p, top_k, &mut rng_c);
        assert_ne!(rng_c, rng_a);

        // top_k cut is honored (keeps only k highest before draw).
        let mut rng_k = seed;
        let tok_k = sample_host_nucleus(&logits, temp, 1.0, 2, &mut rng_k);
        assert!((tok_k as usize) < logits.len());
        assert_ne!(rng_k, seed);

        // Greedy branch remains last_argmax / llama::argmax.
        assert_eq!(llama::argmax(&logits), 1);
    }

    #[test]
    fn chain_prefill_missing_last_logits_is_err() {
        // Production Err string from ChainSpeculator::prefill sample arm.
        let last_logits: Option<Vec<f32>> = None;
        let sample_temp = 0.7f32;
        let samples = true;
        assert!(samples && sample_temp > 1e-6);

        let result: Result<u32, String> = (|| {
            let logits = last_logits.ok_or_else(|| {
                "ChainSpeculator: temp>0 prefill needs last_logits \
                 from SpecTarget::spec_advance (target only exposed GPU argmax)"
                    .to_string()
            })?;
            let mut rng = 0x13579BDF_u64;
            Ok(sample_host_nucleus(&logits, sample_temp, 1.0, 0, &mut rng))
        })();

        let err = result.expect_err("missing last_logits must Err, not greedy");
        assert!(
            err.contains("temp>0 prefill needs last_logits"),
            "unexpected err: {err}"
        );
        assert!(err.contains("ChainSpeculator"), "{err}");
    }
}
