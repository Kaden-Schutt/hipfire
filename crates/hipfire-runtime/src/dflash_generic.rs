// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Target-generic chain-mode DFlash speculator.
//!
//! This is the arch-free twin of `hipfire_arch_qwen35::speculative::spec_step_dflash`
//! / `dflash_spec::DflashSpeculator`. It drives the SAME block-diffusion drafter
//! forward ([`crate::dflash`]) but verifies through the arch-generic
//! [`SpecTarget`] trait instead of a concrete qwen35 `ModelSlot`.
//!
//! Because the target is reached only through `SpecTarget`, the whole
//! DeltaNet-specific apparatus the qwen35 path carries — recurrent snapshot
//! ([`DeltaNetSnapshot`]), the GDN innovation tape ([`GdnTape`]), the hidden-state
//! ring buffer, and the post-verify rewind/replay — DISAPPEARS. For a stateless
//! dense-attention target (LLaMA / plain Qwen3) verify is one block-parallel
//! forward whose accepted-prefix KV is already correct; nothing to rewind.
//!
//! Mechanically the generic skeleton extracted from `spec_step_dflash` is:
//!   1. build the masked block `[seed, MASK, …, MASK]`,
//!   2. draft it in ONE [`draft_forward`] (broadcast the target's mask-token
//!      embedding as the noise input; positions per the qwen35 path),
//!   3. derive `drafts` by applying the TARGET lm_head to the draft hidden rows
//!      (`draft_scratch.x` rows `1..b`) and argmax-ing,
//!   4. verify `[seed, drafts…]` through [`SpecTarget::verify_block`], which
//!      returns the per-position target argmax AND the per-position hidden rows,
//!   5. greedy-accept the longest matching prefix + bonus, append ONLY the
//!      committed-prefix hidden to `target_hidden_host`, and advance.
//!
//! **Linear chain is the SHIPPED DEFAULT.** The DDTree tree-SWOR arm raises τ
//! (acceptance) but costs more per cycle, and net loses to chain on every
//! drafter measured (DeltaNet + non-DeltaNet qwen3-8b/Bielik) — the DFlash
//! drafter emits independent per-position marginals, so a tree branch has no
//! joint to exploit. Opt IN to the tree with the CLI `--ddtree` flag (sets
//! `HIPFIRE_DFLASH_TREE=1`). Both arms are lossless at temp 0 and
//! distribution-exact at temp>0 (chain via naive sampling, tree via SWOR).
//!
//! The chain (linear) verify supports BOTH greedy (temp≈0) and a
//! distribution-EXACT temp>0 path: SpecInfer NAIVE sampling (draw `x ~
//! softmax(target_logits/temp)` per position, accept the drafted token it lands
//! on, else emit the draw as the bonus — see [`crate::ddtree::naive_sample_chain`]).
//! So [`supports_temp_verify`](Speculator::supports_temp_verify) is `true` for the
//! chain.
//!
//! The tree arm (opt-in; `--ddtree` / `HIPFIRE_DFLASH_TREE=1`) verifies the WHOLE bounded
//! DDTree in ONE tree-masked target forward ([`SpecTarget::verify_tree_logits`])
//! and walks the per-node logits with the q-exploiting without-replacement
//! speculative sampler ([`crate::ddtree::sample_verified_tree_swor`]). This is
//! ALSO distribution-exact at temp>0, so a tree-enabled speculator reports
//! `supports_temp_verify = true` and temp>0 routes
//! through the single-pass tree-SWOR rather than the chain naive sampler. At
//! temp 0 the greedy argmax walk through the tree is lossless == AR.

use crate::ddtree::{
    build_ddtree_tree_bounded, linearize_tree_with_parents, naive_sample_chain,
    sample_verified_tree, sample_verified_tree_swor, swor_draft_candidates, topk_from_logits,
};
use crate::dflash::{draft_forward, DflashConfig, DflashScratch, DflashWeights};
use crate::hfq::HfqFile;
use crate::llama;
use crate::spec::{
    accept_greedy_prefix, PrefillOutcome, SpecAdvance, SpecGrammar, SpecScratch, SpecStep,
    SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::path::Path;

/// Tree-verify defaults (overridable via `HIPFIRE_DDTREE_BUDGET` / `_TOPK`).
const DEFAULT_TREE_BUDGET: usize = 8;
const DEFAULT_TREE_TOPK: usize = 2;

/// Effective tree node budget for this config, clamped so the linearized tree
/// (`budget` nodes + 1 seed) can never exceed the dense verify kernel's batch
/// ceiling. Single source of truth shared by the tree builder ([`TreeMode`])
/// and the verify-scratch sizer ([`dense_tree_verify_nodes`]) so the two can't
/// disagree.
fn clamped_tree_budget(flags: &rdna_compute::FeatureFlags) -> usize {
    flags
        .ddtree_budget
        .unwrap_or(DEFAULT_TREE_BUDGET)
        .clamp(1, crate::llama::PREFILL_MAX_BATCH - 1)
}

/// Linearized node count (clamped `budget` + seed) the dense DFlash tree-verify
/// scratch must hold, or `0` when the tree arm is disabled. `SpecTarget::
/// new_spec_scratch` sizes `PrefillBatchScratch` to at least this, so a large
/// `HIPFIRE_DDTREE_BUDGET` (or `--draft-max`) can't overflow the verify batch —
/// the `forward_prefill_batch_tree: tree size N exceeds max_batch` panic.
pub fn dense_tree_verify_nodes(flags: &rdna_compute::FeatureFlags) -> usize {
    if flags.dflash_tree {
        clamped_tree_budget(flags) + 1
    } else {
        0
    }
}

/// Resolved tree-mode policy, read once at build time from the environment.
///
/// `enabled` gates the whole arm (`HIPFIRE_DFLASH_TREE=1`). When off the
/// speculator is byte-for-byte the original chain path. `budget` caps the tree
/// node count (passed as `max_nodes` to [`build_ddtree_tree_bounded`]); `topk`
/// is the per-position breadth (the second dim of the top-K marginal arrays).
#[derive(Debug, Clone, Copy)]
struct TreeMode {
    enabled: bool,
    budget: usize,
    topk: usize,
}

impl TreeMode {
    fn from_flags(flags: &rdna_compute::FeatureFlags) -> Self {
        TreeMode {
            enabled: flags.dflash_tree,
            budget: clamped_tree_budget(flags),
            topk: flags.ddtree_topk.unwrap_or(DEFAULT_TREE_TOPK),
        }
    }
}

/// Target-generic chain-mode DFlash speculator.
///
/// Owns the loaded draft weights/scratch/config, the cumulative target-hidden
/// host buffer (`[committed_rows × num_extract × hidden]` f32, row-major,
/// extract-layers ascending), and the arch-specific verify scratch the target
/// minted via [`SpecTarget::new_spec_scratch`]. The target itself is borrowed
/// per call — never owned — exactly like the qwen35 `DflashSpeculator`.
pub struct GenericDflashSpeculator {
    weights: DflashWeights,
    scratch: DflashScratch,
    config: DflashConfig,
    /// Cumulative committed target-hidden rows. Authoritative CPU shadow handed
    /// to `draft_forward` each cycle (the generic path always uses the host
    /// buffer — there is no D2D scatter fast path here, unlike qwen35's GPU-side
    /// hidden_rb). Grows by exactly `accept+1` rows per accepted cycle.
    target_hidden_host: Vec<f32>,
    /// Per-target verify scratch, minted by `SpecTarget::new_spec_scratch`.
    /// `Option` so `free` can move it out for explicit GPU release.
    verify_scratch: Option<Box<dyn SpecScratch>>,
    block_size: usize,
    ctx_capacity: usize,
    /// Tree-verify policy (default OFF; opt in via `--ddtree` / `HIPFIRE_DFLASH_TREE=1`). When
    /// `enabled` is false the `step` takes the original single-path chain verify.
    tree: TreeMode,
    /// Xorshift64* RNG state for the temp>0 chain naive sampler. Bit-compatible
    /// with the qwen35 spec sampler's seed so the chain draws are deterministic.
    rng_state: u64,
    /// Per-request temperature from [`Speculator::set_sampling`]. Used only for
    /// the post-prefill first-token draw (`step` receives `temp` as an argument).
    /// Default 0 → historical greedy argmax. top_p/top_k/cactus are accepted by
    /// `set_sampling` for API parity with qwen35 but the shipped chain path is
    /// temperature-only via [`naive_sample_chain`] (same as later verify/bonus).
    sample_temp: f32,
}

impl GenericDflashSpeculator {
    fn num_extract(&self) -> usize {
        self.config.num_extract()
    }

    /// True one-token window: verify only `[seed]`, emit exactly one target
    /// token, commit prefix length 0 (full accept of a b=1 block). No draft
    /// forward, no post-commit `cap_emit` truncation of a longer commit.
    ///
    /// Greedy (`temp≈0`): argmax of the seed row. Sampled: one draw from
    /// `softmax(logits/temp)` via [`naive_sample_chain`] with empty drafts —
    /// distribution-exact, never an argmax substitute.
    fn step_one_token(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        temp: f32,
    ) -> Result<SpecStep, String> {
        let ne = self.num_extract();
        let h = self.config.hidden;
        let vocab = self.config.vocab_size;
        let block = [seed];

        let ctx_elems = position * ne * h;
        assert_eq!(
            self.target_hidden_host.len(),
            ctx_elems,
            "target_hidden_host len {} != position {} * ne {} * h {}",
            self.target_hidden_host.len(),
            position,
            ne,
            h
        );

        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
        let mut block_hidden: Vec<f32> = Vec::with_capacity(ne * h);

        let token = if temp > 1e-6 {
            let logits = target.verify_block_logits(
                gpu,
                &block,
                position,
                vs.as_mut(),
                Some(&mut block_hidden),
            )?;
            debug_assert_eq!(logits.len(), vocab);
            // Empty drafts → single bonus draw at row 0 (distribution-exact).
            let (accepted, bonus) =
                naive_sample_chain(&logits, &[], vocab, temp, &mut self.rng_state);
            debug_assert_eq!(
                accepted, 0,
                "empty-draft naive_sample_chain accepts nothing"
            );
            bonus
        } else {
            let pick =
                target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut block_hidden))?;
            debug_assert_eq!(pick.len(), 1);
            pick[0]
        };
        debug_assert_eq!(block_hidden.len(), ne * h);

        // b=1 full accept: accept_len == block.len()-1 == 0.
        target.commit_prefix(gpu, &block, 0, position, vs.as_mut())?;

        // H2: seed row only (accepted drafts = 0).
        let keep_elems = committed_block_hidden_elems(0, ne, h);
        self.target_hidden_host
            .extend_from_slice(&block_hidden[..keep_elems]);
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, 0, ne, h),
            "host buffer length mismatch after one-token commit"
        );

        let step = SpecStep::new(std::iter::once(token), token, 0, 0);
        debug_assert_eq!(step.emit.len(), 1);
        // Defensive only — one-token path never overshoots.
        Ok(step.cap_emit(1))
    }

    /// Single-pass SWOR tree-verify step. Builds a bounded DDTree from the
    /// per-position draft marginals, LINEARIZES it, verifies the WHOLE tree in
    /// ONE tree-masked target forward ([`SpecTarget::verify_tree_logits`]), then
    /// walks the per-node target logits with the distribution-exact
    /// without-replacement speculative sampler ([`sample_verified_tree_swor`] at
    /// temp>0; [`sample_verified_tree`]'s greedy argmax walk at temp 0).
    ///
    /// This is the q-exploiting SWOR that supersedes the prior linear-n-best
    /// (N forwards/cycle, 2.4× loss) AND the host-side `naive_sample_chain`
    /// temp>0 (which collapses on high-entropy prompts because it ignores `q`).
    /// The whole verify is ONE forward.
    ///
    /// LOSSLESS at temp 0: the greedy walk follows the target argmax through the
    /// tree exactly as a chain verify would, so every committed token is the
    /// target's own greedy continuation. DISTRIBUTION-EXACT at temp>0: every
    /// emitted token is a draw from the (residual) target, so the output marginal
    /// equals `softmax(target_logits / temp)` regardless of the draft `q`
    /// (validated by the `*_preserves_target_distribution` MC tests).
    ///
    /// Hidden capture (H2): `verify_tree_logits` captures one residual-hidden row
    /// per linearized slot. We append, in committed order, the seed slot (0) plus
    /// each accepted node's slot (`node_idx + 1`) — `accept_len + 1` rows total —
    /// so the cumulative `target_hidden_host` grows by exactly `accept + 1`,
    /// matching `draft_forward`'s incremental contract. The bonus row is NOT
    /// appended (its hidden materializes next cycle when it forwards as slot 0).
    #[allow(clippy::too_many_arguments)]
    fn step_tree(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        draft_logits: &[f32],
        vocab: usize,
        b: usize,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        let ne = self.num_extract();
        let h = self.config.hidden;
        let depth = b - 1; // drafted positions = block_size - 1

        // Per-position top-K marginals from the draft logits, then the bounded
        // best-first DDTree (Algorithm 1). topk is capped to vocab defensively.
        let topk = self.tree.topk.min(vocab).max(1);
        let (top_tokens, top_log_probs) = topk_from_logits(draft_logits, depth, vocab, topk);
        let tree = build_ddtree_tree_bounded(
            &top_tokens,
            &top_log_probs,
            depth,
            topk,
            0,
            self.tree.budget,
            f32::NEG_INFINITY,
        );

        // Linearize: slot 0 = seed; slots 1.. = tree.nodes (topological order).
        // mask_block is the [big_n × big_n] additive ancestor-visibility bias;
        // depth_positions carry each slot's DEPTH RoPE position (position + depth)
        // — the verify rotates Q/K at these so the bushy-tree verify is lossless.
        let (verify_tokens, depth_positions, mask_block, _parents) =
            linearize_tree_with_parents(&tree, seed, position as u32);
        let big_n = verify_tokens.len();
        debug_assert_eq!(big_n, 1 + tree.num_nodes());

        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;

        // ── ONE tree-masked verify forward → per-node target logits + hidden ──
        let mut block_hidden: Vec<f32> = Vec::with_capacity(big_n * ne * h);
        let logits_per_slot = target.verify_tree_logits(
            gpu,
            &verify_tokens,
            &mask_block,
            &depth_positions,
            position,
            vs.as_mut(),
            Some(&mut block_hidden),
        )?;
        debug_assert_eq!(logits_per_slot.len(), big_n * vocab);
        debug_assert_eq!(block_hidden.len(), big_n * ne * h);

        // ── Accept walk: greedy (temp≈0) or q-exploiting SWOR (temp>0) ────────
        let (accepted_nodes, bonus): (Vec<usize>, u32) = if temp <= 1e-6 {
            sample_verified_tree(&tree, &logits_per_slot, vocab, 0.0, &mut self.rng_state)
        } else {
            // Draw the draft's per-position SWOR candidates + full q from the SAME
            // draft logits the tree was built from (sequential SWOR mirror of the
            // device Gumbel-top-k sampler). The walk then reuses target draws that
            // land on a drafted child; the output marginal stays target-exact.
            let (draft_q, pos_cands) =
                swor_draft_candidates(draft_logits, depth, vocab, topk, temp, &mut self.rng_state);
            sample_verified_tree_swor(
                &tree,
                &logits_per_slot,
                &draft_q,
                &pos_cands,
                depth,
                topk,
                vocab,
                temp,
                &mut self.rng_state,
            )
        };
        let mut accepted_nodes = accepted_nodes;
        let mut accept_len = accepted_nodes.len();
        let mut bonus = bonus;
        // Cap accept walk BEFORE KV/hidden commit so emit (= accept_len + 1)
        // never exceeds max_emit. When the walk accepted past the budget, the
        // next accepted node's token IS the walk's boundary draw — keep it.
        // Re-argmax here would bias every length-capped temp>0 request.
        let max_accept = max_emit.saturating_sub(1);
        if accept_len > max_accept {
            let boundary = tree.nodes[accepted_nodes[max_accept]].token;
            accepted_nodes.truncate(max_accept);
            accept_len = max_accept;
            bonus = boundary;
        }

        // Build the committed token block [seed, accepted node tokens…].
        let mut committed_block: Vec<u32> = Vec::with_capacity(accept_len + 1);
        committed_block.push(seed);
        for &ni in &accepted_nodes {
            committed_block.push(tree.nodes[ni].token);
        }

        // ── KV commit fixup (the bushy-tree correctness gate) ─────────────────
        // The tree forward wrote every node's KV at CONTIGUOUS physical slots
        // [position .. position+big_n) in linearized order, each RoPE'd at its
        // DEPTH. The next cycle reads physical slot `position+k` as the k-th
        // COMMITTED token. That alignment holds ONLY when the accepted path is
        // the linearized SPINE prefix (`accepted_nodes[i] == i`): then linear
        // slot == commit index == depth, so the committed KV is already correct
        // (verified byte-identical to AR at topk=1, where every accept is a
        // spine accept). When the greedy/SWOR walk detours to a non-spine
        // sibling, the committed token's KV sits at a DIFFERENT physical slot
        // with a DIFFERENT depth-RoPE phase → the next cycle reads stale KV and
        // emits a duplicate/garbage token. Re-commit those cycles with ONE
        // causal verify over the committed block at contiguous absolute
        // positions (overwrites the scattered KV with correct phases). Pure
        // attention ⇒ no recurrent state to restore; the rejected-tail KV is
        // overwritten next cycle as usual. Cost: +1 forward only on detour
        // cycles (spine accepts — the common case — stay single-pass).
        let spine_accept = accepted_nodes.iter().enumerate().all(|(i, &ni)| ni == i);
        if spine_accept {
            target.commit_prefix(gpu, &committed_block, accept_len, position, vs.as_mut())?;
        } else {
            // Causal re-verify over [seed, accepted…] fixes the committed KV.
            // Discard its argmax/hidden — the accept decision + hidden already
            // came from the (correct, depth-RoPE) tree forward above.
            let _ = target.verify_block(gpu, &committed_block, position, vs.as_mut(), None)?;
            target.commit_prefix(gpu, &committed_block, accept_len, position, vs.as_mut())?;
        }

        // ── H2 hidden truncation: append the committed slots' hidden in order ─
        // Committed linearized slots: seed = slot 0, accepted node i = slot
        // (node_idx + 1). Gather their per-slot residual-hidden rows (ne × h each)
        // → exactly accept_len + 1 rows. Bonus row excluded.
        let row_stride = ne * h;
        let push_slot = |host: &mut Vec<f32>, slot: usize| {
            host.extend_from_slice(&block_hidden[slot * row_stride..(slot + 1) * row_stride]);
        };
        push_slot(&mut self.target_hidden_host, 0); // seed
        for &ni in &accepted_nodes {
            push_slot(&mut self.target_hidden_host, ni + 1);
        }
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, accept_len, ne, h),
            "host buffer length mismatch after tree-SWOR commit"
        );

        // Lower: emit = accepted node tokens + bonus (seed dropped); next_seed = bonus.
        // Pre-commit clamp above guarantees emit.len() <= max_emit; cap_emit is defense.
        let emit = accepted_nodes
            .iter()
            .map(|&ni| tree.nodes[ni].token)
            .chain(std::iter::once(bonus));
        let step = SpecStep::new(emit, bonus, depth, accept_len);
        debug_assert!(
            step.emit.len() <= max_emit && !step.emit.is_empty(),
            "tree emit len {} vs max_emit {}",
            step.emit.len(),
            max_emit
        );
        Ok(step.cap_emit(max_emit))
    }

    /// Shared chain-verify tail: H2 hidden truncation + SpecStep construction.
    ///
    /// Called after both the greedy and temp>0 chain arms have computed `accepted`
    /// and `bonus` and committed the prefix via `commit_prefix`. Appends the first
    /// `accepted + 1` rows of `block_hidden` to `target_hidden_host` (H2: seed +
    /// accepted drafts; the bonus row is NOT appended — it materialises next cycle
    /// as the block seed) and constructs the `SpecStep` for the daemon.
    ///
    /// MUST be called with identical arguments from both arms: the call order
    /// (extend_from_slice → debug_assert_eq → SpecStep::new) is the same in both.
    fn finish_chain(
        &mut self,
        block_hidden: &[f32],
        drafts: &[u32],
        accepted: usize,
        bonus: u32,
        position: usize,
        ne: usize,
        h: usize,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        // Append ONLY the committed-prefix hidden — the first accepted+1 rows of
        // block_hidden (seed + accepted drafts). The bonus's hidden is NOT appended:
        // its proper hidden materialises on the NEXT cycle's verify when it is
        // forwarded as block[0]. This grows the prefix by accept+1, matching
        // draft_forward's contract.
        // draft_forward owns the uploaded_target_hidden_rows cursor (it delta-uploads
        // the appended host rows next step); the generic path never scatters to GPU
        // itself, so we must NOT set it here.
        let keep_elems = committed_block_hidden_elems(accepted, ne, h);
        self.target_hidden_host
            .extend_from_slice(&block_hidden[..keep_elems]);
        debug_assert_eq!(
            self.target_hidden_host.len(),
            committed_host_len(position, accepted, ne, h),
            "host buffer length mismatch after chain commit"
        );

        // ── Lower to SpecStep: emit = committed[1..] (accepted drafts + bonus,
        // seed dropped), next_seed = bonus. emit.len() == accepted + 1 drives the
        // daemon's position += emit.len(). Pre-commit budget guarantees
        // emit.len() <= max_emit; cap_emit is defensive only.
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        let step = SpecStep::new(emit, bonus, drafts.len(), accepted);
        debug_assert!(
            step.emit.len() <= max_emit && !step.emit.is_empty(),
            "chain emit len {} vs max_emit {}",
            step.emit.len(),
            max_emit
        );
        Ok(step.cap_emit(max_emit))
    }
}

/// Pure (testable) truncation math for the partial-accept host-buffer update.
///
/// After a cycle that committed `committed_len` tokens (= `accepted + 2`:
/// `[seed, accepted drafts…, bonus]`), `draft_forward`'s incremental contract
/// grows the cached prefix by exactly `accepted + 1` rows. So the host buffer
/// must hold `(position + accepted + 1) × row_stride` floats, where
/// `row_stride = num_extract × hidden`. The verify produced `block_hidden` for
/// all `b+1 = drafts+1` positions; we keep the first `accepted + 1` rows and
/// discard the rejected tail.
///
/// Returns the number of f32 elements the host buffer should have after the
/// append. Factored out so the truncation invariant is unit-testable without a
/// GPU.
fn committed_host_len(
    position: usize,
    accepted: usize,
    num_extract: usize,
    hidden: usize,
) -> usize {
    (position + accepted + 1) * num_extract * hidden
}

/// Number of leading f32 elements of a verify `block_hidden` buffer that belong
/// to the committed prefix (the first `accepted + 1` rows). The rejected tail
/// (rows `accepted+1 ..= drafts`) is discarded.
fn committed_block_hidden_elems(accepted: usize, num_extract: usize, hidden: usize) -> usize {
    (accepted + 1) * num_extract * hidden
}

/// Clamp an accept/bonus pair so `emit_len = accepted + 1` ≤ `max_emit`.
///
/// When the unclamped accept already fits, returns it unchanged. When it must
/// shrink, `boundary_token` becomes the new bonus — callers pass:
/// - greedy chain: `target_pick[max_accept]` (argmax at the boundary row);
/// - sampled chain: `drafts[max_accept]` (the walk's matched draw at that row);
/// - tree SWOR/greedy: `tree.nodes[accepted_nodes[max_accept]].token`.
///
/// Never substitutes argmax for a sampled/SWOR boundary draw.
fn clamp_accept_to_budget(
    accepted: usize,
    bonus: u32,
    max_emit: usize,
    boundary_token: u32,
) -> (usize, u32) {
    let max_accept = max_emit.saturating_sub(1);
    if accepted > max_accept {
        (max_accept, boundary_token)
    } else {
        (accepted, bonus)
    }
}

impl Speculator for GenericDflashSpeculator {
    fn name(&self) -> &'static str {
        "dflash"
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
        // Mirror the qwen35 prefill cache-hit/miss split, minus the DeltaNet /
        // hidden-ring machinery: on a miss we re-seed the whole prompt and reset
        // the draft's incremental-upload cursor; on a hit we advance only the
        // suffix from `prefill_start` and keep the cached prefix projections.
        let (fill_tokens, start_pos): (&[u32], usize) = if cache_hit {
            (prefill_tokens, prefill_start)
        } else {
            (prompt_tokens, 0)
        };

        if !cache_hit {
            // Cold start: drop both the cumulative host shadow and the draft's
            // upload/projection tracking so the first step re-uploads from row 0.
            self.target_hidden_host.clear();
            self.scratch.reset_upload_tracking();
        }

        // Advance the target AND capture its residual hidden into the cumulative
        // host buffer in one pass. `reset = !cache_hit` zeroes the target's KV
        // (and recurrent, for arches that have it) on a miss. Capture only fires
        // when the target's `dflash_extract_layers()` is `Some` (set at build).
        let adv = target.spec_advance(
            gpu,
            fill_tokens,
            start_pos,
            !cache_hit,
            abort,
            Some(&mut self.target_hidden_host),
        )?;
        // First emit = target draw at the final prompt position.
        // temp≈0 keeps historical `last_argmax` (byte-identical greedy).
        // temp>0 reuses the empty-draft `naive_sample_chain` pipeline that
        // `step_one_token` / later chain verify use — one RNG advance, no
        // second distribution path. Requires host last-row logits from the
        // target (LLaMA/generic path populates `SpecAdvance::Ready.last_logits`).
        let first_token = match adv {
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
            SpecAdvance::Ready {
                last_argmax,
                last_logits,
            } => {
                if self.sample_temp <= 1e-6 {
                    last_argmax
                } else {
                    let logits = last_logits.ok_or_else(|| {
                        "GenericDflashSpeculator: temp>0 prefill needs last_logits \
                         from SpecTarget::spec_advance (target only exposed GPU argmax)"
                            .to_string()
                    })?;
                    let vocab = self.config.vocab_size;
                    if logits.len() != vocab {
                        return Err(format!(
                            "GenericDflashSpeculator: last_logits len {} != vocab {}",
                            logits.len(),
                            vocab
                        ));
                    }
                    let (accepted, bonus) = naive_sample_chain(
                        &logits,
                        &[],
                        vocab,
                        self.sample_temp,
                        &mut self.rng_state,
                    );
                    debug_assert_eq!(
                        accepted, 0,
                        "empty-draft naive_sample_chain accepts nothing"
                    );
                    bonus
                }
            }
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        // emit ≤ max_emit: a window of block size b yields up to b tokens
        // (accepted drafts + bonus ≤ b). Size b so full-accept emit cannot
        // exceed the remaining budget; max_emit==1 takes the true one-token
        // path (no draft, verify [seed] only).
        let cfg_b = self.config.block_size.max(2);
        if max_emit == 0 {
            return Err("GenericDflashSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        if max_emit == 1 {
            return self.step_one_token(gpu, target, position, seed, temp);
        }
        // max_emit >= 2: b ∈ [2, min(cfg, max_emit)] so emit ≤ b ≤ max_emit
        // on full accept without post-commit truncation.
        let b = cfg_b.min(max_emit).max(2);
        assert!(b >= 2, "dflash block size must be >= 2");
        let h = self.config.hidden;
        let ne = self.num_extract();

        // ── 1. Build the masked block: [seed, MASK, …, MASK] ────────────────
        let mut block: Vec<u32> = vec![self.config.mask_token_id; b];
        block[0] = seed;

        // ── 2. Build the block-diffusion draft INPUT ────────────────────────
        // noise_embedding mirrors the z-lab reference `embed_tokens([seed, MASK,
        // …, MASK])` (dflash.py:235-237): position 0 carries the COMMITTED seed
        // token's embedding (it anchors the bidirectional block — every drafted
        // slot attends to it), positions 1..b are the mask-token embedding.
        // Broadcasting MASK to slot 0 too (the old behavior) stripped the anchor
        // and collapsed the drafts into degenerate repetition → low acceptance.
        let mask_row = target.embed_row(gpu, self.config.mask_token_id)?;
        let seed_row = target.embed_row(gpu, seed)?;
        debug_assert_eq!(mask_row.len(), h, "embed_row length != hidden");
        let mut noise_embedding: Vec<f32> = Vec::with_capacity(b * h);
        noise_embedding.extend_from_slice(&seed_row);
        for _ in 1..b {
            noise_embedding.extend_from_slice(&mask_row);
        }

        // positions_q: absolute positions of the block slots [position .. position+b).
        // positions_k: context positions [0 .. position) then block [position .. position+b).
        // (The generic path has no FlashCASK eviction, so positions are contiguous —
        // matching the qwen35 pre-eviction layout byte-for-byte.)
        let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
        let positions_k: Vec<i32> = (0i32..(position + b) as i32).collect();

        // ── 3. Draft forward over the cumulative target-hidden prefix ───────
        // The host buffer is authoritative: hand draft_forward rows [0..position).
        // Its incremental-upload fast path keys off scratch.thlog.uploaded_rows().
        let ctx_elems = position * ne * h;
        assert_eq!(
            self.target_hidden_host.len(),
            ctx_elems,
            "target_hidden_host len {} != position {} * ne {} * h {}",
            self.target_hidden_host.len(),
            position,
            ne,
            h
        );
        draft_forward(
            gpu,
            &self.weights,
            &self.config,
            Some(&noise_embedding),
            Some(&self.target_hidden_host[..ctx_elems]),
            &positions_q,
            &positions_k,
            b,
            position,
            &mut self.scratch,
        )
        .map_err(|e| format!("draft_forward: {e}"))?;

        // ── 3b. Draft → tokens: apply the TARGET lm_head to draft hidden rows ─
        // The draft's final-hidden rows live in draft_scratch.x; row 0 is the seed
        // slot, rows 1..b are the drafted positions (mirrors qwen35's
        // draft_scratch.x.sub_offset(h, batch*h)). We argmax the target lm_head
        // over those b-1 rows to get the drafted tokens.
        let batch = b - 1;
        let draft_hidden = self.scratch.x.sub_offset(h, batch * h);
        let draft_logits = target.lm_head_logits(gpu, &draft_hidden, batch)?;
        debug_assert_eq!(draft_logits.len(), batch * self.config.vocab_size);
        let vocab = self.config.vocab_size;
        let mut drafts: Vec<u32> = Vec::with_capacity(batch);
        for i in 0..batch {
            drafts.push(llama::argmax(&draft_logits[i * vocab..(i + 1) * vocab]));
        }
        for (i, &d) in drafts.iter().enumerate() {
            block[i + 1] = d;
        }

        // ── Tree-verify arm (opt-in; --ddtree / HIPFIRE_DFLASH_TREE=1) ─────
        // Build a bounded DDTree from the per-position draft marginals and verify
        // the WHOLE tree in ONE tree-masked forward (`verify_tree_logits`), then
        // walk it with the distribution-exact SWOR sampler. temp 0 → greedy
        // (lossless == AR); temp>0 → q-exploiting SWOR (distribution-exact). This
        // arm CARRIES temp>0 — the chain naive-sampling path below is skipped when
        // the tree is enabled.
        if self.tree.enabled {
            return self.step_tree(
                gpu,
                target,
                position,
                seed,
                &draft_logits,
                vocab,
                b,
                temp,
                max_emit,
            );
        }

        // ── 4t. temp>0 chain naive-sampling verify (distribution-exact) ─────
        // When the request temperature is >0,
        // verify the chain by drawing x ~ softmax(target_logits/temp) per position
        // and accepting the longest prefix where draft[i] == x_i (SpecInfer NAIVE
        // sampling — distribution-EXACT at any temperature; see
        // `ddtree::naive_sample_chain`). We need the FULL per-position target
        // logits, so this arm uses `verify_block_logits` instead of `verify_block`
        // (which only returns the argmax). The same H2 truncation applies: append
        // only the first accepted+1 rows of the captured hidden.
        if temp > 1e-6 {
            let vs = self
                .verify_scratch
                .as_mut()
                .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
            let mut block_hidden: Vec<f32> = Vec::with_capacity(b * ne * h);
            let logits_per_pos = target.verify_block_logits(
                gpu,
                &block,
                position,
                vs.as_mut(),
                Some(&mut block_hidden),
            )?;
            debug_assert_eq!(logits_per_pos.len(), b * vocab);
            debug_assert_eq!(block_hidden.len(), b * ne * h);

            // `drafts` (len b-1) are block[1..b]; logits_per_pos has b rows (one
            // bonus row past the last draft). naive_sample_chain draws the target
            // token per row, accepts the matching draft prefix, and returns the
            // bonus at the divergence position. All emitted tokens are genuine
            // target draws → distribution-exact.
            let (accepted, bonus) =
                naive_sample_chain(&logits_per_pos, &drafts, vocab, temp, &mut self.rng_state);
            // Pre-commit budget: emit = accepted + 1 ≤ max_emit. With b ≤ max_emit
            // this is defensive; if it binds, keep the walk's own boundary draw
            // (drafts[max_accept] — matched, since accepted > max_accept), never argmax.
            let boundary = if accepted > max_emit.saturating_sub(1) {
                drafts[max_emit.saturating_sub(1)]
            } else {
                bonus
            };
            let (accepted, bonus) = clamp_accept_to_budget(accepted, bonus, max_emit, boundary);

            target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
            return self.finish_chain(
                &block_hidden,
                &drafts,
                accepted,
                bonus,
                position,
                ne,
                h,
                max_emit,
            );
        }

        // ── 4. Verify + accept + truncation (review finding H2) ─────────────
        // Verify [seed, drafts…] through the target. Returns the per-position
        // greedy argmax (length b) AND fills block_hidden with the per-position
        // residual hidden (b rows × ne × h). The target leaves its state advanced
        // by `b`; commit_prefix fixes it to the committed prefix afterward.
        let vs = self
            .verify_scratch
            .as_mut()
            .ok_or("GenericDflashSpeculator: verify scratch already freed")?;
        let mut block_hidden: Vec<f32> = Vec::with_capacity(b * ne * h);
        let target_pick =
            target.verify_block(gpu, &block, position, vs.as_mut(), Some(&mut block_hidden))?;
        debug_assert_eq!(target_pick.len(), b, "verify_block returned != b argmax");
        debug_assert_eq!(
            block_hidden.len(),
            b * ne * h,
            "verify_block hidden != b*ne*h"
        );

        // Greedy accept: drafts = block[1..b], target_pick is the verifier's argmax
        // after each of the b positions. eos=None — DFlash never early-stops on
        // EOS here (the daemon handles EOS downstream). committed = accepted prefix
        // + bonus (= target_pick[accepted]).
        let acc = accept_greedy_prefix(&drafts, &target_pick, None);
        let accepted = acc.accepted;
        let bonus = *acc.committed.last().expect("eos=None yields a bonus");
        // Pre-commit budget: emit = accepted + 1 ≤ max_emit. Greedy boundary
        // re-pick is argmax at the clamped row (temp-0 correct).
        let boundary = if accepted > max_emit.saturating_sub(1) {
            target_pick[max_emit.saturating_sub(1)]
        } else {
            bonus
        };
        let (accepted, bonus) = clamp_accept_to_budget(accepted, bonus, max_emit, boundary);

        // Fix the target state to the committed prefix [seed, accepted…, bonus].
        // For a stateless target this is a no-op; for a recurrent one it restores
        // the snapshot verify_block stashed in `vs`. accept_len passed is the
        // number of accepted DRAFTS (block[1..=accept_len] accepted).
        target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
        self.finish_chain(
            &block_hidden,
            &drafts,
            accepted,
            bonus,
            position,
            ne,
            h,
            max_emit,
        )
    }

    /// Forced tokens (think-budget force-close) must land in the drafter's
    /// cumulative `target_hidden_host`, not just the target's KV. The default
    /// trait hook returns `false`, so the daemon's plain `spec_advance(..., None)`
    /// would leave a hole: the next `draft_forward` asserts host len == position
    /// and would either panic or condition on a stale/short prefix.
    ///
    /// Same one-row ownership as qwen35 `DflashSpeculator::on_forced_advance`:
    /// advance the target WITH hidden extraction into our host buffer and return
    /// `true` so the caller does not double-advance. Fail closed before mutation
    /// when extract layers are missing (capture would be a silent no-op).
    fn on_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        let extract = target.dflash_extract_layers().unwrap_or(&[]);
        if extract.is_empty() {
            // No capture possible → would leave host misaligned. Refuse before
            // any target mutation so the daemon fail-closed path can roll back.
            return Err(
                "GenericDflashSpeculator: forced advance requires dflash extract \
                 layers for target-hidden ownership"
                    .into(),
            );
        }
        let ne = self.num_extract();
        let h = self.config.hidden;
        let expected_len = self.target_hidden_host.len() + tokens.len() * ne * h;
        // Host is authoritative at `start_pos` rows going in (same invariant as step).
        let expected_prefix = start_pos * ne * h;
        if self.target_hidden_host.len() != expected_prefix {
            return Err(format!(
                "GenericDflashSpeculator: forced advance host len {} != start_pos {} * ne {} * h {}",
                self.target_hidden_host.len(),
                start_pos,
                ne,
                h
            ));
        }
        match target.spec_advance(
            gpu,
            tokens,
            start_pos,
            false,
            abort,
            Some(&mut self.target_hidden_host),
        )? {
            SpecAdvance::Aborted => {
                // Caller tears the request down; leaving a partial append is fine
                // because the drafter state is reset on the way out.
                Ok(true)
            }
            SpecAdvance::Ready { .. } => {
                if self.target_hidden_host.len() != expected_len {
                    return Err(format!(
                        "GenericDflashSpeculator: forced advance captured {} floats, expected {}",
                        self.target_hidden_host.len(),
                        expected_len
                    ));
                }
                Ok(true)
            }
        }
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset: clear the cumulative host shadow + the draft's
        // upload/projection tracking. The target's KV/recurrent reset is the
        // daemon's job (it owns the bundle).
        self.target_hidden_host.clear();
        self.scratch.reset_upload_tracking();
        Ok(())
    }

    fn reset_state_evidence(&self) -> Option<crate::spec::SpecResetEvidence> {
        let th = &self.scratch.thlog;
        Some(crate::spec::SpecResetEvidence {
            drafter_reset: self.target_hidden_host.is_empty()
                && th.uploaded_rows() == 0
                && th.proj_cached_rows() == 0
                && th.full_cached_rows() == 0,
            // Generic path has no divergent-render checkpoint ring.
            checkpoint_empty: true,
        })
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn supports_temp_verify(&self) -> bool {
        // BOTH the chain path (SpecInfer naive sampling) and the tree arm
        // (q-exploiting SWOR — `step_tree` + `sample_verified_tree_swor`) are
        // distribution-EXACT at temp>0, so temp>0 always routes through spec.
        true
    }

    fn requires_greedy(&self) -> bool {
        // Chain and tree-SWOR both honor temp>0 distribution-correctly.
        false
    }

    fn set_sampling(&mut self, temp: f32, _top_p: f32, _top_k: usize, _cactus_delta: f32) {
        // Store temp for the post-prefill first-token draw. Re-seed RNG to the
        // same fixed value qwen35 uses per request so sampled runs are
        // deterministic given seed; greedy does not consume it. top_p/top_k are
        // accepted for Speculator API parity — chain verify remains temperature-
        // only via `naive_sample_chain` (identical to later step/bonus draws).
        self.sample_temp = temp;
        self.rng_state = 0x13579BDF;
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        let GenericDflashSpeculator {
            weights,
            scratch,
            verify_scratch,
            ..
        } = *self;
        weights.free_gpu(gpu);
        scratch.free_gpu(gpu);
        if let Some(vs) = verify_scratch {
            vs.free(gpu);
        }
    }
}

/// Build a target-generic chain DFlash speculator from a converted draft HFQ.
///
/// The draft must be the F16 product of `dflash_convert` (`has_mq = false`), so
/// the scratch is built with [`DflashScratch::new`] (the `new_with_mq` path is
/// only for an MQ-quantized draft — review finding L3; qwen35 regressed exactly
/// this once at `dflash_spec.rs:91`).
///
/// The verify scratch is minted by the TARGET (`new_spec_scratch`) so no arch
/// type leaks here, and the target is told the draft's extract layers via
/// [`SpecTarget::set_dflash_extract_layers`] so hidden capture matches the
/// drafter's `fc` input layout.
pub fn build_generic_dflash_speculator(
    gpu: &mut Gpu,
    draft_hfq_path: &str,
    target: &mut dyn SpecTarget,
    ctx_capacity: usize,
) -> Result<Box<dyn Speculator>, String> {
    let draft_hfq = HfqFile::open(Path::new(draft_hfq_path)).map_err(|e| format!("{e}"))?;
    let config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let weights = DflashWeights::load(gpu, &draft_hfq, &config).map_err(|e| format!("{e}"))?;
    let block_size = config.block_size;
    // L3: F16 drafts (dflash_convert) → has_mq=false → DflashScratch::new.
    // new_with_mq only for an MQ-quantized draft.
    let scratch = if weights.has_mq {
        DflashScratch::new_with_mq(gpu, &config, block_size, ctx_capacity, true)
            .map_err(|e| format!("{e}"))?
    } else {
        DflashScratch::new(gpu, &config, block_size, ctx_capacity).map_err(|e| format!("{e}"))?
    };
    let _ = draft_hfq;

    // Tell the target which residual-hidden layers to capture (the drafter's
    // target_layer_ids), and mint the per-target verify scratch.
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    let verify_scratch = target.new_spec_scratch(gpu, block_size)?;

    Ok(Box::new(GenericDflashSpeculator {
        weights,
        scratch,
        config,
        target_hidden_host: Vec::new(),
        verify_scratch: Some(verify_scratch),
        block_size,
        ctx_capacity,
        tree: TreeMode::from_flags(&gpu.flags),
        // Fixed deterministic seed (matches the qwen35 dflash_spec default).
        rng_state: 0x13579BDF,
        // Greedy until a request calls `set_sampling`.
        sample_temp: 0.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::SpecStep;
    use smallvec::SmallVec;

    // The SpecStep lowering the chain step produces: emit = accepted drafts +
    // bonus (seed dropped), next_seed = bonus = emit.last(). Mirrors
    // `spec.rs::emit_len_drives_advance_not_accepted` — pins the load-bearing
    // loop contract that the daemon advances `position` by emit.len(), not by
    // `accepted`.
    fn lower_chain(drafts: &[u32], accepted: usize, bonus: u32) -> SpecStep {
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        SpecStep::new(emit, bonus, drafts.len(), accepted)
    }

    #[test]
    fn chain_lowering_emit_len_is_accepted_plus_one() {
        // 4 drafts, accepted 2, bonus 99 → emit = [d0, d1, 99] (len 3).
        let drafts = [10u32, 11, 12, 13];
        let step = lower_chain(&drafts, 2, 99);
        assert_eq!(step.emit.as_slice(), &[10, 11, 99]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(*step.emit.last().unwrap(), step.next_seed);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 4);
        assert_eq!(step.accepted, 2);
    }

    #[test]
    fn chain_lowering_full_accept() {
        // all 3 drafts accepted, bonus 77 → emit = [d0,d1,d2,77] (len 4).
        let drafts = [10u32, 11, 12];
        let step = lower_chain(&drafts, 3, 77);
        assert_eq!(step.emit.as_slice(), &[10, 11, 12, 77]);
        assert_eq!(step.emit.len(), step.accepted + 1);
        assert_eq!(step.next_seed, 77);
    }

    #[test]
    fn chain_lowering_zero_accept() {
        // 0 accepted → emit = [bonus] only (still non-empty, forward progress).
        let drafts = [10u32, 11];
        let step = lower_chain(&drafts, 0, 42);
        assert_eq!(step.emit.as_slice(), &[42]);
        assert_eq!(step.emit.len(), 1);
        assert_eq!(step.next_seed, 42);
        assert_eq!(step.accepted, 0);
    }

    // The lowering is byte-equivalent to building a SmallVec by hand (guards the
    // SpecStep::new IntoIterator path against an accidental reorder).
    #[test]
    fn chain_lowering_matches_manual_smallvec() {
        let drafts = [5u32, 6, 7, 8];
        let step = lower_chain(&drafts, 1, 99);
        let manual: SmallVec<[u32; 8]> = SmallVec::from_slice(&[5, 99]);
        assert_eq!(step.emit, manual);
    }

    // Partial-accept truncation math (review finding H2): after a step that
    // accepted `accepted` drafts at absolute `position`, the cumulative host
    // buffer must hold (position + accepted + 1) committed rows × ne × h, and we
    // keep exactly (accepted + 1) rows of the verify's block_hidden.
    #[test]
    fn truncation_keeps_accept_plus_one_rows() {
        let ne = 5usize;
        let h = 4096usize;
        // Simulate appending the committed-prefix hidden onto a host buffer that
        // already holds `position` rows.
        for &(position, drafts, accepted) in
            &[(0usize, 16usize, 0usize), (10, 16, 7), (100, 16, 16)]
        {
            let row_stride = ne * h;
            // verify produced b = drafts+1 rows of hidden.
            let b = drafts + 1;
            let block_hidden = vec![1.0f32; b * row_stride];
            let keep = committed_block_hidden_elems(accepted, ne, h);
            assert_eq!(keep, (accepted + 1) * row_stride);
            // Host buffer pre-step holds `position` rows.
            let mut host = vec![0.0f32; position * row_stride];
            host.extend_from_slice(&block_hidden[..keep]);
            assert_eq!(host.len(), committed_host_len(position, accepted, ne, h));
            assert_eq!(host.len(), (position + accepted + 1) * row_stride);
        }
    }

    // Full-accept and zero-accept boundaries of the truncation math.
    #[test]
    fn truncation_boundaries() {
        let ne = 2usize;
        let h = 8usize;
        // zero accept → keep exactly 1 row (the seed position's hidden).
        assert_eq!(committed_block_hidden_elems(0, ne, h), ne * h);
        // full accept of 15 drafts → keep 16 rows.
        assert_eq!(committed_block_hidden_elems(15, ne, h), 16 * ne * h);
        // host length advances by accept+1 rows from position.
        assert_eq!(committed_host_len(50, 3, ne, h), (50 + 4) * ne * h);
    }

    // Pre-commit budget clamp: emit = accepted + 1 must fit max_emit, and the
    // boundary token must be the caller's supplied draw (never silently swapped).
    #[test]
    fn clamp_accept_preserves_fitting_window() {
        assert_eq!(clamp_accept_to_budget(3, 99, 4, 7), (3, 99));
        assert_eq!(clamp_accept_to_budget(0, 42, 1, 7), (0, 42));
        assert_eq!(clamp_accept_to_budget(5, 11, 6, 7), (5, 11));
    }

    #[test]
    fn clamp_accept_uses_boundary_token_not_old_bonus() {
        // accepted=4 → emit would be 5; max_emit=3 → keep 2 drafts + boundary.
        assert_eq!(clamp_accept_to_budget(4, 999, 3, 55), (2, 55));
        // max_emit=1 → zero drafts + boundary as the sole emit token.
        assert_eq!(clamp_accept_to_budget(3, 999, 1, 77), (0, 77));
    }

    #[test]
    fn precommit_budget_emit_never_exceeds_max_emit() {
        // Mirrors finish_chain lowering after clamp_accept_to_budget.
        let drafts = [10u32, 11, 12, 13, 14];
        for max_emit in 1usize..=6 {
            let raw_accepted = 4usize; // would emit 5 without clamp
            let raw_bonus = 99u32;
            let boundary = drafts[max_emit.saturating_sub(1).min(drafts.len() - 1)];
            let (accepted, bonus) =
                clamp_accept_to_budget(raw_accepted, raw_bonus, max_emit, boundary);
            let step = lower_chain(&drafts, accepted, bonus);
            assert!(
                !step.emit.is_empty() && step.emit.len() <= max_emit,
                "emit {:?} vs max_emit {}",
                step.emit.as_slice(),
                max_emit
            );
            assert_eq!(step.emit.len(), accepted + 1);
            assert_eq!(step.next_seed, bonus);
            // Defensive cap_emit is a no-op when pre-commit clamp held.
            let capped = step.clone().cap_emit(max_emit);
            assert_eq!(capped.emit, step.emit);
            assert_eq!(capped.accepted, step.accepted);
            assert_eq!(capped.next_seed, step.next_seed);
        }
    }

    #[test]
    fn sampled_boundary_keeps_matched_draft_not_argmax() {
        // Simulate temp>0 clamp: walk accepted past budget; boundary = drafts[max_accept]
        // (the matched draw), never a fresh argmax substitute.
        let drafts = [10u32, 20, 30, 40];
        let walk_accepted = 3usize;
        let walk_bonus = 999u32; // would-be full-accept bonus
        let max_emit = 2usize;
        let max_accept = max_emit - 1;
        let boundary = drafts[max_accept]; // matched draft at clamped row
        let (accepted, bonus) =
            clamp_accept_to_budget(walk_accepted, walk_bonus, max_emit, boundary);
        assert_eq!(accepted, 1);
        assert_eq!(bonus, 20); // drafts[1], not argmax, not walk_bonus
        let step = lower_chain(&drafts, accepted, bonus);
        assert_eq!(step.emit.as_slice(), &[10, 20]);
        assert_eq!(step.cap_emit(max_emit).emit.as_slice(), &[10, 20]);
    }

    #[test]
    fn one_token_emit_shape() {
        // max_emit==1 path: proposed=0, accepted=0, emit=[token], next_seed=token.
        let step = SpecStep::new(std::iter::once(42u32), 42, 0, 0);
        assert_eq!(step.emit.as_slice(), &[42]);
        assert_eq!(step.accepted, 0);
        assert_eq!(step.proposed, 0);
        assert_eq!(step.next_seed, 42);
        let capped = step.cap_emit(1);
        assert_eq!(capped.emit.as_slice(), &[42]);
        assert_eq!(capped.accepted, 0);
    }

    // ── Task 4: generic first-token sampling seams ─────────────────────────
    //
    // Prefill first token (Speculator::prefill after set_sampling) and
    // step_one_token share the same already-configured sampling semantics:
    //   sample_temp > 1e-6 → naive_sample_chain empty-draft bonus draw
    //   sample_temp ≤ 1e-6 → last_argmax / llama::argmax (byte-identical)
    // set_sampling reseeds rng_state to 0x13579BDF before the first draw.

    /// temp≈0 first token is the seed-row argmax — byte-identical to greedy
    /// `naive_sample_chain(..., temp=0, empty drafts)` and `llama::argmax`.
    /// Mirrors prefill `sample_temp <= 1e-6 → last_argmax` and step_one_token
    /// greedy branch.
    #[test]
    fn generic_first_token_temp_zero_is_argmax_byte_identical() {
        let vocab = 5usize;
        // Distinct logits; argmax must be index 3 (highest).
        let logits = [1.0f32, -2.0, 0.5, 4.25, 4.0];
        assert_eq!(logits.len(), vocab);

        // Prefill greedy keeps last_argmax; chain temp=0 is the same fold.
        let last_argmax = crate::llama::argmax(&logits);
        assert_eq!(last_argmax, 3);

        let mut rng = 0xDEAD_BEEF_u64;
        let (accepted, bonus) =
            crate::ddtree::naive_sample_chain(&logits, &[], vocab, 0.0, &mut rng);
        assert_eq!(accepted, 0, "empty-draft chain accepts nothing");
        assert_eq!(
            bonus, last_argmax,
            "temp=0 empty-draft bonus == last_argmax"
        );
        // temp=0 does not consume RNG (argmax fold only) — prefill greedy path
        // likewise leaves rng_state untouched.
        assert_eq!(rng, 0xDEAD_BEEF_u64);

        // Tie-break: first strict-max wins (matches GPU kernel `>` fold).
        let tied = [1.0f32, 5.0, 5.0, 0.0];
        assert_eq!(crate::llama::argmax(&tied), 1);
        let mut rng_tie = 1u64;
        let (_, tbonus) =
            crate::ddtree::naive_sample_chain(&tied, &[], tied.len(), 0.0, &mut rng_tie);
        assert_eq!(tbonus, 1);
        assert_eq!(rng_tie, 1);

        // Threshold gate used by prefill / step_one_token: ≤ 1e-6 is greedy.
        // (naive_sample_chain itself treats any temp>0 as softmax — the gate
        // lives in the caller, which we pin by asserting argmax equivalence.)
        const GREEDY_TEMP: f32 = 1e-6;
        assert!(GREEDY_TEMP <= 1e-6);
        assert!(0.0f32 <= 1e-6);
    }

    /// temp>0 first token uses the same empty-draft `naive_sample_chain` draw
    /// as later target tokens (`step_one_token` / prefill after set_sampling).
    /// set_sampling reseeds to 0x13579BDF; fixed seed ⇒ deterministic selection
    /// and in-place RNG advancement shared with subsequent steps.
    #[test]
    fn generic_first_token_temp_gt_zero_seeded_sample_advances_rng() {
        let vocab = 4usize;
        // Non-degenerate logits under temp sampling (would be last_logits from
        // SpecAdvance::Ready on the LLaMA/generic prefill path).
        let logits = [0.0f32, 2.0, 0.5, -1.0];
        let temp = 0.8f32;
        assert!(temp > 1e-6, "must take the sample branch, not last_argmax");
        // Matches GenericDflashSpeculator::set_sampling reseed + construction default.
        let seed = 0x13579BDF_u64;

        let mut rng_a = seed;
        let (acc_a, tok_a) =
            crate::ddtree::naive_sample_chain(&logits, &[], vocab, temp, &mut rng_a);
        assert_eq!(acc_a, 0, "first-token empty-draft never accepts drafts");
        assert!(
            (tok_a as usize) < vocab,
            "sampled token {tok_a} out of vocab {vocab}"
        );
        // RNG must advance (xorshift_unit mutates state on every draw).
        assert_ne!(rng_a, seed, "empty-draft sample must advance rng_state");

        // Same seed → identical draw and identical post-state (deterministic).
        let mut rng_b = seed;
        let (acc_b, tok_b) =
            crate::ddtree::naive_sample_chain(&logits, &[], vocab, temp, &mut rng_b);
        assert_eq!(acc_b, 0);
        assert_eq!(tok_b, tok_a);
        assert_eq!(rng_b, rng_a);

        // Stream progresses from the advanced state (not a constant of logits).
        // Later step_one_token / chain verify continue from this rng_state.
        let mut rng_c = rng_a;
        let (_acc_c, _tok_c) =
            crate::ddtree::naive_sample_chain(&logits, &[], vocab, temp, &mut rng_c);
        assert_ne!(rng_c, rng_a, "second draw continues the RNG stream");

        // temp≈0 branch remains last_argmax / llama::argmax (byte-identical).
        let last_argmax = crate::llama::argmax(&logits);
        assert_eq!(last_argmax, 1);
        let mut rng_g = seed;
        let (_, gbonus) = crate::ddtree::naive_sample_chain(&logits, &[], vocab, 0.0, &mut rng_g);
        assert_eq!(gbonus, last_argmax);
        assert_eq!(rng_g, seed, "greedy first token must not advance rng");
    }
}
