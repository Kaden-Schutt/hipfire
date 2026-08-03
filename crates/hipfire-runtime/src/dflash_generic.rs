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

/// Allocation boundaries exposed for deterministic generic-DFlash rollback tests.
#[cfg(feature = "dflash-fault-inject")]
#[derive(Clone, Copy)]
pub enum GenericDflashConstructionStage {
    DraftWeights,
    DraftScratch,
    VerifyScratch,
    TargetWeights,
    TargetKv,
    VerifyScratchAllocation(usize),
    TargetWeightsAllocation(usize),
    TargetKvAllocation(usize),
    ParoWeightUpload(usize),
    DsparkAllocation(usize),
    AwqScaleUpload(usize),
    F32KvAllocation(usize),
    /// Fault injection for Qwen auxiliary construction (DeltaNet,
    /// scratch, bundle build transaction staging).
    QwenAuxAllocation(usize),
}

#[cfg(feature = "dflash-fault-inject")]
impl GenericDflashConstructionStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::DraftWeights => "draft weights",
            Self::DraftScratch => "draft scratch",
            Self::VerifyScratch => "verify scratch",
            Self::TargetWeights => "target weights",
            Self::TargetKv => "target KV",
            Self::VerifyScratchAllocation(_) => "verify scratch allocation",
            Self::TargetWeightsAllocation(_) => "target weights allocation",
            Self::TargetKvAllocation(_) => "target KV allocation",
            Self::ParoWeightUpload(_) => "Paro weight upload",
            Self::DsparkAllocation(_) => "DSpark allocation",
            Self::AwqScaleUpload(_) => "AWQ scale upload",
            Self::F32KvAllocation(_) => "F32 KV allocation",
            Self::QwenAuxAllocation(_) => "Qwen aux allocation",
        }
    }

    fn code(self) -> usize {
        match self {
            Self::DraftWeights => 1,
            Self::DraftScratch => 2,
            Self::VerifyScratch => 3,
            Self::TargetWeights => 4,
            Self::TargetKv => 5,
            Self::VerifyScratchAllocation(_) => 6,
            Self::TargetWeightsAllocation(_) => 7,
            Self::TargetKvAllocation(_) => 8,
            Self::ParoWeightUpload(_) => 9,
            Self::DsparkAllocation(_) => 10,
            Self::AwqScaleUpload(_) => 11,
            Self::F32KvAllocation(_) => 12,
            Self::QwenAuxAllocation(_) => 13,
        }
    }
}

#[cfg(feature = "dflash-fault-inject")]
mod generic_dflash_fault_inject {
    use super::GenericDflashConstructionStage;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());
    static STAGE: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_TARGET: AtomicUsize = AtomicUsize::new(usize::MAX);
    static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            STAGE.store(0, Ordering::SeqCst);
            ALLOCATION_TARGET.store(usize::MAX, Ordering::SeqCst);
            ALLOCATION_COUNT.store(0, Ordering::SeqCst);
        }
    }

    pub(super) fn with_fault<T>(stage: GenericDflashConstructionStage, f: impl FnOnce() -> T) -> T {
        let _lock = LOCK.lock().expect("generic DFlash fault lock poisoned");
        STAGE.store(stage.code(), Ordering::SeqCst);
        ALLOCATION_TARGET.store(
            match stage {
                GenericDflashConstructionStage::VerifyScratchAllocation(allocation)
                | GenericDflashConstructionStage::TargetWeightsAllocation(allocation)
                | GenericDflashConstructionStage::TargetKvAllocation(allocation)
                | GenericDflashConstructionStage::ParoWeightUpload(allocation)
                | GenericDflashConstructionStage::DsparkAllocation(allocation)
                | GenericDflashConstructionStage::AwqScaleUpload(allocation)
                | GenericDflashConstructionStage::F32KvAllocation(allocation)
                | GenericDflashConstructionStage::QwenAuxAllocation(allocation) => allocation,
                _ => usize::MAX,
            },
            Ordering::SeqCst,
        );
        ALLOCATION_COUNT.store(0, Ordering::SeqCst);
        let _reset = Reset;
        f()
    }

    pub(super) fn after_stage(stage: GenericDflashConstructionStage) -> Result<(), String> {
        if matches!(
            stage,
            GenericDflashConstructionStage::VerifyScratchAllocation(_)
                | GenericDflashConstructionStage::TargetWeightsAllocation(_)
                | GenericDflashConstructionStage::TargetKvAllocation(_)
                | GenericDflashConstructionStage::ParoWeightUpload(_)
                | GenericDflashConstructionStage::DsparkAllocation(_)
                | GenericDflashConstructionStage::AwqScaleUpload(_)
                | GenericDflashConstructionStage::F32KvAllocation(_)
                | GenericDflashConstructionStage::QwenAuxAllocation(_)
        ) {
            return Ok(());
        }
        if STAGE.load(Ordering::SeqCst) == stage.code() {
            return Err(format!("test fault after generic DFlash {}", stage.label()));
        }
        Ok(())
    }

    pub(super) fn after_allocation(stage: GenericDflashConstructionStage) -> Result<(), String> {
        if STAGE.load(Ordering::SeqCst) == stage.code()
            && ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst)
                == ALLOCATION_TARGET.load(Ordering::SeqCst)
        {
            return Err(format!("test fault after generic DFlash {}", stage.label()));
        }
        Ok(())
    }
}

/// Run `f` with a deterministic fault injected at one generic-DFlash
/// construction boundary. Available only in builds with
/// `dflash-fault-inject` enabled.
#[cfg(feature = "dflash-fault-inject")]
pub fn with_generic_dflash_construction_fault<T>(
    stage: GenericDflashConstructionStage,
    f: impl FnOnce() -> T,
) -> T {
    generic_dflash_fault_inject::with_fault(stage, f)
}

#[cfg(feature = "dflash-fault-inject")]
pub fn generic_dflash_construction_boundary(
    stage: GenericDflashConstructionStage,
) -> Result<(), String> {
    generic_dflash_fault_inject::after_stage(stage)?;
    Ok(())
}

#[cfg(feature = "dflash-fault-inject")]
pub fn generic_dflash_allocation_boundary(
    stage: GenericDflashConstructionStage,
) -> Result<(), String> {
    generic_dflash_fault_inject::after_allocation(stage)
}

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
}

/// Constructor-local owner for generic DFlash resources. GPU buffers have no
/// Drop, so this keeps each completed allocation published until construction
/// succeeds and frees them explicitly on every error path.
struct GenericDflashStaging {
    weights: Option<DflashWeights>,
    scratch: Option<DflashScratch>,
    verify_scratch: Option<Box<dyn SpecScratch>>,
}

impl GenericDflashStaging {
    fn free_gpu(&mut self, gpu: &mut Gpu) {
        if let Some(verify_scratch) = self.verify_scratch.take() {
            verify_scratch.free(gpu);
        }
        if let Some(scratch) = self.scratch.take() {
            scratch.free_gpu(gpu);
        }
        if let Some(weights) = self.weights.take() {
            weights.free_gpu(gpu);
        }
    }

    fn into_speculator(
        mut self,
        config: DflashConfig,
        block_size: usize,
        ctx_capacity: usize,
        tree: TreeMode,
    ) -> Box<dyn Speculator> {
        Box::new(GenericDflashSpeculator {
            weights: self.weights.take().expect("staged generic DFlash weights"),
            scratch: self.scratch.take().expect("staged generic DFlash scratch"),
            config,
            target_hidden_host: Vec::new(),
            verify_scratch: Some(
                self.verify_scratch
                    .take()
                    .expect("staged generic DFlash verify scratch"),
            ),
            block_size,
            ctx_capacity,
            tree,
            rng_state: 0x13579BDF,
        })
    }
}

impl GenericDflashSpeculator {
    fn num_extract(&self) -> usize {
        self.config.num_extract()
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
        let accept_len = accepted_nodes.len();

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
        let emit = accepted_nodes
            .iter()
            .map(|&ni| tree.nodes[ni].token)
            .chain(std::iter::once(bonus));
        Ok(SpecStep::new(emit, bonus, depth, accept_len))
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
        // daemon's position += emit.len().
        let emit = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::once(bonus));
        Ok(SpecStep::new(emit, bonus, drafts.len(), accepted))
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

impl Speculator for GenericDflashSpeculator {
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
        // host buffer in one pass. The daemon's central cache-miss lifecycle has
        // already performed the authoritative total reset. Capture only fires
        // when the target's `dflash_extract_layers()` is `Some` (set at build).
        let adv = target.spec_advance_cold_start(
            gpu,
            fill_tokens,
            start_pos,
            !cache_hit,
            abort,
            Some(&mut self.target_hidden_host),
        )?;
        let first_token = match adv {
            SpecAdvance::Ready { last_argmax } => last_argmax,
            SpecAdvance::Aborted => return Ok(PrefillOutcome::Aborted),
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
    ) -> Result<SpecStep, String> {
        let b = self.config.block_size;
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
            return self.step_tree(gpu, target, position, seed, &draft_logits, vocab, b, temp);
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

            target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
            return self.finish_chain(&block_hidden, &drafts, accepted, bonus, position, ne, h);
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

        // Fix the target state to the committed prefix [seed, accepted…, bonus].
        // For a stateless target this is a no-op; for a recurrent one it restores
        // the snapshot verify_block stashed in `vs`. accept_len passed is the
        // number of accepted DRAFTS (block[1..=accept_len] accepted).
        target.commit_prefix(gpu, &block, accepted, position, vs.as_mut())?;
        self.finish_chain(&block_hidden, &drafts, accepted, bonus, position, ne, h)
    }

    fn advance_forced(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        position: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        // Forced continuations become normal draft context for the following
        // window, so capture their target-hidden rows rather than advancing only
        // the target KV/recurrent state.
        target.spec_advance(
            gpu,
            tokens,
            position,
            abort,
            Some(&mut self.target_hidden_host),
        )
    }

    fn reset(&mut self, gpu: &mut Gpu) {
        let _ = self.reset_checked(gpu);
    }

    fn reset_checked(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset: clear the cumulative host shadow + the draft's
        // upload/projection tracking. The target's KV/recurrent reset is the
        // daemon's job (it owns the bundle).
        self.target_hidden_host.clear();
        self.scratch.reset_upload_tracking();
        Ok(())
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
    let mut staged = GenericDflashStaging {
        weights: None,
        scratch: None,
        verify_scratch: None,
    };
    let weights = match DflashWeights::load(gpu, &draft_hfq, &config) {
        Ok(weights) => weights,
        Err(error) => return Err(format!("{error}")),
    };
    staged.weights = Some(weights);
    #[cfg(feature = "dflash-fault-inject")]
    {
        if let Err(error) =
            generic_dflash_construction_boundary(GenericDflashConstructionStage::DraftWeights)
        {
            staged.free_gpu(gpu);
            return Err(error);
        }
    }
    let block_size = config.block_size;
    // L3: F16 drafts (dflash_convert) → has_mq=false → DflashScratch::new.
    // new_with_mq only for an MQ-quantized draft.
    let scratch = if staged
        .weights
        .as_ref()
        .expect("staged generic DFlash weights")
        .has_mq
    {
        DflashScratch::new_with_mq(gpu, &config, block_size, ctx_capacity, true)
            .map_err(|e| format!("{e}"))
    } else {
        DflashScratch::new(gpu, &config, block_size, ctx_capacity).map_err(|e| format!("{e}"))
    };
    let scratch = match scratch {
        Ok(scratch) => scratch,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(error);
        }
    };
    staged.scratch = Some(scratch);
    #[cfg(feature = "dflash-fault-inject")]
    {
        if let Err(error) =
            generic_dflash_construction_boundary(GenericDflashConstructionStage::DraftScratch)
        {
            staged.free_gpu(gpu);
            return Err(error);
        }
    }
    let _ = draft_hfq;

    // Tell the target which residual-hidden layers to capture (the drafter's
    // target_layer_ids), and mint the per-target verify scratch.
    target.set_dflash_extract_layers(config.target_layer_ids.clone());
    let verify_scratch = match target.new_spec_scratch(gpu, block_size) {
        Ok(verify_scratch) => verify_scratch,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(error);
        }
    };
    staged.verify_scratch = Some(verify_scratch);
    #[cfg(feature = "dflash-fault-inject")]
    {
        if let Err(error) =
            generic_dflash_construction_boundary(GenericDflashConstructionStage::VerifyScratch)
        {
            staged.free_gpu(gpu);
            return Err(error);
        }
    }

    Ok(staged.into_speculator(
        config,
        block_size,
        ctx_capacity,
        TreeMode::from_flags(&gpu.flags),
    ))
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
}
