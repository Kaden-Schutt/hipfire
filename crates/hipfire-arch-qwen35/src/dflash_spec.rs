// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5 DFlash / DDTree speculative-decode state and `Speculator` impl.
//!
//! Contents: [`DflashState`] (the loaded draft weights/scratch + target
//! snapshot/tape + optional [`DdtreeState`]), [`load_dflash_state`] (its
//! load-time constructor), the [`DflashSpeculator`] impl (which owns
//! `DflashState` + the divergent-render checkpoint ring) behind the arch-generic
//! [`Speculator`] trait, and [`build_dflash_speculator`] (its env-resolving
//! constructor). All types here are qwen35 + runtime types — no loader types —
//! so the loader only calls in; it never owns the DFlash mechanics.

use crate::qwen35::{self, DeltaNetState, LayerType, Qwen35Config};
use crate::speculative::{
    apply_eviction_retain_to_draft, scatter_hidden_block_to_interleaved,
    seed_target_hidden_from_prompt_abortable, seed_target_hidden_suffix_abortable,
    spec_step_ddtree_batched, spec_step_dflash, DdtreeScratch, DeltaNetSnapshot, GdnTape,
    HiddenStateRingBuffer, ModelSlot, SpecStepResult, VerifyScratch,
};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{
    EvictRetain, PrefillOutcome, SpecGrammar, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::path::Path;

// ─── DDTree side state ────────────────────────────────────────────────

/// Side state for DDTree-mode speculative decoding.
pub struct DdtreeState {
    pub post_seed_snap: DeltaNetSnapshot,
    pub scratch: DdtreeScratch,
    pub budget: usize,
    pub topk: usize,
}

// ─── DFlash state ─────────────────────────────────────────────────────

/// Optional DFlash speculative-decoding state.
pub struct DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub hidden_rb: HiddenStateRingBuffer,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    pub target_hidden_host: Vec<f32>,
    pub ctx_capacity: usize,
    pub block_size: usize,
    pub ddtree: Option<DdtreeState>,
}

// ─── DFlash state load ────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
/// Default ceiling for the DFlash draft's context-indexed structures
/// (`target_hidden` [L × extract×hidden], the per-layer K/V caches, the
/// hidden ring, `mq_x_rot`, and the host hidden log). Serve loads default
/// `max_seq` to 32768+, which sized ALL of these to 32K rows — on a 27B
/// target with a 5-layer-extract MQ4 draft that is ~11 GB of draft-side
/// VRAM, vs ~1.4 GB at the ≤4K contexts DFlash benches actually run. The
/// draft only affects acceptance rate (verify is target-gated), so a
/// request that outgrows the cap simply falls back to AR in the daemon —
/// emitted tokens are never at risk. `HIPFIRE_DFLASH_CTX_CAP=0` opts out
/// (legacy uncapped behaviour); any other value overrides the ceiling.
pub const DEFAULT_DFLASH_CTX_CAP: usize = 8192;

pub fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut Gpu,
    // DDTree draft tuning forwarded by the loader from the unified spec config
    // (CLI `--ddtree-budget` / `--ddtree-topk`). Env wins, else these, else default.
    ddtree_budget_param: Option<usize>,
    ddtree_topk_param: Option<usize>,
) -> Result<DflashState, String> {
    let requested_ctx = ctx_capacity;
    let ctx_capacity = match std::env::var("HIPFIRE_DFLASH_CTX_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(0) => ctx_capacity, // explicit opt-out: legacy uncapped
        Some(cap) => ctx_capacity.min(cap),
        None => ctx_capacity.min(DEFAULT_DFLASH_CTX_CAP),
    };
    if ctx_capacity < requested_ctx {
        eprintln!(
            "  DFlash draft ctx capped: {} -> {} rows (draft-side VRAM scales with this; \
             HIPFIRE_DFLASH_CTX_CAP=0 for uncapped, or set a larger cap)",
            requested_ctx, ctx_capacity
        );
    }
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let draft_weights =
        DflashWeights::load(gpu, &draft_hfq, &draft_config).map_err(|e| format!("{e}"))?;
    let block_size = draft_config.block_size;
    // DDTree verify batches up to `budget + 1` slots (seed + budget nodes), which
    // can exceed the chain block_size+1. Size verify_scratch / GdnTape / hidden
    // staging for the larger of the two so ddtree-mode serve doesn't overflow
    // ("verify_scratch max_n < b" panic). budget=0 ⇒ chain-only, unchanged.
    // Resolved through FeatureFlags (env override) so the ddtree budget has a
    // single parser shared with the dense path — env wins, else the CLI param,
    // else 0 (chain-only). An explicit `HIPFIRE_DDTREE_BUDGET=0` reads as None
    // (unset) here and falls through to the param, matching the dense semantics.
    let ddtree_budget: usize = gpu.flags.ddtree_budget.or(ddtree_budget_param).unwrap_or(0);
    let max_n = (block_size + 1).max(ddtree_budget + 1);
    // `with_mq` allocates the FWHT rotation scratch (mq_x_rot) that
    // `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft weights. The carrier
    // refactor regressed this to the `with_mq=false` `::new` constructor →
    // panic "MQ4 dispatch requires mq_x_rot scratch" on any MQ-quantized draft.
    let draft_scratch = DflashScratch::new_with_mq(
        gpu,
        &draft_config,
        block_size,
        ctx_capacity,
        draft_weights.has_mq,
    )
    .map_err(|e| format!("{e}"))?;
    let _ = draft_hfq;
    // The hidden-ring STAGING buffers must hold one prefill chunk. Verify
    // cycles seed only `max_n` (= block_size+1) rows, but the prompt seed
    // (`seed_target_hidden_from_prompt_abortable`) prefills the prompt in
    // chunks of up to `PREFILL_MAX_BATCH` and captures each into staging via
    // `write_rows_to_staging` (whose `n <= max_batch` guard is a debug_assert,
    // silent in release). Sizing staging to only `max_n` overflowed the d2d
    // copy on any prompt longer than block_size+1 tokens. Size it to the
    // larger of the two so both paths fit.
    let staging_max_batch = max_n.max(qwen35::PREFILL_MAX_BATCH);
    let hidden_rb = HiddenStateRingBuffer::new(
        gpu,
        target_config.n_layers,
        draft_config.num_extract(),
        target_config.dim,
        ctx_capacity,
        staging_max_batch,
    )
    .map_err(|e| format!("HiddenStateRingBuffer::new: {e}"))?;
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        max_n,
        target_config.dim,
        target_config.vocab_size,
        hidden_k,
        target_config,
    )
    .map_err(|e| format!("VerifyScratch::with_prefill: {e}"))?;
    let target_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
        .map_err(|e| format!("DeltaNetSnapshot::new_for: {e}"))?;
    let gdn_tape = GdnTape::new_for_config(gpu, target_config, max_n)
        .map_err(|e| format!("GdnTape::new_for_config: {e}"))?;
    let target_hidden_host = vec![0.0f32; ctx_capacity * target_config.dim];
    // DDTree (budget read once above, used for scratch sizing).
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = gpu.flags.ddtree_topk.or(ddtree_topk_param).unwrap_or(4);
        let post_seed_snap =
            DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("{e}"))?;
        let scratch = DdtreeScratch::new(gpu, ddtree_budget)
            .map_err(|e| format!("DdtreeScratch::new: {e}"))?;
        Some(DdtreeState {
            post_seed_snap,
            scratch,
            budget: ddtree_budget,
            topk,
        })
    } else {
        None
    };
    Ok(DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ctx_capacity,
        block_size,
        ddtree,
    })
}

// ─── DflashSpeculator ───────────────────────────────────────────────────

/// Lower a qwen35 `SpecStepResult` onto the arch-generic `SpecStep`.
///
/// The daemon-called `spec_step_*` build `committed = [seed, drafts.., bonus]`,
/// so `committed[1..]` is exactly the daemon's `committed_tail` (the tokens
/// emitted this window) and its length is `accepted + 1` — which is why the
/// unified loop advances `position` by `emit.len()`.
fn lower_qwen35(r: SpecStepResult) -> SpecStep {
    SpecStep::new(
        r.committed[1..].iter().copied(),
        r.bonus_token,
        r.drafted.len(),
        r.accepted,
    )
}

/// DFlash / DDTree speculator: wraps the qwen35 `spec_step_*` chain/tree
/// kernels behind the arch-generic [`Speculator`] trait. Chain-vs-tree is an
/// internal detail resolved at build (`ddtree` presence comes from the loaded
/// `DflashState`).
///
/// Owns the `DflashState` moved out of `LoadedModel.dflash`, plus the divergent-
/// render DeltaNet checkpoint ring folded in from `LoadedModel.dflash_checkpoints`.
pub struct DflashSpeculator {
    df: DflashState,
    rng_state: u64,
    /// Per-request sampling, set via `set_sampling` before each step loop and
    /// applied in the chain-mode `spec_step_dflash` branch of `step`. Default
    /// greedy (temp 0 / top_p 1 / top_k 0 / cactus 0) → argmax-accept, the
    /// historical DFlash posture, so an unconfigured speculator (or the
    /// greedy-only DDTree branches) decode greedily. Mirrors spec-graph's old
    /// inline `generate_dflash` call, which threaded the request temp/top_p/top_k
    /// into the same four `spec_step_dflash` args.
    sample_temp: f32,
    sample_top_p: f32,
    sample_top_k: usize,
    sample_cactus: f32,
    /// Divergent-render checkpoint ring. Populated by `prefill`'s seed when
    /// `resume_enabled`; freed on `reset`/`free`.
    checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    resume_enabled: bool,
    ck_interval: usize,
    ck_cap: usize,
}

impl DflashSpeculator {
    /// `resume_enabled`/`ck_interval`/`ck_cap` mirror the daemon's
    /// `ckpt_resume_enabled()`/`ckpt_interval()`/`ckpt_max()` — passed in by
    /// `build_dflash_speculator` so `new` itself is env-free (and unit-testable).
    pub fn new(df: DflashState, resume_enabled: bool, ck_interval: usize, ck_cap: usize) -> Self {
        Self {
            df,
            // Same fixed seed the daemon's DFlash loop used. `set_sampling`
            // re-seeds it to this value per request (matching spec-graph's local
            // `let mut rng_state = 0x13579BDF` per `generate_dflash` call) so a
            // sampled request is deterministic given its seed; greedy decode does
            // not consume it.
            rng_state: 0x13579BDF,
            // Greedy by default until a request calls `set_sampling`.
            sample_temp: 0.0,
            sample_top_p: 1.0,
            sample_top_k: 0,
            sample_cactus: 0.0,
            checkpoints: Vec::new(),
            resume_enabled,
            ck_interval,
            ck_cap,
        }
    }
}

impl Speculator for DflashSpeculator {
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
        resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Mirror the daemon's pre-seed drafter setup (generate_dflash 4064-4072):
        // always clear the host hidden buffer; on a full prefill drop the draft's
        // upload/projection tracking. On a cache HIT it is PRESERVED so the draft
        // reuses the cached [0..start_pos] projections and only projects the suffix.
        self.df.target_hidden_host.clear();
        if !cache_hit {
            self.df.draft_scratch.reset_upload_tracking();
        }

        // Seed the target's hidden state into the drafter ring (chunked prefill
        // with hidden extraction). Cache hit → seed only the suffix from
        // `prefill_start`, reusing the prior turn's KV + recurrent state; miss →
        // seed the full prompt (the seed fn resets target state itself).
        let (ck_interval, ck_cap) = (self.ck_interval, self.ck_cap);
        let ckpt_sink = if self.resume_enabled {
            Some(&mut self.checkpoints)
        } else {
            None
        };
        let aborted = if cache_hit {
            seed_target_hidden_suffix_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                prefill_tokens,
                prefill_start,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        } else {
            seed_target_hidden_from_prompt_abortable(
                gpu,
                slot,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                prefill_tokens,
                abort,
                ckpt_sink,
                ck_interval,
                ck_cap,
            )
        }
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller resets conversation state + emits aborted/done; the slot
            // guard restores the target bundle on the way out.
            return Ok(PrefillOutcome::Aborted);
        }

        // Prime/extend the draft's GPU target_hidden buffer. On a hit, scatter
        // only the suffix rows at `prefill_start` (the prefix is preserved);
        // on a miss, scatter all prompt rows from 0.
        let (scatter_off, scatter_len) = if cache_hit {
            (prefill_start, prefill_tokens.len())
        } else {
            (0, prompt_tokens.len())
        };
        if let Err(e) = scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            scatter_off,
            scatter_len,
            scatter_len,
        ) {
            eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
        }
        self.df.draft_scratch.thlog.seed_prompt(prompt_tokens.len());
        if let Some(ckpt) = resume_from {
            // Divergent rows [ckpt..len) were just overwritten; drop the draft's
            // projection cursor so the first spec step re-projects from `ckpt`.
            self.df.draft_scratch.thlog.set_resume_checkpoint(ckpt);
        }

        // First emit = target argmax at the final prompt position (seed already
        // ran the per-token forward; scratch.logits holds the post-prompt logits).
        let first_logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        let first_token = first_logits
            .iter()
            .enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv {
                    (i as u32, v)
                } else {
                    (best, bv)
                }
            })
            .0;
        Ok(PrefillOutcome::Ready { first_token })
    }

    /// Temp>0 verify is distribution-correct only on the ddtree-batched arm
    /// (SWOR); chain mode is greedy, so a non-ddtree drafter must NOT receive
    /// temp>0 routing.
    fn supports_temp_verify(&self) -> bool {
        self.df.ddtree.is_some()
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        emitted: &[u32],
        _grammar: Option<&mut dyn SpecGrammar>,
        temp: f32,
    ) -> Result<SpecStep, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        // Two-way dispatch from the daemon's old generate_dflash loop: DDTree-
        // batched (SWOR) verify when a tree is configured, else chain-mode DFlash.
        // The grammar arg is ignored — qwen35 enforces tool-call grammar post-hoc
        // in the daemon.
        let result = if let Some(dd) = self.df.ddtree.as_mut() {
            spec_step_ddtree_batched(
                gpu,
                slot,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                &mut self.df.target_snap,
                &mut dd.post_seed_snap,
                &mut self.df.gdn_tape,
                &dd.scratch,
                &self.df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                dd.budget,
                dd.topk,
                // Request temperature → distribution-preserving SWOR verify at
                // temp>0 (greedy/argmax at temp 0). The ddtree-batched arm is the
                // only DFlash mode with sampled verify; the chain below stays
                // greedy, so `supports_temp_verify` gates serve routing to ddtree.
                temp,
                &mut self.rng_state,
            )
        } else {
            spec_step_dflash(
                gpu,
                slot,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &mut self.df.hidden_rb,
                &mut self.df.target_hidden_host,
                &mut self.df.target_snap,
                &self.df.verify_scratch,
                position,
                seed,
                None, // ctx_slice = full history
                Some(&mut self.df.gdn_tape),
                // Sampling threaded from the request via `set_sampling` (#477
                // merge re-wire). These four positions reproduce spec-graph's old
                // inline `generate_dflash` call verbatim: temp 0 ⇒ greedy/argmax;
                // temp>0 ⇒ lossless rejection sampling with the IDENTICAL
                // (top_k,top_p) nucleus truncation on draft + target. The DDTree
                // branches above stay greedy (tree-verify is greedy by
                // construction) and ignore these.
                self.sample_temp,
                self.sample_top_p, // top_p (1.0 = no truncation)
                self.sample_top_k, // top_k (0 = top_p-only)
                &mut self.rng_state,
                None, // block_size override
                None, // ngram_cache
                emitted,
                self.sample_cactus, // 0.0 = lossless; >0 = deliberately lossy
                None,               // pld_spine
                1.0_f32,            // repeat_penalty (off)
                0,                  // repeat_window
            )
        };

        result.map(lower_qwen35).map_err(|e| e.to_string())
    }

    fn on_evict(&mut self, gpu: &mut Gpu, retain: &EvictRetain) -> Result<(), String> {
        // Compact the drafter's cached target-hidden rows to match the target KV
        // after the FlashCASK eviction the daemon already applied to the target.
        let ne = self.df.draft_config.num_extract();
        let h = self.df.draft_config.hidden;
        apply_eviction_retain_to_draft(
            gpu,
            &mut self.df.draft_scratch,
            &retain.retain_mask,
            ne,
            h,
            retain.pre_phys,
        )
        .map_err(|e| e.to_string())
    }

    fn reset(&mut self, gpu: &mut Gpu) {
        // Drafter-local reset: invalidate cached suffix projections and free the
        // divergent-render checkpoint ring (the target KV/recurrent reset is the
        // daemon's job — it owns the bundle).
        self.df.draft_scratch.reset_upload_tracking();
        for (_, snap) in self.checkpoints.drain(..) {
            snap.free_gpu(gpu);
        }
    }

    fn block_size(&self) -> usize {
        self.df.block_size
    }

    fn ctx_capacity(&self) -> usize {
        self.df.ctx_capacity
    }

    fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints.iter().map(|(p, _)| *p).collect()
    }

    fn rewind_to(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
    ) -> Result<usize, String> {
        // Restore the target's DeltaNet recurrent state to the checkpoint at
        // `position` and drop the now-stale tail of the ring (mirrors the old
        // divergent-render resume at generate_dflash 4021-4036). Caller rewinds
        // seq_pos / conversation_tokens to match.
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        if let Some(idx) = self.checkpoints.iter().rposition(|(p, _)| *p == position) {
            let _ = self.checkpoints[idx].1.restore_to(&mut slot.dn_state, gpu);
            for (_, snap) in self.checkpoints.drain(idx + 1..) {
                snap.free_gpu(gpu);
            }
        }
        Ok(position)
    }

    fn set_sampling(&mut self, temp: f32, top_p: f32, top_k: usize, cactus_delta: f32) {
        // Store the request's sampling config for the chain-mode branch of
        // `step`. Re-seed the RNG to the same fixed value spec-graph used per
        // `generate_dflash` call (a fresh `let mut rng_state = 0x13579BDF`), so a
        // sampled request is deterministic given its seed and two identical
        // requests in one session produce identical output — preserving
        // spec-graph's behavior rather than letting the seed drift across turns.
        self.sample_temp = temp;
        self.sample_top_p = top_p;
        self.sample_top_k = top_k;
        self.sample_cactus = cactus_delta;
        self.rng_state = 0x13579BDF;
    }

    fn requires_greedy(&self) -> bool {
        // DFlash supports faithful temp>0 decode via lossless rejection sampling
        // (set_sampling + the sampled `spec_step_dflash` path), so it does NOT
        // require greedy verification. The daemon dispatch consults this (via
        // `spec_can_sample`) to decide whether a temp>0 request may take the spec
        // path or must fall to AR — returning `false` here is what lets sampled
        // DFlash engage while greedy-only drafters (MTP/n-gram) stay on AR.
        false
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        // Mirrors the `unload_model` dflash teardown + the checkpoint-ring free.
        let DflashSpeculator {
            df, checkpoints, ..
        } = *self;
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Construct the DFlash speculator from a freshly-loaded `DflashState`, resolving
/// the env config the daemon's old `generate_dflash` read inline: checkpoint
/// resume (`HIPFIRE_DFLASH_CKPT_RESUME` + no-eviction) and interval/cap
/// (`HIPFIRE_CACHE_CKPT_INTERVAL`/`_MAX`, matching the daemon's
/// `ckpt_interval()`/`ckpt_max()` defaults). Called once at load.
pub fn build_dflash_speculator(df: DflashState, eviction_is_none: bool) -> Box<dyn Speculator> {
    let resume_enabled = hipfire_config::developer_var("HIPFIRE_DFLASH_CKPT_RESUME").ok().as_deref() != Some("0")
        && eviction_is_none;
    let ck_interval = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize)
        .max(256);
    let ck_cap = hipfire_config::developer_var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    Box::new(DflashSpeculator::new(
        df,
        resume_enabled,
        ck_interval,
        ck_cap,
    ))
}
