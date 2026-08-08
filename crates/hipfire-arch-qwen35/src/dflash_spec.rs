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

use crate::qwen35::{self, DeltaNetState, Qwen35Config};
use crate::speculative::{
    apply_eviction_retain_to_draft, apply_host_nucleus, sample_categorical,
    scatter_hidden_block_to_interleaved, seed_target_hidden_from_prompt_abortable,
    seed_target_hidden_suffix_abortable, softmax_temp_into, spec_step_ddtree_batched,
    spec_step_dflash, xorshift_next_unit, DdtreeScratch, DeltaNetSnapshot, GdnTape,
    HiddenStateRingBuffer, ModelSlot, SpecStepResult, VerifyScratch,
};
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{
    EvictRetain, PrefillOutcome, SpecAdvance, SpecGrammar, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::Gpu;
use std::path::Path;

#[derive(Clone, Copy)]
pub(crate) enum DflashConstructionStage {
    DraftWeights,
    DraftScratch,
    HiddenRing,
    VerifyScratch,
    TargetSnapshot,
    GdnTape,
    DdtreeSnapshot,
    DdtreeScratch,
}

impl DflashConstructionStage {
    #[cfg(test)]
    const ALL: [Self; 8] = [
        Self::DraftWeights,
        Self::DraftScratch,
        Self::HiddenRing,
        Self::VerifyScratch,
        Self::TargetSnapshot,
        Self::GdnTape,
        Self::DdtreeSnapshot,
        Self::DdtreeScratch,
    ];

    #[cfg(test)]
    fn label(self) -> &'static str {
        match self {
            Self::DraftWeights => "draft weights",
            Self::DraftScratch => "draft scratch",
            Self::HiddenRing => "hidden ring",
            Self::VerifyScratch => "verify scratch",
            Self::TargetSnapshot => "target snapshot",
            Self::GdnTape => "GdnTape",
            Self::DdtreeSnapshot => "DDTree snapshot",
            Self::DdtreeScratch => "DDTree scratch",
        }
    }

    #[cfg(test)]
    fn code(self) -> usize {
        match self {
            Self::DraftWeights => 1,
            Self::DraftScratch => 2,
            Self::HiddenRing => 3,
            Self::VerifyScratch => 4,
            Self::TargetSnapshot => 5,
            Self::GdnTape => 6,
            Self::DdtreeSnapshot => 7,
            Self::DdtreeScratch => 8,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum DflashAllocationSite {
    DeltaNetSnapshot,
    GdnTape,
    DdtreeScratch,
    VerifyScratch,
    HiddenStateRing,
    PrefillBatchScratch,
}

#[cfg(test)]
impl DflashAllocationSite {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::DeltaNetSnapshot => "DeltaNetSnapshot",
            Self::GdnTape => "GdnTape",
            Self::DdtreeScratch => "DdtreeScratch",
            Self::VerifyScratch => "VerifyScratch",
            Self::HiddenStateRing => "HiddenStateRingBuffer",
            Self::PrefillBatchScratch => "PrefillBatchScratch",
        }
    }

    fn code(self) -> usize {
        match self {
            Self::DeltaNetSnapshot => 1,
            Self::GdnTape => 2,
            Self::DdtreeScratch => 3,
            Self::VerifyScratch => 4,
            Self::HiddenStateRing => 5,
            Self::PrefillBatchScratch => 6,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum DflashTestFault {
    AfterStage(DflashConstructionStage),
    AfterAllocation {
        site: DflashAllocationSite,
        allocation: usize,
    },
}

#[cfg(test)]
mod dflash_test_fault {
    use super::{DflashAllocationSite, DflashConstructionStage, DflashTestFault};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());
    static STAGE: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_SITE: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_TARGET: AtomicUsize = AtomicUsize::new(usize::MAX);
    static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            STAGE.store(0, Ordering::SeqCst);
            ALLOCATION_SITE.store(0, Ordering::SeqCst);
            ALLOCATION_TARGET.store(usize::MAX, Ordering::SeqCst);
            ALLOCATION_COUNT.store(0, Ordering::SeqCst);
        }
    }

    pub(super) fn with_fault<T>(fault: DflashTestFault, f: impl FnOnce() -> T) -> T {
        let _lock = LOCK.lock().expect("DFlash fault lock poisoned");
        match fault {
            DflashTestFault::AfterStage(stage) => STAGE.store(stage.code(), Ordering::SeqCst),
            DflashTestFault::AfterAllocation { site, allocation } => {
                ALLOCATION_SITE.store(site.code(), Ordering::SeqCst);
                ALLOCATION_TARGET.store(allocation, Ordering::SeqCst);
                ALLOCATION_COUNT.store(0, Ordering::SeqCst);
            }
        }
        let _reset = Reset;
        f()
    }

    pub(super) fn after_stage(stage: DflashConstructionStage) -> Result<(), String> {
        if STAGE.load(Ordering::SeqCst) == stage.code() {
            return Err(format!("test fault after {}", stage.label()));
        }
        Ok(())
    }

    pub(super) fn after_allocation(site: DflashAllocationSite) -> hip_bridge::HipResult<()> {
        if ALLOCATION_SITE.load(Ordering::SeqCst) == site.code()
            && ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst)
                == ALLOCATION_TARGET.load(Ordering::SeqCst)
        {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("test fault after {} allocation", site.label()),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn with_dflash_test_fault<T>(fault: DflashTestFault, f: impl FnOnce() -> T) -> T {
    dflash_test_fault::with_fault(fault, f)
}

#[cfg(test)]
pub(crate) fn dflash_test_after_allocation(
    site: DflashAllocationSite,
) -> hip_bridge::HipResult<()> {
    dflash_test_fault::after_allocation(site)
}

fn dflash_construction_boundary(stage: DflashConstructionStage) -> Result<(), String> {
    #[cfg(test)]
    dflash_test_fault::after_stage(stage)?;
    #[cfg(not(test))]
    let _ = stage;
    Ok(())
}

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

impl DflashState {
    fn free_gpu(self, gpu: &mut Gpu) {
        let DflashState {
            draft_weights,
            draft_scratch,
            hidden_rb,
            verify_scratch,
            target_snap,
            gdn_tape,
            ddtree,
            ..
        } = self;
        draft_weights.free_gpu(gpu);
        draft_scratch.free_gpu(gpu);
        for tensor in hidden_rb
            .layer_bufs
            .into_iter()
            .chain(hidden_rb.staging_bufs)
        {
            let _ = gpu.free_tensor(tensor);
        }
        verify_scratch.free_gpu(gpu);
        target_snap.free_gpu(gpu);
        gdn_tape.free_gpu(gpu);
        if let Some(DdtreeState {
            post_seed_snap,
            scratch,
            ..
        }) = ddtree
        {
            post_seed_snap.free_gpu(gpu);
            scratch.free_gpu(gpu);
        }
    }
}

/// Constructor-local owner for DFlash resources that have been allocated but
/// have not yet become published `DflashState`. It is explicitly drained on
/// every error path because GPU buffers deliberately have no global `Drop`.
struct DflashStateStaging {
    draft_weights: Option<DflashWeights>,
    draft_scratch: Option<DflashScratch>,
    hidden_rb: Option<HiddenStateRingBuffer>,
    verify_scratch: Option<VerifyScratch>,
    target_snap: Option<DeltaNetSnapshot>,
    gdn_tape: Option<GdnTape>,
    ddtree: Option<DdtreeStaging>,
}

struct DdtreeStaging {
    post_seed_snap: Option<DeltaNetSnapshot>,
    scratch: Option<DdtreeScratch>,
    budget: usize,
    topk: usize,
}

impl DflashStateStaging {
    fn free_gpu(&mut self, gpu: &mut Gpu) {
        if let Some(ddtree) = self.ddtree.take() {
            ddtree.free_gpu(gpu);
        }
        if let Some(gdn_tape) = self.gdn_tape.take() {
            gdn_tape.free_gpu(gpu);
        }
        if let Some(target_snap) = self.target_snap.take() {
            target_snap.free_gpu(gpu);
        }
        if let Some(verify_scratch) = self.verify_scratch.take() {
            verify_scratch.free_gpu(gpu);
        }
        if let Some(hidden_rb) = self.hidden_rb.take() {
            for tensor in hidden_rb
                .layer_bufs
                .into_iter()
                .chain(hidden_rb.staging_bufs)
            {
                let _ = gpu.free_tensor(tensor);
            }
        }
        if let Some(draft_scratch) = self.draft_scratch.take() {
            draft_scratch.free_gpu(gpu);
        }
        if let Some(draft_weights) = self.draft_weights.take() {
            draft_weights.free_gpu(gpu);
        }
    }

    fn into_state(
        mut self,
        draft_config: DflashConfig,
        target_hidden_host: Vec<f32>,
        ctx_capacity: usize,
        block_size: usize,
    ) -> DflashState {
        DflashState {
            draft_config,
            draft_weights: self.draft_weights.take().expect("staged draft weights"),
            draft_scratch: self.draft_scratch.take().expect("staged draft scratch"),
            hidden_rb: self.hidden_rb.take().expect("staged hidden ring"),
            verify_scratch: self.verify_scratch.take().expect("staged verify scratch"),
            target_snap: self.target_snap.take().expect("staged target snapshot"),
            gdn_tape: self.gdn_tape.take().expect("staged GdnTape"),
            target_hidden_host,
            ctx_capacity,
            block_size,
            ddtree: self.ddtree.take().map(DdtreeStaging::into_state),
        }
    }
}

impl DdtreeStaging {
    fn free_gpu(mut self, gpu: &mut Gpu) {
        if let Some(scratch) = self.scratch.take() {
            scratch.free_gpu(gpu);
        }
        if let Some(post_seed_snap) = self.post_seed_snap.take() {
            post_seed_snap.free_gpu(gpu);
        }
    }

    fn into_state(mut self) -> DdtreeState {
        DdtreeState {
            post_seed_snap: self.post_seed_snap.take().expect("staged DDTree snapshot"),
            scratch: self.scratch.take().expect("staged DDTree scratch"),
            budget: self.budget,
            topk: self.topk,
        }
    }
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
    // CASK eviction active for this load. Windowed draft mode refuses the
    // combination (the eviction rebuild re-projects rows the window has
    // already dropped) and falls back to Legacy — gather-compact over the
    // rings is a follow-up.
    eviction_active: bool,
) -> Result<DflashState, String> {
    let requested_ctx = ctx_capacity;
    // Open the draft container up-front: its declared SWA window is the
    // DEFAULT window (below), so the artifact must be parsed before the
    // windowed-vs-Legacy decision.
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;

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
    let ddtree_topk: usize = gpu.flags.ddtree_topk.or(ddtree_topk_param).unwrap_or(4);

    // Keep malformed draft metadata and host capacity failures out of the GPU
    // transaction. Once the first GPU allocation happens, every error below
    // routes through DflashStateStaging::free_gpu.
    if ctx_capacity == 0 {
        return Err("DFlash context capacity must be nonzero".into());
    }
    if block_size == 0 {
        return Err("DFlash draft block_size must be nonzero".into());
    }
    if draft_config.hidden != target_config.dim {
        return Err(format!(
            "DFlash draft hidden size {} does not match target dimension {}",
            draft_config.hidden, target_config.dim
        ));
    }
    if draft_config.num_target_layers != target_config.n_layers {
        return Err(format!(
            "DFlash draft expects {} target layers, target has {}",
            draft_config.num_target_layers, target_config.n_layers
        ));
    }
    if draft_config.target_layer_ids.is_empty()
        || draft_config
            .target_layer_ids
            .iter()
            .any(|&layer| layer >= target_config.n_layers)
    {
        return Err("DFlash draft target_layer_ids are empty or out of range".into());
    }
    let max_n = block_size
        .checked_add(1)
        .ok_or_else(|| "DFlash draft block_size overflows max_n".to_string())?
        .max(
            ddtree_budget
                .checked_add(1)
                .ok_or_else(|| "DDTree budget overflows max_n".to_string())?,
        );
    let target_hidden_len = ctx_capacity
        .checked_mul(target_config.dim)
        .ok_or_else(|| "DFlash target hidden host buffer size overflows".to_string())?;
    let hidden_k = target_config
        .dim
        .checked_next_power_of_two()
        .ok_or_else(|| "DFlash target dimension overflows hidden_k".to_string())?;
    let mut target_hidden_host = Vec::new();
    target_hidden_host
        .try_reserve_exact(target_hidden_len)
        .map_err(|e| format!("reserve DFlash target hidden host buffer: {e}"))?;
    target_hidden_host.resize(target_hidden_len, 0.0);

    let mut staged = DflashStateStaging {
        draft_weights: None,
        draft_scratch: None,
        hidden_rb: None,
        verify_scratch: None,
        target_snap: None,
        gdn_tape: None,
        ddtree: None,
    };
    let draft_weights = match DflashWeights::load(gpu, &draft_hfq, &draft_config) {
        Ok(weights) => weights,
        Err(error) => return Err(format!("DflashWeights::load: {error}")),
    };
    staged.draft_weights = Some(draft_weights);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::DraftWeights) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    // `with_mq` allocates the FWHT rotation scratch (mq_x_rot) that
    // `gemm_dispatch` requires for MQ4/MQ3/MQ6 draft weights. The carrier
    // refactor regressed this to the `with_mq=false` `::new` constructor →
    // panic "MQ4 dispatch requires mq_x_rot scratch" on any MQ-quantized draft.
    let draft_scratch = match DflashScratch::new_with_mq(
        gpu,
        &draft_config,
        block_size,
        ctx_capacity,
        staged
            .draft_weights
            .as_ref()
            .expect("staged draft weights")
            .has_mq,
    ) {
        Ok(scratch) => scratch,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(format!("DflashScratch::new_with_mq: {error}"));
        }
    };
    staged.draft_scratch = Some(draft_scratch);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::DraftScratch) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    // The hidden-ring STAGING buffers must hold one prefill chunk. Verify
    // cycles seed only `max_n` (= block_size+1) rows, but the prompt seed
    // (`seed_target_hidden_from_prompt_abortable`) prefills the prompt in
    // chunks of up to `PREFILL_MAX_BATCH` and captures each into staging via
    // `write_rows_to_staging` (whose `n <= max_batch` guard is a debug_assert,
    // silent in release). Sizing staging to only `max_n` overflowed the d2d
    // copy on any prompt longer than block_size+1 tokens. Size it to the
    // larger of the two so both paths fit.
    let staging_max_batch = max_n.max(qwen35::PREFILL_MAX_BATCH);
    let hidden_rb = match HiddenStateRingBuffer::new(
        gpu,
        target_config.n_layers,
        draft_config.num_extract(),
        target_config.dim,
        ctx_capacity,
        staging_max_batch,
    ) {
        Ok(hidden_rb) => hidden_rb,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(format!("HiddenStateRingBuffer::new: {error}"));
        }
    };
    staged.hidden_rb = Some(hidden_rb);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::HiddenRing) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    let verify_scratch = match VerifyScratch::with_prefill(
        gpu,
        max_n,
        target_config.dim,
        target_config.vocab_size,
        hidden_k,
        target_config,
    ) {
        Ok(verify_scratch) => verify_scratch,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(format!("VerifyScratch::with_prefill: {error}"));
        }
    };
    staged.verify_scratch = Some(verify_scratch);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::VerifyScratch) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    let target_snap = match DeltaNetSnapshot::new_for(gpu, target_dn) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(format!("DeltaNetSnapshot::new_for: {error}"));
        }
    };
    staged.target_snap = Some(target_snap);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::TargetSnapshot) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    let gdn_tape = match GdnTape::new_for_config(gpu, target_config, max_n) {
        Ok(tape) => tape,
        Err(error) => {
            staged.free_gpu(gpu);
            return Err(format!("GdnTape::new_for_config: {error}"));
        }
    };
    staged.gdn_tape = Some(gdn_tape);
    if let Err(error) = dflash_construction_boundary(DflashConstructionStage::GdnTape) {
        staged.free_gpu(gpu);
        return Err(error);
    }
    // DDTree (budget read once above, used for scratch sizing).
    if ddtree_budget > 0 {
        staged.ddtree = Some(DdtreeStaging {
            post_seed_snap: None,
            scratch: None,
            budget: ddtree_budget,
            topk: ddtree_topk,
        });
        let post_seed_snap = match DeltaNetSnapshot::new_for(gpu, target_dn) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                staged.free_gpu(gpu);
                return Err(format!("DDTree DeltaNetSnapshot::new_for: {error}"));
            }
        };
        staged
            .ddtree
            .as_mut()
            .expect("staged DDTree state")
            .post_seed_snap = Some(post_seed_snap);
        if let Err(error) = dflash_construction_boundary(DflashConstructionStage::DdtreeSnapshot) {
            staged.free_gpu(gpu);
            return Err(error);
        }
        let scratch = match DdtreeScratch::new(gpu, ddtree_budget) {
            Ok(scratch) => scratch,
            Err(error) => {
                staged.free_gpu(gpu);
                return Err(format!("DdtreeScratch::new: {error}"));
            }
        };
        staged.ddtree.as_mut().expect("staged DDTree state").scratch = Some(scratch);
        if let Err(error) = dflash_construction_boundary(DflashConstructionStage::DdtreeScratch) {
            staged.free_gpu(gpu);
            return Err(error);
        }
    }
    Ok(staged.into_state(draft_config, target_hidden_host, ctx_capacity, block_size))
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
            self.df.draft_scratch.ctx_modulus(),
        ) {
            eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
        }
        // Windowed mode, cold prefill longer than the SWA window: the last
        // (full-attention) draft layer still needs K/V for every prompt row,
        // but hidden_rb and the draft ring only retain the last W. Backfill
        // the last layer's long-reach ring from the host shadow (cumulative
        // on the cold path) before the first spec step.
        if !cache_hit {
            hipfire_runtime::dflash::draft_seed_backfill(
                gpu,
                &self.df.draft_weights,
                &self.df.draft_config,
                &mut self.df.draft_scratch,
                &self.df.target_hidden_host,
                prompt_tokens.len(),
            )
            .map_err(|e| e.to_string())?;
        }
        self.df.draft_scratch.thlog.seed_prompt(prompt_tokens.len());
        if let Some(ckpt) = resume_from {
            // Divergent rows [ckpt..len) were just overwritten; drop the draft's
            // projection cursor so the first spec step re-projects from `ckpt`.
            self.df.draft_scratch.thlog.set_resume_checkpoint(ckpt);
        }

        // First emit = target draw at the final prompt position (seed already
        // ran the per-token forward; scratch.logits holds the post-prompt logits).
        // temp≈0 stays the historical host argmax fold (byte-identical greedy).
        // temp>0 uses the same host nucleus sampler as chain DFlash verify so the
        // post-prefill seed is not a special greedy exception on distribution-
        // preserving requests.
        let first_logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        let first_token = if self.sample_temp <= 1e-6 {
            first_logits
                .iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                    if v > bv {
                        (i as u32, v)
                    } else {
                        (best, bv)
                    }
                })
                .0
        } else {
            let mut probs = Vec::with_capacity(first_logits.len());
            softmax_temp_into(&first_logits, self.sample_temp, &mut probs);
            // DDTree SWOR honors temperature only (matches step's tree arm).
            // Chain mode applies the same host top_k + nucleus cuts as
            // `spec_step_dflash` so the seed is AR-at-(top_k,top_p).
            if self.df.ddtree.is_none() {
                if self.sample_top_k > 0 && self.sample_top_k < probs.len() {
                    let mut order: Vec<usize> = (0..probs.len()).collect();
                    order.sort_by(|&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let keep = self.sample_top_k;
                    let mut kept_mass = 0.0f32;
                    for (rank, &idx) in order.iter().enumerate() {
                        if rank < keep {
                            kept_mass += probs[idx];
                        } else {
                            probs[idx] = 0.0;
                        }
                    }
                    if kept_mass > 0.0 {
                        let inv = 1.0 / kept_mass;
                        for p in probs.iter_mut() {
                            *p *= inv;
                        }
                    }
                }
                if self.sample_top_p < 0.999 {
                    apply_host_nucleus(&mut probs, self.sample_top_p);
                }
            }
            let u = xorshift_next_unit(&mut self.rng_state);
            sample_categorical(&probs, u)
        };
        Ok(PrefillOutcome::Ready { first_token })
    }

    /// Forced tokens (think-budget force-close) must land in the drafter's
    /// per-position `target_hidden` cache, not just the target's KV. Seeding via
    /// the same suffix path the prompt-cache HIT uses advances the target WITH
    /// hidden extraction, so the rows exist and `thlog` stays contiguous.
    ///
    /// Skipping this is what previously left an uninitialized (NaN) hole at the
    /// forced positions: the next draft forward read it, produced all-NaN logits,
    /// and `argmax` collapsed to token 0 — τ went to 0 for the rest of the
    /// session and stayed dead across prompt-cache HITs.
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
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        let aborted = seed_target_hidden_suffix_abortable(
            gpu,
            slot,
            &mut self.df.hidden_rb,
            tokens,
            start_pos,
            abort,
            None,
            self.ck_interval,
            self.ck_cap,
        )
        .map_err(|e| e.to_string())?;
        if aborted {
            // Caller tears the request down; leaving the rows unwritten is fine
            // because the drafter state is reset on the way out.
            return Ok(true);
        }
        scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            start_pos,
            tokens.len(),
            tokens.len(),
            self.df.draft_scratch.ctx_modulus(),
        )
        .map_err(|e| e.to_string())?;
        let co = slot.kv_cache_mut().map(|kv| kv.compact_offset).unwrap_or(0) as i32;
        self.df
            .draft_scratch
            .thlog
            .append_committed(start_pos, tokens.len(), co);
        Ok(true)
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
        max_emit: usize,
    ) -> Result<SpecStep, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;

        if max_emit == 0 {
            return Err("DflashSpeculator: max_emit=0 (no remaining output budget)".into());
        }
        // Chain DFlash: emit ≤ b (accepted drafts + bonus). Cap block size so the
        // verify window cannot commit past remaining client budget. b >= 2.
        // emit = accept + 1 ≤ b when seed is excluded. Prefer b = max_emit
        // (uniform for max_emit >= 1); max_accept clamps accept before commit
        // so max_emit == 1 is a true one-token path (accept 0 + bonus).
        let block_override = {
            let cfg_b = self.df.block_size.max(2);
            let want = max_emit.max(2);
            let b = cfg_b.min(want);
            if b < cfg_b {
                Some(b)
            } else {
                None
            }
        };
        // accepted drafts + bonus = emit; max accepted drafts = max_emit - 1.
        let max_accept = Some(max_emit.saturating_sub(1));

        // Two-way dispatch from the daemon's old generate_dflash loop: DDTree-
        // batched (SWOR) verify when a tree is configured, else chain-mode DFlash.
        // The grammar arg is ignored — qwen35 enforces tool-call grammar post-hoc
        // in the daemon.
        let result = if let Some(dd) = self.df.ddtree.as_mut() {
            // Tree node budget is structural; max_accept is the commit bound.
            // Keep at least 1 node so the tree builder stays well-formed; the
            // accept clamp drops to 0 drafts when max_emit == 1.
            let tree_budget = dd.budget.min(max_emit.saturating_sub(1).max(1));
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
                tree_budget,
                dd.topk,
                // Request temperature → distribution-preserving SWOR verify at
                // temp>0 (greedy/argmax at temp 0). The ddtree-batched arm is the
                // only DFlash mode with sampled verify; the chain below stays
                // greedy, so `supports_temp_verify` gates serve routing to ddtree.
                temp,
                &mut self.rng_state,
                max_accept,
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
                block_override, // remaining-output budget
                None,           // ngram_cache
                emitted,
                self.sample_cactus, // 0.0 = lossless; >0 = deliberately lossy
                None,               // pld_spine
                1.0_f32,            // repeat_penalty (off)
                0,                  // repeat_window
                max_accept,
            )
        };

        result
            .map(lower_qwen35)
            // Defense only — accept stage already committed ≤ max_emit.
            .map(|s| s.cap_emit(max_emit))
            .map_err(|e| e.to_string())
    }

    fn advance_forced(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        position: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<SpecAdvance, String> {
        let slot = target
            .as_any_mut()
            .downcast_mut::<ModelSlot>()
            .ok_or("DflashSpeculator: target is not a Qwen3.5 ModelSlot")?;
        if tokens.is_empty() {
            return Ok(SpecAdvance::Ready { last_argmax: 0, last_logits: None });
        }

        // The target-hidden buffer is the DFlash draft's authoritative context.
        // A target-only advance here would leave the next draft's cached rows and
        // projection cursor behind the target KV/recurrent state.
        if seed_target_hidden_suffix_abortable(
            gpu,
            slot,
            &mut self.df.hidden_rb,
            tokens,
            position,
            abort,
            None,
            self.ck_interval,
            self.ck_cap,
        )
        .map_err(|e| e.to_string())?
        {
            return Ok(SpecAdvance::Aborted);
        }
        scatter_hidden_block_to_interleaved(
            gpu,
            &self.df.hidden_rb,
            &self.df.draft_scratch.target_hidden,
            position,
            tokens.len(),
            tokens.len(),
            self.df.draft_scratch.ctx_modulus(),
        )
        .map_err(|e| e.to_string())?;
        self.df.draft_scratch.thlog.append_committed(
            position,
            tokens.len(),
            slot.kv_cache.compact_offset as i32,
        );
        let logits = gpu
            .download_f32(&slot.scratch.logits)
            .map_err(|e| e.to_string())?;
        let last_argmax = logits
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
        Ok(SpecAdvance::Ready { last_argmax, last_logits: None })
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

    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Drafter-local reset: invalidate cached suffix projections and free the
        // divergent-render checkpoint ring (the target KV/recurrent reset is the
        // daemon's job — it owns the bundle).
        self.df.draft_scratch.reset_upload_tracking();
        let mut first_error = None;
        let mut remaining = Vec::new();
        for (position, mut snap) in self.checkpoints.drain(..) {
            if let Err(error) = snap.free_gpu_checked(gpu) {
                first_error.get_or_insert(error);
                remaining.push((position, snap));
            }
        }
        Ok(())
    }

    fn reset_state_evidence(&self) -> Option<hipfire_runtime::spec::SpecResetEvidence> {
        let th = &self.df.draft_scratch.thlog;
        Some(hipfire_runtime::spec::SpecResetEvidence {
            drafter_reset: th.uploaded_rows() == 0
                && th.proj_cached_rows() == 0
                && th.full_cached_rows() == 0,
            checkpoint_empty: self.checkpoints.is_empty(),
        })
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
            self.checkpoints[idx]
                .1
                .restore_to(&mut slot.dn_state, gpu)
                .map_err(|e| format!("DeltaNetSnapshot::restore_to: {e}"))?;
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
        df.free_gpu(gpu);
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
    let resume_enabled = hipfire_config::developer_var("HIPFIRE_DFLASH_CKPT_RESUME")
        .ok()
        .as_deref()
        != Some("0")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::LayerType;
    use crate::speculative::ModelSlotConfig;
    use std::sync::{Mutex, MutexGuard};

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn dflash_gpu_test_lock() -> MutexGuard<'static, ()> {
        GPU_TEST_LOCK.lock().expect("DFlash GPU test lock poisoned")
    }

    fn expect_exact_vram_baseline(
        gpu: &mut Gpu,
        before: usize,
        context: &str,
    ) -> Result<(), String> {
        gpu.drain_pool();
        let after = gpu
            .hip
            .get_vram_info()
            .map_err(|e| format!("measure after {context}: {e}"))?
            .0;
        if after != before {
            return Err(format!(
                "{context} changed free VRAM from {before} to {after} bytes"
            ));
        }
        Ok(())
    }

    /// Requires real Qwen3.6 target/draft fixtures because DFlash owns GPU-only
    /// scratch and DeltaNet snapshots. DDTree is enabled to cover both snapshot
    /// owners, and every target allocation is released before reporting failures.
    #[test]
    #[ignore = "requires an AMD GPU plus qwen3.6-27b.mq4 and qwen36-27b-dflash-mq4.hfq"]
    fn dflash_free_reclaims_target_and_ddtree_snapshots() {
        let _gpu_test_lock = dflash_gpu_test_lock();
        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_DFLASH_FREE_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        let draft_path = std::env::var("HIPFIRE_DFLASH_FREE_DRAFT")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen36-27b-dflash-mq4.hfq"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_DFLASH_FREE_TARGET"
        );
        assert!(
            Path::new(&draft_path).is_file(),
            "draft fixture not found: {draft_path}; set HIPFIRE_DFLASH_FREE_DRAFT"
        );

        let mut gpu = Gpu::init().expect("Gpu::init");
        let slot = ModelSlot::load(
            &mut gpu,
            Path::new(&target_path),
            "dflash-free-test",
            ModelSlotConfig {
                max_seq: 64,
                ..Default::default()
            },
        )
        .expect("load Qwen target slot");

        let result = (|| -> Result<(), String> {
            if slot.dn_state.s_matrices.is_empty() {
                return Err("target fixture has no DeltaNet state to snapshot".into());
            }
            let before = gpu
                .hip
                .get_vram_info()
                .map_err(|e| format!("measure before DFlash load: {e}"))?
                .0;
            let mut state = Some(load_dflash_state(
                &draft_path,
                64,
                &slot.config,
                &slot.dn_state,
                &mut gpu,
                Some(1),
                Some(1),
                false,
            )?);
            let result = (|| -> Result<(), String> {
                if state
                    .as_ref()
                    .map(|state| state.ddtree.is_none())
                    .unwrap_or(true)
                {
                    return Err("DDTree post-seed snapshot was not constructed".into());
                }
                let loaded = gpu
                    .hip
                    .get_vram_info()
                    .map_err(|e| format!("measure after DFlash load: {e}"))?
                    .0;
                if loaded >= before {
                    return Err("DFlash state did not allocate observable GPU memory".into());
                }
                let dflash_state = state
                    .take()
                    .ok_or("DFlash state disappeared before speculator publication")?;
                let spec: Box<dyn Speculator> =
                    Box::new(DflashSpeculator::new(dflash_state, false, 256, 1));
                spec.free(&mut gpu);
                expect_exact_vram_baseline(&mut gpu, before, "DFlash speculator free")?;
                Ok(())
            })();
            // `load_dflash_state` publishes GPU-only resources. If any check above
            // rejects that state before it becomes a speculator, free it before
            // propagating the failure to the outer ModelSlot cleanup.
            if let Some(state) = state.take() {
                state.free_gpu(&mut gpu);
            }
            result
        })();

        let ModelSlot {
            weights,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = slot;
        scratch.free_gpu(&mut gpu);
        dn_state.free_gpu(&mut gpu);
        kv_cache.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        if let Err(error) = result {
            panic!("{error}");
        }
    }

    /// Exercises the staged owner at every point a completed DFlash resource
    /// is waiting to be published. The fault fires after that resource is
    /// owned, so successful cleanup proves the unpublished state cannot leak.
    #[test]
    #[ignore = "requires an AMD GPU plus qwen3.6-27b.mq4 and qwen36-27b-dflash-mq4.hfq"]
    fn dflash_construction_rolls_back_each_completed_resource() {
        let _gpu_test_lock = dflash_gpu_test_lock();
        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_DFLASH_FREE_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        let draft_path = std::env::var("HIPFIRE_DFLASH_FREE_DRAFT")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen36-27b-dflash-mq4.hfq"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_DFLASH_FREE_TARGET"
        );
        assert!(
            Path::new(&draft_path).is_file(),
            "draft fixture not found: {draft_path}; set HIPFIRE_DFLASH_FREE_DRAFT"
        );

        let mut gpu = Gpu::init().expect("Gpu::init");
        let slot = ModelSlot::load(
            &mut gpu,
            Path::new(&target_path),
            "dflash-construction-rollback-test",
            ModelSlotConfig {
                max_seq: 64,
                ..Default::default()
            },
        )
        .expect("load Qwen target slot");

        let result = (|| -> Result<(), String> {
            let before = gpu
                .hip
                .get_vram_info()
                .map_err(|e| format!("measure before DFlash load: {e}"))?
                .0;
            for &stage in &DflashConstructionStage::ALL {
                let loaded = with_dflash_test_fault(DflashTestFault::AfterStage(stage), || {
                    load_dflash_state(
                        &draft_path,
                        64,
                        &slot.config,
                        &slot.dn_state,
                        &mut gpu,
                        Some(1),
                        Some(1),
                        false,
                    )
                });
                if let Ok(state) = loaded {
                    state.free_gpu(&mut gpu);
                    return Err(format!(
                        "fault after {} unexpectedly constructed DFlash state",
                        stage.label()
                    ));
                }
                expect_exact_vram_baseline(
                    &mut gpu,
                    before,
                    &format!("DFlash staging rollback after {}", stage.label()),
                )?;
            }
            Ok(())
        })();

        let ModelSlot {
            weights,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = slot;
        scratch.free_gpu(&mut gpu);
        dn_state.free_gpu(&mut gpu);
        kv_cache.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        if let Err(error) = result {
            panic!("{error}");
        }
    }

    /// Directly faults every allocation in the constructors that allocate their
    /// GPU buffers incrementally. Each constructor must reclaim its local
    /// partial state before returning the injected allocation error.
    #[test]
    #[ignore = "requires an AMD GPU and qwen3.5-0.8b.mq4"]
    fn dflash_component_constructors_rollback_every_allocation() {
        let _gpu_test_lock = dflash_gpu_test_lock();
        let model = std::env::var("HIPFIRE_QWEN35_RESET_STATE_MODEL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME is required for the default model path");
            format!("{home}/.hipfire/models/qwen3.5-0.8b.mq4")
        });
        assert!(
            Path::new(&model).is_file(),
            "model fixture not found: {model}; set HIPFIRE_QWEN35_RESET_STATE_MODEL"
        );

        let mut gpu = Gpu::init().expect("Gpu::init");
        let slot = ModelSlot::load(
            &mut gpu,
            Path::new(&model),
            "dflash-component-rollback-test",
            ModelSlotConfig {
                max_seq: 16,
                ..Default::default()
            },
        )
        .expect("load Qwen target slot");

        let result = (|| -> Result<(), String> {
            let before = gpu
                .hip
                .get_vram_info()
                .map_err(|e| format!("measure before component allocation: {e}"))?
                .0;
            let snapshot_allocations = slot.dn_state.s_matrices.len()
                + slot.dn_state.s_scales.len()
                + slot.dn_state.conv_states.len()
                + slot.dn_state.s_ef_residual.len();
            let gdn_allocations = slot
                .config
                .layer_types
                .iter()
                .filter(|layer| **layer == LayerType::LinearAttention)
                .count()
                * 3
                + 6;
            for (site, allocations) in [
                (DflashAllocationSite::DeltaNetSnapshot, snapshot_allocations),
                (DflashAllocationSite::GdnTape, gdn_allocations),
                (DflashAllocationSite::DdtreeScratch, 2),
                (DflashAllocationSite::VerifyScratch, 4),
                (DflashAllocationSite::HiddenStateRing, 4),
            ] {
                for allocation in 0..allocations {
                    let loaded = with_dflash_test_fault(
                        DflashTestFault::AfterAllocation { site, allocation },
                        || match site {
                            DflashAllocationSite::DeltaNetSnapshot => {
                                DeltaNetSnapshot::new_for(&mut gpu, &slot.dn_state)
                                    .map(|snapshot| ComponentAllocation::Snapshot(snapshot))
                            }
                            DflashAllocationSite::GdnTape => {
                                GdnTape::new_for_config(&mut gpu, &slot.config, 2)
                                    .map(|tape| ComponentAllocation::GdnTape(tape))
                            }
                            DflashAllocationSite::DdtreeScratch => DdtreeScratch::new(&mut gpu, 1)
                                .map(|scratch| ComponentAllocation::DdtreeScratch(scratch)),
                            DflashAllocationSite::VerifyScratch => {
                                VerifyScratch::new(&mut gpu, 2, 32, 64, 32)
                                    .map(|scratch| ComponentAllocation::VerifyScratch(scratch))
                            }
                            DflashAllocationSite::HiddenStateRing => {
                                HiddenStateRingBuffer::new(&mut gpu, 4, 2, 32, 16, 2)
                                    .map(|ring| ComponentAllocation::HiddenStateRing(ring))
                            }
                            DflashAllocationSite::PrefillBatchScratch => {
                                unreachable!("PrefillBatchScratch is checked below")
                            }
                        },
                    );
                    if let Ok(component) = loaded {
                        component.free_gpu(&mut gpu);
                        return Err(format!(
                            "fault after {} allocation {allocation} unexpectedly succeeded",
                            site.label()
                        ));
                    }
                    expect_exact_vram_baseline(
                        &mut gpu,
                        before,
                        &format!("{} allocation {allocation}", site.label()),
                    )?;
                }
            }
            let hidden_k =
                slot.config.dim.checked_next_power_of_two().ok_or_else(|| {
                    "target dimension overflows nested verify hidden_k".to_string()
                })?;
            let mut nested_completed = false;
            for allocation in 0..=128 {
                let loaded = with_dflash_test_fault(
                    DflashTestFault::AfterAllocation {
                        site: DflashAllocationSite::PrefillBatchScratch,
                        allocation,
                    },
                    || {
                        VerifyScratch::with_prefill(
                            &mut gpu,
                            2,
                            slot.config.dim,
                            slot.config.vocab_size,
                            hidden_k,
                            &slot.config,
                        )
                    },
                );
                match loaded {
                    Err(_) => expect_exact_vram_baseline(
                        &mut gpu,
                        before,
                        &format!("PrefillBatchScratch allocation {allocation}"),
                    )?,
                    Ok(scratch) => {
                        scratch.free_gpu(&mut gpu);
                        expect_exact_vram_baseline(
                            &mut gpu,
                            before,
                            "PrefillBatchScratch success/free",
                        )?;
                        nested_completed = true;
                        break;
                    }
                }
            }
            if !nested_completed {
                return Err("nested PrefillBatchScratch fault loop did not reach success".into());
            }
            Ok(())
        })();

        let ModelSlot {
            weights,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = slot;
        scratch.free_gpu(&mut gpu);
        dn_state.free_gpu(&mut gpu);
        kv_cache.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        if let Err(error) = result {
            panic!("{error}");
        }
    }

    /// Faults every resource returned by the runtime draft constructors. The
    /// missing runtime injector makes this test RED until those constructors
    /// stage their own allocations instead of relying on outer DFlash state.
    #[cfg(feature = "dflash-fault-inject")]
    #[test]
    #[ignore = "requires an AMD GPU plus qwen3.6-27b.mq4 and qwen36-27b-dflash-mq4.hfq"]
    fn draft_runtime_constructors_rollback_every_allocation() {
        const MAX_RUNTIME_ALLOCATIONS: usize = 256;

        let _gpu_test_lock = dflash_gpu_test_lock();
        let home = std::env::var("HOME").expect("HOME is required for default model paths");
        let target_path = std::env::var("HIPFIRE_DFLASH_FREE_TARGET")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen3.6-27b.mq4"));
        let draft_path = std::env::var("HIPFIRE_DFLASH_FREE_DRAFT")
            .unwrap_or_else(|_| format!("{home}/.hipfire/models/qwen36-27b-dflash-mq4.hfq"));
        assert!(
            Path::new(&target_path).is_file(),
            "target fixture not found: {target_path}; set HIPFIRE_DFLASH_FREE_TARGET"
        );
        assert!(
            Path::new(&draft_path).is_file(),
            "draft fixture not found: {draft_path}; set HIPFIRE_DFLASH_FREE_DRAFT"
        );

        let mut gpu = Gpu::init().expect("Gpu::init");
        let slot = ModelSlot::load(
            &mut gpu,
            Path::new(&target_path),
            "dflash-runtime-rollback-test",
            ModelSlotConfig {
                max_seq: 64,
                ..Default::default()
            },
        )
        .expect("load Qwen target slot");

        let result = (|| -> Result<(), String> {
            let draft_hfq = HfqFile::open(Path::new(&draft_path)).map_err(|e| format!("{e}"))?;
            let draft_config = DflashConfig::from_hfq(&draft_hfq)
                .ok_or_else(|| "parse DFlash fixture config".to_string())?;

            // MQ sign tables belong to Gpu, not DflashWeights. Warm them before
            // taking the baseline so constructor-local accounting is exact.
            let warm = DflashWeights::load(&mut gpu, &draft_hfq, &draft_config)
                .map_err(|e| format!("warm DflashWeights: {e}"))?;
            let has_mq = warm.has_mq;
            warm.free_gpu(&mut gpu);
            gpu.drain_pool();
            let before = gpu
                .hip
                .get_vram_info()
                .map_err(|e| format!("measure before runtime constructor fault: {e}"))?
                .0;

            let mut weights_completed = false;
            for allocation in 0..=MAX_RUNTIME_ALLOCATIONS {
                let loaded = hipfire_runtime::dflash::with_dflash_allocation_fault(
                    hipfire_runtime::dflash::DflashAllocationFault {
                        site: hipfire_runtime::dflash::DflashAllocationSite::Weights,
                        allocation,
                    },
                    || DflashWeights::load(&mut gpu, &draft_hfq, &draft_config),
                );
                match loaded {
                    Err(_) => expect_exact_vram_baseline(
                        &mut gpu,
                        before,
                        &format!("draft weights allocation {allocation}"),
                    )?,
                    Ok(weights) => {
                        weights.free_gpu(&mut gpu);
                        expect_exact_vram_baseline(&mut gpu, before, "draft weights success/free")?;
                        weights_completed = true;
                        break;
                    }
                }
            }
            if !weights_completed {
                return Err("draft weights fault loop did not reach constructor success".into());
            }

            let mut scratch_completed = false;
            for allocation in 0..=MAX_RUNTIME_ALLOCATIONS {
                let loaded = hipfire_runtime::dflash::with_dflash_allocation_fault(
                    hipfire_runtime::dflash::DflashAllocationFault {
                        site: hipfire_runtime::dflash::DflashAllocationSite::Scratch,
                        allocation,
                    },
                    || DflashScratch::new_with_mq(&mut gpu, &draft_config, 16, 64, has_mq),
                );
                match loaded {
                    Err(_) => expect_exact_vram_baseline(
                        &mut gpu,
                        before,
                        &format!("draft scratch allocation {allocation}"),
                    )?,
                    Ok(scratch) => {
                        scratch.free_gpu(&mut gpu);
                        expect_exact_vram_baseline(&mut gpu, before, "draft scratch success/free")?;
                        scratch_completed = true;
                        break;
                    }
                }
            }
            if !scratch_completed {
                return Err("draft scratch fault loop did not reach constructor success".into());
            }
            Ok(())
        })();

        let ModelSlot {
            weights,
            kv_cache,
            dn_state,
            scratch,
            ..
        } = slot;
        scratch.free_gpu(&mut gpu);
        dn_state.free_gpu(&mut gpu);
        kv_cache.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        if let Err(error) = result {
            panic!("{error}");
        }
    }

    enum ComponentAllocation {
        Snapshot(DeltaNetSnapshot),
        GdnTape(GdnTape),
        DdtreeScratch(DdtreeScratch),
        VerifyScratch(VerifyScratch),
        HiddenStateRing(HiddenStateRingBuffer),
    }

    impl ComponentAllocation {
        fn free_gpu(self, gpu: &mut Gpu) {
            match self {
                Self::Snapshot(snapshot) => snapshot.free_gpu(gpu),
                Self::GdnTape(tape) => tape.free_gpu(gpu),
                Self::DdtreeScratch(scratch) => scratch.free_gpu(gpu),
                Self::VerifyScratch(scratch) => scratch.free_gpu(gpu),
                Self::HiddenStateRing(ring) => {
                    for tensor in ring.layer_bufs.into_iter().chain(ring.staging_bufs) {
                        let _ = gpu.free_tensor(tensor);
                    }
                }
            }
        }
    }
}

// ── Send-bound assertions ──────────────────────────────────────────────
#[cfg(test)]
mod send_assertions {
    fn _assert_send<T: Send>() {}

    #[test]
    fn dflash_speculator_is_send() {
        _assert_send::<super::DflashSpeculator>();
    }
}
