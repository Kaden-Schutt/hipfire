use crate::qwen35::{
    DeltaNetState, LayerType, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use crate::store::Qwen35LoadError;
use crate::Qwen35;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_adaptive::{KvAdaptive, Preset};
use hipfire_runtime::kv_mode::{self, ResolveResult};
use hipfire_runtime::llama::{self, KvCache, KvDims, KvLayers, KvTarget};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use rdna_compute::{Gpu, GpuTensor};

/// qwen35 pipeline-parallel scratch, present iff this bundle is served PP (pp>1).
/// One payload so scratch + layer→device map are inseparable (no "one without the
/// other" illegal state).
pub struct Qwen35PipelineState {
    pub scratch_set: crate::qwen35::Qwen35ScratchSet,
    pub dn_la_to_device: Vec<u8>,
}

pub struct Qwen35Bundle {
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub scratch: Qwen35Scratch,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    /// Optional native MTP (NextN) head — present only when a `.mq4-mtp` trailer
    /// or `.mtp` sidecar was loaded. Lives here (not on `LoadedModel`) so it
    /// travels with the arch state through reset/reload; the loader folds it in
    /// after `load_bundle` returns. `None` on every non-MTP construction.
    pub mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
    /// PP scratch — `Some` only for the qwen35 pp>1 serve path; `None` single-GPU.
    /// Carried here (not on `LoadedModel`) so it travels with arch state and its
    /// teardown is arm-local to `ModelState::Qwen35`.
    pub pipeline: Option<Qwen35PipelineState>,
}

// ═════════════════════════════════════════════════════════════════════
// Explicit bundle build transaction (replaces BundleStagingGuard)
// ═════════════════════════════════════════════════════════════════════

/// Typed error from [`BundleBuildTransaction::abort`] that preserves every
/// auxiliary owner the transaction held at the time of failure.
///
/// The receiver MUST free each complete auxiliary owner (via that type's
/// `abort_checked` method) and must NOT drop any of them.
///
/// # Ownership contract
///
/// * `weights` — always present (the transaction starts with weights).
/// * `kv_cache`, `dn_state`, `scratch` — `Some` only if construction
///   completed that stage before the failure.  Each is a COMPLETE owner
///   that can be freed independently.
/// * `mtp_head` — optional MTP head, if allocated.
///
/// The receiver should attempt cleanup of each `Some` domain independently
/// using the domain's checked free method (e.g. `KvCache::free_checked`,
/// `DeltaNetState::abort_checked`, `Qwen35Scratch::abort_checked`).
/// Any tensors that cannot be freed are returned as typed errors for retry.
///
/// Crate-private: no external consumer exists; the loader reaches it only
/// through `load_bundle`'s string error surface.
pub(crate) struct Qwen35BundleBuildError {
    /// Human-readable description of the failure point.
    pub message: String,
    /// Common weights (always present — the transaction begins with weights).
    pub weights: Qwen35Weights,
    /// KV cache, if allocated before the failure.
    pub kv_cache: Option<KvCache>,
    /// DeltaNet recurrent state, if allocated before the failure.
    pub dn_state: Option<DeltaNetState>,
    /// Forward scratch, if allocated before the failure.
    pub scratch: Option<Qwen35Scratch>,
    /// MTP head, if allocated.
    pub mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
}

impl Qwen35BundleBuildError {
    /// Attempt checked cleanup of EVERY present domain, returning retained
    /// failures for each.  Domains are attempted independently — a failure
    /// in one does not prevent cleanup of others.
    ///
    /// Returns `(message, kv_failures, dn_failures, scratch_failures,
    /// weights_failure)` where:
    /// - `kv_failures`: `(label, GpuTensor)` from KvCache
    /// - `dn_failures`: `RetainedQwenTensor` from DeltaNetState
    /// - `scratch_failures`: `RetainedQwenTensor` from Qwen35Scratch
    /// - `weights_failure`: `Option<Qwen35CleanupFailure>` from Qwen35Weights
    ///   (None = freed successfully, Some = some weights retained)
    ///
    /// The `message` and all failure collectors are ALWAYS returned — no
    /// owner can be inadvertently dropped.
    #[expect(
        clippy::type_complexity,
        reason = "the tuple preserves each rollback-owner category distinctly (message, KV label+tensor pairs, DN retainers, scratch retainers, weights cleanup failure) so the caller folds every category whole into the retry aggregate without cross-category flattening"
    )]
    pub fn try_free(
        self,
        gpu: &mut Gpu,
    ) -> (
        String,
        Vec<(String, GpuTensor)>,
        Vec<crate::qwen35::RetainedQwenTensor>,
        Vec<crate::qwen35::RetainedQwenTensor>,
        Option<crate::qwen35::Qwen35CleanupFailure>,
    ) {
        // Destructure self to extract every owner field.
        let Qwen35BundleBuildError {
            message,
            weights,
            kv_cache,
            dn_state,
            scratch,
            mtp_head,
        } = self;

        // Phase A: KV cache — checked free, retain failures.
        let msg = message;
        let kv_failures = match kv_cache {
            Some(kv) => match kv.free_checked(gpu) {
                Ok(()) => Vec::new(),
                Err(f) => f,
            },
            None => Vec::new(),
        };

        // Phase B: DeltaNet state — checked free, retain failures.
        let dn_failures = match dn_state {
            Some(dn) => match dn.abort_checked(gpu) {
                Ok(()) => Vec::new(),
                Err(f) => f,
            },
            None => Vec::new(),
        };

        // Phase C: Scratch — checked free, retain failures.
        let scratch_failures = match scratch {
            Some(s) => match s.abort_checked(gpu) {
                Ok(()) => Vec::new(),
                Err(f) => f,
            },
            None => Vec::new(),
        };

        // Phase D: MTP head — checked free (via free_checked which
        // returns retained GpuTensors as RetainedQwenTensor).
        let mtp_failures: Vec<crate::qwen35::RetainedQwenTensor> = match mtp_head {
            Some(mtp) => mtp.free_checked(gpu),
            None => Vec::new(),
        };

        // Phase E: Weights — checked free via free_gpu_checked.
        // Done LAST so prior domain failures are already collected.
        let weights_failure = match weights.free_gpu_checked(gpu) {
            Ok(()) => {
                // Weights freed successfully — still need to surface MTP failures.
                if mtp_failures.is_empty() {
                    None
                } else {
                    let mut cf = crate::qwen35::Qwen35CleanupFailure::empty();
                    for r in mtp_failures {
                        cf.add_retained(r);
                    }
                    Some(cf)
                }
            }
            Err(mut cf) => {
                // Merge MTP failures into weights failure.
                for r in mtp_failures {
                    cf.add_retained(r);
                }
                Some(cf)
            }
        };

        (
            msg,
            kv_failures,
            dn_failures,
            scratch_failures,
            weights_failure,
        )
    }
}

/// Transactional bundle build state.  Every allocated GPU resource (weights,
/// KV cache, DeltaNet state, scratch) is tracked so that a mid-construction
/// failure returns every owner through [`BundleBuildTransaction::abort`].
///
/// The transaction starts with `weights` (always present).  Auxiliary domains
/// are added via `set_*` as construction progresses.
///
/// # Correctness
///
/// * On success: call [`into_bundle`](Self::into_bundle) — consumes the
///   transaction and all resources into the final [`Qwen35Bundle`].
/// * On failure: call [`abort`](Self::abort) — returns
///   [`Qwen35BundleBuildError`] with every owner the transaction held.
///
/// If dropped without either call, the transaction panics to catch
/// programming errors (this is a diagnostic abort, NOT a cleanup path).
///
/// Crate-private: constructed only inside `build_qwen35_bundle`.
#[must_use = "BundleBuildTransaction must be consumed via into_bundle() or abort()"]
pub(crate) struct BundleBuildTransaction {
    weights: Option<Qwen35Weights>,
    kv_cache: Option<KvCache>,
    dn_state: Option<DeltaNetState>,
    scratch: Option<Qwen35Scratch>,
    mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
}

impl BundleBuildTransaction {
    /// Create a new transaction with the given weights (always present).
    pub fn new(weights: Qwen35Weights) -> Self {
        Self {
            weights: Some(weights),
            kv_cache: None,
            dn_state: None,
            scratch: None,
            mtp_head: None,
        }
    }

    /// Read-only borrow of weights (for screening etc.).
    pub fn weights(&self) -> &Qwen35Weights {
        self.weights.as_ref().expect("transaction: weights missing")
    }

    /// Read-only borrow of KV cache (for reconfiguration etc.).
    pub fn kv_cache(&self) -> &KvCache {
        self.kv_cache
            .as_ref()
            .expect("transaction: KV cache missing")
    }

    /// Mutable borrow of KV cache (for reconfiguration via realloc etc.).
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        self.kv_cache
            .as_mut()
            .expect("transaction: KV cache missing")
    }

    /// Set the KV cache.
    pub fn set_kv_cache(&mut self, kv: KvCache) {
        self.kv_cache = Some(kv);
    }

    /// Set the DeltaNet state.
    pub fn set_dn_state(&mut self, dn: DeltaNetState) {
        self.dn_state = Some(dn);
    }

    /// Set the scratch.
    pub fn set_scratch(&mut self, scratch: Qwen35Scratch) {
        self.scratch = Some(scratch);
    }

    /// Consume the transaction into a final [`Qwen35Bundle`] on success.
    ///
    /// # Panics
    ///
    /// Panics if any required field is missing (programming error — all
    /// mandatory fields must be set before calling this).
    pub fn into_bundle(mut self, config: Qwen35Config) -> Qwen35Bundle {
        Qwen35Bundle {
            config,
            weights: self.weights.take().expect("transaction: weights missing"),
            scratch: self.scratch.take().expect("transaction: scratch missing"),
            kv_cache: self.kv_cache.take().expect("transaction: KV cache missing"),
            dn_state: self.dn_state.take().expect("transaction: DN state missing"),
            mtp_head: self.mtp_head.take(),
            pipeline: None,
        }
    }

    /// Abort the transaction, returning every owner for cleanup.
    pub fn abort(mut self, message: impl Into<String>) -> Qwen35BundleBuildError {
        Qwen35BundleBuildError {
            message: message.into(),
            weights: self.weights.take().expect("transaction: weights missing"),
            kv_cache: self.kv_cache.take(),
            dn_state: self.dn_state.take(),
            scratch: self.scratch.take(),
            mtp_head: self.mtp_head.take(),
        }
    }
}

/// Drop is a safety net — logs a warning if the transaction is dropped
/// without being consumed via `into_bundle()` or `abort()`.  This is NOT
/// a cleanup path: GPU resources would leak, which is why the transaction
/// is `#[must_use]`.
impl Drop for BundleBuildTransaction {
    fn drop(&mut self) {
        if self.weights.is_some() {
            eprintln!(
                "WARN: BundleBuildTransaction dropped without calling into_bundle() or abort() — \
                 weights and any auxiliary GPU resources would leak. \
                 This is a programming bug."
            );
        }
    }
}

/// Build the Qwen35 GPU bundle from already-loaded weights, config, and
/// the HFQ source (needed for MMQ screening and tiny-model warnings).
///
/// Shared by the Legacy and Frozen load paths.  On failure returns
/// [`Qwen35BundleBuildError`] which preserves weights, every complete
/// auxiliary owner, and any partial-construction retained owners.
#[expect(
    clippy::result_large_err,
    reason = "Err transports every complete GPU owner (weights, KV cache, DN state, scratch, MTP head) for exact rollback; flattening would leak on failure"
)]
fn build_qwen35_bundle(
    hfq: &HfqFile,
    config: Qwen35Config,
    weights: Qwen35Weights,
    ctx: &mut LoadCtx,
) -> Result<Qwen35Bundle, Qwen35BundleBuildError> {
    let mut tx = BundleBuildTransaction::new(weights);

    // Helper: on allocation failure, abort the tx and return the error.
    macro_rules! try_or_abort {
        ($expr:expr, $msg:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return Err(tx.abort(format!("{}: {}", $msg, e))),
            }
        };
    }

    // Helper: on parse/validation failure (no GPU alloc yet), abort tx.
    macro_rules! try_or_abort_msg {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return Err(tx.abort(format!("{}", e))),
            }
        };
    }

    // ── MMQ screening (read-only) ────────────────────────────
    if ctx.gpu.mmq_screen.enabled
        && matches!(
            ctx.gpu.arch.as_str(),
            "gfx906"
                | "gfx1100"
                | "gfx1101"
                | "gfx1102"
                | "gfx1103"
                | "gfx1150"
                | "gfx1151"
                | "gfx1152"
        )
    {
        let t0 = std::time::Instant::now();
        let weights_ref = tx.weights();
        let (n_safe, n_unsafe) = screen_weights_qwen35(weights_ref, ctx.gpu);
        let elapsed = t0.elapsed();
        eprintln!(
            "  MMQ screening: {n_safe} safe, {n_unsafe} unsafe (threshold={:.2}, {:.1}ms)",
            ctx.gpu.mmq_screen.threshold,
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    // ── KV mode ──────────────────────────────────────────────
    let kv_mode = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());

    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();

    let ResolveResult { mode, warning } =
        kv_mode::resolve(&kv_mode, &kv_mode::QWEN35_HFQ_POLICY, config.head_dim);
    if let Some(w) = warning {
        eprintln!("  KV cache: {w} (site {})", kv_mode::QWEN35_HFQ_POLICY.site);
    }
    let dims = KvDims {
        layers: KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: ctx.kv_physical_cap.or(Some(ctx.max_seq)),
    };
    let kv = try_or_abort!(
        KvCache::from_mode(mode, KvTarget::Single(ctx.gpu), &dims),
        "KV cache construction"
    );
    tx.set_kv_cache(kv);

    // ── V-mode override via env ──────────────────────────────
    let kv_v_env = std::env::var("HIPFIRE_KV_V").unwrap_or_default();
    let v_mode_override = match kv_v_env.as_str() {
        "lloyd2" => Some(llama::VMode::Lloyd2),
        "lloyd3" => Some(llama::VMode::Lloyd3),
        "lloyd4" => Some(llama::VMode::Lloyd4),
        "q8" | "" => None,
        other => {
            eprintln!("[hipfire-arch-qwen35] HIPFIRE_KV_V='{other}' unknown — ignoring (expected q8|lloyd2|lloyd3|lloyd4)");
            None
        }
    };
    if let Some(vm) = v_mode_override {
        let kv = tx.kv_cache();
        if (kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4) && kv.quant_fwht {
            let kv_mut = tx.kv_cache_mut();
            try_or_abort!(kv_mut.set_v_mode_realloc(ctx.gpu, vm), "V-mode realloc");
            eprintln!(
                    "[hipfire-arch-qwen35] V-cache mode override → {kv_v_env} (256-wide lloyd-V on fwht K)"
                );
        } else {
            eprintln!("[hipfire-arch-qwen35] HIPFIRE_KV_V={kv_v_env} ignored — lloyd-V requires an FWHT K mode (fwht2/3/4); cache is a different mode");
        }
    }

    // ── KV adaptive ──────────────────────────────────────────
    let kv_adaptive_spec = ctx
        .kv_adaptive_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_ADAPTIVE").unwrap_or_default());

    let _kv_adaptive: Option<KvAdaptive> = {
        match parse_kv_adaptive(&kv_adaptive_spec) {
            None => None,
            Some((preset, k_floor, v_floor)) => {
                let ad = match preset {
                    Some(p) => {
                        KvAdaptive::from_preset(p, ctx.max_seq, config.n_kv_heads, config.head_dim)
                    }
                    None => KvAdaptive::new(
                        ctx.max_seq,
                        config.n_kv_heads,
                        config.head_dim,
                        k_floor,
                        v_floor,
                    ),
                };
                let kv = tx.kv_cache();
                if !((kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4) && kv.quant_fwht) {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — adaptive KV requires an FWHT K mode (fwht2/3/4); cache is a different mode");
                    None
                } else if ctx.cask.sidecar.is_some() {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — adaptive KV is a no-eviction capacity strategy and CASK eviction is active (mutually exclusive)");
                    None
                } else if ad.current_cap() < hipfire_runtime::llama::PREFILL_MAX_BATCH {
                    eprintln!(
                            "[hipfire-arch-qwen35] kv_adaptive={kv_adaptive_spec} ignored — max_seq={} too small: start-tier capacity {} < prefill chunk {}",
                            ctx.max_seq, ad.current_cap(), hipfire_runtime::llama::PREFILL_MAX_BATCH,
                        );
                    None
                } else {
                    if !kv.quant_asym4 {
                        eprintln!("[hipfire-arch-qwen35] kv_adaptive: adaptive works best with kv_mode=fwht4 (K starts at fwht4); current K mode is not fwht4");
                    }
                    let k_floor_bph = k_floor.bytes_per_head(config.head_dim);
                    let kv_mut = tx.kv_cache_mut();
                    try_or_abort!(
                        kv_mut.set_adaptive_floor_alloc(ctx.gpu, v_floor, k_floor_bph),
                        "KV adaptive floor alloc"
                    );
                    eprintln!(
                            "[adaptive-kv] engaged: pattern={:?} k_floor={:?} v_floor={:?} thresholds={:?} start_cap={} (max_seq={}, V buffer sized at floor)",
                            ad.steps, ad.k_floor, ad.v_floor, ad.thresholds, ad.current_cap(), ctx.max_seq,
                        );
                    Some(ad)
                }
            }
        }
    };

    // ── DeltaNet state ───────────────────────────────────────
    let dn_quant = try_or_abort_msg!(parse_state_quant(ctx.state_quant_override));
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    warn_tiny_model_state(hfq, dn_quant);
    let dn = try_or_abort!(
        DeltaNetState::new_with_quant(ctx.gpu, &config, dn_quant),
        "DeltaNet state construction"
    );
    tx.set_dn_state(dn);

    // ── Scratch ──────────────────────────────────────────────
    let scratch = try_or_abort!(
        Qwen35Scratch::new_with_kv_max(ctx.gpu, &config, 2048, ctx.max_seq),
        "Scratch construction"
    );
    tx.set_scratch(scratch);

    Ok(tx.into_bundle(config))
}

/// Build the Qwen35 GPU bundle from an HFQ source (Legacy path).
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen35Bundle, String> {
    let ModelSource::Hfq(hfq) = src else {
        return Err("qwen35: directory source unsupported".into());
    };

    let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let weights = crate::store::load_qwen35_hfq_weights(&hfq, &config, ctx.gpu)?;
    build_qwen35_bundle(&hfq, config, weights, ctx).map_err(|e| e.message)
}

/// RAII guard owning the vision owner until the planned Frozen load
/// resolves.
///
/// On ANY exit without success — an `Err` from the load steps OR a
/// panic unwinding through this frame — `Drop` invokes the abort
/// closure EXACTLY ONCE, moving the owner in (it is never dropped
/// silently).  Abort-on-Drop runs during unwinding too, so a panicking
/// load step cannot leak the vision owner.  On success the caller
/// disarms the guard with [`Self::disarm`], taking the owner back and
/// suppressing the abort.  Panics from the guarded steps PROPAGATE —
/// the guard does not catch them.
///
/// The exactly-once property is structural: the owner lives in
/// `Option<V>` and is moved out exactly once — into `abort` by `Drop`,
/// or back to the caller by `disarm`.
struct PlannedVisionOwnerGuard<V, F: FnOnce(V)> {
    owner: Option<V>,
    abort: Option<F>,
}

impl<V, F: FnOnce(V)> PlannedVisionOwnerGuard<V, F> {
    fn new(owner: V, abort: F) -> Self {
        Self {
            owner: Some(owner),
            abort: Some(abort),
        }
    }

    /// Success path: take the owner back and suppress the abort.
    fn disarm(mut self) -> V {
        self.abort.take();
        self.owner
            .take()
            .expect("PlannedVisionOwnerGuard owner present until disarm")
    }
}

impl<V, F: FnOnce(V)> Drop for PlannedVisionOwnerGuard<V, F> {
    fn drop(&mut self) {
        // Runs on normal scope exit AND on panic unwind.
        if let (Some(owner), Some(abort)) = (self.owner.take(), self.abort.take()) {
            abort(owner);
        }
    }
}

/// Outcome of the arch-owned planned Frozen load: the bundle plus the
/// vision-tower result (uploaded after target verification from the
/// SEALED plan source).
pub struct Qwen35FrozenPlannedOutcome<V> {
    pub bundle: Qwen35Bundle,
    pub vision: V,
}

/// Build the Qwen35 GPU bundle from a preflighted Frozen plan,
/// performing the vision-tower upload inside this arch-owned operation.
///
/// The plan was produced by [`preflight_qwen35_frozen`]; it OWNS the
/// exact HFQ source (SEALED — no public source access), the parsed
/// config, the validated partitioned manifest, and the dispatch
/// eligibility snapshot — this entry accepts ONLY the plan and the load
/// context, with no independent source or config argument.  No manifest
/// work or source payload read is repeated here.  Once this entry point
/// is reached the selection is final — an operational error returns
/// [`Qwen35LoadError`] and the caller MUST NOT fall back to the Legacy
/// path.
///
/// Sequence (all inside this function, BEFORE any Frozen allocation):
/// 1. Target verification — the GPU arch must match the plan's
///    eligibility snapshot.
/// 2. `vision` — invoked with an IMMUTABLE borrow of the plan's sealed
///    source (the caller can neither mutate nor replace the artifact)
///    plus the GPU; it returns the vision owner/result.
/// 3. The planned Frozen load consumes the plan (source + config +
///    prepared manifest + dispatch snapshot — no re-resolution, no env
///    re-read after allocation).
///
/// If the bundle load fails after the vision upload succeeded,
/// `vision_abort` is invoked with the vision owner so the caller frees
/// it — the owner is never dropped silently.
///
/// Pre-publication common/auxiliary rollback inside this path is
/// best-effort per the accepted STEP-002R debt (see the tracker): any
/// owner surfaced by the existing rollback API is retained; exact
/// failed-free retention is NOT claimed.
#[expect(
    clippy::result_large_err,
    reason = "Err preserves the complete cleanup aggregate (retained tensors + frozen SingleFreeFailed owners) for the loader backlog; flattening would drop owners"
)]
pub fn load_bundle_frozen_planned<V>(
    plan: crate::store::Qwen35FrozenPlan,
    ctx: &mut LoadCtx,
    vision: impl FnOnce(&HfqFile, &mut Gpu) -> Result<V, String>,
    vision_abort: impl FnOnce(V, &mut Gpu),
) -> Result<Qwen35FrozenPlannedOutcome<V>, Qwen35LoadError> {
    // 1. Verify the target GPU matches the plan's selection BEFORE any
    //    allocation (vision or Frozen): the GPU arch must equal the arch
    //    the eligibility snapshot was resolved for.
    plan.verify_target(&ctx.gpu.arch)
        .map_err(Qwen35LoadError::common_failure)?;

    // 2. Vision upload from the SEALED source (immutable borrow only).
    let vision_owner = match vision(&plan.hfq, ctx.gpu) {
        Ok(v) => v,
        Err(e) => {
            return Err(Qwen35LoadError::common_failure(format!("vision load: {e}")));
        }
    };

    // 3. Planned Frozen load — consumes the plan.  The vision owner is
    // wrapped in an abort-on-Drop guard for the ENTIRE post-vision
    // sequence: weight-load failure, bundle-build failure, AND a panic
    // unwinding through this frame all abort the owner exactly once;
    // success disarms the guard and returns the owner with the bundle.
    let crate::store::Qwen35FrozenPlan {
        hfq,
        config,
        prepared,
        dispatch_ctx,
        moe_awq_enabled,
    } = plan;

    // The guard's abort closure and the load steps both need the GPU.
    // Route it through a cell so the borrows are strictly sequential
    // (the steps' borrows are released before the guard's Drop runs —
    // on unwind, inner locals drop before outer ones — so the abort
    // closure's borrow cannot be active; `try_borrow_mut` is a
    // defensive no-double-panic fallback).
    let gpu_cell = std::cell::RefCell::new(&mut *ctx.gpu);
    let guard = PlannedVisionOwnerGuard::new(vision_owner, |vision_owner| {
        match gpu_cell.try_borrow_mut() {
            Ok(mut gpu) => vision_abort(vision_owner, &mut gpu),
            Err(_) => eprintln!(
                "[hipfire-arch-qwen35] BUG: vision abort during an active GPU borrow;                  owner dropped unfreed"
            ),
        }
    });

    // Sequential GPU access through the cell; rebuild a step context
    // sharing the cell-borrowed GPU (all other LoadCtx fields are
    // Copy).  Declared AFTER the guard so unwind drops the borrow
    // before the guard's abort runs.
    let mut gpu_ref = gpu_cell.borrow_mut();
    let mut step_ctx = hipfire_runtime::loader_api::LoadCtx {
        gpu: *gpu_ref,
        path: ctx.path,
        max_seq: ctx.max_seq,
        draft_path: ctx.draft_path,
        kv_mode_override: ctx.kv_mode_override,
        kv_adaptive_override: ctx.kv_adaptive_override,
        state_quant_override: ctx.state_quant_override,
        cask: ctx.cask,
        pp: ctx.pp,
        pp_bands: ctx.pp_bands,
        mtp_mode: ctx.mtp_mode,
        mtp_k: ctx.mtp_k,
        spec: ctx.spec,
        kv_physical_cap: ctx.kv_physical_cap,
    };

    let weights = crate::store::load_qwen35_hfq_weights_frozen_prepared(
        prepared,
        &hfq,
        &config,
        &dispatch_ctx,
        moe_awq_enabled,
        step_ctx.gpu,
    )?;

    // Use the transaction-based bundle build. On failure the error
    // carries weights + every complete/partial auxiliary owner in
    // Qwen35BundleBuildError.  Convert to Qwen35LoadError preserving
    // ownership. The COMPLETE cleanup aggregate — failed_tensors AND
    // frozen SingleFreeFailed owners — is preserved wholesale (never
    // flattened into tensor retainers).
    let bundle = match build_qwen35_bundle(&hfq, config, weights, &mut step_ctx) {
        Ok(bundle) => bundle,
        Err(build_err) => {
            // Build error carries weights + complete auxiliaries. Free
            // every domain independently; collect ALL retained failures.
            let (msg, kv_failures, dn_failures, scratch_failures, weights_failure) =
                build_err.try_free(step_ctx.gpu);

            // Convert the KV/DN/scratch retained domains to
            // RetainedQwenTensor.
            let mut all_retained: Vec<crate::qwen35::RetainedQwenTensor> = Vec::new();
            for (label, tensor) in kv_failures {
                all_retained.push(crate::qwen35::RetainedQwenTensor {
                    label,
                    tensor,
                    last_error: "KvCache::free_checked failed".into(),
                });
            }
            all_retained.extend(dn_failures);
            all_retained.extend(scratch_failures);

            // `weights_failure` is the COMPLETE Qwen35CleanupFailure from
            // the abort (including its frozen SingleFreeFailed owners).
            // Pass it through whole — the loader enqueues it wholesale.
            return Err(Qwen35LoadError::common_failure_with_cleanup_aggregate(
                format!("bundle build after frozen weight load: {msg}"),
                all_retained,
                weights_failure,
            ));
        }
    };

    // Success: take the owner back and suppress the abort.
    let vision = guard.disarm();
    Ok(Qwen35FrozenPlannedOutcome { bundle, vision })
}

// ─── Helper: StateQuant parsing ─────────────────────────────────────

fn parse_state_quant(mode: Option<&str>) -> Result<StateQuant, String> {
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

fn state_quant_label(q: StateQuant) -> &'static str {
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

// ─── Helper: MMQ screening (inline from hipfire-loader) ───────────

fn screen_weights_qwen35(weights: &Qwen35Weights, gpu: &mut rdna_compute::Gpu) -> (usize, usize) {
    use crate::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;
    for layer in &weights.layers {
        let wts: Vec<&hipfire_runtime::llama::WeightTensor> = match layer {
            LayerWeights::DeltaNet(l) => {
                vec![
                    &l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up, &l.wo,
                ]
            }
            LayerWeights::FullAttn(l) => {
                vec![&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up, &l.wo]
            }
            LayerWeights::DeltaNetMoe(l) => {
                vec![&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.wo]
            }
            LayerWeights::FullAttnMoe(l) => {
                vec![&l.wq, &l.wk, &l.wv, &l.wo]
            }
        };
        for wt in wts {
            if !matches!(
                wt.gpu_dtype,
                rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256
            ) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }
    (n_safe, n_unsafe)
}

// ─── Helper: parameter count + tiny-model warning ─────────────────

fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

fn warn_tiny_model_state(hfq: &HfqFile, q: StateQuant) {
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS && q != StateQuant::FP32 {
        eprintln!(
            "  warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {})",
            params as f64 / 1.0e9,
            state_quant_label(q)
        );
    }
}

// ─── Helper: KV adaptive parsing ──────────────────────────────────

fn parse_kv_adaptive(
    s: &str,
) -> Option<(
    Option<Preset>,
    hipfire_runtime::kv_adaptive::KMode,
    hipfire_runtime::llama::VMode,
)> {
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;
    match s {
        "" | "off" => None,
        "conservative" => Some((Some(Preset::Conservative), KMode::Fwht4, VMode::Lloyd4)),
        "balanced" => Some((Some(Preset::Balanced), KMode::Fwht2, VMode::Lloyd2)),
        "aggressive" => Some((Some(Preset::Aggressive), KMode::Fwht2, VMode::Lloyd2)),
        other if other.starts_with("advanced:") => {
            let spec = &other["advanced:".len()..];
            let mut k = None;
            let mut v = None;
            for kvp in spec.split(',') {
                let mut it = kvp.splitn(2, '=');
                match (it.next(), it.next()) {
                    (Some("k"), Some("fwht4")) => k = Some(KMode::Fwht4),
                    (Some("k"), Some("fwht3")) => k = Some(KMode::Fwht3),
                    (Some("k"), Some("fwht2")) => k = Some(KMode::Fwht2),
                    (Some("v"), Some("lloyd4")) => v = Some(VMode::Lloyd4),
                    (Some("v"), Some("lloyd3")) => v = Some(VMode::Lloyd3),
                    (Some("v"), Some("lloyd2")) => v = Some(VMode::Lloyd2),
                    _ => {}
                }
            }
            match (k, v) {
                (Some(k), Some(v)) => Some((None, k, v)),
                _ => {
                    eprintln!("[hipfire-arch-qwen35] kv_adaptive='{other}' malformed — expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>; ignoring");
                    None
                }
            }
        }
        other => {
            eprintln!("[hipfire-arch-qwen35] kv_adaptive='{other}' unknown — expected off|conservative|balanced|aggressive|advanced:k=..,v=..; ignoring");
            None
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Vision-owner guard tests (pure, CPU-only)
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::PlannedVisionOwnerGuard;
    use std::cell::Cell;

    #[test]
    fn vision_owner_aborted_exactly_once_on_err() {
        // An Err from the guarded load steps: the guard's Drop aborts
        // the owner exactly once (covers BOTH post-vision failure sites
        // — weight-load failure and bundle-build failure).
        let abort_count = Cell::new(0u32);
        let abort_seen = Cell::new(None);
        let result = {
            let _guard = PlannedVisionOwnerGuard::new(42u64, |v| {
                abort_count.set(abort_count.get() + 1);
                abort_seen.set(Some(v));
            });
            Err::<u32, &str>("weight load failed")
        };
        assert_eq!(result, Err("weight load failed"));
        assert_eq!(abort_count.get(), 1, "abort must run exactly once on Err");
        assert_eq!(
            abort_seen.get(),
            Some(42),
            "abort must receive the exact owner"
        );
    }

    #[test]
    fn vision_owner_aborted_exactly_once_on_panic() {
        // A panic unwinding through the guarded frame: Drop still aborts
        // the owner exactly once, and the panic PROPAGATES.
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let abort_count = Cell::new(0u32);
        let abort_seen = Cell::new(None);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _guard = PlannedVisionOwnerGuard::new(99u64, |v| {
                abort_count.set(abort_count.get() + 1);
                abort_seen.set(Some(v));
            });
            panic!("load step panicked");
        }));
        assert!(
            caught.is_err(),
            "the guarded panic must propagate out of catch_unwind"
        );
        assert_eq!(
            abort_count.get(),
            1,
            "abort must run exactly once on panic unwind"
        );
        assert_eq!(
            abort_seen.get(),
            Some(99),
            "abort must receive the exact owner on panic unwind"
        );
    }

    #[test]
    fn vision_owner_returned_on_success_via_disarm() {
        // Success: disarm takes the owner back and suppresses the abort.
        let abort_count = Cell::new(0u32);
        let abort_seen = Cell::new(None);
        let owner = {
            let guard = PlannedVisionOwnerGuard::new(7u64, |v| {
                abort_count.set(abort_count.get() + 1);
                abort_seen.set(Some(v));
            });
            guard.disarm()
        };
        assert_eq!(owner, 7);
        assert_eq!(abort_count.get(), 0, "abort must never run after disarm");
        assert_eq!(abort_seen.get(), None);
    }

    #[test]
    fn vision_owner_abort_never_runs_twice_for_same_owner() {
        // Exactly-once across repeated scenarios, including a panic.
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let abort_err = Cell::new(0u32);
        {
            let _guard = PlannedVisionOwnerGuard::new(1u64, |_| abort_err.set(abort_err.get() + 1));
            let _: Result<u32, &str> = Err("e");
        }
        let abort_ok = Cell::new(0u32);
        {
            let guard = PlannedVisionOwnerGuard::new(2u64, |_| abort_ok.set(abort_ok.get() + 1));
            let _ = guard.disarm();
        }
        let abort_panic = Cell::new(0u32);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _guard =
                PlannedVisionOwnerGuard::new(3u64, |_| abort_panic.set(abort_panic.get() + 1));
            panic!("boom");
        }));
        assert_eq!(
            (abort_err.get(), abort_ok.get(), abort_panic.get()),
            (1, 0, 1),
            "exactly once per scenario: Err, success (none), panic"
        );
    }
}
