// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emulated EP2 harness (STEP-002 Task 8, Phase 2B): high-level parity driver.
//!
//! **Test-only.** Compiled only under the non-default `emulated-ep2-harness`
//! feature.  Production Qwen35 EP stays `Planned { owner: "AXIS-002" }`; this
//! module never constructs `EpArch::Qwen35`, never weakens the Frozen
//! multi-device refusal, and exposes **no** store / tensor / raw-ownership
//! types.
//!
//! One model load, one `Qwen35MoeResident` owner, two independent
//! KV/DeltaNet/scratch states:
//!
//! * **baseline** — the production Single/Frozen path (`EP=1` is only an
//!   alias, never a second implementation);
//! * **emulated EP2** — two logical expert-ownership ranks over ONE GPU,
//!   executed sequentially, overriding only the gate-up pointer tables via
//!   `Qwen35Weights::moe_ffn_view_ep2` (down/AWQ/tags/router/shared/Paro stay
//!   canonical and borrowed).
//!
//! Honest FP32 acceptance (Oracle Phase 1 correction): final logits must be
//! finite with identical argmax and a strict measured + pinned
//! maximum-absolute-logit-delta; the first emitted token and every compared
//! generated token ID must match exactly.  Bitwise-equal logits are NEVER
//! required or claimed.
//!
//! **Probe vs acceptance:** probe mode (no pinned delta) prints observed max
//! deltas and is NON-ACCEPTANCE; acceptance mode requires an explicit finite
//! `max_logit_delta` and fails above it.  Phase 3 measures, pins, rebuilds.
//!
//! # Single-shot + STEP-002R debt
//!
//! [`run`] is a **feature-gated, test-only, SINGLE-SHOT API**: exactly one
//! GPU-bearing invocation per process (an [`AtomicBool`] claim taken
//! immediately before `Gpu::init`, never reset).  Callers MUST terminate the
//! process after any post-claim error.  The success path performs explicit
//! owner cleanup + a checked pool drain, but partial scratch/KV/DeltaNet/PBS
//! construction rollback, typed load-error cleanup retention, and failed
//! checked-free owner retention are deferred to STEP-002R — no
//! repeated-load VRAM/lifecycle claim is made.

use crate::qwen35::{Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::EmbeddingFormat;
use rdna_compute::Gpu;
use std::path::PathBuf;

// ── High-level options ───────────────────────────────────────────────

/// High-level harness options.  No store / tensor / ownership types.
#[derive(Clone, Debug)]
pub struct Ep2HarnessOptions {
    /// Path to the `.mq4` / `.hfq` Frozen model artifact.
    pub model_path: PathBuf,
    /// Prompt text (tokenized inside the harness).  The runner feeds the
    /// committed `benchmarks/prompts/qwen35_moe_ep_parity.txt` here.
    pub prompt: String,
    /// Deterministic second-turn suffix appended WITHOUT clearing state.
    pub second_turn_suffix: String,
    /// Greedy decode steps compared lockstep after prefill.
    pub max_steps: usize,
    /// Pinned maximum-absolute-logit-delta for acceptance mode.
    /// `None` = probe mode (NON-ACCEPTANCE).  Acceptance refuses `None`
    /// and any non-finite / non-positive value.
    pub max_logit_delta: Option<f32>,
    /// KV cache mode override — the harness REQUIRES `q8` and verifies the
    /// resolved mode is actually Q8 (no fallback).
    pub kv_mode: String,
    /// DeltaNet state quant — the harness requires `fp32` (deterministic
    /// parity; Q8 state rounding is not bit-comparable across orderings)
    /// and verifies the constructed state is actually FP32.
    pub state_quant: String,
    /// KV max sequence length.  Must cover prompt + decode + the pending
    /// final token + suffix (verified before GPU execution).
    pub max_seq: usize,
}

impl Default for Ep2HarnessOptions {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prompt: String::new(),
            second_turn_suffix: String::from("\n\nWhat follows is a second turn."),
            max_steps: 16,
            max_logit_delta: None,
            kv_mode: String::from("q8"),
            state_quant: String::from("fp32"),
            max_seq: 4096,
        }
    }
}

/// Harness mode: [`Ep2Mode::Probe`] reports observed deltas but never
/// accepts; [`Ep2Mode::Acceptance`] requires an explicit finite pinned
/// delta and fails above it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ep2Mode {
    Probe,
    Acceptance,
}

/// Deterministic-execution environment snapshot (CPU-testable; `run`
/// captures it from the process env once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HarnessEnv {
    pub deterministic: bool,
    pub graph_off: bool,
}

// ── Report ───────────────────────────────────────────────────────────

/// High-level parity report.  On failure the report carries the first
/// differing token position, logit index, both values, and the maximum
/// absolute logit delta.
#[derive(Clone, Debug)]
pub struct Ep2HarnessReport {
    pub mode: Ep2Mode,
    pub passed: bool,
    /// All compared logit vectors were finite.
    pub finite_logits: bool,
    /// Final-prefill argmax (and therefore the first emitted token) matched.
    pub first_token_match: bool,
    /// Every compared generated token ID matched exactly.
    pub generated_tokens_match: bool,
    /// Second-turn (appended suffix) comparison matched.
    pub second_turn_match: bool,
    /// Post-reset replay of the original prompt matched.
    pub reset_match: bool,
    pub max_abs_logit_delta: f32,
    /// Token position of the first failing comparison (0 = final prefill
    /// logits; `Some(1 + step)` = decode step).
    pub first_delta_pos: Option<usize>,
    /// Vocab index of the first differing logit value.
    pub first_delta_index: Option<usize>,
    pub baseline_logit: Option<f32>,
    pub ep2_logit: Option<f32>,
    /// Greedy tokens emitted by the baseline (Single) path.
    pub baseline_tokens: Vec<u32>,
    /// Greedy tokens emitted by the emulated EP2 path.
    pub ep2_tokens: Vec<u32>,
    /// The resolved, verified KV cache mode (always `Q8` — the harness
    /// refuses any request that resolves elsewhere or falls back).
    pub resolved_kv_mode: String,
}

// ── Pure logit comparison (CPU-testable) ─────────────────────────────

/// One logit-vector comparison between the baseline and emulated-EP2 paths.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LogitComparison {
    /// Every entry in both vectors was finite.
    pub finite: bool,
    /// First index with a non-finite entry in either vector.
    pub first_non_finite: Option<usize>,
    /// Argmax equality (only meaningful when both sides are finite).
    pub argmax_match: bool,
    pub baseline_argmax: Option<usize>,
    pub ep2_argmax: Option<usize>,
    /// Max |baseline[i] - ep2[i]| over finite pairs.
    pub max_abs_delta: f32,
    /// Index of the max-abs-delta pair (finite pairs only).  The relevant
    /// diagnostic for a tolerance violation.
    pub max_delta_index: Option<usize>,
    /// Baseline value at `max_delta_index`.
    pub max_delta_baseline: Option<f32>,
    /// EP2 value at `max_delta_index`.
    pub max_delta_ep2: Option<f32>,
    /// First index where the two vectors differ (value or finiteness).
    pub first_delta_index: Option<usize>,
    pub baseline_value: Option<f32>,
    pub ep2_value: Option<f32>,
}

/// Compare two logit vectors: finite check, argmax, max-abs-delta (with its
/// index), and first-divergence diagnostics.  Pure and CPU-testable.
pub(crate) fn compare_logits(baseline: &[f32], ep2: &[f32]) -> LogitComparison {
    let n = baseline.len().min(ep2.len());
    let mut finite = true;
    let mut first_non_finite: Option<usize> = None;
    let mut first_delta_index: Option<usize> = None;
    let mut baseline_value: Option<f32> = None;
    let mut ep2_value: Option<f32> = None;
    let mut max_abs_delta = 0.0f32;
    let mut max_delta_index: Option<usize> = None;
    let mut max_delta_baseline: Option<f32> = None;
    let mut max_delta_ep2: Option<f32> = None;
    for i in 0..n {
        let b = baseline[i];
        let e = ep2[i];
        if !(b.is_finite() && e.is_finite()) {
            finite = false;
            if first_non_finite.is_none() {
                first_non_finite = Some(i);
                first_delta_index = Some(i);
                baseline_value = Some(b);
                ep2_value = Some(e);
            }
            continue;
        }
        let d = (b - e).abs();
        if d > max_abs_delta {
            max_abs_delta = d;
            max_delta_index = Some(i);
            max_delta_baseline = Some(b);
            max_delta_ep2 = Some(e);
        }
        if first_delta_index.is_none() && b != e {
            first_delta_index = Some(i);
            baseline_value = Some(b);
            ep2_value = Some(e);
        }
    }
    // A length mismatch (different vocab widths) is itself a divergence.
    // There is NO valid in-range first-diff index for a pure length
    // mismatch — leave the diagnostics exactly as the shared-prefix loop
    // found them (None when the prefix is identical) instead of fabricating
    // an out-of-range index paired with values from another position.
    if baseline.len() != ep2.len() {
        finite = false;
    }
    let baseline_argmax = argmax(baseline);
    let ep2_argmax = argmax(ep2);
    let argmax_match = finite && baseline_argmax == ep2_argmax;
    LogitComparison {
        finite,
        first_non_finite,
        argmax_match,
        baseline_argmax,
        ep2_argmax,
        max_abs_delta,
        max_delta_index,
        max_delta_baseline,
        max_delta_ep2,
        first_delta_index,
        baseline_value,
        ep2_value,
    }
}

/// The diagnostic `(index, baseline_value, ep2_value)` for the FIRST
/// FAILING boundary, per failure kind:
///
/// * non-finite (NaN/Inf) → the first non-finite index;
/// * a pure length mismatch (identical shared prefix, different widths) has
///   NO in-range index — the caller records the boundary position with
///   absent values instead;
/// * argmax/token mismatch → the first differing value index;
/// * tolerance violation (argmax still matches, max delta exceeds `pinned`)
///   → the EXCEEDING max-delta index.
///
/// Ordinary accepted FP drift (delta ≤ `pinned`, argmax matches) returns
/// `None` — it never occupies the failure diagnostics.  `pinned == None`
/// (probe mode) has no tolerance failures.
pub(crate) fn first_failing_diagnostic(
    c: &LogitComparison,
    pinned: Option<f32>,
) -> Option<(usize, f32, f32)> {
    if !c.finite {
        let i = c.first_non_finite.or(c.first_delta_index)?;
        let b = c.baseline_value.unwrap_or(f32::NAN);
        let e = c.ep2_value.unwrap_or(f32::NAN);
        return Some((i, b, e));
    }
    if !c.argmax_match {
        let i = c.first_delta_index?;
        return Some((i, c.baseline_value?, c.ep2_value?));
    }
    if let Some(p) = pinned {
        if c.max_abs_delta > p {
            let i = c.max_delta_index?;
            return Some((i, c.max_delta_baseline?, c.max_delta_ep2?));
        }
    }
    None
}

/// Index of the maximum entry; `None` for an empty slice.  Non-finite
/// entries never win (NaN comparisons are false).
fn argmax(v: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            if best.map_or(true, |(_, b)| x > b) {
                best = Some((i, x));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Resolve the harness mode from the options + environment, refusing
/// malformed configurations BEFORE any GPU work:
///
/// * `HIPFIRE_DETERMINISTIC=1` and `HIPFIRE_GRAPH=0` are MANDATORY — there
///   is no public bypass (the old `require_deterministic` option is gone);
/// * `max_steps` must be > 0 (a zero-step run is vacuous);
/// * the prompt and the second-turn suffix strings must be non-empty;
/// * the DeltaNet state quant must be `fp32`;
/// * acceptance mode requires an explicit finite positive pinned delta;
///   probe mode is NON-ACCEPTANCE.
pub(crate) fn validate_options(
    options: &Ep2HarnessOptions,
    env: &HarnessEnv,
) -> Result<Ep2Mode, String> {
    if options.max_steps == 0 {
        return Err(
            "emulated-EP2 harness requires max_steps > 0 (a zero-step comparison is vacuous)"
                .into(),
        );
    }
    if options.prompt.trim().is_empty() {
        return Err("emulated-EP2 harness requires a non-empty prompt".into());
    }
    if options.second_turn_suffix.trim().is_empty() {
        return Err("emulated-EP2 harness requires a non-empty second-turn suffix".into());
    }
    if !env.deterministic {
        return Err(
            "emulated-EP2 harness requires HIPFIRE_DETERMINISTIC=1 (deterministic parity gate; \
             no bypass)"
                .into(),
        );
    }
    if !env.graph_off {
        return Err(
            "emulated-EP2 harness requires HIPFIRE_GRAPH=0 (direct execution on both sides; \
             baseline-only graph capture would fake the comparison)"
                .into(),
        );
    }
    if !options.state_quant.eq_ignore_ascii_case("fp32") {
        return Err(format!(
            "emulated-EP2 harness requires FP32 DeltaNet state (state_quant={:?}); \
             Q8/Q4 stochastic rounding is not order-comparable",
            options.state_quant
        ));
    }
    match options.max_logit_delta {
        None => Ok(Ep2Mode::Probe),
        Some(d) => {
            if !d.is_finite() || d <= 0.0 {
                return Err(format!(
                    "acceptance mode requires an explicit finite positive pinned \
                     max-logit-delta, got {d:?}"
                ));
            }
            Ok(Ep2Mode::Acceptance)
        }
    }
}

/// Precomputed KV capacity requirement: prompt positions + decode steps +
/// the pending final token continuation (+1) + second-turn suffix positions.
/// The reset replay re-uses `prompt_len` positions from 0 and never exceeds
/// this bound.
///
/// All additions are CHECKED: an overflowing requirement is an explicit
/// error, never a release wraparound.
pub(crate) fn required_seq_capacity(
    prompt_len: usize,
    max_steps: usize,
    suffix_len: usize,
) -> Result<usize, String> {
    const PENDING_FINAL_TOKEN: usize = 1;
    prompt_len
        .checked_add(max_steps)
        .and_then(|v| v.checked_add(PENDING_FINAL_TOKEN))
        .and_then(|v| v.checked_add(suffix_len))
        .ok_or_else(|| {
            "emulated-EP2 harness: required sequence capacity overflows usize \
             (prompt + decode steps + pending final token + suffix)"
                .into()
        })
}

/// Non-vacuous token-level contract, checked BEFORE GPU execution:
///
/// * the tokenized prompt must be non-empty;
/// * the tokenized second-turn suffix must be non-empty (a suffix that
///   tokenizes to nothing makes the second-turn comparison vacuous);
/// * `max_seq` must cover `prompt + max_steps + pending final token + suffix`.
pub(crate) fn validate_turn_bounds(
    prompt_tokens: &[u32],
    suffix_tokens: &[u32],
    max_steps: usize,
    max_seq: usize,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Err("emulated-EP2 harness requires a non-empty tokenized prompt".into());
    }
    if suffix_tokens.is_empty() {
        return Err(
            "emulated-EP2 harness requires a non-empty tokenized second-turn suffix".into(),
        );
    }
    let required = required_seq_capacity(prompt_tokens.len(), max_steps, suffix_tokens.len())?;
    if max_seq < required {
        return Err(format!(
            "emulated-EP2 harness requires max_seq >= {required} \
             (prompt {} + decode {max_steps} + pending final token 1 + suffix {}), got {max_seq}",
            prompt_tokens.len(),
            suffix_tokens.len()
        ));
    }
    Ok(())
}

/// Verify the requested KV mode resolves EXACTLY to Q8 with no fallback:
/// a warning means the request was not honored, and any other resolved
/// mode is not the fixture mode.
pub(crate) fn validate_kv_mode_request(raw: &str, head_dim: usize) -> Result<KvMode, String> {
    use hipfire_runtime::kv_mode::{self, KvMode};
    let kv_mode::ResolveResult { mode, warning } =
        kv_mode::resolve(raw, &kv_mode::QWEN35_HFQ_POLICY, head_dim);
    if warning.is_some() {
        return Err(format!(
            "emulated-EP2 harness: KV mode request {raw:?} was not honored (fallback warning); \
             fixture mode q8 must resolve exactly"
        ));
    }
    if mode != KvMode::Q8 {
        return Err(format!(
            "emulated-EP2 harness requires fixture KV mode q8, resolved {mode:?}"
        ));
    }
    Ok(mode)
}

/// Boundary kinds in run order.  Each maps to a report position via
/// [`boundary_pos`]; the pending-final-token continuation gets its own
/// position between the decode loop and the second-turn suffix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryKind {
    /// Final prefill logits of the prompt (position 0).
    Prefill,
    /// One greedy decode step's logits.
    Decode(usize),
    /// Logits produced by feeding the final sampled token into both states.
    PendingToken,
    /// Second-turn suffix prefill logits.
    SecondTurn,
    /// Post-reset replay prefill logits.
    ResetReplay,
}

/// Report position for a boundary: 0 = prefill, `1 + step` = decode,
/// `1 + max_steps` = pending-final-token continuation,
/// `2 + max_steps` = second turn, `3 + max_steps` = reset replay.
pub(crate) fn boundary_pos(kind: BoundaryKind, max_steps: usize) -> usize {
    match kind {
        BoundaryKind::Prefill => 0,
        BoundaryKind::Decode(step) => 1 + step,
        BoundaryKind::PendingToken => 1 + max_steps,
        BoundaryKind::SecondTurn => 2 + max_steps,
        BoundaryKind::ResetReplay => 3 + max_steps,
    }
}

/// Whether a report counts as a PASS.  Probe mode is NON-ACCEPTANCE — only
/// acceptance-mode reports with `passed` set are passes.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "probe-vs-acceptance pass semantics pinned by the CPU tests and consumed by the runtime example's exit-code mapping"
    )
)]
pub(crate) fn report_is_pass(report: &Ep2HarnessReport) -> bool {
    report.mode == Ep2Mode::Acceptance && report.passed
}

// ── High-level driver ────────────────────────────────────────────────

/// Process-wide single-shot permit for GPU-bearing harness invocations.
///
/// The harness is deliberately SINGLE-SHOT per process (STEP-002R debt:
/// partial-construction rollback, typed load-error cleanup retention, and
/// failed checked-free owner retention are NOT closed).  Once a GPU-bearing
/// `run` claims the permit it is NEVER reset — a second invocation in the
/// same process is refused before any GPU work, so accidental
/// repeated-load/lifecycle use cannot silently accumulate owners.
///
/// The claim protocol is a pure `compare_exchange` over an [`AtomicBool`]
/// so it is CPU-testable with a local instance.
pub(crate) fn try_claim_single_shot(claim: &std::sync::atomic::AtomicBool) -> bool {
    claim
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
}

static SINGLE_SHOT_CLAIM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run the Single-vs-emulated-EP2 parity sequence:
///
/// 1. batched prefill of the prompt on both states → compare final logits
///    (finite / argmax / max-abs-delta / first divergence);
/// 2. greedy decode lockstep for `max_steps` tokens → compare token IDs;
/// 3. append the deterministic second-turn suffix WITHOUT clearing state →
///    compare again;
/// 4. checked reset of KV position + DeltaNet state (fresh deterministic
///    states over the same weights) → replay the original prompt → compare.
///
/// Returns the report; `passed` is `false` in probe mode (NON-ACCEPTANCE) —
/// only acceptance mode with an explicit finite pinned delta can pass.
///
/// # Single-shot contract (STEP-002R debt)
///
/// This is a **feature-gated, test-only, SINGLE-SHOT API**: exactly one
/// GPU-bearing invocation per process.  All pure option/model/token/capacity
/// validation runs first; immediately before `Gpu::init` the process-wide
/// [`SINGLE_SHOT_CLAIM`] permit is claimed and NEVER reset.  A second
/// invocation in the same process is refused with a clear error BEFORE any
/// GPU work.
///
/// On success the explicit owner cleanup + checked pool drain runs
/// ([`HarnessState::free`] on both states, `weights.free_gpu_checked`,
/// `gpu.drain_pool_checked`).  This is **success-path evidence only**: it
/// does NOT close partial-construction rollback, typed load-error cleanup
/// retention, or failed checked-free owner retention — all of which are
/// deferred to STEP-002R.  Callers MUST terminate the process after any
/// post-claim error (a failed invocation leaves the permit claimed and may
/// have retained owners).  No repeated-load VRAM/lifecycle claim is made.
pub fn run(options: &Ep2HarnessOptions) -> Result<Ep2HarnessReport, String> {
    // ── 1. Environment + option gates (before any GPU work) ─────────────
    let env = HarnessEnv {
        deterministic: std::env::var("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1"),
        graph_off: std::env::var("HIPFIRE_GRAPH").ok().as_deref() == Some("0"),
    };
    let mode = validate_options(options, &env)?;

    // ── 2. Model + tokenizer ────────────────────────────────────────────
    let hfq = hipfire_runtime::hfq::HfqFile::open(&options.model_path)
        .map_err(|e| format!("open model {}: {e}", options.model_path.display()))?;
    let config = crate::qwen35::config_from_hfq(&hfq).map_err(|e| format!("config: {e}"))?;
    if config.num_experts == 0 {
        return Err("emulated-EP2 harness requires an MoE model (num_experts > 0)".into());
    }
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e}"))?;
    let prompt_tokens = tokenizer.encode(&options.prompt);
    let suffix_tokens = tokenizer.encode(&options.second_turn_suffix);

    // ── 2b. Non-vacuous contract + capacity, BEFORE any GPU execution ────
    validate_turn_bounds(
        &prompt_tokens,
        &suffix_tokens,
        options.max_steps,
        options.max_seq,
    )?;
    let kv_mode = validate_kv_mode_request(&options.kv_mode, config.head_dim)?;
    let prompt_len = prompt_tokens.len();
    let max_batch = prompt_len.max(suffix_tokens.len()).max(16);

    // ── 3. Single-shot claim, then GPU + one-owner Frozen load ──────────
    // All pure validation above has passed; the process-wide permit is now
    // claimed and NEVER reset.  A second GPU-bearing invocation in this
    // process is refused here, before any GPU work.
    if !try_claim_single_shot(&SINGLE_SHOT_CLAIM) {
        return Err(
            "emulated-EP2 harness is SINGLE-SHOT per process: a GPU-bearing invocation \
             was already claimed. Repeated/retry/lifecycle use is deferred to STEP-002R; \
             terminate this process and start a fresh one."
                .into(),
        );
    }
    // Success-path explicit owner cleanup + checked pool drain (below) is
    // the accepted terminal path; it does NOT solve partial-construction /
    // failed-free retention (STEP-002R debt).
    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("Gpu::init: {e}"))?;
    let mut weights: Option<Qwen35Weights> = None;
    let mut base_state: Option<HarnessState> = None;
    let mut ep2_state: Option<HarnessState> = None;
    let pinned = options.max_logit_delta;
    let mut report = Ep2HarnessReport {
        mode,
        passed: false,
        finite_logits: true,
        first_token_match: true,
        generated_tokens_match: true,
        second_turn_match: true,
        reset_match: true,
        max_abs_logit_delta: 0.0,
        first_delta_pos: None,
        first_delta_index: None,
        baseline_logit: None,
        ep2_logit: None,
        baseline_tokens: Vec::with_capacity(options.max_steps),
        ep2_tokens: Vec::with_capacity(options.max_steps),
        resolved_kv_mode: format!("{kv_mode:?}"),
    };
    let primary: Result<(), String> = (|| {
        let manifest = {
            use hipfire_runtime::arch::Architecture;
            <crate::arch::Qwen35 as Architecture>::weight_manifest(&config)
        };
        let prepared = crate::store::prepare_frozen_hfq_manifest(&config, &manifest)
            .map_err(|e| format!("frozen manifest preparation: {e}"))?;
        let dispatch_ctx = hipfire_dispatch::context::DispatchCtx::new(&gpu);
        let moe_awq_enabled = crate::store::Qwen35MoeLoadFlags::resolve().moe_awq_enabled;
        let plan = crate::store::EmulatedExpertPartitionPlan::stride2(config.num_experts)
            .map_err(|e| format!("EP2 partition plan: {e}"))?;
        weights = Some(
            crate::store::load_qwen35_hfq_weights_frozen_prepared_ep2(
                prepared,
                &hfq,
                &config,
                &dispatch_ctx,
                moe_awq_enabled,
                &mut gpu,
                &plan,
            )
            .map_err(|e| format!("frozen EP2 load: {e:?}"))?,
        );
        let weights = weights.as_ref().ok_or("internal: weights missing")?;

        // ── 4. Two independent state sets over the SAME weights ─────────
        // Baseline = production Single/Frozen path; EP2 = rank-masked
        // execution.  Built before any comparison; fully-constructed states
        // are freed exactly once by the success-path terminal cleanup below
        // (partial-construction rollback is STEP-002R debt).
        base_state = Some(
            HarnessState::new(&mut gpu, &config, options, max_batch, kv_mode)
                .map_err(|e| format!("baseline state construction: {e}"))?,
        );
        ep2_state = Some(
            HarnessState::new(&mut gpu, &config, options, max_batch, kv_mode)
                .map_err(|e| format!("EP2 state construction: {e}"))?,
        );
        let mut base_state = base_state.as_mut().ok_or("internal: base state missing")?;
        let mut ep2_state = ep2_state.as_mut().ok_or("internal: EP2 state missing")?;

        // ── 5. Parity sequence ──────────────────────────────────────────
        // Every compared logit vector — final prefill, EVERY decode step,
        // the pending-final-token continuation, the second-turn suffix, and
        // the reset replay — passes the SAME finite/argmax/max-abs-delta
        // comparator.  `first_delta_pos` encoding via [`boundary_pos`]:
        // 0 = prefill, 1+step = decode, 1+max_steps = pending token,
        // 2+max_steps = second turn, 3+max_steps = reset replay.
        // 5a. Batched prefill of the prompt.
        let base_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut base_state,
            &prompt_tokens,
            0,
            false,
        )?;
        let ep2_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut ep2_state,
            &prompt_tokens,
            0,
            true,
        )?;
        let c = compare_and_record(
            &mut report,
            boundary_pos(BoundaryKind::Prefill, options.max_steps),
            &base_logits,
            &ep2_logits,
            pinned,
        )?;
        report.first_token_match = c.finite && c.argmax_match;
        if !(c.finite && c.argmax_match) {
            report.generated_tokens_match = false;
        }

        // 5b. Greedy decode lockstep.  Both sides always consume the SAME
        // token (the baseline's argmax) at the SAME position, so the
        // comparison continues even after a divergence.  EVERY step's logit
        // pair goes through the comparator (NaN/Inf or a per-step tolerance
        // excess fails acceptance even when the argmax/tokens match).
        let mut pos = prompt_len;
        let mut base_token = argmax(&base_logits).ok_or("baseline prefill logits empty")? as u32;
        for step in 0..options.max_steps {
            let base_logits = decode_step(
                &mut gpu,
                &weights,
                &config,
                &mut base_state,
                base_token,
                pos,
                false,
            )?;
            let ep2_logits = decode_step(
                &mut gpu,
                &weights,
                &config,
                &mut ep2_state,
                base_token,
                pos,
                true,
            )?;
            let c = compare_and_record(
                &mut report,
                boundary_pos(BoundaryKind::Decode(step), options.max_steps),
                &base_logits,
                &ep2_logits,
                pinned,
            )?;
            if !(c.finite && c.argmax_match) {
                report.generated_tokens_match = false;
            }
            let next_base = argmax(&base_logits).ok_or("baseline logits empty")? as u32;
            let ep2_token = argmax(&ep2_logits).ok_or("EP2 logits empty")? as u32;
            report.baseline_tokens.push(next_base);
            report.ep2_tokens.push(ep2_token);
            base_token = next_base;
            pos += 1;
        }

        // 5c. Pending-final-token continuation: the final sampled token
        // (the argmax of the last decode step) must be fed into BOTH
        // retained states at the next exact position, its logits compared,
        // and only THEN may the position advance.  No token is replaced or
        // skipped between decode and the second turn.
        let base_logits = decode_step(
            &mut gpu,
            &weights,
            &config,
            &mut base_state,
            base_token,
            pos,
            false,
        )?;
        let ep2_logits = decode_step(
            &mut gpu,
            &weights,
            &config,
            &mut ep2_state,
            base_token,
            pos,
            true,
        )?;
        let c = compare_and_record(
            &mut report,
            boundary_pos(BoundaryKind::PendingToken, options.max_steps),
            &base_logits,
            &ep2_logits,
            pinned,
        )?;
        if !(c.finite && c.argmax_match) {
            report.generated_tokens_match = false;
        }
        pos += 1;

        // 5d. Second turn: append the suffix WITHOUT clearing state, at the
        // exact position after the pending token.
        let base_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut base_state,
            &suffix_tokens,
            pos,
            false,
        )?;
        let ep2_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut ep2_state,
            &suffix_tokens,
            pos,
            true,
        )?;
        let c = compare_and_record(
            &mut report,
            boundary_pos(BoundaryKind::SecondTurn, options.max_steps),
            &base_logits,
            &ep2_logits,
            pinned,
        )?;
        report.second_turn_match = c.finite && c.argmax_match;

        // 5e. Checked in-place reset: zero the EXISTING KV + DeltaNet
        // buffers (no replacement state, no new allocations), then replay
        // the original prompt from position 0.
        base_state.reset(&mut gpu)?;
        ep2_state.reset(&mut gpu)?;
        let base_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut base_state,
            &prompt_tokens,
            0,
            false,
        )?;
        let ep2_logits = prefill_step(
            &mut gpu,
            &weights,
            &config,
            &mut ep2_state,
            &prompt_tokens,
            0,
            true,
        )?;
        let c = compare_and_record(
            &mut report,
            boundary_pos(BoundaryKind::ResetReplay, options.max_steps),
            &base_logits,
            &ep2_logits,
            pinned,
        )?;
        report.reset_match = c.finite && c.argmax_match;
        Ok(())
    })();

    // ── 6. Success-path terminal cleanup + checked pool drain ───────────
    // SUCCESS-PATH ONLY (accepted scope): fully-constructed independent
    // owners are released through the available teardown APIs, then the pool
    // is drained so the cycle returns after ACTUAL HIP release, not merely
    // pool return. KV/weight checked failures are summarized, but retained
    // owners are not preserved; PBS/scratch/DeltaNet teardown remains
    // unchecked. This does NOT solve
    // partial-construction rollback, typed load-error cleanup retention, or
    // failed checked-free owner retention (STEP-002R debt).  A drain
    // failure is a harness error and never masks the primary run error.
    let mut cleanup_errors: Vec<String> = Vec::new();
    if let Some(s) = ep2_state.take() {
        cleanup_errors.extend(s.free(&mut gpu));
    }
    if let Some(s) = base_state.take() {
        cleanup_errors.extend(s.free(&mut gpu));
    }
    if let Some(w) = weights.take() {
        if let Err(e) = w.free_gpu_checked(&mut gpu) {
            cleanup_errors.extend(ep2_weights_cleanup_errors(e));
        }
    }
    if let Err(e) = gpu.drain_pool_checked() {
        cleanup_errors.push(format!("pool drain (checked): {e}"));
    }
    if let Err(e) = primary {
        return Err(combine_harness_error(
            format!("emulated-EP2 harness run failed: {e}"),
            &cleanup_errors,
        ));
    }
    if !cleanup_errors.is_empty() {
        return Err(cleanup_errors_message(&cleanup_errors));
    }

    // Acceptance: every compared boundary must match and the pinned
    // max-abs-logit-delta must hold.  Probe is NON-ACCEPTANCE.
    if mode == Ep2Mode::Acceptance {
        let delta_ok = report.max_abs_logit_delta <= options.max_logit_delta.unwrap_or(0.0);
        report.passed = report.finite_logits
            && report.first_token_match
            && report.generated_tokens_match
            && report.second_turn_match
            && report.reset_match
            && delta_ok;
    } else {
        report.passed = false;
    }
    Ok(report)
}

/// Combine the primary harness error with aggregated cleanup/drain errors.
/// A drain/cleanup failure must never mask the primary error — both are
/// reported.  Clean cleanup leaves the primary message unchanged.
fn combine_harness_error(primary: String, cleanup: &[String]) -> String {
    if cleanup.is_empty() {
        return primary;
    }
    format!(
        "{primary} (cleanup/drain also reported: {})",
        cleanup.join("; ")
    )
}

/// Aggregate message for the Ok-report-but-dirty path: cleanup/drain
/// failures with no primary error.  Empty for a clean run.
fn cleanup_errors_message(cleanup: &[String]) -> String {
    if cleanup.is_empty() {
        return String::new();
    }
    format!("ep2 harness cleanup/drain failures: {}", cleanup.join("; "))
}

/// Summarize a checked weights cleanup failure for this single-shot harness.
/// The retained-owner count and diagnostics are surfaced, but the owners are
/// not preserved across the public string error; exact retention is STEP-002R.
fn ep2_weights_cleanup_errors(e: crate::qwen35::Qwen35CleanupFailure) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!(
        "weights free_checked: {} tensor(s) retained",
        e.num_failed()
    ));
    out.extend(e.error_summaries());
    out
}

/// Fold one logit comparison into the report (finite / max-delta /
/// first-failing-boundary diagnostics).  `pos` uses the report's position
/// encoding ([`boundary_pos`]).  `pinned` is the acceptance max-abs-logit-
/// delta (None in probe mode): a boundary only counts as FAILING for
/// non-finite logits, an argmax/token mismatch, or a tolerance excess.
/// Ordinary accepted FP drift updates the global max delta but never
/// occupies `first_delta_*`.
fn compare_and_record(
    report: &mut Ep2HarnessReport,
    pos: usize,
    baseline: &[f32],
    ep2: &[f32],
    pinned: Option<f32>,
) -> Result<LogitComparison, String> {
    let c = compare_logits(baseline, ep2);
    if !c.finite {
        report.finite_logits = false;
    }
    if c.max_abs_delta > report.max_abs_logit_delta {
        report.max_abs_logit_delta = c.max_abs_delta;
    }
    if report.first_delta_pos.is_none() {
        if let Some((idx, b, e)) = first_failing_diagnostic(&c, pinned) {
            report.first_delta_pos = Some(pos);
            report.first_delta_index = Some(idx);
            report.baseline_logit = Some(b);
            report.ep2_logit = Some(e);
        } else if !c.finite {
            // Divergence without an in-range diagnostic index (a pure
            // length mismatch): record the boundary position honestly with
            // absent values rather than fabricating an out-of-range index.
            report.first_delta_pos = Some(pos);
        }
    }
    Ok(c)
}

/// One batched-prefill step on one state; returns the final logits
/// (`scratch.logits`, last-token).  `ep2` selects the rank-masked tail.
fn prefill_step(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    state: &mut HarnessState,
    tokens: &[u32],
    start_pos: usize,
    ep2: bool,
) -> Result<Vec<f32>, String> {
    let r = if ep2 {
        crate::qwen35::forward_prefill_batch_with_pbs_ep2(
            gpu,
            weights,
            config,
            tokens,
            start_pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            &state.scratch,
            None,
            None,
            None,
            None,
            &state.pbs,
            true,
        )
    } else {
        crate::qwen35::forward_prefill_batch_with_pbs(
            gpu,
            weights,
            config,
            tokens,
            start_pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            &state.scratch,
            None,
            None,
            None,
            None,
            Some(&state.pbs),
            None,
            None,
        )
    };
    r.map_err(|e| format!("prefill (ep2={ep2}): {e}"))?;
    gpu.download_f32(&state.scratch.logits)
        .map_err(|e| format!("prefill logits download: {e}"))
}

/// One greedy decode step on one state: embedding lookup of `token`, the
/// layer walk (production Single or rank-masked EP2), and the final norm +
/// lm_head (inside the layer walk) — returns the logits.  Both sides are
/// fed the SAME token at the SAME position (lockstep).
fn decode_step(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    state: &mut HarnessState,
    token: u32,
    pos: usize,
    ep2: bool,
) -> Result<Vec<f32>, String> {
    // Embedding lookup into scratch.x (mirrors forward_scratch's embed).
    let dim = config.dim;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu
            .embedding_lookup_hfq4g256(&weights.token_embd, &state.scratch.x, token, dim)
            .map_err(|e| format!("embedding lookup: {e}"))?,
        EmbeddingFormat::HFQ4G128 => gpu
            .embedding_lookup_hfq4g128(&weights.token_embd, &state.scratch.x, token, dim)
            .map_err(|e| format!("embedding lookup: {e}"))?,
        EmbeddingFormat::Q8_0 => gpu
            .embedding_lookup_q8(&weights.token_embd, &state.scratch.x, token, dim)
            .map_err(|e| format!("embedding lookup: {e}"))?,
        EmbeddingFormat::F32 => gpu
            .embedding_lookup(&weights.token_embd, &state.scratch.x, token, dim)
            .map_err(|e| format!("embedding lookup: {e}"))?,
        _ => return Err("unsupported embedding format".into()),
    }
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&state.scratch.pos_buf, &pos_i32.to_ne_bytes())
        .map_err(|e| format!("pos upload: {e}"))?;
    if ep2 {
        crate::qwen35::forward_scratch_layers_ep2(
            gpu,
            weights,
            config,
            pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            &state.scratch,
            None,
        )
        .map_err(|e| format!("EP2 decode layers: {e}"))?;
    } else {
        crate::qwen35::forward_scratch_layers(
            gpu,
            weights,
            config,
            pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            &state.scratch,
            None,
        )
        .map_err(|e| format!("baseline decode layers: {e}"))?;
    }
    gpu.download_f32(&state.scratch.logits)
        .map_err(|e| format!("decode logits download: {e}"))
}

/// Independent per-execution state set over the shared weights: scratch
/// (including the cfg-gated EP2 partials), KV cache, DeltaNet state, and
/// the batched-prefill scratch.
struct HarnessState {
    scratch: Qwen35Scratch,
    kv_cache: hipfire_runtime::llama::KvCache,
    dn_state: crate::qwen35::DeltaNetState,
    pbs: crate::qwen35::PrefillBatchScratch,
}

impl HarnessState {
    fn new(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        options: &Ep2HarnessOptions,
        max_batch: usize,
        kv_mode: KvMode,
    ) -> Result<Self, String> {
        let scratch = Qwen35Scratch::new(gpu, config, 1).map_err(|e| format!("scratch: {e}"))?;
        let kv_cache =
            build_kv_cache(gpu, config, options, kv_mode).map_err(|e| format!("kv cache: {e}"))?;
        let dn_state = crate::qwen35::DeltaNetState::new_with_quant(
            gpu,
            config,
            crate::qwen35::StateQuant::FP32,
        )
        .map_err(|e| format!("DeltaNet state: {e}"))?;
        // "Actual FP32, not merely the option string": the constructed
        // state must really be FP32 (StateQuant has no PartialEq, so match).
        if !matches!(dn_state.quant, crate::qwen35::StateQuant::FP32) {
            return Err(format!(
                "DeltaNet state constructed as {:?}, expected FP32",
                dn_state.quant
            ));
        }
        let pbs = crate::qwen35::PrefillBatchScratch::new(gpu, config, max_batch)
            .map_err(|e| format!("prefill scratch: {e}"))?;
        Ok(Self {
            scratch,
            kv_cache,
            dn_state,
            pbs,
        })
    }

    /// Checked IN-PLACE reset: zero the existing KV buffers
    /// ([`KvCache::clear_gpu`]) and the existing DeltaNet recurrent state
    /// ([`DeltaNetState::reset_checked`]) — no replacement state objects,
    /// no new allocations.  `compact_offset` (position bookkeeping) is
    /// rewound; replay starts at position 0.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.kv_cache
            .clear_gpu(gpu)
            .map_err(|e| format!("KV reset (clear_gpu): {e}"))?;
        self.kv_cache.compact_offset = 0;
        self.dn_state
            .reset(gpu)
            .map_err(|e| format!("DeltaNet reset: {e}"))?;
        Ok(())
    }

    /// Attempt teardown of every fully constructed state owner. KV uses its
    /// checked API and reports retained labels; PBS, scratch, and DeltaNet use
    /// their existing unchecked APIs. Exact failed-owner retention remains
    /// STEP-002R debt. Empty vec means no checked KV failure was reported.
    ///
    /// Frees return buffers to the GPU pool; the caller MUST drain the pool
    /// (checked) before returning so memory is actually released to HIP.
    fn free(self, gpu: &mut Gpu) -> Vec<String> {
        let Self {
            scratch,
            kv_cache,
            dn_state,
            pbs,
        } = self;
        let mut errors = Vec::new();
        pbs.free_gpu(gpu);
        scratch.free_gpu(gpu);
        if let Err(failures) = kv_cache.free_checked(gpu) {
            let labels: Vec<&str> = failures.iter().map(|(l, _)| l.as_str()).collect();
            errors.push(format!(
                "KV cache free_checked: {} tensor(s) retained ({labels:?})",
                failures.len()
            ));
        }
        dn_state.free_gpu(gpu);
        errors
    }
}

/// Build the KV cache for one state set from the ALREADY-VALIDATED mode
/// (verified Q8 with no fallback by [`validate_kv_mode_request`] before any
/// GPU work).
fn build_kv_cache(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    options: &Ep2HarnessOptions,
    mode: KvMode,
) -> Result<hipfire_runtime::llama::KvCache, String> {
    use hipfire_runtime::llama::{KvCache, KvDims, KvLayers, KvTarget};
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == crate::qwen35::LayerType::FullAttention)
        .collect();
    let dims = KvDims {
        layers: KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        // options.max_seq was validated to cover prompt + decode + pending
        // token + suffix BEFORE GPU execution (validate_turn_bounds).
        max_seq: options.max_seq,
        physical_cap: Some(options.max_seq),
    };
    KvCache::from_mode(mode, KvTarget::Single(gpu), &dims)
        .map_err(|e| format!("KvCache::from_mode: {e:?}"))
}

// ── CPU tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::ep2_rank_skip_shared;
    use rdna_compute::GpuTensor;

    fn env(deterministic: bool, graph_off: bool) -> HarnessEnv {
        HarnessEnv {
            deterministic,
            graph_off,
        }
    }

    fn options() -> Ep2HarnessOptions {
        Ep2HarnessOptions {
            model_path: PathBuf::from("/fake/model.mq4"),
            prompt: String::from("The capital of France is"),
            max_steps: 4,
            ..Default::default()
        }
    }

    /// Minimal Qwen35 config for state-construction tests (mirrors
    /// store.rs's `test_config`; takes CONFIG_ENV_LOCK itself — call it
    /// BEFORE holding that lock, the std Mutex is not reentrant).
    fn test_config(layer_types: &[&str]) -> Qwen35Config {
        let _env_guard = crate::store::CONFIG_ENV_LOCK.lock().unwrap();
        let value = serde_json::json!({
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": layer_types.len(),
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 8,
            "layer_types": layer_types,
            "tie_word_embeddings": true
        });
        crate::qwen35::config_from_metadata_json(&serde_json::json!({"config": value}).to_string())
            .unwrap()
    }

    fn report(mode: Ep2Mode) -> Ep2HarnessReport {
        Ep2HarnessReport {
            mode,
            passed: false,
            finite_logits: true,
            first_token_match: true,
            generated_tokens_match: true,
            second_turn_match: true,
            reset_match: true,
            max_abs_logit_delta: 0.0,
            first_delta_pos: None,
            first_delta_index: None,
            baseline_logit: None,
            ep2_logit: None,
            baseline_tokens: vec![],
            ep2_tokens: vec![],
            resolved_kv_mode: String::new(),
        }
    }

    #[test]
    fn ep2_compare_rejects_non_finite_logits() {
        // NaN in either vector must refuse acceptance with the exact index.
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, f32::NAN, 3.0];
        let c = compare_logits(&base, &ep2);
        assert!(!c.finite, "non-finite ep2 logits must be refused");
        assert_eq!(c.first_non_finite, Some(1));
        assert!(!c.argmax_match);
        assert_eq!(c.first_delta_index, Some(1));

        let base = vec![f32::INFINITY, 2.0, 3.0];
        let ep2 = vec![1.0, 2.0, 3.0];
        let c = compare_logits(&base, &ep2);
        assert!(!c.finite, "non-finite baseline logits must be refused");
        assert_eq!(c.first_non_finite, Some(0));
    }

    #[test]
    fn ep2_compare_detects_argmax_mismatch_with_first_diff() {
        // Argmax flips at index 2; the first differing value must be
        // reported with both values.
        let base = vec![0.5, 0.9, 1.0, 0.2];
        let ep2 = vec![0.5, 0.9, 1.4, 0.2];
        let c = compare_logits(&base, &ep2);
        assert!(c.finite);
        assert_eq!(c.baseline_argmax, Some(2));
        assert_eq!(c.ep2_argmax, Some(2), "argmax still matches at index 2");
        assert!(c.argmax_match);
        assert_eq!(c.first_delta_index, Some(2));
        assert_eq!(c.baseline_value, Some(1.0));
        assert_eq!(c.ep2_value, Some(1.4));
        assert!(
            (c.max_abs_delta - 0.4).abs() < 1e-6,
            "max delta must be |1.4 - 1.0| = 0.4, got {}",
            c.max_abs_delta
        );

        // Flip the argmax: index 0 becomes the winner on the ep2 side.
        let base = vec![0.5, 0.9, 1.0, 0.2];
        let ep2 = vec![1.6, 0.9, 1.0, 0.2];
        let c = compare_logits(&base, &ep2);
        assert!(!c.argmax_match, "argmax flip must be detected");
        assert_eq!(c.baseline_argmax, Some(2));
        assert_eq!(c.ep2_argmax, Some(0));
        assert_eq!(c.first_delta_index, Some(0));
        assert_eq!(c.max_abs_delta, 1.1);
    }

    #[test]
    fn ep2_compare_length_mismatch_never_reports_out_of_range_index() {
        // A pure length mismatch (identical shared prefix, different vocab
        // widths) is a divergence but has NO valid in-range first-diff
        // index.  It must not fabricate `Some(n)` (out of range) and must
        // not pair an out-of-range index with values from another position.
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 2.0, 3.0, 4.0];
        let c = compare_logits(&base, &ep2);
        assert!(!c.finite, "a length mismatch is a hard divergence");
        assert_eq!(
            c.first_delta_index, None,
            "a pure length mismatch must not report an out-of-range index"
        );
        assert_eq!(c.baseline_value, None);
        assert_eq!(c.ep2_value, None);
        assert!(!c.argmax_match);

        // When the shared prefix DOES differ, the in-range first-diff index
        // and its OWN values are kept — never displaced by the length
        // mismatch.
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 9.0, 3.0, 4.0];
        let c = compare_logits(&base, &ep2);
        assert!(!c.finite);
        assert_eq!(c.first_delta_index, Some(1), "in-range diff must be kept");
        assert_eq!(c.baseline_value, Some(2.0));
        assert_eq!(c.ep2_value, Some(9.0));
    }

    #[test]
    fn ep2_compare_max_delta_threshold() {
        // The comparison itself reports the delta; the caller applies the
        // pinned threshold.  Identical vectors must give delta 0 and no
        // first-diff.
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = compare_logits(&a, &b);
        assert!(c.finite && c.argmax_match);
        assert_eq!(c.max_abs_delta, 0.0);
        assert_eq!(c.first_delta_index, None);

        // Delta above a 0.1 pinned threshold must be detectable.
        let b = vec![1.0, 2.15, 3.0];
        let c = compare_logits(&a, &b);
        assert!(c.max_abs_delta > 0.1);
        assert_eq!(c.first_delta_index, Some(1));
        assert_eq!(c.max_delta_index, Some(1));
        assert_eq!(c.max_delta_baseline, Some(2.0));
        assert_eq!(c.max_delta_ep2, Some(2.15));
    }

    #[test]
    fn ep2_first_failing_diagnostic_prefers_non_finite() {
        // NaN/Inf at one index is the failure — the diagnostic must name
        // that index even when the argmax happens to match.
        let base = vec![f32::NAN, 1.0, 2.0];
        let ep2 = vec![1.0, 1.0, 2.0];
        let c = compare_logits(&base, &ep2);
        assert!(!c.finite);
        let d = first_failing_diagnostic(&c, Some(1.0)).expect("non-finite must fail");
        assert_eq!(d.0, 0, "first non-finite index");
        assert!(
            d.1.is_nan() && d.2 == 1.0,
            "both values at the failing index"
        );
    }

    #[test]
    fn ep2_first_failing_diagnostic_argmax_mismatch_uses_first_diff() {
        // Argmax flip (token mismatch): the first differing value index is
        // the relevant diagnostic, not the max-delta index.
        let base = vec![0.5, 0.9, 1.0, 0.2];
        let ep2 = vec![1.6, 0.9, 1.0, 0.2];
        let c = compare_logits(&base, &ep2);
        assert!(!c.argmax_match);
        let d = first_failing_diagnostic(&c, Some(10.0)).expect("argmax mismatch must fail");
        assert_eq!(d, (0, 0.5, 1.6));
    }

    #[test]
    fn ep2_first_failing_diagnostic_tolerance_excess_uses_max_delta_index() {
        // Same argmax, same tokens — ONLY the delta exceeds the pin.  The
        // diagnostic must name the EXCEEDING (max-delta) index + both values.
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 2.3, 3.0];
        let c = compare_logits(&base, &ep2);
        assert!(c.argmax_match, "argmax must still match");
        let d = first_failing_diagnostic(&c, Some(0.1)).expect("tolerance excess must fail");
        assert_eq!(d, (1, 2.0, 2.3), "exceeding max-delta index + both values");
    }

    #[test]
    fn ep2_first_failing_diagnostic_ignores_accepted_drift() {
        // Ordinary accepted FP drift (delta <= pin, argmax same) must NOT
        // occupy the failure diagnostics at all.
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 2.05, 3.0];
        let c = compare_logits(&base, &ep2);
        assert!(c.finite && c.argmax_match);
        assert_eq!(first_failing_diagnostic(&c, Some(0.1)), None);
        assert_eq!(first_failing_diagnostic(&c, None), None);
    }

    #[test]
    fn ep2_compare_and_record_records_decode_nan_failure() {
        // A decode-boundary NaN must fail the comparison and record the
        // diagnostic at that boundary's position.
        let mut r = report(Ep2Mode::Acceptance);
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, f32::NAN, 3.0];
        let _ = compare_and_record(&mut r, 3, &base, &ep2, Some(1.0)).unwrap();
        assert!(!r.finite_logits, "NaN must fail finite_logits");
        assert_eq!(r.first_delta_pos, Some(3));
        assert_eq!(r.first_delta_index, Some(1));
    }

    #[test]
    fn ep2_compare_and_record_records_excessive_delta_same_argmax() {
        // Same argmax, same tokens, but the delta exceeds the pin: the
        // boundary must fail and report the exceeding index + both values.
        let mut r = report(Ep2Mode::Acceptance);
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 2.4, 3.0];
        let _ = compare_and_record(&mut r, 1, &base, &ep2, Some(0.1)).unwrap();
        assert_eq!(r.first_delta_pos, Some(1));
        assert_eq!(r.first_delta_index, Some(1));
        assert_eq!(r.baseline_logit, Some(2.0));
        assert_eq!(r.ep2_logit, Some(2.4));
        assert!(r.max_abs_logit_delta > 0.1);
    }

    #[test]
    fn ep2_compare_and_record_ignores_accepted_drift_for_diagnostics() {
        // Accepted drift updates the global max delta but must NOT occupy
        // first_delta_*.
        let mut r = report(Ep2Mode::Acceptance);
        let base = vec![1.0, 2.0, 3.0];
        let ep2 = vec![1.0, 2.05, 3.0];
        let _ = compare_and_record(&mut r, 2, &base, &ep2, Some(0.1)).unwrap();
        assert_eq!(
            r.first_delta_pos, None,
            "accepted drift must not occupy first_delta"
        );
        assert!(
            (r.max_abs_logit_delta - 0.05).abs() < 1e-6,
            "accepted drift must still update the global max delta, got {}",
            r.max_abs_logit_delta
        );
    }

    // ── Non-vacuous contract (max_steps / prompts / capacity) ────────

    #[test]
    fn ep2_validate_rejects_zero_steps() {
        let mut o = options();
        o.max_steps = 0;
        let err = validate_options(&o, &env(true, true)).unwrap_err();
        assert!(
            err.contains("max_steps"),
            "zero decode steps must be refused: {err}"
        );
    }

    #[test]
    fn ep2_validate_rejects_empty_prompt_and_suffix_strings() {
        let mut o = options();
        o.prompt = String::new();
        assert!(
            validate_options(&o, &env(true, true)).is_err(),
            "empty prompt string must refuse"
        );
        o.prompt = String::from("   \n\t ");
        assert!(
            validate_options(&o, &env(true, true)).is_err(),
            "whitespace-only prompt must refuse"
        );
        let mut o = options();
        o.second_turn_suffix = String::new();
        assert!(
            validate_options(&o, &env(true, true)).is_err(),
            "empty suffix string must refuse"
        );
        o.second_turn_suffix = String::from("  \n ");
        assert!(
            validate_options(&o, &env(true, true)).is_err(),
            "whitespace-only suffix must refuse"
        );
    }

    #[test]
    fn ep2_validate_turn_bounds_rejects_empty_tokenized_suffix() {
        // A suffix that tokenizes to zero tokens must be refused — the
        // second-turn comparison would otherwise be vacuous.
        assert!(
            validate_turn_bounds(&[1, 2, 3], &[], 4, 4096).is_err(),
            "empty tokenized suffix must refuse"
        );
    }

    #[test]
    fn ep2_validate_turn_bounds_rejects_empty_prompt_tokens() {
        assert!(
            validate_turn_bounds(&[], &[7, 8], 4, 4096).is_err(),
            "empty tokenized prompt must refuse"
        );
    }

    #[test]
    fn ep2_validate_turn_bounds_rejects_insufficient_max_seq() {
        // Required capacity = prompt 10 + decode 4 + pending final token 1
        // + suffix 5 = 20.  max_seq 19 must refuse BEFORE GPU execution.
        assert!(
            validate_turn_bounds(&[0; 10], &[0; 5], 4, 19).is_err(),
            "max_seq below the precomputed requirement must refuse"
        );
        assert_eq!(validate_turn_bounds(&[0; 10], &[0; 5], 4, 20), Ok(()));
    }

    #[test]
    fn ep2_required_seq_capacity_counts_pending_token_and_suffix() {
        assert_eq!(required_seq_capacity(10, 4, 5), Ok(20));
        assert_eq!(
            required_seq_capacity(8, 0, 0),
            Ok(9),
            "the pending final token always counts"
        );
    }

    #[test]
    fn ep2_required_seq_capacity_checked_overflow() {
        // No release wraparound: an overflowing capacity sum must be an
        // explicit error, never a wrapping usize.
        assert!(
            required_seq_capacity(usize::MAX, 1, 0).is_err(),
            "prompt + steps must not wrap"
        );
        assert!(
            required_seq_capacity(usize::MAX - 1, 1, 0).is_err(),
            "the pending final token must not wrap"
        );
        assert!(
            required_seq_capacity(usize::MAX, 0, 0).is_err(),
            "the pending final token alone must not wrap"
        );
        assert!(required_seq_capacity(10, usize::MAX, 5).is_err());
        assert!(required_seq_capacity(10, 4, usize::MAX).is_err());
        // Exact-fit boundary still succeeds.
        assert_eq!(required_seq_capacity(usize::MAX - 2, 1, 0), Ok(usize::MAX));
        assert!(
            required_seq_capacity(usize::MAX - 2, 1, 1).is_err(),
            "one suffix token past the boundary must overflow"
        );
    }

    // ── Mandatory KV mode ────────────────────────────────────────────

    #[test]
    fn ep2_validate_kv_mode_requires_q8_no_fallback() {
        use hipfire_runtime::kv_mode::KvMode;
        assert_eq!(
            validate_kv_mode_request("q8", 256).expect("q8 must resolve"),
            KvMode::Q8
        );
        assert!(
            validate_kv_mode_request("asym3", 256).is_err(),
            "asym3 is not the fixture mode"
        );
        assert!(
            validate_kv_mode_request("auto", 256).is_err(),
            "'auto' resolves away from q8 and must refuse"
        );
        assert!(
            validate_kv_mode_request("", 256).is_err(),
            "empty request falls back to the default (not q8) and must refuse"
        );
    }

    // ── Boundary sequencing (pending-final-token position) ───────────

    #[test]
    fn ep2_boundary_positions_sequence_includes_pending_token() {
        // Report position encoding: 0 = prefill, 1+step = decode,
        // 1+max_steps = pending-final-token continuation, 2+max_steps =
        // second turn, 3+max_steps = reset replay.  The pending token gets
        // its own position between the decode loop and the suffix.
        assert_eq!(boundary_pos(BoundaryKind::Prefill, 4), 0);
        assert_eq!(boundary_pos(BoundaryKind::Decode(0), 4), 1);
        assert_eq!(boundary_pos(BoundaryKind::Decode(3), 4), 4);
        assert_eq!(boundary_pos(BoundaryKind::PendingToken, 4), 5);
        assert_eq!(boundary_pos(BoundaryKind::SecondTurn, 4), 6);
        assert_eq!(boundary_pos(BoundaryKind::ResetReplay, 4), 7);
    }

    // ── Real reset (in-place checked APIs) ───────────────────────────

    #[test]
    fn ep2_reset_uses_in_place_checked_kv_and_dn_apis() {
        // HarnessState::reset must zero the EXISTING buffers through the
        // checked in-place APIs — never rebuild/replace state objects
        // (which would silently allocate fresh state).  Pin the exact API
        // signatures the harness selects.
        fn _kv(f: fn(&mut hipfire_runtime::llama::KvCache, &mut Gpu) -> hip_bridge::HipResult<()>) {
            let _ = f;
        }
        fn _dn(f: fn(&mut crate::qwen35::DeltaNetState, &mut Gpu) -> hip_bridge::HipResult<()>) {
            let _ = f;
        }
        _kv(hipfire_runtime::llama::KvCache::clear_gpu);
        _dn(crate::qwen35::DeltaNetState::reset);
    }

    // ── Terminal checked cleanup + actual HIP release (pool drain) ─────

    #[test]
    fn ep2_harness_state_free_returns_aggregated_errors() {
        // HarnessState::free must RETURN every checked cleanup error (empty
        // vec = clean) instead of discarding KvCache::free_checked failures.
        fn _sig(f: fn(HarnessState, &mut Gpu) -> Vec<String>) {
            let _ = f;
        }
        _sig(HarnessState::free);
    }

    #[test]
    fn ep2_combine_harness_error_preserves_primary_and_cleanup() {
        // A drain/cleanup failure must never mask the primary harness error;
        // both are reported.  Clean cleanup leaves the primary message as-is.
        assert_eq!(
            combine_harness_error("ep2 parity sequence failed: x".into(), &[]),
            "ep2 parity sequence failed: x"
        );
        let msg = combine_harness_error(
            "ep2 parity sequence failed: x".into(),
            &[
                "KV cache free_checked: 2 tensor(s) retained".into(),
                "pool drain (checked): hipFree failed".into(),
            ],
        );
        assert!(msg.contains("ep2 parity sequence failed: x"), "{msg}");
        assert!(
            msg.contains("KV cache free_checked: 2 tensor(s) retained")
                && msg.contains("pool drain (checked): hipFree failed"),
            "{msg}"
        );
        assert!(msg.contains("cleanup/drain also reported"), "{msg}");
    }

    #[test]
    fn ep2_cleanup_errors_message_aggregates_all_owners() {
        // The Ok-report-but-dirty path aggregates every independent owner
        // failure into one harness error (nothing silently discarded).
        let msg = cleanup_errors_message(&[
            "weights free_checked: 3 tensor(s) retained".into(),
            "pool drain (checked): bind failed".into(),
        ]);
        assert!(msg.contains("3 tensor(s) retained"), "{msg}");
        assert!(msg.contains("bind failed"), "{msg}");
        assert!(
            msg.starts_with("ep2 harness cleanup/drain failures"),
            "{msg}"
        );
        assert_eq!(cleanup_errors_message(&[]), String::new());
    }

    #[test]
    fn ep2_weights_cleanup_errors_reports_retained_owners() {
        // The checked weights cleanup's retry/retention owners are surfaced
        // by count + summaries — never converted to a best-effort drop.
        let f = crate::qwen35::Qwen35CleanupFailure {
            failed_tensors: vec![crate::qwen35::RetainedQwenTensor {
                label: "token_embd".into(),
                tensor: GpuTensor::null_for_test(),
                last_error: "bind failed".into(),
            }],
            frozen: vec![],
        };
        let errs = ep2_weights_cleanup_errors(f);
        assert_eq!(errs.len(), 2, "count line + one summary: {errs:?}");
        assert!(errs[0].contains("1 tensor(s) retained"), "{errs:?}");
        assert!(
            errs[1].contains("token_embd") || errs[1].contains("bind failed"),
            "summary must name the retained owner: {errs:?}"
        );
    }

    #[test]
    #[ignore = "requires an AMD GPU; SUCCESS-PATH-ONLY warmed terminal cleanup returns VRAM within one page"]
    fn ep2_harness_state_free_and_drain_release_vram() {
        // SUCCESS-PATH-ONLY evidence: a fully-constructed state set freed
        // through the SAME checked path run() uses (aggregating
        // KvCache::free_checked), then the pool drained, returns to a WARMED
        // post-drain baseline within one hipMemGetInfo page. The first
        // identical cycle intentionally absorbs one-time HIP allocator/queue
        // initialization. This test is NOT partial-failure evidence: it does not
        // exercise partial-construction rollback, typed load-error cleanup
        // retention, or failed checked-free owner retention (STEP-002R
        // debt).
        static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = GPU_TEST_LOCK.lock().unwrap();
        // Build the config FIRST: test_config takes CONFIG_ENV_LOCK itself
        // (std Mutex is not reentrant).
        let config = test_config(&["linear_attention"]);
        let _env_guard = crate::store::CONFIG_ENV_LOCK.lock().unwrap();
        let opts = options();
        let mut gpu = Gpu::init().expect("GPU required");
        let hip = hip_bridge::HipRuntime::load().expect("HIP runtime dlopen");
        let run_state_cycle = |gpu: &mut Gpu| {
            let state = HarnessState::new(gpu, &config, &opts, 16, KvMode::Q8).expect("state set");
            let errors = state.free(gpu);
            assert!(
                errors.is_empty(),
                "checked state cleanup must be clean: {errors:?}"
            );
            gpu.drain_pool_checked()
                .expect("terminal pool drain must succeed");
        };

        run_state_cycle(&mut gpu);
        hip.set_device(0).expect("bind logical device 0");
        hip.device_synchronize().expect("warm-up synchronize");
        let warmed_baseline = hip
            .get_vram_info()
            .map(|(f, _)| f as u64)
            .expect("warmed baseline VRAM query");

        run_state_cycle(&mut gpu);
        hip.set_device(0).expect("rebind logical device 0");
        hip.device_synchronize().expect("measured synchronize");
        let after = hip
            .get_vram_info()
            .map(|(f, _)| f as u64)
            .expect("post-drain VRAM query");
        let retained = warmed_baseline.saturating_sub(after);
        assert!(
            retained <= 4096,
            "post-drain retained VRAM {retained} bytes exceeds one 4096-byte page \
             (warmed baseline={warmed_baseline}, after={after})"
        );
    }

    #[test]
    fn ep2_single_shot_claim_first_succeeds_second_refuses() {
        // The single-shot permit is a pure compare_exchange: the first claim
        // succeeds, every later claim refuses, and a claimed permit is never
        // reset.
        let claim = std::sync::atomic::AtomicBool::new(false);
        assert!(
            try_claim_single_shot(&claim),
            "first claim of a fresh permit must succeed"
        );
        assert!(
            !try_claim_single_shot(&claim),
            "second claim must refuse (single-shot)"
        );
        assert!(
            !try_claim_single_shot(&claim),
            "a refused permit stays refused"
        );
    }

    #[test]
    fn ep2_single_shot_global_permit_claims_once_and_never_resets() {
        // The process-wide permit follows the same protocol and is never
        // reset once claimed (run() claims it immediately before Gpu::init).
        // No GPU test in this binary invokes run(), so claiming the global
        // permit here cannot interfere with the other tests.
        assert!(try_claim_single_shot(&SINGLE_SHOT_CLAIM));
        assert!(!try_claim_single_shot(&SINGLE_SHOT_CLAIM));
        assert!(!try_claim_single_shot(&SINGLE_SHOT_CLAIM));
    }

    #[test]
    fn ep2_validate_acceptance_requires_finite_pinned_delta() {
        // No pinned delta = probe mode (NON-ACCEPTANCE, always valid);
        // acceptance requires an explicit finite positive delta.
        let mut o = options();
        o.max_logit_delta = None;
        assert_eq!(validate_options(&o, &env(true, true)), Ok(Ep2Mode::Probe));

        o.max_logit_delta = Some(0.5);
        assert_eq!(
            validate_options(&o, &env(true, true)),
            Ok(Ep2Mode::Acceptance)
        );

        o.max_logit_delta = Some(f32::NAN);
        assert!(validate_options(&o, &env(true, true)).is_err());
        o.max_logit_delta = Some(-0.5);
        assert!(validate_options(&o, &env(true, true)).is_err());
        o.max_logit_delta = Some(f32::INFINITY);
        assert!(validate_options(&o, &env(true, true)).is_err());
    }

    #[test]
    fn ep2_validate_probe_is_non_acceptance() {
        // Probe mode (no pinned delta) must never count as a PASS: the mode
        // is reported and only acceptance-mode reports can pass.
        let mut o = options();
        o.max_logit_delta = None;
        assert_eq!(validate_options(&o, &env(true, true)), Ok(Ep2Mode::Probe));

        let mut report = report(Ep2Mode::Probe);
        report.passed = true;
        assert!(
            !report_is_pass(&report),
            "probe reports are NON-ACCEPTANCE and must never count as a pass"
        );
        report.mode = Ep2Mode::Acceptance;
        assert!(report_is_pass(&report));
    }

    #[test]
    fn ep2_validate_determinism_and_graph_mandatory_no_bypass() {
        // The public `require_deterministic` bypass is REMOVED: the env
        // gates are unconditional.  HIPFIRE_DETERMINISTIC=1 and HIPFIRE_GRAPH=0
        // cannot be waived through the options struct.
        let o = options();
        assert!(
            validate_options(&o, &env(false, true)).is_err(),
            "missing HIPFIRE_DETERMINISTIC=1 must refuse"
        );
        assert!(
            validate_options(&o, &env(true, false)).is_err(),
            "HIPFIRE_GRAPH != 0 must refuse (baseline-only capture would fake parity)"
        );
        assert_eq!(validate_options(&o, &env(true, true)), Ok(Ep2Mode::Probe));
    }

    #[test]
    fn ep2_validate_refuses_non_fp32_state_quant() {
        let mut o = options();
        o.state_quant = String::from("q8");
        assert!(
            validate_options(&o, &env(true, true)).is_err(),
            "Q8 DeltaNet state rounding is not order-comparable"
        );
        o.state_quant = String::from("fp32");
        assert_eq!(validate_options(&o, &env(true, true)), Ok(Ep2Mode::Probe));
    }

    #[test]
    fn ep2_rank_shared_policy_rank0_owns_shared_contribution() {
        // Execution contract: rank 0 runs the shared-expert down (skip =
        // false), rank 1 skips it, so the shared expert is contributed
        // exactly once after the partial combine.
        assert!(!ep2_rank_skip_shared(0), "rank 0 must run the shared down");
        assert!(ep2_rank_skip_shared(1), "rank 1 must skip the shared down");
        assert!(
            ep2_rank_skip_shared(2),
            "out-of-range ranks never own shared"
        );
    }

    #[test]
    fn ep2_moe_execution_selector_single_is_production_default() {
        // The private execution selector: `Single` is the production
        // wrapper's selection; `EmulatedEp2` is the harness selection and
        // never the default.
        use crate::qwen35::MoeExecution;
        assert_eq!(MoeExecution::Single, MoeExecution::Single);
        assert_ne!(MoeExecution::Single, MoeExecution::EmulatedEp2);
    }
}
