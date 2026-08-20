// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Retained-PM4 route state for the fixed **B=16** DFlash2 target-verify
//! forward.
//!
//! This module owns the *pure* half of the route: phase, admission binding,
//! counters, and the state machine that decides which forward route a verify
//! window takes. It performs no GPU work of its own — the caller in
//! [`crate::speculative`] owns capture, preparation, replay, and every HIP
//! fallback, and reports each outcome back here.
//!
//! Scope is deliberately narrow (see the design record): the tape covers only
//! the fixed B=16 chain target forward. Token/position upload, the recurrent
//! snapshot, hidden staging commit, lm-head, argmax, acceptance, snapshot
//! restore, and committed-prefix GDN replay all stay outside it.
//!
//! ## Why the controller is optional
//!
//! [`DflashVerifyPm4Phase::Disabled`] must structurally guarantee that no
//! retained controller holds a model pointer. Storing the controller as an
//! `Option` makes that an ownership fact rather than a convention, and lets
//! [`DflashVerifyPm4::shutdown`] release it before any captured allocation is
//! freed.

use rdna_compute::replay::{
    PreparedReplayIdentity, RecordedKernargSnapshot, ReplayController, ReplayQuiescence,
};
use serde::Serialize;

/// The one admitted verify block size. Every other `b` stays on the existing
/// HIP route; final partial blocks are never captured, prepared, or replayed.
pub const DFLASH_VERIFY_PM4_BLOCK: usize = 16;

/// Externally visible route phase.
///
/// Capturing and preparing are stack-local transitions inside a single call,
/// so a half-completed capture can never be observed here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DflashVerifyPm4Phase {
    /// Never admitted for this model/config. Carries the admission reason.
    Disabled { reason: String },
    /// Admitted, but no window has materialized lazy code/scratch yet.
    Armed,
    /// One direct capture-safe HIP window has run; pointer identity is stable.
    Primed,
    /// One calibration capture is held; a second capture at a *different*
    /// position is required before a route may be prepared. Two recordings of
    /// the same tape are what let preparation tell a position-tracking scalar
    /// kernarg apart from one that must never be retained.
    Calibrating,
    /// A prepared single-IB PM4 route exists and matches the live binding.
    Ready,
    /// Sticky route failure. HIP remains correct and authoritative.
    Poisoned { reason: String },
    /// Replay failed without proven queue quiescence. The model must not be
    /// used or freed until an explicit quiesce succeeds.
    Quarantined { reason: String },
}

impl DflashVerifyPm4Phase {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Disabled { .. } => "disabled",
            Self::Armed => "armed",
            Self::Primed => "primed",
            Self::Calibrating => "calibrating",
            Self::Ready => "ready",
            Self::Poisoned { .. } => "poisoned",
            Self::Quarantined { .. } => "quarantined",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Disabled { reason } | Self::Poisoned { reason } | Self::Quarantined { reason } => {
                Some(reason.as_str())
            }
            _ => None,
        }
    }

    /// True while the route may still do retained-PM4 work.
    pub fn is_live(&self) -> bool {
        matches!(
            self,
            Self::Armed | Self::Primed | Self::Calibrating | Self::Ready
        )
    }
}

/// Monotonic route-proof counters for the loaded model.
///
/// No performance or correctness claim may rest on [`DflashVerifyPm4Phase::Ready`]
/// alone: `replays` must be non-zero and the replay positions must match the
/// windows under test.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DflashVerifyPm4Counters {
    /// Eligible-shape windows that ran the existing HIP route.
    pub full_hip: u64,
    /// Windows that ran HIP because `b != DFLASH_VERIFY_PM4_BLOCK`.
    pub partial_hip: u64,
    /// Direct capture-safe windows run to materialize lazy code/scratch.
    pub prime_windows: u64,
    /// Windows on which a recording capture was started.
    pub capture_attempts: u64,
    /// Captures that completed and produced launch blobs.
    pub captures: u64,
    /// Calibration captures taken (a prepared route needs two, at distinct
    /// positions).
    pub calibration_captures: u64,
    /// AQL contract probe rejections.
    pub contract_failures: u64,
    /// PM4 prefix preparation rejections.
    pub prepare_failures: u64,
    /// Successful retained-PM4 submissions.
    pub replays: u64,
    /// Retained-PM4 submissions that failed.
    pub replay_failures: u64,
    /// Direct-HIP reruns after a proven-quiescent replay failure.
    pub safe_hip_retries: u64,
    /// Sticky poison transitions.
    pub poison_count: u64,
    /// Binding/layout rearms (expected growth, not poison).
    pub rearms: u64,
    /// Position-tracking kernarg scalars the calibration differencing found and
    /// bound. Zero on a prepared route means the tape declared no scalar that
    /// varies with position — treat that as a claim to verify, not a default.
    pub position_bindings: u64,
    pub first_replay_position: Option<usize>,
    pub last_replay_position: Option<usize>,
}

/// Identity a prepared route is valid for.
///
/// Fingerprints cover shapes, dtypes, extraction layers, and every captured
/// allocation base — never buffer contents.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DflashVerifyBinding {
    pub batch: usize,
    pub arch: String,
    pub model_fingerprint: u64,
    pub layout_generation: u64,
    /// Largest `start_pos + batch` the prepared geometry is sized for.
    pub max_position: usize,
    pub kv_mode: String,
    pub dn_state_quant: String,
}
impl DflashVerifyBinding {
    pub fn new(
        batch: usize,
        arch: impl Into<String>,
        model_fingerprint: u64,
        layout_generation: u64,
        max_position: usize,
    ) -> Self {
        Self {
            batch,
            arch: arch.into(),
            model_fingerprint,
            layout_generation,
            max_position,
            kv_mode: "q8".to_string(),
            dn_state_quant: "q8".to_string(),
        }
    }

    /// Identity match ignoring `max_position`, which only bounds admission.
    pub fn same_route(&self, other: &Self) -> bool {
        self.batch == other.batch
            && self.arch == other.arch
            && self.model_fingerprint == other.model_fingerprint
            && self.layout_generation == other.layout_generation
    }
}

/// Stable 64-bit FNV-1a over captured allocation bases and shape scalars.
///
/// Deliberately not `DefaultHasher`: the fingerprint is logged as route-proof
/// evidence and must be reproducible across builds.
pub fn fingerprint_u64(values: &[u64]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The shape of one verify window, as seen before any GPU work.
#[derive(Clone, Debug)]
pub struct DflashVerifyWindow<'a> {
    pub batch: usize,
    pub tree: bool,
    pub want_full_logits: bool,
    /// `start_pos` of this window.
    pub position: usize,
    pub binding: &'a DflashVerifyBinding,
}

impl DflashVerifyWindow<'_> {
    fn eligible_shape(&self) -> bool {
        self.batch == DFLASH_VERIFY_PM4_BLOCK && !self.tree && !self.want_full_logits
    }
}

/// Route selected for one verify window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DflashVerifyRoute {
    /// Existing shipping behavior, including HipGraph auto-selection.
    HipAuto,
    /// Forced direct capture-safe HIP, no graph, no recording.
    PrimeDirect,
    /// Forced direct capture-safe HIP under an active recording capture.
    CaptureRecord,
    /// Prepared single-IB retained PM4.
    Pm4,
}

/// Per-speculator retained-PM4 verify route.
pub struct DflashVerifyPm4 {
    phase: DflashVerifyPm4Phase,
    controller: Option<ReplayController>,
    binding: Option<DflashVerifyBinding>,
    identity: Option<PreparedReplayIdentity>,
    /// First calibration recording and the position it was taken at.
    calibration: Option<(RecordedKernargSnapshot, usize)>,
    counters: DflashVerifyPm4Counters,
}

impl DflashVerifyPm4 {
    /// Never-admitted route. Holds no controller and no model pointer.
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            phase: DflashVerifyPm4Phase::Disabled {
                reason: reason.into(),
            },
            controller: None,
            binding: None,
            identity: None,
            calibration: None,
            counters: DflashVerifyPm4Counters::default(),
        }
    }

    /// Admitted route with a dedicated manual-PM4 controller. Allocates no GPU
    /// resource until the first capture.
    pub fn armed() -> Self {
        Self {
            phase: DflashVerifyPm4Phase::Armed,
            controller: Some(ReplayController::new_manual_pm4()),
            binding: None,
            identity: None,
            calibration: None,
            counters: DflashVerifyPm4Counters::default(),
        }
    }

    pub fn phase(&self) -> &DflashVerifyPm4Phase {
        &self.phase
    }

    pub fn counters(&self) -> &DflashVerifyPm4Counters {
        &self.counters
    }

    pub fn binding(&self) -> Option<&DflashVerifyBinding> {
        self.binding.as_ref()
    }

    pub fn prepared_identity(&self) -> Option<PreparedReplayIdentity> {
        self.identity
    }

    /// Borrow the dedicated controller for the `gpu.replay` swap. `None`
    /// whenever the route holds no captured state.
    pub fn controller_mut(&mut self) -> Option<&mut ReplayController> {
        self.controller.as_mut()
    }

    /// Decide this window's route and record the decision.
    ///
    /// Pure with respect to the GPU: a `Rearm` is only a host-side reset of the
    /// prepared route, which is safe because the previous replay waited to
    /// completion (a failed replay poisons or quarantines instead).
    pub fn plan_route(&mut self, window: &DflashVerifyWindow<'_>) -> DflashVerifyRoute {
        if !self.phase.is_live() {
            self.note_hip_window(window);
            return DflashVerifyRoute::HipAuto;
        }
        if !window.eligible_shape() {
            self.note_hip_window(window);
            return DflashVerifyRoute::HipAuto;
        }
        match self.phase {
            DflashVerifyPm4Phase::Armed => {
                self.counters.prime_windows += 1;
                DflashVerifyRoute::PrimeDirect
            }
            DflashVerifyPm4Phase::Primed => {
                if self.route_matches(window) {
                    self.counters.capture_attempts += 1;
                    DflashVerifyRoute::CaptureRecord
                } else {
                    self.rearm("binding changed before capture");
                    self.counters.prime_windows += 1;
                    DflashVerifyRoute::PrimeDirect
                }
            }
            DflashVerifyPm4Phase::Calibrating => {
                let calibrated_at = self.calibration.as_ref().map(|(_, at)| *at);
                if !self.route_matches(window) {
                    self.rearm("binding changed between calibration captures");
                    self.counters.prime_windows += 1;
                    DflashVerifyRoute::PrimeDirect
                } else if calibrated_at == Some(window.position) {
                    // Differencing two recordings taken at the SAME position
                    // proves nothing. Run direct and wait for the position to
                    // move; generation advances it every window.
                    self.counters.prime_windows += 1;
                    DflashVerifyRoute::PrimeDirect
                } else {
                    self.counters.capture_attempts += 1;
                    DflashVerifyRoute::CaptureRecord
                }
            }
            DflashVerifyPm4Phase::Ready => {
                if self.route_matches(window) && self.position_admitted(window) {
                    DflashVerifyRoute::Pm4
                } else {
                    self.rearm("binding or position range changed before submission");
                    self.counters.prime_windows += 1;
                    DflashVerifyRoute::PrimeDirect
                }
            }
            _ => {
                self.note_hip_window(window);
                DflashVerifyRoute::HipAuto
            }
        }
    }

    fn route_matches(&self, window: &DflashVerifyWindow<'_>) -> bool {
        self.binding
            .as_ref()
            .is_some_and(|bound| bound.same_route(window.binding))
    }

    fn position_admitted(&self, window: &DflashVerifyWindow<'_>) -> bool {
        let Some(bound) = self.binding.as_ref() else {
            return false;
        };
        window
            .position
            .checked_add(window.batch)
            .is_some_and(|end| end <= bound.max_position)
    }

    fn note_hip_window(&mut self, window: &DflashVerifyWindow<'_>) {
        if window.batch != DFLASH_VERIFY_PM4_BLOCK {
            self.counters.partial_hip += 1;
        } else {
            self.counters.full_hip += 1;
        }
    }

    /// A direct capture-safe window completed; pointer identity is now stable.
    pub fn note_prime_success(&mut self, binding: DflashVerifyBinding) {
        if matches!(self.phase, DflashVerifyPm4Phase::Armed) {
            self.phase = DflashVerifyPm4Phase::Primed;
        }
        self.binding = Some(binding);
    }

    /// Capture completed and produced launch blobs.
    pub fn note_capture(&mut self) {
        self.counters.captures += 1;
    }

    /// Hold the first calibration recording. A second capture at a different
    /// position must difference against it before a route may be prepared.
    pub fn note_calibration_capture(&mut self, snapshot: RecordedKernargSnapshot, position: usize) {
        self.counters.captures += 1;
        self.counters.calibration_captures += 1;
        self.calibration = Some((snapshot, position));
        self.phase = DflashVerifyPm4Phase::Calibrating;
    }

    /// The held calibration recording and the position it was taken at.
    pub fn calibration(&self) -> Option<(&RecordedKernargSnapshot, usize)> {
        self.calibration
            .as_ref()
            .map(|(snapshot, at)| (snapshot, *at))
    }

    /// Record how many position-tracking kernarg scalars the calibration
    /// differencing bound for this tape.
    pub fn note_position_bindings(&mut self, count: usize) {
        self.counters.position_bindings = count as u64;
    }

    /// Capture + contract probe + PM4 preparation all succeeded.
    pub fn note_ready(&mut self, binding: DflashVerifyBinding, identity: PreparedReplayIdentity) {
        self.binding = Some(binding);
        self.identity = Some(identity);
        // The recording has served its purpose; holding it would pin bytes for
        // the model's whole lifetime.
        self.calibration = None;
        self.phase = DflashVerifyPm4Phase::Ready;
    }

    /// AQL contract probe rejected the capture. The current HIP result stands.
    pub fn note_contract_failure(&mut self, reason: impl Into<String>) {
        self.counters.contract_failures += 1;
        self.poison(reason);
    }

    /// PM4 preparation failed. No model kernel ran; the HIP result stands.
    pub fn note_prepare_failure(&mut self, reason: impl Into<String>) {
        self.counters.prepare_failures += 1;
        self.poison(reason);
    }

    pub fn note_replay_success(&mut self, position: usize) {
        self.counters.replays += 1;
        if self.counters.first_replay_position.is_none() {
            self.counters.first_replay_position = Some(position);
        }
        self.counters.last_replay_position = Some(position);
    }

    /// A retained submission failed.
    ///
    /// `Proven` quiescence permits the caller to restore the pre-window
    /// recurrent snapshot and rerun the forward on direct HIP. `Unknown`
    /// quarantines the model: nothing downstream may run and no captured
    /// allocation may be freed.
    pub fn note_replay_failure(
        &mut self,
        position: usize,
        quiescence: ReplayQuiescence,
        reason: impl Into<String>,
    ) {
        self.counters.replay_failures += 1;
        self.counters.last_replay_position = Some(position);
        let reason = reason.into();
        match quiescence {
            ReplayQuiescence::Proven => self.poison(reason),
            ReplayQuiescence::Unknown => {
                self.counters.poison_count += 1;
                self.identity = None;
                self.phase = DflashVerifyPm4Phase::Quarantined { reason };
            }
        }
    }

    pub fn note_safe_hip_retry(&mut self) {
        self.counters.safe_hip_retries += 1;
    }

    /// Sticky poison. The route stops; HIP behavior is unchanged and correct.
    pub fn poison(&mut self, reason: impl Into<String>) {
        if matches!(self.phase, DflashVerifyPm4Phase::Quarantined { .. }) {
            return;
        }
        let reason = reason.into();
        if let Some(controller) = self.controller.as_mut() {
            controller.poison(reason.clone());
        }
        self.counters.poison_count += 1;
        self.identity = None;
        self.calibration = None;
        self.phase = DflashVerifyPm4Phase::Poisoned { reason };
    }

    /// Drop a prepared route because the admitted layout changed. Not poison.
    fn rearm(&mut self, reason: &str) {
        let _ = reason;
        if let Some(controller) = self.controller.as_mut() {
            controller.rearm_after_layout_growth();
        }
        self.counters.rearms += 1;
        self.identity = None;
        self.binding = None;
        // A recording taken against the old layout can no longer be differenced
        // against a new one.
        self.calibration = None;
        self.phase = DflashVerifyPm4Phase::Armed;
    }

    /// Prove no retained IB is in flight and release the controller.
    ///
    /// MUST complete before any captured allocation — verify scratch, PBS,
    /// hidden ring, GDN tape, snapshot, KV, DeltaNet state, or target weights —
    /// is freed. An `Err` means quiescence is unknown: the model stays
    /// quarantined and nothing it names may be deallocated.
    pub fn shutdown(&mut self) -> Result<(), String> {
        let Some(mut controller) = self.controller.take() else {
            self.identity = None;
            self.binding = None;
            return Ok(());
        };
        match controller.shutdown() {
            Ok(()) => {
                drop(controller);
                self.identity = None;
                self.binding = None;
                if self.phase.is_live() {
                    self.phase = DflashVerifyPm4Phase::Disabled {
                        reason: "shut down".to_string(),
                    };
                }
                Ok(())
            }
            Err(failure) => {
                // Quiescence is unknown. Retain the controller so its Drop can
                // keep retrying inactivation, and refuse to release pointers.
                let reason = format!("dflash verify PM4 shutdown failed: {}", failure.error);
                self.controller = Some(controller);
                self.phase = DflashVerifyPm4Phase::Quarantined {
                    reason: reason.clone(),
                };
                Err(reason)
            }
        }
    }

    /// Route-proof evidence for the daemon harness and request markers.
    pub fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": self.phase.label(),
            "reason": self.phase.reason(),
            "binding": self.binding,
            "kv_mode": "q8",
            "dn_state_quant": "q8",
            "prepared_dispatch_count": self.identity.map(|id| id.dispatch_count),
            "prepared_packet_count": self.identity.and_then(|id| id.packet_count),
            "prepared_queue_id": self.identity.map(|id| id.queue_id),
            "prepared_queue_count": self.identity.map(|id| id.queue_count),
            "prepared_phase_count": self.identity.map(|id| id.phase_count),
            "counters": self.counters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(generation: u64, max_position: usize) -> DflashVerifyBinding {
        DflashVerifyBinding::new(DFLASH_VERIFY_PM4_BLOCK, "gfx1201", 0xabc, generation, max_position)
    }

    fn identity(dispatch_count: usize) -> PreparedReplayIdentity {
        PreparedReplayIdentity {
            dispatch_count,
            packet_count: Some(dispatch_count),
            queue_id: 1,
            command_dwords: None,
            queue_count: 1,
            phase_count: 1,
        }
    }

    fn window<'a>(bound: &'a DflashVerifyBinding, batch: usize, position: usize) -> DflashVerifyWindow<'a> {
        DflashVerifyWindow {
            batch,
            tree: false,
            want_full_logits: false,
            position,
            binding: bound,
        }
    }

    #[test]
    fn disabled_route_never_leaves_hip() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::disabled("unsupported arch");
        assert_eq!(
            route.plan_route(&window(&bound, 16, 0)),
            DflashVerifyRoute::HipAuto
        );
        assert!(route.controller_mut().is_none());
        assert_eq!(route.counters().full_hip, 1);
    }

    #[test]
    fn partial_blocks_are_counted_separately_and_stay_hip() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::armed();
        assert_eq!(
            route.plan_route(&window(&bound, 4, 0)),
            DflashVerifyRoute::HipAuto
        );
        assert_eq!(route.counters().partial_hip, 1);
        assert_eq!(route.counters().full_hip, 0);
        assert_eq!(*route.phase(), DflashVerifyPm4Phase::Armed);
    }

    #[test]
    fn tree_and_full_logits_windows_stay_hip() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::armed();
        let mut tree = window(&bound, 16, 0);
        tree.tree = true;
        assert_eq!(route.plan_route(&tree), DflashVerifyRoute::HipAuto);
        let mut logits = window(&bound, 16, 0);
        logits.want_full_logits = true;
        assert_eq!(route.plan_route(&logits), DflashVerifyRoute::HipAuto);
        assert_eq!(route.counters().full_hip, 2);
    }

    #[test]
    fn arm_prime_capture_ready_replay_progression() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::armed();
        assert_eq!(
            route.plan_route(&window(&bound, 16, 0)),
            DflashVerifyRoute::PrimeDirect
        );
        route.note_prime_success(bound.clone());
        assert_eq!(*route.phase(), DflashVerifyPm4Phase::Primed);
        assert_eq!(
            route.plan_route(&window(&bound, 16, 16)),
            DflashVerifyRoute::CaptureRecord
        );
        route.note_capture();
        route.note_ready(
            bound.clone(),
            identity(1154),
        );
        assert_eq!(*route.phase(), DflashVerifyPm4Phase::Ready);
        assert_eq!(
            route.plan_route(&window(&bound, 16, 32)),
            DflashVerifyRoute::Pm4
        );
        route.note_replay_success(32);
        assert_eq!(route.counters().replays, 1);
        assert_eq!(route.counters().first_replay_position, Some(32));
        assert_eq!(route.counters().last_replay_position, Some(32));
    }

    #[test]
    fn position_beyond_prepared_max_rearms_instead_of_replaying() {
        let bound = binding(1, 64);
        let mut route = DflashVerifyPm4::armed();
        route.note_prime_success(bound.clone());
        route.note_ready(
            bound.clone(),
            identity(8),
        );
        // Exactly at the prepared bound (48 + 16 == 64) is still admitted.
        assert_eq!(
            route.plan_route(&window(&bound, 16, 48)),
            DflashVerifyRoute::Pm4
        );
        // One window past it must rearm rather than replay geometry that was
        // prepared too small.
        assert_eq!(
            route.plan_route(&window(&bound, 16, 56)),
            DflashVerifyRoute::PrimeDirect
        );
        assert_eq!(*route.phase(), DflashVerifyPm4Phase::Armed);
        assert_eq!(route.counters().rearms, 1);
        assert_eq!(route.counters().poison_count, 0);
    }

    #[test]
    fn layout_generation_change_rearms_without_poison() {
        let bound = binding(1, 4096);
        let grown = binding(2, 8192);
        let mut route = DflashVerifyPm4::armed();
        route.note_prime_success(bound.clone());
        route.note_ready(
            bound,
            identity(8),
        );
        assert_eq!(
            route.plan_route(&window(&grown, 16, 0)),
            DflashVerifyRoute::PrimeDirect
        );
        assert_eq!(route.counters().rearms, 1);
        assert!(route.prepared_identity().is_none());
    }

    #[test]
    fn proven_replay_failure_poisons_and_allows_hip_retry() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::armed();
        route.note_prime_success(bound.clone());
        route.note_ready(
            bound.clone(),
            identity(8),
        );
        route.note_replay_failure(64, ReplayQuiescence::Proven, "signal timeout");
        assert!(matches!(
            route.phase(),
            DflashVerifyPm4Phase::Poisoned { .. }
        ));
        route.note_safe_hip_retry();
        assert_eq!(
            route.plan_route(&window(&bound, 16, 64)),
            DflashVerifyRoute::HipAuto
        );
        assert_eq!(route.counters().safe_hip_retries, 1);
        assert_eq!(route.counters().replay_failures, 1);
    }

    #[test]
    fn unknown_quiescence_quarantines_and_poison_cannot_downgrade_it() {
        let bound = binding(1, 4096);
        let mut route = DflashVerifyPm4::armed();
        route.note_prime_success(bound.clone());
        route.note_ready(
            bound,
            identity(8),
        );
        route.note_replay_failure(80, ReplayQuiescence::Unknown, "teardown failed");
        assert!(matches!(
            route.phase(),
            DflashVerifyPm4Phase::Quarantined { .. }
        ));
        route.poison("later failure");
        assert!(matches!(
            route.phase(),
            DflashVerifyPm4Phase::Quarantined { .. }
        ));
    }

    #[test]
    fn fingerprint_is_order_sensitive_and_stable() {
        assert_eq!(fingerprint_u64(&[1, 2, 3]), fingerprint_u64(&[1, 2, 3]));
        assert_ne!(fingerprint_u64(&[1, 2, 3]), fingerprint_u64(&[3, 2, 1]));
        assert_eq!(fingerprint_u64(&[]), 0xcbf2_9ce4_8422_2325);
    }
}
