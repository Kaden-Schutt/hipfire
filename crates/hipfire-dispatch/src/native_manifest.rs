// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Native-required manifest: enforcement + telemetry for the wave32-native
//! dp4a decode-GEMV family (`gate_up` / `down` / `qkvza`,
//! `ArchCaps::dp4a_decode_gemv_coverage`, Task 7). Declare-only in Phase A —
//! every arch reports all-false coverage, so every call site that dispatches
//! the scalar decode-GEMV today is an *expected* fallback, not a regression.
//!
//! Two entry points:
//! - [`NativeManifest::snapshot`] — pure data: per-family native/scalar
//!   coverage for one arch, computed from `ArchCaps`.
//! - [`report_fallback`] — a call-site hook: logs (`eprintln!`) once per
//!   distinct `(family, reason)` key (no per-layer spam across a token loop
//!   calling the same scalar kernel thousands of times), and enforces
//!   severity: `NativeLost` is always a hard block (and, under the
//!   `native_manifest_strict` feature — dev/CI only — panics at report
//!   time); everything else is warn+degrade.
//!
//! See `docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md`
//! Task 8.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use rdna_compute::arch_caps::ArchCaps;

/// The three dp4a-decode-GEMV families tracked by the manifest (Task 7:
/// `Dp4aDecodeGemvCoverage { gate_up, down, qkvza }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    GateUp,
    Down,
    Qkvza,
}

impl Family {
    pub fn as_str(&self) -> &'static str {
        match self {
            Family::GateUp => "gate_up",
            Family::Down => "down",
            Family::Qkvza => "qkvza",
        }
    }
}

/// Why a family took the scalar (non-native) decode-GEMV path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackReason {
    /// The manifest never claimed native coverage for this (arch, family) —
    /// expected, not a regression. Phase A: every arch, every family.
    ExpectedInManifest,
    /// The manifest claims native coverage for this (arch, family), but a
    /// runtime precondition (shape/dtype/batch) forced the scalar path this
    /// call.
    PreconditionFailed,
    /// The manifest claims native coverage, but the dispatched kernel was
    /// scalar anyway (e.g. an intrinsic-scan regression) — a real bug, not
    /// an expected fallback. Hard-blocks regardless of build profile.
    NativeLost,
}

impl FallbackReason {
    /// `NativeLost` is always a hard block; every other reason is a
    /// warn+degrade (or a dev/CI panic under `native_manifest_strict`).
    pub fn is_hard_block(&self) -> bool {
        matches!(self, FallbackReason::NativeLost)
    }
}

/// Severity returned by [`report_fallback`] — the enforcement action taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Warn,
    HardBlock,
}

/// Per-family coverage entry produced by [`NativeManifest::snapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyCoverage {
    pub family: Family,
    /// Does a native (wave32-native dp4a) decode-GEMV kernel exist for this
    /// (arch, family) today, per the Task 7 manifest?
    pub native: bool,
    /// The reason a call to this family currently takes the scalar path.
    /// `None` when `native` is true (no fallback expected).
    pub reason: Option<FallbackReason>,
}

/// A point-in-time snapshot of dp4a-decode-GEMV coverage for one arch.
#[derive(Debug, Clone)]
pub struct CoverageSnapshot {
    pub arch: String,
    pub entries: Vec<FamilyCoverage>,
}

impl CoverageSnapshot {
    pub fn entry(&self, family: Family) -> Option<&FamilyCoverage> {
        self.entries.iter().find(|e| e.family == family)
    }

    /// One-line human summary, e.g.
    /// `"native-manifest gfx1201: expected scalar fallback: gate_up, down, qkvza"`.
    pub fn summary(&self) -> String {
        let scalar: Vec<&str> = self
            .entries
            .iter()
            .filter(|e| !e.native)
            .map(|e| e.family.as_str())
            .collect();
        if scalar.is_empty() {
            format!("native-manifest {}: full native coverage", self.arch)
        } else {
            format!(
                "native-manifest {}: expected scalar fallback: {}",
                self.arch,
                scalar.join(", ")
            )
        }
    }
}

pub struct NativeManifest;

impl NativeManifest {
    /// Build the coverage snapshot for `caps`'s arch from
    /// `ArchCaps::dp4a_decode_gemv_coverage` (Task 7 — all-false today).
    /// Pure data computation; never panics.
    pub fn snapshot(caps: &ArchCaps) -> CoverageSnapshot {
        let coverage = caps.dp4a_decode_gemv_coverage();
        let mk = |family: Family, native: bool| FamilyCoverage {
            family,
            native,
            reason: if native {
                None
            } else {
                Some(FallbackReason::ExpectedInManifest)
            },
        };
        CoverageSnapshot {
            arch: caps.arch().to_string(),
            entries: vec![
                mk(Family::GateUp, coverage.gate_up),
                mk(Family::Down, coverage.down),
                mk(Family::Qkvza, coverage.qkvza),
            ],
        }
    }
}

fn reported_set() -> &'static Mutex<HashSet<(Family, FallbackReason)>> {
    static REPORTED: OnceLock<Mutex<HashSet<(Family, FallbackReason)>>> = OnceLock::new();
    REPORTED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Returns `true` the FIRST time this `(family, reason)` key is seen
/// (process-global), `false` on every repeat. Split out from
/// [`report_fallback`] so the dedup logic itself — not just the
/// `Severity` it happens to return — is directly unit-testable.
fn should_log(family: Family, reason: FallbackReason) -> bool {
    let mut set = reported_set().lock().unwrap_or_else(|e| e.into_inner());
    set.insert((family, reason))
}

/// Report a scalar-kernel fallback for `family` on `arch` with `reason`.
///
/// Logs once per distinct `(family, reason)` key — repeat calls with the
/// SAME family+reason are silent (a decode loop calling this every token
/// does not spam stderr); a *different* reason for the same family logs
/// again (the split the manifest exists to preserve — `ExpectedInManifest`
/// and `PreconditionFailed` are not interchangeable).
///
/// `NativeLost` is always [`Severity::HardBlock`]; under the
/// `native_manifest_strict` feature (dev/CI only) it additionally panics.
pub fn report_fallback(family: Family, arch: &str, reason: FallbackReason) -> Severity {
    let severity = if reason.is_hard_block() {
        Severity::HardBlock
    } else {
        Severity::Warn
    };
    if should_log(family, reason) {
        eprintln!(
            "[native-manifest] {arch}/{}: {reason:?} ({})",
            family.as_str(),
            match severity {
                Severity::Warn => "warn+degrade",
                Severity::HardBlock => "HARD BLOCK",
            },
        );
    }
    #[cfg(feature = "native_manifest_strict")]
    if matches!(reason, FallbackReason::NativeLost) {
        panic!(
            "native-manifest NATIVE-LOST: {family:?} on {arch} — a previously-native \
             decode-GEMV kernel dispatched scalar; see \
             docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md Task 8"
        );
    }
    severity
}

/// Returns `true` the FIRST time `arch` is seen by [`emit_summary_once`]
/// (process-global, but keyed per-arch — NOT a single process-wide flag), and
/// `false` on every repeat for that same arch. Split out for direct unit
/// testing (see `emit_summary_once_per_arch_not_process`).
fn should_emit_summary(arch: &str) -> bool {
    static EMITTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let set = EMITTED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut set = set.lock().unwrap_or_else(|e| e.into_inner());
    set.insert(arch.to_string())
}

/// Emit the coverage summary exactly once per (process, arch) — idempotent
/// across repeated `DispatchCtx::new` calls in a per-layer hot loop, but a
/// SECOND distinct arch seen by the same process (e.g. a heterogeneous
/// multi-GPU session) still gets its own summary line rather than being
/// silently swallowed by a single process-wide flag.
pub fn emit_summary_once(caps: &ArchCaps) {
    if should_emit_summary(caps.arch()) {
        eprintln!("{}", NativeManifest::snapshot(caps).summary());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DispatchCtx;

    fn caps_for(arch: &str) -> ArchCaps {
        DispatchCtx::for_test(arch).arch
    }

    #[test]
    fn snapshot_gfx1201_all_expected_scalar_fallback() {
        let caps = caps_for("gfx1201");
        let snap = NativeManifest::snapshot(&caps);
        assert_eq!(snap.arch, "gfx1201");
        for family in [Family::GateUp, Family::Down, Family::Qkvza] {
            let entry = snap.entry(family).unwrap_or_else(|| {
                panic!("snapshot missing entry for {family:?}");
            });
            assert!(!entry.native, "{family:?} unexpectedly native on gfx1201");
            assert_eq!(
                entry.reason,
                Some(FallbackReason::ExpectedInManifest),
                "{family:?} reason mismatch"
            );
        }
        assert!(snap.summary().contains("expected scalar fallback"));
        assert!(snap.summary().contains("gate_up"));
        assert!(snap.summary().contains("down"));
        assert!(snap.summary().contains("qkvza"));
    }

    // Under `native_manifest_strict`, NativeLost panics at report time
    // (see below) instead of returning — so this variant of the assertion
    // only holds for the default (release-shaped) build.
    #[cfg(not(feature = "native_manifest_strict"))]
    #[test]
    fn native_lost_is_hard_block() {
        let severity = report_fallback(
            Family::GateUp,
            "gfx1201-synthetic",
            FallbackReason::NativeLost,
        );
        assert_eq!(severity, Severity::HardBlock);
    }

    #[cfg(feature = "native_manifest_strict")]
    #[test]
    #[should_panic(expected = "NATIVE-LOST")]
    fn native_lost_panics_under_strict_feature() {
        report_fallback(
            Family::GateUp,
            "gfx1201-synthetic-strict",
            FallbackReason::NativeLost,
        );
    }

    #[test]
    fn expected_and_precondition_are_not_the_same_reason() {
        assert_ne!(
            FallbackReason::ExpectedInManifest,
            FallbackReason::PreconditionFailed
        );
        // Both are non-hard-block (warn+degrade) — only NativeLost hard-blocks.
        assert!(!FallbackReason::ExpectedInManifest.is_hard_block());
        assert!(!FallbackReason::PreconditionFailed.is_hard_block());
        assert!(FallbackReason::NativeLost.is_hard_block());

        // `Severity` alone doesn't distinguish first-seen from a repeat (both
        // are Warn) — that's exercised for real below via `should_log`. This
        // just checks report_fallback doesn't blow up across repeats/reasons.
        let first = report_fallback(
            Family::Down,
            "gfx1201-synthetic-2",
            FallbackReason::ExpectedInManifest,
        );
        let second = report_fallback(
            Family::Down,
            "gfx1201-synthetic-2",
            FallbackReason::ExpectedInManifest,
        );
        let third = report_fallback(
            Family::Down,
            "gfx1201-synthetic-2",
            FallbackReason::PreconditionFailed,
        );
        assert_eq!(first, Severity::Warn);
        assert_eq!(second, Severity::Warn);
        assert_eq!(third, Severity::Warn);
    }

    // Direct test of the dedup logic itself (not just the Severity it
    // happens to return, which is identical whether or not the log line
    // fires). Uses (Qkvza, PreconditionFailed) — a (family, reason) pair no
    // other test in this module touches — so it's independent of test
    // execution order on the process-global dedup set.
    #[test]
    fn should_log_dedups_by_family_reason_key() {
        assert!(
            should_log(Family::Qkvza, FallbackReason::PreconditionFailed),
            "first report of a (family, reason) key must log"
        );
        assert!(
            !should_log(Family::Qkvza, FallbackReason::PreconditionFailed),
            "repeat of the SAME (family, reason) key must be deduped/silent"
        );
        assert!(
            should_log(Family::Qkvza, FallbackReason::ExpectedInManifest),
            "a DIFFERENT reason for the same family is a distinct key and logs again"
        );
    }

    // Regression test for the process-global-not-per-arch bug: a naive
    // `OnceLock<()>` implementation would only ever emit for the FIRST arch
    // seen by the process and silently swallow every other arch. Uses
    // arch strings no other test touches.
    #[test]
    fn emit_summary_once_per_arch_not_process() {
        assert!(
            should_emit_summary("gfx1201-emit-test-a"),
            "first time this arch is seen must emit"
        );
        assert!(
            !should_emit_summary("gfx1201-emit-test-a"),
            "repeat of the SAME arch must be deduped/silent"
        );
        assert!(
            should_emit_summary("gfx1100-emit-test-b"),
            "a DIFFERENT arch in the same process must still emit its own summary"
        );
    }
}
