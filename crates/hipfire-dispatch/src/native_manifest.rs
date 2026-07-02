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
    let first_seen = {
        let mut set = reported_set().lock().unwrap_or_else(|e| e.into_inner());
        set.insert((family, reason))
    };
    if first_seen {
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

/// Emit the coverage summary exactly once per process (idempotent across
/// repeated `DispatchCtx::new` calls in a per-layer hot loop).
pub fn emit_summary_once(caps: &ArchCaps) {
    static EMITTED: OnceLock<()> = OnceLock::new();
    EMITTED.get_or_init(|| {
        eprintln!("{}", NativeManifest::snapshot(caps).summary());
    });
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

        // Telemetry split: reporting the SAME (family, reason) key twice is
        // deduped (second call is silent); a DIFFERENT reason for the same
        // family is a distinct key and reports again. Use a family unused by
        // other tests in this module to avoid cross-test interference on the
        // process-global dedup set.
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
}
