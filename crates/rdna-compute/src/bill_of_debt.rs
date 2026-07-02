// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The bill-of-debt ledger (Oracle Phase 0, spec §4d) — one row per
//! `(arch × fitting-model × kernel × domain)` recording an RDNA kernel's
//! recoverable in-model time as a [`DebtRow`] variant:
//!
//!   - [`DebtRow::Measured`] — a real on-device measurement: an in-model
//!     wall-time and how far off the roofline it ran.
//!   - [`DebtRow::Structural`] — a known fallback penalty (a slow-path the
//!     arch is forced onto), carried as a fixed cost.
//!   - [`DebtRow::Withheld`] — a cell we could NOT honestly measure
//!     (absent kernel / OOM / unverified capture). It is recorded, never
//!     faked, and contributes no debt magnitude.
//!
//! Mirrors the [`crate::kernel_ledger`] AtlasRow-JSONL pattern (reusing
//! `hipfire_atlas::schema::AtlasRow` read-only from this stable home,
//! keeping ledger code OUT of the revert-bound `hipfire-atlas` crate).
//!
//! **Data-not-tags invariant:** the *debt magnitude* and any `bound_class`
//! verdict are DERIVED at query time ([`DebtRow::debt_magnitude_ms`]), never
//! persisted. The committed JSONL carries raw fields ONLY — round-trip must
//! never leak a `debt_magnitude` / `recoverable` / `bound_class` /
//! `unevenness` column onto disk.

use hipfire_atlas::schema::AtlasRow;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Below this magnitude a per-arch debt delta is floating-point noise, not a
/// real change — used by [`BillOfDebt::no_arch_clobber_delta`] as the ONLY
/// tolerance (there is deliberately NO percentage "noise band": spec §4d
/// requires candidate per-arch debt `<=` baseline for EVERY arch).
const DEBT_EPS_MS: f64 = 1e-9;

/// Identity of one bill-of-debt row: `(arch × fitting-model × kernel ×
/// domain)`. `model` is the model the kernel was *fitting* when measured
/// (in-model attribution is per-model); `domain` is the workload domain
/// (attention / moe_gate_up / lm_head / …).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DebtKey {
    pub arch: String,
    pub model: String,
    pub kernel: String,
    pub domain: String,
}

/// One bill-of-debt row. See the module docs for the three-variant
/// provenance model. All numeric fields are RAW measurements — the debt
/// magnitude is derived by [`Self::debt_magnitude_ms`], never stored.
#[derive(Debug, Clone, PartialEq)]
pub enum DebtRow {
    /// A real on-device measurement.
    Measured {
        key: DebtKey,
        /// Wall-time this kernel spent inside the model's forward pass (ms).
        in_model_walltime_ms: f64,
        /// The theoretical roofline time for this kernel/shape (ms) — carried
        /// for provenance; NOT used in the derived debt magnitude.
        roofline_ms: f64,
        /// How far off the roofline the kernel actually ran (percent). The
        /// recoverable fraction of `in_model_walltime_ms`.
        pct_off_roofline: f64,
    },
    /// A known structural fallback penalty (a slow path the arch is forced
    /// onto). The whole penalty is recoverable debt.
    Structural {
        key: DebtKey,
        fallback_penalty_ms: f64,
    },
    /// A cell we could not honestly measure — recorded, never faked.
    Withheld { key: DebtKey, reason: String },
}

impl DebtRow {
    /// The `(arch × model × kernel × domain)` identity of this row.
    pub fn key(&self) -> &DebtKey {
        match self {
            DebtRow::Measured { key, .. }
            | DebtRow::Structural { key, .. }
            | DebtRow::Withheld { key, .. } => key,
        }
    }

    /// Query-DERIVED recoverable debt magnitude (ms). Never persisted.
    ///
    ///   - `Measured` → `in_model_walltime_ms * pct_off_roofline / 100`
    ///     (the recoverable slice of the in-model wall-time).
    ///   - `Structural` → `fallback_penalty_ms` (the whole penalty).
    ///   - `Withheld` → `None` (no honest magnitude exists).
    pub fn debt_magnitude_ms(&self) -> Option<f64> {
        match self {
            DebtRow::Measured {
                in_model_walltime_ms,
                pct_off_roofline,
                ..
            } => Some(in_model_walltime_ms * pct_off_roofline / 100.0),
            DebtRow::Structural {
                fallback_penalty_ms,
                ..
            } => Some(*fallback_penalty_ms),
            DebtRow::Withheld { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(
        arch: &str,
        kernel: &str,
        domain: &str,
        in_model_walltime_ms: f64,
        roofline_ms: f64,
        pct_off_roofline: f64,
    ) -> DebtRow {
        DebtRow::Measured {
            key: DebtKey {
                arch: arch.to_string(),
                model: String::new(),
                kernel: kernel.to_string(),
                domain: domain.to_string(),
            },
            in_model_walltime_ms,
            roofline_ms,
            pct_off_roofline,
        }
    }

    fn structural(arch: &str, kernel: &str, domain: &str, fallback_penalty_ms: f64) -> DebtRow {
        DebtRow::Structural {
            key: DebtKey {
                arch: arch.to_string(),
                model: String::new(),
                kernel: kernel.to_string(),
                domain: domain.to_string(),
            },
            fallback_penalty_ms,
        }
    }

    fn withheld(arch: &str, kernel: &str, domain: &str, reason: &str) -> DebtRow {
        DebtRow::Withheld {
            key: DebtKey {
                arch: arch.to_string(),
                model: String::new(),
                kernel: kernel.to_string(),
                domain: domain.to_string(),
            },
            reason: reason.to_string(),
        }
    }

    #[test]
    fn debt_magnitude_derived_per_variant() {
        // Measured: in_model_walltime_ms * pct_off_roofline / 100 = 200 * 50 / 100 = 100.
        let m = measured("gfx1100", "k", "d", 200.0, 50.0, 50.0);
        assert_eq!(m.debt_magnitude_ms(), Some(100.0));
        // Structural: the fallback penalty is the whole debt.
        let s = structural("gfx1100", "k", "d", 12.5);
        assert_eq!(s.debt_magnitude_ms(), Some(12.5));
        // Withheld: no honest magnitude.
        let w = withheld("gfx1100", "k", "d", "oom");
        assert_eq!(w.debt_magnitude_ms(), None);
    }
}
