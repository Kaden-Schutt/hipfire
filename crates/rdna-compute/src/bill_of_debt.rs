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

    /// Map into `AtlasRow` for committed JSONL — RAW fields ONLY. `phase` is
    /// the fixed `"bill_of_debt"` tag; `workload_kind = domain`,
    /// `model_size = model`; `arch`/`kernel`/`debt_kind`/`withheld_reason` go
    /// through `set_extra`; numeric raw fields through `set_metric_f64`. The
    /// DERIVED debt magnitude / bound_class / unevenness are NEVER written.
    pub fn to_atlas_row(&self) -> AtlasRow {
        let key = self.key();
        let mut row = AtlasRow::new("bill_of_debt", key.domain.clone());
        row.model_size = key.model.clone();
        row.set_extra("arch", Value::String(key.arch.clone()));
        row.set_extra("kernel", Value::String(key.kernel.clone()));
        match self {
            DebtRow::Measured {
                in_model_walltime_ms,
                roofline_ms,
                pct_off_roofline,
                ..
            } => {
                row.set_extra("debt_kind", Value::String("measured".to_string()));
                row.set_metric_f64("in_model_walltime_ms", *in_model_walltime_ms);
                row.set_metric_f64("roofline_ms", *roofline_ms);
                row.set_metric_f64("pct_off_roofline", *pct_off_roofline);
            }
            DebtRow::Structural {
                fallback_penalty_ms,
                ..
            } => {
                row.set_extra("debt_kind", Value::String("structural".to_string()));
                row.set_metric_f64("fallback_penalty_ms", *fallback_penalty_ms);
            }
            DebtRow::Withheld { reason, .. } => {
                row.set_extra("debt_kind", Value::String("withheld".to_string()));
                row.set_extra("withheld_reason", Value::String(reason.clone()));
            }
        }
        row
    }

    /// Inverse of [`Self::to_atlas_row`]. `Err` on any missing/malformed
    /// required field — a corrupted committed row must fail loud, not
    /// silently coerce to a default.
    pub fn from_atlas_row(row: &AtlasRow) -> Result<Self, String> {
        let key = DebtKey {
            arch: extra_str(row, "arch")?,
            model: row.model_size.clone(),
            kernel: extra_str(row, "kernel")?,
            domain: row.workload_kind.clone(),
        };
        let debt_kind = extra_str(row, "debt_kind")?;
        match debt_kind.as_str() {
            "measured" => Ok(DebtRow::Measured {
                key,
                in_model_walltime_ms: metric(row, "in_model_walltime_ms")?,
                roofline_ms: metric(row, "roofline_ms")?,
                pct_off_roofline: metric(row, "pct_off_roofline")?,
            }),
            "structural" => Ok(DebtRow::Structural {
                key,
                fallback_penalty_ms: metric(row, "fallback_penalty_ms")?,
            }),
            "withheld" => Ok(DebtRow::Withheld {
                key,
                reason: extra_str(row, "withheld_reason")?,
            }),
            other => Err(format!(
                "DebtRow::from_atlas_row: unknown debt_kind '{other}'"
            )),
        }
    }
}

fn extra_str(row: &AtlasRow, key: &str) -> Result<String, String> {
    row.extra
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("DebtRow::from_atlas_row: missing extra.{key}"))
}

fn metric(row: &AtlasRow, key: &str) -> Result<f64, String> {
    row.metric_f64(key)
        .ok_or_else(|| format!("DebtRow::from_atlas_row: missing metrics.{key}"))
}

/// A committed corpus of [`DebtRow`]s — the per-arch "bill of debt". Query
/// methods derive debt magnitude / ranking / per-arch totals / cross-arch
/// unevenness on the fly; nothing derived is stored.
pub struct BillOfDebt {
    pub rows: Vec<DebtRow>,
}

impl BillOfDebt {
    /// Load a committed bill-of-debt JSONL. NO GPU — pure file I/O via
    /// `hipfire_atlas::schema::load_rows`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let atlas_rows = hipfire_atlas::schema::load_rows(path)?;
        let rows = atlas_rows
            .iter()
            .map(DebtRow::from_atlas_row)
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { rows })
    }

    /// Emit this bill as JSONL (truncating any existing file), one raw-fields
    /// row per [`DebtRow`]. NO derived column is ever written.
    pub fn emit(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        hipfire_atlas::schema::truncate_jsonl(path)
            .map_err(|e| format!("BillOfDebt::emit: truncate {}: {e}", path.display()))?;
        for row in &self.rows {
            row.to_atlas_row()
                .append_to_jsonl(path)
                .map_err(|e| format!("BillOfDebt::emit: append {}: {e}", path.display()))?;
        }
        Ok(())
    }

    /// Rows ranked by recoverable REAL time (`debt_magnitude_ms`, descending),
    /// excluding withheld rows (no honest magnitude). This ranks by the
    /// actual lever — recoverable wall-time — NOT by roofline efficiency: a
    /// kernel 90% off roofline but only 1ms in-model is a smaller lever than
    /// one 20% off roofline but 500ms in-model.
    pub fn ranked_by_lever(&self) -> Vec<&DebtRow> {
        let mut ranked: Vec<(&DebtRow, f64)> = self
            .rows
            .iter()
            .filter_map(|r| r.debt_magnitude_ms().map(|m| (r, m)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.into_iter().map(|(r, _)| r).collect()
    }

    /// The withheld cells — targets for future measurement (the honest gaps
    /// in the corpus). Recorded, never faked.
    pub fn withheld_targets(&self) -> Vec<&DebtRow> {
        self.rows
            .iter()
            .filter(|r| matches!(r, DebtRow::Withheld { .. }))
            .collect()
    }

    /// Total recoverable debt per arch (ms). Every arch that appears in the
    /// corpus gets an entry (a fully-withheld arch appears with `0.0`);
    /// withheld rows contribute NOTHING to the magnitude. Ordered for
    /// deterministic iteration via `BTreeMap`.
    pub fn per_arch_total_debt(&self) -> BTreeMap<String, f64> {
        let mut totals: BTreeMap<String, f64> = BTreeMap::new();
        for row in &self.rows {
            let entry = totals.entry(row.key().arch.clone()).or_insert(0.0);
            if let Some(d) = row.debt_magnitude_ms() {
                *entry += d;
            }
        }
        totals
    }

    /// Cross-arch unevenness score `(max - min) / mean` over per-arch total
    /// debt — how lopsidedly the recoverable debt is distributed across the
    /// fleet. `0.0` for fewer than 2 arches or a zero mean (nothing to
    /// compare / no debt at all).
    pub fn cross_arch_unevenness(&self) -> f64 {
        let totals = self.per_arch_total_debt();
        if totals.len() < 2 {
            return 0.0;
        }
        let vals: Vec<f64> = totals.values().copied().collect();
        let max = vals.iter().copied().fold(f64::MIN, f64::max);
        let min = vals.iter().copied().fold(f64::MAX, f64::min);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        if mean.abs() < DEBT_EPS_MS {
            return 0.0;
        }
        (max - min) / mean
    }

    /// No-arch-clobber delta (spec §4d): per-arch debt delta (candidate -
    /// baseline). §4d requires candidate debt <= baseline for EVERY arch, so an
    /// arch is *worsened* whenever its debt rises by more than DEBT_EPS_MS —
    /// there is NO tolerated percentage band. The CI invariant Phase-2 enforces
    /// is `any_arch_worsened == false`.
    pub fn no_arch_clobber_delta(baseline: &BillOfDebt, candidate: &BillOfDebt) -> ClobberReport {
        let base_totals = baseline.per_arch_total_debt();
        let cand_totals = candidate.per_arch_total_debt();
        let mut arches: Vec<String> = base_totals
            .keys()
            .chain(cand_totals.keys())
            .cloned()
            .collect();
        arches.sort();
        arches.dedup();
        let mut per_arch = Vec::new();
        let mut any_arch_worsened = false;
        for arch in arches {
            let b = base_totals.get(&arch).copied().unwrap_or(0.0);
            let c = cand_totals.get(&arch).copied().unwrap_or(0.0);
            let delta_ms = c - b;
            let worsened = delta_ms > DEBT_EPS_MS; // any real growth is a clobber (§4d)
            if worsened {
                any_arch_worsened = true;
            }
            per_arch.push(ArchDebtDelta {
                arch,
                baseline_debt_ms: b,
                candidate_debt_ms: c,
                delta_ms,
                worsened,
            });
        }
        ClobberReport {
            per_arch,
            any_arch_worsened,
        }
    }
}

/// One arch's baseline→candidate recoverable-debt delta.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchDebtDelta {
    pub arch: String,
    pub baseline_debt_ms: f64,
    pub candidate_debt_ms: f64,
    pub delta_ms: f64,
    /// `true` iff the candidate's debt for this arch grew beyond
    /// `DEBT_EPS_MS` — a §4d clobber (no tolerated percentage band).
    pub worsened: bool,
}

/// The result of a no-arch-clobber check. `any_arch_worsened == false` is the
/// CI invariant a candidate must hold to land (spec §4d).
#[derive(Debug, Clone, PartialEq)]
pub struct ClobberReport {
    pub per_arch: Vec<ArchDebtDelta>,
    pub any_arch_worsened: bool,
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
    fn ranked_by_lever_uses_real_time_not_efficiency() {
        let bill = BillOfDebt {
            rows: vec![
                // 90% off roofline but only 1ms in-model → tiny recoverable time.
                measured("gfx1100", "efficiency_trap", "d", 1.0, 0.1, 90.0), // debt = 0.9
                // 20% off roofline but 500ms in-model → large recoverable time.
                measured("gfx1100", "real_lever", "d", 500.0, 400.0, 20.0), // debt = 100.0
                // Withheld: no honest magnitude → must be excluded from ranking.
                withheld("gfx1100", "unmeasured", "d", "oom"),
            ],
        };
        let ranked = bill.ranked_by_lever();
        assert_eq!(
            ranked.len(),
            2,
            "withheld rows must be excluded from ranking"
        );
        assert_eq!(
            ranked[0].key().kernel,
            "real_lever",
            "the 100ms lever must outrank the 0.9ms efficiency trap (real time, not % efficiency)"
        );
        assert_eq!(ranked[1].key().kernel, "efficiency_trap");

        let withheld_kernels: Vec<&str> = bill
            .withheld_targets()
            .iter()
            .map(|r| r.key().kernel.as_str())
            .collect();
        assert_eq!(withheld_kernels, vec!["unmeasured"]);
    }

    #[test]
    fn per_arch_totals_and_unevenness() {
        let bill = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k1", "d", 100.0, 0.0, 10.0), // debt 10
                measured("gfx1010", "k2", "d", 100.0, 0.0, 20.0), // debt 20 → gfx1010 = 30
                measured("gfx1100", "k1", "d", 100.0, 0.0, 90.0), // debt 90 → gfx1100 = 90
                withheld("gfx1100", "kw", "d", "oom"),            // contributes nothing
            ],
        };
        let totals = bill.per_arch_total_debt();
        assert_eq!(totals.get("gfx1010").copied(), Some(30.0));
        assert_eq!(totals.get("gfx1100").copied(), Some(90.0));
        // (max - min) / mean = (90 - 30) / ((30 + 90) / 2) = 60 / 60 = 1.0
        let u = bill.cross_arch_unevenness();
        assert!(
            (u - 1.0).abs() < 1e-9,
            "unevenness = (90-30)/60 = 1.0, got {u}"
        );
    }

    #[test]
    fn unevenness_zero_when_even() {
        let even = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k", "d", 100.0, 0.0, 50.0), // debt 50
                measured("gfx1100", "k", "d", 100.0, 0.0, 50.0), // debt 50
            ],
        };
        assert!(even.cross_arch_unevenness().abs() < 1e-9);
        // <2 arches → 0 (no cross-arch spread to score).
        let single = BillOfDebt {
            rows: vec![measured("gfx1010", "k", "d", 100.0, 0.0, 50.0)],
        };
        assert_eq!(single.cross_arch_unevenness(), 0.0);
        // Empty / zero-mean → 0.
        let empty = BillOfDebt { rows: vec![] };
        assert_eq!(empty.cross_arch_unevenness(), 0.0);
    }

    #[test]
    fn no_arch_clobber_delta_flags_worsened_arch() {
        let baseline = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k1", "d1", 100.0, 50.0, 100.0),
                measured("gfx1100", "k2", "d2", 100.0, 50.0, 100.0),
            ],
        };
        let candidate = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k1", "d1", 100.0, 50.0, 150.0), // +50 -> clobber
                measured("gfx1100", "k2", "d2", 100.0, 50.0, 80.0),
            ],
        }; // -20 -> improvement
        let report = BillOfDebt::no_arch_clobber_delta(&baseline, &candidate);
        assert!(report.any_arch_worsened);
        let g1010 = report
            .per_arch
            .iter()
            .find(|d| d.arch == "gfx1010")
            .unwrap();
        assert_eq!(g1010.delta_ms, 50.0);
        assert!(g1010.worsened);
        let g1100 = report
            .per_arch
            .iter()
            .find(|d| d.arch == "gfx1100")
            .unwrap();
        assert_eq!(g1100.delta_ms, -20.0);
        assert!(!g1100.worsened);
    }
    #[test]
    fn no_arch_clobber_delta_clean_when_all_improve() {
        let baseline = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k1", "d1", 100.0, 50.0, 100.0),
                measured("gfx1100", "k2", "d2", 100.0, 50.0, 100.0),
            ],
        };
        let candidate = BillOfDebt {
            rows: vec![
                measured("gfx1010", "k1", "d1", 100.0, 50.0, 90.0),
                measured("gfx1100", "k2", "d2", 100.0, 50.0, 95.0),
            ],
        };
        assert!(!BillOfDebt::no_arch_clobber_delta(&baseline, &candidate).any_arch_worsened);
    }
    #[test]
    fn no_arch_clobber_delta_any_positive_growth_is_a_clobber() {
        // Spec §4d: candidate per-arch debt must be <= baseline for EVERY arch.
        // Even a small +2% growth is a clobber — there is NO tolerated noise band.
        let baseline = BillOfDebt {
            rows: vec![measured("gfx1100", "k", "d", 100.0, 50.0, 100.0)],
        };
        let candidate = BillOfDebt {
            rows: vec![measured("gfx1100", "k", "d", 100.0, 50.0, 102.0)],
        };
        let report = BillOfDebt::no_arch_clobber_delta(&baseline, &candidate);
        assert!(
            report.any_arch_worsened,
            "any per-arch debt growth violates §4d, no band"
        );
    }

    #[test]
    fn bill_round_trips_through_jsonl_raw_fields_only() {
        let bill = BillOfDebt {
            rows: vec![
                measured("gfx1100", "k1", "attention", 100.0, 50.0, 90.0),
                structural("gfx1010", "k2", "moe_gate_up", 42.0),
                withheld("gfx1201", "k3", "lm_head", "oom"),
            ],
        };
        let tmp = std::env::temp_dir().join(format!(
            "hipfire-bill-of-debt-roundtrip-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        bill.emit(&tmp).expect("emit must succeed");

        // Raw fields only — no DERIVED column may ever be persisted.
        let raw = std::fs::read_to_string(&tmp).expect("read raw jsonl");
        for forbidden in [
            "recoverable",
            "debt_magnitude",
            "debt_ms",
            "bound_class",
            "unevenness",
        ] {
            assert!(
                !raw.contains(forbidden),
                "derived field '{forbidden}' must NOT be persisted; found in:\n{raw}"
            );
        }

        let loaded = BillOfDebt::load(&tmp).expect("load must succeed");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(loaded.rows.len(), bill.rows.len());
        for (orig, round) in bill.rows.iter().zip(loaded.rows.iter()) {
            assert_eq!(orig, round, "every row must round-trip value-for-value");
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
