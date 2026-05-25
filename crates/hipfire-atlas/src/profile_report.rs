//! AtlasRow extension for rocprofv3 coverage reports.
//!
//! Defines a plain-data [`AtlasProfileReport`] (all owned types, no GPU
//! dependency) and an [`AtlasRow::set_profile_report`] method that serializes
//! it into the metrics/artifacts maps following the conventions below.
//!
//! # Metric keys written into `AtlasRow.metrics`
//!
//! | Key | Type | Description |
//! |-----|------|-------------|
//! | `internal_kernel_total_ms` | f64 | Sum of `HIPFIRE_PROFILE` timer entries |
//! | `rocprof_kernel_total_ms` | f64 | Ground-truth total from rocprofv3 |
//! | `rocprof_coverage_pct` | f64 | Fraction of rocprof time covered by internal profile |
//! | `rocprof_blindspot_total_ms` | f64 | GPU time invisible to internal profiling |
//! | `rocprof_blindspot_count` | u64 | Number of distinct un-tracked kernels |
//!
//! # Artifact keys written into `AtlasRow.artifacts`
//!
//! | Key | Type | Description |
//! |-----|------|-------------|
//! | `rocprof_blindspots` | JSON array | `{name, calls, duration_us, percent}` per blindspot |
//! | `rocprof_top_kernels` | JSON array | Top-10 rocprof entries by `duration_us` |

use crate::schema::AtlasRow;
use serde_json::{json, Value};

/// Plain-data summary of a rocprofv3 kernel entry, suitable for serialization.
/// Mirrors `rdna_compute::profile_rocprof::RocprofKernel` but with no GPU deps.
#[derive(Debug, Clone)]
pub struct AtlasRocprofKernel {
    pub name: String,
    pub calls: u64,
    pub duration_us: f64,
    pub percent: f64,
}

impl AtlasRocprofKernel {
    fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "calls": self.calls,
            "duration_us": self.duration_us,
            "percent": self.percent,
        })
    }
}

/// Plain-data coverage report for use with [`AtlasRow::set_profile_report`].
/// Construct from `rdna_compute::profile_rocprof::ProfileReport` by field copy.
#[derive(Debug, Clone)]
pub struct AtlasProfileReport {
    pub internal_total_us: f64,
    pub rocprof_total_us: f64,
    pub coverage_pct: f64,
    pub blindspot_total_us: f64,
    pub blindspots: Vec<AtlasRocprofKernel>,
    /// All rocprof kernels, sorted by duration descending (for top-10 slice).
    pub rocprof_all: Vec<AtlasRocprofKernel>,
}

impl AtlasRow {
    /// Serialize a coverage report into this row's `metrics` and `artifacts` maps.
    ///
    /// This method is additive: it does not clear existing metrics/artifacts.
    /// Call it after all other `set_metric_*` calls to avoid key collisions.
    pub fn set_profile_report(&mut self, report: &AtlasProfileReport) -> &mut Self {
        // Metrics
        self.metrics.insert(
            "internal_kernel_total_ms".to_string(),
            Value::from(report.internal_total_us / 1_000.0),
        );
        self.metrics.insert(
            "rocprof_kernel_total_ms".to_string(),
            Value::from(report.rocprof_total_us / 1_000.0),
        );
        self.metrics.insert(
            "rocprof_coverage_pct".to_string(),
            Value::from(report.coverage_pct),
        );
        self.metrics.insert(
            "rocprof_blindspot_total_ms".to_string(),
            Value::from(report.blindspot_total_us / 1_000.0),
        );
        self.metrics.insert(
            "rocprof_blindspot_count".to_string(),
            Value::from(report.blindspots.len() as u64),
        );

        // Artifacts
        let blindspot_json: Vec<Value> =
            report.blindspots.iter().map(|k| k.to_json()).collect();
        self.artifacts.insert(
            "rocprof_blindspots".to_string(),
            Value::Array(blindspot_json),
        );

        // Top-10 by duration (rocprof_all is already sorted descending by caller).
        let top10: Vec<Value> = report
            .rocprof_all
            .iter()
            .take(10)
            .map(|k| k.to_json())
            .collect();
        self.artifacts
            .insert("rocprof_top_kernels".to_string(), Value::Array(top10));

        self
    }
}
