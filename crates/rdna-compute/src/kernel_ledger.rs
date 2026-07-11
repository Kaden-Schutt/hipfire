// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Coherence-stamped, shape-keyed kernel-perf ledger — composes
//! [`crate::isa_histogram::IsaHistogram`] + [`crate::roofline::Roofline`]
//! into a committed-JSONL-able record ([`LedgerRow`]) over
//! `hipfire_atlas::schema::AtlasRow`, plus a static NO-GPU JIT
//! module-name collision scanner ([`module_collision_scan`]) for a
//! directory of `.hsaco` fixtures/cache blobs.
//!
//! Two things this module deliberately does NOT do:
//!   - Live in the `hipfire-atlas` crate (Global Constraint: "Ledger code
//!     stays OUT of the revert-bound hipfire-atlas crate — reuse
//!     `AtlasRow` read-only from a stable home"). `hipfire-atlas` is a
//!     dependency here, not the other way round.
//!   - Decide, on its own, that a throughput GAIN is real: any dynamic
//!     "gain" delta without a `coherence` stamp on the *current* row is
//!     downgraded to [`LedgerDelta::RejectedGainNoCoherence`] rather than
//!     accepted — see the mandatory "Δ ≥ 5%/coherence-gate" rule in
//!     `CLAUDE.md`.

use crate::isa_histogram::{self, IsaHistogram};
use crate::roofline::BoundClass;
use hipfire_atlas::schema::AtlasRow;
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `±3%` — the same session A/B noise band `docs/methodology/perf-benchmarking.md`
/// establishes for gfx11; used here as the "is this dynamic metric actually
/// different" threshold before a ledger delta is even considered
/// Soft-regression/Gain (vs. plain noise).
const DYNAMIC_BAND_PCT: f64 = 3.0;

/// Identity of one ledger row — mirrors the `scripts/kernel_atlas.py`
/// grouping grammar (`arch`, `workload`, `phase`, `shape_bucket`, `quant`)
/// plus the specific kernel symbol, since one (workload, phase, shape,
/// quant) tuple can be served by more than one kernel across arches/tiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LedgerKey {
    pub arch: String,
    pub kernel: String,
    pub shape_bucket: String,
    pub quant: String,
    pub workload: String,
    pub phase: String,
}

/// How to reproduce the measurement behind a [`LedgerRow`] — the fixture
/// or live command, and (per the mandatory prompt-md5 rule) the exact
/// prompt bytes hash when the row came from a live decode/prefill run
/// rather than a static fixture.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reproducer {
    pub cmd: String,
    pub fixture_path: Option<String>,
    pub prompt_md5: Option<String>,
}

impl Reproducer {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "cmd": self.cmd,
            "fixture_path": self.fixture_path,
            "prompt_md5": self.prompt_md5,
        })
    }

    fn from_json(v: &Value) -> Result<Self, String> {
        let cmd = v["cmd"]
            .as_str()
            .ok_or_else(|| "Reproducer JSON missing string field 'cmd'".to_string())?
            .to_string();
        Ok(Self {
            cmd,
            fixture_path: v["fixture_path"].as_str().map(str::to_string),
            prompt_md5: v["prompt_md5"].as_str().map(str::to_string),
        })
    }
}

/// A coherence-gate stamp — per CLAUDE.md's mandatory rule, a "gain" ledger
/// row is rejected without one of these attached to the CURRENT measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceStamp {
    pub passed: bool,
    pub gate: String,
    pub report_path: Option<String>,
}

impl CoherenceStamp {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "passed": self.passed,
            "gate": self.gate,
            "report_path": self.report_path,
        })
    }

    fn from_json(v: &Value) -> Result<Self, String> {
        let passed = v["passed"]
            .as_bool()
            .ok_or_else(|| "CoherenceStamp JSON missing bool field 'passed'".to_string())?;
        let gate = v["gate"]
            .as_str()
            .ok_or_else(|| "CoherenceStamp JSON missing string field 'gate'".to_string())?
            .to_string();
        Ok(Self {
            passed,
            gate,
            report_path: v["report_path"].as_str().map(str::to_string),
        })
    }
}

/// One ledger row: the identity ([`LedgerKey`]), static ISA/occupancy shape
/// (`isa_fingerprint`/`vgpr`/`sgpr`/`lds`/`scratch` — compared EXACT by
/// [`KernelLedger::diff`]), dynamic throughput (`achieved_bw`/`roofline_pct`
/// — compared banded by `±3%`), the [`BoundClass`] verdict, how to
/// reproduce it, and an optional coherence stamp gating any claimed gain.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerRow {
    pub key: LedgerKey,
    pub isa_fingerprint: u64,
    pub vgpr: u32,
    pub sgpr: u32,
    pub lds: u32,
    pub scratch: u32,
    pub achieved_bw: Option<f64>,
    pub roofline_pct: Option<f64>,
    pub bound_class: BoundClass,
    pub reproducer: Reproducer,
    pub coherence: Option<CoherenceStamp>,
}

impl LedgerRow {
    /// Build a `LedgerRow` from a real committed `.hsaco` fixture (no GPU):
    /// [`IsaHistogram::from_hsaco`] for the ISA fingerprint, plus
    /// `profiler::profile_kernels` for vgpr/sgpr/lds/scratch (via
    /// [`crate::roofline::Roofline`]'s own upstream deps) — the static
    /// half of the pipeline, matching `kernel_perf_instrument --self-check`.
    pub fn from_fixture(
        key: LedgerKey,
        hist: &IsaHistogram,
        kprofile: &crate::profiler::KernelProfile,
        reproducer: Reproducer,
    ) -> Self {
        Self {
            key,
            isa_fingerprint: isa_fingerprint(hist),
            vgpr: kprofile.vgprs,
            sgpr: kprofile.sgprs,
            lds: kprofile.lds_bytes,
            scratch: kprofile.scratch_bytes,
            achieved_bw: None,
            roofline_pct: None,
            bound_class: BoundClass::ValuIssue, // overwritten by caller once Roofline::analyze runs
            reproducer,
            coherence: None,
        }
    }

    /// Map into `AtlasRow`. `phase`/`workload_kind`/`quant`/`shape_bucket`
    /// use `AtlasRow`'s native fields (shared schema with
    /// `scripts/kernel_atlas.py`); `arch`/`kernel`/`isa_fingerprint`/
    /// `vgpr`/`sgpr`/`lds`/`scratch`/`bound_class`/`reproducer`/
    /// `coherence` (none of which `AtlasRow` has a native slot for) go
    /// through `set_metric_*`/`set_extra`.
    pub fn to_atlas_row(&self) -> AtlasRow {
        let mut row = AtlasRow::new(self.key.phase.clone(), self.key.workload.clone());
        row.quant = self.key.quant.clone();
        row.shape_bucket = self.key.shape_bucket.clone();

        row.set_extra("arch", Value::String(self.key.arch.clone()));
        row.set_extra("kernel", Value::String(self.key.kernel.clone()));
        row.set_metric_u64("isa_fingerprint", self.isa_fingerprint);
        row.set_metric_u64("vgpr", self.vgpr as u64);
        row.set_metric_u64("sgpr", self.sgpr as u64);
        row.set_metric_u64("lds", self.lds as u64);
        row.set_metric_u64("scratch", self.scratch as u64);
        if let Some(bw) = self.achieved_bw {
            row.set_metric_f64("achieved_bw_gbps", bw);
        }
        if let Some(pct) = self.roofline_pct {
            row.set_metric_f64("roofline_pct", pct);
        }
        row.set_extra(
            "bound_class",
            Value::String(bound_class_to_str(self.bound_class).to_string()),
        );
        row.set_extra("reproducer", self.reproducer.to_json());
        row.set_extra(
            "coherence",
            match &self.coherence {
                Some(c) => c.to_json(),
                None => Value::Null,
            },
        );
        row
    }

    /// Inverse of [`Self::to_atlas_row`]. `Err` on any missing/malformed
    /// required field — a corrupted committed ledger row must fail loud,
    /// not silently coerce to a default.
    pub fn from_atlas_row(row: &AtlasRow) -> Result<Self, String> {
        let arch = row
            .extra
            .get("arch")
            .and_then(Value::as_str)
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing extra.arch".to_string())?
            .to_string();
        let kernel = row
            .extra
            .get("kernel")
            .and_then(Value::as_str)
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing extra.kernel".to_string())?
            .to_string();
        let key = LedgerKey {
            arch,
            kernel,
            shape_bucket: row.shape_bucket.clone(),
            quant: row.quant.clone(),
            workload: row.workload_kind.clone(),
            phase: row.phase.clone(),
        };

        let isa_fingerprint = row.metric_u64("isa_fingerprint").ok_or_else(|| {
            "LedgerRow::from_atlas_row: missing metrics.isa_fingerprint".to_string()
        })?;
        let vgpr = row
            .metric_u64("vgpr")
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing metrics.vgpr".to_string())?
            as u32;
        let sgpr = row
            .metric_u64("sgpr")
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing metrics.sgpr".to_string())?
            as u32;
        let lds = row
            .metric_u64("lds")
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing metrics.lds".to_string())?
            as u32;
        let scratch = row
            .metric_u64("scratch")
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing metrics.scratch".to_string())?
            as u32;
        let achieved_bw = row.metric_f64("achieved_bw_gbps");
        let roofline_pct = row.metric_f64("roofline_pct");

        let bound_class_str = row
            .extra
            .get("bound_class")
            .and_then(Value::as_str)
            .ok_or_else(|| "LedgerRow::from_atlas_row: missing extra.bound_class".to_string())?;
        let bound_class = bound_class_from_str(bound_class_str)?;

        let reproducer = match row.extra.get("reproducer") {
            Some(v) => Reproducer::from_json(v)?,
            None => Reproducer::default(),
        };
        let coherence = match row.extra.get("coherence") {
            None | Some(Value::Null) => None,
            Some(v) => Some(CoherenceStamp::from_json(v)?),
        };

        Ok(Self {
            key,
            isa_fingerprint,
            vgpr,
            sgpr,
            lds,
            scratch,
            achieved_bw,
            roofline_pct,
            bound_class,
            reproducer,
            coherence,
        })
    }
}

fn bound_class_to_str(b: BoundClass) -> &'static str {
    match b {
        BoundClass::Bandwidth => "bandwidth",
        BoundClass::ValuIssue => "valu_issue",
        BoundClass::Latency => "latency",
    }
}

fn bound_class_from_str(s: &str) -> Result<BoundClass, String> {
    match s {
        "bandwidth" => Ok(BoundClass::Bandwidth),
        "valu_issue" => Ok(BoundClass::ValuIssue),
        "latency" => Ok(BoundClass::Latency),
        other => Err(format!("unknown bound_class '{other}'")),
    }
}

/// A deterministic fingerprint of an [`IsaHistogram`]'s instruction-mix
/// shape — the "static metrics compared EXACT" half of
/// [`KernelLedger::diff`]. Any change to the compiled ISA (different
/// dequant chain, a spill appearing, a dp4a kernel taking over) flips
/// this, independent of `vgpr`/`sgpr`/`lds`/`scratch` (which are already
/// tracked as their own exact fields).
pub fn isa_fingerprint(hist: &IsaHistogram) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hist.v_bfe.hash(&mut hasher);
    hist.v_cvt.hash(&mut hasher);
    hist.f32_fma.hash(&mut hasher);
    hist.v_dot4.hash(&mut hasher);
    hist.v_wmma.hash(&mut hasher);
    hist.global_load_b128.hash(&mut hasher);
    hist.s_delay_alu.hash(&mut hasher);
    hist.s_wait.hash(&mut hasher);
    hist.private_segment_red.hash(&mut hasher);
    hist.vmem_valu_ratio.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// One row-level diff outcome between a committed baseline and a current
/// measurement. Static fields (`vgpr`/`sgpr`/`lds`/`scratch`/
/// `isa_fingerprint`) are compared EXACT — any change is
/// [`LedgerDelta::RegressionHard`], full stop (a real ISA/occupancy shape
/// change, not noise). Dynamic fields (`achieved_bw_gbps`/`roofline_pct`)
/// are banded at `±3%` (session A/B noise floor); outside the band, a
/// worse number is [`LedgerDelta::RegressionSoft`] and a better one is
/// [`LedgerDelta::Gain`] — UNLESS the current row lacks a
/// [`CoherenceStamp`], in which case the gain is downgraded to
/// [`LedgerDelta::RejectedGainNoCoherence`] per CLAUDE.md's mandatory
/// coherence-before-any-perf-gain-claim rule.
#[derive(Debug, Clone, PartialEq)]
pub enum LedgerDelta {
    RegressionHard {
        field: String,
        committed: String,
        current: String,
    },
    RegressionSoft {
        field: String,
        pct_change: f64,
    },
    Gain {
        field: String,
        pct_change: f64,
    },
    RejectedGainNoCoherence {
        field: String,
        pct_change: f64,
    },
}

/// A committed-baseline JSONL corpus of [`LedgerRow`]s, keyed by
/// [`LedgerKey`].
pub struct KernelLedger {
    pub rows: Vec<LedgerRow>,
}

impl KernelLedger {
    /// Load a committed ledger JSONL. NO GPU — pure file I/O via
    /// `hipfire_atlas::schema::load_rows`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let atlas_rows = hipfire_atlas::schema::load_rows(path)?;
        let rows = atlas_rows
            .iter()
            .map(LedgerRow::from_atlas_row)
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { rows })
    }

    pub fn find(&self, key: &LedgerKey) -> Option<&LedgerRow> {
        self.rows.iter().find(|r| &r.key == key)
    }

    /// Diff a committed baseline row against a freshly-measured current
    /// row. See [`LedgerDelta`] doc comment for the exact/banded split.
    /// Returns an empty `Vec` when nothing crosses either threshold.
    pub fn diff(committed: &LedgerRow, current: &LedgerRow) -> Vec<LedgerDelta> {
        let mut deltas = Vec::new();

        hard_check(&mut deltas, "vgpr", committed.vgpr, current.vgpr);
        hard_check(&mut deltas, "sgpr", committed.sgpr, current.sgpr);
        hard_check(&mut deltas, "lds", committed.lds, current.lds);
        hard_check(&mut deltas, "scratch", committed.scratch, current.scratch);
        hard_check(
            &mut deltas,
            "isa_fingerprint",
            committed.isa_fingerprint,
            current.isa_fingerprint,
        );

        banded_check(
            &mut deltas,
            "achieved_bw_gbps",
            committed.achieved_bw,
            current.achieved_bw,
            current.coherence.is_some(),
        );
        banded_check(
            &mut deltas,
            "roofline_pct",
            committed.roofline_pct,
            current.roofline_pct,
            current.coherence.is_some(),
        );

        deltas
    }
}

fn hard_check<T: PartialEq + std::fmt::Debug>(
    deltas: &mut Vec<LedgerDelta>,
    field: &str,
    committed: T,
    current: T,
) {
    if committed != current {
        deltas.push(LedgerDelta::RegressionHard {
            field: field.to_string(),
            committed: format!("{committed:?}"),
            current: format!("{current:?}"),
        });
    }
}

fn banded_check(
    deltas: &mut Vec<LedgerDelta>,
    field: &str,
    committed: Option<f64>,
    current: Option<f64>,
    has_coherence: bool,
) {
    let (Some(c), Some(cur)) = (committed, current) else {
        return; // one side unmeasured — nothing to diff
    };
    if c == 0.0 {
        return; // avoid div-by-zero; a zero baseline can't be percentage-banded
    }
    let pct_change = (cur - c) / c * 100.0;
    if pct_change.abs() <= DYNAMIC_BAND_PCT {
        return; // within session A/B noise floor — not a delta
    }
    if pct_change > 0.0 {
        if has_coherence {
            deltas.push(LedgerDelta::Gain {
                field: field.to_string(),
                pct_change,
            });
        } else {
            deltas.push(LedgerDelta::RejectedGainNoCoherence {
                field: field.to_string(),
                pct_change,
            });
        }
    } else {
        deltas.push(LedgerDelta::RegressionSoft {
            field: field.to_string(),
            pct_change,
        });
    }
}

/// A detected JIT module-name collision: two or more `.hsaco` files in a
/// directory whose EMBEDDED kernel symbol (the ELF `.kd`-suffixed symbol —
/// the real JIT module identity that `compiler.rs::guard_module_collision`
/// keys its cache on at runtime) is identical, but whose file bytes
/// differ. This is the static, directory-scan analog of the Task 2 bug
/// (gfx12 MMQ HFQ4G256: RDNA3 source pre-loaded under the gfx12 module
/// name → empty-stub kernels → ~100% NRMSE).
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleCollision {
    pub module_name: String,
    pub files: Vec<PathBuf>,
}

/// Scan every `.hsaco` file directly inside `dir`, group by embedded
/// kernel symbol name, and report any group whose members have differing
/// file bytes. NO GPU — shells out to `clang-offload-bundler` +
/// `llvm-readelf` (offline LLVM toolchain), same tool-discovery path as
/// [`crate::isa_histogram::IsaHistogram::from_hsaco`].
pub fn module_collision_scan(
    dir: impl AsRef<Path>,
    arch: &str,
) -> Result<Vec<ModuleCollision>, String> {
    let dir = dir.as_ref();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("module_collision_scan: failed to read {dir:?}: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("hsaco"))
        .collect();
    entries.sort();

    let mut by_symbol: HashMap<String, Vec<(PathBuf, u64)>> = HashMap::new();
    for path in entries {
        let data = std::fs::read(&path)
            .map_err(|e| format!("module_collision_scan: failed to read {path:?}: {e}"))?;
        let symbol = kernel_symbol_name(&path, arch)?;
        by_symbol
            .entry(symbol)
            .or_default()
            .push((path, content_hash(&data)));
    }

    let mut collisions: Vec<ModuleCollision> = by_symbol
        .into_iter()
        .filter_map(|(symbol, members)| {
            let first_hash = members[0].1;
            let differs = members.iter().any(|(_, h)| *h != first_hash);
            if members.len() > 1 && differs {
                Some(ModuleCollision {
                    module_name: symbol,
                    files: members.into_iter().map(|(p, _)| p).collect(),
                })
            } else {
                None
            }
        })
        .collect();
    collisions.sort_by(|a, b| a.module_name.cmp(&b.module_name));
    Ok(collisions)
}

fn content_hash(data: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Extract the embedded `.kd`-suffixed kernel symbol name (with the `.kd`
/// suffix stripped) from a raw `.hsaco` ELF or a
/// `__CLANG_OFFLOAD_BUNDLE__` fat binary, via `llvm-readelf -s`. Reuses
/// `isa_histogram`'s private tool-discovery/unbundle helpers rather than
/// re-deriving them.
fn kernel_symbol_name(path: &Path, arch: &str) -> Result<String, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("kernel_symbol_name: failed to read {path:?}: {e}"))?;

    let readelf = isa_histogram::find_llvm_tool("llvm-readelf").ok_or_else(|| {
        "kernel_symbol_name: llvm-readelf not found (checked PATH and $HIP_PATH/llvm/bin, \
         default /opt/rocm/llvm/bin)"
            .to_string()
    })?;

    let (disasm_input, _guard) = if isa_histogram::is_offload_bundle(&data) {
        let bundler = isa_histogram::find_llvm_tool("clang-offload-bundler").ok_or_else(|| {
            "kernel_symbol_name: clang-offload-bundler not found (checked PATH and \
             $HIP_PATH/llvm/bin, default /opt/rocm/llvm/bin)"
                .to_string()
        })?;
        let targets = isa_histogram::list_bundle_targets(&bundler, path)?;
        let chosen = isa_histogram::choose_bundle_target(&targets, arch).ok_or_else(|| {
            format!(
                "kernel_symbol_name({path:?}): no bundle target for arch '{arch}' \
                 (available targets: {targets:?})"
            )
        })?;
        let extracted = isa_histogram::unbundle(&bundler, path, &chosen)?;
        let guard = isa_histogram::TempExtract(extracted.clone());
        (extracted, Some(guard))
    } else {
        (path.to_path_buf(), None)
    };

    let output = Command::new(&readelf)
        .arg("-s")
        .arg(&disasm_input)
        .output()
        .map_err(|e| format!("kernel_symbol_name: failed to run llvm-readelf: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "kernel_symbol_name({path:?}): llvm-readelf failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if let Some(name) = line.split_whitespace().last() {
            if let Some(stripped) = name.strip_suffix(".kd") {
                return Ok(stripped.to_string());
            }
        }
    }
    Err(format!(
        "kernel_symbol_name({path:?}): no .kd symbol found in llvm-readelf output"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/kernel-fixtures/gfx1201")
    }

    fn sample_row() -> LedgerRow {
        LedgerRow {
            key: LedgerKey {
                arch: "gfx1201".to_string(),
                kernel: "gemv_hfq4g256_moe_gate_up_k8_indexed_batched".to_string(),
                shape_bucket: "decode_gemv".to_string(),
                quant: "hfq4g256".to_string(),
                workload: "moe_gate_up".to_string(),
                phase: "decode".to_string(),
            },
            isa_fingerprint: 0xDEAD_BEEF_u64,
            vgpr: 64,
            sgpr: 32,
            lds: 0,
            scratch: 0,
            achieved_bw: Some(120.0),
            roofline_pct: None,
            bound_class: BoundClass::ValuIssue,
            reproducer: Reproducer {
                cmd: "kernel_perf_instrument --self-check".to_string(),
                fixture_path: Some(
                    "tests/kernel-fixtures/gfx1201/gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco"
                        .to_string(),
                ),
                prompt_md5: None,
            },
            coherence: None,
        }
    }

    #[test]
    fn ledger_row_round_trips_through_atlas_row() {
        let row = sample_row();
        let atlas_row = row.to_atlas_row();

        let tmp = std::env::temp_dir().join(format!(
            "hipfire-kernel-ledger-test-roundtrip-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        atlas_row
            .append_to_jsonl(&tmp)
            .expect("append_to_jsonl must succeed");
        let loaded = hipfire_atlas::schema::load_rows(&tmp).expect("load_rows must succeed");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.len(), 1);
        let round = LedgerRow::from_atlas_row(&loaded[0]).expect("from_atlas_row must succeed");
        assert_eq!(
            round.isa_fingerprint, row.isa_fingerprint,
            "isa_fingerprint must round-trip"
        );
        assert_eq!(
            round.bound_class, row.bound_class,
            "bound_class must round-trip"
        );
        assert_eq!(round.key, row.key, "key must round-trip");
        assert_eq!(round.vgpr, row.vgpr);
        assert_eq!(round.sgpr, row.sgpr);
        assert_eq!(round.lds, row.lds);
        assert_eq!(round.scratch, row.scratch);
        assert_eq!(round.achieved_bw, row.achieved_bw);
    }

    #[test]
    fn diff_rejects_gain_without_coherence_stamp() {
        let committed = sample_row();
        let mut current = committed.clone();
        current.achieved_bw = Some(committed.achieved_bw.unwrap() * 1.5); // +50%, no coherence
        current.coherence = None;

        let deltas = KernelLedger::diff(&committed, &current);
        assert_eq!(
            deltas.len(),
            1,
            "only achieved_bw changed — expected exactly one delta, got {deltas:?}"
        );
        assert!(
            matches!(deltas[0], LedgerDelta::RejectedGainNoCoherence { .. }),
            "got {:?}",
            deltas[0]
        );
    }

    #[test]
    fn diff_accepts_gain_with_coherence_stamp() {
        let committed = sample_row();
        let mut current = committed.clone();
        current.achieved_bw = Some(committed.achieved_bw.unwrap() * 1.5); // +50%
        current.coherence = Some(CoherenceStamp {
            passed: true,
            gate: "coherence-gate.sh".to_string(),
            report_path: None,
        });

        let deltas = KernelLedger::diff(&committed, &current);
        assert_eq!(deltas.len(), 1, "got {deltas:?}");
        assert!(
            matches!(deltas[0], LedgerDelta::Gain { .. }),
            "got {:?}",
            deltas[0]
        );
    }

    #[test]
    fn diff_flags_regression_soft_below_band_without_coherence_requirement() {
        let committed = sample_row();
        let mut current = committed.clone();
        current.achieved_bw = Some(committed.achieved_bw.unwrap() * 0.5); // -50%, a loss
        current.coherence = None;

        let deltas = KernelLedger::diff(&committed, &current);
        assert_eq!(deltas.len(), 1, "got {deltas:?}");
        assert!(
            matches!(deltas[0], LedgerDelta::RegressionSoft { .. }),
            "a loss must never require a coherence stamp — got {:?}",
            deltas[0]
        );
    }

    #[test]
    fn diff_within_noise_band_is_unchanged() {
        let committed = sample_row();
        let mut current = committed.clone();
        current.achieved_bw = Some(committed.achieved_bw.unwrap() * 1.01); // +1%, inside ±3%

        let deltas = KernelLedger::diff(&committed, &current);
        assert!(
            deltas.is_empty(),
            "a +1% change must fall inside the ±3% noise band, got {deltas:?}"
        );
    }

    #[test]
    fn diff_flags_hard_regression_on_vgpr_change() {
        let committed = sample_row();
        let mut current = committed.clone();
        current.vgpr = committed.vgpr + 16; // static-field shape change

        let deltas = KernelLedger::diff(&committed, &current);
        assert!(
            deltas
                .iter()
                .any(|d| matches!(d, LedgerDelta::RegressionHard { field, .. } if field == "vgpr")),
            "a vgpr change must be flagged as RegressionHard EXACT, got {deltas:?}"
        );
    }

    #[test]
    fn load_committed_gfx1201_ledger_seed() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/kernel-ledger/gfx1201.jsonl");
        let ledger = KernelLedger::load(&path)
            .unwrap_or_else(|e| panic!("committed ledger seed must load: {e}"));
        assert_eq!(
            ledger.rows.len(),
            2,
            "expected the gate_up + down seed rows"
        );
        let gate_up = ledger
            .find(&LedgerKey {
                arch: "gfx1201".to_string(),
                kernel: "gemv_hfq4g256_moe_gate_up_indexed_batched".to_string(),
                shape_bucket: "decode_gemv".to_string(),
                quant: "hfq4g256".to_string(),
                workload: "moe_gate_up".to_string(),
                phase: "decode".to_string(),
            })
            .expect("gate_up row must be present in the committed seed");
        assert_eq!(gate_up.bound_class, BoundClass::ValuIssue);
        assert!(gate_up.vgpr > 0);
    }

    /// `find` (and therefore `diff`, which callers should always gate
    /// behind a `find`) is a FULL `LedgerKey` match — same `kernel` symbol
    /// but a different `quant`/`workload` must NOT be treated as the same
    /// baseline. This is the exact hazard a kernel-name-only lookup (the
    /// bug `kernel_perf_instrument`'s driver had before shape-keying the
    /// self-check row via `resolve_ledger_key`) would silently paper over.
    #[test]
    fn find_does_not_match_same_kernel_with_different_quant() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/kernel-ledger/gfx1201.jsonl");
        let ledger = KernelLedger::load(&path)
            .unwrap_or_else(|e| panic!("committed ledger seed must load: {e}"));

        let wrong_quant_key = LedgerKey {
            arch: "gfx1201".to_string(),
            kernel: "gemv_hfq4g256_moe_gate_up_indexed_batched".to_string(),
            shape_bucket: "decode_gemv".to_string(),
            quant: "mq4".to_string(), // WRONG — committed seed row is "hfq4g256"
            workload: "moe_gate_up".to_string(),
            phase: "decode".to_string(),
        };
        assert!(
            ledger.find(&wrong_quant_key).is_none(),
            "a same-kernel-name row with a different quant must not match — \
             a kernel-name-only lookup would incorrectly return the hfq4g256 row here"
        );

        let wrong_workload_key = LedgerKey {
            workload: "moe_down".to_string(), // WRONG — this kernel's real workload is moe_gate_up
            ..wrong_quant_key.clone()
        };
        let wrong_workload_key = LedgerKey {
            quant: "hfq4g256".to_string(),
            ..wrong_workload_key
        };
        assert!(
            ledger.find(&wrong_workload_key).is_none(),
            "a same-kernel-name row with a different workload must not match"
        );
    }

    #[test]
    fn module_collision_scan_finds_deliberate_pair() {
        let collisions = module_collision_scan(fixture_dir(), "gfx1201")
            .expect("module_collision_scan must succeed against committed fixtures");
        assert_eq!(
            collisions.len(),
            1,
            "expected exactly 1 collision (the deliberate collision_probe_variant_a/b pair), \
             got {collisions:?}"
        );
        let collision = &collisions[0];
        assert_eq!(collision.module_name, "collision_probe_kernel");
        assert_eq!(collision.files.len(), 2);
    }

    #[test]
    fn module_collision_scan_clean_pair_reports_no_collision() {
        // The two real gate_up/down fixtures have distinct embedded kernel
        // symbol names — scanning a directory containing ONLY those two
        // (a temp copy, so the deliberate collision pair doesn't leak in)
        // must report zero collisions.
        let tmp_dir = std::env::temp_dir().join(format!(
            "hipfire-kernel-ledger-test-clean-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        for name in [
            "gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco",
            "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hsaco",
        ] {
            std::fs::copy(fixture_dir().join(name), tmp_dir.join(name)).unwrap();
        }

        let collisions =
            module_collision_scan(&tmp_dir, "gfx1201").expect("module_collision_scan must succeed");
        let _ = std::fs::remove_dir_all(&tmp_dir);

        assert!(
            collisions.is_empty(),
            "gate_up and down have distinct kernel symbols — expected no collisions, got {collisions:?}"
        );
    }
}
