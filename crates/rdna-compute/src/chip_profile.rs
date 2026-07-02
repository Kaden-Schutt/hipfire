// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-arch chip record — composes `profiler::GpuCapability` (runtime-probed)
//! and `arch_caps::ArchCaps` (intrinsic set) into a single committed-JSON-able
//! `ChipProfile`, with honesty guards so a stale/mismatched committed row
//! fails loud instead of silently mis-informing the roofline instrument
//! (Task 5) with wrong hardware constants.
//!
//! Two construction paths:
//!   - [`ChipProfile::load_committed`] — reads `tests/chip-profiles/<arch>.json`
//!     (NO GPU). This is the reference row checked into the repo.
//!   - [`ChipProfile::for_unprofiled`] — builds static-only fields for an arch
//!     that has never been measured (NO GPU); `mem_latency_ns`/`peak_bw_gbps`
//!     are `None`, which downstream roofline analysis (Task 5) must treat as
//!     "WITHHELD", never as zero.
//!
//! [`ChipProfile::verify_live`] cross-checks a committed row against a
//! live-probed `GpuCapability` (e.g. from `Gpu::init()` on hiptrx) and
//! returns `Err` on ANY static-field mismatch — a committed row going stale
//! after a driver/firmware/SKU change must fail loud, not silently drift.

use crate::arch_caps::ArchCaps;
use crate::feature_flags::FeatureFlags;
use crate::profiler::GpuCapability;
use std::sync::Arc;

/// Provenance metadata for a `ChipProfile` measurement. All fields are
/// `Option` — a purely-static (`for_unprofiled`) profile has all `None`;
/// a live-measured profile fills in what was actually sampled at capture
/// time. `measured_at_utc` is caller-supplied (no `SystemTime::now()` /
/// wall-clock read inside this library — see `docs/methodology/perf-benchmarking.md`
/// on avoiding hidden non-determinism in library code).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChipProfileProvenance {
    pub rocm_version: Option<String>,
    pub sclk_mhz: Option<u32>,
    pub mclk_mhz: Option<u32>,
    pub dpm_state: Option<String>,
    pub temp_c: Option<f32>,
    pub measured_at_utc: Option<String>,
}

impl ChipProfileProvenance {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "rocm_version": self.rocm_version,
            "sclk_mhz": self.sclk_mhz,
            "mclk_mhz": self.mclk_mhz,
            "dpm_state": self.dpm_state,
            "temp_c": self.temp_c,
            "measured_at_utc": self.measured_at_utc,
        })
    }

    fn from_json(v: &serde_json::Value) -> Self {
        Self {
            rocm_version: v["rocm_version"].as_str().map(str::to_string),
            sclk_mhz: v["sclk_mhz"].as_u64().map(|x| x as u32),
            mclk_mhz: v["mclk_mhz"].as_u64().map(|x| x as u32),
            dpm_state: v["dpm_state"].as_str().map(str::to_string),
            temp_c: v["temp_c"].as_f64().map(|x| x as f32),
            measured_at_utc: v["measured_at_utc"].as_str().map(str::to_string),
        }
    }
}

/// Chip-tied static + measured record for a single `gfxNNNN` arch. See the
/// module docs for the honesty-guard rationale.
#[derive(Debug, Clone, PartialEq)]
pub struct ChipProfile {
    pub arch: String,
    pub cu_count: u32,
    pub vgprs_per_simd: u32,
    pub max_waves_per_simd: u32,
    pub lds_bytes_per_cu: u32,
    pub wavefront_size: u32,
    /// DRAM pointer-chase latency in nanoseconds. `None` until the Task-3
    /// microbench (`pointer_chase_latency` example, GPU-required) has
    /// actually measured this arch — NEVER filled with an unlabeled
    /// estimate (per the mandatory "mem_latency is MEASURED" rule).
    pub mem_latency_ns: Option<f64>,
    /// Theoretical peak memory bandwidth in GB/s. `None` withholds any
    /// roofline BW-bound verdict downstream (Task 5) rather than silently
    /// treating an unknown chip as zero-bandwidth.
    pub peak_bw_gbps: Option<f64>,
    /// `ArchCaps::dump_json()` — the full atom/molecule/capability/tuning
    /// set for this arch, so the committed row is self-describing without
    /// re-deriving intrinsics from `FeatureFlags::from_env` at read time.
    pub intrinsics: serde_json::Value,
    pub provenance: ChipProfileProvenance,
}

/// Wavefront size for `arch`, derived the same way `ArchCaps::is_wave32()`
/// derives it — NOT re-hardcoded independently (single source of truth
/// stays in `arch_caps.rs`).
fn wavefront_size_for_arch(arch: &str) -> u32 {
    let flags = Arc::new(FeatureFlags::from_env(arch));
    let caps = ArchCaps::new(arch, flags);
    if caps.is_wave32() {
        32
    } else {
        64
    }
}

impl ChipProfile {
    /// Relative path (from the workspace root) of the committed chip-profile
    /// JSON for `arch`.
    fn committed_path(arch: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/chip-profiles")
            .join(format!("{arch}.json"))
    }

    /// Load the committed reference row for `arch` from
    /// `tests/chip-profiles/<arch>.json`. NO GPU required — pure file I/O +
    /// JSON parse. `Err` if the file is missing or malformed (fail loud
    /// rather than silently falling back to `for_unprofiled`).
    pub fn load_committed(arch: &str) -> Result<Self, String> {
        let path = Self::committed_path(arch);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!("ChipProfile::load_committed({arch}): failed to read {path:?}: {e}")
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            format!("ChipProfile::load_committed({arch}): invalid JSON in {path:?}: {e}")
        })?;
        Self::from_json(&value)
    }

    /// Build a static-only profile for `arch` with no measured dynamic
    /// fields (`mem_latency_ns`/`peak_bw_gbps` both `None` — the roofline
    /// instrument (Task 5) MUST treat these as WITHHELD, not zero). NO GPU
    /// required and no sysfs access — deterministic regardless of host.
    pub fn for_unprofiled(arch: &str) -> Self {
        let flags = Arc::new(FeatureFlags::from_env(arch));
        let caps = ArchCaps::new(arch, flags);
        // Purely arch-keyed static data — NO sysfs, NO live GPU, so the
        // result is identical regardless of which host builds it.
        let static_cap = GpuCapability::static_capability(arch);

        Self {
            arch: arch.to_string(),
            cu_count: static_cap.cu_count,
            vgprs_per_simd: static_cap.vgprs_per_simd,
            max_waves_per_simd: static_cap.max_waves_per_simd,
            lds_bytes_per_cu: static_cap.lds_per_cu_bytes,
            wavefront_size: if caps.is_wave32() { 32 } else { 64 },
            mem_latency_ns: None,
            peak_bw_gbps: None,
            intrinsics: caps.dump_json(),
            provenance: ChipProfileProvenance::default(),
        }
    }

    /// Cross-check this (typically committed) profile's static fields
    /// against a live-probed `GpuCapability`. `Err` on ANY mismatch —
    /// honesty guard against a stale committed row silently mis-informing
    /// the roofline instrument after a driver/firmware/SKU change.
    pub fn verify_live(&self, live: &GpuCapability) -> Result<(), String> {
        let mut mismatches = Vec::new();
        if self.arch != live.arch {
            mismatches.push(format!("arch: committed={} live={}", self.arch, live.arch));
        }
        if self.cu_count != live.cu_count {
            mismatches.push(format!(
                "cu_count: committed={} live={}",
                self.cu_count, live.cu_count
            ));
        }
        if self.vgprs_per_simd != live.vgprs_per_simd {
            mismatches.push(format!(
                "vgprs_per_simd: committed={} live={}",
                self.vgprs_per_simd, live.vgprs_per_simd
            ));
        }
        if self.lds_bytes_per_cu != live.lds_per_cu_bytes {
            mismatches.push(format!(
                "lds_bytes_per_cu: committed={} live={}",
                self.lds_bytes_per_cu, live.lds_per_cu_bytes
            ));
        }
        let live_wavefront = wavefront_size_for_arch(&live.arch);
        if self.wavefront_size != live_wavefront {
            mismatches.push(format!(
                "wavefront_size: committed={} live(derived)={}",
                self.wavefront_size, live_wavefront
            ));
        }
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "ChipProfile::verify_live({}): committed row stale vs live GpuCapability: {}",
                self.arch,
                mismatches.join("; ")
            ))
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "arch": self.arch,
            "cu_count": self.cu_count,
            "vgprs_per_simd": self.vgprs_per_simd,
            "max_waves_per_simd": self.max_waves_per_simd,
            "lds_bytes_per_cu": self.lds_bytes_per_cu,
            "wavefront_size": self.wavefront_size,
            "mem_latency_ns": self.mem_latency_ns,
            "peak_bw_gbps": self.peak_bw_gbps,
            "intrinsics": self.intrinsics,
            "provenance": self.provenance.to_json(),
        })
    }

    fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let arch = v["arch"]
            .as_str()
            .ok_or_else(|| "ChipProfile JSON missing string field 'arch'".to_string())?
            .to_string();
        let cu_count = v["cu_count"]
            .as_u64()
            .ok_or_else(|| format!("ChipProfile({arch}) JSON missing u64 field 'cu_count'"))?
            as u32;
        let vgprs_per_simd = v["vgprs_per_simd"]
            .as_u64()
            .ok_or_else(|| format!("ChipProfile({arch}) JSON missing u64 field 'vgprs_per_simd'"))?
            as u32;
        let max_waves_per_simd = v["max_waves_per_simd"].as_u64().ok_or_else(|| {
            format!("ChipProfile({arch}) JSON missing u64 field 'max_waves_per_simd'")
        })? as u32;
        let lds_bytes_per_cu = v["lds_bytes_per_cu"].as_u64().ok_or_else(|| {
            format!("ChipProfile({arch}) JSON missing u64 field 'lds_bytes_per_cu'")
        })? as u32;
        let wavefront_size = v["wavefront_size"]
            .as_u64()
            .ok_or_else(|| format!("ChipProfile({arch}) JSON missing u64 field 'wavefront_size'"))?
            as u32;
        let mem_latency_ns = v["mem_latency_ns"].as_f64();
        let peak_bw_gbps = v["peak_bw_gbps"].as_f64();
        let intrinsics = v["intrinsics"].clone();
        let provenance = ChipProfileProvenance::from_json(&v["provenance"]);

        Ok(Self {
            arch,
            cu_count,
            vgprs_per_simd,
            max_waves_per_simd,
            lds_bytes_per_cu,
            wavefront_size,
            mem_latency_ns,
            peak_bw_gbps,
            intrinsics,
            provenance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_committed_gfx1201() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        assert_eq!(profile.wavefront_size, 32, "gfx1201 is wave32-native RDNA4");
        assert_eq!(profile.cu_count, 56, "R9700 gfx1201 = 56 CU");
        assert_eq!(profile.vgprs_per_simd, 1536, "RDNA4 VGPR file per SIMD");
    }

    #[test]
    fn verify_live_fails_loud_on_mismatch() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        // Deliberately-corrupted live capability (wrong CU count — as if a
        // different SKU / a stale committed row were plugged in).
        let live = GpuCapability {
            arch: "gfx1201".to_string(),
            generation: "RDNA4",
            cu_count: 99,
            simds_per_cu: 2,
            max_waves_per_simd: 16,
            vgprs_per_simd: 1536,
            lds_per_cu_bytes: 65536,
            l2_cache_mb: 4.0,
            infinity_cache_mb: 64.0,
            peak_bw_gbs: 800.0,
            boost_clock_mhz: 2900,
            mem_clock_mhz: 2500,
            mem_bus_width_bits: 256,
            vram_mb: 32768,
        };
        let result = profile.verify_live(&live);
        assert!(
            result.is_err(),
            "verify_live must fail loud on a cu_count mismatch, got Ok"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("cu_count"),
            "error message should name the mismatched field: {msg}"
        );
    }

    #[test]
    fn verify_live_passes_on_exact_match() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        let live = GpuCapability {
            arch: "gfx1201".to_string(),
            generation: "RDNA4",
            cu_count: profile.cu_count,
            simds_per_cu: 2,
            max_waves_per_simd: profile.max_waves_per_simd,
            vgprs_per_simd: profile.vgprs_per_simd,
            lds_per_cu_bytes: profile.lds_bytes_per_cu,
            l2_cache_mb: 4.0,
            infinity_cache_mb: 64.0,
            peak_bw_gbs: 800.0,
            boost_clock_mhz: 2900,
            mem_clock_mhz: 2500,
            mem_bus_width_bits: 256,
            vram_mb: 32768,
        };
        assert!(
            profile.verify_live(&live).is_ok(),
            "verify_live must pass when every static field matches"
        );
    }

    #[test]
    fn for_unprofiled_withholds_roofline_inputs() {
        let profile = ChipProfile::for_unprofiled("gfx9999");
        assert!(
            profile.peak_bw_gbps.is_none(),
            "unprofiled arch must withhold peak_bw (roofline BW-bound verdict WITHHELD)"
        );
        assert!(
            profile.mem_latency_ns.is_none(),
            "unprofiled arch must withhold mem_latency (roofline latency-bound verdict WITHHELD)"
        );
        assert_eq!(profile.arch, "gfx9999");
    }

    #[test]
    fn round_trip_json() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        let json = profile.to_json();
        let reloaded = ChipProfile::from_json(&json).expect("round-trip JSON must parse");
        assert_eq!(profile, reloaded);
    }
}
