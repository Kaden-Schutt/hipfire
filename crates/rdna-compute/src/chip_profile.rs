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
    /// VRAM capacity in MB. `None` withholds the check (e.g. `for_unprofiled`,
    /// or an older committed row from before this field existed) rather than
    /// comparing against a bogus zero. When `Some`, `verify_live` fails loud
    /// on a mismatch — VRAM capacity is a SKU-identifying static field (the
    /// gfx1201 family alone spans 16GB RX 9070/9070 XT vs 32GB Radeon AI PRO
    /// R9700), exactly the "driver/firmware/SKU change" case the module docs
    /// call out.
    pub vram_mb: Option<u32>,
    /// DRAM pointer-chase latency in nanoseconds. `None` until the Task-3
    /// microbench (`pointer_chase_latency` example, GPU-required) has
    /// actually measured this arch — NEVER filled with an unlabeled
    /// estimate (per the mandatory "mem_latency is MEASURED" rule).
    pub mem_latency_ns: Option<f64>,
    /// Theoretical peak memory bandwidth in GB/s. `None` withholds any
    /// roofline BW-bound verdict downstream (Task 5) rather than silently
    /// treating an unknown chip as zero-bandwidth.
    ///
    /// NOTE (2026-07-02, `peak_bw_probe` multi-tier finding): despite the
    /// field name this is the DRAM-bound large-buffer PLATEAU, i.e. the
    /// correct roofline denominator — never the small-buffer cache-resident
    /// ceiling. Prior to the multi-tier fix, `peak_bw_probe` folded over a
    /// `GpuCapability::l2_cache_mb + infinity_cache_mb` threshold that is
    /// wrong for unified-memory APUs (gfx1151/Strix Halo), so a committed row
    /// authored against that probe could actually hold the CACHE tier's BW
    /// mislabeled as DRAM peak. See [`detect_bw_tiers`] and `cache_bw_gbps`.
    pub peak_bw_gbps: Option<f64>,
    /// Small-buffer (cache-resident) bandwidth ceiling in GB/s, when the
    /// swept curve shows a distinct tier below the DRAM plateau (see
    /// [`detect_bw_tiers`]). `None` for a chip whose sweep never showed a
    /// meaningful cache tier (e.g. a dGPU where even the smallest swept
    /// buffer is already DRAM-bound) — NOT the same as "not yet measured";
    /// `for_unprofiled` also returns `None` for that latter reason.
    pub cache_bw_gbps: Option<f64>,
    /// Empirically-inferred effective cache size in MiB: the largest swept
    /// buffer size that still landed in the cache tier (i.e. the last row
    /// before the curve transitions to the DRAM plateau). This EMPIRICALLY
    /// CORRECTS `GpuCapability::l2_cache_mb + infinity_cache_mb`, which is
    /// wrong for unified-memory APUs — see [`detect_bw_tiers`] module docs.
    /// `None` alongside `cache_bw_gbps == None`.
    pub effective_cache_mib: Option<f64>,
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
            // `static_capability` sets vram_mb=0 as its "unknown" sentinel
            // (no sysfs/HIP query performed) — withhold rather than commit
            // to a bogus zero-byte card.
            vram_mb: None,
            mem_latency_ns: None,
            peak_bw_gbps: None,
            cache_bw_gbps: None,
            effective_cache_mib: None,
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
        if let Some(vram_mb) = self.vram_mb {
            if vram_mb != live.vram_mb {
                mismatches.push(format!(
                    "vram_mb: committed={} live={}",
                    vram_mb, live.vram_mb
                ));
            }
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
            "vram_mb": self.vram_mb,
            "mem_latency_ns": self.mem_latency_ns,
            "peak_bw_gbps": self.peak_bw_gbps,
            "cache_bw_gbps": self.cache_bw_gbps,
            "effective_cache_mib": self.effective_cache_mib,
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
        // Optional: absent (older committed rows / for_unprofiled) => None,
        // i.e. verify_live withholds the VRAM check rather than comparing
        // against a fabricated value.
        let vram_mb = v["vram_mb"].as_u64().map(|x| x as u32);
        let mem_latency_ns = v["mem_latency_ns"].as_f64();
        let peak_bw_gbps = v["peak_bw_gbps"].as_f64();
        // Optional, and absent on any committed row authored before the
        // multi-tier bandwidth finding (2026-07-02) — withhold (None) rather
        // than fabricate a cache-tier figure for an older row.
        let cache_bw_gbps = v["cache_bw_gbps"].as_f64();
        let effective_cache_mib = v["effective_cache_mib"].as_f64();
        let intrinsics = v["intrinsics"].clone();
        let provenance = ChipProfileProvenance::from_json(&v["provenance"]);

        Ok(Self {
            arch,
            cu_count,
            vgprs_per_simd,
            max_waves_per_simd,
            lds_bytes_per_cu,
            wavefront_size,
            vram_mb,
            mem_latency_ns,
            peak_bw_gbps,
            cache_bw_gbps,
            effective_cache_mib,
            intrinsics,
            provenance,
        })
    }
}

/// Minimum pointer-chase buffer size in **bytes** for `arch` such that the
/// working set is genuinely DRAM-resident: >= 4x the on-chip (L2 + Infinity
/// Cache) footprint from `arch_spec`, floored at 128 MiB.
///
/// Issue #490: the microbench shipped a FIXED 128 MiB buffer, which fits inside
/// the Infinity Cache on every IC-bearing RDNA part. Sizing to >= 4x(L2+IC)
/// spills the working set well past the last cache level so each hop is a true
/// DRAM round trip. The 128 MiB floor keeps the historically proven size on
/// no-IC parts (gfx1010, gfx1151).
///
/// PURE + NO-GPU: reads only `GpuCapability::static_capability(arch)`'s
/// arch_spec cache constants — deterministic, unit-testable without a device.
pub fn pointer_chase_buffer_bytes(arch: &str) -> usize {
    const MIB: usize = 1024 * 1024;
    const FLOOR_MIB: f64 = 128.0;
    const SAFETY_MULT: f64 = 4.0;
    let cap = GpuCapability::static_capability(arch);
    let on_chip_mib = (cap.l2_cache_mb + cap.infinity_cache_mb) as f64;
    let target_mib = (SAFETY_MULT * on_chip_mib).ceil().max(FLOOR_MIB);
    target_mib as usize * MIB
}

/// Cache-residency guard for a pointer-chase latency measurement (issue #490).
///
/// The bare `[50, 2000) ns` plausibility assert CANNOT prove the working set
/// reached DRAM — a fully cache-resident chase still lands in that window. A
/// genuine DRAM round trip has per-hop latency substantially HIGHER than a
/// deliberately cache-resident chase on the same device. If the DRAM-sized
/// chase is not at least `min_dram_over_cache_ratio`x the cache-resident chase,
/// the large buffer never left cache and the number is cache-DEFLATED -> `Err`
/// (WITHHELD, never shipped as a clean DRAM latency).
///
/// PURE + NO-GPU: operates on two already-measured latencies.
pub fn check_residency(
    cache_resident_ns: f64,
    candidate_dram_ns: f64,
    min_dram_over_cache_ratio: f64,
) -> Result<(), String> {
    if !cache_resident_ns.is_finite() || cache_resident_ns <= 0.0 {
        return Err(format!(
            "check_residency: cache_resident_ns must be finite and positive, got {cache_resident_ns}"
        ));
    }
    if !candidate_dram_ns.is_finite() || candidate_dram_ns <= 0.0 {
        return Err(format!(
            "check_residency: candidate_dram_ns must be finite and positive, got {candidate_dram_ns}"
        ));
    }
    if !min_dram_over_cache_ratio.is_finite() || min_dram_over_cache_ratio <= 1.0 {
        return Err(format!(
            "check_residency: min_dram_over_cache_ratio must be finite and > 1.0, got {min_dram_over_cache_ratio}"
        ));
    }
    let required = cache_resident_ns * min_dram_over_cache_ratio;
    if candidate_dram_ns >= required {
        Ok(())
    } else {
        Err(format!(
            "check_residency: DRAM-sized chase {candidate_dram_ns:.3} ns is not >= \
             {min_dram_over_cache_ratio:.2}x the cache-resident chase {cache_resident_ns:.3} ns \
             (required >= {required:.3} ns) — the 'DRAM' buffer is still CACHE-RESIDENT and this \
             latency is cache-DEFLATED (issue #490). WITHHELD, not shipped."
        ))
    }
}

/// One (buffer-size-in-MiB, achieved-GB/s) sample from a bandwidth sweep
/// (e.g. `peak_bw_probe`'s `SweepRow`), ordered ascending by `mib`. Used only
/// by [`detect_bw_tiers`] — kept as a small standalone struct (rather than
/// depending on the example's private `SweepRow`) so both a library
/// consumer and the `peak_bw_probe` example can share one tier-detection
/// implementation instead of hand-rolling separate copies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BwSweepPoint {
    pub mib: f64,
    pub gbps: f64,
}

/// Output of [`detect_bw_tiers`]: the DRAM-bound plateau bandwidth (the
/// correct roofline denominator, i.e. what `ChipProfile::peak_bw_gbps`
/// should hold) plus, when the swept curve shows a distinct small-size
/// cache-resident tier, that tier's ceiling BW and its empirical size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BwTiers {
    /// DRAM-bound plateau bandwidth in GB/s — the BW at the largest swept
    /// size, once verified that the top of the curve has actually
    /// converged (see module docs on the `plateau_tolerance_pct` check).
    pub dram_peak_gbps: f64,
    /// Max achieved GB/s among the sweep rows below the DRAM plateau's
    /// start. `None` when the DRAM plateau covers the whole swept range
    /// (no distinct cache tier observed).
    pub cache_bw_gbps: Option<f64>,
    /// MiB of the largest swept size that is still cache-resident (i.e. the
    /// row immediately below the DRAM plateau's start) — the empirical
    /// cache-size estimate. `None` alongside `cache_bw_gbps == None`.
    pub effective_cache_mib: Option<f64>,
}

/// Detect the DRAM-bound plateau (and optional cache tier) in a bandwidth
/// sweep, WITHOUT relying on `GpuCapability::l2_cache_mb`/`infinity_cache_mb`
/// — those arch_spec cache-footprint constants are correct for discrete GPUs
/// but WRONG for unified-memory APUs (Strix Halo gfx1151 measured
/// `peak_bw_gbps=952` under the old L2+IC-threshold fold, which was actually
/// the *cache* tier; DRAM is ~230 GB/s — see `tests/chip-profiles/gfx1201.json`
/// provenance and `peak_bw_probe.rs`).
///
/// Algorithm: `points` MUST be sorted strictly ascending by `mib`. Starting
/// from the largest swept size, greedily extend a "DRAM plateau" backward
/// while each additional point's GB/s agrees with the largest size's GB/s
/// within `plateau_tolerance_pct` (empirically ~2-3%: the gfx1151 64/128/
/// 256/512 MiB rows — 229/230/231/232 GB/s — agree within ~1.3%, while the
/// jump from the 16 MiB cache row (952 GB/s) is >300%). The plateau must
/// span >= 2 points; if even the two largest swept sizes disagree by more
/// than the tolerance, the curve hasn't converged (still climbing/monotonic
/// at the top of the sweep) and this fails loud rather than mis-reporting
/// the top row as a trustworthy DRAM peak.
///
/// `dram_peak_gbps` is the GB/s at the largest swept size (once the plateau
/// check above passes). `cache_bw_gbps` is the max GB/s among the rows
/// below the plateau's start (`None` if the plateau starts at row 0, i.e.
/// no distinct cache tier showed up in the sweep at all — e.g. a dGPU).
/// `effective_cache_mib` is the MiB of the last cache-resident row before
/// the plateau begins.
pub fn detect_bw_tiers(
    points: &[BwSweepPoint],
    plateau_tolerance_pct: f64,
) -> Result<BwTiers, String> {
    if points.len() < 2 {
        return Err(format!(
            "detect_bw_tiers: need >= 2 sweep points to detect a DRAM plateau, got {}",
            points.len()
        ));
    }
    for w in points.windows(2) {
        // `partial_cmp` (not a bare `<`/`>=`) so a NaN `mib` (which cannot
        // compare `Less` to anything) is correctly rejected as non-ascending
        // rather than silently accepted.
        if w[0].mib.partial_cmp(&w[1].mib) != Some(std::cmp::Ordering::Less) {
            return Err(format!(
                "detect_bw_tiers: points must be strictly ascending by mib, got {} then {}",
                w[0].mib, w[1].mib
            ));
        }
    }

    let n = points.len();
    let last = points[n - 1];

    let mut start = n - 1;
    for i in (0..n - 1).rev() {
        let diff_pct = (points[i].gbps - last.gbps).abs() / last.gbps * 100.0;
        if diff_pct <= plateau_tolerance_pct {
            start = i;
        } else {
            break;
        }
    }

    if start == n - 1 {
        return Err(format!(
            "detect_bw_tiers: no DRAM plateau found — the two largest swept sizes \
             ({} MiB={:.1} GB/s, {} MiB={:.1} GB/s) differ by more than \
             plateau_tolerance_pct={plateau_tolerance_pct}% — the curve is still \
             climbing/monotonic at the top of the sweep. Extend the sweep to larger \
             sizes before trusting a DRAM peak_bw figure; do NOT fall back to \
             reporting the top row as peak.",
            points[n - 2].mib,
            points[n - 2].gbps,
            last.mib,
            last.gbps
        ));
    }

    let dram_peak_gbps = last.gbps;

    let (cache_bw_gbps, effective_cache_mib) = if start == 0 {
        (None, None)
    } else {
        let cache_rows = &points[..start];
        let cache_peak = cache_rows.iter().copied().fold(cache_rows[0], |best, p| {
            if p.gbps > best.gbps {
                p
            } else {
                best
            }
        });
        // A real cache tier must be FASTER than the DRAM plateau — that's
        // the whole point of it being a cache. A sub-plateau row that is
        // merely SLOWER (e.g. small-buffer ramp-up noise on a chip with no
        // Infinity Cache, like RDNA1/gfx1010: cache-row peak 351.8 GB/s <
        // DRAM plateau 384.1 GB/s) is not a meaningful distinct tier — report
        // it as withheld (None) rather than fabricating a "cache tier" out of
        // a row that's actually just slower than DRAM.
        if cache_peak.gbps > dram_peak_gbps {
            (Some(cache_peak.gbps), Some(points[start - 1].mib))
        } else {
            (None, None)
        }
    };

    Ok(BwTiers {
        dram_peak_gbps,
        cache_bw_gbps,
        effective_cache_mib,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // Explicit import: the parent module's `use crate::profiler::GpuCapability;`
    // is a private alias, so we bind it directly here rather than rely on the
    // `use super::*` glob re-exporting it into this child module (issue #490 /
    // adversarial-review fix).
    use crate::profiler::GpuCapability;

    #[test]
    fn pointer_chase_buffer_exceeds_4x_onchip_cache() {
        // Every arch's chase buffer must clear 4x its (L2 + Infinity Cache)
        // footprint so the working set is genuinely DRAM-resident (issue #490).
        for arch in ["gfx1010", "gfx1030", "gfx1100", "gfx1151", "gfx1201"] {
            let cap = GpuCapability::static_capability(arch);
            let onchip_bytes =
                ((cap.l2_cache_mb + cap.infinity_cache_mb) as f64 * 1024.0 * 1024.0) as usize;
            let buf = pointer_chase_buffer_bytes(arch);
            assert!(
                buf >= 4 * onchip_bytes,
                "{arch}: pointer-chase buffer {buf} must be >= 4x on-chip cache {}",
                4 * onchip_bytes
            );
        }
    }

    #[test]
    fn pointer_chase_buffer_rdna2_exceeds_512mib() {
        // gfx1030 (RDNA2): L2 4 MiB + IC 128 MiB -> 4x(132) = 528 MiB, clearing
        // the spec's ">= 512 MiB on RDNA2" floor. The OLD fixed 128 MiB buffer
        // fit ENTIRELY inside the 128 MiB IC — the exact #490 deflation.
        let buf = pointer_chase_buffer_bytes("gfx1030");
        assert_eq!(buf, 528 * 1024 * 1024, "gfx1030 = 4*(4+128) = 528 MiB");
        assert!(
            buf >= 512 * 1024 * 1024,
            "RDNA2 buffer must clear the 512 MiB floor, got {buf}"
        );
        assert!(
            buf > 128 * 1024 * 1024,
            "must exceed the deflated 128 MiB buffer that fit inside the 128 MiB IC"
        );
    }

    #[test]
    fn pointer_chase_buffer_floors_at_128mib_for_no_ic_parts() {
        // gfx1010 (RDNA1, no IC) and gfx1151 (RDNA3.5, no discrete IC) fall
        // below the floor, so the 128 MiB minimum applies — these rows were
        // never cache-deflated (128 MiB already >> their L2).
        assert_eq!(pointer_chase_buffer_bytes("gfx1010"), 128 * 1024 * 1024);
        assert_eq!(pointer_chase_buffer_bytes("gfx1151"), 128 * 1024 * 1024);
    }

    #[test]
    fn pointer_chase_buffer_rdna4_and_rdna3() {
        // gfx1201 (RDNA4): 4x(4+64) = 272 MiB; gfx1100 (RDNA3): 4x(6+96) = 408 MiB.
        assert_eq!(pointer_chase_buffer_bytes("gfx1201"), 272 * 1024 * 1024);
        assert_eq!(pointer_chase_buffer_bytes("gfx1100"), 408 * 1024 * 1024);
    }

    #[test]
    fn check_residency_flags_cache_deflated_gfx1030() {
        // The #490 signature: the shipped "DRAM" latency is barely above a
        // genuine L2-hit reference — within noise, NOT a real >1.5x DRAM step.
        let result = check_residency(140.0, 148.035, 1.5);
        assert!(
            result.is_err(),
            "a within-noise DRAM/cache ratio must fail, got {result:?}"
        );
        assert!(result.unwrap_err().contains("cache-DEFLATED"));
    }

    #[test]
    fn check_residency_passes_genuine_dram() {
        assert!(check_residency(140.0, 300.0, 1.5).is_ok());
    }

    #[test]
    fn check_residency_boundary_is_inclusive() {
        assert!(check_residency(100.0, 150.0, 1.5).is_ok());
        assert!(check_residency(100.0, 149.999, 1.5).is_err());
    }

    #[test]
    fn check_residency_rejects_bad_inputs() {
        assert!(check_residency(0.0, 300.0, 1.5).is_err()); // non-positive cache
        assert!(check_residency(140.0, f64::NAN, 1.5).is_err()); // non-finite dram
        assert!(check_residency(140.0, 300.0, 1.0).is_err()); // ratio must be > 1.0
    }

    #[test]
    fn load_committed_gfx1201() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        assert_eq!(profile.wavefront_size, 32, "gfx1201 is wave32-native RDNA4");
        assert_eq!(
            profile.cu_count, 64,
            "R9700 gfx1201 = 64 CU (full Navi 48; verified via rocminfo on the \
             live R9700 and the local RX 9070 XT). 56 is the RX 9070 non-XT SKU."
        );
        assert_eq!(profile.vgprs_per_simd, 1536, "RDNA4 VGPR file per SIMD");
        assert_eq!(
            profile.lds_bytes_per_cu, 65536,
            "64 KiB per CU — matches profiler.rs::arch_spec (the single source \
             of truth for every RDNA1-4 arch), i.e. what ROCm/HIP reports as the \
             per-CU LDS partition. NOT 128 KiB (131072): that's the physical LDS \
             size per WGP, shared by the 2 CUs in the pair, not per-CU."
        );
        assert_eq!(
            profile.vram_mb,
            Some(32768),
            "R9700 (Radeon AI PRO, gfx1201 hiptrx fleet card) = 32GB"
        );
        assert_eq!(
            profile.peak_bw_gbps,
            Some(611.8),
            "DRAM-plateau peak, per peak_bw_probe multi-tier detection"
        );
        assert_eq!(
            profile.cache_bw_gbps,
            Some(1352.4),
            "R9700's 64MB Infinity Cache DOES show a real cache tier in a wider \
             re-sweep (2026-07-02 correction) — the cache-resident row's peak BW \
             (1352.4 GB/s) clears the DRAM plateau (611.8 GB/s), so it is a genuine \
             tier per the detect_bw_tiers 'cache peak must exceed DRAM plateau' guard, \
             not the earlier no-GPU deduction that withheld it as null"
        );
        assert_eq!(profile.effective_cache_mib, Some(64.0));
    }

    /// Load EVERY committed `tests/chip-profiles/*.json` row (not just
    /// gfx1201) and assert each parses cleanly and carries its expected
    /// arch-identifying static/measured fields — a regression guard so a
    /// future committed row can't silently bit-rot (bad JSON, wrong
    /// cu_count, a withheld `mem_latency_ns`, or a cache tier that
    /// mysteriously vanishes/appears) without a test noticing. Deliberately
    /// enumerates the directory (rather than hardcoding an arch list) so a
    /// newly-added `tests/chip-profiles/<gfx>.json` MUST also get an arm
    /// added to `expected` below — an unrecognized file panics loud instead
    /// of silently passing unchecked.
    #[test]
    fn load_all_committed_chip_profiles() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/chip-profiles");
        let mut archs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("failed to read {dir:?}: {e}"))
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                name.strip_suffix(".json").map(str::to_string)
            })
            .collect();
        archs.sort();

        assert!(
            !archs.is_empty(),
            "tests/chip-profiles/*.json must contain at least one committed row"
        );

        for arch in &archs {
            let profile = ChipProfile::load_committed(arch)
                .unwrap_or_else(|e| panic!("tests/chip-profiles/{arch}.json must parse: {e}"));
            assert_eq!(&profile.arch, arch, "arch field must match filename stem");

            // (expected cu_count, expected mem_latency_ns, expected cache_bw_gbps)
            // mem_latency_ns RE-MEASURED 2026-07-02 on hipx warm force-high with
            // the per-arch pointer-chase buffer (issue #490 CLOSED): the three
            // Infinity-Cache parts (gfx1030/1100/1201) were previously WITHHELD
            // because the old 128 MiB buffer fit inside their IC (cache-DEFLATED);
            // now un-WITHHELD with the real DRAM latency (median of 3, residency
            // guard passed). gfx1010/gfx1151 (no IC deflation) unchanged.
            let (expected_cu, expected_latency, expected_cache_bw): (u32, Option<f64>, Option<f64>) =
                match arch.as_str() {
                    "gfx1010" => (40, Some(276.058), None),
                    "gfx1030" => (80, Some(248.744), Some(1586.5)),
                    "gfx1100" => (96, Some(214.124), Some(1778.8)),
                    "gfx1151" => (40, Some(219.932), Some(952.5)),
                    "gfx1201" => (64, Some(237.999), Some(1352.4)),
                    other => panic!(
                        "load_all_committed_chip_profiles: unrecognized committed row {other:?} — \
                         add an expected-value arm here"
                    ),
                };

            assert_eq!(profile.cu_count, expected_cu, "{arch}: cu_count");
            assert_eq!(
                profile.mem_latency_ns, expected_latency,
                "{arch}: mem_latency_ns (WITHHELD/None on IC parts per issue #490 until re-measured)"
            );
            assert_eq!(
                profile.cache_bw_gbps, expected_cache_bw,
                "{arch}: cache_bw_gbps"
            );
            assert_eq!(
                profile.effective_cache_mib.is_some(),
                expected_cache_bw.is_some(),
                "{arch}: effective_cache_mib must be Some iff cache_bw_gbps is Some"
            );
        }
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
    fn verify_live_fails_loud_on_vram_mismatch() {
        let profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        // Every static field matches EXCEPT vram — as if the same die were
        // plugged into a different-VRAM SKU (e.g. RX 9070 XT 16GB card
        // reporting under a committed row meant for the 32GB R9700).
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
            vram_mb: 16384,
        };
        let result = profile.verify_live(&live);
        assert!(
            result.is_err(),
            "verify_live must fail loud on a vram_mb mismatch, got Ok"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("vram_mb"),
            "error message should name the mismatched field: {msg}"
        );
    }

    #[test]
    fn verify_live_withholds_vram_check_when_committed_row_lacks_it() {
        // Simulate an older-format committed row (pre-vram_mb field) by
        // round-tripping through JSON with the key stripped: `from_json`
        // must leave `vram_mb: None`, and `verify_live` must NOT fail loud
        // on a live vram_mb it was never told to expect.
        let mut json = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load")
            .to_json();
        json.as_object_mut().unwrap().remove("vram_mb");
        let profile = ChipProfile::from_json(&json).expect("must parse without vram_mb");
        assert!(profile.vram_mb.is_none());

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
            vram_mb: 999, // any value — must be ignored, not compared
        };
        assert!(
            profile.verify_live(&live).is_ok(),
            "verify_live must withhold (not fail) the vram check when the \
             committed row doesn't carry vram_mb"
        );
    }

    #[test]
    fn verify_live_fails_loud_on_wavefront_mismatch() {
        let mut profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        // Corrupt only the committed wavefront_size (as if the row were
        // authored against the wrong arch's wave width) — every other field,
        // including live.arch, stays gfx1201/wave32-correct.
        profile.wavefront_size = 64;
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
        let result = profile.verify_live(&live);
        assert!(
            result.is_err(),
            "verify_live must fail loud on a wavefront_size mismatch, got Ok"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("wavefront_size"),
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
        assert!(
            profile.cache_bw_gbps.is_none(),
            "unprofiled arch must withhold cache_bw (never measured)"
        );
        assert!(
            profile.effective_cache_mib.is_none(),
            "unprofiled arch must withhold effective_cache_mib (never measured)"
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

    #[test]
    fn round_trip_json_with_cache_tier() {
        // Round-trip a profile that DOES carry a cache tier (unlike the
        // committed gfx1201/R9700 row, which is dGPU-cacheless) — covers the
        // `Some`/`Some` branch of the new fields' serde path.
        let mut profile = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        profile.cache_bw_gbps = Some(952.0);
        profile.effective_cache_mib = Some(16.0);
        let json = profile.to_json();
        let reloaded = ChipProfile::from_json(&json).expect("round-trip JSON must parse");
        assert_eq!(profile, reloaded);
        assert_eq!(reloaded.cache_bw_gbps, Some(952.0));
        assert_eq!(reloaded.effective_cache_mib, Some(16.0));
    }

    #[test]
    fn from_json_missing_cache_fields_withholds() {
        // Older committed rows authored before the multi-tier fix don't
        // carry cache_bw_gbps/effective_cache_mib at all — from_json must
        // withhold (None), not error or fabricate a value.
        let mut json = ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load")
            .to_json();
        json.as_object_mut().unwrap().remove("cache_bw_gbps");
        json.as_object_mut().unwrap().remove("effective_cache_mib");
        let profile = ChipProfile::from_json(&json)
            .expect("must parse without cache_bw_gbps/effective_cache_mib");
        assert!(profile.cache_bw_gbps.is_none());
        assert!(profile.effective_cache_mib.is_none());
    }

    /// Empirical gfx1151 (Strix Halo) `peak_bw_probe` sweep (multi-tier
    /// bandwidth finding, 2026-07-02): 1MiB=593, 4MiB=799, 16MiB=952
    /// (cache-resident tier), 64/128/256/512 MiB = 229/230/231/232
    /// (DRAM-bound plateau). The OLD fold-max-over-arch_spec-threshold logic
    /// reported `peak_bw_gbps=952` — the cache tier, not DRAM — because
    /// `GpuCapability::l2_cache_mb + infinity_cache_mb` (derived from
    /// `arch_spec`) is wrong for this unified-memory APU. See
    /// `tests/chip-profiles/gfx1201.json`'s provenance note
    /// ("gfx1151=950 (SUSPECT...)").
    #[test]
    fn detect_bw_tiers_gfx1151_curve() {
        let points = [
            BwSweepPoint {
                mib: 1.0,
                gbps: 593.0,
            },
            BwSweepPoint {
                mib: 4.0,
                gbps: 799.0,
            },
            BwSweepPoint {
                mib: 16.0,
                gbps: 952.0,
            },
            BwSweepPoint {
                mib: 64.0,
                gbps: 229.0,
            },
            BwSweepPoint {
                mib: 128.0,
                gbps: 230.0,
            },
            BwSweepPoint {
                mib: 256.0,
                gbps: 231.0,
            },
            BwSweepPoint {
                mib: 512.0,
                gbps: 232.0,
            },
        ];
        let tiers = detect_bw_tiers(&points, 3.0).expect("gfx1151 curve must yield a DRAM plateau");
        assert!(
            (tiers.dram_peak_gbps - 232.0).abs() < 1e-9,
            "DRAM peak must be the 512 MiB plateau BW (232), NOT the cache-tier max \
             (952) that the old threshold-fold mis-reported: got {}",
            tiers.dram_peak_gbps
        );
        assert_eq!(
            tiers.cache_bw_gbps,
            Some(952.0),
            "cache tier ceiling = max GB/s among the sub-plateau rows (16 MiB)"
        );
        assert_eq!(
            tiers.effective_cache_mib,
            Some(16.0),
            "effective cache size = last cache-resident sweep row (16 MiB) before the \
             DRAM plateau begins (64 MiB)"
        );
    }

    #[test]
    fn detect_bw_tiers_monotonic_curve_fails_loud() {
        // A curve that is still climbing at the top of the sweep (no two
        // largest sizes agree) must fail loud, not silently report the top
        // row as "peak" — HAZARD requirement #6.
        let points = [
            BwSweepPoint {
                mib: 1.0,
                gbps: 100.0,
            },
            BwSweepPoint {
                mib: 4.0,
                gbps: 200.0,
            },
            BwSweepPoint {
                mib: 16.0,
                gbps: 400.0,
            },
        ];
        let result = detect_bw_tiers(&points, 3.0);
        assert!(
            result.is_err(),
            "monotonic/no-plateau curve must fail loud, got {result:?}"
        );
    }

    #[test]
    fn detect_bw_tiers_no_cache_tier_for_dgpu() {
        // A dGPU-style curve where every swept size is already DRAM-bound
        // (no small-size cache advantage large enough to show up) — the
        // whole range is one plateau, so cache_bw_gbps/effective_cache_mib
        // must both withhold (None), not fabricate a value.
        let points = [
            BwSweepPoint {
                mib: 64.0,
                gbps: 610.0,
            },
            BwSweepPoint {
                mib: 128.0,
                gbps: 611.0,
            },
            BwSweepPoint {
                mib: 256.0,
                gbps: 611.8,
            },
            BwSweepPoint {
                mib: 512.0,
                gbps: 612.0,
            },
        ];
        let tiers = detect_bw_tiers(&points, 3.0).expect("all-plateau curve must still succeed");
        assert!((tiers.dram_peak_gbps - 612.0).abs() < 1e-9);
        assert!(tiers.cache_bw_gbps.is_none());
        assert!(tiers.effective_cache_mib.is_none());
    }

    /// Empirical gfx1010 (RDNA1, RX 5700 XT) `peak_bw_probe` sweep (2026-07-02
    /// cross-arch campaign): a sub-plateau row (351.8 GB/s) does NOT clear the
    /// DRAM plateau (384.1 GB/s) — RDNA1 has no Infinity Cache
    /// (`profiler.rs::arch_spec` sets `infinity_cache_mb=0.0` for this family),
    /// so the "cache row" is just small-buffer ramp-up noise, not a real
    /// cache-resident bandwidth advantage. `detect_bw_tiers` must withhold
    /// `cache_bw_gbps`/`effective_cache_mib` (None) here, NOT report 351.8 as
    /// a fabricated "cache tier" — a real cache tier is by definition FASTER
    /// than DRAM. See `tests/chip-profiles/gfx1010.json`'s provenance note.
    #[test]
    fn detect_bw_tiers_gfx1010_sub_dram_row_is_not_a_cache_tier() {
        let points = [
            BwSweepPoint {
                mib: 16.0,
                gbps: 351.8,
            },
            BwSweepPoint {
                mib: 128.0,
                gbps: 384.0,
            },
            BwSweepPoint {
                mib: 256.0,
                gbps: 384.1,
            },
        ];
        let tiers = detect_bw_tiers(&points, 3.0).expect("gfx1010 curve must yield a DRAM plateau");
        assert!(
            (tiers.dram_peak_gbps - 384.1).abs() < 1e-9,
            "DRAM peak must be the 256 MiB plateau BW (384.1): got {}",
            tiers.dram_peak_gbps
        );
        assert_eq!(
            tiers.cache_bw_gbps, None,
            "the 16 MiB row (351.8 GB/s) is SLOWER than the DRAM plateau (384.1 GB/s) \
             — not a real cache tier, must be withheld rather than fabricated"
        );
        assert_eq!(tiers.effective_cache_mib, None);
    }

    #[test]
    fn detect_bw_tiers_rejects_too_few_points() {
        let points = [BwSweepPoint {
            mib: 1.0,
            gbps: 100.0,
        }];
        assert!(detect_bw_tiers(&points, 3.0).is_err());
    }

    #[test]
    fn detect_bw_tiers_rejects_unsorted_points() {
        let points = [
            BwSweepPoint {
                mib: 16.0,
                gbps: 400.0,
            },
            BwSweepPoint {
                mib: 4.0,
                gbps: 200.0,
            },
        ];
        let result = detect_bw_tiers(&points, 3.0);
        assert!(
            result.is_err(),
            "unsorted input must fail loud, not silently mis-sort"
        );
    }
}
