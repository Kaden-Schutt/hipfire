// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Static, no-GPU 3-bound-class roofline model — BW, VALU-issue, and
//! DRAM-latency (Little's Law) — composed from [`crate::isa_histogram::IsaHistogram`],
//! [`crate::profiler::KernelProfile`], and [`crate::chip_profile::ChipProfile`].
//!
//! This is a diagnostic heuristic, not a certified perf predictor: with
//! `achieved_bw: None` (the `--self-check` no-GPU path) it can only reason
//! from static ISA shape + occupancy, never from measured throughput. The
//! dynamic half (Task 9, GPU-required) feeds a real `achieved_bw` and raises
//! `trust_score` accordingly.

use crate::chip_profile::ChipProfile;
use crate::isa_histogram::IsaHistogram;
use crate::profiler::KernelProfile;

/// Which of the three closed-form bounds is tightest for a given
/// kernel/hardware pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundClass {
    /// The kernel is (or would be, at its current shape) limited by DRAM
    /// bandwidth — cutting bytes moved is the only lever.
    Bandwidth,
    /// The kernel's static ISA shape is dominated by VALU work (scalar
    /// dequant unpack/convert/accumulate chains) relative to its memory
    /// traffic — issue-slot pressure, not bytes, is the ceiling.
    ValuIssue,
    /// Little's-Law in-flight-bytes shortfall: too few waves in flight
    /// (occupancy) to hide DRAM latency at the given VMEM footprint.
    Latency,
}

/// A single kernel's static 3-bound-class roofline verdict. See the module
/// docs for the "diagnostic heuristic, not a certified predictor" caveat.
#[derive(Debug, Clone, PartialEq)]
pub struct Roofline {
    /// BW-bound score in `[0.0, 1.0]` — `achieved_bw / peak_bw_gbps` when
    /// both are known, else `0.0` (WITHHELD, never a spurious zero-means-
    /// "not BW bound" claim dressed up as a measurement).
    pub bw: f64,
    /// VALU-issue-bound score in `[0.0, 1.0]` — derived from
    /// `hist.vmem_valu_ratio` (VALU share of the vmem+valu instruction
    /// mix); `0.0` when the kernel has neither VALU nor VMEM instructions
    /// at all (degenerate).
    pub valu_issue: f64,
    /// Latency-bound score — Little's Law: `required_in_flight_bytes =
    /// peak_bw * mem_latency` vs. `available_in_flight_bytes` (this
    /// kernel's occupancy × its `global_load_b128` VMEM footprint proxy).
    /// `0.0` (WITHHELD) whenever `chip.peak_bw_gbps`/`chip.mem_latency_ns`
    /// is `None` (unprofiled arch — mem_latency is MEASURED, never
    /// estimated) rather than a fabricated non-zero score.
    pub latency: f64,
    /// The tightest (highest-scoring) of the three bounds above.
    pub binding: BoundClass,
    /// Ratio of the binding score to the runner-up score — how confidently
    /// separated the verdict is. `f64::INFINITY` when the runner-up is
    /// `0.0` (no ambiguity at all).
    pub second_wall_margin: f64,
    /// One-line human verdict, always carrying a "fix hint" appropriate to
    /// `binding`.
    pub verdict: String,
    /// How much to trust this verdict: `1.0` when `achieved_bw` was
    /// supplied (a real measurement backs the BW score), `0.5` for a fully
    /// static (`achieved_bw: None`) estimate. Task 6's ledger refines this
    /// further from an analytic-vs-rocprof-counter cross-check — this is
    /// just the Task 5 static-vs-measured floor.
    pub trust_score: f64,
}

impl Roofline {
    /// Compute the 3-bound-class roofline for one kernel. `achieved_bw` is
    /// `Some(gbps)` on the dynamic (GPU-measured or byte-model-analytic)
    /// path, `None` on the static `--self-check` (no-GPU) path.
    pub fn analyze(
        hist: &IsaHistogram,
        kprofile: &KernelProfile,
        chip: &ChipProfile,
        achieved_bw: Option<f64>,
    ) -> Self {
        let bw = bw_score(achieved_bw, chip.peak_bw_gbps);
        let valu_issue = valu_issue_score(hist);
        let latency = latency_score(hist, kprofile, chip);

        let scores = [
            (BoundClass::Bandwidth, bw),
            (BoundClass::ValuIssue, valu_issue),
            (BoundClass::Latency, latency),
        ];
        // Highest score wins; ties keep the first (Bandwidth > ValuIssue >
        // Latency) — an intentional bias toward the "hardest to fix"
        // explanation when scores are exactly equal (e.g. all-zero
        // degenerate input).
        let (binding, top) =
            scores
                .iter()
                .copied()
                .fold((BoundClass::Bandwidth, f64::MIN), |acc, cur| {
                    if cur.1 > acc.1 {
                        cur
                    } else {
                        acc
                    }
                });
        let runner_up = scores
            .iter()
            .filter(|(b, _)| *b != binding)
            .map(|(_, s)| *s)
            .fold(f64::MIN, f64::max);
        let second_wall_margin = if runner_up > 0.0 {
            top / runner_up
        } else {
            f64::INFINITY
        };

        let verdict = verdict_for(binding, hist, top);
        let trust_score = if achieved_bw.is_some() { 1.0 } else { 0.5 };

        Self {
            bw,
            valu_issue,
            latency,
            binding,
            second_wall_margin,
            verdict,
            trust_score,
        }
    }
}

/// `achieved_bw / peak_bw_gbps`, clamped to `[0.0, 1.0]`. `0.0` (WITHHELD)
/// when either input is unknown.
fn bw_score(achieved_bw: Option<f64>, peak_bw_gbps: Option<f64>) -> f64 {
    match (achieved_bw, peak_bw_gbps) {
        (Some(achieved), Some(peak)) if peak > 0.0 => (achieved / peak).clamp(0.0, 1.0),
        _ => 0.0,
    }
}

/// VALU share of the vmem+valu instruction mix:
/// `valu / (valu + vmem) == 1 / (1 + vmem_valu_ratio)`. A kernel with zero
/// VMEM traffic but nonzero dequant/compute opcodes is treated as maximally
/// VALU-issue-bound (`1.0`); a kernel with neither is `0.0` (degenerate).
fn valu_issue_score(hist: &IsaHistogram) -> f64 {
    if hist.vmem_valu_ratio > 0.0 {
        (1.0 / (1.0 + hist.vmem_valu_ratio)).clamp(0.0, 1.0)
    } else if hist.v_bfe + hist.v_cvt + hist.f32_fma + hist.v_wmma > 0 {
        1.0
    } else {
        0.0
    }
}

/// Little's Law: bytes that must be simultaneously in flight to fully hide
/// DRAM latency at peak bandwidth, compared against what this kernel's
/// occupancy can actually keep in flight (using `global_load_b128` — 16B
/// per lane — as the per-wave VMEM footprint proxy, times `wavefront_size`
/// lanes, times waves-in-flight). `0.0` (WITHHELD) when the chip's
/// `peak_bw_gbps`/`mem_latency_ns` haven't been measured, or `0.0`
/// (genuinely not latency-bound) when the kernel issues no VMEM traffic at
/// all.
fn latency_score(hist: &IsaHistogram, kprofile: &KernelProfile, chip: &ChipProfile) -> f64 {
    let (peak_gbps, lat_ns) = match (chip.peak_bw_gbps, chip.mem_latency_ns) {
        (Some(p), Some(l)) if p > 0.0 => (p, l),
        _ => return 0.0,
    };
    let required_bytes = peak_gbps * 1e9 * (lat_ns * 1e-9);
    let bytes_per_wave = hist.global_load_b128 as f64 * 16.0 * chip.wavefront_size as f64;
    if bytes_per_wave <= 0.0 {
        return 0.0; // no VMEM traffic — cannot be latency-bound
    }
    // Approximation: `occupancy_waves` (per-SIMD, from KernelProfile) times
    // total CU count as a proxy for chip-wide waves-in-flight. Static-model
    // simplification — ChipProfile doesn't carry simds_per_cu, and this is
    // a diagnostic heuristic, not a cycle-accurate occupancy model.
    let waves_in_flight = kprofile.occupancy_waves as f64 * chip.cu_count as f64;
    let available_bytes = bytes_per_wave * waves_in_flight;
    if available_bytes <= 0.0 {
        return 1.0; // can't cover ANY of the required in-flight bytes
    }
    (required_bytes / available_bytes).clamp(0.0, 1.0)
}

fn verdict_for(binding: BoundClass, hist: &IsaHistogram, score: f64) -> String {
    let base = match binding {
        BoundClass::Bandwidth => format!(
            "BW-bound ({:.0}% of peak) — genuinely memory-bandwidth-limited; \
             fix hint: cut bytes moved (tighter quant / avoid re-reads), a \
             faster decode kernel alone won't help",
            score * 100.0
        ),
        BoundClass::ValuIssue => {
            if hist.v_dot4 == 0 && hist.v_wmma == 0 {
                format!(
                    "VALU-issue-bound ({:.0}% VALU share, zero v_dot4/v_wmma) — \
                     scalar dequant (v_bfe/v_cvt/f32-FMA chain) dominates issue \
                     slots; fix hint: a native dp4a/wmma decode-GEMV kernel would \
                     collapse this VALU chain",
                    score * 100.0
                )
            } else {
                format!(
                    "VALU-issue-bound ({:.0}% VALU share) — fix hint: reduce \
                     scalar/vector ALU work per byte moved",
                    score * 100.0
                )
            }
        }
        BoundClass::Latency => format!(
            "Latency-bound ({:.0}% Little's-Law in-flight-bytes shortfall) — \
             fix hint: raise occupancy (cut VGPR/LDS pressure) or restructure \
             to prefetch further ahead and hide DRAM latency",
            score * 100.0
        ),
    };
    // `private_segment_red` (register spill to the private/scratch segment,
    // sourced from the hsaco kernel descriptor — see `IsaHistogram::from_hsaco`)
    // caps occupancy independently of which bound class won, so it's surfaced
    // as an always-on suffix rather than folded into any one bound's score:
    // a BW- or VALU-bound verdict computed on a spilling kernel is still
    // pessimistically-biased (the spill is itself suppressing achievable
    // occupancy/throughput), and a spill fix is a standing fix hint
    // regardless of today's binding class.
    if hist.private_segment_red {
        format!(
            "{base} — ALSO register-spilled (private_segment_fixed_size > 0): \
             occupancy is artificially capped, so this verdict is pessimistically \
             biased until the spill is fixed"
        )
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_serialized_binds_valu_or_latency() {
        // "Dequant-heavy scalar GEMV" shape, matching the ACTUAL measured
        // histogram of the committed gfx1201 gate_up fixture
        // (`tests/kernel-fixtures/gfx1201/gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco`,
        // cross-checked via `clang-offload-bundler`+`llvm-objdump` outside
        // this test — see `real_fixture_binds_valu_issue_under_committed_chip`
        // below for the live from-disassembly version of this same shape):
        // lots of v_bfe/f32_fma/s_delay_alu, NO v_dot4, and VALU heavily
        // dominating the vmem+valu instruction mix (vmem=29, valu=281).
        let hist = IsaHistogram {
            v_bfe: 42,
            v_cvt: 56,
            f32_fma: 98,
            v_dot4: 0,
            v_wmma: 0,
            global_load_b128: 14,
            s_delay_alu: 49,
            s_wait: 60,
            vmem_valu_ratio: 29.0 / 281.0, // VALU heavily dominates VMEM traffic
            private_segment_red: false,
        };
        let kprofile = synthetic_kprofile();
        let chip = synthetic_chip(
            /* peak_bw_gbps */ Some(800.0),
            /* mem_latency_ns */ Some(280.0),
        );

        // achieved_bw is far below peak — this kernel is NOT saturating BW.
        let roofline = Roofline::analyze(&hist, &kprofile, &chip, Some(120.0));

        assert!(
            matches!(
                roofline.binding,
                BoundClass::ValuIssue | BoundClass::Latency
            ),
            "low-BW-utilization + heavy VALU/scalar-dequant shape must bind on \
             ValuIssue or Latency, got {:?}",
            roofline.binding
        );
        assert!(
            roofline.verdict.to_lowercase().contains("fix"),
            "verdict must carry a fix hint, got: {}",
            roofline.verdict
        );
    }

    #[test]
    fn bw_saturated_binds_bandwidth() {
        // Same static shape, but achieved_bw is pinned at (near) peak — the
        // kernel IS saturating the bus, so BW must win regardless of the
        // VALU-heavy disassembly shape.
        let hist = IsaHistogram {
            v_bfe: 42,
            v_cvt: 56,
            f32_fma: 98,
            v_dot4: 0,
            v_wmma: 0,
            global_load_b128: 14,
            s_delay_alu: 49,
            s_wait: 60,
            vmem_valu_ratio: 29.0 / 281.0,
            private_segment_red: false,
        };
        let kprofile = synthetic_kprofile();
        let chip = synthetic_chip(Some(800.0), Some(280.0));

        let roofline = Roofline::analyze(&hist, &kprofile, &chip, Some(784.0)); // 98% of peak

        assert_eq!(
            roofline.binding,
            BoundClass::Bandwidth,
            "achieved_bw at 98% of peak must bind on Bandwidth, got {:?} (bw={}, valu_issue={}, latency={})",
            roofline.binding,
            roofline.bw,
            roofline.valu_issue,
            roofline.latency
        );
    }

    #[test]
    fn unprofiled_chip_withholds_latency_bound() {
        // for_unprofiled semantics: peak_bw/mem_latency both None must NOT
        // silently compute as zero-is-fine — the latency score must come
        // out at its withheld floor (0.0), not spuriously bind.
        let hist = IsaHistogram {
            v_bfe: 1,
            v_cvt: 1,
            f32_fma: 1,
            v_dot4: 0,
            v_wmma: 0,
            global_load_b128: 1,
            s_delay_alu: 1,
            s_wait: 1,
            vmem_valu_ratio: 1.0,
            private_segment_red: false,
        };
        let kprofile = synthetic_kprofile();
        let chip = crate::chip_profile::ChipProfile::for_unprofiled("gfx9999");

        let roofline = Roofline::analyze(&hist, &kprofile, &chip, None);
        assert_eq!(
            roofline.bw, 0.0,
            "no achieved_bw / no peak_bw => BW score withheld at 0"
        );
        assert_eq!(
            roofline.latency, 0.0,
            "unprofiled chip (mem_latency_ns=None) => latency score withheld at 0"
        );
    }

    #[test]
    fn verdict_flags_register_spill_regardless_of_binding() {
        // A cross-model review flagged that `private_segment_red` was parsed
        // by `IsaHistogram` but never consumed by `Roofline` (the plan's
        // Task 5 spec calls for the verdict to be a function of
        // "binding × v_dot4==0? × private_segment_red"). Assert the spill
        // note actually shows up in the rendered verdict string.
        let hist = IsaHistogram {
            v_bfe: 42,
            v_cvt: 56,
            f32_fma: 98,
            v_dot4: 0,
            v_wmma: 0,
            global_load_b128: 14,
            s_delay_alu: 49,
            s_wait: 60,
            vmem_valu_ratio: 29.0 / 281.0,
            private_segment_red: true,
        };
        let kprofile = synthetic_kprofile();
        let chip = synthetic_chip(Some(800.0), Some(280.0));

        let roofline = Roofline::analyze(&hist, &kprofile, &chip, Some(120.0));

        assert!(
            roofline.verdict.to_lowercase().contains("spill"),
            "verdict must flag register spill (private_segment_red) \
             regardless of which bound class won, got: {}",
            roofline.verdict
        );
    }

    #[test]
    fn verdict_omits_spill_note_when_not_spilling() {
        let hist = IsaHistogram {
            v_bfe: 42,
            v_cvt: 56,
            f32_fma: 98,
            v_dot4: 0,
            v_wmma: 0,
            global_load_b128: 14,
            s_delay_alu: 49,
            s_wait: 60,
            vmem_valu_ratio: 29.0 / 281.0,
            private_segment_red: false,
        };
        let kprofile = synthetic_kprofile();
        let chip = synthetic_chip(Some(800.0), Some(280.0));

        let roofline = Roofline::analyze(&hist, &kprofile, &chip, Some(120.0));

        assert!(
            !roofline.verdict.to_lowercase().contains("spill"),
            "verdict must NOT mention spill when private_segment_red is false, got: {}",
            roofline.verdict
        );
    }

    #[test]
    fn real_fixture_binds_valu_issue_under_committed_chip() {
        // End-to-end (no-GPU): real `IsaHistogram` + real `KernelProfile`,
        // both derived from the committed gfx1201 gate_up `.hsaco` fixture
        // (the same file `isa_histogram::tests::from_hsaco_gate_up_gfx1201`
        // exercises), fed through `Roofline::analyze` under the committed
        // `tests/chip-profiles/gfx1201.json` chip row. A cross-model review
        // flagged that no test previously ran real fixture data end-to-end
        // through the roofline model to prove the classification lands on
        // VALU-issue rather than Bandwidth.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/kernel-fixtures/gfx1201")
            .join("gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco");
        let hist = IsaHistogram::from_hsaco(&path, "gfx1201")
            .expect("committed gfx1201 fixture must disassemble");

        let mut map = std::collections::HashMap::new();
        map.insert("real_fixture_probe".to_string(), path.clone());
        let (_cap, profiles) = crate::profiler::profile_kernels("gfx1201", 0, &map);
        let kprofile = profiles
            .into_iter()
            .next()
            .expect("profile_kernels must find the fixture's kernel descriptor");

        let chip = crate::chip_profile::ChipProfile::load_committed("gfx1201")
            .expect("tests/chip-profiles/gfx1201.json must load");
        assert!(
            chip.mem_latency_ns.is_some(),
            "committed gfx1201 row now carries a MEASURED mem_latency (warm \
             pointer-chase, 205.9 ns, 2026-07-02) — the latency/Little's-Law \
             roofline is ACTIVE here, not withheld"
        );

        // Static path (`achieved_bw: None`, matching the `--self-check`
        // no-GPU flow): BW is withheld (no achieved_bw) and latency is
        // withheld (no measured mem_latency), so only the real VALU-issue
        // shape carries signal — it must win.
        let roofline = Roofline::analyze(&hist, &kprofile, &chip, None);
        assert_eq!(
            roofline.binding,
            BoundClass::ValuIssue,
            "real gfx1201 gate_up fixture (v_bfe={}, f32_fma={}, vmem_valu_ratio={:.4}) \
             with no achieved_bw / no measured mem_latency must classify as \
             VALU-issue-bound (bw={}, valu_issue={}, latency={}), got {:?}",
            hist.v_bfe,
            hist.f32_fma,
            hist.vmem_valu_ratio,
            roofline.bw,
            roofline.valu_issue,
            roofline.latency,
            roofline.binding
        );
        assert!(
            !hist.private_segment_red,
            "committed-good fixture must not show a register spill"
        );
        assert!(
            !roofline.verdict.to_lowercase().contains("spill"),
            "non-spilling fixture must not carry a spill note, got: {}",
            roofline.verdict
        );
    }

    fn synthetic_kprofile() -> crate::profiler::KernelProfile {
        crate::profiler::KernelProfile {
            name: "synthetic".to_string(),
            vgprs: 64,
            sgprs: 32,
            lds_bytes: 0,
            scratch_bytes: 0,
            kernarg_bytes: 256,
            occupancy_waves: 8,
            max_waves: 16,
            occupancy_limiter: "VGPRs",
        }
    }

    fn synthetic_chip(
        peak_bw_gbps: Option<f64>,
        mem_latency_ns: Option<f64>,
    ) -> crate::chip_profile::ChipProfile {
        let mut chip = crate::chip_profile::ChipProfile::for_unprofiled("gfx1201");
        chip.peak_bw_gbps = peak_bw_gbps;
        chip.mem_latency_ns = mem_latency_ns;
        chip
    }
}
