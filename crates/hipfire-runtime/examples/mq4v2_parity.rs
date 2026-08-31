//! Host-vs-GPU decode oracle for MQ4-G256-V2 (qt=44) — the correctness gate for
//! v2's dedicated kernels.
//!
//! ## What makes v2 different, and what this test is for
//!
//! v1 (qt=13) stores ONE affine grid per 256-weight group: `[f32 scale][f32 zero]`.
//! v2 (qt=44) spends the same 8 header bytes on TWO INDEPENDENT grids —
//! `[fp16 s0][fp16 z0][fp16 s1][fp16 z1]` — where `(s0,z0)` governs weights 0..127
//! and `(s1,z1)` governs weights 128..255. Payload nibbles and the 136 B group
//! stride are byte-identical to v1.
//!
//! Because the two formats differ ONLY in how those 8 bytes are interpreted, every
//! cheap check is blind to a mis-decode:
//!   * byte count is identical by construction,
//!   * the dtype census is identical,
//!   * a v1 kernel fed v2 bytes still runs at full speed,
//!   * and it compiles, because the dtype was simply added to v1's match arms.
//! A first v2 artifact scored WT2 KLD 12.137559 against a 0.043776 baseline
//! (PPL ~1e6, pure noise) at 163 tok/s while passing every one of those.
//!
//! ## The discriminating construction
//!
//! This oracle builds each group so the two halves have DELIBERATELY DISJOINT
//! ranges: weights 0..127 land in `[-1, 1]`, weights 128..255 in `[+96, +160]`.
//! Consequences, all of which a single-header read cannot survive:
//!   * a kernel that reads only `(s0,z0)` and applies it to all 256 weights
//!     reconstructs half 1 as ~0 instead of ~128 — the half-select bug that an
//!     "equal halves" artifact provably cannot detect, since there both headers
//!     hold the same value;
//!   * a kernel that reads the header as v1's `[f32 scale][f32 zero]` reinterprets
//!     four fp16 fields as two f32s and produces noise;
//!   * a kernel that swaps the halves reconstructs 128 where it should see 0.
//!
//! Passing therefore means the kernel reads BOTH headers, assigns each to the
//! correct 128 weights, and decodes fp16 — the full v2 contract.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq4v2_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
/// 8 header bytes + 128 nibble bytes. Identical to v1's stride.
const GROUP_BYTES: usize = 136;

/// Deterministic pseudo-random in [0,1) — no rand dependency, and reproducible
/// across hosts so a failure is comparable between machines.
fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Build weights whose halves occupy disjoint ranges (see module docs).
/// Returns row-major `[m, k]` values already in the "rotated" domain — this is a
/// pure decode test, so no FWHT is applied on either side.
fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                // half 0: [-1, 1]
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                // half 1: [96, 160] — two orders of magnitude away from half 0,
                // so any header cross-contamination is unmissable.
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

/// Host encoder mirroring `quantize_mq4g256v2` in hipfire-quantize, minus the FWHT.
/// Per half: `step = (hi-lo)/15`, stored as fp16 scale; `lo` stored as fp16 zero;
/// codes are `round((w - zero_rt) / scale_rt)` against the ROUND-TRIPPED header, so
/// the encoder commits to exactly the values the kernel will read back.
fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be a multiple of 256");
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let z_bits = f16::from_f32(lo).to_bits();
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16::from_bits(s_bits).to_f32();
                let z_rt = f16::from_bits(z_bits).to_f32();
                if s_rt == 0.0 {
                    continue; // degenerate half → all codes 0
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            // Nibble packing identical to v1: weight 2i low, weight 2i+1 high.
            // Weights 0..127 therefore occupy payload bytes 0..63 and weights
            // 128..255 occupy bytes 64..127 — this is what makes the half-select
            // predicate layout-derived rather than a lane test.
            for i in 0..HALF {
                let lo_q = codes[2 * i] & 0xF;
                let hi_q = codes[2 * i + 1] & 0xF;
                blob[dst + 8 + i] = lo_q | (hi_q << 4);
            }
        }
    }
    blob
}

/// Decode the blob per the v2 spec and compute `y = W · x` in f64. This is the
/// oracle: it reads BOTH headers by construction.
fn ref_gemv_f64(blob: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut hs = [0.0f32; 2];
            let mut hz = [0.0f32; 2];
            for h in 0..2 {
                let s = u16::from_le_bytes([blob[dst + h * 4], blob[dst + h * 4 + 1]]);
                let z = u16::from_le_bytes([blob[dst + h * 4 + 2], blob[dst + h * 4 + 3]]);
                hs[h] = f16::from_bits(s).to_f32();
                hz[h] = f16::from_bits(z).to_f32();
            }
            for i in 0..HALF {
                let byte = blob[dst + 8 + i];
                let q_lo = (byte & 0xF) as f32;
                let q_hi = (byte >> 4) as f32;
                let idx_lo = 2 * i;
                let idx_hi = 2 * i + 1;
                let h_lo = idx_lo / HALF;
                let h_hi = idx_hi / HALF;
                let w_lo = hz[h_lo] + hs[h_lo] * q_lo;
                let w_hi = hz[h_hi] + hs[h_hi] * q_hi;
                acc += w_lo as f64 * x[g * GROUP + idx_lo] as f64;
                acc += w_hi as f64 * x[g * GROUP + idx_hi] as f64;
            }
        }
        y[r] = acc;
    }
    y
}

/// What a kernel that reads ONLY half 0's header would produce. Reported as the
/// "bug baseline": the real error must be orders of magnitude below this, or the
/// test proves nothing about the half select.
fn ref_gemv_single_header_f64(blob: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let s = u16::from_le_bytes([blob[dst], blob[dst + 1]]);
            let z = u16::from_le_bytes([blob[dst + 2], blob[dst + 3]]);
            let s0 = f16::from_bits(s).to_f32();
            let z0 = f16::from_bits(z).to_f32();
            for i in 0..HALF {
                let byte = blob[dst + 8 + i];
                let w_lo = z0 + s0 * (byte & 0xF) as f32;
                let w_hi = z0 + s0 * (byte >> 4) as f32;
                acc += w_lo as f64 * x[g * GROUP + 2 * i] as f64;
                acc += w_hi as f64 * x[g * GROUP + 2 * i + 1] as f64;
            }
        }
        y[r] = acc;
    }
    y
}

fn rel_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut worst_i = 0;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-6);
        let e = ((g as f64) - w).abs() / denom;
        if e > worst {
            worst = e;
            worst_i = i;
        }
    }
    (worst, worst_i)
}

fn check_gfx1100_residual_r1_exact(gpu: &mut Gpu) {
    assert!(
        gpu.arch_caps.is_gfx1100(),
        "HIPFIRE_MQ4V2_RESIDUAL_R1=1 parity requires exact gfx1100, got {}",
        gpu.arch
    );

    // This is the exact Ornith attention-output shape admitted by the candidate.
    // Starting Y at +0 makes residual addition bit-identical to the plain V2
    // store, so every output bit is a direct oracle for unchanged decode and
    // accumulation order. The padded tail catches an over-wide row grid.
    let (m, k) = (2_048usize, 4_096usize);
    let w = build_disjoint_halves(m, k);
    let blob = pack_mq4g256v2(&w, m, k);
    let x: Vec<f32> = (0..k).map(|i| prng(i, 0x51D3_0001) * 2.0 - 1.0).collect();
    let canary = f32::from_bits(0x4B5A_A55A);
    let mut y_init = vec![0.0f32; m + 16];
    y_init[m..].fill(canary);

    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_plain = gpu.upload_f32(&y_init, &[m + 16]).unwrap();
    let d_residual = gpu.upload_f32(&y_init, &[m + 16]).unwrap();
    gpu.gemv_mq4g256v2(&d_a, &d_x, &d_plain, m, k)
        .expect("plain gemv_mq4g256v2 launch failed");
    gpu.gemv_hfq4g256_residual_mq4v2(&d_a, &d_x, &d_residual, m, k)
        .expect("residual R1 MQ4V2 launch failed");
    gpu.hip.device_synchronize().unwrap();

    let plain = gpu.download_f32(&d_plain).unwrap();
    let residual = gpu.download_f32(&d_residual).unwrap();
    if let Some(i) = (0..m).find(|&i| plain[i].to_bits() != residual[i].to_bits()) {
        panic!(
            "gfx1100 residual R1 differs from plain V2 at row {i}: \
             plain={:#010x}, residual={:#010x}",
            plain[i].to_bits(),
            residual[i].to_bits()
        );
    }
    assert!(
        plain[m..].iter().all(|&v| v.to_bits() == canary.to_bits())
            && residual[m..]
                .iter()
                .all(|&v| v.to_bits() == canary.to_bits()),
        "gfx1100 residual R1 wrote past M={m}"
    );
    eprintln!(
        "gfx1100 residual R1 K4096: PASS — {m} outputs bit-exact vs plain V2; canaries intact"
    );
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq4v2_parity: no GPU ({e}) — skipping");
            return;
        }
    };

    // k=512 gives 2 groups per row, so a group-stride error shows up too.
    let (m, k) = (64usize, 512usize);
    let w = build_disjoint_halves(m, k);
    let blob = pack_mq4g256v2(&w, m, k);
    assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);

    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0).collect();

    let want = ref_gemv_f64(&blob, &x, m, k);
    let bug = ref_gemv_single_header_f64(&blob, &x, m, k);

    // How far apart the correct decode and the single-header bug are. If this is
    // small the fixture is not discriminating and the test is worthless.
    let (sep, _) = {
        let bug_f32: Vec<f32> = bug.iter().map(|&v| v as f32).collect();
        rel_err(&bug_f32, &want)
    };
    eprintln!("fixture separation (single-header vs correct): {sep:.3e} relative");
    assert!(
        sep > 1.0,
        "fixture is not discriminating: a single-header read differs from the \
         correct decode by only {sep:.3e}. Rebuild the halves with more disjoint \
         ranges; otherwise a passing parity result proves nothing."
    );

    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_hfq4g256_residual_mq4v2(&d_a, &d_x, &d_y, m, k)
        .expect("gemv_hfq4g256_residual_mq4v2 launch failed");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_y).unwrap();

    let (worst, worst_i) = rel_err(&got, &want);
    eprintln!(
        "worst relative error: {worst:.3e} at row {worst_i} (got {:.6}, want {:.6})",
        got[worst_i], want[worst_i]
    );

    // fp32 accumulation over 512 terms with values ~128 — 1e-4 is loose enough to
    // absorb summation order, and ~4 orders tighter than the single-header bug.
    const TOL: f64 = 1e-4;
    assert!(
        worst < TOL,
        "MQ4G256V2 GPU decode disagrees with the host oracle: worst relative error \
         {worst:.3e} at row {worst_i} exceeds {TOL:.0e}.\n\
         A single-header (half-0-only) read would give {sep:.3e}; a v1 header read \
         would give noise. Compare against those to identify which mis-decode this \
         is. Do NOT relax this tolerance — it is the only check that distinguishes \
         a correct per-128 decode from v1 wearing v2's header."
    );

    if std::env::var("HIPFIRE_MQ4V2_RESIDUAL_R1").as_deref() == Ok("1") {
        check_gfx1100_residual_r1_exact(&mut gpu);
    }

    eprintln!("mq4v2_parity: PASS — both half headers read, correct 128-weight assignment, fp16 decode exact");
}
