//! Deterministic CPU-vs-GPU parity for MQ4G256V2 (qt=44) residual / batched-lmhead
//! WMMA paths — the gfx11 slice (`gemm_mq4g256v2_residual_wmma`).
//!
//! ## Why this harness exists
//!
//! MQ4V2 differs from HFQ4 only in its 8 header bytes: v2 stores two
//! independent fp16 grids `[s0 z0]` for weights 0..127 and `[s1 z1]` for
//! 128..255 of each 256-weight group (136 B stride, payload at +8). The new
//! gfx11 kernel must therefore:
//!   * read both headers per group,
//!   * apply `kt<8 ? (s0,z0) : (s1,z1)` (or `quarter<2` in the ldsstage
//!     variant), and
//!   * preserve the gfx11 half16/WMMA/lane/C contracts.
//!
//! A fixture with equal halves cannot discriminate a half-select bug: both
//! headers hold the same value so a single-header read is invisible. This
//! harness builds DISJOINT halves — half0 in `[-1,1]`, half1 in `[96,160]`
//! — so a kernel that decodes half1 with half0's header reconstructs
//! `~0` instead of `~128` and fails by >100% relative error. The same
//! construction also defeats a v1-as-f32 mis-decode (noise) and a swapped
//! half (128 where 0 expected).
//!
//! Shapes: M × K × N with K∈{256,512} (1 and 2 groups/row) and N∈{16,64}
//! (one and four 16-batch tiles) to cover the residual grid's row and batch
//! dimensions. M is a multiple of 16. Each shape is run twice:
//!   * residual: `Y += W·X` with a non-zero Y init, through
//!     `gemm_hfq4g256_residual_mq4v2` (gfx11 selects `has_wmma_w32`,
//!     gfx12 through the existing `_gfx12` path);
//!   * batched lmhead: `Y = W·X` via `gemm_mq4g256v2_batched_lmhead` (zeroed
//!     internally, overwrite semantics), same discriminating payload.
//! Both exercise the public dispatchers directly. A half1-with-half0 bug is
//! reported as the "bug baseline" and the test `assert!` fails if the GPU
//! error is not orders of magnitude below it.
//!
//! Tolerances mirror `mq4v2_gemm_parity` / `mq4v2_parity`: worst relative
//! error < 5e-4 for the correct decode on this fixture (FP32 accumulation
//! over ≤512 terms of magnitude ~100), and single-header bug separation >1.0.
//! The harness exits non-zero on failure.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq4v2_residual_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            let gi = c % GROUP;
            let idx = r * k + c;
            if gi < HALF {
                // [-1, 1]
                w[idx] = prng(idx, 0xA5A5_0001) * 2.0 - 1.0;
            } else {
                // [96, 160] — disjoint from half0 by two orders of magnitude
                w[idx] = 96.0 + prng(idx, 0x5A5A_0002) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut q = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let s = &w[src + off..src + off + HALF];
                let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let sb = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let zb = f16::from_f32(lo).to_bits();
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&sb.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&zb.to_le_bytes());
                let st = f16::from_bits(sb).to_f32();
                let z = f16::from_bits(zb).to_f32();
                if st == 0.0 {
                    continue;
                }
                let inv = 1.0 / st;
                for i in 0..HALF {
                    q[off + i] = ((s[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

fn decode_w(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let h_b =
                u32::from_le_bytes([blob[src + 4], blob[src + 5], blob[src + 6], blob[src + 7]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            let sc1 = f16::from_bits((h_b & 0xFFFF) as u16).to_f32();
            let zp1 = f16::from_bits((h_b >> 16) as u16).to_f32();
            for kt in 0..16 {
                let (sc, zp) = if kt < 8 { (sc0, zp0) } else { (sc1, zp1) };
                let k_off = kt * 16;
                for lane in 0..16 {
                    let byte = blob[src + 8 + k_off / 2 + lane / 2];
                    let nib = if lane % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let wk = g * GROUP + k_off + lane;
                    w[r * k + wk] = sc * (nib as f32) + zp;
                }
            }
        }
    }
    w
}

fn decode_w_single_header(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            for kt in 0..16 {
                let k_off = kt * 16;
                for lane in 0..16 {
                    let byte = blob[src + 8 + k_off / 2 + lane / 2];
                    let nib = if lane % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let wk = g * GROUP + k_off + lane;
                    w[r * k + wk] = sc0 * (nib as f32) + zp0;
                }
            }
        }
    }
    w
}

fn ref_gemm(w: &[f32], x: &[f32], y_init: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    // y = y_init + W·X, W [m,k] row-major, X [n,k] row-major, Y [n,m] row-major
    let y = y_init.to_vec();
    let mut out = vec![0.0f64; n * m];
    for nn in 0..n {
        for mm in 0..m {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += w[mm * k + kk] as f64 * x[nn * k + kk] as f64;
            }
            out[nn * m + mm] = y[nn * m + mm] as f64 + acc;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut wi = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let e = ((g as f64 - w).abs()) / denom;
        if e > worst {
            worst = e;
            wi = i;
        }
    }
    (worst, wi)
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq4v2_residual_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!(
        "mq4v2_residual_parity: arch={} has_wmma_w32={} has_wmma_w32_gfx12={}",
        gpu.arch,
        gpu.arch_caps.has_wmma_w32(),
        gpu.arch_caps.has_wmma_w32_gfx12()
    );

    // Shapes covering K=256 (1 group) and K=512 (2 groups), N=16/64 batched tiles.
    let shapes: &[(usize, usize, usize)] = &[
        (32, 256, 16),
        (32, 256, 64),
        (32, 512, 16),
        (32, 512, 64),
        (64, 256, 16),
        (64, 512, 16),
        (16, 512, 64),
    ];

    let mut failures = Vec::new();

    for &(m, k, n) in shapes {
        assert!(k % GROUP == 0);
        assert!(m % 16 == 0);
        assert!(n % 16 == 0);
        let w_f32 = build_disjoint_halves(m, k);
        let blob = pack_mq4g256v2(&w_f32, m, k);
        assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);

        // Verify fixture discriminates: half0 vs half1 headers differ, and
        // single-header decode is far from correct.
        let w_correct = decode_w(&blob, m, k);
        let w_bug = decode_w_single_header(&blob, m, k);
        let x: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        let y_init: Vec<f32> = (0..n * m).map(|i| prng(i, 0xDEAD_BEEF) * 0.5).collect();

        let want = ref_gemm(&w_correct, &x, &y_init, m, k, n);
        let want_bug = ref_gemm(&w_bug, &x, &y_init, m, k, n);
        let bug_rel = {
            let bug_f32: Vec<f32> = want_bug.iter().map(|&v| v as f32).collect();
            rel_err(&bug_f32, &want).0
        };
        eprintln!("shape M={m} K={k} N={n}: bug baseline rel {bug_rel:.3e}");
        assert!(bug_rel > 1.0, "fixture not discriminating at M={m} K={k} N={n}: bug_rel {bug_rel:.3e} — halves overlap");

        // ── residual path (Y += W·X) ──
        {
            let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
            // Upload Y with residual init; residual kernel does Y += ...
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_hfq4g256_residual_mq4v2(&d_a, &d_x, &d_y, m, k, n)
                .unwrap_or_else(|e| panic!("residual launch M={m} K={k} N={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let got = gpu.download_f32(&d_y).unwrap();
            let (worst, wi) = rel_err(&got, &want);
            let status = if worst < 5e-4 { "ok" } else { "FAIL" };
            eprintln!("  residual M={m} K={k} N={n}: worst {worst:.3e} at {wi} {status} (bug {bug_rel:.3e})");
            if worst >= 5e-4 {
                eprintln!(
                    "    got {} want {} y_init {}",
                    got[wi], want[wi], y_init[wi]
                );
                failures.push(format!("residual M={m} K={k} N={n} worst {worst:.3e}"));
            }
            assert!(
                worst < bug_rel * 0.01,
                "half1 decoded with half0 at residual M={m} K={k} N={n}: worst {worst:.3e} not below bug {bug_rel:.3e}"
            );
        }

        // ── batched lmhead path (Y = W·X, zeroed internally) ──
        {
            // lmhead reference is y = W·X (no residual). Build want0 with zero init.
            let y_zero = vec![0.0f32; n * m];
            let want0 = ref_gemm(&w_correct, &x, &y_zero, m, k, n);
            let want0_bug = ref_gemm(&w_bug, &x, &y_zero, m, k, n);
            let bug_rel0 = {
                let b: Vec<f32> = want0_bug.iter().map(|&v| v as f32).collect();
                rel_err(&b, &want0).0
            };
            let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
            let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
            let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
            gpu.gemm_mq4g256v2_batched_lmhead(&d_a, &d_x, &d_y, m, k, n)
                .unwrap_or_else(|e| panic!("lmhead launch M={m} K={k} N={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let got = gpu.download_f32(&d_y).unwrap();
            let (worst, wi) = rel_err(&got, &want0);
            let status = if worst < 5e-4 { "ok" } else { "FAIL" };
            eprintln!("  lmhead   M={m} K={k} N={n}: worst {worst:.3e} at {wi} {status} (bug {bug_rel0:.3e})");
            if worst >= 5e-4 {
                failures.push(format!("lmhead M={m} K={k} N={n} worst {worst:.3e}"));
            }
            assert!(
                worst < bug_rel0 * 0.01,
                "half1 decoded with half0 at lmhead M={m} K={k} N={n}: worst {worst:.3e} not below bug {bug_rel0:.3e}"
            );
        }
    }

    if failures.is_empty() {
        eprintln!("\nmq4v2_residual_parity: PASS — gfx11 residual + lmhead both half-select correct at all shapes");
    } else {
        eprintln!("\nmq4v2_residual_parity: FAIL — {:?}", failures);
        std::process::exit(1);
    }
}
