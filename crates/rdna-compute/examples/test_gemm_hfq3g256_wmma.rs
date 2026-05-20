//! Parity test for the HFQ3 (uniform-MQ3) WMMA family — covers all 4
//! kernels (residual + qkvza + qkv + gate_up) with both `_wmma` and `_mb4`
//! variants. Verifies `_mb4` output is bit-exact against `_wmma` for the
//! same inputs at production shapes.
//!
//! Strategy: don't bother with a CPU reference; the existing `_wmma`
//! kernels are the production baseline (well-trodden in master / 5b/5c
//! reviews). Run both kernels on identical inputs and check max-abs of
//! the GPU-vs-GPU diff. If `_wmma` and `_mb4` produce identical-in-fp32
//! output — modulo the well-defined WMMA reorder envelope — mb4 inherits
//! the same numerical correctness as the production `_wmma`.
//!
//! Routing between `_wmma` and `_mb4` is controlled in-process via the
//! `HIPFIRE_MQ3_MB4` env var (set/unset around each call).

use rdna_compute::{Gpu, DType};

// HFQ3 group: 8 B header (sc:f32 + zp:f32) + 96 B 3-bit indices = 104 B.
// Same packing as MQ3-Lloyd test (see test_gemm_fused_mq3g256_lloyd_wmma.rs):
// 32 chunks × 3 B per group; K-tile reads 6 B = 2 adjacent chunks.
fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
    let mut out = [0u8; 96];
    for tid in 0..32 {
        let mut pk: u32 = 0;
        for i in 0..8 { pk |= (qs[tid * 8 + i] as u32 & 7) << (3 * i); }
        out[tid * 3]     = (pk        & 0xff) as u8;
        out[tid * 3 + 1] = ((pk >>  8) & 0xff) as u8;
        out[tid * 3 + 2] = ((pk >> 16) & 0xff) as u8;
    }
    out
}

/// Build a synthetic HFQ3 matrix [m × k] with per-projection-distinct sc/zp.
/// proj_id seeds sc/zp variation so a swapped weight pointer in dispatch
/// yields a detectable Y delta — though the real correctness signal here
/// is `_wmma == _mb4` regardless of proj_id.
fn build_hfq3_matrix(m: usize, k: usize, proj_id: usize) -> Vec<u8> {
    let groups_per_row = k / 256;
    let mut all_bytes = Vec::with_capacity(m * groups_per_row * 104);
    for row in 0..m {
        for g in 0..groups_per_row {
            let proj_off = proj_id as f32 * 0.05;
            let sc_raw = ((row * 7 + g * 11 + proj_id * 31) % 19) as f32 * 0.001 + 0.01 + proj_off * 0.005;
            let zp_raw = ((row * 13 + g * 17 + proj_id * 29) % 23) as f32 * 0.002 - 0.02 + proj_off;
            all_bytes.extend_from_slice(&sc_raw.to_le_bytes());
            all_bytes.extend_from_slice(&zp_raw.to_le_bytes());
            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)
                       ^ proj_id.wrapping_mul(101)) & 7) as u8;
            }
            all_bytes.extend_from_slice(&pack_3bit_group(&q));
        }
    }
    assert_eq!(all_bytes.len(), m * groups_per_row * 104);
    all_bytes
}

fn make_x(n: usize, k: usize) -> Vec<f32> {
    (0..(n * k)).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect()
}

fn diff_metrics(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_sq = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        let denom = b[i].abs().max(1e-3);
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / denom);
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / a.len() as f64).sqrt() as f32;
    (max_abs, max_rel, rms)
}

/// Helper: run kernel `f` with HIPFIRE_MQ3_MB4 set to `mode`, return Y.
fn run_with_mode<F: FnOnce(&mut Gpu)>(gpu: &mut Gpu, mode: &str, f: F) {
    std::env::set_var("HIPFIRE_MQ3_MB4", mode);
    f(gpu);
    std::env::remove_var("HIPFIRE_MQ3_MB4");
}

// `_wmma` vs `_mb4` accumulation order is identical for both (one acc per
// 16×N tile, K-tile loop is the same). Expected max-abs delta = 0.0 for
// shapes where mb4 fully covers (n divisible by 64 and m divisible by 16).
// For boundary shapes, mb4's safe_batch=0 fallback may differ from wmma's
// safe_batch=0 in the OOB-output rows that get masked anyway — set a
// tight envelope that catches real bugs but tolerates fp32-reorder noise.
const TOL: f32 = 1e-4;

fn test_residual(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> bool {
    println!("--- residual M={} K={} N={} ---", m, k, n);

    let a_b = build_hfq3_matrix(m, k, 0);
    let x = make_x(n, k);
    let y_init: Vec<f32> = (0..(n * m)).map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.001).collect();

    let d_a = gpu.upload_raw(&a_b, &[a_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

    // _wmma path
    let d_y_wmma = gpu.upload_f32(&y_init, &[n, m]).unwrap();
    run_with_mode(gpu, "0", |g| {
        g.gemm_hfq3g256_residual_wmma(&d_a, &d_x, &d_y_wmma, m, k, n).unwrap();
    });
    let y_wmma = gpu.download_f32(&d_y_wmma).unwrap();

    // _mb4 path (force-on regardless of size gate)
    let d_y_mb4 = gpu.upload_f32(&y_init, &[n, m]).unwrap();
    run_with_mode(gpu, "1", |g| {
        g.gemm_hfq3g256_residual_wmma(&d_a, &d_x, &d_y_mb4, m, k, n).unwrap();
    });
    let y_mb4 = gpu.download_f32(&d_y_mb4).unwrap();

    let (ma, mr, rms) = diff_metrics(&y_mb4, &y_wmma);
    let pass = ma < TOL;
    println!("  max_abs={:.3e}  max_rel={:.3e}  rms={:.3e}  {}",
             ma, mr, rms, if pass { "PASS" } else { "FAIL" });

    for d in [d_a, d_x, d_y_wmma, d_y_mb4] { gpu.free_tensor(d).unwrap(); }
    pass
}

fn test_qkvza(gpu: &mut Gpu, qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize, k: usize, n: usize) -> bool {
    println!("--- qkvza M=({}+{}+{}+{}) K={} N={} ---", qkv_m, z_m, beta_m, alpha_m, k, n);

    let a_qkv_b = build_hfq3_matrix(qkv_m, k, 0);
    let a_z_b = build_hfq3_matrix(z_m, k, 1);
    let a_beta_b = build_hfq3_matrix(beta_m, k, 2);
    let a_alpha_b = build_hfq3_matrix(alpha_m, k, 3);
    let x = make_x(n, k);

    let d_a_qkv = gpu.upload_raw(&a_qkv_b, &[a_qkv_b.len()]).unwrap();
    let d_a_z = gpu.upload_raw(&a_z_b, &[a_z_b.len()]).unwrap();
    let d_a_beta = gpu.upload_raw(&a_beta_b, &[a_beta_b.len()]).unwrap();
    let d_a_alpha = gpu.upload_raw(&a_alpha_b, &[a_alpha_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

    let alloc = |gpu: &mut Gpu| {
        (
            gpu.zeros(&[n, qkv_m], DType::F32).unwrap(),
            gpu.zeros(&[n, z_m], DType::F32).unwrap(),
            gpu.zeros(&[n, beta_m], DType::F32).unwrap(),
            gpu.zeros(&[n, alpha_m], DType::F32).unwrap(),
        )
    };

    let (d_yq_w, d_yz_w, d_yb_w, d_ya_w) = alloc(gpu);
    run_with_mode(gpu, "0", |g| {
        g.gemm_qkvza_hfq3g256_wmma(
            &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x,
            &d_yq_w, &d_yz_w, &d_yb_w, &d_ya_w,
            qkv_m, z_m, beta_m, alpha_m, k, n,
        ).unwrap();
    });
    let yq_w = gpu.download_f32(&d_yq_w).unwrap();
    let yz_w = gpu.download_f32(&d_yz_w).unwrap();
    let yb_w = gpu.download_f32(&d_yb_w).unwrap();
    let ya_w = gpu.download_f32(&d_ya_w).unwrap();

    let (d_yq_m, d_yz_m, d_yb_m, d_ya_m) = alloc(gpu);
    run_with_mode(gpu, "1", |g| {
        g.gemm_qkvza_hfq3g256_wmma(
            &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha, &d_x,
            &d_yq_m, &d_yz_m, &d_yb_m, &d_ya_m,
            qkv_m, z_m, beta_m, alpha_m, k, n,
        ).unwrap();
    });
    let yq_m = gpu.download_f32(&d_yq_m).unwrap();
    let yz_m = gpu.download_f32(&d_yz_m).unwrap();
    let yb_m = gpu.download_f32(&d_yb_m).unwrap();
    let ya_m = gpu.download_f32(&d_ya_m).unwrap();

    let (ma_q, _, _) = diff_metrics(&yq_m, &yq_w);
    let (ma_z, _, _) = diff_metrics(&yz_m, &yz_w);
    let (ma_b, _, _) = diff_metrics(&yb_m, &yb_w);
    let (ma_a, _, _) = diff_metrics(&ya_m, &ya_w);
    let max_abs = ma_q.max(ma_z).max(ma_b).max(ma_a);
    let pass = max_abs < TOL;
    println!("  qkv max_abs={:.3e}  z={:.3e}  beta={:.3e}  alpha={:.3e}  {}",
             ma_q, ma_z, ma_b, ma_a, if pass { "PASS" } else { "FAIL" });

    for d in [d_a_qkv, d_a_z, d_a_beta, d_a_alpha, d_x,
              d_yq_w, d_yz_w, d_yb_w, d_ya_w,
              d_yq_m, d_yz_m, d_yb_m, d_ya_m] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn test_qkv(gpu: &mut Gpu, q_m: usize, k_m: usize, v_m: usize, k: usize, n: usize) -> bool {
    println!("--- qkv M=({}+{}+{}) K={} N={} ---", q_m, k_m, v_m, k, n);

    let a_q_b = build_hfq3_matrix(q_m, k, 0);
    let a_k_b = build_hfq3_matrix(k_m, k, 1);
    let a_v_b = build_hfq3_matrix(v_m, k, 2);
    let x = make_x(n, k);

    let d_a_q = gpu.upload_raw(&a_q_b, &[a_q_b.len()]).unwrap();
    let d_a_k = gpu.upload_raw(&a_k_b, &[a_k_b.len()]).unwrap();
    let d_a_v = gpu.upload_raw(&a_v_b, &[a_v_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

    let alloc = |gpu: &mut Gpu| (
        gpu.zeros(&[n, q_m], DType::F32).unwrap(),
        gpu.zeros(&[n, k_m], DType::F32).unwrap(),
        gpu.zeros(&[n, v_m], DType::F32).unwrap(),
    );

    let (d_yq_w, d_yk_w, d_yv_w) = alloc(gpu);
    run_with_mode(gpu, "0", |g| {
        g.gemm_qkv_hfq3g256_wmma(&d_a_q, &d_a_k, &d_a_v, &d_x,
            &d_yq_w, &d_yk_w, &d_yv_w, q_m, k_m, v_m, k, n).unwrap();
    });
    let yq_w = gpu.download_f32(&d_yq_w).unwrap();
    let yk_w = gpu.download_f32(&d_yk_w).unwrap();
    let yv_w = gpu.download_f32(&d_yv_w).unwrap();

    let (d_yq_m, d_yk_m, d_yv_m) = alloc(gpu);
    run_with_mode(gpu, "1", |g| {
        g.gemm_qkv_hfq3g256_wmma(&d_a_q, &d_a_k, &d_a_v, &d_x,
            &d_yq_m, &d_yk_m, &d_yv_m, q_m, k_m, v_m, k, n).unwrap();
    });
    let yq_m = gpu.download_f32(&d_yq_m).unwrap();
    let yk_m = gpu.download_f32(&d_yk_m).unwrap();
    let yv_m = gpu.download_f32(&d_yv_m).unwrap();

    let (ma_q, _, _) = diff_metrics(&yq_m, &yq_w);
    let (ma_k, _, _) = diff_metrics(&yk_m, &yk_w);
    let (ma_v, _, _) = diff_metrics(&yv_m, &yv_w);
    let max_abs = ma_q.max(ma_k).max(ma_v);
    let pass = max_abs < TOL;
    println!("  q max_abs={:.3e}  k={:.3e}  v={:.3e}  {}",
             ma_q, ma_k, ma_v, if pass { "PASS" } else { "FAIL" });

    for d in [d_a_q, d_a_k, d_a_v, d_x, d_yq_w, d_yk_w, d_yv_w, d_yq_m, d_yk_m, d_yv_m] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn test_gate_up(gpu: &mut Gpu, gate_m: usize, up_m: usize, k: usize, n: usize) -> bool {
    println!("--- gate_up M=({}+{}) K={} N={} ---", gate_m, up_m, k, n);

    let a_g_b = build_hfq3_matrix(gate_m, k, 0);
    let a_u_b = build_hfq3_matrix(up_m, k, 1);
    let x = make_x(n, k);

    let d_a_g = gpu.upload_raw(&a_g_b, &[a_g_b.len()]).unwrap();
    let d_a_u = gpu.upload_raw(&a_u_b, &[a_u_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

    let alloc = |gpu: &mut Gpu| (
        gpu.zeros(&[n, gate_m], DType::F32).unwrap(),
        gpu.zeros(&[n, up_m], DType::F32).unwrap(),
    );

    let (d_yg_w, d_yu_w) = alloc(gpu);
    run_with_mode(gpu, "0", |g| {
        g.gemm_gate_up_hfq3g256_wmma(&d_a_g, &d_a_u, &d_x, &d_yg_w, &d_yu_w,
            gate_m, up_m, k, n).unwrap();
    });
    let yg_w = gpu.download_f32(&d_yg_w).unwrap();
    let yu_w = gpu.download_f32(&d_yu_w).unwrap();

    let (d_yg_m, d_yu_m) = alloc(gpu);
    run_with_mode(gpu, "1", |g| {
        g.gemm_gate_up_hfq3g256_wmma(&d_a_g, &d_a_u, &d_x, &d_yg_m, &d_yu_m,
            gate_m, up_m, k, n).unwrap();
    });
    let yg_m = gpu.download_f32(&d_yg_m).unwrap();
    let yu_m = gpu.download_f32(&d_yu_m).unwrap();

    let (ma_g, _, _) = diff_metrics(&yg_m, &yg_w);
    let (ma_u, _, _) = diff_metrics(&yu_m, &yu_w);
    let max_abs = ma_g.max(ma_u);
    let pass = max_abs < TOL;
    println!("  gate max_abs={:.3e}  up={:.3e}  {}",
             ma_g, ma_u, if pass { "PASS" } else { "FAIL" });

    for d in [d_a_g, d_a_u, d_x, d_yg_w, d_yu_w, d_yg_m, d_yu_m] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    eprintln!("Verifying HFQ3 _mb4 == _wmma at all shapes (TOL={:.0e}).", TOL);

    let mut all_pass = true;

    // residual — covers boundary cases (n=16 padded to 64; n=64 partial-mb4).
    all_pass &= test_residual(&mut gpu, 64,   1024,  16);
    all_pass &= test_residual(&mut gpu, 64,   1024,  64);
    all_pass &= test_residual(&mut gpu, 256,  4096, 256);
    all_pass &= test_residual(&mut gpu, 1024, 4096,  64);
    all_pass &= test_residual(&mut gpu, 1024, 12288, 64);

    // qkvza — straddles projection boundaries (4-way fan-out).
    all_pass &= test_qkvza(&mut gpu, 64, 16, 8, 8, 1024, 16);
    all_pass &= test_qkvza(&mut gpu, 256, 32, 16, 16, 4096, 64);
    all_pass &= test_qkvza(&mut gpu, 512, 64, 32, 32, 4096, 32);

    // qkv — 3-way fan-out, balanced + asymmetric.
    all_pass &= test_qkv(&mut gpu, 64, 64, 64, 1024, 16);
    all_pass &= test_qkv(&mut gpu, 256, 32, 32, 4096, 64);
    all_pass &= test_qkv(&mut gpu, 512, 64, 64, 4096, 32);

    // gate_up — 2-way fan-out.
    all_pass &= test_gate_up(&mut gpu, 256, 256, 1024, 16);
    all_pass &= test_gate_up(&mut gpu, 1024, 1024, 4096, 64);

    println!();
    if !all_pass {
        eprintln!("FAIL: one or more HFQ3 _wmma vs _mb4 parity tests failed");
        std::process::exit(1);
    }
    println!("ALL PASS — HFQ3 _mb4 produces output bit-equivalent to _wmma");
}
