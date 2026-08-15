//! GPU wiring validation for the `fused_qkv_<dtype>_with_bias` kernels
//! (HIPFIRE_FUSE_QKV_BIAS). Format-agnostic: we upload ZEROED weight bytes for
//! each format, so every kernel's GEMV body produces exactly 0.0 (scale/codebook
//! = 0 → 0 contribution, finite). The fused store is then `y[row] = 0 + bias[row]`,
//! so the with-bias output MUST equal the per-projection bias bit-exactly, and
//! the no-bias call MUST produce all zeros.
//!
//! This validates, without needing a real model, exactly the three things the
//! bias fold can get wrong: (1) the kernarg ABI (count/order) — a mismatch
//! crashes or corrupts; (2) the q/k/v→bias routing — each row must read its own
//! projection's bias; (3) the `+bias` epilogue. Run on the target GPU:
//!   cargo run --release --example test_fused_qkv_bias_parity -p rdna-compute

use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

const Q_M: usize = 32;
const K_M: usize = 16;
const V_M: usize = 16;
const K: usize = 4096; // 16 groups of 256 (and 128 blocks of 32 for q8_0)

/// Per-row byte stride for each format (× number of groups/blocks per row).
fn row_bytes(dtype: &str) -> usize {
    match dtype {
        "hfq4g256" => (K / 256) * 136,
        "hfq6g256" => (K / 256) * 200,
        "mq4g256_lloyd" => (K / 256) * 160,
        "mq3g256_lloyd" => (K / 256) * 112,
        "q4k" => (K / 256) * 144,
        "q8_0" => (K / 32) * 34,
        _ => unreachable!(),
    }
}

fn zeros_weight(gpu: &mut Gpu, m: usize, dtype: &str) -> HipResult<GpuTensor> {
    let bytes = vec![0u8; m * row_bytes(dtype)];
    gpu.upload_raw(&bytes, &[bytes.len()])
}

/// Distinct, deterministic bias values per projection (offset keeps q/k/v apart
/// so a mis-routed bias is caught).
fn bias_vec(m: usize, offset: f32) -> Vec<f32> {
    (0..m).map(|i| offset + (i as f32) * 0.5 - 1.0).collect()
}

fn check(label: &str, got: &[f32], want: &[f32]) -> bool {
    let mut max_abs = 0f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - want[i]).abs());
    }
    // Exact: gemv=0 then +bias is a single fp32 store of the bias value.
    let pass = max_abs == 0.0;
    println!(
        "  {label:40}  max_abs={max_abs:.3e}  {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Run one dtype: no-bias must be all-zeros; with-bias must equal the bias.
fn run_dtype(
    gpu: &mut Gpu,
    dtype: &str,
    call: impl Fn(
        &mut Gpu,
        &GpuTensor,                                   // a_q
        &GpuTensor,                                   // a_k
        &GpuTensor,                                   // a_v
        &GpuTensor,                                   // x
        &GpuTensor,                                   // y_q
        &GpuTensor,                                   // y_k
        &GpuTensor,                                   // y_v
        Option<(&GpuTensor, &GpuTensor, &GpuTensor)>, // bias
    ) -> HipResult<()>,
) -> HipResult<bool> {
    let a_q = zeros_weight(gpu, Q_M, dtype)?;
    let a_k = zeros_weight(gpu, K_M, dtype)?;
    let a_v = zeros_weight(gpu, V_M, dtype)?;
    let x = gpu.upload_f32(&vec![0.3f32; K], &[K])?;
    let y_q = gpu.zeros(&[Q_M], DType::F32)?;
    let y_k = gpu.zeros(&[K_M], DType::F32)?;
    let y_v = gpu.zeros(&[V_M], DType::F32)?;

    // No-bias → all zeros.
    call(gpu, &a_q, &a_k, &a_v, &x, &y_q, &y_k, &y_v, None)?;
    let nb_q = gpu.download_f32(&y_q)?;
    let nb_k = gpu.download_f32(&y_k)?;
    let nb_v = gpu.download_f32(&y_v)?;
    let p_nb = check(&format!("{dtype} no-bias q"), &nb_q, &vec![0.0; Q_M])
        & check(&format!("{dtype} no-bias k"), &nb_k, &vec![0.0; K_M])
        & check(&format!("{dtype} no-bias v"), &nb_v, &vec![0.0; V_M]);

    // With-bias → equals bias.
    let bq = bias_vec(Q_M, 10.0);
    let bk = bias_vec(K_M, 100.0);
    let bv = bias_vec(V_M, 1000.0);
    let d_bq = gpu.upload_f32(&bq, &[Q_M])?;
    let d_bk = gpu.upload_f32(&bk, &[K_M])?;
    let d_bv = gpu.upload_f32(&bv, &[V_M])?;
    call(
        gpu,
        &a_q,
        &a_k,
        &a_v,
        &x,
        &y_q,
        &y_k,
        &y_v,
        Some((&d_bq, &d_bk, &d_bv)),
    )?;
    let wb_q = gpu.download_f32(&y_q)?;
    let wb_k = gpu.download_f32(&y_k)?;
    let wb_v = gpu.download_f32(&y_v)?;
    let p_wb = check(&format!("{dtype} with-bias q"), &wb_q, &bq)
        & check(&format!("{dtype} with-bias k"), &wb_k, &bk)
        & check(&format!("{dtype} with-bias v"), &wb_v, &bv);

    for t in [a_q, a_k, a_v, x, y_q, y_k, y_v, d_bq, d_bk, d_bv] {
        gpu.free_tensor(t)?;
    }
    Ok(p_nb & p_wb)
}

/// Build `m` rows of HFQ6-G256 weight bytes with FINITE scale/zero (so the GEMV
/// produces finite, deterministic values) and a deterministic 6-bit data pattern.
/// Group layout: f32 scale | f32 zero | 192 data bytes (32 lanes × 6 B).
fn build_hfq6(m: usize, seed: u32) -> Vec<u8> {
    let groups = K / 256;
    let mut out = Vec::with_capacity(m * groups * 200);
    for row in 0..m {
        for g in 0..groups {
            let scale = 0.01f32 + ((row * 3 + g * 5 + seed as usize) % 7) as f32 * 0.001;
            let zero = -0.1f32 + ((row + g) % 5) as f32 * 0.01;
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&zero.to_le_bytes());
            for b in 0..192usize {
                let h = row
                    .wrapping_mul(31)
                    .wrapping_add(g.wrapping_mul(53))
                    .wrapping_add(b.wrapping_mul(7))
                    .wrapping_add(seed as usize);
                out.push((h & 0xFF) as u8);
            }
        }
    }
    out
}

/// Per-row fused_qkv_hfq6g256 (no bias) must be BIT-EXACT with three separate
/// gemv_hfq6g256 calls (same row math, same accumulation order).
fn test_hfq6_fused_vs_gemv(gpu: &mut Gpu) -> HipResult<bool> {
    let wq = gpu.upload_raw(&build_hfq6(Q_M, 1), &[Q_M * (K / 256) * 200])?;
    let wk = gpu.upload_raw(&build_hfq6(K_M, 2), &[K_M * (K / 256) * 200])?;
    let wv = gpu.upload_raw(&build_hfq6(V_M, 3), &[V_M * (K / 256) * 200])?;
    let xv: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let x = gpu.upload_f32(&xv, &[K])?;

    // Fused.
    let yqf = gpu.zeros(&[Q_M], DType::F32)?;
    let ykf = gpu.zeros(&[K_M], DType::F32)?;
    let yvf = gpu.zeros(&[V_M], DType::F32)?;
    gpu.fused_qkv_hfq6g256(&wq, &wk, &wv, &x, &yqf, &ykf, &yvf, Q_M, K_M, V_M, K)?;
    let fq = gpu.download_f32(&yqf)?;
    let fk = gpu.download_f32(&ykf)?;
    let fv = gpu.download_f32(&yvf)?;

    // Three separate gemv_hfq6g256.
    let yqg = gpu.zeros(&[Q_M], DType::F32)?;
    let ykg = gpu.zeros(&[K_M], DType::F32)?;
    let yvg = gpu.zeros(&[V_M], DType::F32)?;
    gpu.gemv_hfq6g256(&wq, &x, &yqg, Q_M, K)?;
    gpu.gemv_hfq6g256(&wk, &x, &ykg, K_M, K)?;
    gpu.gemv_hfq6g256(&wv, &x, &yvg, V_M, K)?;
    let gq = gpu.download_f32(&yqg)?;
    let gk = gpu.download_f32(&ykg)?;
    let gv = gpu.download_f32(&yvg)?;

    let p = check("hfq6 fused_q == gemv_q", &fq, &gq)
        & check("hfq6 fused_k == gemv_k", &fk, &gk)
        & check("hfq6 fused_v == gemv_v", &fv, &gv);

    for t in [wq, wk, wv, x, yqf, ykf, yvf, yqg, ykg, yvg] {
        gpu.free_tensor(t)?;
    }
    Ok(p)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    let mut all = true;

    macro_rules! ptrs {
        ($bias:expr) => {
            match $bias {
                Some((bq, bk, bv)) => (bq.buf.as_ptr(), bk.buf.as_ptr(), bv.buf.as_ptr()),
                None => (
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
            }
        };
    }

    println!("--- hfq4g256 ---");
    all &= run_dtype(
        &mut gpu,
        "hfq4g256",
        |g, aq, ak, av, x, yq, yk, yv, bias| {
            let (bq, bk, bv) = ptrs!(bias);
            g.fused_qkv_hfq4g256_with_bias(aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv)
        },
    )
    .expect("hfq4g256");

    println!("--- mq4g256_lloyd ---");
    all &= run_dtype(
        &mut gpu,
        "mq4g256_lloyd",
        |g, aq, ak, av, x, yq, yk, yv, bias| {
            let (bq, bk, bv) = ptrs!(bias);
            g.fused_qkv_mq4g256_lloyd_with_bias(
                aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv,
            )
        },
    )
    .expect("mq4g256_lloyd");

    println!("--- mq3g256_lloyd ---");
    all &= run_dtype(
        &mut gpu,
        "mq3g256_lloyd",
        |g, aq, ak, av, x, yq, yk, yv, bias| {
            let (bq, bk, bv) = ptrs!(bias);
            g.fused_qkv_mq3g256_lloyd_with_bias(
                aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv,
            )
        },
    )
    .expect("mq3g256_lloyd");

    println!("--- q4k ---");
    all &= run_dtype(&mut gpu, "q4k", |g, aq, ak, av, x, yq, yk, yv, bias| {
        let (bq, bk, bv) = ptrs!(bias);
        g.fused_qkv_q4k_with_bias(aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv)
    })
    .expect("q4k");

    println!("--- q8_0 ---");
    all &= run_dtype(&mut gpu, "q8_0", |g, aq, ak, av, x, yq, yk, yv, bias| {
        let (bq, bk, bv) = ptrs!(bias);
        g.fused_qkv_q8_0_with_bias(aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv)
    })
    .expect("q8_0");

    println!("--- hfq6g256 (Family B, new per-row kernel) ---");
    all &= run_dtype(
        &mut gpu,
        "hfq6g256",
        |g, aq, ak, av, x, yq, yk, yv, bias| {
            let (bq, bk, bv) = ptrs!(bias);
            g.fused_qkv_hfq6g256_with_bias(aq, ak, av, x, yq, yk, yv, Q_M, K_M, V_M, K, bq, bk, bv)
        },
    )
    .expect("hfq6g256");

    // Family B extra: the per-row fused_qkv_hfq6g256 (no bias) must be bit-exact
    // with three separate gemv_hfq6g256 calls on REAL (finite, non-zero) weights —
    // the handover's "validate new fused-per-row == per-row GEMV path" gate.
    println!("--- hfq6g256 fused == 3×gemv_hfq6g256 (finite weights) ---");
    all &= test_hfq6_fused_vs_gemv(&mut gpu).expect("hfq6 fused-vs-gemv");

    if !all {
        eprintln!("\nFAIL: at least one fused_qkv_*_with_bias kernel failed bias parity");
        std::process::exit(1);
    }
    println!("\nALL PASS — every fused_qkv_*_with_bias folds bias bit-exactly (no-bias==0, with-bias==bias)");
}
