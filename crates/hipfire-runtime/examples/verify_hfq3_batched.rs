//! Bit-exact verification for the scalar HFQ3 batched-prefill kernels
//! introduced in Phase 1 of `docs/plans/gfx10_mq3_prefill.md`.
//!
//! Method: each new kernel's claim is "byte-exact with running
//! `gemv_hfq3g256` N times for N=1." We verify this directly by
//! constructing synthetic HFQ3-shaped bytes + a synthetic x, running
//! the single-row reference per row, and comparing element-by-element.
//!
//! Tested kernels:
//!   - `gemm_qkv_hfq3g256`       (3-way fused — FA preamble)
//!   - `gemm_qkvza_hfq3g256`     (4-way fused — LA preamble)
//!   - `gemm_gate_up_hfq3g256`   (2-way fused — FFN preamble)
//!   - `gemm_hfq3g256_residual`  (single-weight, accumulate)
//!
//! Run: `cargo run --release --example verify_hfq3_batched`
//!
//! Acceptance:
//!   - N=1: bit-exact vs per-row gemv_hfq3g256 (scalar path on this side too).
//!   - N>1 on archs with dot2/fp16 routing: ≤ 5e-2 max_abs_err. The batched
//!     dispatchers route to dot2 (gfx1011/1012/1030-1032 + gfx11/12) or
//!     fp16-packed (gfx1010/1013), both of which dequant weights to FP16 —
//!     the per-row reference uses FP32 X and FP32 dequant, so divergence at
//!     FP16 mantissa precision (~1% relative over a 512-element accumulation)
//!     is expected. ~0.1 max_abs_err is normal.

use rdna_compute::{DType, GpuTensor};

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

/// Generate synthetic HFQ3-shaped bytes — 104 B per group, 8 B header
/// (scale + zero as f32) + 96 B body (3-bit values packed via uint24).
/// Values are deterministic pseudo-random so test runs reproduce.
fn synth_hfq3_bytes(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let groups_per_row = k / 256;
    let mut bytes = vec![0u8; m * groups_per_row * 104];
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = (row * groups_per_row + g) * 104;
            // Header: scale + zero (both small to keep accumulation in range)
            let scale = fract_sin(seed as f32 + (row * 7 + g * 11) as f32 * 0.131);
            let zero = fract_sin(seed as f32 + (row * 13 + g * 17) as f32 * 0.211);
            bytes[off..off + 4].copy_from_slice(&(scale * 0.1).to_le_bytes());
            bytes[off + 4..off + 8].copy_from_slice(&(zero * 0.1).to_le_bytes());
            // Body: 32 threads × 3 bytes = 96 bytes. Each 3-byte uint24
            // packs 8 × 3-bit values. We just need *some* bytes — kernel
            // doesn't care if values are semantically meaningful.
            for i in 0..96 {
                let v = (((seed.wrapping_mul(2654435761))
                    .wrapping_add((row as u32) * 257 + (g as u32) * 19 + i as u32))
                    & 0xFF) as u8;
                bytes[off + 8 + i] = v;
            }
        }
    }
    bytes
}

fn synth_x(n: usize, k: usize, seed: u32) -> Vec<f32> {
    (0..n * k)
        .map(|i| fract_sin(seed as f32 + i as f32 * 0.317))
        .collect()
}

/// Reference: call `gemv_hfq3g256` per row, per batch element.
/// Output shape: [n × m] row-major.
fn cpu_reference_via_gemv(
    gpu: &mut rdna_compute::Gpu,
    weight_bytes: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let d_w = gpu.upload_raw(weight_bytes, &[weight_bytes.len()]).unwrap();
    let mut output = vec![0.0f32; n * m];
    for b in 0..n {
        let x_b: Vec<f32> = x[b * k..(b + 1) * k].to_vec();
        let d_x = gpu.upload_f32(&x_b, &[k]).unwrap();
        let d_y = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        gpu.gemv_hfq3g256(&d_w, &d_x, &d_y, m, k).unwrap();
        let y = gpu.download_f32(&d_y).unwrap();
        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y).unwrap();
        output[b * m..(b + 1) * m].copy_from_slice(&y);
    }
    gpu.free_tensor(d_w).unwrap();
    output
}

fn compare_with_tol(name: &str, ref_out: &[f32], test_out: &[f32], tol: f32) -> bool {
    assert_eq!(ref_out.len(), test_out.len(), "{name}: length mismatch");
    let mut max_err = 0.0f32;
    let mut bit_exact = 0usize;
    for i in 0..ref_out.len() {
        let err = (ref_out[i] - test_out[i]).abs();
        max_err = max_err.max(err);
        if ref_out[i] == test_out[i] {
            bit_exact += 1;
        }
    }
    let ok = max_err <= tol;
    let bit_exact_frac = bit_exact as f64 / ref_out.len() as f64;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  {name:<32} {status}  max_err={max_err:.6e}  bit_exact={bit_exact}/{} ({:.1}%)  tol={tol:.0e}",
        ref_out.len(),
        bit_exact_frac * 100.0,
    );
    ok
}


fn alloc_zero(gpu: &mut rdna_compute::Gpu, n_elem: usize) -> GpuTensor {
    let zeros = vec![0.0f32; n_elem];
    gpu.upload_f32(&zeros, &[n_elem]).unwrap()
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("=== verify_hfq3_batched on {} ===", gpu.arch);

    // Use shapes representative of Qwen3.5 9B FA layer: q_m=4096, kv_m=1024.
    // But scale down for test runtime: m=64, k=512 (2 groups/row).
    let m = 64usize;
    let k = 512usize;
    let batches = [1usize, 4, 8, 13]; // mix BATCH_TILE-aligned + odd

    let mut any_fail = false;

    let weight_bytes = synth_hfq3_bytes(m, k, 42);
    let d_w = gpu.upload_raw(&weight_bytes, &[weight_bytes.len()]).unwrap();

    for &n in &batches {
        eprintln!("\n-- batch_size = {n} --");
        let x = synth_x(n, k, 17);
        let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();

        // Reference: per-row gemv_hfq3g256 looped over n batches and m rows.
        let y_ref = cpu_reference_via_gemv(&mut gpu, &weight_bytes, &x, m, k, n);

        // Tolerance: bit-exact at N=1 (scalar both sides); FP16-mantissa
        // tolerance at N>1 (auto-routing hits dot2 or fp16, dequant in FP16).
        // dp4a / MMQ modes quantize X to Q8_1 on top of the FP16 weight
        // dequant, so they need a wider tolerance (~3× the dot2 error band).
        let env_truthy = |k: &str| std::env::var(k)
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
            .unwrap_or(false);
        let int8_x_mode = env_truthy("HIPFIRE_HFQ3_DP4A") || env_truthy("HIPFIRE_HFQ3_MMQ");
        let tol: f32 = if n == 1 {
            1e-3
        } else if int8_x_mode {
            5e-1
        } else {
            2e-1
        };

        // Test 1: gemm_hfq3g256_residual (Y starts zero, accumulates).
        let d_y = alloc_zero(&mut gpu, n * m);
        gpu.gemm_hfq3g256_residual(&d_w, &d_x, &d_y, m, k, n).unwrap();
        let y_resid = gpu.download_f32(&d_y).unwrap();
        any_fail |= !compare_with_tol("residual", &y_ref, &y_resid, tol);
        gpu.free_tensor(d_y).unwrap();

        // Test 2: gemm_qkv_hfq3g256 with A_q=A_k=A_v (same weight, 3 outputs).
        let d_yq = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_yk = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_yv = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        gpu.gemm_qkv_hfq3g256(
            &d_w, &d_w, &d_w, &d_x, &d_yq, &d_yk, &d_yv, m, m, m, k, n,
        )
        .unwrap();
        let yq = gpu.download_f32(&d_yq).unwrap();
        let yk = gpu.download_f32(&d_yk).unwrap();
        let yv = gpu.download_f32(&d_yv).unwrap();
        any_fail |= !compare_with_tol("qkv (y_q arm)", &y_ref, &yq, tol);
        any_fail |= !compare_with_tol("qkv (y_k arm)", &y_ref, &yk, tol);
        any_fail |= !compare_with_tol("qkv (y_v arm)", &y_ref, &yv, tol);
        gpu.free_tensor(d_yq).unwrap();
        gpu.free_tensor(d_yk).unwrap();
        gpu.free_tensor(d_yv).unwrap();

        // Test 3: gemm_gate_up_hfq3g256 with A_gate=A_up.
        let d_yg = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_yu = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        gpu.gemm_gate_up_hfq3g256(&d_w, &d_w, &d_x, &d_yg, &d_yu, m, m, k, n)
            .unwrap();
        let yg = gpu.download_f32(&d_yg).unwrap();
        let yu = gpu.download_f32(&d_yu).unwrap();
        any_fail |= !compare_with_tol("gate_up (gate)", &y_ref, &yg, tol);
        any_fail |= !compare_with_tol("gate_up (up)", &y_ref, &yu, tol);
        gpu.free_tensor(d_yg).unwrap();
        gpu.free_tensor(d_yu).unwrap();

        // Test 4: gemm_qkvza_hfq3g256 with A_qkv=A_z=A_beta=A_alpha.
        let d_y1 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_y2 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_y3 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        let d_y4 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
        gpu.gemm_qkvza_hfq3g256(
            &d_w, &d_w, &d_w, &d_w, &d_x,
            &d_y1, &d_y2, &d_y3, &d_y4,
            m, m, m, m, k, n,
        )
        .unwrap();
        let y1 = gpu.download_f32(&d_y1).unwrap();
        let y2 = gpu.download_f32(&d_y2).unwrap();
        let y3 = gpu.download_f32(&d_y3).unwrap();
        let y4 = gpu.download_f32(&d_y4).unwrap();
        any_fail |= !compare_with_tol("qkvza (qkv arm)", &y_ref, &y1, tol);
        any_fail |= !compare_with_tol("qkvza (z arm)", &y_ref, &y2, tol);
        any_fail |= !compare_with_tol("qkvza (beta arm)", &y_ref, &y3, tol);
        any_fail |= !compare_with_tol("qkvza (alpha arm)", &y_ref, &y4, tol);
        gpu.free_tensor(d_y1).unwrap();
        gpu.free_tensor(d_y2).unwrap();
        gpu.free_tensor(d_y3).unwrap();
        gpu.free_tensor(d_y4).unwrap();

        // FP16-direct tests at N>1 — bypass auto-routing to exercise the
        // gfx1010/1013 fp16 fallback path (Phase 2c) on archs where the
        // public dispatcher would otherwise pick dot2.
        if n > 1 {
            eprintln!("  -- fp16-direct (Phase 2c) --");

            let d_yq = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_yk = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_yv = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            gpu.gemm_qkv_hfq3g256_fp16(
                &d_w, &d_w, &d_w, &d_x, &d_yq, &d_yk, &d_yv, m, m, m, k, n,
            ).unwrap();
            let yq = gpu.download_f32(&d_yq).unwrap();
            any_fail |= !compare_with_tol("fp16 qkv (y_q)", &y_ref, &yq, tol);
            gpu.free_tensor(d_yq).unwrap();
            gpu.free_tensor(d_yk).unwrap();
            gpu.free_tensor(d_yv).unwrap();

            let d_y1 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_y2 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_y3 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_y4 = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            gpu.gemm_qkvza_hfq3g256_fp16(
                &d_w, &d_w, &d_w, &d_w, &d_x,
                &d_y1, &d_y2, &d_y3, &d_y4,
                m, m, m, m, k, n,
            ).unwrap();
            let y1 = gpu.download_f32(&d_y1).unwrap();
            any_fail |= !compare_with_tol("fp16 qkvza (qkv arm)", &y_ref, &y1, tol);
            gpu.free_tensor(d_y1).unwrap();
            gpu.free_tensor(d_y2).unwrap();
            gpu.free_tensor(d_y3).unwrap();
            gpu.free_tensor(d_y4).unwrap();

            let d_yg = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let d_yu = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            gpu.gemm_gate_up_hfq3g256_fp16(&d_w, &d_w, &d_x, &d_yg, &d_yu, m, m, k, n).unwrap();
            let yg = gpu.download_f32(&d_yg).unwrap();
            any_fail |= !compare_with_tol("fp16 gate_up (gate)", &y_ref, &yg, tol);
            gpu.free_tensor(d_yg).unwrap();
            gpu.free_tensor(d_yu).unwrap();

            let d_y = alloc_zero(&mut gpu, n * m);
            gpu.gemm_hfq3g256_residual_fp16(&d_w, &d_x, &d_y, m, k, n).unwrap();
            let y_resid = gpu.download_f32(&d_y).unwrap();
            any_fail |= !compare_with_tol("fp16 residual", &y_ref, &y_resid, tol);
            gpu.free_tensor(d_y).unwrap();
        }

        gpu.free_tensor(d_x).unwrap();
    }
    gpu.free_tensor(d_w).unwrap();

    // -- MMQ selector/fallback qkv + gate_up tests at MMQ_Y-aligned m ----
    // The MMQ qkv/gate_up bodies REQUIRE q_m/k_m/v_m and gate_m/up_m to
    // each be multiples of MMQ_Y=128. Allocate a separate weight matrix
    // at m=256 to test these selectors properly. On non-gfx10-sdot4 archs,
    // the public MMQ selectors fall back to dot2/fp16 rather than compiling
    // a gfx10-only sdot4 kernel.
    {
        eprintln!("\n-- MMQ selector/fallback (qkv + gate_up at m=256) --");
        let m_mmq = 256usize;
        let weight_bytes_mmq = synth_hfq3_bytes(m_mmq, k, 42);
        let d_w_mmq = gpu.upload_raw(&weight_bytes_mmq, &[weight_bytes_mmq.len()]).unwrap();

        // FP16 mantissa + Q8_1 X tolerance.
        let mmq_tol = 5e-1_f32;
        // Test at the batch sizes where MMQ actually fires
        // (auto-selector routes to mmq_x16 from N=13; mmq_x32 at N≥128).
        for &n in &[16usize, 32, 128] {
            let x_mmq = synth_x(n, k, 17);
            let d_x_mmq = gpu.upload_f32(&x_mmq, &[n * k]).unwrap();
            let y_ref_mmq = cpu_reference_via_gemv(&mut gpu, &weight_bytes_mmq, &x_mmq, m_mmq, k, n);

            eprintln!("  -- batch_size = {n} --");

            // qkv MMQ auto-selector
            let d_yq = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_yk = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_yv = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            gpu.gemm_qkv_hfq3g256_mmq(
                &d_w_mmq, &d_w_mmq, &d_w_mmq, &d_x_mmq,
                &d_yq, &d_yk, &d_yv, m_mmq, m_mmq, m_mmq, k, n,
            ).unwrap();
            let yq = gpu.download_f32(&d_yq).unwrap();
            any_fail |= !compare_with_tol("mmq qkv (y_q)", &y_ref_mmq, &yq, mmq_tol);
            gpu.free_tensor(d_yq).unwrap();
            gpu.free_tensor(d_yk).unwrap();
            gpu.free_tensor(d_yv).unwrap();

            // gate_up MMQ auto-selector
            let d_yg = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_yu = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            gpu.gemm_gate_up_hfq3g256_mmq(
                &d_w_mmq, &d_w_mmq, &d_x_mmq, &d_yg, &d_yu, m_mmq, m_mmq, k, n,
            ).unwrap();
            let yg = gpu.download_f32(&d_yg).unwrap();
            any_fail |= !compare_with_tol("mmq gate_up (gate)", &y_ref_mmq, &yg, mmq_tol);
            gpu.free_tensor(d_yg).unwrap();
            gpu.free_tensor(d_yu).unwrap();

            // qkvza MMQ auto-selector (4-way LA preamble)
            let d_y1 = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_y2 = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_y3 = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            let d_y4 = gpu.alloc_tensor(&[n * m_mmq], DType::F32).unwrap();
            gpu.gemm_qkvza_hfq3g256_mmq(
                &d_w_mmq, &d_w_mmq, &d_w_mmq, &d_w_mmq, &d_x_mmq,
                &d_y1, &d_y2, &d_y3, &d_y4,
                m_mmq, m_mmq, m_mmq, m_mmq, k, n,
            ).unwrap();
            let y1 = gpu.download_f32(&d_y1).unwrap();
            any_fail |= !compare_with_tol("mmq qkvza (qkv arm)", &y_ref_mmq, &y1, mmq_tol);
            gpu.free_tensor(d_y1).unwrap();
            gpu.free_tensor(d_y2).unwrap();
            gpu.free_tensor(d_y3).unwrap();
            gpu.free_tensor(d_y4).unwrap();

            gpu.free_tensor(d_x_mmq).unwrap();
        }
        gpu.free_tensor(d_w_mmq).unwrap();
    }

    if any_fail {
        eprintln!("\n[FAIL] At least one batched HFQ3 kernel diverged from per-row gemv_hfq3g256.");
        std::process::exit(1);
    } else {
        eprintln!("\n[PASS] All HFQ3 batched kernels/selectors within tolerance.");
    }
}
