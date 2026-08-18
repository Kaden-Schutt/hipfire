// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gate 3 slice GpuLinear — first runtime validation of
//! `gemm_bf16_mfma_gfx942` plus `parent/linear.rs` residency path.
//!
//! Must run on gfx942 (mi300x). Dev hosts cannot execute the kernel.
//!
//! ```text
//! cargo run --release -p hipfire-arch-deepseek4 --example ds4_parent_gpu_linear_smoke
//! ```

use hipfire_arch_deepseek4::parent::linear::{
    parent_linear_dense, parent_linear_expert, ParentDenseWeight, ParentExpertWeight,
};
use hipfire_arch_deepseek4::parent::{
    Ds4ParentBackend, ParentQuantConfig, PARENT_EXPERT_DTYPE, PARENT_MODEL_TYPE,
    PARENT_QUANT_METHOD, PARENT_SCALE_FMT, PARENT_WEIGHT_BLOCK, PARENT_WEIGHT_FMT,
};
use hipfire_runtime::model_source::{ModelSource, QuantConfig, TensorInfo};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

// ── BF16 helpers ─────────────────────────────────────────────────────────────

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7fc0;
    }
    let lsb = (bits >> 16) & 1;
    let lower = bits & 0xffff;
    let round_bit = (lower >> 15) & 1;
    let sticky = if (lower & 0x7fff) != 0 { 1 } else { 0 };
    let mut top = bits >> 16;
    if round_bit == 1 && (sticky == 1 || lsb == 1) {
        top = top.wrapping_add(1);
    }
    top as u16
}

#[inline]
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn pack_f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

fn upload_bf16(gpu: &Gpu, vals: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    let bytes = pack_f32_to_bf16_bytes(vals);
    let t = gpu
        .upload_raw(&bytes, shape)
        .map_err(|e| format!("upload_bf16: {e:?}"))?;
    // Retag as BF16 (upload_raw stamps Raw).
    Ok(GpuTensor {
        buf: t.buf,
        shape: t.shape,
        dtype: DType::BF16,
    })
}

fn free(gpu: &mut Gpu, t: GpuTensor) {
    let _ = gpu.free_tensor(t);
}

// ── F32 reference GEMM: D[batch,M] = B[batch,K] @ A[M,K]^T ───────────────────

fn gemm_f32_ref(a: &[f32], b: &[f32], m: usize, k: usize, batch: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; batch * m];
    for bi in 0..batch {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[bi * k + ki];
            }
            d[bi * m + mi] = acc;
        }
    }
    d
}

fn rel_stats(gpu: &[f32], refer: &[f32]) -> (f64, f64, f64, f64) {
    assert_eq!(gpu.len(), refer.len());
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_rel = 0.0f64;
    let n = gpu.len() as f64;
    for (&g, &r) in gpu.iter().zip(refer.iter()) {
        let g = g as f64;
        let r = r as f64;
        let abs = (g - r).abs();
        let denom = r.abs().max(1e-6);
        let rel = abs / denom;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        sum_abs += abs;
        sum_rel += rel;
    }
    (max_abs, max_rel, sum_abs / n, sum_rel / n)
}

fn mean_signed_z(gpu: &[f32], refer: &[f32]) -> f64 {
    let n = gpu.len() as f64;
    let mut mean = 0.0f64;
    for (&g, &r) in gpu.iter().zip(refer.iter()) {
        mean += (g as f64) - (r as f64);
    }
    mean /= n;
    let mut var = 0.0f64;
    for (&g, &r) in gpu.iter().zip(refer.iter()) {
        let d = ((g as f64) - (r as f64)) - mean;
        var += d * d;
    }
    let std = (var / n).sqrt().max(1e-30);
    mean / std
}

fn fill_bf16_grid(n: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as i32 - 8;
        let v = (x as f32) * 0.25; // multiples of 0.25 in [-2, 2], BF16-exact
        out.push(bf16_bits_to_f32(f32_to_bf16_bits(v)));
    }
    out
}

// ── Part 1: raw BF16 MFMA GEMM vs F32 ref (+ rocBLAS if present) ─────────────

struct GemmCase {
    name: &'static str,
    /// Weight rows (output features) — kernel M.
    m: usize,
    k: usize,
    /// Activation rows (tokens) — kernel batch / conventional N.
    batch: usize,
    bench: bool,
}

fn run_gemm_case(gpu: &mut Gpu, c: &GemmCase) -> Result<(), String> {
    println!(
        "  case {name}: weight M={m} K={k} batch N={batch}",
        name = c.name,
        m = c.m,
        k = c.k,
        batch = c.batch
    );

    let a_f = fill_bf16_grid(c.m * c.k, 0xA11E);
    let b_f = fill_bf16_grid(c.batch * c.k, 0xB0B0);
    // F32 ref on the already-BF16-rounded values (matches kernel inputs).
    let refer = gemm_f32_ref(&a_f, &b_f, c.m, c.k, c.batch);

    let a = upload_bf16(gpu, &a_f, &[c.m, c.k])?;
    let b = upload_bf16(gpu, &b_f, &[c.batch, c.k])?;
    let d = gpu
        .zeros(&[c.batch, c.m], DType::F32)
        .map_err(|e| format!("zeros D: {e:?}"))?;

    // Warmup / JIT
    gpu.gemm_bf16_mfma_gfx942(&a.buf, &b.buf, &d.buf, c.m, c.k, c.batch)
        .map_err(|e| format!("mfma launch: {e:?}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;

    let t0 = Instant::now();
    let iters = if c.bench { 20 } else { 1 };
    for _ in 0..iters {
        gpu.gemm_bf16_mfma_gfx942(&a.buf, &b.buf, &d.buf, c.m, c.k, c.batch)
            .map_err(|e| format!("mfma launch: {e:?}"))?;
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let elapsed = t0.elapsed();

    let gpu_out = gpu
        .download_f32(&d)
        .map_err(|e| format!("download: {e:?}"))?;
    let (max_abs, max_rel, mean_abs, mean_rel) = rel_stats(&gpu_out, &refer);
    let bias_z = mean_signed_z(&gpu_out, &refer);

    let flops = 2.0 * (c.m as f64) * (c.k as f64) * (c.batch as f64) * (iters as f64);
    let tflops = flops / elapsed.as_secs_f64() / 1e12;

    println!(
        "    MFMA vs F32: max_abs={max_abs:.3e} max_rel={max_rel:.3e} mean_abs={mean_abs:.3e} mean_rel={mean_rel:.3e} bias_z={bias_z:.3e}"
    );
    println!(
        "    throughput: {tflops:.2} TFLOP/s  ({iters} iter, {ms:.2} ms wall)",
        ms = elapsed.as_secs_f64() * 1e3
    );

    // rocBLAS oracle when available
    if gpu.rocblas.is_some() {
        let d_rb = gpu
            .zeros(&[c.batch, c.m], DType::F32)
            .map_err(|e| format!("zeros roc D: {e:?}"))?;
        let _ = gpu.rocblas_gemm_bf16_prefill(&a.buf, &b.buf, &d_rb.buf, c.m, c.batch, c.k);
        gpu.hip.device_synchronize().ok();
        let t1 = Instant::now();
        let r_iters = if c.bench { 20 } else { 1 };
        for _ in 0..r_iters {
            gpu.rocblas_gemm_bf16_prefill(&a.buf, &b.buf, &d_rb.buf, c.m, c.batch, c.k)
                .map_err(|e| format!("rocblas launch: {e:?}"))?;
        }
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("roc sync: {e:?}"))?;
        let r_elapsed = t1.elapsed();
        let rb_out = gpu
            .download_f32(&d_rb)
            .map_err(|e| format!("roc download: {e:?}"))?;
        let (rmax_abs, rmax_rel, _, rmean_rel) = rel_stats(&rb_out, &refer);
        let (g2r_abs, g2r_rel, _, _) = rel_stats(&gpu_out, &rb_out);
        let r_flops = 2.0 * (c.m as f64) * (c.k as f64) * (c.batch as f64) * (r_iters as f64);
        let r_tflops = r_flops / r_elapsed.as_secs_f64() / 1e12;
        println!(
            "    rocBLAS vs F32: max_abs={rmax_abs:.3e} max_rel={rmax_rel:.3e} mean_rel={rmean_rel:.3e}  | {r_tflops:.2} TFLOP/s"
        );
        println!("    MFMA vs rocBLAS: max_abs={g2r_abs:.3e} max_rel={g2r_rel:.3e}");
        free(gpu, d_rb);

        if g2r_rel > 1e-2 && g2r_abs > 1e-2 {
            free(gpu, a);
            free(gpu, b);
            free(gpu, d);
            return Err(format!(
                "FINDING: MFMA disagrees with rocBLAS at {}: max_rel={g2r_rel:.3e} max_abs={g2r_abs:.3e}",
                c.name
            ));
        }
    } else {
        println!("    rocBLAS: not loaded on this Gpu");
    }

    if c.m <= 4 && c.batch <= 4 && c.k <= 16 {
        println!("    ref[0..]: {:?}", &refer[..refer.len().min(8)]);
        println!("    gpu[0..]: {:?}", &gpu_out[..gpu_out.len().min(8)]);
    }

    // Catastrophic-error gate (not a tuned tolerance — BF16 noise budget × 20).
    let rel_budget = ((c.k as f64).sqrt() * (1.0 / 64.0)).max(5e-3);
    if max_rel > rel_budget * 20.0 && max_abs > 1e-2 {
        free(gpu, a);
        free(gpu, b);
        free(gpu, d);
        return Err(format!(
            "FINDING: MFMA vs F32 ref catastrophic at {}: max_rel={max_rel:.3e} (budget≈{rel_budget:.3e}) max_abs={max_abs:.3e}",
            c.name
        ));
    }

    free(gpu, a);
    free(gpu, b);
    free(gpu, d);
    Ok(())
}

// ── Part 2: ParentDenseWeight / ParentExpertWeight residency + linear ────────

struct MetaSource {
    meta: String,
}

impl ModelSource for MetaSource {
    fn metadata_json(&self) -> &str {
        &self.meta
    }
    fn arch_id(&self) -> u32 {
        0
    }
    fn quant_config(&self) -> Option<&QuantConfig> {
        None
    }
    fn tensor_data(&self, _name: &str) -> Option<(&TensorInfo, &[u8])> {
        None
    }
    fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
        None
    }
    fn tensor_names(&self) -> Vec<&str> {
        Vec::new()
    }
    fn path(&self) -> &Path {
        Path::new("/tmp/ds4-parent-gpu-linear-smoke")
    }
}

fn parent_meta_json() -> String {
    serde_json::json!({
        "config": {
            "model_type": PARENT_MODEL_TYPE,
            "expert_dtype": PARENT_EXPERT_DTYPE,
            "num_hidden_layers": 43,
            "num_hash_layers": 3,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "compress_ratios": vec![0u32; 46],
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": PARENT_WEIGHT_FMT,
                "quant_method": PARENT_QUANT_METHOD,
                "scale_fmt": PARENT_SCALE_FMT,
                "weight_block_size": PARENT_WEIGHT_BLOCK,
            }
        }
    })
    .to_string()
}

fn e4m3_code_for_quarter(q: i8) -> u8 {
    match q {
        0 => 0x00,
        1 => 0x38,  // 1.0
        -1 => 0xB8, // -1.0
        2 => 0x40,  // 2.0
        -2 => 0xC0,
        _ => 0x30, // 0.5
    }
}

fn run_residency_and_linear(gpu: &mut Gpu) -> Result<(), String> {
    let src = MetaSource {
        meta: parent_meta_json(),
    };
    let _cfg = ParentQuantConfig::from_metadata_json(src.metadata_json())?;
    let (backend, _) = Ds4ParentBackend::admit(&src, gpu)?;

    // Dense: real parent shape wq_b = [32768, 1024]
    let n_d = 32768usize;
    let k_d = 1024usize;
    let codes_d: Vec<u8> = (0..n_d * k_d)
        .map(|i| e4m3_code_for_quarter(((i % 5) as i8) - 2))
        .collect();
    let s_rows = n_d.div_ceil(128);
    let s_cols = k_d.div_ceil(128);
    let scales_d = vec![127u8; s_rows * s_cols]; // 2^0 = 1.0

    let (free0, total) = gpu
        .hip
        .get_vram_info()
        .map_err(|e| format!("vram: {e:?}"))?;
    let w_dense = ParentDenseWeight::decode_resident(gpu, backend, &codes_d, &scales_d, n_d, k_d)?;
    let (free1, _) = gpu
        .hip
        .get_vram_info()
        .map_err(|e| format!("vram: {e:?}"))?;
    let dense_resident = w_dense.resident_bytes();
    let dense_expected = n_d * k_d * 2;
    let vram_delta = free0.saturating_sub(free1);
    println!(
        "  DenseWeight [{n_d},{k_d}]: resident_bytes={dense_resident} (expect {dense_expected} = 2× stored {})  vram_delta≈{vram_delta}  total_vram={total}",
        n_d * k_d
    );
    if dense_resident != dense_expected {
        return Err(format!(
            "dense resident_bytes {dense_resident} != expected {dense_expected}"
        ));
    }

    // Expert: logical [2048, 4096], packed codes [2048, 2048], scales [2048, 128]
    let n_e = 2048usize;
    let k_e = 4096usize;
    let codes_e = vec![0x11u8; n_e * (k_e / 2)]; // nibbles 1,1 → 0.5,0.5
    let scales_e = vec![127u8; n_e * (k_e / 32)];
    let stored_e = codes_e.len() + scales_e.len();
    let w_exp =
        ParentExpertWeight::upload_compressed(gpu, backend, &codes_e, &scales_e, n_e, k_e)?;
    let exp_bytes = w_exp.compressed_bytes();
    println!(
        "  ExpertWeight logical[{n_e},{k_e}]: compressed_bytes={exp_bytes} (expect {stored_e} = 1× stored codes+scales)"
    );
    if exp_bytes != stored_e {
        return Err(format!(
            "expert compressed_bytes {exp_bytes} != expected {stored_e}"
        ));
    }

    let scratch = gpu
        .alloc_tensor(&[n_e, k_e], DType::BF16)
        .map_err(|e| format!("expert scratch: {e:?}"))?;
    w_exp.decode_into(gpu, &scratch)?;

    // Small dense linear smoke (not the full 32k×1k GEMM — covered in Part 1).
    let n_s = 64usize;
    let k_s = 128usize;
    let codes_s: Vec<u8> = (0..n_s * k_s)
        .map(|i| e4m3_code_for_quarter(if i % 2 == 0 { 1 } else { -1 }))
        .collect();
    let scales_s = vec![127u8; n_s.div_ceil(128) * k_s.div_ceil(128)];
    let w_small =
        ParentDenseWeight::decode_resident(gpu, backend, &codes_s, &scales_s, n_s, k_s)?;

    let m = 32usize;
    let x_f = fill_bf16_grid(m * k_s, 0x0C01);
    let x = upload_bf16(gpu, &x_f, &[m, k_s])?;
    let out = gpu
        .zeros(&[m, n_s], DType::F32)
        .map_err(|e| format!("out: {e:?}"))?;
    parent_linear_dense(gpu, backend, &w_small, &x, m, &out)?;
    let y = gpu
        .download_f32(&out)
        .map_err(|e| format!("download y: {e:?}"))?;
    let y_norm: f64 = y
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    println!(
        "  parent_linear_dense m={m} n={n_s} k={k_s}: ||y||_2={y_norm:.4}  y[0..4]={:?}",
        &y[..4]
    );
    if !y.iter().all(|v| v.is_finite()) {
        return Err("parent_linear_dense produced non-finite output".into());
    }

    let x2_f = fill_bf16_grid(m * k_e, 0x0E01);
    let x2 = upload_bf16(gpu, &x2_f, &[m, k_e])?;
    let out2 = gpu
        .zeros(&[m, n_e], DType::F32)
        .map_err(|e| format!("out2: {e:?}"))?;
    parent_linear_expert(gpu, backend, &scratch, n_e, k_e, &x2, m, &out2)?;
    let y2 = gpu
        .download_f32(&out2)
        .map_err(|e| format!("download y2: {e:?}"))?;
    let y2_norm: f64 = y2
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    println!(
        "  parent_linear_expert m={m} n={n_e} k={k_e}: ||y||_2={y2_norm:.4}  y[0..4]={:?}",
        &y2[..4]
    );
    if !y2.iter().all(|v| v.is_finite()) {
        return Err("parent_linear_expert produced non-finite output".into());
    }

    free(gpu, x);
    free(gpu, out);
    free(gpu, x2);
    free(gpu, out2);
    free(gpu, scratch);
    // Weight structs hold private GpuTensors; process exit reclaims VRAM.
    std::mem::forget(w_dense);
    std::mem::forget(w_small);
    std::mem::forget(w_exp);

    println!("  residency + linear smoke: OK");
    Ok(())
}

fn main() -> ExitCode {
    println!("ds4_parent_gpu_linear_smoke — BF16 MFMA GEMM + parent linear path");
    println!("host target: gfx942 (MI300X)\n");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("FATAL: Gpu::init failed: {e:?}");
            return ExitCode::from(2);
        }
    };
    println!("gpu arch: {}  rocblas={}", gpu.arch, gpu.rocblas.is_some());
    if !gpu.arch_caps.is_gfx942() {
        eprintln!("FATAL: requires gfx942; got {}", gpu.arch);
        return ExitCode::from(2);
    }

    let mut failed = false;

    println!("\n── Part 1: gemm_bf16_mfma_gfx942 vs F32 ref ──");
    // Kernel args: A[M,K] weights, B[batch,K] acts, D[batch,M].
    // Acceptance shapes use m=batch tokens, n=weight rows in one wording and
    // the reverse in another — cover both real parent orientations.
    let cases = [
        GemmCase {
            name: "hand_checkable",
            m: 2,
            k: 4,
            batch: 3,
            bench: false,
        },
        GemmCase {
            name: "k_not_mult_of_128",
            m: 32,
            k: 200,
            batch: 32,
            bench: false,
        },
        GemmCase {
            name: "m_not_mult_of_32",
            m: 40,
            k: 128,
            batch: 48,
            bench: false,
        },
        // wq_a-like: weight [1024,4096], batch 32  (acceptance m=32,n=1024,k=4096)
        GemmCase {
            name: "parent_wq_a_m1024_k4096_b32",
            m: 1024,
            k: 4096,
            batch: 32,
            bench: true,
        },
        // wq_b-like thin batch: weight [32768,1024], batch 32
        GemmCase {
            name: "parent_wq_b_m32768_k1024_b32",
            m: 32768,
            k: 1024,
            batch: 32,
            bench: true,
        },
        // Large-batch stress named in acceptance: n=32768,k=1024,m=32
        // (batch=32768, weight rows=32, k=1024)
        GemmCase {
            name: "throughput_n32768_k1024_m32",
            m: 32,
            k: 1024,
            batch: 32768,
            bench: true,
        },
    ];
    for c in &cases {
        match run_gemm_case(&mut gpu, c) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("  FAIL {}: {e}", c.name);
                failed = true;
            }
        }
    }

    println!("\n── Part 2: ParentDense/Expert residency + linear ──");
    match run_residency_and_linear(&mut gpu) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("  FAIL residency/linear: {e}");
            failed = true;
        }
    }

    if failed {
        println!("\nRESULT: FAIL");
        ExitCode::from(1)
    } else {
        println!("\nRESULT: PASS");
        ExitCode::SUCCESS
    }
}
