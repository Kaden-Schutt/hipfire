// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S3-f16-projection-inputs gate: exact-FP16 projection-input producers on
//! gfx1100.
//!
//! For N in {1,2,8,16} x hidden K in {4096,5120} x AWQ {absent, present}:
//!  1. F16 memcmp: `fused_rmsnorm_rotate_mq[_awq]_f16_batched` bytes vs the
//!     old F32 producer + `cast_f32_to_f16` (same `(_Float16)` cast body the
//!     GEMM-path `convert_f32_to_f16` inlines) — must be bit-identical.
//!  2. Projection-output memcmp: old `*_mq4g256v2_wmma` (F32 x) vs new
//!     `*_wmma_f16` (candidate F16 x) for qkvza / qkv / gate_up with
//!     synthetic MQ4V2 weights — F32 outputs must be bit-identical.
//! Also: the `llama::fused_rmsnorm_rotate_mq_f16_batched_for` wrapper routes
//! AWQ identically (byte-match vs the direct producer call), and a non-F16
//! `x_f16` input is rejected with `Err` (never silently converted).
//!
//! On any non-gfx1100 arch the harness SKIPs cleanly (exit 0, no GPU work).

use hipfire_runtime::llama::{fused_rmsnorm_rotate_mq_f16_batched_for, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
const EPS: f32 = 1e-6;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
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
                    f32_to_f16_bits_round(step)
                };
                let z_bits = f32_to_f16_bits_round(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_bits_to_f32(s_bits);
                let z_rt = f16_bits_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                let lo_q = codes[2 * i] & 0xF;
                let hi_q = codes[2 * i + 1] & 0xF;
                blob[dst + 8 + i] = lo_q | (hi_q << 4);
            }
        }
    }
    blob
}

/// Host-side round-to-nearest-even f32->f16 (packing only — the GPU oracle
/// for producer bytes is `cast_f32_to_f16`, never this function).
fn f32_to_f16_bits_round(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x007F_FFFF;
    if exp <= 0 {
        return sign; // flush subnormals (packing scales never land here)
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    // Round to nearest, ties to even: look at the dropped 13 bits.
    let half = (mant >> 13) as u16;
    let dropped = mant & 0x1FFF;
    let bump = if dropped > 0x1000 || (dropped == 0x1000 && (half & 1) == 1) {
        1
    } else {
        0
    };
    let rounded = half + bump;
    if rounded == 0x0400 {
        // Mantissa overflow carries into the exponent.
        if exp + 1 >= 31 {
            return sign | 0x7C00;
        }
        return sign | (((exp + 1) as u16) << 10);
    }
    sign | ((exp as u16) << 10) | (rounded & 0x03FF)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn htod_f32(gpu: &Gpu, dst: &GpuTensor, host: &[f32]) {
    assert_eq!(dst.numel(), host.len());
    gpu.hip
        .memcpy_htod(&dst.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod f32");
}

fn dtoh_bytes(gpu: &Gpu, src: &GpuTensor) -> Vec<u8> {
    let n_bytes = src.numel() * src.dtype.size();
    let mut out = vec![0u8; n_bytes];
    gpu.hip.memcpy_dtoh(&mut out, &src.buf).expect("dtoh bytes");
    out
}

fn fill_f32_quiet_nan(gpu: &mut Gpu, tensor: &GpuTensor, payload_bits: u32) {
    let host: Vec<u32> = vec![payload_bits; tensor.numel()];
    gpu.hip
        .memcpy_htod(&tensor.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod fill quiet NaN");
    gpu.hip.device_synchronize().expect("sync fill");
}

fn check(label: &str, got: &[u8], want: &[u8], ok: &mut bool) {
    if got.len() != want.len() {
        eprintln!("FAIL {label}: len {} != {}", got.len(), want.len());
        *ok = false;
        return;
    }
    if got != want {
        let mut first = None;
        let mut count = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                if first.is_none() {
                    first = Some(i);
                }
                count += 1;
            }
        }
        eprintln!("FAIL {label}: {count} bytes differ, first at {first:?}");
        *ok = false;
    } else {
        eprintln!("ok {label} ({} bytes identical)", got.len());
    }
}

fn mk_mq4v2_weight(gpu: &Gpu, m: usize, k: usize, seed: u32) -> GpuTensor {
    let w: Vec<f32> = (0..m * k).map(|i| prng(i, seed) * 2.0 - 1.0).collect();
    let blob = pack_mq4g256v2(&w, m, k);
    assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);
    gpu.upload_raw(&blob, &[blob.len()]).expect("upload mq4v2")
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let arch = gpu.arch.clone();
    if !(gpu.arch_caps.is_gfx1100() && arch == "gfx1100") {
        eprintln!("SKIP: arch {arch} is not exact gfx1100");
        return;
    }
    if gpu.active_capture.is_some() {
        eprintln!("SKIP: active_capture is Some");
        return;
    }
    eprintln!("arch {arch} confirmed exact gfx1100 — running S3 F16 producer gate");

    let mut all_ok = true;
    // Routing-sensitive but 16-aligned row counts (base kernels handle tails;
    // aligned rows keep this gate focused on F16 exactness, not tail guards).
    let (qkv_m, z_m, beta_m, alpha_m) = (64usize, 32, 32, 16);
    let (q_m, k_m, v_m) = (64usize, 32, 32);
    let (gate_m, up_m) = (128usize, 128);

    for &n in &[1usize, 2, 8, 16] {
        for &k in &[4096usize, 5120] {
            for &awq in &[false, true] {
                let tag = format!("N={n} K={k} awq={awq}");
                eprintln!("--- {tag} ---");
                // Activations with rich mantissas across the exponent range;
                // row 0 scaled up to exercise F16 rounding away from 1.0.
                let x_host: Vec<f32> = (0..n * k)
                    .map(|i| {
                        let v = prng(i, 0xF16_0000 + n as u32) * 8.0 - 4.0;
                        if i < k {
                            v * 16.0
                        } else {
                            v
                        }
                    })
                    .collect();
                let w_host: Vec<f32> = (0..k).map(|i| 0.8 + 0.4 * prng(i, 0x9E37_0001)).collect();
                let a_host: Vec<f32> = (0..k).map(|i| 0.5 + 1.5 * prng(i, 0xA9A9_0002)).collect();

                let d_x = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
                let d_w = gpu.alloc_tensor(&[k], DType::F32).expect("alloc w");
                let d_awq = gpu.alloc_tensor(&[k], DType::F32).expect("alloc awq");
                let d_rot_f32 = gpu
                    .alloc_tensor(&[n * k], DType::F32)
                    .expect("alloc rot f32");
                let d_oracle_f16 = gpu
                    .alloc_tensor(&[n * k], DType::F16)
                    .expect("alloc oracle");
                let d_cand_f16 = gpu.alloc_tensor(&[n * k], DType::F16).expect("alloc cand");
                let d_wrap_f16 = gpu.alloc_tensor(&[n * k], DType::F16).expect("alloc wrap");
                htod_f32(&gpu, &d_x, &x_host);
                htod_f32(&gpu, &d_w, &w_host);
                htod_f32(&gpu, &d_awq, &a_host);
                gpu.hip.device_synchronize().expect("sync htod");

                // Old path oracle: F32 producer, then the same cast body the
                // GEMM-path convert inlines.
                if awq {
                    gpu.fused_rmsnorm_rotate_mq_awq_batched(
                        &d_x, &d_w, &d_awq, &d_rot_f32, k, EPS, n,
                    )
                    .expect("old awq producer");
                    gpu.fused_rmsnorm_rotate_mq_awq_f16_batched(
                        &d_x,
                        &d_w,
                        &d_awq,
                        &d_cand_f16,
                        k,
                        EPS,
                        n,
                    )
                    .expect("new awq producer");
                } else {
                    gpu.fused_rmsnorm_rotate_mq_batched(&d_x, &d_w, &d_rot_f32, k, EPS, n)
                        .expect("old producer");
                    gpu.fused_rmsnorm_rotate_mq_f16_batched(&d_x, &d_w, &d_cand_f16, k, EPS, n)
                        .expect("new producer");
                }
                gpu.cast_f32_to_f16(&d_rot_f32, &d_oracle_f16)
                    .expect("oracle cast");
                gpu.hip.device_synchronize().expect("sync producers");

                // Wrapper routing must match the direct producer call.
                let anchor = WeightTensor {
                    buf: gpu.upload_raw(&[0u8; 8], &[8]).expect("anchor buf"),
                    gpu_dtype: DType::MQ4G256V2,
                    m: 8,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: if awq { Some(d_awq) } else { None },
                };
                // NOTE: anchor takes ownership of d_awq in the AWQ arm; the
                // direct-producer oracle above already ran, so reuse the
                // wrapper output only for the routing check.
                fused_rmsnorm_rotate_mq_f16_batched_for(
                    &mut gpu,
                    &d_x,
                    &d_w,
                    &anchor,
                    &d_wrap_f16,
                    k,
                    EPS,
                    n,
                )
                .expect("wrapper producer");
                gpu.hip.device_synchronize().expect("sync wrapper");

                let oracle = dtoh_bytes(&gpu, &d_oracle_f16);
                let cand = dtoh_bytes(&gpu, &d_cand_f16);
                check(
                    &format!("{tag} producer-f16-memcmp"),
                    &cand,
                    &oracle,
                    &mut all_ok,
                );
                let wrap = dtoh_bytes(&gpu, &d_wrap_f16);
                check(
                    &format!("{tag} wrapper-routing-memcmp"),
                    &wrap,
                    &oracle,
                    &mut all_ok,
                );

                // Projection-output memcmp per family. Synthetic MQ4V2
                // weights (distinct seeds so swapped routing cannot match).
                {
                    let w_qkv = mk_mq4v2_weight(&gpu, qkv_m, k, 0x1111_2222);
                    let w_z = mk_mq4v2_weight(&gpu, z_m, k, 0x3333_4444);
                    let w_b = mk_mq4v2_weight(&gpu, beta_m, k, 0x5555_6666);
                    let w_a = mk_mq4v2_weight(&gpu, alpha_m, k, 0x7777_8888);
                    let outs_old: Vec<GpuTensor> = [qkv_m, z_m, beta_m, alpha_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    let outs_new: Vec<GpuTensor> = [qkv_m, z_m, beta_m, alpha_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    for (o, s) in
                        outs_old
                            .iter()
                            .zip([0x7fc0_0001, 0x7fc0_0002, 0x7fc0_0003, 0x7fc0_0004])
                    {
                        fill_f32_quiet_nan(&mut gpu, o, s);
                    }
                    for (o, s) in
                        outs_new
                            .iter()
                            .zip([0x7fc0_0011, 0x7fc0_0012, 0x7fc0_0013, 0x7fc0_0014])
                    {
                        fill_f32_quiet_nan(&mut gpu, o, s);
                    }
                    gpu.gemm_qkvza_mq4g256v2_wmma(
                        &w_qkv,
                        &w_z,
                        &w_b,
                        &w_a,
                        &d_rot_f32,
                        &outs_old[0],
                        &outs_old[1],
                        &outs_old[2],
                        &outs_old[3],
                        qkv_m,
                        z_m,
                        beta_m,
                        alpha_m,
                        k,
                        n,
                    )
                    .expect("old qkvza gemm");
                    gpu.gemm_qkvza_mq4g256v2_wmma_f16(
                        &w_qkv,
                        &w_z,
                        &w_b,
                        &w_a,
                        &d_cand_f16,
                        &outs_new[0],
                        &outs_new[1],
                        &outs_new[2],
                        &outs_new[3],
                        qkv_m,
                        z_m,
                        beta_m,
                        alpha_m,
                        k,
                        n,
                    )
                    .expect("new qkvza gemm");
                    gpu.hip.device_synchronize().expect("sync qkvza");
                    for (i, nm) in ["qkv", "z", "beta", "alpha"].iter().enumerate() {
                        let a = gpu.download_f32(&outs_old[i]).expect("dl old");
                        let b = gpu.download_f32(&outs_new[i]).expect("dl new");
                        let ab: &[u8] = unsafe {
                            std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                        };
                        let bb: &[u8] = unsafe {
                            std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4)
                        };
                        assert!(
                            a.iter().all(|v| v.is_finite()),
                            "{tag} qkvza/{nm} old not finite"
                        );
                        assert!(
                            b.iter().all(|v| v.is_finite()),
                            "{tag} qkvza/{nm} new not finite"
                        );
                        check(
                            &format!("{tag} qkvza/{nm}-output-memcmp"),
                            bb,
                            ab,
                            &mut all_ok,
                        );
                    }
                }
                // qkv
                {
                    let w_q = mk_mq4v2_weight(&gpu, q_m, k, 0x2222_1111);
                    let w_k = mk_mq4v2_weight(&gpu, k_m, k, 0x4444_3333);
                    let w_v = mk_mq4v2_weight(&gpu, v_m, k, 0x6666_5555);
                    let outs_old: Vec<GpuTensor> = [q_m, k_m, v_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    let outs_new: Vec<GpuTensor> = [q_m, k_m, v_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    for o in outs_old.iter().chain(outs_new.iter()) {
                        fill_f32_quiet_nan(&mut gpu, o, 0x7fc0_0021);
                    }
                    gpu.gemm_qkv_mq4g256v2_wmma(
                        &w_q,
                        &w_k,
                        &w_v,
                        &d_rot_f32,
                        &outs_old[0],
                        &outs_old[1],
                        &outs_old[2],
                        q_m,
                        k_m,
                        v_m,
                        k,
                        n,
                    )
                    .expect("old qkv gemm");
                    gpu.gemm_qkv_mq4g256v2_wmma_f16(
                        &w_q,
                        &w_k,
                        &w_v,
                        &d_cand_f16,
                        &outs_new[0],
                        &outs_new[1],
                        &outs_new[2],
                        q_m,
                        k_m,
                        v_m,
                        k,
                        n,
                    )
                    .expect("new qkv gemm");
                    gpu.hip.device_synchronize().expect("sync qkv");
                    for (i, nm) in ["q", "k", "v"].iter().enumerate() {
                        let a = gpu.download_f32(&outs_old[i]).expect("dl old");
                        let b = gpu.download_f32(&outs_new[i]).expect("dl new");
                        let ab: &[u8] = unsafe {
                            std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                        };
                        let bb: &[u8] = unsafe {
                            std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4)
                        };
                        assert!(
                            a.iter().all(|v| v.is_finite()),
                            "{tag} qkv/{nm} old not finite"
                        );
                        assert!(
                            b.iter().all(|v| v.is_finite()),
                            "{tag} qkv/{nm} new not finite"
                        );
                        check(
                            &format!("{tag} qkv/{nm}-output-memcmp"),
                            bb,
                            ab,
                            &mut all_ok,
                        );
                    }
                }
                // gate_up
                {
                    let w_g = mk_mq4v2_weight(&gpu, gate_m, k, 0xABCD_0001);
                    let w_u = mk_mq4v2_weight(&gpu, up_m, k, 0xABCD_0002);
                    let outs_old: Vec<GpuTensor> = [gate_m, up_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    let outs_new: Vec<GpuTensor> = [gate_m, up_m]
                        .iter()
                        .map(|&m| gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc out"))
                        .collect();
                    for o in outs_old.iter().chain(outs_new.iter()) {
                        fill_f32_quiet_nan(&mut gpu, o, 0x7fc0_0031);
                    }
                    gpu.gemm_gate_up_mq4g256v2_wmma(
                        &w_g,
                        &w_u,
                        &d_rot_f32,
                        &outs_old[0],
                        &outs_old[1],
                        gate_m,
                        up_m,
                        k,
                        n,
                    )
                    .expect("old gate_up gemm");
                    gpu.gemm_gate_up_mq4g256v2_wmma_f16(
                        &w_g,
                        &w_u,
                        &d_cand_f16,
                        &outs_new[0],
                        &outs_new[1],
                        gate_m,
                        up_m,
                        k,
                        n,
                    )
                    .expect("new gate_up gemm");
                    gpu.hip.device_synchronize().expect("sync gate_up");
                    for (i, nm) in ["gate", "up"].iter().enumerate() {
                        let a = gpu.download_f32(&outs_old[i]).expect("dl old");
                        let b = gpu.download_f32(&outs_new[i]).expect("dl new");
                        let ab: &[u8] = unsafe {
                            std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                        };
                        let bb: &[u8] = unsafe {
                            std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4)
                        };
                        assert!(
                            a.iter().all(|v| v.is_finite()),
                            "{tag} gate_up/{nm} old not finite"
                        );
                        assert!(
                            b.iter().all(|v| v.is_finite()),
                            "{tag} gate_up/{nm} new not finite"
                        );
                        check(
                            &format!("{tag} gate_up/{nm}-output-memcmp"),
                            bb,
                            ab,
                            &mut all_ok,
                        );
                    }
                }
            }
        }
    }

    // Negative gate: F32 input to an F16 entry must Err, never convert.
    {
        let d_f32 = gpu.alloc_tensor(&[16], DType::F32).expect("alloc neg");
        let d_f16 = gpu.alloc_tensor(&[16], DType::F16).expect("alloc neg16");
        let r = gpu.gemm_qkv_mq4g256v2_wmma_f16(
            &d_f32, &d_f32, &d_f32, &d_f32, &d_f32, &d_f32, &d_f32, 1, 1, 1, 16, 1,
        );
        if r.is_ok() {
            eprintln!("FAIL dtype-gate: F32 x_f16 accepted");
            all_ok = false;
        } else {
            eprintln!("ok dtype-gate rejects F32 x_f16");
        }
        let r2 = gpu.fused_rmsnorm_rotate_mq_f16_batched(&d_f32, &d_f32, &d_f32, 16, EPS, 1);
        if r2.is_ok() {
            eprintln!("FAIL dtype-gate: F32 x_rot_f16 accepted");
            all_ok = false;
        } else {
            eprintln!("ok dtype-gate rejects F32 x_rot_f16");
        }
        let _ = d_f16;
    }

    if all_ok {
        eprintln!("PASS test_mq_f16_projection_producers_gfx1100");
    } else {
        eprintln!("FAIL test_mq_f16_projection_producers_gfx1100");
        std::process::exit(1);
    }
}
