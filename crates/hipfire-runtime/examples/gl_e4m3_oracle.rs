// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Host-vs-device E4M3 encoding oracle for `rdna_compute::GL_CB4`.
//!
//! Converts the 16 GL levels on-device with `__builtin_amdgcn_cvt_pk_fp8_f32`
//! (same packing convention as `kernels/src/gemv_hfp4g32_fp8.gfx12.hip` /
//! `mq_rotate_x_dual.gfx12.hip`) and compares each byte against a host OCP
//! E4M3 RNE encoder. Byte-for-byte agreement is required if an fp8 fallback
//! path is ever used: later slices assume host and device share one encoding.
//!
//! Note: int8 (max→127 scale) is the PRIMARY packed carrier for qt=40 prefill;
//! this probe only locks the E4M3 fallback conversion. int8 packing is a plain
//! multiply-and-round with no arch-specific builtin.
//!
//! Run:
//!   cargo build --release -p hipfire-runtime --example gl_e4m3_oracle
//!   ./target/release/examples/gl_e4m3_oracle

use rdna_compute::{Gpu, GL_CB4};

/// OCP E4M3FN RNE (1/4/3, bias 7, max finite 448). Mirrors the host oracle in
/// `hipfire-quantize` tests — kept local because hipfire-quantize is a bin crate.
fn e4m3_decode(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (bits >> 3) & 0x0F;
    let mant = bits & 0x07;
    if exp == 0 {
        return sign * (mant as f32) * (2.0f32).powi(-9);
    }
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN.copysign(sign);
    }
    sign * (1.0 + (mant as f32) / 8.0) * (2.0f32).powi(exp as i32 - 7)
}

fn e4m3_rne(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let ax = x.abs();
    if ax >= 448.0 {
        return sign | 0x7E;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..0x7F {
        let y = e4m3_decode(code);
        let err = (y - ax).abs();
        if err < best_err {
            best_err = err;
            best = code;
        } else if err == best_err && (code & 1) == 0 && (best & 1) == 1 {
            best = code;
        }
    }
    sign | best
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("arch={}", gpu.arch);
    if !(gpu.arch.eq_ignore_ascii_case("gfx1200") || gpu.arch.eq_ignore_ascii_case("gfx1201")) {
        eprintln!(
            "gl_e4m3_oracle: requires gfx1200/gfx1201 (native OCP cvt_pk_fp8_f32); got {}",
            gpu.arch
        );
        std::process::exit(2);
    }

    // Device converts 16 f32 levels → 16 E4M3 bytes via cvt_pk_fp8_f32 pairs.
    // Packing: lo word gets pairs (0,1) then (2,3) into low/high halves; same
    // for the next 4 — matches mq_rotate_x_dual_fp8 / pack_f32_to_fp8.
    let src = r#"
#include <hip/hip_runtime.h>

#if !defined(__gfx1200__) && !defined(__gfx1201__)
#error "gl_e4m3_oracle requires __gfx1200__ or __gfx1201__"
#endif

extern "C" __global__ void gl_e4m3_cvt(
    const float* __restrict__ in,
    unsigned char* __restrict__ out
) {
    // Single-thread probe: convert all 16 values with the production packing
    // builtin. Byte order within each u32 is little-endian on device memory,
    // matching how gemv_hfp4g32_fp8 loads FP8 packs as unsigned int.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(in[0],  in[1],  w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(in[2],  in[3],  w0, true);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(in[4],  in[5],  w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(in[6],  in[7],  w1, true);
    w2 = __builtin_amdgcn_cvt_pk_fp8_f32(in[8],  in[9],  w2, false);
    w2 = __builtin_amdgcn_cvt_pk_fp8_f32(in[10], in[11], w2, true);
    w3 = __builtin_amdgcn_cvt_pk_fp8_f32(in[12], in[13], w3, false);
    w3 = __builtin_amdgcn_cvt_pk_fp8_f32(in[14], in[15], w3, true);

    ((unsigned int*)out)[0] = w0;
    ((unsigned int*)out)[1] = w1;
    ((unsigned int*)out)[2] = w2;
    ((unsigned int*)out)[3] = w3;
}
"#;

    gpu.ensure_kernel_public("gl_e4m3_cvt", src, "gl_e4m3_cvt")
        .expect("compile gl_e4m3_cvt");

    let host_in: Vec<f32> = GL_CB4.to_vec();
    let d_in = gpu.upload_f32(&host_in, &[16]).expect("upload in");
    let d_out = gpu
        .zeros(&[16], rdna_compute::DType::Raw)
        .expect("alloc out");

    let mut kernargs = hip_bridge::KernargBlob::new();
    kernargs.push_ptr(d_in.buf.as_ptr());
    kernargs.push_ptr(d_out.buf.as_ptr());
    gpu.launch_kernel_blob(
        "gl_e4m3_cvt",
        [1, 1, 1],
        [1, 1, 1],
        0,
        kernargs.as_mut_slice(),
    )
    .expect("launch");
    gpu.hip.device_synchronize().expect("sync");

    let mut device_bytes = vec![0u8; 16];
    gpu.hip
        .memcpy_dtoh(&mut device_bytes, &d_out.buf)
        .expect("download");

    let host_bytes: Vec<u8> = host_in.iter().copied().map(e4m3_rne).collect();

    eprintln!("GL_CB4 host RNE vs device cvt_pk_fp8_f32:");
    eprintln!(
        "  {:>4} {:>10} {:>8} {:>8} {:>10} {:>8}",
        "idx", "gl", "host", "device", "decoded", "rel%"
    );
    let mut mismatches = 0usize;
    for i in 0..16 {
        let v = host_in[i];
        let h = host_bytes[i];
        let d = device_bytes[i];
        let dec = e4m3_decode(d);
        let rel = if v == 0.0 {
            0.0
        } else {
            ((dec - v).abs() / v.abs()) * 100.0
        };
        let mark = if h == d { "OK" } else { "MISMATCH" };
        if h != d {
            mismatches += 1;
        }
        eprintln!(
            "  {i:>4} {v:>+10.4}  0x{h:02x}    0x{d:02x}   {dec:>+10.4} {rel:>7.3}  {mark}"
        );
    }

    if mismatches == 0 {
        eprintln!("RESULT: host and device agree byte-for-byte on all 16 levels");
    } else {
        eprintln!("RESULT: FAIL — {mismatches}/16 host/device byte mismatches (no tolerance)");
        std::process::exit(1);
    }
}
