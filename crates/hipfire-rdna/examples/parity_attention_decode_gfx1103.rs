// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the gfx1103 no-LDS flash-decode attention sibling kernels:
//! f32, q4kv, hfq4, int8c, int8c_f16, int8, hfq8.
//!
//! For each format it builds a KV cache with the real write kernel (identical
//! in both runs), runs the attention dispatcher, and dumps all output vectors
//! to a file. Run once on gfx1103 (arch-selected no-LDS kernels) and once with
//! HIPFIRE_FORCE_GENERIC=1 (generic LDS kernels), then diff the two dumps: the
//! no-LDS kernels must equal their trusted generic baselines. F32 additionally
//! carries an f64 CPU reference for absolute correctness.
//!
//!   A=/tmp/.../a.bin B=/tmp/.../b.bin
//!   cargo run --release -p hipfire-rdna --example parity_attention_decode_gfx1103 -- $A
//!   HIPFIRE_FORCE_GENERIC=1 cargo run ... --example parity_attention_decode_gfx1103 -- $B
//!   # then byte/tolerance-diff $A vs $B

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::io::Write;

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const SEQ: usize = 200;
const MAXS: usize = 256;
const KVDIM: usize = NKV * HD;

fn set_pos(gpu: &Gpu, pos_buf: &hip_bridge::DeviceBuffer, p: i32) {
    gpu.hip.memcpy_htod(pos_buf, &p.to_ne_bytes()).unwrap();
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-1000/parity_attn_decode.bin".to_string()
    });
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "force_generic={forced} (path: {})",
        if forced { "generic LDS kernels" } else { "arch-selected kernels" }
    );

    let q = lcg(0xa5a5, NH * HD);
    let d_q = gpu.upload_f32(&q, &[NH * HD]).unwrap();
    let pos_buf = gpu.hip.malloc(4).unwrap();

    // Per-position source KV (f32), deterministic across runs.
    let k_src: Vec<Vec<f32>> = (0..SEQ).map(|t| lcg(0x1000 + t as u64, KVDIM)).collect();
    let v_src: Vec<Vec<f32>> = (0..SEQ).map(|t| lcg(0x9000 + t as u64, KVDIM)).collect();

    let mut dump: Vec<f32> = Vec::new();
    let record = |gpu: &Gpu, out: &GpuTensor, name: &str| -> Vec<f32> {
        let o = gpu.download_f32(out).unwrap();
        let mag = o.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("  {name:16} out_mag={mag:.4}");
        o
    };
    let d_out = gpu.zeros(&[NH * HD], DType::F32).unwrap();

    // Helper: write SEQ positions into a co-located byte cache via `writer`.
    // writer(gpu, dst, src_tensor) writes one position (pos_buf already set).

    // ── 1. f32 (plain cache, CPU ref) ────────────────────────────────────
    {
        let mut kflat = vec![0f32; MAXS * KVDIM];
        let mut vflat = vec![0f32; MAXS * KVDIM];
        for t in 0..SEQ {
            kflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&k_src[t]);
            vflat[t * KVDIM..(t + 1) * KVDIM].copy_from_slice(&v_src[t]);
        }
        let dk = gpu.upload_f32(&kflat, &[MAXS * KVDIM]).unwrap();
        let dv = gpu.upload_f32(&vflat, &[MAXS * KVDIM]).unwrap();
        set_pos(&gpu, &pos_buf, (SEQ - 1) as i32);
        gpu.attention_f32(&d_q, &dk, &dv, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap();
        gpu.hip.device_synchronize().unwrap();
        // CPU ref
        let scale = 1.0f64 / (HD as f64).sqrt();
        let mut refv = vec![0f32; NH * HD];
        for h in 0..NH {
            let kv_h = h / (NH / NKV);
            let mut sc = vec![0f64; SEQ];
            let mut mx = f64::MIN;
            for t in 0..SEQ {
                let mut dot = 0f64;
                for d in 0..HD {
                    dot += q[h * HD + d] as f64 * k_src[t][kv_h * HD + d] as f64;
                }
                sc[t] = dot * scale;
                mx = mx.max(sc[t]);
            }
            let mut den = 0f64;
            for x in sc.iter_mut() { *x = (*x - mx).exp(); den += *x; }
            for d in 0..HD {
                let mut acc = 0f64;
                for t in 0..SEQ { acc += sc[t] * v_src[t][kv_h * HD + d] as f64; }
                refv[h * HD + d] = (acc / den) as f32;
            }
        }
        let got = gpu.download_f32(&d_out).unwrap();
        let err = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("  {:16} out_mag={:.4}  cpu_err={err:.3e}", "f32", got.iter().map(|v| v.abs()).fold(0.0, f32::max));
        assert!(err < 3e-4, "f32 CPU parity FAIL err={err:.3e}");
        dump.extend_from_slice(&got);
    }

    // ── co-located byte caches (q4/hfq4, int8c, int8c_f16) ───────────────
    let colocated = |gpu: &mut Gpu, bph: usize, fmt: u8, out_name: &str, run_kernel: u8| {
        let dst_bytes = MAXS * NKV * bph;
        let dk = gpu.alloc_tensor(&[dst_bytes], DType::Raw).unwrap();
        let dv = gpu.alloc_tensor(&[dst_bytes], DType::Raw).unwrap();
        for t in 0..SEQ {
            set_pos(gpu, &pos_buf, t as i32);
            let sk = gpu.upload_f32(&k_src[t], &[KVDIM]).unwrap();
            let sv = gpu.upload_f32(&v_src[t], &[KVDIM]).unwrap();
            match fmt {
                0 => { gpu.kv_cache_write_hfq4(&dk, &sk, &pos_buf, NKV, HD).unwrap();
                       gpu.kv_cache_write_hfq4(&dv, &sv, &pos_buf, NKV, HD).unwrap(); }
                1 => { gpu.kv_cache_write_int8c(&dk, &sk, &pos_buf, NKV, HD).unwrap();
                       gpu.kv_cache_write_int8c(&dv, &sv, &pos_buf, NKV, HD).unwrap(); }
                _ => { gpu.kv_cache_write_int8c_f16(&dk, &sk, &pos_buf, NKV, HD).unwrap();
                       gpu.kv_cache_write_int8c_f16(&dv, &sv, &pos_buf, NKV, HD).unwrap(); }
            }
        }
        set_pos(gpu, &pos_buf, (SEQ - 1) as i32);
        match run_kernel {
            0 => gpu.attention_q4kv(&d_q, &dk, &dv, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap(),
            1 => gpu.attention_hfq4_kv(&d_q, &dk, &dv, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap(),
            2 => gpu.attention_int8c_kv(&d_q, &dk, &dv, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap(),
            _ => gpu.attention_int8c_f16_kv(&d_q, &dk, &dv, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap(),
        }
        gpu.hip.device_synchronize().unwrap();
        let _ = out_name;
    };

    colocated(&mut gpu, 8 + HD / 2, 0, "q4kv", 0);
    dump.extend(record(&gpu, &d_out, "q4kv"));
    colocated(&mut gpu, 8 + HD / 2, 0, "hfq4_kv", 1);
    dump.extend(record(&gpu, &d_out, "hfq4_kv"));
    colocated(&mut gpu, 8 + HD, 1, "int8c_kv", 2);
    dump.extend(record(&gpu, &d_out, "int8c_kv"));
    colocated(&mut gpu, 4 + HD, 2, "int8c_f16_kv", 3);
    dump.extend(record(&gpu, &d_out, "int8c_f16_kv"));

    // ── split-scale caches (int8, hfq8) ──────────────────────────────────
    {
        // int8: vals [MAXS*KVDIM] bytes, scales [MAXS*NKV] f32
        let dvals_k = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Raw).unwrap();
        let dvals_v = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Raw).unwrap();
        let dsc_k = gpu.alloc_tensor(&[MAXS * NKV], DType::F32).unwrap();
        let dsc_v = gpu.alloc_tensor(&[MAXS * NKV], DType::F32).unwrap();
        for t in 0..SEQ {
            set_pos(&gpu, &pos_buf, t as i32);
            let sk = gpu.upload_f32(&k_src[t], &[KVDIM]).unwrap();
            let sv = gpu.upload_f32(&v_src[t], &[KVDIM]).unwrap();
            gpu.kv_cache_write_int8(&dvals_k, &dsc_k, &sk, &pos_buf, NKV, HD).unwrap();
            gpu.kv_cache_write_int8(&dvals_v, &dsc_v, &sv, &pos_buf, NKV, HD).unwrap();
        }
        set_pos(&gpu, &pos_buf, (SEQ - 1) as i32);
        gpu.attention_int8_kv(&d_q, &dvals_k, &dsc_k, &dvals_v, &dsc_v, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap();
        gpu.hip.device_synchronize().unwrap();
        dump.extend(record(&gpu, &d_out, "int8_kv"));
    }
    {
        // hfq8: data [MAXS*KVDIM] bytes, scales [MAXS*NKV*2] f32
        let dk = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Raw).unwrap();
        let dv = gpu.alloc_tensor(&[MAXS * KVDIM], DType::Raw).unwrap();
        let dsc_k = gpu.alloc_tensor(&[MAXS * NKV * 2], DType::F32).unwrap();
        let dsc_v = gpu.alloc_tensor(&[MAXS * NKV * 2], DType::F32).unwrap();
        for t in 0..SEQ {
            set_pos(&gpu, &pos_buf, t as i32);
            let sk = gpu.upload_f32(&k_src[t], &[KVDIM]).unwrap();
            let sv = gpu.upload_f32(&v_src[t], &[KVDIM]).unwrap();
            gpu.kv_cache_write_hfq8(&dk, &dsc_k, &sk, &pos_buf, NKV, HD).unwrap();
            gpu.kv_cache_write_hfq8(&dv, &dsc_v, &sv, &pos_buf, NKV, HD).unwrap();
        }
        set_pos(&gpu, &pos_buf, (SEQ - 1) as i32);
        gpu.attention_hfq8_kv(&d_q, &dk, &dsc_k, &dv, &dsc_v, &d_out, &pos_buf, SEQ, NH, NKV, HD, MAXS).unwrap();
        gpu.hip.device_synchronize().unwrap();
        dump.extend(record(&gpu, &d_out, "hfq8_kv"));
    }

    let bytes: Vec<u8> = dump.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::File::create(&out_path).unwrap().write_all(&bytes).unwrap();
    println!("wrote {} floats -> {out_path}", dump.len());
}
