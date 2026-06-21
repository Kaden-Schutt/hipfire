// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the routed batched KVarN attention (microbatching,
//! `attention_kvarn_routed_batched`) against the validated single-session fused
//! flash (`attention_flash_kvarn_batched_masked`). Builds 3 sessions with
//! distinct positions (different n_full_blocks / tail), routes a batch of rows to
//! them in SHUFFLED order via session-major pointer tables, and checks each
//! routed row matches its session's single-session flash. Proves the per-session
//! routing + per-row n_full_blocks + inline dequant.
//!
//!   cargo run --release -p rdna-compute --example parity_kvarn_routed

use rdna_compute::{DType, Gpu, GpuTensor};

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f { return sign | 0x7c00; }
    if exp <= 0 {
        if exp < -10 { return sign; }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (mant >> shift) as u16;
        if (mant >> (shift - 1)) & 1 == 1 {
            let sticky = mant & ((1 << (shift - 1)) - 1);
            if sticky != 0 || (h & 1) == 1 { h += 1; }
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    if (mant >> 12) & 1 == 1 {
        let sticky = mant & 0xfff;
        if sticky != 0 || (h_mant & 1) == 1 { h_mant += 1; if h_mant == 0x400 { h_mant = 0; exp += 1; } }
    }
    sign | ((exp as u16) << 10) | h_mant
}

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || { s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff; (s as f32 + 0.5) / 2_147_483_648.0 };
    (0..n).map(|_| { let u1 = u().max(1e-7); let u2 = u(); (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() }).collect()
}

const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const GROUP: usize = 128;
const MAX_SEQ: usize = 768;

struct Session { rd: GpuTensor, wd: GpuTensor, vd: GpuTensor, pos: usize }

fn build_session(gpu: &mut Gpu, seed: u32, n_full_blocks: usize, tail_len: usize) -> Session {
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let n_full = n_full_blocks * GROUP;
    let seq_len = (n_full + tail_len).max(1);
    let blocks_per_head = HEAD_DIM / 32;
    let v_row_stride = N_KV_HEADS * blocks_per_head * 34;
    let tile_elems = HEAD_DIM * GROUP;
    let record_bytes = tile_elems.div_ceil(2) + HEAD_DIM * 2 * 2 + GROUP * 2;

    // records
    let n_blocks_alloc = MAX_SEQ.div_ceil(GROUP);
    let rec_buf_bytes = (n_blocks_alloc * N_KV_HEADS * record_bytes).next_multiple_of(4);
    let rd = gpu.upload_raw(&vec![0u8; rec_buf_bytes], &[rec_buf_bytes / 4]).unwrap();
    if n_full_blocks > 0 {
        let kbase = lcg(seed, n_full * kv_dim);
        let mut k = vec![0.0f32; n_full * kv_dim];
        for t in 0..n_full {
            let ts = 0.1f32 * 20f32.powf(t as f32 / n_full as f32);
            for j in 0..kv_dim { let cs = 0.05f32 * 40f32.powf((j % HEAD_DIM) as f32 / HEAD_DIM as f32); k[t * kv_dim + j] = kbase[t * kv_dim + j] * ts * cs; }
        }
        let kd = gpu.upload_raw(&k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[n_full * kv_dim]).unwrap();
        let n_tiles = n_full_blocks * N_KV_HEADS;
        let td = gpu.upload_raw(&vec![0u8; n_tiles * tile_elems * 4], &[n_tiles * tile_elems]).unwrap();
        gpu.kvarn_gather_k_tiles(&kd, &td, n_full_blocks, N_KV_HEADS, HEAD_DIM, GROUP).unwrap();
        gpu.kvarn_quantize_tile(&td, &rd, n_tiles, HEAD_DIM, GROUP, record_bytes).unwrap();
    }
    // window
    let mut window = vec![0.0f32; GROUP * kv_dim];
    let wbase = lcg(seed.wrapping_add(1), tail_len.max(1) * kv_dim);
    for t in 0..tail_len { for j in 0..kv_dim { window[t * kv_dim + j] = wbase[t * kv_dim + j] * 0.3; } }
    let wd = gpu.upload_raw(&window.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[GROUP * kv_dim]).unwrap();
    // Q8 V
    let vbase = lcg(seed.wrapping_add(2), seq_len * kv_dim);
    let mut v_cache = vec![0u8; MAX_SEQ * v_row_stride];
    for t in 0..seq_len { for kvh in 0..N_KV_HEADS { for bb in 0..blocks_per_head {
        let mut amax = 0.0f32;
        for e in 0..32 { amax = amax.max((vbase[t * kv_dim + kvh * HEAD_DIM + bb * 32 + e] * 0.3).abs()); }
        let scale = (amax / 127.0).max(1e-8);
        let blk = t * v_row_stride + (kvh * blocks_per_head + bb) * 34;
        v_cache[blk..blk + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        for e in 0..32 { let x = vbase[t * kv_dim + kvh * HEAD_DIM + bb * 32 + e] * 0.3; v_cache[blk + 2 + e] = ((x / scale).round().clamp(-127.0, 127.0) as i8) as u8; }
    }}}
    let vd = gpu.upload_raw(&v_cache, &[MAX_SEQ * v_row_stride]).unwrap();
    Session { rd, wd, vd, pos: seq_len - 1 }
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let kv_dim = N_KV_HEADS * HEAD_DIM;

    let sessions = vec![
        build_session(&mut gpu, 11, 1, 20),  // pos 147
        build_session(&mut gpu, 31, 0, 70),  // pos 69
        build_session(&mut gpu, 53, 2, 5),   // pos 260
    ];

    // Reference: single-session fused flash per session (one query at its pos).
    let q_all: Vec<Vec<f32>> = (0..sessions.len())
        .map(|i| lcg(100 + i as u32, N_HEADS * HEAD_DIM).iter().map(|v| v * 0.2).collect())
        .collect();
    let max_tiles = MAX_SEQ.div_ceil(GROUP);
    let partials = gpu.zeros(&[N_HEADS * max_tiles * (2 + HEAD_DIM)], DType::F32).unwrap();
    let record_bytes = (HEAD_DIM * GROUP).div_ceil(2) + HEAD_DIM * 2 * 2 + GROUP * 2;
    let mut ref_out = vec![vec![0.0f32; N_HEADS * HEAD_DIM]; sessions.len()];
    for (i, s) in sessions.iter().enumerate() {
        let qd = gpu.upload_raw(&q_all[i].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[N_HEADS * HEAD_DIM]).unwrap();
        let posd = gpu.upload_raw(&(s.pos as i32).to_le_bytes(), &[1]).unwrap();
        let outd = gpu.upload_raw(&vec![0u8; N_HEADS * HEAD_DIM * 4], &[N_HEADS * HEAD_DIM]).unwrap();
        let seq_len = s.pos + 1;
        let n_full = seq_len / GROUP;
        gpu.attention_flash_kvarn_batched_masked(&qd, &s.rd, &s.wd, &s.vd, &outd, &posd,
            N_HEADS, N_KV_HEADS, HEAD_DIM, MAX_SEQ, seq_len, 1, &partials, None, 0, 0, n_full, record_bytes).unwrap();
        gpu.device_synchronize().unwrap();
        ref_out[i] = gpu.download_f32(&outd).unwrap();
    }

    // Routed batch: rows in SHUFFLED session order.
    let row_sessions: Vec<i32> = vec![2, 0, 1];
    let n_rows = row_sessions.len();
    let mut q_batch = vec![0.0f32; n_rows * N_HEADS * HEAD_DIM];
    let mut pos_batch = vec![0i32; n_rows];
    for (r, &sess) in row_sessions.iter().enumerate() {
        q_batch[r * N_HEADS * HEAD_DIM..(r + 1) * N_HEADS * HEAD_DIM].copy_from_slice(&q_all[sess as usize]);
        pos_batch[r] = sessions[sess as usize].pos as i32;
    }
    let qd = gpu.upload_raw(&q_batch.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[n_rows * N_HEADS * HEAD_DIM]).unwrap();
    let posd = gpu.upload_raw(&pos_batch.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[n_rows]).unwrap();
    let rsid = gpu.upload_raw(&row_sessions.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[n_rows]).unwrap();

    // Pointer tables (ptr_layer_stride=1, layer 0): session-major device ptrs.
    let to_u64 = |t: &GpuTensor| (t.buf.as_ptr() as usize as u64).to_le_bytes();
    let rec_ptrs: Vec<u8> = sessions.iter().flat_map(|s| to_u64(&s.rd)).collect();
    let win_ptrs: Vec<u8> = sessions.iter().flat_map(|s| to_u64(&s.wd)).collect();
    let v_ptrs: Vec<u8> = sessions.iter().flat_map(|s| to_u64(&s.vd)).collect();
    let recp = gpu.upload_raw(&rec_ptrs, &[sessions.len() * 2]).unwrap(); // u64 = 2×u32 elems
    let winp = gpu.upload_raw(&win_ptrs, &[sessions.len() * 2]).unwrap();
    let vp = gpu.upload_raw(&v_ptrs, &[sessions.len() * 2]).unwrap();

    let out_b = gpu.upload_raw(&vec![0u8; n_rows * N_HEADS * HEAD_DIM * 4], &[n_rows * N_HEADS * HEAD_DIM]).unwrap();
    let max_ctx = pos_batch.iter().map(|&p| p as usize + 1).max().unwrap();
    gpu.attention_kvarn_routed_batched(&qd, &recp, &winp, &vp, &out_b, &rsid, &posd, 1, 0,
        N_HEADS, N_KV_HEADS, HEAD_DIM, MAX_SEQ, max_ctx, n_rows).unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_b).unwrap();

    let mut max_abs = 0.0f32;
    for (r, &sess) in row_sessions.iter().enumerate() {
        let rr = &got[r * N_HEADS * HEAD_DIM..(r + 1) * N_HEADS * HEAD_DIM];
        let rf = &ref_out[sess as usize];
        for i in 0..N_HEADS * HEAD_DIM { max_abs = max_abs.max((rr[i] - rf[i]).abs()); }
    }
    let _ = kv_dim;
    let pass = max_abs < 2e-3;
    println!("parity_kvarn_routed on {}: routed-vs-single-session max-abs-err={max_abs:.2e} (rows->sessions {row_sessions:?}) -> {}",
        gpu.arch, if pass { "PASS" } else { "FAIL" });
    if !pass { std::process::exit(1); }
}
