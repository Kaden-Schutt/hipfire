// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Correctness test for SLIDING-WINDOW masking in the batched Q8 flash attention
// (`attention_flash_q8_0_batched_masked_windowed`). Used by cohere2moe's
// `sliding_attention` layers so that, at context > sliding_window, a query at
// position p attends ONLY to keys in [p - window + 1, p] (the last `window`).
//
// No reference implementation needed — we assert three behavioral properties
// that together pin the masking semantics:
//
//   (1) DIFFERS:    windowed(W) != full-causal(W=0) when seq_len > W
//                   (the window actually clips something).
//   (2) INVARIANT:  overwriting the OUT-OF-WINDOW keys [0, seq_len-W) with huge
//                   garbage does NOT change the windowed output (they are truly
//                   excluded — the strongest check; a broken mask that leaks
//                   them would be dominated by the huge scores and diverge).
//   (3) EQUIVALENT: windowed(W >= seq_len) == full-causal (no clipping).
//
// Exit 0 = all pass; exit 1 = a property failed (the assertion message says
// which). Run: cargo run --release -p rdna-compute --example q8_windowed_attn_test

use rdna_compute::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const S: usize = 384; // sequence length (3 tiles of 128)
const W: usize = 128; // sliding window
const BLK: usize = 34; // Q8_0 block: fp16 scale + 32 i8 codes
const TILE: usize = 128;

// Build a Q8_0 K/V cache row pattern; `huge` rows get large-magnitude codes so
// that if they leaked into a windowed softmax they would dominate it.
fn fill_kv(ctx: usize, huge_below: usize) -> Vec<u8> {
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;
    let mut kv = vec![0u8; ctx * bytes_per_pos];
    for pos in 0..ctx {
        let huge = pos < huge_below;
        for blk_i in 0..(NKV * blocks_per_head) {
            let off = pos * bytes_per_pos + blk_i * BLK;
            // fp16 scale: 1.0 (0x3C00) normally, ~100.0 (0x5640) for huge rows.
            let scale_bits: u16 = if huge { 0x5640 } else { 0x3C00 };
            kv[off] = (scale_bits & 0xFF) as u8;
            kv[off + 1] = (scale_bits >> 8) as u8;
            for j in 0..32 {
                // varied small codes normally; max (127) for huge rows.
                kv[off + 2 + j] = if huge {
                    127i8 as u8
                } else {
                    (((pos + blk_i + j) % 7) as i32 - 3) as i8 as u8
                };
            }
        }
    }
    kv
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;

    // Q: [1 × NH × HD], a varied pattern.
    let q_data: Vec<f32> = (0..NH * HD)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[NH * HD]).expect("q");

    // positions: single query at the tail (pos = S-1).
    let pos_data: Vec<i32> = vec![(S - 1) as i32];
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, 4) };
    let positions = gpu.upload_raw(pos_bytes, &[1]).expect("pos");

    let max_tiles = S.div_ceil(TILE);
    let partials = gpu
        .zeros(&[NH * max_tiles * (2 + HD)], DType::F32)
        .expect("partials");

    // Helper: run one windowed attention and return the host output vector.
    let mut run = |gpu: &mut Gpu, kv: &[u8], window: i32| -> Vec<f32> {
        let k = gpu.upload_raw(kv, &[kv.len()]).expect("k");
        let v = gpu.upload_raw(kv, &[kv.len()]).expect("v");
        let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
        gpu.attention_flash_q8_0_batched_masked_windowed(
            &q, &k, &v, &out, &positions, NH, NKV, HD, S, S, 1, &partials, None, 0, 0, window,
        )
        .expect("windowed attn launch");
        gpu.download_f32(&out).expect("download")
    };

    let kv_normal = fill_kv(S, 0);
    // Overwrite the OUT-OF-WINDOW region [0, S-W) with huge garbage.
    let kv_huge_outside = fill_kv(S, S - W);

    let out_full = run(&mut gpu, &kv_normal, 0); // full causal
    let out_win = run(&mut gpu, &kv_normal, W as i32); // windowed
    let out_win_huge = run(&mut gpu, &kv_huge_outside, W as i32); // windowed, huge outside
    let out_bigw = run(&mut gpu, &kv_normal, (S + 16) as i32); // window >= seq

    let d_clip = max_abs_diff(&out_full, &out_win);
    let d_invar = max_abs_diff(&out_win, &out_win_huge);
    let d_equiv = max_abs_diff(&out_full, &out_bigw);

    println!("seq={S} window={W} heads={NH} head_dim={HD}");
    println!("(1) clip   |full - win|            = {d_clip:.6}  (want > 0.001)");
    println!("(2) invar  |win  - win_huge_out|   = {d_invar:.6}  (want < 0.001)");
    println!("(3) equiv  |full - win(W>=seq)|    = {d_equiv:.6}  (want < 0.001)");

    let mut ok = true;
    if !(d_clip > 1e-3) {
        eprintln!(
            "FAIL (1): windowed output == full-causal; window had NO effect (masking not applied)"
        );
        ok = false;
    }
    if !(d_invar < 1e-3) {
        eprintln!(
            "FAIL (2): huge OUT-OF-WINDOW keys changed the output; they leaked past the mask"
        );
        ok = false;
    }
    if !(d_equiv < 1e-3) {
        eprintln!("FAIL (3): window>=seq differs from full causal; mask over-clips in-window keys");
        ok = false;
    }

    // ── Same three properties for the DECODE path (non-batched
    //    attention_flash_q8_0_windowed — a different kernel file). ──
    let mut run_nb = |gpu: &mut Gpu, kv: &[u8], window: i32| -> Vec<f32> {
        let k = gpu.upload_raw(kv, &[kv.len()]).expect("k");
        let v = gpu.upload_raw(kv, &[kv.len()]).expect("v");
        let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
        gpu.attention_flash_q8_0_windowed(
            &q,
            &k,
            &v,
            &out,
            &positions.buf,
            S,
            NH,
            NKV,
            HD,
            S,
            &partials,
            window,
        )
        .expect("nb windowed attn launch");
        gpu.download_f32(&out).expect("download")
    };
    let nb_full = run_nb(&mut gpu, &kv_normal, 0);
    let nb_win = run_nb(&mut gpu, &kv_normal, W as i32);
    let nb_win_huge = run_nb(&mut gpu, &kv_huge_outside, W as i32);
    let nb_bigw = run_nb(&mut gpu, &kv_normal, (S + 16) as i32);
    let nd_clip = max_abs_diff(&nb_full, &nb_win);
    let nd_invar = max_abs_diff(&nb_win, &nb_win_huge);
    let nd_equiv = max_abs_diff(&nb_full, &nb_bigw);
    println!("decode (non-batched): clip={nd_clip:.6} invar={nd_invar:.6} equiv={nd_equiv:.6}");
    if !(nd_clip > 1e-3) {
        eprintln!("FAIL decode(1): windowed == full; window had no effect");
        ok = false;
    }
    if !(nd_invar < 1e-3) {
        eprintln!("FAIL decode(2): huge out-of-window keys leaked past the mask");
        ok = false;
    }
    if !(nd_equiv < 1e-3) {
        eprintln!("FAIL decode(3): window>=seq over-clips in-window keys");
        ok = false;
    }

    if ok {
        println!("PASS: sliding-window masking correct (prefill + decode)");
    } else {
        std::process::exit(1);
    }
}
