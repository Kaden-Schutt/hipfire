// SPDX-License-Identifier: Apache-2.0
//! End-to-end deferred-hierarchical KV read parity (Phase 2b sub-task 4): the
//! FULL production decode call sequence on real GPU vs the validated CPU oracle
//! `hipfire_kvquant::ColdTier::two_tier_attend`.
//!
//! Key fact exploited: for a single decode query at the LAST position, BOTH tiers
//! are all-visible — the hot tier holds the most recent W tokens (all causally
//! visible to the last query) and the cold tier holds compacted older tokens
//! (all visible). So `attention_cold_slots` is the correct primitive for both,
//! and `flash_tier_merge` folds them into one softmax — exactly what the CPU
//! oracle computes as a single concatenated-softmax.
//!
//! Pipeline (mirrors what the live forward would do per decode step):
//!   1. hot  = attention_cold_slots over the recent W raw tokens  → (out_h, m_h, l_h)
//!   2. cold = compact_cold_kv(older N-W tokens) [REAL compaction, KVarN-quantized,
//!             FWHT-rotated] → dequant_head per kv-head → upload f32 →
//!             attention_cold_slots                                  → (out_c, m_c, l_c)
//!   3. merge = flash_tier_merge(hot, cold)                          → out
//!   oracle = two_tier_attend(q, hot_k, hot_v, W, dequant_head(...)) per q-head.
//!
//! Tolerance is loose-ish (the cold tier is 4-bit + merged + rotated, lossy) but
//! the GPU pipeline must reproduce the CPU oracle to f32-arithmetic tolerance,
//! since BOTH consume the SAME dequantized cold slots — the only difference is
//! GPU vs CPU softmax/merge arithmetic. So expect bit-close, not just cos-close.
//!
//!   cargo run --release -p hipfire-rdna --example parity_two_tier_e2e [n_total] [w_hot]

use hipfire_kvquant::kv_compact::compact_cold_kv;
use hipfire_rdna::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 256;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 / 2_147_483_648.0 - 0.5) * 2.0
        })
        .collect()
}

/// Pack a per-kv-head [n_kv_heads][slots×HD] set of slabs into one contiguous
/// [n_kv_heads × slots × HD] GPU-layout buffer.
fn pack_kv(per_head: &[Vec<f32>], slots: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; NKV * slots * HD];
    for (h, slab) in per_head.iter().enumerate() {
        let base = h * slots * HD;
        out[base..base + slots * HD].copy_from_slice(slab);
    }
    out
}

fn main() {
    let nt: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    assert!(
        w < nt,
        "hot window must be smaller than total (nt={nt} w={w})"
    );
    let n_cold_tok = nt - w;
    let scale = 1.0 / (HD as f32).sqrt();
    let kv_dim = NKV * HD;

    // Full sequence K/V (token-major [nt × kv_dim]) + a single decode query.
    let q = lcg(1, NH * HD);
    let k_all = lcg(2, nt * kv_dim);
    let v_all = lcg(3, nt * kv_dim);

    // Split: cold = first n_cold_tok tokens (pos 0..n_cold_tok), hot = last w.
    let cold_k: Vec<f32> = k_all[..n_cold_tok * kv_dim].to_vec();
    let cold_v: Vec<f32> = v_all[..n_cold_tok * kv_dim].to_vec();
    let hot_off = n_cold_tok * kv_dim;
    // Importance: recency-ish weighting so the merge is non-uniform (older = lower).
    let importance: Vec<f32> = (0..n_cold_tok)
        .map(|t| 0.1 + (t as f32 / n_cold_tok as f32))
        .collect();

    // REAL cold compaction: KVarN 4-bit, FWHT-rotated, fold_m=4, 12.5% core exact.
    let cold = compact_cold_kv(
        &cold_k,
        &cold_v,
        n_cold_tok,
        NKV,
        HD,
        &importance,
        0.125,
        4,
        true,
        false,
        15.0,
    );
    let nvc = cold.n_valid;

    // Hot tier raw [nkv × w × HD] (regroup token-major → head-major slabs).
    let mut hot_slabs: Vec<Vec<f32>> = vec![vec![0.0f32; w * HD]; NKV];
    for t in 0..w {
        for h in 0..NKV {
            for d in 0..HD {
                hot_slabs[h][t * HD + d] = k_all[hot_off + t * kv_dim + h * HD + d];
            }
        }
    }
    let mut hot_v_slabs: Vec<Vec<f32>> = vec![vec![0.0f32; w * HD]; NKV];
    for t in 0..w {
        for h in 0..NKV {
            for d in 0..HD {
                hot_v_slabs[h][t * HD + d] = v_all[hot_off + t * kv_dim + h * HD + d];
            }
        }
    }

    // Cold tier dequantized [nkv × nvc × HD] (what BOTH tiers attend over).
    let cold_deq: Vec<(Vec<f32>, Vec<f32>)> = (0..NKV).map(|h| cold.dequant_head(h)).collect();
    let cold_k_packed = pack_kv(
        &cold_deq.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        nvc,
    );
    let cold_v_packed = pack_kv(
        &cold_deq.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>(),
        nvc,
    );
    let hot_k_packed = pack_kv(&hot_slabs, w);
    let hot_v_packed = pack_kv(&hot_v_slabs, w);

    // ── CPU oracle: two_tier_attend per q-head.
    let q_pos = nt - 1;
    let hot_base_pos = n_cold_tok;
    let mut oracle = vec![0.0f32; NH * HD];
    for hq in 0..NH {
        let kvh = hq / (NH / NKV);
        let (ck, cv) = &cold_deq[kvh];
        let o = cold.two_tier_attend(
            &q[hq * HD..hq * HD + HD],
            &hot_slabs[kvh],
            &hot_v_slabs[kvh],
            w,
            ck,
            cv,
            q_pos,
            hot_base_pos,
            HD,
        );
        oracle[hq * HD..hq * HD + HD].copy_from_slice(&o);
    }

    // ── GPU pipeline.
    let mut gpu = Gpu::init().unwrap();
    let qd = gpu.upload_f32(&q, &[NH, HD]).unwrap();

    let run_tier = |gpu: &mut Gpu, kp: &[f32], vp: &[f32], ns: usize| {
        let kd = gpu.upload_f32(kp, &[NKV, ns, HD]).unwrap();
        let vd = gpu.upload_f32(vp, &[NKV, ns, HD]).unwrap();
        let od = gpu.alloc_tensor(&[NH * HD], DType::F32).unwrap();
        let md = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
        let ld = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
        gpu.attention_cold_slots(&qd, &kd, &vd, &od, &md, &ld, NH, NKV, ns, scale, 0, 0, None)
            .unwrap();
        (od, md, ld)
    };

    let (oh, mh, lh) = run_tier(&mut gpu, &hot_k_packed, &hot_v_packed, w);
    let (oc, mc, lc) = run_tier(&mut gpu, &cold_k_packed, &cold_v_packed, nvc);

    let om = gpu.alloc_tensor(&[NH * HD], DType::F32).unwrap();
    let mm = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let lm = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    gpu.flash_tier_merge(&oh, &mh, &lh, &oc, &mc, &lc, &om, &mm, &lm, NH)
        .unwrap();

    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&om).unwrap();

    let mut maxd = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..NH * HD {
        maxd = maxd.max((got[i] - oracle[i]).abs());
        mag = mag.max(oracle[i].abs());
    }
    let tol = 5e-4 * mag.max(1.0);
    let pass = maxd <= tol;
    println!(
        "two-tier E2E parity (GPU pipeline vs CPU two_tier_attend) nt={nt} w_hot={w} n_cold_slots={nvc} (from {n_cold_tok} tok) on {}: max_abs={maxd:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
