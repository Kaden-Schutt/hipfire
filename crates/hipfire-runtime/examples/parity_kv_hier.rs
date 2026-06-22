// SPDX-License-Identifier: Apache-2.0
//! End-to-end parity for the hierarchical KV state machine (Phase 2b sub-task 4c):
//! feed N tokens through `HierKvState::append_token` (triggering several hot→cold
//! migrations), then `two_tier_read` a query, and compare against a CPU softmax
//! over the SAME data actually stored on GPU (hot ring slots + every cold segment
//! dequantized from its 4-bit records). This validates the append/migrate/multi-
//! segment-read/merge orchestration; the underlying kernels + compaction are each
//! separately parity-locked. f16 cold dequant → f16 tolerance.
//!
//!   cargo run --release -p hipfire-runtime --example parity_kv_hier [n_tokens]

use hipfire_kvquant::kvarn::{dequantize_tile, unpack_kvarn_tile_bits};
use hipfire_runtime::kv_hier::HierKvState;
use rdna_compute::{DType, Gpu};

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

fn main() {
    let n_tok: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let kv_dim = NKV * HD;
    let scale = 1.0f32 / (HD as f32).sqrt();

    std::env::set_var("HIPFIRE_KV_HIERARCHICAL", "1");
    std::env::set_var("HIPFIRE_KV_HOT_BUDGET", "16");
    std::env::set_var("HIPFIRE_KV_MIGRATE_BATCH", "8");

    let mut gpu = Gpu::init().unwrap();
    let mut hier = HierKvState::from_env(&mut gpu, 1, NH, NKV, HD).unwrap();
    assert!(hier.enabled, "hierarchical not enabled");

    // Feed N tokens (layer 0). Keep host copies for nothing — we read back GPU state.
    for t in 0..n_tok {
        let k = lcg(1000 + t as u32, kv_dim);
        let v = lcg(5000 + t as u32, kv_dim);
        let kd = gpu.upload_f32(&k, &[kv_dim]).unwrap();
        let vd = gpu.upload_f32(&v, &[kv_dim]).unwrap();
        hier.append_token(&mut gpu, 0, &kd, &vd).unwrap();
    }

    // Optional: exercise the deferred between-turns drain. idle_compact moves hot
    // tokens into cold; the read+oracle below validate it preserves attention.
    if std::env::var("HIPFIRE_KV_HIER_TEST_IDLE").ok().as_deref() == Some("1") {
        let keep: usize = std::env::var("HIPFIRE_KV_HIER_KEEP").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        hier.idle_compact(&mut gpu, keep).unwrap();
        eprintln!("after idle_compact(keep={keep}): hot_count={} cold_segs={}", hier.hot_count[0], hier.cold[0].len());
    }

    let q = lcg(7, NH * HD);
    let qd = gpu.upload_f32(&q, &[NH, HD]).unwrap();
    let out = gpu.alloc_tensor(&[NH * HD], DType::F32).unwrap();
    hier.two_tier_read(&mut gpu, 0, &qd, &out).unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&out).unwrap();

    // ── CPU oracle over the SAME stored data.
    // Hot: first hot_count slots of the slot-major ring [nkv × hot_budget × HD].
    let hot_count = hier.hot_count[0];
    let hb = hier.hot_budget;
    let hk = gpu.download_f32(&hier.hot_k[0]).unwrap();
    let hv = gpu.download_f32(&hier.hot_v[0]).unwrap();
    // Per kv-head list of (k,v) slots in attention order (hot then each cold seg).
    let mut k_slots: Vec<Vec<[f32; HD]>> = vec![Vec::new(); NKV];
    let mut v_slots: Vec<Vec<[f32; HD]>> = vec![Vec::new(); NKV];
    for kv in 0..NKV {
        for s in 0..hot_count {
            let base = (kv * hb + s) * HD;
            let mut ka = [0.0f32; HD];
            let mut va = [0.0f32; HD];
            ka.copy_from_slice(&hk[base..base + HD]);
            va.copy_from_slice(&hv[base..base + HD]);
            k_slots[kv].push(ka);
            v_slots[kv].push(va);
        }
    }
    // Cold: dequantize each segment's 4-bit records (rotate=false → directly usable).
    let n_seg = hier.cold[0].len();
    for seg in &hier.cold[0] {
        let krec_f32 = gpu.download_f32(&seg.k_recs).unwrap();
        let vrec_f32 = gpu.download_f32(&seg.v_recs).unwrap();
        let krec: Vec<u8> = krec_f32.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        let vrec: Vec<u8> = vrec_f32.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        for kv in 0..NKV {
            let off = kv * seg.rec_bytes;
            let kt = dequantize_tile(&unpack_kvarn_tile_bits(&krec[off..off + seg.rec_bytes], HD, seg.n_slots, seg.bits)); // [HD × n_slots]
            let vt = dequantize_tile(&unpack_kvarn_tile_bits(&vrec[off..off + seg.rec_bytes], HD, seg.n_slots, seg.bits));
            for s in 0..seg.n_valid {
                let mut ka = [0.0f32; HD];
                let mut va = [0.0f32; HD];
                for d in 0..HD {
                    ka[d] = kt[d * seg.n_slots + s];
                    va[d] = vt[d * seg.n_slots + s];
                }
                k_slots[kv].push(ka);
                v_slots[kv].push(va);
            }
        }
    }

    let mut oracle = vec![0.0f32; NH * HD];
    for hq in 0..NH {
        let kvh = hq / (NH / NKV);
        let qh = &q[hq * HD..hq * HD + HD];
        let ks = &k_slots[kvh];
        let vs = &v_slots[kvh];
        let n = ks.len();
        let mut logits = vec![0.0f32; n];
        for s in 0..n {
            logits[s] = (0..HD).map(|i| qh[i] * ks[s][i]).sum::<f32>() * scale;
        }
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|x| (x - mx).exp()).collect();
        let z: f32 = p.iter().sum();
        for x in &mut p {
            *x /= z;
        }
        for s in 0..n {
            for i in 0..HD {
                oracle[hq * HD + i] += p[s] * vs[s][i];
            }
        }
    }

    let mut maxd = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..NH * HD {
        maxd = maxd.max((got[i] - oracle[i]).abs());
        mag = mag.max(oracle[i].abs());
    }
    let tol = 2.5e-3f32;
    let pass = maxd <= tol;
    println!(
        "kv_hier E2E parity n_tok={n_tok} hot_count={hot_count} cold_segs={n_seg} (total slots={}) on {}: max_abs={maxd:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        k_slots[0].len(),
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
