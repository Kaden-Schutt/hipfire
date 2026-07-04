// SPDX-License-Identifier: Apache-2.0
//! Parity for `flash_partials_ml` + `flash_tier_merge` composing with the real
//! flash partials layout (deferred-hierarchical KV, Phase 2b sub-task 4).
//!
//! The hot KVarN/asym flash writes per-tile partials [m, l, acc...]; its reduce
//! folds them into a normalized output but discards the final (m, l). To merge
//! the hot tier with the cold tier we re-extract (m, l) via `flash_partials_ml`.
//!
//! We synthesize two partials buffers A (hot-role) and B (cold-role), take each
//! tier's normalized output from a trusted CPU reduce, get each tier's (m, l)
//! from the GPU `flash_partials_ml`, `flash_tier_merge` them on GPU, and compare
//! against a CPU reduce over the CONCATENATION A∪B — the flash invariant. The
//! oracle is independent of both GPU kernels under test, so a wrong (m, l) or a
//! wrong merge fails the check. Also directly verifies the extracted (m, l).
//!
//!   cargo run --release -p hipfire-rdna --example parity_flash_partials_ml

use hipfire_rdna::{DType, Gpu};

const NH: usize = 8;
const HD: usize = 256;
const TS: usize = 128; // tile_size == KVARN_GROUP
const STRIDE: usize = 2 + HD;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 / 2_147_483_648.0 - 0.5) * 2.0
        })
        .collect()
}

/// Synthesize a [NH × n_tiles × STRIDE] partials buffer: per (head,tile) a max
/// in [-2,2], a strictly-positive denom, and a random acc vector.
fn synth_partials(seed: u32, n_tiles: usize) -> Vec<f32> {
    let r = lcg(seed, NH * n_tiles * STRIDE);
    let mut p = vec![0.0f32; NH * n_tiles * STRIDE];
    for h in 0..NH {
        for t in 0..n_tiles {
            let base = (h * n_tiles + t) * STRIDE;
            p[base] = r[base] * 2.0; // m in [-2,2]
            p[base + 1] = 0.25 + (r[base + 1] + 1.0); // l in (0.25, 2.25], >0
            for d in 0..HD {
                p[base + 2 + d] = r[base + 2 + d];
            }
        }
    }
    p
}

/// CPU mirror of attention_flash_asym_reduce_batched (+ its (m,l)): fold the
/// per-tile partials of one buffer into (out[NH*HD], m[NH], l[NH]).
fn reduce_cpu(p: &[f32], n_tiles: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; NH * HD];
    let mut mv = vec![0.0f32; NH];
    let mut lv = vec![0.0f32; NH];
    for h in 0..NH {
        let mut gmax = f32::MIN;
        for t in 0..n_tiles {
            let b = (h * n_tiles + t) * STRIDE;
            if p[b + 1] > 0.0 {
                gmax = gmax.max(p[b]);
            }
        }
        let mut gsum = 0.0f32;
        let mut acc = vec![0.0f32; HD];
        for t in 0..n_tiles {
            let b = (h * n_tiles + t) * STRIDE;
            let l = p[b + 1];
            if l <= 0.0 {
                continue;
            }
            let corr = (p[b] - gmax).exp();
            for d in 0..HD {
                acc[d] += p[b + 2 + d] * corr;
            }
            gsum += l * corr;
        }
        let inv = if gsum > 0.0 { 1.0 / gsum } else { 0.0 };
        for d in 0..HD {
            out[h * HD + d] = acc[d] * inv;
        }
        mv[h] = gmax;
        lv[h] = gsum;
    }
    (out, mv, lv)
}

/// Concatenate A's tiles and B's tiles per head into one [NH × (ta+tb) × STRIDE].
fn concat_tiles(a: &[f32], ta: usize, b: &[f32], tb: usize) -> Vec<f32> {
    let tt = ta + tb;
    let mut out = vec![0.0f32; NH * tt * STRIDE];
    for h in 0..NH {
        for t in 0..ta {
            let src = (h * ta + t) * STRIDE;
            let dst = (h * tt + t) * STRIDE;
            out[dst..dst + STRIDE].copy_from_slice(&a[src..src + STRIDE]);
        }
        for t in 0..tb {
            let src = (h * tb + t) * STRIDE;
            let dst = (h * tt + ta + t) * STRIDE;
            out[dst..dst + STRIDE].copy_from_slice(&b[src..src + STRIDE]);
        }
    }
    out
}

fn maxabs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut md = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..a.len() {
        md = md.max((a[i] - b[i]).abs());
        mag = mag.max(b[i].abs());
    }
    (md, mag)
}

fn main() {
    let (ta, tb) = (3usize, 5usize);

    let pa = synth_partials(11, ta);
    let pb = synth_partials(22, tb);
    let pab = concat_tiles(&pa, ta, &pb, tb);

    let (out_a, ma_c, la_c) = reduce_cpu(&pa, ta);
    let (out_b, _mb_c, _lb_c) = reduce_cpu(&pb, tb);
    let (out_ab, _m_ab, _l_ab) = reduce_cpu(&pab, ta + tb);

    let mut gpu = Gpu::init().unwrap();

    // GPU extract (m,l) for A and B from their partials. positions so n_tiles match.
    let pa_d = gpu.upload_f32(&pa, &[NH * ta * STRIDE]).unwrap();
    let pb_d = gpu.upload_f32(&pb, &[NH * tb * STRIDE]).unwrap();
    let pos_a = gpu
        .upload_raw(&((ta * TS) as i32 - 1).to_le_bytes(), &[1])
        .unwrap();
    let pos_b = gpu
        .upload_raw(&((tb * TS) as i32 - 1).to_le_bytes(), &[1])
        .unwrap();

    let ma_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let la_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let mb_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let lb_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    gpu.flash_partials_ml(&pa_d, &pos_a, &ma_d, &la_d, NH, HD, TS, ta, 1, 0, 0, 0)
        .unwrap();
    gpu.flash_partials_ml(&pb_d, &pos_b, &mb_d, &lb_d, NH, HD, TS, tb, 1, 0, 0, 0)
        .unwrap();

    // Upload the trusted CPU tier outputs, then GPU-merge using the GPU-extracted (m,l).
    let oa_d = gpu.upload_f32(&out_a, &[NH * HD]).unwrap();
    let ob_d = gpu.upload_f32(&out_b, &[NH * HD]).unwrap();
    let om_d = gpu.alloc_tensor(&[NH * HD], DType::F32).unwrap();
    let mm_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    let lm_d = gpu.alloc_tensor(&[NH], DType::F32).unwrap();
    gpu.flash_tier_merge(
        &oa_d, &ma_d, &la_d, &ob_d, &mb_d, &lb_d, &om_d, &mm_d, &lm_d, NH,
    )
    .unwrap();

    gpu.device_synchronize().unwrap();

    // Direct check: GPU-extracted (m,l) for A vs CPU.
    let ma_g = gpu.download_f32(&ma_d).unwrap();
    let la_g = gpu.download_f32(&la_d).unwrap();
    let (mld, _) = maxabs(&ma_g, &ma_c);
    let (lld, lmag) = maxabs(&la_g, &la_c);
    let ml_ok = mld <= 1e-5 && lld <= 1e-4 * lmag.max(1.0);

    // Composed check: GPU merge == CPU reduce over A∪B.
    let merged = gpu.download_f32(&om_d).unwrap();
    let (md, mag) = maxabs(&merged, &out_ab);
    let tol = 2e-4 * mag.max(1.0);
    let merge_ok = md <= tol;

    println!(
        "flash_partials_ml (m,l) extract vs CPU: m_maxabs={mld:.3e} l_maxabs={lld:.3e} -> {}",
        if ml_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "hot⊕cold compose (GPU merge vs CPU reduce A∪B) nh={NH} ta={ta} tb={tb} on {}: max_abs={md:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        gpu.arch,
        if merge_ok { "PASS" } else { "FAIL" }
    );
    if !(ml_ok && merge_ok) {
        std::process::exit(1);
    }
}
