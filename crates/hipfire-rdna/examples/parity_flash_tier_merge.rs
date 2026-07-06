// SPDX-License-Identifier: Apache-2.0
//! Parity for `flash_tier_merge` (deferred-hierarchical KV, Phase 2b hot+cold merge).
//!
//! The merge folds two flash tiers' (out, m, l) partials into one via online
//! softmax. The flash invariant: merging partials over a PARTITION of the keys
//! reproduces a single full-softmax attention over all the keys. So we split the
//! cold slots into group A (the "hot"-role tier) and group B (the "cold"-role
//! tier), run `attention_cold_slots` on each to get partials, `flash_tier_merge`
//! them, and compare against `attention_cold_slots` over ALL slots at once. This
//! validates both the kernel's (m,l) emission and the merge math on real GPU.
//! f32 throughout → expect bit-close. The merge is mask-agnostic, so this is a
//! faithful proxy for hot(causal KVarN flash) ⊕ cold(all-visible) in production.
//!
//!   cargo run --release -p hipfire-rdna --example parity_flash_tier_merge [n_total] [n_a]

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s as f32 / 2_147_483_648.0 - 0.5) * 2.0
        })
        .collect()
}

/// Copy a slot sub-range [s0, s0+ns) out of a [nkv, ns_total, d] buffer into a
/// fresh contiguous [nkv, ns, d] buffer.
fn slice_slots(
    src: &[f32],
    nkv: usize,
    ns_total: usize,
    d: usize,
    s0: usize,
    ns: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; nkv * ns * d];
    for kv in 0..nkv {
        for s in 0..ns {
            let so = ((kv * ns_total) + s0 + s) * d;
            let do_ = ((kv * ns) + s) * d;
            out[do_..do_ + d].copy_from_slice(&src[so..so + d]);
        }
    }
    out
}

fn main() {
    let nt: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(88);
    let na: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(nt / 3);
    let nb = nt - na;
    assert!(
        na > 0 && nb > 0,
        "need both groups non-empty (nt={nt} na={na})"
    );
    let (nh, nkv, d) = (8usize, 2usize, 256usize); // qwen3.5-0.8b FA shape
    let scale = 1.0 / (d as f32).sqrt();

    let q = lcg(1, nh * d);
    let k = lcg(2, nkv * nt * d);
    let v = lcg(3, nkv * nt * d);

    let ka = slice_slots(&k, nkv, nt, d, 0, na);
    let va = slice_slots(&v, nkv, nt, d, 0, na);
    let kb = slice_slots(&k, nkv, nt, d, na, nb);
    let vb = slice_slots(&v, nkv, nt, d, na, nb);

    let mut gpu = Gpu::init().unwrap();
    let qd = gpu.upload_f32(&q, &[nh, d]).unwrap();

    // helper: run cold-slot attention on a [nkv, ns, d] buffer → (out, m, l) tensors
    let run_tier = |gpu: &mut Gpu, kbuf: &[f32], vbuf: &[f32], ns: usize| {
        let kd = gpu.upload_f32(kbuf, &[nkv, ns, d]).unwrap();
        let vd = gpu.upload_f32(vbuf, &[nkv, ns, d]).unwrap();
        let od = gpu.alloc_tensor(&[nh * d], DType::F32).unwrap();
        let md = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
        let ld = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
        gpu.attention_cold_slots(
            &qd, &kd, &vd, &od, &md, &ld, nh, nkv, ns, scale, 0, 0, 0, None,
        )
        .unwrap();
        (od, md, ld)
    };

    let (oa, ma, la) = run_tier(&mut gpu, &ka, &va, na);
    let (ob, mb, lb) = run_tier(&mut gpu, &kb, &vb, nb);

    // Merge the two partials.
    let omr = gpu.alloc_tensor(&[nh * d], DType::F32).unwrap();
    let mmr = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
    let lmr = gpu.alloc_tensor(&[nh], DType::F32).unwrap();
    gpu.flash_tier_merge(&oa, &ma, &la, &ob, &mb, &lb, &omr, &mmr, &lmr, nh)
        .unwrap();

    // Reference: full attention over all nt slots.
    let (of, _mf, _lf) = run_tier(&mut gpu, &k, &v, nt);

    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&omr).unwrap();
    let want = gpu.download_f32(&of).unwrap();

    let mut maxd = 0.0f32;
    let mut mag = 0.0f32;
    for i in 0..nh * d {
        maxd = maxd.max((got[i] - want[i]).abs());
        mag = mag.max(want[i].abs());
    }
    let tol = 2e-4 * mag.max(1.0);
    let pass = maxd <= tol;
    println!(
        "flash_tier_merge parity nh={nh} nkv={nkv} nt={nt} (a={na}+b={nb}) d={d} on {}: max_abs={maxd:.6} (mag={mag:.3}) tol={tol:.6} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
