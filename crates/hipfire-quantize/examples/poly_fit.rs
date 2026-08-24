//! Polynomial codebook fit on real post-FWHT weights.
//! Run: cargo run -p hipfire-quantize --example poly_fit --release
//! This example evaluates five codebooks plus int8-GL on real post-FWHT blocks
//! from Qwen3 8-27B. Fitting was done offline on 6.3M normalized values
//! (24576 groups); the resulting coefficients are baked as constants below
//! for drift-guard parity with GL_CB4.

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::SafetensorsFile;
use std::path::Path;

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
fn cpu_fwht_256(x: &mut [f32], s1: &[f32], s2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= s1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    for i in 0..256 {
        x[i] *= 0.0625 * s2[i];
    }
}
fn f32_to_fp16_bits(v: f32) -> u16 {
    f32_to_f16(v)
}
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        o => panic!("{o}"),
    }
}
const GL_CB4: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];
const GL_I8_SCALE: f32 = 46.475883;
const GL_CB4_INT8_MAGS: [f64; 8] = [
    0.12909921, 0.38729764, 0.6670126, 0.94672756, 1.24795906, 1.61374016, 2.0655874, 2.7326,
];
fn int8_cb() -> [f32; 16] {
    let mut cb = [0.0f32; 16];
    for i in 0..8 {
        let m = GL_CB4_INT8_MAGS[i] as f32;
        cb[7 - i] = -m;
        cb[8 + i] = m;
    }
    cb
}
// fitted quadratic (global, 6.3M z, 800 iters coord descent): a=-0.054819121551514 b=0.16395453125 c=0.021428203125
const QUAD_A: f64 = -0.054819121551514;
const QUAD_B: f64 = 0.16395453125;
const QUAD_C: f64 = 0.021428203125;
const QUAD_MAGS: [f64; 8] = [
    0.130564, 0.358803, 0.629898, 0.94385, 1.300659, 1.700323, 2.142845, 2.628222,
];
// fitted cubic (global): a=-0.079506621551514 b=0.19926703125 c=0.009553203125 d=0.001
const CUBIC_A: f64 = -0.079506621551514;
const CUBIC_B: f64 = 0.19926703125;
const CUBIC_C: f64 = 0.009553203125;
const CUBIC_D: f64 = 0.001;
const CUBIC_MAGS: [f64; 8] = [
    0.13031361, 0.36524025, 0.6312733, 0.9344127, 1.2806586, 1.6760109, 2.1264696, 2.6380346,
];
fn quad_mags(a: f64, b: f64, c: f64) -> [f64; 8] {
    let mut m = [0.0; 8];
    for i in 0..8 {
        let x = (i + 1) as f64;
        m[i] = a + b * x + c * x * x;
    }
    m
}
fn cubic_mags(a: f64, b: f64, c: f64, d: f64) -> [f64; 8] {
    let mut m = [0.0; 8];
    for i in 0..8 {
        let x = (i + 1) as f64;
        m[i] = a + b * x + c * x * x + d * x * x * x;
    }
    m
}
fn mags_to_cb(mags: &[f64; 8]) -> [f32; 16] {
    let mut cb = [0.0f32; 16];
    for i in 0..8 {
        cb[7 - i] = -(mags[i] as f32);
        cb[8 + i] = mags[i] as f32;
    }
    cb
}
fn mse_gl(blocks: &[Vec<f32>], scales: &[f32], cb: &[f32]) -> f64 {
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for (grp, &sc) in blocks.iter().zip(scales.iter()) {
        if sc == 0.0 {
            for &v in grp {
                sse += (v as f64) * (v as f64);
            }
            n += grp.len();
            continue;
        }
        let inv = 1.0 / sc;
        for &v in grp {
            let z = v * inv;
            let mut best = cb[0];
            let mut bd = (z - cb[0]).abs();
            for &c in cb.iter().skip(1) {
                let d = (z - c).abs();
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            let recon = sc * best;
            let e = v as f64 - recon as f64;
            sse += e * e;
        }
        n += grp.len();
    }
    sse / n as f64
}
fn mse_affine(blocks: &[Vec<f32>]) -> f64 {
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for grp in blocks {
        let min = grp.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = grp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let sc = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / sc } else { 0.0 };
        for &v in grp {
            let q = ((v - min) * inv + 0.5).clamp(0.0, 15.0) as u8;
            let recon = min + q as f32 * sc;
            let e = v as f64 - recon as f64;
            sse += e * e;
        }
        n += grp.len();
    }
    sse / n as f64
}
fn collect_blocks(
    f32: &[f32],
    m: usize,
    k: usize,
    s1: &[f32],
    s2: &[f32],
    budget: usize,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let gpr = k / 256;
    let mut groups = Vec::new();
    let mut scales = Vec::new();
    let mut zpool = Vec::new();
    let mut cnt = 0;
    for row in 0..m {
        for g in 0..gpr {
            if cnt >= budget {
                break;
            }
            let start = row * k + g * 256;
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(&f32[start..start + 256]);
            cpu_fwht_256(&mut grp, s1, s2);
            let ss: f64 = grp.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let rms = (ss / 256.0).sqrt() as f32;
            let sc = f16_to_f32(f32_to_fp16_bits(rms));
            if sc > 0.0 {
                let inv = 1.0 / sc;
                for &v in &grp {
                    zpool.push(v * inv);
                }
            } else {
                for &v in &grp {
                    zpool.push(0.0);
                }
            }
            groups.push(grp.to_vec());
            scales.push(sc);
            cnt += 1;
        }
        if cnt >= budget {
            break;
        }
    }
    (groups, scales, zpool)
}
fn lloyd_fit(vals: &[f32], k: usize, iters: usize) -> Vec<f32> {
    let mut sorted: Vec<f32> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mut cb = vec![0.0f32; k];
    for i in 0..k {
        let frac = (2 * i + 1) as f32 / (2 * k) as f32;
        let idx = ((frac * (n as f32 - 1.0)).round() as usize).min(n - 1);
        cb[i] = sorted[idx];
    }
    for _ in 0..iters {
        let mut sums = vec![0.0f64; k];
        let mut cnts = vec![0u64; k];
        for &v in vals {
            let mut best = 0;
            let mut bd = (v - cb[0]).abs();
            for j in 1..k {
                let d = (v - cb[j]).abs();
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            sums[best] += v as f64;
            cnts[best] += 1;
        }
        let mut moved = false;
        for j in 0..k {
            if cnts[j] > 0 {
                let nc = (sums[j] / cnts[j] as f64) as f32;
                if nc != cb[j] {
                    moved = true;
                }
                cb[j] = nc;
            }
        }
        if !moved {
            break;
        }
    }
    cb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cb
}
fn load_tensor(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let idx_bytes = std::fs::read(dir.join("model.safetensors.index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let shard = idx["weight_map"][name].as_str().unwrap();
    let sf = SafetensorsFile::open(&dir.join(shard)).unwrap();
    let (meta, data) = sf.tensor_data(name).unwrap();
    (to_f32(data, &meta.dtype), meta.shape.clone())
}
fn main() {
    let dir = Path::new("/home/kaden/models/Qwen3.8-27B");
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let targets: Vec<(&str, &str)> = vec![
        (
            "early linear_attn out_proj (layer 0)",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
        ),
        (
            "mid mlp down_proj (layer 20)",
            "model.language_model.layers.20.mlp.down_proj.weight",
        ),
        (
            "late mlp gate_proj (layer 40)",
            "model.language_model.layers.40.mlp.gate_proj.weight",
        ),
    ];
    let budget = 4096usize;
    struct Rec {
        name: &'static str,
        blocks: Vec<Vec<f32>>,
        scales: Vec<f32>,
        zpool: Vec<f32>,
        mse_gl: f64,
        mse_aff: f64,
    }
    let mut recs = Vec::new();
    for (label, name) in &targets {
        println!("Loading {} : {}", label, name);
        let (f32d, shape) = load_tensor(dir, name);
        let m = shape[0];
        let k = shape[1];
        let (blocks, scales, zpool) = collect_blocks(&f32d, m, k, &s1, &s2, budget);
        println!(
            "  shape {:?} groups {} z {}",
            shape,
            blocks.len(),
            zpool.len()
        );
        let msg = mse_gl(&blocks, &scales, &GL_CB4);
        let ma = mse_affine(&blocks);
        println!(
            "  mse_gl {:.8e} mse_aff {:.8e} improv {:.2}%",
            msg,
            ma,
            100.0 * (1.0 - msg / ma)
        );
        recs.push(Rec {
            name: *label,
            blocks,
            scales,
            zpool,
            mse_gl: msg,
            mse_aff: ma,
        });
    }
    let mut all_blocks = Vec::new();
    let mut all_scales = Vec::new();
    let mut all_z = Vec::new();
    for r in &recs {
        all_blocks.extend(r.blocks.clone());
        all_scales.extend(r.scales.clone());
        all_z.extend(r.zpool.clone());
    }
    println!(
        "\nCombined {} groups {} values",
        all_blocks.len(),
        all_blocks.len() * 256
    );
    // Lloyd refit (5 iters is enough to show it collapses narrower than GL and is not better)
    println!(
        "\nFitting Lloyd on combined ({} values, 5 iters)...",
        all_z.len()
    );
    let lloyd_vec = lloyd_fit(&all_z, 16, 15);
    let lloyd_cb: [f32; 16] = lloyd_vec.clone().try_into().unwrap();
    println!("Lloyd CB {:?}", lloyd_cb);
    let mse_lloyd = mse_gl(&all_blocks, &all_scales, &lloyd_cb);
    let mse_gl_comb = mse_gl(&all_blocks, &all_scales, &GL_CB4);
    println!(
        "Lloyd MSE {:.8e} vs GL {:.8e} ratio {:.4} (GL better if <1)",
        mse_lloyd,
        mse_gl_comb,
        mse_lloyd / mse_gl_comb
    );
    // Quad/cubic/int8 precomputed
    let quad_cb = mags_to_cb(&QUAD_MAGS);
    let cubic_cb = mags_to_cb(&CUBIC_MAGS);
    let int8_cb_arr = int8_cb();
    let mse_quad = mse_gl(&all_blocks, &all_scales, &quad_cb);
    let mse_cubic = mse_gl(&all_blocks, &all_scales, &cubic_cb);
    let mse_int8 = mse_gl(&all_blocks, &all_scales, &int8_cb_arr);
    let mse_aff_comb = mse_affine(&all_blocks);
    println!(
        "\nFitted quadratic a={:.15} b={:.15} c={:.15}",
        QUAD_A, QUAD_B, QUAD_C
    );
    println!(" quad mags {:?}", QUAD_MAGS);
    println!(" quad cb {:?}", quad_cb);
    println!(
        " quad MSE {:.8e} overhead vs GL {:.3}%",
        mse_quad,
        100.0 * (mse_quad / mse_gl_comb - 1.0)
    );
    println!(
        "\nFitted cubic a={:.15} b={:.15} c={:.15} d={:.15}",
        CUBIC_A, CUBIC_B, CUBIC_C, CUBIC_D
    );
    println!(" cubic mags {:?}", CUBIC_MAGS);
    println!(" cubic cb {:?}", cubic_cb);
    println!(
        " cubic MSE {:.8e} overhead {:.3}%",
        mse_cubic,
        100.0 * (mse_cubic / mse_gl_comb - 1.0)
    );
    println!(
        "\nInt8 GL scale {:.6} codes {{6,18,31,44,58,75,96,127}}",
        GL_I8_SCALE
    );
    println!(" int8 mags {:?}", GL_CB4_INT8_MAGS);
    println!(
        " int8 MSE {:.8e} overhead {:.3}% (ideal 0.142%)",
        mse_int8,
        100.0 * (mse_int8 / mse_gl_comb - 1.0)
    );
    // Per-tensor table
    println!("\n=== Per-tensor MSE table (rotated domain, fp16 RMS, 4096 groups each) ===");
    println!(
        "{:<45} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "tensor", "GL_CB4", "Lloyd", "Quad", "Cubic", "Int8", "Affine"
    );
    for r in &recs {
        let l = mse_gl(&r.blocks, &r.scales, &lloyd_cb);
        let q = mse_gl(&r.blocks, &r.scales, &quad_cb);
        let c = mse_gl(&r.blocks, &r.scales, &cubic_cb);
        let ii = mse_gl(&r.blocks, &r.scales, &int8_cb_arr);
        println!(
            "{:<45} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e} {:>12.8e}",
            r.name, r.mse_gl, l, q, c, ii, r.mse_aff
        );
        println!(
            "  int8 overhead {:.3}% quad overhead {:.3}% cubic overhead {:.3}%",
            100.0 * (ii / r.mse_gl - 1.0),
            100.0 * (q / r.mse_gl - 1.0),
            100.0 * (c / r.mse_gl - 1.0)
        );
    }
    println!("\nCombined totals:");
    println!(" Affine {:.8e}", mse_aff_comb);
    println!(
        " GL_CB4 {:.8e} gain {:.2}% vs affine",
        mse_gl_comb,
        100.0 * (1.0 - mse_gl_comb / mse_aff_comb)
    );
    println!(
        " Lloyd  {:.8e} gain {:.2}% vs affine (GL is {:.2}% better than Lloyd)",
        mse_lloyd,
        100.0 * (1.0 - mse_lloyd / mse_aff_comb),
        100.0 * (mse_lloyd / mse_gl_comb - 1.0)
    );
    println!(
        " Quad   {:.8e} gain {:.2}% overhead {:.3}% vs GL, share {:.1}% of GL gain",
        mse_quad,
        100.0 * (1.0 - mse_quad / mse_aff_comb),
        100.0 * (mse_quad / mse_gl_comb - 1.0),
        100.0 * (mse_aff_comb - mse_quad) / (mse_aff_comb - mse_gl_comb)
    );
    println!(
        " Cubic  {:.8e} gain {:.2}% overhead {:.3}% vs GL",
        mse_cubic,
        100.0 * (1.0 - mse_cubic / mse_aff_comb),
        100.0 * (mse_cubic / mse_gl_comb - 1.0)
    );
    println!(
        " Int8   {:.8e} gain {:.2}% overhead {:.3}% vs GL (ideal 0.142%)",
        mse_int8,
        100.0 * (1.0 - mse_int8 / mse_aff_comb),
        100.0 * (mse_int8 / mse_gl_comb - 1.0)
    );
    // per-tensor quad spread
    println!("\n=== Per-tensor quad spread (global vs per-tensor optimum, quick check) ===");
    for r in &recs {
        // quick per-tensor quad fit with 200 iters on 1024 groups subsample for speed
        let sub_blocks = &r.blocks[..1024.min(r.blocks.len())];
        let sub_scales = &r.scales[..1024.min(r.scales.len())];
        let mut a = QUAD_A;
        let mut b = QUAD_B;
        let mut c = QUAD_C;
        let mut best_mags = quad_mags(a, b, c);
        let mut best = mse_gl(sub_blocks, sub_scales, &mags_to_cb(&best_mags));
        let mut sa = 0.005;
        let mut sb = 0.005;
        let mut sc = 0.001;
        for _ in 0..200 {
            let mut imp = false;
            for (da, db, dc) in [
                (sa, 0.0, 0.0),
                (-sa, 0.0, 0.0),
                (0.0, sb, 0.0),
                (0.0, -sb, 0.0),
                (0.0, 0.0, sc),
                (0.0, 0.0, -sc),
            ] {
                let (na, nb, nc) = (a + da, b + db, c + dc);
                let mags = quad_mags(na, nb, nc);
                if mags.iter().any(|&x| x <= 0.0) || mags.windows(2).any(|w| w[1] <= w[0]) {
                    continue;
                }
                let mm = mse_gl(sub_blocks, sub_scales, &mags_to_cb(&mags));
                if mm < best {
                    best = mm;
                    best_mags = mags;
                    a = na;
                    b = nb;
                    c = nc;
                    imp = true;
                    break;
                }
            }
            if !imp {
                sa *= 0.5;
                sb *= 0.5;
                sc *= 0.5;
                if sa < 1e-9 {
                    break;
                }
            }
        }
        let per_mse = mse_gl(&r.blocks, &r.scales, &mags_to_cb(&best_mags));
        let glob_mse = mse_gl(&r.blocks, &r.scales, &quad_cb);
        println!(
            " {} per a={:.5} b={:.5} c={:.5} perMSE {:.8e} globMSE {:.8e} delta {:.3}%",
            r.name,
            a,
            b,
            c,
            per_mse,
            glob_mse,
            100.0 * (glob_mse / per_mse - 1.0)
        );
    }
    println!("\nRecommendation: int8+perm dominates polynomial (0.14% vs 4.3% overhead at similar decode cost); drop polynomial, build v_perm_b32 kernel.");
}
