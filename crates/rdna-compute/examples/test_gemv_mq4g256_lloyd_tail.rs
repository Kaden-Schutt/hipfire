//! Tail K-sweep parity test for gemv_mq4g256_lloyd.
//!
//! Sweeps groups_per_row ∈ {4, 5, 6, 7, 8, 16, 48} to exercise quad-clean
//! (4, 8, 16, 48) and all three tail cases (5, 6, 7) of the K4-unrolled
//! gfx1100 fast kernel — plus K=4096 (groups=16) and K=12288 (groups=48)
//! to mirror the real Qwen3.5-9B layer dims.
//!
//! Kernel selection follows the dispatch matcher: HIPFIRE_LLOYD_FORCE_BASELINE=1
//! routes to the slow generic kernel, unset (default) routes to the fast
//! variant on gfx1100/1101/1102/1151. Run both modes to verify the fast
//! kernel matches CPU reference *and* the slow kernel matches CPU reference.
//!
//! CPU reference is the bit-exact per-row formula from the Lloyd-MQ4 block
//! layout: 16 fp16 centroids per group, 256 weights per group, packed 4-bit
//! nibble-pair indices at byte_off = 32 + tid*4.
//!
//! Fails if max-abs error > 5e-3 (fp32 summation reorder noise scales with K;
//! K=12288 has ~3× more accumulation than the MQ3 test's K=2048).

use rdna_compute::{Gpu, DType};

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h: u16 = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) {
            new_mant += 1;
        }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 {
            new_mant = 0;
            exp_bits += 1;
        }
        (sign << 15) | (exp_bits << 10) | new_mant
    };
    h.to_le_bytes()
}

fn f16_le_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let mant32 = (m & 0x3ff) << 13;
            let exp32 = (e + 127 - 14) as u32;
            (sign << 31) | (exp32 << 23) | mant32
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// MQ4-Lloyd nibble-pair packing: 256 indices → 128 bytes.
/// byte[i] low nibble = idx[2i], high nibble = idx[2i+1].
fn pack_4bit_group(qs: &[u8; 256]) -> [u8; 128] {
    let mut out = [0u8; 128];
    for i in 0..128 {
        let lo = qs[2 * i]     & 0x0F;
        let hi = qs[2 * i + 1] & 0x0F;
        out[i] = lo | (hi << 4);
    }
    out
}

fn build_lloyd_mq4_row(
    groups_per_row: usize,
    codebooks: &[[f32; 16]],
    indices: &[[u8; 256]],
) -> (Vec<u8>, Vec<[f32; 16]>) {
    let mut out = Vec::with_capacity(groups_per_row * 160);
    let mut roundtripped = Vec::with_capacity(groups_per_row);
    for g in 0..groups_per_row {
        let cb = &codebooks[g];
        let mut cb_rt = [0.0f32; 16];
        for (i, &v) in cb.iter().enumerate() {
            let bytes = f32_to_f16_le(v);
            out.extend_from_slice(&bytes);
            cb_rt[i] = f16_le_to_f32(bytes);
        }
        roundtripped.push(cb_rt);
        let packed = pack_4bit_group(&indices[g]);
        out.extend_from_slice(&packed);
    }
    assert_eq!(out.len(), groups_per_row * 160);
    (out, roundtripped)
}

fn cpu_reference(
    groups_per_row: usize,
    m: usize,
    x: &[f32],
    codebooks_per_row: &[Vec<[f32; 16]>],
    indices_per_row: &[Vec<[u8; 256]>],
) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let cbs = &codebooks_per_row[row];
        let idxs = &indices_per_row[row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let cb = &cbs[g];
            let qs = &idxs[g];
            for i in 0..256 {
                let q = qs[i] as usize & 0xF;
                acc += cb[q] * x[g * 256 + i];
            }
        }
        y[row] = acc;
    }
    y
}

fn run_one(gpu: &mut Gpu, groups_per_row: usize) -> (f32, f32) {
    let m = 64;
    let k = groups_per_row * 256;

    let mut a_rows = Vec::with_capacity(m);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            // Synthetic 16-entry codebook around zero, varies per (row, g).
            let base = ((row * 7 + g * 11) % 19) as f32 * 0.013 - 0.1;
            let cb: [f32; 16] = std::array::from_fn(|i| base + (i as f32 - 7.5) * 0.018);
            cbs.push(cb);

            // Synthetic indices in [0, 16).
            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)) & 0xF) as u8;
            }
            idxs.push(q);
        }
        let (row_bytes, cbs_rt) = build_lloyd_mq4_row(groups_per_row, &cbs, &idxs);
        a_rows.push(row_bytes);
        codebooks_per_row.push(cbs_rt);
        indices_per_row.push(idxs);
    }

    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    let mut a_flat: Vec<u8> = Vec::with_capacity(m * groups_per_row * 160);
    for row in &a_rows {
        a_flat.extend_from_slice(row);
    }

    let d_a = gpu.upload_raw(&a_flat, &[a_flat.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();

    gpu.gemv_mq4g256_lloyd(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let y_ref = cpu_reference(groups_per_row, m, &x, &codebooks_per_row, &indices_per_row);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..m {
        let abs = (y_gpu[i] - y_ref[i]).abs();
        let denom = y_ref[i].abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
    }
    gpu.free_tensor(d_a).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
    (max_abs, max_rel)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    eprintln!(
        "HIPFIRE_LLOYD_FORCE_BASELINE: {:?}",
        std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE").unwrap_or_default()
    );

    let mut all_pass = true;
    let cases: &[(usize, &str)] = &[
        (4,  "K= 1024  (1 quad,  0 tail)"),
        (5,  "K= 1280  (1 quad,  1 tail)"),
        (6,  "K= 1536  (1 quad,  2 tail)"),
        (7,  "K= 1792  (1 quad,  3 tail)"),
        (8,  "K= 2048  (2 quads, 0 tail)"),
        (16, "K= 4096  (4 quads, 0 tail)  ← Qwen3.5-9B attn proj K"),
        (48, "K=12288  (12 quads, 0 tail) ← Qwen3.5-9B FFN K"),
    ];
    for &(gpr, tag) in cases {
        let (max_abs, max_rel) = run_one(&mut gpu, gpr);
        let pass = max_abs < 5e-3;
        let verdict = if pass { "PASS" } else { "FAIL" };
        println!(
            "groups_per_row={gpr:2}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  {verdict}  {tag}",
        );
        if !pass { all_pass = false; }
    }
    if !all_pass {
        eprintln!("\nFAIL: one or more cases produced max_abs > 5e-3");
        std::process::exit(1);
    }
    println!("\nALL PASS — fast kernel matches CPU reference across the K sweep");
}
