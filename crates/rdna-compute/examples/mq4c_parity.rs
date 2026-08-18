//! Correctness gate for MQ4C / v1.5 (qt=45, 132 B/group) decode.
//!
//! v1 and mq4c encode the SAME per-256 affine grid; mq4c just narrows the header to
//! one fp16 pair, which moves the payload from +8 to +4 and the stride from 136 to
//! 132. A repack keeps every nibble (measured: 0 flips in 95,119,360 real groups), so
//! the two containers must decode to the same numbers within fp16 header rounding —
//! measured at 0.0909% weight RMS on a real artifact.
//!
//! That gives a cross-check needing no host model of the kernel: pack one weight set
//! both ways, run each through its own GEMV, and require agreement at the header
//! rounding scale. The failure modes this catches are exactly the ones that are
//! otherwise silent:
//!   * wrong stride (136 vs 132) — every group after the first reads misaligned
//!     bytes, and the result is noise rather than an error;
//!   * wrong payload offset (+8 vs +4) — nibbles read from the header, or the header
//!     read from nibbles;
//!   * f32 header read where an fp16 pair lives — bit_casts a small fp16 pair to
//!     ~1e-14 and zeroes the tensor at full speed. This is the qt=44 failure that
//!     cost four measurement cycles this week.
//!
//! Run: `cargo run --release -p rdna-compute --example mq4c_parity`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const V1_BYTES: usize = 136;
const MQ4C_BYTES: usize = 132;

const CACHE_POLICY: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc")
);
const V1_SRC: &str = include_str!("../../../kernels/src/gemv_hfq4g256_residual.hip");
const MQ4C_SRC: &str = include_str!("../../../kernels/src/gemv_mq4cg256_residual.hip");

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = b & 0x007F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1F {
        return sign | 0x7C00;
    }
    let mut m = (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0FFF) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1F) as u32;
    let m = (bits & 0x3FF) as u32;
    let out = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal — real artifacts DO hit this (min scale measured 1.370e-06,
            // 45x below the fp16 normal floor), so it must be handled correctly.
            let mut e2 = -1i32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            m2 &= 0x3FF;
            (s << 31) | (((e2 + 15 + 112) as u32) << 23) | (m2 << 13)
        }
    } else if e == 0x1F {
        (s << 31) | 0x7F80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(out)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Pack the same weights into both containers, sharing one set of nibbles exactly as
/// the repack does.
fn pack_both(w: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
    let gpr = k / GROUP;
    let mut b1 = vec![0u8; m * gpr * V1_BYTES];
    let mut bc = vec![0u8; m * gpr * MQ4C_BYTES];
    let mut deq_c = vec![0.0f64; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let d1 = (r * gpr + g) * V1_BYTES;
            let dc = (r * gpr + g) * MQ4C_BYTES;
            let s = &w[src..src + GROUP];
            let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            b1[d1..d1 + 4].copy_from_slice(&step.to_le_bytes());
            b1[d1 + 4..d1 + 8].copy_from_slice(&lo.to_le_bytes());
            let sb = f16_bits(step);
            let zb = f16_bits(lo);
            bc[dc..dc + 2].copy_from_slice(&sb.to_le_bytes());
            bc[dc + 2..dc + 4].copy_from_slice(&zb.to_le_bytes());
            let inv = if step > 0.0 { 1.0 / step } else { 0.0 };
            let (sc, zc) = (f16_to_f32(sb), f16_to_f32(zb));
            let mut q = [0u8; GROUP];
            for i in 0..GROUP {
                q[i] = ((s[i] - lo) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                deq_c[src + i] = (zc + sc * q[i] as f32) as f64;
            }
            for i in 0..128 {
                let byte = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
                b1[d1 + 8 + i] = byte;
                bc[dc + 4 + i] = byte; // identical nibbles — this IS the repack
            }
        }
    }
    (b1, bc, deq_c)
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq4c_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!("mq4c_parity: arch={}", gpu.arch);

    let (m, k) = (512usize, 1024usize);
    let w: Vec<f32> = (0..m * k)
        .map(|i| {
            let u1 = prng(i, 0x1234_5678).max(1e-7);
            let u2 = prng(i, 0x9ABC_DEF0);
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * 0.011
        })
        .collect();
    let (b1, bc, deq_c) = pack_both(&w, m, k);
    assert_eq!(bc.len() * 136, b1.len() * 132, "container size ratio");

    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xC0FF_EE) * 2.0 - 1.0).collect();

    // Host reference for the mq4c container, from its own exact dequant.
    let gpr = k / GROUP;
    let _ = gpr;
    let mut want = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for c in 0..k {
            acc += deq_c[r * k + c] * x[c] as f64;
        }
        want[r] = acc;
    }

    let run = |gpu: &mut Gpu, sym: &str, src: &str, blob: &[u8]| -> Vec<f32> {
        let full = format!("#define HIPFIRE_RESIDUAL_KERNEL {sym}\n#define HIPFIRE_MQ4C_RESIDUAL_KERNEL {sym}\n{CACHE_POLICY}{src}");
        gpu.ensure_kernel_public(&format!("mod_{sym}"), &full, sym)
            .unwrap_or_else(|e| panic!("compile {sym}: {e}"));
        let d_a = gpu.upload_raw(blob, &[blob.len()]).unwrap();
        let d_x = gpu.upload_f32(&x, &[k]).unwrap();
        let d_y = gpu.zeros(&[m], DType::F32).unwrap();
        let mut kb = KernargBlob::new();
        kb.push_ptr(d_a.buf.as_ptr());
        kb.push_ptr(d_x.buf.as_ptr());
        kb.push_ptr(d_y.buf.as_ptr());
        kb.push_i32(m as i32);
        kb.push_i32(k as i32);
        gpu.launch_kernel_blob(sym, [((m as u32) + 1) / 2, 1, 1], [32, 1, 1], 0, kb.as_mut_slice())
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.download_f32(&d_y).unwrap()
    };

    let y_v1 = run(&mut gpu, "parity_v1_res", V1_SRC, &b1);
    let y_c = run(&mut gpu, "parity_mq4c_res", MQ4C_SRC, &bc);

    let rel = |a: &[f32], b: &[f64]| -> f64 {
        let (mut n, mut d) = (0.0f64, 0.0f64);
        for (&p, &q) in a.iter().zip(b.iter()) {
            n += ((p as f64) - q) * ((p as f64) - q);
            d += q * q;
        }
        (n / d.max(1e-30)).sqrt()
    };
    let e_c = rel(&y_c, &want);

    // v1 decodes a very slightly different grid (f32 header), so it will not match
    // mq4c's oracle exactly. Its error against that oracle is the header-rounding
    // scale, ~1e-3, and serves as the sanity band.
    let e_v1 = rel(&y_v1, &want);

    eprintln!("mq4c vs its own exact dequant : {e_c:.3e}");
    eprintln!("v1   vs mq4c's dequant        : {e_v1:.3e}   (header-rounding scale)");

    const TOL: f64 = 1e-4;
    assert!(
        e_c < TOL,
        "MQ4C decode disagrees with its own exact dequant: {e_c:.3e} > {TOL:.0e}.\n\
         A stride error (132 vs 136), a payload-offset error (+4 vs +8), or an f32 read \
         of the fp16 header all land here, and none of them produce a runtime error — \
         the qt=44 version of this bug ran at full speed and returned noise."
    );
    assert!(
        e_v1 < 1e-2,
        "v1 and mq4c disagree far beyond header rounding ({e_v1:.3e}); the two \
         containers are supposed to encode the same grid."
    );
    eprintln!("mq4c_parity: PASS — 132 B stride, +4 payload, fp16 header all decode correctly");
}
