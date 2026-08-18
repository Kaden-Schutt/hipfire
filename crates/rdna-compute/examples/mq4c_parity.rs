//! Correctness gate for MQ4C / v1.5 (qt=45, 136 B/group) PAD decode.
//!
//! v1 and mq4c encode the SAME per-256 affine grid; mq4c just narrows the header to
//! one fp16 pair. PAD layout (v1's exact geometry with header re-encoded):
//!   Per weight tensor with m rows, gpr=K/256, n=m*gpr:
//!     per group, 136 B stride: `[0..4)` fp16 header, `[4..8)` zero padding, `[8..136)` 128 B nibbles.
//!   total n*136 = m*gpr*136, the same size as v1 (`MQ4C_GROUP_BYTES` = 136). The 4 padding
//!   bytes are the deliberate price of putting the payload at +8 where v1 has it, because a
//!   132 B stride left the payload 4-byte aligned half the time.
//!   header dword low 16 bits fp16 scale, high 16 bits fp16 zero, payload at +8.
//!
//! A repack keeps every nibble (0 flips in 95M groups), so the two containers must
//! decode to the same numbers within fp16 rounding.  This gate packs one weight set
//! both ways, runs each through its own GEMV, and requires agreement.  Failure modes
//! it catches (otherwise silent, full-speed noise):
//!   * wrong stride (136 vs 132) — misaligned groups after first
//!   * payload offset error (+4 vs +8) — payload bytes as fp16 gives ~1e-14
//!   * f32 header read where fp16 pair lives — the qt=44 bug that cost 4 cycles
//!
//! The fixture is intentionally DISCRIMINATING: per-group scale and zero vary widely
//! so a header read from the wrong group or from the wrong offset produces large error
//! and the gate FAILS.  An assertion checks that before checking the GPU result.
//!
//! Run: `cargo run --release -p rdna-compute --example mq4c_parity`

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};

const GROUP: usize = 256;
const V1_BYTES: usize = 136;
const MQ4C_BYTES: usize = 136;

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
/// the repack does.  v1 is interleaved 136 B/group, mq4c is PAD 136 B/group
/// (header at +0, zero padding at +4, payload at +8).
fn pack_both(w: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
    let gpr = k / GROUP;
    let n = m * gpr;
    let mut b1 = vec![0u8; n * V1_BYTES];
    let mut bc = vec![0u8; n * MQ4C_BYTES];
    let mut deq_c = vec![0.0f64; m * k];
    assert_eq!(bc.len(), n * 136, "pad size");
    assert_eq!(b1.len(), n * 136);
    for r in 0..m {
        for g in 0..gpr {
            let idx = r * gpr + g;
            let src = r * k + g * GROUP;
            let d1 = idx * V1_BYTES;
            let base = idx * MQ4C_BYTES;
            let s = &w[src..src + GROUP];
            let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            b1[d1..d1 + 4].copy_from_slice(&step.to_le_bytes());
            b1[d1 + 4..d1 + 8].copy_from_slice(&lo.to_le_bytes());
            let sb = f16_bits(step);
            let zb = f16_bits(lo);
            // pad layout: header at base+0, zeros at base+4, nibbles at base+8
            bc[base..base + 2].copy_from_slice(&sb.to_le_bytes());
            bc[base + 2..base + 4].copy_from_slice(&zb.to_le_bytes());
            // bc[base+4..base+8] stays zero (vec zeroed)
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
                bc[base + 8 + i] = byte; // same nibbles, payload at +8
            }
        }
    }
    // Both containers now same size: pad deliberately gives up size win
    assert_eq!(b1.len(), n * V1_BYTES);
    assert_eq!(bc.len(), n * MQ4C_BYTES);
    assert_eq!(b1.len(), bc.len(), "v1 and mq4c pad must be same size");
    (b1, bc, deq_c)
}

/// Host dequant reading the PAD mq4c blob — header at group base, payload at +8.
fn host_dequant_pad(blob: &[u8], m: usize, k: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut out = vec![0.0f64; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let idx = r * gpr + g;
            let base = idx * MQ4C_BYTES;
            let hdr = u32::from_le_bytes([
                blob[base],
                blob[base + 1],
                blob[base + 2],
                blob[base + 3],
            ]);
            let sc = f16_to_f32((hdr & 0xFFFF) as u16);
            let zp = f16_to_f32(((hdr >> 16) & 0xFFFF) as u16);
            let pay_base = base + 8;
            let dst_base = r * k + g * GROUP;
            for i in 0..GROUP {
                let byte = blob[pay_base + i / 2];
                let q = if i % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
                out[dst_base + i] = (zp + sc * q as f32) as f64;
            }
        }
    }
    out
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
    let gpr = k / GROUP;
    // Discriminating fixture: per-group range and offset vary widely so that
    // a header read from the wrong group or wrong offset produces large error.
    // Old Gaussian (sigma 0.011) gave all groups ~same lo/hi/step and could not
    // catch an off-by-one header stride bug.
    let w: Vec<f32> = (0..m * k)
        .map(|i| {
            let r = i / k;
            let c = i % k;
            let g = c / GROUP;
            let idx = r * gpr + g;
            // Per-group distinct range: 0.015 .. 0.365, offset -2 .. 2
            let group_range = 0.015 + prng(idx, 0xA002) * 0.35;
            let group_lo = prng(idx, 0xA001) * 4.0 - 2.0 - group_range * 0.0;
            // Uniform inside the group's interval
            let u = prng(i, 0x1234_5678);
            group_lo + u * group_range
        })
        .collect();

    let (b1, bc, deq_c) = pack_both(&w, m, k);
    assert_eq!(bc.len(), b1.len(), "container size must be equal (pad 136 vs v1 136)");
    // Verify pad host dequant matches the pack-time dequant (proves host reads pad correctly)
    let deq_blob = host_dequant_pad(&bc, m, k);
    let max_blob_err = deq_c
        .iter()
        .zip(deq_blob.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_blob_err < 1e-12,
        "host pad dequant diverged from pack-time dequant: {max_blob_err:.3e}"
    );

    // --- Discriminating-fixture assertions (must hold BEFORE checking GPU) ---
    let n = m * gpr;
    let mut scales = Vec::with_capacity(n);
    let mut zeros = Vec::with_capacity(n);
    for idx in 0..n {
        let base = idx * MQ4C_BYTES;
        let hdr = u32::from_le_bytes([
            bc[base],
            bc[base + 1],
            bc[base + 2],
            bc[base + 3],
        ]);
        scales.push(f16_to_f32((hdr & 0xFFFF) as u16));
        zeros.push(f16_to_f32(((hdr >> 16) & 0xFFFF) as u16));
        // padding must be zero
        assert_eq!(
            &bc[base + 4..base + 8],
            &[0, 0, 0, 0],
            "group {idx} padding not zero"
        );
    }
    let sc_min = scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let sc_max = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let zp_min = zeros.iter().cloned().fold(f32::INFINITY, f32::min);
    let zp_max = zeros.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sc_range = sc_max - sc_min;
    let zp_range = zp_max - zp_min;
    eprintln!(
        "fixture header range: scale [{sc_min:.4e} .. {sc_max:.4e}] range {sc_range:.4e}, zero [{zp_min:.4e} .. {zp_max:.4e}] range {zp_range:.4e}"
    );
    assert!(
        sc_range > 0.018,
        "fixture NOT discriminating: scales too similar (range {sc_range:.3e}); a wrong-offset or off-by-one header read would not be caught. Need per-group varied weights."
    );
    assert!(
        zp_range > 0.8,
        "fixture NOT discriminating: zeros too similar (range {zp_range:.3e}); need per-group varied offset."
    );
    // Payload-as-header test: first payload bytes reinterpreted as header must differ sharply
    // In pad layout payload starts at base+8
    let fake_hdr = u32::from_le_bytes([bc[8], bc[9], bc[10], bc[11]]);
    let fake_sc = f16_to_f32((fake_hdr & 0xFFFF) as u16);
    let fake_zp = f16_to_f32(((fake_hdr >> 16) & 0xFFFF) as u16);
    eprintln!(
        "payload bytes as header would decode scale {fake_sc:.4e} zero {fake_zp:.4e} vs true scale {sc_min:.4e}..{sc_max:.4e}"
    );
    // The fake decode is typically garbage (nibbles 0..15 map to tiny fp16 values); require it differs
    let payload_mimics_header = (fake_sc - scales[0]).abs() < 1e-4 && (fake_zp - zeros[0]).abs() < 0.02;
    assert!(
        !payload_mimics_header,
        "fixture NOT discriminating: payload bytes mimic header values — wrong-offset read would not be caught"
    );
    // Adjacent-header distinctness: off-by-one stride must be detectable
    if n > 1 {
        let hdr1 = {
            let off = MQ4C_BYTES;
            u32::from_le_bytes([bc[off], bc[off + 1], bc[off + 2], bc[off + 3]])
        };
        let sc1 = f16_to_f32((hdr1 & 0xFFFF) as u16);
        let sc0 = scales[0];
        assert!(
            (sc1 - sc0).abs() > 5e-5,
            "fixture NOT discriminating: adjacent headers too similar to catch off-by-one"
        );
    }
    eprintln!("fixture discriminating: OK — wrong-offset / wrong-stride would be caught");

    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xC0FF_EE) * 2.0 - 1.0).collect();
    // Host reference for the mq4c container, from its own exact dequant (now verified pad).
    let mut want = vec![0.0f64; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for c in 0..k {
            acc += deq_blob[r * k + c] * x[c] as f64;
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
    eprintln!("mq4c_parity: PASS — pad 136, header at +0, payload at +8, fp16 pair all decode correctly");
}
