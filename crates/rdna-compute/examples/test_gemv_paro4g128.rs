//! GPU vs CPU correctness test for `gemv_paro4g128`.
//!
//! Builds a deterministic native Paro/AWQ payload:
//!   qweight:int32[K,M/8], qzeros:int32[K/128,M/8], scales:f16[K/128,M],
//!   pairs:int16[8,K], theta:f16[8,K/2], channel_scales:f16[K].
//! The CPU reference mirrors ParoQuant's PyTorch runtime semantics:
//! activation scaling + KROT pairwise rotations, then AWQ W4 dequant GEMV.

use rdna_compute::{DType, Gpu};

const GROUP_SIZE: usize = 128;
const KROT: usize = 8;
const PACK: usize = 8;
const AWQ_INV_REORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

fn push_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_i16(out: &mut Vec<u8>, v: i16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
}

fn push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
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
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
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
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn read_i32(payload: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(payload[off..off + 4].try_into().unwrap())
}

fn read_i16(payload: &[u8], off: usize) -> i16 {
    i16::from_le_bytes(payload[off..off + 2].try_into().unwrap())
}

fn read_f16(payload: &[u8], off: usize) -> f32 {
    f16_bits_to_f32(u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()))
}

fn offsets(m: usize, k: usize) -> (usize, usize, usize, usize, usize, usize) {
    let groups = k / GROUP_SIZE;
    let m_pack = m / PACK;
    let qweight_off = 0;
    let qzeros_off = qweight_off + k * m_pack * 4;
    let scales_off = qzeros_off + groups * m_pack * 4;
    let pairs_off = scales_off + groups * m * 2;
    let theta_off = pairs_off + KROT * k * 2;
    let channel_scales_off = theta_off + KROT * (k / 2) * 2;
    (qweight_off, qzeros_off, scales_off, pairs_off, theta_off, channel_scales_off)
}

fn rotate_group(xg: &mut [f32], payload: &[u8], group: usize, k: usize, pairs_off: usize, theta_off: usize) {
    for r in 0..KROT {
        let pair_base = pairs_off + (r * k + group * GROUP_SIZE) * 2;
        let theta_base = theta_off + (r * (k / 2) + group * (GROUP_SIZE / 2)) * 2;
        for p in 0..(GROUP_SIZE / 2) {
            let i = read_i16(payload, pair_base + 2 * (2 * p)) as usize;
            let j = read_i16(payload, pair_base + 2 * (2 * p + 1)) as usize;
            let (s, c) = read_f16(payload, theta_base + p * 2).sin_cos();
            let xi = xg[i];
            let xj = xg[j];
            xg[i] = c.mul_add(xi, s * xj);
            xg[j] = c.mul_add(xj, -s * xi);
        }
    }
}

fn pack_awq_lanes(lanes: [u8; PACK]) -> i32 {
    let mut word = 0i32;
    for logical in 0..PACK {
        let slot = AWQ_INV_REORDER[logical];
        word |= ((lanes[logical] & 0xF) as i32) << (4 * slot);
    }
    word
}

fn build_payload_with_seed(m: usize, k: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % GROUP_SIZE, 0);
    assert_eq!(m % PACK, 0);
    let groups = k / GROUP_SIZE;
    let m_pack = m / PACK;
    let mut payload = Vec::new();

    // qweight [K, M/8]
    for kk in 0..k {
        for mc in 0..m_pack {
            let mut lanes = [0u8; PACK];
            for (lane, v) in lanes.iter_mut().enumerate() {
                *v = ((kk * 3 + mc * 5 + lane * 7 + seed * 11 + 1) & 0xF) as u8;
            }
            push_i32(&mut payload, pack_awq_lanes(lanes));
        }
    }

    // qzeros [groups, M/8]
    for g in 0..groups {
        for mc in 0..m_pack {
            let mut lanes = [0u8; PACK];
            for (lane, v) in lanes.iter_mut().enumerate() {
                *v = ((g * 2 + mc * 3 + lane + seed * 5) & 0xF) as u8;
            }
            push_i32(&mut payload, pack_awq_lanes(lanes));
        }
    }

    // scales [groups, M]
    for g in 0..groups {
        for row in 0..m {
            push_f16(
                &mut payload,
                0.03125 + (row % 7) as f32 * 0.0025 + g as f32 * 0.001 + seed as f32 * 0.0003,
            );
        }
    }

    // pairs [8, K], local indices per G128 group.
    for r in 0..KROT {
        for _g in 0..groups {
            for p in 0..(GROUP_SIZE / 2) {
                let a = (2 * p + r + seed) % GROUP_SIZE;
                let b = (2 * p + 1 + r + seed) % GROUP_SIZE;
                push_i16(&mut payload, a as i16);
                push_i16(&mut payload, b as i16);
            }
        }
    }

    // theta [8, K/2]
    for i in 0..KROT * (k / 2) {
        push_f16(&mut payload, (((i + seed * 3) % 17) as f32 - 8.0) * 0.011);
    }

    // channel_scales [K]
    for i in 0..k {
        push_f16(&mut payload, 0.75 + (i % 23) as f32 * 0.01 + seed as f32 * 0.001);
    }

    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 29) as f32 - 14.0) * 0.027).collect();
    (payload, x)
}

fn build_payload(m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    build_payload_with_seed(m, k, 0)
}

fn retile_qweight_payload(payload: &[u8], m: usize, k: usize) -> Vec<u8> {
    let m_pack = m / PACK;
    let (qweight_off, qzeros_off, _scales_off, _pairs_off, theta_off, channel_scales_off) = offsets(m, k);
    let mut out = Vec::with_capacity(payload.len() + KROT * (k / 2) * 6);
    for mc in 0..m_pack {
        for kk in 0..k {
            let src = qweight_off + (kk * m_pack + mc) * 4;
            out.extend_from_slice(&payload[src..src + 4]);
        }
    }
    out.extend_from_slice(&payload[qzeros_off..theta_off]);
    for i in 0..(KROT * (k / 2)) {
        let theta = read_f16(payload, theta_off + i * 2);
        push_f32(&mut out, theta.sin());
        push_f32(&mut out, theta.cos());
    }
    out.extend_from_slice(&payload[channel_scales_off..]);
    out
}

fn cpu_reference(payload: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let groups = k / GROUP_SIZE;
    let m_pack = m / PACK;
    let (qweight_off, qzeros_off, scales_off, pairs_off, theta_off, channel_scales_off) = offsets(m, k);
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let m_col = row / PACK;
        let m_slot = AWQ_INV_REORDER[row & 7];
        let mut acc = 0.0f32;
        for g in 0..groups {
            let base = g * GROUP_SIZE;
            let mut xg = [0.0f32; GROUP_SIZE];
            for i in 0..GROUP_SIZE {
                xg[i] = x[base + i] * read_f16(payload, channel_scales_off + (base + i) * 2);
            }
            rotate_group(&mut xg, payload, g, k, pairs_off, theta_off);

            let zp_word = read_i32(payload, qzeros_off + (g * m_pack + m_col) * 4);
            let zp = (zp_word >> (4 * m_slot)) & 0xF;
            let scale = read_f16(payload, scales_off + (g * m + row) * 2);
            let bias = -scale * zp as f32;
            for (i, &xv) in xg.iter().enumerate() {
                let qw_word = read_i32(payload, qweight_off + ((base + i) * m_pack + m_col) * 4);
                let q = (qw_word >> (4 * m_slot)) & 0xF;
                acc += (scale * q as f32 + bias) * xv;
            }
        }
        y[row] = acc;
    }
    y
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let m = 40usize;
    let k = 256usize;
    let (payload, x) = build_payload(m, k);
    let y_ref = cpu_reference(&payload, &x, m, k);

    let d_a = gpu.upload_raw(&payload, &[payload.len()]).unwrap();
    let payload_tiled = retile_qweight_payload(&payload, m, k);
    let d_a_tiled = gpu.upload_raw(&payload_tiled, &[payload_tiled.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_paro4g128(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..m {
        let abs = (y_gpu[i] - y_ref[i]).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / y_ref[i].abs().max(1.0));
    }

    println!("max_abs={:.6e} max_rel={:.6e}", max_abs, max_rel);
    if max_abs > 5e-5 || max_rel > 5e-5 {
        std::process::exit(1);
    }

    let d_x_rot = gpu.zeros(&[k], DType::F32).unwrap();
    let d_y_pre = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.paro4g128_rotate(&d_a, &d_x, &d_x_rot, m, k).unwrap();
    gpu.gemv_paro4g128_prerotated(&d_a, &d_x_rot, &d_y_pre, m, k).unwrap();
    let y_pre_gpu = gpu.download_f32(&d_y_pre).unwrap();
    let mut max_abs_pre = 0.0f32;
    let mut max_rel_pre = 0.0f32;
    for i in 0..m {
        let abs = (y_pre_gpu[i] - y_ref[i]).abs();
        max_abs_pre = max_abs_pre.max(abs);
        max_rel_pre = max_rel_pre.max(abs / y_ref[i].abs().max(1.0));
    }
    println!("prerotated max_abs={:.6e} max_rel={:.6e}", max_abs_pre, max_rel_pre);
    if max_abs_pre > 5e-5 || max_rel_pre > 5e-5 {
        std::process::exit(1);
    }

    let d_x_rot_tiled = gpu.zeros(&[k], DType::F32).unwrap();
    gpu.paro4g128t_rotate(&d_a_tiled, &d_x, &d_x_rot_tiled, m, k).unwrap();
    let d_y_tiled = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_paro4g128t_prerotated(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled, m, k).unwrap();
    let y_tiled_gpu = gpu.download_f32(&d_y_tiled).unwrap();
    let mut max_abs_tiled = 0.0f32;
    let mut max_rel_tiled = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_gpu[i] - y_ref[i]).abs();
        max_abs_tiled = max_abs_tiled.max(abs);
        max_rel_tiled = max_rel_tiled.max(abs / y_ref[i].abs().max(1.0));
    }
    println!("tiled prerotated max_abs={:.6e} max_rel={:.6e}", max_abs_tiled, max_rel_tiled);
    if max_abs_tiled > 5e-5 || max_rel_tiled > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_pack4 = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_paro4g128t_prerotated_pack4(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_pack4, m, k).unwrap();
    let y_tiled_pack4_gpu = gpu.download_f32(&d_y_tiled_pack4).unwrap();
    let mut max_abs_tiled_pack4 = 0.0f32;
    let mut max_rel_tiled_pack4 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_pack4_gpu[i] - y_ref[i]).abs();
        max_abs_tiled_pack4 = max_abs_tiled_pack4.max(abs);
        max_rel_tiled_pack4 = max_rel_tiled_pack4.max(abs / y_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack4 prerotated max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_pack4,
        max_rel_tiled_pack4
    );
    if max_abs_tiled_pack4 > 5e-5 || max_rel_tiled_pack4 > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_pack2 = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_paro4g128t_prerotated_pack2(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_pack2, m, k).unwrap();
    let y_tiled_pack2_gpu = gpu.download_f32(&d_y_tiled_pack2).unwrap();
    let mut max_abs_tiled_pack2 = 0.0f32;
    let mut max_rel_tiled_pack2 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_pack2_gpu[i] - y_ref[i]).abs();
        max_abs_tiled_pack2 = max_abs_tiled_pack2.max(abs);
        max_rel_tiled_pack2 = max_rel_tiled_pack2.max(abs / y_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack2 prerotated max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_pack2,
        max_rel_tiled_pack2
    );
    if max_abs_tiled_pack2 > 5e-5 || max_rel_tiled_pack2 > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_pack1 = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_paro4g128t_prerotated_pack1(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_pack1, m, k).unwrap();
    let y_tiled_pack1_gpu = gpu.download_f32(&d_y_tiled_pack1).unwrap();
    let mut max_abs_tiled_pack1 = 0.0f32;
    let mut max_rel_tiled_pack1 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_pack1_gpu[i] - y_ref[i]).abs();
        max_abs_tiled_pack1 = max_abs_tiled_pack1.max(abs);
        max_rel_tiled_pack1 = max_rel_tiled_pack1.max(abs / y_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack1 prerotated max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_pack1,
        max_rel_tiled_pack1
    );

    let (payload_up, _) = build_payload_with_seed(m, k, 7);
    let y_up_ref = cpu_reference(&payload_up, &x, m, k);
    let payload_up_tiled = retile_qweight_payload(&payload_up, m, k);
    let d_a_up_tiled = gpu.upload_raw(&payload_up_tiled, &[payload_up_tiled.len()]).unwrap();
    let d_y_gate_fused = gpu.zeros(&[m], DType::F32).unwrap();
    let d_y_up_fused = gpu.zeros(&[m], DType::F32).unwrap();
    let d_x_rot_gate_fused = gpu.zeros(&[k], DType::F32).unwrap();
    gpu.fused_gate_up_paro4g128t(
        &d_a_tiled,
        &d_a_up_tiled,
        &d_x,
        &d_y_gate_fused,
        &d_y_up_fused,
        &d_x_rot_gate_fused,
        m,
        k,
    )
    .unwrap();
    let y_gate_fused_gpu = gpu.download_f32(&d_y_gate_fused).unwrap();
    let y_up_fused_gpu = gpu.download_f32(&d_y_up_fused).unwrap();
    let mut max_abs_fused_gate = 0.0f32;
    let mut max_rel_fused_gate = 0.0f32;
    let mut max_abs_fused_up = 0.0f32;
    let mut max_rel_fused_up = 0.0f32;
    for i in 0..m {
        let abs_gate = (y_gate_fused_gpu[i] - y_ref[i]).abs();
        max_abs_fused_gate = max_abs_fused_gate.max(abs_gate);
        max_rel_fused_gate = max_rel_fused_gate.max(abs_gate / y_ref[i].abs().max(1.0));
        let abs_up = (y_up_fused_gpu[i] - y_up_ref[i]).abs();
        max_abs_fused_up = max_abs_fused_up.max(abs_up);
        max_rel_fused_up = max_rel_fused_up.max(abs_up / y_up_ref[i].abs().max(1.0));
    }
    println!(
        "fused gate/up max_abs_gate={:.6e} max_rel_gate={:.6e} max_abs_up={:.6e} max_rel_up={:.6e}",
        max_abs_fused_gate,
        max_rel_fused_gate,
        max_abs_fused_up,
        max_rel_fused_up
    );
    if max_abs_fused_gate > 5e-5
        || max_rel_fused_gate > 5e-5
        || max_abs_fused_up > 5e-5
        || max_rel_fused_up > 5e-5
    {
        std::process::exit(1);
    }

    let y_seed: Vec<f32> = (0..m).map(|i| 0.125 + i as f32 * 0.003).collect();
    let y_res_ref: Vec<f32> = y_seed.iter().zip(y_ref.iter()).map(|(a, b)| a + b).collect();
    let d_y_res = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128_residual(&d_a, &d_x, &d_y_res, m, k).unwrap();
    let y_res_gpu = gpu.download_f32(&d_y_res).unwrap();
    let mut max_abs_res = 0.0f32;
    let mut max_rel_res = 0.0f32;
    for i in 0..m {
        let abs = (y_res_gpu[i] - y_res_ref[i]).abs();
        max_abs_res = max_abs_res.max(abs);
        max_rel_res = max_rel_res.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!("residual max_abs={:.6e} max_rel={:.6e}", max_abs_res, max_rel_res);
    if max_abs_res > 5e-5 || max_rel_res > 5e-5 {
        std::process::exit(1);
    }

    let d_y_pre_res = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128_prerotated_residual(&d_a, &d_x_rot, &d_y_pre_res, m, k).unwrap();
    let y_pre_res_gpu = gpu.download_f32(&d_y_pre_res).unwrap();
    let mut max_abs_pre_res = 0.0f32;
    let mut max_rel_pre_res = 0.0f32;
    for i in 0..m {
        let abs = (y_pre_res_gpu[i] - y_res_ref[i]).abs();
        max_abs_pre_res = max_abs_pre_res.max(abs);
        max_rel_pre_res = max_rel_pre_res.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!(
        "prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_pre_res,
        max_rel_pre_res
    );
    if max_abs_pre_res > 5e-5 || max_rel_pre_res > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_res = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_res, m, k).unwrap();
    let y_tiled_res_gpu = gpu.download_f32(&d_y_tiled_res).unwrap();
    let mut max_abs_tiled_res = 0.0f32;
    let mut max_rel_tiled_res = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_res_gpu[i] - y_res_ref[i]).abs();
        max_abs_tiled_res = max_abs_tiled_res.max(abs);
        max_rel_tiled_res = max_rel_tiled_res.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!(
        "tiled prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_res,
        max_rel_tiled_res
    );
    if max_abs_tiled_res > 5e-5 || max_rel_tiled_res > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_res_pack4 = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual_pack4(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_res_pack4, m, k).unwrap();
    let y_tiled_res_pack4_gpu = gpu.download_f32(&d_y_tiled_res_pack4).unwrap();
    let mut max_abs_tiled_res_pack4 = 0.0f32;
    let mut max_rel_tiled_res_pack4 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_res_pack4_gpu[i] - y_res_ref[i]).abs();
        max_abs_tiled_res_pack4 = max_abs_tiled_res_pack4.max(abs);
        max_rel_tiled_res_pack4 = max_rel_tiled_res_pack4.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack4 prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_res_pack4,
        max_rel_tiled_res_pack4
    );
    if max_abs_tiled_res_pack4 > 5e-5 || max_rel_tiled_res_pack4 > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_res_pack2 = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual_pack2(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_res_pack2, m, k).unwrap();
    let y_tiled_res_pack2_gpu = gpu.download_f32(&d_y_tiled_res_pack2).unwrap();
    let mut max_abs_tiled_res_pack2 = 0.0f32;
    let mut max_rel_tiled_res_pack2 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_res_pack2_gpu[i] - y_res_ref[i]).abs();
        max_abs_tiled_res_pack2 = max_abs_tiled_res_pack2.max(abs);
        max_rel_tiled_res_pack2 = max_rel_tiled_res_pack2.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack2 prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_res_pack2,
        max_rel_tiled_res_pack2
    );
    if max_abs_tiled_res_pack2 > 5e-5 || max_rel_tiled_res_pack2 > 5e-5 {
        std::process::exit(1);
    }

    let d_y_tiled_res_pack1 = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual_pack1(&d_a_tiled, &d_x_rot_tiled, &d_y_tiled_res_pack1, m, k).unwrap();
    let y_tiled_res_pack1_gpu = gpu.download_f32(&d_y_tiled_res_pack1).unwrap();
    let mut max_abs_tiled_res_pack1 = 0.0f32;
    let mut max_rel_tiled_res_pack1 = 0.0f32;
    for i in 0..m {
        let abs = (y_tiled_res_pack1_gpu[i] - y_res_ref[i]).abs();
        max_abs_tiled_res_pack1 = max_abs_tiled_res_pack1.max(abs);
        max_rel_tiled_res_pack1 = max_rel_tiled_res_pack1.max(abs / y_res_ref[i].abs().max(1.0));
    }
    println!(
        "tiled pack1 prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_tiled_res_pack1,
        max_rel_tiled_res_pack1
    );

    let gate: Vec<f32> = (0..k).map(|i| ((i as i32 % 31) as f32 - 15.0) * 0.019).collect();
    let up: Vec<f32> = (0..k).map(|i| ((i as i32 % 37) as f32 - 18.0) * 0.017).collect();
    let hidden = silu_mul(&gate, &up);
    let y_swiglu_ref_base = cpu_reference(&payload, &hidden, m, k);
    let y_swiglu_ref: Vec<f32> = y_seed
        .iter()
        .zip(y_swiglu_ref_base.iter())
        .map(|(a, b)| a + b)
        .collect();
    let d_gate = gpu.upload_f32(&gate, &[k]).unwrap();
    let d_up = gpu.upload_f32(&up, &[k]).unwrap();
    let d_x_rot_swiglu = gpu.zeros(&[k], DType::F32).unwrap();
    let d_y_swiglu_pre = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.paro4g128_swiglu_rotate(&d_a, &d_gate, &d_up, &d_x_rot_swiglu, m, k).unwrap();
    gpu.gemv_paro4g128_prerotated_residual(&d_a, &d_x_rot_swiglu, &d_y_swiglu_pre, m, k).unwrap();
    let y_swiglu_pre_gpu = gpu.download_f32(&d_y_swiglu_pre).unwrap();
    let mut max_abs_swiglu_pre = 0.0f32;
    let mut max_rel_swiglu_pre = 0.0f32;
    for i in 0..m {
        let abs = (y_swiglu_pre_gpu[i] - y_swiglu_ref[i]).abs();
        max_abs_swiglu_pre = max_abs_swiglu_pre.max(abs);
        max_rel_swiglu_pre = max_rel_swiglu_pre.max(abs / y_swiglu_ref[i].abs().max(1.0));
    }
    println!(
        "swiglu prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_swiglu_pre,
        max_rel_swiglu_pre
    );
    if max_abs_swiglu_pre > 5e-5 || max_rel_swiglu_pre > 5e-5 {
        std::process::exit(1);
    }

    let d_x_rot_swiglu_tiled = gpu.zeros(&[k], DType::F32).unwrap();
    gpu.paro4g128t_swiglu_rotate(&d_a_tiled, &d_gate, &d_up, &d_x_rot_swiglu_tiled, m, k).unwrap();
    let d_y_swiglu_tiled = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual(&d_a_tiled, &d_x_rot_swiglu_tiled, &d_y_swiglu_tiled, m, k).unwrap();
    let y_swiglu_tiled_gpu = gpu.download_f32(&d_y_swiglu_tiled).unwrap();
    let mut max_abs_swiglu_tiled = 0.0f32;
    let mut max_rel_swiglu_tiled = 0.0f32;
    for i in 0..m {
        let abs = (y_swiglu_tiled_gpu[i] - y_swiglu_ref[i]).abs();
        max_abs_swiglu_tiled = max_abs_swiglu_tiled.max(abs);
        max_rel_swiglu_tiled = max_rel_swiglu_tiled.max(abs / y_swiglu_ref[i].abs().max(1.0));
    }
    println!(
        "swiglu tiled prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_swiglu_tiled,
        max_rel_swiglu_tiled
    );
    if max_abs_swiglu_tiled > 5e-5 || max_rel_swiglu_tiled > 5e-5 {
        std::process::exit(1);
    }

    let d_y_swiglu_tiled_pack4 = gpu.upload_f32(&y_seed, &[m]).unwrap();
    gpu.gemv_paro4g128t_prerotated_residual_pack4(&d_a_tiled, &d_x_rot_swiglu_tiled, &d_y_swiglu_tiled_pack4, m, k).unwrap();
    let y_swiglu_tiled_pack4_gpu = gpu.download_f32(&d_y_swiglu_tiled_pack4).unwrap();
    let mut max_abs_swiglu_tiled_pack4 = 0.0f32;
    let mut max_rel_swiglu_tiled_pack4 = 0.0f32;
    for i in 0..m {
        let abs = (y_swiglu_tiled_pack4_gpu[i] - y_swiglu_ref[i]).abs();
        max_abs_swiglu_tiled_pack4 = max_abs_swiglu_tiled_pack4.max(abs);
        max_rel_swiglu_tiled_pack4 = max_rel_swiglu_tiled_pack4.max(abs / y_swiglu_ref[i].abs().max(1.0));
    }
    println!(
        "swiglu tiled pack4 prerotated residual max_abs={:.6e} max_rel={:.6e}",
        max_abs_swiglu_tiled_pack4,
        max_rel_swiglu_tiled_pack4
    );
    if max_abs_swiglu_tiled_pack4 > 5e-5 || max_rel_swiglu_tiled_pack4 > 5e-5 {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
