// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal roundtrip test for Q8_0 KV cache quantization.
fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let n_kv_heads: usize = 1;
    let head_dim: usize = 32;
    let max_seq: usize = 4;

    // Known input: [0.1, 0.2, ..., 3.2]
    let kv_data: Vec<f32> = (0..head_dim).map(|i| 0.1 * (i + 1) as f32).collect();
    let d_src = gpu.upload_f32(&kv_data, &[head_dim]).unwrap();

    // Q8_0 cache: 1 block * 34 bytes per position
    let total_blocks = n_kv_heads * (head_dim / 32);
    let cache_bytes = max_seq * total_blocks * 34;
    let cache_elems = (cache_bytes + 3) / 4;
    let d_cache = gpu.zeros(&[cache_elems], rdna_compute::DType::F32).unwrap();

    // Write at pos=0
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_cache, &d_src, &pos_buf, n_kv_heads, head_dim)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Read back raw bytes
    let mut raw = vec![0u8; 34];
    gpu.hip.memcpy_dtoh(&mut raw, &d_cache.buf).unwrap();

    // Parse f16 scale
    let scale_bits = u16::from_le_bytes([raw[0], raw[1]]);
    let scale = f16_to_f32(scale_bits);

    eprintln!("Input:  {:?}", &kv_data[..8]);
    eprintln!("scale_f16=0x{:04x} scale_f32={:.6}", scale_bits, scale);
    eprintln!(
        "Expected amax={:.3}, expected scale≈{:.6}",
        3.2,
        3.2 / 127.0
    );

    let mut max_err = 0.0f32;
    for i in 0..head_dim {
        let q = raw[2 + i] as i8;
        let dequant = scale * q as f32;
        let err = (kv_data[i] - dequant).abs();
        max_err = max_err.max(err);
        if i < 8 || i >= head_dim - 2 {
            eprintln!(
                "  [{i:>2}] input={:.3} q={:>4} dequant={:.4} err={:.4}",
                kv_data[i], q, dequant, err
            );
        }
    }
    eprintln!("Max roundtrip error: {:.6}", max_err);

    if max_err < 0.05 {
        eprintln!("PASS: single block roundtrip correct");
    } else {
        eprintln!("FAIL: single block roundtrip error too large");
        std::process::exit(1);
    }

    // Test 2: Multi-head, multi-position
    eprintln!("\n=== Multi-head multi-position test ===");
    let n_kv_heads2: usize = 2;
    let head_dim2: usize = 32; // 1 block per head
    let kv_dim2 = n_kv_heads2 * head_dim2;
    let max_seq2: usize = 4;
    let total_blocks2 = n_kv_heads2 * (head_dim2 / 32); // 2
    let cache_bytes2 = max_seq2 * total_blocks2 * 34;
    let cache_elems2 = (cache_bytes2 + 3) / 4;
    let d_cache2 = gpu
        .zeros(&[cache_elems2], rdna_compute::DType::F32)
        .unwrap();

    // Write pos=0: head0=[1.0]*32, head1=[2.0]*32
    let kv0: Vec<f32> = (0..kv_dim2)
        .map(|i| if i < head_dim2 { 1.0 } else { 2.0 })
        .collect();
    let d_src0 = gpu.upload_f32(&kv0, &[kv_dim2]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_cache2, &d_src0, &pos_buf, n_kv_heads2, head_dim2)
        .unwrap();

    // Write pos=1: head0=[3.0]*32, head1=[4.0]*32
    let kv1: Vec<f32> = (0..kv_dim2)
        .map(|i| if i < head_dim2 { 3.0 } else { 4.0 })
        .collect();
    let d_src1 = gpu.upload_f32(&kv1, &[kv_dim2]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_cache2, &d_src1, &pos_buf, n_kv_heads2, head_dim2)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Read back ALL cache data
    let mut raw2 = vec![0u8; cache_bytes2];
    gpu.hip.memcpy_dtoh(&mut raw2, &d_cache2.buf).unwrap();

    // Verify: pos=0 head=0 should have scale≈1.0/127, all qvals≈127
    // pos=0 head=1 should have scale≈2.0/127, all qvals≈127
    // pos=1 head=0 should have scale≈3.0/127, all qvals≈127
    // pos=1 head=1 should have scale≈4.0/127, all qvals≈127
    let stride = total_blocks2 * 34; // 68 bytes per position
    for pos in 0..2 {
        for h in 0..n_kv_heads2 {
            let off = pos * stride + h * 34;
            let s_bits = u16::from_le_bytes([raw2[off], raw2[off + 1]]);
            let s = f16_to_f32(s_bits);
            let q0 = raw2[off + 2] as i8;
            let expected_val = if h == 0 {
                (pos * 2 + 1) as f32
            } else {
                (pos * 2 + 2) as f32
            };
            let dequant0 = s * q0 as f32;
            eprintln!("  pos={pos} head={h}: scale={s:.4} q[0]={q0} dequant={dequant0:.4} expected={expected_val:.1}");
        }
    }
    eprintln!("Multi-head test complete");

    // Test 3: Attention with Q8_0 KV cache
    eprintln!("\n=== Attention Q8_0 KV test ===");
    // 1 head, head_dim=32, 2 positions in KV cache
    // Q = [1.0]*32
    // K cache: pos0=[1.0]*32, pos1=[0.0]*32 (only pos0 should have high score)
    // V cache: pos0=[1.0]*32, pos1=[0.0]*32
    // Expected: attention output ≈ [1.0]*32 (all weight on pos0)
    let n_heads_a: usize = 1;
    let n_kv_a: usize = 1;
    let hd_a: usize = 32;
    let total_b = n_kv_a * (hd_a / 32); // 1

    // Allocate K and V caches (2 positions)
    let cache_b = 2 * total_b * 34;
    let cache_e = (cache_b + 3) / 4;
    let d_kcache = gpu.zeros(&[cache_e], rdna_compute::DType::F32).unwrap();
    let d_vcache = gpu.zeros(&[cache_e], rdna_compute::DType::F32).unwrap();

    // Write K pos0 = [1.0]*32, V pos0 = [1.0]*32
    let ones32 = vec![1.0f32; hd_a];
    let d_ones = gpu.upload_f32(&ones32, &[hd_a]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kcache, &d_ones, &pos_buf, n_kv_a, hd_a)
        .unwrap();
    gpu.kv_cache_write_q8_0(&d_vcache, &d_ones, &pos_buf, n_kv_a, hd_a)
        .unwrap();

    // Write K pos1 = [0.5]*32, V pos1 = [2.0]*32
    let half32 = vec![0.5f32; hd_a];
    let twos32 = vec![2.0f32; hd_a];
    let d_half = gpu.upload_f32(&half32, &[hd_a]).unwrap();
    let d_twos = gpu.upload_f32(&twos32, &[hd_a]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kcache, &d_half, &pos_buf, n_kv_a, hd_a)
        .unwrap();
    gpu.kv_cache_write_q8_0(&d_vcache, &d_twos, &pos_buf, n_kv_a, hd_a)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Q = [1.0]*32
    let d_q = gpu.upload_f32(&ones32, &[hd_a]).unwrap();
    let d_out = gpu.zeros(&[hd_a], rdna_compute::DType::F32).unwrap();

    // Run attention at pos=1 (seq_len=2: see both positions)
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.attention_q8_0_kv(
        &d_q, &d_kcache, &d_vcache, &d_out, &pos_buf, 2, n_heads_a, n_kv_a, hd_a, 4,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let out_vals = gpu.download_f32(&d_out).unwrap();
    eprintln!("Attention output[0..4]: {:?}", &out_vals[..4]);
    // Q·K for pos0: 1.0*1.0*32 = 32.0, scaled by 1/sqrt(32) = 32/5.66 = 5.66
    // Q·K for pos1: 1.0*0.5*32 = 16.0, scaled = 16/5.66 = 2.83
    // softmax([5.66, 2.83]) ≈ [0.944, 0.056]
    // output ≈ 0.944 * [1.0] + 0.056 * [2.0] ≈ [1.056]
    let expected = 0.944 * 1.0 + 0.056 * 2.0;
    eprintln!("Expected ≈ {expected:.3}");
    if (out_vals[0] - expected).abs() < 0.2 {
        eprintln!("PASS: Q8_0 attention output correct");
    } else {
        eprintln!(
            "FAIL: Q8_0 attention output wrong (got {:.4}, expected ~{:.3})",
            out_vals[0], expected
        );
    }

    // Test 4: 8B dimensions (n_heads=32, n_kv_heads=8, head_dim=128)
    eprintln!("\n=== 8B dimensions test ===");
    let n_h = 32usize;
    let n_kv = 8usize;
    let hd = 128usize;
    let kv_dim_8b = n_kv * hd;
    let tb = n_kv * (hd / 32); // 32 blocks per pos
    let cb = 4 * tb * 34;
    let ce = (cb + 3) / 4;
    let d_kc8 = gpu.zeros(&[ce], rdna_compute::DType::F32).unwrap();
    let d_vc8 = gpu.zeros(&[ce], rdna_compute::DType::F32).unwrap();

    // Write 2 positions with distinct values
    let k0: Vec<f32> = (0..kv_dim_8b).map(|i| 0.01 * ((i % 128) as f32)).collect();
    let v0: Vec<f32> = vec![1.0f32; kv_dim_8b];
    let d_k0 = gpu.upload_f32(&k0, &[kv_dim_8b]).unwrap();
    let d_v0 = gpu.upload_f32(&v0, &[kv_dim_8b]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kc8, &d_k0, &pos_buf, n_kv, hd)
        .unwrap();
    gpu.kv_cache_write_q8_0(&d_vc8, &d_v0, &pos_buf, n_kv, hd)
        .unwrap();

    let k1: Vec<f32> = (0..kv_dim_8b).map(|i| -0.01 * ((i % 128) as f32)).collect();
    let v1: Vec<f32> = vec![2.0f32; kv_dim_8b];
    let d_k1 = gpu.upload_f32(&k1, &[kv_dim_8b]).unwrap();
    let d_v1 = gpu.upload_f32(&v1, &[kv_dim_8b]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kc8, &d_k1, &pos_buf, n_kv, hd)
        .unwrap();
    gpu.kv_cache_write_q8_0(&d_vc8, &d_v1, &pos_buf, n_kv, hd)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Q: 32 heads, each [0.01]*128 (aligned with K pos0)
    let q8b: Vec<f32> = vec![0.01f32; n_h * hd];
    let d_q8 = gpu.upload_f32(&q8b, &[n_h * hd]).unwrap();
    let d_out8 = gpu.zeros(&[n_h * hd], rdna_compute::DType::F32).unwrap();

    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.attention_q8_0_kv(
        &d_q8, &d_kc8, &d_vc8, &d_out8, &pos_buf, 2, n_h, n_kv, hd, 4,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let out8 = gpu.download_f32(&d_out8).unwrap();
    // Q·K for pos0: positive dot product (aligned), Q·K for pos1: negative (anti-aligned)
    // So softmax should put most weight on pos0, output ≈ V[0] = 1.0
    eprintln!("head0 out[0..4]: {:?}", &out8[..4]);
    eprintln!("head0 out should be close to 1.0 (V pos0 dominates)");
    if out8[0] > 0.5 && out8[0] < 1.5 && !out8[0].is_nan() {
        eprintln!("PASS: 8B dimensions correct");
    } else {
        eprintln!("FAIL: 8B dimensions wrong (got {:.4})", out8[0]);
    }

    // Test 5: Flash (tile+reduce online-softmax) vs baseline single-workgroup
    // parity. The flash path (attention_flash_q8_0) is what minimax uses for
    // its native context window — its LDS is O(tile+head_dim), independent of
    // seq_len, where the baseline materializes scores[seq_len] in LDS and caps
    // at ~16K on gfx11/gfx12. Both compute the same math (different FP
    // reduction order), so we require cosine ≥ 0.9999 / small max-abs-err at
    // sizes where the baseline can still run, then prove flash runs (finite,
    // sane) at a size the baseline cannot.
    eprintln!("\n=== Flash vs baseline parity (GQA, head_dim=128) ===");
    let n_h5 = 8usize;
    let n_kv5 = 2usize;
    let hd5 = 128usize;
    let kv_dim5 = n_kv5 * hd5;
    let tb5 = n_kv5 * (hd5 / 32); // blocks per position
    let mut seed: u32 = 0x1234_5678;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((seed >> 9) as f32 / 8_388_608.0) - 1.0 // ~[-1, 1)
    };
    let to_bytes = |v: &[f32]| -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4).to_vec() }
    };

    let mut all_pass = true;
    for &seq in &[4usize, 64, 512, 2048] {
        let physical_cap = seq;
        let cb = physical_cap * tb5 * 34;
        let ce = (cb + 3) / 4;
        let d_kc = gpu.zeros(&[ce], rdna_compute::DType::F32).unwrap();
        let d_vc = gpu.zeros(&[ce], rdna_compute::DType::F32).unwrap();
        // Reuse one upload buffer per kind; overwrite per position.
        let dk = gpu.zeros(&[kv_dim5], rdna_compute::DType::F32).unwrap();
        let dv = gpu.zeros(&[kv_dim5], rdna_compute::DType::F32).unwrap();
        for p in 0..seq {
            let kbuf: Vec<f32> = (0..kv_dim5).map(|_| next()).collect();
            let vbuf: Vec<f32> = (0..kv_dim5).map(|_| next()).collect();
            gpu.hip.memcpy_htod(&dk.buf, &to_bytes(&kbuf)).unwrap();
            gpu.hip.memcpy_htod(&dv.buf, &to_bytes(&vbuf)).unwrap();
            gpu.hip
                .memcpy_htod(&pos_buf, &(p as i32).to_ne_bytes())
                .unwrap();
            gpu.kv_cache_write_q8_0(&d_kc, &dk, &pos_buf, n_kv5, hd5)
                .unwrap();
            gpu.kv_cache_write_q8_0(&d_vc, &dv, &pos_buf, n_kv5, hd5)
                .unwrap();
        }
        let qbuf: Vec<f32> = (0..n_h5 * hd5).map(|_| next()).collect();
        let dq = gpu.upload_f32(&qbuf, &[n_h5 * hd5]).unwrap();
        let out_base = gpu.zeros(&[n_h5 * hd5], rdna_compute::DType::F32).unwrap();
        let out_flash = gpu.zeros(&[n_h5 * hd5], rdna_compute::DType::F32).unwrap();
        let max_tiles = (physical_cap + 127) / 128;
        let partials = gpu
            .zeros(&[n_h5 * max_tiles * (2 + hd5)], rdna_compute::DType::F32)
            .unwrap();
        gpu.hip
            .memcpy_htod(&pos_buf, &((seq - 1) as i32).to_ne_bytes())
            .unwrap();
        gpu.attention_q8_0_kv(
            &dq,
            &d_kc,
            &d_vc,
            &out_base,
            &pos_buf,
            seq,
            n_h5,
            n_kv5,
            hd5,
            physical_cap,
        )
        .unwrap();
        gpu.attention_flash_q8_0(
            &dq,
            &d_kc,
            &d_vc,
            &out_flash,
            &pos_buf,
            seq,
            n_h5,
            n_kv5,
            hd5,
            physical_cap,
            &partials,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let a = gpu.download_f32(&out_base).unwrap();
        let b = gpu.download_f32(&out_flash).unwrap();
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut maxerr = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
            maxerr = maxerr.max((a[i] - b[i]).abs());
        }
        let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        let pass = cosine > 0.9999 && maxerr < 1e-2 && !b.iter().any(|x| x.is_nan());
        eprintln!(
            "  seq={seq:>5}: cosine={cosine:.6} max_abs_err={maxerr:.2e} -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        all_pass &= pass;
    }

    // Large-context flash-only: a size the baseline single-workgroup kernel
    // cannot serve (scores[seq]*4 + ... > 64 KB LDS). All K/V positions get
    // an identical random vector, so softmax is uniform and the output must
    // equal that V vector (mean of identical values). Proves flash runs and
    // is numerically sane far beyond the 16K LDS wall.
    eprintln!("\n=== Flash-only at >16K (baseline cannot run here) ===");
    let big_seq = 20000usize;
    let cb_big = big_seq * tb5 * 34;
    let ce_big = (cb_big + 3) / 4;
    let d_kc_big = gpu.zeros(&[ce_big], rdna_compute::DType::F32).unwrap();
    let d_vc_big = gpu.zeros(&[ce_big], rdna_compute::DType::F32).unwrap();
    let kfix: Vec<f32> = (0..kv_dim5).map(|_| next()).collect();
    let vfix: Vec<f32> = (0..kv_dim5).map(|_| next()).collect();
    let dk_big = gpu.upload_f32(&kfix, &[kv_dim5]).unwrap();
    let dv_big = gpu.upload_f32(&vfix, &[kv_dim5]).unwrap();
    for p in 0..big_seq {
        gpu.hip
            .memcpy_htod(&pos_buf, &(p as i32).to_ne_bytes())
            .unwrap();
        gpu.kv_cache_write_q8_0(&d_kc_big, &dk_big, &pos_buf, n_kv5, hd5)
            .unwrap();
        gpu.kv_cache_write_q8_0(&d_vc_big, &dv_big, &pos_buf, n_kv5, hd5)
            .unwrap();
    }
    let q_big: Vec<f32> = (0..n_h5 * hd5).map(|_| next()).collect();
    let dq_big = gpu.upload_f32(&q_big, &[n_h5 * hd5]).unwrap();
    let out_big = gpu.zeros(&[n_h5 * hd5], rdna_compute::DType::F32).unwrap();
    let max_tiles_big = (big_seq + 127) / 128;
    let partials_big = gpu
        .zeros(&[n_h5 * max_tiles_big * (2 + hd5)], rdna_compute::DType::F32)
        .unwrap();
    gpu.hip
        .memcpy_htod(&pos_buf, &((big_seq - 1) as i32).to_ne_bytes())
        .unwrap();
    gpu.attention_flash_q8_0(
        &dq_big,
        &d_kc_big,
        &d_vc_big,
        &out_big,
        &pos_buf,
        big_seq,
        n_h5,
        n_kv5,
        hd5,
        big_seq,
        &partials_big,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let ob = gpu.download_f32(&out_big).unwrap();
    // Expected: uniform attention over identical V → out[d] ≈ vfix dequantized.
    // Compare head 0's first 32 dims against the V block-0 dequant of vfix.
    let finite = !ob.iter().any(|x| x.is_nan() || x.is_infinite());
    let mut maxerr_big = 0.0f32;
    for d in 0..hd5 {
        maxerr_big = maxerr_big.max((ob[d] - vfix[d]).abs());
    }
    let big_pass = finite && maxerr_big < 0.05;
    eprintln!(
        "  seq={big_seq}: head0[0..4]={:?} max|out-V|={:.3e} finite={finite} -> {}",
        &ob[..4],
        maxerr_big,
        if big_pass { "PASS" } else { "FAIL" }
    );
    all_pass &= big_pass;

    if all_pass {
        eprintln!("\nPASS: flash parity + large-context flash all correct");
    } else {
        eprintln!("\nFAIL: flash parity/large-context check failed");
        std::process::exit(1);
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let v = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let v = 2.0f32.powi(exp - 15) * (1.0 + frac as f32 / 1024.0);
    if sign == 1 {
        -v
    } else {
        v
    }
}
