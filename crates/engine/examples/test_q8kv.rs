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
    gpu.kv_cache_write_q8_0(&d_cache, &d_src, &pos_buf, n_kv_heads, head_dim).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Read back raw bytes
    let mut raw = vec![0u8; 34];
    gpu.hip.memcpy_dtoh(&mut raw, &d_cache.buf).unwrap();

    // Parse f16 scale
    let scale_bits = u16::from_le_bytes([raw[0], raw[1]]);
    let scale = f16_to_f32(scale_bits);

    eprintln!("Input:  {:?}", &kv_data[..8]);
    eprintln!("scale_f16=0x{:04x} scale_f32={:.6}", scale_bits, scale);
    eprintln!("Expected amax={:.3}, expected scale≈{:.6}", 3.2, 3.2 / 127.0);

    let mut max_err = 0.0f32;
    for i in 0..head_dim {
        let q = raw[2 + i] as i8;
        let dequant = scale * q as f32;
        let err = (kv_data[i] - dequant).abs();
        max_err = max_err.max(err);
        if i < 8 || i >= head_dim - 2 {
            eprintln!("  [{i:>2}] input={:.3} q={:>4} dequant={:.4} err={:.4}",
                kv_data[i], q, dequant, err);
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
    let d_cache2 = gpu.zeros(&[cache_elems2], rdna_compute::DType::F32).unwrap();

    // Write pos=0: head0=[1.0]*32, head1=[2.0]*32
    let kv0: Vec<f32> = (0..kv_dim2).map(|i| if i < head_dim2 { 1.0 } else { 2.0 }).collect();
    let d_src0 = gpu.upload_f32(&kv0, &[kv_dim2]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_cache2, &d_src0, &pos_buf, n_kv_heads2, head_dim2).unwrap();

    // Write pos=1: head0=[3.0]*32, head1=[4.0]*32
    let kv1: Vec<f32> = (0..kv_dim2).map(|i| if i < head_dim2 { 3.0 } else { 4.0 }).collect();
    let d_src1 = gpu.upload_f32(&kv1, &[kv_dim2]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_cache2, &d_src1, &pos_buf, n_kv_heads2, head_dim2).unwrap();
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
            let expected_val = if h == 0 { (pos * 2 + 1) as f32 } else { (pos * 2 + 2) as f32 };
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
    gpu.kv_cache_write_q8_0(&d_kcache, &d_ones, &pos_buf, n_kv_a, hd_a).unwrap();
    gpu.kv_cache_write_q8_0(&d_vcache, &d_ones, &pos_buf, n_kv_a, hd_a).unwrap();

    // Write K pos1 = [0.5]*32, V pos1 = [2.0]*32
    let half32 = vec![0.5f32; hd_a];
    let twos32 = vec![2.0f32; hd_a];
    let d_half = gpu.upload_f32(&half32, &[hd_a]).unwrap();
    let d_twos = gpu.upload_f32(&twos32, &[hd_a]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kcache, &d_half, &pos_buf, n_kv_a, hd_a).unwrap();
    gpu.kv_cache_write_q8_0(&d_vcache, &d_twos, &pos_buf, n_kv_a, hd_a).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Q = [1.0]*32
    let d_q = gpu.upload_f32(&ones32, &[hd_a]).unwrap();
    let d_out = gpu.zeros(&[hd_a], rdna_compute::DType::F32).unwrap();

    // Run attention at pos=1 (seq_len=2: see both positions)
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.attention_q8_0_kv(&d_q, &d_kcache, &d_vcache, &d_out, &pos_buf, 2, n_heads_a, n_kv_a, hd_a, 4).unwrap();
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
        eprintln!("FAIL: Q8_0 attention output wrong (got {:.4}, expected ~{:.3})", out_vals[0], expected);
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
    gpu.kv_cache_write_q8_0(&d_kc8, &d_k0, &pos_buf, n_kv, hd).unwrap();
    gpu.kv_cache_write_q8_0(&d_vc8, &d_v0, &pos_buf, n_kv, hd).unwrap();

    let k1: Vec<f32> = (0..kv_dim_8b).map(|i| -0.01 * ((i % 128) as f32)).collect();
    let v1: Vec<f32> = vec![2.0f32; kv_dim_8b];
    let d_k1 = gpu.upload_f32(&k1, &[kv_dim_8b]).unwrap();
    let d_v1 = gpu.upload_f32(&v1, &[kv_dim_8b]).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.kv_cache_write_q8_0(&d_kc8, &d_k1, &pos_buf, n_kv, hd).unwrap();
    gpu.kv_cache_write_q8_0(&d_vc8, &d_v1, &pos_buf, n_kv, hd).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Q: 32 heads, each [0.01]*128 (aligned with K pos0)
    let q8b: Vec<f32> = vec![0.01f32; n_h * hd];
    let d_q8 = gpu.upload_f32(&q8b, &[n_h * hd]).unwrap();
    let d_out8 = gpu.zeros(&[n_h * hd], rdna_compute::DType::F32).unwrap();

    gpu.hip.memcpy_htod(&pos_buf, &1i32.to_ne_bytes()).unwrap();
    gpu.attention_q8_0_kv(&d_q8, &d_kc8, &d_vc8, &d_out8, &pos_buf, 2, n_h, n_kv, hd, 4).unwrap();
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
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let v = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN };
    }
    let v = 2.0f32.powi(exp - 15) * (1.0 + frac as f32 / 1024.0);
    if sign == 1 { -v } else { v }
}
