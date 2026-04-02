//! Comprehensive kernel test harness. Tests every dispatch path with synthetic data.
//! No model loading required — validates kernels independently.
//! Usage: cargo run --release --features deltanet --example test_kernels

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {} ({:.1} GB VRAM)", gpu.arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    macro_rules! test {
        ($name:expr, $body:expr) => {{
            eprint!("  {:50} ", $name);
            let t = Instant::now();
            let mut closure = || -> Result<(), String> { $body };
            match closure() {
                Ok(()) => { passed += 1; eprintln!("OK ({:.1}ms)", t.elapsed().as_secs_f64() * 1000.0); }
                Err(e) => { failed += 1; eprintln!("FAIL: {e}"); }
            }
        }};
    }

    macro_rules! skip {
        ($name:expr, $reason:expr) => {
            eprint!("  {:50} ", $name);
            skipped += 1;
            eprintln!("SKIP ({})", $reason);
        };
    }

    eprintln!("\n--- Basic ops ---");
    test!("alloc + free", {
        let t = gpu.alloc_tensor(&[1024], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("upload + download f32", {
        let data = vec![1.0f32; 256];
        let t = gpu.upload_f32(&data, &[256]).map_err(|e| format!("{e}"))?;
        let back = gpu.download_f32(&t).map_err(|e| format!("{e}"))?;
        assert_eq!(back.len(), 256);
        assert!((back[0] - 1.0).abs() < 1e-6, "got {}", back[0]);
        gpu.free_tensor(t).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("add_inplace_f32", {
        let a = gpu.upload_f32(&vec![1.0f32; 64], &[64]).map_err(|e| format!("{e}"))?;
        let b = gpu.upload_f32(&vec![2.0f32; 64], &[64]).map_err(|e| format!("{e}"))?;
        gpu.add_inplace_f32(&a, &b).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&a).map_err(|e| format!("{e}"))?;
        assert!((r[0] - 3.0).abs() < 1e-6);
        gpu.free_tensor(a).map_err(|e| format!("{e}"))?; gpu.free_tensor(b).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("rmsnorm_f32", {
        let x = gpu.upload_f32(&vec![1.0f32; 128], &[128]).map_err(|e| format!("{e}"))?;
        let w = gpu.upload_f32(&vec![1.0f32; 128], &[128]).map_err(|e| format!("{e}"))?;
        let o = gpu.alloc_tensor(&[128], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.rmsnorm_f32(&x, &w, &o, 1e-6).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
        assert!(r[0].is_finite(), "rmsnorm produced NaN");
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?; gpu.free_tensor(w).map_err(|e| format!("{e}"))?; gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("softmax_f32", {
        let x = gpu.upload_f32(&vec![1.0f32; 32], &[1, 32]).map_err(|e| format!("{e}"))?;
        gpu.softmax_f32(&x).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&x).map_err(|e| format!("{e}"))?;
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "softmax sum={sum}");
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    eprintln!("\n--- Attention kernels ---");
    for (label, n_heads, n_kv, hd, seq) in [
        ("attention_f32 hd=128 h=8 kv=2", 8, 2, 128, 16),
        ("attention_f32 hd=256 h=16 kv=4", 16, 4, 256, 16),
        ("attention_f32 hd=256 h=10 kv=2", 10, 2, 256, 16),
    ] {
        test!(label, {
            let q = gpu.upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let kv_dim = n_kv * hd;
            let k = gpu.upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim]).map_err(|e| format!("{e}"))?;
            let v = gpu.upload_f32(&vec![0.1f32; seq * kv_dim], &[seq * kv_dim]).map_err(|e| format!("{e}"))?;
            let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| format!("{e}"))?;
            let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("{e}"))?;
            let pos_val = (seq - 1) as i32;
            gpu.hip.memcpy_htod(&pos_buf, &pos_val.to_ne_bytes()).map_err(|e| format!("{e}"))?;
            gpu.attention_f32(&q, &k, &v, &o, &pos_buf, seq, n_heads, n_kv, hd, seq).map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "attention produced NaN");
            gpu.free_tensor(q).map_err(|e| format!("{e}"))?; gpu.free_tensor(k).map_err(|e| format!("{e}"))?; gpu.free_tensor(v).map_err(|e| format!("{e}"))?; gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
            gpu.hip.free(pos_buf).map_err(|e| format!("{e}"))?;
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- Q8 KV kernels ---");
    for (label, n_kv, hd) in [
        ("q8 write+attn hd=128 kv=8", 8, 128),
        ("q8 write+attn hd=256 kv=4", 4, 256),
        ("q8 write+attn hd=256 kv=2", 2, 256),
    ] {
        test!(label, {
            let n_heads = n_kv * 4; // GQA ratio 4
            let seq = 8;
            let q8_blocks = hd / 32;
            let q8_bytes_per_pos = n_kv * q8_blocks * 34;
            let cache_bytes = seq * q8_bytes_per_pos;
            let cache_elems = (cache_bytes + 3) / 4;
            let k_cache = gpu.zeros(&[cache_elems], DType::F32).map_err(|e| format!("{e}"))?;
            let v_cache = gpu.zeros(&[cache_elems], DType::F32).map_err(|e| format!("{e}"))?;
            let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("{e}"))?;

            // Write a few positions
            for p in 0..4 {
                let kv_data = gpu.upload_f32(&vec![0.1f32; n_kv * hd], &[n_kv * hd]).map_err(|e| format!("{e}"))?;
                let pv = p as i32;
                gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_q8_0(&k_cache, &kv_data, &pos_buf, n_kv, hd).map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_q8_0(&v_cache, &kv_data, &pos_buf, n_kv, hd).map_err(|e| format!("{e}"))?;
                gpu.free_tensor(kv_data).map_err(|e| format!("{e}"))?;
            }

            // Attention at pos 3
            let q = gpu.upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| format!("{e}"))?;
            let pv = 3i32;
            gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| format!("{e}"))?;
            gpu.attention_q8_0_kv(&q, &k_cache, &v_cache, &o, &pos_buf, 4, n_heads, n_kv, hd, seq).map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "q8 attention produced NaN at r[0]={}", r[0]);
            gpu.free_tensor(q).map_err(|e| format!("{e}"))?; gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(k_cache).map_err(|e| format!("{e}"))?; gpu.free_tensor(v_cache).map_err(|e| format!("{e}"))?;
            gpu.hip.free(pos_buf).map_err(|e| format!("{e}"))?;
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- Asymmetric turbo (q8-K + turbo4-V, hd=256) ---");
    for (label, n_heads, n_kv, hd) in [
        ("asym hd=256 h=16 kv=4 (9B config)", 16, 4, 256),
        ("asym hd=256 h=10 kv=2 (4B config)", 10, 2, 256),
    ] {
        test!(label, {
            let seq = 8;
            // K cache: Q8 format
            let q8_blocks = hd / 32;
            let q8_bytes_per_pos = n_kv * q8_blocks * 34;
            let k_bytes = seq * q8_bytes_per_pos;
            let k_cache = gpu.zeros(&[(k_bytes + 3) / 4], DType::F32).map_err(|e| format!("{e}"))?;

            // V cache: turbo4 format
            let t4_bytes_per_head = 4 + hd / 2;
            let v_bytes = seq * n_kv * t4_bytes_per_head;
            let v_cache = gpu.zeros(&[(v_bytes + 3) / 4], DType::F32).map_err(|e| format!("{e}"))?;

            // Sign tables
            let signs1 = gpu.upload_f32(&vec![1.0f32; hd], &[hd]).map_err(|e| format!("{e}"))?;
            let signs2 = gpu.upload_f32(&vec![1.0f32; hd], &[hd]).map_err(|e| format!("{e}"))?;
            let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("{e}"))?;

            // Write a few positions
            for p in 0..4 {
                let k_data = gpu.upload_f32(&vec![0.1f32; n_kv * hd], &[n_kv * hd]).map_err(|e| format!("{e}"))?;
                let v_data = gpu.upload_f32(&vec![0.1f32; n_kv * hd], &[n_kv * hd]).map_err(|e| format!("{e}"))?;
                let pv = p as i32;
                gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_q8_0(&k_cache, &k_data, &pos_buf, n_kv, hd).map_err(|e| format!("{e}"))?;
                gpu.kv_cache_write_turbo4_v256(&v_cache, &v_data, &pos_buf, &signs1, &signs2, n_kv, hd).map_err(|e| format!("{e}"))?;
                gpu.free_tensor(k_data).map_err(|e| format!("{e}"))?; gpu.free_tensor(v_data).map_err(|e| format!("{e}"))?;
            }

            // Asymmetric attention at pos 3
            let q = gpu.upload_f32(&vec![0.1f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| format!("{e}"))?;
            let pv = 3i32;
            gpu.hip.memcpy_htod(&pos_buf, &pv.to_ne_bytes()).map_err(|e| format!("{e}"))?;
            gpu.attention_q8k_turbo4v_256(
                &q, &k_cache, &v_cache, &o, &pos_buf, &signs1, &signs2,
                4, n_heads, n_kv, hd, seq,
            ).map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "asym attention produced NaN at r[0]={}", r[0]);
            gpu.free_tensor(q).map_err(|e| format!("{e}"))?; gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(k_cache).map_err(|e| format!("{e}"))?; gpu.free_tensor(v_cache).map_err(|e| format!("{e}"))?;
            gpu.free_tensor(signs1).map_err(|e| format!("{e}"))?; gpu.free_tensor(signs2).map_err(|e| format!("{e}"))?;
            gpu.hip.free(pos_buf).map_err(|e| format!("{e}"))?;
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- GDN (tiled LDS) ---");
    for (label, n_heads, hd) in [
        ("gdn_q8 h=32 hd=128 (9B DeltaNet)", 32, 128),
        ("gdn_q8 h=16 hd=128 (4B DeltaNet)", 16, 128),
    ] {
        test!(label, {
            let s_size = n_heads * hd * hd;
            let scale_size = n_heads * hd;
            let s_q8 = gpu.zeros(&[s_size], DType::F32).map_err(|e| format!("{e}"))?; // int8 but alloc as bytes
            let s_scales = gpu.upload_f32(&vec![1.0f32; scale_size], &[scale_size]).map_err(|e| format!("{e}"))?;
            let q = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let k = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let v = gpu.upload_f32(&vec![0.01f32; n_heads * hd], &[n_heads * hd]).map_err(|e| format!("{e}"))?;
            let alpha = gpu.upload_f32(&vec![0.5f32; n_heads], &[n_heads]).map_err(|e| format!("{e}"))?;
            let beta = gpu.upload_f32(&vec![0.5f32; n_heads], &[n_heads]).map_err(|e| format!("{e}"))?;
            let o = gpu.alloc_tensor(&[n_heads * hd], DType::F32).map_err(|e| format!("{e}"))?;
            gpu.gated_delta_net_q8(&q, &k, &v, &alpha, &beta, &s_q8, &s_scales, &o, 1, n_heads, hd).map_err(|e| format!("{e}"))?;
            let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
            assert!(r[0].is_finite(), "gdn produced NaN");
            for t in [q, k, v, alpha, beta, s_q8, s_scales, o] { gpu.free_tensor(t).map_err(|e| format!("{e}"))?; }
            Ok::<(), String>(())
        });
    }

    eprintln!("\n--- Vision encoder kernels ---");
    test!("gemm_f16 (vision GEMM)", {
        let m = 32; let k = 64; let n = 8;
        let w_data: Vec<u8> = vec![0; m * k * 2]; // F16 zeros
        let w = gpu.upload_raw(&w_data, &[w_data.len()]).map_err(|e| format!("{e}"))?;
        let x = gpu.upload_f32(&vec![1.0f32; n * k], &[n * k]).map_err(|e| format!("{e}"))?;
        let y = gpu.alloc_tensor(&[m * n], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.gemm_f16(&w, &x, &y, m, k, n).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&y).map_err(|e| format!("{e}"))?;
        assert_eq!(r.len(), m * n);
        assert!(r[0].is_finite());
        gpu.free_tensor(w).map_err(|e| format!("{e}"))?; gpu.free_tensor(x).map_err(|e| format!("{e}"))?; gpu.free_tensor(y).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("layernorm_batched", {
        let batch = 4; let dim = 64;
        let x = gpu.upload_f32(&vec![1.0f32; batch * dim], &[batch * dim]).map_err(|e| format!("{e}"))?;
        let w = gpu.upload_f32(&vec![1.0f32; dim], &[dim]).map_err(|e| format!("{e}"))?;
        let b = gpu.upload_f32(&vec![0.0f32; dim], &[dim]).map_err(|e| format!("{e}"))?;
        let o = gpu.alloc_tensor(&[batch * dim], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.layernorm_batched(&x, &w, &b, &o, batch, dim, 1e-6).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&o).map_err(|e| format!("{e}"))?;
        assert!(r[0].is_finite());
        gpu.free_tensor(x).map_err(|e| format!("{e}"))?; gpu.free_tensor(w).map_err(|e| format!("{e}"))?; gpu.free_tensor(b).map_err(|e| format!("{e}"))?; gpu.free_tensor(o).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });
    test!("transpose_f32", {
        let rows = 4; let cols = 8;
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src = gpu.upload_f32(&data, &[rows * cols]).map_err(|e| format!("{e}"))?;
        let dst = gpu.alloc_tensor(&[rows * cols], DType::F32).map_err(|e| format!("{e}"))?;
        gpu.transpose_f32(&src, &dst, rows, cols).map_err(|e| format!("{e}"))?;
        let r = gpu.download_f32(&dst).map_err(|e| format!("{e}"))?;
        // r[0] = data[0*cols+0]=0, r[1] = data[1*cols+0]=8, r[2] = data[2*cols+0]=16
        assert!((r[1] - 8.0).abs() < 0.01, "transpose: r[1]={} expected 8", r[1]);
        gpu.free_tensor(src).map_err(|e| format!("{e}"))?; gpu.free_tensor(dst).map_err(|e| format!("{e}"))?;
        Ok::<(), String>(())
    });

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed:  {passed}");
    eprintln!("  Failed:  {failed}");
    eprintln!("  Skipped: {skipped}");
    if failed > 0 { std::process::exit(1); }
}
