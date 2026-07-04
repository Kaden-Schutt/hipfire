//! Numerical parity: `vit_attention_opt` vs the known-good `vit_attention_f32`.
//!
//! Decode-independent fidelity check for the optimized SigLIP attention kernel
//! (tiled K/V + shared-Q). Both kernels compute the same bidirectional
//! softmax(QKᵀ/√d)·V over a fused `qkv[N, 3·hidden]`; the only difference is the
//! tiling, so they must agree to float-reordering tolerance. Runs the SigLIP
//! shape (hidden=1152, heads=16, head_dim=72) at a modest N.
//!
//! Usage: cargo run --release --example vit_attn_parity -p hipfire-rdna

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);

    // SigLIP-so400m head structure; N kept modest so the naive kernel is quick.
    let n = 512usize;
    let num_heads = 16usize;
    let head_dim = 72usize;
    let hidden = num_heads * head_dim; // 1152
    eprintln!("shape: N={n} hidden={hidden} heads={num_heads} head_dim={head_dim}");

    let qkv = lcg(0x1234_5678, n * 3 * hidden);
    let d_qkv = gpu.upload_f32(&qkv, &[n * 3 * hidden]).unwrap();
    let d_ref = gpu.zeros(&[n * hidden], DType::F32).unwrap();
    let d_opt = gpu.zeros(&[n * hidden], DType::F32).unwrap();

    gpu.vit_attention_f32(&d_qkv, &d_ref, n, hidden, num_heads, head_dim)
        .unwrap();
    gpu.vit_attention_opt(&d_qkv, &d_opt, n, hidden, num_heads, head_dim)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let a = gpu.download_f32(&d_ref).unwrap();
    let b = gpu.download_f32(&d_opt).unwrap();
    assert_eq!(a.len(), b.len());

    let mut max_abs = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut sum_abs_ref = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs_diff += d as f64;
        sum_abs_ref += x.abs() as f64;
    }
    let rel_l1 = sum_abs_diff / sum_abs_ref.max(1e-12);
    println!("max_abs_diff = {max_abs:.3e}");
    println!("rel_L1       = {rel_l1:.3e}");

    // Float-reordering tolerance: tiled vs naive accumulate in a different order,
    // so expect ~1e-5 rel-L1, never the ~1.0 of the uninitialised-Q bug.
    assert!(
        rel_l1 < 1e-3,
        "vit_attention_opt diverges from vit_attention_f32 (rel_L1={rel_l1:.3e}) — not parity"
    );
    println!("PARITY OK (rel_L1 < 1e-3)");

    gpu.free_tensor(d_qkv).unwrap();
    gpu.free_tensor(d_ref).unwrap();
    gpu.free_tensor(d_opt).unwrap();
}
