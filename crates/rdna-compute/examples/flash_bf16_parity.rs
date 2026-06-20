//! Parity: `flash_attn_bf16` vs `vit_attention_f32` on bf16-precision data.
//!
//! Both compute the same bidirectional softmax(QKᵀ/√d)·V over a fused qkv at the
//! SigLIP shape (N, 16 heads, head_dim=72). To isolate the kernel from bf16
//! rounding, the reference is run on the **same** bf16-rounded values (upcast to
//! f32), so the only remaining difference is f32-accumulation order — expect
//! ~1e-5, never the ~1.0 of a kernel bug.
//!
//! Usage: cargo run --release --example flash_bf16_parity -p rdna-compute

use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

// Round an f32 to bf16 precision (RNE), returned as f32 (top 16 bits kept).
fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    let lsb = (u >> 16) & 1;
    f32::from_bits((u + 0x7fff + lsb) & 0xffff_0000)
}
fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let lsb = (u >> 16) & 1;
    ((u + 0x7fff + lsb) >> 16) as u16
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let n = 512usize;
    let heads = 16usize;
    let hd = 72usize;
    let hidden = heads * hd;

    let qkv = lcg(0xBEEF, n * 3 * hidden);
    // bf16-rounded f32 (for the f32 reference) and packed bf16 (for the kernel).
    let qkv_round: Vec<f32> = qkv.iter().map(|&x| bf16_round(x)).collect();
    let qkv_bf16_bytes: Vec<u8> = qkv
        .iter()
        .flat_map(|&x| bf16_bits(x).to_le_bytes())
        .collect();

    let d_round = gpu.upload_f32(&qkv_round, &[n * 3 * hidden]).unwrap();
    let mut d_bf16 = gpu.upload_raw(&qkv_bf16_bytes, &[n * 3 * hidden]).unwrap();
    d_bf16.dtype = DType::BF16;

    let d_ref = gpu.zeros(&[n * hidden], DType::F32).unwrap();
    let d_opt = gpu.zeros(&[n * hidden], DType::F32).unwrap();

    gpu.vit_attention_f32(&d_round, &d_ref, n, hidden, heads, hd)
        .unwrap();
    gpu.flash_attn_bf16(&d_bf16, &d_opt, n, hidden, heads, hd)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let a = gpu.download_f32(&d_ref).unwrap();
    let b = gpu.download_f32(&d_opt).unwrap();
    let (mut max_abs, mut sd, mut sr) = (0.0f32, 0.0f64, 0.0f64);
    for (p, q) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((p - q).abs());
        sd += (p - q).abs() as f64;
        sr += p.abs() as f64;
    }
    let rel = sd / sr.max(1e-12);
    println!("max_abs_diff = {max_abs:.3e}");
    println!("rel_L1       = {rel:.3e}");
    assert!(rel < 1e-3, "flash_attn_bf16 diverges (rel_L1={rel:.3e})");
    println!("PARITY OK (rel_L1 < 1e-3)");
}
