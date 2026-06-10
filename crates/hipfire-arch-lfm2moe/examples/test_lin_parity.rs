// fwd parity: lin() with MFMA flag vs naive. Layout bug -> O(1); bf16 -> ~1e-2.
use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};
fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 4000) as f32 / 2000.0) - 1.0 }
fn main() {
    let mut gpu = Gpu::init().unwrap();
    let mut ok = true;
    for (m, k, n) in [(16usize, 1024usize, 3072usize), (4, 32, 96), (16, 5120, 1024), (32, 1024, 1024)] {
        let x: Vec<f32> = (0..m * k).map(|i| frand(i + 3)).collect();
        let w: Vec<f32> = (0..n * k).map(|i| frand(i + 77)).collect();
        let xg = gpu.upload_f32(&x, &[m, k]).unwrap();
        let wg = gpu.upload_f32(&w, &[n, k]).unwrap();
        let yn = gpu.zeros(&[m, n], DType::F32).unwrap();
        gpu.linear_fwd_f32(&xg, &wg, &yn, m, k, n).unwrap();
        let wb = gpu.zeros(&[n, k], DType::F16).unwrap();
        gpu.to_bf16_f32(&wg, &wb, n * k).unwrap();
        let xb = gpu.zeros(&[m, k], DType::F16).unwrap();
        gpu.to_bf16_f32(&xg, &xb, m * k).unwrap();
        let ym = gpu.zeros(&[m, n], DType::F32).unwrap();
        gpu.gemm_bf16_mfma_splitk(&wb, &xb, &ym, n, k, m).unwrap();
        let a = gpu.download_f32(&yn).unwrap(); let b = gpu.download_f32(&ym).unwrap();
        let (mut nu, mut de) = (0f64, 0f64);
        for i in 0..a.len() { let d = (a[i]-b[i]) as f64; nu += d*d; de += (a[i] as f64).powi(2); }
        let l2 = (nu/de.max(1e-30)).sqrt();
        println!("  m={m} k={k} n={n} rel_L2={l2:.3e} {}", if l2 < 5e-2 {"ok"} else {"FAIL"});
        ok &= l2 < 5e-2;
    }
    let _ = dt::dflash_use_mfma();
    if ok { println!("lin_parity: PASS"); } else { println!("lin_parity: FAIL"); std::process::exit(1); }
}
