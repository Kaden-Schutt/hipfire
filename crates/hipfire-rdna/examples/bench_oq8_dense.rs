// De-risk bench for wiring Opus-Quant W8A8 (oq8) into the diffusion linear path.
//
// The Krea2 DiT warm step is ~92% bf16 GEMM at ~0.23 TFLOP/s effective on
// gfx1103 (the naive gemm_bf16_x_bf16_wmma — one wave per 16x16 output tile, no
// large-tile reuse; the tiled _m128 kernel is gfx1151-only). Before doing the
// multi-crate oq8 plumbing, this measures whether gemm_oq8_grouped_wmma +
// quantize_act_oq8 actually beat that baseline on the real DiT GEMM shapes, and
// how much accuracy plain per-group symmetric int8 (no FWHT rotation) costs.
//
// Run under the GPU lock:
//   hipfire lock acquire bench-oq8 && \
//   cargo run --release -p hipfire-rdna --example bench_oq8_dense ; \
//   hipfire lock release

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::time::Instant;

const GROUP: usize = 256;

// Deterministic LCG in [-1, 1) so runs are reproducible without a rng crate.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    // Round-to-nearest-even truncation of the f32 mantissa to bf16.
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}

// Per-group (256) symmetric int8 quant of a row-major [M,K] f32 matrix.
// Returns (int8 [M*K], scales f32 [M*ng]).
fn quantize_w_oq8(w: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / GROUP;
    let mut q = vec![0i8; m * k];
    let mut s = vec![0f32; m * ng];
    for row in 0..m {
        for g in 0..ng {
            let base = row * k + g * GROUP;
            let mut amax = 0f32;
            for i in 0..GROUP {
                amax = amax.max(w[base + i].abs());
            }
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            s[row * ng + g] = scale;
            let inv = 1.0 / scale;
            for i in 0..GROUP {
                let v = (w[base + i] * inv).round().clamp(-127.0, 127.0);
                q[base + i] = v as i8;
            }
        }
    }
    (q, s)
}

fn time_kernel<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f(gpu);
    }
    gpu.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn bench_shape(gpu: &mut Gpu, m: usize, k: usize, b: usize) {
    let mut rng = Lcg(0x1234_5678 ^ ((m * 131 + k * 17 + b) as u64));
    let w: Vec<f32> = (0..m * k).map(|_| rng.next_f32() * 0.05).collect();
    let x: Vec<f32> = (0..b * k).map(|_| rng.next_f32()).collect();

    // ---- bf16 weight tensor + f32 activation (the current diffusion path) ----
    let w_bf16_bytes: Vec<u8> = w
        .iter()
        .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
        .collect();
    let mut w_bf16 = gpu.upload_raw(&w_bf16_bytes, &[m, k]).unwrap();
    w_bf16.dtype = DType::BF16;
    let x_f32 = gpu.upload_f32(&x, &[b, k]).unwrap();
    let y_bf16 = gpu.alloc_tensor(&[b, m], DType::F32).unwrap();

    let bf16_s = time_kernel(gpu, 3, 15, |g| {
        g.gemm_bf16_x_bf16_wmma(&w_bf16, &x_f32, &y_bf16, m, k, b).unwrap();
    });

    // ---- register-tiled bf16 WMMA (the ILP/reuse fix), several tilings ----
    let y_tiled = gpu.alloc_tensor(&[b, m], DType::F32).unwrap();
    let mut tiled = Vec::new();
    for &(mb, nb) in &[(2usize, 2usize), (4, 2), (4, 4)] {
        let s = time_kernel(gpu, 3, 15, |g| {
            g.gemm_bf16_tiled_wmma(&w_bf16, &x_f32, &y_tiled, m, k, b, mb, nb)
                .unwrap();
        });
        tiled.push(((mb, nb), s));
    }

    // ---- oq8: per-group int8 weights + f32 scales, dynamic int8 activation ----
    let (wq, ws) = quantize_w_oq8(&w, m, k);
    let ng = k / GROUP;
    let wq_bytes: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
    let w_i8 = gpu.upload_raw(&wq_bytes, &[m * k]).unwrap();
    let w_scales = gpu.upload_f32(&ws, &[m * ng]).unwrap();
    let xq_i8 = gpu.alloc_tensor(&[b * k], DType::Raw).unwrap();
    let xs = gpu.alloc_tensor(&[b * ng], DType::F32).unwrap();
    let y_oq8 = gpu.alloc_tensor(&[b, m], DType::F32).unwrap();

    let oq8_s = time_kernel(gpu, 3, 15, |g| {
        g.quantize_act_oq8(&x_f32, &xq_i8, &xs, b, k, GROUP).unwrap();
        g.gemm_oq8_grouped_wmma(&w_i8, &w_scales, &xq_i8, &xs, &y_oq8, m, k, b, GROUP)
            .unwrap();
    });

    // ---- register-tiled oq8 (w8a8: tiling win + int8 footprint) ----
    let mut oq8_tiled = Vec::new();
    for &(mb, nb) in &[(2usize, 2usize), (4, 2), (2, 4)] {
        let s = time_kernel(gpu, 3, 15, |g| {
            g.quantize_act_oq8(&x_f32, &xq_i8, &xs, b, k, GROUP).unwrap();
            g.gemm_oq8_tiled_wmma(&w_i8, &w_scales, &xq_i8, &xs, &y_oq8, m, k, b, GROUP, mb, nb)
                .unwrap();
        });
        oq8_tiled.push(((mb, nb), s));
    }
    // validate the 2x4 oq8-tiled config numerics (the fastest, at the 255-VGPR
    // edge); reuses the sampled f32 reference below via y_oq8.
    gpu.quantize_act_oq8(&x_f32, &xq_i8, &xs, b, k, GROUP).unwrap();
    gpu.gemm_oq8_tiled_wmma(&w_i8, &w_scales, &xq_i8, &xs, &y_oq8, m, k, b, GROUP, 2, 4)
        .unwrap();
    gpu.device_synchronize().unwrap();

    // ---- accuracy vs an exact f32 reference on a sampled set of outputs ----
    let sample_rows = [0usize, b / 3, (2 * b) / 3, b - 1];
    let sample_cols = [0usize, m / 3, (2 * m) / 3, m - 1];
    let rel_rmse_of = |y_host: &[f32]| -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for &bi in &sample_rows {
            for &mi in &sample_cols {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += (w[mi * k + kk] as f64) * (x[bi * k + kk] as f64);
                }
                let got = y_host[bi * m + mi] as f64;
                num += (got - acc) * (got - acc);
                den += acc * acc;
            }
        }
        (num / den.max(1e-12)).sqrt()
    };
    let rel_rmse = rel_rmse_of(&gpu.download_f32(&y_oq8).unwrap());
    // Re-run the 4x4 tiled kernel once to validate its numerics too.
    gpu.gemm_bf16_tiled_wmma(&w_bf16, &x_f32, &y_tiled, m, k, b, 4, 4)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let tiled_rmse = rel_rmse_of(&gpu.download_f32(&y_tiled).unwrap());

    let flops = 2.0 * m as f64 * k as f64 * b as f64;
    let tf = |s: f64| flops / s / 1e12;
    println!(
        "M={m:>5} K={k:>5} B={b:>5} | bf16-naive {:>5.2} TF/s | oq8 {:>5.2} TF/s ({:>4.2}x, relRMSE {:.4})",
        tf(bf16_s),
        tf(oq8_s),
        bf16_s / oq8_s,
        rel_rmse,
    );
    for ((mb, nb), s) in &tiled {
        let acc = if (*mb, *nb) == (4, 4) {
            format!(", relRMSE {tiled_rmse:.4}")
        } else {
            String::new()
        };
        println!(
            "                                 | tiled {mb}x{nb}   {:>5.2} TF/s ({:>4.2}x vs naive{acc})",
            tf(*s),
            bf16_s / *s,
        );
    }
    for ((mb, nb), s) in &oq8_tiled {
        let acc = if (*mb, *nb) == (2, 4) {
            format!(", relRMSE {rel_rmse:.4}")
        } else {
            String::new()
        };
        println!(
            "                                 | oq8-tiled {mb}x{nb} {:>5.2} TF/s ({:>4.2}x vs naive{acc})",
            tf(*s),
            bf16_s / *s,
        );
    }

    for t in [w_bf16, x_f32, y_bf16, y_tiled, w_i8, w_scales, xq_i8, xs, y_oq8] {
        let _ = gpu.free_tensor(t);
    }
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    println!("arch = {}", gpu.arch);
    println!("Krea2 DiT dense GEMM shapes (seq B=1040):");
    // qkv/out/gate: MxKxB with M in {6144, 1536}; ffn gate/up: 16384x6144; down: 6144x16384.
    bench_shape(&mut gpu, 6144, 6144, 1040); // to_q / to_out / to_gate
    bench_shape(&mut gpu, 1536, 6144, 1040); // to_k / to_v
    bench_shape(&mut gpu, 16384, 6144, 1040); // ffn gate / up
    bench_shape(&mut gpu, 6144, 16384, 1040); // ffn down
}
