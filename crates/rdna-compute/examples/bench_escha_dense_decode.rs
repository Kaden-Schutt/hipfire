// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Escha native GEMV at the shapes the 27B DENSE model decodes at.
//!
//! WHY THIS EXISTS SEPARATELY FROM `bench_escha_grouped_gemm`: that bench runs
//! one real A3B prefill chunk — 2048 slots, ic=2048, oc=1024 — and at that
//! shape the slot-parallel kernel already moves 2.147 GB at ~175 GB/s against
//! this box's ~209-220 GB/s ceiling. It is 80%+ BANDWIDTH-BOUND, so no
//! instruction-level change can move it, and using it to evaluate one is a
//! category error. Decoding the dense 27B is the opposite regime: ONE slot,
//! weights streamed once, measured at 86 GB/s. That is where decode cost shows.
//!
//! Shapes are the real 27B projections (`hidden 5120`, 64 layers), with the
//! documented per-projection K split: `gate_proj` is K=2, `up_proj` K=3.

use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) - 0.5
    }
}

fn ptr_table(gpu: &Gpu, owner: &GpuTensor, n_exp: usize, stride_bytes: usize) -> GpuTensor {
    let bytes: Vec<u8> = (0..n_exp)
        .map(|e| owner.buf.as_ptr() as u64 + (e * stride_bytes) as u64)
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[2 * n_exp]).expect("ptr table")
}

struct Case {
    name: &'static str,
    ic: usize,
    oc: usize,
    trellis_k: u32,
}

fn main() {
    // The 27B dense projections. `gate_proj` K=2 / `up_proj` K=3 is the
    // documented trap: they cannot share one kernel call.
    let cases = [
        Case {
            name: "gate_proj",
            ic: 5120,
            oc: 17408,
            trellis_k: 2,
        },
        Case {
            name: "up_proj",
            ic: 5120,
            oc: 17408,
            trellis_k: 3,
        },
        Case {
            name: "down_proj",
            ic: 17408,
            oc: 5120,
            trellis_k: 3,
        },
        Case {
            name: "in_proj_z",
            ic: 5120,
            oc: 6144,
            trellis_k: 2,
        },
    ];
    let slots = 1usize; // decode: one token, one "expert"
    let n_exp = 1usize;
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    let mut gpu = Gpu::init().expect("gpu");
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    let ids: Vec<u8> = (0..slots).flat_map(|_| 0i32.to_le_bytes()).collect();
    let d_ids = gpu.upload_raw(&ids, &[slots]).expect("ids");

    println!("escha native GEMV, DENSE DECODE regime (slots=1)");
    println!("box ceiling ~209-220 GB/s; the A3B prefill bench sits at ~175 GB/s\n");

    for case in &cases {
        let words_per_expert = (case.ic / 16) * (case.oc / 16) * 16 * case.trellis_k as usize;
        let expert_bytes = words_per_expert * 2;

        let mut crng = Rng(0x5EED_0000_0000_0001 ^ case.ic as u64);
        let mut code_bytes = vec![0u8; n_exp * expert_bytes];
        for chunk in code_bytes.chunks_exact_mut(4) {
            chunk.copy_from_slice(&crng.next_u32().to_le_bytes());
        }
        let d_code = gpu
            .upload_raw(&code_bytes, &[code_bytes.len()])
            .expect("upload code");
        drop(code_bytes);
        let code_ptrs = ptr_table(&gpu, &d_code, n_exp, expert_bytes);

        let x_bytes: Vec<u8> = (0..slots * case.ic)
            .flat_map(|_| rng.next_f32().to_le_bytes())
            .collect();
        let d_x = gpu.upload_raw(&x_bytes, &[slots * case.ic]).expect("x");
        drop(x_bytes);
        let d_y = gpu.alloc_tensor(&[slots * case.oc], DType::F32).expect("y");

        // Warm up, then take the MINIMUM over `iters` — the fastest observed
        // launch is the one least polluted by other work on the box.
        for _ in 0..20 {
            gpu.escha_gemv_native_moe_k8_indexed_batched(
                &code_ptrs,
                &d_ids,
                &d_x,
                &d_y,
                case.oc,
                case.ic,
                slots,
                case.trellis_k,
            )
            .expect("gemv warmup");
        }
        gpu.hip.device_synchronize().expect("sync");

        let mut best = f64::INFINITY;
        for _ in 0..iters {
            gpu.hip.device_synchronize().expect("sync");
            let t = std::time::Instant::now();
            gpu.escha_gemv_native_moe_k8_indexed_batched(
                &code_ptrs,
                &d_ids,
                &d_x,
                &d_y,
                case.oc,
                case.ic,
                slots,
                case.trellis_k,
            )
            .expect("gemv");
            gpu.hip.device_synchronize().expect("sync");
            best = best.min(t.elapsed().as_secs_f64());
        }

        let weights = case.ic * case.oc;
        let code_gb = expert_bytes as f64 / 1e9;
        let wide = case.ic <= 1536;
        println!(
            "  {:<10} ic={:<6} oc={:<6} K={}  {:<6}  {:8.1} us  \
             {:.3} GB code -> {:6.1} GB/s  ({:.2} G weights/s)",
            case.name,
            case.ic,
            case.oc,
            case.trellis_k,
            if wide { "wide" } else { "narrow" },
            best * 1e6,
            code_gb,
            code_gb / best,
            weights as f64 / best / 1e9,
        );
    }
}
