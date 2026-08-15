//! Kernel-level A/B throughput bench: per-token indexed GEMV vs grouped-WMMA
//! for the E8 MoE gate_up projection, on a realistic A3B prefill shape
//! (K=2048 hidden, M=768 expert intermediate, E=128, k_top=8). Sweeps prefill
//! token count. NOT a correctness check (see test_moe_grouped_wmma_e8) — pure
//! wall-clock. Runs on gfx1151 (RDNA3) + gfx1200/1201 (RDNA4, the .gfx12
//! sibling kernel selected by the launcher). The per-token path is what the
//! whole-model E8 prefill currently falls back to; the grouped path is the
//! ported kernel under test.
//!
//! Run: HIP_VISIBLE_DEVICES=0 cargo run --release -p rdna-compute \
//!        --example bench_moe_e8_prefill

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state & 0x7fff_ffff
}

fn build_expert_weight_e8(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert!(k % 256 == 0);
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut buf = vec![0u8; m * row_bytes];
    let mut s = seed;
    let rs_bits = [0x3800u16, 0x3C00, 0x3E00, 0x4000];
    for row in 0..m {
        let row_off = row * row_bytes;
        let rs = rs_bits[row % 4];
        buf[row_off] = (rs & 0xff) as u8;
        buf[row_off + 1] = (rs >> 8) as u8;
        for b in 0..n_blocks {
            let boff = row_off + 16 + b * 17;
            buf[boff] = (0x30 + (lcg(&mut s) & 0xf)) as u8;
            for c in 0..4 {
                let cw = lcg(&mut s).wrapping_mul(2654435761) ^ lcg(&mut s);
                let cwoff = boff + 1 + c * 4;
                buf[cwoff] = (cw & 0xff) as u8;
                buf[cwoff + 1] = ((cw >> 8) & 0xff) as u8;
                buf[cwoff + 2] = ((cw >> 16) & 0xff) as u8;
                buf[cwoff + 3] = ((cw >> 24) & 0xff) as u8;
            }
        }
    }
    buf
}

fn build_x_f32(n: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0f32; n * k];
    for v in out.iter_mut() {
        *v = -1.0 + (lcg(&mut s) as f32 / 0x7fff_ffff as f32) * 2.0;
    }
    out
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, data).unwrap();
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu.alloc_tensor(&[data.len()], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu.alloc_tensor(&[data.len() * 4], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu.alloc_tensor(&[data.len() * 8], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
    t
}
fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    gpu.hip.memset(&t.buf, 0, n * 4).unwrap();
    t
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    let ok = gpu.arch_caps.has_wmma_w32() || matches!(arch.as_str(), "gfx1200" | "gfx1201");
    if !ok {
        println!("SKIP — arch {} lacks RDNA3/RDNA4 WMMA", arch);
        return;
    }
    if std::env::var_os("HIPFIRE_DS4_MOE_UNSCATTER_ONLY").is_some() {
        run_ds4_unscatter_fusion(&mut gpu);
        return;
    }
    println!(
        "arch={}  (gate_up A/B: per-token GEMV[1 launch] vs grouped-WMMA[x2])",
        arch
    );

    let k = 2048usize; // hidden
    let m = 768usize; // expert intermediate
    let e = 128usize; // experts
    let k_top = 8usize;

    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for ei in 0..e {
        let bytes = build_expert_weight_e8(m, k, 0x1234_5678u32.wrapping_add(ei as u32 * 9973));
        let t = upload_u8(&mut gpu, &bytes);
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);

    let iters = 100usize;
    for &n in &[128usize, 512, 1024, 2048] {
        let m_total = n * k_top;
        let sorted: Vec<i32> = (0..m_total as i32).collect();
        let sorted_t = upload_i32(&mut gpu, &sorted);
        let tile_ids: Vec<i32> = (0..(m_total / 16)).map(|t| (t % e) as i32).collect();
        let tile_t = upload_i32(&mut gpu, &tile_ids);
        let x = build_x_f32(n, k, 0xCAFE_0000u32 ^ n as u32);
        let x_t = upload_f32(&mut gpu, &x);
        let y_grouped = alloc_f32_zeros(&mut gpu, m_total * m);

        let mut s = 0x99u32 ^ n as u32;
        let topk: Vec<i32> = (0..n * k_top)
            .map(|_| (lcg(&mut s) % e as u32) as i32)
            .collect();
        let topk_t = upload_i32(&mut gpu, &topk);
        let y_gate = alloc_f32_zeros(&mut gpu, n * k_top * m);
        let y_up = alloc_f32_zeros(&mut gpu, n * k_top * m);

        // warmup (JIT both kernels + DPM)
        for _ in 0..10 {
            gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                &expert_ptrs,
                &topk_t,
                &x_t,
                &y_gate,
                &y_up,
                m,
                k,
                k_top,
                n,
            )
            .unwrap();
            gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
                &expert_ptrs,
                &tile_t,
                &sorted_t,
                &x_t,
                &y_grouped,
                m,
                k,
                k_top,
                m_total,
                n,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // per-token GEMV: gate+up in one launch
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.gemv_mfp4g32_e8_moe_gate_up_k8_indexed_batched(
                &expert_ptrs,
                &topk_t,
                &x_t,
                &y_gate,
                &y_up,
                m,
                k,
                k_top,
                n,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let gemv_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // grouped-WMMA: 2 launches (gate + up) to match the GEMV's two projections
        let t1 = Instant::now();
        for _ in 0..iters {
            gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
                &expert_ptrs,
                &tile_t,
                &sorted_t,
                &x_t,
                &y_grouped,
                m,
                k,
                k_top,
                m_total,
                n,
            )
            .unwrap();
            gpu.gemm_mfp4g32_e8_moe_grouped_wmma(
                &expert_ptrs,
                &tile_t,
                &sorted_t,
                &x_t,
                &y_grouped,
                m,
                k,
                k_top,
                m_total,
                n,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let grp_us = t1.elapsed().as_micros() as f64 / iters as f64;

        // FLOPs for gate+up = 2 projections × 2 flop/MAC × (n*k_top) slots × M × K
        let flops = 2.0 * 2.0 * (n * k_top) as f64 * m as f64 * k as f64;
        println!(
            "N={:5} m_total={:6} | GEMV {:8.1} us ({:5.1} TF) | grouped {:8.1} us ({:6.1} TF) | speedup {:.2}x",
            n, m_total, gemv_us, flops / gemv_us / 1e6, grp_us, flops / grp_us / 1e6, gemv_us / grp_us
        );
    }
}

fn run_ds4_unscatter_fusion(gpu: &mut Gpu) {
    assert_eq!(gpu.arch, "gfx1151", "DS4 fusion screen is gfx1151-only");
    const BATCH: usize = 1024;
    const K_TOP: usize = 6;
    const IM: usize = 2048;
    const N_EXP: usize = 256;
    const BLOCK_M: usize = 16;
    const TRIALS: usize = 20;
    const SWIGLU_LIMIT: f32 = 10.0;

    let valid_slots = BATCH * K_TOP;
    let m_total = valid_slots + N_EXP * BLOCK_M;
    let y_grouped_host = build_x_f32(m_total, 2 * IM, 0xD54F_0510);
    let mut sorted = vec![-1i32; m_total];
    for (slot, flat) in sorted.iter_mut().take(valid_slots).enumerate() {
        *flat = ((slot * 5) % valid_slots) as i32;
    }

    let y_grouped = upload_f32(gpu, &y_grouped_host);
    let sorted_slot_index = upload_i32(gpu, &sorted);
    let gate = alloc_f32_zeros(gpu, valid_slots * IM);
    let up = alloc_f32_zeros(gpu, valid_slots * IM);
    let fused = alloc_f32_zeros(gpu, valid_slots * IM);

    let mut run_baseline = |gpu: &mut Gpu| {
        gpu.moe_gate_up_unscatter_k8(
            &y_grouped,
            &sorted_slot_index,
            &gate,
            &up,
            IM,
            K_TOP,
            m_total,
        )
        .expect("baseline unscatter");
        gpu.deepseek4_silu_mul_clamp_f32_batched(&gate, &up, &gate, IM, valid_slots, SWIGLU_LIMIT)
            .expect("baseline SwiGLU");
    };
    let mut run_fused = |gpu: &mut Gpu| {
        gpu.moe_unscatter_silu_clamp_k8(
            &y_grouped,
            &sorted_slot_index,
            &fused,
            IM,
            K_TOP,
            m_total,
            SWIGLU_LIMIT,
        )
        .expect("fused unscatter SwiGLU");
    };

    for _ in 0..5 {
        run_baseline(gpu);
        run_fused(gpu);
    }
    gpu.hip.device_synchronize().expect("warm synchronize");

    let mut baseline_host = vec![0.0f32; valid_slots * IM];
    let mut fused_host = vec![0.0f32; valid_slots * IM];
    gpu.hip
        .memcpy_dtoh(
            unsafe {
                std::slice::from_raw_parts_mut(
                    baseline_host.as_mut_ptr() as *mut u8,
                    baseline_host.len() * 4,
                )
            },
            &gate.buf,
        )
        .expect("download baseline");
    gpu.hip
        .memcpy_dtoh(
            unsafe {
                std::slice::from_raw_parts_mut(
                    fused_host.as_mut_ptr() as *mut u8,
                    fused_host.len() * 4,
                )
            },
            &fused.buf,
        )
        .expect("download fused");
    let exact = baseline_host
        .iter()
        .zip(&fused_host)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert_eq!(
        exact,
        baseline_host.len(),
        "fused unscatter SwiGLU must be raw-bit exact"
    );

    let time_arm = |gpu: &mut Gpu, run: &mut dyn FnMut(&mut Gpu)| -> f64 {
        let start = Instant::now();
        for _ in 0..TRIALS {
            run(gpu);
        }
        gpu.hip.device_synchronize().expect("timing synchronize");
        start.elapsed().as_secs_f64() * 1.0e6 / TRIALS as f64
    };
    let a0 = time_arm(gpu, &mut run_baseline);
    let b0 = time_arm(gpu, &mut run_fused);
    let b1 = time_arm(gpu, &mut run_fused);
    let a1 = time_arm(gpu, &mut run_baseline);
    let baseline_us = 0.5 * (a0 + a1);
    let fused_us = 0.5 * (b0 + b1);

    println!("=== DS4 grouped-MoE unscatter + SwiGLU fusion ===");
    println!(
        "arch={} B={} K_TOP={} IM={} m_total={} trials={}",
        gpu.arch, BATCH, K_TOP, IM, m_total, TRIALS
    );
    println!("raw_bit_exact={exact}/{}", baseline_host.len());
    println!(
        "baseline_us={baseline_us:.3} fused_us={fused_us:.3} speedup={:.4}x delta={:.3}%",
        baseline_us / fused_us,
        (baseline_us / fused_us - 1.0) * 100.0
    );
}
