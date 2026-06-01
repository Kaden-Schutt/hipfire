use rdna_compute::{DType, Gpu};
/// Micro-bench: compare MQ4-Lloyd gate_up baseline vs nosync on gfx1100.
/// Usage:
///   cargo run --release -p rdna-compute --example bench_gate_up_nosync
///   HIPFIRE_GATE_UP_NOSYNC=1 cargo run --release -p rdna-compute --example bench_gate_up_nosync
use std::time::Instant;

fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let ne = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut nm = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (nm & 1) != 0) {
            nm += 1;
        }
        let mut eb = ne;
        if nm == 0x400 {
            nm = 0;
            eb += 1;
        }
        (sign << 15) | (eb << 10) | nm
    };
    h.to_le_bytes()
}

/// Build MQ4-Lloyd weight data (160 B/group = 32 B codebook + 128 B indices).
fn build_mq4_lloyd(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let groups = k / 256;
    let mut data = vec![0u8; m * groups * 160];
    let mut r = seed as u64;
    for row in 0..m {
        for g in 0..groups {
            let off = (row * groups + g) * 160;
            for i in 0..16 {
                r = r
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = 0.5 + ((r >> 32) as f32) * 0.001;
                let b = f32_to_f16_bytes(v);
                data[off + i * 2..off + i * 2 + 2].copy_from_slice(&b);
            }
            for p in 0..32 {
                let mut pk = 0u32;
                for _ in 0..8 {
                    r = r
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    pk = (pk >> 4) | (((r >> 32) as u32 & 0xF) << 28);
                }
                data[off + 32 + p * 4..off + 32 + p * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    data
}

fn main() {
    let m = 27648usize; // Qwen 9B FFN total rows (gate_m + up_m)
    let k = 4096usize;
    let gm = m / 2;
    let um = m - gm;
    let n = 64usize; // prefill batch

    let a_gate = build_mq4_lloyd(gm, k, 42);
    let a_up = build_mq4_lloyd(um, k, 99);
    let x_f32: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.001).collect();
    let x_u8: Vec<u8> = x_f32.iter().flat_map(|&v| f32_to_f16_bytes(v)).collect();

    let w_bytes = (m * (k / 256) * 160) as f64;
    let x_bytes = (n * k * 2) as f64;
    let y_bytes = (n * m * 4) as f64;
    let total = w_bytes + x_bytes + y_bytes;

    let mut gpu = Gpu::init_with_device(0).expect("GPU init");
    let d_ag = gpu
        .upload_raw(&a_gate, &[a_gate.len()])
        .expect("upload gate");
    let d_au = gpu.upload_raw(&a_up, &[a_up.len()]).expect("upload up");
    let d_x = gpu.upload_raw(&x_u8, &[n * k * 2]).expect("upload x");
    let d_yg = gpu.zeros(&[n * gm], DType::F32).expect("yg zeros");
    let d_yu = gpu.zeros(&[n * um], DType::F32).expect("yu zeros");

    let nosync = std::env::var("HIPFIRE_GATE_UP_NOSYNC").is_ok();
    let label = if nosync { "NOSYNC" } else { "BASELINE" };
    println!(
        "=== MQ4-Lloyd gate_up: gfx1100 | {} | M={} K={} N={} ===",
        label, m, k, n
    );

    // Warmup (3 iters + compile)
    for i in 0..5 {
        gpu.gemm_gate_up_mq4g256_lloyd_wmma(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gm, um, k, n)
            .unwrap_or_else(|e| panic!("kernel failed at warmup iter {i}: {e:?}"));
    }
    gpu.hip.device_synchronize().expect("sync after warmup");

    let runs = 30;
    let t0 = Instant::now();
    for _ in 0..runs {
        gpu.gemm_gate_up_mq4g256_lloyd_wmma(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gm, um, k, n)
            .expect("kernel failed");
    }
    gpu.hip.device_synchronize().expect("sync after bench");
    let elapsed = t0.elapsed().as_secs_f64();
    let avg_us = elapsed * 1_000_000.0 / runs as f64;
    let gbps = total / elapsed / 1_073_741_824.0;
    println!("[{}] {:.1} µs/call  {:.1} GiB/s", label, avg_us, gbps);
}
