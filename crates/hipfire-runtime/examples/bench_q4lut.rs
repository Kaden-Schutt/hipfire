// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Autokernel benchmark: all Q4 variants + Q8 baseline.
//! Tests: Q4_K, Q4_LUT, Q4_WAVE, Q4-as-Q8, Q8_0

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");

    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();
    let ti = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_q4k = gguf.tensor_data(ti);
    let m = 2048usize;
    let k = 2048usize;

    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_q4k, m * k);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();

    // Prepare all formats
    let q4lut = convert_q4k_to_q4lut(raw_q4k, m * k);
    let q4f16g32 = hipfire_runtime::llama::convert_q4k_to_q4f16_g32(raw_q4k, m * k);
    let q8 = quantize_q8(&a_f32);
    let q4as8 = quantize_q4_as_q8(&a_f32); // 4-bit precision in Q8 storage

    eprintln!("=== Format sizes ===");
    eprintln!("Q4_K:      {} bytes ({:.4} B/w)", raw_q4k.len(), raw_q4k.len() as f64 / (m*k) as f64);
    eprintln!("Q4_F16_G32:{} bytes ({:.4} B/w)", q4f16g32.len(), q4f16g32.len() as f64 / (m*k) as f64);
    eprintln!("Q4_LUT:    {} bytes ({:.4} B/w)", q4lut.len(), q4lut.len() as f64 / (m*k) as f64);
    eprintln!("Q8_0:      {} bytes ({:.4} B/w)", q8.len(), q8.len() as f64 / (m*k) as f64);
    eprintln!("Q4-as-Q8:  {} bytes ({:.4} B/w)", q4as8.len(), q4as8.len() as f64 / (m*k) as f64);

    let d_q4k = gpu.upload_raw(raw_q4k, &[raw_q4k.len()]).unwrap();
    let d_q4lut = gpu.upload_raw(&q4lut, &[q4lut.len()]).unwrap();
    let d_q4g32 = gpu.upload_raw(&q4f16g32, &[q4f16g32.len()]).unwrap();
    let d_q8 = gpu.upload_raw(&q8, &[q8.len()]).unwrap();
    let d_q4as8 = gpu.upload_raw(&q4as8, &[q4as8.len()]).unwrap();

    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    let n_warmup = 50;
    let n_iter = 500;

    // Warmup all
    for _ in 0..n_warmup {
        gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap();
        gpu.gemv_q4lut(&d_q4lut, &d_x, &d_y, m, k).unwrap();
        gpu.gemv_q4f16_g32(&d_q4g32, &d_x, &d_y, m, k).unwrap();
        gpu.gemv_q4wave(&d_q4g32, &d_x, &d_y, m, k).unwrap();
        gpu.gemv_q8_0(&d_q8, &d_x, &d_y, m, k).unwrap();
        gpu.gemv_q4as8(&d_q4as8, &d_x, &d_y, m, k).unwrap();
    }

    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();

    eprintln!("\n=== GEMV {m}x{k} Benchmark ({n_iter} iterations) ===\n");

    struct Result { name: &'static str, bpw: f64, us: f32, bw: f64 }
    let mut results = Vec::new();

    // Macro to reduce boilerplate
    macro_rules! bench {
        ($name:expr, $bpw:expr, $data_bytes:expr, $call:expr) => {{
            gpu.hip.event_record(&start, None).unwrap();
            for _ in 0..n_iter { $call.unwrap(); }
            gpu.hip.event_record(&stop, None).unwrap();
            gpu.hip.event_synchronize(&stop).unwrap();
            let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
            let us = ms * 1000.0 / n_iter as f32;
            let bytes = ($data_bytes + k * 4) as f64;
            let bw = bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
            results.push(Result { name: $name, bpw: $bpw, us, bw });
        }};
    }

    bench!("Q4_K",      0.5625, m*(k/256)*144, gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k));
    bench!("Q4_F16_G32", 0.625, m*(k/32)*20,  gpu.gemv_q4f16_g32(&d_q4g32, &d_x, &d_y, m, k));
    bench!("Q4_WAVE",   0.625,  m*(k/32)*20,  gpu.gemv_q4wave(&d_q4g32, &d_x, &d_y, m, k));
    bench!("Q4_LUT",    1.5,    m*(k/32)*48,  gpu.gemv_q4lut(&d_q4lut, &d_x, &d_y, m, k));
    bench!("Q4-as-Q8",  1.0625, m*(k/32)*34,  gpu.gemv_q4as8(&d_q4as8, &d_x, &d_y, m, k));
    bench!("Q8_0",      1.0625, m*(k/32)*34,  gpu.gemv_q8_0(&d_q8, &d_x, &d_y, m, k));

    // Sort by us (fastest first)
    results.sort_by(|a, b| a.us.partial_cmp(&b.us).unwrap());

    eprintln!("{:<12} {:>6} {:>8} {:>10} {:>7}", "Format", "B/w", "us/call", "GB/s", "% peak");
    eprintln!("{}", "-".repeat(50));
    for r in &results {
        eprintln!("{:<12} {:>6.4} {:>8.1} {:>10.1} {:>6.1}%",
            r.name, r.bpw, r.us, r.bw, r.bw / 448.0 * 100.0);
    }

    // Correctness check: all vs Q4_K reference
    eprintln!("\n=== Correctness (max abs error vs Q4_K) ===");
    let d_ref = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4k(&d_q4k, &d_x, &d_ref, m, k).unwrap();
    let y_ref = gpu.download_f32(&d_ref).unwrap();

    for (name, kernel_fn) in [
        ("Q4_LUT", Box::new(|gpu: &mut rdna_compute::Gpu| gpu.gemv_q4lut(&d_q4lut, &d_x, &d_y, m, k)) as Box<dyn FnMut(&mut rdna_compute::Gpu) -> hip_bridge::HipResult<()>>),
        ("Q4_WAVE", Box::new(|gpu: &mut rdna_compute::Gpu| gpu.gemv_q4wave(&d_q4g32, &d_x, &d_y, m, k))),
    ] {
        let mut f = kernel_fn;
        f(&mut gpu).unwrap();
        let y_test = gpu.download_f32(&d_y).unwrap();
        let max_err: f32 = y_ref.iter().zip(y_test.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  {name}: {max_err:.8}");
    }
}

fn convert_q4k_to_q4lut(q4k_data: &[u8], n_elements: usize) -> Vec<u8> {
    let q4k_block_bytes = 144;
    let q4k_block_elems = 256;
    let lut_block_bytes = 48;
    let nblocks = (n_elements + q4k_block_elems - 1) / q4k_block_elems;
    let mut output = vec![0u8; nblocks * 8 * lut_block_bytes];

    for b in 0..nblocks {
        let off = b * q4k_block_bytes;
        if off + q4k_block_bytes > q4k_data.len() { break; }

        let d = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([q4k_data[off], q4k_data[off + 1]]));
        let dmin = hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([q4k_data[off + 2], q4k_data[off + 3]]));

        let sc_data = &q4k_data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &q4k_data[off + 16..off + 16 + 128];

        for group in 0..4 {
            for sub in 0..2 {
                let sb_idx = group * 2 + sub;
                let eff_scale = d * scales[sb_idx] as f32;
                let eff_min = dmin * mins[sb_idx] as f32;

                let out_off = (b * 8 + sb_idx) * lut_block_bytes;

                for n in 0..16u8 {
                    let val = eff_scale * n as f32 - eff_min;
                    let f16_val = hipfire_runtime::llama::f32_to_f16(val);
                    let co = out_off + (n as usize) * 2;
                    output[co..co + 2].copy_from_slice(&f16_val.to_le_bytes());
                }

                for i in 0..16 {
                    let nib_lo = if sub == 0 { qdata[group * 32 + i] & 0xF } else { qdata[group * 32 + i] >> 4 };
                    let nib_hi = if sub == 0 { qdata[group * 32 + 16 + i] & 0xF } else { qdata[group * 32 + 16 + i] >> 4 };
                    output[out_off + 32 + i] = nib_lo | (nib_hi << 4);
                }
            }
        }
    }
    output
}

fn quantize_q8(f32_data: &[f32]) -> Vec<u8> {
    let mut output = Vec::new();
    for block in f32_data.chunks(32) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        output.extend_from_slice(&hipfire_runtime::llama::f32_to_f16(scale).to_le_bytes());
        for &v in block { output.push((v * inv_scale).round().max(-128.0).min(127.0) as i8 as u8); }
    }
    output
}

fn quantize_q4_as_q8(f32_data: &[f32]) -> Vec<u8> {
    // 4-bit precision (values -8..7) stored in Q8_0 format (1 byte per weight)
    let mut output = Vec::new();
    for block in f32_data.chunks(32) {
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 7.0; // 4-bit range: -8 to 7
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        output.extend_from_slice(&hipfire_runtime::llama::f32_to_f16(scale).to_le_bytes());
        for &v in block { output.push((v * inv_scale).round().max(-8.0).min(7.0) as i8 as u8); }
    }
    output
}
