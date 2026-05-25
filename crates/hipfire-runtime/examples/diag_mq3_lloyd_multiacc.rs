//! Sibling of diag_mq4_lloyd_multiacc.rs — same diagnostic but for MQ3-Lloyd
//! (qt=20, 112 B/group, 8 entries, 3-bit cross-byte indices). Runs the
//! production gemv_mq3g256_lloyd via the dispatcher (which on gfx1100 +
//! gfx1151 routes to the K4 multi-acc fast kernel) and compares per-row
//! against a CPU reference. The CPU reference matches the slow generic
//! kernel byte-equal, so this measures the magnitude of multi-acc-vs-slow
//! drift on REAL Qwen3.5-9B MQ3-Lloyd weights — directly comparable to
//! the MQ4-Lloyd diag's output for the same model and tensor name.

use rdna_compute::Gpu;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn f16_to_f32(bits: u16) -> f32 {
    hipfire_runtime::llama::f16_to_f32(bits)
}

fn cpu_gemv_mq3(row_bytes: &[u8], groups_per_row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..groups_per_row {
        let off = g * 112;
        let mut cb = [0.0f32; 8];
        for k in 0..8 {
            let bits = u16::from_le_bytes([row_bytes[off + 2 * k], row_bytes[off + 2 * k + 1]]);
            cb[k] = f16_to_f32(bits);
        }
        for chunk in 0..32 {
            let bo = off + 16 + chunk * 3;
            let pk = (row_bytes[bo] as u32)
                | ((row_bytes[bo + 1] as u32) << 8)
                | ((row_bytes[bo + 2] as u32) << 16);
            let base = g * 256 + chunk * 8;
            acc += cb[((pk      ) & 7) as usize] * x[base];
            acc += cb[((pk >>  3) & 7) as usize] * x[base + 1];
            acc += cb[((pk >>  6) & 7) as usize] * x[base + 2];
            acc += cb[((pk >>  9) & 7) as usize] * x[base + 3];
            acc += cb[((pk >> 12) & 7) as usize] * x[base + 4];
            acc += cb[((pk >> 15) & 7) as usize] * x[base + 5];
            acc += cb[((pk >> 18) & 7) as usize] * x[base + 6];
            acc += cb[((pk >> 21) & 7) as usize] * x[base + 7];
        }
    }
    acc
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: diag_mq3_lloyd_multiacc <model.hfq> <tensor> [--rows N]");
    let tensor_name = args.next().expect("missing tensor name");
    let mut max_rows: usize = usize::MAX;
    while let Some(flag) = args.next() {
        if flag == "--rows" {
            max_rows = args.next().expect("--rows N").parse().expect("N");
        }
    }

    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let (info, bytes) = hfq.tensor_data(&tensor_name).expect("tensor not found");
    assert_eq!(info.quant_type, 20, "tensor {tensor_name} is qt={}, expected MQ3G256Lloyd (20)", info.quant_type);
    assert_eq!(info.shape.len(), 2, "expected 2D weight, got {:?}", info.shape);
    let m_full = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let m = m_full.min(max_rows);
    let groups_per_row = k / 256;
    let row_stride = groups_per_row * 112;
    eprintln!("MQ3-Lloyd Tensor: {tensor_name}  shape=[{}, {}]  testing {m} rows", m_full, k);

    let mut state = 0xC0FFEEu32;
    let x: Vec<f32> = (0..k).map(|_| {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.5
    }).collect();

    let t0 = std::time::Instant::now();
    let y_cpu: Vec<f32> = (0..m)
        .map(|row| {
            let row_bytes = &bytes[row * row_stride..(row + 1) * row_stride];
            cpu_gemv_mq3(row_bytes, groups_per_row, &x)
        })
        .collect();
    eprintln!("  CPU reference: {:.2}s", t0.elapsed().as_secs_f64());

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {} — running gemv_mq3g256_lloyd (multi-acc fast on gfx1100/1151, slow elsewhere)", gpu.arch);
    let upload_bytes = &bytes[..m * row_stride];
    let d_a = gpu.upload_raw(upload_bytes, &[upload_bytes.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_mq3g256_lloyd(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let mut diffs: Vec<(usize, f32)> = (0..m)
        .map(|i| (i, (y_gpu[i] - y_cpu[i]).abs()))
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_abs = diffs[0].1;
    let total: f64 = (0..m).map(|i| (y_gpu[i] - y_cpu[i]).abs() as f64).sum();
    let mean_abs = (total / m as f64) as f32;
    let big = diffs.iter().take_while(|(_, e)| *e > 1e-3).count();
    let medium = diffs.iter().take_while(|(_, e)| *e > 1e-4).count();
    let tiny = diffs.iter().take_while(|(_, e)| *e > 1e-5).count();

    println!("\n=== diff summary (MQ3-Lloyd) ===");
    println!("M tested:     {}", m);
    println!("Max abs:      {:.6e}", max_abs);
    println!("Mean abs:     {:.6e}", mean_abs);
    println!("Rows >1e-3:   {} / {}", big, m);
    println!("Rows >1e-4:   {} / {}", medium, m);
    println!("Rows >1e-5:   {} / {}", tiny, m);
}
