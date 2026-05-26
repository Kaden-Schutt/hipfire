//! Diagnostic for the open MQ3-Lloyd-vs-MQ4-Lloyd multi-acc question.
//!
//! Loads one MQ4G256Lloyd weight tensor from a real .hfq file, generates
//! deterministic random x, runs the diagnostic K4 multi-accumulator kernel
//! (gemv_mq4g256_lloyd_multiacc_diag), and compares per-row against a CPU
//! reference. The CPU reference is the same per-row formula the slow generic
//! kernel computes, so divergence here = divergence of the multi-acc fast
//! variant from the slow generic.
//!
//! Usage:
//!   diag_mq4_lloyd_multiacc <model.hfq> <tensor_name> [--rows N]
//!
//! Reports: max-abs error, top-K rows by |err|, and for those rows the
//! codebook span + index histogram so we can spot what triggers the drift.

use rdna_compute::Gpu;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn f16_to_f32(bits: u16) -> f32 {
    hipfire_runtime::llama::f16_to_f32(bits)
}

/// CPU dequant of one MQ4-Lloyd-formatted row → flat f32 weights, FWHT
/// rotation BAKED IN (so it's the GPU-side view of the row, not the
/// pre-quantization values). 256 weights × groups_per_row.
fn cpu_dequant_row(row_bytes: &[u8], groups_per_row: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(groups_per_row * 256);
    for g in 0..groups_per_row {
        let off = g * 160;
        let mut cb = [0.0f32; 16];
        for k in 0..16 {
            let bits = u16::from_le_bytes([row_bytes[off + 2 * k], row_bytes[off + 2 * k + 1]]);
            cb[k] = f16_to_f32(bits);
        }
        // 128 nibble-pair bytes → 256 indices → 256 weights.
        for i in 0..128 {
            let byte_val = row_bytes[off + 32 + i] as usize;
            let lo = byte_val & 0xF;
            let hi = (byte_val >> 4) & 0xF;
            out.push(cb[lo]);
            out.push(cb[hi]);
        }
    }
    out
}

fn cpu_gemv(row_bytes: &[u8], groups_per_row: usize, x: &[f32]) -> f32 {
    // Same accumulation order as the slow generic GEMV: linear sum across
    // groups, with each group's 8-thread×8-weight inner loop unrolled.
    let mut acc = 0.0f32;
    for g in 0..groups_per_row {
        let off = g * 160;
        let mut cb = [0.0f32; 16];
        for k in 0..16 {
            let bits = u16::from_le_bytes([row_bytes[off + 2 * k], row_bytes[off + 2 * k + 1]]);
            cb[k] = f16_to_f32(bits);
        }
        for i in 0..128 {
            let byte_val = row_bytes[off + 32 + i] as usize;
            let lo = byte_val & 0xF;
            let hi = (byte_val >> 4) & 0xF;
            acc += cb[lo] * x[g * 256 + 2 * i];
            acc += cb[hi] * x[g * 256 + 2 * i + 1];
        }
    }
    acc
}

fn report_row(row_bytes: &[u8], groups_per_row: usize) -> (f32, f32, [u32; 16]) {
    let mut min_cb = f32::INFINITY;
    let mut max_cb = f32::NEG_INFINITY;
    let mut hist = [0u32; 16];
    for g in 0..groups_per_row {
        let off = g * 160;
        for k in 0..16 {
            let bits = u16::from_le_bytes([row_bytes[off + 2 * k], row_bytes[off + 2 * k + 1]]);
            let v = f16_to_f32(bits);
            if v < min_cb { min_cb = v; }
            if v > max_cb { max_cb = v; }
        }
        for i in 0..128 {
            let byte_val = row_bytes[off + 32 + i] as usize;
            hist[byte_val & 0xF] += 1;
            hist[(byte_val >> 4) & 0xF] += 1;
        }
    }
    (min_cb, max_cb, hist)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: diag_mq4_lloyd_multiacc <model.hfq> <tensor> [--rows N]");
    let tensor_name = args.next().expect("missing tensor name");
    let mut max_rows: usize = usize::MAX;
    while let Some(flag) = args.next() {
        if flag == "--rows" {
            max_rows = args.next().expect("--rows N").parse().expect("N");
        }
    }

    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let (info, bytes) = hfq.tensor_data(&tensor_name).expect("tensor not found");
    assert_eq!(info.quant_type, 21, "tensor {tensor_name} is qt={}, expected MQ4G256Lloyd (21)", info.quant_type);
    assert_eq!(info.shape.len(), 2, "expected 2D weight, got {:?}", info.shape);
    let m_full = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let m = m_full.min(max_rows);
    let groups_per_row = k / 256;
    let row_stride = groups_per_row * 160;
    eprintln!("Tensor: {tensor_name}  shape=[{}, {}]  testing {m} rows", m_full, k);
    eprintln!("Block: {groups_per_row} groups/row × 160 B = {row_stride} B/row");

    // Deterministic x (cheap PRNG).
    let mut state = 0xC0FFEEu32;
    let x: Vec<f32> = (0..k).map(|_| {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.5
    }).collect();

    // CPU reference.
    eprintln!("Computing CPU reference...");
    let t0 = std::time::Instant::now();
    let y_cpu: Vec<f32> = (0..m)
        .map(|row| {
            let row_bytes = &bytes[row * row_stride..(row + 1) * row_stride];
            cpu_gemv(row_bytes, groups_per_row, &x)
        })
        .collect();
    eprintln!("  CPU reference: {:.2}s", t0.elapsed().as_secs_f64());

    // GPU multi-acc.
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {} — running multi-acc diag kernel", gpu.arch);
    // Upload only the rows we're testing.
    let upload_bytes = &bytes[..m * row_stride];
    let d_a = gpu.upload_raw(upload_bytes, &[upload_bytes.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_mq4g256_lloyd_multiacc_diag(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    // Diff.
    let mut diffs: Vec<(usize, f32)> = (0..m)
        .map(|i| (i, (y_gpu[i] - y_cpu[i]).abs()))
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_abs = diffs[0].1;
    let total: f64 = (0..m).map(|i| (y_gpu[i] - y_cpu[i]).abs() as f64).sum();
    let mean_abs = (total / m as f64) as f32;

    println!("\n=== diff summary ===");
    println!("M tested:     {}", m);
    println!("Max abs:      {:.6e}", max_abs);
    println!("Mean abs:     {:.6e}", mean_abs);
    let big = diffs.iter().take_while(|(_, e)| *e > 1e-3).count();
    println!("Rows >1e-3:   {} / {}", big, m);
    let medium = diffs.iter().take_while(|(_, e)| *e > 1e-4).count();
    println!("Rows >1e-4:   {} / {}", medium, m);

    println!("\n=== top-10 worst rows ===");
    println!("  row    cpu          gpu          err          cb_min       cb_max       cb_span      hist (idx 0..15 counts)");
    for (rank, (row, err)) in diffs.iter().take(10).enumerate() {
        let row_bytes = &bytes[row * row_stride..(row + 1) * row_stride];
        let (mn, mx, hist) = report_row(row_bytes, groups_per_row);
        let span = mx - mn;
        let total_indices: u32 = hist.iter().sum();
        let hist_str: String = hist
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let pct = (*c as f64 * 100.0 / total_indices as f64) as u32;
                format!("{}:{}", i, pct)
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "  {:5}  {:11.4}  {:11.4}  {:11.4e}  {:11.4}  {:11.4}  {:11.4}  {}",
            row, y_cpu[*row], y_gpu[*row], err, mn, mx, span, hist_str,
        );
        if rank == 0 && *err < 1e-4 {
            println!("  ... (remaining errors below 1e-4, skipping)");
            break;
        }
    }
}
