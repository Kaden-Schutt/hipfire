//! Real-data correctness test for the gfx906 dp4a MMQ residual kernel.
//!
//! Loads dump files produced by HIPFIRE_MMQ_DUMP=N from a real model run,
//! runs the dp4a MMQ kernel against the same inputs, and compares to the
//! dumped FP16 wave64 reference output.
//!
//! Usage: cargo run --release -p rdna-compute --example test_gfx906_mmq_realdata \
//!        -- /tmp/mmq_dump_0
//!
//! Reads from <dir>:
//!   shape.txt   — "M K N" line
//!   a_raw.bin   — HFQ4 weights (raw bytes)
//!   x.f32       — FP32 activations [N × K]
//!   y_in.f32    — FP32 Y (input residual stream)
//!   y_out.f32   — FP32 Y after FP16 wave64 reference (Y_in + A·X^T)
//!
//! Ours: starts from y_in, runs gemm_hfq4g256_residual_mmq_gfx906, compares
//! to y_out.

use rdna_compute::{DType, Gpu};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/mmq_dump_0");

    let shape_str = std::fs::read_to_string(format!("{dir}/shape.txt"))
        .expect("read shape.txt");
    let dims: Vec<usize> = shape_str.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    assert_eq!(dims.len(), 3, "shape.txt must have 3 numbers");
    let m = dims[0];
    let k = dims[1];
    let n = dims[2];
    eprintln!("=== gfx906 MMQ real-data correctness test ===");
    eprintln!("dump dir: {dir}");
    eprintln!("M={m} K={k} N={n}");

    let weight_bytes = std::fs::read(format!("{dir}/a_raw.bin")).expect("read a_raw.bin");
    let expected_w_bytes = m * (k / 256) * 136;
    assert_eq!(weight_bytes.len(), expected_w_bytes,
        "weight file size mismatch: got {} expected {}", weight_bytes.len(), expected_w_bytes);

    let x_host = read_f32(&format!("{dir}/x.f32"), n * k);
    let y_in_host = read_f32(&format!("{dir}/y_in.f32"), n * m);
    let y_ref_host = read_f32(&format!("{dir}/y_out.f32"), n * m);

    eprintln!("x range:    [{:.4e}, {:.4e}]",
        x_host.iter().copied().fold(f32::INFINITY, f32::min),
        x_host.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("y_in range: [{:.4e}, {:.4e}]",
        y_in_host.iter().copied().fold(f32::INFINITY, f32::min),
        y_in_host.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("y_ref range:[{:.4e}, {:.4e}]",
        y_ref_host.iter().copied().fold(f32::INFINITY, f32::min),
        y_ref_host.iter().copied().fold(f32::NEG_INFINITY, f32::max));

    // Spot-check a few weight scale/zp values
    eprintln!("\n--- Weight scale/zp samples (row 0..3, group 0) ---");
    for row in 0..4.min(m) {
        let gp = row * (k / 256) * 136;
        let scale = f32::from_le_bytes([weight_bytes[gp], weight_bytes[gp+1],
            weight_bytes[gp+2], weight_bytes[gp+3]]);
        let zp = f32::from_le_bytes([weight_bytes[gp+4], weight_bytes[gp+5],
            weight_bytes[gp+6], weight_bytes[gp+7]]);
        let nibbles_first_byte = weight_bytes[gp+8];
        eprintln!("  row {row}: scale={scale:.4e}  zp={zp:.4e}  byte0=0x{nibbles_first_byte:02x}");
    }

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("\narch: {}", gpu.arch);
    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    let a_raw = gpu.upload_raw(&weight_bytes, &[weight_bytes.len()]).expect("upload weights");
    let x_tensor = gpu.upload_f32(&x_host, &[n * k]).expect("upload x");

    // Run MMQ kernel starting from y_in.
    let y_mmq = gpu.upload_f32(&y_in_host, &[n * m]).expect("upload y_in for mmq");
    eprintln!("\n--- Running gemm_hfq4g256_residual_mmq_gfx906 ---");
    gpu.gemm_hfq4g256_residual_mmq_gfx906(&a_raw, &x_tensor, &y_mmq, m, k, n)
        .expect("mmq gfx906 launch");
    gpu.hip.device_synchronize().expect("sync after mmq");
    let y_mmq_host = gpu.download_f32(&y_mmq).expect("download mmq output");

    // Compare MMQ output to dumped FP16 reference.
    eprintln!("\n--- Comparing dp4a MMQ vs dumped FP16 wave64 reference ---");
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut worst_idx = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for i in 0..n * m {
        let r = y_ref_host[i];
        let q = y_mmq_host[i];
        let err = (r - q).abs();
        if err > max_abs_err {
            max_abs_err = err;
            worst_idx = i;
            worst_pair = (r, q);
        }
        let rel = if r.abs() > 1e-6 { err / r.abs() } else { 0.0 };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (r as f64).powi(2);
    }
    let rms_err = (sum_sq_err / (n * m) as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / (n * m) as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);

    eprintln!("max_abs_err  = {:.6e}", max_abs_err);
    eprintln!("max_rel_err  = {:.4}%", max_rel_err * 100.0);
    eprintln!("rms_err      = {:.6e}", rms_err);
    eprintln!("rms_ref      = {:.6e}", rms_ref);
    eprintln!("NRMSE        = {:.4}%", nrmse * 100.0);

    let worst_col = worst_idx / m;
    let worst_row = worst_idx % m;
    eprintln!("worst at (col,row)=({worst_col},{worst_row}): ref={:.4e} mmq={:.4e}",
        worst_pair.0, worst_pair.1);

    // Histogram of per-element errors to find hot spots
    eprintln!("\n--- Error histogram (abs error) ---");
    let mut bins = [0usize; 8];
    let edges = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, f32::INFINITY];
    for i in 0..n * m {
        let e = (y_ref_host[i] - y_mmq_host[i]).abs();
        for (b, &edge) in edges.iter().enumerate() {
            if e < edge { bins[b] += 1; break; }
        }
    }
    let total = (n * m) as f64;
    for (b, &edge) in edges.iter().enumerate() {
        let count = bins[b];
        let pct = count as f64 / total * 100.0;
        let lo = if b == 0 { 0.0 } else { edges[b-1] };
        eprintln!("  [{:.0e}, {:.0e}): {:>10} ({:5.2}%)", lo, edge, count, pct);
    }

    // Per-row max abs error
    eprintln!("\n--- Per-row max abs error (top 20 worst rows) ---");
    let mut row_max = vec![0f32; m];
    for i in 0..n*m {
        let row = i % m;
        let e = (y_ref_host[i] - y_mmq_host[i]).abs();
        if e > row_max[row] { row_max[row] = e; }
    }
    let mut rows_sorted: Vec<(usize, f32)> = row_max.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    rows_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(row, err)) in rows_sorted.iter().take(20).enumerate() {
        eprintln!("  #{rank}: row={row} max_err={err:.4e}");
    }

    // Show a few cells with absolute error >0.01
    eprintln!("\n--- Top 10 worst-error cells (col, row, ref, mmq, abs_err) ---");
    let mut errs: Vec<(usize, f32)> = (0..n*m)
        .map(|i| (i, (y_ref_host[i] - y_mmq_host[i]).abs()))
        .collect();
    errs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(i, e)) in errs.iter().take(10).enumerate() {
        let col = i / m;
        let row = i % m;
        eprintln!("  #{rank}: col={col} row={row} ref={:.4e} mmq={:.4e} err={:.4e}",
            y_ref_host[i], y_mmq_host[i], e);
    }

    eprintln!("\n--- First 16 rows, col=0 ---");
    for row in 0..16.min(m) {
        let r = y_ref_host[row];
        let q = y_mmq_host[row];
        let yi = y_in_host[row];
        eprintln!("  row {row}: y_in={yi:.4e}  ref={r:.4e}  mmq={q:.4e}  diff={:.4e}",
            (r - q).abs());
    }

    if nrmse > 1e-2 {
        eprintln!("\nFAIL: NRMSE exceeds 1% — kernel produces wrong output on real data");
        std::process::exit(1);
    } else {
        eprintln!("\nPASS");
    }
}

fn read_f32(path: &str, n: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
    assert_eq!(bytes.len(), n * 4, "size mismatch on {path}: got {} expected {}",
        bytes.len(), n * 4);
    let mut out = vec![0f32; n];
    for i in 0..n {
        out[i] = f32::from_le_bytes([bytes[4*i], bytes[4*i+1], bytes[4*i+2], bytes[4*i+3]]);
    }
    out
}

#[allow(dead_code)]
fn _path_check(p: &str) -> bool { Path::new(p).exists() }
