// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Diff two raw f32 binary files (host-order). Used to compare tensor
//! captures from the single-gpu and hetero MTP paths to isolate
//! numerical-drift sources in RDNA2 kernels.
//!
//! Reports: count of elements compared, max abs diff, mean abs diff,
//! RMS diff, top-10 diverging indices with both values, and (for vector
//! inputs) the argmax of each — useful for "did the predicted token
//! change?" questions.
//!
//! Run: ./target/release/examples/diff_f32_bin <a.bin> <b.bin>

use std::path::PathBuf;

fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", path.display());
        std::process::exit(2);
    });
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} not multiple of 4 bytes",
        path.display()
    );
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    {
        let dst: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
        dst.copy_from_slice(&bytes);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: diff_f32_bin <a.bin> <b.bin>");
        std::process::exit(2);
    }
    let a = read_f32_bin(&PathBuf::from(&args[1]));
    let b = read_f32_bin(&PathBuf::from(&args[2]));

    println!("file A: {} ({} f32)", args[1], a.len());
    println!("file B: {} ({} f32)", args[2], b.len());
    if a.len() != b.len() {
        eprintln!("LENGTH MISMATCH: a={} b={}", a.len(), b.len());
        std::process::exit(3);
    }

    let n = a.len();
    let mut max_abs_diff = 0f32;
    let mut max_idx = 0usize;
    let mut sum_abs = 0f64;
    let mut sum_sq = 0f64;
    let mut nan_count = 0usize;
    let mut bit_equal = 0usize;

    for i in 0..n {
        if a[i].is_nan() || b[i].is_nan() {
            nan_count += 1;
            continue;
        }
        if a[i].to_bits() == b[i].to_bits() {
            bit_equal += 1;
        }
        let d = (a[i] - b[i]).abs();
        sum_abs += d as f64;
        sum_sq += (d as f64) * (d as f64);
        if d > max_abs_diff {
            max_abs_diff = d;
            max_idx = i;
        }
    }

    let mean_abs = sum_abs / n as f64;
    let rms = (sum_sq / n as f64).sqrt();

    println!();
    println!("elements:        {n}");
    println!(
        "bit-equal:       {bit_equal} ({:.2}%)",
        100.0 * bit_equal as f64 / n as f64
    );
    println!("nan-skipped:     {nan_count}");
    println!("max abs diff:    {max_abs_diff:.6e}  at idx {max_idx}");
    println!(
        "                 A[{max_idx}] = {:.6e}  B[{max_idx}] = {:.6e}",
        a[max_idx], b[max_idx]
    );
    println!("mean abs diff:   {mean_abs:.6e}");
    println!("RMS diff:        {rms:.6e}");

    // Top-K diverging indices.
    let mut diffs: Vec<(usize, f32)> = (0..n).map(|i| (i, (a[i] - b[i]).abs())).collect();
    diffs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\ntop-10 diverging indices:");
    for (i, d) in diffs.iter().take(10) {
        println!(
            "  idx={i:>8} diff={d:.6e}  A={:.6e}  B={:.6e}",
            a[*i], b[*i]
        );
    }

    // Argmax (useful for the compressed-logits case — does the picked
    // token change?). For prev_hidden / t_mtp_out this is just the
    // largest-magnitude index; for logits it's the predicted token.
    let mut a_argmax = 0usize;
    let mut a_max = f32::NEG_INFINITY;
    let mut b_argmax = 0usize;
    let mut b_max = f32::NEG_INFINITY;
    for i in 0..n {
        if a[i] > a_max {
            a_max = a[i];
            a_argmax = i;
        }
        if b[i] > b_max {
            b_max = b[i];
            b_argmax = i;
        }
    }
    println!("\nargmax: A={a_argmax} (val={a_max:.6e})  B={b_argmax} (val={b_max:.6e})");
    if a_argmax != b_argmax {
        println!("** ARGMAX DIFFERS **");
        println!("   A[B's argmax={b_argmax}] = {:.6e}", a[b_argmax]);
        println!("   B[A's argmax={a_argmax}] = {:.6e}", b[a_argmax]);
    } else {
        println!("argmax MATCHES");
    }
}
