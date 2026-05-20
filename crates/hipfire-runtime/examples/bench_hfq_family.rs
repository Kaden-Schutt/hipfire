// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark the complete HFQ family: Q2, Q3, Q4, Q6, Q8 at G256.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    let peak_bw = 448.0f64;

    let sizes: &[(usize, usize, &str)] = &[
        (4096, 4096, "attn 4096²"),
        (12288, 4096, "FFN 12288×4096"),
    ];

    // Format: (name, bits, block_bytes, group_size, bench_fn_index)
    struct Fmt { name: &'static str, id: u32, block_bytes: usize, group_size: usize }
    let formats = [
        Fmt { name: "HFQ2-G256",  id: 2,  block_bytes: 72,  group_size: 256 },
        Fmt { name: "HFQ3-G256",  id: 3,  block_bytes: 104, group_size: 256 },
        Fmt { name: "HFQ4-G256",  id: 4,  block_bytes: 136, group_size: 256 },
        Fmt { name: "HFQ4-G512",  id: 45, block_bytes: 264, group_size: 512 },
        Fmt { name: "HFQ4-G1024", id: 46, block_bytes: 520, group_size: 1024 },
        Fmt { name: "HFQ6-G256",  id: 6,  block_bytes: 200, group_size: 256 },
        Fmt { name: "HFQ8-G256",  id: 8,  block_bytes: 264, group_size: 256 },
        Fmt { name: "Q4_K(ref)",  id: 0,  block_bytes: 144, group_size: 256 },
    ];

    let n = 200;

    eprintln!("{:<12} {:>14} {:>9} {:>8} {:>6}  {:>9} {:>8} {:>6}",
        "Format", "B/w", "attn µs", "GB/s", "%peak", "FFN µs", "GB/s", "%peak");
    eprintln!("{}", "-".repeat(85));

    for fmt in &formats {
        let bpw = if fmt.id == 0 { 0.5625 } else { fmt.block_bytes as f64 / fmt.group_size as f64 };
        let mut results = Vec::new();

        for &(m, k, _) in sizes {
            if k % fmt.group_size != 0 { results.push((0.0f32, 0.0f64)); continue; }
            let groups = k / fmt.group_size;
            let row_bytes = groups * fmt.block_bytes;
            let total = m * row_bytes;
            let d_a = gpu.upload_raw(&vec![0x55u8; total], &[total]).unwrap();
            let d_x = gpu.upload_f32(&vec![0.01f32; k], &[k]).unwrap();
            let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

            let run = |gpu: &mut rdna_compute::Gpu| match fmt.id {
                2 => gpu.gemv_hfq2g256(&d_a, &d_x, &d_y, m, k).unwrap(),
                3 => gpu.gemv_hfq3g256(&d_a, &d_x, &d_y, m, k).unwrap(),
                4 => gpu.gemv_hfq4g256(&d_a, &d_x, &d_y, m, k).unwrap(),
                45 => gpu.gemv_hfq4g512(&d_a, &d_x, &d_y, m, k).unwrap(),
                46 => gpu.gemv_hfq4g1024(&d_a, &d_x, &d_y, m, k).unwrap(),
                6 => gpu.gemv_hfq6g256(&d_a, &d_x, &d_y, m, k).unwrap(),
                8 => gpu.gemv_hfq8g256(&d_a, &d_x, &d_y, m, k).unwrap(),
                0 => gpu.gemv_q4k(&d_a, &d_x, &d_y, m, k).unwrap(),
                _ => unreachable!(),
            };

            for _ in 0..10 { run(&mut gpu); }

            let start = gpu.hip.event_create().unwrap();
            let stop = gpu.hip.event_create().unwrap();
            gpu.hip.event_record(&start, None).unwrap();
            for _ in 0..n { run(&mut gpu); }
            gpu.hip.event_record(&stop, None).unwrap();
            gpu.hip.event_synchronize(&stop).unwrap();
            let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
            let us = ms * 1000.0 / n as f32;
            let bw = (total + k * 4) as f64 * n as f64 / (ms as f64 / 1000.0) / 1e9;

            results.push((us, bw));

            gpu.free_tensor(d_a).unwrap();
            gpu.free_tensor(d_x).unwrap();
            gpu.free_tensor(d_y).unwrap();
            gpu.hip.event_destroy(start).unwrap();
            gpu.hip.event_destroy(stop).unwrap();
        }

        eprintln!("{:<12} {:>10.4} B/w  {:>8.1} {:>8.1} {:>5.1}%  {:>8.1} {:>8.1} {:>5.1}%",
            fmt.name, bpw,
            results[0].0, results[0].1, results[0].1 / peak_bw * 100.0,
            results[1].0, results[1].1, results[1].1 / peak_bw * 100.0);
    }
}
