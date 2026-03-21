//! Benchmark HFQ4-G256 vs Q4K at multiple matrix sizes.
//! Tests whether HFQ4-G256's 2x occupancy advantage shows at small matrices.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    let peak_bw = 448.0f64; // RX 5700 XT theoretical peak GB/s

    let sizes: &[(usize, usize, &str)] = &[
        (1024, 1024, "Qwen3-0.6B attn"),
        (2048, 2048, "TinyLlama attn"),
        (4096, 4096, "Qwen3-8B attn"),
        (12288, 4096, "Qwen3-8B FFN"),
    ];

    eprintln!("{:<20} {:>6} {:>6}  {:>9} {:>8} {:>6}  {:>9} {:>8} {:>6}  {:>7}",
        "Name", "M", "K", "HFQ4 us", "GB/s", "%peak", "Q4K us", "GB/s", "%peak", "speedup");
    eprintln!("{}", "-".repeat(110));

    let n = 200;

    for &(m, k, name) in sizes {
        // HFQ4-G256: 136 bytes per 256 weights
        let groups = k / 256;
        let row_hfq4 = groups * 136;
        let total_hfq4 = m * row_hfq4;
        let d_hfq4 = gpu.upload_raw(&vec![0x55u8; total_hfq4], &[total_hfq4]).unwrap();

        // Q4K: 144 bytes per 256 weights
        let row_q4k = groups * 144;
        let total_q4k = m * row_q4k;
        let d_q4k = gpu.upload_raw(&vec![0x55u8; total_q4k], &[total_q4k]).unwrap();

        let d_x = gpu.upload_f32(&vec![0.01f32; k], &[k]).unwrap();
        let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

        // Warmup
        for _ in 0..10 {
            gpu.gemv_hfq4g256(&d_hfq4, &d_x, &d_y, m, k).unwrap();
            gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap();
        }

        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();

        // HFQ4-G256
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n { gpu.gemv_hfq4g256(&d_hfq4, &d_x, &d_y, m, k).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms_hfq4 = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let us_hfq4 = ms_hfq4 * 1000.0 / n as f32;
        let bw_hfq4 = (total_hfq4 + k * 4) as f64 * n as f64 / (ms_hfq4 as f64 / 1000.0) / 1e9;

        // Q4K
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n { gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms_q4k = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let us_q4k = ms_q4k * 1000.0 / n as f32;
        let bw_q4k = (total_q4k + k * 4) as f64 * n as f64 / (ms_q4k as f64 / 1000.0) / 1e9;

        let pct_hfq4 = bw_hfq4 / peak_bw * 100.0;
        let pct_q4k = bw_q4k / peak_bw * 100.0;

        eprintln!("{:<20} {:>6} {:>6}  {:>9.1} {:>8.1} {:>5.1}%  {:>9.1} {:>8.1} {:>5.1}%  {:>6.2}x",
            name, m, k, us_hfq4, bw_hfq4, pct_hfq4, us_q4k, bw_q4k, pct_q4k, us_q4k as f64 / us_hfq4 as f64);

        gpu.free_tensor(d_hfq4).unwrap();
        gpu.free_tensor(d_q4k).unwrap();
        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y).unwrap();
        gpu.hip.event_destroy(start).unwrap();
        gpu.hip.event_destroy(stop).unwrap();
    }
}
