//! Micro-benchmark: host→device upload throughput from a warm host buffer.
//! Measures sync pageable memcpy (the loader's current path) and, if
//! available, repeated malloc+memcpy (what upload_raw actually does).
//!
//! Run: cargo run --release -p hipfire-runtime --example h2d_microbench

use rdna_compute::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let chunk: usize = 64 << 20; // 64 MiB
    let reps = 16;
    let data = vec![0x5au8; chunk];

    // Warmup
    for _ in 0..2 {
        let t = gpu.upload_raw(&data, &[chunk]).expect("upload");
        gpu.free_tensor(t).expect("free");
    }

    // malloc + sync pageable memcpy (upload_raw path)
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let t = gpu.upload_raw(&data, &[chunk]).expect("upload");
        gpu.free_tensor(t).expect("free");
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "MALLOC_PLUS_MEMCPY {:.2} GB/s ({} x {} MiB in {:.3}s)",
        (chunk * reps) as f64 / dt / 1e9,
        reps,
        chunk >> 20,
        dt
    );
}
