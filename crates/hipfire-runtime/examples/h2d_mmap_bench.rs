//! Micro-benchmark: H2D upload from (a) warm heap buffer vs (b) mmap'd
//! file-backed pages (the zero-copy loader path). Decides where the
//! remaining load-time wall lives.
//!
//! Run: cargo run --release -p hipfire-runtime --example h2d_mmap_bench -- <file>

use rdna_compute::Gpu;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: h2d_mmap_bench <file>");
    let mut gpu = Gpu::init().expect("gpu init");

    let file = std::fs::File::open(&path).expect("open");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let size = mmap.len();
    println!("file: {} MiB", size >> 20);

    let chunk: usize = 64 << 20;
    let reps_per_src = size / chunk;

    // One reusable destination; memcpy_htod overwrites it per iteration.
    let dst = {
        let warm = vec![0u8; chunk];
        gpu.upload_raw(&warm, &[chunk]).expect("alloc dst")
    };

    let reps_per_src = (size / chunk).min(128);
    let heap = vec![0x5au8; chunk];
    let t0 = std::time::Instant::now();
    let mut n = 0;
    while n + chunk <= size {
        gpu.memcpy_htod_auto(&dst.buf, &heap).expect("up");
        n += chunk;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "HEAP_SRC  {:.2} GB/s ({} x {} MiB in {:.3}s)",
        (chunk * reps_per_src) as f64 / dt / 1e9,
        reps_per_src,
        chunk >> 20,
        dt
    );

    // (b0) explicit madvise populates PTEs
    unsafe {
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            size,
            libc::MADV_POPULATE_READ,
        );
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            size,
            libc::MADV_HUGEPAGE,
        );
    }

    // (b) mmap source, sequential sweep (cold PTEs first pass)
    let t0 = std::time::Instant::now();
    let mut n = 0;
    while n + chunk <= size {
        gpu.memcpy_htod_auto(&dst.buf, &mmap[n..n + chunk])
            .expect("up");
        n += chunk;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "MMAP_SRC(cold-ptes) {:.2} GB/s",
        (chunk * reps_per_src) as f64 / dt / 1e9
    );

    // (c) mmap source, second pass (PTEs warm)
    let t0 = std::time::Instant::now();
    let mut n = 0;
    while n + chunk <= size {
        gpu.memcpy_htod_auto(&dst.buf, &mmap[n..n + chunk])
            .expect("up");
        n += chunk;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "MMAP_SRC(warm-ptes) {:.2} GB/s",
        (chunk * reps_per_src) as f64 / dt / 1e9
    );
}
