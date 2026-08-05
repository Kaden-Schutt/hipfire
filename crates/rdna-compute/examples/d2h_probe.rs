fn main() {
    let mut gpu = rdna_compute::Gpu::init_with_device(0).expect("gpu");
    let n = 6usize;
    let t = gpu
        .alloc_tensor(&[n], rdna_compute::DType::F32)
        .expect("alloc");
    let big = gpu
        .alloc_tensor(&[4096 * 4096], rdna_compute::DType::F32)
        .expect("big");
    let mut buf = vec![0u8; n * 4];
    // Cold: D2H with an idle pipeline.
    for _ in 0..20 {
        let _ = gpu.hip.memcpy_dtoh(&mut buf, &t.buf);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..200 {
        gpu.hip.memcpy_dtoh(&mut buf, &t.buf).unwrap();
    }
    let idle = t0.elapsed().as_secs_f64() / 200.0 * 1e6;
    // Hot: same D2H, but with real GPU work queued in front of it.
    let t1 = std::time::Instant::now();
    for _ in 0..200 {
        gpu.hip.memset(&big.buf, 0, 4096 * 4096 * 4).unwrap();
        gpu.hip.memcpy_dtoh(&mut buf, &t.buf).unwrap();
    }
    let busy = t1.elapsed().as_secs_f64() / 200.0 * 1e6;
    println!("  24-byte D2H, idle pipeline : {:8.1} us", idle);
    println!("  24-byte D2H, work queued   : {:8.1} us", busy);
    println!("  => drain accounts for      : {:8.1} us", busy - idle);
}
