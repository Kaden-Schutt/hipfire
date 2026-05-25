// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark: Redline dispatch overhead.
//! Measures per-dispatch latency, multi-dispatch throughput, startup time, and memory.

use redline::device::Device;
use redline::dispatch::{CommandBuffer, DispatchQueue, FastDispatch, KernargBuilder, Kernel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000u32);

    eprintln!("=== Redline Dispatch Benchmark ===");
    eprintln!("Iterations: {}\n", iterations);

    // --- Startup time ---
    let t_start = std::time::Instant::now();
    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    let startup_device = t_start.elapsed();

    // Auto-detect target arch from the open device
    let arch = dev.info.gfx_arch.clone();
    eprintln!("[bench] targeting {}", arch);

    // Compile vector_add with __launch_bounds__ (so no hidden arg overhead)
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_bench_va.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3",
               "-o", "/tmp/redline_bench_va.hsaco", "/tmp/redline_bench_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_bench_va.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").expect("kernel not found");
    let startup_total = t_start.elapsed();

    eprintln!("[startup] device+queue: {:.2}ms, total (incl compile): {:.2}ms",
        startup_device.as_secs_f64() * 1000.0, startup_total.as_secs_f64() * 1000.0);

    // Set up buffers (256 elements = 1KB)
    let n = 256u32;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();

    let mut ka = KernargBuilder::new(28);
    ka.write_ptr(0, a_buf.gpu_addr).write_ptr(8, b_buf.gpu_addr)
      .write_ptr(16, c_buf.gpu_addr).write_u32(24, n);

    // Warm up (first dispatch is always slower)
    for _ in 0..10 {
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
    }

    // --- Per-dispatch latency (includes submit + fence wait) ---
    let mut latencies = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t = std::time::Instant::now();
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
        latencies.push(t.elapsed());
    }

    latencies.sort();
    let to_us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = to_us(latencies[latencies.len() / 2]);
    let mean = to_us(latencies.iter().sum::<std::time::Duration>()) / latencies.len() as f64;
    let p99 = to_us(latencies[(latencies.len() as f64 * 0.99) as usize]);
    let min = to_us(latencies[0]);
    let max = to_us(*latencies.last().unwrap());

    eprintln!("\n[per-dispatch] {} iterations, vector_add 256 elements:", iterations);
    eprintln!("  median: {:.1} µs", median);
    eprintln!("  mean:   {:.1} µs", mean);
    eprintln!("  p99:    {:.1} µs", p99);
    eprintln!("  min:    {:.1} µs", min);
    eprintln!("  max:    {:.1} µs", max);

    // --- Multi-dispatch throughput (sequential submits) ---
    let batch = 200u32;
    let t_batch = std::time::Instant::now();
    for _ in 0..batch {
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
    }
    let batch_time = t_batch.elapsed();
    let per_kernel = batch_time.as_secs_f64() * 1_000_000.0 / batch as f64;
    eprintln!("\n[{}-dispatch sequential] total: {:.2}ms, per-kernel: {:.1} µs",
        batch, batch_time.as_secs_f64() * 1000.0, per_kernel);

    // --- FastDispatch (optimized path: persistent mappings, no per-dispatch alloc) ---
    eprintln!("\n--- FastDispatch (optimized ioctl) ---");
    let fd = FastDispatch::new(&dev, &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();

    // Warm up
    for _ in 0..10 {
        fd.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1], ka.as_bytes()).unwrap();
    }

    let mut fast_latencies = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t = std::time::Instant::now();
        fd.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1], ka.as_bytes()).unwrap();
        fast_latencies.push(t.elapsed());
    }
    fast_latencies.sort();
    let fast_median = to_us(fast_latencies[fast_latencies.len() / 2]);
    let fast_mean = to_us(fast_latencies.iter().sum::<std::time::Duration>()) / fast_latencies.len() as f64;
    let fast_p99 = to_us(fast_latencies[(fast_latencies.len() as f64 * 0.99) as usize]);
    let fast_min = to_us(fast_latencies[0]);

    eprintln!("[fast-dispatch] {} iterations:", iterations);
    eprintln!("  median: {:.1} µs", fast_median);
    eprintln!("  mean:   {:.1} µs", fast_mean);
    eprintln!("  p99:    {:.1} µs", fast_p99);
    eprintln!("  min:    {:.1} µs", fast_min);

    let t_fast_batch = std::time::Instant::now();
    for _ in 0..200 {
        fd.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1], ka.as_bytes()).unwrap();
    }
    let fast_batch = t_fast_batch.elapsed();
    eprintln!("[fast 200-dispatch] total: {:.2}ms, per-kernel: {:.1} µs",
        fast_batch.as_secs_f64() * 1000.0, fast_batch.as_secs_f64() * 1_000_000.0 / 200.0);

    fd.destroy(&dev);

    // --- Chained IB dispatch (single submit, 200 dispatches with barriers) ---
    eprintln!("\n--- Chained IB dispatch ---");
    {
        let fence_buf = dev.alloc_vram(4096).unwrap();
        dev.upload(&fence_buf, &vec![0u8; 4096]).unwrap();

        // Need separate kernarg slots for each dispatch (they all use same args but need distinct VAs)
        let chain_ka = dev.alloc_vram(64 * 1024).unwrap(); // 64KB for kernarg slots
        let chain_fd = FastDispatch::new(&dev, &[&module.code_buf, &a_buf, &b_buf, &c_buf, &fence_buf, &chain_ka]).unwrap();

        // Write same kernarg at 200 offsets (each 256 bytes apart)
        let mut ka_full = vec![0u8; 64 * 1024];
        for i in 0..200usize {
            let off = i * 256;
            ka_full[off..off + 8].copy_from_slice(&a_buf.gpu_addr.to_le_bytes());
            ka_full[off + 8..off + 16].copy_from_slice(&b_buf.gpu_addr.to_le_bytes());
            ka_full[off + 16..off + 24].copy_from_slice(&c_buf.gpu_addr.to_le_bytes());
            ka_full[off + 24..off + 28].copy_from_slice(&n.to_le_bytes());
            // Hidden args
            let h = off + 32;
            ka_full[h..h + 4].copy_from_slice(&1u32.to_le_bytes());
            ka_full[h + 4..h + 8].copy_from_slice(&1u32.to_le_bytes());
            ka_full[h + 8..h + 12].copy_from_slice(&1u32.to_le_bytes());
            ka_full[h + 12..h + 14].copy_from_slice(&256u16.to_le_bytes());
            ka_full[h + 14..h + 16].copy_from_slice(&1u16.to_le_bytes());
            ka_full[h + 16..h + 18].copy_from_slice(&1u16.to_le_bytes());
        }
        dev.upload(&chain_ka, &ka_full).unwrap();

        // Warm up
        let mut warmup_cb = CommandBuffer::new();
        warmup_cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], chain_ka.gpu_addr);
        chain_fd.submit_cmdbuf(&dev, &warmup_cb).unwrap();

        // Build chained IB with barriers — start with 10, scale up
        for chain_count in [10u32, 50, 100, 200] {
            let iters = 50;
            let mut chain_latencies = Vec::with_capacity(iters);
            let mut ok = true;
            for _ in 0..iters {
                dev.upload(&fence_buf, &vec![0u8; 4096]).unwrap();
                let mut cb = CommandBuffer::new();
                for i in 0..chain_count {
                    cb.dispatch(kernel, [1, 1, 1], [256, 1, 1], chain_ka.gpu_addr + (i as u64 * 256));
                    if i < chain_count - 1 {
                        cb.barrier(fence_buf.gpu_addr + (i as u64 * 8), i + 1); // 8-byte spacing
                    }
                }
                let t = std::time::Instant::now();
                match chain_fd.submit_cmdbuf(&dev, &cb) {
                    Ok(()) => chain_latencies.push(t.elapsed()),
                    Err(e) => { eprintln!("  chain {} FAILED at iter: {e}", chain_count); ok = false; break; }
                }
            }
            if ok && !chain_latencies.is_empty() {
                chain_latencies.sort();
                let med = chain_latencies[chain_latencies.len() / 2];
                let total_ms = med.as_secs_f64() * 1000.0;
                let per_kernel = med.as_secs_f64() * 1_000_000.0 / chain_count as f64;
                eprintln!("[chain {}-dispatch] median: {:.2}ms, per-kernel: {:.2} µs",
                    chain_count, total_ms, per_kernel);
                if chain_count == 200 {
                    println!("BENCH_REDLINE_CHAIN_TOTAL_MS={:.2}", total_ms);
                    println!("BENCH_REDLINE_CHAIN_PER_KERNEL_US={:.2}", per_kernel);
                }
            }
        }

        chain_fd.destroy(&dev);
    }

    // --- Memory overhead ---
    let rss = get_rss_kb();
    eprintln!("\n[memory] RSS: {} KB ({:.1} MB)", rss, rss as f64 / 1024.0);

    // --- Verify correctness ---
    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| (c[i] - (i as f32) * 3.0).abs() > 0.001).count();
    eprintln!("\n[verify] vector_add: {}/{} correct", n as usize - bad, n);

    // --- Print machine-readable results ---
    println!("BENCH_REDLINE_MEDIAN_US={:.1}", median);
    println!("BENCH_REDLINE_MEAN_US={:.1}", mean);
    println!("BENCH_REDLINE_P99_US={:.1}", p99);
    println!("BENCH_REDLINE_MIN_US={:.1}", min);
    println!("BENCH_REDLINE_MAX_US={:.1}", max);
    println!("BENCH_REDLINE_BATCH_TOTAL_MS={:.2}", batch_time.as_secs_f64() * 1000.0);
    println!("BENCH_REDLINE_BATCH_PER_KERNEL_US={:.1}", per_kernel);
    println!("BENCH_REDLINE_STARTUP_MS={:.2}", startup_device.as_secs_f64() * 1000.0);
    println!("BENCH_REDLINE_RSS_KB={}", rss);
    println!("BENCH_REDLINE_FAST_MEDIAN_US={:.1}", fast_median);
    println!("BENCH_REDLINE_FAST_MEAN_US={:.1}", fast_mean);
    println!("BENCH_REDLINE_FAST_P99_US={:.1}", fast_p99);
    println!("BENCH_REDLINE_FAST_MIN_US={:.1}", fast_min);

    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn get_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}
