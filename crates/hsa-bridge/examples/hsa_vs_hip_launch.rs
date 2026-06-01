// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 2 benchmark: per-dispatch latency for vector_add through HIP vs HSA.
//!
//! Compiles the same kernel for the current GPU's gfx arch, then runs the
//! same number of iterations through both dispatch paths and reports
//! median/mean/p99 latency, plus the speedup ratio.
//!
//! HIP path: hipModuleLaunchKernel + hipDeviceSynchronize per iteration.
//! HSA path: build AQL packet → ring doorbell → wait_lt(completion_signal).
//!
//! Run with:
//!   cargo run --release -p hsa-bridge --example hsa_vs_hip_launch -- 5000

use hip_bridge::HipRuntime;
use hsa_bridge::{
    build_dispatch_packet, dispatch_packet_header, publish_dispatch_packet, HsaExecutable,
    HsaRuntime, HsaSignal,
};
use std::time::Instant;

const KERNEL_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

const HSA_WAIT_TIMEOUT_NS: u64 = 5_000_000_000;

fn wait_for_completion(signal: &HsaSignal, context: &str) {
    let observed = signal.wait_lt_active(1, HSA_WAIT_TIMEOUT_NS);
    assert!(
        observed < 1,
        "{context}: HSA completion signal timed out with value {observed}"
    );
}

fn main() {
    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);

    eprintln!("=== HSA vs HIP per-dispatch latency benchmark ===");
    eprintln!("iterations: {iters}");

    // ─── 1. Init HSA ──────────────────────────────────────────────────────
    let hsa = HsaRuntime::load().expect("HsaRuntime::load");
    let agent = hsa.find_gpu_agent(None).expect("find_gpu_agent");
    let arch = agent.name().expect("agent name");
    eprintln!("[hsa] agent: {arch}");

    // ─── 2. Compile kernel for this arch ─────────────────────────────────
    let hsaco_path = "/tmp/hsa_vs_hip_va.hsaco";
    std::fs::write("/tmp/hsa_vs_hip_va.hip", KERNEL_SRC).unwrap();
    let out = std::process::Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "-o",
            hsaco_path,
            "/tmp/hsa_vs_hip_va.hip",
        ])
        .output()
        .expect("hipcc");
    assert!(
        out.status.success(),
        "hipcc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hsaco = std::fs::read(hsaco_path).expect("read hsaco");
    eprintln!("[hipcc] compiled {} bytes for {arch}", hsaco.len());

    // ─── 3. Set up HIP path ──────────────────────────────────────────────
    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device");

    let n: u32 = 256;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let hip_a = hip.malloc(nbytes).unwrap();
    let hip_b = hip.malloc(nbytes).unwrap();
    let hip_c = hip.malloc(nbytes).unwrap();
    hip.memcpy_htod(&hip_a, as_bytes(&a_data)).unwrap();
    hip.memcpy_htod(&hip_b, as_bytes(&b_data)).unwrap();

    let hip_module = hip.module_load_data(&hsaco).unwrap();
    let hip_kernel = hip.module_get_function(&hip_module, "vector_add").unwrap();

    // Pack args for HIP launch (pointers passed by-reference)
    let mut hip_a_ptr = hip_a.as_ptr();
    let mut hip_b_ptr = hip_b.as_ptr();
    let mut hip_c_ptr = hip_c.as_ptr();
    let mut n_arg: u32 = n;
    let mut hip_params: Vec<*mut std::ffi::c_void> = vec![
        &mut hip_a_ptr as *mut _ as *mut std::ffi::c_void,
        &mut hip_b_ptr as *mut _ as *mut std::ffi::c_void,
        &mut hip_c_ptr as *mut _ as *mut std::ffi::c_void,
        &mut n_arg as *mut _ as *mut std::ffi::c_void,
    ];

    // ─── 4. Set up HSA path ──────────────────────────────────────────────
    let queue = agent.create_queue(1024).expect("create_queue");
    eprintln!(
        "[hsa] queue size={} doorbell=0x{:x}",
        queue.size(),
        queue.doorbell()
    );

    // Kernarg pool lives on the CPU agent (host-pinned, GPU-readable).
    let cpu_agent = hsa.find_cpu_agent().expect("find_cpu_agent");
    let kernarg_pool = cpu_agent.find_kernarg_pool().expect("kernarg pool");
    let device_pool = agent
        .find_coarse_grained_pool()
        .expect("device pool (coarse)");

    let hsa_a = device_pool.allocate(nbytes).unwrap();
    let hsa_b = device_pool.allocate(nbytes).unwrap();
    let hsa_c = device_pool.allocate(nbytes).unwrap();
    // Coarse-grained device memory is GPU-only by default; allow CPU writes too.
    device_pool
        .allow_access(&[&cpu_agent, &agent], hsa_a)
        .unwrap();
    device_pool
        .allow_access(&[&cpu_agent, &agent], hsa_b)
        .unwrap();
    device_pool
        .allow_access(&[&cpu_agent, &agent], hsa_c)
        .unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(a_data.as_ptr() as *const u8, hsa_a, nbytes);
        std::ptr::copy_nonoverlapping(b_data.as_ptr() as *const u8, hsa_b, nbytes);
    }

    let mut exec = HsaExecutable::from_code_object(&agent, &hsaco).expect("load exec");
    exec.freeze().expect("freeze exec");
    let kernel = exec.kernel(&agent, "vector_add").expect("get kernel");
    eprintln!(
        "[hsa] kernel: object=0x{:x}, kernarg={}B, group={}B, private={}B",
        kernel.kernel_object,
        kernel.kernarg_size,
        kernel.group_segment_size,
        kernel.private_segment_size
    );

    // Allocate kernarg buffer for HSA dispatches. The kernarg pool is CPU-side
    // (KERNARG_INIT flag, fine-grained system memory). Explicitly allow the GPU
    // to read it — without this we get a GPU page fault on the kernarg load.
    let kernarg = kernarg_pool.allocate(kernel.kernarg_size as usize).unwrap();
    kernarg_pool
        .allow_access(&[&cpu_agent, &agent], kernarg)
        .unwrap();
    eprintln!(
        "[hsa] addrs: a=0x{:x} b=0x{:x} c=0x{:x} kernarg=0x{:x}",
        hsa_a as usize, hsa_b as usize, hsa_c as usize, kernarg as usize,
    );
    // Zero the entire kernarg buffer first so unused hidden-arg slots aren't garbage.
    unsafe {
        std::ptr::write_bytes(kernarg, 0, kernel.kernarg_size as usize);
        // Explicit args: [a_ptr (8), b_ptr (8), c_ptr (8), n (4), pad (4), ...]
        let p = kernarg as *mut u64;
        p.add(0).write(hsa_a as u64);
        p.add(1).write(hsa_b as u64);
        p.add(2).write(hsa_c as u64);
        let n_ptr = (kernarg as *mut u8).add(24) as *mut u32;
        n_ptr.write(n);
        // Hidden args at offset 32 (clang code object v5):
        // block_count_x/y/z (u32 × 3), then group_size_x/y/z (u16 × 3)
        let h = (kernarg as *mut u8).add(32);
        let groups = (n + 255) / 256;
        (h.add(0) as *mut u32).write(groups);
        (h.add(4) as *mut u32).write(1);
        (h.add(8) as *mut u32).write(1);
        (h.add(12) as *mut u16).write(256);
        (h.add(14) as *mut u16).write(1);
        (h.add(16) as *mut u16).write(1);
    }

    let signal = HsaSignal::create(&hsa, 1).expect("signal create");
    let header = dispatch_packet_header();

    // ─── 5. Warm-up ──────────────────────────────────────────────────────
    let groups = (n + 255) / 256;
    for _ in 0..50 {
        // HIP warm-up
        unsafe {
            hip.launch_kernel(
                &hip_kernel,
                [groups, 1, 1],
                [256, 1, 1],
                0,
                None,
                &mut hip_params,
            )
            .unwrap();
        }
        hip.device_synchronize().unwrap();

        // HSA warm-up
        signal.store_relaxed(1);
        let idx = queue.load_write_index_relaxed();
        let slot = queue.packet_slot(idx);
        unsafe {
            build_dispatch_packet(
                slot,
                &kernel,
                [groups, 1, 1],
                [256, 1, 1],
                kernarg,
                signal.raw_handle(),
            );
            publish_dispatch_packet(slot, header);
        }
        queue.store_write_index_release(idx + 1);
        queue.ring_doorbell(idx);
        wait_for_completion(&signal, "HSA warm-up");
    }

    // ─── 6. Verify both paths produce the same result ────────────────────
    let mut hip_out = vec![0u8; nbytes];
    let mut hsa_out = vec![0u8; nbytes];
    hip.memcpy_dtoh(&mut hip_out, &hip_c).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(hsa_c, hsa_out.as_mut_ptr(), nbytes);
    }
    let hip_floats: &[f32] =
        unsafe { std::slice::from_raw_parts(hip_out.as_ptr() as *const f32, n as usize) };
    let hsa_floats: &[f32] =
        unsafe { std::slice::from_raw_parts(hsa_out.as_ptr() as *const f32, n as usize) };
    let hip_bad = (0..n as usize)
        .filter(|&i| (hip_floats[i] - (i as f32) * 3.0).abs() > 1e-3)
        .count();
    let hsa_bad = (0..n as usize)
        .filter(|&i| (hsa_floats[i] - (i as f32) * 3.0).abs() > 1e-3)
        .count();
    eprintln!(
        "[verify] HIP {}/{} correct, HSA {}/{} correct",
        n as usize - hip_bad,
        n,
        n as usize - hsa_bad,
        n
    );
    if hip_bad != 0 || hsa_bad != 0 {
        eprintln!("  HIP first 4: {:?}", &hip_floats[..4]);
        eprintln!("  HSA first 4: {:?}", &hsa_floats[..4]);
        std::process::exit(1);
    }

    // ─── 7. Time HIP dispatches ──────────────────────────────────────────
    let mut hip_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        unsafe {
            hip.launch_kernel(
                &hip_kernel,
                [groups, 1, 1],
                [256, 1, 1],
                0,
                None,
                &mut hip_params,
            )
            .unwrap();
        }
        hip.device_synchronize().unwrap();
        hip_lat.push(t.elapsed());
    }

    // ─── 8. Time HSA dispatches ──────────────────────────────────────────
    let mut hsa_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        signal.store_relaxed(1);
        let idx = queue.load_write_index_relaxed();
        let slot = queue.packet_slot(idx);
        unsafe {
            build_dispatch_packet(
                slot,
                &kernel,
                [groups, 1, 1],
                [256, 1, 1],
                kernarg,
                signal.raw_handle(),
            );
            publish_dispatch_packet(slot, header);
        }
        queue.store_write_index_release(idx + 1);
        queue.ring_doorbell(idx);
        wait_for_completion(&signal, "HSA latency iteration");
        hsa_lat.push(t.elapsed());
    }

    // ─── 8b. HIP burst: N launches back-to-back, sync once at the end ───
    // This is the real apples-to-apples comparison vs HSA burst — the
    // engine doesn't sync per launch in production either.
    let mut hip_burst_results: Vec<(u32, f64)> = Vec::new();
    for &burst in &[10u32, 50, 100, 200] {
        let mut burst_lat: Vec<std::time::Duration> = Vec::with_capacity(500);
        for _ in 0..500 {
            let t = Instant::now();
            for _ in 0..burst {
                unsafe {
                    hip.launch_kernel(
                        &hip_kernel,
                        [groups, 1, 1],
                        [256, 1, 1],
                        0,
                        None,
                        &mut hip_params,
                    )
                    .unwrap();
                }
            }
            hip.device_synchronize().unwrap();
            burst_lat.push(t.elapsed());
        }
        burst_lat.sort();
        let total_us = burst_lat[burst_lat.len() / 2].as_secs_f64() * 1_000_000.0;
        let per_dispatch = total_us / burst as f64;
        eprintln!(
            "[HIP burst {burst:3}-launch] median total {total_us:7.1} µs  →  {per_dispatch:6.2} µs/dispatch",
        );
        hip_burst_results.push((burst, per_dispatch));
    }

    // ─── 9. Burst dispatch (HSA only): N packets → 1 doorbell → 1 wait ──
    // This mirrors how the engine actually wants to use HSA: chain many
    // small kernels in flight, only sync at the end of a step. The
    // per-dispatch cost should drop to ~packet build + atomic header store.
    //
    // Optimization: only the LAST packet carries a completion signal.
    // Intermediate packets use signal handle 0 (no signal), so the GPU
    // doesn't pay an atomic decrement per dispatch. The queue is FIFO,
    // so when the last packet completes, all earlier ones are done too.
    let burst_sizes = [10u32, 50, 100, 200];
    let burst_iters = 500u32;
    let mut burst_results: Vec<(u32, f64)> = Vec::new();
    for &burst in &burst_sizes {
        if burst > queue.size() {
            continue;
        }
        let mut burst_lat: Vec<std::time::Duration> = Vec::with_capacity(burst_iters as usize);
        for _ in 0..burst_iters {
            signal.store_relaxed(1); // wait_lt(1) → wait until kernel done
            let t = Instant::now();
            let base_idx = queue.load_write_index_relaxed();
            for i in 0..burst {
                let idx = base_idx + i as u64;
                let slot = queue.packet_slot(idx);
                let completion = if i == burst - 1 {
                    signal.raw_handle()
                } else {
                    0
                };
                unsafe {
                    build_dispatch_packet(
                        slot,
                        &kernel,
                        [groups, 1, 1],
                        [256, 1, 1],
                        kernarg,
                        completion,
                    );
                    publish_dispatch_packet(slot, header);
                }
            }
            queue.store_write_index_release(base_idx + burst as u64);
            queue.ring_doorbell(base_idx + burst as u64 - 1);
            wait_for_completion(&signal, "HSA burst iteration");
            burst_lat.push(t.elapsed());
        }
        burst_lat.sort();
        let total_us = burst_lat[burst_lat.len() / 2].as_secs_f64() * 1_000_000.0;
        let per_dispatch = total_us / burst as f64;
        eprintln!(
            "[burst {burst:3}-dispatch] median total {total_us:7.1} µs  →  {per_dispatch:6.2} µs/dispatch",
        );
        burst_results.push((burst, per_dispatch));
    }

    // ─── 10. Report ──────────────────────────────────────────────────────
    print_stats("HIP (hipModuleLaunchKernel + sync)", &mut hip_lat);
    print_stats("HSA (single dispatch + wait)", &mut hsa_lat);

    let hip_med = median_us(&hip_lat);
    let hsa_med = median_us(&hsa_lat);
    eprintln!("\n[summary]");
    eprintln!("  HIP single dispatch:  {hip_med:6.2} µs");
    eprintln!(
        "  HSA single dispatch:  {hsa_med:6.2} µs   ({:.2}x vs HIP)",
        hip_med / hsa_med
    );
    if let Some(&(_, hip_best)) = hip_burst_results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        eprintln!(
            "  HIP burst (best):     {hip_best:6.2} µs   ({:.2}x vs HIP single)",
            hip_med / hip_best
        );
        if let Some(&(_, hsa_best)) = burst_results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            eprintln!(
                "  HSA burst (best):     {hsa_best:6.2} µs   ({:.2}x vs HIP burst)",
                hip_best / hsa_best
            );
        }
    }
}

fn print_stats(label: &str, lat: &mut Vec<std::time::Duration>) {
    lat.sort();
    let to_us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = to_us(lat[lat.len() / 2]);
    let mean = to_us(lat.iter().sum::<std::time::Duration>()) / lat.len() as f64;
    let p99 = to_us(lat[(lat.len() as f64 * 0.99) as usize]);
    let min = to_us(lat[0]);
    let max = to_us(*lat.last().unwrap());
    eprintln!("\n[{label}]");
    eprintln!("  median: {median:.2} µs");
    eprintln!("  mean:   {mean:.2} µs");
    eprintln!("  p99:    {p99:.2} µs");
    eprintln!("  min:    {min:.2} µs");
    eprintln!("  max:    {max:.2} µs");
}

fn median_us(lat: &[std::time::Duration]) -> f64 {
    let mut sorted = lat.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2].as_secs_f64() * 1_000_000.0
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
