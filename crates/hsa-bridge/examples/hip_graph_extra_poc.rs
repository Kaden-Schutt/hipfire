// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validates that HIP graph capture works correctly when kernargs are
//! passed via the `extra` parameter (HIP_LAUNCH_PARAM_BUFFER_POINTER) with
//! a stable byte buffer, instead of the kernelParams array of stack pointers.
//!
//! Background: rdna-compute's existing dispatch path packs kernargs as
//! `&mut <stack-local>` pointers into a Vec<*mut c_void>. HIP graph capture
//! apparently stores those pointers (not the values they point to) and
//! dereferences them at replay time, by which point the stack is gone —
//! producing garbage outputs. This POC proves that the `extra` parameter
//! path captures cleanly because the kernarg bytes are baked into the graph
//! node at capture time.
//!
//! Run:
//!   cargo run --release -p hsa-bridge --example hip_graph_extra_poc

use hip_bridge::HipRuntime;
use libloading::Library;
use std::ffi::{c_void, CString};
use std::time::Instant;

// HIP launch param sentinels (from hip_runtime_api.h)
const HIP_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut c_void;
const HIP_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut c_void;
const HIP_LAUNCH_PARAM_END: *mut c_void = 3 as *mut c_void;

// FFI types we need beyond what hip-bridge exposes
type HipFunction = *mut c_void;
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipGraph = *mut c_void;
type HipGraphExec = *mut c_void;

// Direct dlopen so we can call hipModuleLaunchKernel with `extra`
struct DirectHip {
    _lib: Library,
    fn_module_launch_kernel: unsafe extern "C" fn(
        HipFunction, u32, u32, u32, u32, u32, u32, u32, HipStream,
        *mut *mut c_void, *mut *mut c_void,
    ) -> u32,
}

impl DirectHip {
    fn load() -> Self {
        let lib = unsafe { Library::new("libamdhip64.so").expect("dlopen") };
        let fn_module_launch_kernel = unsafe {
            let sym: libloading::Symbol<
                unsafe extern "C" fn(
                    HipFunction, u32, u32, u32, u32, u32, u32, u32, HipStream,
                    *mut *mut c_void, *mut *mut c_void,
                ) -> u32,
            > = lib.get(b"hipModuleLaunchKernel").unwrap();
            *sym.into_raw()
        };
        Self { _lib: lib, fn_module_launch_kernel }
    }
}

fn main() {
    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "unknown".into());
    eprintln!("[hip_graph_extra_poc] arch={arch}");

    // Compile a simple vector_add kernel for the local arch.
    let src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/hip_graph_va.hip", src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "-o",
            "/tmp/hip_graph_va.hsaco",
            "/tmp/hip_graph_va.hip",
        ])
        .output()
        .expect("hipcc");
    assert!(
        out.status.success(),
        "hipcc: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hsaco = std::fs::read("/tmp/hip_graph_va.hsaco").unwrap();
    let module = hip.module_load_data(&hsaco).unwrap();
    let kernel = hip.module_get_function(&module, "vector_add").unwrap();
    eprintln!("[poc] kernel loaded");

    // Allocate buffers (4096 floats — same as hipfire hidden_dim 9B).
    let n: u32 = 4096;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_buf = hip.malloc(nbytes).unwrap();
    let b_buf = hip.malloc(nbytes).unwrap();
    let c_buf = hip.malloc(nbytes).unwrap();
    hip.memcpy_htod(&a_buf, as_bytes(&a_data)).unwrap();
    hip.memcpy_htod(&b_buf, as_bytes(&b_data)).unwrap();

    // ─── Pack kernargs into a STABLE byte buffer (heap-allocated) ────────
    // Kernel signature: const float* a, const float* b, float* c, int n
    // ABI: pointers are 8 bytes, int is 4, with 8-byte alignment for ptrs.
    // Layout: [a_ptr 8B][b_ptr 8B][c_ptr 8B][n 4B][pad 4B] = 32 bytes total.
    let mut kernarg_buf: Vec<u8> = vec![0u8; 32];
    let a_ptr = a_buf.as_ptr() as u64;
    let b_ptr = b_buf.as_ptr() as u64;
    let c_ptr = c_buf.as_ptr() as u64;
    kernarg_buf[0..8].copy_from_slice(&a_ptr.to_le_bytes());
    kernarg_buf[8..16].copy_from_slice(&b_ptr.to_le_bytes());
    kernarg_buf[16..24].copy_from_slice(&c_ptr.to_le_bytes());
    kernarg_buf[24..28].copy_from_slice(&n.to_le_bytes());

    // Build the `extra` array. HIP_LAUNCH_PARAM_BUFFER_POINTER points to
    // our flat kernarg buffer; HIP copies the bytes into the kernarg
    // segment for the kernel.
    let mut kernarg_size: usize = kernarg_buf.len();
    let mut extra: Vec<*mut c_void> = vec![
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernarg_buf.as_mut_ptr() as *mut c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &mut kernarg_size as *mut _ as *mut c_void,
        HIP_LAUNCH_PARAM_END,
    ];

    // Direct dlopen for hipModuleLaunchKernel so we can pass `extra`.
    let direct = DirectHip::load();

    // Helper that launches via the `extra` path.
    let mut launch = |stream: HipStream| -> u32 {
        // Per the HIP API contract, kernelParams must be NULL when extra is used.
        unsafe {
            (direct.fn_module_launch_kernel)(
                kernel_handle(&kernel),
                ((n + 255) / 256) as u32, 1, 1,
                256, 1, 1,
                0,
                stream,
                std::ptr::null_mut(),
                extra.as_mut_ptr(),
            )
        }
    };

    // ─── 1. Reference: launch on null stream ─────────────────────────────
    eprintln!("\n--- Reference (null stream, extra-param launch) ---");
    let rc = launch(std::ptr::null_mut());
    assert_eq!(rc, 0, "launch failed: {rc}");
    hip.device_synchronize().unwrap();
    let mut c_raw = vec![0u8; nbytes];
    hip.memcpy_dtoh(&mut c_raw, &c_buf).unwrap();
    let reference: Vec<f32> =
        unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize).to_vec() };
    let bad = (0..n as usize)
        .filter(|&i| (reference[i] - (i as f32) * 3.0).abs() > 1e-3)
        .count();
    eprintln!("  reference {}/{} match", n as usize - bad, n);
    assert_eq!(bad, 0);

    // ─── 2. Capture into a graph ─────────────────────────────────────────
    eprintln!("\n--- Graph capture via extra-param launch ---");
    let stream = hip.stream_create().unwrap();

    // Reset c_buf to a known value so we can detect a successful replay.
    hip.memcpy_htod(&c_buf, &vec![0u8; nbytes]).unwrap();

    hip.stream_begin_capture(&stream, 0).expect("begin_capture");
    let rc = launch(stream_handle(&stream));
    assert_eq!(rc, 0, "graph-mode launch failed: {rc}");
    let graph = hip.stream_end_capture(&stream).expect("end_capture");
    let exec = hip.graph_instantiate(&graph).expect("graph_instantiate");
    eprintln!("  capture + instantiate OK");

    // Replay
    hip.memcpy_htod(&c_buf, &vec![0u8; nbytes]).unwrap();
    hip.graph_launch(&exec, &stream).expect("graph_launch");
    hip.stream_synchronize(&stream).unwrap();

    let mut c_graph = vec![0u8; nbytes];
    hip.memcpy_dtoh(&mut c_graph, &c_buf).unwrap();
    let graph_out: Vec<f32> =
        unsafe { std::slice::from_raw_parts(c_graph.as_ptr() as *const f32, n as usize).to_vec() };
    let bad = (0..n as usize)
        .filter(|&i| (graph_out[i] - reference[i]).abs() > 1e-3)
        .count();
    eprintln!("  graph replay vs reference: {}/{} match", n as usize - bad, n);
    if bad > 0 {
        for i in 0..8 {
            eprintln!(
                "    [{i}] graph={} ref={} delta={}",
                graph_out[i], reference[i], graph_out[i] - reference[i]
            );
        }
        std::process::exit(1);
    }

    // ─── 3. Single-kernel sequential vs graph (apples to apples) ─────────
    eprintln!("\n--- 1-kernel: sequential vs graph ---");
    let iters = 2000u32;

    for _ in 0..50 {
        let _ = launch(stream_handle(&stream));
    }
    hip.stream_synchronize(&stream).unwrap();
    let mut seq_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = launch(stream_handle(&stream));
        hip.stream_synchronize(&stream).unwrap();
        seq_lat.push(t.elapsed());
    }
    print_stats("Sequential 1-kernel + sync", &mut seq_lat);

    for _ in 0..50 {
        hip.graph_launch(&exec, &stream).unwrap();
        hip.stream_synchronize(&stream).unwrap();
    }
    let mut graph_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        hip.graph_launch(&exec, &stream).unwrap();
        hip.stream_synchronize(&stream).unwrap();
        graph_lat.push(t.elapsed());
    }
    print_stats("Graph 1-kernel replay + sync", &mut graph_lat);

    // ─── 4. Multi-kernel graph capture: N identical launches in one graph ─
    // This is the realistic test — what happens when we put many kernels
    // into a single graph. Per-kernel walk inside the graph should be much
    // smaller than the per-launch HIP overhead.
    eprintln!("\n--- N-kernel graph capture ---");
    for &n_kernels in &[10u32, 50, 100, 200] {
        let stream_n = hip.stream_create().unwrap();
        hip.stream_begin_capture(&stream_n, 0).unwrap();
        for _ in 0..n_kernels {
            let rc = launch(stream_handle(&stream_n));
            assert_eq!(rc, 0, "{n_kernels}-kernel capture failed at launch");
        }
        let graph_n = hip.stream_end_capture(&stream_n).unwrap();
        let exec_n = hip.graph_instantiate(&graph_n).unwrap();

        // Verify still produces correct output
        hip.memcpy_htod(&c_buf, &vec![0u8; nbytes]).unwrap();
        hip.graph_launch(&exec_n, &stream_n).unwrap();
        hip.stream_synchronize(&stream_n).unwrap();
        let mut c_check = vec![0u8; nbytes];
        hip.memcpy_dtoh(&mut c_check, &c_buf).unwrap();
        let cf: &[f32] = unsafe {
            std::slice::from_raw_parts(c_check.as_ptr() as *const f32, n as usize)
        };
        let bad = (0..n as usize)
            .filter(|&i| (cf[i] - (i as f32) * 3.0).abs() > 1e-3)
            .count();
        assert_eq!(bad, 0, "{n_kernels}-kernel graph output wrong");

        // Time replays
        for _ in 0..20 {
            hip.graph_launch(&exec_n, &stream_n).unwrap();
        }
        hip.stream_synchronize(&stream_n).unwrap();

        let replays = 200u32;
        let t = Instant::now();
        for _ in 0..replays {
            hip.graph_launch(&exec_n, &stream_n).unwrap();
        }
        hip.stream_synchronize(&stream_n).unwrap();
        let total = t.elapsed().as_secs_f64() * 1_000_000.0;
        let per_replay = total / replays as f64;
        let per_kernel = per_replay / n_kernels as f64;
        eprintln!(
            "[graph {n_kernels:3}-kernel × {replays} replays] total {total:7.1} µs   per-replay {per_replay:6.2} µs   per-kernel {per_kernel:.3} µs"
        );

        hip.graph_exec_destroy(exec_n).unwrap();
        hip.graph_destroy(graph_n).unwrap();
        hip.stream_destroy(stream_n).unwrap();
    }

    // ─── 5. MIXED-kernel graph: alternate between 3 distinct kernels ─
    // Real production has ~362 different kernels; the per-kernel cost in
    // the bandwidth profiler is ~10 µs not 3 µs because of kernel-switch
    // overhead between distinct kernels. The same-kernel test above can't
    // tell us whether hipGraph helps with that. Compile two more kernels
    // (vector_mul, vector_scale_add) and capture an alternating sequence.
    eprintln!("\n--- MIXED-kernel graph capture ---");
    // Each kernel READS c AND WRITES c — creating a true dependency chain.
    // This matches production where every non-GEMV kernel feeds the next.
    let mul_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = c[i] * b[i] + a[i];   // c depends on c
}
"#;
    let scale_add_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_scale_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = c[i] * 0.5f + b[i] * 0.25f + a[i] * 0.1f;   // c depends on c
}
"#;
    std::fs::write("/tmp/hip_graph_mul.hip", mul_src).unwrap();
    std::fs::write("/tmp/hip_graph_sa.hip", scale_add_src).unwrap();
    for (src, out_path) in [
        ("/tmp/hip_graph_mul.hip", "/tmp/hip_graph_mul.hsaco"),
        ("/tmp/hip_graph_sa.hip", "/tmp/hip_graph_sa.hsaco"),
    ] {
        let out = std::process::Command::new("hipcc")
            .args([
                "--genco",
                &format!("--offload-arch={arch}"),
                "-O3",
                "-o",
                out_path,
                src,
            ])
            .output()
            .expect("hipcc");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    let mul_hsaco = std::fs::read("/tmp/hip_graph_mul.hsaco").unwrap();
    let sa_hsaco = std::fs::read("/tmp/hip_graph_sa.hsaco").unwrap();
    let mul_module = hip.module_load_data(&mul_hsaco).unwrap();
    let sa_module = hip.module_load_data(&sa_hsaco).unwrap();
    let mul_kernel = hip.module_get_function(&mul_module, "vector_mul").unwrap();
    let sa_kernel = hip.module_get_function(&sa_module, "vector_scale_add").unwrap();

    // Build helper closures for each kernel — same kernarg layout (32 bytes).
    // We need separate kernarg buffers per kernel since the values differ.
    let mut ka_add: Vec<u8> = kernarg_buf.clone();
    let mut ka_mul: Vec<u8> = kernarg_buf.clone();
    let mut ka_sa: Vec<u8> = kernarg_buf.clone();
    // All three kernels use (a, b, c, n) — same layout, same args, no rebuild needed.

    let mut sz_add: usize = 32;
    let mut sz_mul: usize = 32;
    let mut sz_sa: usize = 32;
    let mut extra_add: Vec<*mut c_void> = vec![
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ka_add.as_mut_ptr() as *mut c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &mut sz_add as *mut _ as *mut c_void,
        HIP_LAUNCH_PARAM_END,
    ];
    let mut extra_mul: Vec<*mut c_void> = vec![
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ka_mul.as_mut_ptr() as *mut c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &mut sz_mul as *mut _ as *mut c_void,
        HIP_LAUNCH_PARAM_END,
    ];
    let mut extra_sa: Vec<*mut c_void> = vec![
        HIP_LAUNCH_PARAM_BUFFER_POINTER, ka_sa.as_mut_ptr() as *mut c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &mut sz_sa as *mut _ as *mut c_void,
        HIP_LAUNCH_PARAM_END,
    ];

    let direct2 = DirectHip::load();
    let mut launch_add = |s: HipStream| -> u32 {
        unsafe {
            (direct2.fn_module_launch_kernel)(
                kernel_handle(&kernel),
                ((n + 255) / 256) as u32, 1, 1, 256, 1, 1, 0, s,
                std::ptr::null_mut(), extra_add.as_mut_ptr(),
            )
        }
    };
    let mut launch_mul = |s: HipStream| -> u32 {
        unsafe {
            (direct2.fn_module_launch_kernel)(
                kernel_handle(&mul_kernel),
                ((n + 255) / 256) as u32, 1, 1, 256, 1, 1, 0, s,
                std::ptr::null_mut(), extra_mul.as_mut_ptr(),
            )
        }
    };
    let mut launch_sa = |s: HipStream| -> u32 {
        unsafe {
            (direct2.fn_module_launch_kernel)(
                kernel_handle(&sa_kernel),
                ((n + 255) / 256) as u32, 1, 1, 256, 1, 1, 0, s,
                std::ptr::null_mut(), extra_sa.as_mut_ptr(),
            )
        }
    };

    for &n_kernels in &[30u32, 90, 150, 300] {
        let stream_m = hip.stream_create().unwrap();

        // Sequential mixed timing FIRST (no graph) for comparison
        for _ in 0..30 {
            for i in 0..n_kernels {
                match i % 3 {
                    0 => { let _ = launch_add(stream_handle(&stream_m)); }
                    1 => { let _ = launch_mul(stream_handle(&stream_m)); }
                    _ => { let _ = launch_sa(stream_handle(&stream_m)); }
                }
            }
        }
        hip.stream_synchronize(&stream_m).unwrap();

        let replays = 100u32;
        let t = Instant::now();
        for _ in 0..replays {
            for i in 0..n_kernels {
                match i % 3 {
                    0 => { let _ = launch_add(stream_handle(&stream_m)); }
                    1 => { let _ = launch_mul(stream_handle(&stream_m)); }
                    _ => { let _ = launch_sa(stream_handle(&stream_m)); }
                }
            }
        }
        hip.stream_synchronize(&stream_m).unwrap();
        let total_seq = t.elapsed().as_secs_f64() * 1_000_000.0;
        let per_seq = total_seq / (replays * n_kernels) as f64;

        // Now capture the same sequence into a graph
        hip.stream_begin_capture(&stream_m, 0).unwrap();
        for i in 0..n_kernels {
            let rc = match i % 3 {
                0 => launch_add(stream_handle(&stream_m)),
                1 => launch_mul(stream_handle(&stream_m)),
                _ => launch_sa(stream_handle(&stream_m)),
            };
            assert_eq!(rc, 0, "mixed graph capture failed at kernel {i}");
        }
        let graph_m = hip.stream_end_capture(&stream_m).unwrap();
        let exec_m = hip.graph_instantiate(&graph_m).unwrap();

        for _ in 0..30 {
            hip.graph_launch(&exec_m, &stream_m).unwrap();
        }
        hip.stream_synchronize(&stream_m).unwrap();

        let t = Instant::now();
        for _ in 0..replays {
            hip.graph_launch(&exec_m, &stream_m).unwrap();
        }
        hip.stream_synchronize(&stream_m).unwrap();
        let total_graph = t.elapsed().as_secs_f64() * 1_000_000.0;
        let per_graph = total_graph / (replays * n_kernels) as f64;

        let speedup = per_seq / per_graph;
        eprintln!(
            "[mixed {n_kernels:3}-kernel × {replays} replays]  seq {per_seq:5.2} µs/k   graph {per_graph:5.2} µs/k   ({speedup:.2}x)"
        );

        hip.graph_exec_destroy(exec_m).unwrap();
        hip.graph_destroy(graph_m).unwrap();
        hip.stream_destroy(stream_m).unwrap();
    }

    let seq_med = median_us(&seq_lat);
    let graph_med = median_us(&graph_lat);
    eprintln!("\n=== Summary ===");
    eprintln!("  Single launch (sync each):  seq {seq_med:.2} µs   graph {graph_med:.2} µs");
    eprintln!("  See N-kernel + mixed-kernel tables above.");

    // Cleanup
    hip.graph_exec_destroy(exec).unwrap();
    hip.graph_destroy(graph).unwrap();
    hip.stream_destroy(stream).unwrap();
    eprintln!("\n=== POC PASSED ===");
}

fn kernel_handle(f: &hip_bridge::Function) -> *mut c_void {
    // hip_bridge::Function is a pub struct wrapping HipFunction. We can't access
    // the inner pointer through the public API, but we can transmute since
    // both are #[repr(transparent)] over a *mut c_void.
    unsafe { *(f as *const hip_bridge::Function as *const *mut c_void) }
}
fn stream_handle(s: &hip_bridge::Stream) -> *mut c_void {
    unsafe { *(s as *const hip_bridge::Stream as *const *mut c_void) }
}

fn print_stats(label: &str, lat: &mut Vec<std::time::Duration>) {
    lat.sort();
    let to_us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = to_us(lat[lat.len() / 2]);
    let mean = to_us(lat.iter().sum::<std::time::Duration>()) / lat.len() as f64;
    let p99 = to_us(lat[(lat.len() as f64 * 0.99) as usize]);
    let min = to_us(lat[0]);
    let max = to_us(*lat.last().unwrap());
    eprintln!("[{label}]");
    eprintln!("  median {median:7.2} µs   mean {mean:7.2}   p99 {p99:7.2}   min {min:7.2}   max {max:7.2}");
}

fn median_us(lat: &[std::time::Duration]) -> f64 {
    let mut sorted = lat.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2].as_secs_f64() * 1_000_000.0
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[allow(dead_code)]
fn _ensure_compile_check() {
    // Pull in CString to keep the import for the proof-of-concept structure
    // even though we don't currently use it (kept for future kernel-name lookups).
    let _ = CString::new("noop").unwrap();
}
