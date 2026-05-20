// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validates hipGraph replay savings on a real hipfire kernel (not a toy
//! vector_add). Uses gemv_hfq4g256 at Qwen3.5 0.8B-realistic sizes to
//! measure whether hipGraph forward-pass integration is worth the
//! dispatch refactor cost on gfx1013 / gfx1010 (BC-250).
//!
//! ══════════════════════════════════════════════════════════════════════
//! RESULT: NO-GO on forward-pass hipGraph integration for BC-250.
//! ══════════════════════════════════════════════════════════════════════
//!
//! Measured on BC-250 (gfx1013 running as gfx1010), session 2 (2026-04-10):
//!
//!   === Individual kernel shapes ===
//!     wide   M=1024 K=1024   SEQ  9.30 µs   GRAPH  9.36 µs   Δ  -0.1%  (noise)
//!     narrow M=1024 K=1024   SEQ  8.33 µs   GRAPH  7.03 µs   Δ +15.6%  (small win)
//!     wide   M=2816 K=1024   SEQ  5.10 µs   GRAPH 17.39 µs   Δ -241%   (big loss)
//!
//!   === Realistic Qwen3.5 0.8B mixed shape (186 GEMVs/step) ===
//!     SEQ    1888.6 µs/step    GRAPH 1854.8 µs/step    Δ +1.8%
//!     Projected forward-pass Δ: +0.9% (235.8 → 238.1 tok/s)
//!
//! Why hipGraph loses on big BW-bound kernels:
//!
//! HIP burst mode on gfx1013 aggressively pipelines back-to-back kernels
//! of the same shape — 138 × M=2816 launches take 701 µs wall time,
//! which is SHORTER than 138 × single-kernel compute time (786 µs).
//! Kernel N+1 starts reading weights while N is still writing. The RDNA1
//! command processor + HIP runtime gets this overlap for free in burst
//! mode — no explicit concurrency management needed.
//!
//! hipGraph replay does NOT achieve the same pipelining. The same 138
//! launches take 2400 µs — 3.4× slower. That's ~12 µs of fixed per-node
//! overhead that burst mode hides via overlap. The graph replay path
//! apparently serializes more aggressively than stream launches on
//! gfx1013's command processor.
//!
//! On small launch-overhead-limited kernels (narrow 1024×1024), burst
//! is bottlenecked by launch dispatch cost (8.33 µs/call, ~3 µs of
//! which is actual compute). Graph's fixed per-node overhead is
//! slightly LOWER than burst's per-launch cost in this regime → graph
//! wins modestly (+15.6%).
//!
//! The real forward pass is dominated by big BW-bound GEMVs
//! (w_gate / w_up / w_down = 72 large launches/step). Small-kernel wins
//! (1-2 µs each × ~100 small launches) cancel against big-kernel losses
//! (12 µs each × 72 big launches) → net +1.8% on GEMV time, +0.9% on
//! forward-pass tok/s. Far below the dispatch refactor threshold.
//!
//! The session 1 projection of +20% from hipGraph was based on the
//! earlier hip_graph_extra_poc.rs measurement on a toy vector_add
//! kernel, which is pure launch-overhead. Real hipfire kernels are
//! fundamentally different: bigger per-launch work, BW-bound, tightly
//! pipelined already. The burst path on gfx1013 is essentially optimal
//! for this workload shape.
//!
//! Any future revisit should test on a different generation (gfx1100
//! already tested negative for different reasons, gfx1030/gfx1200 untested).
//! For BC-250 specifically, move on to other levers.
//!
//! KEY TECHNICAL NOTE on kernargs stability during graph capture:
//!
//! Every pointer passed into hipModuleLaunchKernel during capture must
//! live past the point where HIP records it — otherwise graph replay
//! dereferences stack-gone memory and crashes with
//! HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION. The `extras` array AND the
//! `kernarg_size` slot AND the flat kernarg byte buffer ALL need to
//! be heap-allocated (Box<>) or owned in a Vec that lives until replay.
//! Stack-local `extras` inside a closure will build cleanly, pass
//! sequential launches fine, then crash under graph capture. See the
//! Box<[u8; 32]> / Box<usize> / Box<[*mut c_void; 5]> triple used here.
//!
//! Run:
//!   cargo run --release -p hsa-bridge --example hip_graph_gemv_poc

use hip_bridge::HipRuntime;
use libloading::Library;
use std::ffi::c_void;
use std::time::Instant;

const HIP_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut c_void;
const HIP_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut c_void;
const HIP_LAUNCH_PARAM_END: *mut c_void = 3 as *mut c_void;

type HipFunction = *mut c_void;
type HipStream = *mut c_void;

struct DirectHip {
    _lib: Library,
    fn_module_launch_kernel: unsafe extern "C" fn(
        HipFunction, u32, u32, u32, u32, u32, u32, u32, HipStream,
        *mut *mut c_void, *mut *mut c_void,
    ) -> u32,
}

impl DirectHip {
    fn load() -> Self {
        let lib = unsafe { Library::new("libamdhip64.so").expect("dlopen libamdhip64.so") };
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

// Cast a hip_bridge Function to the raw HipFunction pointer. Relies on
// the bridge's Function/Stream being newtypes over *mut c_void.
fn function_handle(f: &hip_bridge::Function) -> HipFunction {
    unsafe { *(f as *const hip_bridge::Function as *const *mut c_void) }
}
fn stream_handle(s: &hip_bridge::Stream) -> HipStream {
    unsafe { *(s as *const hip_bridge::Stream as *const *mut c_void) }
}

fn print_stats(label: &str, samples: &mut [std::time::Duration]) {
    samples.sort();
    let n = samples.len();
    let us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = us(samples[n / 2]);
    let p99 = us(samples[(n * 99) / 100]);
    let min = us(samples[0]);
    let max = us(samples[n - 1]);
    let mean: f64 = samples.iter().map(|d| us(*d)).sum::<f64>() / n as f64;
    eprintln!(
        "  {label:36}  median {median:7.2} µs   mean {mean:7.2}   p99 {p99:7.2}   min {min:7.2}   max {max:7.2}"
    );
}

fn bench_kernel(
    hip: &HipRuntime,
    direct: &DirectHip,
    name: &str,
    src: &str,
    m: u32,
    k: u32,
    block: [u32; 3],
    grid: [u32; 3],
    n_launches: u32,
    arch: &str,
) {
    eprintln!("\n========================================");
    eprintln!("  KERNEL: {name}   M={m}  K={k}");
    eprintln!("  block=[{}, {}, {}]  grid=[{}, {}, {}]  launches/batch={n_launches}",
        block[0], block[1], block[2], grid[0], grid[1], grid[2]);
    eprintln!("========================================");

    // Compile fresh
    let src_path = format!("/tmp/hip_graph_gemv_{name}.hip");
    let hsaco_path = format!("/tmp/hip_graph_gemv_{name}.hsaco");
    std::fs::write(&src_path, src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "-I",
            "kernels/src",
            "-o",
            &hsaco_path,
            &src_path,
        ])
        .output()
        .expect("hipcc");
    if !out.status.success() {
        eprintln!("hipcc stderr: {}", String::from_utf8_lossy(&out.stderr));
        panic!("compile failed for {name}");
    }
    let hsaco = std::fs::read(&hsaco_path).unwrap();
    let module = hip.module_load_data(&hsaco).unwrap();
    let kernel = hip.module_get_function(&module, name).unwrap();

    // Allocate realistic buffers
    let groups_per_row = k / 256;
    let row_bytes = (groups_per_row * 136) as usize;
    let a_bytes = (m as usize) * row_bytes;
    let x_bytes = (k as usize) * 4;
    let y_bytes = (m as usize) * 4;

    let a_buf = hip.malloc(a_bytes).expect("malloc A");
    let x_buf = hip.malloc(x_bytes).expect("malloc x");
    let y_buf = hip.malloc(y_bytes).expect("malloc y");

    // Fill with non-zero junk so the kernel sees realistic memory traffic
    // patterns (compiler can't dead-code eliminate).
    let a_junk: Vec<u8> = (0..a_bytes).map(|i| ((i & 0x7f) | 0x10) as u8).collect();
    let x_junk: Vec<f32> = (0..k as usize).map(|i| (i as f32) * 0.01).collect();
    hip.memcpy_htod(&a_buf, &a_junk).unwrap();
    hip.memcpy_htod(&x_buf, unsafe {
        std::slice::from_raw_parts(x_junk.as_ptr() as *const u8, x_bytes)
    }).unwrap();
    hip.memcpy_htod(&y_buf, &vec![0u8; y_bytes]).unwrap();

    eprintln!("  A={:.2} MiB, x={:.1} KiB, y={:.1} KiB",
        a_bytes as f64 / (1024.0*1024.0),
        x_bytes as f64 / 1024.0,
        y_bytes as f64 / 1024.0);
    eprintln!("  A streaming BW per launch: {:.2} MiB (close to per-GEMV weight traffic)",
        a_bytes as f64 / (1024.0*1024.0));

    // Pack kernargs: gemv_hfq4g256(const char* A, const float* x, float* y, int M, int K)
    // 3 × 8B pointer + 2 × 4B int = 32 bytes. Pointers are 8-byte aligned.
    // Layout: [A 0..8][x 8..16][y 16..24][M 24..28][K 28..32]
    let mut kernarg_buf: Vec<u8> = vec![0u8; 32];
    let a_ptr = a_buf.as_ptr() as u64;
    let x_ptr = x_buf.as_ptr() as u64;
    let y_ptr = y_buf.as_ptr() as u64;
    kernarg_buf[0..8].copy_from_slice(&a_ptr.to_le_bytes());
    kernarg_buf[8..16].copy_from_slice(&x_ptr.to_le_bytes());
    kernarg_buf[16..24].copy_from_slice(&y_ptr.to_le_bytes());
    kernarg_buf[24..28].copy_from_slice(&(m as i32).to_le_bytes());
    kernarg_buf[28..32].copy_from_slice(&(k as i32).to_le_bytes());

    let mut kernarg_size: usize = kernarg_buf.len();
    let mut extra: Vec<*mut c_void> = vec![
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernarg_buf.as_mut_ptr() as *mut c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &mut kernarg_size as *mut _ as *mut c_void,
        HIP_LAUNCH_PARAM_END,
    ];

    let kfunc = function_handle(&kernel);
    let mut launch = |stream: HipStream| -> u32 {
        unsafe {
            (direct.fn_module_launch_kernel)(
                kfunc,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0,
                stream,
                std::ptr::null_mut(),
                extra.as_mut_ptr(),
            )
        }
    };

    // Warmup (JIT, driver state, cache)
    let stream = hip.stream_create().unwrap();
    for _ in 0..50 {
        let _ = launch(stream_handle(&stream));
    }
    hip.stream_synchronize(&stream).unwrap();

    // ─── Sequential: N launches, sync at end (HIP burst baseline) ─────
    let iters = 50u32;
    let mut seq_per_launch: Vec<std::time::Duration> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..n_launches {
            let rc = launch(stream_handle(&stream));
            assert_eq!(rc, 0);
        }
        hip.stream_synchronize(&stream).unwrap();
        let per_call = t.elapsed() / n_launches;
        seq_per_launch.push(per_call);
    }
    print_stats(&format!("[SEQ   {n_launches} launches × {iters}]"), &mut seq_per_launch);

    // Single-launch sync-per-call latency (worst-case)
    let mut single_lat: Vec<std::time::Duration> = Vec::with_capacity(500);
    for _ in 0..500 {
        let t = Instant::now();
        let _ = launch(stream_handle(&stream));
        hip.stream_synchronize(&stream).unwrap();
        single_lat.push(t.elapsed());
    }
    print_stats("[SEQ     1 launch sync-each]", &mut single_lat);

    // ─── Graph capture: N launches, one graph, replay R times ─────────
    let graph_stream = hip.stream_create().unwrap();
    hip.stream_begin_capture(&graph_stream, 0).unwrap();
    for _ in 0..n_launches {
        let rc = launch(stream_handle(&graph_stream));
        assert_eq!(rc, 0, "graph capture launch failed");
    }
    let graph = hip.stream_end_capture(&graph_stream).unwrap();
    let exec = hip.graph_instantiate(&graph).unwrap();
    eprintln!("  [GRAPH] captured {n_launches}-launch graph");

    // Warmup replays
    for _ in 0..20 {
        hip.graph_launch(&exec, &graph_stream).unwrap();
    }
    hip.stream_synchronize(&graph_stream).unwrap();

    // Timed replays: do R replays with single sync at the end
    let replays = 50u32;
    let mut graph_per_launch: Vec<std::time::Duration> = Vec::with_capacity(replays as usize);
    for _ in 0..replays {
        let t = Instant::now();
        hip.graph_launch(&exec, &graph_stream).unwrap();
        hip.stream_synchronize(&graph_stream).unwrap();
        let per_call = t.elapsed() / n_launches;
        graph_per_launch.push(per_call);
    }
    print_stats(&format!("[GRAPH {n_launches} launches × {replays}]"), &mut graph_per_launch);

    // Summary
    graph_per_launch.sort();
    seq_per_launch.sort();
    let seq_median_us = seq_per_launch[iters as usize / 2].as_secs_f64() * 1_000_000.0;
    let graph_median_us = graph_per_launch[replays as usize / 2].as_secs_f64() * 1_000_000.0;
    let saving = seq_median_us - graph_median_us;
    let pct = saving / seq_median_us * 100.0;
    let speedup = seq_median_us / graph_median_us;
    eprintln!(
        "  ==> per-launch SEQ {seq_median_us:.3} µs  GRAPH {graph_median_us:.3} µs  ΔSAVING {saving:.3} µs ({pct:+.1}%)   speedup {speedup:.2}×"
    );

    // Project to real forward-pass impact
    let gemv_per_step: u32 = 138 + 48; // gemv + residual on 0.8B
    let total_save_per_step_us = saving * gemv_per_step as f64;
    let step_ms_current = 4.62; // measured R=2 0.8B
    let new_step_ms = step_ms_current - total_save_per_step_us / 1000.0;
    let old_tps = 1000.0 / step_ms_current;
    let new_tps = 1000.0 / new_step_ms;
    let tps_pct = (new_tps - old_tps) / old_tps * 100.0;
    eprintln!(
        "  projection: {gemv_per_step} gemv/step × {saving:.3} µs = {total_save_per_step_us:.0} µs/step"
    );
    eprintln!(
        "              step {step_ms_current:.2} ms → {new_step_ms:.2} ms  ({old_tps:.1} tok/s → {new_tps:.1} tok/s, {tps_pct:+.1}%)"
    );

    hip.graph_exec_destroy(exec).unwrap();
    hip.graph_destroy(graph).unwrap();
    hip.stream_destroy(graph_stream).unwrap();
    hip.stream_destroy(stream).unwrap();
    // Module leaks until process exit — hip-bridge doesn't expose unload.
    let _ = module;
    hip.free(a_buf).unwrap();
    hip.free(x_buf).unwrap();
    hip.free(y_buf).unwrap();
}

fn main() {
    let hip = HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "unknown".into());
    eprintln!("[hip_graph_gemv_poc] arch={arch}");

    let direct = DirectHip::load();

    // Load the real gemv_hfq4g256_wide source from the repo. This is the
    // hot-path GEMV kernel on gfx1010/gfx1013 for any M >= 64.
    let wide_src = std::fs::read_to_string("kernels/src/gemv_hfq4g256_wide.hip")
        .expect("read gemv_hfq4g256_wide.hip");
    let narrow_src = std::fs::read_to_string("kernels/src/gemv_hfq4g256.hip")
        .expect("read gemv_hfq4g256.hip");

    // Qwen3.5 0.8B real sizes:
    //   dim=1024, hidden_dim=2816
    //   wq/wk/wv/wo: M=1024 (or similar), K=1024
    //   w_gate/w_up: M=2816, K=1024
    //   w_down:      M=1024, K=2816

    // Pick the most representative — 1024×1024 (wo, small projections).
    let m = 1024u32;
    let k = 1024u32;

    // gemv_hfq4g256_wide uses block=[64, 1, 1] (2 warps per block, one row per warp)
    // and grid = [ceil(M/2), 1, 1].
    let block_wide: [u32; 3] = [64, 1, 1];
    let grid_wide: [u32; 3] = [(m + 1) / 2, 1, 1];
    // Narrow uses block=[32, 1, 1] and grid = [M, 1, 1].
    let block_narrow: [u32; 3] = [32, 1, 1];
    let grid_narrow: [u32; 3] = [m, 1, 1];

    // 0.8B forward pass has ~138 gemv_hfq4g256 calls per step — reproduce
    // that exactly as the N-launches-per-graph parameter.
    let n_launches = 138u32;

    bench_kernel(
        &hip, &direct,
        "gemv_hfq4g256_wide",
        &wide_src,
        m, k,
        block_wide,
        grid_wide,
        n_launches,
        &arch,
    );

    bench_kernel(
        &hip, &direct,
        "gemv_hfq4g256",
        &narrow_src,
        m, k,
        block_narrow,
        grid_narrow,
        n_launches,
        &arch,
    );

    // Also test a larger matrix — 2816×1024 like w_gate/w_up, the biggest
    // GEMV in the 0.8B forward pass.
    let m2 = 2816u32;
    let k2 = 1024u32;
    let grid_wide_big: [u32; 3] = [(m2 + 1) / 2, 1, 1];
    bench_kernel(
        &hip, &direct,
        "gemv_hfq4g256_wide",
        &wide_src,
        m2, k2,
        block_wide,
        grid_wide_big,
        n_launches,
        &arch,
    );

    // ─── Realistic forward-pass shape ─────────────────────────────────
    // Capture a representative mix of GEMV sizes matching one Qwen3.5 0.8B
    // layer and replay 24 times (one per layer). Measures whether the
    // heterogeneous mix of small/medium/big kernels favors graph or seq.
    eprintln!("\n========================================");
    eprintln!("  REALISTIC FORWARD-PASS SHAPE (mixed sizes)");
    eprintln!("========================================");
    realistic_forward_shape(&hip, &direct, &wide_src, &narrow_src, &arch);

    eprintln!("\n=== DONE ===");
}

fn realistic_forward_shape(
    hip: &HipRuntime,
    direct: &DirectHip,
    wide_src: &str,
    narrow_src: &str,
    arch: &str,
) {
    // Compile both kernels once
    let compile = |name: &str, src: &str| -> hip_bridge::Function {
        let src_path = format!("/tmp/hip_graph_real_{name}.hip");
        let hsaco_path = format!("/tmp/hip_graph_real_{name}.hsaco");
        std::fs::write(&src_path, src).unwrap();
        let out = std::process::Command::new("hipcc")
            .args([
                "--genco",
                &format!("--offload-arch={arch}"),
                "-O3",
                "-I", "kernels/src",
                "-o", &hsaco_path,
                &src_path,
            ])
            .output()
            .expect("hipcc");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let hsaco = std::fs::read(&hsaco_path).unwrap();
        let module = hip.module_load_data(&hsaco).unwrap();
        let f = hip.module_get_function(&module, name).unwrap();
        std::mem::forget(module); // leak
        f
    };
    let wide = compile("gemv_hfq4g256_wide", wide_src);
    let narrow = compile("gemv_hfq4g256", narrow_src);

    // Qwen3.5 0.8B layer shapes. 7 GEMVs per LA layer, 7 per FA layer.
    // Represented as (kernel_kind, M, K) where kind = 0 wide, 1 narrow.
    // Use wide when M >= 64 (matches dispatch.rs).
    //
    // LA layer: wqkv (k*2+v = 768×1024), wz (v=256×1024), w_beta (32×1024),
    //           w_alpha (32×1024), wo (1024×256), w_gate (2816×1024),
    //           w_up (2816×1024), w_down (1024×2816)
    // 8 GEMVs per LA layer.
    let la_gemvs: Vec<(u32, u32, u32)> = vec![
        (0, 768, 1024),   // wqkv
        (0, 256, 1024),   // wz
        (1,  32, 1024),   // w_beta (narrow — small M)
        (1,  32, 1024),   // w_alpha
        (0, 1024, 256),   // wo (residual, small K)
        (0, 2816, 1024),  // w_gate  ← BIG
        (0, 2816, 1024),  // w_up    ← BIG
        (0, 1024, 2816),  // w_down  ← BIG (residual)
    ];
    // FA layer: wq (1024×1024), wk (256×1024), wv (256×1024),
    //           wo (1024×1024), w_gate, w_up, w_down. 7 GEMVs.
    let fa_gemvs: Vec<(u32, u32, u32)> = vec![
        (0, 1024, 1024),  // wq
        (0,  256, 1024),  // wk
        (0,  256, 1024),  // wv
        (0, 1024, 1024),  // wo (residual)
        (0, 2816, 1024),  // w_gate
        (0, 2816, 1024),  // w_up
        (0, 1024, 2816),  // w_down (residual)
    ];

    // Qwen3.5 0.8B layer pattern: ~18 LA + 6 FA = 24 layers
    // Total: 18 × 8 + 6 × 7 = 144 + 42 = 186 GEMV launches per step
    // which matches our measured 138 + 48 = 186 from the profile.
    let mut seq: Vec<(u32, u32, u32)> = Vec::new();
    // LA layers
    for _ in 0..18 {
        seq.extend_from_slice(&la_gemvs);
    }
    // FA layers
    for _ in 0..6 {
        seq.extend_from_slice(&fa_gemvs);
    }
    eprintln!("  {} GEMV calls/step total  (18 LA × 8 + 6 FA × 7)", seq.len());

    // Find max sizes so we can allocate a single set of buffers large enough
    // for every shape. Point all kernels at the same buffers.
    let max_m = seq.iter().map(|t| t.1).max().unwrap();
    let max_k = seq.iter().map(|t| t.2).max().unwrap();
    let max_groups = max_k / 256;
    let max_row_bytes = (max_groups * 136) as usize;
    let a_bytes = (max_m as usize) * max_row_bytes;
    let x_bytes = (max_k as usize) * 4;
    let y_bytes = (max_m as usize) * 4;
    let a_buf = hip.malloc(a_bytes).unwrap();
    let x_buf = hip.malloc(x_bytes).unwrap();
    let y_buf = hip.malloc(y_bytes).unwrap();
    let junk: Vec<u8> = (0..a_bytes).map(|i| ((i & 0x7f) | 0x10) as u8).collect();
    hip.memcpy_htod(&a_buf, &junk).unwrap();
    let x_junk: Vec<f32> = (0..max_k as usize).map(|i| (i as f32) * 0.01).collect();
    hip.memcpy_htod(&x_buf, unsafe {
        std::slice::from_raw_parts(x_junk.as_ptr() as *const u8, x_bytes)
    }).unwrap();
    hip.memcpy_htod(&y_buf, &vec![0u8; y_bytes]).unwrap();

    // Stable per-launch state. EVERY pointer we pass into hipModuleLaunchKernel
    // must live past the point where HIP graph capture records it — otherwise
    // graph replay dereferences stack-gone memory and crashes with
    // HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION. We Box each piece so the backing
    // memory has a stable address that Vec operations won't invalidate.
    let a_ptr = a_buf.as_ptr() as u64;
    let x_ptr = x_buf.as_ptr() as u64;
    let y_ptr = y_buf.as_ptr() as u64;
    let mut kernarg_bufs: Vec<Box<[u8; 32]>> = Vec::with_capacity(seq.len());
    let mut kernarg_sizes: Vec<Box<usize>> = Vec::with_capacity(seq.len());
    let mut extras: Vec<Box<[*mut c_void; 5]>> = Vec::with_capacity(seq.len());
    for &(_, m, k) in &seq {
        let mut buf: [u8; 32] = [0; 32];
        buf[0..8].copy_from_slice(&a_ptr.to_le_bytes());
        buf[8..16].copy_from_slice(&x_ptr.to_le_bytes());
        buf[16..24].copy_from_slice(&y_ptr.to_le_bytes());
        buf[24..28].copy_from_slice(&(m as i32).to_le_bytes());
        buf[28..32].copy_from_slice(&(k as i32).to_le_bytes());
        let mut kbuf = Box::new(buf);
        let mut ksize = Box::new(32usize);
        let extra: [*mut c_void; 5] = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            kbuf.as_mut_ptr() as *mut c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            (&mut *ksize) as *mut usize as *mut c_void,
            HIP_LAUNCH_PARAM_END,
        ];
        let extra_box = Box::new(extra);
        kernarg_bufs.push(kbuf);
        kernarg_sizes.push(ksize);
        extras.push(extra_box);
    }

    // Grid/block per launch (constant across iterations, independent of buffers).
    let grids_blocks: Vec<([u32; 3], [u32; 3], HipFunction)> = seq
        .iter()
        .map(|&(kind, m, _k)| {
            if kind == 0 {
                // wide: block=[64,1,1], grid=[(m+1)/2,1,1]
                ([(m + 1) / 2, 1, 1], [64, 1, 1], function_handle(&wide))
            } else {
                // narrow: block=[32,1,1], grid=[m,1,1]
                ([m, 1, 1], [32, 1, 1], function_handle(&narrow))
            }
        })
        .collect();

    // Sequential baseline
    let stream = hip.stream_create().unwrap();
    let do_launch = |i: usize, stream_raw: HipStream| -> u32 {
        let (grid, block, f) = grids_blocks[i];
        unsafe {
            (direct.fn_module_launch_kernel)(
                f,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0,
                stream_raw,
                std::ptr::null_mut(),
                extras[i].as_ptr() as *mut *mut c_void,
            )
        }
    };

    // Warmup
    for _ in 0..5 {
        for i in 0..seq.len() {
            let rc = do_launch(i, stream_handle(&stream));
            assert_eq!(rc, 0, "warmup launch {i} failed: {rc}");
        }
    }
    hip.stream_synchronize(&stream).unwrap();

    // Sequential timed (seq.len() launches × K iters)
    let iters = 30u32;
    let t = Instant::now();
    for _ in 0..iters {
        for i in 0..seq.len() {
            let _ = do_launch(i, stream_handle(&stream));
        }
    }
    hip.stream_synchronize(&stream).unwrap();
    let seq_total_us = t.elapsed().as_secs_f64() * 1_000_000.0;
    let seq_per_step = seq_total_us / iters as f64;
    let seq_per_call = seq_per_step / seq.len() as f64;
    eprintln!(
        "  [SEQ] {iters} steps × {} calls = {} launches   total {seq_total_us:.0} µs   per-step {seq_per_step:.1} µs   per-call {seq_per_call:.3} µs",
        seq.len(), iters as usize * seq.len()
    );

    // Graph capture: ONE step = one graph
    let graph_stream = hip.stream_create().unwrap();
    hip.stream_begin_capture(&graph_stream, 0).unwrap();
    for i in 0..seq.len() {
        let rc = do_launch(i, stream_handle(&graph_stream));
        assert_eq!(rc, 0, "graph capture launch {i} failed: {rc}");
    }
    let graph = hip.stream_end_capture(&graph_stream).unwrap();
    let exec = hip.graph_instantiate(&graph).unwrap();
    eprintln!("  [GRAPH] captured {}-launch graph", seq.len());

    // Warmup
    for _ in 0..5 {
        hip.graph_launch(&exec, &graph_stream).unwrap();
    }
    hip.stream_synchronize(&graph_stream).unwrap();

    // Timed
    let t = Instant::now();
    for _ in 0..iters {
        hip.graph_launch(&exec, &graph_stream).unwrap();
    }
    hip.stream_synchronize(&graph_stream).unwrap();
    let graph_total_us = t.elapsed().as_secs_f64() * 1_000_000.0;
    let graph_per_step = graph_total_us / iters as f64;
    let graph_per_call = graph_per_step / seq.len() as f64;
    eprintln!(
        "  [GRAPH] {iters} steps × {} calls = {} launches   total {graph_total_us:.0} µs   per-step {graph_per_step:.1} µs   per-call {graph_per_call:.3} µs",
        seq.len(), iters as usize * seq.len()
    );

    // Delta
    let delta_us = seq_per_step - graph_per_step;
    let pct = delta_us / seq_per_step * 100.0;
    let speedup = seq_per_step / graph_per_step;
    eprintln!(
        "  ==> per-step SEQ {seq_per_step:.1} µs  GRAPH {graph_per_step:.1} µs  ΔSAVING {delta_us:+.1} µs ({pct:+.1}%)   speedup {speedup:.2}×"
    );

    // Project to tok/s on top of the measured 4.24 ms/step baseline
    let baseline_step_ms = 4.24f64;
    // GEMV is ~51.8% of that = 2.20 ms. The remaining ~2.04 ms is non-GEMV.
    // We've only measured GEMV portion; assume non-GEMV is unchanged for now.
    let gemv_step_ms = 2.20f64;
    let non_gemv_step_ms = baseline_step_ms - gemv_step_ms;
    // Scale our measured seq_per_step (GEMV-only) to match the real GEMV time
    // by applying the graph delta as a ratio.
    let gemv_save_ratio = delta_us / seq_per_step;
    let new_gemv_ms = gemv_step_ms * (1.0 - gemv_save_ratio);
    let new_step_ms = new_gemv_ms + non_gemv_step_ms;
    let old_tps = 1000.0 / baseline_step_ms;
    let new_tps = 1000.0 / new_step_ms;
    let tps_pct = (new_tps - old_tps) / old_tps * 100.0;
    eprintln!(
        "  projection: gemv {gemv_step_ms:.2} ms → {new_gemv_ms:.2} ms  (non-gemv {non_gemv_step_ms:.2} unchanged)"
    );
    eprintln!(
        "              step {baseline_step_ms:.2} ms → {new_step_ms:.2} ms  ({old_tps:.1} tok/s → {new_tps:.1} tok/s, {tps_pct:+.1}%)"
    );

    hip.graph_exec_destroy(exec).unwrap();
    hip.graph_destroy(graph).unwrap();
    hip.stream_destroy(graph_stream).unwrap();
    hip.stream_destroy(stream).unwrap();
    hip.free(a_buf).unwrap();
    hip.free(x_buf).unwrap();
    hip.free(y_buf).unwrap();
}
