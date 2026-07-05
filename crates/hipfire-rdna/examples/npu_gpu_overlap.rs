// Concurrency probe: do the Phoenix iGPU (gfx1103) and the XDNA1 NPU make progress
// SIMULTANEOUSLY, or does one stall the other? This is the make-or-break for the "NPU as
// a parallel unit" thesis (parallel prefill offload; a pipelined spec-decode draft) — the
// value is concurrency, not raw NPU speed.
//
// Method: a prefill-shaped GPU WMMA GEMM (F16) and the R14 whole_array NPU GEMM dispatch,
// each run in a time-boxed busy loop. Measure each engine's own throughput SOLO, then run
// them CONCURRENTLY (NPU on its own thread, GPU on main) started via a barrier. If both
// throughputs are ~unchanged under concurrency, the engines overlap; if each ~halves, they
// serialize / contend. Reports per-engine slowdown and an overlap efficiency.
//
// Usage: cargo run --release -p hipfire-rdna --example npu_gpu_overlap -- \
//          <npu-cache-dir> <asz> <wsz> <csz> [secs] [macs_per_dispatch]
#![allow(clippy::too_many_arguments, clippy::useless_vec)]

use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_xdna::NpuKernel;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

fn wrap(ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dt: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
        shape,
        dtype: dt,
    }
}

fn npu_busy(xclbin: &[u8], insts: &[u8], asz: usize, wsz: usize, csz: usize, dur: Duration) -> u64 {
    let k = NpuKernel::load(xclbin, insts).expect("npu load");
    let mut a = k.alloc_arg(asz).expect("A");
    let mut w = k.alloc_arg(wsz).expect("W");
    let mut c = k.alloc_arg(csz).expect("C");
    a.as_mut_slice().fill(1);
    w.as_mut_slice().fill(0x11);
    c.as_mut_slice().fill(0);
    for _ in 0..2 {
        k.dispatch(&[&a, &w, &c]).expect("warmup");
    }
    let t0 = Instant::now();
    let mut n = 0u64;
    while t0.elapsed() < dur {
        k.dispatch(&[&a, &w, &c]).expect("dispatch");
        n += 1;
    }
    n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: npu_gpu_overlap <npu-dir> <asz> <wsz> <csz> [secs] [macs/dispatch]");
        std::process::exit(2);
    }
    let dir = args[1].clone();
    let asz: usize = args[2].parse().unwrap();
    let wsz: usize = args[3].parse().unwrap();
    let csz: usize = args[4].parse().unwrap();
    let secs: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let npu_macs: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let dur = Duration::from_secs_f64(secs);

    // ── GPU: prefill-shaped WMMA GEMM (hipfire's own F16 kernel) ─────────────────
    let mut gpu = Gpu::init().expect("gpu init");
    println!("GPU arch: {}  |  probe {secs}s per phase", gpu.arch);
    let (m, k, b) = (1024usize, 7168usize, 256usize);
    let wg = gpu.hip.malloc(m * k * 2).unwrap();
    let xg = gpu.hip.malloc(b * k * 2).unwrap();
    let yg = gpu.hip.malloc(b * m * 4).unwrap();
    gpu.hip.memcpy_htod(&wg, &vec![1u8; m * k * 2]).unwrap();
    gpu.hip.memcpy_htod(&xg, &vec![1u8; b * k * 2]).unwrap();
    let wt = wrap(wg.as_ptr(), m * k * 2, vec![m, k], DType::F16);
    let xt = wrap(xg.as_ptr(), b * k * 2, vec![b, k], DType::F16);
    let yt = wrap(yg.as_ptr(), b * m * 4, vec![b, m], DType::F32);
    let gflop_per = 2.0 * m as f64 * k as f64 * b as f64 / 1e9;
    for _ in 0..8 {
        gpu.gemm_f16_x_f16_wmma(&wt, &xt, &yt, m, k, b).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let mut gpu_busy = |dur: Duration| -> u64 {
        let t0 = Instant::now();
        let mut n = 0u64;
        while t0.elapsed() < dur {
            for _ in 0..16 {
                gpu.gemm_f16_x_f16_wmma(&wt, &xt, &yt, m, k, b).unwrap();
                n += 1;
            }
            gpu.hip.device_synchronize().unwrap();
        }
        n
    };

    let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");

    // ── SOLO baselines ───────────────────────────────────────────────────────────
    let gpu_solo = gpu_busy(dur);
    let npu_solo = npu_busy(&xclbin, &insts, asz, wsz, csz, dur);
    let gpu_solo_gflops = gpu_solo as f64 * gflop_per / secs;
    let npu_solo_rate = npu_solo as f64 / secs;

    // ── CONCURRENT: NPU on its own thread, GPU on main, started together ──────────
    let barrier = Arc::new(Barrier::new(2));
    let (xc, ic, b2) = (xclbin.clone(), insts.clone(), barrier.clone());
    let npu_h = std::thread::spawn(move || {
        b2.wait();
        npu_busy(&xc, &ic, asz, wsz, csz, dur)
    });
    barrier.wait();
    let gpu_conc = gpu_busy(dur);
    let npu_conc = npu_h.join().unwrap();
    let gpu_conc_gflops = gpu_conc as f64 * gflop_per / secs;
    let npu_conc_rate = npu_conc as f64 / secs;

    // ── Report ───────────────────────────────────────────────────────────────────
    let gpu_keep = gpu_conc_gflops / gpu_solo_gflops;
    let npu_keep = npu_conc_rate / npu_solo_rate;
    // Overlap efficiency: 1.0 = both keep full throughput (perfect overlap); 0 = fully
    // serialized (concurrent sum of fractions = 1). frac_sum in [1,2]; (sum-1) in [0,1].
    let overlap = (gpu_keep + npu_keep - 1.0).clamp(0.0, 1.0);
    println!("\n                    SOLO            CONCURRENT      keeps");
    println!(
        "GPU WMMA GEMM   {:8.1} GFLOP/s   {:8.1} GFLOP/s   {:.0}%",
        gpu_solo_gflops,
        gpu_conc_gflops,
        gpu_keep * 100.0
    );
    println!(
        "NPU dispatch    {:8.1} disp/s     {:8.1} disp/s     {:.0}%",
        npu_solo_rate,
        npu_conc_rate,
        npu_keep * 100.0
    );
    if npu_macs > 0.0 {
        println!(
            "  (NPU: {:.2} TOPS solo -> {:.2} TOPS concurrent)",
            npu_solo_rate * 2.0 * npu_macs / 1e12,
            npu_conc_rate * 2.0 * npu_macs / 1e12
        );
    }
    println!(
        "\nOverlap efficiency: {:.0}%  ({})",
        overlap * 100.0,
        if overlap > 0.75 {
            "engines run in parallel — parallel prefill / spec-draft is viable"
        } else if overlap > 0.3 {
            "partial overlap — some contention"
        } else {
            "serialized — one engine stalls the other"
        }
    );
    std::mem::forget(wt);
    std::mem::forget(xt);
    std::mem::forget(yt);
}
