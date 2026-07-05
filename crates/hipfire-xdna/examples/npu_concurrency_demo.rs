//! Proof-of-premise for the concurrent NPU ‖ (GPU/host) offload: an async NPU dispatch
//! (`submit` → do other work → `wait`) overlaps with concurrent host work, so the NPU GEMM
//! is hidden behind it instead of adding to the critical path. The whole offload thesis
//! rests on this — if the host can't run useful work during the NPU `wait`, offload is a
//! net loss regardless of NPU TOPS.
//!
//! Measures three schedules on the M-parallel whole-GEMM xclbin + a CPU workload proxy for
//! "GPU-dispatch-issuing host work":
//!   T_npu     : blocking NPU dispatch alone (submit+wait).
//!   T_host    : the host workload alone.
//!   T_serial  : submit; wait; host_work         (no overlap)  ~ T_npu + T_host
//!   T_overlap : submit; host_work; wait          (overlapped) ~ max(T_npu, T_host)
//! Overlap saving ≈ min(T_npu, T_host) proves the NPU ran concurrently with the host.
//!
//! Run: cargo run --release -p hipfire-xdna --example npu_concurrency_demo -- <mp-xclbin-dir> [asz wsz csz]

// Host workload proxy for "GPU-dispatch-issuing host work": a memory-touching busy loop.
#[cfg(target_os = "linux")]
fn host_work(iters: u64) -> u64 {
    let mut acc = 0u64;
    for i in 0..iters {
        acc = acc.wrapping_add(i.wrapping_mul(2654435761)).rotate_left(7);
        std::hint::black_box(acc);
    }
    std::hint::black_box(acc)
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        // Defaults = r6mp_8x4x32_c8_nb64_r3 (whole-GEMM, ~0.9 ms dispatch).
        let (asz, wsz, csz) = (p(2, 393216), p(3, 3145728), p(4, 12582912));

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");
        let mut aw = k.alloc_arg(asz).expect("A");
        let mut ww = k.alloc_arg(wsz).expect("W");
        let cw = k.alloc_arg(csz).expect("C");
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);
        let args = [&aw, &ww, &cw];

        let time = |n: u32, mut f: Box<dyn FnMut()>| -> f64 {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        };

        // warm up
        for _ in 0..10 {
            k.dispatch(&args).expect("warm");
        }
        let iters = 30u32;

        let t_npu = time(
            iters,
            Box::new(|| {
                k.dispatch(&args).expect("d");
            }),
        );

        // Calibrate host_work to ~t_npu.
        let mut hi = 1u64 << 20;
        loop {
            let t = time(
                5,
                Box::new(move || {
                    host_work(hi);
                }),
            );
            if t >= t_npu || hi > (1u64 << 34) {
                break;
            }
            hi = (hi as f64 * (t_npu / t.max(1e-3)) * 1.1) as u64 + (1 << 20);
        }
        let t_host = time(
            iters,
            Box::new(|| {
                host_work(hi);
            }),
        );

        let t_serial = time(
            iters,
            Box::new(|| {
                let seq = k.submit(&args).expect("s");
                k.wait(seq).expect("w");
                host_work(hi);
            }),
        );
        let t_overlap = time(
            iters,
            Box::new(|| {
                let seq = k.submit(&args).expect("s"); // NPU runs async
                host_work(hi); // host runs concurrently
                k.wait(seq).expect("w"); // collect NPU
            }),
        );

        let saved = t_serial - t_overlap;
        let hidden = 100.0 * saved / t_npu.max(1e-9);
        println!("T_npu    (dispatch alone)    = {t_npu:.3} ms");
        println!("T_host   (host work alone)   = {t_host:.3} ms");
        println!("T_serial (submit;wait;host)  = {t_serial:.3} ms");
        println!("T_overlap(submit;host;wait)  = {t_overlap:.3} ms");
        println!(
            "=> overlap saves {saved:.3} ms/iter ({hidden:.0}% of the NPU GEMM hidden behind host work)"
        );
        if saved > 0.15 * t_npu.min(t_host) {
            println!("CONCURRENCY CONFIRMED — NPU dispatch overlaps concurrent host work");
        } else {
            println!("NO overlap observed (host cannot run during NPU wait on this path)");
        }
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
