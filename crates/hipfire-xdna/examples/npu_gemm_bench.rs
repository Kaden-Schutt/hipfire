//! Measure aggregate W4A8 GEMM throughput on the NPU through the hipfire NpuKernel
//! dispatch path (no XRT). Loads a compiled mlir-aie kernel from a cache dir, binds
//! A/W/C, validates the all-ones result, then times a dispatch loop for TOPS.
//!
//! Run: cargo run -p hipfire-xdna --example npu_gemm_bench -- <cache-dir> \
//!        <asize> <wsize> <csize> <macs-per-dispatch> [iters] [expect_c0]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        use std::time::Instant;

        let a: Vec<String> = std::env::args().collect();
        if a.len() < 6 {
            eprintln!("usage: npu_gemm_bench <dir> <asz> <wsz> <csz> <macs> [iters] [expect_c0]");
            std::process::exit(2);
        }
        let dir = &a[1];
        let asz: usize = a[2].parse().unwrap();
        let wsz: usize = a[3].parse().unwrap();
        let csz: usize = a[4].parse().unwrap();
        let macs: f64 = a[5].parse().unwrap();
        let iters: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(500);
        let expect: Option<i32> = a.get(7).and_then(|s| s.parse().ok());

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut aw = k.alloc_arg(asz).expect("A");
        let mut ww = k.alloc_arg(wsz).expect("W");
        let mut cw = k.alloc_arg(csz).expect("C");
        aw.as_mut_slice().fill(1);
        ww.as_mut_slice().fill(0x11);
        cw.as_mut_slice().fill(0);

        // Correctness gate: all-ones W4A8 compute.
        k.dispatch(&[&aw, &ww, &cw]).expect("dispatch");
        let c0 = unsafe { *(cw.as_slice().as_ptr() as *const i32) };
        println!(
            "all-ones C[0] = {c0}{}",
            match expect {
                Some(e) => format!(" (expect {e})"),
                None => String::new(),
            }
        );
        if let Some(e) = expect {
            if c0 != e {
                eprintln!("correctness FAIL");
                std::process::exit(4);
            }
        }

        // Warm up, then time the dispatch loop.
        for _ in 0..20 {
            k.dispatch(&[&aw, &ww, &cw]).expect("warmup");
        }
        let t = Instant::now();
        for _ in 0..iters {
            k.dispatch(&[&aw, &ww, &cw]).expect("bench");
        }
        let dt = t.elapsed().as_secs_f64();
        let per = dt / iters as f64;
        let tops = 2.0 * macs / per / 1e12;
        println!(
            "iters={iters} total={:.3}s per_dispatch={:.1}us  MACs/dispatch={macs:.0}  => {tops:.2} TOPS",
            dt,
            per * 1e6
        );
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
