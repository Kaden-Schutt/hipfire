//! End-to-end NPU dispatch through the reusable [`NpuKernel`] API (W5). Loads a
//! compiled mlir-aie kernel from a cache dir (`final.xclbin` + `insts.bin`), binds
//! A/W/C buffers, dispatches on hardware, and reads back C.
//!
//! Targets the R2a single-core W4A8 GEMM (NACC=8, INNER=64, N_BTILES=4):
//! A=all-1s int8 × W=all-1s int4 → C lane = 16·(INNER+1) = 1040.
//!
//! Run: `cargo run -p hipfire-xdna --example run_smoke -- <npu-cache-dir>`

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;

        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("usage: run_smoke <npu-cache-dir> [asize wsize csize]");
            eprintln!("  default sizes are R2a's (512 512 2048); pass 3 sizes for other kernels");
            std::process::exit(2);
        }
        let dir = &args[1];
        // A/W/C byte sizes; default to the R2a W4A8 GEMM signature.
        let (asz, wsz, csz) = if args.len() >= 5 {
            (
                args[2].parse().expect("asize"),
                args[3].parse().expect("wsize"),
                args[4].parse().expect("csize"),
            )
        } else {
            (512usize, 512usize, 2048usize)
        };
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");

        let kernel = NpuKernel::load(&xclbin, &insts).expect("load kernel");
        println!(
            "kernel loaded ({} B insts), A={asz} W={wsz} C={csz}",
            insts.len()
        );

        let mut a = kernel.alloc_arg(asz).expect("A");
        let mut w = kernel.alloc_arg(wsz).expect("W");
        let mut c = kernel.alloc_arg(csz).expect("C");
        a.as_mut_slice().fill(1); // int8 activations = 1
        w.as_mut_slice().fill(0x11); // two int4 = 1 per byte
        c.as_mut_slice().fill(0);

        let read_c0 = |c: &hipfire_xdna::DeviceBuffer| -> i32 {
            unsafe { *(c.as_slice().as_ptr() as *const i32) }
        };

        // Dispatch twice with A=1 then A=2 to prove the hwctx/program is reusable,
        // per-dispatch command BOs do not leak, and the math is right: any linear
        // W4A8 GEMM/GEMV must satisfy C(A=2) = 2·C(A=1) (kernel-agnostic invariant).
        kernel.dispatch(&[&a, &w, &c]).expect("dispatch 1");
        let c0 = read_c0(&c);

        a.as_mut_slice().fill(2);
        c.as_mut_slice().fill(0);
        kernel.dispatch(&[&a, &w, &c]).expect("dispatch 2");
        let c1 = read_c0(&c);

        println!("dispatch1 C[0]={c0}, dispatch2 C[0]={c1} (expect 2×)");
        if c0 == 0 || c1 != 2 * c0 {
            eprintln!("unexpected results — kernel math off (C should be nonzero and double)");
            std::process::exit(4);
        }
        println!("NPU dispatch OK via NpuKernel (reuse + linear W4A8 math)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
