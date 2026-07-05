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

        let dir = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!("usage: run_smoke <npu-cache-dir with final.xclbin + insts.bin>");
            std::process::exit(2);
        });
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("read final.xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");

        let kernel = NpuKernel::load(&xclbin, &insts).expect("load kernel");
        println!("kernel loaded ({} B insts)", insts.len());

        // R2a signature: A (512 i8), W (512 B packed int4), C (2048 B = 512 i32).
        let mut a = kernel.alloc_arg(512).expect("A");
        let mut w = kernel.alloc_arg(512).expect("W");
        let mut c = kernel.alloc_arg(2048).expect("C");
        a.as_mut_slice().fill(1); // int8 activations = 1
        w.as_mut_slice().fill(0x11); // two int4 = 1 per byte
        c.as_mut_slice().fill(0);

        let read_c = |c: &hipfire_xdna::DeviceBuffer| -> i32 {
            let out: &[i32] =
                unsafe { std::slice::from_raw_parts(c.as_slice().as_ptr() as *const i32, 512) };
            out[0]
        };

        // Dispatch twice on the same kernel with different activations to prove the
        // hwctx/program is reusable and per-dispatch command BOs do not leak.
        kernel.dispatch(&[&a, &w, &c]).expect("dispatch 1");
        let c0 = read_c(&c); // A=1: 16·65 = 1040

        a.as_mut_slice().fill(2);
        c.as_mut_slice().fill(0);
        kernel.dispatch(&[&a, &w, &c]).expect("dispatch 2");
        let c1 = read_c(&c); // A=2: 32·65 = 2080

        println!("dispatch1 C[0]={c0} (expect 1040), dispatch2 C[0]={c1} (expect 2080)");
        if c0 != 1040 || c1 != 2080 {
            eprintln!("unexpected results — kernel math off");
            std::process::exit(4);
        }
        println!("NPU dispatch OK via NpuKernel (reuse + correct W4A8 math)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
