//! R6 W4A8 GEMM numerical correctness check (real random data vs a CPU reference).
//! The benches only prove the all-ones throughput ceiling; this proves the kernel
//! computes a *correct* GEMM and pins the tile layout (all row-major), which the
//! runtime `NpuGemm` marshaling depends on.
//!
//! Point it at a workdir built for **MT=1 NT=4 KCHUNK=1** (one M-block × 4 N-blocks ×
//! one K-tile) — see benchmarks/npu_gemm_tuning/r6/README.md for the build recipe.
//!
//! Run: cargo run -p hipfire-xdna --example r6_verify -- <workdir>

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;

        let dir = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!(
                "usage: r6_verify <workdir with final.xclbin + insts.bin (MT=1 NT=4 KCHUNK=1)>"
            );
            std::process::exit(2);
        });
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let k = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut a = k.alloc_arg(64).expect("A"); // 4x16 int8, row-major a[m*16+k]
        let mut w = k.alloc_arg(512).expect("W"); // 4 tiles x 16x16 int4, row-major w[k*16+n], 2/byte
        let c = k.alloc_arg(1024).expect("C"); // 4 tiles x 4x16 int32, row-major c[m*16+n]

        // Deterministic pseudo-random small values in [-8, 7], keyed by index.
        fn rnd(i: usize) -> i32 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            ((s >> 13) & 0xf) as i32 - 8
        }

        let mut aref = [[0i32; 16]; 4];
        {
            let s = a.as_mut_slice();
            for (m, row) in aref.iter_mut().enumerate() {
                for (kk, cell) in row.iter_mut().enumerate() {
                    let v = rnd(m * 16 + kk);
                    *cell = v;
                    s[m * 16 + kk] = v as i8 as u8;
                }
            }
        }
        let mut wref = [[[0i32; 16]; 16]; 4];
        {
            let s = w.as_mut_slice();
            s.fill(0);
            for (nt, wt) in wref.iter_mut().enumerate() {
                for (kk, wrow) in wt.iter_mut().enumerate() {
                    for (n, cell) in wrow.iter_mut().enumerate() {
                        let idx = nt * 256 + kk * 16 + n; // int4 element index
                        let v = rnd(1000 + idx);
                        *cell = v;
                        let u = (v & 0xf) as u8;
                        s[idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                    }
                }
            }
        }

        k.dispatch(&[&a, &w, &c]).expect("dispatch");
        let out: &[i32] =
            unsafe { std::slice::from_raw_parts(c.as_slice().as_ptr() as *const i32, 256) };

        let mut mism = 0;
        for (nt, wt) in wref.iter().enumerate() {
            for (m, arow) in aref.iter().enumerate() {
                for n in 0..16 {
                    let acc: i32 = (0..16).map(|kk| arow[kk] * wt[kk][n]).sum();
                    if out[nt * 64 + m * 16 + n] != acc {
                        mism += 1;
                    }
                }
            }
        }
        println!("{}/256 mismatches", mism);
        if mism != 0 {
            eprintln!("R6 GEMM numerically WRONG");
            std::process::exit(4);
        }
        println!("R6 real-data W4A8 GEMM CORRECT (all tiles row-major)");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
