//! Validate the R6-TS kernel (r6_gemm_ts.cc): a full W4A8 GEMM that reads ROW-MAJOR A and
//! writes ROW-MAJOR C via in-core tensor buffer streams (W pre-packed tile-major). Feeds a
//! row-major A block + pre-packed W, reads a row-major C block, compares to a CPU int8xint4
//! reference. If this passes, NpuGemm can drop the dynamic A-pack / C-unpack marshaling and
//! pass activations/outputs row-major.
//!
//! Build:  R6_KERNEL_SRC=<r6 dir>/r6_gemm_ts.cc R6_OUT_TAG=r6ts <r6 dir>/r6_cache.sh MT 4 KCHUNK
//! Run:    cargo run -p hipfire-xdna --example r6_ts_verify -- <workdir> [MT] [KCHUNK]

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;

        let args: Vec<String> = std::env::args().collect();
        let dir = args.get(1).cloned().unwrap_or_else(|| {
            eprintln!("usage: r6_ts_verify <workdir> [MT] [KCHUNK]  (NT=4)");
            std::process::exit(2);
        });
        let mt: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let kc: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
        const NT: usize = 4;
        const MR: usize = 4;
        const MK: usize = 16;
        const MN: usize = 16;
        let m = mt * MR; // rows of A / C
        let k = kc * MK; // cols of A / rows of W
        let n = NT * MN; // cols of C

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let kern = NpuKernel::load(&xclbin, &insts).expect("load");

        let mut a = kern.alloc_arg(m * k).expect("A"); // row-major int8 a[mm*k+kk]
        let mut w = kern.alloc_arg(k * n / 2).expect("W"); // pre-packed tile-major int4
        let c = kern.alloc_arg(m * n * 4).expect("C"); // row-major int32 c[mm*n+nn]

        fn rnd(i: usize) -> i32 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            ((s >> 13) & 0xf) as i32 - 8
        }

        // Row-major A.
        let aref: Vec<i32> = (0..m * k).map(rnd).collect();
        {
            let s = a.as_mut_slice();
            for (i, &v) in aref.iter().enumerate() {
                s[i] = v as i8 as u8;
            }
        }

        // W pre-packed tile-major: tile (nt,ki) at (nt*kc+ki), within-tile w[kk*16+nn],
        // int4 index (nt*kc+ki)*256 + kk*16 + nn, 2 nibbles/byte low-first. wref indexed
        // by global [k_global][n_global] for the CPU GEMM.
        let mut wref = vec![0i32; k * n];
        {
            let s = w.as_mut_slice();
            s.fill(0);
            for nt in 0..NT {
                for ki in 0..kc {
                    for kk in 0..MK {
                        for nn in 0..MN {
                            let idx = (nt * kc + ki) * 256 + kk * MN + nn;
                            let v = rnd(1000 + idx);
                            let kg = ki * MK + kk;
                            let ng = nt * MN + nn;
                            wref[kg * n + ng] = v;
                            let u = (v & 0xf) as u8;
                            s[idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                        }
                    }
                }
            }
        }

        kern.dispatch(&[&a, &w, &c]).expect("dispatch");
        let out: &[i32] =
            unsafe { std::slice::from_raw_parts(c.as_slice().as_ptr() as *const i32, m * n) };

        let mut mism = 0usize;
        for mm in 0..m {
            for nn in 0..n {
                let acc: i32 = (0..k).map(|kk| aref[mm * k + kk] * wref[kk * n + nn]).sum();
                if out[mm * n + nn] != acc {
                    mism += 1;
                }
            }
        }
        println!(
            "{mism}/{} mismatches (MT={mt} NT={NT} KCHUNK={kc}, M={m} K={k} N={n})",
            m * n
        );
        if mism != 0 {
            eprintln!("R6-TS GEMM (row-major A/C via tensor streams) WRONG");
            std::process::exit(4);
        }
        println!("R6-TS W4A8 GEMM CORRECT — row-major A in, row-major C out, zero marshaling");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
