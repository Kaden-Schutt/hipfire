//! Validate the M-parallel W-broadcast R6 array (r6_gen_mp.py + r6_gemm_ts.cc): COLS cores
//! each compute a DISTINCT M-block over full N, all sharing ONE broadcast W stream. Checks
//! that every core's M-block is correct against the shared W (all-ones can't distinguish
//! M-blocks; this uses random data). Layouts:
//!   A: COLS row-major (MT*MR)x(KCHUNK*MK) M-blocks, core c at c*AW.
//!   W: NB N-slabs, slab j at j*WW, each NT*KCHUNK tile-major int4 tiles (the TS kernel's
//!      W layout) covering N-cols [j*NT*MN, +NT*MN).
//!   C: COLS*NB row-major (MT*MR)x(NT*MN) blocks, core c slab j at (c*NB+j)*CW.
//!
//! Build: R6_KERNEL_SRC=<r6>/r6_gemm_ts.cc R6_GEN=r6_gen_mp.py R6_OUT_TAG=r6mp <r6>/r6_cache.sh MT 4 KCHUNK COLS NB
//! Run:   cargo run -p hipfire-xdna --example r6_mp_verify -- <dir> COLS MT KCHUNK NB

fn main() {
    #[cfg(target_os = "linux")]
    {
        use hipfire_xdna::NpuKernel;
        let a: Vec<String> = std::env::args().collect();
        let dir = &a[1];
        let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (cols, mt, kc, nb) = (p(2, 2), p(3, 8), p(4, 8), p(5, 2));
        const NT: usize = 4;
        const MR: usize = 4;
        const MK: usize = 16;
        const MN: usize = 16;
        let m = cols * mt * MR;
        let k = kc * MK;
        let n = nb * NT * MN;
        let aw = mt * kc * MR * MK; // bytes per A M-block
        let ww = NT * kc * MK * MN / 2; // bytes per W N-slab (int4 packed)
        let cw = mt * NT * MR * MN; // i32 per C block

        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).expect("xclbin");
        let insts = std::fs::read(format!("{dir}/insts.bin")).expect("insts");
        let kern = NpuKernel::load(&xclbin, &insts).expect("load");
        let mut abuf = kern.alloc_arg(cols * aw).expect("A");
        let mut wbuf = kern.alloc_arg(nb * ww).expect("W");
        let cbuf = kern.alloc_arg(cols * nb * cw * 4).expect("C");

        let rnd = |i: usize| -> i32 {
            let s = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e37_79b9);
            ((s >> 13) & 0xf) as i32 - 8
        };

        // A: row-major per M-block. aref[m][k].
        let aref: Vec<i32> = (0..m * k).map(rnd).collect();
        {
            let s = abuf.as_mut_slice();
            for c in 0..cols {
                for r in 0..mt * MR {
                    for kk in 0..k {
                        s[c * aw + r * k + kk] = aref[(c * mt * MR + r) * k + kk] as i8 as u8;
                    }
                }
            }
        }
        // W: NB slabs, tile-major int4. wref[kg][ng].
        let mut wref = vec![0i32; k * n];
        {
            let s = wbuf.as_mut_slice();
            s.fill(0);
            for j in 0..nb {
                for nt in 0..NT {
                    for ki in 0..kc {
                        for kk in 0..MK {
                            for nn in 0..MN {
                                let kg = ki * MK + kk;
                                let ng = j * NT * MN + nt * MN + nn;
                                let v = rnd(1000 + kg * n + ng);
                                wref[kg * n + ng] = v;
                                let idx = (nt * kc + ki) * (MK * MN) + kk * MN + nn; // int4 in slab
                                let byte = j * ww + idx / 2;
                                let u = (v & 0xf) as u8;
                                s[byte] |= if idx % 2 == 0 { u } else { u << 4 };
                            }
                        }
                    }
                }
            }
        }

        kern.dispatch(&[&abuf, &wbuf, &cbuf]).expect("dispatch");
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(cbuf.as_slice().as_ptr() as *const i32, cols * nb * cw)
        };

        // C: core c slab j block at (c*nb+j)*cw, row-major (mt*MR)x(NT*MN). Global
        // C[c*mt*MR + r][j*NT*MN + col].
        let mut mism = 0usize;
        for c in 0..cols {
            for j in 0..nb {
                for r in 0..mt * MR {
                    for col in 0..NT * MN {
                        let mg = c * mt * MR + r;
                        let ng = j * NT * MN + col;
                        let acc: i32 = (0..k).map(|kk| aref[mg * k + kk] * wref[kk * n + ng]).sum();
                        let got = out[(c * nb + j) * cw + r * (NT * MN) + col];
                        if got != acc {
                            mism += 1;
                        }
                    }
                }
            }
        }
        println!("{mism}/{} mismatches (COLS={cols} MT={mt} NT={NT} KCHUNK={kc} NB={nb}, M={m} K={k} N={n})", cols * nb * cw);
        if mism != 0 {
            eprintln!("M-parallel W-broadcast array WRONG");
            std::process::exit(4);
        }
        println!("M-parallel W-broadcast R6 array CORRECT — {cols} distinct M-blocks share one broadcast W");
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("amdxdna is Linux-only");
}
